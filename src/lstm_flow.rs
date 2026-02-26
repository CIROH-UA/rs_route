use anyhow::{Context, Result};
use burn::nn::LstmState;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use glob::glob;
use netcdf::open;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub static USE_HARD_CODED_WEIGHTS: bool = true;

#[derive(Clone)]
pub struct NgenLstmConfig {
    pub root_dir: PathBuf,
    pub forcing_path: PathBuf,
    pub divide_index: HashMap<String, usize>,
    pub total_timesteps: usize,
}

pub fn find_forcing_file(root_dir: &Path) -> Result<PathBuf> {
    // forcing dir should contain single .nc file
    // name is *usually* forcings.nc, but is different for datastream
    let forcing_dir = root_dir.join("forcings");
    let mut nc_files = glob(&format!("{}/*.nc", forcing_dir.display()))?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    if nc_files.len() != 1 {
        return Err(anyhow::anyhow!(
            "Expected exactly one .nc file in forcing directory {}, found {}",
            forcing_dir.display(),
            nc_files.len()
        ));
    }
    Ok(nc_files.remove(0))
}

pub fn find_lstm_config_directory(root_dir: &Path) -> Result<PathBuf> {
    let cat_config_dir = root_dir.join("config").join("cat_config");
    // lstm subdir can be capitalized differently depending on the source of the config, so we look for any subdir that contains
    // a variation of "lstm" in the name, i.e. lstm, LSTM
    let mut lstm_dirs = glob(&format!("{}/*/", cat_config_dir.display()))?
        .filter_map(Result::ok)
        .filter(|p| {
            p.file_name()
                .and_then(|name| name.to_str())
                .map(|name_str| name_str.to_lowercase().contains("lstm"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if lstm_dirs.is_empty() {
        let other_dirs = glob(&format!("{}/*/", cat_config_dir.display()))?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        return Err(anyhow::anyhow!(
            "No LSTM config directory found in {}. Expected a subdirectory with 'lstm' in the name. Found these other subdirectories: {:?}",
            cat_config_dir.display(),
            other_dirs.iter().map(|p| p.display()).collect::<Vec<_>>()
        ));
    } else if lstm_dirs.len() != 1 {
        return Err(anyhow::anyhow!(
            "Expected exactly one LSTM config directory in {}, found {}: {:?}",
            cat_config_dir.display(),
            lstm_dirs.len(),
            lstm_dirs.iter().map(|p| p.display()).collect::<Vec<_>>()
        ));
    }
    Ok(lstm_dirs.remove(0))
}

impl NgenLstmConfig {
    pub fn new(root_dir: PathBuf) -> Result<Self> {
        // let forcing_path = root_dir.join("forcings").join("forcings.nc");
        let forcing_path = find_forcing_file(&root_dir)
            .with_context(|| format!("Failed to find forcing file in directory: {:?}", root_dir))?;

        // Open forcing file temporarily to get metadata
        let forcing_file = open(&forcing_path)
            .with_context(|| format!("Failed to open forcing file: {:?}", forcing_path))?;

        // Load divide index mapping
        let divide_index_var = forcing_file
            .variable("ids")
            .context("Couldn't find variable 'ids'")?;

        let mut divide_index = HashMap::new();
        for i in 0..divide_index_var.len() {
            let id = divide_index_var
                .get_string(i)
                .with_context(|| format!("Failed to get string at index {}", i))?;
            divide_index.insert(id, i);
        }

        // Get total timesteps
        let total_timesteps = forcing_file
            .dimension_len("time")
            .context("Couldn't find variable 'Time'")?;

        Ok(Self {
            root_dir,
            forcing_path,
            divide_index,
            total_timesteps,
        })
    }
}
// Copy the NextgenLstm module structure
mod lstm_model {
    use burn::{
        module::Param,
        nn::{
            GateController, Initializer, Linear, LinearConfig, LinearRecord, Lstm, LstmConfig,
            LstmState,
        },
        prelude::*,
        tensor::Bytes,
    };
    use serde_json::Value;
    use std::fs;

    use crate::{lstm_flow::USE_HARD_CODED_WEIGHTS, weights::WeightConfig};

    #[derive(Module, Debug)]
    pub struct NextgenLstm<B: Backend> {
        pub lstm: Lstm<B>,
        pub head: Linear<B>,
    }

    pub fn vec_to_tensor(input_vec: &Vec<f32>, shape: Vec<usize>) -> TensorData {
        let bytes_vec = input_vec
            .iter()
            .flat_map(|&value| f32::to_le_bytes(value))
            .collect();
        let bytes = Bytes::from_bytes_vec(bytes_vec);
        TensorData {
            bytes,
            shape,
            dtype: burn::tensor::DType::F32,
        }
    }

    fn create_with_weights<B: Backend>(
        d_input: usize,
        d_output: usize,
        bias: bool,
        initializer: Initializer,
        input_record: LinearRecord<B>,
        hidden_record: LinearRecord<B>,
    ) -> GateController<B> {
        let l1 = LinearConfig {
            d_input,
            d_output,
            bias,
            initializer: initializer.clone(),
        }
        .init(&input_record.weight.device())
        .load_record(input_record);

        let l2 = LinearConfig {
            d_input,
            d_output,
            bias,
            initializer,
        }
        .init(&hidden_record.weight.device())
        .load_record(hidden_record);

        GateController {
            input_transform: l1,
            hidden_transform: l2,
        }
    }

    fn create_gate_controller<B: Backend>(
        input_weights: &Vec<f32>,
        input_biases: &Vec<f32>,
        hidden_weights: &Vec<f32>,
        hidden_biases: &Vec<f32>,
        device: &Device<B>,
        input_length: usize,
        hidden_size: usize,
    ) -> GateController<B> {
        let input_record = LinearRecord {
            weight: Param::from_data(
                vec_to_tensor(input_weights, vec![input_length, hidden_size]),
                device,
            ),
            bias: Some(Param::from_data(
                vec_to_tensor(input_biases, vec![hidden_size]),
                device,
            )),
        };
        let hidden_record = LinearRecord {
            weight: Param::from_data(
                vec_to_tensor(hidden_weights, vec![hidden_size, hidden_size]),
                device,
            ),
            bias: Some(Param::from_data(
                vec_to_tensor(hidden_biases, vec![hidden_size]),
                device,
            )),
        };

        create_with_weights(
            input_length,
            hidden_size,
            true,
            Initializer::Zeros,
            input_record,
            hidden_record,
        )
    }

    impl<B: Backend> NextgenLstm<B> {
        pub fn init(
            device: &B::Device,
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
        ) -> NextgenLstm<B> {
            let lstm = LstmConfig::new(input_size, hidden_size, true)
                .with_initializer(nn::Initializer::Zeros)
                .init(device);
            let head = LinearConfig::new(hidden_size, output_size)
                .with_bias(true)
                .init(device);
            Self { lstm, head }
        }

        fn try_get_builtin_weights(&self, weight_name: &str) -> Option<WeightConfig> {
            match weight_name {
                "nh_AORC_hourly_25yr_1210_112435_7" => WeightConfig::nh_AORC_hourly_25yr_1210_112435_7(),
                "nh_AORC_hourly_25yr_1210_112435_8" => WeightConfig::nh_AORC_hourly_25yr_1210_112435_8(),
                "nh_AORC_hourly_25yr_1210_112435_9" => WeightConfig::nh_AORC_hourly_25yr_1210_112435_9(),
                "nh_AORC_hourly_25yr_seq999_seed101_0701_143442" => WeightConfig::nh_AORC_hourly_25yr_seq999_seed101_0701_143442(),
                "nh_AORC_hourly_25yr_seq999_seed103_2701_171540" => WeightConfig::nh_AORC_hourly_25yr_seq999_seed103_2701_171540(),
                "nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806" => WeightConfig::nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806(),
                _ => None,
            }
        }

        fn load_weight_config(&mut self, device: &B::Device, weight_config: WeightConfig) {
            let input_size = weight_config.input_size;
            let hidden_size = weight_config.hidden_size;

            self.lstm.input_gate = create_gate_controller(
                &weight_config.input_gate_input_weights,
                &weight_config.input_gate_input_biases,
                &weight_config.input_gate_hidden_weights,
                &weight_config.input_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );

            self.lstm.forget_gate = create_gate_controller(
                &weight_config.forget_gate_input_weights,
                &weight_config.forget_gate_input_biases,
                &weight_config.forget_gate_hidden_weights,
                &weight_config.forget_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );

            self.lstm.cell_gate = create_gate_controller(
                &weight_config.cell_gate_input_weights,
                &weight_config.cell_gate_input_biases,
                &weight_config.cell_gate_hidden_weights,
                &weight_config.cell_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );

            self.lstm.output_gate = create_gate_controller(
                &weight_config.output_gate_input_weights,
                &weight_config.output_gate_input_biases,
                &weight_config.output_gate_hidden_weights,
                &weight_config.output_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );
        }

        pub fn load_json_weights(&mut self, device: &B::Device, weight_path: &str) {
            // rustify_test section 0 start. 
            // experimental functionality for optimizing weight loading.
            let weight_dirname: &str = weight_path
                .split('/')
                .rev()
                .nth(2)
                .unwrap_or_else(|| panic!("Failed to parse weight path: {}", weight_path));
            // Using builtin_weights: 10.0 seconds
            let builtin_weights: Option<WeightConfig> = self.try_get_builtin_weights(weight_dirname);
            if let Some(weight_config) = builtin_weights && USE_HARD_CODED_WEIGHTS {
                // println!("Using built-in weights for {}", weight_dirname);
                self.load_weight_config(device, weight_config);
                return;
            }
            // let target_path: String = format!("./rustify_test/{}.rs", weight_dirname);
            // let need_conversion = !Path::new(&target_path).exists();
            // let need_conversion: bool = !fs::exists(target_path.clone()).unwrap();
            // let mut conversion_string: String = "use crate::rustify_test::example_weights::WeightConfig;\n\n".to_string();
            // let first_indent = "    ";
            // let second_indent = "        ";
            // if need_conversion {
            //     println!("Doing conversion for {}", weight_dirname);
            //     fs::create_dir_all("./rustify_test").unwrap();
            //     // fs::write(
            //     //     &target_path,
            //     //     "".to_string()
            //     //     )
            //     // .expect("Failed to write Rust file");
            //     // implement the static weights as an impl for WeightConfig that returns a constructed WeightConfig struct with all the weights hardcoded in. This way we can load weights without parsing JSON in the future.
            //     // conversion_string.push_str(&format!("impl WeightConfig {{\n    pub fn {}() -> Self {{\n        Self {{\n", weight_dirname));
            //     conversion_string.push_str(&format!("impl WeightConfig {{\n{}pub fn {}() -> Self {{\n{}Self {{\n", first_indent, weight_dirname, second_indent));
            // }
            // rustify_test section 0 end.
            let json_str = fs::read_to_string(weight_path).expect("Failed to read file");
            let weights: Value = serde_json::from_str(&json_str).unwrap();

            fn to_vec(value: &Value) -> Vec<f32> {
                value
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            }

            let input_size = weights["input_size"].as_u64().unwrap() as usize;
            let output_size = weights["output_size"].as_u64().unwrap() as usize;
            let hidden_size = weights["hidden_size"].as_u64().unwrap() as usize;

            // if need_conversion {
            //     conversion_string.push_str(&format!("{}input_size: {},\n", second_indent, input_size));
            //     conversion_string.push_str(&format!("{}hidden_size: {},\n", second_indent, hidden_size));
            //     conversion_string.push_str(&format!("{}output_size: {},\n", second_indent, output_size));
            // }

            // Load all gate weights
            let input_gate_input_weights =
                to_vec(&weights["lstm.input_gate.input_transform.weight"]);
            let input_gate_input_biases = to_vec(&weights["lstm.input_gate.input_transform.bias"]);
            let input_gate_hidden_weights =
                to_vec(&weights["lstm.input_gate.hidden_transform.weight"]);
            let input_gate_hidden_biases =
                to_vec(&weights["lstm.input_gate.hidden_transform.bias"]);

            // if need_conversion {
            //     // There are megabytes of weights, so we *do not* want individual values per line.
            //     conversion_string.push_str(&format!("{}input_gate_input_weights: vec!{:?},\n", second_indent, input_gate_input_weights));
            //     conversion_string.push_str(&format!("{}input_gate_input_biases: vec!{:?},\n", second_indent, input_gate_input_biases));
            //     conversion_string.push_str(&format!("{}input_gate_hidden_weights: vec!{:?},\n", second_indent, input_gate_hidden_weights));
            //     conversion_string.push_str(&format!("{}input_gate_hidden_biases: vec!{:?},\n", second_indent, input_gate_hidden_biases));
            // }

            self.lstm.input_gate = create_gate_controller(
                &input_gate_input_weights,
                &input_gate_input_biases,
                &input_gate_hidden_weights,
                &input_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );

            let forget_gate_input_weights =
                to_vec(&weights["lstm.forget_gate.input_transform.weight"]);
            let forget_gate_input_biases =
                to_vec(&weights["lstm.forget_gate.input_transform.bias"]);
            let forget_gate_hidden_weights =
                to_vec(&weights["lstm.forget_gate.hidden_transform.weight"]);
            let forget_gate_hidden_biases =
                to_vec(&weights["lstm.forget_gate.hidden_transform.bias"]);

            // if need_conversion {
            //     conversion_string.push_str(&format!("{}forget_gate_input_weights: vec!{:?},\n", second_indent, forget_gate_input_weights));
            //     conversion_string.push_str(&format!("{}forget_gate_input_biases: vec!{:?},\n", second_indent, forget_gate_input_biases));
            //     conversion_string.push_str(&format!("{}forget_gate_hidden_weights: vec!{:?},\n", second_indent, forget_gate_hidden_weights));
            //     conversion_string.push_str(&format!("{}forget_gate_hidden_biases: vec!{:?},\n", second_indent, forget_gate_hidden_biases));
            // }

            self.lstm.forget_gate = create_gate_controller(
                &forget_gate_input_weights,
                &forget_gate_input_biases,
                &forget_gate_hidden_weights,
                &forget_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );

            let cell_gate_input_weights = to_vec(&weights["lstm.cell_gate.input_transform.weight"]);
            let cell_gate_input_biases = to_vec(&weights["lstm.cell_gate.input_transform.bias"]);
            let cell_gate_hidden_weights =
                to_vec(&weights["lstm.cell_gate.hidden_transform.weight"]);
            let cell_gate_hidden_biases = to_vec(&weights["lstm.cell_gate.hidden_transform.bias"]);

            // if need_conversion {
            //     conversion_string.push_str(&format!("{}cell_gate_input_weights: vec!{:?},\n", second_indent, cell_gate_input_weights));
            //     conversion_string.push_str(&format!("{}cell_gate_input_biases: vec!{:?},\n", second_indent, cell_gate_input_biases));
            //     conversion_string.push_str(&format!("{}cell_gate_hidden_weights: vec!{:?},\n", second_indent, cell_gate_hidden_weights));
            //     conversion_string.push_str(&format!("{}cell_gate_hidden_biases: vec!{:?},\n", second_indent, cell_gate_hidden_biases));
            // }

            self.lstm.cell_gate = create_gate_controller(
                &cell_gate_input_weights,
                &cell_gate_input_biases,
                &cell_gate_hidden_weights,
                &cell_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );

            let output_gate_input_weights =
                to_vec(&weights["lstm.output_gate.input_transform.weight"]);
            let output_gate_input_biases =
                to_vec(&weights["lstm.output_gate.input_transform.bias"]);
            let output_gate_hidden_weights =
                to_vec(&weights["lstm.output_gate.hidden_transform.weight"]);
            let output_gate_hidden_biases =
                to_vec(&weights["lstm.output_gate.hidden_transform.bias"]);

            // if need_conversion {
            //     conversion_string.push_str(&format!("{}output_gate_input_weights: vec!{:?},\n", second_indent, output_gate_input_weights));
            //     conversion_string.push_str(&format!("{}output_gate_input_biases: vec!{:?},\n", second_indent, output_gate_input_biases));
            //     conversion_string.push_str(&format!("{}output_gate_hidden_weights: vec!{:?},\n", second_indent, output_gate_hidden_weights));
            //     conversion_string.push_str(&format!("{}output_gate_hidden_biases: vec!{:?},\n", second_indent, output_gate_hidden_biases));
            // }

            self.lstm.output_gate = create_gate_controller(
                &output_gate_input_weights,
                &output_gate_input_biases,
                &output_gate_hidden_weights,
                &output_gate_hidden_biases,
                device,
                input_size,
                hidden_size,
            );

            // if need_conversion {
            //     conversion_string.push_str(&format!("    }}\n}}\n"));
            //     fs::write(
            //         &target_path,
            //         conversion_string,
            //      )
            //     .expect("Failed to write Rust file");
            // }
        }

        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            state: Option<LstmState<B, 2>>,
        ) -> (Tensor<B, 3>, LstmState<B, 2>) {
            let (output, state) = self.lstm.forward(input, state);
            let [batch_size, seq_length, hidden_size] = output.dims();
            let output_reshaped = output.reshape([batch_size * seq_length, hidden_size]);
            let prediction = self.head.forward(output_reshaped);
            let prediction = prediction.reshape([batch_size, seq_length, 1]);
            (prediction, state)
        }
    }
}

use lstm_model::{NextgenLstm, vec_to_tensor};

#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingScalars {
    input_mean: Vec<f32>,
    input_std: Vec<f32>,
    output_mean: f32,
    output_std: f32,
}

struct ModelInstance<B: Backend> {
    model: NextgenLstm<B>,
    metadata: ModelMetadata,
    scalars: TrainingScalars,
    lstm_state: Option<LstmState<B, 2>>,
}

pub struct LstmFlowGenerator<B: Backend> {
    root_dir: PathBuf,
    forcing_file: netcdf::File,
    divide_index: HashMap<String, usize>,
    total_timesteps: usize,
    device: B::Device,
}

impl<B: Backend> LstmFlowGenerator<B> {
    pub fn new(root_dir: PathBuf) -> Result<Self> {
        // Open forcing file
        // let forcing_path = root_dir.join("forcings").join("forcings.nc");
        let forcing_path = find_forcing_file(&root_dir)
            .with_context(|| format!("Failed to find forcing file in directory: {:?}", root_dir))?;
        let forcing_file = open(&forcing_path)
            .with_context(|| format!("Failed to open forcing file: {:?}", forcing_path))?;

        // Load divide index mapping
        let divide_index_var = forcing_file
            .variable("ids")
            .context("Couldn't find variable 'ids'")?;

        let mut divide_index = HashMap::new();
        for i in 0..divide_index_var.len() {
            let id = divide_index_var
                .get_string(i)
                .with_context(|| format!("Failed to get string at index {}", i))?;
            divide_index.insert(id, i);
        }

        // Get total timesteps from forcing file
        let total_timesteps = forcing_file
            .dimension_len("time")
            .context("Couldn't find variable 'Time'")?;

        Ok(Self {
            root_dir,
            forcing_file,
            divide_index,
            total_timesteps,
            device: Default::default(),
        })
    }

    fn load_single_model(&self, training_config_path: &Path) -> Result<ModelInstance<B>> {
        let training_config = fs::read_to_string(training_config_path)?;
        let training_config: serde_yaml::Value = serde_yaml::from_str(&training_config)?;

        // Find model path
        let model_dir = training_config["run_dir"]
            .as_str()
            .ok_or(anyhow::anyhow!("Missing run_dir"))?
            .replace(
                "..",
                training_config_path
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .to_str()
                    .unwrap(),
            );

        let model_path = glob(&format!("{}/model_*.pt", model_dir))?
            .next()
            .ok_or(anyhow::anyhow!("No model file found"))??;

        // Convert weights if needed
        let model_folder = model_path.parent().unwrap();
        let burn_dir = model_folder.join("burn");
        let converted_path = burn_dir.join(model_path.file_name().unwrap());
        let lock_file_path = burn_dir.join(".conversion.lock");

        // Create burn directory if it doesn't exist
        if !burn_dir.exists() {
            fs::create_dir_all(&burn_dir)?;
        }

        // Check if conversion is needed
        let needs_conversion = !converted_path.exists()
            || !converted_path.with_extension("json").exists()
            || !burn_dir.join("train_data_scaler.json").exists()
            || !burn_dir.join("weights.json").exists();

        if needs_conversion {
            // Try to acquire lock
            let mut lock_acquired = false;
            let process_id = std::process::id();

            loop {
                // Try to create lock file atomically
                match fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&lock_file_path)
                {
                    Ok(mut file) => {
                        // Write process ID to lock file for debugging
                        use std::io::Write;
                        writeln!(file, "Locked by process {}", process_id)?;
                        lock_acquired = true;
                        break;
                    }
                    Err(_) => {
                        // Lock file exists, another process is converting

                        std::thread::sleep(std::time::Duration::from_millis(1000));

                        // Check if conversion is complete
                        if converted_path.exists()
                            && converted_path.with_extension("json").exists()
                            && burn_dir.join("train_data_scaler.json").exists()
                            && burn_dir.join("weights.json").exists()
                        {
                            // println!("Process {} found completed conversion", process_id);
                            break;
                        }
                    }
                }
            }

            // If we acquired the lock, do the conversion
            if lock_acquired {
                println!(
                    "Process {} converting PyTorch weights to Burn format for model: {}",
                    process_id,
                    model_path.display()
                );

                // Perform conversion
                match self.convert_model_weights(&model_path, &training_config_path) {
                    Ok(_) => {
                        println!("Process {} completed model conversion", process_id);
                    }
                    Err(e) => {
                        // Clean up lock file on error
                        fs::remove_file(&lock_file_path)?;
                        return Err(e);
                    }
                }

                // Remove lock file after successful conversion
                fs::remove_file(&lock_file_path)?;
                println!("Process {} released conversion lock", process_id);
            }
        } else {
            // println!("Model already converted, skipping conversion");
        }

        // Load metadata
        let metadata_str = fs::read_to_string(converted_path.with_extension("json"))?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_str)?;

        // Load scalars
        let scalars_str = fs::read_to_string(burn_dir.join("train_data_scaler.json"))?;
        let scalars: TrainingScalars = serde_json::from_str(&scalars_str)?;

        // Load model
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(converted_path.into(), &self.device)?;

        let mut model = NextgenLstm::init(
            &self.device,
            metadata.input_size,
            metadata.hidden_size,
            metadata.output_size,
        );
        model = model.load_record(record);
        model.load_json_weights(
            &self.device,
            burn_dir.join("weights.json").to_str().unwrap(),
        );

        Ok(ModelInstance {
            model,
            metadata,
            scalars,
            lstm_state: None,
        })
    }

    fn convert_model_weights(&self, weights_path: &Path, config_path: &Path) -> Result<()> {
        // Use the Python conversion script
        println!(
            "Converting PyTorch weights to Burn format for: {}",
            weights_path.display()
        );
        let script = include_str!("../python_convert.py");
        let script_path = std::env::temp_dir().join("convert_weights.py");
        fs::write(&script_path, script)?;

        // The Python script content would be the same as in python.rs
        let output = Command::new("uv")
            .arg("run")
            .arg("-p")
            .arg("3.9")
            .arg("--with")
            .arg("pyyaml")
            .arg("--with")
            .arg("numpy")
            .arg("--with")
            .arg("torch")
            .arg("--extra-index-url")
            .arg("https://download.pytorch.org/whl/cpu")
            .arg(&script_path)
            .arg(&weights_path)
            .arg(&config_path)
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "Weight conversion failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
    }

    pub fn generate_flows_for_node(
        &self,
        node_id: u32,
        area_sqkm: f32,
        max_timesteps: usize,
    ) -> Result<VecDeque<f32>> {
        // Construct config path
        // let config_path = self
        //     .root_dir
        //     .join("config")
        //     .join("cat_config")
        //     .join("lstm")
        //     .join(format!("cat-{}.yml", node_id));
        let lstm_config_dir = find_lstm_config_directory(&self.root_dir)?;
            // .with_context(|| format!("Failed to find LSTM config directory in {:?}", self.root_dir))?;
        let config_path = lstm_config_dir.join(format!("cat-{}.yml", node_id));

        if !config_path.exists() {
            return Err(anyhow::anyhow!(
                "LSTM config not found for node {}: {:?}",
                node_id,
                config_path
            ));
        }

        // Load config
        let config_str = fs::read_to_string(&config_path)?;
        let config: serde_yaml::Value = serde_yaml::from_str(&config_str)?;

        // Get node index in forcing file
        let node_key = format!("cat-{}", node_id);
        let current_index = *self
            .divide_index
            .get(&node_key)
            .with_context(|| format!("Node {} not found in forcing file", node_key))?;

        // Load all models in ensemble
        let training_configs = config["train_cfg_file"]
            .as_sequence()
            .ok_or(anyhow::anyhow!("train_cfg_file should be an array"))?;

        let mut models = Vec::new();
        for config_value in training_configs {
            let training_config_path = Path::new(
                config_value
                    .as_str()
                    .ok_or(anyhow::anyhow!("train_cfg_file entry not a string"))?,
            );
            models.push(self.load_single_model(training_config_path)?);
        }

        // Create a map of all available forcing values
        let mut forcing_map: HashMap<String, Vec<f32>> = HashMap::new();

        // Load forcing variables
        let forcing_timesteps = self.total_timesteps;
        let time_range = 0..forcing_timesteps;

        // Load all forcing data into the map using internal names
        forcing_map.insert(
            "APCP_surface".to_string(),
            self.forcing_file
                .variable("APCP_surface")
                .ok_or(anyhow::anyhow!("Unable to load APCP_surface"))?
                .get_values((current_index, time_range.clone()))?,
        );
        forcing_map.insert(
            "TMP_2maboveground".to_string(),
            self.forcing_file
                .variable("TMP_2maboveground")
                .ok_or(anyhow::anyhow!("Unable to load TMP_2maboveground"))?
                .get_values((current_index, time_range.clone()))?,
        );
        forcing_map.insert(
            "DLWRF_surface".to_string(),
            self.forcing_file
                .variable("DLWRF_surface")
                .ok_or(anyhow::anyhow!("Unable to load DLWRF_surface"))?
                .get_values((current_index, time_range.clone()))?,
        );
        forcing_map.insert(
            "PRES_surface".to_string(),
            self.forcing_file
                .variable("PRES_surface")
                .ok_or(anyhow::anyhow!("Unable to load PRES_surface"))?
                .get_values((current_index, time_range.clone()))?,
        );
        forcing_map.insert(
            "SPFH_2maboveground".to_string(),
            self.forcing_file
                .variable("SPFH_2maboveground")
                .ok_or(anyhow::anyhow!("Unable to load SPFH_2maboveground"))?
                .get_values((current_index, time_range.clone()))?,
        );
        forcing_map.insert(
            "DSWRF_surface".to_string(),
            self.forcing_file
                .variable("DSWRF_surface")
                .ok_or(anyhow::anyhow!("Unable to load DSWRF_surface"))?
                .get_values((current_index, time_range.clone()))?,
        );
        forcing_map.insert(
            "UGRD_10maboveground".to_string(),
            self.forcing_file
                .variable("UGRD_10maboveground")
                .ok_or(anyhow::anyhow!("Unable to load UGRD_10maboveground"))?
                .get_values((current_index, time_range.clone()))?,
        );
        forcing_map.insert(
            "VGRD_10maboveground".to_string(),
            self.forcing_file
                .variable("VGRD_10maboveground")
                .ok_or(anyhow::anyhow!("Unable to load VGRD_10maboveground"))?
                .get_values((current_index, time_range))?,
        );

        // Build a map of static attributes from config (any key that parses as f64)
        let mut static_attributes: HashMap<String, f32> = HashMap::new();
        if let Some(config_map) = config.as_mapping() {
            for (key, value) in config_map {
                if let Some(key_str) = key.as_str() {
                    if let Some(val) = value.as_f64() {
                        static_attributes.insert(key_str.to_string(), val as f32);
                    }
                }
            }
        }

        // Add static attributes to forcing_map (expanded to all timesteps)
        for (attr_name, attr_value) in &static_attributes {
            forcing_map.insert(attr_name.clone(), vec![*attr_value; forcing_timesteps]);
        }

        // Check for missing inputs before the main loop and warn once per missing input
        for model_instance in &models {
            for input_name in &model_instance.metadata.input_names {
                if !forcing_map.contains_key(input_name) {
                    eprintln!(
                        "Warning: Missing input '{}' for model, will use 0.0",
                        input_name
                    );
                }
            }
        }

        // Calculate output scale factor
        let output_scale_factor_cms =
            (1.0 / 1000.0) * (area_sqkm * 1000.0 * 1000.0) * (1.0 / 3600.0);

        // Now run each timestep
        let mut all_flows = VecDeque::new();
        for time_idx in 0..forcing_timesteps {
            // Run ensemble
            let mut ensemble_outputs = Vec::new();
            for model_instance in &mut models {
                // Get the model's expected input order from metadata
                let input_names = &model_instance.metadata.input_names;

                // Gather inputs in the correct order that the model expects
                let mut forcing_values = Vec::new();
                for input_name in input_names {
                    // Get the value for this timestep from our forcing map
                    let value = forcing_map
                        .get(input_name)
                        .and_then(|values| values.get(time_idx))
                        .copied()
                        .unwrap_or(0.0);
                    forcing_values.push(value);
                }

                let output = self.run_single_model(model_instance, &forcing_values)?;
                ensemble_outputs.push(output);
            }

            // Average ensemble outputs
            let mean_surface_runoff_mm = if !ensemble_outputs.is_empty() {
                ensemble_outputs.iter().sum::<f32>() / ensemble_outputs.len() as f32
            } else {
                0.0
            };

            // Convert to m3/s
            let surface_runoff_volume_m3_s = mean_surface_runoff_mm * output_scale_factor_cms;
            all_flows.push_back(surface_runoff_volume_m3_s);
        }

        // Pad if needed
        if all_flows.len() < max_timesteps {
            let last_flow = all_flows.back().copied().unwrap_or(0.0);
            while all_flows.len() < max_timesteps {
                all_flows.push_back(last_flow);
            }
        }

        Ok(all_flows)
    }

    fn run_single_model(
        &self,
        model_instance: &mut ModelInstance<B>,
        inputs: &[f32],
    ) -> Result<f32> {
        // Scale inputs
        let scaled_inputs: Vec<f32> = inputs
            .iter()
            .zip(&model_instance.scalars.input_mean)
            .zip(&model_instance.scalars.input_std)
            .map(
                |((val, mean), std)| {
                    if *std != 0.0 { (val - mean) / std } else { 0.0 }
                },
            )
            .collect();

        // Create input tensor
        let input_tensor_data = vec_to_tensor(
            &scaled_inputs,
            vec![1, 1, model_instance.metadata.input_size],
        );
        let input_tensor = Tensor::from_data(input_tensor_data, &self.device);

        // Forward pass
        let (output, new_state) = model_instance
            .model
            .forward(input_tensor, model_instance.lstm_state.take());
        model_instance.lstm_state = Some(new_state);

        // Process output
        let output_vec: Vec<f32> = output.into_data().to_vec().unwrap();
        let output_value = output_vec[0];

        // Denormalize
        let surface_runoff_mm = (output_value * model_instance.scalars.output_std
            + model_instance.scalars.output_mean)
            .max(0.0);

        Ok(surface_runoff_mm)
    }
}
