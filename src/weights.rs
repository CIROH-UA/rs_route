use hardcode_weights::hardcode_weights;

#[allow(dead_code)] // suppress warning about the unused output_size field, since it might get used later
pub struct WeightConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub input_gate_input_weights: Vec<f32>,
    pub input_gate_input_biases: Vec<f32>,
    pub input_gate_hidden_weights: Vec<f32>,
    pub input_gate_hidden_biases: Vec<f32>,
    pub forget_gate_input_weights: Vec<f32>,
    pub forget_gate_input_biases: Vec<f32>,
    pub forget_gate_hidden_weights: Vec<f32>,
    pub forget_gate_hidden_biases: Vec<f32>,
    pub cell_gate_input_weights: Vec<f32>,
    pub cell_gate_input_biases: Vec<f32>,
    pub cell_gate_hidden_weights: Vec<f32>,
    pub cell_gate_hidden_biases: Vec<f32>,
    pub output_gate_input_weights: Vec<f32>,
    pub output_gate_input_biases: Vec<f32>,
    pub output_gate_hidden_weights: Vec<f32>,
    pub output_gate_hidden_biases: Vec<f32>,
}

impl WeightConfig {
    #[hardcode_weights("/ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_25yr_1210_112435_7/burn/weights.json")]
    pub fn nh_AORC_hourly_25yr_1210_112435_7() -> Option<Self> {}
    #[hardcode_weights("/ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_25yr_1210_112435_8/burn/weights.json")]
    pub fn nh_AORC_hourly_25yr_1210_112435_8() -> Option<Self> {}
    #[hardcode_weights("/ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_25yr_1210_112435_9/burn/weights.json")]
    pub fn nh_AORC_hourly_25yr_1210_112435_9() -> Option<Self> {}
    #[hardcode_weights("/ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_25yr_seq999_seed101_0701_143442/burn/weights.json")]
    pub fn nh_AORC_hourly_25yr_seq999_seed101_0701_143442() -> Option<Self> {}
    #[hardcode_weights("/ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_25yr_seq999_seed103_2701_171540/burn/weights.json")]
    pub fn nh_AORC_hourly_25yr_seq999_seed103_2701_171540() -> Option<Self> {}
    #[hardcode_weights("/ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806/burn/weights.json")]
    pub fn nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806() -> Option<Self> {}
}