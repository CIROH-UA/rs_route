use crate::io::results::SimulationResults;
use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use netcdf::{self, FileMut};
use std::path::PathBuf;

// Silence HDF5 diagnostic output (e.g. "unable to determine if file is accessible")
// that occurs when netcdf::create checks for an existing file.
unsafe extern "C" {
    fn H5Eset_auto2(
        estack_id: i64,
        func: Option<unsafe extern "C" fn() -> i32>,
        client_data: *mut std::ffi::c_void,
    ) -> i32;
}

// (name, long_name, units) for each per-feature, per-timestep output variable
const DATA_VARIABLES: [(&str, &str, &str); 3] = [
    ("flow", "Flow", "m3 s-1"),
    ("velocity", "Velocity", "m/s"),
    ("depth", "Depth", "m"),
];

// NetCDF output file that tracks its own append position, so callers can
// write batches of any size without managing feature offsets.
pub struct NetCdfWriter {
    file: FileMut,
    next_feature_idx: usize,
}

pub fn init_netcdf_output(
    output_dir: PathBuf,
    filename: &str,
    num_flowpaths: usize,
    timesteps: Vec<f64>,
    reference_time: &NaiveDateTime,
) -> Result<NetCdfWriter> {
    // Suppress HDF5 diagnostic messages during file creation
    unsafe {
        H5Eset_auto2(0, None, std::ptr::null_mut());
    }

    // Create NetCDF file
    let mut file = netcdf::create(output_dir.join(filename))
        .with_context(|| format!("Failed to create NetCDF file: {}", filename))?;

    // Add dimensions
    file.add_dimension("feature_id", num_flowpaths)
        .context("Failed to add feature_id dimension")?;
    file.add_dimension("time", timesteps.len())
        .context("Failed to add time dimension")?;

    // Add variables
    // Time variable
    let mut time_var = file
        .add_variable::<f64>("time", &["time"])
        .context("Failed to add time variable")?;
    time_var.put_attribute("_FillValue", -9999.0)?;
    time_var.put_attribute("long_name", "valid output time")?;
    time_var.put_attribute("standard_name", "time")?;
    time_var.put_attribute(
        "units",
        format!(
            "seconds since {}",
            reference_time.format("%Y-%m-%d %H:%M:%S")
        ),
    )?;
    time_var.put_attribute("missing_value", -9999.0)?;
    time_var
        .put_values(&timesteps, ..)
        .context("Failed to write time values")?;

    // Feature ID variable
    let mut feature_var = file
        .add_variable::<i64>("feature_id", &["feature_id"])
        .context("Failed to add feature_id variable")?;
    feature_var.put_attribute("long_name", "Segment ID")?;

    // Flow, velocity and depth variables
    for (name, long_name, units) in DATA_VARIABLES {
        let mut var = file
            .add_variable::<f32>(name, &["feature_id", "time"])
            .with_context(|| format!("Failed to add {} variable", name))?;
        var.put_attribute("_FillValue", -9999.0f32)?;
        var.put_attribute("long_name", long_name)?;
        var.put_attribute("units", units)?;
        var.put_attribute("missing_value", -9999.0f32)?;
    }

    // Global attributes
    file.add_attribute("TITLE", "OUTPUT FROM RS-ROUTE")?;
    file.add_attribute(
        "file_reference_time",
        reference_time.format("%Y-%m-%d_%H:%M:%S").to_string(),
    )?;
    file.add_attribute("code_version", "")?;

    // Additional expected variables
    let _ = file.add_variable::<f32>("type", &["feature_id"])?;
    let _ = file.add_variable::<f32>("nudge", &["feature_id"])?;

    Ok(NetCdfWriter {
        file,
        next_feature_idx: 0,
    })
}

impl NetCdfWriter {
    // Append a batch of (already downsampled) results after the last written feature
    pub fn append(&mut self, batch: &[SimulationResults]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let start_idx = self.next_feature_idx;
        let end_idx = start_idx + batch.len();

        let feature_ids: Vec<u32> = batch.iter().map(|r| r.feature_id).collect();
        let mut feature_var = self
            .file
            .variable_mut("feature_id")
            .ok_or_else(|| anyhow::anyhow!("feature_id variable not found"))?;
        feature_var
            .put_values(&feature_ids, start_idx..end_idx)
            .context("Failed to write feature_ids")?;

        // Concatenating per-feature rows gives row-major (feature, time) order,
        // matching the variable layout, so each can be written in a single call.
        let fields: [(&str, fn(&SimulationResults) -> &[f32]); 3] = [
            ("flow", |r| &r.flow_data),
            ("velocity", |r| &r.velocity_data),
            ("depth", |r| &r.depth_data),
        ];
        for (name, field) in fields {
            let data: Vec<f32> = batch.iter().flat_map(|r| field(r).iter().copied()).collect();
            let mut var = self
                .file
                .variable_mut(name)
                .ok_or_else(|| anyhow::anyhow!("{} variable not found", name))?;
            var.put_values(&data, (start_idx..end_idx, ..))
                .with_context(|| format!("Failed to write {} data", name))?;
        }

        self.next_feature_idx = end_idx;
        Ok(())
    }
}
