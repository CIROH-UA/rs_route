use anyhow::{Context, Result};
use chrono::{Duration, NaiveDateTime};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::sync::Arc;

mod cli;
mod config;
mod io;

mod network;
mod routing;
mod state;
pub mod kernel {
    pub mod muskingum;
}

use cli::get_args;
use config::{ChannelParams, ColumnConfig, OutputFormat};
use io::netcdf::init_netcdf_output;
use network::build_network_topology;
use routing::process_routing_parallel;

static OUTPUT_TYPE: &str = "NetCDF"; // or "CSV" or "Both"

fn main() -> Result<()> {
    // Configuration
    //let (_, csv_dir, db_path, internal_timestep_seconds, output_dir)
    let config: cli::Config = get_args()?;
    let dt: f32 = config.internal_timestep_seconds as f32;
    let db_path: std::path::PathBuf = config.gpkg_file;
    let csv_dir: std::path::PathBuf = config.csv_dir;
    let _config_dir: std::path::PathBuf = config.config_dir;
    // let output_format: OutputFormat = OutputFormat::NetCdf;
    let output_format: OutputFormat = match OUTPUT_TYPE {
        "CSV" => OutputFormat::Csv,
        "NetCDF" => OutputFormat::NetCdf,
        "Both" => OutputFormat::Both,
        _ => return Err(anyhow::anyhow!("Invalid output type: {}", OUTPUT_TYPE)),
    };

    // Initialize SQLite connection
    let conn = rusqlite::Connection::open(&db_path)
        .with_context(|| format!("Failed to open database: {:?}", db_path))?;

    let column_config = ColumnConfig::new();

    // Build network topology
    println!("Building network topology...");
    let topology = build_network_topology(&conn, &column_config, &csv_dir)?;

    // Load channel parameters
    println!("Loading channel parameters...");
    let channel_params_map = network::load_channel_parameters(&conn, &topology, &column_config)?;

    // Set up CSV output if needed
    let csv_writer = if matches!(output_format, OutputFormat::Csv | OutputFormat::Both) {
        Some(io::csv::create_csv_writer("network_routing_results.csv")?)
    } else {
        None
    };

    // Get simulation parameters
    let (max_external_steps, reference_time) =
        get_simulation_params(&csv_dir, &channel_params_map)?;

    let start_time = reference_time;
    let end_time = start_time + Duration::seconds((3600 * max_external_steps) as i64);

    let external_timestep_seconds = 3600;
    let total_timesteps =
        (max_external_steps) * (external_timestep_seconds / config.internal_timestep_seconds);

    println!("\nSimulation Configuration:");
    println!("  Period: {} to {}", start_time, end_time);
    println!(
        "  Internal timestep: {} seconds",
        config.internal_timestep_seconds
    );
    println!("  Network nodes: {}", topology.routing_order.len());
    println!("  Total timesteps: {}", total_timesteps);

    // Initialize NetCDF output
    // skip the 0th timestep
    let timesteps: Vec<f64> = (1..=max_external_steps)
        .map(|step| (step * external_timestep_seconds) as f64)
        .collect();

    let nc_filename = format!("troute_output_{}.nc", reference_time.format("%Y%m%d%H%M"));
    let netcdf_writer = init_netcdf_output(
        config.output_dir,
        &nc_filename,
        topology.routing_order.len(),
        timesteps,
        &reference_time,
    )?;

    // Create progress bar
    let pb = ProgressBar::new(topology.routing_order.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} nodes ({eta})")?
            .progress_chars("#>-")
    );

    // Run parallel routing
    println!("\nStarting parallel wave-front routing...");
    process_routing_parallel(
        config.kernel,
        &topology,
        &channel_params_map,
        total_timesteps,
        dt,
        netcdf_writer,
        Arc::new(pb),
    )?;

    // Final flush for CSV
    if let Some(mut wtr) = csv_writer {
        wtr.flush().context("Failed to flush CSV writer")?;
        println!("CSV results saved to network_routing_results.csv");
    }

    println!(
        "\nNetwork routing complete. Output saved to {}",
        nc_filename
    );
    Ok(())
}

fn get_simulation_params(
    csv_dir: &std::path::PathBuf,
    features: &HashMap<u32, ChannelParams>,
) -> Result<(usize, NaiveDateTime)> {
    let first_id = features
        .keys()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No features found"))?;

    let file_name = csv_dir.join(format!("cat-{}.csv", first_id));
    let content = std::fs::read_to_string(&file_name)
        .with_context(|| format!("Failed to read file: {:?}", file_name))?;

    if content.lines().count() == 0 {
        return Err(anyhow::anyhow!("CSV file is empty: {:?}", file_name))
            .with_context(|| format!("Failed to read CSV file: {:?}", file_name));
    } else if content.lines().count() == 1 {
        return Err(anyhow::anyhow!(
            "CSV file only contains header: {:?}",
            file_name
        ))
        .with_context(|| format!("Failed to read CSV file: {:?}", file_name));
    }

    let max_external_steps = content.lines().count().saturating_sub(2);

    let line = content
        .lines()
        .nth(1)
        .with_context(|| format!("Failed to read second line of CSV file: {:?}", file_name))?;
    let time = line
        .split(',')
        .nth(1)
        .with_context(|| format!("Failed to parse time from CSV line: {:?}", line))?;

    let reference_time = NaiveDateTime::parse_from_str(time, "%Y-%m-%d %H:%M:%S")
        .context("Failed to parse reference time")?;

    Ok((max_external_steps, reference_time))
}
