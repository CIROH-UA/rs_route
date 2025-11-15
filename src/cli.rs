use anyhow::{Context, Result};
use clap::{Parser, command};
use std::path::PathBuf;

use crate::kernel::muskingum::MuskingumCungeKernel;

/// Network routing simulation tool
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Route directory path
    route_dir: PathBuf,

    /// Internal timestep in seconds
    #[arg(short, long, default_value_t = 300)]
    internal_timestep_seconds: usize,
    #[arg(short, long, default_value_t = MuskingumCungeKernel::TRouteModernized)]
    kernel: MuskingumCungeKernel,
    /// Use LSTM for flow generation instead of CSV files
    #[arg(short = 'l', long, default_value_t = false)]
    use_lstm: bool,
}

pub struct Config {
    pub config_dir: PathBuf,
    pub csv_dir: PathBuf,
    pub gpkg_file: PathBuf,
    pub internal_timestep_seconds: usize,
    pub output_dir: PathBuf,
    pub kernel: MuskingumCungeKernel,
    pub use_lstm: bool,
}

pub fn get_args() -> Result<Config> {
    let args = Args::parse();

    let root_dir = args.route_dir;
    let csv_dir = root_dir.join("outputs").join("ngen");
    let config_dir = root_dir.join("config");
    let output_dir = root_dir.join("outputs").join("troute");

    // Find the .gpkg file in the config directory
    let gpkg_file = config_dir
        .read_dir()
        .context("Failed to read config directory")?
        .filter_map(Result::ok)
        .find(|entry| entry.path().extension().map_or(false, |ext| ext == "gpkg"))
        .ok_or_else(|| anyhow::anyhow!("No .gpkg file found in config directory"))?
        .path();

    Ok(Config {
        config_dir,
        csv_dir,
        gpkg_file,
        internal_timestep_seconds: args.internal_timestep_seconds,
        output_dir,
        kernel: args.kernel,
        use_lstm: args.use_lstm,
    })
}
