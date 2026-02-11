use anyhow::{Context, Result};
use clap::Parser;
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
}

pub struct Config {
    pub config_dir: PathBuf,
    pub csv_dir: PathBuf,
    pub gpkg_file: PathBuf,
    pub internal_timestep_seconds: usize,
    pub output_dir: PathBuf,
    pub kernel: MuskingumCungeKernel,
}

pub fn get_args() -> Result<Config> {
    let args = Args::parse();

    let root_dir = args.route_dir;
    let csv_dir = root_dir.join("outputs").join("ngen");
    let config_dir = root_dir.join("config");
    let output_dir = root_dir.join("outputs").join("troute");

    // Check directories valid
    if !root_dir.exists() || !root_dir.is_dir() {
        return Err(anyhow::anyhow!("Given root directory does not exist or is not a directory: {:?}", root_dir))
            .with_context(|| format!("Failed to access root directory: {:?}", root_dir));
    }

    let mut missing_dirs = Vec::new();
    for dir in [&csv_dir, &config_dir, &output_dir] {
        if !dir.exists() || !dir.is_dir() {
            missing_dirs.push(dir);
        }
    }
    if !missing_dirs.is_empty() {
        return Err(anyhow::anyhow!("Missing required directories: {:?}", missing_dirs))
            .with_context(|| format!("Failed to access required directories: {:?}", missing_dirs));
    }
    
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
    })
}

#[cfg(test)]
mod tests {
    // Same-file tests for CLI module
    use super::*;

    // Test Args parsing with default values
    #[test]
    fn test_args_parsing_defaults() {
        let args = Args::parse_from(["test", "test_route_dir"]);
        assert_eq!(args.route_dir, PathBuf::from("test_route_dir"));
        assert_eq!(args.internal_timestep_seconds, 300);
        match args.kernel {
            MuskingumCungeKernel::TRouteModernized => {},
            _ => panic!("Expected default kernel to be TRouteModernized"),
        }
    }
    // Test Args parsing with custom values
    #[test]
    fn test_args_parsing_custom() {
        let args = Args::parse_from(["test", "test_route_dir", "-i", "600", "-k", "t-route-legacy"]);
        assert_eq!(args.route_dir, PathBuf::from("test_route_dir"));
        assert_eq!(args.internal_timestep_seconds, 600);
        match args.kernel {
            MuskingumCungeKernel::TRouteLegacy => {},
            _ => panic!("Expected kernel to be TRouteLegacy"),
        }
    }
    // Impossible to test get_args(), as it pulls from the program's actual command line arguments, which we can't easily manipulate.
    // #[test]
    // fn test_get_args_invalid_root() {
    //     let result = get_args();
    //     assert!(result.is_err());
    // }
}
