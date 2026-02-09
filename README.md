# Experimental MC routing in rust
Imitating t-route to help teach myself rust
the code in here isn't very correct or good

# Network Routing

A Rust implementation of network flow routing using the Muskingum-Cunge method.

## Project Structure

```
src/
├── main.rs         # Main entry point
├── config.rs       # Configuration structures
├── network.rs      # Network topology and database operations
├── state.rs        # Network state management
├── routing.rs      # Core routing logic
├── mc_kernel.rs    # Muskingum-Cunge kernel (existing)
└── io/             # I/O operations
    ├── mod.rs      # Module declarations
    ├── csv.rs      # CSV reading/writing
    ├── netcdf.rs   # NetCDF output
    └── results.rs  # Simulation results storage
```
## Dependencies
* hdf5
* netcdf
* sqlite3

### on ubuntu
```bash
sudo apt install -y libhdf5-dev libnetcdf-dev libsqlite3-dev
```
## Building and Running

```bash
# Build in release mode for optimal performance
cargo build --release

# Run the simulation
cargo run --release --bin route_rs [target_directory]
```

## Performance Optimizations

- Parallel loading of external flow CSV files
- Serial routing computation (due to dependencies)
- Efficient topological sorting for correct processing order

## Future Improvements

- Command-line argument parsing
- Configuration file support
- Additional output formats
- Performance profiling and optimization
- Unit tests for each module
