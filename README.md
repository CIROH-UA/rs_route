# Experimental MC routing in rust
Imitating t-route to help teach myself rust
the code in here isn't very correct or good

# Network Routing

A Rust implementation of network flow routing using the Muskingum-Cunge method.

## Project Structure

```
src/
├── main.rs         # Main entry point
├── cli.rs          # Command-line interface
├── config.rs       # Configuration structures
├── network.rs      # Network topology and database operations
├── state.rs        # Network state management
├── routing.rs      # Core routing logic
├── io/             # I/O operations
│   ├── mod.rs      # Module declarations
│   ├── csv.rs      # CSV reading/writing
│   ├── netcdf.rs   # NetCDF output
│   └── results.rs  # Simulation results storage
└── kernel/
    └── muskingum/
        ├── mod.rs          # Muskingum module
        ├── c_mc.rs         # Rust interface to C Muskingum-Cunge implementation
        ├── t_route.rs      # Rust interface to Fortran T-Route implementation
        ├── c_mc/
        │   ├── muskingumcunge.c
        │   └── muskingumcunge.h
        ├── route_rs/
        │   └── mc_kernel.rs    # Rust implementation of Muskingum-Cunge
        └── t-route/
            ├── bind.f90
            ├── muskingum_cunge.f90
            └── t-route-legacy/
                ├── MCsingleSegStime_f2py_NOLOOP.f90
                └── varPrecision.f90
```
<!-- Above is accurate to src/ as of Feb 2026 -->

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
