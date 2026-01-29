# Experimental MC routing in rust
immitating t-route to help teach myself rust

# Network Routing
A Rust implementation of network flow routing using the Muskingum-Cunge method.
+ rust lstm

## Project Structure

```
src/
├── main.rs         # Main entry point
├── lstm_flow.rs    # all the rust lstm crammed into one file
├── config.rs       # Configuration structures for routing
├── network.rs      # Network topology and database operations
├── state.rs        # Network state management
├── routing.rs      # Core routing logic
├── kernel          # Various routing kernels
└── io/             # I/O operations
    ├── mod.rs      # Module declarations
    ├── csv.rs      # CSV reading/writing
    ├── netcdf.rs   # NetCDF output
    └── results.rs  # Simulation results storage
python_convert.py   # The python script that converts model weights to rust/burn format
```
## Dependencies
* cargo & rust
* gcc
* gfortran
* hdf5
* netcdf
* sqlite3
* UV (optional for automatic weight conversion)

The rust code uses UV to run the python_convert.py script if the converted weights are missing.    
`uv run -p 3.9 --with pyyaml --with numpy --with torch --extra-index-url https://download.pytorch.org/whl/cpu /path/to/model_007.pt /path/to/config.yml`    

### on ubuntu
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# install cargo + rust
curl https://sh.rustup.rs -sSf | sh
# install the rest of the deps
sudo apt install -y libhdf5-dev libnetcdf-dev libsqlit3-dev gfortran gcc
```

## Building and Running

```bash
# Build in release mode for optimal performance
cargo build --release

# Run the simulation      -l for running lstm
./target/release/route_rs -l /path/to/ngiab/style/data
```
## install without cloning

```bash
# Build in release mode for optimal performance
cargo install --git https://github.com/JoshCu/route_rs.git --branch lstm_dev
# Run the simulation      -l for running lstm
route_rs -l /path/to/ngiab/style/data
```

# Hardcoded paths
the following files and folders are expected.
```
ngiab-data
├── config
│   ├── cat_config
│   │   └── lstm
│   │       ├── cat-1.yml
│   │       ├── cat-2.yml
│   │       └── cat-3.yml
│   └── some.gpkg
├── forcings
│   └── forcings.nc
└── outputs
    └── troute
```

### cat-1.yml
```yaml
train_cfg_file:
 - /ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_25yr_1210_112435_7/config.yml
```
### in that train_cfg_file ..../nh_AORC_hourly_25yr_1210_112435_7/config.yml
```yaml
run_dir: ../trained_neuralhydrology_models/nh_AORC_hourly_25yr_1210_112435_7
##!!!!!! ^^ the .. above is interpreted by the code as (config.yml).parent.parent.parent  
# aka /ngen/ngen/extern/lstm/    this to avoid having to modify config files after training in pytorch
```

### in that run_dir 
```
nh_AORC_hourly_25yr_1210_112435_7/
├── config.yml
├── model_*.pt # wildcard get first match
└── train_data
    └── train_data_scaler.yml
```

### after the weights have been converted there will be an extra burn folder
```
nh_AORC_hourly_25yr_1210_112435_7/
├── burn
│   ├── model_epoch007.json
│   ├── model_epoch007.pt
│   ├── train_data_scaler.json
│   └── weights.json
├── config.yml
├── model_epoch007.pt
└── train_data
    └── train_data_scaler.yml
```

# lstm inputs
during conversion lstm inputs AND ORDER are fetched from `train_cfg_file["dynamic_inputs"] + train_cfg_file["static_attributes"]`.    
they're saved to this json file
```
nh_AORC_hourly_25yr_1210_112435_7/
├── burn
│   ├── model_epoch007.json
```

```json
// model_epoch007.json
{
  "hidden_size": 64,
  "input_size": 10,
  "output_size": 64,
  "input_names": [
    "APCP_surface",
    "TMP_2maboveground",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "elev_mean", // these need to match the names in cat_config/lstm/cat-1.yml 
    "slope_mean" 
  ],
  "output_names": [
    "QObs(mm/h)"
  ]
}
```

# GPKG format
This is designed to work with the v2.2hf and needs the following
```
flowpaths 
  - areasqkm
  - id
  - toid
  - Length_m
  - n
  - nCC
  - So
  - BtmWdth
  - TopWdth
  - TopWdthCC
  - ChSlp
// names of these can be changed in src/config.rs 
```
none of the other tables are actually needed    
catchments are just flowpaths with the prefix changes e.g. wb-1 -> cat-1    
`divides.areasqkm` is the same as `flowpaths.sqkm` so it's just fetched from flowpaths
