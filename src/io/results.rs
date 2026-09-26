// Structure to store results for NetCDF output
#[derive(Debug)]
pub struct SimulationResults {
    pub feature_id: u32,
    pub flow_data: Vec<f32>,
    pub velocity_data: Vec<f32>,
    pub depth_data: Vec<f32>,
}

impl SimulationResults {
    pub fn with_capacity(feature_id: u32, capacity: usize) -> Self {
        SimulationResults {
            feature_id,
            flow_data: Vec::with_capacity(capacity),
            velocity_data: Vec::with_capacity(capacity),
            depth_data: Vec::with_capacity(capacity),
        }
    }
}
