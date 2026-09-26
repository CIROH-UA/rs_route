use anyhow::{Context, Result};
use csv::{ReaderBuilder, StringRecord};
use std::collections::VecDeque;
use std::io::ErrorKind;
use std::path::Path;

// Function to load external flows for a specific nexus/catchment
pub fn load_external_flows(
    csv_file: &Path,
    id: u32,
    var_name: &str,
    area: f32,
) -> Result<VecDeque<f32>> {
    let mut external_flows = VecDeque::new();

    let mut rdr = match ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(csv_file)
    {
        Ok(rdr) => rdr,
        Err(err) => {
            if let csv::ErrorKind::Io(io_err) = err.kind() {
                if io_err.kind() == ErrorKind::NotFound {
                    println!(
                        "No external flow file found for {}: {}",
                        id,
                        csv_file.display()
                    );
                    return Ok(external_flows);
                }
            }
            return Err(err)
                .with_context(|| format!("Failed to open CSV file: {}", csv_file.display()));
        }
    };

    let headers = rdr.headers().context("Failed to read CSV headers")?;
    let qlat_index = headers.iter().position(|h| h == var_name).unwrap_or(2);

    let mut record = StringRecord::new();
    let mut i = 0;
    while rdr
        .read_record(&mut record)
        .with_context(|| format!("Failed to read record {} in file {}", i, csv_file.display()))?
    {
        let ql_str = record
            .get(qlat_index)
            .ok_or_else(|| anyhow::anyhow!("Missing column {} in record {}", qlat_index, i))?;

        let ql = ql_str
            .parse::<f32>()
            .with_context(|| format!("Failed to parse flow value '{}' in record {}", ql_str, i))?;

        // https://github.com/CIROH-UA/ngen/blob/ed2a903730467fa631716c033b757c3dff5fa2bb/include/core/Layer.hpp#L142
        let adjusted_flow = (ql * (area * 1_000_000.0)) / 3600.0;
        external_flows.push_back(adjusted_flow);
        i += 1;
    }

    Ok(external_flows)
}
