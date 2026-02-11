// Configuration structure for column name mapping
#[derive(Debug, Clone)]
pub struct ColumnConfig {
    pub key: String,
    pub downstream: String,
    pub dx: String,
    pub n: String,
    pub ncc: String,
    pub s0: String,
    pub bw: String,
    pub tw: String,
    pub twcc: String,
    pub cs: String,
}

impl Default for ColumnConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnConfig {
    pub fn new() -> Self {
        ColumnConfig {
            key: "id".to_string(),
            downstream: "toid".to_string(),
            dx: "Length_m".to_string(),
            n: "n".to_string(),
            ncc: "nCC".to_string(),
            s0: "So".to_string(),
            bw: "BtmWdth".to_string(),
            tw: "TopWdth".to_string(),
            twcc: "TopWdthCC".to_string(),
            cs: "ChSlp".to_string(),
        }
    }
}

// Output format configuration
#[derive(Debug, Clone)]
pub enum OutputFormat {
    Csv,
    NetCdf,
    Both,
}

// Channel parameters from SQLite
#[derive(Debug, Clone)]
pub struct ChannelParams {
    pub dx: f32,
    pub n: f32,
    pub ncc: f32,
    pub s0: f32,
    pub bw: f32,
    pub tw: f32,
    pub twcc: f32,
    pub cs: f32,
}

#[cfg(test)]
mod tests {
    // Same-file tests for config module
    use super::*;

    // Test ColumnConfig defaults
    #[test]
    fn test_column_config_defaults() {
        fn assert_column_config_defaults(config: &ColumnConfig) {
            assert_eq!(config.key, "id", "Key column name should be 'id', not '{}'", config.key);
            assert_eq!(config.downstream, "toid", "Downstream column name should be 'toid', not '{}'", config.downstream);
            assert_eq!(config.dx, "Length_m", "DX column name should be 'Length_m', not '{}'", config.dx);
            assert_eq!(config.n, "n", "Manning's n column name should be 'n', not '{}'", config.n);
            assert_eq!(config.ncc, "nCC", "Manning's nCC column name should be 'nCC', not '{}'", config.ncc);
            assert_eq!(config.s0, "So", "Channel slope column name should be 'So', not '{}'", config.s0);
            assert_eq!(config.bw, "BtmWdth", "Bottom width column name should be 'BtmWdth', not '{}'", config.bw);
            assert_eq!(config.tw, "TopWdth", "Top width column name should be 'TopWdth', not '{}'", config.tw);
            assert_eq!(config.twcc, "TopWdthCC", "Top width at critical condition column name should be 'TopWdthCC', not '{}'", config.twcc);
            assert_eq!(config.cs, "ChSlp", "Channel slope column name should be 'ChSlp', not '{}'", config.cs);
        }
        let config_default: ColumnConfig = ColumnConfig::default();
        assert_column_config_defaults(&config_default);
        let config_new = ColumnConfig::new();
        assert_column_config_defaults(&config_new);
    }
    // Test OutputFormat enum variants
    #[test]
    fn test_output_format_variants() {
        let format_csv = OutputFormat::Csv;
        let format_netcdf = OutputFormat::NetCdf;
        let format_both = OutputFormat::Both;
        match format_csv {
            OutputFormat::Csv => (),
            _ => panic!("Expected OutputFormat::Csv variant"),
        }
        match format_netcdf {
            OutputFormat::NetCdf => (),
            _ => panic!("Expected OutputFormat::NetCdf variant"),
        }
        match format_both {
            OutputFormat::Both => (),
            _ => panic!("Expected OutputFormat::Both variant"),
        }
    }
    // Test ChannelParams struct initialization
    #[test]
    fn test_channel_params_initialization() {
        let params = ChannelParams {
            dx: 1.0,
            n: 2.0,
            ncc: 3.0,
            s0: 4.0,
            bw: 5.0,
            tw: 6.0,
            twcc: 7.0,
            cs: 8.0,
        };
        assert_eq!(params.dx, 1.0, "DX value should be 1.0, not {}", params.dx);
        assert_eq!(params.n, 2.0, "Manning's n value should be 2.0, not {}", params.n);
        assert_eq!(params.ncc, 3.0, "Manning's nCC value should be 3.0, not {}", params.ncc);
        assert_eq!(params.s0, 4.0, "Channel slope value should be 4.0, not {}", params.s0);
        assert_eq!(params.bw, 5.0, "Bottom width value should be 5.0, not {}", params.bw);
        assert_eq!(params.tw, 6.0, "Top width value should be 6.0, not {}", params.tw);
        assert_eq!(params.twcc, 7.0, "Top width at critical condition value should be 7.0, not {}", params.twcc);
        assert_eq!(params.cs, 8.0, "Channel slope value should be 8.0, not {}", params.cs);
    }
}