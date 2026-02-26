extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree};
use std::fs;
use serde_json::Value;

#[proc_macro]
pub fn test_macro(_item: TokenStream) -> TokenStream {
    "fn test() { println!(\"Hello, world!\"); }".parse().unwrap()
}

#[proc_macro_attribute]
pub fn refresh_hardcoded_weights(target: TokenStream, item: TokenStream) -> TokenStream {
    // Load json weights from file at target path,
    // overwrite given WeightConfig construction with new values, 
    // and return the modified item.
    let target_str: String = target.to_string().trim_matches('"').to_string();
    let json_str: Result<String, std::io::Error> = fs::read_to_string(&target_str);
    if let Err(e) = json_str {
        // Just return the original item if there's an error reading the file
        eprintln!("Error reading weights file at ({}): {}", target_str, e);
        return item;
    }
    let weights: Value = serde_json::from_str(&json_str.unwrap()).unwrap();

    let input_size = weights["input_size"].as_u64().unwrap() as usize;
    let output_size = weights["output_size"].as_u64().unwrap() as usize;
    let hidden_size = weights["hidden_size"].as_u64().unwrap() as usize;

    fn to_vec(value: &Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }

    let input_gate_input_weights =
        to_vec(&weights["lstm.input_gate.input_transform.weight"]);
    let input_gate_input_biases = to_vec(&weights["lstm.input_gate.input_transform.bias"]);
    let input_gate_hidden_weights =
        to_vec(&weights["lstm.input_gate.hidden_transform.weight"]);
    let input_gate_hidden_biases =
        to_vec(&weights["lstm.input_gate.hidden_transform.bias"]);

    let forget_gate_input_weights =
        to_vec(&weights["lstm.forget_gate.input_transform.weight"]);
    let forget_gate_input_biases =
        to_vec(&weights["lstm.forget_gate.input_transform.bias"]);
    let forget_gate_hidden_weights =
        to_vec(&weights["lstm.forget_gate.hidden_transform.weight"]);
    let forget_gate_hidden_biases =
        to_vec(&weights["lstm.forget_gate.hidden_transform.bias"]);

    let cell_gate_input_weights = to_vec(&weights["lstm.cell_gate.input_transform.weight"]);
    let cell_gate_input_biases = to_vec(&weights["lstm.cell_gate.input_transform.bias"]);
    let cell_gate_hidden_weights =
        to_vec(&weights["lstm.cell_gate.hidden_transform.weight"]);
    let cell_gate_hidden_biases = to_vec(&weights["lstm.cell_gate.hidden_transform.bias"]);

    let output_gate_input_weights =
        to_vec(&weights["lstm.output_gate.input_transform.weight"]);
    let output_gate_input_biases =
        to_vec(&weights["lstm.output_gate.input_transform.bias"]);
    let output_gate_hidden_weights =
        to_vec(&weights["lstm.output_gate.hidden_transform.weight"]);
    let output_gate_hidden_biases =
        to_vec(&weights["lstm.output_gate.hidden_transform.bias"]);

    // Provided item will be as:
    // Self {
    //     input_size: [number],
    //     hidden_size: [number],
    //     output_size: [number],
    //     input_gate_input_weights: vec![values],
    //     input_gate_input_biases: vec![values],
    //     input_gate_hidden_weights: vec![values],
    //     input_gate_hidden_biases: vec![values],
    //     forget_gate_input_weights: vec![values],
    //     forget_gate_input_biases: vec![values],
    //     forget_gate_hidden_weights: vec![values],
    //     forget_gate_hidden_biases: vec![values],
    //     cell_gate_input_weights: vec![values],
    //     cell_gate_input_biases: vec![values],
    //     cell_gate_hidden_weights: vec![values],
    //     cell_gate_hidden_biases: vec![values],
    //     output_gate_input_weights: vec![values],
    //     output_gate_input_biases: vec![values],
    //     output_gate_hidden_weights: vec![values],
    //     output_gate_hidden_biases: vec![values],
    // }
    let innermost_open_brace_index = item.to_string().rfind('{').unwrap();
    let innermost_close_brace_index = item.to_string().find('}').unwrap();
    let pre_brace = item.to_string().chars().take(innermost_open_brace_index + 1).collect::<String>();
    let post_brace = item.to_string().chars().skip(innermost_close_brace_index).collect::<String>();
    let inner_brace = item.to_string().chars().skip(innermost_open_brace_index + 1).take(innermost_close_brace_index - innermost_open_brace_index - 1).collect::<String>();
    let inner_indentation = inner_brace.chars().take_while(|c| c.is_whitespace()).collect::<String>().trim_start_matches('\n').to_string();
    let mut new_inner_brace: String = "".to_string();
    new_inner_brace += &format!("{}input_size: {},\n", inner_indentation, input_size);
    new_inner_brace += &format!("{}hidden_size: {},\n", inner_indentation, hidden_size);
    new_inner_brace += &format!("{}output_size: {},\n", inner_indentation, output_size);
    new_inner_brace += &format!("{}input_gate_input_weights: vec!{:?},\n", inner_indentation, input_gate_input_weights);
    new_inner_brace += &format!("{}input_gate_input_biases: vec!{:?},\n", inner_indentation, input_gate_input_biases);
    new_inner_brace += &format!("{}input_gate_hidden_weights: vec!{:?},\n", inner_indentation, input_gate_hidden_weights);
    new_inner_brace += &format!("{}input_gate_hidden_biases: vec!{:?},\n", inner_indentation, input_gate_hidden_biases);
    new_inner_brace += &format!("{}forget_gate_input_weights: vec!{:?},\n", inner_indentation, forget_gate_input_weights);
    new_inner_brace += &format!("{}forget_gate_input_biases: vec!{:?},\n", inner_indentation, forget_gate_input_biases);
    new_inner_brace += &format!("{}forget_gate_hidden_weights: vec!{:?},\n", inner_indentation, forget_gate_hidden_weights);
    new_inner_brace += &format!("{}forget_gate_hidden_biases: vec!{:?},\n", inner_indentation, forget_gate_hidden_biases);
    new_inner_brace += &format!("{}cell_gate_input_weights: vec!{:?},\n", inner_indentation, cell_gate_input_weights);
    new_inner_brace += &format!("{}cell_gate_input_biases: vec!{:?},\n", inner_indentation, cell_gate_input_biases);
    new_inner_brace += &format!("{}cell_gate_hidden_weights: vec!{:?},\n", inner_indentation, cell_gate_hidden_weights);
    new_inner_brace += &format!("{}cell_gate_hidden_biases: vec!{:?},\n", inner_indentation, cell_gate_hidden_biases);
    new_inner_brace += &format!("{}output_gate_input_weights: vec!{:?},\n", inner_indentation, output_gate_input_weights);
    new_inner_brace += &format!("{}output_gate_input_biases: vec!{:?},\n", inner_indentation, output_gate_input_biases);
    new_inner_brace += &format!("{}output_gate_hidden_weights: vec!{:?},\n", inner_indentation, output_gate_hidden_weights);
    new_inner_brace += &format!("{}output_gate_hidden_biases: vec!{:?},\n", inner_indentation, output_gate_hidden_biases);
    new_inner_brace += &format!("{}{}", inner_indentation, "");
    let new_item_str: String = format!("{}{}{}", pre_brace, new_inner_brace, post_brace);
    // eprintln!("New item string with refreshed weights:\n{}", new_item_str);
    let new_item_lines: Vec<String> = new_item_str.lines().map(|line| line.to_string()).collect();
    // eprintln!("First 4 lines of the new item string with refreshed weights:\n{:?}", new_item_lines.iter().take(6).cloned().collect::<Vec<String>>());
    // eprintln!("Last 3 lines of the new item string with refreshed weights:\n{:?}", new_item_lines.iter().rev().take(3).rev().cloned().collect::<Vec<String>>());
    let joined_output: TokenStream = new_item_str.parse().unwrap();
    // eprintln!("OUTPUT ITEM:\n{}", joined_output.to_string());
    joined_output
}


fn to_vec(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn build_weights_function(func_sig: String, weights: Value) -> String {
    let mut output: String = String::from(func_sig.trim()); // pub fn [function_name] -> Option<Self>
    output += &format!(" {{\n    return Some(Self {{\n");
    // This method is technically more opaque than just doing a line-by-line string construction of the function body,
    // but that method is extremely tedious and hard to read/maintain, so this is a good compromise that
    // accomplishes the same thing more concisely.
    for size in ["input_size", "hidden_size", "output_size"] {
        output += &format!("        {}: {},\n", size, weights[size].as_u64().unwrap() as usize);
    }
    for gate in ["input_gate", "forget_gate", "cell_gate", "output_gate"] {
        output += &format!("        {}_input_weights: vec!{:?},\n", gate, to_vec(&weights[&format!("lstm.{}.input_transform.weight", gate)]));
        output += &format!("        {}_input_biases: vec!{:?},\n", gate, to_vec(&weights[&format!("lstm.{}.input_transform.bias", gate)]));
        output += &format!("        {}_hidden_weights: vec!{:?},\n", gate, to_vec(&weights[&format!("lstm.{}.hidden_transform.weight", gate)]));
        output += &format!("        {}_hidden_biases: vec!{:?},\n", gate, to_vec(&weights[&format!("lstm.{}.hidden_transform.bias", gate)]));
    }
    output += "    });\n";
    output += "}\n";
    output
}

#[proc_macro_attribute]
pub fn hardcode_weights(target: TokenStream, item: TokenStream) -> TokenStream {
    // Load json weights from file at target path,
    // overwrite given WeightConfig construction function with one created from scratch.
    let target_str: String = target.to_string().trim_matches('"').to_string();
    // grab the part of the string up to the first {, which should be the function signature (e.g. "pub fn nh_AORC_hourly_25yr_1210_112435_7() -> Option<Self>")
    let func_sig: String = item.to_string().split('{').next().unwrap().to_string();

    let json_str: Result<String, std::io::Error> = fs::read_to_string(&target_str);
    if let Err(e) = json_str {
        // Just return the original item if there's an error reading the file
        eprintln!("Error reading weights file at ({}): {}", target_str, e);
        return (func_sig.trim().to_string() + " { return None; }\n").parse().unwrap();
    }
    let weights: Value = serde_json::from_str(&json_str.unwrap()).unwrap();
    
    // eprintln!("Parsed function signature:\n{}", func_sig);
    // pass the function signature and weights to a helper function that builds the full function string with the weights hardcoded in, then parse that string into a TokenStream and return it
    let output_function_str: String = build_weights_function(func_sig, weights);
    // eprintln!("Generated function:\n{}", output_function_str);
    return output_function_str.parse().unwrap()
}