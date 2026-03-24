use crate::config::ChannelParams;
use crate::io::csv::load_external_flows;
use crate::io::netcdf::write_batch;
use crate::io::results::SimulationResults;
use crate::kernel::muskingum::{MuskingumCungeInput, MuskingumCungeKernel, MuskingumCungeResult};
use crate::network::NetworkTopology;
use crate::state::NodeStatus;
use anyhow::{Context, Result};
use indicatif::ProgressBar;
use netcdf::FileMut;
use std::cmp::min;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

// Message types
enum WriterMessage {
    WriteResults(Arc<SimulationResults>),
    Shutdown,
}

enum WorkerMessage {
    ProcessNode(u32),
    Shutdown,
}

enum SchedulerMessage {
    NodeCompleted(u32),
    Shutdown,
}

/// Linearly interpolate a single f32 value.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Interpolate channel parameters between two sets at fraction `t` (0→a, 1→b).
/// `dx` is NOT interpolated – it is set by the subdivision logic.
fn lerp_params(a: &ChannelParams, b: &ChannelParams, t: f32) -> ChannelParams {
    ChannelParams {
        dx: 0.0, // caller sets this
        n: lerp(a.n, b.n, t),
        ncc: lerp(a.ncc, b.ncc, t),
        s0: lerp(a.s0, b.s0, t),
        bw: lerp(a.bw, b.bw, t),
        tw: lerp(a.tw, b.tw, t),
        twcc: lerp(a.twcc, b.twcc, t),
        cs: lerp(a.cs, b.cs, t),
    }
}

/// Average channel parameters from multiple upstream nodes.
fn average_params(params_list: &[&ChannelParams]) -> ChannelParams {
    let n = params_list.len() as f32;
    ChannelParams {
        dx: params_list.iter().map(|p| p.dx).sum::<f32>() / n,
        n: params_list.iter().map(|p| p.n).sum::<f32>() / n,
        ncc: params_list.iter().map(|p| p.ncc).sum::<f32>() / n,
        s0: params_list.iter().map(|p| p.s0).sum::<f32>() / n,
        bw: params_list.iter().map(|p| p.bw).sum::<f32>() / n,
        tw: params_list.iter().map(|p| p.tw).sum::<f32>() / n,
        twcc: params_list.iter().map(|p| p.twcc).sum::<f32>() / n,
        cs: params_list.iter().map(|p| p.cs).sum::<f32>() / n,
    }
}

// Process all timesteps for a single node, optionally subdividing the reach
// into shorter sections when `subdivision_target_length` > 0.
// When subdivided, channel parameters are interpolated from upstream_params
// through channel_params to downstream_params along the reach.
fn process_node_all_timesteps(
    kernel: MuskingumCungeKernel,
    node_id: &u32,
    topology: &NetworkTopology,
    channel_params: &ChannelParams,
    upstream_params: Option<&ChannelParams>,
    downstream_params: Option<&ChannelParams>,
    max_timesteps: usize,
    dt: f32,
    subdivision_target_length: f32,
) -> Result<SimulationResults> {
    let node = topology
        .nodes
        .get(node_id)
        .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;

    let mut results = SimulationResults::new(node.id as i64);

    let area = node
        .area_sqkm
        .ok_or_else(|| anyhow::anyhow!("Node {} has no area defined", node_id))?;

    let mut external_flows =
        load_external_flows(node.qlat_file.clone(), &node.id, Some(&"Q_OUT"), area)?;

    let mut inflow = node
        .inflow_storage
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock inflow storage: {}", e))?;

    if inflow.len() == 0 && external_flows.len() == 0 {
        // if these are both empty then just return all zeros to the results
        results.flow_data = vec![0.0; max_timesteps];
        results.velocity_data = vec![0.0; max_timesteps];
        results.depth_data = vec![0.0; max_timesteps];
        return Ok(results);
    }

    // if headwater then upstream inflow is 0.0
    if inflow.len() == 0 {
        inflow.resize(max_timesteps, 0.0);
    }

    if external_flows.len() == 0 {
        external_flows.resize(max_timesteps, 0.0);
    } else if external_flows.len() == 1 {
        // Only a single external flow value breaks the upsampling logic,
        // so we throw an error if the file only contains one value (which is likely a mistake)
        return Err(anyhow::anyhow!(
            "External flow file for node {} only contains one value, which is not sufficient for routing. Please check the file: {:?}",
            node_id,
            node.qlat_file
        )).with_context(|| format!("Failed to load external flows for node {}: {:?}", node_id, node.qlat_file));
    }

    // Determine number of subdivisions
    let total_length = channel_params.dx;
    let num_sections = if subdivision_target_length > 0.0 {
        (total_length / subdivision_target_length).ceil().max(1.0) as usize
    } else {
        1
    };
    let section_dx = total_length / num_sections as f32;

    // Precompute interpolated channel params for each section.
    // The interpolation goes: upstream_params → channel_params → downstream_params
    // with channel_params at the midpoint of the reach.
    let up = upstream_params.unwrap_or(channel_params);
    let down = downstream_params.unwrap_or(channel_params);
    let section_params: Vec<ChannelParams> = (0..num_sections)
        .map(|s| {
            if num_sections == 1 {
                channel_params.clone()
            } else {
                // t goes from 0 (upstream end) to 1 (downstream end)
                // section center position
                let t = (s as f32 + 0.5) / num_sections as f32;
                if t <= 0.5 {
                    // First half: interpolate from upstream_params to channel_params
                    // t=0 → upstream, t=0.5 → current
                    lerp_params(up, channel_params, t * 2.0)
                } else {
                    // Second half: interpolate from channel_params to downstream_params
                    // t=0.5 → current, t=1.0 → downstream
                    lerp_params(channel_params, down, (t - 0.5) * 2.0)
                }
            }
        })
        .collect();

    // Per-section state arrays
    let mut section_qup = vec![0.0_f32; num_sections];
    let mut section_qdp = vec![0.0_f32; num_sections];
    let mut section_depthp = vec![0.0_f32; num_sections];

    // -1 because the input files have one additional timestep
    let upsampling = max_timesteps / (external_flows.len() - 1);

    let mut external_flow = 0.0;

    for _timestep in 0..max_timesteps {
        if _timestep % upsampling == 0 {
            external_flow = external_flows.pop_front().ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to fetch qlateral from file for: {} at timestep {}",
                    node_id,
                    _timestep
                )
            })?;
        }
        let upstream_flow = inflow.pop_front().unwrap();

        // Scale lateral flow proportionally for each section
        let section_ql = external_flow / num_sections as f32;

        let mut current_upstream = upstream_flow;
        let mut last_result = MuskingumCungeResult::default();

        for s in 0..num_sections {
            let sp = &section_params[s];
            let section_s0 = if sp.s0 == 0.0 { 0.00001 } else { sp.s0 };
            last_result = kernel.exec(
                &MuskingumCungeInput {
                    dt,
                    qup: section_qup[s],
                    quc: current_upstream,
                    qdp: section_qdp[s],
                    ql: section_ql,
                    dx: section_dx,
                    bw: sp.bw,
                    tw: sp.tw,
                    tw_cc: sp.twcc,
                    n: sp.n,
                    n_cc: sp.ncc,
                    cs: sp.cs,
                    s0: section_s0,
                    velp: 0.0, // unused
                    depthp: section_depthp[s],
                },
                false,
            );

            // Update section state for next timestep
            section_qup[s] = current_upstream;
            section_qdp[s] = last_result.qdc;
            section_depthp[s] = last_result.depthc;

            // Output of this section feeds the next
            current_upstream = last_result.qdc;
        }

        // Only record the final section's output
        results.flow_data.push(last_result.qdc);
        results.velocity_data.push(last_result.velc);
        results.depth_data.push(last_result.depthc);
    }

    Ok(results)
}

fn writer_thread(
    receiver: Receiver<WriterMessage>,
    output_file: Arc<Mutex<FileMut>>,
    batch_size: usize, // e.g., 100 nodes
) -> Result<()> {
    let mut batch = Vec::new();

    loop {
        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(WriterMessage::WriteResults(results)) => {
                batch.push(results);

                // Write when batch is full
                if batch.len() >= batch_size {
                    write_batch(&output_file, &batch)?;
                    batch.clear();
                }
            }
            Ok(WriterMessage::Shutdown) => {
                // Write remaining batch
                if !batch.is_empty() {
                    write_batch(&output_file, &batch)?;
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Write partial batch on timeout to avoid holding data too long
                if !batch.is_empty() {
                    write_batch(&output_file, &batch)?;
                    batch.clear();
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // All senders dropped — normal shutdown
                if !batch.is_empty() {
                    write_batch(&output_file, &batch)?;
                }
                break;
            }
        }
    }
    Ok(())
}

// Scheduler thread that tracks dependencies and sends ready work
fn scheduler_thread(
    topology: Arc<NetworkTopology>,
    scheduler_rx: Receiver<SchedulerMessage>,
    worker_tx: Vec<Sender<WorkerMessage>>,
    total_nodes: usize,
    _completed_count: Arc<AtomicUsize>,
) -> Result<()> {
    // Track which nodes are ready to process
    let mut ready_nodes = VecDeque::new();
    let mut processed_nodes = HashSet::new();
    let mut pending_downstream_count: HashMap<u32, usize> = HashMap::new();

    // Initialize with leaf nodes (no upstream dependencies)
    for (&node_id, node) in &topology.nodes {
        if node.upstream_ids.is_empty() {
            ready_nodes.push_back(node_id);
        } else {
            // Count how many upstream nodes need to complete
            pending_downstream_count.insert(node_id, node.upstream_ids.len());
        }
    }

    let num_workers = worker_tx.len();
    let mut next_worker = 0;

    loop {
        // Send ready work to workers
        while let Some(node_id) = ready_nodes.pop_front() {
            // Round-robin distribution to workers
            if let Err(e) = worker_tx[next_worker].send(WorkerMessage::ProcessNode(node_id)) {
                eprintln!("Failed to send work to worker {}: {}", next_worker, e);
            }
            next_worker = (next_worker + 1) % num_workers;
        }

        // Wait for completion messages
        match scheduler_rx.recv() {
            Ok(SchedulerMessage::NodeCompleted(node_id)) => {
                processed_nodes.insert(node_id);

                // Check if this enables any downstream nodes
                if let Some(node) = topology.nodes.get(&node_id) {
                    if let Some(downstream_id) = node.downstream_id {
                        if let Some(count) = pending_downstream_count.get_mut(&downstream_id) {
                            *count = count.saturating_sub(1);
                            if *count == 0 {
                                // All upstream nodes are complete, this node is ready
                                ready_nodes.push_back(downstream_id);
                                pending_downstream_count.remove(&downstream_id);
                            }
                        }
                    }
                }

                // Check if we're done
                if processed_nodes.len() >= total_nodes {
                    break;
                }
            }
            Ok(SchedulerMessage::Shutdown) => break,
            Err(e) => {
                eprintln!("Scheduler channel error: {}", e);
                break;
            }
        }
    }

    // Send shutdown to all workers
    for tx in &worker_tx {
        let _ = tx.send(WorkerMessage::Shutdown);
    }

    Ok(())
}

// Worker thread - now just receives work and processes it
fn worker_thread(
    kernel: MuskingumCungeKernel,
    work_rx: Receiver<WorkerMessage>,
    scheduler_tx: Sender<SchedulerMessage>,
    topology: Arc<NetworkTopology>,
    channel_params_map: Arc<HashMap<u32, ChannelParams>>,
    max_timesteps: usize,
    dt: f32,
    writer_tx: Sender<WriterMessage>,
    progress_bar: Arc<ProgressBar>,
    subdivision_target_length: f32,
) -> Result<()> {
    loop {
        match work_rx.recv() {
            Ok(WorkerMessage::ProcessNode(node_id)) => {
                // Process the node
                if let Some(params) = channel_params_map.get(&node_id) {
                    // Look up upstream/downstream params for interpolation
                    let upstream_params = topology
                        .nodes
                        .get(&node_id)
                        .and_then(|node| {
                            let up_params: Vec<&ChannelParams> = node
                                .upstream_ids
                                .iter()
                                .filter_map(|uid| channel_params_map.get(uid))
                                .collect();
                            if up_params.is_empty() {
                                None
                            } else {
                                Some(average_params(&up_params))
                            }
                        });
                    let downstream_params = topology
                        .nodes
                        .get(&node_id)
                        .and_then(|node| node.downstream_id)
                        .and_then(|did| channel_params_map.get(&did))
                        .cloned();

                    match process_node_all_timesteps(
                        kernel,
                        &node_id,
                        &topology,
                        params,
                        upstream_params.as_ref(),
                        downstream_params.as_ref(),
                        max_timesteps,
                        dt,
                        subdivision_target_length,
                    ) {
                        Ok(results) => {
                            let results_arc = Arc::new(results);

                            // Send results to writer
                            if let Err(e) = writer_tx
                                .send(WriterMessage::WriteResults(Arc::clone(&results_arc)))
                            {
                                eprintln!("Failed to send results to writer: {}", e);
                            }

                            // Pass flow to downstream node
                            if let Some(node) = topology.nodes.get(&node_id) {
                                if let Some(downstream_id) = node.downstream_id {
                                    if let Some(downstream_node) =
                                        topology.nodes.get(&downstream_id)
                                    {
                                        let mut buffer =
                                            downstream_node.inflow_storage.lock().map_err(|e| {
                                                anyhow::anyhow!(
                                                    "Failed to lock downstream buffer: {}",
                                                    e
                                                )
                                            })?;
                                        if buffer.is_empty() {
                                            buffer.resize(results_arc.flow_data.len(), 0.0);
                                        }
                                        for (i, &flow) in results_arc.flow_data.iter().enumerate() {
                                            if i < buffer.len() {
                                                buffer[i] += flow;
                                            }
                                        }
                                    }
                                }

                                // Update status
                                let mut status = node.status.write().map_err(|e| {
                                    anyhow::anyhow!("Failed to acquire status write lock: {}", e)
                                })?;
                                *status = NodeStatus::Ready;

                                // Clear inflow storage
                                let mut old_inflow = node.inflow_storage.lock().map_err(|e| {
                                    anyhow::anyhow!("Failed to lock inflow storage: {}", e)
                                })?;
                                old_inflow.clear();
                            }
                        }
                        Err(e) => {
                            let mut error_message =
                                format!("Error processing node {}: {}", node_id, e);
                            // if error context, elaborate on it
                            if let Some(context) = e.chain().skip(1).next() {
                                error_message.push_str(&format!("\nContext: {}", context));
                            }
                            eprintln!("{}", error_message);
                            writer_tx.send(WriterMessage::Shutdown).ok();
                            scheduler_tx.send(SchedulerMessage::Shutdown).ok();
                        }
                    }

                    progress_bar.inc(1);
                }

                // Notify scheduler that node is complete
                if let Err(e) = scheduler_tx.send(SchedulerMessage::NodeCompleted(node_id)) {
                    eprintln!("Failed to notify scheduler of completion: {}", e);
                }
            }
            Ok(WorkerMessage::Shutdown) => break,
            Err(e) => {
                eprintln!("Worker channel error: {}", e);
                break;
            }
        }
    }
    Ok(())
}

// Main parallel routing function
pub fn process_routing_parallel(
    kernel: MuskingumCungeKernel,
    topology: &NetworkTopology,
    channel_params_map: &HashMap<u32, ChannelParams>,
    max_timesteps: usize,
    dt: f32,
    output_file: Arc<Mutex<FileMut>>,
    progress_bar: Arc<ProgressBar>,
    subdivision_target_length: f32,
) -> Result<()> {
    let total_nodes = topology.nodes.len();
    let completed_count = Arc::new(AtomicUsize::new(0));
    let topology_arc = Arc::new(topology.clone());
    let channel_params_arc = Arc::new(channel_params_map.clone());

    // Create channels
    let (writer_tx, writer_rx) = mpsc::channel();
    let (scheduler_tx, scheduler_rx) = mpsc::channel();

    // Create worker channels
    let num_threads = num_cpus::get();
    println!(
        "Using {} worker threads for parallel processing",
        num_threads
    );

    let mut worker_txs = Vec::new();
    let mut worker_handles = Vec::new();

    // Spawn worker threads
    for i in 0..num_threads {
        let (work_tx, work_rx) = mpsc::channel();
        worker_txs.push(work_tx);

        let topo = Arc::clone(&topology_arc);
        let params = Arc::clone(&channel_params_arc);
        let writer = writer_tx.clone();
        let scheduler = scheduler_tx.clone();
        let pb = Arc::clone(&progress_bar);

        let handle = thread::spawn(move || {
            if let Err(e) = worker_thread(
                kernel,
                work_rx,
                scheduler,
                topo,
                params,
                max_timesteps,
                dt,
                writer,
                pb,
                subdivision_target_length,
            ) {
                eprintln!("Worker {} error: {}", i, e);
            }
        });
        worker_handles.push(handle);
    }

    // Spawn writer thread
    let output_file_clone = Arc::clone(&output_file);
    let writer_handle = thread::spawn(move || {
        if let Err(e) = writer_thread(writer_rx, output_file_clone, min(100, total_nodes)) {
            eprintln!("Writer thread error: {}", e);
        }
    });

    // Spawn scheduler thread
    let topo = Arc::clone(&topology_arc);
    let completed = Arc::clone(&completed_count);
    let scheduler_handle = thread::spawn(move || {
        if let Err(e) = scheduler_thread(topo, scheduler_rx, worker_txs, total_nodes, completed) {
            eprintln!("Scheduler thread error: {}", e);
        }
    });

    // Drop original senders
    drop(writer_tx);
    drop(scheduler_tx);

    // Wait for all threads to complete
    scheduler_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Scheduler thread panicked: {:?}", e))?;

    for (i, handle) in worker_handles.into_iter().enumerate() {
        handle
            .join()
            .map_err(|e| anyhow::anyhow!("Worker thread {} panicked: {:?}", i, e))?;
    }

    writer_handle
        .join()
        .map_err(|e| anyhow::anyhow!("Writer thread panicked: {:?}", e))?;

    progress_bar.finish_with_message("Complete");
    println!("Successfully processed all {} nodes", total_nodes);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ColumnConfig;
    use crate::network::build_network_topology;

    fn setup_test_topology_and_params() -> (NetworkTopology, HashMap<u32, ChannelParams>) {
        let conn = rusqlite::Connection::open("./tests/one_cat/config/cat-486888_subset.gpkg")
            .expect("Failed to open test database");
        let column_config = ColumnConfig::new();
        let csv_dir = std::path::PathBuf::from("./tests/one_cat/outputs/ngen");

        let topology =
            build_network_topology(&conn, &column_config, &csv_dir).expect("Failed to build topology");
        let channel_params_map =
            crate::network::load_channel_parameters(&conn, &topology, &column_config)
                .expect("Failed to load channel params");

        (topology, channel_params_map)
    }

    /// Subdivision with target_length == total dx should produce identical results
    /// to no subdivision (target_length = -1).
    #[test]
    fn test_subdivision_single_section_matches_no_subdivision() {
        let (topology, channel_params_map) = setup_test_topology_and_params();
        let node_id: u32 = 486888;
        let params = channel_params_map.get(&node_id).unwrap();
        let kernel = MuskingumCungeKernel::TRouteModernized;
        let dt = 300.0;
        let max_timesteps = 24 * (3600 / 300); // 24 hours at 300s

        // No subdivision
        let result_no_sub = process_node_all_timesteps(
            kernel, &node_id, &topology, params, None, None, max_timesteps, dt, -1.0,
        )
        .unwrap();

        // Reset inflow storage (consumed by previous call)
        topology
            .nodes
            .get(&node_id)
            .unwrap()
            .inflow_storage
            .lock()
            .unwrap()
            .clear();

        // Subdivision with target == full length (1 section)
        let result_one_section = process_node_all_timesteps(
            kernel,
            &node_id,
            &topology,
            params,
            None,
            None,
            max_timesteps,
            dt,
            params.dx, // target = full length => 1 section
        )
        .unwrap();

        assert_eq!(result_no_sub.flow_data.len(), result_one_section.flow_data.len());
        let tolerance = 0.01; // 1% relative tolerance (f32 rounding from ceil(dx/dx)=1)
        for i in 0..result_no_sub.flow_data.len() {
            let expected = result_no_sub.flow_data[i];
            let actual = result_one_section.flow_data[i];
            if expected != 0.0 {
                let rel_diff = ((expected - actual) / expected).abs();
                assert!(
                    rel_diff < tolerance,
                    "Flow mismatch at timestep {}: {} vs {} ({:.2}%)",
                    i, expected, actual, rel_diff * 100.0,
                );
            }
        }
    }

    /// Subdivision with a small target length should produce valid, non-zero
    /// results that differ from the undivided case (since the kernel behaves
    /// differently with shorter dx).
    #[test]
    fn test_subdivision_produces_valid_output() {
        let (topology, channel_params_map) = setup_test_topology_and_params();
        let node_id: u32 = 486888;
        let params = channel_params_map.get(&node_id).unwrap();
        let kernel = MuskingumCungeKernel::TRouteModernized;
        let dt = 300.0;
        let max_timesteps = 24 * (3600 / 300);

        // No subdivision
        let result_no_sub = process_node_all_timesteps(
            kernel, &node_id, &topology, params, None, None, max_timesteps, dt, -1.0,
        )
        .unwrap();

        // Reset inflow storage
        topology
            .nodes
            .get(&node_id)
            .unwrap()
            .inflow_storage
            .lock()
            .unwrap()
            .clear();

        // Subdivision with 300m target (node is ~6910m, so ~24 sections)
        let result_subdivided = process_node_all_timesteps(
            kernel, &node_id, &topology, params, None, None, max_timesteps, dt, 300.0,
        )
        .unwrap();

        assert_eq!(result_subdivided.flow_data.len(), max_timesteps);

        // Results should be non-zero (the test data has real flows)
        let total_flow: f32 = result_subdivided.flow_data.iter().sum();
        assert!(
            total_flow > 0.0,
            "Subdivided total flow should be positive, got {}",
            total_flow,
        );

        // Results should differ from the undivided case
        let mut any_different = false;
        for i in 0..max_timesteps {
            if (result_no_sub.flow_data[i] - result_subdivided.flow_data[i]).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Subdivided results should differ from undivided (dx={}, 300m target => {} sections)",
            params.dx,
            (params.dx / 300.0).ceil() as usize,
        );
    }
}
