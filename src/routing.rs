use crate::config::ChannelParams;
use crate::io::csv::load_external_flows;
use crate::io::netcdf::NetCdfWriter;
use crate::io::results::SimulationResults;
use crate::kernel::muskingum::{MuskingumCungeInput, MuskingumCungeKernel};
use crate::network::NetworkTopology;
use anyhow::{Context, Result};
use crossbeam_channel::{self as cb, Receiver, Sender};
use indicatif::ProgressBar;
use rustc_hash::FxHashMap;
use std::cmp::min;
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;

// Message types
enum WriterMessage {
    WriteResults(SimulationResults),
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

// Process all timesteps for a single node
fn process_node_all_timesteps(
    kernel: MuskingumCungeKernel,
    node_id: u32,
    topology: &NetworkTopology,
    channel_params: &ChannelParams,
    max_timesteps: usize,
    dt: f32,
) -> Result<SimulationResults> {
    let node = topology
        .nodes
        .get(&node_id)
        .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;

    let mut results = SimulationResults::with_capacity(node.id, max_timesteps);

    let mut external_flows =
        load_external_flows(&node.qlat_file, node.id, "Q_OUT", node.area_sqkm)?;

    // Empty for headwaters, whose upstream inflow is 0.0
    let inflow = node.take_inflow()?;

    if inflow.is_empty() && external_flows.is_empty() {
        // if these are both empty then just return all zeros to the results
        results.flow_data.resize(max_timesteps, 0.0);
        results.velocity_data.resize(max_timesteps, 0.0);
        results.depth_data.resize(max_timesteps, 0.0);
        return Ok(results);
    }

    if external_flows.is_empty() {
        external_flows.resize(max_timesteps, 0.0);
    } else if external_flows.len() == 1 {
        // Only a single external flow value breaks the upsampling logic,
        // so we throw an error if the file only contains one value (which is likely a mistake)
        anyhow::bail!(
            "External flow file for node {} only contains one value, which is not sufficient for routing. Please check the file: {:?}",
            node_id,
            node.qlat_file
        );
    }

    let mut qup = 0.0;
    let mut qdp = 0.0;
    let mut depth_p = 0.0;
    // -1 because the input files have one additional timestep
    let upsampling = max_timesteps / (external_flows.len() - 1);

    let mut external_flow = 0.0;

    for timestep in 0..max_timesteps {
        if timestep % upsampling == 0 {
            external_flow = external_flows.pop_front().ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to fetch qlateral from file for: {} at timestep {}",
                    node_id,
                    timestep
                )
            })?;
        }
        let upstream_flow = inflow.get(timestep).copied().unwrap_or(0.0);

        let result = kernel.exec(
            &MuskingumCungeInput {
                dt,
                qup,
                quc: upstream_flow,
                qdp,
                ql: external_flow,
                dx: channel_params.dx,
                bw: channel_params.bw,
                tw: channel_params.tw,
                tw_cc: channel_params.twcc,
                n: channel_params.n,
                n_cc: channel_params.ncc,
                cs: channel_params.cs,
                s0: channel_params.s0,
                velp: 0.0, // unused
                depthp: depth_p,
            },
            false,
        );

        results.flow_data.push(result.qdc);
        results.velocity_data.push(result.velc);
        results.depth_data.push(result.depthc);

        qup = upstream_flow;
        qdp = result.qdc;
        depth_p = result.depthc;
    }

    Ok(results)
}

// Buffers results into batches and appends them to the output file. Runs until
// told to shut down or until every sender has been dropped.
fn writer_thread(
    receiver: Receiver<WriterMessage>,
    mut output: NetCdfWriter,
    batch_size: usize, // e.g., 100 nodes
) -> Result<()> {
    let mut batch = Vec::with_capacity(batch_size);
    while let Ok(WriterMessage::WriteResults(results)) = receiver.recv() {
        batch.push(results);
        if batch.len() >= batch_size {
            output.append(&batch)?;
            batch.clear();
        }
    }
    output.append(&batch)
}

// Scheduler that tracks dependencies and sends ready work to the shared queue
fn run_scheduler(
    topology: &NetworkTopology,
    scheduler_rx: Receiver<SchedulerMessage>,
    work_tx: Sender<WorkerMessage>,
    num_workers: usize,
) -> Result<()> {
    // Track which nodes are ready to process
    let mut ready_nodes = VecDeque::new();
    let mut pending_upstreams_count: FxHashMap<u32, usize> = FxHashMap::default();

    // Initialize with leaf nodes (no upstream dependencies)
    for node_id in topology.nodes.keys() {
        let upstream_count = topology.upstream_counts.get(node_id).copied().unwrap_or(0);
        if upstream_count == 0 {
            ready_nodes.push_back(*node_id);
        } else {
            // Count how many upstream nodes need to complete
            pending_upstreams_count.insert(*node_id, upstream_count);
        }
    }

    let mut pending_runs = 0;

    loop {
        // Send ready work to workers
        while let Some(node_id) = ready_nodes.pop_front() {
            // Idle workers pull from the shared queue
            if let Err(e) = work_tx.send(WorkerMessage::ProcessNode(node_id)) {
                eprintln!("Failed to send work to workers: {}", e);
            }
            pending_runs += 1;
        }

        // Wait for completion messages
        match scheduler_rx.recv() {
            Ok(SchedulerMessage::NodeCompleted(node_id)) => {
                pending_runs -= 1;
                // Check if this enables any downstream nodes
                if let Some(node) = topology.nodes.get(&node_id) {
                    if let Some(count) = pending_upstreams_count.get_mut(&node.downstream_id) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            // Prioritize downstream nodes to free inflow buffers sooner
                            ready_nodes.push_front(node.downstream_id);
                            pending_upstreams_count.remove(&node.downstream_id);
                        }
                    }
                }
            }
            Ok(SchedulerMessage::Shutdown) => break,
            Err(e) => {
                eprintln!("Scheduler channel error: {}", e);
                break;
            }
        }
        if pending_runs == 0 && ready_nodes.is_empty() {
            break;
        }
    }

    // Send shutdown to all workers
    for _ in 0..num_workers {
        let _ = work_tx.send(WorkerMessage::Shutdown);
    }

    Ok(())
}

// Downsample full-resolution results to output frequency
fn downsample_results(results: SimulationResults, downsampling: usize) -> SimulationResults {
    if downsampling <= 1 {
        return results;
    }
    let actual_timesteps = results.flow_data.len();
    let mut downsampled =
        SimulationResults::with_capacity(results.feature_id, actual_timesteps / downsampling);
    for i in (downsampling - 1..actual_timesteps).step_by(downsampling) {
        downsampled.flow_data.push(results.flow_data[i]);
        downsampled.velocity_data.push(results.velocity_data[i]);
        downsampled.depth_data.push(results.depth_data[i]);
    }
    downsampled
}

// Shared, read-only state each worker needs to route a node
struct WorkerContext {
    kernel: MuskingumCungeKernel,
    topology: Arc<NetworkTopology>,
    channel_params_map: Arc<FxHashMap<u32, ChannelParams>>,
    max_timesteps: usize,
    dt: f32,
    downsampling: usize,
    writer_tx: Sender<WriterMessage>,
    scheduler_tx: Sender<SchedulerMessage>,
    progress_bar: ProgressBar,
}

impl WorkerContext {
    // Route one node, pass its flow downstream and send its results to the writer
    fn process_node(&self, node_id: u32) -> Result<()> {
        let Some(params) = self.channel_params_map.get(&node_id) else {
            return Ok(());
        };
        match process_node_all_timesteps(
            self.kernel,
            node_id,
            &self.topology,
            params,
            self.max_timesteps,
            self.dt,
        ) {
            Ok(results) => {
                // Pass full-resolution flow to downstream node
                if let Some(downstream_node) = self
                    .topology
                    .nodes
                    .get(&node_id)
                    .and_then(|node| self.topology.nodes.get(&node.downstream_id))
                {
                    downstream_node
                        .add_inflow(&results.flow_data)
                        .context("Failed to update downstream inflow")?;
                }

                // Downsample then send to writer
                let downsampled = downsample_results(results, self.downsampling);
                if let Err(e) = self.writer_tx.send(WriterMessage::WriteResults(downsampled)) {
                    eprintln!("Failed to send results to writer: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Error processing node {}: {:#}", node_id, e);
                self.writer_tx.send(WriterMessage::Shutdown).ok();
                self.scheduler_tx.send(SchedulerMessage::Shutdown).ok();
            }
        }
        self.progress_bar.inc(1);
        Ok(())
    }
}

// Worker thread - receives work from the shared queue and processes it
fn worker_thread(ctx: WorkerContext, work_rx: Receiver<WorkerMessage>) -> Result<()> {
    loop {
        match work_rx.recv() {
            Ok(WorkerMessage::ProcessNode(node_id)) => {
                ctx.process_node(node_id)?;

                // Notify scheduler that node is complete
                if let Err(e) = ctx.scheduler_tx.send(SchedulerMessage::NodeCompleted(node_id)) {
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
    topology: Arc<NetworkTopology>,
    channel_params_map: Arc<FxHashMap<u32, ChannelParams>>,
    max_timesteps: usize,
    dt: f32,
    downsampling: usize,
    output: NetCdfWriter,
    progress_bar: ProgressBar,
    num_threads: usize,
) -> Result<()> {
    let total_nodes = topology.nodes.len();

    // Create channels
    let (writer_tx, writer_rx) = cb::unbounded();
    let (scheduler_tx, scheduler_rx) = cb::unbounded();

    println!(
        "Using {} worker threads for parallel processing",
        num_threads
    );

    // Single work queue shared by all workers
    let (work_tx, work_rx) = cb::unbounded();
    let mut worker_handles = Vec::new();

    // Spawn worker threads
    for i in 0..num_threads {
        let ctx = WorkerContext {
            kernel,
            topology: Arc::clone(&topology),
            channel_params_map: Arc::clone(&channel_params_map),
            max_timesteps,
            dt,
            downsampling,
            writer_tx: writer_tx.clone(),
            scheduler_tx: scheduler_tx.clone(),
            progress_bar: progress_bar.clone(),
        };
        let work_rx = work_rx.clone();

        let handle = thread::spawn(move || {
            if let Err(e) = worker_thread(ctx, work_rx) {
                eprintln!("Worker {} error: {}", i, e);
            }
        });
        worker_handles.push(handle);
    }

    // Spawn writer thread; it owns the output file from here on
    let writer_handle = thread::spawn(move || {
        if let Err(e) = writer_thread(writer_rx, output, min(100, total_nodes)) {
            eprintln!("Writer thread error: {}", e);
        }
    });

    // Drop original senders/receivers
    drop(writer_tx);
    drop(scheduler_tx);
    drop(work_rx);

    // Run the scheduler on this thread until all nodes are done
    run_scheduler(&topology, scheduler_rx, work_tx, num_threads)?;

    // Wait for all threads to complete
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
