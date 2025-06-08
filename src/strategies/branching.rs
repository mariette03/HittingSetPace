use std::time::Instant;
use log::{info, trace, warn};
use crate::instance::{Instance, InstanceType, NodeIdx};
use crate::{reductions};
use crate::reductions::ReductionResult;
use crate::report::Report;
use crate::solve::{State, Status};
use crate::utils::select_vertex;

pub(crate) fn branch_on(
    node: NodeIdx,
    instance: &mut Instance,
    state: &mut State,
    report: &mut Report,
    vertex_importance: &mut Vec<f64>,
    depth: usize
) -> Status {
    trace!("Branching on {}", node);
    report.branching_steps += 1;
    instance.delete_node(node);

    instance.delete_incident_edges(node);
    state.partial_hs.push(node);
    let status_with_vertex_in_hs = solve_recursive(instance, state, report, vertex_importance, depth+1);
    debug_assert_eq!(state.partial_hs.last().copied(), Some(node));
    state.partial_hs.pop();
    instance.restore_incident_edges(node);

    if status_with_vertex_in_hs == Status::Stop {
        instance.restore_node(node);
        return Status::Stop;
    }

    let status_without_vertex_in_hs = solve_recursive(instance, state, report, vertex_importance, depth+1);
    instance.restore_node(node);

    status_without_vertex_in_hs
}

pub(crate) fn solve_recursive(instance: &mut Instance, state: &mut State, report: &mut Report, vertex_importance: &mut Vec<f64>, depth: usize) -> Status {
    // let now = Instant::now();
    // if (now - state.last_log_time).as_secs() >= ITERATION_LOG_INTERVAL_SECS {
    //     info!(
    //         "Running on {} for {} branching steps",
    //         &report.file_name, report.branching_steps
    //     );
    //     state.last_log_time = now;
    // }

    if state.minimum_hs.len() <= state.global_lower_bound {
        assert_eq!(state.minimum_hs.len(), state.global_lower_bound);
        info!("Hit global lower bound computed by LP");
        return Status::Stop;
    }

    if state.partial_hs.len() >= state.minimum_hs.len() {
        return Status::Continue;
    }

    let (reduction_result, reduction) = reductions::reduce(instance, state, report, depth);

    let status = match reduction_result {
        ReductionResult::Solved => {
            if state.partial_hs.len() < state.minimum_hs.len() {
                info!("Found HS of size {} by branching |V|={}, |E|={}", state.partial_hs.len(), instance.num_nodes(), instance.num_edges());
                state.minimum_hs.clear();
                state.minimum_hs.extend(state.partial_hs.iter().copied());
                // report.upper_bound_improvements.push(UpperBoundImprovement {
                //     new_bound: state.minimum_hs.len(),
                //     branching_steps: report.branching_steps,
                //     runtime: state.solve_start_time.elapsed(),
                // });
            } else {
                warn!(
                    "Found HS is not smaller than best known ({} vs. {}), should have been pruned",
                    state.partial_hs.len(),
                    state.minimum_hs.len(),
                );
            }

            if state.minimum_hs.len() <= report.settings.stop_at {
                Status::Stop
            } else {
                Status::Continue
            }
        }
        ReductionResult::Unsolvable => Status::Continue,
        ReductionResult::Stop => Status::Stop,
        ReductionResult::Finished => {
            let mut use_ilp : bool = false;
            use_ilp |= instance.num_nodes() <= report.settings.ilp_size && (report.instance_type & InstanceType::Dense as u32) != 0 && (report.instance_type & InstanceType::Graph as u32)== 0;
            // // use_ilp |= instance.num_nodes() <= 40 && (report.instance_type & InstanceType::Sparse as u32) != 0;
            // use_ilp |= report.settings.ilp_size != 0 && instance.num_nodes() <= 200 && (report.instance_type & InstanceType::Graph as u32) != 0;

            if use_ilp {
                let before = Instant::now();
                let (ilp_result, ilp_solution) = crate::lp_solver::solve_ilp_exact(instance);
                let elapsed_time = before.elapsed();
                report.runtimes.ilp_exact += elapsed_time;

                if ilp_result != usize::MAX {
                    if state.partial_hs.len() + ilp_result < state.minimum_hs.len() {
                        info!("Found HS of size {} by ILP in {:?} |V|={}, |E|={}", state.partial_hs.len() + ilp_result, elapsed_time, instance.num_nodes(), instance.num_edges());
                        state.minimum_hs.clear();
                        state.minimum_hs.extend(state.partial_hs.iter().copied().chain(ilp_solution.iter().copied()));
                        report.reductions.ilp_improvements += 1;
                    }
                    report.reductions.ilp_breaks += 1;
                    reduction.restore(instance, &mut state.partial_hs);
                    return Status::Continue;
                }
            }
            let node = select_vertex(instance, &vertex_importance);
            branch_on(node, instance, state, report, vertex_importance, depth)
        }
    };

    reduction.restore(instance, &mut state.partial_hs);
    status
}
