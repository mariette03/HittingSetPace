use crate::{
    instance::{Instance, NodeIdx, EdgeIdx, InstanceType},
    reductions::{self, ReductionResult},
    report::{ReductionStats, Report, RuntimeStats, Settings, UpperBoundImprovement},
    small_indices::{IdxHashSet, SmallIdx},
    utils::{is_hitting_set, select_vertex},
    strategies,
    lp_solver};
use anyhow::{ensure, Result};
use log::{debug, info, trace, warn};
use std::time::Instant;
use crate::reductions::optimistic_reductions;

#[derive(Debug, Clone)]
pub struct State {
    pub partial_hs: Vec<NodeIdx>,
    pub minimum_hs: Vec<NodeIdx>,
    pub solve_start_time: Instant,
    pub last_log_time: Instant,
    pub global_lower_bound: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Status {
    /// Continue solving to search for smaller hitting sets
    Continue,

    /// A hitting set smaller or equal to the stopping size has been found
    Stop,
}





pub fn solve(
    mut instance: Instance,
    file_name: String,
    settings: Settings,
) -> Result<(Vec<NodeIdx>, Report)> {
    let packing_from_scratch_limit = settings.packing_from_scratch_limit;
    let use_first_lp = settings.lp_guided;

    let mut report = Report {
        file_name,
        opt: 0,
        branching_steps: 0,
        settings,
        runtimes: RuntimeStats::default(),
        reductions: ReductionStats::new(packing_from_scratch_limit),
        upper_bound_improvements: Vec::new(),
        instance_type: 0,
    };

    if instance.num_edges_total() == 0 || instance.num_nodes_total() == 0 {
        return Ok((Vec::new(), report));
    }

    let instance_type = instance.get_instance_type();
    info!("Instance has type: {:#b}", instance_type);

    let mut hard_instance : bool = instance.is_hard_instance();
    info!("Is instance hard? {}", hard_instance);

    report.instance_type = instance_type;

    let mut initial_hs = Vec::new();
    let mut global_lower_bound = 0;
    let mut vertex_importance = Vec::new();

    // if use_first_lp && !hard_instance {
    if use_first_lp {
        let before = Instant::now();
        let (lp_bound, mut vertex_importance_lp) = lp_solver::solve_lp(&instance);
        let time_spend_lp = before.elapsed();

        global_lower_bound = lp_bound;

        initial_hs = reductions::calc_greedy_approximation_from_vec(&mut instance, &mut vertex_importance_lp);
        ensure!(
            is_hitting_set(&initial_hs, &mut instance),
            "LP-guided greedy HS is not valid"
        );

        vertex_importance = vertex_importance_lp;

        info!("Size of LP-guided greedy HS: {}, took {:.2?}", initial_hs.len(), time_spend_lp);
    } else {
        initial_hs.reserve(instance.num_nodes_total());
        vertex_importance.reserve(instance.num_nodes_total());

        for i in 0..instance.num_nodes_total() {
            initial_hs.push(NodeIdx::from(i));
            vertex_importance.push(1.0);
        }
    }

    report.opt = initial_hs.len();

    let mut state = State {
        partial_hs: Vec::with_capacity(global_lower_bound),
        minimum_hs: initial_hs,
        last_log_time: Instant::now(),
        solve_start_time: Instant::now(),
        global_lower_bound: global_lower_bound,
    };

    /*if hard_instance && (global_lower_bound + 5 < state.minimum_hs.len()) {
        let (_ilp_result, ilp_solution) = lp_solver::solve_ilp_exact(&instance);

        state.minimum_hs.clear();
        state.minimum_hs.extend(ilp_solution.iter().copied());
    } else {*/
    let be_optimistic = true;
    if be_optimistic {
        optimistic_reductions(&mut instance, &mut state, &vertex_importance);
    }
    
    let _ = strategies::branching::solve_recursive(&mut instance, &mut state, &mut report, &mut vertex_importance, 0);
    // }
    
    report.runtimes.total = state.solve_start_time.elapsed();
    report.opt = state.minimum_hs.len();

    // info!("Validating found hitting set");
    assert_eq!(instance.num_nodes_total(), instance.nodes().len());
    assert_eq!(instance.num_edges_total(), instance.edges().len());
    assert!(is_hitting_set(&state.minimum_hs, &instance));

    info!(
        "Found minimum hitting set in {:.2?} and {} branching steps",
        report.runtimes.total, report.branching_steps
    );
    debug!("Final HS (size {}): {:?}", report.opt, &state.minimum_hs);

    Ok((state.minimum_hs, report))
}
