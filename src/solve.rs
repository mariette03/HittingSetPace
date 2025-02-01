use crate::{
    instance::{Instance, NodeIdx, EdgeIdx},
    lower_bound::{self, PackingBound},
    reductions::{self, ReductionResult},
    report::{ReductionStats, Report, RootBounds, RuntimeStats, Settings, UpperBoundImprovement},
    small_indices::{IdxHashSet, SmallIdx},
};
use anyhow::{ensure, Result};
use log::{debug, info, trace, warn};
use std::time::Instant;

use std::collections::HashSet;

const ILP_LIMIT: usize = 75;

/// ***************************************************************
/// LP STUFF

extern crate glpk_sys as glpk;

use std::ffi::CString;
use std::ptr;
use std::os::raw::*;

const GLP_MIN: i32 = 1;  // Minimization objective
// const GLP_MAX: i32 = 2;  // Maximization objective

const GLP_LO: i32 = 2;   // Lower bound
// const GLP_UP: i32 = 3;   // Upper bound
const GLP_FX: i32 = 5;   // FIXED

const GLP_BV: i32 = 3;   // Binary variable
const GLP_DB: i32 = 4;   // Double bounded
// const GLP_OPT: i32 = 5;
/// ***************************************************************

const ITERATION_LOG_INTERVAL_SECS: u64 = 60;

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct State {
    pub partial_hs: Vec<NodeIdx>,
    pub minimum_hs: Vec<NodeIdx>,
    pub solve_start_time: Instant,
    pub last_log_time: Instant,
    pub conflict_scores: HashMap<NodeIdx, f64>,
    pub decay_factor: f64,
    pub global_lower_bound: usize,
    pub ilp_matrix: ILPMatrix
}

impl State {
    fn apply_decay(&mut self) {
        for score in self.conflict_scores.values_mut() {
            *score *= self.decay_factor;
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Status {
    /// Continue solving to search for smaller hitting sets
    Continue,

    /// A hitting set smaller or equal to the stopping size has been found
    Stop,
}

fn branch_on(
    node: NodeIdx,
    instance: &mut Instance,
    state: &mut State,
    report: &mut Report,
    vertex_importance: &mut Vec<f64>,
) -> Status {
    trace!("Branching on {}", node);
    report.branching_steps += 1;

    if report.branching_steps % 64 == 0 {
        state.apply_decay();
    }

    instance.delete_node(node);

    instance.delete_incident_edges(node);
    state.partial_hs.push(node);
    let status_with_vertex_in_hs = solve_recursive(instance, state, report, vertex_importance);
    debug_assert_eq!(state.partial_hs.last().copied(), Some(node));
    state.partial_hs.pop();
    instance.restore_incident_edges(node);

    // Apply conflict score
    if status_with_vertex_in_hs == Status::Stop {
        *state.conflict_scores.entry(node).or_insert(0.0) += 1.0;
    }

    if status_with_vertex_in_hs == Status::Stop {
        instance.restore_node(node);
        return Status::Stop;
    }

    let status_without_vertex_in_hs = solve_recursive(instance, state, report, vertex_importance);
    instance.restore_node(node);

    // // Apply conflict score
    // if status_with_vertex_in_hs == Status::Stop {
    //     *state.conflict_scores.entry(node).or_insert(0.0) += 1.0;
    // }

    status_without_vertex_in_hs
}

fn select_vertex(instance: &Instance, state: &State, vertex_importance: &[f64]) -> NodeIdx {
    instance
        .nodes()
        .iter()
        .copied()
        .max_by(|&a, &b| {
            let score_a = vertex_importance[a.idx()] * instance.node_degree(a) as f64;
            let score_b = vertex_importance[b.idx()] * instance.node_degree(b) as f64;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("Branching on an empty instance")
    // instance
    //     .nodes()
    //     .iter()
    //     .copied()
    //     .max_by(|&a, &b| {
    //         let score_a = vertex_importance[a.idx()] * (instance.node_degree(a) as f64 + state.conflict_scores.get(&a).copied().unwrap_or(1.0));
    //         let score_b = vertex_importance[b.idx()] * (instance.node_degree(b) as f64 + state.conflict_scores.get(&b).copied().unwrap_or(1.0));

    //         (score_a).partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
    //     })
    //     .expect("Branching on an empty instance")
}

fn solve_recursive(instance: &mut Instance, state: &mut State, report: &mut Report, vertex_importance: &mut Vec<f64>) -> Status {
    let now = Instant::now();
    if (now - state.last_log_time).as_secs() >= ITERATION_LOG_INTERVAL_SECS {
        info!(
            "Running on {} for {} branching steps",
            &report.file_name, report.branching_steps
        );
        state.last_log_time = now;
    }

    if state.minimum_hs.len() <= state.global_lower_bound {
        assert!(state.minimum_hs.len() == state.global_lower_bound);
        info!("Hit global lower bound computed by LP");
        return Status::Stop;
    }

    if state.partial_hs.len() >= state.minimum_hs.len() {
        return Status::Continue;
    }

    let (reduction_result, reduction) = reductions::reduce(instance, state, report);

    let status = match reduction_result {
        ReductionResult::Solved => {
            if state.partial_hs.len() < state.minimum_hs.len() {
                info!("Found HS of size {} by branching", state.partial_hs.len());
                state.minimum_hs.clear();
                state.minimum_hs.extend(state.partial_hs.iter().copied());
                report.upper_bound_improvements.push(UpperBoundImprovement {
                    new_bound: state.minimum_hs.len(),
                    branching_steps: report.branching_steps,
                    runtime: state.solve_start_time.elapsed(),
                });
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
            if instance.num_edges() <= ILP_LIMIT {
                let before = Instant::now();
                let (ilp_result, ilp_solution) = solve_ilp_exact(instance, state);
                report.runtimes.ilp_exact += before.elapsed();

                if ilp_result > 0 {
                    if state.partial_hs.len() + ilp_result < state.minimum_hs.len() {
                        info!("Found HS of size {} by ILP", state.partial_hs.len() + ilp_result);
                        state.minimum_hs.clear();
                        state.minimum_hs.extend(state.partial_hs.iter().copied().chain(ilp_solution.iter().copied()));
                    }
                    report.reductions.ilp_breaks += 1;
                    reduction.restore(instance, &mut state.partial_hs);
                    return Status::Continue;
                }
            }
            let node = select_vertex(instance, state, &vertex_importance);
            branch_on(node, instance, state, report, vertex_importance)
        }
    };

    reduction.restore(instance, &mut state.partial_hs);
    status
}

fn is_hitting_set(hs: &[NodeIdx], instance: &Instance) -> bool {
    let hs_set: IdxHashSet<_> = hs.iter().copied().collect();
    instance
        .edges()
        .iter()
        .all(|&edge| instance.edge(edge).any(|node| hs_set.contains(&node)))
}

fn calculate_root_bounds(instance: &Instance, settings: &Settings) -> RootBounds {
    let num_nodes = instance.num_nodes_total();
    let root_packing = PackingBound::new(instance, settings);
    RootBounds {
        max_degree: lower_bound::calc_max_degree_bound(instance).unwrap_or(num_nodes),
        sum_degree: lower_bound::calc_sum_degree_bound(instance),
        efficiency: lower_bound::calc_efficiency_bound(instance)
            .0
            .round()
            .unwrap_or(num_nodes),
        packing: root_packing.bound(),
        sum_over_packing: root_packing.calc_sum_over_packing_bound(instance),
        greedy_upper: reductions::calc_greedy_approximation(instance).len(),
    }
}

pub fn solve_lp(instance: &Instance) -> (usize, Vec<f64>) {
    unsafe {
        glpk::glp_term_out(0);

        let lp = glpk::glp_create_prob();

        glpk::glp_set_prob_name(lp, CString::new("hitting_set_lp").unwrap().as_ptr());
        glpk::glp_set_obj_dir(lp, GLP_MIN);

        let num_sets = instance.num_edges_total();
        glpk::glp_add_rows(lp, num_sets as i32);

        let num_elements = instance.num_nodes_total();
        glpk::glp_add_cols(lp, num_elements as i32);

        let mut total_size = 0;

        for i in 0..num_elements {
            let col_idx = (i + 1) as i32;

            glpk::glp_set_col_bnds(lp, col_idx, GLP_DB, 0.0, 1.0);
            glpk::glp_set_obj_coef(lp, col_idx, 1.0);

            total_size += instance.node_degree(NodeIdx::from(i));
        }

        for j in 0..num_sets {
            let row_idx = (j + 1) as i32;

            glpk::glp_set_row_bnds(lp, row_idx, GLP_LO, 1.0, f64::INFINITY);
        }

        let mut ia: Vec<c_int> = Vec::with_capacity(total_size);
        let mut ja: Vec<c_int> = Vec::with_capacity(total_size);
        let mut ar: Vec<c_double> = Vec::with_capacity(total_size);

        ia.push(0 as c_int);
        ja.push(0 as c_int);
        ar.push(0.0 as c_double);

        for j in 0..num_sets {
            let row_idx = (j + 1) as c_int;

            for element in instance.edge(EdgeIdx::from(j)) {
                let col_idx = (element.idx() + 1) as c_int;

                ia.push(row_idx);
                ja.push(col_idx);
                ar.push(1.0);
            }
        }

        assert!(ia.len() == ja.len());
        assert!(ia.len() == ar.len());

        glpk::glp_load_matrix(lp, (ia.len() - 1) as i32, ia.as_ptr(), ja.as_ptr(), ar.as_ptr());
        
        // glpk::glp_write_lp(lp, ptr::null(), CString::new("init_debug.lp").unwrap().as_ptr());

        glpk::glp_simplex(lp, ptr::null());

        let z = glpk::glp_get_obj_val(lp).ceil() as usize;
        info!("LP: Objective value (z) = {}", z);

        let mut vertex_importance = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let col_idx = (i + 1) as i32;
            let value = glpk::glp_get_col_prim(lp, col_idx);
            vertex_importance.push(value);
        }

        glpk::glp_delete_prob(lp);

        (z, vertex_importance)
    }
}

/*
pub fn solve_lp_mu(instance: &Instance, eta: f64, iterations: usize) -> Vec<f64> {
    let num_elements = instance.num_nodes_total();

    // Initialize weights for each element to 1.0
    let mut weights: Vec<f64> = vec![1.0; num_elements];

    for _ in 0..iterations {
        // Sum of the weights
        let total_weight: f64 = weights.iter().sum();

        // Iterate over each set and apply MWU
        for (j, edge) in instance.edges().iter().enumerate() {
            // _row_idx is unused, so prefix with an underscore
            let _row_idx = (j + 1) as c_int;

            // Directly collect the set from edge without iter()
            let set: Vec<usize> = instance.edge(*edge).map(|node| node.idx()).collect();

            // Check if the set is covered
            let is_covered = set.iter().any(|&node_idx| weights[node_idx] > 0.5 * total_weight);

            if !is_covered {
                // If the set is not covered, boost weights of the elements in the set
                for &node_idx in &set {
                    weights[node_idx] *= 1.0 + eta; // Boost the weight
                }
            } else {
                // If the set is covered, decay the weights of the elements in the set
                for &node_idx in &set {
                    weights[node_idx] *= 1.0 - eta; // Decay the weight
                }
            }
        }
    }

    let total_weight: f64 = weights.iter().sum();
    let vertex_importance: Vec<f64> = weights
        .iter()
        .map(|&weight| weight / total_weight)
        .collect();

    vertex_importance
}
*/

/// ILP SOLVER
#[derive(Debug, Clone)] // Derive Clone for ILPMatrix
pub struct ILPMatrix {
    pub ia: Vec<c_int>,  // Make fields public
    pub ja: Vec<c_int>,
    pub ar: Vec<c_double>,
    pub num_rows: i32,   // Make fields public
    pub num_cols: i32,   // Make fields public
}

impl ILPMatrix {
    pub fn new(instance: &Instance) -> Self {  // Make method public
        let num_sets = instance.num_edges();
        let num_elements = instance.num_nodes_total();

        let mut matrix = Self {
            ia: vec![0],
            ja: vec![0],
            ar: vec![0.0],
            num_rows: num_sets as i32,
            num_cols: num_elements as i32,
        };

        for (j, &edge) in instance.edges().iter().enumerate() {
            let row_idx = (j + 1) as c_int;
            for node in instance.edge(edge) {
                let col_idx = (node.idx() + 1) as c_int;
                matrix.ia.push(row_idx);
                matrix.ja.push(col_idx);
                matrix.ar.push(1.0);
            }
        }

        matrix
    }

    pub fn load_into_glpk(&self, lp: *mut glpk::glp_prob) {  // Make method public
        unsafe {
            glpk::glp_load_matrix(lp, (self.ia.len() - 1) as i32, self.ia.as_ptr(), self.ja.as_ptr(), self.ar.as_ptr());
        }
    }
}

pub fn solve_ilp_exact(instance: &mut Instance, state: &mut State) -> (usize, Vec<NodeIdx>) {
    unsafe {
        let partial_hs_map: HashSet<NodeIdx> = state.partial_hs.iter().cloned().collect();

        glpk::glp_term_out(0);

        let lp = glpk::glp_create_prob();
        glpk::glp_set_prob_name(lp, CString::new("hitting_set_ilp").unwrap().as_ptr());
        glpk::glp_set_obj_dir(lp, GLP_MIN);

        glpk::glp_add_rows(lp, state.ilp_matrix.num_rows);
        glpk::glp_add_cols(lp, state.ilp_matrix.num_cols);

        for i in 0..state.ilp_matrix.num_cols {
            let col_idx = (i + 1) as i32;

            let node = NodeIdx::from(i as usize);

            if partial_hs_map.contains(&node) {
                glpk::glp_set_col_bnds(lp, col_idx, GLP_FX, 1.0, 1.0);
            } else if instance.is_node_deleted(node) {
                glpk::glp_set_col_bnds(lp, col_idx, GLP_FX, 0.0, 0.0);
            } else {
                glpk::glp_set_col_bnds(lp, col_idx, GLP_DB, 0.0, 1.0);
                glpk::glp_set_obj_coef(lp, col_idx, 1.0);
                glpk::glp_set_col_kind(lp, col_idx, GLP_BV);
            }
        }

        for j in 0..state.ilp_matrix.num_rows {
            let row_idx = (j + 1) as i32;
            glpk::glp_set_row_bnds(lp, row_idx, GLP_LO, 1.0, f64::INFINITY);
        }

        state.ilp_matrix.load_into_glpk(lp);
        glpk::glp_scale_prob(lp, 0x80);

        let mut smcp: glpk::glp_smcp = std::mem::zeroed();
        glpk::glp_init_smcp(&mut smcp);
        smcp.presolve = 1;
        if glpk::glp_simplex(lp, &mut smcp) != 0 {
            glpk::glp_delete_prob(lp);
            return (usize::MAX, vec![]);
        }

        let mut iocp: glpk::glp_iocp = std::mem::zeroed();
        glpk::glp_init_iocp(&mut iocp);
        iocp.presolve = 1;
        iocp.mip_gap = 1e-6;
        iocp.tm_lim = 300000;
        iocp.msg_lev = 3;

        if glpk::glp_intopt(lp, &mut iocp) != 0 {
            glpk::glp_delete_prob(lp);
            return (usize::MAX, vec![]);
        }

        let z = glpk::glp_mip_obj_val(lp).ceil() as usize;
        let mut result = Vec::with_capacity(z);
        for i in 0..state.ilp_matrix.num_cols {
            let col_idx = (i + 1) as i32;
            let node = NodeIdx::from(i as usize);

            if glpk::glp_mip_col_val(lp, col_idx) as i32 == 1 && !partial_hs_map.contains(&node) {
                result.push((i as usize).into());
            }
        }

        glpk::glp_delete_prob(lp);
        (z, result)
    }
}

/*
pub fn solve_ilp_exact(instance: &Instance) -> (usize, Vec<NodeIdx>) {
    unsafe {
        glpk::glp_term_out(0);

        let lp = glpk::glp_create_prob();
        glpk::glp_set_prob_name(lp, CString::new("hitting_set_ilp").unwrap().as_ptr());
        glpk::glp_set_obj_dir(lp, GLP_MIN);

        let num_sets = instance.num_edges();
        glpk::glp_add_rows(lp, num_sets as i32);

        let num_elements = instance.num_nodes_total();
        glpk::glp_add_cols(lp, num_elements as i32);

        for i in 0..num_elements {
            let col_idx = (i + 1) as i32;

            glpk::glp_set_col_bnds(lp, col_idx, GLP_DB, 0.0, 1.0);
            glpk::glp_set_obj_coef(lp, col_idx, 1.0);
            glpk::glp_set_col_kind(lp, col_idx, GLP_BV);
        }

        for j in 0..num_sets {
            let row_idx = (j + 1) as i32;

            glpk::glp_set_row_bnds(lp, row_idx, GLP_LO, 1.0, f64::INFINITY);
        }

        let mut ia: Vec<c_int> = Vec::new();
        let mut ja: Vec<c_int> = Vec::new();
        let mut ar: Vec<c_double> = Vec::new();

        ia.push(0 as c_int);
        ja.push(0 as c_int);
        ar.push(0.0 as c_double);

        for (j, &edge) in instance.edges().into_iter().enumerate() {
            let row_idx = (j + 1) as c_int;
            assert!(j+1 <= num_sets);

            for node in instance.edge(edge) {
                let col_idx = (node.idx() + 1) as c_int;
                assert!(node.idx() + 1 <= num_elements);

                ia.push(row_idx);
                ja.push(col_idx);
                ar.push(1.0);
            }
        }

        assert!(ia.len() == ja.len());
        assert!(ia.len() == ar.len());

        glpk::glp_load_matrix(lp, (ia.len() - 1) as i32, ia.as_ptr(), ja.as_ptr(), ar.as_ptr());

        glpk::glp_scale_prob(lp, 0x80); 

        // glpk::glp_write_lp(lp, ptr::null(), CString::new("debug.ilp").unwrap().as_ptr());

        let mut smcp: glpk::glp_smcp = std::mem::zeroed();
        glpk::glp_init_smcp(&mut smcp);
        smcp.presolve = 1;
        let simplex_status = glpk::glp_simplex(lp, &mut smcp);
        if simplex_status != 0 {
            info!("Simplex phase failed with status: {}", simplex_status);
            return (usize::MAX, vec![]);
        }

        let mut iocp: glpk::glp_iocp = std::mem::zeroed();
        glpk::glp_init_iocp(&mut iocp);
        iocp.presolve = 1;
        iocp.mip_gap = 1e-6;
        iocp.tm_lim = 300000;
        iocp.msg_lev = 3;

        let ilp_status = glpk::glp_intopt(lp, &mut iocp);
        if ilp_status != 0 {
            info!("GLPK Integer Optimization failed. Status: {}", ilp_status);
            return (usize::MAX, vec![]);
        }

        let z = glpk::glp_mip_obj_val(lp).ceil() as usize;
        // info!("ILP Objective value (z) = {}", z);

        let mut result = Vec::with_capacity(z);
        for i in 0..num_elements {
            let col_idx = (i + 1) as i32;
            if glpk::glp_mip_col_val(lp, col_idx) as i32 == 1 {
                result.push(NodeIdx::from(i));
            }
        }

        glpk::glp_delete_prob(lp);
        (z, result)
    }
}
*/

/*
pub fn solve_ilp(instance: &Instance) -> (usize, Vec<NodeIdx>) {
    unsafe {
        glpk::glp_term_out(0);

        let lp = glpk::glp_create_prob();

        glpk::glp_set_prob_name(lp, CString::new("hitting_set_ilp").unwrap().as_ptr());
        glpk::glp_set_obj_dir(lp, GLP_MIN);

        let num_sets = instance.num_edges_total();
        glpk::glp_add_rows(lp, num_sets as i32);

        let num_elements = instance.num_nodes_total();
        glpk::glp_add_cols(lp, num_elements as i32);

        let mut total_size = 0;
        for i in 0..num_elements {
            let col_idx = (i + 1) as i32;

            glpk::glp_set_col_bnds(lp, col_idx, GLP_DB, 0.0, 1.0);
            glpk::glp_set_obj_coef(lp, col_idx, 1.0);
            glpk::glp_set_col_kind(lp, col_idx, GLP_BV);

            total_size += instance.node_degree(NodeIdx::from(i));
        }

        for j in 0..num_sets {
            let row_idx = (j + 1) as i32;
            glpk::glp_set_row_bnds(lp, row_idx, GLP_LO, 1.0, f64::INFINITY);
        }

        let mut ia: Vec<c_int> = Vec::with_capacity(total_size + 1);
        let mut ja: Vec<c_int> = Vec::with_capacity(total_size + 1);
        let mut ar: Vec<c_double> = Vec::with_capacity(total_size + 1);

        ia.push(0 as c_int);
        ja.push(0 as c_int);
        ar.push(0.0 as c_double);

        for j in 0..num_sets {
            let row_idx = (j + 1) as c_int;

            for element in instance.edge(EdgeIdx::from(j)) {
                let col_idx = (element.idx() + 1) as c_int;

                ia.push(row_idx);
                ja.push(col_idx);
                ar.push(1.0);
            }
        }

        assert!(ia.len() == ja.len());
        assert!(ia.len() == ar.len());

        glpk::glp_load_matrix(lp, (ia.len()-1) as i32, ia.as_ptr(), ja.as_ptr(), ar.as_ptr());

        // glpk::glp_scale_prob(lp, 0x80);
        // glpk::glp_write_lp(lp, ptr::null(), CString::new("debug.ilp").unwrap().as_ptr());

        // let simplex_status = glpk::glp_simplex(lp, ptr::null());
        // if simplex_status != 0 {
        //     info!("Simplex phase failed with status: {}", simplex_status);
        //     return (usize::MAX, vec![]);
        // }
        
        let ilp_status = glpk::glp_intopt(lp, ptr::null());
        if ilp_status != 0 {
            info!("GLPK Integer Optimization failed. Status: {}", ilp_status);
            return (usize::MAX, vec![]);
        }

        let z = glpk::glp_mip_obj_val(lp).ceil() as usize;
        info!("ILP Objective value (z) = {}", z);

        let mut result = Vec::with_capacity(z);

        for i in 0..num_elements {
            let col_idx = (i + 1) as i32;
            let value = glpk::glp_mip_col_val(lp, col_idx) as i32;

            if value == 1 {
                result.push(NodeIdx::from(i));
            }
        }

        glpk::glp_delete_prob(lp);

        (z, result)
    }
}
*/

pub fn solve(
    mut instance: Instance,
    file_name: String,
    settings: Settings,
) -> Result<(Vec<NodeIdx>, Report)> {
    let before = Instant::now();
    let (lp_bound, mut vertex_importance) = solve_lp(&instance);
    let time_spend_lp = before.elapsed();

    let initial_hs = reductions::calc_greedy_approximation_from_vec(&mut instance, &mut vertex_importance);
    ensure!(
        is_hitting_set(&initial_hs, &mut instance),
        "LP-guided greedy HS is not valid"
    );

    info!("Size of LP-guided greedy HS: {}, took {:.2?}", initial_hs.len(), time_spend_lp);

    // let initial_hs = get_initial_hitting_set(&instance, &settings)?;
    let root_bounds = calculate_root_bounds(&instance, &settings);
    let packing_from_scratch_limit = settings.packing_from_scratch_limit;
    let mut report = Report {
        file_name,
        opt: initial_hs.len(),
        branching_steps: 0,
        settings,
        root_bounds,
        runtimes: RuntimeStats::default(),
        reductions: ReductionStats::new(packing_from_scratch_limit),
        upper_bound_improvements: Vec::new(),
    };

    let mut state = State {
        partial_hs: Vec::new(),
        minimum_hs: initial_hs,
        last_log_time: Instant::now(),
        solve_start_time: Instant::now(),
        conflict_scores: HashMap::new(),
        decay_factor: 0.95,
        global_lower_bound: lp_bound,
        ilp_matrix: ILPMatrix::new(&instance),
    };

    let _ = solve_recursive(&mut instance, &mut state, &mut report, &mut vertex_importance);
    report.runtimes.total = state.solve_start_time.elapsed();
    report.opt = state.minimum_hs.len();

    info!("Validating found hitting set");
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
