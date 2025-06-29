use crate::{
    data_structures::{subset_trie::SubsetTrie, superset_trie::SupersetTrie},
    instance::{EdgeIdx, Instance, NodeIdx},
    lower_bound::{self, EfficiencyBound, PackingBound},
    lp_solver,
    report::{GreedyMode, Report, Settings},
    small_indices::{IdxHashSet, SmallIdx},
    solve::State,
};
use log::info;
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    time::{Duration, Instant},
};

#[derive(Copy, Clone, Debug)]
pub enum ReducedItem {
    RemovedNode(NodeIdx),
    RemovedEdge(EdgeIdx),
    ForcedNode(NodeIdx),
}

impl ReducedItem {
    pub fn apply(self, instance: &mut Instance, partial_hs: &mut Vec<NodeIdx>) {
        match self {
            Self::RemovedNode(node) => instance.delete_node(node),
            Self::RemovedEdge(edge) => instance.delete_edge(edge),
            Self::ForcedNode(node) => {
                instance.delete_node(node);
                instance.delete_incident_edges(node);
                partial_hs.push(node);
            }
        }
    }

    fn restore(self, instance: &mut Instance, partial_hs: &mut Vec<NodeIdx>) {
        match self {
            Self::RemovedNode(node) => instance.restore_node(node),
            Self::RemovedEdge(edge) => instance.restore_edge(edge),
            Self::ForcedNode(node) => {
                instance.restore_incident_edges(node);
                instance.restore_node(node);
                debug_assert_eq!(partial_hs.last().copied(), Some(node));
                partial_hs.pop();
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Reduction(Vec<ReducedItem>);

impl Reduction {
    pub fn restore(&self, instance: &mut Instance, partial_hs: &mut Vec<NodeIdx>) {
        for item in self.0.iter().rev() {
            item.restore(instance, partial_hs);
        }
    }
}

#[derive(Debug, Clone)]
pub enum ReductionResult {
    /// The instance was fully solved through reductions.
    ///
    /// The current partial hitting set is now a valid hitting set and should
    /// be compared to the current minimum hitting set.
    Solved,

    /// There exists no smaller hitting set than the current minimum hitting set.
    ///
    /// This occurs when a lower bound meets or exceeds the size of the current
    /// minimum hitting set (the upper bound).
    Unsolvable,

    /// A minimum hitting set smaller or equal to the stopping size was found.
    Stop,

    /// No further progress can be made using reductions.
    Finished,
}

pub fn optimistic_reductions(
    instance: &mut Instance,
    state: &mut State,
    vertex_importance: &Vec<f64>,
) -> impl Iterator<Item = ReducedItem> {
    let res: Vec<ReducedItem> = instance.nodes().iter().filter_map(
            | node | {
                if vertex_importance[node.idx()] == 0f64 {
                    Some(ReducedItem::RemovedNode(*node))
                } else if vertex_importance[node.idx()] == 1f64 {
                    Some(ReducedItem::ForcedNode(*node))
                } else {
                    None
                }      
            }
        ).collect();
    
        
    res.into_iter()
}

pub fn optimistic_lp_reductions(instance: &mut Instance) -> impl Iterator<Item = ReducedItem> + '_ {
    let mut res: Vec<_> = Vec::new();
    if instance.num_edges() > 0 && instance.num_nodes() > 0 {
        let (lp_bound, vertex_importance_lp) = lp_solver::solve_lp(&instance);
        //info!("A: {}, B: {}", instance.num_nodes(), vertex_importance_lp.len());
        res = instance.nodes().iter().filter_map(
            | node | {
                if vertex_importance_lp[node.idx()] == 0f64 {
                    Some(ReducedItem::RemovedNode(*node))
                } else if vertex_importance_lp[node.idx()] == 1f64 {
                    Some(ReducedItem::ForcedNode(*node))
                } else {
                    None
                }      
            }
        ).collect();
    }
    
    res.into_iter()
    
    // 
    // let (lp_bound, vertex_importance_lp) = lp_solver::solve_lp(&instance);
    // vertex_importance_lp
    //     .into_iter()
    //     .enumerate()
    //     .filter_map(|(idx, importance)| {
    //         let node = NodeIdx::from(idx);
    //         if importance == 0f64 {
    //             Some(ReducedItem::RemovedNode(node))
    //         } else if importance == 1f64 {
    //             Some(ReducedItem::ForcedNode(node))
    //         } else {
    //             None
    //         }
    //     })
}

fn find_dominated_nodes(instance: &Instance) -> impl Iterator<Item = ReducedItem> + '_ {
    let mut nodes = instance.nodes().to_vec();
    nodes.sort_unstable_by_key(|&node| Reverse(instance.node_degree(node)));
    let mut trie = SupersetTrie::new(instance.num_edges_total());
    nodes.into_iter().filter_map(move |node| {
        if trie.contains_superset(instance.node(node)) {
            Some(ReducedItem::RemovedNode(node))
        } else {
            trie.insert(instance.node(node));
            None
        }
    })
}

fn find_dominated_edges(instance: &Instance) -> impl Iterator<Item = ReducedItem> + '_ {
    let mut edges = instance.edges().to_vec();
    edges.sort_unstable_by_key(|&edge| instance.edge_size(edge));
    let mut trie = SubsetTrie::new(instance.num_nodes_total());
    edges.into_iter().filter_map(move |edge| {
        if trie.find_subset(instance.edge(edge)) {
            Some(ReducedItem::RemovedEdge(edge))
        } else {
            trie.insert(true, instance.edge(edge));
            None
        }
    })
}

use std::collections::HashSet;

fn find_discard_and_forced_vertex(instance: &Instance) -> impl Iterator<Item = ReducedItem> {
    let mut marked_for_removal = HashSet::new();
    let mut forced_nodes = HashSet::new();

    let discard_candidates: Vec<_> = instance
        .nodes()
        .iter()
        .copied()
        .filter(|&node| instance.node_degree(node) <= 1)
        .collect();

    for &node in &discard_candidates {
        if instance.node_degree(node) == 0 {
            marked_for_removal.insert(node);
            continue;
        }
        let mut edges = instance.node(node);
        let edge = edges.next().unwrap();
        let edge_nodes: Vec<_> = instance.edge(edge).collect();

        if edge_nodes
            .iter()
            .any(|&w| w != node && !marked_for_removal.contains(&w))
        {
            marked_for_removal.insert(node);
        } else {
            forced_nodes.insert(node);
        }
    }

    marked_for_removal
        .into_iter()
        .map(ReducedItem::RemovedNode)
        .chain(forced_nodes.into_iter().map(ReducedItem::ForcedNode))
}

fn find_discard_vertex(instance: &Instance) -> impl Iterator<Item = ReducedItem> {
    let mut marked_for_removal = HashSet::new();

    let discard_candidates: Vec<_> = instance
        .nodes()
        .iter()
        .copied()
        .filter(|&node| instance.node_degree(node) <= 1)
        .collect();

    for &node in &discard_candidates {
        if instance.node_degree(node) == 0 {
            marked_for_removal.insert(node);
            continue;
        }
        let mut edges = instance.node(node);
        let edge = edges.next().unwrap();
        let edge_nodes: Vec<_> = instance.edge(edge).collect();

        if edge_nodes
            .iter()
            .any(|&w| w != node && !marked_for_removal.contains(&w))
        {
            marked_for_removal.insert(node);
        }
    }

    marked_for_removal.into_iter().map(ReducedItem::RemovedNode)
}

fn find_forced_nodes(instance: &Instance) -> impl Iterator<Item = ReducedItem> {
    let forced: IdxHashSet<_> = instance
        .edges()
        .iter()
        .copied()
        .filter_map(|edge| {
            let mut edge_nodes_iter = instance.edge(edge);
            edge_nodes_iter.next().and_then(|first_node| {
                if edge_nodes_iter.next().is_some() {
                    None
                } else {
                    Some(first_node)
                }
            })
        })
        .collect();
    forced.into_iter().map(ReducedItem::ForcedNode)
}

fn find_costly_discards_using_efficiency_bound<'a>(
    instance: &'a Instance,
    lower_bound_breakpoint: usize,
    discard_efficieny_bounds: &'a [EfficiencyBound],
) -> impl Iterator<Item = ReducedItem> + 'a {
    instance
        .nodes()
        .iter()
        .copied()
        .filter(move |node| {
            discard_efficieny_bounds[node.idx()]
                .round()
                .unwrap_or(usize::MAX)
                >= lower_bound_breakpoint
        })
        .map(ReducedItem::ForcedNode)
}

fn find_costly_discards_using_packing_update<'a>(
    instance: &'a Instance,
    lower_bound_breakpoint: usize,
    packing_bound: &'a PackingBound,
) -> impl Iterator<Item = ReducedItem> + 'a {
    packing_bound
        .calc_discard_bounds(instance)
        .filter_map(move |(node, new_bound)| {
            if new_bound >= lower_bound_breakpoint {
                Some(ReducedItem::ForcedNode(node))
            } else {
                None
            }
        })
}

fn find_costly_discard_using_packing_from_scratch(
    instance: &mut Instance,
    lower_bound_breakpoint: usize,
    settings: &Settings,
) -> Option<(ReducedItem, usize)> {
    if settings.packing_from_scratch_limit == 0 {
        return None;
    }

    let mut nodes = instance.nodes().to_vec();
    nodes.sort_unstable_by_key(|&node| Reverse(instance.node_degree(node)));
    nodes
        .into_iter()
        .take(settings.packing_from_scratch_limit)
        .enumerate()
        .find_map(|(idx, node)| {
            instance.delete_node(node);
            let packing_bound = PackingBound::new(instance, settings);
            let new_lower_bound = if settings.enable_sum_over_packing_bound {
                packing_bound.calc_sum_over_packing_bound(instance)
            } else {
                packing_bound.bound()
            };
            instance.restore_node(node);

            if new_lower_bound >= lower_bound_breakpoint {
                Some((ReducedItem::ForcedNode(node), idx))
            } else {
                None
            }
        })
}

use std::cmp::Ordering;

#[derive(PartialEq, PartialOrd)]
struct NodeEntry {
    importance: f64,
    node: NodeIdx,
}

impl Eq for NodeEntry {}

impl Ord for NodeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal).reverse()
    }
}

pub fn calc_greedy_approximation_from_vec(instance: &Instance, importance: &[f64]) -> Vec<NodeIdx> {
    let mut hit = vec![true; instance.num_edges_total()];
    for edge in instance.edges() {
        hit[edge.idx()] = false;
    }

    let mut node_queue = BinaryHeap::from(
        instance
            .nodes()
            .into_iter()
            .map(|&node| NodeEntry {
                importance: importance[node.idx()] * (instance.node_degree(node) as f64),
                // importance: importance[node.idx()],
                node,
            })
            .collect::<Vec<_>>(),
    );

    let mut hs = Vec::new();
    while let Some(NodeEntry {
        importance: _,
        node,
    }) = node_queue.pop()
    {
        if instance.node(node).all(|edge| hit[edge.idx()]) {
            continue;
        }

        hs.push(node);

        for edge in instance.node(node) {
            hit[edge.idx()] = true;
        }
    }

    hs
}

pub fn calc_greedy_approximation(instance: &Instance) -> Vec<NodeIdx> {
    let mut hit = vec![true; instance.num_edges_total()];
    for edge in instance.edges() {
        hit[edge.idx()] = false;
    }

    let mut node_degrees = vec![0; instance.num_nodes_total()];
    let mut node_queue = BinaryHeap::from(
        instance
            .nodes()
            .into_iter()
            .map(|&node| {
                let degree = instance.node_degree(node);
                node_degrees[node.idx()] = degree;
                (degree, node)
            })
            .collect::<Vec<_>>(),
    );

    let mut hs = Vec::new();
    while let Some((degree, node)) = node_queue.pop() {
        if degree == 0 {
            break;
        }
        if degree > node_degrees[node.idx()] {
            continue;
        }

        hs.push(node);
        node_degrees[node.idx()] = 0;
        for edge in instance.node(node) {
            if hit[edge.idx()] {
                continue;
            }

            hit[edge.idx()] = true;
            for edge_node in instance.edge(edge) {
                if node_degrees[edge_node.idx()] > 0 {
                    node_degrees[edge_node.idx()] -= 1;
                    node_queue.push((node_degrees[edge_node.idx()], edge_node));
                }
            }
        }
    }

    hs
}

fn recalculate_greedy_upper_bound(instance: &Instance, state: &mut State, report: &mut Report) {
    report.reductions.greedy_runs += 1;
    // let improvements_list_ref = &mut report.upper_bound_improvements;
    // let branching_steps = report.branching_steps;
    collect_time_info(&mut report.runtimes.greedy, || {
        let greedy = calc_greedy_approximation(instance);
        if let Ok(mut min_hs_guard) = state.minimum_hs.lock() {
            if state.partial_hs.len() + greedy.len() < min_hs_guard.len() {
                min_hs_guard.clear();
                min_hs_guard.extend(state.partial_hs.iter().copied());
                min_hs_guard.extend(greedy.iter().copied());
                // improvements_list_ref.push(UpperBoundImprovement {
                //     new_bound: state.minimum_hs.len(),
                //     branching_steps,
                //     runtime: state.solve_start_time.elapsed(),
                // });
                info!(
                    "Found HS of size {} using greedy (partial {} + greedy {}) |V|={}, |E|={}",
                    min_hs_guard.len(),
                    state.partial_hs.len(),
                    greedy.len(),
                    instance.num_nodes(),
                    instance.num_edges()
                );
            }
        }
    });
}

pub fn collect_time_info<T>(runtime: &mut Duration, func: impl FnOnce() -> T) -> T {
    let before = Instant::now();
    let result = func();
    *runtime += before.elapsed();
    result
}

pub fn run_reduction<I>(
    reduced_items: &mut Vec<ReducedItem>,
    runtime: &mut Duration,
    runs: &mut usize,
    item_counter: &mut usize,
    func: impl FnOnce() -> I,
) where
    I: IntoIterator<Item = ReducedItem>,
{
    let len_before = reduced_items.len();
    *runs += 1;
    collect_time_info(runtime, || {
        reduced_items.extend(func());
    });
    *item_counter += reduced_items.len() - len_before;
}

#[allow(clippy::too_many_lines)]
pub fn reduce(
    instance: &mut Instance,
    state: &mut State,
    report: &mut Report,
    depth: usize,
    vertex_importance: &Vec<f64>,
) -> (ReductionResult, Reduction) {
    if report.settings.greedy_mode == GreedyMode::Once {
        recalculate_greedy_upper_bound(instance, state, report);
        if let Ok(mut min_hs_guard) = state.minimum_hs.lock() {
            if min_hs_guard.len() <= report.settings.stop_at {
                return (ReductionResult::Stop, Reduction(vec![]));
            }
        }
        // else should NEVER happen
    }

    let mut reduced_items = Vec::new();
    let mut applied_lp_already_once = false;
    let result = loop {
        if let Ok(mut min_hs_guard) = state.minimum_hs.lock() {
            if state.partial_hs.len() >= min_hs_guard.len() {
                break ReductionResult::Unsolvable;
            }
        }

        if instance.num_edges() == 0 || instance.num_nodes() == 0 {
            break ReductionResult::Solved;
        }

        if report.settings.greedy_mode == GreedyMode::AlwaysBeforeBounds {
            recalculate_greedy_upper_bound(instance, state, report);
            if let Ok(mut min_hs_guard) = state.minimum_hs.lock() {
                if min_hs_guard.len() <= report.settings.stop_at {
                    break ReductionResult::Stop;
                }
                if state.partial_hs.len() >= min_hs_guard.len() {
                    break ReductionResult::Unsolvable;
                }
            }
            // else should never happen
        }

        let mut lower_bound_breakpoint = instance.num_nodes_total(); // this might be source of error if if statement is not executed (which should NEVER happen)
        if let Ok(mut min_hs_guard) = state.minimum_hs.lock() {
            let mut lower_bound_breakpoint = min_hs_guard.len() - state.partial_hs.len();
        }

        if report.settings.enable_max_degree_bound {
            let max_degree_bound = collect_time_info(&mut report.runtimes.max_degree_bound, || {
                lower_bound::calc_max_degree_bound(instance).unwrap_or(0)
            });
            if max_degree_bound >= lower_bound_breakpoint {
                report.reductions.max_degree_bound_breaks += 1;
                break ReductionResult::Unsolvable;
            }
        }

        if report.settings.enable_sum_degree_bound {
            let sum_degree_bound = collect_time_info(&mut report.runtimes.sum_degree_bound, || {
                lower_bound::calc_sum_degree_bound(instance)
            });
            if sum_degree_bound >= lower_bound_breakpoint {
                report.reductions.sum_degree_bound_breaks += 1;
                break ReductionResult::Unsolvable;
            }
        }

        let discard_efficiency_bounds = if report.settings.enable_efficiency_bound {
            let (efficiency_bound, discard_efficiency_bounds) =
                collect_time_info(&mut report.runtimes.efficiency_bound, || {
                    lower_bound::calc_efficiency_bound(instance)
                });
            if efficiency_bound.round().unwrap_or(0) >= lower_bound_breakpoint {
                report.reductions.efficiency_degree_bound_breaks += 1;
                break ReductionResult::Unsolvable;
            }
            discard_efficiency_bounds
        } else {
            Vec::new()
        };

        let packing_bound = if report.settings.enable_packing_bound {
            let settings_ref = &report.settings;
            let packing_bound = collect_time_info(&mut report.runtimes.packing_bound, || {
                PackingBound::new(instance, settings_ref)
            });
            if packing_bound.bound() >= lower_bound_breakpoint {
                report.reductions.packing_bound_breaks += 1;
                break ReductionResult::Unsolvable;
            }
            packing_bound
        } else {
            PackingBound::default()
        };

        if report.settings.enable_packing_bound && report.settings.enable_sum_over_packing_bound {
            let sum_over_packing_bound =
                collect_time_info(&mut report.runtimes.sum_over_packing_bound, || {
                    packing_bound.calc_sum_over_packing_bound(instance)
                });
            if sum_over_packing_bound >= lower_bound_breakpoint {
                report.reductions.sum_over_packing_bound_breaks += 1;
                break ReductionResult::Unsolvable;
            }
        }

        if report.settings.enable_lp_lower_bound {
            let lp_bound = collect_time_info(&mut report.runtimes.lp_bound, || {
                lower_bound::calc_lp_bound(instance).unwrap_or(0)
            });
            if lp_bound >= lower_bound_breakpoint {
                report.reductions.lp_bound_breaks += 1;
                break ReductionResult::Unsolvable;
            }
        }

        let unchanged_len = reduced_items.len();

        if depth % 5 == 0 && report.settings.enable_lp_reduction && !applied_lp_already_once{ // optimistic code!
            //info!("Doing LP optimistic reduction...");
            if (depth == 0){
                //info!("Depth is {} for lp reduction", depth);
                run_reduction(
                    &mut reduced_items,
                    &mut report.runtimes.vertex_domination, // this does not make sense, however
                    &mut report.reductions.vertex_dominations_runs,
                    &mut report.reductions.vertex_dominations_vertices_found,
                    || optimistic_reductions(instance, state, vertex_importance),
                );               
            }
            else {
                run_reduction(
                    &mut reduced_items,
                    &mut report.runtimes.vertex_domination,
                    &mut report.reductions.vertex_dominations_runs,
                    &mut report.reductions.vertex_dominations_vertices_found,
                    || optimistic_lp_reductions(instance),
                );
            }
            //info!("Apply optimistic reduction...");
            for reduced_item in &reduced_items[unchanged_len..] {
                reduced_item.apply(instance, &mut state.partial_hs);
            } 
            applied_lp_already_once = true;
        }
        // let unchanged_len = reduced_items.len();
        
        if report.settings.degree_one_removal && reduced_items.len() == unchanged_len {
            // Remove degree one vertices which have a neighbour
            run_reduction(
                &mut reduced_items,
                &mut report.runtimes.discarded_vertex,
                &mut report.reductions.discard_vertex_runs,
                &mut report.reductions.discard_vertices_found,
                || find_discard_and_forced_vertex(instance),
            );
        }

        run_reduction(
            &mut reduced_items,
            &mut report.runtimes.forced_vertex,
            &mut report.reductions.forced_vertex_runs,
            &mut report.reductions.forced_vertices_found,
            || find_forced_nodes(instance),
        );

        if reduced_items.len() == unchanged_len && report.settings.enable_efficiency_bound {
            // Do not time this step as all costly parts are integrated into the
            // calculation of the efficiency bound above. This steps just checks
            // the already calculated discard bounds against the breakpoint
            let mut dummy_duration = Duration::default();
            run_reduction(
                &mut reduced_items,
                &mut dummy_duration,
                &mut report.reductions.costly_discard_efficiency_runs,
                &mut report.reductions.costly_discard_efficiency_vertices_found,
                || {
                    find_costly_discards_using_efficiency_bound(
                        instance,
                        lower_bound_breakpoint,
                        &discard_efficiency_bounds,
                    )
                },
            );
        }

        if reduced_items.len() == unchanged_len && report.settings.enable_packing_bound {
            run_reduction(
                &mut reduced_items,
                &mut report.runtimes.costly_discard_packing_update,
                &mut report.reductions.costly_discard_packing_update_runs,
                &mut report
                    .reductions
                    .costly_discard_packing_update_vertices_found,
                || {
                    find_costly_discards_using_packing_update(
                        instance,
                        lower_bound_breakpoint,
                        &packing_bound,
                    )
                },
            );
        }

        if reduced_items.len() == unchanged_len
            && report.settings.greedy_mode == GreedyMode::AlwaysBeforeExpensiveReductions
        {
            recalculate_greedy_upper_bound(instance, state, report);
            if let Ok(mut min_hs_guard) = state.minimum_hs.lock() {
                if min_hs_guard.len() <= report.settings.stop_at {
                    break ReductionResult::Stop;
                }
                if state.partial_hs.len() >= min_hs_guard.len() {
                    break ReductionResult::Unsolvable;
                }
                lower_bound_breakpoint = min_hs_guard.len() - state.partial_hs.len();
            }
        }

        if reduced_items.len() == unchanged_len && report.settings.enable_packing_bound {
            let table_ref = &mut report
                .reductions
                .costly_discard_packing_from_scratch_steps_per_run;
            let settings_ref = &report.settings;
            let mut dummy_counter = 0;
            run_reduction(
                &mut reduced_items,
                &mut report.runtimes.costly_discard_packing_from_scratch,
                &mut report.reductions.costly_discard_packing_from_scratch_runs,
                &mut dummy_counter,
                || {
                    let result = find_costly_discard_using_packing_from_scratch(
                        instance,
                        lower_bound_breakpoint,
                        settings_ref,
                    );
                    match result {
                        None => {
                            table_ref[settings_ref.packing_from_scratch_limit] += 1;
                            None
                        }
                        Some((item, idx)) => {
                            table_ref[idx] += 1;
                            Some(item)
                        }
                    }
                },
            );
        }

        if reduced_items.len() == unchanged_len {
            run_reduction(
                &mut reduced_items,
                &mut report.runtimes.vertex_domination,
                &mut report.reductions.vertex_dominations_runs,
                &mut report.reductions.vertex_dominations_vertices_found,
                || find_dominated_nodes(instance),
            );
        }

        if reduced_items.len() == unchanged_len {
            run_reduction(
                &mut reduced_items,
                &mut report.runtimes.edge_domination,
                &mut report.reductions.edge_dominations_runs,
                &mut report.reductions.edge_dominations_edges_found,
                || find_dominated_edges(instance),
            );
        }

        if reduced_items.len() == unchanged_len {
            break ReductionResult::Finished;
        }

        collect_time_info(&mut report.runtimes.applying_reductions, || {
            for reduced_item in &reduced_items[unchanged_len..] {
                reduced_item.apply(instance, &mut state.partial_hs);
            }
        });
    };

    (result, Reduction(reduced_items))
}

pub fn reduce_for_ilp(instance: &mut Instance) -> (usize, usize) {
    let mut reduced = Vec::new();
    let mut dummy_partial_hs = Vec::new();
    let mut reduced_nodes = 0;
    let mut reduced_edges = 0;
    loop {
        let mut changed = false;

        reduced.extend(find_discard_vertex(instance));
        reduced_nodes += reduced.len();
        changed |= !reduced.is_empty();
        for item in reduced.drain(..) {
            item.apply(instance, &mut dummy_partial_hs);
        }

        reduced.extend(find_dominated_nodes(instance));
        reduced_nodes += reduced.len();
        changed |= !reduced.is_empty();
        for item in reduced.drain(..) {
            item.apply(instance, &mut dummy_partial_hs);
        }

        reduced.extend(find_dominated_edges(instance));
        reduced_edges += reduced.len();
        changed |= !reduced.is_empty();
        for item in reduced.drain(..) {
            item.apply(instance, &mut dummy_partial_hs);
        }

        if !changed {
            break;
        }
    }

    (reduced_nodes, reduced_edges)
}
