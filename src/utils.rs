use crate::instance::{Instance, NodeIdx};
use crate::small_indices::{IdxHashSet, SmallIdx};

pub fn is_hitting_set(hs: &[NodeIdx], instance: &Instance) -> bool {
    let hs_set: IdxHashSet<_> = hs.iter().copied().collect();
    instance
        .edges()
        .iter()
        .all(|&edge| instance.edge(edge).any(|node| hs_set.contains(&node)))
}

pub fn select_vertex(instance: &Instance, vertex_importance: &[f64]) -> NodeIdx {
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
}
