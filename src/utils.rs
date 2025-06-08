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
    *instance
        .nodes()
        .iter()
        .max_by(|&a, &b| { 
            vertex_importance[a.idx()].partial_cmp(&vertex_importance[b.idx()]).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("Branching on an empty instance")
}
