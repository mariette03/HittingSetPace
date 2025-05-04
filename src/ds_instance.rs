use crate::{
    create_idx_struct,
    data_structures::{cont_idx_vec::ContiguousIdxVec, skipvec::SkipVec},
    small_indices::SmallIdx,
    instance::{GeneralInstance, AnalysisInstance, NodeIdx, EdgeIdx, EntryIdx},
};
use anyhow::{anyhow, ensure, Result};
use itertools::{all, Itertools};
use log::{debug, info, trace};
use rustworkx_core::petgraph::algo::connected_components;
use rustworkx_core::petgraph::dot::{Config, Dot};
use rustworkx_core::petgraph::graph::{Edge, UnGraph};
use rustworkx_core::petgraph::Graph;
use rustworkx_core::planar::is_planar;
use serde::Deserialize;
use serde::Serialize;
use std::cmp::PartialEq;
use std::collections::VecDeque;
use std::{
    fmt::{self, Display, Write as _},
    io::{BufRead, Write},
    mem,
    time::Instant,
};


#[derive(Debug, Clone, Serialize)]
pub enum InstanceType {
    FlatDegree = (1 << 0),
    VariedDegree = (1 << 1),
    Dense = (1 << 2),
    Sparse = (1 << 3),
    Graph = (1 << 4),
}


#[derive(Clone, Debug)]
pub struct GraphInstanceX {
    nodes: ContiguousIdxVec<NodeIdx>,
    node_incidences: Vec<SkipVec<(NodeIdx, EntryIdx)>>,
}

impl GraphInstanceX {
    /// Alive nodes in the instance, in arbitrary order.
    pub fn nodes(&self) -> &[NodeIdx] {
        &self.nodes
    }

    /// Alive edges in the instance, in arbitrary order.
    pub fn edges(&self) -> &[EdgeIdx] {
        todo!()
    }

    pub fn is_planar(&self) -> bool {
        todo!();
    }

    fn simple_edge(&self, edge: EdgeIdx) -> (NodeIdx, NodeIdx) {
        todo!();
    }

    /// we may want to call this library
    pub fn into_petgraph(&self) -> UnGraph<u32, ()> {
        let edges: Vec<(_, _)> = self
            .edges()
            .iter()
            .map(|edge_idx| {
                let simple_edge = self.simple_edge(*edge_idx);
                (simple_edge.0.idx() as u32, simple_edge.1.idx() as u32)
            })
            .collect();
        UnGraph::<u32, ()>::from_edges(edges)
    }

    fn from_petgraph(petgraph: &Graph<u32, ()>) -> Self {
        let mut adj_list: Vec<Vec<NodeIdx>> = Vec::new();
        for node in petgraph.node_indices() {
            let neighbors = petgraph
                .neighbors(node)
                .map(|x| NodeIdx(x.index() as u32))
                .collect();
            adj_list.push(neighbors);
        }

        // let mut node_incidences: Vec<_> = node_degrees
        //     .iter()
        //     .map(|&len| SkipVec::with_len(len))
        //     .collect();
        // let mut rem_node_degrees = node_degrees;
        // for (edge, incidences) in edge_incidences.iter_mut().enumerate() {
        //     let edge = EdgeIdx::from(edge);
        //     for (edge_entry_idx, edge_entry) in incidences.iter_mut() {
        //         let node = edge_entry.0.idx();
        //         let node_entry_idx = node_incidences[node].len() - rem_node_degrees[node];
        //         rem_node_degrees[node] -= 1;
        //         edge_entry.1 = EntryIdx::from(node_entry_idx);
        //         node_incidences[node][node_entry_idx] = (edge, EntryIdx::from(edge_entry_idx));
        //     }
        // }
        todo!()
    }

    /// get graphviz visualisation
    pub fn to_graphviz(&self) -> String {
        format!(
            "{:?}",
            Dot::with_config(
                &self.into_petgraph(),
                &[Config::EdgeNoLabel, Config::NodeIndexLabel]
            )
        )
    }
}

impl GeneralInstance for GraphInstanceX {
    fn load_from_buffer(reader: impl BufRead) -> Result<Self> {
        todo!()
    }

    fn partial_instance(&self, new_nodes: &Vec<bool>) -> Self
    where
        Self: Sized,
    {
        todo!()
    }

    fn nodes(&self) -> &[NodeIdx] {
        todo!()
    }

    fn num_nodes(&self) -> usize {
        todo!()
    }

    fn adjacent_nodes(&self, node: NodeIdx) -> impl Iterator<Item = NodeIdx> + Clone + '_ {
        self.node_incidences[node.idx()]
            .iter()
            .map(|(_, (x, _))| *x)
    }

    fn delete_node(&mut self, node_idx: NodeIdx) {
        todo!()
    }

    fn delete_edge(&mut self, edge_idx: EdgeIdx) {
        todo!()
    }

    fn restore_node(&mut self, node_idx: NodeIdx) {
        todo!()
    }

    fn restore_edge(&mut self, edge_idx: EdgeIdx) {
        todo!()
    }

    fn is_node_deleted(&self, node: NodeIdx) -> bool {
        todo!()
    }

    fn export_to_ilp(&self, writer: impl Write) -> Result<()> {
        todo!()
    }

    fn export_to_max_sat(&self, writer: impl Write) -> Result<()> {
        todo!()
    }
}
