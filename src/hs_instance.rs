use crate::data_structures::cont_idx_vec::ContiguousIdxVec;
use crate::data_structures::skipvec::SkipVec;
use crate::instance::{
    AnalysisInstance, CompressedIlpName, EdgeIdx, EntryIdx, GeneralInstance, Instance,
    InstanceType, NodeIdx,
};
use crate::small_indices::SmallIdx;
use anyhow::{anyhow, ensure, Result};
use itertools::Itertools;
use log::{debug, info, trace};
use std::io::{BufRead, Write};
use std::mem;
use std::time::Instant;

#[derive(Debug)]
struct ParsedEdgeHandler {
    edge_incidences: Vec<SkipVec<(NodeIdx, EntryIdx)>>,
    node_degrees: Vec<usize>,
}

impl ParsedEdgeHandler {
    fn handle_edge(&mut self, node_indices: impl IntoIterator<Item = Result<usize>>) -> Result<()> {
        let incidences = SkipVec::try_sorted_from(node_indices.into_iter().map(|idx_result| {
            idx_result.and_then(|node_idx| {
                ensure!(
                    node_idx < self.node_degrees.len(),
                    "invalid node idx in edge: {}",
                    node_idx
                );
                Ok((NodeIdx::from(node_idx), EntryIdx::INVALID))
            })
        }))?;
        ensure!(incidences.len() > 0, "edges may not be empty");
        for (_, (node, _)) in &incidences {
            self.node_degrees[node.idx()] += 1;
        }
        self.edge_incidences.push(incidences);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct HGRInstanceX {
    nodes: ContiguousIdxVec<NodeIdx>,
    edges: ContiguousIdxVec<EdgeIdx>,
    node_incidences: Vec<SkipVec<(EdgeIdx, EntryIdx)>>,
    edge_incidences: Vec<SkipVec<(NodeIdx, EntryIdx)>>,
}

impl HGRInstanceX {
    fn build_instance(
        num_nodes: usize,
        num_edges: usize,
        read_edges: impl FnOnce(&mut ParsedEdgeHandler) -> Result<()>,
    ) -> Result<Self> {
        let mut handler = ParsedEdgeHandler {
            edge_incidences: Vec::with_capacity(num_edges),
            node_degrees: vec![0; num_nodes],
        };
        read_edges(&mut handler)?;
        let ParsedEdgeHandler {
            mut edge_incidences,
            node_degrees,
        } = handler;

        let mut node_incidences: Vec<_> = node_degrees
            .iter()
            .map(|&len| SkipVec::with_len(len))
            .collect();
        let mut rem_node_degrees = node_degrees;
        for (edge, incidences) in edge_incidences.iter_mut().enumerate() {
            let edge = EdgeIdx::from(edge);
            for (edge_entry_idx, edge_entry) in incidences.iter_mut() {
                let node = edge_entry.0.idx();
                let node_entry_idx = node_incidences[node].len() - rem_node_degrees[node];
                rem_node_degrees[node] -= 1;
                edge_entry.1 = EntryIdx::from(node_entry_idx);
                node_incidences[node][node_entry_idx] = (edge, EntryIdx::from(edge_entry_idx));
            }
        }

        Ok(Self {
            nodes: (0..num_nodes).map(NodeIdx::from).collect(),
            edges: (0..num_edges).map(EdgeIdx::from).collect(),
            node_incidences,
            edge_incidences,
        })
    }

    /// Alive edges in the instance, in arbitrary order.
    pub fn edges(&self) -> &[EdgeIdx] {
        &self.edges
    }

    /// Edges incident to a node, sorted by increasing indices.
    pub fn incident_edges(
        &self,
        node: NodeIdx,
    ) -> impl Iterator<Item = EdgeIdx> + ExactSizeIterator + Clone + '_ {
        self.node_incidences[node.idx()]
            .iter()
            .map(|(_, (edge, _))| *edge)
    }

    /// Nodes incident to an edge, sorted by increasing indices.
    pub fn incident_nodes(
        &self,
        edge: EdgeIdx,
    ) -> impl Iterator<Item = NodeIdx> + ExactSizeIterator + Clone + '_ {
        self.edge_incidences[edge.idx()]
            .iter()
            .map(|(_, (node, _))| *node)
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_nodes_total(&self) -> usize {
        self.node_incidences.len()
    }

    pub fn num_edges_total(&self) -> usize {
        self.edge_incidences.len()
    }

    pub fn node_degree(&self, node: NodeIdx) -> usize {
        self.node_incidences[node.idx()].len()
    }

    pub fn edge_degree(&self, edge: crate::instance::EdgeIdx) -> usize {
        self.edge_incidences[edge.idx()].len()
    }

    pub fn adjacent_edges(&self, edge: EdgeIdx) -> impl Iterator<Item = EdgeIdx> + Clone + '_ {
        self.incident_nodes(edge)
            .flat_map(|node_idx| self.incident_edges(node_idx).filter(|e| e != edge))
            .sorted()
            .dedup()
    }

    /// Deletes all edges incident to a node.
    ///
    /// The node itself must have already been deleted.
    pub fn delete_incident_edges(&mut self, node: crate::instance::NodeIdx) {
        // We want to iterate over the incidence of `node` while deleting
        // edges, which in turn changes node incidences. This is safe, since
        // `node` itself was already deleted. To make the borrow checker
        // accept this, we temporarily move `node` incidence to a local
        // variable, replacing it with an empty list. This should not be much
        // slower than unsafe alternatives, since an incidence list is only
        // 28 bytes large.
        trace!("Deleting all edges incident to {}", node);
        debug_assert!(
            self.nodes.is_deleted(node.idx()),
            "Node passed to delete_incident_edges must be deleted"
        );
        let incidence = mem::take(&mut self.node_incidences[node.idx()]);
        for (_, (edge, _)) in &incidence {
            self.delete_edge(*edge);
        }
        self.node_incidences[node.idx()] = incidence;
    }

    /// Restores all incident edges to a node.
    ///
    /// This reverses the effect of `delete_incident_edges`. As with all other
    /// `restore_*` methods, this must be done in reverse order of deletions.
    /// In particular, the node itself must still be deleted.
    pub fn restore_incident_edges(&mut self, node: crate::instance::NodeIdx) {
        trace!("Restoring all edges incident to {}", node);
        debug_assert!(
            self.nodes.is_deleted(node.idx()),
            "Node passed to restore_incident_edges must be deleted"
        );

        // See `delete_incident_edges` for an explanation of this swapping around
        let incidence = mem::take(&mut self.node_incidences[node.idx()]);

        // It is important that we restore the edges in reverse order
        for (_, (edge, _)) in incidence.iter().rev() {
            self.restore_edge(*edge);
        }
        self.node_incidences[node.idx()] = incidence;
    }

    /// with output of the Instance and two mappings, new to old
    fn partial_instance(
        &self,
        new_nodes: &Vec<bool>,
        new_edges: &Vec<bool>,
    ) -> (Self, Vec<(NodeIdx, NodeIdx)>, Vec<(EdgeIdx, EdgeIdx)>) {
        assert_eq!(new_nodes.len(), self.num_nodes_total());
        let new_nodes_size = new_nodes.iter().filter(|x| **x).count();
        let mut node_mapping = Vec::with_capacity(new_nodes_size);

        let new_node_idx = 0;
        new_nodes.iter().enumerate().for_each(|(i, x)| {
            if *x {
                node_mapping.push((NodeIdx(new_node_idx), NodeIdx(i as u32)))
            }
        });

        let new_edges_size = new_edges.iter().filter(|x| **x).count();
        let mut edge_mapping = Vec::with_capacity(new_edges_size);

        let new_edge_idx = 0;
        new_edges.iter().enumerate().for_each(|(i, x)| {
            if *x {
                edge_mapping.push((EdgeIdx(new_edge_idx), EdgeIdx(i as u32)))
            }
        });

        let mut new_idx = 0;
        let mut old_nodes_mapping = new_nodes
            .iter()
            .map(|x| {
                if *x {
                    new_idx += 1;
                    new_idx - 1
                } else {
                    -1
                }
            })
            .collect::<Vec<_>>();

        let mut handler = ParsedEdgeHandler {
            edge_incidences: Vec::with_capacity(new_edges_size),
            node_degrees: vec![0; new_nodes_size],
        };

        for edge in self.edges() {
            if !(new_edges[edge.idx()]) {
                continue;
            };
            handler
                .handle_edge(
                    self.incident_nodes(*edge)
                        .map(|node| old_nodes_mapping[node.idx()])
                        .filter(|&val| val >= 0)
                        .map(|val| Ok(val as usize)),
                )
                .unwrap();
        }

        let mut node_incidences: Vec<_> = handler
            .node_degrees
            .iter()
            .map(|&len| SkipVec::with_len(len))
            .collect();
        let mut rem_node_degrees = handler.node_degrees;
        for (edge, incidences) in handler.edge_incidences.iter_mut().enumerate() {
            let edge = EdgeIdx::from(edge);
            for (edge_entry_idx, edge_entry) in incidences.iter_mut() {
                let node = edge_entry.0.idx();
                let node_entry_idx = node_incidences[node].len() - rem_node_degrees[node];
                rem_node_degrees[node] -= 1;
                edge_entry.1 = EntryIdx::from(node_entry_idx);
                node_incidences[node][node_entry_idx] = (edge, EntryIdx::from(edge_entry_idx));
            }
        }
        let new_edge_incidences = handler.edge_incidences;

        (
            Self {
                nodes: (0..new_nodes_size).map(NodeIdx::from).collect(),
                edges: (0..new_edges_size).map(EdgeIdx::from).collect(),
                node_incidences,
                edge_incidences: new_edge_incidences,
            },
            node_mapping,
            edge_mapping,
        )
    }
}

impl GeneralInstance for HGRInstanceX {
    fn load_from_buffer(mut reader: impl BufRead) -> Result<Self> {
        let time_before = Instant::now();
        let mut line = String::new();

        loop {
            line.clear();
            reader.read_line(&mut line)?;
            if line.starts_with('c') {
                continue;
            }
            if line.starts_with('p') {
                break;
            }
            return Err(anyhow!("Expected problem line starting with 'p'"));
        }

        let mut parts = line.split_ascii_whitespace();
        ensure!(
            parts.next() == Some("p"),
            "Expected 'p' at start of problem line"
        );
        // ensure!(parts.next() == Some("hs"), "Expected 'hs' in problem line");
        let instance_typ = parts.next().unwrap();
        debug!("Read: {instance_typ:?}");
        let num_nodes: usize = parts
            .next()
            .ok_or_else(|| anyhow!("Missing node count"))?
            .parse()?;
        let num_edges: usize = parts
            .next()
            .ok_or_else(|| anyhow!("Missing edge count"))?
            .parse()?;
        ensure!(parts.next().is_none(), "Too many numbers in problem line");

        let instance = Self::build_instance(num_nodes, num_edges, |handler| {
            for _ in 0..num_edges {
                line.clear();
                reader.read_line(&mut line)?;

                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('c') {
                    continue;
                }

                let node_indices = trimmed.split_ascii_whitespace().map(|s| {
                    let idx: usize = s.parse()?;
                    ensure!(
                        idx >= 1 && idx <= num_nodes,
                        "Invalid node index in edge: {}",
                        idx
                    );
                    Ok(idx - 1)
                });

                handler.handle_edge(node_indices)?;
            }
            Ok(())
        })?;

        info!(
            "Loaded HGR instance with {} nodes, {} edges in {:.2?}",
            num_nodes,
            num_edges,
            time_before.elapsed(),
        );

        Ok(instance)
    }

    fn partial_instance(&self, new_nodes: &Vec<bool>) -> Self {
        todo!("Call the inner partial_instance function")
    }

    fn nodes(&self) -> &[NodeIdx] {
        &self.nodes
    }

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn adjacent_nodes(&self, node: NodeIdx) -> impl Iterator<Item = NodeIdx> + Clone + '_ {
        self.incident_edges(node)
            .flat_map(|edge| self.incident_nodes(edge).filter(|n| n != node))
            .sorted()
            .dedup()
    }

    /// Deletes a node from the instance.
    fn delete_node(&mut self, node: NodeIdx) {
        trace!("Deleting node {}", node);
        for (_idx, (edge, entry_idx)) in &self.node_incidences[node.idx()] {
            self.edge_incidences[edge.idx()].delete(entry_idx.idx());
        }
        self.nodes.delete(node.idx());
    }

    fn is_node_deleted(&self, node: NodeIdx) -> bool {
        self.nodes.is_deleted(node.idx())
    }

    /*
    pub fn is_edge_deleted(&mut self, edge: EdgeIdx) -> bool {
        self.edges.is_deleted(edge.idx())
    }
    */

    /// Deletes an edge from the instance.
    fn delete_edge(&mut self, edge: EdgeIdx) {
        trace!("Deleting edge {}", edge);
        for (_idx, (node, entry_idx)) in &self.edge_incidences[edge.idx()] {
            self.node_incidences[node.idx()].delete(entry_idx.idx());
        }
        self.edges.delete(edge.idx());
    }

    /// Restores a previously deleted node.
    ///
    /// All restore operations (node or edge) must be done in reverse order of
    /// the corresponding deletions to produce sensible results.
    fn restore_node(&mut self, node: NodeIdx) {
        trace!("Restoring node {}", node);
        for (_idx, (edge, entry_idx)) in self.node_incidences[node.idx()].iter().rev() {
            self.edge_incidences[edge.idx()].restore(entry_idx.idx());
        }
        self.nodes.restore(node.idx());
    }

    /// Restores a previously deleted edge.
    ///
    /// All restore operations (node or edge) must be done in reverse order of
    /// the corresponding deletions to produce sensible results.
    fn restore_edge(&mut self, edge: EdgeIdx) {
        trace!("Restoring edge {}", edge);
        for (_idx, (node, entry_idx)) in self.edge_incidences[edge.idx()].iter().rev() {
            self.node_incidences[node.idx()].restore(entry_idx.idx());
        }
        self.edges.restore(edge.idx());
    }

    fn export_to_ilp(&self, mut writer: impl Write) -> anyhow::Result<()> {
        writeln!(writer, "Minimize")?;
        write!(writer, "  v{}", CompressedIlpName(self.nodes()[0]))?;
        for &node in &self.nodes()[1..] {
            write!(writer, " + v{}", CompressedIlpName(node))?;
        }
        writeln!(writer)?;

        writeln!(writer, "Subject To")?;
        for &edge in self.edges() {
            write!(writer, "  e{}: ", CompressedIlpName(edge))?;
            for (idx, node) in self.incident_nodes(edge).enumerate() {
                if idx > 0 {
                    write!(writer, " + ")?;
                }
                write!(writer, "v{}", CompressedIlpName(node))?;
            }
            writeln!(writer, " >= 1")?;
        }

        writeln!(writer, "Binaries")?;
        write!(writer, "  v{}", CompressedIlpName(self.nodes()[0]))?;
        for &node in &self.nodes()[1..] {
            write!(writer, " v{}", CompressedIlpName(node))?;
        }
        writeln!(writer)?;

        writeln!(writer, "End")?;
        Ok(())
    }

    /// Exports the hitting set instance as a weighted MaxSAT (wcnf) instance.
    ///
    /// Each node is represented by a Boolean variable. For every node we add a
    /// soft clause (with weight 1) preferring that the node is *not* chosen, and
    /// for every edge we add a hard clause (with weight `top`) enforcing that at least one
    /// incident node is chosen. The DIMACS wcnf header is:
    ///
    ///     p wcnf <num_vars> <num_clauses> <top>
    ///
    /// # Parameters
    ///
    /// - `writer`: An output sink to which the DIMACS representation will be written.
    ///
    /// # Returns
    ///
    /// A Result with an empty tuple on success.
    fn export_to_max_sat(&self, mut writer: impl Write) -> anyhow::Result<()> {
        // We assume that self.nodes() returns the list of alive nodes.
        let alive_nodes = self.nodes();
        let num_vars = alive_nodes.len();
        let num_soft_clauses = num_vars;
        let num_hard_clauses = self.edges().len();
        let total_clauses = num_soft_clauses + num_hard_clauses;
        // The top weight must exceed the sum of soft clause weights.
        let top = num_soft_clauses + 1;

        // In DIMACS the variables are 1-indexed. Because our NodeIdx values may be
        // sparse (or not in order) due to deletions, we create a mapping from NodeIdx to
        // new variable numbers.
        let mut node_var = vec![None; self.node_incidences.len()];
        for (i, &node) in alive_nodes.iter().enumerate() {
            node_var[node.idx()] = Some(i + 1);
        }

        // Write the header line.
        writeln!(writer, "p wcnf {} {} {}", num_vars, total_clauses, top)?;

        // Write one soft clause per alive node:
        // Each soft clause is: 1 -<var> 0
        // (i.e. we “prefer” that the node is not chosen).
        for &node in alive_nodes {
            let var = node_var[node.idx()].ok_or_else(|| anyhow!("Alive node missing mapping"))?;
            writeln!(writer, "1 -{} 0", var)?;
        }

        // Write one hard clause per edge:
        // For each edge, we create a clause with weight top that is the disjunction
        // of the (positive) node variables in the edge.
        for &edge in self.edges() {
            // Gather the variable numbers for all nodes incident to this edge.
            let clause_vars: Vec<_> = self
                .incident_nodes(edge)
                .map(|node| {
                    node_var[node.idx()].ok_or_else(|| anyhow!("Edge contains a deleted node"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            ensure!(!clause_vars.is_empty(), "Edge clause is empty");
            write!(writer, "{} ", top)?;
            for var in clause_vars {
                write!(writer, "{} ", var)?;
            }
            writeln!(writer, "0")?;
        }

        Ok(())
    }
}

impl AnalysisInstance for HGRInstanceX {
    /// degree distribution, diameter, treewidth, große Cliquen/ große Gitter?
    fn get_instance_type(&self) -> u32 {
        let mut result = 0;

        let node_degrees = &self.node_degrees();

        let min_degree = node_degrees.iter().min().unwrap_or(&0);
        let max_degree = node_degrees.iter().max().unwrap_or(&0);

        let num_nodes = self.num_nodes_total();
        let num_edges = self.num_edges_total();

        if *max_degree - *min_degree <= 8 && num_edges <= num_nodes * 2 {
            // magic numbers TODO
            result |= InstanceType::FlatDegree as u32;
        } else {
            result |= InstanceType::VariedDegree as u32;
        }

        if num_edges > num_nodes * 2 {
            result |= InstanceType::Dense as u32;
        } else {
            result |= InstanceType::Sparse as u32;
        }

        let max_edge_size = self
            .edges
            .iter()
            .map(|&edge| self.edge_incidences[edge.idx()].len())
            .max()
            .unwrap_or(0);

        if max_edge_size <= 2 {
            result |= InstanceType::Graph as u32;
        }

        result
    }

    fn node_degrees(&self) -> Vec<usize> {
        self.node_incidences.iter().map(|inc| inc.len()).collect()
    }

    fn show_stats(&self) {
        let mut degrees: Vec<usize> = self
            .nodes
            .iter()
            .map(|&node| self.node_degree(node))
            .collect();

        if degrees.is_empty() {
            println!("No nodes available.");
            return;
        }

        degrees.sort_unstable();

        for percentile in (0..=100).step_by(10) {
            let index = (percentile as f64 / 100.0 * (degrees.len() as f64 - 1.0)).round() as usize;
            println!("{}% percentile: {}", percentile, degrees[index]);
        }
    }

    /// Computes the variance of node degrees.
    fn degree_variance(&self) -> f64 {
        let num_nodes = self.num_nodes();
        if num_nodes == 0 {
            return 0.0;
        }

        let mean_degree = self.edges.len() as f64 / num_nodes as f64;
        let variance = self
            .nodes()
            .iter()
            .map(|&node| {
                let degree = self.node_degree(node) as f64;
                (degree - mean_degree).powi(2)
            })
            .sum::<f64>()
            / num_nodes as f64;

        info!("Variance {}", variance);
        variance
    }

    /// Computes the entropy of the node degree distribution.
    fn degree_entropy(&self) -> f64 {
        let num_nodes = self.num_nodes();
        if num_nodes == 0 {
            return 0.0;
        }

        let mut degree_counts = vec![0; self.num_edges_total()];
        for &node in self.nodes() {
            degree_counts[self.node_degree(node)] += 1;
        }

        let mut entropy = 0.0;
        for &count in &degree_counts {
            if count > 0 {
                let p = count as f64 / num_nodes as f64;
                entropy -= p * p.log2();
            }
        }

        info!("Entropy {}", entropy);

        entropy
    }

    /// Computes the edge-to-node ratio.
    fn edge_to_node_ratio(&self) -> f64 {
        if self.num_nodes() == 0 {
            return 0.0;
        }
        info!(
            "edge / node {}",
            self.num_edges() as f64 / self.num_nodes() as f64
        );

        self.num_edges() as f64 / self.num_nodes() as f64
    }

    fn is_hard_instance(&self) -> bool {
        let degree_var = self.degree_variance();
        let degree_entropy = self.degree_entropy();
        let edge_node_ratio = self.edge_to_node_ratio();

        let low_variance = degree_var < 50.0;
        let low_entropy = degree_entropy < 3.5;
        let sparse_graph = edge_node_ratio < 1.5;

        low_variance && low_entropy && sparse_graph
    }
}
