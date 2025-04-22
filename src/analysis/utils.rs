use std::collections::HashMap;
use crate::instance::{Instance};
use crate::small_indices::SmallIdx;
use itertools::Itertools;

pub fn get_degree_distribution(instance: &Instance) -> Vec<(usize, usize)> {
    let mut degrees = instance
        .nodes()
        .iter()
        .map(|node_idx| instance.node_degree(*node_idx))
        .collect::<Vec<_>>();
    degrees.sort_unstable();
    degrees
        .into_iter()
        .chunk_by(|(deg)| *deg)
        .into_iter()
        .map(|(deg, elements)| (deg, elements.count()))
        .collect()
}

pub fn apx_treewidth(instance: &Instance) -> usize {
    todo!("Find a library for approximate treewidth (or call the treewidth computation from the main code")
}

pub fn centrality_distribution(instance: &Instance) -> Vec<(usize, usize)> {
    let dists = instance.get_dist_matrix();
    let mut radius_vec: Vec<usize> = Vec::new();
    for row in &dists{
        let radius_i = *row.iter().filter(|x| **x <= instance.num_nodes()).max().unwrap();
        radius_vec.push(radius_i);
    }
    
    radius_vec.sort_unstable();
    radius_vec
        .into_iter()
        .chunk_by(|deg| *deg)
        .into_iter()
        .map(|(deg, elements)| (deg, elements.count()))
        .collect()
}

use rustworkx_core::centrality::betweenness_centrality;
use rustworkx_core::petgraph::graph::UnGraph;


pub fn pet_functionality(g: UnGraph<u32, ()>) {
    // Calculate the betweenness centrality
    let output = betweenness_centrality(&g, false, false, 200);
    assert_eq!(
        vec![Some(0.0), Some(0.5), Some(0.5), Some(0.5), Some(0.5)],
        output
    );
    todo!();
}

use std::fs::File;
use std::io::Write;

/// Generates a report of the graph data in JSON format. Not very performant, but should suffice
pub fn graph_data_report(instance: &Instance, filename: &str) -> Result<String, std::io::Error> {
    let mut graphviz = File::create(format!("output/{filename}_graphviz.dot"))?;
    graphviz.write_all(instance.to_graphviz().as_bytes())?;
    
    let mut file = File::create(format!("output/{filename}_graph-data.json"))?;
    let mut output = HashMap::new();

    output.insert("name".to_string(), filename.to_string());
    output.insert("degrees".to_string(), serde_json::to_string(&get_degree_distribution(instance))?);
    output.insert("radii".to_string(), serde_json::to_string(&centrality_distribution(instance))?);
    
    let json = serde_json::to_string_pretty(&output)?;
    file.write_all(json.as_bytes())?;
    Ok(json)
}
