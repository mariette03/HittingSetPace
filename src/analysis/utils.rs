use crate::instance::{Instance, NodeIdx};
use itertools::Itertools;
use rand::random;
use std::cmp::{max, max_by_key};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

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

pub fn apx_diameter(instance: &Instance) -> (NodeIdx, usize) {
    let mut result = (Default::default(), 0);
    for _ in 0..3 {
        let rand_start = random::<usize>() % instance.num_nodes();
        let mut candidate = NodeIdx::from(rand_start);
        let mut candidate_bfs = instance.bfs(candidate);
        for _ in 0..4 {
            candidate = NodeIdx::from(
                candidate_bfs
                    .iter()
                    .enumerate()
                    .filter(|(_, dist)| **dist < instance.num_nodes() + 2)
                    .max_by_key(|(_, dist)| **dist)
                    .unwrap()
                    .0,
            );
            candidate_bfs = instance.bfs(candidate);
        }
        result = max_by_key(
            result,
            candidate_bfs
                .into_iter()
                .enumerate()
                .map(|(idx, dist)| (NodeIdx::from(idx), dist))
                .filter(|(_, dist)| *dist < instance.num_nodes() + 2)
                .max_by_key(|(_, v)| *v)
                .unwrap(),
            |(_, x)| *x,
        );
    }
    result
}

pub fn centrality_distribution(instance: &Instance) -> Vec<(usize, usize)> {
    let dists = instance.get_dist_matrix();
    let mut radius_vec: Vec<usize> = Vec::new();
    for row in &dists {
        let radius_i = *row
            .iter()
            .filter(|x| **x <= instance.num_nodes())
            .max()
            .unwrap();
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

pub fn number_connected_components(instance: &Instance) -> usize {
    connected_components(&instance.into_petgraph())
}

use rustworkx_core::centrality::betweenness_centrality;
use rustworkx_core::petgraph::algo::connected_components;
use rustworkx_core::petgraph::graph::UnGraph;
use serde_json::{json, Map, Value};

pub fn pet_functionality(g: UnGraph<u32, ()>) {
    // Calculate the betweenness centrality
    let output = betweenness_centrality(&g, false, false, 200);
    assert_eq!(
        vec![Some(0.0), Some(0.5), Some(0.5), Some(0.5), Some(0.5)],
        output
    );
    todo!();
}

/// Generates a report of the graph data in JSON format. Not very performant, but should suffice
pub fn graph_data_report(
    instance: &Instance,
    filename: &str,
    additional_info: Option<String>,
) -> Result<String, std::io::Error> {
    let mut graphviz = File::create(format!("output/{filename}_graphviz.dot"))?;
    graphviz.write_all(instance.to_graphviz().as_bytes())?;

    let mut file = File::create(format!("output/{filename}_graph-data.json"))?;
    let mut output = Map::new();

    output.insert("name".to_string(), json!(filename));
    output.insert("#nodes".to_string(), json!(instance.num_nodes()));
    output.insert("#edges".to_string(), json!(instance.num_edges()));

    let sqrt_n = (instance.num_nodes() as f64).sqrt().round() as usize;
    let degree_dist = get_degree_distribution(instance);

    output.insert(
        "#CCs".to_string(),
        json!(number_connected_components(instance)),
    );
    output.insert(
        "#low_degree_nodes".to_string(),
        json!(degree_dist.iter().filter(|(_, deg)| *deg <= 2).count()),
    );
    output.insert(
        "#high_degree_nodes".to_string(),
        json!(degree_dist.iter().filter(|(_, deg)| *deg >= sqrt_n).count()),
    );

    output.insert(
        "degrees".to_string(),
        json!(serde_json::to_string(&degree_dist)?),
    );

    //// too slow since code is not optimized yet
    // output.insert(
    //     "radii".to_string(),
    //     serde_json::to_string(&centrality_distribution(instance))?,
    // );

    output.insert("apx-diameter".to_string(), json!(apx_diameter(instance).1));

    if let Some(info) = additional_info {
        output.insert(
            "additional_info".to_string(),
            json!(info),
        );
    }

    let json = serde_json::to_string_pretty(&Value::Object(output))?;
    file.write_all(json.as_bytes())?;
    Ok(json)
}
