use crate::instance::Instance;
use crate::solve::Status;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// Structure representing the tree decomposition.
struct TreeDecomposition {
    num_bags: usize,
    bag_width: usize, // width + 1, as provided by the file
    num_vertices: usize,
    bags: HashMap<usize, Vec<usize>>, // Map bag id to its vertices
    edges: Vec<(usize, usize)>,       // Edges connecting bag ids
}

#[derive(Debug)]
enum NiceTreeNode {
    Leaf { bag: Vec<usize> },
    Introduce { vertex: usize, child: Box<NiceTreeNode>, bag: Vec<usize> },
    Forget { vertex: usize, child: Box<NiceTreeNode>, bag: Vec<usize> },
    Join { left: Box<NiceTreeNode>, right: Box<NiceTreeNode>, bag: Vec<usize> },
}

/// A helper structure representing a node in the raw (rooted) tree decomposition.
#[derive(Debug)]
struct RawNode {
    id: usize,
    bag: Vec<usize>,
    children: Vec<RawNode>,
}


/// Solve the instance assuming it has bounded treewidth
fn solve_via_treewidth(instance: &Instance) -> Status {
    //TODO: Use a TW-approach, needs to compute tw before, then do a dp on that
    let nice_tree_root = compute_nice_tree_decomposition(instance);
    /// TODO: solve using DP
    Status::Stop
}

/// Compute decomposition
fn compute_nice_tree_decomposition(instance: &Instance) -> Box<NiceTreeNode> {
    // TODO call to function of htd
    let root_id = 1;
    let tree_decomposition = parse_resulting_td_file("TODO").expect("Failed to load file");
    let raw_tree_root = build_raw_tree(&tree_decomposition, root_id).expect("Something went wrong converting tree decomp to raw_tree");
    let nice_tree_root = assign_node_types(&raw_tree_root, None);
    nice_tree_root    
}

fn parse_resulting_td_file(file_path: &str) -> io::Result<TreeDecomposition> {
    let file = File::open(file_path)?;
    let reader = io::BufReader::new(file);

    let mut num_bags = 0;
    let mut bag_width = 0;
    let mut num_vertices = 0;
    let mut bags = HashMap::new();
    let mut edges = Vec::new();
    let mut header_parsed = false;

    for line in reader.lines() {
        let line = line?; // unwrap the line
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            // Skip comments and empty lines
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if !header_parsed {
            // We expect the header line to start with "s td"
            if tokens.len() < 4 || tokens[0] != "s" || tokens[1] != "td" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Expected header line starting with 's td'",
                ));
            }
            // Parse number of bags, bag width (width+1) and number of vertices
            num_bags = tokens[2].parse().map_err(|_| io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid number for bags",
            ))?;
            bag_width = tokens[3].parse().map_err(|_| io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid bag width (width+1)",
            ))?;
            // tokens[4] holds the number of vertices (optional check)
            num_vertices = tokens.get(4)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing vertex count"))?
                .parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid number for vertices"))?;
            header_parsed = true;
        } else if tokens[0] == "b" {
            // Bag line: Format: b <bag_id> <vertex1> <vertex2> ...
            if tokens.len() < 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Bag line is too short",
                ));
            }
            let bag_id: usize = tokens[1].parse().map_err(|_| io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid bag id",
            ))?;
            // Parse the vertices. Bags can be empty.
            let vertices: Vec<usize> = tokens.iter().skip(2)
                .map(|s| s.parse::<usize>().map_err(|_| io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid vertex id",
                )))
                .collect::<Result<Vec<_>, _>>()?;
            bags.insert(bag_id, vertices);
        } else {
            // If not a header or bag line, it's an edge line.
            // Edge line: Two integers (bag ids) with the first smaller than the second.
            if tokens.len() != 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Edge line does not contain exactly two values",
                ));
            }
            let bag1: usize = tokens[0].parse().map_err(|_| io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid bag id in edge",
            ))?;
            let bag2: usize = tokens[1].parse().map_err(|_| io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid bag id in edge",
            ))?;
            edges.push((bag1, bag2));
        }
    }
    
    // (Optional) Check that we have exactly num_bags bags parsed.
    if bags.len() != num_bags {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Number of bag lines does not match header information",
        ));
    }
    
    Ok(TreeDecomposition {
        num_bags,
        bag_width,
        num_vertices,
        bags,
        edges,
    })
}

/// Given the parsed TreeDecomposition (which is an undirected tree),
/// construct a rooted RawNode tree using the provided root bag id.
fn build_raw_tree(td: &TreeDecomposition, root_id: usize) -> Option<RawNode> {
    // Build an adjacency list for the bags.
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(u, v) in &td.edges {
        adj.entry(u).or_default().push(v);
        adj.entry(v).or_default().push(u);
    }

    // Start a DFS from the root.
    let mut visited = HashSet::new();

    fn dfs(current: usize, adj: &HashMap<usize, Vec<usize>>, 
           bags: &HashMap<usize, Vec<usize>>, visited: &mut HashSet<usize>) -> RawNode {
        visited.insert(current);
        let mut children = Vec::new();
        if let Some(neighbors) = adj.get(&current) {
            for &nbr in neighbors {
                if !visited.contains(&nbr) {
                    // Recursively build the subtree for each unvisited neighbor.
                    children.push(dfs(nbr, adj, bags, visited));
                }
            }
        }
        RawNode {
            id: current,
            bag: bags.get(&current).cloned().unwrap_or_default(),
            children,
        }
    }

    if td.bags.contains_key(&root_id) {
        Some(dfs(root_id, &adj, &td.bags, &mut visited))
    } else {
        None
    }
}

/// Convert the raw tree (RawNode) into a nice tree decomposition (NiceTreeNode). 
/// (Note that this does not change the decomposition, just assigns node types to each node)
/// This function assumes that the difference between a parent and child bag is exactly one vertex,
/// or that a node with two children represents a join with identical bags.
fn assign_node_types(node: &RawNode, parent_bag: Option<&Vec<usize>>) -> Box<NiceTreeNode> {
    // Base: if node has no children, it's a Leaf.
    if node.children.is_empty() {
        return Box::new(NiceTreeNode::Leaf { bag: node.bag.clone() });
    }

    // If there is one child, we decide if it's an Introduce or Forget.
    if node.children.len() == 1 {
        let child = &node.children[0];
        let mut nice_child = assign_node_types(child, Some(&node.bag));

        // Compare the parent's bag (node.bag) with the child's bag.
        let parent = &node.bag;
        let child_bag = &child.bag;
        if child_bag.len() == parent.len() + 1 {
            // Child introduced one vertex.
            let vertex = *child_bag.iter().find(|v| !parent.contains(v))
                .expect("Expected one new vertex");
            nice_child = Box::new(NiceTreeNode::Introduce {
                vertex,
                child: nice_child,
                bag: child_bag.clone(),
            });
        } else if child_bag.len() + 1 == parent.len() {
            // Child forgot one vertex.
            let vertex = *parent.iter().find(|v| !child_bag.contains(v))
                .expect("Expected one vertex to be forgotten");
            nice_child = Box::new(NiceTreeNode::Forget {
                vertex,
                child: nice_child,
                bag: child_bag.clone(),
            });
        }
        // If the bags are identical, no introduce/forget node is needed.
        return nice_child;
    }

    // If the node has two children, assume it's a join node.
    if node.children.len() == 2 {
        let left = assign_node_types(&node.children[0], Some(&node.bag));
        let right = assign_node_types(&node.children[1], Some(&node.bag));
        return Box::new(NiceTreeNode::Join {
            left,
            right,
            bag: node.bag.clone(),
        });
    }

    // If more than two children, combine them pairwise as join nodes.
    let mut iter: std::slice::Iter<'_, RawNode> = node.children.iter();
    let first = assign_node_types(iter.next().unwrap(), Some(&node.bag));
    let join_node = iter.fold(first, |acc, child| {
        let right = assign_node_types(child, Some(&node.bag));
        Box::new(NiceTreeNode::Join {
            left: acc,
            right,
            bag: node.bag.clone(),
        })
    });
    join_node
}


