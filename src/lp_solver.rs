use crate::instance::{EdgeIdx, Instance, NodeIdx};
use crate::small_indices::SmallIdx;

use std::ffi::CString;
use std::ptr;
use std::os::raw::*;
use log::info;

extern crate glpk_sys as glpk;

const GLP_MIN: i32 = 1;  // Minimization objective
const GLP_LO: i32 = 2;   // Lower bound
const GLP_BV: i32 = 3;   // Binary variable
const GLP_DB: i32 = 4;   // Double bounded
// const ITERATION_LOG_INTERVAL_SECS: u64 = 60;



pub fn solve_lp(instance: &Instance) -> (usize, Vec<f64>) {
    if (instance.num_nodes() == 0 || instance.num_edges() == 0) {
        return (0, vec![0f64; instance.num_nodes_total()])
    }
    
    unsafe {
        glpk::glp_term_out(0);

        info!("Solving lp with number of rows {}", instance.num_edges());

        let lp = glpk::glp_create_prob();

        glpk::glp_set_prob_name(lp, CString::new("hitting_set_lp").unwrap().as_ptr());
        glpk::glp_set_obj_dir(lp, GLP_MIN);

        let num_sets = instance.num_edges();
        glpk::glp_add_rows(lp, num_sets as i32);

        let num_elements = instance.num_nodes_total();
        glpk::glp_add_cols(lp, num_elements as i32);

        let mut total_size = 0; //TODO is it an issue that we cast this to int32 later on?

        for i in 0..num_elements {
            let col_idx = (i + 1) as i32;

            glpk::glp_set_col_bnds(lp, col_idx, GLP_DB, 0.0, 1.0); // each node is chosen (1) or not chosen (0) - or in between
            glpk::glp_set_obj_coef(lp, col_idx, 1.0); // in the objective funciton, all nodes are added up

            let add = instance.node_degree(NodeIdx::from(i)); // for something, we are adding all node degrees

            // info!("add nd {add} {i}");
            total_size += add;
        }

        for j in 0..num_sets { // for each edge
            let row_idx = (j + 1) as i32;

            glpk::glp_set_row_bnds(lp, row_idx, GLP_LO, 1.0, f64::INFINITY); // the edge needs to be covered
        }

        // info!("Total sze: {total_size}");
        
        let mut ia: Vec<c_int> = Vec::with_capacity(total_size);
        let mut ja: Vec<c_int> = Vec::with_capacity(total_size);
        let mut ar: Vec<c_double> = Vec::with_capacity(total_size);

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
        assert_eq!(ia.len(), ja.len());
        assert_eq!(ia.len(), ar.len());

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

        info!("test nach lp solving");

        (z, vertex_importance)
    }
}

// Erkenntnisse: Knoten mehrfach gelöscht, das lag daran, dass er zu optimistischen Code geschrieben hat (genauso viele Variablen wie Knoten)
// im lp möchten wir aktuelle Anzahl edges und gesamtanzahl knoten

pub fn solve_ilp_exact(instance: &Instance) -> (usize, Vec<NodeIdx>) {
    if (instance.num_nodes() == 0 || instance.num_edges() == 0) {
        return (0, Vec::new());
    }
    unsafe {
        glpk::glp_term_out(0);
        info!("Solving ILP with number of rows {}", instance.num_edges());

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
            glpk::glp_delete_prob(lp);
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
            glpk::glp_delete_prob(lp);
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
