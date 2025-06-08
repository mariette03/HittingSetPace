#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::similar_names, clippy::cast_possible_truncation)]
use crate::{instance::Instance, report::IlpReductionReport};
use anyhow::{anyhow, Result};
use log::{debug, info};
use std::{
    ffi::OsStr,
    fs::File,
    io::{self, BufReader, BufWriter},
    path::PathBuf,
    time::Instant,
};
use std::ffi::OsString;
use serde_json::{json, Map, Value};
use structopt::{clap::AppSettings, StructOpt};
use crate::analysis::utils::graph_data_report;
use crate::report::GreedyMode;
use crate::report::Settings;



mod ds_instance;
mod data_structures;
mod instance;
mod lower_bound;
mod reductions;
mod report;
mod small_indices;
mod solve;
mod strategies;
mod utils;
pub(crate) mod lp_solver;
mod analysis;
mod hs_instance;

const APP_SETTINGS: &[AppSettings] = &[
    AppSettings::DisableHelpSubcommand,
    AppSettings::SubcommandRequiredElseHelp,
    AppSettings::VersionlessSubcommands,
];
const GLOBAL_APP_SETTINGS: &[AppSettings] =
    &[AppSettings::ColoredHelp, AppSettings::UnifiedHelpMessage];

#[cfg(not(feature = "optilio"))]
#[derive(Debug, StructOpt)]
#[structopt(settings = APP_SETTINGS, global_settings = GLOBAL_APP_SETTINGS)]
/*
struct CliOpts {
    #[structopt(subcommand)]
    cmd: Option<Commands>,
}
    */

// #[derive(Debug, StructOpt)]
enum CliOpts {
    /// Run the solver on a given hypergraph
    Solve(SolveOpts),

    /// Analysis of the given (hyper)graph
    Analysis(AnalysisOpts),

    /// Convert a hypergraph into an equivalent ILP
    Ilp(IlpOpts),
}


#[derive(Debug, StructOpt)]
struct CommonOpts {
    /// Input hypergraph
    #[structopt(parse(from_os_str), value_name = "hypergraph-file")]
    hypergraph: PathBuf,

    /// Use the json format for the input hypergraph rather than the text-based one.
    #[structopt(short, long)]
    json: bool,
}

impl CommonOpts {
    #[cfg(not(feature = "optilio"))]
    fn load_instance(&self) -> Result<Instance> {
        let reader = BufReader::new(File::open(&self.hypergraph)?);
        // for PaceChallenge
        Instance::load_from_hgr(reader)
        // if self.json {
        //     Instance::load_from_json(reader)
        // } else {
        //     Instance::load_from_text(reader)
        // }
    }

    #[cfg(feature = "optilio")]
    fn load_instance(&self) -> Result<Instance> {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        Instance::load_from_hgr(reader)
    }

}

#[derive(Debug, StructOpt)]
struct AnalysisOpts {
    #[structopt(flatten)]
    common: CommonOpts,

    /// Solver settings
    #[structopt(long, parse(from_os_str), value_name = "settings-file")]
    settings: Option<PathBuf>,

    /// Write the final hitting set to this file as a json array
    #[structopt(long, parse(from_os_str), value_name = "solution-file")]
    solution: Option<PathBuf>,

    /// Write a detailed statistics report to this file formatted as json
    #[structopt(long, parse(from_os_str), value_name = "report-file")]
    report: Option<PathBuf>,
}

#[derive(Debug, StructOpt)]
struct IlpOpts {
    #[structopt(flatten)]
    common: CommonOpts,

    /// Reduce the hypergraph first by applying vertex and edge domination rules
    #[structopt(long)]
    reduced: bool,

    /// Write a json report about the applied reductions to this file
    #[structopt(
        short,
        long,
        parse(from_os_str),
        requires("reduced"),
        value_name = "file"
    )]
    report: Option<PathBuf>,
}

#[derive(Debug, StructOpt)]
struct SolveOpts {
    #[structopt(flatten)]
    common: CommonOpts,

    /// Solver settings
    #[structopt(parse(from_os_str), value_name = "settings-file")]
    settings: PathBuf,

    /// Write the final hitting set to this file as a json array
    #[structopt(short, long, parse(from_os_str), value_name = "file")]
    solution: Option<PathBuf>,

    /// Write a detailed statistics report to this file formatted as json
    #[structopt(short, long, parse(from_os_str), value_name = "file")]
    report: Option<PathBuf>,
}

impl SolveOpts {
    fn new(common: CommonOpts, settings: PathBuf, solution: Option<PathBuf>, report: Option<PathBuf>) -> Self {
        SolveOpts {
            common,
            settings,
            solution,
            report,
        }
    }

    fn hardcoded_solve_opts() -> Self {
        SolveOpts {
            common: CommonOpts {
                hypergraph: PathBuf::from("default.hg"), // hardcoded todo this solution is super ugly
                json: false,
            },
            settings: PathBuf::from("default_settings.conf"), // hardcoded
            solution: None,
            report: None,
        }
    }
}

fn solve(opts: SolveOpts) -> Result<()> {
    debug!("Solving...");
    #[cfg(not(feature = "optilio"))]
    let file_name = opts
        .common
        .hypergraph
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| anyhow!("File name can't be extracted"))?
        .to_string();
    #[cfg(feature = "optilio")]
    let file_name = "hardcode".to_string(); // TODO this is an intermediate solution

    let instance = opts.common.load_instance()?;

    #[cfg(not(feature = "optilio"))]
    let settings = {
        let reader = BufReader::new(File::open(&opts.settings)?);
        serde_json::from_reader(reader)?
    };
    #[cfg(feature = "optilio")]
    let settings = get_hardcoded_settings();

    info!("Solving {:?}", &opts.common.hypergraph);
    let (final_hs, report) = solve::solve(instance, file_name, settings)?;

    // PaceChal Output
    print!("{}\n", final_hs.len());
    for h in &final_hs {
        print!("{}\n", ((usize::from(*h)) + 1));
    }
    
    #[cfg(not(feature = "optilio"))]
    {
        if let Some(solution_file) = opts.solution {
            debug!("Writing solution to {}", solution_file.display());
            let writer = BufWriter::new(File::create(&solution_file)?);
            serde_json::to_writer(writer, &final_hs)?;
        }
        if let Some(report_file) = opts.report {
            debug!("Writing report to {}", report_file.display());
            let writer = BufWriter::new(File::create(&report_file)?);
            serde_json::to_writer(writer, &report)?;
        }
    }

    Ok(())
}

#[cfg(feature = "optilio")]
fn get_hardcoded_settings() -> Settings {
    Settings{
        enable_local_search: false,
        enable_max_degree_bound: true,
        enable_sum_degree_bound: false,
        enable_efficiency_bound: true,
        enable_packing_bound: true,
        enable_sum_over_packing_bound: true,
        packing_from_scratch_limit: 3,
        greedy_mode: GreedyMode::Once, // Never, Once, AlwaysBeforeBounds, AlwaysBeforeExpensiveReductions,
        initial_hitting_set: None,
        enable_lp_lower_bound: true,
        ilp_size: 100,
        degree_one_removal: true,
        lp_guided: true,
        stop_at: 0,
    }
}

fn convert_to_ilp(opts: IlpOpts) -> Result<()> {
    debug!("get ILP formulation ...");
    let mut instance = opts.common.load_instance()?;

    if opts.reduced {
        let time_before = Instant::now();
        let (reduced_vertices, reduced_edges) = reductions::reduce_for_ilp(&mut instance);
        if let Some(report_file) = opts.report {
            let report = IlpReductionReport {
                runtime: time_before.elapsed(),
                reduced_vertices,
                reduced_edges,
            };
            let log_writer = BufWriter::new(File::create(&report_file)?);
            serde_json::to_writer(log_writer, &report)?;
        }
    }

    let stdout = io::stdout();
    instance.export_as_ilp(stdout.lock())
}

fn instance_analysis(opts: AnalysisOpts) -> Result<()> {
    debug!("analyse instance ...");
    let file_name = opts
        .common
        .hypergraph
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| anyhow!("File name can't be extracted"))?
        .to_string();
    let instance = opts.common.load_instance()?;
    // Ok(graph_data_report(&instance, file_name.as_str())?)

    let solver_info = match (opts.settings, opts.solution, opts.report) {
        (Some(settings_path), Some(solution_path), Some(report_path)) => {
            debug!("running solver");
            let mut additional_info = Map::new();
            additional_info.insert("settings_path".to_string(), json!(settings_path));
            additional_info.insert("solution_path".to_string(), json!(solution_path));
            additional_info.insert("report_path".to_string(), json!(report_path));
            solve(SolveOpts::new(
                opts.common, settings_path, Some(solution_path), Some(report_path),
            ))?;
            Some(serde_json::to_string_pretty(&Value::Object(additional_info))?)
        }
        _ => None
    };

    graph_data_report(&instance, file_name.as_str(), solver_info)?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::new().filter_or("FINDMINHS_LOG", "info"))
        .format_timestamp_millis()
        .init();

    /*
    let cli = CliOpts::from_args();

    match cli.cmd.unwrap_or(Commands::Solve(SolveOpts::our_default())) {
        Commands::Solve(solve_opts) => solve(solve_opts),
        Commands::Ilp(ilp_opts) => convert_to_ilp(ilp_opts),
        Commands::Analysis(analysis_opts) => instance_analysis(analysis_opts),
    }   
     */

    #[cfg(feature = "optilio")]
    {
        let solve_opts = SolveOpts::hardcoded_solve_opts();
        return solve(solve_opts);
    }
    
    #[cfg(not(feature = "optilio"))]
    {
        let opts = CliOpts::from_args();
        match opts {
            CliOpts::Solve(solve_opts) => solve(solve_opts),
            CliOpts::Ilp(ilp_opts) => convert_to_ilp(ilp_opts),
            CliOpts::Analysis(analysis_opts) => instance_analysis(analysis_opts),
        }   
    }

}
