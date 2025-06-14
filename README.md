# PACE 2025 - Hitting Set
This codebase calculates a hitting set for a given hypergraph - not necissarily the minimum one, as it is a heuristic solver.

## Installing/ compiling
Make sure cargo and rust are up-to-date.
First option: Run `cargo build --release` for usage with graph.hgr and settings.json files
Second option: Run `cargo build --release --features optilio` to take the input from stdin and print the solution to stdout (as required for the PACE challenge)

### Install Dependencies
Install glpk (you can find it [here](https://www.gnu.org/software/glpk/)). Under ubuntu, you may use ```apt install glpk-utils glpk-doc libglpk-dev```

## Usage
For the first option: Use `hitdomsolver solve <hypergraph-file.hgr> <settings-file>`. A standard `settings.json` file is already in the repo - when looking for an exact dominating set, enable_lp_reduction should be set to false.
For the second option: Give the inputfile as stdin to the solver, for example like this: `cat ./test_hs/bremen_subgraph_100.hgr | ./target/release/hitdomsolver`

## Instances
Under `test_hs/` you can find a selection of `*.hgr` files, which we used to benchmark our algorithm.

## Reference
This codebase is based on this [solver](https://github.com/Felerius/findminhs), which is described in this paper:
> Thomas Bläsius, Tobias Friedrich, David Stangl, and Christopher Weyand.  
> *An Efficient Branch-and-Bound Solver for Hitting Set*.  
> In Cynthia A. Phillips and Bettina Speckmann (Eds.), Proceedings of the Symposium on Algorithm Engineering and Experiments (ALENEX 2022), Alexandria, VA, USA, January 9–10, 2022, pages 209–220. SIAM, 2022.  
> [https://doi.org/10.1137/1.9781611977042.17](https://doi.org/10.1137/1.9781611977042.17)