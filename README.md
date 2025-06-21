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
For the second option: Give the inputfile to the solver via stdin, for example like this: `cat ./test_hs/bremen_subgraph_100.hgr | ./target/release/hitdomsolver`

## Instances
Under `test_hs/` you can find a selection of `*.hgr` files, which we used to benchmark our algorithm.

## Reference
This codebase is based on this [solver](https://github.com/Felerius/findminhs), which is described in this paper:
> Thomas Bläsius, Tobias Friedrich, David Stangl, and Christopher Weyand.  
> *An Efficient Branch-and-Bound Solver for Hitting Set*.  
> In Cynthia A. Phillips and Bettina Speckmann (Eds.), Proceedings of the Symposium on Algorithm Engineering and Experiments (ALENEX 2022), Alexandria, VA, USA, January 9–10, 2022, pages 209–220. SIAM, 2022.  
> [https://doi.org/10.1137/1.9781611977042.17](https://doi.org/10.1137/1.9781611977042.17)
>

## Sovler description
Our solver is a branch and bound solver, meaning we make a decision to include or exclude a vertex and solve the remaining instances, respectively. Before bracnhing, we check whether the upper and lower bound for a hitting set are equal, meaning we have found the optimal size for the hitting set in this branch already. We also reduce the instance using different simple reduction rules before branching. See the referenced paper for a more detailed description of the bounding- and reduction rules. Our solver includes a possibility to turn them on or off based on a settings file you give to the solver along with your input graph (provided you don't use the optil mode). A hyperparameter search could also be conducted to tune these parameters.

Aside from a very minor tweaks, we introduced the usage of a LP and ILP sovler: 
- We use the LP to prioritize how to branch: Vertices with a higher value in the LP are first included in the hitting set (this property is called vertex importance in our code)
- The LP solution is also used to reduce the instance. We first aimed for the exact track, having the hypothesis that one could use a similar approach as in the Vertex Cover problem. To use the LP solution, which can be turned into a half integral solution in the vertex cover case (in the case of hitting set into a d!-integral solution, at least from our rough thoughts), to reduce the instance as follows: The vertices assigned with a 1 get chosen for the vertex cover, the ones assigned with a 0 get excluded, and the remaining part of the graph gets solved on its own. We tried to proof the same procedure for the hitting set problem, but unfortunately, applying this reduction rule in combination with the rest of the solver yielded suboptimal results, which is why we think it is not proofable. However, we decided to include this very fast reduction in the heuristic path which this solver is for. You can read more about the reduction rule for vertex cover and its proof of correctness here (https://doi.org/10.1007/BF01580444)
- We use an ILP solver for instances that are "small enough" (also a setting you set in your settings file). We do this in order to avoid many small branching steps without many possible reductions.
