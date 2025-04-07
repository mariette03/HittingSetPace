# PACE 2025 - Hitting Set
Patrick Steil (100% of the added coding), Mariette Vasen

## Installing
Make sure cargo and rust are up-to-date.
Run `cargo build <--release>`

### Install Dependencies
Install glpk (you can find it [here](https://www.gnu.org/software/glpk/)). Under ubuntu, you may use ```apt install glpk-utils glpk-doc libglpk-dev```


## Usage
Use `findminhs solve <hypergraph-file.hgr> <settings-file>`. A `settings.json` file is already in the repo.

## Instances
Under `test_hs/` you can find a selection of `*.hgr` files, which we used to benchmark our algorithm.