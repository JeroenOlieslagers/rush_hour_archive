# Rush Hour
Code to reproduce figures from paper.

```
rush_hour
├── README.md
├── Project.toml              : tells julia which packages to install
├── Manifest.toml
├── demos
├── figures                   : contains all figures from paper
├── data
│   ├── processed_data
│   │   ├── binned_stats.csv
│   │   ├── d_goals_prbs.jld3
│   │   ├── df_models.csv
│   │   ├── df_stats.csv
│   │   ├── dict.jld2
│   │   ├── mc_dict.jld2
│   │   ├── params.jld2
│   │   └── stuff.jld2
│   └── raw_data
│       ├── problems          : directory containing all puzzles for all move data from experiment in .json format
│       └── all_subjects.csv  : raw data
└── src
    ├── and_or_model.jl       : AND-OR tree traversal model
    ├── and_or_trees.jl       : generate AND-OR trees
    ├── data_analysis.jl      : additional functions to analyze data for supplementary figures
    ├── load_scripts.jl       : load all functions, scripts, and data
    ├── main.jl               : central script
    ├── markov_chains.jl      : additional functions for forwar search model
    ├── model_fitting.jl      : function to fit and cross-validated models
    ├── models.jl             : all models to be tested
    ├── plotting.jl           : functions to produce figures from paper
    ├── pre_process.jl        : filter and process data into suitable format
    ├── representation.jl     : types, structs, and converter functions
    ├── rush_hour.jl          : game logic
    ├── search.jl             : BFS to compute d_goal for every state
    ├── summary_statistics.jl : compute summary statistics for every state
    └── visualization.jl      : board state and AND-OR tree visualizers



```

# Installation
Instructions for how to install the necessary files

# Usage (Experiment)
Instructions for how to run experiments

# Usage (Analysis)
Instructions for how to run analysis

## Demo(s)
Description of demos

# Authors and Acknowledgements
Jeroen Olieslagers

# License
Contact lab about licensing code
