include("data_analysis.jl")
using FileIO

data = load("analysis/processed_data/filtered_data.jld2")["data"]
