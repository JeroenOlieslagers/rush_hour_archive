module RushHour
    using JSON
    using StaticArrays
    using BenchmarkTools
    using DataStructures
    using DataFrames
    using CSV
    using JLD2
    using StatsPlots
    using StatsBase
    using ProgressBars
    using Random
    using MLUtils
    using Plots
    using GraphViz
    using LaTeXStrings
    using Optim
    using LinearAlgebra
    using SparseArrays
    include("representations.jl")
    include("rush_hour.jl")
    include("visualization.jl")
    include("search.jl")
    include("pre_process.jl")
    include("data_analysis.jl")
    include("and_or_trees.jl")
    include("and_or_model.jl")
    include("models.jl")
    include("model_fitting.jl")
    include("model_validation.jl")
    include("plotting.jl")
end