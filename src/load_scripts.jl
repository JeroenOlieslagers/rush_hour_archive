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
include("markov_chains.jl")
include("summary_statistics.jl")
include("plotting.jl")



function load_processed_data()
    # Raw data must be available
    println("Loading raw data...")
    df_raw = load_raw_data()
    df_filtered = filter_subjects(df_raw)
    # Dictionary containing distance to goal for every state in every puzzle
    # Keys are puzzles, value is dictionary with keys state, and values d_goal
    if isfile("data/processed_data/d_goals_prbs.jld2")
        println("Loading distance to goal...")
        d_goals_prbs = load("data/processed_data/d_goals_prbs.jld2")
    else
        println("d_goal dictionary not found, creating now...")
        d_goals_prbs = get_d_goals(prbs)
        save("data/processed_data/d_goals_prbs.jld2", d_goals_prbs)
    end
    df = pre_process(df_filtered, d_goals_prbs)

    # Dictionary with AND-OR trees, neighbours of states, and features
    if isfile("data/processed_data/stuff.jld2")
        println("Loading AND-OR trees...")
        stuff = load("data/processed_data/stuff.jld2")
    else
        println("File containing AND-OR trees and other details not found, creating now...")
        println("(this file is 840MB, do it might take ~15min to finish this)")
        stuff = first_pass(df);
        save("data/processed_data/stuff.jld2", stuff)
    end

    df[!, :tree] = stuff["trees"]
    df[!, :dict] = stuff["dicts"]
    df[!, :all_moves] = stuff["all_moves"]
    df[!, :neighs] = stuff["neighs"]
    df[!, :features] = stuff["features"]

    # Q and R matrices for Markov Chain fitting of forward model
    if isfile("data/processed_data/dict.jld2")
        println("Loading MC matrices...")
        dict = load("data/processed_data/dict.jld2")
    else
        println("File containing Matrices for forward search model not found, creating now...")
        dict = get_QR_dict(prbs)
        save("data/processed_data/dict.jld2", dict)
    end

    # NLL, cross-validated NLL, and parameters of all models
    if isfile("data/processed_data/df_models.csv") && isfile("data/processed_data/params.jld2")
        println("Loading models...")
        df_models = CSV.read("data/processed_data/df_models.csv", DataFrame)
        params = load("data/processed_data/params.jld2")["params"]
    else
        println("File containing model parameters and cross-validated NLL not found, creating now...")
        println("WARNING: THIS IS VERY EXPENSIVE AND TAKES A VERY LONG TIME TO FINISH")
        println("I SUGGEST YOU DOWNLOAD THESE FILES AND START AGAIN")
        println("Because fitting the forward search model is so expensive, this will only run for one subject.")
        
        df_models, params = fit_all_models(df, d_goals_prbs, dict; m=1)
        CSV.write("data/processed_data/df_models.csv", df_models)
        save("data/processed_data/params.jld2", "params", params)
    end
    df_models[!, :params] = params

    # Summary statistics of subjects and all models
    if isfile("data/processed_data/df_stats.csv")
        println("Loading summary stats...")
        df_stats = CSV.read("data/processed_data/df_stats.csv", DataFrame)
    else
        println("File containing summary statistics not found, creating now...")
        println("WARNING: THIS MIGHT TAKE ~15min")
        # We have to fit the forward model on the cluster since it is expensive,
        # so we assume the fitted parameters are in `params`.
        # This obtains the final column needed for inference in the forward search model.
        if isfile("data/processed_data/mc_dict.jld2")
            println("Loading MC column...")
            mc_dict = load("data/processed_data/mc_dict.jld2")
        else
            println("File containing final column for forward search model not found, creating now...")
            mc_dict = get_mc_dict(df, params, dict)
            save("data/processed_data/mc_dict.jld2", mc_dict)
        end
        df_stats = calculate_summary_statistics(df, df_models, d_goals_prbs, mc_dict, dict)
        CSV.write("data/processed_data/df_stats.csv", df_stats)
    end

    # Binned summary stats where the x-axis is distance to goal
    if isfile("data/processed_data/binned_stats.csv")
        println("Loading binned stats...")
        binned_stats = CSV.read("data/processed_data/binned_stats.csv", DataFrame)
    else
        println("File containing binned d_goal statistics not found, creating now...")
        binned_stats = bin_stats(df_stats, :X_d_goal)
        CSV.write("data/processed_data/binned_stats.csv", binned_stats)
    end

    return df, df_filtered, d_goals_prbs, df_models, df_stats, binned_stats
end

function get_subjs_prbs(df)
    prbs = unique(df.puzzle)[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in unique(df.puzzle)])]
    subjs = unique(df.subject)
    return subjs, prbs
end