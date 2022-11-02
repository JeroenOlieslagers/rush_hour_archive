include("data_analysis.jl")
include("solvers.jl")
# using FileIO
# using Distributions
# using StatsPlots
# using MixedModels
using StatsModels
using StatsBase
using GLM
# using MLJLinearModels
# using ColorSchemes
# using LaTeXStrings
using Combinatorics
using ProgressBars
using Colors
# using MLDataUtils


"""
    get_Ls(data)

Return dictionary where keys are problem ids and values are arrays of lengths of people's solution.
"""
function get_Ls(data::Dict{String, DataFrame})::Tuple{DefaultDict{String, Array{Int64}, Vector{Any}}, DefaultDict{String, Dict{String, Int64}, Dict{Any, Any}}}
    Ls = DefaultDict{String, Array{Int}}([])
    dict = DefaultDict{String, Dict{String, Int}}(Dict())
    # Loop over all subjects
    for subj in keys(data)
        # Get L for each problem subject attempted
        _, _, attempts = analyse_subject(data[subj]);
        for prb in keys(attempts)
            # Push L to array for each problem
            push!(Ls[prb], attempts[prb])
            dict[subj][prb] = attempts[prb]
        end
    end
    return Ls, dict
end


"""
    get_state_spaces(prbs)

Fully traverses every game by BFS tree search and returns number of solutions,
tree structure, parents and children and number of visits for each node. From these,
returns all necessary features:
1. optimal length
2. average branching factor on whole graph of moving downstream (average number of children)
(equal to average outdegree)
3. average indegree on optimal paths to optimal solution (averaged over optimal solutions and paths)
4. average outdegree on optimal paths to optimal solution (averaged over optimal solutions and paths)
5. number of solutions
6. number of optimal paths to any solution
7. size of state space
8. average width of state space
9. minimum width of state space
10. maximum width of state space
11. depth of state space
12. number of nodes in constrained MAG
"""
function get_state_spaces(prbs::Vector{String})::Dict{String, Vector{Float64}}
    state_space_features = Dict{String, Vector{Float64}}()
    for prb in ProgressBar(prbs)
        board = load_data(prb)
        tree, seen, stat, dict, parents, children, solutions = bfs_path_counters(board, traverse_full=true)
        optimal_length = minimum([stat[solution][1] for solution in solutions])
        average_children = mean([length(childs) for childs in values(children)])
        # Calculate in and out degree for optimal paths to optimal end states
        optimal_ends = [stat[solution][1] for solution in solutions]
        optimal_ends = solutions[optimal_ends .== optimal_length]
        in_n_out = [get_av_branchs(optimal_end, parents, children, optimal_length) for optimal_end in optimal_ends]
        in_degree = mean([t[1] for t in in_n_out])
        out_degree = mean([t[2] for t in in_n_out])
        
        n_solutions = length(solutions)
        n_optimal_paths = sum([seen[solution] for solution in solutions])
        # Make sure everything is working
        if !(sum(values(tree)) == length(seen) == length(stat) == length(dict) == length(parents) == length(children))
            println("ERROR, STATE SPACES DO NOT AGREE")
            break
        end
        size_state_space = length(stat)
        tree_shape = collect(values(tree))[2:end]
        av_width = mean(tree_shape)
        min_width = minimum(tree_shape)
        max_width = maximum(tree_shape)
        depth = length(tree_shape)
        arr = get_board_arr(board)
        mag_size = mag_size_nodes(board, arr)
        state_space_features[prb] = [optimal_length, average_children, in_degree, out_degree, n_solutions, n_optimal_paths, size_state_space, av_width, min_width, max_width, depth, mag_size]
    end
    return state_space_features
end

"""
    get_av_branchs(optimal_moves)

Returns average in and out degree of random optimal path to leaf.
Can average over random paths by setting N>1.
"""
function get_av_branchs(leaf, parents, children, depth; N=1)::Tuple{Float64, Float64}
    # Initialise
    total_in = 0.0
    total_out = 0.0
    for n in 1:N
        outdegree = 0
        indegree = 0
        parent = sample(parents[leaf])
        for i in 1:depth
            outdegree += length(children[parent])
            # Choose random parent if more than 1
            if i < depth
                indegree += length(parents[parent])
                parent = sample(parents[parent])
            end
        end
        total_in += indegree / (depth-1)
        total_out += outdegree / depth
    end
    return total_in / N, total_out / N
end

"""
get_random_agent_stats(prbs, N=100)

Returns mean and standard deviation of moves a random agent performs to solve the problem
over N iterations.
"""
function get_random_agent_stats(prbs::Vector{String}; N=100)::Dict{String, Vector{Float64}}
    dict = Dict{String, Vector{Float64}}()
    for prb in prbs
        println(prb)
        exps = zeros(Int, N)
        for i in 1:N
            board = load_data(prb)
            exps[i] = random_agent(board)
        end
        # A star solvers
        board = load_data(prb)
        _, _, expansions_bfs = a_star(board)
        board = load_data(prb)
        _, _, expansions_red = a_star(board, h=red_distance)
        board = load_data(prb)
        _, _, expansions_forest = a_star(board, h=multi_mag_size_nodes)
        dict[prb] = [mean(exps), std(exps), expansions_bfs, expansions_red, expansions_forest]
    end
    return dict
end

"""
    get_dataframe(prbs, dict, state_space_features, random_agent_stats)

Creates dataframe for each subject and puzzle index with each of 13 features as well as subject solution length.
"""
function get_dataframe(prbs::Vector{String}, dict::DefaultDict{String, Dict{String, Int64}, Dict{Any, Any}}, state_space_features::Dict{String, Vector{Float64}}, random_agent_stats::Dict{String, Vector{Float64}})
    df = DataFrame(
        subj=String[], 
        prb=String[],
        L=Int[],
        opt_L=Int[], 
        branch=Float64[], 
        indegree=Float64[], 
        outdegree=Float64[],
        n_sol=Int[],
        paths=Int[],
        ss_size=Int[],
        av_width=Float64[],
        max_width=Int[],
        min_width=Int[],
        depth=Int[],
        mag_nodes=Int[],
        mean_random=Float64[],
        std_random=Float64[],
        exp_bfs=Int[],
        exp_red=Int[],
        exp_forest=Int[])

    for subject in keys(dict)
        subj_prbs = dict[subject]
        for prb in keys(subj_prbs)
            push!(df, [subject, prb, subj_prbs[prb], state_space_features[prb]..., random_agent_stats[prb]...])
        end
    end
    # Normalise variance
    for column in names(df)[4:end]
        df[!, column] ./= std(df[!, column])
    end
    
    return df
end

"""
    lasso_analysis(XX, av_L)

Fits lasso regressions for all features XX using range of lamba values.
Also plots correlation heatmap.
"""
function lasso_analysis(XX, av_L)

    lambdas = 10 .^(range(-3,stop=log10(2),length=10))

    colors = ColorScheme([RGB(n/100, (100-n)/100, 0) for n in 1:100])
    ps = []
    for (n, l) in enumerate(lambdas)
        en = LassoRegression(l)
        p = MLJLinearModels.fit(en, XX, av_L)
        push!(ps, p[1:end-1])
    end
    plot(ps, ylabel="Coefficient", xlabel="Feature", xticks=(1:13, names(df)[4:end]), xrotation=45, size=(400, 350), palette=colors, legend=nothing, colorbar=true, title="Lasso regression for different values of "*latexstring("\\lambda"), titlefontsize=10, dpi=400)
    plot!([2, 2.001], [0, 0], zcolor=[0, lambdas[end]], color=palette(colors), dpi=400)

    heatmap(abs.(cor(XX)), size=(400,450), yflip=true, xticks=(1:13, names(df)[4:end]), legend=nothing, xrotation=60, yticks=(1:13, names(df)[4:end]), tick_direction=:out, title="Absolute correlation coefficient", dpi=400, c=:greys)

end

"""
    exhaustive_linear_bic(df)

Fits a range of linear regression models over powerset of features. 
Returns AIC, BIC and 5-fold cross validated log likelihood.
"""
function exhaustive_linear(df)
    feature_sets = collect(powerset([feature for feature in names(df) if feature ∉ ["subj", "prb", "L"]]))
    BICs = zeros(length(collect(feature_sets)))
    AICs = zeros(length(collect(feature_sets)))
    CVs = zeros(length(collect(feature_sets)))
    for n in ProgressBar(eachindex(feature_sets))
        # Skip empty set
        if n == 1
            continue
        end
        fm = Term(:L) ~ term(1) +([Term(Symbol(name)) for name in feature_sets[n]]...)
        M = fit(LinearModel, fm, df)
        BICs[n] = GLM.bic(M)
        AICs[n] = GLM.aic(M)
        # 5 fold Cross Validation (CV)
        folds = kfolds(shuffle(1:size(df)[1]), 5)
        for fold in folds
            # Fit on 80% of data
            M = fit(LinearModel, fm, df[fold[1], :])
            # Test on 20% of data
            P = predict(M, df[fold[2], :])
            y = df[fold[2], "L"]
            CVs[n] += loglik(y, P)
        end
    end
    return AICs, BICs, CVs
end

"""
    loglik(y, P)

Returns loglikelihood of linear model on predictions P and data y.
"""
function loglik(y::Vector, P::Vector)::Float64
    dev = sum((y .- P).^2)
    n = length(y)
    return -n/2 * (log(2π * dev/n) + 1)
end

data = load("analysis/processed_data/filtered_data.jld2")["data"]

Ls, dict = get_Ls(data)
prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])]

state_space_features = get_state_spaces(jsons)

random_agent_stats = get_random_agent_stats(prbs, N=10000)

df = get_dataframe(prbs, dict, state_space_features, random_agent_stats)

#save("analysis/processed_data/all_prbs_stat_space_features.jld2", state_space_features)
#save("analysis/processed_data/new_norm_features.jld2", "data", XX)
X = load("analysis/processed_data/new_features.jld2")["data"]
state_space_features_all = load("analysis/processed_data/all_prbs_stat_space_features.jld2")
XX = load("analysis/processed_data/new_norm_features.jld2")["data"]

L = [mean(Ls[prb]) for prb in prbs]
LL = L./std(L)

plot(layout=grid(2,2), size=(700, 700))
scatter!(XX[:, 13], LL, sp=1, legend=nothing, xscale=:log10, xlabel="Mean random agent moves", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 13]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 13])), ordinalrank(LL))), sigdigits=1))*")")
scatter!(XX[:, 15], LL, sp=2, legend=nothing, xscale=:log10, xlabel="BFS expansions", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 15]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 15])), ordinalrank(LL))), sigdigits=1))*")")
scatter!(XX[:, 16], LL, sp=3, legend=nothing, xscale=:log10, xlabel="A* red distance expansions", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 16]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 16])), ordinalrank(LL))), sigdigits=1))*")")
scatter!(XX[:, 17], LL, sp=4, legend=nothing, xscale=:log10, xlabel="A* forest expansions", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 17]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 17])), ordinalrank(LL))), sigdigits=1))*")")

#plot(layout=grid(4, 3), size=(1000, 1400))
cors = zeros(12, 12)
for i in 1:12
    #f=scatter!(XX[:, i], LL, sp=i, legend=nothing, xscale=:log10, xlabel=names(df)[i+3], title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, i]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, i])), ordinalrank(LL))), sigdigits=1))*")")
    #display(f)
    for j in 1:12
        cors[i, j] = cor(ordinalrank(XX_F[:, i]), ordinalrank(XX_F[:, j]))
    end
end

cmap = [i < 500 ? RGB(0, 0, 1) : i < 1500 ? RGB(1, 1, 0) : RGB(1, 0, 0) for i in 1:2000]

heatmap(cors, yflip=true, clim=(-1, 1), size=(600, 500), xmirror=true, xticks=(1:12, names(df)[4:16]), yticks=(1:12, names(df)[4:16]), right_margin = 5Plots.mm, colorbar_title="\nSpearman correlation", xrotation=60, c=cmap)


df = DataFrame(CSV.File("analysis/processed_data/13_features.csv"))

#df = DataFrame(load("analysis/processed_data/13_features.jld2"))
#prbs = unique(df[!, "prb"])[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in unique(df[!, "prb"])])]

dff = DataFrame(XXX, names(df)[3:15])
AICs, BICs, CVs = exhaustive_linear(dff)
#save("analysis/processed_data/exhaustive_linear_BICs.jld2", "data", AICs)
#save("analysis/processed_data/exhaustive_linear_AICs.jld2", "data", BICs)
#save("analysis/processed_data/exhaustive_linear_CVs.jld2", "data", CVs)
BICs = load("analysis/processed_data/exhaustive_linear_BICs.jld2")["data"]
AICs = load("analysis/processed_data/exhaustive_linear_AICs.jld2")["data"]
CVs = load("analysis/processed_data/exhaustive_linear_CVs.jld2")["data"]

fm = @formula(L ~ opt_L)
fm = Term(:L) ~ (term(1) | Term(:subj)) +([Term(Symbol(name)) for name in names(df) if name ∉ ["subj", "prb", "L"]]...)
M = fit(MixedModel, fm, df)

M = fit(LinearModel, fm, df)

av_L = [mean(Ls[prb]) for prb in prbs]


jsons = readdir("experiment/raw_data/problems")[14:end]
jsons = [js[1:end-5] for js in jsons]


en = ElasticNetRegression(0, 0)
p = MLJLinearModels.fit(en, XX, av_L)

board = load_data("example_4")

X = zeros(70, 12)
for (i, prb) in enumerate(prbs)
    for j in 1:12
        if j <= 12
            X[i, j] = state_space_features[prb][j]
        elseif j > 12
            X[i, j] = random_agent_stats[prb][j-12]
        end
    end
end

XX = zeros(size(X))
for i in 1:17
    #XX[:, i] = X[:, i] .- mean(X[:, i])
    #XX[:, i] = XX[:, i] / norm(XX[:, i])
    XX[:, i] = X[:, i] / std(X[:, i])
end





