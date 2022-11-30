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
using HypothesisTests
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
        _, _, _, attempts = analyse_subject(data[subj]);
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
#8. average width of state space
9. minimum width of state space
10. maximum width of state space
11. depth of state space
#12. number of nodes in MAG
"""
function get_state_spaces(prbs::Vector{String})::Dict{String, Vector{Float64}}
    state_space_features = Dict{String, Vector{Float64}}()
    for prb in ProgressBar(prbs)
        board = load_data(prb)
        state_space_features[prb] = get_state_space_features(board)
    end
    return state_space_features
end

function get_state_space_features(board)
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
        return nothing
    end
    size_state_space = length(stat)
    tree_shape = collect(values(tree))[2:end]
    #av_width = mean(tree_shape)
    min_width = minimum(tree_shape)
    max_width = maximum(tree_shape)
    depth = length(tree_shape)
    #arr = get_board_arr(board)
    #mag_size = mag_size_nodes(board, arr)
    return [optimal_length, average_children, in_degree, out_degree, n_solutions, n_optimal_paths, size_state_space, min_width, max_width, depth]
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
        indegree = length(parents[leaf])
        parent = sample(parents[leaf])
        for i in 1:depth
            outdegree += length(children[parent])
            # Choose random parent if more than 1
            if i < depth
                indegree += length(parents[parent])
                parent = sample(parents[parent])
            end
        end
        total_in += indegree / depth
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
    for prb in ProgressBar(prbs)
        #println(prb)
        exps = zeros(Int, N)
        for i in 1:N
            board = load_data(prb)
            exps[i] = random_agent(board)
        end
        # A star solvers
        board = load_data(prb)
        _, _, expansions_bfs = a_star(board)
        board = load_data(prb)
        _, _, expansions_dfs = dfs(board)
        board = load_data(prb)
        _, _, expansions_red = a_star(board, h=red_distance)
        board = load_data(prb)
        _, _, expansions_forest = a_star(board, h=multi_mag_size_nodes)
        dict[prb] = [mean(exps), expansions_bfs, expansions_dfs, expansions_red, expansions_forest]
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
        min_width=Int[],
        max_width=Int[],
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

function scatter_corr_plot(X; L=[], prbs=[], full=false)
    dff = DataFrame(prb=String[],
                L=Float64[],
                opt_L=Float64[],
                depth=Float64[],
                indegree=Float64[],
                min_width=Float64[],
                branch=Float64[], 
                outdegree=Float64[],
                n_sol=Float64[],
                paths=Float64[],
                ss_size=Float64[],
                max_width=Float64[])
    
    stats = DataFrame(Feature=String[], Pearson=Float64[], Pearson_p=Float64[], Spearman=Float64[], Spearman_p=Float64[], Lesion=Float64[])
    nms = names(dff)[3:end]
    XX = zeros(size(X))
    if !full
        if length(L) > 0
            for i in 1:70
                push!(dff, [prbs[i], log10.(L[i]), log10.(X[i, 1]), X[i, 2:4]..., log10.(X[i, 5:7])..., log10.(X[i, 9:11])...])#, log10.(X[i, 10]), log10.(X[i, 11])])# X[i, [1:7..., 9:11...]]...])
            end
            fm = Term(:L) ~ +([Term(Symbol(trm)) for trm in nms]...)
            ols = lm(fm, dff)
            r2a = adjr2(ols)
        else
            for i in 1:70
                push!(dff, ["", 0, 
                log10.(X[i, 1]), 
                log10.(X[i, 11]),
                X[i, 3],
                log10.(X[i, 9]),
                X[i, 2], 
                X[i, 4],
                log10.(X[i, 5:7])..., 
                log10.(X[i, 10])])# X[i, [1:7..., 9:11...]]...])
            end
        end
    else
        XX[:, 1] = log10.(X[:, 1])
        XX[:, 2] = log10.(X[:, 10])
        XX[:, 3] = X[:, 3]
        XX[:, 4] = log10.(X[:, 8])
        XX[:, 5] = X[:, 2]
        XX[:, 6] = X[:, 4]
        XX[:, 7] = log10.(X[:, 5])
        XX[:, 8] = log10.(X[:, 6])
        XX[:, 9] = log10.(X[:, 7])
        XX[:, 10] = log10.(X[:, 9])
    end
    corr = zeros(10, 10)

    l = @layout [grid(10, 10) a{0.035w}]
    plot(layout=l, size=(800, 600), dpi=200)#, plot_title="Scatterplots")
    for i in 1:10
        if length(L) > 0
            trms = names(dff)[[3:3+i-2..., 3+i:end...]]
            fm = Term(:L) ~ +([Term(Symbol(trm)) for trm in trms]...)
            ols = lm(fm, dff)
            push!(stats, [
                nms[i], 
                cor(dff.L, dff[!, nms[i]]), 
                pvalue(CorrelationTest(dff.L, dff[!, nms[i]])),
                cor(ordinalrank(dff.L), ordinalrank(dff[!, nms[i]])), 
                pvalue(CorrelationTest(ordinalrank(dff.L), ordinalrank(dff[!, nms[i]]))),
                adjr2(ols)-r2a])
        end
        for j in 1:10
            if full
                corr[i, j] = abs(cor(XX[:, i], XX[:, j]))
            else
                corr[i, j] = -log10(pvalue(CorrelationTest(dff[!, nms[i]], dff[!, nms[j]])))
            end
            yl = ""
            xl = ""
            tm = -2Plots.mm
            lm = -2Plots.mm
            rm = 2Plots.mm
            if j == 1
                yl = nms[i]*" "^length(nms[i])
                lm = 25Plots.mm
            end
            if i == 1
                xl = (" "^length(nms[j]))*nms[j]
                tm = 20Plots.mm
            end
            if j == 10
                rm = 20Plots.mm
            end
            if full
                c = ceil(Int, corr[i, j]*100)
                c = palette(cgrad(:Spectral_11, rev=true), 100)[c]
            else
                c = corr[i, j] > 25 ? 25 : corr[i, j]
                c = ceil(Int, c)
                c = palette(cgrad(:Spectral_11, rev=true), 25)[c]
            end            
            if full
                scatter!([], [], sp=10*(i-1) + j, xticks=nothing, yticks=nothing, label=nothing, color=:black, markersize=1.5, yguide=yl, xguide=xl, left_margin = lm, right_margin = rm, top_margin = tm, background_color_subplot=c, xguidefontrotation=60, yguidefontrotation=-90, xmirror=true)
            else
                scatter!(dff[!, nms[i]], dff[!, nms[j]], sp=10*(i-1) + j, xticks=nothing, yticks=nothing, label=nothing, color=:black, markersize=1.5, yguide=yl, xguide=xl, left_margin = lm, right_margin = rm, top_margin = tm, background_color_subplot=c, xguidefontrotation=60, yguidefontrotation=-90, xmirror=true)
            end
        end
    end
    display(scatter!([], [], sp=101, xticks=nothing, yticks=nothing, marker_z=[0], label=nothing, clim=(0, 25), c=cgrad(:Spectral_11, rev=true), ticks=false, colorbar_title="-log(Pearson correlation p-value)", colorbar_titlefontsize=18, right_margin=2Plots.mm))
    return stats
end

data = load("analysis/processed_data/filtered_data.jld2")["data"]

Ls, dict = get_Ls(data)
prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])]

state_space_features = get_state_spaces(prbs)

random_agent_stats = get_random_agent_stats(prbs, N=10000)

df = get_dataframe(prbs, dict, state_space_features, random_agent_stats)

#save("analysis/processed_data/all_prbs_stat_space_features.jld2", state_space_features)
#save("analysis/processed_data/new_norm_features.jld2", "data", XX)
X = load("analysis/processed_data/new_features.jld2")["data"]
state_space_features_all = load("analysis/processed_data/full_all_prbs_stat_space_features.jld2")
XX = load("analysis/processed_data/new_norm_features.jld2")["data"]

L = [mean(Ls[prb]) for prb in prbs]
LL = L./std(L)

plot(layout=grid(2,2), size=(700, 700))
scatter!(XX[:, 13], LL, sp=1, legend=nothing, xscale=:log10, xlabel="Mean random agent moves", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 13]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 13])), ordinalrank(LL))), sigdigits=1))*")")
scatter!(XX[:, 15], LL, sp=2, legend=nothing, xscale=:log10, xlabel="BFS expansions", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 15]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 15])), ordinalrank(LL))), sigdigits=1))*")")
scatter!(XX[:, 16], LL, sp=3, legend=nothing, xscale=:log10, xlabel="A* red distance expansions", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 16]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 16])), ordinalrank(LL))), sigdigits=1))*")")
scatter!(XX[:, 17], LL, sp=4, legend=nothing, xscale=:log10, xlabel="A* forest expansions", title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, 17]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, 17])), ordinalrank(LL))), sigdigits=1))*")")

plot(layout=grid(4, 3), size=(1000, 1400), plot_title="Log feature histograms across 5082 puzzles")
cors = zeros(12, 12)
for i in 1:12
    #f=scatter!(XX[:, i], LL, sp=i, legend=nothing, xscale=:log10, xlabel=names(df)[i+3], title="Spearman r="*string(round(cor(ordinalrank(L), ordinalrank(log10.(XX[:, i]))), digits=3))*"("*string(round(pvalue(CorrelationTest(ordinalrank(log10.(XX[:, i])), ordinalrank(LL))), sigdigits=1))*")")
    histogram!(log10.(X[:, i]), sp=i, legend=nothing, xlabel=names(df)[i+3], yticks=[])
    # for j in 1:12
    #     cors[i, j] = cor(ordinalrank(XX[:, i]), ordinalrank(XX[:, j]))
    # end
end

cmap = [i < 500 ? RGB(0, 0, 1) : i < 1500 ? RGB(1, 1, 0) : RGB(1, 0, 0) for i in 1:2000]

heatmap(cors, yflip=true, clim=(-1, 1), size=(600, 500), xmirror=true, xticks=(1:12, names(df)[4:16]), yticks=(1:12, names(df)[4:16]), right_margin = 5Plots.mm, colorbar_title="\nSpearman correlation", xrotation=60, c=cgrad(:Spectral_11, rev=true))


df = DataFrame(CSV.File("analysis/processed_data/new_17_features.csv"))

#CSV.write("analysis/processed_data/new_17_features.csv", df)
#df = DataFrame(load("analysis/processed_data/13_features.jld2"))
#prbs = unique(df[!, "prb"])[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in unique(df[!, "prb"])])]


heatmap(corr, yflip=true, clim=(0, 24), size=(600, 500), xmirror=true, xticks=(1:10, nms), yticks=(1:10, nms), right_margin = 5Plots.mm, colorbar_title="log (Pearson correlation p-value)", xrotation=60, c=cgrad(:Spectral_11, rev=true))


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


jsons = readdir("experiment/raw_data/problems")[15:end]
jsons = [js[1:end-5] for js in jsons]


en = ElasticNetRegression(0, 0)
p = MLJLinearModels.fit(en, XX, av_L)

board = load_data("example_4")

X = zeros(length(state_space_features_all), 10)
for (i, prb) in enumerate(keys(state_space_features_all))
    for j in 1:10
        X[i, j] = state_space_features_all[prb][j]
    end
end
for (i, prb) in enumerate(prbs)
    for j in 1:12
        if j <= 12
            X[i, j] = state_space_features_all[prb][j]
        elseif j > 12
            X[i, j] = random_agent_stats[prb][j-12]
        end
    end
end

XX = zeros(size(X))
for i in 1:12
    #XX[:, i] = X[:, i] .- mean(X[:, i])
    #XX[:, i] = XX[:, i] / norm(XX[:, i])
    XX[:, i] = X[:, i] / std(X[:, i])
end

for i in eachindex(X[1, :])
    println(names(df)[3+i])
    println(extrema(X[:, i]))
    println("-----")
end

agent_stats = DataFrame(Feature=String[], Pearson=Float64[], Pearson_p=Float64[], Spearman=Float64[], Spearman_p=Float64[])
nms = ["10k_random_mean", "bfs_expansions", "dfs_expansions", "a_star_red_dist", "a_star_forest"]
for i in eachindex(XXX[1, :])
    push!(agent_stats, [
                nms[i], 
                cor(log10.(L), log10.(XXX[:, i])), 
                pvalue(CorrelationTest(log10.(L), log10.(XXX[:, i]))),
                cor(ordinalrank(L), ordinalrank(XXX[:, i])), 
                pvalue(CorrelationTest(ordinalrank(L), ordinalrank(XXX[:, i])))])
end

for row in findall(isnan, X)
    X[row[1], 3] = 1
end

stats = scatter_corr_plot(X)#, full=true)#, L=L, prbs=prbs)


l = @layout [grid(3, 3) a{0.07w}]
plot(layout=l, size=(550, 500), dpi=200)
for i in 2:10
    yl = ""
    if (i-2) % 3 == 0
        yl = latexstring("L")
    end
    if i < 5
        c = ceil(Int, -log10(pvalue(CorrelationTest(X[:, i], L))))
        c = palette(cgrad(:Spectral_11, rev=true), 10)[c]
        scatter!(X[:, i], L, sp=i-1, label=nothing, xlabel=names(df)[3+i], xticks=[], yticks=[], color=:black, background_color_subplot=c, ylabel=yl)
    else
        c = ceil(Int, -log10(pvalue(CorrelationTest(log10.(X[:, i]), L))))
        c = palette(cgrad(:Spectral_11, rev=true), 10)[c]
        if i < 8
            scatter!(log10.(X[:, i]), L, sp=i-1, label=nothing, xlabel="log("*names(df)[3+i]*")", xticks=[], yticks=[], color=:black, background_color_subplot=c, ylabel=yl)
        else
            scatter!(log10.(X[:, i]), L, sp=i-1, label=nothing, xlabel="log("*names(df)[4+i]*")", xticks=[], yticks=[], color=:black, background_color_subplot=c, ylabel=yl)
        end
    end
end
display(scatter!([], [], sp=10, xticks=nothing, yticks=nothing, marker_z=[0], label=nothing, clim=(0, 10), c=cgrad(:Spectral_11, rev=true), ticks=false, colorbar_title="-log(Pearson correlation p-value)", colorbar_titlefontsize=18, right_margin=2Plots.mm))


scatter(X[:, 1], L, size=(350, 350), c=:black, label="Data", ylabel="Average human solution length ("*latexstring("L")*")", xlabel="Optimal solution length", dpi=200, grid=false)
bhat = [X[:, 1] ones(70)]\L
Plots.abline!(bhat..., label="Linear fit", linewidth=3)
annotate!(7, 65, latexstring("R^2=")*"0.651")
annotate!(8, 58, latexstring("\\rho=")*"0.807(3e-17)")