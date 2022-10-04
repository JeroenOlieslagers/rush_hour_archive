include("data_analysis.jl")
include("solvers.jl")
using FileIO
using Distributions
using StatsPlots


data = load("analysis/processed_data/filtered_data.jld2")["data"]

"""
    get_Ls(data)

Return dictionary where keys are problem ids and values are arrays of lengths of people's solution.
"""
function get_Ls(data::Dict{String, DataFrame})::DefaultDict{String, Array{Int64}, Vector{Any}}
    dict = DefaultDict{String, Array{Int}}([])
    # Loop over all subjects
    for subj in keys(data)
        # Get L for each problem subject attempted
        _, _, attempts = analyse_subject(data[subj]);
        for prb in keys(attempts)
            # Push L to array for each problem
            push!(dict[prb], attempts[prb])
        end
    end
    return dict
end


"""
    get_optimal_lens(prbs)

Return dictionary where keys are problem ids and values are optimal lengths of solving that puzzle
Also return dictionary where values are the optimal moves themselves
"""
function get_optimal_lens(prbs::Vector{String})::Tuple{Dict{String, Int}, Dict{String, Array{Array{Int, 1}}}}
    dict = Dict{String, Int}()
    opt_moves = Dict{String, Vector{Any}}()
    #exps = 0
    for prb in prbs
        board = load_data(prb)
        _, past_moves, exp = a_star(board, h=red_distance)
        dict[prb] = length(past_moves)
        opt_moves[prb] = past_moves
        #exps += exp
    end
    #println(exps)
    return dict, opt_moves
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
"""
function get_state_spaces(prbs::Vector{String})
    state_space_features = Dict{String, Vector{Float64}}()
    for prb in prbs
        println(prb)
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
        depth = length(tree_shape) + 1
        state_space_features[prb] = [optimal_length, average_children, in_degree, out_degree, n_solutions, n_optimal_paths, size_state_space, av_width, min_width, max_width, depth]
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
function get_random_agent_stats(prbs::Vector{String}; N=100)
    dict = Dict{String, Vector{Float64}}()
    for prb in prbs
        println(prb)
        exps = zeros(Int, N)
        for i in 1:N
            board = load_data(prb)
            exps[i] = random_agent(board)
        end
        dict[prb] = [mean(exps), std(exps)]
    end
    return dict
end

    
Ls = get_Ls(data)
prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])]

state_space_features = get_state_spaces(prbs)

random_agent_stats = get_random_agent_stats(prbs)

x = [[] for _ in 1:13]
y = []
for prb in prbs
    for i in 1:11
        push!(x[i], [state_space_features[prb][i] for _ in 1:length(Ls[prb])]...)
    end
    for i in 12:13
        push!(x[i], [random_agent_stats[prb][i-11] for _ in 1:length(Ls[prb])]...)
    end
    push!(y, Ls[prb]...)
end

scatter(x[12], y, legend=nothing)

X = zeros(70, 13)
for (i, prb) in enumerate(prbs)
    for j in 1:13
        if j <= 11
            X[i, j] = state_space_features[prb][j]
        elseif j > 11
            X[i, j] = random_agent_stats[prb][j-11]
        end
    end
end

XX = zeros(size(X))
for i in 1:13
    XX[:, i] = X[:, i] .- mean(X[:, i])
    XX[:, i] = XX[:, i] / norm(XX[:, i])
end
