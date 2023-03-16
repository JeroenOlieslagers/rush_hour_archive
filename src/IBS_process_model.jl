using Polylogarithms

function rollout_board(board, γ; max_iter=10000)
    first_move_idx = nothing
    prev_move = [0, 0]
    for n in 1:max_iter
        if rand() > γ
            arr = get_board_arr(board)
            # Check if complete
            if check_solved(arr)
                return true, first_move_idx
            end
            # Expand current node by getting all available moves
            available_moves = get_all_available_moves(board, arr)
            filter!(e -> e != [prev_move[1], -prev_move[2]], available_moves)
            # Randomly choose a move
            selected_move_idx = rand(1:length(available_moves))
            prev_move = available_moves[selected_move_idx]
            # Make move
            make_move!(board, prev_move)
            if n == 1
                first_move_idx = selected_move_idx
            end
        else
            return false, first_move_idx
        end
    end
    return false, first_move_idx
end

function ibs_board(board, γ, k, action_index; max_iter=1000)
    for n in 1:max_iter
        move_index = nothing
        for i in 1:k
            boardd = deepcopy(board)
            solved, first_move_idx = rollout_board(boardd, γ)
            if solved 
                move_index = first_move_idx
                break
            end
        end
        if move_index === nothing
            arr = get_board_arr(board)
            available_moves = get_all_available_moves(board, arr)
            move_index = rand(1:length(available_moves))
        end
        if move_index == action_index
            return n
        end
    end
    return max_iter
end

function rollout(start, graph, solutions, γ; max_iter=10000)
    first_state = nothing
    prev_node = nothing
    current_node = start
    for n in 1:max_iter
        if rand() > γ
            # Get all neighbouring nodes
            neighbours = graph[current_node]
            # Dont undo if other options available
            if length(neighbours) > 1
                neighbours = filter(e -> e != prev_node, neighbours)
            end
            if length(neighbours) < 1
                println("wtf")
            end
            prev_node = current_node
            # Randomly choose a neighbour
            current_node = sample(neighbours)
            if n == 1
                first_state = current_node
            end
            if current_node in solutions
                return true, first_state, n
            end
        else
            return false, first_state, n
        end
    end
    return false, first_state, n
end

function ibs(start, graph, solutions, subject_state, γ, k; max_iter=1000)
    for n in 1:max_iter
        model_state = nothing
        for _ in 1:k
            solved, first_state = rollout(start, graph, solutions, γ)
            if solved 
                model_state = first_state
                break
            end
        end
        if model_state === nothing
            model_state = sample(graph[start])
        end
        if model_state == subject_state
            return n
        end
    end
    return max_iter
end

function ibs_estimator(K)
    if K == 1
        return 0
    else
        return harmonic(K-1)
    end
end

function run_subject(x, states_prb, graphs_prb, solutions_prb; N=1)
    logγ, k_fl = x
    γ = exp(logγ)
    k = Int(round(k_fl))
    # if γ > 1.0 || γ < 0.0001 || k > 10 || k < 1
    #     return Inf
    # end
    nll = 0
    for prb in keys(states_prb)
        graph = graphs_prb[prb]
        solutions = solutions_prb[prb]
        for r in eachindex(states_prb[prb])
            states = states_prb[prb][r]
            for i in 1:length(states)-1
                this_state = states[i]
                next_state = states[i+1]
                specific_nll = 0
                for _ in 1:N
                    specific_nll += ibs_estimator(ibs(this_state, graph, solutions, next_state, γ, k))
                end
                nll += specific_nll/N
                if this_state in solutions
                    break
                end
            end
        end
    end
    return nll
end

function get_graphs(prbs)
    graphs_prb = load("data/processed_data/graphs_prb.jld2")["data"]
    solutions_prb = load("data/processed_data/solutions_prb.jld2")["data"]
    # graphs_prb = Dict{String, Dict{BigInt, Vector{BigInt}}}()
    # solutions_prb = Dict{String, Vector{BigInt}}()
    # for prb in ProgressBar(prbs)
    #     board = load_data(prb)
    #     _, _, _, _, graphs_prb[prb], _, solutions_prb[prb] = bfs_path_counters(board; traverse_full=true, all_parents=true)
    # end
    return graphs_prb, solutions_prb
end

graphs_prb, solutions_prb = get_graphs(prbs);

pybads = pyimport("pybads")
BADS = pybads.BADS

easy_problem = Dict(prbs[1] => [visited_states[subjs[2]][prbs[1]][1]]);

s = easy_problem[prbs[1]][1][1];
graph = graphs_prb[prbs[1]];
solutions = solutions_prb[prbs[1]];

arr = [];
for st in ProgressBar(easy_problem[prbs[1]][1][1:end-1])
    neighs = DefaultDict{BigInt, Array{Int64}}([]);
    for i in 1:100000
        _, neighbour, n = rollout(st, graph, solutions, 0.00)
        push!(neighs[neighbour], n)
    end
    push!(arr, neighs)
end

plot(layout=(5, 1), size=(400, 800), left_margin = 5Plots.mm, bottom_margin=10Plots.mm, xticks=[], ylabel="n", grid=false, ylim=(0, 15))
for i in 1:5
    # dotplot!(1:length(arr[i]), [arr[i][kk] for kk in keys(arr[i])], sp=i, label=nothing, title="Distance from goal:"*string(i), alpha=0.05, markersize=3)
    colors = ifelse.(collect(keys(arr[i])) .== states[i+1], :red, :black)
    nn = 0
    for (n, kk) in enumerate(keys(arr[i]))
        if length(arr[i][kk]) > 0
            nn += 1
            dotplot!([nn], [arr[i][kk]], sp=i, label=nothing, title="Distance from goal:"*string(i), alpha=0.05, markersize=3, color=colors[n])
        end
    end
    if i == 5
        xlabel!(sp=i, "Neighbour")
    end
    # means = [mean(arr[i][kk]) for kk in keys(arr[i])]
    # errs = [2 .* sem(arr[i][kk]) for kk in keys(arr[i])]
    # bar!(1:length(arr[i]), means, yerr=errs, sp=i, label=nothing, title="Distance from goal:"*string(i), ylim=(minimum(means) - 10, maximum(means) + 10), fillcolor = colors)
end
plot!()

bads_target = (x) -> run_subject(x, easy_problem, graphs_prb, solutions_prb; N=100);#visited_states[subj]
lb = [-100, 0.0];
ub = [0.0, 100];
plb = [-10, 0.0];
pub = [0.0, 20.0];
x0 = [-4.2, 5.0];
options = Dict("tolfun"=> 1e-3, "max_fun_evals"=>400);
bads = BADS(bads_target, x0, lb, ub, plb, pub, options=options)
res = bads.optimize();

nlls = zeros(length(visited_states));
gammas = ones(length(visited_states)) .* -4.61;
ks = ones(Int, length(visited_states));
params = zeros(42, 2)

Threads.@threads for n in ProgressBar(1:42)
    # x0 = [-2.0, 1.0];
    # res = optimize((x) -> run_subject(x, visited_states[subjs[n]], graphs_prb, solutions_prb), [-10.0, 0.0], [0.0, 10.0], x0, Fminbox(); autodiff=:forward)
    # params[n, :] = Optim.minimizer(res)
    # nlls[n] = Optim.minimum(res)
    nlls[n] = run_subject((gammas[n], ks[n]), visited_states[subjs[n]], graphs_prb, solutions_prb)
end

