using Polylogarithms
using PythonCall

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

function rollout(start, graph, solutions, γ; max_iter=100000)
    prev_node = start
    first_neighbours = graph[start]
    first_node = sample(first_neighbours)
    current_node = first_node
    if rand() > γ
        if first_node in solutions
            return true, first_node, 1
        end
    else
        return false, first_node, 1
    end
    for n in 2:max_iter
        if rand() > γ
            # Get all neighbouring nodes
            neighbours = graph[current_node]
            prev_node = current_node
            # Randomly choose a neighbour
            @label sample_label
            current_node = sample(neighbours)
            # Resample if undo move
            if current_node == prev_node
                @goto sample_label
            end
            if current_node in solutions
                return true, first_node, n
            end
        else
            return false, first_node, n
        end
    end
    return throw(ErrorException("Could not find solution"))
end

function select_store_car(cars, cars_moved, β)
    c = cars_moved[cars]
    num = exp.(-β*c)
    p = num / sum(num)
    return wsample(p)
end

function rollout_store_car(start, graph, graph_cars, solutions, γ, τ, β; max_iter=100000, K=9)
    prev_node = start
    cars_moved = zeros(K)
    cars = graph_cars[start]
    move_idx = select_store_car(cars, cars_moved, β)
    first_node = graph[start][move_idx]
    cars_moved[cars[move_idx]] = 1
    current_node = first_node
    if rand() > γ
        if first_node in solutions
            return true, first_node, 1
        end
    else
        return false, first_node, 1
    end
    for n in 2:max_iter
        if rand() > γ
            # Update eligibility traces
            cars_moved .*= τ
            prev_node = current_node
            cars = graph_cars[current_node]
            # Choose a neighbour according to which cars have been visited
            @label sample_label
            move_idx = select_store_car(cars, cars_moved, β)
            current_node = graph[current_node][move_idx]
            # Resample if undo move
            if current_node == prev_node
                @goto sample_label
            end
            cars_moved[cars[move_idx]] = 1
            if current_node in solutions
                return true, first_node, n
            end
        else
            return false, first_node, n
        end
    end
    return throw(ErrorException("Could not find solution"))
end

function ibs(start, graph, solutions, subject_state, γ, k; graph_cars=nothing, τ=0.9, β=2.0, max_iter=1000)
    for n in 1:max_iter
        model_state = nothing
        for _ in 1:k
            if graph_cars === nothing
                solved, first_state, iters = rollout(start, graph, solutions, γ)
            else
                solved, first_state, iters = rollout_store_car(start, graph, graph_cars, solutions, γ, τ, β)
            end
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

function run_subject(x, states_prb, graphs_prb, graph_cars_prb, solutions_prb; N=1)
    if length(x) == 2
        logγ, k_fl = x
        γ = exp(logγ)
        k = Int(round(k_fl))
    elseif length(x) == 4
        logγ, k_fl, logτ, β = x
        γ = exp(logγ)
        k = Int(round(k_fl))
        τ = exp(logτ)
    end
    # if γ > 1.0 || γ < 0.0001 || k > 10 || k < 1
    #     return Inf
    # end
    nll = 0
    for prb in keys(states_prb)
        graph = graphs_prb[prb]
        graph_cars = graph_cars_prb[prb]
        solutions = solutions_prb[prb]
        for r in eachindex(states_prb[prb])
            states = states_prb[prb][r]
            for i in 1:length(states)-1
                this_state = states[i]
                next_state = states[i+1]
                specific_nll = 0
                for _ in 1:N
                    if length(x) == 2
                        specific_nll += ibs_estimator(ibs(this_state, graph, solutions, next_state, γ, k))
                    elseif length(x) == 4
                        specific_nll += ibs_estimator(ibs(this_state, graph, solutions, next_state, γ, k; graph_cars=graph_cars, τ=τ, β=β))
                    end
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
    # graphs_prb = load("data/processed_data/graphs_prb.jld2")["data"]
    # solutions_prb = load("data/processed_data/solutions_prb.jld2")["data"]
    graphs_prb = Dict{String, Dict{BigInt, Vector{BigInt}}}()
    solutions_prb = Dict{String, Vector{BigInt}}()
    graph_cars_prb = Dict{String, Dict{BigInt, Vector{Int}}}()
    for prb in ProgressBar(prbs)
        board = load_data(prb)
        _, _, _, _, graphs_prb[prb], _, solutions_prb[prb], graph_cars_prb[prb] = bfs_path_counters(board; traverse_full=true, all_parents=true)
    end
    return graphs_prb, solutions_prb, graph_cars_prb
end

function simulate_rollouts(x, visited_states, graphs_prb, graph_cars_prb, solutions_prb)
    # Get parameters in normal form
    if length(x) == 2
        logγ, k_fl = x
        γ = exp(logγ)
        k = Int(round(k_fl))
    elseif length(x) == 4
        logγ, k_fl, logτ, β = x
        γ = exp(logγ)
        k = Int(round(k_fl))
        τ = exp(logτ)
    end
    # Load optimal actions in every state
    optimal_a = load("data/processed_data/optimal_a.jld2")
    # Load independent variables in every state
    IDV = load("data/processed_data/IDV_OLD.jld2")
    # Initialise
    error_l_data = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_data = Dict{String, Array{Array{Int, 1}, 1}}()
    error_l_model = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_model = Dict{String, Array{Array{Int, 1}, 1}}()
    chance_l = Dict{String, Array{Array{Float64, 1}, 1}}()
    chance_a = Dict{String, Array{Array{Float64, 1}, 1}}()
    subjs = collect(keys(visited_states))
    Threads.@threads for subj_idx in ProgressBar(1:42)
        subj = subjs[subj_idx]
        # Initialise arrays
        error_l_data[subj] = [[] for _ in 1:35]
        error_a_data[subj] = [[] for _ in 1:35]
        error_l_model[subj] = [[] for _ in 1:35]
        error_a_model[subj] = [[] for _ in 1:35]
        chance_l[subj] = [[] for _ in 1:35]
        chance_a[subj] = [[] for _ in 1:35]
        for prb in keys(visited_states[subj])
            # Get optimal actions
            opt = optimal_a[prb]
            # Get parents and solutions
            graph = graphs_prb[prb]
            graph_cars = graph_cars_prb[prb]
            solutions = solutions_prb[prb]
            visited_prb = visited_states[subj][prb]
            for r in eachindex(visited_prb)
                for i in 1:length(visited_prb[r])-1
                    s = visited_prb[r][i]
                    ns = graph[s]
                    # Don't calculate error on solved positions
                    if IDV[prb][s][1] == 0
                        continue
                    end
                    ## ERROR AND COUNTS FROM DATA
                    # Add error count to independent variable category
                    if visited_prb[r][i+1] ∉ opt[s]
                        push!(error_l_data[subj][IDV[prb][s][1]], 1)
                        push!(error_a_data[subj][IDV[prb][s][2]], 1)
                    else
                        push!(error_l_data[subj][IDV[prb][s][1]], 0)
                        push!(error_a_data[subj][IDV[prb][s][2]], 0)
                    end
                    ## ERROR AND COUNTS FROM MODEL
                    # Perform rollouts
                    s_model = nothing
                    for j in 1:k
                        if length(x) == 2
                            finished, first_state, n_expansions = rollout(s, graph, solutions, γ)
                        elseif length(x) == 4
                            finished, first_state, n_expansions = rollout_store_car(s, graph, graph_cars, solutions, γ, τ, β)
                        end
                        # If rollout reaches solution, choose first move
                        if finished
                            s_model = first_state
                            break
                        end
                    end
                    # Else choose random move
                    if s_model === nothing
                        s_model = sample(ns)
                    end
                    # Add error count to independent variable category
                    if s_model ∉ opt[s]
                        push!(error_l_model[subj][IDV[prb][s][1]], 1)
                        push!(error_a_model[subj][IDV[prb][s][2]], 1)
                    else
                        push!(error_l_model[subj][IDV[prb][s][1]], 0)
                        push!(error_a_model[subj][IDV[prb][s][2]], 0)
                    end
                    ## Chance calculations
                    push!(chance_l[subj][IDV[prb][s][1]], 1 - (length(opt[s])/IDV[prb][s][2]))
                    push!(chance_a[subj][IDV[prb][s][2]], 1 - (length(opt[s])/IDV[prb][s][2]))
                end
            end
        end
    end
    return error_l_data, error_a_data, error_l_model, error_a_model, chance_l, chance_a
end

function summary_statistics_plot_random_rollouts(x, visited_states, graphs_prb, graph_cars_prb, solutions_prb)
    # Simulate model
    error_l_data, error_a_data, error_l_model, error_a_model, chance_l, chance_a = simulate_rollouts(x, visited_states, graphs_prb, graph_cars_prb, solutions_prb);
    # Quantile bin data
    error_l_data_qb = quantile_binning(error_l_data);
    error_a_data_qb = quantile_binning(error_a_data);
    error_l_model_qb = quantile_binning(error_l_model);
    error_a_model_qb = quantile_binning(error_a_model);
    chance_l_qb = quantile_binning(chance_l);
    chance_a_qb = quantile_binning(chance_a);
    # Calculate errorbars
    mu_hat_bar_l_data, error_bar_l_data = get_errorbar(error_l_data_qb);
    mu_hat_bar_a_data, error_bar_a_data = get_errorbar(error_a_data_qb);
    mu_hat_bar_l_model, error_bar_l_model = get_errorbar(error_l_model_qb);
    mu_hat_bar_a_model, error_bar_a_model = get_errorbar(error_a_model_qb);
    mu_hat_bar_chance_l, error_bar_chance_l = get_errorbar(chance_l_qb);
    mu_hat_bar_chance_a, error_bar_chance_a = get_errorbar(chance_a_qb);
    # PLOTTING
    # Get parameters in normal form
    if length(x) == 2
        logγ, k_fl = x
        γ = exp(logγ)
        k = Int(round(k_fl))
    elseif length(x) == 4
        logγ, k_fl, logτ, β = x
        γ = exp(logγ)
        k = Int(round(k_fl))
        τ = exp(logτ)
    end
    #plot(layout=grid(2, 1), size=(600, 500), legend=:bottom, ylim=(0, 1), grid=false, foreground_color_legend = nothing, title=latexstring("\\gamma = ")*string(round(γ, sigdigits=3))*latexstring(",   k = ")*string(k)*latexstring(",   \\tau = ")*string(round(τ, sigdigits=3))*latexstring(",   \\beta = ")*string(β))
    #scatter!(1:7, mu_hat_bar_a_data, yerr=2*error_bar_a_data, sp=2, markersize=3, c=:black, label="Data", xlabel=latexstring("|A|"), ylabel="p(error)")
    #scatter!(1:7, mu_hat_bar_l_data, yerr=2*error_bar_l_data, sp=1, markersize=3, c=:black, label="Data", xlabel="opt_L", ylabel="p(error)")
    plot!(1:7, mu_hat_bar_l_model, ribbon=2*error_bar_l_model, sp=1, c=palette(:default)[6], lw=0.0, label=nothing)
    plot!([], [], sp=1, c=palette(:default)[6], label="Car rollout model beta=2")
    plot!(1:7, mu_hat_bar_a_model, ribbon=2*error_bar_a_model, sp=2, c=palette(:default)[6], lw=0.0, label=nothing)
    plot!([], [], sp=2, c=palette(:default)[6], label="Car rollout model beta=2")
    #plot!(1:7, mu_hat_bar_chance_l, ribbon=2*error_bar_chance_l, sp=1, c=palette(:default)[3], lw=0.0, label=nothing)
    #plot!([], [], sp=1, c=palette(:default)[3], label="Chance")
    #plot!(1:7, mu_hat_bar_chance_a, ribbon=2*error_bar_chance_a, sp=2, c=palette(:default)[3], lw=0.0, label=nothing)
    #display(plot!([], [], sp=2, c=palette(:default)[3], label="Chance"))
end

graphs_prb, solutions_prb, graph_cars_prb = get_graphs(prbs);

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

summary_statistics_plot_random_rollouts(x0, visited_states, graphs_prb, graph_cars_prb, solutions_prb)

bads_target = (x) -> run_subject(x, visited_states[subjs[1]], graphs_prb, graph_cars_prb, solutions_prb; N=100);#visited_states[subj]
lb = [-100, 0.0];
ub = [0.0, 100];
plb = [-10, 0.0];
pub = [0.0, 20.0];
x0 = [-2.0, 1000.0, -0.01, 2.0];
options = Dict("tolfun"=> 1e-3, "max_fun_evals"=>10);
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
    #nlls[n] = run_subject((gammas[n], ks[n]), visited_states[subjs[n]], graphs_prb, solutions_prb)
    nlls[n] = run_subject(x0, visited_states[subjs[n]], graphs_prb, graph_cars_prb, solutions_prb)
end

