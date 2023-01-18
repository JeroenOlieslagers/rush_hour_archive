include("engine.jl")
include("solvers.jl")
include("data_analysis.jl")
using StatsBase
using LaTeXStrings


"""
    softmin(β, a, A)

``\\frac{e^{-βa}}{\\sum_{a_i∈A}e^{-βa_i}}``
"""
function softmin(β::Float64, a::Float64, A::Vector{Float64})::Float64
    return exp(-β*a)/sum(exp.(-β*A))
end

function softmin(β::Float64, A::Vector{Float64})::Vector{Float64}
    return exp.(-β*A) ./ sum(exp.(-β*A))
end

function lapsed_softmin(λ::Float64, β::Float64, a::Float64, A::Vector{Float64})::Float64
    return λ/length(A) + (1 - λ)*softmin(β, a, A)
end

function lapsed_softmin(λ::Float64, β::Float64, A::Vector{Float64})::Vector{Float64}
    return λ/length(A) .+ (1 - λ)*softmin(β, A)
end

function get_state_data(data, full_heur_dict; heur=4)
    subjs = collect(keys(data));

    qqs = Dict{String, DefaultDict{String, Array{Array{Float64, 1}, 1}}}()
    QQs = Dict{String, DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}}()
    visited_states = Dict{String, DefaultDict{String, Array{Array{BigInt, 1}, 1}}}()
    neighbour_states = Dict{String, DefaultDict{String, Array{Array{Array{BigInt, 1}, 1}, 1}}}()

    for subj in ProgressBar(subjs)
        tot_arrs, tot_move_tuples, tot_states_visited, attempts = analyse_subject(data[subj]);

        subj_dict_qs = DefaultDict{String, Array{Array{Float64, 1}, 1}}([])
        subj_dict_Qs = DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}([])
        subj_dict_visited = DefaultDict{String, Array{Array{BigInt, 1}, 1}}([])
        subj_dict_neigh = DefaultDict{String, Array{Array{Array{BigInt, 1}, 1}, 1}}([])

        for prb in keys(tot_move_tuples)
            restart_count = 0
            board = load_data(prb)
            arr = get_board_arr(board)
            for (j, move) in enumerate(tot_move_tuples[prb])
                # Reset board
                if move == (-1, 0)
                    restart_count += 1
                    # Initialise new reset
                    push!(subj_dict_qs[prb], [])
                    push!(subj_dict_Qs[prb], [])
                    push!(subj_dict_visited[prb], [])
                    push!(subj_dict_neigh[prb], [])
                    board = load_data(prb)
                    arr = get_board_arr(board)
                    s_init = board_to_int(arr, BigInt)
                    push!(subj_dict_visited[prb][restart_count], s_init)
                    continue
                end
                # Get all neighbours
                A = get_all_available_moves(board, arr)
                neighs = zeros(BigInt, length(A))
                Q = zeros(length(A))
                # Calculate Q value for each neighbour
                for (n, a) in enumerate(A)
                    make_move!(board, a)
                    arr_a = get_board_arr(board)
                    s = board_to_int(arr_a, BigInt)
                    # Ignore if move outside state space
                    if string(s) ∉ keys(full_heur_dict)
                        undo_moves!(board, [a])
                        break
                    end
                    # Get specific heuristic value as Q value
                    Q[n] = full_heur_dict[string(s)][heur]
                    neighs[n] = s
                    undo_moves!(board, [a])
                end
                # Calculate Q value for move made
                make_move!(board, move)
                arr = get_board_arr(board)
                s = board_to_int(arr, BigInt)
                # If action leads to state in state space, add it
                if string(s) ∈ keys(full_heur_dict)
                    push!(subj_dict_qs[prb][restart_count], full_heur_dict[string(s)][heur])
                    push!(subj_dict_Qs[prb][restart_count], Q)
                    push!(subj_dict_visited[prb][restart_count], s)
                    push!(subj_dict_neigh[prb][restart_count], neighs)
                end
            end
        end
        qqs[subj] = subj_dict_qs
        QQs[subj] = subj_dict_Qs
        visited_states[subj] = subj_dict_visited
        neighbour_states[subj] = subj_dict_neigh
    end
    return qqs, QQs, visited_states, neighbour_states
end

### OLD FITTING

# N = 100

# mll = zeros(length(subjs))
# bts = zeros(length(subjs))
# lmds = zeros(length(subjs))

# betas = 10 .^(range(-0.3,stop=0.7,length=N))
# lambdas = range(0,stop=0.5,length=N)

# for (m, subj) in ProgressBar(enumerate(subjs))
#     qs = qqs[m]
#     Qs = QQs[m]
#     nlls = zeros(N, N)
#     for i in eachindex(betas)
#         for j in eachindex(lambdas)
#             for k in eachindex(Qs)
#                 nlls[i, j] += log(lapsed_softmin(lambdas[j], betas[i], qs[k], Qs[k]))
#             end
#         end
#     end
#     ma = argmax(nlls)
#     mll[m] = nlls[ma]
#     bts[m] = betas[ma[1]]
#     lmds[m] = lambdas[ma[2]]
#     #plot!(lambdas, nlls[m, :], c=:black, label=nothing)
# end

# plot()
# names = ["red_dist", "mag_nodes", "min_forest_nodes", "mean_leave_one_out",  "rand_leave_one_out", "max_leave_one_out"]
# for i in 1:6
#     if i == 5
#         continue
#     end
#     m = argmax(probs[i, :])
#     plot!([betas[m], betas[m]], [-120000, maximum(nlls[i, :])], label=nothing, c=:black)
#     plot!(betas, nlls[i, :], xscale=:log10, label=names[i], legend=:bottomleft)
# end
# plot!(xlabel=latexstring("\\beta"), ylabel="Log likelihood")


### FIT WITH BLACKBOXOPTIM

function subject_fit(x::Vector{Float64}, qs, Qs)::Float64
    λ, β = x
    r = 0
    for prb in keys(qs)
        qs_prb = qs[prb]
        Qs_prb = Qs[prb]
        for i in eachindex(qs_prb)
            qs_prb_restart = qs_prb[i]
            Qs_prb_restart = Qs_prb[i]
            for j in eachindex(qs_prb_restart)
                q = qs_prb_restart[j]
                Q = Qs_prb_restart[j]
                if isnan(lapsed_softmin(λ, β, q, Q))
                    println(lapsed_softmin(λ, β, Q))
                end
                r -= log(lapsed_softmin(λ, β, q, Q))
            end
        end
    end
    return r
end

subject_fit(params[end, :], qqs[subjs[end]], QQs[subjs[end]])

function fit_all_subjs(qqs, QQs; max_time=30)
    M = length(qqs)
    params = zeros(M, 2)
    fitness = 0
    iter = ProgressBar(enumerate(keys(qqs)))
    for (m, subj) in iter
        # Initial guess
        x0 = [0.2, 2.0]
        # Optimization
        res = bboptimize((x) -> subject_fit(x, qqs[subj], QQs[subj]), x0; SearchRange = [(0.0, 1.0), (0.0, 50.0)], NumDimensions = 2, TraceMode=:silent, MaxTime=max_time/M);
        params[m, :] = best_candidate(res)
        fitness += best_fitness(res)
        set_description(iter, "Fitting subject $(m)/$(M)")
    end
    return params, fitness
end

function sample_action(x::Vector{Float64}, A::Vector)::Int
    λ, β = x
    p = lapsed_softmin(λ, β, A)
    return wsample(p)
end

##### CALCULATE CONFIDENCE AND ACCURACY
function get_accuracy_and_confidence(qqs, QQs, params)
    confidences = zeros(length(subjs))
    accuracies = zeros(length(subjs))
    chances = zeros(length(subjs))

    for (m, subj) in enumerate(keys(QQs))
        confidence = 0
        accuracy = 0
        chance = 0
        ii = 0 
        qs = qqs[subj]
        Qs = QQs[subj]
        λ, β = params[m, :]
        for prb in keys(Qs)
            for k in eachindex(Qs[prb])
                for j in eachindex(Qs[prb][k])
                    q = qs[prb][k][j]
                    Q = Qs[prb][k][j]
                    p = lapsed_softmin(λ, β, q, Q)
                    confidence += p
                    ps = lapsed_softmin(λ, β, Q)
                    accuracy += p == maximum(ps) ? 1 : 0
                    chance += 1/length(Q)
                    ii += 1
                end
            end
        end
        confidences[m] = confidence /= ii
        accuracies[m] = accuracy /= ii
        chances[m] = chance /= ii
    end
    return confidences, accuracies, chances
end



##### CALCULATE p(ERROR) FOR SUBJECT'S DATA

# Ls, dict = get_Ls(data)
# prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])]
# optimal_a = Dict{String, Dict{BigInt, Array{BigInt, 1}}}()
# IDV = Dict{String, Dict{BigInt, Array{Int, 1}}}()

# for prb in ProgressBar(prbs)
#     board = load_data(prb);
#     tree, seen, stat, dict, all_parents, children, solutions = bfs_path_counters(board, traverse_full=true, all_parents=true);
#     idvs = invert_tree(all_parents, solutions)
#     optimal_a[prb] = get_optimal_actions(idvs, all_parents);
#     IDV[prb] = idvs
# end

# save("data/processed_data/IDV.jld2", IDV)
function calculate_p_error(QQs, visited_states, neighbour_states, params)
    # Load optimal actions in every state
    optimal_a = load("data/processed_data/optimal_a.jld2")
    # Load independent variables in every state
    IDV = load("data/processed_data/IDV.jld2")
    # Initialise
    error_l_data = Dict{String, Array{Any, 1}}()
    error_a_data = Dict{String, Array{Any, 1}}()
    error_l_model = Dict{String, Array{Any, 1}}()
    error_a_model = Dict{String, Array{Any, 1}}()
    cnt = 0    
    # Loop over subjects
    for (m, subj) in ProgressBar(enumerate(keys(visited_states)))
        # Initialise arrays
        error_l_data[subj] = [[] for _ in 1:35]
        error_a_data[subj] = [[] for _ in 1:35]
        error_l_model[subj] = [[] for _ in 1:35]
        error_a_model[subj] = [[] for _ in 1:35]
        # Model parameters
        x_subj = params[m, :]
        # Loop over puzzles
        for prb in keys(visited_states[subj])
            # Get optimal actions
            opt = optimal_a[prb]
            # Loop over set of states in between restarts
            for r in eachindex(visited_states[subj][prb])
                states = visited_states[subj][prb][r]
                neighs = neighbour_states[subj][prb][r]
                # Loop over states (ignore last state as it has no action)
                for i in 1:length(states)-1
                    s = states[i]
                    ns = neighs[i]
                    # Don't calculate error on solved positions
                    if IDV[prb][s][1] == 0
                        continue
                    end
                    ## ERROR AND COUNTS FROM DATA
                    # Add error count to independent variable category
                    if states[i+1] ∉ opt[s]
                        push!(error_l_data[subj][IDV[prb][s][1]], 1)
                        push!(error_a_data[subj][IDV[prb][s][2]], 1)
                    else
                        push!(error_l_data[subj][IDV[prb][s][1]], 0)
                        push!(error_a_data[subj][IDV[prb][s][2]], 0)
                    end

                    ## ERROR AND COUNTS FROM MODEL
                    for j in 1:1
                        a_model = sample_action(x_subj, QQs[subj][prb][r][i])
                        s_model = ns[a_model]
                        # Add error count to independent variable category
                        if s_model ∉ opt[s]
                            push!(error_l_model[subj][IDV[prb][s][1]], 1)
                            push!(error_a_model[subj][IDV[prb][s][2]], 1)
                        else
                            push!(error_l_model[subj][IDV[prb][s][1]], 0)
                            push!(error_a_model[subj][IDV[prb][s][2]], 0)
                        end
                    end
                    cnt += 1
                end
            end
        end
    end
    return error_l_data, error_a_data, error_l_model, error_a_model, cnt_l, cnt_a
end

### FANCY ERROR BARS


function get_errorbar(error_dict)
    error_bar_1 = zeros(35);
    error_bar_2 = zeros(35);
    mu_hat_bar = zeros(35);
    mu_hat_bar_w = zeros(35);
    error_bar_w = zeros(35);

    for j in 1:35
        mu_hat = []
        sigma = []
        N_i = []
        for i in eachindex(subjs)
            subj = subjs[i]
            if length(error_dict[subj][j]) > 1
                push!(mu_hat, mean(error_dict[subj][j]))
                push!(sigma, std(error_dict[subj][j]))
                push!(N_i, length(error_dict[subj][j]))
            end
        end
        if length(mu_hat) > 1
            mu_hat_bar[j] = mean(mu_hat)
            sigma_bar = std(mu_hat)
            M = length(mu_hat)
            error_bar_1[j] = sigma_bar/sqrt(M)
            error_bar_2[j] = sqrt(sigma_bar*sigma_bar/M + sum((sigma .* sigma) ./ N_i)/(M*M))
            
            #Z = sum(1 ./ sigma)
            #mu_hat_bar_w[j] = sum(mu_hat ./ sigma) / Z
            #error_bar_w[j] = sqrt(sum(sigma_bar*sigma_bar ./ (sigma .* sigma)) + sum(1 ./ N_i))/Z        Z = sum(sqrt.(N_i))
            Z = sum(sqrt.(N_i))
            mu_hat_bar_w[j] = sum(mu_hat .* sqrt.(N_i)) / Z
            error_bar_w[j] = sqrt(sigma_bar*sum(N_i) + sum(sigma .* sigma))/Z
        elseif length(mu_hat) == 1
            mu_hat_bar[j] = mean(mu_hat)
            # Z = sum(1 ./ sigma)
            # mu_hat_bar_w[j] = sum(mu_hat ./ sigma) / Z
            Z = sum(sqrt.(N_i))
            mu_hat_bar_w[j] = sum(mu_hat .* sqrt.(N_i)) / Z
        end
    end
    return mu_hat_bar, error_bar_2
end

data = load("data/processed_data/filtered_data.jld2")["data"];
subjs = collect(keys(data));

full_heur_dict = load("data/processed_data/full_heur_dict.jld2")

qqs, QQs, visited_states, neighbour_states = get_state_data(data, full_heur_dict, heur=4);

params, fitness = fit_all_subjs(qqs, QQs, max_time=30);

confidences, accuracies, chances = get_accuracy_and_confidence(qqs, QQs, params);
dotplot([["Confidence"], ["Accuracy"]], [confidences, accuracies], color=:black, label=["Subjects" ""], ylim=(0, 1), size=(200, 300), markersize=3)
plot!([0, 2.2], [mean(chances), mean(chances)], color=:red, label="Chance")

error_l_data, error_a_data, error_l_model, error_a_model, cnt_l, cnt_a = calculate_p_error(QQs, visited_states, neighbour_states, params);


mu_hat_bar_l_data, error_bar_l_data = get_errorbar(error_l_data);
mu_hat_bar_a_data, error_bar_a_data = get_errorbar(error_a_data);
mu_hat_bar_l_model, error_bar_l_model = get_errorbar(error_l_model);
mu_hat_bar_a_model, error_bar_a_model = get_errorbar(error_a_model);

plot(layout=grid(2, 1), size=(600, 500))
plot!(1:35, mu_hat_bar_l_data, ribbon=error_bar_l_data, sp=1, label="Data", xlabel="opt_L", ylabel="p(error)")
plot!(1:35, mu_hat_bar_l_model, ribbon=error_bar_l_model, sp=1, label="Model")
plot!(1:35, mu_hat_bar_a_data, ribbon=error_bar_a_data, sp=2, label="Data", xlabel="|A|", ylabel="p(error)")
plot!(1:35, mu_hat_bar_a_model, ribbon=error_bar_a_model, sp=2, label="Model")

plot(layout=grid(2, 1), size=(600, 500))
histogram!(params[:, 1], bins=30, sp=1, xlabel=latexstring("\\lambda"), label=nothing)
histogram!(params[:, 2], bins=30, sp=2, xlabel=latexstring("\\beta"), label=nothing)

