include("engine.jl")
include("solvers.jl")
include("data_analysis.jl")
include("generate_fake_data.jl")
using StatsBase
using LaTeXStrings
using BlackBoxOptim
using Optim


"""
    softmin(β, q, Q)

``\\frac{e^{-βq}}{\\sum_{q_i∈Q}e^{-βq_i}}``
"""
function softmin(β::Float64, q::Float64, Q::Vector{Float64})::Float64
    return exp(-β*q)/sum(exp.(-β*Q))
end

"""
    softmin(β, Q)

``\\{\\frac{e^{-βq_j}}{\\sum_{q_i∈Q}e^{-βq_i}}\\}_{q_j∈Q}``
"""
function softmin(β::Float64, Q::Vector{Float64})::Vector{Float64}
    return exp.(-β*Q) ./ sum(exp.(-β*Q))
end

"""
    lapsed_softmin(λ, β, q, Q)

``\\frac{λ}{|Q|} + (1-λ)softmin(β, q, Q)``
"""
function lapsed_softmin(λ::Float64, β::Float64, q::Float64, Q::Vector{Float64})::Float64
    return λ/length(Q) + (1 - λ)*softmin(β, q, Q)
end

"""
    lapsed_softmin(λ, β, Q)

``\\frac{λ}{|Q|} + (1-λ)softmin(β, Q)``
"""
function lapsed_softmin(λ::Float64, β::Float64, Q::Vector{Float64})::Vector{Float64}
    return λ/length(Q) .+ (1 - λ)*softmin(β, Q)
end

function softmin(β, q, Q)
    return exp(-β*q)/sum(exp.(-β*Q))
end

function lapsed_softmin(λ, β, q, Q)
    return λ/length(Q) + (1 - λ)*softmin(β, q, Q)
end


"""
    get_state_data(data, full_heur_dict; heur=4)

Return four dictionaries with structure 
{subj:
  {prb: 
    [
      [values for first attempt]
      [values for second attempt]
    ]
  }
}
where values = 
- Q value of every visited state (Float64)
- Q value of each neighbour of every visited state (Float64[])
- every visited state (BigInt)
- each neighbour of every visited state (BigInt[])

Q values are determined by full_heur dictionary input, where
heur specifies which heuristic is being used.
"""
function get_state_data(data, full_heur_dict; heur=4)
    subjs = collect(keys(data));
    # Initialise output dicts
    qqs = Dict{String, DefaultDict{String, Array{Array{Float64, 1}, 1}}}()
    QQs = Dict{String, DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}}()
    visited_states = Dict{String, DefaultDict{String, Array{Array{BigInt, 1}, 1}}}()
    neighbour_states = Dict{String, DefaultDict{String, Array{Array{Array{BigInt, 1}, 1}, 1}}}()
    # Loop over subjects
    for subj in ProgressBar(subjs)
        # Get subject-specific data in dicts
        tot_arrs, tot_move_tuples, tot_states_visited, attempts = analyse_subject(data[subj]);
        # Initialise subject-specific output dicts
        subj_dict_qs = DefaultDict{String, Array{Array{Float64, 1}, 1}}([])
        subj_dict_Qs = DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}([])
        subj_dict_visited = DefaultDict{String, Array{Array{BigInt, 1}, 1}}([])
        subj_dict_neigh = DefaultDict{String, Array{Array{Array{BigInt, 1}, 1}, 1}}([])
        # Loop over puzzles
        for prb in keys(tot_move_tuples)
            # Keep track of which attempt is being indexed
            restart_count = 0
            # Initialise board
            board = load_data(prb)
            arr = get_board_arr(board)
            # Loop over all moves in puzzle
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
                    # Initialise visited states at root
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
                # Make move
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

function subject_fit(x, qs, Qs)
    λ, logb = x
    β = exp(logb)
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
                r -= log(lapsed_softmin(λ, β, q, Q))
            end
        end
    end
    return r
end

function fit_all_subjs(qqs, QQs; max_time=30)
    M = length(qqs)
    params = zeros(M, 2)
    fitness = 0
    iter = ProgressBar(enumerate(keys(qqs)))
    for (m, subj) in iter
        # Initial guess
        x0 = [0.1, 1.0]
        # Optimization
        # res = bboptimize((x) -> subject_fit(x, qqs[subj], QQs[subj]), x0; SearchRange = [(0.0, 1.0), (-10.0, 4.0)], NumDimensions = 2, TraceMode=:silent, MaxTime=max_time/M);
        # params[m, :] = best_candidate(res)
        # fitness += best_fitness(res)
        res = optimize((x) -> subject_fit(x, qqs[subj], QQs[subj]), [0.0, -10.0], [1.0, 3.0], x0, Fminbox(); autodiff=:forward)
        # df = TwiceDifferentiable((x) -> subject_fit(x, qqs[subj], QQs[subj]), x0; autodiff=:forward)
        # dfc = TwiceDifferentiableConstraints([0.0, -10.0], [1.0, 4.0])
        # res = optimize(df, dfc, x0, IPNewton())
        params[m, :] = Optim.minimizer(res)
        fitness += Optim.minimum(res)
        set_description(iter, "Fitting subject $(m)/$(M)")
    end
    return params, fitness
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
        λ, logb = params[m, :]
        β = exp(logb)
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
function calculate_p_error(QQs, visited_states, neighbour_states, params; heur=4)
    # Load optimal actions in every state
    optimal_a = load("data/processed_data/optimal_a.jld2")
    # Load independent variables in every state
    IDV = load("data/processed_data/IDV_OLD.jld2")
    # Initialise
    error_l_data = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_data = Dict{String, Array{Array{Int, 1}, 1}}()
    error_l_fake = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_fake = Dict{String, Array{Array{Int, 1}, 1}}()
    error_l_model = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_model = Dict{String, Array{Array{Int, 1}, 1}}()
    chance_l = Dict{String, Array{Array{Float64, 1}, 1}}()
    chance_a = Dict{String, Array{Array{Float64, 1}, 1}}()
    cnt = 0
    # Fake data
    fake_qqs = Dict{String, DefaultDict{String, Array{Array{Float64, 1}, 1}}}()
    # Loop over subjects
    for (m, subj) in ProgressBar(enumerate(keys(visited_states)))
        # Initialise arrays
        error_l_data[subj] = [[] for _ in 1:35]
        error_a_data[subj] = [[] for _ in 1:35]
        error_l_fake[subj] = [[] for _ in 1:35]
        error_a_fake[subj] = [[] for _ in 1:35]
        error_l_model[subj] = [[] for _ in 1:35]
        error_a_model[subj] = [[] for _ in 1:35]
        chance_l[subj] = [[] for _ in 1:35]
        chance_a[subj] = [[] for _ in 1:35]
        # Fake data
        fake_subj_dict_qs = DefaultDict{String, Array{Array{Float64, 1}, 1}}([])
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
                # Fake data
                push!(fake_subj_dict_qs[prb], [])
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
                        # # RANDOM MODEL
                        # s_rand = sample(ns)
                        # # Add error count to independent variable category
                        # if s_rand ∉ opt[s]
                        #     push!(chance_l[subj][IDV[prb][s][1]], 1)
                        #     push!(chance_a[subj][IDV[prb][s][2]], 1)
                        # else
                        #     push!(chance_l[subj][IDV[prb][s][1]], 0)
                        #     push!(chance_a[subj][IDV[prb][s][2]], 0)
                        # end
                    end

                    ## FAKE DATA
                    # Fake data
                    a_fake = sample_action(x_subj, QQs[subj][prb][r][i])
                    s_fake = ns[a_fake]
                    push!(fake_subj_dict_qs[prb][r], full_heur_dict[string(s_fake)][heur])
                    if s_fake ∉ opt[s]
                        push!(error_l_fake[subj][IDV[prb][s][1]], 1)
                        push!(error_a_fake[subj][IDV[prb][s][2]], 1)
                    else
                        push!(error_l_fake[subj][IDV[prb][s][1]], 0)
                        push!(error_a_fake[subj][IDV[prb][s][2]], 0)
                    end

                    ## Chance calculations
                    push!(chance_l[subj][IDV[prb][s][1]], 1 - (length(opt[s])/IDV[prb][s][2]))
                    push!(chance_a[subj][IDV[prb][s][2]], 1 - (length(opt[s])/IDV[prb][s][2]))
                    cnt += 1
                end
            end
        end
        fake_qqs[subj] = fake_subj_dict_qs
    end
    return error_l_data, error_a_data, error_l_model, error_a_model, chance_l, chance_a, cnt, error_l_fake, error_a_fake, fake_qqs
end

### FANCY ERROR BARS


function get_errorbar(error_dict)
    bins = length(error_dict[first(keys(error_dict))])
    error_bar_1 = zeros(bins);
    error_bar_2 = zeros(bins);
    mu_hat_bar = zeros(bins);
    mu_hat_bar_w = zeros(bins);
    error_bar_w = zeros(bins);

    for j in 1:bins
        mu_hat = []
        sigma = []
        N_i = []
        for subj in keys(error_dict)
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

#full_heur_dict = load("data/processed_data/full_heur_dict.jld2");
full_heur_dict_opt = load("data/processed_data/full_heur_dict_opt.jld2");


qqs, QQs, visited_states, neighbour_states = get_state_data(data, full_heur_dict_opt, heur=4);

params, fitness = fit_all_subjs(qqs, QQs, max_time=600);

X = zeros(420, 2)
Y = zeros(420, 2)
fitnesses = zeros(10)

for i in 0:9
    error_l_data, error_a_data, error_l_model, error_a_model, chance_l, chance_a, cnt, error_l_fake, error_a_fake, fake_qqs = calculate_p_error(QQs, visited_states, neighbour_states, params);
    fparams, ffitness = fit_all_subjs(fake_qqs, QQs, max_time=600);
    X[1 + i*42:(i+1)*42, :] = params
    Y[1 + i*42:(i+1)*42, :] = fparams
    fitnesses[i+1] = ffitness
end


confidences, accuracies, chances = get_accuracy_and_confidence(qqs, QQs, params);
dotplot([["Confidence"], ["Accuracy"]], [confidences, accuracies], color=:black, label=["Subjects" ""], ylim=(0, 1), size=(200, 300), markersize=3)
plot!([0, 2.2], [mean(chances), mean(chances)], color=:red, label="Chance")


fake_params = zeros(42, 2)
fake_params[:, 2] .= 1.1
fake_params[:, 1] .= 1.0

fit1 = 0
fit2 = 0
for i in 1:42
    fit1 += subject_fit(fake_params[i, :], qqs[subjs[i]], QQs[subjs[i]])
    fit2 += subject_fit(params[i, :], qqs[subjs[i]], QQs[subjs[i]])
end

error_l_data, error_a_data, error_l_model, error_a_model, chance_l, chance_a, cnt, error_l_fake, error_a_fake, fake_qqs = calculate_p_error(QQs, visited_states, neighbour_states, fparams);

error_l_data_qb = quantile_binning(error_l_data);
error_a_data_qb = quantile_binning(error_a_data);
error_l_model_qb = quantile_binning(error_l_model);
error_a_model_qb = quantile_binning(error_a_model);
chance_l_qb = quantile_binning(chance_l);
chance_a_qb = quantile_binning(chance_a);

error_l_fake_qb = quantile_binning(error_l_fake);
error_a_fake_qb = quantile_binning(error_a_fake);



mu_hat_bar_l_data, error_bar_l_data = get_errorbar(error_l_data_qb);
mu_hat_bar_a_data, error_bar_a_data = get_errorbar(error_a_data_qb);
mu_hat_bar_l_model, error_bar_l_model = get_errorbar(error_l_model_qb);
mu_hat_bar_a_model, error_bar_a_model = get_errorbar(error_a_model_qb);
mu_hat_bar_chance_l, error_bar_chance_l = get_errorbar(chance_l_qb);
mu_hat_bar_chance_a, error_bar_chance_a = get_errorbar(chance_a_qb);

mu_hat_bar_l_fake, error_bar_l_fake = get_errorbar(error_l_fake_qb);
mu_hat_bar_a_fake, error_bar_a_fake = get_errorbar(error_a_fake_qb);

plot(layout=grid(2, 1), size=(600, 500), legend=:bottom)
for i in 0:10
    #params[:, 1] .= i/10
    #params[:, 1] .= 0.0
    #params[:, 2] .= (i*3 - 10)/10
    #params[:, 2] .= 2.0
    error_l_data, error_a_data, error_l_model, error_a_model, cnt = calculate_p_error(QQs, visited_states, neighbour_states, params);

    error_l_model_qb = quantile_binning(error_l_model);
    error_a_model_qb = quantile_binning(error_a_model);

    mu_hat_bar_l_model, error_bar_l_model = get_errorbar(error_l_model_qb);
    mu_hat_bar_a_model, error_bar_a_model = get_errorbar(error_a_model_qb);

    if i == 0 || i == 10
        plot!(1:7, mu_hat_bar_l_model, ribbon=error_bar_l_model, sp=1, label=latexstring("\\lambda=")*string(params[1, 1]), palette=:RdYlGn, lw=0.0)
        plot!(1:7, mu_hat_bar_a_model, ribbon=error_bar_a_model, sp=2, label=latexstring("\\lambda=")*string(params[1, 1]), palette=:RdYlGn, lw=0.0)
    else
        plot!(1:7, mu_hat_bar_l_model, ribbon=error_bar_l_model, sp=1, label=nothing, palette=:RdYlGn, lw=0.0)
        plot!(1:7, mu_hat_bar_a_model, ribbon=error_bar_a_model, sp=2, label=nothing, palette=:RdYlGn, lw=0.0)
    end
end
scatter!(1:7, mu_hat_bar_a_fake, yerr=2*error_bar_a_fake, sp=2, markersize=3, c=:gray, label="Fake data", xlabel=latexstring("|A|"), ylabel="p(error)")
scatter!(1:7, mu_hat_bar_l_fake, yerr=2*error_bar_l_fake, sp=1, markersize=3, c=:gray, label="Fake data", xlabel="opt_L", ylabel="p(error)")

plot!(1:7, mu_hat_bar_l_model, ribbon=2*error_bar_l_model, sp=1, label="Model fit to fake data", lw=0.0)
plot!(1:7, mu_hat_bar_a_model, ribbon=2*error_bar_a_model, sp=2, label="Model fit to fake data", lw=0.0)

plot!(1:7, mu_hat_bar_chance_l, ribbon=2*error_bar_chance_l, sp=1, label="Chance", lw=0.0)
plot!(1:7, mu_hat_bar_chance_a, ribbon=2*error_bar_chance_a, sp=2, label="Chance", lw=0.0)


true_x = [0.0, 1.0];
qqs, QQs, visited_states, neighbour_states, cnt = generate_fake_data(40, prbs, true_x, 1, full_heur_dict_opt; heur=4);
params, fitness = fit_all_subjs(qqs, QQs, max_time=600);
error_l_data, error_a_data, error_l_model, error_a_model, chance_l, chance_a, cnt = calculate_p_error(QQs, visited_states, neighbour_states, params);
error_l_data_qb = quantile_binning(error_l_data);
error_a_data_qb = quantile_binning(error_a_data);
error_l_model_qb = quantile_binning(error_l_model);
error_a_model_qb = quantile_binning(error_a_model);
chance_l_qb = quantile_binning(chance_l);
chance_a_qb = quantile_binning(chance_a);
mu_hat_bar_l_data, error_bar_l_data = get_errorbar(error_l_data_qb);
mu_hat_bar_a_data, error_bar_a_data = get_errorbar(error_a_data_qb);
mu_hat_bar_l_model, error_bar_l_model = get_errorbar(error_l_model_qb);
mu_hat_bar_a_model, error_bar_a_model = get_errorbar(error_a_model_qb);
mu_hat_bar_chance_l, error_bar_chance_l = get_errorbar(chance_l_qb);
mu_hat_bar_chance_a, error_bar_chance_a = get_errorbar(chance_a_qb);
plot(layout=grid(2, 1), size=(600, 500))
plot!(1:7, mu_hat_bar_l_model, ribbon=error_bar_l_model, sp=1, label="Model", lw=0.0)
plot!(1:7, mu_hat_bar_a_model, ribbon=error_bar_a_model, sp=2, label="Model", lw=0.0)
plot!(1:7, mu_hat_bar_chance_l, ribbon=2*error_bar_chance_l, sp=1, label="Chance", lw=0.0)
plot!(1:7, mu_hat_bar_chance_a, ribbon=2*error_bar_chance_a, sp=2, label="Chance", lw=0.0)
scatter!(1:7, mu_hat_bar_a_data, yerr=error_bar_a_data, sp=2, markersize=3, c=:black, label="Fake data", xlabel=latexstring("|A|"), ylabel="p(error)")
scatter!(1:7, mu_hat_bar_l_data, yerr=error_bar_l_data, sp=1, markersize=3, c=:black, label="Fake data", xlabel="opt_L", ylabel="p(error)")


Rs = [1, 10, 100]
ppppparams = []
for R in Rs
    println("R=$(R)")
    x = [0.1, 0.0];
    qqs, QQs, cnt = generate_fake_data(40, prbs, x, R, full_heur_dict_opt);
    params, fitness = fit_all_subjs(qqs, QQs, max_time=600);
    push!(ppppparams, params)
end

plot(layout=grid(6, 2), size=(600, 1000))
for i in 1:6
    histogram!(ppparams[i][:, 1], bins=30, sp=2*i - 1, xlabel=latexstring("\\lambda"), label=nothing, link=:x, yticks=nothing)
    #plot!([0.1, 0.1], [0.0, 10.0], sp=2*i - 1, c=:red, label="True parameter", linewidth=2.0)
    #plot!([mean(pppparams[i][:, 1]), mean(pppparams[i][:, 1])], [0.0, 10.0], sp=2*i - 1, c=:green, label="Estimated parameter", linewidth=2.0)
    histogram!(ppparams[i][:, 2], bins=30, sp=2*i, xlabel=latexstring("log(\\beta)"), label=nothing, link=:x, yticks=nothing)
end
plot!()


function quantile_binning(error_dict; bins=7)
    new_error_dict = Dict{String, Array{Any, 1}}()

    for subj in keys(error_dict)
        ls = []
        for i in 1:35
            push!(ls, [i for j in 1:length(error_dict[subj][i])]...)
        end
        new_error_dict[subj] = [[] for _ in 1:bins]
        for bin in 1:bins
            l = Int(ceil(quantile(ls, (bin-1)/bins)))
            u = Int(floor(quantile(ls, bin/bins))) - 1
            if bin == bins
                u += 1
            end
            for i in l:u
                push!(new_error_dict[subj][bin], error_dict[subj][i]...)
            end
        end
    end
    return new_error_dict
end

xs = [[] for _ in 1:42];
ys = [[] for _ in 1:42];
IDV = load("data/processed_data/IDV_OLD.jld2")
for (n, subj) in enumerate(keys(visited_states))
    for prb in keys(visited_states[subj])
        for r in eachindex(visited_states[subj][prb])
            for state in visited_states[subj][prb][r]
                push!(xs[n], IDV[prb][state][1])
                push!(ys[n], IDV[prb][state][2])
            end
        end
    end
end

bins = 7;
x = reduce(vcat, xs);
y = reduce(vcat, ys);
h1 = zeros(bins);
h2 = zeros(bins);
plot(layout=grid(4, 1, heights=[0.5, 0.17, 0.16, 0.16]), size=(400, 700))
histogram2d!(x, y, sp=1, bins=(maximum(x), maximum(y)), color=cgrad(:grays, rev=true), xlabel="opt_L", ylabel=latexstring("|A|"), colorbar_title="Counts", left_margin = 3Plots.mm)
for i in 1:bins
    l = Int(ceil(quantile(x, (i-1)/bins)))
    u = Int(floor(quantile(x, i/bins)))
    if i == bins
        plot!([u, u], [0, 25], sp=1, c=:red, label="opt_L bin bounds")
    end
    plot!([l, l], [0, 25], sp=1, c=:red, label=nothing)
    if i == bins
        h1[i] = length(findall(xx -> l<=xx<=u, x))
    else
        h1[i] = length(findall(xx -> l<=xx<u, x))
    end

    l = Int(ceil(quantile(y, (i-1)/bins)))
    u = Int(floor(quantile(y, i/bins)))
    if i == bins
        plot!([0, 34], [u, u], sp=1, c=:blue, label=latexstring("\$|A|\$bin bounds"))
    end
    plot!([0, 34], [l, l], sp=1, c=:blue, label=nothing)
    if i == bins
        h2[i] = length(findall(yy -> l<=yy<=u, y))
    else
        h2[i] = length(findall(yy -> l<=yy<u, y))
    end
end
plot!(h1, sp=2, ylabel="Total bin counts", ylim=(0, maximum(reduce(vcat, [h1, h2]))*1.1), label="opt_L", left_margin = 10Plots.mm, yguidefontsize=8)
plot!(h2, sp=2, label=latexstring("|A|"))

for j in 1:42
    h1 = zeros(bins);
    h2 = zeros(bins);
    for i in 1:bins
        l = Int(ceil(quantile(xs[j], (i-1)/bins)))
        u = Int(floor(quantile(xs[j], i/bins)))
        if i == bins
            h1[i] = length(findall(xx -> l<=xx<=u, xs[j]))
        else
            h1[i] = length(findall(xx -> l<=xx<u, xs[j]))
        end

        l = Int(ceil(quantile(ys[j], (i-1)/bins)))
        u = Int(floor(quantile(ys[j], i/bins)))
        if i == bins
            h2[i] = length(findall(yy -> l<=yy<=u, ys[j]))
        else
            h2[i] = length(findall(yy -> l<=yy<u, ys[j]))
        end
    end
    plot!(h1 ./ sum(h1), sp=3, c=:black, alpha=0.2, label=nothing)
    plot!(h2 ./ sum(h2), sp=4, c=:black, alpha=0.2, label=nothing)
end
plot!(sp=3, ylabel="Norm. opt_L bin cnts", yguidefontsize=6)
plot!(sp=4, xlabel="Bin index", ylabel="Norm. |A| bin cnts", yguidefontsize=6)




plot(layout=grid(2, 1, heights=[0.8, 0.2]), size=(500, 700))
scatter!(X[:, 1], Y[:, 1], sp=1, label=latexstring("\\log(\\beta)"), xlim=(0, 2.0), ylim=(0, 2.0), ylabel="Fit on fake data", xlabel="Fit on real data", legend=:bottomright)
scatter!(X[:, 2], Y[:, 2], sp=1, label=latexstring("\\beta"))
plot!([0, 2.0], [0, 2.0], sp=1, label="Diagonal", c=:red)
histogram!(fitnesses, sp=2, bins=20, label="NLL of fit on fake data")
plot!([fitness, fitness], [0, 2.0], sp=2, c=:red, label="NLL of fit on true data")


