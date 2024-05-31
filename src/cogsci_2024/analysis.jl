using Distributions
using StatsBase
using Random
include("model.jl");

function X_histogram(tree, all_moves, d_goal)
    return 1
end

function X_d_goal(tree, all_moves, d_goal)
    return d_goal
end

function X_n_A(tree, all_moves, d_goal)
    return length(all_moves)
end

function y_d_goal(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    sampled_move_idx = wsample(ps)
    sampled_s = neighs[sampled_move_idx]
    d_goal_next = d_goals[sampled_s]
    return d_goal_next+1
end

function y_p_in_tree(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    sampled_move = wsample(all_moves, ps)
    AND_root, AND, OR, parents_moves = tree
    return sampled_move ∈ keys(parents_moves)
end

function y_p_undo(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    sampled_move = wsample(all_moves, ps)
    if prev_move == (0, 0)
        return 0
    else
        undo_move = (prev_move[1], -prev_move[2])
        return sampled_move == undo_move
    end
end

function y_p_same_car(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    sampled_move = wsample(all_moves, ps)
    if prev_move == (0, 0)
        return 0
    else
        prev_car = prev_move[1]
        curr_car = sampled_move[1]
        return curr_car == prev_car
    end
end

function y_d_tree(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    AND_root, AND, OR, parents_moves = tree
    sampled_move = wsample(all_moves, ps)
    if sampled_move in keys(parents_moves)
        ds = [or_node[3] for or_node in parents_moves[sampled_move]]
        return minimum(ds)
    else
        return 1000
    end
end

function y_d_tree_ranked(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    AND_root, AND, OR, parents_moves = tree
    sampled_move = wsample(all_moves, ps)
    all_unique_depths = unique(reduce(vcat, [[or_node[3] for or_node in parents_moves[move]] for move in keys(parents_moves)]))
    ranked_depths = denserank(all_unique_depths)
    if sampled_move in keys(parents_moves)
        # look at depths of all possible OR nodes, find their position in unique depths,
        # and let this index the rank
        ranked_ds = [ranked_depths[findfirst(x->x==or_node[3], all_unique_depths)] for or_node in parents_moves[sampled_move]]
        return minimum(ranked_ds)
    else
        return 1000
    end
end

function y_p_worse(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    sampled_move_idx = wsample(ps)
    sampled_s = neighs[sampled_move_idx]
    d_goal_sampled = d_goals[sampled_s]
    return d_goal_sampled > d_goal
end

function y_p_same(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    sampled_move_idx = wsample(ps)
    sampled_s = neighs[sampled_move_idx]
    d_goal_sampled = d_goals[sampled_s]
    return d_goal_sampled == d_goal
end

function y_p_better(ps, all_moves, neighs, tree, prev_move, d_goal, d_goals)
    sampled_move_idx = wsample(ps)
    sampled_s = neighs[sampled_move_idx]
    d_goal_sampled = d_goals[sampled_s]
    return d_goal_sampled < d_goal
end

function random_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    N = length(neigh)
    return ones(N) / N
end

function means_end_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    b0, b1, b2, β = params
    #b0, b1, b2, b3, β = params
    f1 = fs[:, 1]
    f2 = fs[:, 2]
    #f3 = fs[:, 3]
    ex = exp.((b0 .+ (b1 .* f1) .+ (b2 .* f2)))# ./ β)
    #ex = exp.((b0 .+ (b1 .* f1) .+ (b2 .* f2) .+ (b3 .* f3)) ./ β)
    return ex ./ sum(ex)
end

function forward_search(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    function rollout(board, γ; max_iter=10000)
        first_move = nothing
        prev_move = [0, 0]
        for n in 1:max_iter
            if rand() > γ || n == 1
                arr = get_board_arr(board)
                # Check if complete
                if check_solved(arr)
                    return true, first_move
                end
                # Expand current node by getting all available moves
                available_moves = get_all_available_moves(board, arr)
                filter!(e -> e[1] != prev_move[1], available_moves)
                if isempty(available_moves)
                    available_moves = get_all_available_moves(board, arr)
                end
                # Randomly choose a move
                selected_move_idx = rand(1:length(available_moves))
                prev_move = available_moves[selected_move_idx]
                # Make move
                make_move!(board, prev_move)
                if n == 1
                    first_move = prev_move
                end
            else
                return false, first_move
            end
        end
        return false, first_move
    end
    logγ, k = params
    γ = exp(-logγ)
    k = round(Int, k)
    N = length(neigh)
    ps = zeros(N)
    N_REPEATS = 100
    array = get_board_arr(board)
    for n in 1:N_REPEATS
        f_move = nothing
        for i in 1:k
            solved, first_move = rollout(arr_to_board(array), γ)
            if solved
                f_move = first_move
                break
            end
        end
        if f_move === nothing
            ps .+= 1/N
        else
            move_index = findfirst(x->x==Tuple(f_move), all_moves)
            ps[move_index] += 1
        end
    end
    return ps ./= N_REPEATS
end
# ff = (x) -> forward_search(x, tree_datas[subjs[3]][1][138], tree_datas[subjs[3]][2][138],  tree_datas[subjs[3]][3][138], boards[subjs[3]][138], prev_moves[subjs[3]][138], neighs[subjs[3]][138], 3, features[subjs[3]][138], d_goals)[findfirst(x->x==tree_datas[subjs[3]][4][138], tree_datas[subjs[3]][3][138])]
# a=[[ff([ii, jj]) for ii in 1:0.5:4] for jj in 100:100:2000]
# ff([1.0, 1000])
# options = Dict("tolfun"=> 0.01, "max_fun_evals"=>100, "display"=>"iter");
# bads = BADS(ff, [2.0, 100.0], [0.0, 0.0], [10.0, 2000.0], [1.0, 50.0], [5.0, 1000.0], options=options)
# ress = bads.optimize();

function gamma_k_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    γ = params
    # EXPENSIVE
    dict = propagate_ps(γ, tree)
    same_car_moves = prev_move == (0, 0) ? [(0, 0)] : [(prev_move[1], j) for j in -4:4]
    # modulate depths by (1-γ)^d
    new_dict = apply_gamma(dict, γ)
    # turn dict into probability over actions
    ps, dropout, N = process_dict(all_moves, new_dict, same_car_moves, 0.0, 1.0, board)
    return ps
end

function gamma_mus_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    γ, μ_same, μ_block = params
    same_car_moves = prev_move == (0, 0) ? [(0, 0)] : [(prev_move[1], j) for j in -4:4]
    # modulate depths by (1-γ)^d
    new_dict = apply_gamma(dict, γ)
    # turn dict into probability over actions
    ps, dropout, N = process_dict(all_moves, new_dict, same_car_moves, μ_same, μ_block, board)
    return ps
end

function gamma_only_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    γ = params
    same_car_moves = prev_move == (0, 0) ? [(0, 0)] : [(prev_move[1], j) for j in -4:4]
    # modulate depths by (1-γ)^d
    new_dict = apply_gamma(dict, γ)
    # turn dict into probability over actions
    ps, dropout, N = process_dict(all_moves, new_dict, same_car_moves, 0.0, 1.0, board)
    return ps
end

function gamma_0_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    λ = params
    same_car_moves = prev_move == (0, 0) ? [(0, 0)] : [(prev_move[1], j) for j in -4:4]
    # turn dict into probability over actions
    ps, dropout, N = process_dict(all_moves, dict, same_car_moves, 0.0, 1.0, board)
    ps = λ/length(ps) .+ (1-λ) * ps
    return ps
end

function gamma_no_same_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    γ = params
    same_car_moves = [(0, 0)]
    # modulate depths by (1-γ)^d
    new_dict = apply_gamma(dict, γ)
    # turn dict into probability over actions
    ps, dropout, N = process_dict(all_moves, new_dict, same_car_moves, 0.0, 1.0, board)
    return ps
end

function eureka_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    d, λ = params
    N = length(neigh)
    ps = zeros(N)
    if d_goal <= d
        opt_idx = [d_goals[s_p] < d_goal for s_p in neigh]
        ps[opt_idx] .= 1/sum(opt_idx)
    else
        ps = ones(N) ./ N
    end
    ps = λ/N .+ (1-λ) * ps
    return ps
end

function opt_rand_model(params, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
    λ = params
    N = length(neigh)
    ps = zeros(N)
    opt_idx = [d_goals[s_p] < d_goal for s_p in neigh]
    ps[opt_idx] .= 1/sum(opt_idx)
    ps = λ/N .+ (1-λ) * ps
    return ps
end

function summary_stats_per_subj(models, params, independent, dependent, tree_data, states, boards, neighs, prev_moves, features, d_goals, problems; seed=nothing, N_repeats=1000)
    if seed !== nothing
        Random.seed!(seed)
    end
    N = length(states)
    MM = length(models) + 3
    lik = zeros(MM, N)
    trees, dicts, all_all_moves, moves = tree_data
    independents = zeros(N)
    dependents = zeros(MM, N)
    for i in 1:N
        # prb = problems_subj[i]
        # if split(prb, "_")[2] != "16"
        #     continue
        # end
        # if i > 100
        #     break
        # end
        tree, dict, all_moves, move = trees[i], dicts[i], all_all_moves[i], moves[i]
        s, board, neigh, prev_move, fs = states[i], boards[i], neighs[i], prev_moves[i], features[i]
        d_goal = d_goals[s]
        n_A = length(all_moves)
        # subj move
        ps_subj = zeros(n_A)
        subj_move_idx = findfirst(x->x==move, all_moves)
        ps_subj[subj_move_idx] = 1
        # chance
        ps_rand = ones(n_A) / n_A
        # optimal
        ps_opt = [d_goals[nei] < d_goal ? 1 : 0 for nei in neigh]
        ps_opt = ps_opt / sum(ps_opt)
        ps = [ps_subj, ps_rand, ps_opt]
        # model probabilities
        for (n, model) in enumerate(models)
            ps_model = model(params[n], tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
            push!(ps, ps_model)
        end
        # summary stats
        independents[i] = independent(tree, all_moves, d_goal)
        for j in 1:MM
            samples = [dependent(ps[j], all_moves, neigh, tree, prev_move, d_goal, d_goals) for _ in 1:N_repeats]
            dependents[j, i] = mean(samples[samples .< 1000])
            # likelihood of subject's move
            lik[j, i] = ps[j][subj_move_idx]
        end
    end
    idxs = independents .!= 0
    return independents[idxs], dependents[:, idxs], lik[:, idxs]
end

function quantile_binning(x, y; bins=10)
    binned_x = []
    binned_y = []
    x_sorted = sort(x)
    for bin in 1:bins
        # bin bounds
        lower = quantile(x_sorted, (bin-1)/bins; sorted=true)
        upper = quantile(x_sorted, bin/bins; sorted=true)
        # bin elements
        idxs = findall(x-> lower<= x < upper, x)
        if bin == bins
            idxs = findall(x-> lower <= x <= upper, x)
        end
        idxs = intersect(idxs, findall(x->x<1000, y))
        push!(binned_x, x[idxs])
        push!(binned_y, y[idxs])
    end
    return binned_x, binned_y
end

function across_subject_summary_stats(independent, dependent, models, params, tree_datas, states, boards, neighs, prev_moves, features, d_goals, problems; bins=10, seed=nothing)
    subjs = collect(keys(tree_datas))
    M = length(subjs)
    MM = length(models) + 3
    x_means, x_sems, y_means, y_sems = [[] for _ in 1:bins], [[] for _ in 1:bins], [[[] for _ in 1:bins] for _ in 1:MM], [[[] for _ in 1:bins] for _ in 1:MM]
    for (m, subj) in ProgressBar(enumerate(subjs))
        param = params[m]
        tree_data = tree_datas[subj]
        states_, boards_, neighs_, prev_moves_, features_, problems_ = states[subj], boards[subj], neighs[subj], prev_moves[subj], features[subj], problems[subj]
        # calculate summary statistics
        X, y, lik = summary_stats_per_subj(models, param, independent, dependent, tree_data, states_, boards_, neighs_, prev_moves_, features_, d_goals, problems_; seed=seed)
        for j in 1:MM
            # quantile binning
            if isempty(X)
                continue
            end
            binned_x, binned_y = quantile_binning(X, y[j, :]; bins=bins)
            # subject specific means and SEMs
            for bin in 1:bins
                N_i = length(binned_x[bin])
                if N_i > 0
                    if j == 1
                        push!(x_means[bin], mean(binned_x[bin]))
                        push!(x_sems[bin], var(binned_x[bin]) / N_i)
                    end
                    push!(y_means[j][bin], mean(binned_y[bin]))
                    push!(y_sems[j][bin], var(binned_y[bin]) / N_i)
                end
            end
        end
    end
    println("Missing $(M - minimum(length.(x_means))) bins")
    x_mean = zeros(bins)
    y_mean = zeros(MM, bins)
    x_sem = zeros(bins)
    y_sem = zeros(MM, bins)
    for j in 1:MM
        # population means
        y_mean[j, :] = mean.(y_means[j])
        # population standard error of means
        # error due to intra-subject variance
        Ms = length.(x_means)
        y_var = var.(y_means[j]) ./ Ms
        # error due to inter-subject variance
        y_extra_var = sum.(y_sems[j]) ./ (Ms .^ 2)
        # total error
        y_sem[j, :] = sqrt.(y_var + y_extra_var)
        if j == 1
            x_mean = mean.(x_means)
            x_var = var.(x_means) ./ Ms
            x_extra_var = sum.(x_sems) ./ (Ms .^ 2)
            x_sem = sqrt.(x_var + x_extra_var)
        end
    end
    return x_mean, x_sem, y_mean, y_sem
end

function across_subject_histogram(dependent, models, params, tree_datas, states, boards, neighs, prev_moves, features, d_goals, problems; bins=10)
    subjs = collect(keys(tree_datas))
    MM = length(models) + 3
    hist = [[[] for _ in 1:bins] for _ in 1:MM]
    for (m, subj) in ProgressBar(enumerate(subjs))
        param = params[m]
        tree_data = tree_datas[subj]
        states_, boards_, neighs_, prev_moves_, features_, problems_ = states[subj], boards[subj], neighs[subj], prev_moves[subj], features[subj], problems[subj]
        # calculate summary statistics
        _, y, _ = summary_stats_per_subj(models, param, X_histogram, dependent, tree_data, states_, boards_, neighs_, prev_moves_, features_, d_goals, problems_; N_repeats=1)
        for j in 1:MM
            # histogram counting
            hist_subj = countmap(y[j, :])
            N_i = sum(values(hist_subj))
            for k in keys(hist_subj)
                if k !== NaN
                    if k >= bins
                        push!(hist[j][bins], hist_subj[k]/N_i)
                    else
                        push!(hist[j][Int(k)], hist_subj[k]/N_i)
                    end
                end
            end
        end
    end
    y_mean = zeros(MM, bins)
    y_sem = zeros(MM, bins)
    for j in 1:MM
        # population means
        empty_idxs = isempty.(hist[j])
        non_empty_idxs = .!(empty_idxs)        
        one_length_idxs = length.(hist[j]) .== 1
        data_ = hist[j][non_empty_idxs]
        y_mean[j, non_empty_idxs] = mean.(data_)
        y_mean[j, empty_idxs] .= NaN
        # population standard error of means
        # error due to intra-subject variance
        Ms = length.(data_)
        y_var = var.(data_) ./ Ms
        # total error
        y_sem[j, non_empty_idxs] = sqrt.(y_var)
        y_sem[j, empty_idxs] .= NaN
        y_sem[j, one_length_idxs] .= 0
    end
    return y_mean, y_sem
end

function plot1_data(;bins=10)
    M = 42
    params = [[params_gamma_only[m], params_gamma_0[m], params_gamma_no_same[m], params_opt_rand[m]] for m in 1:M]
    models = [gamma_only_model, gamma_0_model, gamma_no_same_model, opt_rand_model]
    #params = [[params_gamma_only[m], params_eureka[m, :], params_opt_rand[m], params_means_ends[m, :]] for m in 1:M]
    #models = [gamma_only_model, eureka_model, opt_rand_model, means_end_model]
    independent = X_d_goal
    dependents = [y_p_in_tree, y_p_undo, y_p_same_car, y_d_tree]
    d = length(dependents)
    MM = length(models) + 3
    MM = length(models) + 3
    X, Xerr, y, yerr = zeros(d, bins), zeros(d, bins), zeros(d, MM, bins), zeros(d, MM, bins)
    for i in 1:d
        println("Doing $(i) plot")
        flush(stdout)
        X[i, :], Xerr[i, :], y[i, :, :], yerr[i, :, :] = across_subject_summary_stats(independent, dependents[i], models, params, tree_datas, states, boards, neighs, prev_moves, features, d_goals, problems; bins=bins);
    end
    return X, Xerr, y, yerr
end

function plot2_data(;bins=12)
    M = 42
    params = [[params_gamma_only[m], params_gamma_0[m], params_gamma_no_same[m], params_opt_rand[m]] for m in 1:M]
    models = [gamma_only_model, gamma_0_model, gamma_no_same_model, opt_rand_model]
    # params = [[params_gamma_only[m], params_eureka[m, :], params_opt_rand[m], params_means_ends[m, :]] for m in 1:M]
    # models = [gamma_only_model, eureka_model, opt_rand_model, means_end_model]
    dependents = [y_d_tree, y_d_tree_ranked]
    d = length(dependents)
    MM = length(models) + 3
    y, yerr = zeros(d, MM, bins), zeros(d, MM, bins)
    for i in 1:d
        println("Doing $(i) plot")
        flush(stdout)
        y[i, :, :], yerr[i, :, :] = across_subject_histogram(dependents[i], models, params, tree_datas, states, boards, neighs, prev_moves, features, d_goals, problems; bins=bins)
    end
    return y, yerr
end

function plot3_data(;bins=10)
    M = 42
    params = [[params_gamma_only[m], params_gamma_0[m], params_gamma_no_same[m], params_opt_rand[m]] for m in 1:M]
    models = [gamma_only_model, gamma_0_model, gamma_no_same_model, opt_rand_model]
    # params = [[params_gamma_only[m], params_eureka[m, :], params_opt_rand[m], params_means_ends[m, :]] for m in 1:M]
    # models = [gamma_only_model, eureka_model, opt_rand_model, means_end_model]
    independent = X_d_goal
    dependents = [y_p_worse, y_p_same, y_p_better]
    d = length(dependents)
    MM = length(models) + 3
    X, Xerr, y, yerr = zeros(d, bins), zeros(d, bins), zeros(d, MM, bins), zeros(d, MM, bins)
    for i in 1:d
        println("Doing $(i) plot")
        flush(stdout)
        X[i, :], Xerr[i, :], y[i, :, :], yerr[i, :, :] = across_subject_summary_stats(independent, dependents[i], models, params, tree_datas, states, boards, neighs, prev_moves, features, d_goals, problems; bins=bins);
    end
    return X, Xerr, y, yerr
end

function plot4_data(;bins=20)
    M = 42
    params = [[params_gamma[m]] for m in 1:M]
    #params = [[params_gamma[m], params_eureka[m, :]] for m in 1:M]
    models = [gamma_only_model]#, eureka_model]
    dependents = [y_d_goal]
    d = length(dependents)
    MM = length(models) + 3
    y, yerr = zeros(d, MM, bins), zeros(d, MM, bins)
    for i in 1:d
        y[i, :, :], yerr[i, :, :] = across_subject_histogram(dependents[i], models, params, tree_datas, states, boards, neighs, prev_moves, features, d_goals, problems; bins=bins)
    end
    return y, yerr
end

function plot4_1_data(;bins=21)
    M = length(all_subj_states)
    ds = zeros(4, bins, M)
    difficulty = x -> x == 7 ? 1 : x == 11 ? 2 : x == 14 ? 3 : x == 16 ? 4 : 0
    for (m, subj) in enumerate(keys(all_subj_states))
        Ts = zeros(4)
        for prb in keys(all_subj_states[subj])
            # if prb ∉ [prbs[15], prbs[41], prbs[69]]
            #     continue
            # end
            states = all_subj_states[subj][prb]
            idx = difficulty(parse(Int, split(prb, "_")[2]))
            for state in states
                d = d_goals[state]
                if d > 20
                    d = 20
                end
                ds[idx, d+1, m] += 1
                Ts[idx] += 1
                if d == 0
                    break
                end
                #end
            end
        end
        for i in 1:4
            ds[i, :, m] ./= Ts[i]
        end
    end
    return ds
end

