using Distributions
using StatsBase
using Random
include("model.jl");

function X_histogram(tree, all_moves, d_goal)
    return 0
end

function X_d_goal(tree, all_moves, d_goal)
    return d_goal
end

function X_n_A(tree, all_moves, d_goal)
    return length(all_moves)
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

function gamma_only_model(params, dict, all_moves, board, prev_move, neigh, d_goal, d_goals)
    γ = params
    same_car_moves = prev_move == (0, 0) ? [(0, 0)] : [(prev_move[1], j) for j in -4:4]
    # modulate depths by (1-γ)^d
    new_dict = apply_gamma(dict, γ)
    # turn dict into probability over actions
    ps, dropout, N = process_dict(all_moves, new_dict, same_car_moves, 0.0, 1.0, board)
    return ps
end

function eureka_model(params, dict, all_moves, board, prev_move, neigh, d_goal, d_goals)
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

function summary_stats_per_subj(models, params, independent, dependent, tree_data, states, boards, neighs, first_moves, d_goals; seed=nothing, N_repeats=1000)
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
        tree, dict, all_moves, move = trees[i], dicts[i], all_all_moves[i], moves[i]
        s, board, neigh, first_move = states[i], boards[i], neighs[i], first_moves[i]
        d_goal = d_goals[s]
        n_A = length(all_moves)
        prev_move = first_move ? (0, 0) : moves[i-1]
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
            ps_model = model(params[n], dict, all_moves, board, prev_move, neigh, d_goal, d_goals)
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
    return independents, dependents, lik
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

function across_subject_summary_stats(independent, dependent, models, params, tree_datas, states, boards, neighs, first_moves, d_goals; bins=10, seed=nothing)
    subjs = collect(keys(tree_datas))
    M = length(subjs)
    MM = length(models) + 3
    x_means, x_sems, y_means, y_sems = [[] for _ in 1:bins], [[] for _ in 1:bins], [[[] for _ in 1:bins] for _ in 1:MM], [[[] for _ in 1:bins] for _ in 1:MM]
    for (m, subj) in ProgressBar(enumerate(subjs))
        param = params[m]
        tree_data = tree_datas[subj]
        states_, boards_, neighs_, first_moves_ = states[subj], boards[subj], neighs[subj], first_moves[subj]
        # calculate summary statistics
        X, y, lik = summary_stats_per_subj(models, param, independent, dependent, tree_data, states_, boards_, neighs_, first_moves_, d_goals; seed=seed)
        for j in 1:MM
            # quantile binning
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

function across_subject_histogram(dependent, models, params, tree_datas, states, boards, neighs, first_moves, d_goals; bins=10)
    subjs = collect(keys(tree_datas))
    MM = length(models) + 3
    hist = [[[] for _ in 1:bins] for _ in 1:MM]
    for (m, subj) in ProgressBar(enumerate(subjs))
        param = params[m]
        tree_data = tree_datas[subj]
        states_, boards_, neighs_, first_moves_ = states[subj], boards[subj], neighs[subj], first_moves[subj]
        # calculate summary statistics
        _, y, _ = summary_stats_per_subj(models, param, X_histogram, dependent, tree_data, states_, boards_, neighs_, first_moves_, d_goals)
        for j in 1:MM
            # histogram counting
            hist_subj = countmap(y[j, :])
            N_i = sum(values(hist_subj))
            for k in keys(hist_subj)
                if k != 1000
                    push!(hist[j][Int(k)], hist_subj[k]/N_i)
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
    params = [[params_gamma[m]] for m in 1:M]
    #params = [[params_gamma[m], params_eureka[m, :]] for m in 1:M]
    models = [gamma_only_model]#, eureka_model]
    independent = X_d_goal
    dependents = [y_p_in_tree, y_p_undo, y_p_same_car, y_d_tree]
    d = length(dependents)
    MM = length(models) + 3
    MM = length(models) + 3
    X, Xerr, y, yerr = zeros(d, bins), zeros(d, bins), zeros(d, MM, bins), zeros(d, MM, bins)
    for i in 1:d
        X[i, :], Xerr[i, :], y[i, :, :], yerr[i, :, :] = across_subject_summary_stats(independent, dependents[i], models, params, tree_datas, states, boards, neighs, first_moves, d_goals; bins=bins);
    end
    return X, Xerr, y, yerr
end

function plot2_data(;bins=12)
    M = 42
    params = [[params_gamma[m]] for m in 1:M]
    #params = [[params_gamma[m], params_eureka[m, :]] for m in 1:M]
    models = [gamma_only_model]#, eureka_model]
    dependents = [y_d_tree, y_d_tree_ranked]
    d = length(dependents)
    MM = length(models) + 3
    y, yerr = zeros(d, MM, bins), zeros(d, MM, bins)
    for i in 1:d
        y[i, :, :], yerr[i, :, :] = across_subject_histogram(dependents[i], models, params, tree_datas, states, boards, neighs, first_moves, d_goals; bins=bins)
    end
    return y, yerr
end

function plot3_data(;bins=10)
    M = 42
    params = [[params_gamma[m]] for m in 1:M]
    #params = [[params_gamma[m], params_eureka[m, :]] for m in 1:M]
    models = [gamma_only_model]#, eureka_model]
    independent = X_d_goal
    dependents = [y_p_worse, y_p_same, y_p_better]
    d = length(dependents)
    MM = length(models) + 3
    X, Xerr, y, yerr = zeros(d, bins), zeros(d, bins), zeros(d, MM, bins), zeros(d, MM, bins)
    for i in 1:d
        X[i, :], Xerr[i, :], y[i, :, :], yerr[i, :, :] = across_subject_summary_stats(independent, dependents[i], models, params, tree_datas, states, boards, neighs, first_moves, d_goals; bins=bins, seed=123);
    end
    return X, Xerr, y, yerr
end
