

function X_histogram(move, row, d_goals)
    return 1
end

function X_d_goal(move, row, d_goals)
    return row.d_goal
end

function X_n_A(move, row, d_goals)
    return length(row.all_moves)
end

function X_diff(move, row, d_goals)
    return row.Lopt
end

function y_d_goal(move, row, d_goals)
    move_idx = findfirst(x->x==move, row.all_moves)
    sampled_s = row.neighs[move_idx]
    d_goal_next = d_goals[row.puzzle][sampled_s]
    return d_goal_next+1
end

function y_p_in_tree(move, row, d_goals)
    AND_root, AND, OR, parents_moves = row.tree
    return move ∈ keys(parents_moves)
end

function y_p_undo(move, row, d_goals)
    if row.prev_move == (0, 0)
        return 0
    else
        undo_move = (row.prev_move[1], -row.prev_move[2])
        return move == undo_move
    end
end

function y_p_same_car(move, row, d_goals)
    if row.prev_move == (0, 0)
        return 0
    else
        prev_car = row.prev_move[1]
        curr_car = move[1]
        return curr_car == prev_car
    end
end

function y_d_tree(move, row, d_goals)
    AND_root, AND, OR, parents_moves = row.tree
    if move in keys(parents_moves)
        ds = [or_node[1] for or_node in parents_moves[move]]
        return minimum(ds)
    else
        return 1000
    end
end

function y_d_tree_ranked(move, row, d_goals)
    AND_root, AND, OR, parents_moves = row.tree
    all_unique_depths = unique(reduce(vcat, [[or_node[1] for or_node in parents_moves[move]] for move in keys(parents_moves)]))
    ranked_depths = denserank(all_unique_depths)
    if move in keys(parents_moves)
        # look at depths of all possible OR nodes, find their position in unique depths,
        # and let this index the rank
        ranked_ds = [ranked_depths[findfirst(x->x==or_node[1], all_unique_depths)] for or_node in parents_moves[move]]
        return minimum(ranked_ds)
    else
        return 1000
    end
end

function y_p_worse(move, row, d_goals)
    move_idx = findfirst(x->x==move, row.all_moves)
    sampled_s = row.neighs[move_idx]
    d_goal_sampled = d_goals[row.puzzle][sampled_s]
    return d_goal_sampled > row.d_goal
end

function y_p_same(move, row, d_goals)
    move_idx = findfirst(x->x==move, row.all_moves)
    sampled_s = row.neighs[move_idx]
    d_goal_sampled = d_goals[row.puzzle][sampled_s]
    return d_goal_sampled == row.d_goal
end

function y_p_better(move, row, d_goals)
    move_idx = findfirst(x->x==move, row.all_moves)
    sampled_s = row.neighs[move_idx]
    d_goal_sampled = d_goals[row.puzzle][sampled_s]
    return d_goal_sampled < row.d_goal
end

function calculate_summary_statistics(df, df_models, d_goals_prbs; iters=100)
    summary_stats = [X_d_goal, X_n_A, X_diff, y_d_goal, y_p_in_tree, y_p_undo, y_p_same_car, y_d_tree, y_d_tree_ranked, y_p_worse, y_p_same, y_p_better, y_d_tree, y_d_tree_ranked]
    N_stats = length(summary_stats)
    models = [random_model, optimal_model, gamma_only_model, gamma_0_model, gamma_no_same_model, eureka_model, forward_search, opt_rand_model, hill_climbing_model]
    df_stats = DataFrame(subject=String[], puzzle=String[], model=String[], X_d_goal=Int[], X_n_A=Int[], X_diff=Int[], y_d_goal=Float64[], y_p_in_tree=Float64[], y_p_undo=Float64[], y_p_same_car=Float64[], y_d_tree=Float64[], y_d_tree_ranked=Float64[], y_p_worse=Float64[], y_p_same=Float64[], y_p_better=Float64[], h_d_tree=Int[], h_d_tree_ranked=Int[])
    df_ = df[df.event .== "move", :]
    i = 0
    for row in ProgressBar(eachrow(df_))
        i += 1
        # if i > 2000
        #     break
        # end
        # subject data
        s = zeros(N_stats)
        for (n, sum_stat) in enumerate(summary_stats)
            s[n] = sum_stat(row.move, row, d_goals_prbs)
        end
        stats = [row.subject, row.puzzle, "data"]
        push!(df_stats, vcat(stats, s))
        # model simulations
        for model in models
            stats = [row.subject, row.puzzle, string(model)]
            if model == optimal_model || model == forward_search
                params = 0
            else
                params = df_models[df_models.subject .== row.subject .&& df_models.model .== string(model), :params][1]
            end
            if length(params) == 1
                params = params[1]
            end
            s = zeros(N_stats - 2, iters)
            for i in 1:iters
                if model == forward_search
                    ps = random_model(params, row, d_goals_prbs)
                else
                    ps = model(params, row, d_goals_prbs)
                end
                move = wsample(row.all_moves, ps)
                for (n, sum_stat) in enumerate(summary_stats)
                    s[n, i] = sum_stat(move, row, d_goals_prbs)
                    if n == N_stats - 2
                        break
                    end
                end
            end
            stats = vcat(stats, [mean(ss[ss .< 1000]) for ss in eachrow(s)])
            if model == forward_search
                first_ps = random_model(params, row, d_goals_prbs)
            else
                first_ps = model(params, row, d_goals_prbs)
            end
            first_move = wsample(row.all_moves, first_ps)
            stats = vcat(stats, [summary_stats[end-1](first_move, row, d_goals_prbs), summary_stats[end](first_move, row, d_goals_prbs)])
            push!(df_stats, stats)
        end
    end
    return df_stats
end

function bin_stats(df_stats, independent_var::Symbol; nbins=10)
    input_column_names = propertynames(df_stats)[4:end]
    output_column_names = Symbol[]
    for n in input_column_names
        push!(output_column_names, Symbol("mean_"*string(n)))
        push!(output_column_names, Symbol("std_"*string(n)))
    end
    final_column_names = Symbol[]
    for n in input_column_names
        push!(final_column_names, Symbol("m_"*string(n)))
        push!(final_column_names, Symbol("sem_"*string(n)))
    end
    binned_stats_subj = []
    for subj in unique(df_stats.subject)
        df_subj = df_stats[df_stats.subject .== subj, :]
        transform!(df_subj, independent_var => (x -> add_bin_number(x; nbins=nbins)) => :bin_number)
        gdf_subj = groupby(df_subj, [:bin_number, :model])
        dummy = combine(gdf_subj, input_column_names => calculate_mean_sem_1 => output_column_names)
        dummy[!, :subject] .= subj
        push!(binned_stats_subj, dummy)
    end
    binned_stats_subj_ = vcat(binned_stats_subj...)
    gdf_binned_stats = groupby(binned_stats_subj_, [:model, :bin_number])
    return combine(gdf_binned_stats, output_column_names => calculate_mean_sem_2 => final_column_names)
end

function add_bin_number(d_goal; nbins=10)
    bin_n = zeros(Int64, length(d_goal))
    bins = quantile(d_goal, 0:(1/nbins):1)
    for i in 1:nbins
        if i < nbins
            idxs = d_goal .>= bins[i] .&& d_goal .< bins[i+1]
        else
            idxs = d_goal .>= bins[i] .&& d_goal .<= bins[i+1]
        end
        bin_n[idxs] .= i
    end
    return bin_n
end

function calculate_mean_sem_1(cols...)
    ls = []
    for col in cols
        push!(ls, mean(col[col .< 1000]), sem(col[col .< 1000]))
    end
    return [Tuple(ls)]
end

function calculate_mean_sem_2(cols...)
    ls = []
    for i in 1:Int(length(cols)/2)
        m = cols[2*i - 1]
        s = cols[2*i]
        push!(ls, mean(m), 2*sqrt(sem(m)^2 + mean(s .^ 2)/length(s)))
    end
    return [Tuple(ls)]
end


