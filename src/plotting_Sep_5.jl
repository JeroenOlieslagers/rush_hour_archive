function ranked_depth_histogram(params1, params2, tree_datas_prb, QQs_prb, states_prb, optimal_act, d_goals)
    subjs = collect(keys(tree_datas_prb))
    ranked_depth = Dict{String, DefaultDict{Int, Int, Int}}()
    ranked_depth_rand = Dict{String, DefaultDict{Int, Int, Int}}()
    ranked_depth_opt = Dict{String, DefaultDict{Int, Int, Int}}()
    ranked_depth_model1 = Dict{String, DefaultDict{Int, Int, Int}}()
    ranked_depth_model2 = Dict{String, DefaultDict{Int, Int, Int}}()
    tot_moves = Dict{String, Int}()
    for (m, subj) in enumerate(subjs)
        γ = params1[m]
        #γ, μ, λ = params1[m]
        λ, d_fl = params2[m, :]
        d = Int(round(d_fl))
        #β, c = params2[m, :]
        ranked_depth_subj = DefaultDict{Int, Int}(0)
        ranked_depth_chance = DefaultDict{Int, Int}(0)
        ranked_depth_optimal = DefaultDict{Int, Int}(0)
        ranked_depth_mod1 = DefaultDict{Int, Int}(0)
        ranked_depth_mod2 = DefaultDict{Int, Int}(0)
        tree_data_prb = tree_datas_prb[subj]
        QQ = QQs_prb[subj]
        states = states_prb[subj]
        tree, dicts, all_all_moves, moves = tree_data_prb
        prbs = collect(keys(tree))
        for prb in prbs
            opt_a = optimal_act[prb]
            for r in 1:length(tree[prb])
                for i in 1:length(tree[prb][r])
                    s = states[prb][r][i]
                    move = moves[prb][r][i]
                    all_moves = all_all_moves[prb][r][i]
                    move_parents = tree[prb][r][i][6]
                    same_car_moves = i > 1 ? [(moves[prb][r][i-1][1], j) for j in -4:4] : [(0, 0)]
                    rand_move = sample(all_moves)
                    opt_move = sample(opt_a[s])
                    new_dict = only_gamma_dict(dicts[prb][r][i], γ)
                    #new_dict2 = logit_dict(dicts[prb][r][i], β, c)
                    ps1, dropout, N = process_dict(all_moves, new_dict, same_car_moves)
                    ps2 = lapsed_depth_limited_random(λ, d, QQ[prb][r][i])
                    #ps2 =  process_dict(all_moves, new_dict2)
                    model1_move = wsample(collect(all_moves), Float64.(ps1))
                    model2_move = wsample(collect(all_moves), Float64.(ps2))

                    tree_moves = collect(keys(move_parents))
                    mins = []
                    for tree_move in tree_moves
                        push!(mins, minimum([n[3] for n in move_parents[tree_move]]))
                    end
                    ranks = denserank(mins)
                    ranks_dict = Dict{Tuple{Int, Int}, Int}()
                    for m in all_moves
                        if m in tree_moves
                            ranks_dict[m] = mins[findfirst(x->x==m, tree_moves)]
                        else
                            ranks_dict[m] = 1000
                        end
                    end
                    ranked_depth_subj[ranks_dict[move]] += 1
                    ranked_depth_chance[ranks_dict[rand_move]] += 1 
                    ranked_depth_optimal[ranks_dict[opt_move]] += 1 
                    ranked_depth_mod1[ranks_dict[model1_move]] += 1
                    ranked_depth_mod2[ranks_dict[model2_move]] += 1
                end
            end
        end
        ranked_depth[subj] = ranked_depth_subj
        ranked_depth_rand[subj] = ranked_depth_chance
        ranked_depth_opt[subj] = ranked_depth_optimal
        ranked_depth_model1[subj] = ranked_depth_mod1
        ranked_depth_model2[subj] = ranked_depth_mod2
        tot_moves[subj] = sum(values(ranked_depth_subj))
    end
    all_ranked_depth = zeros(Float64, 11, length(subjs))
    all_ranked_depth_rand = zeros(Float64, 11, length(subjs))
    all_ranked_depth_opt = zeros(Float64, 11, length(subjs))
    all_ranked_depth_model1 = zeros(Float64, 11, length(subjs))
    all_ranked_depth_model2 = zeros(Float64, 11, length(subjs))
    for (n, subj) in enumerate(subjs)
        for rd in 1:10
            all_ranked_depth[rd, n] = ranked_depth[subj][rd+1]/tot_moves[subj]
            all_ranked_depth_rand[rd, n] = ranked_depth_rand[subj][rd+1]/tot_moves[subj]
            all_ranked_depth_opt[rd, n] = ranked_depth_opt[subj][rd+1]/tot_moves[subj]
            all_ranked_depth_model1[rd, n] = ranked_depth_model1[subj][rd+1]/tot_moves[subj]
            all_ranked_depth_model2[rd, n] = ranked_depth_model2[subj][rd+1]/tot_moves[subj]
        end
        all_ranked_depth[11, n] = ranked_depth[subj][1000]/tot_moves[subj]
        all_ranked_depth_rand[11, n] = ranked_depth_rand[subj][1000]/tot_moves[subj]
        all_ranked_depth_opt[11, n] = ranked_depth_opt[subj][1000]/tot_moves[subj]
        all_ranked_depth_model1[11, n] = ranked_depth_model1[subj][1000]/tot_moves[subj]
        all_ranked_depth_model2[11, n] = ranked_depth_model2[subj][1000]/tot_moves[subj]
    end
    mean_ranked_depth = mean(all_ranked_depth, dims=2)
    sem_ranked_depth = std(all_ranked_depth, dims=2) ./ sqrt(length(subjs))
    mean_ranked_depth_rand = mean(all_ranked_depth_rand, dims=2)
    sem_ranked_depth_rand = std(all_ranked_depth_rand, dims=2) ./ sqrt(length(subjs))
    mean_ranked_depth_opt = mean(all_ranked_depth_opt, dims=2)
    sem_ranked_depth_opt = std(all_ranked_depth_opt, dims=2) ./ sqrt(length(subjs))
    mean_ranked_depth_model1 = mean(all_ranked_depth_model1, dims=2)
    sem_ranked_depth_model1 = std(all_ranked_depth_model1, dims=2) ./ sqrt(length(subjs))
    mean_ranked_depth_model2 = mean(all_ranked_depth_model2, dims=2)
    sem_ranked_depth_model2 = std(all_ranked_depth_model2, dims=2) ./ sqrt(length(subjs))
    plot(size=(700, 500), grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:outertop, dpi=300, legend_columns=3, 
    legendfont=font(18), 
    xtickfont=font(16), 
    ytickfont=font(16), 
    guidefont=font(32), xlabel=latexstring("d_\\textrm{tree}"), ylabel=latexstring("p"), margin=5Plots.mm, xticks=(push!(collect(2:2:11), 13), push!(string.(2:2:11), "not in tree")))
    #bar!(push!(collect(2:11), 13), mean_ranked_depth_rand, sp=1, yerr=2*sem_ranked_depth_rand, msw=0, c=palette(:default)[1], markerstrokecolor=palette(:default)[1], linecolor=palette(:default)[1], label="Chance", linewidth=4, fillstyle = :/)
    bar!(push!(collect(2:11), 13), mean_ranked_depth_model1, sp=1, yerr=2*sem_ranked_depth_model1, msw=0, c=palette(:default)[3], markerstrokecolor=palette(:default)[3], linecolor=palette(:default)[3], label="AND/OR", linewidth=4, fillstyle = :\)
    bar!(push!(collect(2:11), 13), mean_ranked_depth, sp=1, yerr=2*sem_ranked_depth, c=:black, msw=0, label="Subjects", linewidth=5, fillalpha=0)

    #plot(ylim=(0.0, 0.5), grid=false, ylabel="Proportion of all moves", xlabel="Depth of move", dpi=300, fg_legend = :transparent, size=(400, 500))
    #bar!(x1, y1, yerr=y1s ./ sqrt(length(subjs)), xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Data", linecolor=:match, color=:gray, fillstyle = :/, bar_width=0.4, markerstrokecolor=:gray)
    #plot!(1:10, mean_ranked_depth, yerr=2*sem_ranked_depth, xticks=(push!(collect(1:10), 11), push!(string.(2:11), "not in tree")), label="Data", linecolor=:match, color=:gray, fillstyle = :/, bar_width=0.4, markerstrokecolor=:gray)
    #plot!(2:11, mean_ranked_depth, yerr=2*sem_ranked_depth, label="Data", linecolor=:match, color=:gray, fillstyle = :/, bar_width=0.4, markerstrokecolor=:gray)
    #plot!(2:11, mean_ranked_depth_rand, yerr=2*sem_ranked_depth_rand, label="Chance", linecolor=:blue, color=nothing, legend=:top, bar_width=0.4, markerstrokecolor=:blue)
    #bar!(1:7, mean_ranked_depth_opt, yerr=2*sem_ranked_depth_opt, label="Optimal", linecolor=:red, color=nothing, bar_width=0.4, markerstrokecolor=:red)
    #plot!(2:11, mean_ranked_depth_model1, yerr=2*sem_ranked_depth_model1, label="AND/OR", linecolor=:green, color=nothing, bar_width=0.4, markerstrokecolor=:green)
    #bar!(1:7, mean_ranked_depth_model2, yerr=2*sem_ranked_depth_model2, label="Eureka", linecolor=:orange, color=nothing, bar_width=0.4, markerstrokecolor=:orange)
    display(plot!())
end
ranked_depth_histogram(params11, params3, tree_datas_prb, QQs_prb, states_prb, optimal_act, d_goals)

function quantile_binning(subj_data; bins=7, bounds=false)
    new_subj_data = Dict{String, Array{Any, 1}}()
    bin_bounds = Dict{String, Array{Any, 1}}()
    lens = [length(subj_data[subj]) for subj in keys(subj_data)]
    lim = maximum(lens)
    if lim != minimum(lens)
        throw(ErrorException("lim are different for different subjects"))
    end
    for subj in keys(subj_data)
        ls = []
        for i in 1:lim
            push!(ls, [i for j in 1:length(subj_data[subj][i])]...)
        end
        new_subj_data[subj] = [[] for _ in 1:bins]
        bin_bounds[subj] = [[] for _ in 1:bins]
        for bin in 1:bins
            l = Int(ceil(quantile(ls, (bin-1)/bins)))
            u = Int(floor(quantile(ls, bin/bins))) - 1
            if bin == bins
                u += 1
            end
            for i in l:u
                push!(new_subj_data[subj][bin], subj_data[subj][i]...)
                push!(bin_bounds[subj][bin], (ones(length(subj_data[subj][i]))*i)...)
            end
        end
    end
    if bounds
        av_subj = zeros(bins)
        sem_subj = zeros(bins)
        av_bin = zeros(bins)
        for i in 1:bins
            Z = 0
            means = []
            for (n, subj) in enumerate(keys(new_subj_data))
                if length(new_subj_data[subj][i]) > 0
                    mu_hat = mean(new_subj_data[subj][i])
                    N_i = sqrt(length(new_subj_data[subj][i]))
                    av_subj[i] += mu_hat * N_i
                    mean_bin = mean(bin_bounds[subj][i])
                    av_bin[i] += mean_bin * N_i
                    Z += N_i
                    push!(means, mu_hat)
                end
            end
            av_subj[i] /= Z
            sem_subj[i] = std(means) / sqrt(length(means))
            av_bin[i] /= Z
        end
        return new_subj_data, av_subj, sem_subj, av_bin
    else
        return new_subj_data
    end
end

function continuous_bin(continuous_x, y, bins)
    factor = 1 / bins
    binned_y = [[] for _ in 1:bins]
    for subj in keys(continuous_x)
        for b in 1:bins
            interval = ((b-1)*factor, b*factor)
            idx = (continuous_x[subj] .> interval[1]) .& (continuous_x[subj] .<= interval[2])# .& (y[subj] .> 0)
            push!(binned_y[b], mean(y[subj][idx]))
        end
    end
    return binned_y
end

function depth_tree_progress(params1, params2, tree_datas_prb, QQs_prb, states_prb, boards_prb, optimal_act, d_goals; bins1=10, bins2=10)
    subjs = collect(keys(tree_datas_prb))
    depth = Dict{String, Vector{Float64}}([])
    depth_rand = Dict{String, Vector{Float64}}([])
    depth_opt = Dict{String, Vector{Float64}}([])
    depth_model1 = Dict{String, Vector{Float64}}([])
    depth_model2 = Dict{String, Vector{Float64}}([])
    progress = Dict{String, Vector{Float64}}([])
    depth_d_goal = Dict{String, Vector{Vector{Float64}}}([])
    depth_rand_d_goal = Dict{String, Vector{Vector{Float64}}}([])
    depth_opt_d_goal = Dict{String, Vector{Vector{Float64}}}([])
    depth_model1_d_goal = Dict{String, Vector{Vector{Float64}}}([])
    depth_model2_d_goal = Dict{String, Vector{Vector{Float64}}}([])
    for (m, subj) in enumerate(subjs)
        γ = params1[m]
        γ2, μ_red, μ_same, μ_block, λ = params2[m, :]
        #γ2, μ, λ = params2[m, :]
        #λ, d_fl = params2[m, :]
        #d = Int(round(d_fl))
        #β, c = params2[m, :]
        depth_subj = Vector{Float64}()
        depth_chance = Vector{Float64}()
        depth_optimal = Vector{Float64}()
        depth_mod1 = Vector{Float64}()
        depth_mod2 = Vector{Float64}()
        progress_subj = Vector{Float64}()
        depth_subj_d_goal = [Float64[] for i in 1:35]
        depth_rand_subj_d_goal = [Float64[] for i in 1:35]
        depth_opt_subj_d_goal = [Float64[] for i in 1:35]
        depth_model1_subj_d_goal = [Float64[] for i in 1:35]
        depth_model2_subj_d_goal = [Float64[] for i in 1:35]
        tree_data_prb = tree_datas_prb[subj]
        QQ = QQs_prb[subj]
        states = states_prb[subj]
        tree, dicts, all_all_moves, moves = tree_data_prb
        boards = boards_prb[subj]
        prbs = collect(keys(tree))
        for prb in prbs
            opt_a = optimal_act[prb]
            for r in 1:length(tree[prb])
                for i in 1:length(tree[prb][r])
                    s = states[prb][r][i]
                    move = moves[prb][r][i]
                    board = boards[prb][r][i]
                    undo_move = [i > 1 ? (moves[prb][r][i-1][1], -moves[prb][r][i-1][2]) : (0, 0)]
                    same_car_moves = i > 1 ? [(moves[prb][r][i-1][1], j) for j in -4:4] : [(0, 0)]
                    all_moves = all_all_moves[prb][r][i]
                    move_parents = tree[prb][r][i][6]
                    rand_move = sample(all_moves)
                    opt_move = sample(opt_a[s])
                    new_dict = only_gamma_dict(dicts[prb][r][i], γ)
                    #new_dict2 = logit_dict(dicts[prb][r][i], β, c)
                    ps1, dropout, N = process_dict(all_moves, new_dict, same_car_moves, 0, 1, board)

                    new_dict2 = only_gamma_dict(dicts[prb][r][i], γ2)
                    ps2, dropout, N = process_dict(all_moves, new_dict2, same_car_moves, μ_same, μ_block, board)
                    idx = Int[]
                    for n in eachindex(all_moves)
                        if all_moves[n][1] == 9
                            push!(idx, n)
                        end
                    end
                    if !isempty(idx)
                        #ps2[idx] = sum(ps2[idx])*μ/length(idx) .+ (1-μ)*ps2[idx]
                        ps2 = μ_red*[ci in idx for (ci, _) in enumerate(ps2)]/length(idx) .+ (1-μ_red)*ps2
                    end
                    ps2 = λ/length(ps2) .+ (1-λ)*ps2
                    #ps2 = lapsed_depth_limited_random(λ, d, QQ[prb][r][i])
                    #ps2 =  process_dict(all_moves, new_dict2)
                    model1_move = wsample(collect(all_moves), Float64.(ps1))
                    model2_move = wsample(collect(all_moves), Float64.(ps2))
                    # if move in keys(move_parents)
                    #     #d = mean([n[3] for n in move_parents[move]])
                    #     d = 1
                    # else
                    #     d = 0
                    # end
                    target0 = keys(move_parents)#opt_a[s]#
                    target1 = [all_moves[j] for j in eachindex(all_moves) if QQ[prb][r][i][j] < d_goals[s]]
                    target2 = [all_moves[j] for j in eachindex(all_moves) if QQ[prb][r][i][j] == d_goals[s]]
                    target3 = [all_moves[j] for j in eachindex(all_moves) if QQ[prb][r][i][j] > d_goals[s]]
                    target4 = undo_move
                    target5 = same_car_moves
                    target = target1

                    # if move in keys(move_parents)
                    #     if length(move_parents[move]) > 0
                    #         dd = mean([n[3] for n in move_parents[move]])
                    #         push!(depth_subj_d_goal[d_goals[s]], dd)
                    #     end
                    # end
                    # if rand_move in keys(move_parents)
                    #     if length(move_parents[rand_move]) > 0
                    #         ddr = mean([n[3] for n in move_parents[rand_move]])
                    #         push!(depth_rand_subj_d_goal[d_goals[s]], ddr)
                    #     end
                    # end
                    # if opt_move in keys(move_parents)
                    #     if length(move_parents[opt_move]) > 0
                    #         ddo = mean([n[3] for n in move_parents[opt_move]])
                    #         push!(depth_opt_subj_d_goal[d_goals[s]], ddo)
                    #     end
                    # end
                    # if model1_move in keys(move_parents)
                    #     if length(move_parents[model1_move]) > 0
                    #         ddd1 = mean([n[3] for n in move_parents[model1_move]])
                    #         push!(depth_model1_subj_d_goal[d_goals[s]], ddd1)
                    #     end
                    # end
                    # if model2_move in keys(move_parents)
                    #     ddd2 = mean([n[3] for n in move_parents[model2_move]])
                    #     push!(depth_model2_subj_d_goal[d_goals[s]], ddd2)
                    # end

                    dd = move in target ? 1 : 0
                    ddr = rand_move in target ? 1 : 0
                    ddo = opt_move in target ? 1 : 0
                    ddd1 = model1_move in target ? 1 : 0
                    #ddd1 = model1_move in target2 ? 1 : 0
                    #ddd1 = (1-dropout) + dropout*length(move_parents)/length(all_moves)
                    ddd2 = model2_move in target ? 1 : 0
                    #ddd2 = model1_move in target3 ? 1 : 0
                    push!(depth_subj, dd)
                    push!(depth_chance, ddr)
                    push!(depth_optimal, ddo)
                    push!(depth_mod1, ddd1)
                    push!(depth_mod2, ddd2)
                    push!(progress_subj, i/length(tree[prb][r]))
                    push!(depth_subj_d_goal[d_goals[s]], dd)
                    push!(depth_rand_subj_d_goal[d_goals[s]], ddr)
                    push!(depth_opt_subj_d_goal[d_goals[s]], ddo)
                    push!(depth_model1_subj_d_goal[d_goals[s]], ddd1)
                    push!(depth_model2_subj_d_goal[d_goals[s]], ddd2)
                end
            end
        end
        depth[subj] = depth_subj
        depth_rand[subj] = depth_chance
        depth_opt[subj] = depth_optimal
        depth_model1[subj] = depth_mod1
        depth_model2[subj] = depth_mod2
        progress[subj] = progress_subj
        depth_d_goal[subj] = depth_subj_d_goal
        depth_rand_d_goal[subj] = depth_rand_subj_d_goal
        depth_opt_d_goal[subj] = depth_opt_subj_d_goal
        depth_model1_d_goal[subj] = depth_model1_subj_d_goal
        depth_model2_d_goal[subj] = depth_model2_subj_d_goal
    end
    in_tree = continuous_bin(progress, depth, bins1)
    in_tree_rand = continuous_bin(progress, depth_rand, bins1)
    in_tree_opt = continuous_bin(progress, depth_opt, bins1)
    in_tree_model1 = continuous_bin(progress, depth_model1, bins1)
    in_tree_model2 = continuous_bin(progress, depth_model2, bins1)
    _, av_subj, sem_subj, av_bin = quantile_binning(depth_d_goal; bins=bins2, bounds=true)
    _, av_subj_rand, sem_subj_rand, _ = quantile_binning(depth_rand_d_goal; bins=bins2, bounds=true)
    _, av_subj_opt, sem_subj_opt, _ = quantile_binning(depth_opt_d_goal; bins=bins2, bounds=true)
    _, av_subj_model1, sem_subj_model1, _ = quantile_binning(depth_model1_d_goal; bins=bins2, bounds=true)
    _, av_subj_model2, sem_subj_model2, _ = quantile_binning(depth_model2_d_goal; bins=bins2, bounds=true)

    # plot(layout=grid(2, 1, heights=[0.55, 0.45]), size=(900, 900), grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:outertop, dpi=300, legend_columns=3, 
    # legendfont=font(18), 
    # xtickfont=font(16), 
    # ytickfont=font(16), 
    # guidefont=font(32))
    plot(size=(700, 500), grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:outertop, dpi=300, legend_columns=3, 
    legendfont=font(18), 
    xtickfont=font(16), 
    ytickfont=font(16), 
    guidefont=font(32), xlabel=latexstring("d_\\textrm{goal}"), ylabel=latexstring("p_\\textrm{better}"), margin=5Plots.mm)
    #plot!([], [], yerr=[], sp=1, c=:black, label="Data")
    #plot!(av_bin, av_subj, sp=1, yerr=2*sem_subj, ms=10, l=nothing, markershape=:none, label=nothing, linewidth=4, xflip=true)
    plot!(av_bin, av_subj, sp=1, yerr=2*sem_subj, c=:black, msw=0, label="Subjects", xflip=true, linewidth=5)
    # plot!(av_bin, av_subj, sp=1, msw=0, label="Better", xflip=true, linewidth=5)
    # plot!(av_bin, av_subj_model1, sp=1, msw=0, label="Same", xflip=true, linewidth=5)
    # plot!(av_bin, av_subj_model2, sp=1, msw=0, label="Worse", xflip=true, linewidth=5)
    #areaplot!(av_bin, [av_subj_model2 + av_subj_model1 + av_subj, av_subj_model1 + av_subj, av_subj], xflip=true, linewidth=5, label=["Worse" "Same" "Better"])    
    plot!(av_bin, av_subj_rand, sp=1, ribbon=2*sem_subj_rand, c=palette(:default)[1], label="Chance", linewidth=4, xflip=true)
    #plot!(av_bin, av_subj_opt, sp=1, ribbon=2*sem_subj_opt, c=palette(:default)[2], label="Optimal", linewidth=4, xflip=true)
    plot!(av_bin, av_subj_model1, sp=1, ribbon=2*sem_subj_model1, c=palette(:default)[3], label="AND/OR", linewidth=4, xflip=true)
    plot!(av_bin, av_subj_model2, sp=1, ribbon=2*sem_subj_model2, c=palette(:default)[4], label="AND/OR+", linewidth=4, xflip=true)#, ylabel=latexstring("p_\\textrm{worse}"), xlabel=latexstring("d_\\textrm{goal}"))
    
    #display(plot!())

    #plot(grid=false, background_color_legend=nothing, foreground_color_legend=nothing, ylabel="In tree proportion", xlabel="Progress", xlim=(0, 1), legend=:bottomleft, dpi=300)
    # y = [mean(b) for b in in_tree]
    # yerr = [std(b)/sqrt(length(b)) for b in in_tree]
    # plot!((collect(1:bins1) .- 0.5) ./ bins1, y, sp=2, yerr=2*yerr, ms=10, l=nothing, markershape=:none, label=nothing, linewidth=4)
    # plot!((collect(1:bins1) .- 0.5) ./ bins1, y, sp=2, yerr=2*yerr, c=:transparent, msw=1, label=nothing)

    # y_rand = [mean(b) for b in in_tree_rand]
    # yerr_rand = [std(b)/sqrt(length(b)) for b in in_tree_rand]
    # plot!((collect(1:bins1) .- 0.5) ./ bins1, y_rand, sp=2, ribbon=2*yerr_rand, c=palette(:default)[1], label="Chance", linewidth=4)

    # y_opt = [mean(b) for b in in_tree_opt]
    # yerr_opt = [std(b)/sqrt(length(b)) for b in in_tree_opt]
    # plot!((collect(1:bins1) .- 0.5) ./ bins1, y_opt, sp=2, ribbon=2*yerr_opt, c=palette(:default)[2], label="Optimal", linewidth=4)

    # y_model1 = [mean(b) for b in in_tree_model1]
    # yerr_model1 = [std(b)/sqrt(length(b)) for b in in_tree_model1]
    # plot!((collect(1:bins1) .- 0.5) ./ bins1, y_model1, sp=2, ribbon=2*yerr_model1, c=palette(:default)[3], label="AND/OR", linewidth=4)

    # y_model2 = [mean(b) for b in in_tree_model2]
    # yerr_model2 = [std(b)/sqrt(length(b)) for b in in_tree_model2]
    # plot!((collect(1:bins1) .- 0.5) ./ bins1, y_model2, sp=2, ribbon=2*yerr_model2, c=palette(:default)[4], label="Eureka", linewidth=4, ylabel=latexstring("p_\\textrm{error}"), xlabel="Progress", legend=nothing, xlim=(0, 1))

    # for j in 1:length(in_tree[1])
    #     plot!((collect(1:bins1) .- 0.5) ./ bins1, [in_tree[i][j] for i in 1:bins], sp=2, linewidth=1, color=:black, alpha=0.2, label=nothing)
    # end
    # plot!([], [], sp=2, linewidth=1, color=:black, alpha=0.2, label="Subjects")
    display(plot!())
end

function biggest_residuals(params, params2, tree_datas_prb, states_prb, optimal_act, d_goals)
    lls = []
    pss = []
    pss2 = []
    all_movess = []
    movess = []
    opt_movess = []
    ss = []
    dictss = []
    for (m, subj) in enumerate(subjs)
        #γ = params1[m]
        γ, μ, λ = params[m, :]
        γ2, μ2, λ2 = params2[m, :]
        tree, dicts, all_all_moves, moves = tree_datas_prb[subj]
        states = states_prb[subj]
        prbs = collect(keys(tree))
        for prb in prbs
            opt_a = optimal_act[prb]
            for r in 1:length(tree[prb])
                for i in 1:length(tree[prb][r])
                    s = states[prb][r][i]
                    move = moves[prb][r][i]
                    opt_moves = opt_a[s]
                    #undo_move = [i > 1 ? (moves[prb][r][i-1][1], -moves[prb][r][i-1][2]) : (0, 0)]
                    same_car_moves = i > 1 ? [(moves[prb][r][i-1][1], j) for j in -4:4] : [(0, 0)]
                    all_moves = all_all_moves[prb][r][i]
                    new_dict = only_gamma_dict(dicts[prb][r][i], γ)
                    ps, dropout, N = process_dict(all_moves, new_dict, same_car_moves)
                    idx = Int[]
                    for n in eachindex(all_moves)
                        if all_moves[n][1] == 9
                            push!(idx, n)
                        end
                    end
                    if !isempty(idx)
                        #ps[idx] = sum(ps[idx])*μ/length(idx) .+ (1-μ)*ps[idx]
                        ps = μ*[ci in idx for (ci, _) in enumerate(ps)]/length(idx) .+ (1-μ)*ps
                    end
                    ps = λ/length(ps) .+ (1-λ)*ps

                    new_dict2 = only_gamma_dict(dicts[prb][r][i], γ2)
                    ps2, dropout, N = process_dict(all_moves, new_dict2, [(0, 0)])
                    idx = Int[]
                    unblockable = false
                    for n in eachindex(all_moves)
                        # if all_moves[n][1] == 9
                        #     push!(idx, n)
                        # end
                        if all_moves[n] == (9, 3)
                            push!(idx, n)
                            for k in keys(dicts[prb][r][i])
                                if k[3] == 2 && k[2] != (-1, -1)
                                    unblockable = true
                                end
                            end
                        end
                    end
                    if !isempty(idx) && !unblockable
                        #ps[idx] = sum(ps[idx])*μ/length(idx) .+ (1-μ)*ps[idx]
                        #ps2 = μ2*[ci in idx for (ci, _) in enumerate(ps2)]/length(idx) .+ (1-μ2)*ps2
                        ps2 = μ2*[ci in idx for (ci, _) in enumerate(ps2)]/length(idx) .+ (1-μ2)*ps2
                    end
                    ps2 = λ2/length(ps2) .+ (1-λ2)*ps2

                    # ps2 = []
                    # if i > 1
                    #     ps2 = [mm[1] == moves[prb][r][i-1][1] ? 1 : 0 for mm in all_moves]
                    # end

                    d = d_goals[s]
                    if (d > 5) || (all_moves[argmax(ps)] in opt_moves) || (move ∉ opt_moves)
                        continue
                    end
                    push!(lls, log(ps2[findfirst(x->x==move, all_moves)]))
                    push!(pss, ps)
                    push!(pss2, ps2)
                    push!(all_movess, all_moves)
                    push!(movess, move)
                    push!(opt_movess, opt_moves)
                    push!(ss, s)
                    push!(dictss, new_dict)
                end
            end
        end
    end
    idxs = sortperm(lls)
    #subidx = [1, 5, 15, 17, 18, 19]
    subidx = [14]
    println(ss[idxs[subidx[1]]])
    println(dictss[idxs[subidx[1]]])
    display(move_hist(pss, pss2, all_movess, movess, opt_movess, ss, subidx, idxs))
end
biggest_residuals(params14, params16, tree_datas_prb, states_prb, optimal_act, d_goals)

function depth_tree_progress_by_level(tree_datas_prb; bins=10)
    subjs = collect(keys(tree_datas_prb))
    depth = Dict{String, Any}([])
    depth_rand = Dict{String, Any}([])
    progress = Dict{String, Any}([])
    for subj in subjs
        depth_subj = DefaultDict{String, Vector{Int}}([])
        depth_chance = DefaultDict{String, Vector{Int}}([])
        progress_subj = DefaultDict{String, Vector{Float64}}([])
        tree_data_prb = tree_datas_prb[subj]
        tree, dicts, all_all_moves, moves = tree_data_prb
        prbs = collect(keys(tree))
        for prb in prbs
            for r in 1:length(tree[prb])
                for i in 1:length(tree[prb][r])
                    move = moves[prb][r][i]
                    move_parents = tree[prb][r][i][6]
                    rand_move = sample(all_all_moves[prb][r][i])
                    if move in keys(move_parents)
                        #d = mean([n[3] for n in move_parents[move]])
                        d = 1
                    else
                        d = 0
                    end
                    dd = rand_move in keys(move_parents) ? 1 : 0
                    push!(depth_subj[prb[end:end]], d)
                    push!(depth_chance[prb[end:end]], dd)
                    push!(progress_subj[prb[end:end]], i/length(tree[prb][r]))
                end
            end
        end
        depth[subj] = depth_subj
        depth_rand[subj] = depth_chance
        progress[subj] = progress_subj
    end
    factor = 1 / bins
    in_tree = Dict{String, Vector{Vector{Float64}}}()
    in_tree_rand = Dict{String, Vector{Vector{Float64}}}()
    for diff in keys(progress[subjs[1]])
        in_tree[diff] = [[] for _ in 1:bins]
        in_tree_rand[diff] = [[] for _ in 1:bins]
        for subj in subjs
            if diff in keys(progress[subj])
                for b in 1:bins
                    interval = ((b-1)*factor, b*factor)
                    idx = (progress[subj][diff] .> interval[1]) .& (progress[subj][diff] .<= interval[2])
                    push!(in_tree[diff][b], mean(depth[subj][diff][idx]))
                    push!(in_tree_rand[diff][b], mean(depth_rand[subj][diff][idx]))
                end
            end
        end
    end
    plot(grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend_title="L", ylabel="In tree proportion", xlabel="Progress", xlim=(0, 1), legend=:bottomleft)
    difficulty = Dict("7"=>6, "1"=>10, "4"=>13, "6"=>15)
    for k in ["7", "1", "4", "6"]
        y = [mean(b) for b in in_tree[k]]
        yerr = [std(b)/sqrt(length(b)) for b in in_tree[k]]
        plot!((collect(1:bins) .- 0.5) ./ bins, y, ribbon=yerr, label=difficulty[k], linewidth=4)
        # for j in 1:length(in_tree[k][1])
        #     plot!((collect(1:bins) .- 0.5) ./ bins, [in_tree[k][i][j] for i in 1:5], linewidth=1, color=:black, alpha=0.2, label=nothing)
        # end
    end
    for k in ["7", "1", "4", "6"]
        y_rand = [mean(b) for b in in_tree_rand[k]]
        yerr_rand = [std(b)/sqrt(length(b)) for b in in_tree_rand[k]]
        plot!((collect(1:bins) .- 0.5) ./ bins, y_rand, ribbon=yerr_rand, label=string(difficulty[k])*" chance", linewidth=4, alpha=0.3)
    end
    display(plot!())
end

function RT_plot(RTs_prb, tree_datas_prb)
    ds = []
    rts = []
    for subj in collect(keys(tree_datas_prb))
        tree, dicts, all_all_moves, moves = tree_datas_prb[subj]
        RTs = RTs_prb[subj]
        prbs = collect(keys(tree))
        for prb in prbs
            for r in 1:length(tree[prb])
                for i in 1:length(tree[prb][r])
                    move = moves[prb][r][i]
                    move_parents = tree[prb][r][i][6]
                    # if move in keys(move_parents)
                    #     d = mean([n[3] for n in move_parents[move]])
                    #     rt = RTs[prb][r][i]
                    #     push!(ds, d)
                    #     push!(rts, rt)
                    # end
                    push!(ds, i/length(tree[prb][r]))
                    #push!(ds, move in keys(move_parents))
                    push!(rts, RTs[prb][r][i])
                end
            end
        end
    end
    plot(grid=false, dpi=300, xlabel="Progress", ylabel="log(RT)")
    scatter!(ds, log10.(rts), label=nothing)
    display(plot!())
    return ds, rts
end
a, b = RT_plot(RTs_prb, tree_datas_prb)

optimal_act = load("data/processed_data/optimal_act.jld2");
depth_tree_progress(params11, params20, tree_datas_prb, QQs_prb, states_prb, boards_prb, optimal_act, d_goals; bins2=10);
depth_tree_depth_puzzle(tree_datas_prb, all_subj_states; bins=10)


a = 1