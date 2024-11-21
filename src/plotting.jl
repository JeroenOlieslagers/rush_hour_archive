# MAIN TEXT FIGURES

function fig2A(prbs)
    s = load_data(prbs[41])
    make_move!(s[1], (Int8(7), Int8(-1)))
    draw_board(s)
end

function fig2C(df, prbs)
    df_ = df[df.puzzle .== prbs[41], :]
    tree = df_[2, :].tree
    s = (df_[2, :].s_free, df_[2, :].s_fixed)
    draw_ao_tree(tree[2], tree[3], s)
end

function fig2D(df, prbs)
    plot(size=(350, 450), layout=grid(2, 2), grid=false, dpi=300, xflip=false,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=3Plots.mm, top_margin=0Plots.mm, bottom_margin=3Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, link=:y)

    xticks = [[0, 5, 10, 15, 20], [0, 10, 20, 30], [0, 20, 40, 60, 80], [0, 50, 100, 150]]
    yticks = [[0, 5, 10, 15], [], [0, 5, 10, 15], []]
    titles = ["Length 5", "Length 9", "Length 12", "Length 14"]
    ylabels = ["Distance to goal" "" "Distance to goal" ""]
    xlabels = ["Move number" "Move number" "Move number" "Move number"]

    for (n, prb) in enumerate([prbs[12], prbs[26], prbs[46], prbs[61]])#[15, 20, 48, 69]
        for subj in subjs
            df_subj = df[df.subject .== subj .&& df.event .!= "start" .&& df.event .!= "restart" .&& df.puzzle .== prb, :]
            plot!(df_subj.d_goal, sp=n, label=nothing, c=:black, alpha=0.2, xticks=xticks[n], yticks=yticks[n], xlim=(0, Inf), ylim=(0, 18), ylabel=ylabels[n], xlabel=xlabels[n], title=titles[n])
        end
    end
    display(plot!())
end

function fig4(binned_stats)
    models = ["data", "random_model", "random_model"]
    DVs = ["y_p_in_tree", "y_p_in_tree", "y_d_tree"]
    IDV = "X_d_goal"
    d = length(DVs)
    l = @layout [grid(1, d); a{0.001h}];
    plot(size=(744, 250), grid=false, layout=l, dpi=300, xflip=true,
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"), 
        right_margin=2Plots.mm, top_margin=1Plots.mm, bottom_margin=7Plots.mm, left_margin=7Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 16));

    ylabels = ["Prop. sensible \nactions (human)" "Prop. sensible \nactions (all)" "Average depth \n of sensible action"];
    ytickss = [[0.8, 0.9, 1.0], [0.2, 0.4, 0.6], [2.0, 3.0, 4.0, 5.0]]
    ylimss = [(0.75, 1.0), (0.15, 0.7), (2, 5.3)]

    for i in 1:d
        df_model = binned_stats[binned_stats.model .== models[i], :]
        sort!(df_model, :bin_number)
        ylabel = ylabels[i]
        title = ""
        yticks = ytickss[i]
        xticks = [0, 5, 10, 15]
        ylims = ylimss[i]
        sp = i
        plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], yerr=df_model[:, "sem_"*DVs[i]], sp=sp, c=:black, msw=1.4, label=nothing, xflip=true, linewidth=1, markershape=:none, ms=4, title=title, ylabel=ylabel, xticks=xticks, yticks=yticks, ylims=ylims)
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=d + 1, top_margin=-15Plots.mm, bottom_margin=7Plots.mm)
    display(plot!())
end

function fig5(df_stats)
    plot(layout=grid(1, 3), grid=false, legend=nothing, size=(744, 250), dpi=300,
            legendfont=font(14, "helvetica"), 
            xtickfont=font(12, "helvetica"), 
            ytickfont=font(12, "helvetica"), 
            titlefont=font(14, "helvetica"), 
            guidefont=font(14, "helvetica"), 
            right_margin=2Plots.mm, top_margin=0Plots.mm, bottom_margin=7Plots.mm, left_margin=4Plots.mm, 
            fontfamily="helvetica", tick_direction=:out)

    # We only care about the participants' data
    df_ = df_stats[df_stats.model .== "data", :]
    # Bin by whether move is first move or not
    binned_stats = bin_stats(df_, :X_first_move; nbins=2)
    sort!(binned_stats, :bin_number, rev=true)
    rts = binned_stats[binned_stats.model .== "data", :m_X_rt] ./ 1000
    rts_err = binned_stats[binned_stats.model .== "data", :sem_X_rt] ./ 1000
    bar!(["First\naction", "Other\nactions"], rts, bar_width=0.5, color=nothing, sp=1, yerr=rts_err, grid=false, label=nothing, ylabel="Response time (s)", xlim=(0, 2), yscale=:log10, fillrange=0.9, yminorticks=true, yticks=([1, 10], ["1", "10"]))
    # This is to get individual participant data
    binned_stats = bin_stats(df_, :X_first_move; nbins=2, subject_level=true)
    rts1 = binned_stats[binned_stats.bin_number .== 1, :mean_X_rt] ./ 1000
    rts2 = binned_stats[binned_stats.bin_number .== 2, :mean_X_rt] ./ 1000
    for i in 1:42
        plot!(["First\naction", "Other\nactions"], [rts2[i], rts1[i]], sp=1, label=nothing, c=:black, alpha=0.1)
    end
    # Bin by distance to goal for non-first moves
    binned_stats = bin_stats(df_[df_.X_first_move .== 0, :], :X_d_goal; nbins=10)
    sort!(binned_stats, :bin_number)
    plot!(binned_stats[:, :m_X_d_goal], binned_stats[:, :m_X_rt], sp=2, yerr=binned_stats[:, :sem_X_rt], label=nothing, ylim=(800, 2400), yticks=[1000, 1500, 2000], msw=1.4, ms=4, linewidth=1, markershape=:none, c=:black, xflip=true, xlabel="Distance to goal", ylabel="Response time (ms)", xlim=(0, 16), yminorticks=true)
    # Bin by depth in tree for sensible moves
    binned_stats = bin_stats(df_[df_.y_p_in_tree .== 1, :], :y_d_tree; nbins=7)
    sort!(binned_stats, :bin_number)
    plot!(binned_stats[:, :m_y_d_tree], binned_stats[:, :m_X_rt], sp=3, yerr=binned_stats[:, :sem_X_rt], label=nothing, ylim=(1350, 2550), yticks=[1500, 2000, 2500], msw=1.4, ms=4, linewidth=1, markershape=:none, c=:black, xflip=true, xlabel="Depth in tree", ylabel="Response time (ms)", xlim=(1.5, 6.7), yminorticks=true)
    display(plot!())
end

function fig6AD(binned_stats)
    models = ["random_model", "gamma_only_model"] 
    DVs = ["y_p_in_tree", "y_d_tree", "y_p_undo", "y_p_same_car"]
    IDV = "X_d_goal"
    MM = length(models)
    d = length(DVs)
    l = @layout [grid(1, d); a{0.001h}];
    plot(size=(744, 200), grid=false, layout=l, dpi=300, xflip=true,
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"), 
        right_margin=0Plots.mm, top_margin=1Plots.mm, bottom_margin=7Plots.mm, left_margin=5Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 16));

    ylabels = ["Prop. sensible" "Depth in tree" "Prop. undos" "Prop. same car"];
    ytickss = [[0.4, 0.6, 0.8, 1.0], [2.0, 3.0, 4.0, 5.0], ([0.0, 0.1, 0.2], ["0", "0.1", "0.2"]), ([0.0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])]
    ylimss = [(0.25, 1.0), (2, 5.3), (-Inf, 0.2), (0, 0.3)]
    order = [1, 3]

    for i in 1:d
        for j in 1:MM
            df_data = binned_stats[binned_stats.model .== "data", :]
            df_model = binned_stats[binned_stats.model .== models[j], :]
            sort!(df_data, :bin_number)
            sort!(df_model, :bin_number)
            ylabel = ylabels[i]
            title = ""
            yticks = ytickss[i]
            xticks = [0, 5, 10, 15]
            ylims = ylimss[i]
            sp = i
            o = order[j]

            plot!(df_data[:, "m_"*IDV], df_data[:, "m_"*DVs[i]], yerr=df_data[:, "sem_"*DVs[i]], sp=sp, c=:white, msw=1.4, label=nothing, xflip=true, linewidth=1, markershape=:none, ms=4, ylabel=ylabel, xticks=xticks, yticks=yticks)
            plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], ribbon=df_model[:, "sem_"*DVs[i]], sp=sp, label=nothing, c=palette(:default)[o], l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], ribbon=df_model[:, "sem_"*DVs[i]], sp=sp, label=nothing, c=palette(:default)[o], l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
        end
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=d + 1, top_margin=-15Plots.mm, bottom_margin=7Plots.mm)
    display(plot!())
end

function fig6EF(df_stats)
    models = ["random_model", "gamma_only_model"]
    Vs = [:h_d_tree, :h_d_tree_diff]
    lims = [2:11, 1:9]
    MM = length(models)
    d = length(Vs)
    l = @layout [grid(1, d)];
    plot(size=(298, 200), grid=false, layout=l, dpi=300, xflip=false,
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=1Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out);

    xlabels = ["Depth" latexstring("\\Delta\\textrm{ depth}")]
    order = [1, 3]
    ytickss = [([0.0, 0.1, 0.2], ["0", "0.1", "0.2"]), ([0.0, 0.2, 0.4], ["0", "0.2", "0.4"])]
    xtickss = [[2, 4, 6, 8, 10], [0, 2, 4, 6, 8]]
    for i in 1:d
        r = Vs[i]
        df_ = df_stats[df_stats.h_d_tree .< 1000, :]
        gdf = groupby(df_, [:subject, :model, r])
        count_df = combine(gdf, r => length => :hist_counts)
    
        count_df_norm = normalize_hist_counts(count_df, "subject", "model", r, lims[i])
    
        diff_gdf = groupby(count_df_norm, [:model, r])
        diff_df = combine(diff_gdf, :norm_counts => (x -> [(mean(x), sem(x))]) => [:hist_mean, :hist_sem])

        df_data = diff_df[diff_df.model .== "data", :]
        xlabel = xlabels[i]
        ylabel = i == 1 ? "Proportion" : ""
        title = ""
        yticks = ytickss[i]
        xticks = xtickss[i]
        bar!(df_data[:, r], df_data[:, :hist_mean], yerr=1.96*df_data[:, :hist_sem], sp=i, c=:white, msw=1.4, label=nothing, linewidth=1.4, markershape=:none, ms=0, title=title, ylabel=ylabel, xticks=xticks, yticks=yticks, linecolor=:gray, markercolor=:gray, xlabel=xlabel, left_margin=i==1 ? 0Plots.mm : -4Plots.mm)
        for j in 1:MM
            df_model = diff_df[diff_df.model .== models[j], :]
            sort!(df_model, r)
            o = order[j]
            plot!(df_model[:, r], df_model[:, :hist_mean], ribbon=1.96*df_model[:, :hist_sem], sp=i, label=nothing, c=palette(:default)[o], l=nothing)
            plot!(df_model[:, r], df_model[:, :hist_mean], ribbon=1.96*df_model[:, :hist_sem], sp=i, label=nothing, c=palette(:default)[o], l=nothing)
        end
    end
    display(plot!())
end

function fig6GI(binned_stats)
    models = ["data", "random_model", "gamma_only_model"]
    DVs = ["m_y_p_worse", "m_y_p_same", "m_y_p_better"]
    IDV = "m_X_d_goal"
    MM = length(models)
    d = length(DVs)
    l = @layout [a{0.001h}; grid(1, MM); a{0.001h}];
    plot(size=(446, 200), grid=false, layout=l, dpi=300, xflip=true, link=:both,
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"), 
        right_margin=0Plots.mm, top_margin=1Plots.mm, bottom_margin=4Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 15), ylim=(0, 1), yticks=nothing)

    labels = ["Worse   " "Same   " "Better   "]
    bar!([0 0 0], c=[palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=0, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top, top_margin=-2Plots.mm);
    titles = ["Participants" "Random" "AND-OR"];
    xlabels = ["" "" ""]
    ylabels = ["Proportion" "" ""]
    yticks = [([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"]) for _ in 1:MM]
    xticks = [[0, 5, 10, 15] for _ in 1:MM]
    for i in 1:MM
        df_ = binned_stats[binned_stats.model .== models[i], :]
        areaplot!(df_[:, IDV] .- 1, [df_[:, DVs[1]] + df_[:, DVs[2]] + df_[:, DVs[3]], df_[:, DVs[2]] + df_[:, DVs[3]], df_[:, DVs[3]]], sp=i+1, xflip=true, label=nothing, xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i], yticks=yticks[i], xticks=xticks[i], left_margin=i==1 ? 2Plots.mm : -1Plots.mm)    
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=MM+2, top_margin=-12Plots.mm)
    display(plot!())
end

function fig7(df_models; iters=1000)
    plot(size=(372, 200), grid=false, dpi=300,
        legendfont=font(9, "helvetica"), 
        xtickfont=font(8, "helvetica"), 
        ytickfont=font(8, "helvetica"), 
        titlefont=font(9, "helvetica"), 
        guidefont=font(9, "helvetica"), 
        right_margin=1Plots.mm, top_margin=0Plots.mm, bottom_margin=2Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, background_color_legend=nothing, foreground_color_legend=nothing)

    base_cv_nlls = df_models[df_models.model .== "gamma_only_model", :cv_nll]
    N = length(unique(df_models.model)) - 1
    means = zeros(N)
    err = zeros(N, 2)
    n = 0
    for model in unique(df_models.model)
        if model == "gamma_no_same_model"
            continue
        end
        n += 1
        cv_nlls = df_models[df_models.model .== model, :cv_nll]
        M = length(cv_nlls)
        total = zeros(iters)
        for i in 1:iters
            subsampled_idxs = rand(1:M, M)
            total[i] = sum(cv_nlls[subsampled_idxs] - base_cv_nlls[subsampled_idxs])
        end
        means[n] = mean(total)
        err[n, :] = [mean(total) - quantile(total, 0.025), quantile(total, 0.975) - mean(total)]
    end
    names = ["AND-OR tree", "AND-OR tree ("*latexstring("\\gamma")*"=0)", "Eureka", "Optimal-random", "Forward search", "Hill climbing", "Random"];
    switched_order = [1, 2, 3, 5, 4, 6, 7]
    bar!(names[switched_order], means[switched_order], yerr=(err[switched_order, 1], err[switched_order, 2]), xflip=true, label=nothing, xlim=(0, N), ylim=(0, 25000), bar_width=0.8, permute=(:x, :y), yticks=([0, 4, 8, 12, 16, 20, 24]*1000, [0, 4, 8, 12, 16, 20, 24]), markersize=5, linewidth=1.4, ylabel="\n"*latexstring("\\Delta")*"NLL (x1000)", c=:transparent)
    display(plot!())
end

# EXTENDED DATA FIGURES

function fig_ext1(prbs)
    board = load_data(prbs[41])
    make_move!(board[1], (Int8(7), Int8(-1)))

    mvs = [(5, -2), (3, -2), (4, 2), (6, 3), (1, 1), (7, -3), (1, -1), (6, -3), (4, -3), (6, 3), (3, 2), (5, 3)];

    for (n, mv) in enumerate(mvs)
        make_move!(board[1], (Int8(mv[1]), Int8(mv[2])))
        display(draw_board(board))
        #savefig("move$(n).svg")
    end
end

function fig_ext2(df, messy_data)
    puzzle_df = puzzle_statistics(df)

    subj_diff_gdf = groupby(puzzle_df, [:subject, :Lopt])
    opt_df = combine(subj_diff_gdf, [:Lopt, :L] => ((x, y) -> [(sum(x .== y), length(y))]) => [:opt, :completed])
    add_completion_data!(opt_df, messy_data, prbs)

    diff_gdf = groupby(opt_df, :Lopt)
    diff_df = combine(diff_gdf, [:opt, :attempted, :completed, :total] => difficulty_stats => 
    [:opt_mean, :opt_sem, :attempt_mean, :attempt_sem, :complete_mean, :complete_sem])


    plot(layout=grid(1, 3), size=(372*2, 300), grid=false, dpi=300,         
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"),
        right_margin=0Plots.mm, top_margin=1Plots.mm, bottom_margin=6Plots.mm, left_margin=4Plots.mm, 
        fontfamily="helvetica", tick_direction=:out)

    @df diff_df bar!(:opt_mean, yerr=:opt_sem, sp=1, xticks=(1:4, :Lopt), label=nothing, xlabel="Length", ylabel="Proportion optimal", ylim=(0, 0.32), c=:transparent, ms=10)
    @df diff_df bar!(:attempt_mean, yerr=:attempt_sem, sp=2, xticks=(1:4, :Lopt), label=nothing, xlabel="Length", ylabel="Attempt rate", ylim=(0, 1), c=:transparent, ms=10)
    @df diff_df bar!(:complete_mean, yerr=:complete_sem, sp=3, xticks=(1:4, :Lopt), label=nothing, xlabel="Length", ylabel="Completion rate", ylim=(0, 1), c=:transparent, ms=10)
    display(plot!())
end

function fig_ext3(df)
    # get d_goal where above 20 gets clumped together
    df_max_20 = transform(df[df.event .== "move" .|| df.event .== "win", :], :d_goal => max_20 => :d_goal_20)
    # count how how many of each d_goal appear per subject per difficulty level
    dgoal_diff_gdf = groupby(df_max_20, [:subject, :Lopt, :d_goal_20])
    dgoal_diff_df = combine(dgoal_diff_gdf, :d_goal_20 => length => :hist_counts)
    dgoal_diff_df_norm = normalize_hist_counts(dgoal_diff_df, "subject", "Lopt", "d_goal_20", 0:20)
    # calculate subject error bars
    diff_gdf = groupby(dgoal_diff_df_norm, [:Lopt, :d_goal_20])
    diff_df = combine(diff_gdf, :norm_counts => (x -> [(mean(x), sem(x))]) => [:hist_mean, :hist_sem])

    l = @layout [grid(1, 4); a{0.001h}];
    plot(size=(372, 150), grid=false, layout=l, dpi=300, xflip=false, link=:both,
        legendfont=font(9, "helvetica"), 
        xtickfont=font(8, "helvetica"), 
        ytickfont=font(8, "helvetica"), 
        titlefont=font(9, "helvetica"), 
        guidefont=font(9, "helvetica"), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=3Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xticks=([1, 11, 21], ["0", "10", "20+"]),
        background_color_legend=nothing, foreground_color_legend=nothing, legend=:topright)

    titles = ["Length 5", "Length 9", "Length 12", "Length 14"]
    ytickss = [[0, 0.1, 0.2], [], [], []]
    diffs = [5, 9, 12, 14]
    ylabels = ["Proportion" "" "" ""]
    for i in 1:4
        diff = diffs[i]
        dummy = diff_df[diff_df.Lopt .== diff, :]
        means = dummy.hist_mean
        sems = 1.96*dummy.hist_sem
        bar!(means, yerr=sems, sp=i, linecolor=:gray, markerstrokecolor=:gray, linewidth=1, markersize=0, label=nothing, title=titles[i], ylabel=ylabels[i], fillcolor=:transparent, xlabel="", yticks=ytickss[i])
        plot!([diffs[i]+1, diffs[i]+1], [0, 1/(diffs[i]+1)], sp=i, label=i < 4 ? nothing : "optimal", c=:red, linestyle=:dash)
        plot!([0.5, diffs[i]+1], [1/(diffs[i]+1), 1/(diffs[i]+1)], sp=i, label=nothing, c=:red, linestyle=:dash)
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=5, top_margin=-10Plots.mm, bottom_margin=2Plots.mm)
    display(plot!())
end

function fig_ext4(df, d_goals_prbs)
    # coarseness of surface
    N = 100
    # choice of gammas
    log_gammas = range(-2, 0, N)
    plot([], [], grid=false, c=:black, label="NLL surface", xlabel=latexstring("\\gamma"), ylabel="z-scored NLL", size=(300, 200))
    for subj in ProgressBar(unique(df.subject))
        df_subj = df[df.subject .== subj .&& df.event .== "move", :]
        nlls = zeros(N)
        # calculate NLL for each gamma
        for (n, log_gamma) in enumerate(log_gammas)
            nlls[n] = subject_nll_general(gamma_only_model, 10^log_gamma, df_subj, d_goals_prbs)
        end
        plot!(10 .^(log_gammas), zscore(nlls), label=nothing, c=:black, alpha=0.15)
        vline!([10^log_gammas[argmin(nlls)]], c=:red, alpha=0.3, label=nothing)
    end
    plot!([], [], xscale=:log10, xticks=[0.01, 0.1, 1.0], xlim=(0.01, 1.0), c=:red, label="Minimum", background_color_legend=nothing, foreground_color_legend=nothing)
    display(plot!())
end

function fig_ext6(binned_stats)
    models = ["random_model", "optimal_model", "hill_climbing_model", "forward_search", "eureka_model", "opt_rand_model", "gamma_0_model", "gamma_only_model"]
    DVs = ["y_p_in_tree", "y_d_tree", "y_p_undo", "y_p_same_car"]
    IDV = "X_d_goal"
    MM = length(models)
    d = length(DVs)
    l = @layout [grid(d, MM); a{0.001h}];
    plot(size=(744, 700), grid=false, layout=l, dpi=300, xflip=true, link=:both,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=8Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 16));

    titles = ["Random" "Optimal" "Hill\nclimbing" "Forward" "Eureka" "Optimal-\nrandom" "AND-OR\n(g=0)" "AND-OR"];
    ylabels = ["Prop. sensible" "Depth in tree\n" "Prop. undos" "Prop. same car"];
    ytickss = [[0.4, 0.6, 0.8, 1.0], [2.0, 3.0, 4.0, 5.0], ([0.0, 0.1, 0.2], ["0", "0.1", "0.2"]), ([0.0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])]
    ylimss = [(-Inf, 1.0), (2, 5.3), (-Inf, 0.2), (0, 0.3)]
    order = [2, 3, 7, 6, 5, 8, 9, 4]

    for i in 1:d
        for j in 1:MM
            df_data = binned_stats[binned_stats.model .== "data", :]
            df_model = binned_stats[binned_stats.model .== models[j], :]
            sort!(df_data, :bin_number)
            sort!(df_model, :bin_number)
            xlabel = i == d ? "Distance\nto goal" : ""
            ylabel = j == 1 ? ylabels[i] : ""
            title = i == 1 ? titles[j] : ""
            yticks = j == 1 ? ytickss[i] : nothing
            xticks = i == d ? [0, 5, 10, 15] : [0, 5, 10, 15]
            ylims = ylimss[i]
            sp = (i-1)*MM + j
            o = order[j]

            plot!(df_data[:, "m_"*IDV], df_data[:, "m_"*DVs[i]], yerr=df_data[:, "sem_"*DVs[i]], sp=sp, c=:white, msw=1.4, label=nothing, xflip=true, linewidth=1, markershape=:none, ms=4, ylabel=ylabel, xticks=xticks, yticks=yticks)
            if i == 1 && j == 2
                plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], ribbon=df_model[:, "sem_"*DVs[i]], sp=sp, label=nothing, c=palette(:default)[o-1], ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            else
                plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], ribbon=df_model[:, "sem_"*DVs[i]], sp=sp, label=nothing, c=palette(:default)[o-1], l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            end
            plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], ribbon=df_model[:, "sem_"*DVs[i]], sp=sp, label=nothing, c=palette(:default)[o-1], l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
        end
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=d*MM + 1, top_margin=-18Plots.mm, bottom_margin=0Plots.mm)
    display(plot!())
end

function fig_ext7(df_stats)
    models = ["random_model", "optimal_model", "hill_climbing_model", "forward_search", "eureka_model", "opt_rand_model", "gamma_0_model", "gamma_only_model"]
    Vs = [:h_d_tree, :h_d_tree_diff]
    lims = [2:11, 1:8]
    MM = length(models)
    d = length(Vs)
    l = @layout [grid(1, MM); a{0.001h}; grid(1, MM); a{0.001h}];
    plot(size=(744, 450), grid=false, layout=l, dpi=300, xflip=false, link=:y,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=6Plots.mm, bottom_margin=4Plots.mm, left_margin=2Plots.mm, 
        fontfamily="helvetica", tick_direction=:out);

    titles = ["Random" "Optimal" "Hill\nclimbing" "Forward" "Eureka" "Optimal-\nrandom" "AND-OR\n(g=0)" "AND-OR"];
    xlabels = ["Depth" latexstring("\\Delta\\textrm{ depth}")]
    order = [2, 3, 7, 6, 5, 8, 9, 4]
    ytickss = [([0.0, 0.1, 0.2], ["0", "0.1", "0.2"]), ([0.0, 0.2, 0.4], ["0", "0.2", "0.4"])]
    xtickss = [[2, 4, 6, 8, 10], [0, 2, 4, 6, 8]]
    ii = 0
    for i in 1:d
        r = Vs[i]
        # Select only the sensible moves
        df_ = df_stats[df_stats.h_d_tree .< 1e9, :]
        gdf = groupby(df_, [:subject, :model, r])
        count_df = combine(gdf, r => length => :hist_counts)
        count_df_norm = normalize_hist_counts(count_df, "subject", "model", r, lims[i])
        # Take average of histograms
        diff_gdf = groupby(count_df_norm, [:model, r])
        diff_df = combine(diff_gdf, :norm_counts => (x -> [(mean(x), 1.96*sem(x))]) => [:hist_mean, :hist_sem])
        for j in 1:MM
            df_data = diff_df[diff_df.model .== "data", :]
            df_model = diff_df[diff_df.model .== models[j], :]
            sort!(df_model, r)
            ylabel = j == 1 ? "Proportion" : ""
            title = i == 1 ? titles[j] : ""
            yticks = j == 1 ? ytickss[i] : nothing
            xticks = xtickss[i]
            o = order[j]
            ii += 1
            model_sem = df_model[:, :hist_sem]
            model_sem[isnan.(model_sem)] .= 0
            bar!(df_data[:, r], df_data[:, :hist_mean], yerr=df_data[:, :hist_sem], sp=ii, c=:white, msw=1.4, label=nothing, linewidth=1.4, markershape=:none, ms=0, title=title, ylabel=ylabel, xticks=xticks, yticks=yticks, linecolor=:gray, markercolor=:gray)
            plot!(df_model[:, r], df_model[:, :hist_mean], ribbon=model_sem, sp=ii, label=nothing, c=palette(:default)[o-1])
            plot!(df_model[:, r], df_model[:, :hist_mean], ribbon=model_sem, sp=ii, label=nothing, c=palette(:default)[o-1])
        end
        ii += 1
        plot!(xlabel=xlabels[Int(1+Int(ii > 10))], showaxis=false, grid=false, sp=ii, top_margin=-12Plots.mm)
    end
    display(plot!())
end

function fig_ext8(binned_stats)
    models = ["data", "random_model", "optimal_model", "hill_climbing_model", "forward_search", "eureka_model", "opt_rand_model", "gamma_0_model", "gamma_only_model"]
    DVs = ["m_y_p_worse", "m_y_p_same", "m_y_p_better"]
    IDV = "m_X_d_goal"
    MM = length(models)
    d = length(DVs)
    l = @layout [a{0.001h}; grid(1, MM); a{0.001h}];
    plot(size=(744, 300), grid=false, layout=l, dpi=300, xflip=true, link=:both,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=4Plots.mm, bottom_margin=6Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 15), ylim=(0, 1), yticks=nothing)

    labels = ["Worse" "Same" "Better"]
    bar!([0 0 0], c=[palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=0, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top, top_margin=-2Plots.mm);
    titles = ["Data" "Random" "Optimal" "Hill\nclimbing" "Forward" "Eureka" "Optimal-\nrandom" "AND-OR\n(g=0)" "AND-OR"];
    xlabels = ["" "" "" "" "" "" "" "" ""]
    ylabels = ["Proportion" "" "" "" "" "" "" "" ""]
    yticks = [([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"]), [([0, 0.2, 0.4, 0.6, 0.8, 1], ["", "", ""]) for _ in 1:8]...]
    xticks = [[0, 5, 10, 15] for _ in 1:9]
    for i in 1:MM
        df_ = binned_stats[binned_stats.model .== models[i], :]
        areaplot!(df_[:, IDV] .- 1, [df_[:, DVs[1]] + df_[:, DVs[2]] + df_[:, DVs[3]], df_[:, DVs[2]] + df_[:, DVs[3]], df_[:, DVs[3]]], sp=i+1, xflip=true, label=nothing, xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i], yticks=yticks[i], xticks=xticks[i])    
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=MM+2, top_margin=-14Plots.mm)
    display(plot!())
end

# SUPPLEMENTARY INFORMATION FIGURES

function fig_supp1(xs, ys)
    plot(layout=grid(2, 2), dpi=300, grid=false, legendfont=font(12, "helvetica"), 
    xtickfont=font(10, "helvetica"), 
    ytickfont=font(10, "helvetica"), 
    titlefont=font(8, "helvetica"), 
    guidefont=font(12, "helvetica"), size=(372, 300))

    yticks = [[0, 0.5, 1.0], [0.1, 0.2, 0.3], [0.0, 0.5, 1.0], [0.1, 0.2, 0.3]]
    ylims = [(0, 1.1), (0.08, 0.31), (0, 1.1), (0.08, 0.31)]
    xlabel = [latexstring("M"), latexstring("b_N"), latexstring("\\sigma"), latexstring("\\sigma'")]
    titles = [  latexstring("\\sigma=1, \\sigma'=10, b_N=10"),
                latexstring("\\sigma=1, \\sigma'=10, M=100"),
                latexstring("\\sigma'=10, M=100, b_N=10"),
                latexstring("\\sigma=1, M=100, b_N=10")]

    for i in 1:4
        scatter!(xs[i], ys[:, i, 1], sp=i, label=false, alpha=0.2, color=:blue, xscale=:log10, title=titles[i], xlabel=xlabel[i], ylabel=latexstring("\\epsilon"), yticks=yticks[i], ylim=ylims[i])
        plot!(xs[i], ys[:, i, 2], sp=i, label=false, color=:blue, linewidth=2, xscale=:log10)
        scatter!(xs[i], ys[:, i, 3], sp=i, label=false, alpha=0.2, color=:red, xscale=:log10)
        plot!(xs[i], ys[:, i, 4], sp=i, label=false, color=:red, linewidth=2, xscale=:log10)
    end
    plot!()
end

function fig_supp2(true_params, fitted_params)    
    plot(layout=(1, 2), grid=false, aspect_ratio=1, dpi=300, legendfont=font(12, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), size=(372, 200))
    for i in eachindex(true_params)
        tp = true_params[i]
        fp = fitted_params[i]
        scatter!([tp[1]], [fp[1]], sp=1, c=:red, label=nothing, xlim=(0, 3.5), ylim=(0, 3.5))
        scatter!(log10.([tp[2]]), log10.([fp[2]]), sp=2, c=:blue, label=nothing, xlim=(0, 3.5), ylim=(0, 3.5))
    end
    plot!([0, 3.5], [0, 3.5], sp=1, c=:black, label=nothing, title=latexstring("-\\log_{10}\\gamma"), yticks=[0, 1, 2, 3], xticks=[0, 1, 2, 3])
    plot!([0, 3.5], [0, 3.5], sp=2, c=:black, label=nothing, title=latexstring("\\log_{10}k"), yticks=[0, 1, 2, 3], xticks=[0, 1, 2, 3])
    display(plot!(xlabel="true", ylabel="fitted"))
end
