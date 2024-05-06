using Plots
using LaTeXStrings
using GraphViz
using FileIO
include("analysis.jl")

function create_move_icon(move, board)
    car, m = move
    s = string(car)
    if board.cars[car].is_horizontal
        for i in 1:abs(m)
            if sign(m) == 1
                s *= "→"
            else
                s *= "←"
            end
        end
    else
        for i in 1:abs(m)
            if sign(m) == 1
                s *= "↓"
            else
                s *= "↑"
            end
        end
    end
    return s
end

function draw_ao_tree(AO, board; highlight_ORs=[], outside_moves=[], heatmap_moves=nothing)
    graph = """digraph{graph [pad="0.2",nodesep="0.15",ranksep="0.3"];layout="dot";"""
    drawn_and = []
    drawn_or = []
    AND, OR = AO
    for (n, move) in enumerate(outside_moves)
        graph *= """ "outside_and_$(n)"[fontname="Helvetica",fixedsize=shape,style=filled,fillcolor=white,width=0.6,margin=0,label="$(move[1])",fontsize=20,shape="circle"];"""
        graph *= """ "outside_and_$(n)"->"outside_or_$(n)"[constraint="true"];"""
        if heatmap_moves !== nothing
            alpha = string(round(Int, heatmap_moves[move] * 255 / maximum(values(heatmap_moves))), base = 16)
            graph *= """ "outside_or_$(n)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="#9CB4EE$(alpha)",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
        else
            graph *= """ "outside_or_$(n)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="#9CB4EE",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
        end
    end
    for and in keys(AND)
        if and ∉ drawn_and
            push!(drawn_and, and)
            graph *= """ "$(and)"[fontname="Helvetica",fixedsize=shape,style=filled,fillcolor=white,width=0.6,margin=0,label="$(and[1][1])",fontsize=20,shape="circle"];"""
        end
        for or in AND[and]
            graph *= """ "$(and)"->"$(or)"[constraint="true"];"""
            if or ∉ keys(OR)
                move = or[2]
                graph *= """ "$(or)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="#E05B53",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
            end
        end
    end
    for or in keys(OR)
        if or ∉ drawn_or
            push!(drawn_or, or)
            move = or[2]
            if heatmap_moves !== nothing && move in keys(heatmap_moves)
                alpha = string(round(Int, heatmap_moves[move] * 255 / maximum(values(heatmap_moves))), base = 16)
                graph *= """ "$(or)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(or in highlight_ORs ? "#9CB4EE$(alpha)" : OR[or][1][1] == (0, (0,)) ? "#86D584$(alpha)" : "#DBDBDB$(alpha)")",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
            else
                graph *= """ "$(or)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(or in highlight_ORs ? "#9CB4EE" : OR[or][1][1] == (0, (0,)) ? "#86D584" : "#DBDBDB")",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
            end
        end
        for and in OR[or]
            if and[1] == (0, (0,))
                continue
            end
            if and ∉ drawn_and
                push!(drawn_and, and)
                graph *= """ "$(and)"[fontname="Helvetica",fixedsize=shape,style=filled,fillcolor="#E05B53",width=0.6,margin=0,label="$(and[1][1])",fontsize=20,shape="circle"];"""
            end
            graph *= """ "$(or)"->"$(and)"[constraint="true"];"""
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

"""
    save_graph(g, file_name)

Save graph as svg
"""
function save_graph(g, file_name)
    FileIO.save(file_name * ".svg", g)
end

function draw_board(arr; alpha_map=[], sp=nothing)
    """
    Draws board state as heatmap with custom colormap
    """
    cmap = [
        RGBA(([255, 255, 255]./255)...), 
        RGBA(([147, 190, 103]./255)...), 
        RGBA(([102, 152, 80]./255)...), 
        RGBA(([80, 173, 202]./255)...), 
        RGBA(([219, 130, 57]./255)...), 
        RGBA(([81, 51, 154]./255)...), 
        RGBA(([185, 156, 105]./255)...), 
        RGBA(([126, 74, 51]./255)...), 
        RGBA(([124, 124, 124]./255)...), 
        RGBA(([202, 76, 60]./255)...)
        ]
    for i in eachindex(alpha_map)
        cmap[alpha_map[i]+1].alpha = 0.3
    end
    if sp === nothing
        heatmap(arr, c = cmap, legend = false, yflip = true, xmirror=true, framestyle = :box, size=(200, 200), dpi=300)
        vline!(1.5:5.5, c=:black, linewidth=0.2)
        hline!(1.5:5.5, c=:black, linewidth=0.2)
        for c in sort(unique(arr))[2:end]
            idxs = findall(x->x==c, arr)
            l = length(idxs)
            s = sum(idxs)
            annotate!(s[2]/l, s[1]/l, text(c, :white, 20))
        end
    else
        heatmap!(arr, c = cmap, sp=sp, legend = nothing, yflip = true, xmirror=true, framestyle = :box, dpi=300)
        vline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2, legend=nothing)
        hline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2, legend=nothing)
        for c in sort(unique(arr))[2:end]
            idxs = findall(x->x==c, arr)
            l = length(idxs)
            s = sum(idxs)
            annotate!(s[2]/l, s[1]/l, sp=sp, text(c, :white, 20), legend=nothing)
        end
    end
    plot!()
end

function summary_stats1_plot(X, Xerr, y, yerr)
    d, MM, bins = size(y)
    l = @layout [a{0.01h}; grid(2, 2)];
    plot(size=(500, 450), grid=false, layout=l, dpi=300, xflip=true, #size=(750, 250)
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out);

    labels = ["Subjects" "Random" "Optimal"];
    for i in 1:(MM-3)
        labels = hcat(labels, "Model "*string(i))
    end
    println(labels)
    plot!(zeros(Int, MM)', c=[:black palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=10, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top);
    for i in 1:d
        for j in 1:MM
            # if j == 3
            #     continue
            # end
            if j == 1
                plot!(X[i, :], y[i, j, :], yerr=2*yerr[i, j, :], sp=i+1, c=:white, msw=2, label=nothing, xflip=true, linewidth=2, markershape=:none, ms=6)
            else
                plot!(X[i, :], y[i, j, :], ribbon=2*yerr[i, j, :], sp=i+1, label=nothing, c=palette(:default)[j-1], l=nothing)
            end
        end
    end
    display(plot!())
end

function summary_stats1_1_plot(X, Xerr, y, yerr)
    d, MM, bins = size(y)
    MM -= 1
    plot(size=(150*MM, 150*d), grid=false, layout=grid(d, MM), dpi=300, xflip=true, link=:both,
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 16));

    titles = ["Random" "Optimal" "Eureka model" "AND/OR model"];
    #ylabels = [latexstring("p_\\textrm{in\\_tree}") latexstring("p_\\textrm{undo}") latexstring("p_\\textrm{same\\_car}") latexstring("d_\\textrm{tree}")];
    ylabels = ["Proportion in tree" "Proportion undos" "Proportion same car" "Depth in tree"];
    order = [2, 3, 5, 4]
    ytickss = [[0.4, 0.6, 0.8, 1.0], [0.0, 0.05, 0.1, 0.15], [0.0, 0.1, 0.2, 0.3], [2.0, 3.0, 4.0, 5.0]]
    ylimss = [:native, :native, (0, 0.3), (2, 5.3)]
    for i in 1:d
        for j in 1:MM
            xlabel = i == d ? "Distance to goal" : ""#latexstring("d_\\textrm{goal}") : ""
            ylabel = j == 1 ? ylabels[i] : ""
            title = i == 1 ? titles[j] : ""
            yticks = j == 1 ? ytickss[i] : nothing
            xticks = i == d ? [0, 5, 10, 15] : nothing
            ylims = ylimss[i]
            o = order[j]
            sp = (i-1)*4 + j
            plot!(X[i, :], y[i, 1, :], yerr=2*yerr[i, 1, :], sp=sp, c=:white, msw=1.4, label=nothing, xflip=true, linewidth=1.4, markershape=:none, ms=4, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks)
            if i == 1 && j == 4
                plot!(X[i, :], y[i, o, :], ribbon=2*yerr[i, o, :], sp=sp, label=nothing, c=palette(:default)[o-1], xlabel=xlabel, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            else
                plot!(X[i, :], y[i, o, :], ribbon=2*yerr[i, o, :], sp=sp, label=nothing, c=palette(:default)[o-1], l=nothing, xlabel=xlabel, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            end
        end
    end
    display(plot!())
end

function summary_stats2_plot(y, yerr)
    d, MM, bins = size(y)
    X = 1:size(y)[end]
    l = @layout [a{0.01h}; grid(1, 2)];
    plot(size=(500, 250), grid=false, layout=l, dpi=300, xflip=false, #size=(750, 250)
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=4Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out)

    labels = ["Subjects" "Random" "Optimal"];
    for i in 1:(MM-3)
        labels = hcat(labels, "Model "*string(i))
    end
    xlabels = ["Depth in tree" "Ranked depth"]#[latexstring("d_\\textrm{tree}") "Ranked "*latexstring("d_\\textrm{tree}")]
    plot!(zeros(Int, MM)', c=[:black palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=10, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top)
    for i in 1:d
        for j in 1:MM
            if j == 3
                continue
            end
            if j == 1
                plot!(X, y[i, j, :], yerr=2*yerr[i, j, :], sp=i+1, c=:white, msw=2, label=nothing, linewidth=2, markershape=:none, ms=6, xlabel=xlabels[i])
            else
                plot!(X, y[i, j, :], ribbon=2*yerr[i, j, :], sp=i+1, label=nothing, c=palette(:default)[j-1], l=nothing)
            end
        end
    end
    display(plot!())
end

function summary_stats2_1_plot(y, yerr)
    d, MM, bins = size(y)
    MM -= 1
    Xs = [collect(1:bins)[.!(isnan.(y[1, 1, :]))], collect(1:bins)[.!(isnan.(y[2, 1, :]))]]
    plot(size=(150*MM, 150*d), grid=false, layout=grid(d, MM), dpi=300, xflip=false, link=:y,
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(10), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=2Plots.mm, left_margin=2Plots.mm, 
        fontfamily="helvetica", tick_direction=:out);

    titles = ["Random" "Optimal" "Eureka model" "AND/OR model"];
    xlabels = ["Depth in tree" "Ranked depth"]#[latexstring("d_\\textrm{tree}") "Ranked "*latexstring("d_\\textrm{tree}")]
    order = [2, 3, 5, 4]
    ytickss = [[0.0, 0.1, 0.2], [0.0, 0.2, 0.4, 0.6]]
    xtickss = [[2, 4, 6, 8, 10], [1, 2, 3, 4, 5, 6, 7, 8]]
    for i in 1:d
        for j in 1:MM
            X = Xs[i]
            xlabel = xlabels[i]
            ylabel = j == 1 ? "Proportion" : ""
            title = i == 1 ? titles[j] : ""
            yticks = j == 1 ? ytickss[i] : nothing
            xticks = xtickss[i]
            #ylims = ylimss[i]
            o = order[j]
            sp = (i-1)*4 + j
            plot!(X, y[i, 1, X], yerr=2*yerr[i, 1, X], sp=sp, c=:white, msw=1.4, label=nothing, linewidth=1.4, markershape=:none, ms=4, xlabel=xlabel, title=title, ylabel=ylabel, xticks=xticks, yticks=yticks)
            plot!(X, y[i, o, X], ribbon=2*yerr[i, o, X], sp=sp, label=nothing, c=palette(:default)[o-1], l=nothing)
        end
    end
    display(plot!())
end

function summary_stats3_plot(X, Xerr, y, yerr)
    l = @layout [a{0.01h}; grid(1, 4)];
    plot(size=(800, 250), grid=false, layout=l, dpi=300, xflip=true, link=:both,
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=5Plots.mm, left_margin=6Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 15), ylim=(0, 1), yticks=nothing)

    labels = ["Worse" "Same" "Better"]
    plot!([0 0 0], c=[palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=10, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top);
    titles = ["Subjects" "Random" "Eureka model" "AND/OR model"]
    xlabels = ["Distance to goal" "Distance to goal" "Distance to goal" "Distance to goal"]
    ylabels = ["Proportion" "" "" ""]#latexstring("p")
    yticks = [[0, 1], [], [], []]
    rows = [1, 2, 5, 4]
    for i in eachindex(rows)
        row = rows[i]
        areaplot!(X[1, :] .- 1, [y[1, row, :] + y[2, row, :] + y[3, row, :], y[2, row, :] + y[3, row, :], y[3, row, :]], sp=i+1, xflip=true, label=nothing, xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i], yticks=yticks[i])    
    end
    display(plot!())
end

function summary_stats4_plot(y, yerr)
    X = 1:size(y)[end]
    plot(size=(300, 300), grid=false, dpi=300, xflip=false, #size=(750, 250)
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=-4Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, background_color_legend=nothing, foreground_color_legend=nothing)

    xlabels = ["Distance to goal"]#latexstring("d_\\textrm{goal}")
    all_ds=collect(values(d_goals))
    all_ds[all_ds .> 20] .= 20
    histogram!(all_ds[all_ds .> 0], label="All states", normalize=true, bins=20, c=palette(:default)[1], alpha=0.7)
    hist_opt = zeros(Int, 20)
    for prb in prbs
        for i in 1:(parse(Int, split(prb, "_")[end])-2)
            hist_opt[i] += 1
        end
    end
    plot!(collect(1:20), hist_opt / sum(hist_opt), c=:red, label="Optimal", linetype=:steppost, linewidth=2)
    plot!(X .+ 0.5, y[1, 1, :], yerr=2*yerr[1, 1, :], c=RGBA(1,1,1,0), msw=2, label=nothing, linewidth=2, markershape=:none, ms=4, xlabel=xlabels[1], xticks=([1.5, 5.5, 10.5, 15.5, 20.5], ["1", "5", "10", "15", "20+"]), xlim=(0, 21), ylabel="Proportion")
    display(plot!([],[],c=:black, label="Subjects"))
end

function model_comparison_plot(fitnesses; iters=1000)
    names = ["AND/OR tree", "AND/OR tree lapse", "Eureka", "Optimal lapse", "Random"];
    Nps = [1, 1, 2, 1, 0];
    Nm = length(fitnesses)
    means = zeros(Nm)
    CIs = zeros(Nm, 2)
    for i in 1:Nm
        Np = Nps[i]
        fitness = fitnesses[i]
        total = zeros(iters)
        M = length(fitness)
        for k in 1:iters
            subsampled_idxs = rand(1:M, M)
            # AIC
            #total[k] = sum(2*(Np .+ fitness[subsampled_idxs]) - 2*(Nps[1] .+ fitnesses[1][subsampled_idxs]))
            # BIC
            #total[k] = sum((Np*log(sum(length.(fitness))) .+ 2*fitness[subsampled_idxs]) - (Nps[1]*log(sum(length.(fitnesses[1]))) .+ 2*fitnesses[1][subsampled_idxs]))
            # LL
            total[k] = sum(fitness[subsampled_idxs] - fitnesses[1][subsampled_idxs])
        end
        m = mean(total)
        means[i] = m
        CIs[i, :] = [m - quantile(total, 0.025), quantile(total, 0.975) - m]
    end
    l = @layout [grid(1, 2, widths=(0.47, 0.53)); a{0.001h}];
    plot(size=(600, 300), layout=l, grid=false, dpi=300, xflip=false, link=:both,
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, background_color_legend=nothing, foreground_color_legend=nothing)

    bar!(names, means, yerr=(CIs[:, 1], CIs[:, 2]), sp=1, xflip=true, label=nothing, xlim=(0, Nm), ylim=(0, 6500), bar_width=0.8, permute=(:x, :y), yticks=([0, 2000, 4000, 6000], [0, 2, 4, 6]))
    bar!(names, means, yerr=(CIs[:, 1], CIs[:, 2]), sp=2, xflip=true, label=nothing, xlim=(0, Nm), ylim=(17000, 24000), bar_width=0.8, permute=(:x, :y), xticks=[], yticks=([18000, 20000, 22000, 24000], [18, 20, 22, 24]), xaxis=false)
    plot!(title=latexstring("\\Delta")*"NLL (x1000)", showaxis = false, sp=3, bottom_margin=-10Plots.mm)
    display(plot!())
end

function difficulty_d_plot(ds)
    l = @layout [grid(1, 4)];#a{0.01h}; 
    plot(size=(1000, 250), grid=false, layout=l, dpi=300, xflip=false, link=:both,
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=8Plots.mm, left_margin=6Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xticks=([1, 5, 10, 15, 20], ["1", "5", "10", "15", "20+"]))

    titles = ["L = 5", "L = 9", "L = 12", "L = 14"]
    ytickss = [[0, 0.1, 0.2], [], [], []]
    diffs = [5, 9, 12, 14]
    ylabels = ["Proportion" "" "" ""]
    xlabels = ["Distance to goal", "Distance to goal", "Distance to goal", "Distance to goal"]
    d, bins, M = size(ds)
    for i in 1:d
        ds_diff = ds[i, :, :]
        incl = .!(isnan.(ds_diff[1, :]))
        means = mean(ds_diff[:, incl], dims=2)[:]
        sems = 2*std(ds_diff[:, incl], dims=2)[:] / sqrt(sum(incl))
        bar!(means, yerr=sems, sp=i, linecolor=palette(:default)[1], markerstrokecolor=palette(:default)[1], linewidth=1, markersize=0, label=nothing, title=titles[i], ylabel=ylabels[i], fillcolor=:transparent, xlabel=xlabels[i], yticks=ytickss[i])
        plot!([diffs[i], diffs[i]], [0, 1/diffs[i]], sp=i, label=nothing, c=:red, linestyle=:dash)
        plot!([0.5, diffs[i]], [1/diffs[i], 1/diffs[i]], sp=i, label=nothing, c=:red, linestyle=:dash)
    end
    display(plot!())
end
difficulty_d_plot(ds)

# data
X_1, Xerr_1, y_1, yerr_1 = plot1_data();
summary_stats1_1_plot(X_1, Xerr_1, y_1, yerr_1)

y_2, yerr_2 = plot2_data();
summary_stats2_1_plot(y_2, yerr_2)

X_3, Xerr_3, y_3, yerr_3 = plot3_data();
summary_stats3_1_plot(X_3, Xerr_3, y_3, yerr_3)

y_4, yerr_4 = plot4_data();
summary_stats4_plot(y_4, yerr_4)

fitnesses = [cv_nll_gamma_only, cv_nll_gamma_0, cv_nll_eureka, cv_nll_opt_rand, cv_nll_rand];
model_comparison_plot(fitnesses)

histogram(params_gamma_only[:, 1], label=nothing, grid=false, xlabel=latexstring("\\gamma"), ylabel="Subjects")

ds = plot4_1_data()
difficulty_d_plot(ds)


