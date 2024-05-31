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
        heatmap(arr, c = cmap, legend = false, yflip = true, xmirror=true, framestyle = :box, size=(100, 100), dpi=300, xticks=[], yticks=[], fontfamily="helvetica", fontsize=11)
        vline!(1.5:5.5, c=:black, linewidth=0.2)
        hline!(1.5:5.5, c=:black, linewidth=0.2)
        for c in sort(unique(arr))[2:end]
            idxs = findall(x->x==c, arr)
            l = length(idxs)
            s = sum(idxs)
            annotate!(s[2]/l, s[1]/l, text(c, color=:white, pointsize=8, family="helvetica"))
        end
    else
        heatmap!(arr, c = cmap, sp=sp, legend = nothing, yflip = true, xmirror=true, framestyle = :box, dpi=300)
        vline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2, legend=nothing)
        hline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2, legend=nothing)
        for c in sort(unique(arr))[2:end]
            idxs = findall(x->x==c, arr)
            l = length(idxs)
            s = sum(idxs)
            annotate!(s[2]/l, s[1]/l, sp=sp, text(c, :white, 16, "helvetica"), legend=nothing, fontfamily="helvetica")
        end
    end
    plot!()
end

function summary_stats1_plot(X, Xerr, y, yerr)
    d, MM, bins = size(y)
    l = @layout [a{0.1h}; grid(2, 2)];
    plot(size=(500, 350), grid=false, layout=l, dpi=300, xflip=true, #size=(750, 250)
        legendfont=font(10, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(10, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, link=:x, xticks=[], xlim=(0, 15.5));

    labels = ["Subjects" "Random" "Eureka model" "AND/OR model"];
    ylabels = ["Proportion\nin tree" "Proportion\nundos" "Proportion\nsame car" "Depth in tree"]
    yticks = [[0.4, 0.6, 0.8, 1.0], [0, 0.04, 0.08, 0.12], [0.0, 0.1, 0.2], [2, 3, 4, 5]]
    ylim = [(0.25, 1.0), (-0.005, 0.14), (0.0, 0.25), (2, 5.1)]
    # for i in 1:(MM-3)
    #     labels = hcat(labels, "Model "*string(i))
    # end
    plot!(zeros(Int, MM-1)', c=[:black palette(:default)[1] palette(:default)[4] palette(:default)[3]], labels=labels, legend_columns=2, linewidth=10, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top);
    for i in 1:d
        for j in 1:MM
            if j == 3
                continue
            end
            if j == 1
                plot!(X[i, :], y[i, j, :], yerr=2*yerr[i, j, :], sp=i+1, c=:white, msw=2, label=nothing, xflip=true, linewidth=2, markershape=:none, ms=6, ylabel=ylabels[i], yticks=yticks[i], ylim=ylim[i])
            else
                plot!(X[i, :], y[i, j, :], ribbon=2*yerr[i, j, :], sp=i+1, label=nothing, c=palette(:default)[j-1], l=nothing)
            end
        end
    end
    plot!(xlabel="Distance to goal", sp=4, xticks=[0, 5, 10, 15])
    plot!(xlabel="Distance to goal", sp=5, xticks=[0, 5, 10, 15])
    display(plot!())
end
#summary_stats1_plot(X_1, Xerr_1, y_1, yerr_1)

function summary_stats1_1_plot(X, Xerr, y, yerr)
    d, MM, bins = size(y)
    MM -= 2#1
    l = @layout [grid(d, MM); a{0.001h}];
    plot(size=(744, 700), grid=false, layout=l, dpi=300, xflip=true, link=:both,
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=8Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 16));

    #titles = ["Random" "Optimal" "Hill climbing" "Forward" "Eureka" "AND/OR"];
    titles = ["Random" "Optimal\nrandom" "AND-OR\n(no same car)" "AND-OR\n("*latexstring("\\gamma=0")*")" "AND-OR"];
    #ylabels = [latexstring("p_\\textrm{in\\_tree}") latexstring("p_\\textrm{undo}") latexstring("p_\\textrm{same\\_car}") latexstring("d_\\textrm{tree}")];
    ylabels = ["Proportion\nin tree" "Proportion\nundos" "Proportion\nsame car" "Depth in tree\n"];
    order = [2, 7, 6, 5, 4]
    #order = [2, 3, 7, 6, 5, 4]
    ytickss = [[0.4, 0.6, 0.8, 1.0], ([0.0, 0.1, 0.2], ["0", "0.1", "0.2"]), ([0.0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"]), [2.0, 3.0, 4.0, 5.0]]
    ylimss = [(-Inf, 1.0), (-Inf, 0.2), (0, 0.3), (2, 5.3)]
    for i in 1:d
        for j in 1:MM
            xlabel = i == d ? "Distance\nto goal" : ""#latexstring("d_\\textrm{goal}") : ""
            ylabel = j == 1 ? ylabels[i] : ""
            title = i == 1 ? titles[j] : ""
            yticks = j == 1 ? ytickss[i] : nothing
            xticks = i == d ? [0, 5, 10, 15] : [0, 5, 10, 15]#([0, 5, 10, 15], ["", "", "", ""])
            ylims = ylimss[i]
            o = order[j]
            sp = (i-1)*MM + j
            plot!(X[i, :], y[i, 1, :], yerr=2*yerr[i, 1, :], sp=sp, c=:white, msw=1.4, label=nothing, xflip=true, linewidth=1, markershape=:none, ms=4, ylabel=ylabel, xticks=xticks, yticks=yticks)
            if i == 1 && j == 2
                plot!(X[i, :], y[i, o, :], ribbon=2*yerr[i, o, :], sp=sp, label=nothing, c=palette(:default)[o-1], ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            else
                plot!(X[i, :], y[i, o, :], ribbon=2*yerr[i, o, :], sp=sp, label=nothing, c=palette(:default)[o-1], l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
                plot!(X[i, :], y[i, o, :], ribbon=2*yerr[i, o, :], sp=sp, label=nothing, c=palette(:default)[o-1], l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            end
        end
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=d*MM + 1, top_margin=-18Plots.mm, bottom_margin=0Plots.mm)
    display(plot!())
end
# models = [gamma_only_model, gamma_0_model, gamma_no_same_model, opt_rand_model]
# models = [gamma_only_model, eureka_model, opt_rand_model, means_end_model]
summary_stats1_1_plot(X_1, Xerr_1, y_1, yerr_1)

function summary_stats2_plot(y, yerr)
    d, MM, bins = size(y)
    X = 1:size(y)[end]
    l = @layout [a{0.01h}; grid(2, 1)];
    plot(size=(300, 350), grid=false, layout=l, dpi=300, xflip=false, #size=(750, 250)
        legendfont=font(10, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(10, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=0Plots.mm, left_margin=2Plots.mm, 
        fontfamily="helvetica", tick_direction=:out)

    labels = ["Subjects" "Random" "Eureka model" "AND/OR model"];
    xlabels = ["Depth in tree" "Ranked depth"]#[latexstring("d_\\textrm{tree}") "Ranked "*latexstring("d_\\textrm{tree}")]
    yticks = [[0, 0.1, 0.2], [0, 0.2, 0.4]]
    xticks = [[2, 4, 6, 8, 10], [1, 2, 3, 4, 5, 6, 7, 8]]
    plot!(zeros(Int, MM-1)', c=[:black palette(:default)[1] palette(:default)[4] palette(:default)[3]], labels=labels, legend_columns=2, linewidth=10, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:topright)
    for i in 1:d
        for j in 1:MM
            if j == 3
                continue
            end
            if j == 1
                plot!(X, y[i, j, :], yerr=2*yerr[i, j, :], sp=i+1, c=:white, msw=2, label=nothing, linewidth=2, markershape=:none, ms=6, xlabel=xlabels[i], yticks=yticks[i], xticks=xticks[i])
            else
                plot!(X, y[i, j, :], ribbon=2*yerr[i, j, :], sp=i+1, label=nothing, c=palette(:default)[j-1], l=nothing)
            end
        end
    end
    plot!(ylabel="Proportion", sp=2, xlim=(1.5, 11.5), bottom_margin=-5Plots.mm)
    plot!(ylabel="Proportion", sp=3, bottom_margin=-3Plots.mm)
    plot!(sp=3, xlim=(0.5, 8.5))
    display(plot!())
end
#summary_stats2_plot(y_2, yerr_2)

function summary_stats2_1_plot(y, yerr)
    d, MM, bins = size(y)
    MM -= 2#1
    Xs = [collect(1:bins)[.!(isnan.(y[1, 1, :]))], collect(1:bins)[.!(isnan.(y[2, 1, :]))]]
    l = @layout [grid(1, MM); a{0.001h}; grid(1, MM); a{0.001h}];
    plot(size=(744, 450), grid=false, layout=l, dpi=300, xflip=false, link=:y,
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"), 
        right_margin=0Plots.mm, top_margin=6Plots.mm, bottom_margin=4Plots.mm, left_margin=2Plots.mm, 
        fontfamily="helvetica", tick_direction=:out);

    #titles = ["Random" "Optimal" "Hill climbing" "Forward" "Eureka" "AND/OR"];
    titles = ["Random" "Optimal\nrandom" "AND-OR\n(no same car)" "AND-OR\n("*latexstring("\\gamma=0")*")" "AND-OR"];
    xlabels = ["Depth" "Depth rank"]#[latexstring("d_\\textrm{tree}") "Ranked "*latexstring("d_\\textrm{tree}")]
    #order = [2, 3, 7, 6, 5, 4]
    order = [2, 7, 6, 5, 4]
    ytickss = [([0.0, 0.1, 0.2], ["0", "0.1", "0.2"]), ([0.0, 0.2, 0.4], ["0", "0.2", "0.4"])]
    xtickss = [[2, 4, 6, 8, 10], [1, 3, 5, 7]]
    ii = 0
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
            sp = (i-1)*MM + j
            ii += 1
            bar!(X, y[i, 1, X], yerr=2*yerr[i, 1, X], sp=ii, c=:white, msw=1.4, label=nothing, linewidth=1.4, markershape=:none, ms=0, title=title, ylabel=ylabel, xticks=xticks, yticks=yticks, linecolor=:gray, markercolor=:gray)#, alpha=0.5)
            plot!(X, y[i, o, X], ribbon=2*yerr[i, o, X], sp=ii, label=nothing, c=palette(:default)[o-1], l=nothing)
            plot!(X, y[i, o, X], ribbon=2*yerr[i, o, X], sp=ii, label=nothing, c=palette(:default)[o-1], l=nothing)
            if i > 1
                plot!(X, y[i, o, X], ribbon=2*yerr[i, o, X], sp=ii, label=nothing, c=palette(:default)[o-1], l=nothing)
                plot!(X, y[i, o, X], ribbon=2*yerr[i, o, X], sp=ii, label=nothing, c=palette(:default)[o-1], l=nothing)
            end
        end
        ii += 1
        plot!(xlabel=xlabels[Int(1+Int(ii > 10))], showaxis=false, grid=false, sp=ii, top_margin=-12Plots.mm)
    end
    display(plot!())
end
summary_stats2_1_plot(y_2, yerr_2)

function summary_stats3_plot(X, Xerr, y, yerr)
    l = @layout [a{0.001h}; grid(1, 6); a{0.001h}];
    plot(size=(744, 350), grid=false, layout=l, dpi=300, xflip=true, link=:both,
        legendfont=font(14, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(14, "helvetica"), 
        guidefont=font(14, "helvetica"), 
        right_margin=0Plots.mm, top_margin=4Plots.mm, bottom_margin=6Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 15), ylim=(0, 1), yticks=nothing)

    labels = ["Worse" "Same" "Better"]
    bar!([0 0 0], c=[palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=0, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top, top_margin=-2Plots.mm);
    #titles = ["Subjects\n" "AND/OR\n" "Eureka\n" "Random\n" "Means-\nends" "Forward\nsearch"];
    #titles = ["Subjects" "Random" "Hill climbing" "Forward" "Eureka" "AND/OR"];
    titles = ["Subjects" "Random" "Optimal\nrandom" "AND-OR\n(no same car)" "AND-OR\n("*latexstring("\\gamma=0")*")" "AND-OR"];
    xlabels = ["" "" "" "" "" ""]
    ylabels = ["Proportion" "" "" "" "" ""]#latexstring("p")
    yticks = [([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"]), ([0, 0.2, 0.4, 0.6, 0.8, 1], ["", "", ""]), ([0, 0.2, 0.4, 0.6, 0.8, 1], ["", "", ""]), ([0, 0.2, 0.4, 0.6, 0.8, 1], ["", "", ""]), ([0, 0.2, 0.4, 0.6, 0.8, 1], ["", "", ""]), ([0, 0.2, 0.4, 0.6, 0.8, 1], ["", "", ""])]
    xticks = [[0, 5, 10, 15], [0, 5, 10, 15], [0, 5, 10, 15], [0, 5, 10, 15], [0, 5, 10, 15], [0, 5, 10, 15]]
    #rows = [1, 4, 5, 2, 7, 6]
    rows = [1, 2, 7, 6, 5, 4]
    for i in eachindex(rows)
        row = rows[i]
        areaplot!(X[1, :] .- 1, [y[1, row, :] + y[2, row, :] + y[3, row, :], y[2, row, :] + y[3, row, :], y[3, row, :]], sp=i+1, xflip=true, label=nothing, xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i], yticks=yticks[i], xticks=xticks[i])    
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=8, top_margin=-14Plots.mm)
    display(plot!())
end
summary_stats3_plot(X_3, Xerr_3, y_3, yerr_3)

function summary_stats4_plot(y, yerr)
    X = 1:size(y)[end]
    plot(size=(300, 300), grid=false, dpi=300, xflip=false, #size=(750, 250)
        legendfont=font(10, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(10, "helvetica"), 
        guidefont=font(12, "helvetica"), 
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
    #names = ["AND/OR\ntree", "Eureka", "Forward\nsearch", "Means\nends", "Random"];
    #names = ["AND-OR tree", "AND-OR tree ("*latexstring("\\gamma")*"=0)", "Eureka", "Optimal-random", "Forward search", "Hill climbing", "Random"];
    names = ["AND-OR tree", "AND-OR tree ("*latexstring("\\gamma")*"=0)", "AND-OR tree (no same car)", "Eureka", "Optimal-random", "Forward search", "Hill climbing", "Random"];
    #Nps = [1, 1, 2, 1, 2, 4, 0];
    Nps = [1, 1, 1, 2, 1, 2, 4, 0];
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
    #l = @layout [grid(1, 2, widths=(0.47, 0.53)); a{0.001h}];
    #l = @layout [a{0.001w} [grid(2, 1, heights=(0.47, 0.53))]];
    plot(size=(372, 200), grid=false, dpi=300,
        legendfont=font(9, "helvetica"), 
        xtickfont=font(8, "helvetica"), 
        ytickfont=font(8, "helvetica"), 
        titlefont=font(9, "helvetica"), 
        guidefont=font(9, "helvetica"), 
        right_margin=1Plots.mm, top_margin=0Plots.mm, bottom_margin=2Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, background_color_legend=nothing, foreground_color_legend=nothing)

    #bar!(names, means, yerr=(CIs[:, 1], CIs[:, 2]), sp=3, label=nothing, xlim=(0, Nm), ylim=(0, 5500), bar_width=0.8, yticks=([0, 2000, 4000], [0, 2, 4]), markersize=12, linewidth=2)
    #bar!(names, means, yerr=(CIs[:, 1], CIs[:, 2]), sp=2, label=nothing, xlim=(0, Nm), ylim=(15000, 24000), bar_width=0.8, xticks=[], yticks=([16000, 20000, 24000], [16, 20, 24]), xaxis=false, markersize=12, linewidth=2, linestrokecolor=:none)
    #plot!(ylabel=latexstring("\\Delta")*"NLL (x1000)", showaxis = false, sp=1, right_margin=-10Plots.mm)

    bar!(names, means, yerr=(CIs[:, 1], CIs[:, 2]), xflip=true, label=nothing, xlim=(0, Nm), ylim=(0, 24000), bar_width=0.8, permute=(:x, :y), yticks=([0, 4, 8, 12, 16, 20, 24]*1000, [0, 4, 8, 12, 16, 20, 24]), markersize=5, linewidth=1.4, ylabel="\n"*latexstring("\\Delta")*"NLL (x1000)", c=:transparent)
    #bar!(names, means, yerr=(CIs[:, 1], CIs[:, 2]), sp=2, xflip=true, label=nothing, xlim=(0, Nm), ylim=(7800, 24000), bar_width=0.8, permute=(:x, :y), xticks=[], yticks=([8000, 12000, 16000, 20000, 24000], [8, 12, 16, 20, 24]), xaxis=false, markersize=5, linewidth=1.4)
    #plot!(title=latexstring("\\Delta")*"NLL (x1000)", sp=3, showaxis = false, bottom_margin=-10Plots.mm)
    display(plot!())
end
#fitnesses = [cv_nll_gamma_only, cv_nll_eureka, cv_nll_rand];
#fitnesses = [cv_nll_gamma_only, cv_nll_gamma_0, cv_nll_eureka, cv_nll_opt_rand, sum(cv_nll_forward, dims=2)[:], cv_nll_means_ends, cv_nll_rand];
fitnesses = [cv_nll_gamma_only, cv_nll_gamma_0, cv_nll_gamma_no_same, cv_nll_eureka, cv_nll_opt_rand, sum(cv_nll_forward, dims=2)[:], cv_nll_means_ends, cv_nll_rand];
#fitnesses = [fitness_gamma_only, fitness_gamma_0, fitness_eureka, fitness_opt_rand, fitness_opt_rand, fitness_means_ends, fitness_rand]
model_comparison_plot(fitnesses)

ls = []
for _ in 1:100
    c = rand(1:42, 42)
    push!(ls, sum(a[c] - b[c]))
end

function difficulty_d_plot(ds)
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
    xlabels = ["Distance to goal", "Distance to goal", "Distance to goal", "Distance to goal"]
    d, bins, M = size(ds)
    for i in 1:d
        ds_diff = ds[i, :, :]
        incl = .!(isnan.(ds_diff[1, :]))
        means = mean(ds_diff[:, incl], dims=2)[:]
        sems = 2*std(ds_diff[:, incl], dims=2)[:] / sqrt(sum(incl))
        bar!(means, yerr=sems, sp=i, linecolor=:gray, markerstrokecolor=:gray, linewidth=1, markersize=0, label=nothing, title=titles[i], ylabel=ylabels[i], fillcolor=:transparent, xlabel="", yticks=ytickss[i])
        plot!([diffs[i]+1, diffs[i]+1], [0, 1/(diffs[i]+1)], sp=i, label=i < 4 ? nothing : "optimal", c=:red, linestyle=:dash)
        plot!([0.5, diffs[i]+1], [1/(diffs[i]+1), 1/(diffs[i]+1)], sp=i, label=nothing, c=:red, linestyle=:dash)
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=5, top_margin=-10Plots.mm, bottom_margin=2Plots.mm)
    display(plot!())
end
difficulty_d_plot(ds)

# data
X_1 = load("fig_data/X_1.jld2")["X_1"]
Xerr_1 = load("fig_data/Xerr_1.jld2")["Xerr_1"]
y_1 = load("fig_data/y_1.jld2")["y_1"]
yerr_1 = load("fig_data/yerr_1.jld2")["yerr_1"]
X_1, Xerr_1, y_1, yerr_1 = plot1_data();
summary_stats1_1_plot(X_1, Xerr_1, y_1, yerr_1)

y_2 = load("fig_data/y_2.jld2")["y_2"]
yerr_2 = load("fig_data/yerr_2.jld2")["yerr_2"]
y_2, yerr_2 = plot2_data();
summary_stats2_1_plot(y_2, yerr_2)

X_3 = load("fig_data/X_3.jld2")["X_3"]
Xerr_3 = load("fig_data/Xerr_3.jld2")["Xerr_3"]
y_3 = load("fig_data/y_3.jld2")["y_3"]
yerr_3 = load("fig_data/yerr_3.jld2")["yerr_3"]
X_3, Xerr_3, y_3, yerr_3 = plot3_data();
summary_stats3_plot(X_3, Xerr_3, y_3, yerr_3)
# y_34 = y_3[:, 1, :];
# X_34 = X_3;
idxs = X_31[3, :] .< 6
plot(X_31[3, idxs, ], y_31[3, idxs], grid=false, label="L=5", xflip=true)
idxs = X_32[3, :] .< 6
plot!(X_32[3, idxs, ], y_32[3, idxs], grid=false, label="L=9", xflip=true)
idxs = X_33[3, :] .< 6
plot!(X_33[3, idxs, ], y_33[3, idxs], grid=false, label="L=12", xflip=true)
idxs = X_34[3, :] .< 6
plot!(X_34[3, idxs, ], y_34[3, idxs], grid=false, label="L=14", xflip=true, legend=:topleft)
plot!(xlabel="Distance to goal", ylabel="Proportion of optimal moves")

y_4, yerr_4 = plot4_data();
summary_stats4_plot(y_4, yerr_4)

histogram(params_gamma_only[:, 1], label=nothing, grid=false, xlabel=latexstring("\\gamma"), ylabel="Subjects")

ds = plot4_1_data()
difficulty_d_plot(ds)

difficulty = x -> x == 7 ? 1 : x == 11 ? 2 : x == 14 ? 3 : x == 16 ? 4 : 0
subj = subjs[9]
DD = zeros(4, 10, 42)
TS = zeros(4, 10, 42)
for (m, subj) in enumerate(subjs)
    dd = [[[] for _ in 1:10] for _ in 1:4]
    ts = [[[] for _ in 1:10] for _ in 1:4]
    for i in 1:length(states[subj])-1
        d = difficulty(parse(Int, split(problems[subj][i], "_")[2]))
        move = tree_datas[subj][4][i]
        all_moves = tree_datas[subj][3][i]
        subj_move_idx = findfirst(x->x==move, all_moves)
        next_s = neighs[subj][i][subj_move_idx]
        d_g = d_goals[states[subj][i]]
        d_gp = d_goals[next_s]
        push!(dd[d][d_g > 10 ? 10 : d_g], d_gp < d_g)
        push!(ts[d][d_g > 10 ? 10 : d_g], times[subj][i])
    end
    for i in 1:4
        idxs = .!(isempty.(dd[i]))
        DD[i, idxs, m] = mean.(dd[i][idxs])
        TS[i, idxs, m] = mean.(ts[i][idxs])
    end
end
plot(grid=false, xflip=true, xlabel="Distance to goal", ylabel="Proportion of optimal moves", title="All subjects")
labels = ["L = 5", "L = 9", "L = 12", "L = 14"]
for i in 1:4
    plot!(collect(1:10)[.!(isempty.(dd[i]))], mean.(dd[i][.!(isempty.(dd[i]))]), label=labels[i])
end
for i in 1:4
    plot!(1:10, mean(DD[i, :, :], dims=2), yerr=std(DD[i, :, :], dims=2) ./ sqrt(42), label=labels[i])
end
plot!(legend=:topleft)

plot(grid=false, xflip=true, xlabel="Distance to goal", ylabel="RT (ms)", title="All subjects")
labels = ["L = 5", "L = 9", "L = 12", "L = 14"]
for i in 1:4
    plot!(collect(1:10)[.!(isempty.(ts[i]))], mean.(ts[i][.!(isempty.(ts[i]))]), label=labels[i])
end
for i in 1:4
    plot!(1:10, mean(TS[i, :, :], dims=2), yerr=std(TS[i, :, :], dims=2) ./ sqrt(42), label=labels[i])
end
plot!(legend=:topleft)

cv_nll_forward = zeros(42, 5)
for i in 1:42
    for j in 1:5
        n = (i-1)*5 + (j-1)
        if n == 52
            continue
        end
        cv_nll_forward += load("cv_fitness/cv_fitness$(n).jld2")["cv_fitness"]
    end
end
cv_nll_forward[11, 2:3] .= mean(cv_nll_forward[11, [1, 4, 5]])


times = Dict{String, Vector{Int}}()
for subj in subjs
    ls = []
    for prb in keys(all_subj_states[subj])
        for i in eachindex(all_subj_states[subj][prb])
            if all_subj_moves[subj][prb][i] == (-1, 0)
                continue
            end
            push!(ls, all_subj_times[subj][prb][i] - all_subj_times[subj][prb][i-1])
        end
    end
    times[subj] = ls
end
