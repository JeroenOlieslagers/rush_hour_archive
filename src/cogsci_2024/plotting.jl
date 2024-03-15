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

function draw_ao_tree(AO, board; highlight_ORs=[], outside_moves=[])
    graph = """digraph{graph [pad="0.2",nodesep="0.15",ranksep="0.3"];layout="dot";"""
    drawn_and = []
    drawn_or = []
    AND, OR = AO
    for move in outside_moves
        graph *= """ "outside_and"[fontname="Helvetica",fixedsize=shape,style=filled,fillcolor=white,width=0.6,margin=0,label="$(and[1][1])",fontsize=20,shape="circle"];"""
        graph *= """ "outside_and"->"$(or)"[constraint="true"];"""
        graph *= """ "$(or)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="#E05B53",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
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
            graph *= """ "$(or)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(or in highlight_ORs ? "#9CB4EE" : OR[or][1][1] == (0, (0,)) ? "#86D584" : "#DBDBDB")",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
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

    labels = ["Subjects" "Chance" "Optimal"];
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

    labels = ["Subjects" "Chance" "Optimal"];
    for i in 1:(MM-3)
        labels = hcat(labels, "Model "*string(i))
    end
    xlabels = [latexstring("d_\\textrm{tree}") "Ranked "*latexstring("d_\\textrm{tree}")]
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

function summary_stats3_plot(X, Xerr, y, yerr)
    l = @layout [a{0.01h}; grid(2, 2)];
    plot(size=(750, 750), grid=false, layout=l, dpi=300, xflip=true,#(750, 250)
        legendfont=font(10), 
        xtickfont=font(10), 
        ytickfont=font(10), 
        titlefont=font(10), 
        guidefont=font(14), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=5Plots.mm, left_margin=6Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 15), ylim=(0, 1), yticks=nothing)

    labels = ["Worse" "Same" "Better"]
    plot!([0 0 0], c=[palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=10, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top);
    #titles = ["Subjects" "AND/OR model" "Chance"]
    titles = ["Subjects" "Chance" "AND/OR model" "Eureka model"]
    xlabels = [latexstring("d_\\textrm{goal}") latexstring("d_\\textrm{goal}") latexstring("d_\\textrm{goal}") latexstring("d_\\textrm{goal}")]
    ylabels = [latexstring("p") "" latexstring("p") ""]
    rows = [1, 2, 4, 5]
    for i in eachindex(rows)
        row = rows[i]
        areaplot!(X[1, :] .- 1, [y[1, row, :] + y[2, row, :] + y[3, row, :], y[2, row, :] + y[3, row, :], y[3, row, :]], sp=i+1, xflip=true, label=nothing, xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i])    
    end
    display(plot!())
end

# data
X, Xerr, y, yerr = plot1_data();
summary_stats1_plot(X, Xerr, y, yerr)

y, yerr = plot2_data();
summary_stats2_plot(y, yerr)

X, Xerr, y, yerr = plot3_data();
summary_stats3_plot(X, Xerr, y, yerr)