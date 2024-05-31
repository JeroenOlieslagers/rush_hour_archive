using Plots
using GraphViz

function create_move_icon(move, s)
    s_free, s_fixed = s
    car, m = move
    s = string(car)
    if s_fixed[car].dir == :x
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

function draw_board(s::s_type; sp=nothing, car_ind=[])
    board_arr = board_to_arr(s)
    if !isempty(car_ind)
        board_arr_d = get_board_arr(s)
        for c in sort(unique(board_arr))[2:end]
            board_arr[board_arr_d .== c] .= car_ind[c]
        end
    end
    cmap = [
        RGB(([255, 255, 255]./255)...),
        RGB(([202, 76, 60]./255)...),
        RGB(([185, 156, 105]./255)...), 
        RGB(([102, 152, 80]./255)...), 
        RGB(([80, 173, 202]./255)...), 
        RGB(([219, 130, 57]./255)...), 
        RGB(([81, 51, 154]./255)...), 
        RGB(([147, 190, 103]./255)...), 
        RGB(([126, 74, 51]./255)...), 
        RGB(([124, 124, 124]./255)...),
        RGB(([242, 217, 80]./255)...),
        RGB(([51, 51, 51]./255)...),
        RGB(([158, 156, 79]./255)...),
        RGB(([48, 83, 141]./255)...),
        RGB(([211, 128, 149]./255)...),
        RGB(([84, 255, 198]./255)...)
        ]
    if sp === nothing
        heatmap(board_arr, c = cmap[1:maximum(board_arr)+1], legend = false, yflip = true, xmirror=true, framestyle = :box, size=(200, 200))
        vline!(1.5:5.5, c=:black, linewidth=0.2)
        hline!(1.5:5.5, c=:black, linewidth=0.2)
    else
        heatmap!(board_arr, sp=sp, c = cmap[1:maximum(board_arr)+1], legend = false, yflip = true, xmirror=true, framestyle = :box)
        vline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2)
        hline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2)
    end
    for c in sort(unique(board_arr))[2:end]
        idxs = findall(x->x==c, board_arr)
        l = length(idxs)
        s = sum(idxs)
        annotate!(s[2]/l, s[1]/l, text(c, :white, 16))
    end
    plot!()
end

function draw_ao_tree(AND, OR, s; highlight_ANDs=[])
    graph = """digraph{graph [pad="0.2",nodesep="0.15",ranksep="0.3"];layout="dot";"""
    for and in keys(AND)
        graph *= """ "$(and)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(and in highlight_ANDs ? "#9CB4EE" : AND[and][1][2][1] == 0 ? "#86D584" : "#DBDBDB")",label="$(create_move_icon(and[2], s)[2:end])",height=.5,width=.5,fontsize=14];"""
        for or in AND[and]
            if or[2][1] != 0
                graph *= """ "$(and)"->"$(or)"[constraint="true"];"""
                if or ∉ keys(OR)
                    graph *= """ "$(or)"[fontname="Helvetica",fixedsize=shape,style=filled,fillcolor="#E05B53",width=0.6,margin=0,label="$(or[2][1])",fontsize=20,shape="circle"];"""
                end
            end
        end
    end
    for or in keys(OR)
        graph *= """ "$(or)"[fontname="Helvetica",fixedsize=shape,style=filled,fillcolor=white,width=0.6,margin=0,label="$(or[2][1])",fontsize=20,shape="circle"];"""
        for and in OR[or]
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
