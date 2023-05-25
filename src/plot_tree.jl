using GraphViz, FileIO, ImageIO

function draw_tree(draw_tree_nodes; highlight_node=nothing)
    tree = """graph{layout="dot";"""
    drawn = []
    or_counter = 100000
    child_drawn_counter = 1000000
    highlighted = false
    parents = collect(keys(draw_tree_nodes))
    for (n, parent) in enumerate(parents)
        if parent ∉ drawn
            tree *= """ "$(parent)"[label="$(parent[1])"];"""
            push!(drawn, parent)
        end
        children = draw_tree_nodes[parent]
        chds = zeros(Int, length(children))
        for (m, child) in enumerate(children)
            if child in drawn
                if child[1] == 'm'
                    tree *= """ "$(child_drawn_counter)"[fixedsize=shape;shape=diamond;label="$(child[3:end])"fillcolor="lime";style=filled;height=.5;width=.5;];"""
                else
                    tree *= """ "$(child_drawn_counter)"[label="$(child[child[1]=='v' ? 2 : 1])"fillcolor="red";style=filled;];"""
                end
                chds[m] = child_drawn_counter
                child_drawn_counter += 1
            elseif child[1] == 'm'
                if child == highlight_node
                    highlighted = true
                    tree *= """ "$(child)"[fixedsize=shape;shape=diamond;label="$(child[3:end])"fillcolor="magenta";style=filled;height=.5;width=.5;];"""
                else
                    tree *= """ "$(child)"[fixedsize=shape;shape=diamond;label="$(child[3:end])"fillcolor="lime";style=filled;height=.5;width=.5;];"""
                end
            elseif child[1] == 'v'#child ∉ parents
                tree *= """ "$(child)"[label="$(child[2])"fillcolor="red";style=filled;];"""
            elseif child[end] == '_'#child ∉ parents
                tree *= """ "$(child)"[label="$(child[1])"fillcolor="red";style=filled;];"""
            elseif n == length(draw_tree_nodes)
                tree *= """ "$(child)"[label="$(child[1])"fillcolor="red";style=filled;];"""
            else
                tree *= """ "$(child)"[label="$(child[1])"];"""
            end
            push!(drawn, child)
        end
        childss = split(parent, "_")[2:end]
        counter3 = 1
        for childs in childss
            if children[counter3][1] == 'm'
                if chds[counter3] > 0
                    tree *= """ "$(parent)"--"$(chds[counter3])";"""
                else
                    tree *= """ "$(parent)"--"$(children[counter3])";"""
                end
                or_counter += 1
                counter3 += 1
                continue
            end
            tree *= """ $(or_counter) [shape=diamond,style=filled,label="",height=.3,width=.3];"""
            tree *= """ "$(parent)"--"$(or_counter)";"""
            childs = split(childs, "")
            for chld in childs
                if chds[counter3] > 0
                    tree *= """ "$(or_counter)"--"$(chds[counter3])";"""
                else
                    tree *= """ "$(or_counter)"--"$(children[counter3])";"""
                end
                counter3 += 1
            end
            or_counter += 1
        end
    end
    if highlight_node !== nothing && !highlighted
        tree *= """ "$(or_counter+1)"[label="$(highlight_node[2])"];"""
        tree *= """ "$(highlight_node)"[fixedsize=shape;shape=diamond;label="$(highlight_node[3:end])"fillcolor="magenta";style=filled;height=.5;width=.5;];"""
        tree *= """ "$(or_counter+1)"--"$(highlight_node)";"""
    end
    tree *= "}"
    return GraphViz.Graph(tree)
end

function new_draw_tree(tree, board, visited; highlight_node=nothing)
    graph = """graph{graph [pad="0.2",nodesep="0.1",ranksep="0.2"];layout="dot";"""
    drawn = []
    for node in visited
        if node ∉ keys(tree)
            continue
        end
        push!(drawn, node)
        graph *= """ "$(node)"[fixedsize=shape,style=filled,fillcolor="$(isempty(tree[node]) ? "red" : "white")",width=0.6,margin=0,label="$(node[1])",fontsize=32,shape="circle"];"""
        for (or_node, s) in sort(collect(tree[node]), by=x->x[2])
            graph *= """ "$((node, or_node))" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(isempty(s) ? "lime" : "gray75")",label="$(create_move_icon(or_node, board)[2:end])",height=$(isempty(s) ? .5 : .3),width=$(isempty(s) ? .5 : .3),fontsize=$(isempty(s) ? 18 : 10)];"""
            graph *= """ "$(node)"--"$((node, or_node))";"""
            for grandchild in reverse(s)
                if grandchild in keys(tree) && grandchild ∉ drawn
                    graph *= """ "$((node, or_node))"--"$(grandchild)";"""
                else
                    graph *= """ "$((node, grandchild))"[shape="circle",label="$(grandchild[1])",style=filled,fillcolor="red"];"""
                    graph *= """ "$((node, or_node))"--"$((node, grandchild))";"""
                end
            end
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

board = load_data("prb72800_14")
make_move!(board, (2, -1))
make_move!(board, (6, -1))
make_move!(board, (7, -3))
make_move!(board, (8, -1))
make_move!(board, (3, 3))
make_move!(board, (5, 3))
make_move!(board, (8, 3))
make_move!(board, (1, 3))

board = load_data("prb26567_7")
make_move!(board, (9, 2))
make_move!(board, (7, 1))
make_move!(board, (6, -1))
make_move!(board, (1, -1))
make_move!(board, (2, 2))

board = load_data("prb55384_14")
make_move!(board, (3, 1))
make_move!(board, (9, 1))
make_move!(board, (8, 4))
make_move!(board, (9, -1))
make_move!(board, (2, -1))
make_move!(board, (3, -3))
make_move!(board, (4, -3))

board = load_data("prb23404_14")
make_move!(board, (3, -1))
make_move!(board, (1, 2))
make_move!(board, (5, 2))
make_move!(board, (2, 3))
make_move!(board, (9, 1))
make_move!(board, (4, -4))
make_move!(board, (9, -1))
make_move!(board, (2, 1))

board = load_data("prb15595_16")
make_move!(board, (6, -1))
make_move!(board, (5, 3))
make_move!(board, (4, 1))
make_move!(board, (7, -1))
make_move!(board, (2, 2))
make_move!(board, (9, 1))
make_move!(board, (3, -2))

board = load_data("prb44171_16")
make_move!(board, (5, -2))
make_move!(board, (8, 1))
make_move!(board, (6, -1))
make_move!(board, (3, 3))
make_move!(board, (5, 3))
make_move!(board, (6, 1))
make_move!(board, (7, 4))
make_move!(board, (6, -1))
make_move!(board, (4, -1))

board = load_data(prbs[1])

draw_board(get_board_arr(board))
tree, visited, actions, blockages, trials, parents, move_parents, h = new_and_or_tree(board);
g = new_draw_tree(tree, board, visited)

actions, trials, visited, timeline, draw_tree_nodes = and_or_tree(board);
g = draw_tree(draw_tree_nodes)

function draw_planning(timeline, board)
    # Get full board
    arr = get_board_arr(board)
    display(draw_board(arr))
    nums = sort(unique(arr))[2:end]
    visited = Node[]
    arrs = [(arr, [], :white, :all)]
    for (n, (node, move, next_car)) in enumerate(timeline)
        # AND node
        arrr = copy(arr)
        ls = nums[findall(x->x∉vcat(node.value, node.children...), nums)]
        #draw_board(arrr, alpha_map=ls)
        push!(arrs, (arrr, ls, :white, :and))
        # OR node
        make_move!(board, move)
        arrr = get_board_arr(board)
        ls = nums[findall(x->x∉vcat(node.value, next_car), nums)]
        #draw_board(arrr, alpha_map=ls)
        if next_car == move[1] || n == length(timeline)
            push!(arrs, (arrr, ls, :cyan, :or))
        elseif next_car != timeline[n+1][1].value
            undo_moves!(board, [move])
            arrr = get_board_arr(board)
            if node in visited
                mmove = timeline[findfirst(x->x==node, visited)][2]
                make_move!(board, mmove)
                arrr = get_board_arr(board)
                push!(arrs, (arrr, ls, :white, :or))
                undo_moves!(board, [mmove])
                arrr = copy(arr)
                nnode = timeline[findfirst(x->x==node, visited)+1][1]
                ls = nums[findall(x->x∉vcat(nnode.value, nnode.children...), nums)]
                make_move!(board, move)
                push!(arrs, (arrr, ls, :red, :and))
            else
                make_move!(board, move)
                push!(arrs, (arrr, ls, :red, :or))
            end
        else
            push!(arrs, (arrr, ls, :white, :or))
        end
        undo_moves!(board, [move])
        push!(visited, node)
    end
    anim = @animate for (n, (arr, ls, c, ao)) in enumerate(arrs)
        draw_board(arr, alpha_map=ls)
        title = "Iteration $(n)$(ao==:and ? "(AND)" : "(OR)")"
        if c == :cyan
            title *= "\n possible move!"
        elseif c == :red
            title *= "\n cycle detected..."
        else
            title *= "\n "
        end
        plot!(background_color_outside=c, title=title, size=(200, 250), plot_titlefontsize=4)
    end
    return anim
end

#anim = draw_planning(timeline, board)

#gif(anim, "and_or_plan.gif", fps = 0.5)