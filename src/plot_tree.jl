using GraphViz, FileIO, ImageIO

# function draw_tree(draw_tree_nodes; highlight_node=nothing)
#     tree = """graph{layout="dot";"""
#     drawn = []
#     or_counter = 100000
#     child_drawn_counter = 1000000
#     highlighted = false
#     parents = collect(keys(draw_tree_nodes))
#     for (n, parent) in enumerate(parents)
#         if parent ∉ drawn
#             tree *= """ "$(parent)"[label="$(parent[1])"];"""
#             push!(drawn, parent)
#         end
#         children = draw_tree_nodes[parent]
#         chds = zeros(Int, length(children))
#         for (m, child) in enumerate(children)
#             if child in drawn
#                 if child[1] == 'm'
#                     tree *= """ "$(child_drawn_counter)"[fixedsize=shape;shape=diamond;label="$(child[3:end])"fillcolor="lime";style=filled;height=.5;width=.5;];"""
#                 else
#                     tree *= """ "$(child_drawn_counter)"[label="$(child[child[1]=='v' ? 2 : 1])"fillcolor="red";style=filled;];"""
#                 end
#                 chds[m] = child_drawn_counter
#                 child_drawn_counter += 1
#             elseif child[1] == 'm'
#                 if child == highlight_node
#                     highlighted = true
#                     tree *= """ "$(child)"[fixedsize=shape;shape=diamond;label="$(child[3:end])"fillcolor="magenta";style=filled;height=.5;width=.5;];"""
#                 else
#                     tree *= """ "$(child)"[fixedsize=shape;shape=diamond;label="$(child[3:end])"fillcolor="lime";style=filled;height=.5;width=.5;];"""
#                 end
#             elseif child[1] == 'v'#child ∉ parents
#                 tree *= """ "$(child)"[label="$(child[2])"fillcolor="red";style=filled;];"""
#             elseif child[end] == '_'#child ∉ parents
#                 tree *= """ "$(child)"[label="$(child[1])"fillcolor="red";style=filled;];"""
#             elseif n == length(draw_tree_nodes)
#                 tree *= """ "$(child)"[label="$(child[1])"fillcolor="red";style=filled;];"""
#             else
#                 tree *= """ "$(child)"[label="$(child[1])"];"""
#             end
#             push!(drawn, child)
#         end
#         childss = split(parent, "_")[2:end]
#         counter3 = 1
#         for childs in childss
#             if children[counter3][1] == 'm'
#                 if chds[counter3] > 0
#                     tree *= """ "$(parent)"--"$(chds[counter3])";"""
#                 else
#                     tree *= """ "$(parent)"--"$(children[counter3])";"""
#                 end
#                 or_counter += 1
#                 counter3 += 1
#                 continue
#             end
#             tree *= """ $(or_counter) [shape=diamond,style=filled,label="",height=.3,width=.3];"""
#             tree *= """ "$(parent)"--"$(or_counter)";"""
#             childs = split(childs, "")
#             for chld in childs
#                 if chds[counter3] > 0
#                     tree *= """ "$(or_counter)"--"$(chds[counter3])";"""
#                 else
#                     tree *= """ "$(or_counter)"--"$(children[counter3])";"""
#                 end
#                 counter3 += 1
#             end
#             or_counter += 1
#         end
#     end
#     if highlight_node !== nothing && !highlighted
#         tree *= """ "$(or_counter+1)"[label="$(highlight_node[2])"];"""
#         tree *= """ "$(highlight_node)"[fixedsize=shape;shape=diamond;label="$(highlight_node[3:end])"fillcolor="magenta";style=filled;height=.5;width=.5;];"""
#         tree *= """ "$(or_counter+1)"--"$(highlight_node)";"""
#     end
#     tree *= "}"
#     return GraphViz.Graph(tree)
# end
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

function new_draw_tree(tree, board, root; visited=[], highlight_node=nothing, green_act=[])
    graph = """graph{graph [pad="0.2",nodesep="0.1",ranksep="0.2"];layout="dot";"""
    if isempty(visited)
        visited = collect(keys(tree))
    end
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    frontier = Vector{s_type}()
    drawn = Vector{s_type}()
    push!(frontier, root)
    for i in 1:length(tree)
        if isempty(frontier)
            break
        end
        node = popfirst!(frontier)
        push!(drawn, node)
        graph *= """ "$(node)"[fixedsize=shape,style=filled,fillcolor="$(isempty(tree[node]) ? "red" : "white")",width=0.6,margin=0,label="$(node[1])",fontsize=32,shape="circle"];"""
        for (or_node, s) in sort(collect(tree[node]), by=x->x[2])
            graph *= """ "$((node, or_node))" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(isempty(s) ? or_node in green_act ? "lime" : "red" : "gray75")",label="$(create_move_icon(or_node, board)[2:end])",height=$(isempty(s) ? .5 : .3),width=$(isempty(s) ? .5 : .3),fontsize=$(isempty(s) ? 18 : 10)];"""
            graph *= """ "$(node)"--"$((node, or_node))";"""
            for grandchild in reverse(s)
                if grandchild in keys(tree) && grandchild ∉ drawn
                    graph *= """ "$((node, or_node))"--"$(grandchild)";"""
                    if grandchild ∉ frontier
                        push!(frontier, grandchild)
                    end
                else
                    graph *= """ "$((node, grandchild))"[shape="circle",label="$(grandchild[1])",style=filled,fillcolor="red"];"""
                    graph *= """ "$((node, or_node))"--"$((node, grandchild))";"""
                end
            end
        end
    end
    # for node in visited
    #     if node ∉ keys(tree)
    #         continue
    #     end
    #     graph *= """ "$(node)"[fixedsize=shape,style=filled,fillcolor="$(isempty(tree[node]) ? "red" : "white")",width=0.6,margin=0,label="$(node[1])",fontsize=32,shape="circle"];"""
    #     push!(drawn, node)
    #     for (or_node, s) in sort(collect(tree[node]), by=x->x[2])
    #         graph *= """ "$((node, or_node))" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(isempty(s) ? or_node in green_act ? "lime" : "red" : "gray75")",label="$(create_move_icon(or_node, board)[2:end])",height=$(isempty(s) ? .5 : .3),width=$(isempty(s) ? .5 : .3),fontsize=$(isempty(s) ? 18 : 10)];"""
    #         graph *= """ "$(node)"--"$((node, or_node))";"""
    #         for grandchild in reverse(s)
    #             if grandchild in drawn
    #                 continue
    #             end
    #             if grandchild in keys(tree)
    #                 graph *= """ "$((node, or_node))"--"$(grandchild)";"""
    #             else
    #                 graph *= """ "$((node, grandchild))"[shape="circle",label="$(grandchild[1])",style=filled,fillcolor="red"];"""
    #                 graph *= """ "$((node, or_node))"--"$((node, grandchild))";"""
    #             end
    #         end
    #     end
    # end
    graph *= "}"
    return GraphViz.Graph(graph)
end

function invert_backtrack_tree(ms, root_a)
    tree = DefaultDict{Tuple{Tuple{Int, Int}, Int}, Vector{Tuple{Tuple{Int, Int}, Int}}}([])
    sol_counter = 0
    for level in reverse(sort(collect(keys(ms))))
        for (from, to) in ms[level]
            if to == root_a
                sol_counter += 1
                push!(tree[(to, sol_counter)], (from, sol_counter))
                ls = [from]
                new_ls = []
                for level2 in reverse(collect(1:level-1))
                    for (from2, to2) in ms[level2]
                        if to2 in ls
                            push!(tree[(to2, sol_counter)], (from2, sol_counter))
                            push!(new_ls, from2)
                        end
                    end
                    ls = new_ls
                    new_ls = []
                end
            end
        end
    end
    return tree, sol_counter
end

function draw_backtrack_tree(ms, board, root_a; max_depth=100)
    inverted_tree, n_sol = invert_backtrack_tree(ms, root_a);
    graph = """graph{graph [pad="0.2",nodesep="0.1",ranksep="0.2"];layout="dot";"""
    for i in 3:3
        frontier = Vector{Tuple{Tuple{Int, Int}, Int}}()
        push!(frontier, (root_a, 0))
        graph *= """ "$((root_a, i, 0))" [fixedsize=shape,shape=diamond,style=filled,fillcolor="cyan",label="$(create_move_icon(root_a, board)[1:end])",height=.7,width=.7,fontsize=18];"""
        for j in 1:max_depth
            if isempty(frontier)
                break
            end
            to, level = popfirst!(frontier)
            for (from, _) in inverted_tree[(to, i)]
                graph *= """ "$((from, i, level+1))" [fixedsize=shape,shape=diamond,style=filled,fillcolor="lime",label="$(create_move_icon(from, board)[1:end])",height=.5,width=.5,fontsize=12];"""
                graph *= """ "$((from, i, level+1))"--"$((to, i, level))";"""
                if (from, level+1) ∉ frontier 
                    push!(frontier, (from, level+1))
                end
            end
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

function draw_backtrack_state_space(state_space, action_space, board, root, idv_prb, optimal_a_prb; max_iter=100000, highlight_nodes=[], full=false, subj_states=[])
    frontier = Vector{Tuple{BigInt, BigInt, Int, Int, Tuple{Int, Int}}}()
    visited = Vector{Tuple{BigInt, BigInt}}()
    push!(frontier, (root, 0, idv_prb[root][1], -1, (-1, 0)))
    push!(visited, (root, 0))
    act_counter = 1
    constrained = []
    graph = """digraph{graph [pad="0.2",nodesep="0.5",ranksep="0.8"];layout="dot";"""
    if full
        for s in keys(optimal_a_prb)
            graph *= """ "$(s)"[fixedsize=shape,style=filled,fillcolor="$(s in highlight_nodes ? "orange" : "white")",width=0.3,margin=0,label="",fontsize=16,shape="circle"];"""
            for so in optimal_a_prb[s]
                graph *= """ "$(s)"->"$(so)"[constraint=true, style=invis;];"""
            end
        end
    end
    graph *= """ "$(root)"[fixedsize=shape,style=filled,fillcolor="$(isempty(state_space[root]) ? "lime" : "cyan")",width=0.6,margin=0,label="",fontsize=16,shape="circle"];"""
    for i in 1:max_iter
        if isempty(frontier)
            break
        end
        node, prev_node, d_goal, prev_d_goal, action = popfirst!(frontier)
        if prev_node != 0
            if node != root
                graph *= """ "$(node)"[fixedsize=shape,style=filled,fillcolor="$(0 in state_space[node] ? "lime" : node in highlight_nodes ? "orange" : unique(state_space[node]) == [-1] ? "purple" : unique(state_space[node]) == [-2] ? "red" : unique(state_space[node]) == [-3] ? "brown" : "white")",width=0.6,margin=0,label="",fontsize=16,shape="circle"];"""
            end
            if full
                graph *= """ "$(prev_node)"->"$(node)"[constraint=false,taillabel="$(create_move_icon(action, board)[1:end])";];"""
            else
            #graph *= """ "$(act_counter)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="gray",label="$(create_move_icon(action, board)[1:end])",height=.7,width=.7,fontsize=18];"""
                constraint = false
                if state_space[node] == [-3]
                    graph *= """ "$(node)"->"$(prev_node)"[constraint=true,taillabel="$(create_move_icon((action[1], -action[2]), board)[1:end])";];"""
                end
                for s in optimal_a_prb[prev_node]
                    if s in keys(state_space) && (prev_node, s) ∉ constrained
                        if s == node
                            constraint = true
                            graph *= """ "$(prev_node)"->"$(node)"[constraint=true,taillabel="$(create_move_icon(action, board)[1:end])";];"""
                        else
                            graph *= """ "$(prev_node)"->"$(s)"[constraint=true, style=invis;];"""
                        end
                        push!(constrained, (prev_node, s))
                    end
                end
                if !constraint
                    if prev_node in optimal_a_prb[node] && (prev_node, node) ∉ constrained
                        graph *= """ "$(node)"->"$(prev_node)"[constraint=true, style=invis;];"""
                    end
                    graph *= """ "$(prev_node)"->"$(node)"[constraint=false,taillabel="$(create_move_icon(action, board)[1:end])";];"""
                end
            end
            #graph *= """ "$(prev_node)"--"$(act_counter)"[constraint=$(d_goal < prev_d_goal);];"""
            #graph *= """ "$(act_counter)"--"$(node)"[constraint=$(d_goal < prev_d_goal);];"""
            act_counter += 1
        end
        for (n, s) in enumerate(state_space[node])
            if (s, node) ∉ visited && !(s <= 0)
                push!(frontier, (s, node, idv_prb[s][1], d_goal, action_space[node][n]))
                push!(visited, (s, node))
            end
        end
    end
    subj_edges = []
    for n in eachindex(subj_states)
        for m in 2:length(subj_states[n])
            node = subj_states[n][m]
            prev_node = subj_states[n][m-1]
            if (prev_node, node) ∉ subj_edges
                graph *= """ "$(node)"[fixedsize=shape,style=filled,color="blue",fillcolor="$(node == root ? "cyan" : "white")",width=0.6,margin=0,label="",fontsize=16,shape="circle"];"""
                graph *= """ "$(prev_node)"->"$(node)"[constraint=false,color="blue";];"""
                push!(subj_edges, (prev_node, node))
            end
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

function draw_chain_dict(chain_dict, board, root, optimal_a_prb; highlight_nodes=[], full=false)
    graph = """digraph{graph [pad="0.2",nodesep="0.5",ranksep="0.8"];layout="dot";"""
    if full
        for s in keys(optimal_a_prb)
            graph *= """ "$(s)"[fixedsize=shape,style=filled,fillcolor="$(s in highlight_nodes ? "orange" : "white")",width=0.3,margin=0,label="",fontsize=16,shape="circle"];"""
            for so in optimal_a_prb[s]
                graph *= """ "$(s)"->"$(so)"[constraint=true, style=invis;];"""
            end
        end
    end
    drawn = []
    drawn_edges = []
    constrained = []
    ss = [s[1] for s in keys(chain_dict)]
    for triplet in keys(chain_dict)
        node = triplet[1]
        if node == 0
            continue
        end
        action = triplet[3]
        # from node
        if node != root
            if node ∉ drawn
                push!(drawn, node)
                graph *= """ "$(node)"[fixedsize=shape,style=filled,fillcolor="$((-1, (-1, (-1,)), (-1, -1)) in chain_dict[triplet] ? "purple" : (-3, (-3, (-3,)), (-3, -3)) in chain_dict[triplet] ? "brown" : node in highlight_nodes ? "orange" : "white")",width=0.6,margin=0,label="",fontsize=16,shape="circle"];"""
            end
        else
            graph *= """ "$(node)"[fixedsize=shape,style=filled,fillcolor=cyan,width=0.6,margin=0,label="",fontsize=16,shape="circle"];"""
        end
        for next_triplet in chain_dict[triplet]
            next_node = next_triplet[1]
            if next_node < 0 || next_node == node
                continue
            end
            # to node
            if (next_node == 0 || (next_node != root && next_node ∉ ss)) && (node, next_node) ∉ drawn_edges
                push!(drawn, next_node)
                graph *= """ "$(next_node)"[fixedsize=shape,style=filled,fillcolor=lime,width=0.6,margin=0,label="",fontsize=16,shape="circle"];"""
                graph *= """ "$(node)"->"$(next_node)"[constraint=true,taillabel="$(create_move_icon(action, board)[1:end])";];"""
                push!(drawn_edges, (node, next_node))
                push!(constrained, (node, next_node))
                continue
            end
            # edges
            if full
                if (node, next_node) ∉ drawn_edges
                    push!(drawn_edges, (node, next_node))
                    graph *= """ "$(node)"->"$(next_node)"[constraint=false,taillabel="$(create_move_icon(action, board)[1:end])";];"""
                end
            else
                constraint = false
                for s in optimal_a_prb[node]
                    if s in ss && (node, s) ∉ constrained
                        if s == next_node
                            constraint = true
                            graph *= """ "$(node)"->"$(next_node)"[constraint=true,taillabel="$(create_move_icon(action, board)[1:end])";];"""
                            push!(drawn_edges, (node, s))
                        else
                            graph *= """ "$(node)"->"$(s)"[constraint=true, style=invis;];"""
                        end
                        push!(constrained, (node, s))
                    end
                end
                if !constraint
                    if node in optimal_a_prb[next_node] && (node, next_node) ∉ constrained
                        graph *= """ "$(next_node)"->"$(node)"[constraint=true, style=invis;];"""
                    end
                    if (node, next_node) ∉ drawn_edges
                        push!(drawn_edges, (node, next_node))
                        graph *= """ "$(node)"->"$(next_node)"[constraint=false,taillabel="$(create_move_icon(action, board)[1:end])";];"""
                        if (-3, (-3, (-3,)), (-3, -3)) in chain_dict[next_triplet] || (-1, (-1, (-1,)), (-1, -1)) in chain_dict[next_triplet]
                            graph *= """ "$(next_node)"->"$(node)"[constraint=true, style=invis;];"""
                        end
                    end
                end
            end
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

function new_draw_chain_dict(chain_dict, board, root; max_iters=100000)
    graph = """digraph{graph [pad="0.2",nodesep="0.5",ranksep="0.8"];layout="dot";"""
    drawn = []
    frontier = []
    push!(frontier, (root, [(0, (0,))], (0, 0)))
    for i in 1:max_iters
        if isempty(frontier)
            break
        end
        s = pop!(frontier)
        if s in drawn
            continue
        end
        if length(unique(chain_dict[s])) == 1 && chain_dict[s][1][1] < 0
            continue
        end
        push!(drawn, s)
        graph *= """ "$(s)"[fixedsize=shape,style=filled,fillcolor=white,width=0.3,margin=0,label="",fontsize=16,shape="circle"];"""
        for ss in chain_dict[s]
            if ss[1] <= 0
                continue
            end
            action = ss[3]
            if ss ∉ drawn
                graph *= """ "$(s)"->"$(ss)"[constraint="true",taillabel="$(create_move_icon(action, board)[1:end])";];"""
            else
                graph *= """ "$(s)"->"$(ss)"[constraint="false",taillabel="$(create_move_icon(action, board)[1:end])";];"""
            end
            if ss ∉ keys(chain_dict)
                push!(drawn, ss)
                graph *= """ "$(ss)"[fixedsize=shape,style=filled,fillcolor=red,width=0.3,margin=0,label="",fontsize=16,shape="circle"];"""
            elseif length(unique(chain_dict[ss])) == 1 && chain_dict[ss][1][1] == 0
                push!(drawn, ss)
                graph *= """ "$(ss)"[fixedsize=shape,style=filled,fillcolor=lime,width=0.3,margin=0,label="",fontsize=16,shape="circle"];"""
            elseif length(unique(chain_dict[ss])) == 1 && chain_dict[ss][1][1] == -1
                push!(drawn, ss)
                graph *= """ "$(ss)"[fixedsize=shape,style=filled,fillcolor=purple,width=0.3,margin=0,label="",fontsize=16,shape="circle"];"""
            elseif length(unique(chain_dict[ss])) == 1 && chain_dict[ss][1][1] == -3
                push!(drawn, ss)
                graph *= """ "$(ss)"[fixedsize=shape,style=filled,fillcolor=brown,width=0.3,margin=0,label="",fontsize=16,shape="circle"];"""
            end
            pushfirst!(frontier, ss)
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

function draw_one_step_chains(chains)
    graph = """digraph{graph [pad="0.2",nodesep="0.2",ranksep="0.2"];layout="dot";rankdir="LR";"""
    counter = 0
    for chain in reverse(chains)
        move = chain[end][2]
        counter += 1
        graph *= """ "$(counter)" [fixedsize=shape,shape=diamond,style=filled,fillcolor=lime,label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=$(abs(move[2]) == 1 ? 20 : abs(move[2]) == 2 ? 18 : abs(move[2]) == 3 ? 8 : 6)];"""
        counter += 1
        graph *= """ "$(counter)"->"$(counter-1)"[constraint="true",arrowhead=none;];"""
        graph *= """ "$(counter)"[fixedsize=shape,style=filled,fillcolor=white,width=0.5,margin=0,label="$(move[1])",fontsize=16,shape="circle"];"""
        for (n, node) in enumerate(reverse(chain[1:end-1]))
            move = node[2]
            counter += 1
            graph *= """ "$(counter)"->"$(counter-1)"[constraint="true",arrowhead=none;];"""
            graph *= """ "$(counter)" [fixedsize=shape,shape=diamond,style=filled,fillcolor=gray75,label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=$(abs(move[2]) == 1 ? 20 : abs(move[2]) == 2 ? 18 : abs(move[2]) == 3 ? 8 : 6)];"""
            counter += 1
            graph *= """ "$(counter)"->"$(counter-1)"[constraint="true",arrowhead=none;];"""
            graph *= """ "$(counter)"[fixedsize=shape,style=filled,fillcolor=white,width=0.5,margin=0,label="$(move[1])",fontsize=16,shape="circle"];"""
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

function draw_ao_tree(AO, board)
    graph = """digraph{graph [pad="0.2",nodesep="0.15",ranksep="0.3"];layout="dot";"""
    drawn_and = []
    drawn_or = []
    AND, OR = AO
    for or in keys(AND)
        if or ∉ drawn_or
            push!(drawn_or, or)
            move = or[2]
            graph *= """ "$(or)" [fixedsize=shape,shape=diamond,style=filled,fillcolor="$(AND[or][1][1] == (0, (0,)) ? "lime" : "gray75")",label="$(create_move_icon(move, board)[2:end])",height=.5,width=.5,fontsize=14];"""
        end
        for and in AND[or]
            if and[1] == (0, (0,))
                continue
            end
            graph *= """ "$(or)"->"$(and)"[constraint="true"];"""
        end
    end
    for and in keys(OR)
        if and ∉ drawn_and
            push!(drawn_and, and)
            graph *= """ "$(and)"[fixedsize=shape,style=filled,fillcolor=white,width=0.6,margin=0,label="$(and[1][1])",fontsize=16,shape="circle"];"""
        end
        for or in OR[and]
            graph *= """ "$(and)"->"$(or)"[constraint="true"];"""
        end
    end
    graph *= "}"
    return GraphViz.Graph(graph)
end

#draw_backtrack_state_space(state_space, action_space, board, root, IDV[prb])

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

board = load_data(prbs[20])
board = load_data(prbs[1])
board = arr_to_board(int_to_arr(s))
board = arr_to_board(int_to_arr(sss))
draw_board(get_board_arr(board))
tree, visited, actions, blockages, trials, parents, move_parents, h, repeated_actions = new_and_or_tree(board);
g = new_draw_tree(tree, board)#; visited)#; green_act=actions)

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