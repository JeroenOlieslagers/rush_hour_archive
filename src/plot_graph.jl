using GraphViz, FileIO, ImageIO
using Plots, ProgressBars
using Accessors

"""
    g = initialise_graph(title="")

Creates initial graph outline with title in dummy node
"""
function initialise_graph(title="")
    if title != ""
        return """digraph {layout="dot";splines=false;dummy [label=""" * string(title) * " shape=box]; dummy -> 9 [style=invis];"
    else
        return """digraph {layout="dot";"""
    end
end

"""
    add_layer(g, n, nodes, heurs=[], max_heur=0, root=false, invisible_nodes=false)

Adds layer as subgraph cluster with nodes and n as layer number

# Arguments
- `g::String`:                 graph string to be edited.
- `n::Integer`:                layer number.
- `nodes::Vector`:             list of nodes.
- `heurs::Vector`:             list of heuristic values to be displayed in each node.
- `max_heur::Integer`:         maximum_heuristic to set scale.
- `root::Boolean`:             set layer as root for red car. 
- `invisible_nodes::Boolean`:  set to true to hide node labels.
"""
function add_layer(g, n, nodes; heurs=[], max_heur=0, root=false, invisible_nodes=false)
    g *= "subgraph cluster_" * string(n) * "{peripheries=0;"
    if root
        g *= string(nodes[1]) * """[label=R fillcolor="#ff0000" style=filled];"""
    elseif invisible_nodes
        for (m, node) in enumerate(nodes)
            if length(heurs) > 0
                fracs = [string(round(Int, 255*(heur/max_heur)), base=16) for heur in heurs]
                combs = [(length(frac) == 1 ? "0" * frac : frac) for frac in fracs]
                cols = [""" "#ff""" * comb * comb * """ " """ for comb in combs]
                g *= string(node) * """[label=" """ * string(heurs[m]) * """ "fillcolor=""" * cols[m] * """style=filled];"""
            else
                g *= string(node) * """[label=""];"""
            end
        end
    else
        for (m, node) in enumerate(nodes)
            if length(heurs) > 0
                fracs = [string(round(Int, 255*(heur/max_heur)), base=16) for heur in heurs]
                combs = [(length(frac) == 1 ? "0" * frac : frac) for frac in fracs]
                cols = [""" "#ff""" * comb * comb * """ " """ for comb in combs]
                g *= string(node) * """[fillcolor=""" * cols[n] * """style=filled];"""
            else
                g *= string(node) * ";"
            end
        end
    end
    return g *= "}"
end

"""
    g = add_node(g, node, invisible_nodes=false, color="", label="", invis=false)

Adds node to graph

# Arguments
- `g::String`:                 graph string to be edited.
- `node::Integer`:             node number.
- `invisible_nodes::Boolean`:  set to true to hide node label.
- `color::String`:             color of node.
- `label::String`:             alternate label for node.
- `invis::Boolean`:            set to true to hide node. 
"""
function add_node(g, node; invisible_nodes=false, color="", border_color="", label="", invis=false, width=0)
    g *= string(node)
    if color != "" || invisible_nodes || label != "" || invis || border_color != "" || width != 0
        g *= "[shape = circle;"
    else
        g *= "[shape = circle]"
    end
    if color != ""
        g *= """fillcolor=""" * color * """;style=filled;"""
    end
    if border_color != ""
        g *= """color=""" * border_color * """;"""
    end
    if invis
        g *= "style=invis;"
    end
    if invisible_nodes
        g *= """label="";"""
    elseif label != ""
        g *= "label=" * label * ";"
    end
    if width != 0
        g *= "penwidth=" * string(width) * ";"
    end
    if color != "" || invisible_nodes || label != "" || invis || border_color != "" || width != 0
        g *= "]"
    end
    return g *= ";"
    # if invisible_nodes
    #     if color != ""
    #         return g *= string(node) * """[label="" fillcolor=""" * color * """ style=filled];"""
    #     else
    #         return g *= string(node) * """[label=""];"""
    #     end
    # else
    #     if color != ""
    #         return g *= string(node) * """[fillcolor=""" * color * """ style=filled];"""
    #     else
    #         return g *= string(node) * ";"
    #     end
    # end
end

"""
    g = add_edge(g, a, b, constraint=true, color="", bidirectional=false, invis=false)

Draw edge between nodes a and b, if constraint, enforce to keep hierarchy

# Arguments
- `g::String`:                 graph string to be edited.
- `a::Integer`:                start node.
- `b::Integer`:                end node.
- `constraint::Boolean`:       set to true to exclude from ordering.
- `color::String`:             color of edge.
- `bidirectional::Boolean`:    set to true for bidirectional edge.
- `invis::Boolean`:            set to true to hide edge. 
"""
function add_edge(g, a, b; constraint=true, color="", bidirectional=false, invis=false, width=0, arrow_size=0)
    g *= string(a) * "->" * string(b)
    if color != "" || !constraint || bidirectional || invis || width != 0 || arrow_size != 0
        g *= "["
    end
    if color != ""
        g *= "color=" * color * ";"
    end
    if invis
        g *= "style=invis;"
    end
    if !constraint
        g *= "constraint=false;"
    end
    if bidirectional
        g *= "arrowhead=none;"#"dir=both;"
    end
    if arrow_size != 0
        g *= "arrowsize=" * string(arrow_size) * ";"
    end
    if width != 0
        g *= "penwidth=" * string(width) * ";"
    end
    if color != "" || !constraint || bidirectional || invis || width != 0 || arrow_size != 0
        g *= "]"
    end
    return g *= ";"
    # if !constraint
    #     if color != ""
    #         return g *= """[color=""" * color * """, constraint=false];"""
    #     else
    #         return g *= "[constraint=false, dir=both];"
    #     end
    # else
    #     if color != ""
    #         return g *= """[color=""" * color * """];"""
    #     else
    #         return g *= ";"
    #     end
    # end
end

"""
    g = close_graph(g)

Finalises graph
"""
function close_graph(g)
    return g * "}"
end

function test_graph()
    g = initialise_graph()
    g = add_layer(g, 0, [9], root=true)

    g = add_layer(g, 1, [1, 2, 7])
    g = add_layer(g, 2, [3, 8])
    g = add_layer(g, 3, [6, 5])
    g = add_layer(g, 4, [4])
    g = add_edge(g, 9, 7)
    g = add_edge(g, 9, 2)
    g = add_edge(g, 9, 1)
    
    g = add_edge(g, 1, 3)
    g = add_edge(g, 2, 8)
    g = add_edge(g, 7, 8)
    
    g = add_edge(g, 3, 6)
    g = add_edge(g, 8, 5)
    g = add_edge(g, 8, 1)
    
    g = add_edge(g, 6, 9, constraint=false)
    g = add_edge(g, 6, 4)
    g = add_edge(g, 5, 3)

    g = close_graph(g)

    return GraphViz.Graph(g)
end

"""
    save_graph(g, file_name)

Save graph as svg
"""
function save_graph(g, file_name)
    FileIO.save(file_name * ".svg", g)
end

"""
    g = plot_mag(mag, title="")

Plot hierarchical graph from MAG
"""
function plot_mag(mag; title="")
    # Start graph
    g = initialise_graph(title)
    # Keep track of visited nodes to retain hierarchy
    visited_nodes = []
    for (n, layer) in enumerate(mag)
        # Get nodes per layer
        nodes = [keys(layer)...]
        # Create cluster subgraphs
        g = add_layer(g, n, nodes, root=n == 1)
        for node in nodes
            # Append visited nodes
            push!(visited_nodes, node)
            # Create edges
            for target in layer[node]
                g = add_edge(g, node, target, constraint=!(target in visited_nodes))
            end
        end
    end
    g = close_graph(g)
    return GraphViz.Graph(g)
end

"""
    plot_tree(tree, fake_tree, sp=0)

Creates centered horizontal bar graph to display tree shape.
Fake tree shows optimal paths to solutions
sp is sup plot counter
"""
function plot_tree(tree, fake_tree; sp=0)
    x = collect(keys(tree))
    y = collect(values(tree))./2
    xx = collect(keys(fake_tree))
    yy = collect(values(fake_tree))./2
    if sp > 0
        bar!(x,y,sp=sp,orientation=:h, fillto=-y, yflip=true, grid=false, label=nothing, yticks=[], xticks=[])
        bar!(xx,yy,sp=sp,orientation=:h, fillto=-yy, yflip=true, label=nothing)
    else
        bar(x,y,orientation=:h, fillto=-y, yflip=true, grid=false, xlabel="Number of states", ylabel="Moves from root", label="Total states", xticks=[-round(maximum(y), sigdigits=1), 0, round(maximum(y), sigdigits=1)])
        bar!(xx,yy,orientation=:h, fillto=-yy, yflip=true, label="States on solution path")
    end
end

function draw_solution_paths(solution_paths, parents, stat, max_heur)
    """
    Draws graphs that displays all solution paths
    """
    # Start graph
    g = initialise_graph()
    # Get layers
    x = reverse(collect(keys(solution_paths)))
    for layer in x
        heurs = [stat[nod][3] for nod in solution_paths[layer]]
        g = add_layer(g, layer, solution_paths[layer], heurs=heurs, max_heur=max_heur, invisible_nodes=true)
        if layer < length(x)
            for child in solution_paths[layer + 1]
                for parent in parents[child]
                    g = add_edge(g, parent, child)
                end
            end
        end
    end
    g = close_graph(g)
    return GraphViz.Graph(g)
end

function draw_directed_tree(parents; solution_paths=Dict(), all_parents=[], value_map=[], solutions=[], start=nothing)
    """
    Draws full tree of parents
    """
    # Start graph
    g = initialise_graph()
    # Get parents
    x = reverse(collect(keys(parents)))
    optimal_nodes = Set{BigInt}()
    # All edges drawn so far
    all_edges = DefaultDict{BigInt, Array{BigInt, 1}}([])
    if length(solution_paths) > 0
        # Get list of all nodes on optimal path
        optimal_nodes = Set(reduce(vcat, collect(values(solution_paths))))
    end
    for child in ProgressBar(x)
        node_color = ""
        edge_color = """ "#00000050" """
        edge_width = 1
        if length(solution_paths) > 0
            if child in optimal_nodes
                node_color = """ "#0000ff" """
                edge_color = """ "#0000ff" """
                edge_width = 5
            end
        end
        if length(solutions) > 0
            if child in solutions
                node_color = """ "#00ff00" """
            end
        end
        if child == start
            node_color = """ "#ff0000" """
        end
        if length(value_map) > 0
            g = add_node(g, child, color=node_color, label=string(round(value_map[child], digits=2)))
        else
            g = add_node(g, child, invisible_nodes=true, color=node_color)
        end
        for parent in parents[child]
            if node_color == """ "#00ff00" """
                if parent in optimal_nodes
                    edge_color = """ "#0000ff" """
                    edge_width = 5
                else
                    edge_width = 1
                end
            end
            g = add_edge(g, parent, child, color=edge_color, bidirectional=length(all_parents) > 0, width=edge_width)
            push!(all_edges[parent], child)
            push!(all_edges[child], parent)
        end
    end
    if length(all_parents) > 0
        for child in x
            for parent in all_parents[child]
                if !(parent in all_edges[child])
                    g = add_edge(g, parent, child, constraint=false, bidirectional=true, color=""" "#00000050" """, width=1)
                    push!(all_edges[parent], child)
                    push!(all_edges[child], parent)
                end
            end
        end
    end
    g = close_graph(g)
    return GraphViz.Graph(g)
end


function draw_subgraph(parents, graph, value_map, color_map)
    """
    Draws tree (parents) with full structure of graph so that
    subsequent calls do not change structure. all_parents sets
    all edges, value_map displays values and color_map displays colors
    for each node.
    """
    # Start graph
    g = initialise_graph()
    # Get children of subgraph
    sub_children = reverse(collect(keys(parents)))
    # Get children of full graph
    full_children = reverse(collect(keys(graph)))
    # Draw invisible full graph
    for child in full_children
        # Set color for node in colormap, defaults to none
        node_color = haskey(color_map, child) ? color_map[child] : ""
        if child in sub_children
            # Add nodes from subgraph
            g = add_node(g, child, color=node_color, label=string(round(value_map[child], digits=2)))
        else
            # Add invisible nodes from full graph
            g = add_node(g, child, invisible_nodes=true, invis=true)
        end
        # Add invisible edges from full graph
        for parent in graph[child]
            g = add_edge(g, parent, child, invis=true)
        end
    end
    # Draw visible subgraphe edges
    for child in sub_children
        # Add visible edges from subgraph
        for parent in parents[child]
            g = add_edge(g, parent, child, constraint=false)
        end
    end
    g = close_graph(g)
    return GraphViz.Graph(g)
end


function draw_visited_nodes(nodes, graph, all_parents, solutions, solution_paths)
    # Start graph
    g = initialise_graph()
    # Get children of full graph
    full_children = reverse(collect(keys(graph)))
    # Get children of nodes
    sub_children = cat([all_parents[node] for node in nodes]..., dims=1)
    # Get nodes on optimal paths
    optimal_nodes = Set(reduce(vcat, collect(values(solution_paths))))
    # Draw invisible full graph
    for child in full_children
        if child in nodes
            # Add coloured nodes
            if child in solutions
                g = add_node(g, child, color="lime", invisible_nodes=true)
            elseif child == last(nodes)
                g = add_node(g, child, color="red", invisible_nodes=true)
            elseif child in optimal_nodes
                g = add_node(g, child, color="magenta", invisible_nodes=true)
            else
                g = add_node(g, child, color="orange", invisible_nodes=true)
            end
        elseif child in solutions
            g = add_node(g, child, color="green", invisible_nodes=true)
        elseif child in optimal_nodes
            if child in sub_children
                g = add_node(g, child, color="cyan", invisible_nodes=true)
            else
                g = add_node(g, child, color="blue", invisible_nodes=true)
            end
        elseif child in sub_children
            # Add neighbours of nodes
            g = add_node(g, child, invisible_nodes=true, color="yellow")
        else
            # Add invisible nodes from full graph
            g = add_node(g, child, invisible_nodes=true)#, invis=true)
        end
        # Add invisible edges from full graph
        for parent in all_parents[child]
            if parent in graph[child]
                if parent in optimal_nodes && child in optimal_nodes
                    g = add_edge(g, parent, child, color="blue")
                else
                    g = add_edge(g, parent, child, invis=true)
                end
            else
                g = add_edge(g, parent, child, invis=true, constraint=false)
            end
        end
    end
    # Draw visible subgraph edges
    for (n, child) in enumerate(nodes)
        if n > 1
            if nodes[n-1] in all_parents[child]
                g = add_edge(g, nodes[n-1], child, constraint=false, color="red")
            end
        end
        # Add visible edges from subgraph
        # for parent in all_parents[child]
        #     if n > 1
        #         if parent == nodes[n-1]
        #             g = add_edge(g, parent, child, constraint=false, color="red")
        #         end
        #     end
        # end
    end
    g = close_graph(g)
    return GraphViz.Graph(g)
end

function draw_visited_nodes_simple(nodes, graph, all_parents, solutions, solution_paths)
    # Start graph
    g = initialise_graph()
    # Get children of full graph
    full_children = reverse(collect(keys(graph)))
    # Get children of nodes
    sub_children = cat([all_parents[node] for node in nodes]..., dims=1)
    # Get nodes on optimal paths
    optimal_nodes = Set(reduce(vcat, collect(values(solution_paths))))
    # Draw invisible full graph
    for child in full_children
        if child in nodes
            # Add coloured nodes
            if child == last(nodes)
                if child in solutions
                    g = add_node(g, child, color="red", invisible_nodes=true, border_color="green", width=5)
                elseif child in optimal_nodes
                    g = add_node(g, child, color="red", invisible_nodes=true, border_color="blue", width=5)
                else
                    g = add_node(g, child, color="red", invisible_nodes=true)
                end
            else
                if child in solutions
                    g = add_node(g, child, color="orange", invisible_nodes=true, border_color="green", width=5)
                elseif child in optimal_nodes
                    g = add_node(g, child, color="orange", invisible_nodes=true, border_color="blue", width=5)
                else
                    g = add_node(g, child, color="orange", invisible_nodes=true)
                end
            end
        else
            # Add invisible nodes from full graph
            if child in solutions
                g = add_node(g, child, invisible_nodes=true, border_color="green", width=5)
            elseif child in optimal_nodes
                g = add_node(g, child, invisible_nodes=true, border_color="blue", width=5)
            else
                g = add_node(g, child, invisible_nodes=true)
            end
        end
        # Add invisible edges from full graph
        for parent in all_parents[child]
            if parent in graph[child]
                if parent in optimal_nodes && child in optimal_nodes
                    g = add_edge(g, parent, child, color="blue")
                else
                    g = add_edge(g, parent, child, invis=true)
                end
            else
                g = add_edge(g, parent, child, invis=true, constraint=false)
            end
        end
    end
    # Draw visible subgraph edges
    for (n, child) in enumerate(nodes)
        if n > 1
            if nodes[n-1] in all_parents[child]
                g = add_edge(g, nodes[n-1], child, constraint=false, color="red", width=3, arrow_size=1.5)
            end
        end
    end
    g = close_graph(g)
    return GraphViz.Graph(g)
end

function draw_ss_heatmap(counts, graph, all_parents, solutions, solution_paths)
    # Start graph
    g = initialise_graph()
    # Get children of full graph
    full_children = reverse(collect(keys(graph)))
    # Get nodes on optimal paths
    optimal_nodes = Set(reduce(vcat, collect(values(solution_paths))))
    # Get maximum visits as baseline
    max_count = maximum(values(counts))
    # Draw invisible full graph
    for child in full_children
        # c = 0
        # ct = counts[child]
        # if ct >= 40
        #     c = 6
        # elseif ct >= 20
        #     c = 5
        # elseif ct >= 5
        #     c = 4
        # elseif ct >= 3
        #     c = 3
        # elseif ct >= 2
        #     c = 2
        # elseif ct >= 1
        #     c = 1
        # end
        # alpha = c/6
        offset = 10
        alpha = counts[child] == 0 ? 0 : (offset+counts[child])/max_count > 1 ? 1 : (offset+counts[child])/max_count
        color = """ "#ff0000"""*string(round(Int, 255*alpha), base=16)*"""" """
        if child in solutions
            g = add_node(g, child, color=color, border_color="green", invisible_nodes=true, width=5)
        elseif child in optimal_nodes
            g = add_node(g, child, color=color, border_color="blue", invisible_nodes=true, width=5)
        else
            # Add invisible nodes from full graph
            g = add_node(g, child, color=color, invisible_nodes=true)#, invis=true)
        end
        # Add invisible edges from full graph
        for parent in all_parents[child]
            if parent in graph[child]
                if parent in optimal_nodes && child in optimal_nodes
                    g = add_edge(g, parent, child, color="blue", width=3, arrow_size=1.5)
                else
                    g = add_edge(g, parent, child, invis=true)
                end
            else
                g = add_edge(g, parent, child, invis=true, constraint=false)
            end
        end
    end
    g = close_graph(g)
    return GraphViz.Graph(g)
end


function plot_tree(prb)
    board = load_data(prb)
    arr = get_board_arr(board)
    tree, seen, stat, dict, parents, children, solutions = bfs_path_counters(board, traverse_full=true);
    solution_paths, fake_tree, max_heur = get_solution_paths(solutions, parents, stat);
    plot_tree(tree, fake_tree)
end

function plot_state_space(prb)
    board = load_data(prb)
    arr = get_board_arr(board)
    tree, seen, stat, dict, parents, children, solutions = bfs_path_counters(board, traverse_full=true);
    tree, seen, stat, dict, all_parents, children, solutions = bfs_path_counters(board, traverse_full=true, all_parents=true);
    solution_paths, fake_tree, max_heur = get_solution_paths(solutions, parents, stat);
    g = draw_directed_tree(parents, solution_paths=solution_paths, solutions=solutions, all_parents=all_parents)
    return g
end

function draw_prb(prb)
    board = load_data(prb)
    arr = get_board_arr(board)
    draw_board(arr)
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
        cmap = @set cmap[alpha_map[i]+1].alpha = 0.3
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
        heatmap!(arr, c = cmap, sp=sp, legend = nothing, yflip = true, xmirror=true, framestyle = :box)
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


function all_trees_plot(prbs)

    plot(layout=grid(4, 18), size=(2400, 500))

    for i in ProgressBar(eachindex(prbs))
        board = load_data(prbs[i])
        arr = get_board_arr(board)
        tree, seen, stat, dict, parents, children, solutions = bfs_path_counters(board, traverse_full=true);
        solution_paths, fake_tree, max_heur = get_solution_paths(solutions, parents, stat);
        if i < 18+18+18
            plot_tree(tree, fake_tree, sp=i)
        else
            plot_tree(tree, fake_tree, sp=i+1)
        end
    end
    plot!(xticks=[], yticks=[], sp=18+18+18)
    plot!(xticks=[], yticks=[], sp=18+18+18+18)
end
