using GraphViz, FileIO, ImageIO
using Plots

function initialise_graph(title="")
    """
    Creates initial graph outline with title in dummy node
    """
    if title != ""
        return """digraph {layout="dot";splines=false;dummy [label=""" * string(title) * " shape=box]; dummy -> 9 [style=invis];"
    else
        return """digraph {layout="dot";"""
    end
end

function add_layer(g, n, nodes; heurs=[], max_heur=0, root=false, invisible_nodes=false)
    """
    Adds layer as subgraph cluster with nodes and n as layer number
    """
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
    return g * "}"
end

function add_node(g, node; invisible_nodes=false, color="", label="", invis=false)
    """
    Adds node to graph
    """
    g *= string(node)
    if color != "" || invisible_nodes || label != "" || invis
        g *= "["
    end
    if color != ""
        g *= "fillcolor=" * color * ";style=filled;"
    end
    if invis
        g *= "style=invis;"
    end
    if invisible_nodes
        g *= """label="";"""
    elseif label != ""
        g *= "label=" * label * ";"
    end
    if color != "" || invisible_nodes || label != "" || invis
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

function add_edge(g, a, b; constraint=true, color="", bidirectional=false, invis=false)
    """
    Draw edge between nodes a and b, if constraint, enforce to keep hierarchy
    """
    g *= string(a) * "->" * string(b)
    if color != "" || !constraint || bidirectional || invis
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
        g *= "dir=both;"
    end
    if color != "" || !constraint || bidirectional || invis
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

function close_graph(g)
    """
    Finalises graph
    """
    return g * "}"
end

function test_graph()
    g = initialise_graph("test", "move_0")
    g = add_layer(g, 0, [9], true)

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
    g = add_edge(g, 8, 1, true)
    
    g = add_edge(g, 6, 9, true)
    g = add_edge(g, 6, 4)
    g = add_edge(g, 5, 3, true)

    g = close_graph(g)

    return GraphViz.Graph(g)
end

function save_graph(g, file_name)
    """
    Save graph as png
    """
    FileIO.save(file_name * ".svg", g)
end

function plot_mag(mag; title="")
    """
    Plot hierarchical graph from MAG
    """
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

function plot_tree(tree)
    """
    Creates centered horizontal bar graph to display tree shape
    """
    x = collect(keys(tree))
    y = collect(values(tree))
    bar(x,y,orientation=:h, fillto=-y, yflip=true, grid=false, legend=false, xticks=[-round(maximum(y), sigdigits=1), 0, round(maximum(y), sigdigits=1)])
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

function draw_directed_tree(parents; solution_paths=Dict(), all_parents=[], value_map=[], solutions=[])
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
    for child in x
        node_color = ""
        if length(solution_paths) > 0
            if child in optimal_nodes
                node_color = """ "#0000ff" """
            end
        end
        if length(solutions) > 0
            if child in solutions
                node_color = """ "#00ff00" """
            end
        end
        if length(value_map) > 0
            g = add_node(g, child, color=node_color, label=string(round(value_map[child], digits=2)))
        else
            g = add_node(g, child, invisible_nodes=true, color=node_color)
        end
        for parent in parents[child]
            g = add_edge(g, parent, child, color=node_color, bidirectional=length(all_parents) > 0)
            push!(all_edges[parent], child)
            push!(all_edges[child], parent)
        end
    end
    if length(all_parents) > 0
        for child in x
            for parent in all_parents[child]
                if !(parent in all_edges[child])
                    g = add_edge(g, parent, child, constraint=false, bidirectional=true)
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
