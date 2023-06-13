include("engine.jl")
include("solvers.jl")
include("plot_graph.jl")
using Random


function traverse(board, value_start, γ; h=(x,y)->0, noise=randn)
    """
    Starts at a node and expands with probability γ. 
    When finished, returns best heuristic value encountered.
    h(x, y) is the heuristic function that will give the value to the nodes
    noise is the function that will add stochasticity to the heuristic value function
    """
    # Seed
    Random.seed!(3)
    # Get current state
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    start = board_to_int(arr_start, T)
    # open_set is priority queue that will indicate which node to expand next (best first)
    open_set = PriorityQueue{T, Float64}()
    enqueue!(open_set, start, 0)
    # dict is a map that gives cheapest cost from start to node currently known
    # (initialised with infinity as default value) as first element, the second element
    # is an array with the shortest path in moves to board state
    dict = DefaultDict{T, Array}([100000, []])
    dict[start] = [value_start, []]
    # parents is a map that shows the parents of each node
    parents = DefaultDict{T, Array{T, 1}}([])
    parents[start] = []
    # all_parents is a map that shows all parents of each node
    all_parents = DefaultDict{T, Array{T, 1}}([])
    all_parents[start] = []
    # value_map is a map that associates each node with its value
    value_map = Dict{T, Float64}()
    value_map[start] = h(board, arr_start)# value_start
    # Initialise
    expansions = 0
    # Loops until complete, will end more often due to expansion probability
    while length(open_set) > 0
        # Get state with best f score
        current = dequeue!(open_set)
        f_score_old, past_moves = dict[current]
        # Move to current node
        if length(past_moves) > 0
            for move in past_moves
                make_move(board, move)
            end
        end
        # Get current board int-string
        arr_current = get_board_arr(board)
        # Check if solved
        if check_solved(arr_current)
            println("FOUND SOLUTION")
            #break
        end
        # Expand current node by getting all available moves
        available_moves = get_all_available_moves(board, arr_current)
        # Increment expansion counter
        expansions += 1
        for move in available_moves
            dummy = true
            # Perform available move
            make_move(board, move)
            # Get board int-string
            arr_new = get_board_arr(board)
            new = board_to_int(arr_new, T)
            # Calculates g score of neighbor
            tentative_score = f_score_old + 1
            # If this is the fastest way to get to neighbor, update its costs
            if tentative_score < dict[new][1]
                # Update score
                dict[new] = [tentative_score, vcat(past_moves, [move])]
                heuristic_value = h(board, arr_new) + noise()
                f_score_new = tentative_score + heuristic_value
                # Update heuristic value
                value_map[new] = f_score_new#heuristic_value
                # Update parents
                push!(parents[new], current)
                # If neighbor is not yet queued to be visited, queue it
                if !(new in keys(open_set))
                    # Push node onto heap
                    enqueue!(open_set, new, f_score_new)
                end
                dummy = false
            end
            if dummy
                # Update parents
                push!(all_parents[new], current)
            end
            # Undo move
            undo_moves(board, [move])
        end
        # Undo rest of moves back to initial board
        undo_moves(board, past_moves)
        if rand() < γ
            break
        end
    end
    # for state in keys(open_set)
    #     delete!(parents, state)
    #     delete!(value_map, state)
    # end
    println("Expansions: "* string(expansions))
    return value_map, parents, all_parents
end

function animate_expand_astar(board, value_start, steps, graph; h=(x,y)->0, noise=randn)
    """
    Creates series of png images of graph for animation of a star expansion method
    """
    # Seed
    Random.seed!(3)
    # Get current state
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    start = board_to_int(arr_start, T)
    past = start
    # open_set is priority queue that will indicate which node to expand next (best first)
    open_set = PriorityQueue{T, Float64}()
    enqueue!(open_set, start, 0)
    # dict is a map that gives cheapest cost from start to node currently known
    # (initialised with infinity as default value) as first element, the second element
    # is an array with the shortest path in moves to board state
    dict = DefaultDict{T, Array}([100000, []])
    dict[start] = [value_start, []]
    # parents is a map that shows the parents of each node
    parents = DefaultDict{T, Array{T, 1}}([])
    parents[start] = []
    # value_map is a map that associates each node with its value
    value_map = Dict{T, Float64}()
    value_map[start] = h(board, arr_start)
    # Draw first frame
    color_map = Dict()
    g = draw_subgraph(parents, graph, value_map, color_map)
    #save_graph(g, "puzzle_hard_1_astar_" * string(steps) * "_steps_frame_000")
    # Loop over animation steps
    for step in 1:steps
        # Get state with best value
        current = dequeue!(open_set)
        f_score_old, past_moves = dict[current]
        # Move to current node
        if length(past_moves) > 0
            for move in past_moves
                make_move!(board, move)
            end
        end
        # Color smallest value
        color_map = Dict(current=>""" "#00ff00" """, past=>""" "#0000ff" """)
        g = draw_subgraph(parents, graph, value_map, color_map)
        display(g)
        fill = (2*step)-1 < 10 ? "00" : (2*step)-1 < 100 ? "0" : ""
        #save_graph(g, "puzzle_hard_1_astar_" * string(steps) * "_steps_frame_" * fill * string((2*step)-1))
        delete!(color_map, past)
        # Get current board int-string
        arr_current = get_board_arr(board)
        # Expand current node by getting all available moves
        available_moves = get_all_available_moves(board, arr_current)
        for move in available_moves
            # Perform available move
            make_move!(board, move)
            # Get board int-string
            arr_new = get_board_arr(board)
            new = board_to_int(arr_new, T)
            # Calculates g score of neighbor
            tentative_score = f_score_old + 1
            if tentative_score < dict[new][1]
                # Update score
                dict[new] = [tentative_score, vcat(past_moves, [move])]
                f_score_new = tentative_score + h(board, arr_new) + noise()
                # Update heuristic value
                value_map[new] = f_score_new
                # If neighbor is not yet queued to be visited, queue it
                if !(new in keys(open_set))
                    # Push node onto heap
                    enqueue!(open_set, new, f_score_new)
                end
            end
            # Update parents
            push!(parents[new], current)
            # Update colors
            color_map[new] = """ "#ff0000" """
            undo_moves!(board, [move])
        end
        # Undo rest of moves back to initial board
        undo_moves!(board, past_moves)
        # Color expanded nodes
        g = draw_subgraph(parents, graph, value_map, color_map)
        display(g)
        fill = 2*step < 10 ? "00" : 2*step < 100 ? "0" : ""
        #save_graph(g, "puzzle_hard_1_astar_" * string(steps) * "_steps_frame_" * fill * string(2*step))
        past = current
    end
end

function animate_expand_backprop(board, value_start, steps, graph; h=(x,y)->0, noise=randn)
    """
    Creates series of png images of graph for animation of backprop expansion method
    """
    # Seed
    Random.seed!(3)
    # Get current state
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    start = board_to_int(arr_start, T)
    past = start
    # open_set is priority queue that will indicate which node to expand next (best first)
    open_set = PriorityQueue{T, Float64}()
    enqueue!(open_set, start, 0)
    # dict is a map that gives value of current node
    # (initialised with infinity as default value) as first element, the second element
    # is an array with the path in moves to board state
    dict = DefaultDict{T, Array}([100000, []])
    dict[start] = [value_start, []]
    # parents is a map that shows the parents of each node
    parents = DefaultDict{T, Array{T, 1}}([])
    parents[start] = []
    # value_map is a map that associates each node with its value (including backprop)
    value_map = Dict{T, Float64}()
    value_map[start] = h(board, arr_start)
    # Draw first frame
    color_map = Dict()
    g = draw_subgraph(parents, graph, value_map, color_map)
    save_graph(g, "puzzle_hard_1_backprop_" * string(steps) * "_steps_frame_000")
    # Keep track of frames
    frame = 0
    # Loop over animation steps
    for step in 1:steps
        # Get state with best value
        current = dequeue!(open_set)
        f_score_old, past_moves = dict[current]
        # Move to current node
        if length(past_moves) > 0
            for move in past_moves
                make_move(board, move)
            end
        end
        # Color smallest value
        color_map = Dict(current=>""" "#00ff00" """)#, past=>""" "#0000ff" """)
        g = draw_subgraph(parents, graph, value_map, color_map)
        #display(g)
        frame += 1
        fill = frame < 10 ? "00" : frame < 100 ? "0" : ""
        save_graph(g, "puzzle_hard_1_backprop_" * string(steps) * "_steps_frame_" * fill * string(frame))
        delete!(color_map, past)
        # Get current board int-string
        arr_current = get_board_arr(board)
        # Expand current node by getting all available moves
        available_moves = get_all_available_moves(board, arr_current)
        # Smallest child value
        smallest = current
        for move in available_moves
            # Perform available move
            make_move(board, move)
            # Get board int-string
            arr_new = get_board_arr(board)
            new = board_to_int(arr_new, T)
            # Calculates g score of neighbor
            # tentative_score = f_score_old + 1
            # if tentative_score < dict[new][1]
            #     # Update score
            #     dict[new] = [tentative_score, vcat(past_moves, [move])]
            #     value_new = tentative_score + h(board, arr_new) + noise()
            #     # Update heuristic value
            #     value_map[new] = value_new
            #     # If neighbor is not yet queued to be visited, queue it
            #     if !(new in keys(open_set))
            #         # Push node onto heap
            #         enqueue!(open_set, new, value_new)
            #     end
            # end
            if !(new in keys(dict))
                # Update value
                value_new = h(board, arr_new) + noise()
                dict[new] = [value_new, vcat(past_moves, [move])]
                value_map[new] = value_new
                # Push node onto heap
                enqueue!(open_set, new, value_new)
            end
            # Update smallest value
            if value_map[new] < value_map[smallest]
                smallest = new
            end
            # Update parents
            push!(parents[new], current)
            # Update colors
            color_map[new] = """ "#ff0000" """
            undo_moves(board, [move])
        end
        # Color expanded nodes
        g = draw_subgraph(parents, graph, value_map, color_map)
        #display(g)
        frame += 1
        fill = frame < 10 ? "00" : frame < 100 ? "0" : ""
        save_graph(g, "puzzle_hard_1_backprop_" * string(steps) * "_steps_frame_" * fill * string(frame))
        # Backprop if smallest child value lower than current
        if value_map[smallest] < value_map[current]
            frame = backprop(smallest, parents, value_map, graph, frame, steps)
        end
        # Undo rest of moves back to initial board
        undo_moves(board, past_moves)
        past = current
    end
end

function backprop(child, parents, value_map, graph, frame, steps)
    """
    Takes new smallest value and backpropagates it up the tree
    """
    # List of nodes to backprop
    q = Deque{typeof(child)}()
    push!(q, child)
    # List of nodes updated so far for coloring
    updated = Set{typeof(child)}()
    # Smallest value
    value_new = value_map[child]
    # Keep backpropagating until all parents with larger
    # values have been updated
    while length(q) > 0
        node = pop!(q)
        for parent in parents[node]
            # If parent has larger value, update it
            if value_map[parent] > value_new
                # Update value
                value_map[parent] = value_new
                # Add to frontier
                push!(q, parent)
                # Draw update
                color_map = Dict(child=>""" "#00ff00" """, parent=>""" "#00ffff" """)
                for updated_node in updated
                    color_map[updated_node] = """ "#ff00ff" """
                end
                g = draw_subgraph(parents, graph, value_map, color_map)
                #display(g)
                frame += 1
                fill = frame < 10 ? "00" : frame < 100 ? "0" : ""
                save_graph(g, "puzzle_hard_1_backprop_" * string(steps) * "_steps_frame_" * fill * string(frame))
                # Add for colored path
                push!(updated, parent)
            end
        end
    end
    return frame
end

function noise()
    return 0.1*randn()
end

# board = load_data("hard_puzzle_1");
# v, graph, all_parents = traverse(board, 0, 0, noise=noise);#, h=multi_mag_size_nodes);

# g = draw_directed_tree(graph, value_map=v)#, all_parents=all_parents, all_parents=all_parents)

# board = load_data("hard_puzzle_1");
# animate_expand_astar(board, 0, 5, graph);
# animate_expand_backprop(board, 0, 111, graph);

# data = load("analysis/processed_data/filtered_data.jld2")["data"]
# Ls, dict = get_Ls(data)
# prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])]

# for subj in keys(data)
#     d = data[subj]
#     puzzle = d[d["instance"] == prbs[10]]
