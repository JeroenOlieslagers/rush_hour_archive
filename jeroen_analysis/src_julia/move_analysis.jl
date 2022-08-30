include("rushhour.jl")
include("solvers.jl")
include("plot_graph.jl")


function traverse(board, value_start, γ; h=(x,y)->0, noise=randn)
    """
    Starts at a node and expands with probability γ. 
    When finished, returns best heuristic value encountered.
    h(x, y) is the heuristic function that will give the value to the nodes
    noise is the function that will add stochasticity to the heuristic value function
    """
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
    value_map = Dict{T, Int}()
    value_map[start] = value_start
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
                value_map[new] = heuristic_value
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
    println("Expansions: "* string(expansions))
    return minimum(values(value_map)), parents, all_parents
end


function noise()
    return 0#.001*randn()
end

board = load_data("hard_puzzle_11")
v, graph, all_parents = traverse(board, 999, 0.01, h=multi_mag_size_nodes, noise=noise);

g = draw_directed_tree(graph, all_parents=all_parents)