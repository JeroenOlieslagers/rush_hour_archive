include("rushhour.jl")
include("solvers.jl")
include("data_analysis.jl")
using CSV
using DataFrames
using DataStructures
using ProgressBars


data = load("analysis/processed_data/filtered_data.jld2")["data"];

Ls, dict = get_Ls(data);
prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])]

# prb = "prb55384_14";
# #prb = "prb29414_11";
# #prb = "prb29027_16";
# board = load_data(prb);
# arr = get_board_arr(board);

full_heur_dict = Dict{BigInt, Array{Float64, 1}}()
for prb in ProgressBar(prbs)
    board = load_data(prb)
    heur_dict = bfs_heur(board)
    total_states = collect(keys(full_heur_dict))
    for state in keys(heur_dict)
        if state in total_states
            println("DUPLICATE STATE?????")
        end
    end
    merge!(full_heur_dict, heur_dict)
end

full_heur_dict_strings = Dict{String, Array{Float64, 1}}()
for key in keys(full_heur_dict)
    full_heur_dict_strings[string(key)] = full_heur_dict[key]
end
full_heur_dict_strings_copy = load("analysis/processed_data/full_heur_dict.jld2")

#save("analysis/processed_data/full_heur_dict.jld2", full_heur_dict_strings)

function calculate_heuristics(board, arr)
    opts = leave_one_out(board, arr)
    return [
        red_distance(board, arr), 
        mag_size_nodes(board, arr), 
        multi_mag_size_nodes(board, arr), 
        mean(opts), 
        sample(opts), 
        maximum(opts)]
end

function bfs_heur(board)
    """
    Breadth first search algorithm that gives the value of heuristic functions at every state
    """
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    start = board_to_int(arr_start, T)
    # dict is a map that gives an array with the shortest path in moves to board state
    dict = Dict{T, Array{Array{Int, 1}, 1}}()
    dict[start] = []
    # heur_dict is a map that gives heuristic values for every state
    heur_dict = Dict{T, Array{Float64, 1}}()
    heur_dict[start] = calculate_heuristics(board, arr_start)
    # q is a queue that will indicate which node to expand next
    q = Deque{T}()
    push!(q, start)
    # visited keeps track of all visited nodes
    visited = Set{T}()
    push!(visited, start)
    # count node expansions to prevent infinite loop
    expansions = 0
    available_moves = Array{Array{Int, 1}, 1}()
    while length(q) > 0
        # Exit if too many expansions occured
        if expansions > 100000
            throw(DomainError("Expansion limit reached"))
        end
        # Get next node
        current = pop!(q)
        # Visit node
        push!(visited, current)
        # Get moves to node
        past_moves = dict[current]
        # Move to current node
        for move in past_moves
            make_move!(board, move)
        end
        # Get current board int-string
        arr_current = get_board_arr(board)
        # Update heuristic values
        heur_dict[current] = calculate_heuristics(board, arr_current)
        # Check if solved
        if check_solved(arr_current)
            undo_moves!(board, past_moves)
            continue
        end
        # Expand current node by getting all available moves
        get_all_available_moves!(available_moves, board, arr_current)
        # Increment expansion counter
        expansions += 1
        for move in available_moves
            # Perform available move
            make_move!(board, move)
            # Get board int-string
            arr_new = get_board_arr(board)
            new = board_to_int(arr_new, T)
            # Add to queue if not yet seen
            if !(new in visited || new in q)
                # Shortest path to node
                dict[new] = vcat(past_moves, [move])
                # Push node onto heap
                pushfirst!(q, new)
            end
            # Undo move
            undo_moves!(board, [move])
        end
        # Undo rest of moves back to initial board
        undo_moves!(board, past_moves)
    end
    return heur_dict
end