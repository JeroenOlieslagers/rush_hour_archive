using DataFrames
using JLD2
include("rush_hour.jl")

"""
    analyse_subject(subject_data)

Take in DataFrame of subject data, filter it and return dicts where keys are puzzle IDs
and values are moves, states, time stamps, attempts
"""
function analyse_subject(subject_data)
    # Get problem sets
    probs = subject_data.instance
    uniq = unique(probs)
    # Get timestamps
    tt = (subject_data.t)./(1000*60)
    tt .-= first(tt)
    # Get negative time points (BONUS_FAIL/BONUS_SUCCESS)
    negs = findall(x->x<0, tt)
    # Get moves data
    moves = subject_data.move
    targets = subject_data.target
    pieces = subject_data.piece
    events = subject_data.event
    times = subject_data.t
    # Only include move end and restart data (tells us everything we need)
    excl_events = findall(x->x∉["drag_end", "restart"], events)
    # return dicts
    tot_move_tuples = Dict{String, Array{Tuple{Int, Int}, 1}}()
    tot_states_visited = Dict{String, Array{BigInt, 1}}()
    tot_times = Dict{String, Array{Int, 1}}()
    # Record number of moves per puzzle per attempt
    attempts = Dict{String, Int}()
    for uni in uniq
        # Get indices that correspond to the current problem instance
        idx = findall(x->x==uni, probs)
        # Find first index of specific puzzle
        iddx = findfirst(x->x=="start", events[idx])
        # start time of puzzle
        start_time = 0
        if iddx !== nothing
            start_time = times[idx[iddx]]
        end
        # Remove indices that are BONUS_FAIL/BONUS_SUCCESS and not drag_end
        filter!(x->x∉negs, idx)
        # Remove anything that isn't drag_end or restart
        filter!(x->x∉excl_events, idx)
        # Get moves of current problem
        all_moves = moves[idx]
        # If empty, skip
        if length(all_moves) == 0
            continue
        end
        # Remove moves that do not move any cars
        excl_moves = Array{Int, 1}()
        for (index, move) in enumerate(all_moves)
            # If move number does not change, exclude it
            if index < length(all_moves) && move == all_moves[index+1]
                push!(excl_moves, idx[index])
            end
        end
        filter!(x->x∉excl_moves, idx)
        # Get final move data
        prob_moves = moves[idx]
        prob_targets = targets[idx]
        prob_pieces = pieces[idx]
        prob_events = events[idx]
        prob_times = times[idx]

        if idx[end]+1 <= length(events)
            if events[idx[end]+1] != "win"
                continue
            end
        else
            #continue
            if events[idx[end]] != "win"
                continue
            end
        end
        # Add to attempts dictionary to keep track of total number of moves per attempt
        # INCLUDES RESTARTING AS A MOVE
        attempts[uni] = length(idx)
        # Find restarts
        restarts = findall(x->x=="restart", prob_events)
        # Add restart at beginning and end to create correct intervals
        pushfirst!(restarts, 0)
        push!(restarts, length(prob_moves))
        # Collect data across restarts
        move_tuples = Array{Tuple{Int, Int}, 1}()
        states_visited = Array{BigInt, 1}()
        times_prb = Array{Int, 1}()
        init_time = start_time
        # Loop over attempts
        for i in 1:length(restarts)-1
            # Get indices for intervals between restarts
            idx = (restarts[i]+1):(restarts[i+1]-1)
            # # Add to attempts dictionary to keep track of total number of moves per attempt
            # attempts[uni][i] = length(idx)
            # Initialise board
            board = load_data(uni)
            if i > 1
                init_time = prob_times[idx[1]-1]
            end
            # Loop over moves
            for j in idx
                # If start or restart, reload problem
                move, target, piece, time = prob_moves[j], prob_targets[j], prob_pieces[j], prob_times[j]
                if move == 0
                    # Load problem
                    board = load_data(uni)
                    arr = get_board_arr(board)
                    push!(move_tuples, (-1, 0))
                    push!(states_visited, board_to_int(arr, BigInt))
                    push!(times_prb, 0)
                end
                # Get moving piece
                move_piece = piece + 1
                # Get move amount
                m = get_move_amount(move_piece, target, board)
                # Make move
                make_move!(board, (move_piece, m))
                arr = get_board_arr(board)
                push!(move_tuples, (move_piece, m))
                push!(states_visited, board_to_int(arr, BigInt))
                push!(times_prb, time - init_time)
            end
        end
        tot_move_tuples[uni] = move_tuples
        tot_states_visited[uni] = states_visited
        tot_times[uni] = times_prb
    end
    return tot_move_tuples, tot_states_visited, tot_times, attempts
end

"""
    get_all_subj_moves(data)

Return four dictionaries with structure 
{subj:
  {prb: 
    [
      [values for first attempt]
      [values for second attempt]
      ...
    ]
  }
}
where values are lists of: 
- moves (Tuple{Int, Int})
- states (BigInt)
- times (Int)
"""
function get_all_subj_moves(data)
    all_subj_moves = Dict{String, Dict{String, Vector{Tuple{Int, Int}}}}()
    all_subj_states = Dict{String, Dict{String, Vector{BigInt}}}()
    all_subj_times = Dict{String, Dict{String, Vector{Int}}}()
    Ls = DefaultDict{String, Array{Int}}([])
    for subj in collect(keys(data))
        tot_moves, tot_states_visited, tot_times, attempts = analyse_subject(data[subj]);
        all_subj_moves[subj] = tot_moves
        all_subj_states[subj] = tot_states_visited
        all_subj_times[subj] = tot_times
        for prb in keys(attempts)
            push!(Ls[prb], attempts[prb])
        end
    end
    return all_subj_moves, all_subj_states, all_subj_times, Ls
end


# LOADING VARIABLES THAT I ALWAYS NEED

data = load("data/processed_data/filtered_data.jld2")["data"];
all_subj_moves, all_subj_states, all_subj_times, Ls = get_all_subj_moves(data);
prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])];
subjs = collect(keys(data));