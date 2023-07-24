include("engine.jl")
include("data_analysis.jl")

"""
    is_acyclic(graph, start)

Check whether there are cycles in graph by doing topological sort.
Graph must be in dictionary format (s -> s')
"""
function is_acyclic(g, T; max_iters=100000)
    graph = deepcopy(g)
    clean_graph!(graph)
    inverted_graph = invert_tree(graph, T)
    S = []
    for n in keys(graph)
        if n ∉ keys(inverted_graph)
            push!(S, n)
        end
    end
    L = []
    for i in 1:max_iters
        if isempty(S)
            break
        end
        n = popfirst!(S)
        push!(L, n)
        ls = deepcopy(graph[n])
        for m in ls
            deleteat!(graph[n], findfirst(x->x==m, graph[n]))
            deleteat!(inverted_graph[m], findfirst(x->x==n, inverted_graph[m])) 
            if isempty(inverted_graph[m])
                push!(S, m)
            end
        end
    end
    for n in keys(graph)
        if !isempty(graph[n])
            return false
        end
    end
    for n in keys(inverted_graph)
        if !isempty(inverted_graph[n])
            return false
        end
    end
    return true
end


function clean_graph!(graph)
    for s in keys(graph)
        ls = deepcopy(graph[s])
        for ss in ls
            if typeof(s) == BigInt
                if ss <= 0
                    deleteat!(graph[s], findfirst(x->x==ss, graph[s]))
                end
            else
                if ss[1] <= 0
                    deleteat!(graph[s], findfirst(x->x==ss, graph[s]))
                end
            end
        end
    end
    return nothing
end

function invert_tree(graph, T)
    inverted_graph = DefaultDict{T, Vector{T}}([])
    for s in keys(graph)
        for ss in graph[s]
            push!(inverted_graph[ss], s)
        end
    end
    return inverted_graph
end

function backtracking_forest(d, board; max_iter=1000)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # main AO tree (downward direction, i.e. children)
    tree = Vector{a_type}()
    # keeps track of current AO nodes visited
    visited_states = Vector{s_type}()
    # return dictionaries
    state_space = DefaultDict{BigInt, Vector{BigInt}}([])
    action_space = DefaultDict{BigInt, Vector{a_type}}([])
    # points each (s, parent, child) triplet to a chain and the position in that chain
    chain_dict = DefaultDict{Tuple{BigInt, Vector{s_type}, a_type}, Vector{Tuple{BigInt, Vector{s_type}, a_type}}}([])
    #chain_dict = DefaultDict{Tuple{BigInt, s_type, a_type}, Vector{Tuple{BigInt, s_type, a_type}}}([])
    # records to be kept
    records = Dict(
        "state_space" => state_space,
        "action_space" => action_space,
        "chain_dict" => chain_dict,
    )
    # current planning trajectory (gets copied for each recursion)
    current_chain = Vector{Tuple{BigInt, a_type}}()
    # initialize process
    arr = get_board_arr(board)
    s_root = board_to_int(arr, BigInt)
    recalculate_root!(tree, visited_states, board)
    start_triplet = (s_root, [(0, (0,))], (0, 0))
    #start_triplet = (s_root, (0, (0,)), (0, 0))
    # Expand tree and recurse
    expand_tree!(tree, visited_states, d, records, current_chain, board, arr, 0, s_root, start_triplet, 0, max_iter)
    return records
end

"""
    expand_tree!(tree, visited_states, d, records, current_chain, board, arr, recursion_depth, s_prev, iter, max_iter)

Starts with frontier node and list of actions to expand and-or tree.
If frontier is empty and actions are present, simply extend those, 
otherwise expend tree starting with nodes in frontier
"""
function expand_tree!(tree, visited_states, d, records, current_chain, board, arr, recursion_depth, s_prev, prev_triplet, iter, max_iter)
    move = tree[end]
    # get parent node
    parent_ao = visited_states[end-1]
    # if recursion_depth == 2 && move != (8, 4)
    #     return nothing
    # end
    # reach depth d
    if recursion_depth > d
        update_records!(records, prev_triplet, (-1, [(-1, (-1,))], (-1 ,-1)), current_chain)
        #update_records!(records, prev_triplet, (-1, (-1, (-1,)), (-1 ,-1)), current_chain)
        return nothing
    end
    # If move is invalid, look at parent and re-calculate
    if !is_valid_move(board, move)
        pop!(tree)
        pop!(visited_states)
        if isempty(tree)
            recalculate_root!(tree, visited_states, board)
        end
        expand_tree!(tree, visited_states, d, records, current_chain, board, arr, recursion_depth + 1, s_prev, prev_triplet, 0, max_iter)
        return nothing
    end
    # If move is unblocked, take it and travel up tree
    # (if move not already in chain)
    if !move_has_overlap(arr, board, move)#length(cars) == 0
        if (s_prev, move) ∉ current_chain
            # if move is undo -> dead end
            #if prev_triplet[3] == (move[1], -move[2])
            if prev_triplet[3] == (move[1], -move[2])#prev_triplet[3][1] == move[1]
                update_records!(records, prev_triplet, (-3, [(-3, (-3,))], (-3, -3)), current_chain)
                #update_records!(records, prev_triplet, (-3, (-3, (-3,)), (-3, -3)), current_chain)
                return nothing
            end
            pop!(tree)
            pop!(visited_states)
            # reached root node
            if isempty(tree)
                # solved
                if check_solved(arr)
                    update_records!(records, prev_triplet, (0, [(0, (0,))], (0, 0)), current_chain)
                    #update_records!(records, prev_triplet, (0, (0, (0,)), (0, 0)), current_chain)
                    return nothing
                else # not solved but at the root node
                    recalculate_root!(tree, visited_states, board)
                end
                expand_tree!(tree, visited_states, d, records, current_chain, board, arr, recursion_depth + 1, s_prev, prev_triplet, 0, max_iter)
                return nothing
            end
            # commit move
            make_move!(board, move)
            arr = get_board_arr(board)
            s_next = board_to_int(arr, BigInt)
            # this check shouldn't be necessary
            if board_has_overlap(arr, board)
                throw(DomainError("board has overlap"))
            end
            next_triplet = (s_prev, copy(visited_states), move)
            #next_triplet = (s_next, parent_ao, move)
            if next_triplet ∉ records["chain_dict"][prev_triplet]
                update_records!(records, prev_triplet, next_triplet, current_chain)
                # next iteration up
                expand_tree!(tree, visited_states, d, records, current_chain, board, arr, recursion_depth + 1, s_next, next_triplet, 0, max_iter)
            end
            # undo move to ensure board does not get tangled
            undo_moves!(board, [move])
        end
        return nothing
    end
    car_id, m = move
    # get all blocking cars
    cars = move_blocked_by(board.cars[car_id], m, arr)
    for car in cars
        # Get all OR nodes of next layer
        ms_new, _ = moves_that_unblock(board.cars[car_id], board.cars[car], arr, move_amount=m)
        # Remove impossible moves with cars in same row
        ms_new = ms_new[(in).(ms_new, Ref(possible_moves(arr, board.cars[car])))]
        # If no possible moves, end iteration for this move
        if length(ms_new) == 0
            return nothing
        end
        # new ao state
        ao_new = (car, Tuple(ms_new))
        # Extend tree if new node
        if ao_new ∉ visited_states
            # check all moves for blockages
            for m_new in ms_new
                if iter < max_iter
                    extend_planning_trajectories!((car, m_new), ao_new, copy(tree), copy(visited_states), d, records, copy(current_chain), board, arr, recursion_depth, s_prev, prev_triplet, iter, max_iter)
                else
                    throw(DomainError("max_iter depth reached"))
                end
            end
        end
    end
    return nothing
end

function extend_planning_trajectories!(move_new, ao_new, tree, visited_states, d, records, current_chain, board, arr, recursion_depth, s_prev, prev_triplet, iter, max_iter)
    # grow tree
    push!(tree, move_new)
    push!(visited_states, ao_new)
    # expand new layer
    expand_tree!(tree, visited_states, d, records, current_chain, board, arr, recursion_depth, s_prev, prev_triplet, iter + 1, max_iter)
    return nothing
end

"""
    recalculate_root!(tree, board)

Recalculates root move and adds it to tree
"""
function recalculate_root!(tree, visited_states, board)
    # Recalculate root
    m_init = 6 - (board.cars[9].x+1)
    ao_root = (9, (m_init,))
    root = (9, m_init)
    push!(tree, root)
    push!(visited_states, (0, (0,)))
    push!(visited_states, ao_root)
    return nothing
end

"""
    update_records!(records, prev_triplet, next_triplet, current_chain)

Update records and update current chain

- `records::Dict`               records all that is necessary:
    - `state_space::Dict`:          s ➡ s'.
    - `action_space::Dict`:         s ➡ a.
    - `chain_dict::Dict`:           (s, ao_parent, a) ➡ (s', ao_parent', a').
"""
function update_records!(records, prev_triplet, next_triplet, current_chain)
    # unpacking
    s_prev = prev_triplet[1]
    move = next_triplet[3]
    s_next = next_triplet[1]
    # update records
    state_space = records["state_space"]
    action_space = records["action_space"]
    chain_dict = records["chain_dict"]
    if s_next ∉ state_space[s_prev]
        push!(state_space[s_prev], s_next)
    end
    if move ∉ action_space[s_prev]
        push!(action_space[s_prev], move)
    end
    push!(chain_dict[prev_triplet], next_triplet)
    # update current chain
    push!(current_chain, (s_prev, move))
    return nothing
end


#prb = prbs[sp[24]]
prb = prbs[sp[1]]
board = load_data(prb);
@time records = backtracking_forest(1000, board; max_iter=1000);
chain_dict = records["chain_dict"]

root = board_to_int(get_board_arr(board), BigInt)
rc = (root, [(0, (0,))], (0, 0))
#g = new_draw_chain_dict(records["chain_dict"], board, root, optimal_a[prb]; full=true, highlight_nodes=sols)

g = draw_backtrack_state_space(records["state_space"], records["action_space"], board, root, IDV[prb], optimal_a[prb]; full=true)
draw_board(get_board_arr(board))

g = new_draw_chain_dict(chain_dict, board, root)

sols = []
for s in keys(IDV[prb])
    if IDV[prb][s][1] == 0
        push!(sols, s)
    end
end


ls = []
for prb in prbs
    push!(ls, length(IDV[prb]))
end
sp = sortperm(ls)


# SUBJECT STUFF
subj_states = []
for subj in subjs
    if prb in keys(states_all[subj])
        push!(subj_states, states_all[subj][prb])
    end
end
# first moves
fms = []
for subj in subjs
    if prb in keys(moves_all[subj])
        push!(fms, moves_all[subj][prb][1])
    end
end






