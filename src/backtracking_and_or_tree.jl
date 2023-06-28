function backtracking_and_or_tree(d, board; max_iter=1000)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # BFS stuff
    frontier = Vector{s_type}()
    visited = Vector{s_type}()
    # list of all actions for each level of AO tree (gets cleared after each level)
    actions = Vector{Tuple{Int, Int}}()
    # main AO tree (downward direction, i.e. children)
    children = Dict{s_type, Dict{Tuple{Int, Int}, Vector{s_type}}}()
    # parents AO node for each AO node
    parents = DefaultDict{s_type, Vector{Tuple{s_type, Tuple{Int, Int}}}}([])
    # depth in AO tree for each node
    depths = Dict{s_type, Int}()
    # putting it all together
    tree = Dict(
        "children" => children,
        "parents" => parents,
        "depths" => depths
    )
    tree_type = typeof(tree)
    # parent AO node for each action
    parents_a = DefaultDict{Tuple{Int, Int}, Vector{s_type}}([])
    # return dictionaries
    state_space = DefaultDict{BigInt, Vector{BigInt}}([])
    action_space = DefaultDict{BigInt, Vector{Tuple{Int, Int}}}([])
    tree_space = DefaultDict{BigInt, Vector{tree_type}}([])
    # list of all planning trajectories
    chains = Vector{Vector{Tuple{BigInt, Tuple{Int, Int}, Int}}}()
    # records to be kept
    records = Dict(
        "state_space" => state_space,
        "action_space" => action_space,
        "tree_space" => tree_space,
        "chains" => chains
    )
    # current planning trajectory (gets copied for each recursion)
    current_chain = Vector{Tuple{BigInt, Tuple{Int, Int}, Int}}()
    # initialize process
    arr = get_board_arr(board)
    m_init = 6 - (board.cars[9].x+1)
    root = (9, (m_init,))
    s_root = board_to_int(arr, BigInt)
    push!(frontier, root)
    depths[root] = 0
    current_depth = 1
    # Expand tree and recurse
    expand_tree!(frontier, visited, d, (9, m_init), actions, parents_a, records, current_chain, board, arr, tree, current_depth, 0, root, s_root, max_iter)
    return records
end

function expand_tree!(frontier, visited, d, action, actions, parents_a, records, current_chain, board, arr, tree, current_depth, recursion_depth, root, s_next, max_iter)
    # Unpack tree components
    children = tree["children"]
    parents = tree["parents"]
    depths = tree["depths"]
    for i in 0:max_iter-1
        # if end of AO tree is reached
        if isempty(frontier)
            extend_planning_trajectories(d, actions, parents_a, records, current_chain, board, tree, current_depth, recursion_depth, root, s_next)#replan ? s_prev : s_next
            empty!(actions)
            empty!(parents_a)
            break
        end
        # current AO node
        dict = Dict{Tuple{Int, Int}, Vector{s_type}}()
        # BFS stuff
        s_old = popfirst!(frontier)
        push!(visited, s_old)
        car_id, moves = s_old
        # set depth to one layer deeper since we never go up
        depth_new = depths[s_old] + 1
        # extend planning trajectories after layer has been completed
        if depth_new > current_depth
            extend_planning_trajectories(d, actions, parents_a, records, current_chain, board, tree, current_depth, recursion_depth, root, s_next)#replan ? s_prev : s_next
            # reset actions and parents of actions
            empty!(actions)
            empty!(parents_a)
            # set new layer depth 
            current_depth = depth_new
        end
        # extend all OR nodes
        for m in reverse(moves)
            move = (car_id, m)
            # skip is move no longer valid
            if !is_valid_move(board, move)
                continue
            end
            # get all blocking cars
            cars = move_blocked_by(board.cars[car_id], m, arr)
            # if move not blocked, add as potential action
            if length(cars) == 0
                if move ∉ actions && move[1] != action[1]
                    push!(actions, move)
                end
                # add parents of each action
                # if s_old ∉ parents_a[move]
                #     push!(parents_a[move], s_old)
                # end
            end
            if s_old ∉ parents_a[move]
                push!(parents_a[move], s_old)
            end
            # list of all AND nodes
            ls = Vector{s_type}()
            for car in cars
                # Get all OR nodes of next layer
                moves_new, _ = moves_that_unblock(board.cars[car_id], board.cars[car], arr, move_amount=m)
                # Remove impossible moves with cars in same row
                moves_new = moves_new[(in).(moves_new, Ref(possible_moves(arr, board.cars[car])))]
                s_new = (car, Tuple(moves_new))
                # If no possible moves, end iteration for this move
                if length(moves_new) == 0
                    push!(ls, s_new)
                    break
                end
                # Extend frontier if new node
                if s_new ∉ visited && s_new ∉ frontier 
                    if (s_old, move) ∉ parents[s_new]
                        push!(parents[s_new], (s_old, move))
                    end
                    push!(frontier, s_new)
                    depths[s_new] = depth_new
                end
                push!(ls, s_new)
            end
            # All AND node for given OR (move) node
            dict[move] = ls
        end
        # If node concerns root car, check if moves are contained
        # in root node, don't add if so
        if car_id == 9
            if any([i ∉ root[2] for i in s_old[2]]) || root ∉ keys(tree)
                children[s_old] = dict
            end
        else
            children[s_old] = dict
        end
    end
end

"""
    extend_planning_trajectories(d, actions, parents_a, records, current_chain, board, tree, current_depth, recursion_depth, root, s_current)

Call recursion on all available actions in layer

# Arguments
- `d::Int`:                     recursion depth parameter.
- `actions::Vector`:            list of all available actions in layer.
- `parents_a::Dict`:            parent node of each action.
- `records::Dict`               records all that is necessary:
    - `state_space::Dict`:          s ➡ s'.
    - `action_space::Dict`:         s ➡ a.
    - `tree_space::Dict`:           s ➡ and-or tree.
    - `chains::Vector`:             list of all chains so far.
- `current_chain::Vector`:      current list of (s, a, code) triplets, where code is:
                                    1: normal
                                    -1: recursion depth reached
                                    -2: invalid action
                                    -3: dead end
- `board::Board`:               current game board
- `tree::Dict`:                 unpruned tree
    - `children::Dict`:             main tree structure (a.k.a. children)
    - `parents::Dict`:              inverted and-or tree
    - `depths::Dict`:               depth of each node in and-or tree
- `current_depth::Int`:         depth of current node
- `recursion_depth::Int`:       depth of recursion
- `root::BigInt`:               root state
- `s_current::BigInt`:          current state
"""
function extend_planning_trajectories(d, actions, parents_a, records, current_chain, board, tree, current_depth, recursion_depth, root, s_current)
    parents = tree["parents"]
    # Recurse each action in layer
    for a in actions
        # Recurse each parent of that action (since this gets cleared after
        # each layer, it is mostyl just one parent)
        for s in parents_a[a]
            # Recurse each parent node of the current node (multiple branches)
            for s_startt in parents[s]
                # Only recurse if triplet has not already been visited
                if (s_current, a, 1) ∉ current_chain
                    backtrack!(d, records, copy(current_chain), board, deepcopy(tree), current_depth-1, s_startt, s, a, recursion_depth + 1, root)
                end
            end
        end
    end
end

"""
    copy_dict!(new, old)

Copy content of old dict into new dict (without checking for duplicates)
"""
function copy_dict!(new, old)
    for k in keys(old)
        for item in old[k]
            #if item ∉ new[k]
            push!(new[k], item)
            #end
        end
    end
end

"""
    update_records!(records, s_prev, s_next, action, tree, code, current_chain; new_chain=false)

Update records. If new_chain is set to true, copy current chain into chains

- `records::Dict`               records all that is necessary:
    - `state_space::Dict`:          s ➡ s'.
    - `action_space::Dict`:         s ➡ a.
    - `tree_space::Dict`:           s ➡ and-or tree.
    - `chains::Vector`:             list of all chains so far.
"""
function update_records!(records, s_prev, s_next, action, tree, code, current_chain; new_chain=false)
    state_space = records["state_space"]
    action_space = records["action_space"]
    tree_space = records["tree_space"]
    push!(state_space[s_prev], s_next)
    push!(action_space[s_prev], action)
    push!(tree_space[s_prev], tree)
    push!(current_chain, (s_prev, action, code))
    if new_chain
        chains = records["chains"]
        push!(chains, copy(current_chain))
    end
end

"""
    check_move_violates(board, action)

Check whether action is possible on board
"""
function check_move_violates(board, action)
    arr_o = get_board_arr(board)
    if move_has_overlap(arr_o, board, action)
        return arr_o, true
    end
    make_move!(board, action)
    if !is_valid_board(board)
        undo_moves!(board, [action])
        return arr_o, true
    end
    arr = get_board_arr(board)
    if board_has_overlap(arr, board)
        undo_moves!(board, [action])
        return arr_o, true
    end
    return arr, false
end

"""
    prune_tree!(tree_c, s_old, s_start, actions, parents_a, current_depth)

Prune tree by removing nodes deeper than current depth as well as current node.
Also add available actions to list
"""
function prune_tree!(tree_c, s_old, s_start, actions, parents_a, current_depth)
    children = tree_c["children"]
    parents = tree_c["parents"]
    depths = tree_c["depths"]
    # Remove current node
    delete!(children, s_old)
    delete!(parents, s_old)
    delete!(depths, s_old)
    for a in keys(children[s_start[1]])
        for (n, s) in enumerate(children[s_start[1]][a])
            if s == s_old
                # Add actions that have become immediately possible to list
                if length(unique(children[s_start[1]][a])) == 1
                    push!(actions, a)
                    push!(parents_a[a], s_start[1])
                end
                # Remove current node from parent node
                deleteat!(children[s_start[1]][a], n)
            end
        end
    end
    # Remove nodes deeper than current_depth
    for s in keys(depths)
        if depths[s] > current_depth
            delete!(depths, s)
            delete!(children, s)
            delete!(parents, s)
        end
    end
end


"""
    backtrack!(d, records, current_chain, board, tree, current_depth, s_start, s_old, action, recursion_depth, root; max_iter=1000)

Call recursion on all available actions in layer

# Arguments
- `d::Int`:                     recursion depth parameter
- `records::Dict`               records all that is necessary:
    - `state_space::Dict`:          s ➡ s'.
    - `action_space::Dict`:         s ➡ a.
    - `tree_space::Dict`:           s ➡ and-or tree.
    - `chains::Vector`:             list of all chains so far.
- `current_chain::Vector`:      current list of (s, a, code) triplets, where code is:
                                    1: normal
                                    -1: recursion depth reached
                                    -2: invalid action
                                    -3: dead end
- `board::Board`:               current game board
- `tree::Dict`:                 unpruned tree
    - `children::Dict`:             main tree structure (a.k.a. children)
    - `parents::Dict`:              inverted and-or tree
    - `depths::Dict`:               depth of each node in and-or tree
- `current_depth::Int`:         depth of current node
- `s_start::stype`:             node where to start backtracking from
- `s_old::stype`:               node that corresponds to action
- `action::Tuple{Int, Int}`:    action to be taken
- `recursion_depth::Int`:       depth of recursion
- `root::BigInt`:               root state
"""
function backtrack!(d, records, current_chain, board, tree, current_depth, s_start, s_old, action, recursion_depth, root; max_iter=1000)
    # Check if depths match
    if current_depth != tree["depths"][s_old]
        throw(DomainError("Depths do not match"))
        return nothing
    end
    # Get previous state for records
    s_prev = board_to_int(get_board_arr(board), BigInt)
    # Stop if recursion depth is reached
    if recursion_depth > d
        update_records!(records, s_prev, -1, action, tree, -1, current_chain; new_chain=true)
        return nothing
    end
    # Type of AND-OR node
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # Initialise search
    frontier = Vector{s_type}()
    parents_a = DefaultDict{Tuple{Int, Int}, Vector{s_type}}([])
    actions = Vector{Tuple{Int, Int}}()
    # Add to tree without overriding previous tree
    tree_c = deepcopy(tree)
    children = tree_c["children"]
    parents = tree_c["parents"]
    depths = tree_c["depths"]
    # Check for violations
    arr, replan = check_move_violates(board, action)
    if replan
        # update_records!(records, s_prev, -2, action, tree, -2, current_chain; new_chain=true)
        #return nothing
        @goto replan
    end
    s_next = board_to_int(arr, BigInt)
    # Prune tree
    prune_tree!(tree_c, s_old, s_start, actions, parents_a, current_depth)
    # If parent node is root and root is completely free
    if s_start[1] == root && all([isempty(children[root][a]) for a in keys(children[root])])
        # Actually correctly solved
        if check_solved(arr)
            undo_moves!(board, [action])
            update_records!(records, s_prev, s_next, action, tree, 1, current_chain)
            # Final node
            update_records!(records, s_next, 0, (0, 0), tree_c, 0, current_chain; new_chain=true)
            return nothing
        else # Replan from root
            @label replan
            # Recalculate root node
            m_init = 6 - (board.cars[9].x+1)
            root = (9, (m_init,))
            push!(frontier, root)
            # Reset tree
            empty!(children)
            empty!(parents)
            empty!(depths)
            depths[root] = 0
            current_depth = 1
            empty!(actions)
        end
    end
    if !replan
        # If not replanning, add to records
        update_records!(records, s_prev, s_next, action, tree, 1, current_chain)
    end

    visited = collect(keys(children))
    # If not replanning but no available actions
    # extend frontier with children of parents of current node
    if isempty(actions) && isempty(frontier)
        current_depth += 1
        for s in children[s_start[1]][s_start[2]]
            if depths[s] == current_depth-1
                push!(frontier, s)
                if s ∉ visited
                    push!(visited, s)
                end
            end
        end
    end
    # Expand and recurse
    expand_tree!(frontier, visited, d, action, actions, parents_a, records, current_chain, board, arr, tree_c, current_depth, recursion_depth, root, replan ? s_prev : s_next, max_iter)
    if !replan
        undo_moves!(board, [action])
        # Dead end case
        if isempty(records["state_space"][s_next])
            rev_a = (action[1], -action[2])
            update_records!(records, s_next, -3, rev_a, tree_c, -3, current_chain; new_chain=true)
        end
    end
    return nothing
end

prb = prbs[sp[1]]

board = load_data(prb);
records = backtracking_and_or_tree(30, board);
draw_board(get_board_arr(board))

solution_chains = []
for chain in chains
    if chain[end][3] == 0
        push!(solution_chains, chain)
    end
end


board = arr_to_board(int_to_arr(sss));
board = load_data("prb26567_7");
board = load_data(prbs[end]);
board = load_data(prbs[40]);
board = load_data(prbs[22]);
board = load_data(prbs[1]);
make_move!(board, (2, 1))
make_move!(board, (8, 1))

root = board_to_int(get_board_arr(board), BigInt)

sols = []
for s in keys(IDV[prb])
    if IDV[prb][s][1] == 0
        push!(sols, s)
    end
end
subj_states = []
for subj in subjs
    if prb in keys(states_all[subj])
        push!(subj_states, states_all[subj][prb])
    end
end
g = draw_backtrack_state_space(records["state_space"], records["action_space"], board, root, IDV[prb], optimal_a[prb]; highlight_nodes=sols, full=true)#, subj_states=subj_states)

g = new_draw_tree(tt, board, (9, (4,)))

for chain in chains
    if chain[3][2] == (8, 4)
        println(chain)
    end
end

fms = []
for subj in subjs
    if prb in keys(moves_all[subj])
        push!(fms, moves_all[subj][prb][1])
    end
end

for move in solution_chains[1][1:end-1]
    make_move!(board, move[2])
    display(draw_board(get_board_arr(board)))
end

for s in keys(state_space)
    if sw in state_space[s]
        println(s)
    end
end

ls = []
for prb in prbs
    push!(ls, length(IDV[prb]))
end
sp = sortperm(ls)


## CHECKS
# for s in keys(state_space)
#     if s ∉ collect(keys(IDV[prbs[1]]))
#         display(int_to_arr(s))
#         #println(s)
#     end
# end
# CHECKS FOR DUPLICATES
# for s in keys(state_space)
#     if !(length(state_space[s]) == length(action_space[s]) == length(tree_space[s]))
#         println(s)
#     end
# end
# visited = []
# for s in keys(action_space)
#     for n in eachindex(action_space[s])
#         if state_space[s][n] != -1
#             push!(visited, (s, action_space[s][n], tree_space[s][n]))
#         end
#     end
# end
# if length(visited) == length(unique(visited))
#     println("yas")
# end



