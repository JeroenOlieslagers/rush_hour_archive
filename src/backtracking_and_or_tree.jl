include("engine.jl")
include("data_analysis.jl")

function backtracking_and_or_tree(d, board; max_iter=1000)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # BFS stuff
    frontier = Vector{s_type}()
    visited = Vector{s_type}()
    # list of all actions for each level of AO tree (gets cleared after each level)
    actions = Vector{a_type}()
    # main AO tree (downward direction, i.e. children)
    children = Dict{s_type, Dict{a_type, Vector{s_type}}}()
    # parents AO node for each AO node
    parents = DefaultDict{s_type, Vector{Tuple{s_type, a_type}}}([])
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
    parents_a = DefaultDict{a_type, Vector{s_type}}([])
    # return dictionaries
    state_space = DefaultDict{BigInt, Vector{BigInt}}([])
    action_space = DefaultDict{BigInt, Vector{a_type}}([])
    #tree_space = DefaultDict{BigInt, Vector{tree_type}}([])
    # list of all planning trajectories
    chains = Vector{Vector{Tuple{BigInt, a_type, Int}}}()
    # points each (s, a, tree) triplet to a chain and the position in that chain
    #chain_dict = Dict{Tuple{BigInt, s_type, a_type}, Tuple{Int, Int}}()
    chain_dict = DefaultDict{Tuple{BigInt, s_type, a_type}, Vector{Tuple{BigInt, s_type, a_type}}}([])
    # records to be kept
    records = Dict(
        "state_space" => state_space,
        "action_space" => action_space,
        #"tree_space" => tree_space,
        #"chains" => chains
    )
    # current planning trajectory (gets copied for each recursion)
    current_chain = Vector{Tuple{BigInt, a_type, Int}}()
    # initialize process
    arr = get_board_arr(board)
    s_root = board_to_int(arr, BigInt)
    root, current_depth = initialize_tree(board, frontier, children, parents, depths, actions)
    # Expand tree and recurse
    expand_tree!(frontier, visited, d, (root[1], root[2][1]), actions, parents_a, records, chain_dict, current_chain, board, arr, tree, current_depth, 0, root, s_root, 0, ((0, (0,)), (0, 0)), max_iter)
    return records, chain_dict
end

"""
    expand_tree!(frontier, visited, d, action, actions, parents_a, records, current_chain, board, arr, tree, current_depth, recursion_depth, root, s_next, max_iter)

Starts with frontier node and list of actions to expand and-or tree.
If frontier is empty and actions are present, simply extend those, 
otherwise expend tree starting with nodes in frontier
"""
function expand_tree!(frontier, visited, d, action, actions, parents_a, records, chain_dict, current_chain, board, arr, tree, current_depth, recursion_depth, root, s_next, s_prev, s_start, max_iter)
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    a_type = Tuple{Int, Int}
    # Unpack tree components
    children = tree["children"]
    parents = tree["parents"]
    depths = tree["depths"]
    for i in 0:max_iter-1
        # if end of AO tree is reached
        if isempty(frontier)
            if !isempty(actions)
                extend_planning_trajectories(d, actions, parents_a, records, chain_dict, current_chain, board, tree, current_depth, recursion_depth, root, s_next, s_prev, action, s_start)#replan ? s_prev : s_next
            end
            empty!(actions)
            empty!(parents_a)
            break
        end
        # current AO node
        dict = Dict{a_type, Vector{s_type}}()
        # BFS stuff
        s_old = popfirst!(frontier)
        push!(visited, s_old)
        car_id, moves = s_old
        # set depth to one layer deeper since we never go up
        depth_new = depths[s_old] + 1
        # extend planning trajectories after layer has been completed
        if depth_new > current_depth
            if !isempty(actions)
                extend_planning_trajectories(d, actions, parents_a, records, chain_dict, current_chain, board, tree, current_depth, recursion_depth, root, s_next, s_prev, action, s_start)#replan ? s_prev : s_next
            end
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
                if s_old ∉ parents_a[move]
                    push!(parents_a[move], s_old)
                end
            end
            # if s_old ∉ parents_a[move]
            #     push!(parents_a[move], s_old)
            # end
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
                    # if (s_old, move) ∉ parents[s_new]
                    #     push!(parents[s_new], (s_old, move))
                    # end
                    push!(frontier, s_new)
                    depths[s_new] = depth_new
                end
                if (s_old, move) ∉ parents[s_new]
                    push!(parents[s_new], (s_old, move))
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
    initialize_tree(board, frontier, children, parents, depths, actions)

Recalculates root and empties tree
"""
function initialize_tree(board, frontier, children, parents, depths, actions)
    # Recalculate root
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
    return root, current_depth
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
                                    0: solved
                                    1: re-expand
                                    2: follow-through
                                    3: other chain pointer
                                    -1: recursion depth reached
                                    -2: replanning
                                    -3: dead end
                                    -4: invalid action
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
function extend_planning_trajectories(d, actions, parents_a, records, chain_dict, current_chain, board, tree, current_depth, recursion_depth, root, s_current, s_prev, action, s_startt)
    parents = tree["parents"]
    children = tree["children"]
    no_code_chain = [ch[1:2] for ch in current_chain]
    no_code_chainn = [ch[2] for ch in current_chain]
    # Recurse each action in layer
    for a in actions
        # if recursion_depth == 2 && a != (8, 4)
        #     continue
        # end
        # Recurse each parent of that action (since this gets cleared after
        # each layer, it is mostyl just one parent)
        for s in parents_a[a]
            # Since parents has (AND, OR) node structure, ensure
            # each AND node is only recursed once
            visited_and = []
            # Recurse each parent node of the current node (multiple branches)
            for s_start in parents[s]
                # Only recurse if tuple has not already been visited
                if (s_current, a) ∉ no_code_chain && s_start ∉ visited_and && s_start[1] ∈ keys(children)
                    #if (s_current, s_start[1], a) ∉ keys(chain_dict)
                        # update chain pointers
                        #chain_dict[(s_current, s_start[1], a)] = (length(records["chains"])+1, length(current_chain)+1)
                    if (s_current, s_start[1], a) ∉ chain_dict[(s_prev, s_startt[1], action)]
                        push!(chain_dict[(s_prev, s_startt[1], action)], (s_current, s_start[1], a))
                        push!(visited_and, s_start)
                        backtrack!(d, records, chain_dict, copy(current_chain), board, deepcopy(tree), current_depth-1, s_start, s, a, recursion_depth + 1, root)
                    end
                    #else
                        #pointer = chain_dict[(s_current, s_start[1], a)]
                        #update_records!(records, nothing, nothing, pointer, nothing, 3, copy(current_chain); new_chain=true, create_pointer=true)
                    #end
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
function update_records!(records, s_prev, s_next, action, tree, code, current_chain; new_chain=false, create_pointer=false)
    if !create_pointer
        state_space = records["state_space"]
        action_space = records["action_space"]
        #tree_space = records["tree_space"]
        push!(state_space[s_prev], s_next)
        push!(action_space[s_prev], action)
        #push!(tree_space[s_prev], tree)
        push!(current_chain, (s_prev, action, code))
        # if new_chain
        #     chains = records["chains"]
        #     push!(chains, copy(current_chain))
        # end
    else
        push!(current_chain, (-1, action, code))
        # if new_chain
        #     chains = records["chains"]
        #     #if current_chain ∉ chains
        #     push!(chains, copy(current_chain))
        #     #end
        # end
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
function prune_tree!(tree_c, s_old, s_start, actions, parents_a, current_depth, replan)
    children = tree_c["children"]
    parents = tree_c["parents"]
    depths = tree_c["depths"]
    # Remove current node
    delete!(children, s_old)
    # If replanning, leave current node in tree for re-expansion
    if !replan
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
                                    0: solved
                                    1: re-expand
                                    2: follow-through
                                    3: other chain pointer
                                    -1: recursion depth reached
                                    -2: replanning
                                    -3: dead end
                                    -4: invalid action
- `board::Board`:               current game board
- `tree::Dict`:                 unpruned tree
    - `children::Dict`:             main tree structure (a.k.a. children)
    - `parents::Dict`:              inverted and-or tree
    - `depths::Dict`:               depth of each node in and-or tree
- `current_depth::Int`:         depth of current node
- `s_start::stype`:             node where to start backtracking from
- `s_old::stype`:               node that corresponds to action
- `action::atype`:    action to be taken
- `recursion_depth::Int`:       depth of recursion
- `root::BigInt`:               root state
"""
function backtrack!(d, records, chain_dict, current_chain, board, tree, current_depth, s_start, s_old, action, recursion_depth, root; max_iter=1000)
    # Check if depths match
    if current_depth != tree["depths"][s_start[1]]+1#s_old
        #throw(DomainError("Depths do not match"))
        #return nothing
        #current_depth = tree["depths"][s_old]
        current_depth = tree["depths"][s_start[1]]+1
    end
    # Get previous state for records
    s_prev = board_to_int(get_board_arr(board), BigInt)
    # Stop if recursion depth is reached
    if recursion_depth > d
        update_records!(records, s_prev, -1, action, tree, -1, current_chain; new_chain=true)
        push!(chain_dict[(s_prev, s_start[1], action)], (-1, (-1, (-1,)), (-1, -1)))
        return nothing
    end
    # Type of AND-OR node
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    a_type = Tuple{Int, Int}
    # Initialise search
    frontier = Vector{s_type}()
    parents_a = DefaultDict{a_type, Vector{s_type}}([])
    actions = Vector{a_type}()
    # Add to tree without overriding previous tree
    tree_c = tree#deepcopy(tree)
    children = tree_c["children"]
    parents = tree_c["parents"]
    depths = tree_c["depths"]
    # Check for violations
    arr, replan = check_move_violates(board, action)
    s_next = board_to_int(arr, BigInt)
    # Prune tree
    prune_tree!(tree_c, s_old, s_start, actions, parents_a, current_depth, replan)
    if replan
        # update_records!(records, s_prev, -4, action, tree, -4, current_chain; new_chain=true)
        #return nothing
        # Restart expansion from current node
        push!(frontier, s_old)
        #@goto replan
    end
    # If parent node is root and root is completely free
    if s_start[1] == root && all([isempty(children[root][a]) for a in keys(children[root])])
        # Actually correctly solved
        if check_solved(arr)
            undo_moves!(board, [action])
            update_records!(records, s_prev, s_next, action, tree, 1, current_chain)
            # Final node
            update_records!(records, s_next, 0, (0, 0), tree_c, 0, current_chain; new_chain=true)
            push!(chain_dict[(s_prev, s_start[1], action)], (s_next, root, (root[1], root[2][1])))
            return nothing
        else # Replan from root
            #@label replan
            # Recalculate root node
            root, current_depth = initialize_tree(board, frontier, children, parents, depths, actions)
        end
    end
    if !replan
        # If not replanning, add to records
        update_records!(records, s_prev, s_next, action, tree, isempty(actions) ? 1 : 2, current_chain)
    else
        update_records!(records, s_prev, -2, action, tree, -2, current_chain)
    end
    visited = collect(keys(children))
    # If not replanning but no available actions
    # extend frontier with children of parents of current node]
    if isempty(actions) && isempty(frontier)
        current_depth += 1
        for s in children[s_start[1]][s_start[2]]
            # Causes problems if s is not in keys of depths
            # due to red node (can condition on this)
            # if depths[s] == current_depth-1
            #     push!(frontier, s)
            #     if s ∉ visited
            #         push!(visited, s)
            #     end
            # end
            ##################
            # This adds all children, regardless of red node or not
            # Runs into problem that children of children etc dont have
            # updated depths -> new model that only looks at one branch
            # push!(frontier, s)
            # if s ∉ visited
            #     push!(visited, s)
            # end
            # depths[s] = current_depth-1
            # push!(parents[s], s_start)
            ##################
            # Can get blocked early because doesn't expand all
            # children
            if s ∈ keys(depths)
                if depths[s] == current_depth-1
                    push!(frontier, s)
                    if s ∉ visited
                        push!(visited, s)
                    end
                end
            end
        end
        # println(children[s_start[1]])
        # println(s_start)
        # println(collect(keys(depths)))
    end
    # Expand and recurse
    expand_tree!(frontier, visited, d, action, actions, parents_a, records, chain_dict, current_chain, board, arr, tree_c, current_depth, recursion_depth, root, replan ? s_prev : s_next, s_prev, s_start, max_iter)
    if !replan
        undo_moves!(board, [action])
        # Dead end case
        if isempty(records["state_space"][s_next])
            rev_a = (action[1], -action[2])
            update_records!(records, s_next, -3, rev_a, tree_c, -3, current_chain; new_chain=true)
            push!(chain_dict[(s_prev, s_start[1], action)], (s_next, s_start[1], rev_a))
            push!(chain_dict[(s_next, s_start[1], rev_a)], (-3, (-3, (-3,)), (-3, -3)))
        end
    end
    return nothing
end

# function complete_chains!(chains)
#     for chain in chains
#         if chain[end][3] == 3
#             idx = chain[end][2]
#             push!(chain, chains[idx[1]][idx[2]:end]...)
#         end
#     end
# end

prb = prbs[sp[24]]
board = load_data(prb);
@time records, chain_dict = backtracking_and_or_tree(1000, board);

root = board_to_int(get_board_arr(board), BigInt)
g = draw_chain_dict(chain_dict, board, root, optimal_a[prb]; full=false)

a
# ls = []
# ss = []
# cs = []
# ms = []
# for prb_id in ProgressBar(sp)
#     records = nothing
#     chain_dict = nothing
#     board = load_data(prbs[prb_id]);
#     records, chain_dict = backtracking_and_or_tree(1000, board);
#     state_space = records["state_space"];
#     chains = records["chains"];
#     push!(ls, length(state_space))
#     solution_chains = []
#     for chain in chains
#         if chain[end][3] == 0
#             push!(solution_chains, chain)
#         end
#     end
#     push!(ss, length(solution_chains))
#     push!(cs, length(chains))
#     push!(ms, maximum([length(c) for c in chains]))
# end
# action_space = records["action_space"];
# tree_space = records["tree_space"];
# chains = records["chains"];

root = board_to_int(get_board_arr(board), BigInt)
g = draw_backtrack_state_space(state_space, action_space, board, root, IDV[prb], optimal_a[prb]; highlight_nodes=sols, full=true)#, subj_states=subj_states)
draw_board(get_board_arr(board))

full_chains = []
for ch in chains
    if ch[end][3] != 3
        push!(full_chains, ch)
    end
end

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

g = new_draw_tree(t1["children"], board, (9, (2,)))
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



