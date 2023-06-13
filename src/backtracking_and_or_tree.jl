function backtracking_and_or_tree(board; max_iter=100)
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    tree = Dict{s_type, Dict{Tuple{Int, Int}, Vector{s_type}}}()
    frontier = Vector{s_type}()
    visited = Vector{s_type}()
    actions = Vector{Tuple{Int, Int}}()
    tried_actions = Vector{Tuple{Int, Int}}()
    parents = DefaultDict{s_type, Vector{Tuple{s_type, Tuple{Int, Int}}}}([])
    parents_a = DefaultDict{Tuple{Int, Int}, Vector{s_type}}([])
    depths = Dict{s_type, Int}()
    #ms_new = DefaultDict{Int, Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}}}}([])
    state_space = DefaultDict{BigInt, Vector{BigInt}}([])
    action_space = DefaultDict{BigInt, Vector{Tuple{Int, Int}}}([])
    tree_space = DefaultDict{BigInt, Vector{Dict{s_type, Dict{Tuple{Int, Int}, Vector{s_type}}}}}([])

    arr = get_board_arr(board)
    m_init = 6 - (board.cars[9].x+1)
    root = (9, (m_init,))
    s_root = board_to_int(arr, BigInt)
    push!(frontier, root)
    depths[root] = 0
    current_depth = 1

    for i in 0:max_iter-1
        if isempty(frontier)
            for action in actions
                if action ∉ tried_actions
                    for s in parents_a[action]
                        for s_startt in parents[s]
                            #ms_old = backtrack(board, tree, parents, depths, current_depth, s_start, s, action, 1, root)
                            # state_space_old, action_space_old, tree_space_old, ms_old = backtrack(board, tree, parents, depths, current_depth-1, s_start, s, action, 1, root)
                            # copy_dict!(state_space_new, state_space_old)
                            # copy_dict!(action_space_new, action_space_old)
                            # copy_dict!(tree_space_new, tree_space_old)
                            # copy_dict!(ms_new, ms_old)
                            recurse = true
                            for n in eachindex(state_space[s_root])
                                if action_space[s_root][n] == action && tree_space[s_root][n] == tree
                                    recurse = false
                                end
                            end
                            if recurse
                                backtrack!(state_space, action_space, tree_space, board, deepcopy(tree), deepcopy(parents), deepcopy(depths), current_depth-1, s_startt, s, action, 1, root)
                            end
                        end
                    end
                end
            end
            empty!(actions)
            empty!(parents_a)
            break
        end
        dict = Dict{Tuple{Int, Int}, Vector{s_type}}()
        s_old = popfirst!(frontier)
        car_id, moves = s_old
        depth_new = depths[s_old] + 1
        if depth_new > current_depth
            for action in actions
                if action ∉ tried_actions
                    for s in parents_a[action]
                        for s_startt in parents[s]
                            #ms_old = backtrack(board, tree, parents, depths, current_depth, s_start, s, action, 1, root)
                            # state_space_old, action_space_old, tree_space_old, ms_old = backtrack(board, tree, parents, depths, current_depth-1, s_start, s, action, 1, root)
                            # copy_dict!(state_space_new, state_space_old)
                            # copy_dict!(action_space_new, action_space_old)
                            # copy_dict!(tree_space_new, tree_space_old)
                            # copy_dict!(ms_new, ms_old)
                            #return state_space_new, action_space_new, ms_new
                            recurse = true
                            for n in eachindex(state_space[s_root])
                                if action_space[s_root][n] == action && tree_space[s_root][n] == tree
                                    recurse = false
                                end
                            end
                            if recurse
                                backtrack!(state_space, action_space, tree_space, board, deepcopy(tree), deepcopy(parents), deepcopy(depths), current_depth-1, s_startt, s, action, 1, root)
                            end
                        end
                    end
                    #push!(tried_actions, action)
                end
            end
            empty!(actions)
            empty!(parents_a)
            current_depth = depth_new
            #return state_space_new, action_space_new, ms_new
            #return ms_new#tree, visited, actions, trials, parents
        end
        push!(visited, s_old)
        for m in reverse(moves)
            move = (car_id, m)
            cars = move_blocked_by(board.cars[car_id], m, arr)
            if length(cars) == 0 && move ∉ actions
                push!(actions, move)
                #push!(trials, depth_new)
            end
            if s_old ∉ parents_a[move]
                push!(parents_a[move], s_old)
            end
            ls = Vector{s_type}()
            for car in cars
                moves_new, _ = moves_that_unblock(board.cars[car_id], board.cars[car], arr, move_amount=m)
                # Remove impossible moves with cars in same row
                moves_new = moves_new[(in).(moves_new, Ref(possible_moves(arr, board.cars[car])))]
                s_new = (car, Tuple(moves_new))
                if length(moves_new) == 0
                    push!(ls, s_new)
                    break
                end
                # if (s_old, move) ∉ parents[s_new]
                #     push!(parents[s_new], (s_old, move))
                # end
                if s_new ∉ visited && s_new ∉ frontier 
                    if (s_old, move) ∉ parents[s_new]
                        push!(parents[s_new], (s_old, move))
                    end
                    push!(frontier, s_new)
                    depths[s_new] = depth_new
                end
                push!(ls, s_new)
            end
            dict[move] = ls
        end
        if car_id == 9
            if any([i ∉ root[2] for i in s_old[2]]) || root ∉ keys(tree)
                tree[s_old] = dict
            end
        else
            tree[s_old] = dict
        end
    end
    #return tree, visited, parents, depths
    #return ms_new#tree, visited, actions, trials, parents
    #return state_space_new, action_space_new, tree_space_new, ms_new
    return state_space, action_space, tree_space
end

function copy_dict!(new, old)
    for k in keys(old)
        for item in old[k]
            #if item ∉ new[k]
            push!(new[k], item)
            #end
        end
    end
end

#function backtrack(board, tree, parents, depths, current_depth, s_start, s_old, action, recursion_depth, root; max_iter=100)
function backtrack!(state_space, action_space, tree_space, board, tree, parents, depths, current_depth, s_start, s_old, action, recursion_depth, root; max_iter=100)
    if current_depth != depths[s_old]
        return nothing
    end
    cd = current_depth
    s_prev = board_to_int(get_board_arr(board), BigInt)
    if recursion_depth > 200
        #return Dict(s_prev => [-1]), Dict(s_prev => [action]), Dict(s_prev => [tree]), Dict(recursion_depth => [(action, (-1, 0))])
        push!(state_space[s_prev], -1)
        push!(action_space[s_prev], action)
        push!(tree_space[s_prev], tree)
        return nothing
    end
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    frontier = Vector{s_type}()
    visited = Vector{s_type}()
    parents_a = DefaultDict{Tuple{Int, Int}, Vector{s_type}}([])
    actions = Vector{Tuple{Int, Int}}()
    tried_actions = Vector{Tuple{Int, Int}}()
    trials = Vector{Int}()
    #ms_new = DefaultDict{Int, Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}}}}([])
    #state_space_new = DefaultDict{BigInt, Vector{BigInt}}([])
    #action_space_new = DefaultDict{BigInt, Vector{Tuple{Int, Int}}}([])
    #tree_space_new = DefaultDict{BigInt, Vector{Dict{s_type, Dict{Tuple{Int, Int}, Vector{s_type}}}}}([])

    make_move!(board, action)
    if !is_valid_board(board)
        undo_moves!(board, [action])
        #return Dict(s_prev => [-2]), Dict(s_prev => [action]), Dict(s_prev => [tree]), Dict(recursion_depth => [(action, (-2, 0))])
        push!(state_space[s_prev], -2)
        push!(action_space[s_prev], action)
        push!(tree_space[s_prev], tree)
        return nothing
    end
    arr = get_board_arr(board)
    if has_overlap(arr, board)
        undo_moves!(board, [action])
        #return Dict(s_prev => [-2]), Dict(s_prev => [action]), Dict(s_prev => [tree]), Dict(recursion_depth => [(action, (-2, 0))])
        push!(state_space[s_prev], -2)
        push!(action_space[s_prev], action)
        push!(tree_space[s_prev], tree)
        return nothing
    end
    s_next = board_to_int(arr, BigInt)

    tree_c = deepcopy(tree)
    parents_c = deepcopy(parents)
    depths_c = deepcopy(depths)
    delete!(tree_c, s_old)
    delete!(parents_c, s_old)
    delete!(depths_c, s_old)
    for a in keys(tree_c[s_start[1]])
        for (n, s) in enumerate(tree_c[s_start[1]][a])
            if s == s_old
                if length(unique(tree_c[s_start[1]][a])) == 1
                    push!(actions, a)
                    push!(parents_a[a], s_start[1])
                end
                deleteat!(tree_c[s_start[1]][a], n)
            end
        end
    end


    for s in keys(parents_c)
        if depths_c[s] > current_depth
            delete!(parents_c, s)
        end
    end
    for s in keys(tree_c)
        if depths_c[s] > current_depth
            delete!(tree_c, s)
        end
    end
    for s in keys(depths_c)
        if depths_c[s] > current_depth
            delete!(depths_c, s)
        end
    end
    #current_depth = depths[s_start[1]]+2
    if s_start[1] == root && all([isempty(tree_c[root][a]) for a in keys(tree_c[root])])
        # Actually correctly solved
        if check_solved(arr)
            undo_moves!(board, [action])
            #return Dict(s_prev => [s_next]), Dict(s_prev => [action]), Dict(s_prev => [tree_c]), Dict(recursion_depth => [(action, (root[1], root[2][1]))])
            push!(state_space[s_prev], s_next)
            push!(action_space[s_prev], action)
            push!(tree_space[s_prev], tree)
            return nothing
        else # Tree incomplete
            m_init = 6 - (board.cars[9].x+1)
            root = (9, (m_init,))
            push!(frontier, root)
            empty!(tree_c)
            empty!(parents_c)
            empty!(depths_c)
            depths_c[root] = 0
            current_depth = 1
            empty!(actions)
        end
    end
    # RETURN
    push!(state_space[s_prev], s_next)
    push!(action_space[s_prev], action)
    push!(tree_space[s_prev], tree)

    visited = collect(keys(tree_c))

    # if isempty(tree_c[s_start[1]][s_start[2]])
    #     push!(actions, s_start[2])
    #     push!(parents_a[s_start[2]], s_start[1])
    # else
    if isempty(actions) && isempty(frontier)
        current_depth += 1
        for s in tree_c[s_start[1]][s_start[2]]
            if depths_c[s] == current_depth-1
                push!(frontier, s)
                if s ∉ visited
                    push!(visited, s)
                end
            end
        end
    end
    for i in 0:max_iter-1
        if isempty(frontier)
            for a in actions
                if a ∉ tried_actions
                    for s in parents_a[a]
                        for s_startt in parents_c[s]
                            # if (action, a) ∉ ms_new[recursion_depth]
                            #     push!(ms_new[recursion_depth], (action, a))
                            # end
                            # if s_next ∉ state_space_new[s_prev]
                            #     push!(state_space_new[s_prev], s_next)
                            #     push!(action_space_new[s_prev], action)
                            #     push!(tree_space_new[s_prev], tree_c)
                            # end
                            # if a ∉ action_space_new[s_next]
                            #     state_space_old, action_space_old, tree_space_old, ms_old = backtrack(board, tree_c, parents_c, depths_c, current_depth-1, s_start, s, a, recursion_depth + 1, root)
                            #     copy_dict!(state_space_new, state_space_old)
                            #     copy_dict!(action_space_new, action_space_old)
                            #     copy_dict!(tree_space_new, tree_space_old)
                            #     copy_dict!(ms_new, ms_old)
                            # end
                            recurse = true
                            for n in eachindex(state_space[s_next])
                                if action_space[s_next][n] == a && tree_space[s_next][n] == tree_c
                                    recurse = false
                                end
                            end
                            if recurse
                                backtrack!(state_space, action_space, tree_space, board, deepcopy(tree_c), deepcopy(parents_c), deepcopy(depths_c), current_depth-1, s_startt, s, a, recursion_depth + 1, root)
                            end
                        end
                    end
                    #push!(tried_actions, a)
                end
            end
            empty!(actions)
            empty!(parents_a)
            break
        end
        dict = Dict{Tuple{Int, Int}, Vector{s_type}}()
        s_old = popfirst!(frontier)
        car_id, moves = s_old
        depth_new = depths_c[s_old] + 1
        if depth_new > current_depth
            for a in actions
                if a ∉ tried_actions
                    for s in parents_a[a]
                        for s_startt in parents_c[s]
                            # if (action, a) ∉ ms_new[recursion_depth]
                            #     push!(ms_new[recursion_depth], (action, a))
                            # end
                            # if s_next ∉ state_space_new[s_prev]
                            #     push!(state_space_new[s_prev], s_next)
                            #     push!(action_space_new[s_prev], action)
                            #     push!(tree_space_new[s_prev], tree_c)
                            # end
                            # if a ∉ action_space_new[s_next]
                            #     state_space_old, action_space_old, tree_space_old, ms_old = backtrack(board, tree_c, parents_c, depths_c, current_depth-1, s_start, s, a, recursion_depth + 1, root)
                            #     copy_dict!(state_space_new, state_space_old)
                            #     copy_dict!(action_space_new, action_space_old)
                            #     copy_dict!(tree_space_new, tree_space_old)
                            #     copy_dict!(ms_new, ms_old)
                            # end
                            recurse = true
                            for n in eachindex(state_space[s_next])
                                if action_space[s_next][n] == a && tree_space[s_next][n] == tree_c
                                    recurse = false
                                end
                            end
                            if recurse
                                backtrack!(state_space, action_space, tree_space, board, deepcopy(tree_c), deepcopy(parents_c), deepcopy(depths_c), current_depth-1, s_startt, s, a, recursion_depth + 1, root)
                            end
                        end
                    end
                    #push!(tried_actions, a)
                end
            end
            empty!(actions)
            empty!(parents_a)
            current_depth = depth_new
        end
        if s_old ∉ visited
            push!(visited, s_old)
        end
        for m in reverse(moves)
            move = (car_id, m)
            if !is_valid_move(board, move)
                continue
            end
            cars = move_blocked_by(board.cars[car_id], m, arr)
            if length(cars) == 0 && move ∉ actions && move[1] != action[1]
                push!(actions, move)
                push!(trials, depth_new)
            end
            if s_old ∉ parents_a[move]
                push!(parents_a[move], s_old)
            end
            ls = Vector{s_type}()
            for car in cars
                moves_new, _ = moves_that_unblock(board.cars[car_id], board.cars[car], arr, move_amount=m)
                # Remove impossible moves with cars in same row
                moves_new = moves_new[(in).(moves_new, Ref(possible_moves(arr, board.cars[car])))]
                s_new = (car, Tuple(moves_new))
                if length(moves_new) == 0
                    push!(ls, s_new)
                    break
                end
                # if (s_old, move) ∉ parents_c[s_new]
                #     push!(parents_c[s_new], (s_old, move))
                # end
                if s_new ∉ visited && s_new ∉ frontier 
                    if (s_old, move) ∉ parents_c[s_new]
                        push!(parents_c[s_new], (s_old, move))
                    end
                    push!(frontier, s_new)
                    depths_c[s_new] = depth_new
                end
                push!(ls, s_new)
            end
            dict[move] = ls
        end
        if car_id == 9
            if any([i ∉ root[2] for i in s_old[2]]) || root ∉ keys(tree_c)
                tree_c[s_old] = dict
            end
        else
            tree_c[s_old] = dict
        end
    end
    undo_moves!(board, [action])
    # Dead end case
    if isempty(state_space[s_next])
        rev_a = (action[1], -action[2])
        #push!(state_space[s_prev], s_next)
        push!(state_space[s_next], -3)
        #push!(action_space[s_prev], action)
        push!(action_space[s_next], rev_a)
        #push!(tree_space[s_prev], tree)
        push!(tree_space[s_next], tree_c)
        #push!(ms_new[recursion_depth], (action, rev_a))
    end
    return nothing
    #return state_space_new, action_space_new, tree_space_new, ms_new
end

state_space, action_space, tree_space = backtracking_and_or_tree(board);


#tree, visited, actions, trials, parents = backtracking_and_or_tree(board);

board = arr_to_board(int_to_arr(sss));
board = load_data("prb26567_7");
board = load_data(prbs[end]);
board = load_data(prbs[1]);
make_move!(board, (2, 1))
make_move!(board, (8, 1))

root = board_to_int(get_board_arr(board), BigInt)

state_space, action_space, ms = backtracking_and_or_tree(board);

#draw_backtrack_state_space(state_space, action_space, board, root, IDV[prbs[end]], optimal_a[prbs[end]]; highlight_nodes=[])
#g = draw_backtrack_state_space(state_space, action_space, board, root, IDV[prbs[37]], optimal_a[prbs[37]]; highlight_nodes=[])
g = draw_backtrack_state_space(state_space, action_space, board, root, IDV[prbs[1]], optimal_a[prbs[1]]; highlight_nodes=[], full=false)

tree, visited, actions = backtracking_and_or_tree(board);
g = new_draw_tree(tree_space[root][2], board)#, visited; green_act=actions)
g = new_draw_tree(ttt, board, (9, (4,)))
g = new_draw_tree(tree_space[root][2], board, (9, (4,)))

for s in keys(state_space)
    if s ∉ collect(keys(IDV[prbs[1]]))
        display(int_to_arr(s))
    end
end


# CHECKS FOR DUPLICATES
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



