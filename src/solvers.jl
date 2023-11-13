include("engine.jl")
include("plot_graph.jl")
using DataStructures
using BenchmarkTools
using Distributions
using Random

"""
    a_star(board, h=(x,y)->0, graph_search=false, solve=true)

A* algorithm returning (visited states, optimal moves, expansions) with zero heuristic as default heuristic function.
The search is a tree-search, not a graph-search, which means visited nodes can be
revisited but the required heuristic function to ensure optimality only needs to be
admissible, and not monotonic (as would be required for graph search).
"""
function a_star(board; h=(x,y)->0, graph_search=false, solve=true)
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    start = board_to_int(arr_start, T)
    # f_dict is a map that gives cheapest cost from start to node currently known
    f_dict = DefaultDict{T, Int}(10000)
    f_dict[start] = 0
    # dict is an array with the shortest path in moves to board state
    dict = DefaultDict{T, Array{Array{Int, 1}}}(Int[])
    # open_set is priority queue that will indicate which node to expand next
    open_set = PriorityQueue{T, Int}()
    enqueue!(open_set, start, h(board, arr_start))
    # Fifth list for graph_search keeps track of all visited nodes
    closed_set = Set{T}()
    push!(closed_set, start)
    # Visited nodes for logging
    visited = Array{T, 1}()
    push!(visited, start)
    # count node expansions to prevent infinite loop
    expansions = 0
    available_moves = Array{Array{Int, 1}, 1}()
    while length(open_set) > 0
        # Exit if too many expansions occured
        if expansions > 100000
            throw(DomainError("Expansion limit reached"))
        end
        # Get state with lowest f score
        current = dequeue!(open_set)
        f_score_old = f_dict[current]
        past_moves = dict[current]
        # Move to current node
        if length(past_moves) > 0
            for move in past_moves
                make_move!(board, move)
            end
        end
        # Get current board int-string
        arr_current = get_board_arr(board)
        # Add to visited for logging
        if !(current in visited)
            push!(visited, current)
        end
        # Check if solved
        if check_solved(arr_current)
            undo_moves!(board, past_moves)
            if solve
                # println("------")
                # println("Size of dict: " * string(Base.summarysize(dict)) * " bytes")
                # println("Number of node expansions: " * string(expansions))
                # println("Optimal length: " * string(length(past_moves)))
                # println("------")
                # Undo rest of moves back to initial board
                
                return visited, past_moves, expansions
            else
                continue
            end
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
            if graph_search
                # Ignore if neighbor already visited
                if new in closed_set
                    # Undo move
                    undo_moves!(board, [move])
                    continue
                # Add to list if not yet visited
                else
                    push!(closed_set, new)
                end
            end
            # Calculates g score of neighbor
            tentative_score = f_score_old + 1
            # If this is the fastest way to get to neighbor, update its costs
            if tentative_score < f_dict[new]
                # Update score
                f_dict[new] = tentative_score
                dict[new] = vcat(past_moves, [move])
                f_score_new = tentative_score + h(board, arr_new)
                # Set all moves leading to current neighbor
                #move_dict[new] = vcat(past_moves, [move])
                # If neighbor is not yet queued to be visited, queue it
                if !(new in keys(open_set))
                    # Push node onto heap
                    enqueue!(open_set, new, f_score_new)
                end
            end
            # Undo move
            undo_moves!(board, [move])
        end
        # Undo rest of moves back to initial board
        undo_moves!(board, past_moves)
    end
    # println("--------------------------")
    # println("A* did not find a solution")
    # println(expansions)
    # println(Base.summarysize(dict))
    # println("--------------------------")
    # return nothing, nothing
    return visited, nothing, expansions
end

"""
    dfs(board, solve=true)

Depth first search algorithm that returns (visited states, optimal moves, expansions)
"""
function dfs(board; solve=true)
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    start = board_to_int(arr_start, T)
    # dict is an array with the shortest path in moves to board state
    dict = Dict{T, Array{Array{Int, 1}}}()
    dict[start] = []
    # q is an array that will indicate which node to expand next
    q = T[]
    push!(q, start)
    # visited keeps track of all visited nodes
    visited = Array{T, 1}()
    push!(visited, start)
    # count node expansions to prevent infinite loop
    expansions = 0
    available_moves = Array{Array{Int, 1}, 1}()
    while length(q) > 0
        # Exit if too many expansions occured
        if expansions > 100000
            throw(DomainError("Expansion limit reached"))
        end
        # Get depth first node
        current = pop!(q)
        past_moves = dict[current]
        # Move to current node
        if length(past_moves) > 0
            for move in past_moves
                make_move!(board, move)
            end
        end
        # Get current board int-string
        arr_current = get_board_arr(board)
        # Add to visited for logging
        if !(current in visited)
            push!(visited, current)
        end
        # Check if solved
        if solve
            if check_solved(arr_current)
                # Undo rest of moves back to initial board
                undo_moves!(board, past_moves)
                return visited, past_moves, expansions
            end
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
            # Ignore if already visited
            if new in visited || new in q
                # Undo move
                undo_moves!(board, [move])
                continue
            # Remove already existing copy in frontier
            elseif new in q
                deleteat!(q, findfirst(x->x==new, q))
            end
            # Add current node to front of frontier
            push!(q, new)
            # Check if current path is fastest so far
            if haskey(dict, new)
                if length(past_moves)+1 < length(dict[new])
                    dict[new] = vcat(past_moves, [move])
                end
            else
                dict[new] = vcat(past_moves, [move])
            end
            # Undo move
            undo_moves!(board, [move])
        end
        # Undo rest of moves back to initial board
        undo_moves!(board, past_moves)
    end
    return nothing, nothing, expansions
end

"""
    bfs_path_counters(board, traverse_full=false, heuristic=zer, all_parents=false)

Breadth first search algorithm that counts how many paths there are to
all nodes in same layer as optimal solution
"""
function bfs_path_counters(board; traverse_full=false, heuristic=zer, all_parents=false)
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    start = board_to_int(arr_start, T)
    # dict is a map that gives an array with the shortest path in moves to board state
    dict = Dict{T, Array{Array{Int, 1}, 1}}()
    dict[start] = []
    # Stat is a map that gives the layer, whether the board is solved and heuristic value for each state
    stat = Dict{T, Array{Int, 1}}()
    stat[start] = [0, false, heuristic(board, arr_start)]
    # seen is a map that counts how many times a node has been 'seen' by its parents
    seen = DefaultDict{T, Int}(0)
    seen[start] = 1
    # q is a queue that will indicate which node to expand next
    q = Deque{T}()
    push!(q, start)
    # visited keeps track of all visited nodes
    visited = Set{T}()
    push!(visited, start)
    # dictionary to keep track of tree size
    tree = DefaultOrderedDict{Int, Int}(0)
    tree[0] = 1
    # parents keeps track of the parent nodes for each node in tree
    parents = DefaultDict{T, Array{T, 1}}([])
    # parent_cars also keep track of which car is moved
    parent_cars = DefaultDict{T, Array{Int, 1}}([])
    # children keeps track of the children nodes for each node in tree
    children = DefaultDict{T, Array{T, 1}}([])
    # solutions is a list of the nodes that are solved
    solutions = Array{T, 1}()
    # count node expansions to prevent infinite loop
    expansions = 0
    # set optimal length depth
    depth = 10000
    available_moves = Array{Array{Int, 1}, 1}()
    while length(q) > 0
        # Exit if too many expansions occured
        if expansions > 1000000
            throw(DomainError("Expansion limit reached"))
        end
        # Get next node
        current = pop!(q)
        # Visit node
        push!(visited, current)
        # Get moves to node
        past_moves = dict[current]
        # Get layer number
        layer = length(past_moves)
        # Move to current node
        if layer > 0
            for move in past_moves
                make_move!(board, move)
            end
        end
        # Get current board int-string
        arr_current = get_board_arr(board)
        # Check if solved
        is_solved = check_solved(arr_current)
        if (!traverse_full) && is_solved && depth == 10000
            depth = layer
        end
        # Update stat
        stat[current] = [layer, is_solved, heuristic(board, arr_current)]
        # Add solutions to list
        if is_solved
            push!(solutions, current)
            undo_moves!(board, past_moves)
            continue
        end
        if layer < depth
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
                if seen[new] == 0
                    # Shortest path to node
                    dict[new] = vcat(past_moves, [move])
                    # Update tree
                    tree[layer+1] += 1
                    # Push node onto heap
                    pushfirst!(q, new)
                end     
                # Increment seen if not yet visited
                if length(dict[new]) > layer
                    # Increment seen counter
                    seen[new] += seen[current]
                    # Add to parents and children
                    if !all_parents
                        push!(parents[new], current)
                        push!(parent_cars[new], move[1])
                    end
                    push!(children[current], new)
                end  
                if all_parents
                    if !(current in parents[new])
                        push!(parents[new], current)
                        push!(parent_cars[new], move[1])
                    end
                    if !(new in parents[current])
                        push!(parents[current], new)
                        push!(parent_cars[current], move[1])
                    end
                end
                # Undo move
                undo_moves!(board, [move])
            end
        end
        # Undo rest of moves back to initial board
        undo_moves!(board, past_moves)
    end
    # Delete nodes beyond optimal depth
    if !traverse_full
        for key in keys(stat)
            layer, is_solved = stat[key]
            if layer > depth
                delete!(seen, key)
                delete!(stat, key)
                delete!(parents, key)
                delete!(parent_cars, key)
                delete!(children, key)
            end
        end
        delete!(tree, depth+1)
    end
    for child in keys(parents)
        if !(child in keys(children))
            children[child] = []
        end
    end
    return tree, seen, stat, dict, parents, children, solutions, parent_cars
end

"""
    random_agent(board, max_iters=1000000)

Return number of moves an agent which makes moves randomly takes to complete puzzle.
Returns Inf if not solved within `max_iters` expansions.
"""
function random_agent(board, max_iters=1000000)
    # board = load_data("hard_puzzle_40")
    # Random.seed!(2)
    expansions = 0
    # Stores all moves performed
    #moves = Array{Tuple{Int, Int}, 1}()
    # Stores all visited states
    #visited = Array{BigInt, 1}()
    available_moves = Array{Array{Int, 1}, 1}()
    arr = zeros(Int, 6, 6)
    while expansions < max_iters
        #arr_current = get_board_arr(board)
        get_board_arr!(arr, board)
        #push!(visited, board_to_int(arr, BigInt))
        # Check if complete
        if check_solved(arr)
            return expansions
        end
        # Expand current node by getting all available moves
        get_all_available_moves!(available_moves, board, arr)
        # Increment
        expansions += 1
        # Randomly choose a move
        #selected_move_idx = rand(1:length(available_moves))
        # Make move
        make_move!(board, available_moves[rand(1:length(available_moves))])
        #push!(moves, selected_move)
    end
    #return Inf
end

"""
    random_agent_explore(board, max_iters=100000)

Return number of moves an agent which makes moves randomly amongst
moves that bring agent to unseen states until puzzle is complete.
If none of the moves are 'exploratory', choose a move at random.
Returns Inf if not solved within `max_iters` expansions.
"""
function random_agent_explore(board, max_iters=100000)
    # board = load_data("hard_puzzle_40")
    # Random.seed!(2)
    expansions = 0
    # Stores all visited states
    arr_start = get_board_arr(board)
    T = get_type(arr_start)
    visited = Set{T}()
    available_moves = Array{Array{Int, 1}, 1}()
    while expansions < max_iters
        arr_current = get_board_arr(board)
        # Add to visited if not yet visited
        current = board_to_int(arr_current, T)
        if !(current in visited)
            push!(visited, current)
        end
        # Check if complete
        if check_solved(arr_current)
            return expansions
        end
        # Expand current node by getting all available moves
        get_all_available_moves!(available_moves, board, arr_current)
        # Increment
        expansions += 1
        # Check which moves don't lead to already visited states
        choices = []
        for move in available_moves
            make_move!(board, move)
            arr_new = get_board_arr(board)
            new = board_to_int(arr_new, T)
            if !(new in visited)
                push!(choices, move)
            end
            # Undo move
            undo_moves!(board, [move])
        end
        if length(choices) == 0
            # Randomly choose a move
            selected_move_idx = rand(1:length(available_moves))
            selected_move = available_moves[selected_move_idx]
        else
            # Randomly choose a move that leads to unvisited state
            selected_move_idx = rand(1:length(choices))
            selected_move = choices[selected_move_idx]
        end
        # Make move
        make_move!(board, selected_move)
    end
    return Inf
end

"""
    random_agent_no_undo(board, max_iters=1000000)

Return number of moves an agent which makes moves randomly (except moving back)
takes to complete puzzle.
Returns Inf if not solved within `max_iters` expansions.
"""
function random_agent_no_undo(board, max_iters=1000000)
    # board = load_data("hard_puzzle_40")
    # Random.seed!(2)
    expansions = 0
    prev_move = [0, 0]
    # Stores all moves performed
    #moves = Array{Tuple{Int, Int}, 1}()
    available_moves = Array{Array{Int, 1}, 1}()
    while expansions < max_iters
        arr_current = get_board_arr(board)
        # Check if complete
        if check_solved(arr_current)
            return expansions
        end
        # Expand current node by getting all available moves
        get_all_available_moves!(available_moves, board, arr_current)
        choices = []
        for move in available_moves
            if !([move[1], -move[2]] == prev_move)
                push!(choices, move)
            end
        end
        # Increment
        expansions += 1
        if length(choices) == 0
            # Randomly choose a move
            selected_move_idx = rand(1:length(available_moves))
            selected_move = available_moves[selected_move_idx]
        else
            # Randomly choose a move that is not an undo
            selected_move_idx = rand(1:length(choices))
            selected_move = choices[selected_move_idx]
        end
        # Make move
        make_move!(board, selected_move)
        prev_move = selected_move
        #push!(moves, selected_move)
    end
    return Inf
end


"""
    red_distance(board, arr)

Calculate how many vehicles block the red car
"""
function red_distance(board, arr)
    # Get row with red car on it
    row = arr[3, :]
    # Split row/col
    idx = findfirst(x -> x == maximum(arr), row)
    # Get forward moves
    f = row[idx+2:end]
    # Return unique cars
    u = unique(f)
    # Remove zeros
    deleteat!(u, u .== 0)
    return length(u)
end

"""
    red_pos(board, arr)

Calculate how close the red car is to the exit
"""
function red_pos(board, arr)
    # Get row with red car on it
    row = arr[3, :]
    # Find position of red car
    idx = findfirst(x -> x == maximum(arr), row) + 1
    return 6 - idx
end

"""
    mag_size_layers(board, arr)

Calculates size (layers) of MAG
"""
function mag_size_layers(board, arr)
    mag = get_constrained_mag(board, arr)
    return length(mag) - 1
end

"""
    mag_size_nodes(board, arr)

Calculates size (nodes) of MAG
"""
function mag_size_nodes(board, arr)
    mag = get_constrained_mag(board, arr)
    return sum([length(m) for m in mag]) - 1
end

"""
    multi_mag_size_nodes(board, arr)    

Calculate minimum node size from forest
"""
function multi_mag_size_nodes(board, arr)
    # Get forest
    mags = get_multiple_mags(board, arr)
    # Find minimum number of layers
    min = 100000
    for mag in mags
        size = sum([length(m) for m in mag]) - 1#length(unique(reduce(vcat, (reduce(vcat, [collect(values(m)) for m in mag])))))
        if size < min
            min = size
        end
    end
    return min
end

"""
    multi_mag_size_layers(board, arr)    

Calculate minimum layer size from forest
"""
function multi_mag_size_layers(board, arr)
    # Get forest
    mags = get_multiple_mags(board, arr)
    # Find minimum number of layers
    min = 100000
    for mag in mags
        size = length(mag) - 1
        if size < min
            min = size
        end
    end
    return min
end

"""
    leave_one_out(board, arr)

Calculate (mean, rand, min) optimal length when removing each car
"""
function leave_one_out(board, arr)
    # Get number of unique cars
    car_n = length(unique(arr)) - 2
    # Optimal length for removing each car
    opts = zeros(car_n)
    # Remove each car
    for i in 1:car_n
        # Remove car and save it to be added back later
        deleted_car = popat!(board.cars, i)
        # Change car IDs
        for j in i:car_n
            board.cars[j].id -= 1
        end
        # Find optimal length
        _, pm, _ = a_star(board, h=red_distance)
        opts[i] = length(pm)
        # Change back car IDs
        for j in i:car_n
            board.cars[j].id += 1
        end
        # Add deleted car back
        insert!(board.cars, i, deleted_car)
    end
    return opts
end

function zer(board, arr)
    return 0
end

"""
    calculate_blocked_moves(board, arr)

Calculate whether the cars blocking the red car can be moved
"""
function calculate_blocked_moves(board, arr)
    # Get cars blocking exit
    blocking_cars = blocked_by(arr, board.cars[end])
    # Loop over blocking cars
    car_blocked = Array{Bool, 1}()
    for car_id in blocking_cars
        # Get car details
        car = board.cars[car_id]
        len = car.len
        y = car.y
        # Get moves for car
        moves = get_available_moves(arr, car)
        # Initialise car to blocked
        is_blocked = true
        # Loop over all moves
        for move in moves
            m = move[2]
            # If car is small and available move does not keep car in front of exit
            # set blocked to false
            if len == 2
                if m != (2*(y==2)-1)
                    is_blocked = false
                    break
                end
            # If car is large and available move moves car all the way to the bottom
            # set blocked to false
            elseif len == 3
                if m + y == 4
                    is_blocked = false
                    break
                end
            end
        end
        # For each blocked car, set whether it can move out of the way
        push!(car_blocked, is_blocked)
    end
    return (length(car_blocked) + sum(car_blocked))
end

"""
    calculate_heur(board, moves, h)

Calculate h at each of the moves, returns list with values
"""
function calculate_heur(board, moves, h)
    # Undo moves
    undo_moves!(board, moves)
    # Initialise list
    arr = get_board_arr(board)
    heurs = [h(board, arr)]
    for move in moves
        # Commit move
        make_move!(board, move)
        arr = get_board_arr(board)
        # Calculate and append heuristic
        push!(heurs, h(board, arr))
    end
    return heurs
end

function get_solution_paths(solutions, parents, stat)
    # for solution in solutions
    #     soln_len = stat[solution][1]
    #     old_parents = [solution]
    #     for i in 0:soln_len
    #         layer = soln_len - i
    #         println(layer)
    #         dummy_parents = Array{Int128, 1}()
    #         for old_parent in old_parents
    #             new_parents = parents[old_parent]
    #             for new_parent in new_parents
    #                 if !(old_parent in solution_paths[new_parent])
    #                     push!(solution_paths[new_parent], old_parent)
    #                     push!(dummy_parents, new_parent)
    #                     fake_tree[layer] += 1
    #                 end
    #             end
    #         end
    #         old_parents = dummy_parents
    #     end
    # end
    # return solution_paths, fake_tree
    
    solution_paths = DefaultOrderedDict{Int, Array{BigInt, 1}}([])
    for solution in solutions
        push!(solution_paths[stat[solution][1]], solution)
    end
    fake_tree = OrderedDict{Int, Int}()
    max_len = maximum(keys(solution_paths))
    fake_tree[max_len] = length(solution_paths[max_len])
    
    max_heur = 0
    for i in 1:max_len
        layer = max_len - i
        for child in solution_paths[layer + 1]
            for parent in parents[child]
                if !(parent in solution_paths[layer])
                    push!(solution_paths[layer], parent)
                end
            end
        end
        for node in solution_paths[layer]
            if stat[node][3] > max_heur
                max_heur = stat[node][3]
            end
        end
        fake_tree[layer] = length(solution_paths[layer])
    end

    return solution_paths, fake_tree, max_heur
end

function invert_tree(all_parents, solutions, max_dist=1000)
    idvs = Dict{BigInt, Array{Int, 1}}()
    frontier = solutions
    distance = 0

    for i in 1:max_dist
        new_frontier = []
        for state in frontier
            idvs[state] = [distance, length(all_parents[state])]
            for parent in all_parents[state]
                if !(parent in keys(idvs)) && !(parent in new_frontier) && !(parent in frontier)
                    push!(new_frontier, parent)
                end
            end
        end
        if isempty(new_frontier)
            break
        end
        frontier = new_frontier
        distance += 1
    end
    return idvs
end

function get_optimal_actions(distance_from_sol, all_parents)
    optimal_a = DefaultDict{BigInt, Array{BigInt, 1}}([])
    for state in keys(all_parents)
        dist = distance_from_sol[state][1]
        if dist == 0
            optimal_a[state] = []
            continue
        end
        for parent in all_parents[state]
            if distance_from_sol[parent][1] < dist
                push!(optimal_a[state], parent)
            end
        end
    end
    return optimal_a
end

function get_action_from_states(s1, s2)
    arr1 = int_to_arr(s1)
    arr2 = int_to_arr(s2)
    dif = arr2 - arr1
    pos = dif[dif .> 0]
    car = pos[1]
    b1 = arr_to_board(arr1)
    b2 = arr_to_board(arr2)
    if b1.cars[car].is_horizontal
        return (car, b2.cars[car].x - b1.cars[car].x)
    else
        return (car, b2.cars[car].y - b1.cars[car].y)
    end
end

#solution_paths, fake_tree, max_heur = get_solution_paths(solutions, parents, stat);

# board = load_data("hard_puzzle_1")

# arr = get_board_arr(board)

# draw_board(arr)

# mag = get_mag(board, arr)
# g = plot_mag(mag, "two_choice_problem_00", "Move_0")
# mag = get_constrained_mag(board, arr)
# mags = get_multiple_mags(board, arr)

# arr, moves, exp = a_star(board, multi_mag_size_nodes);

# heur = calculate_heur(board, moves, multi_mag_size_nodes)

# for move in moves[1:10]
#     make_move(board, move)
# end

# for mag in mags
#     g = plot_mag(mag)
#     display(g)
# end

# # # mag = get_constrained_mag(board, arr)
# # # mag = get_mag(board, arr)

# # g = plot_mag(mag, "two_choice_problem_00", "Move_0")
# # save_graph(g, "two_choice_problem_00")

# board = load_data([])
# for (n, move) in enumerate(moves[1:12])
#     make_move(board, move)
#     arr = get_board_arr(board)
#     #mag = get_constrained_mag(board, arr)
#     mag = get_mag(board, arr)
#     g = plot_mag(mag, "test_" * (n > 9 ? "" : "0") * string(n), "Move_" * string(n))
#     display(g)
#     #save_graph(g, "test_" * (n > 9 ? "" : "0") * string(n))
# end


# @btime multi_mag_size_nodes(board, arr)

# board = load_data("hard_puzzle_1")
board = load_data(prbs[3])
arr = get_board_arr(board)
T = get_type(arr)
start = board_to_int(arr, T)
tree, seen, stat, dict, all_parents, children, solutions, parent_actions = bfs_path_counters(board, traverse_full=true, all_parents=true);
tree, seen, stat, dict, parents, children, solutions, parent_actions = bfs_path_counters(board, traverse_full=true, all_parents=false);

solution_paths, fake_tree, max_heur = get_solution_paths(solutions[1:2], parents, stat);

# # plot_tree(fake_tree)

# #g = draw_solution_paths(solution_paths, parents, stat, max_heur)

# g = draw_directed_tree(parents, solution_paths=solution_paths, solutions=solutions, all_parents=all_parents, start=start)
parents[start] = []
g = draw_directed_tree(parents, solution_paths=solution_paths, solutions=solutions, all_parents=all_parents, start=start)
draw_subject_paths(visited_states, prbs[2], d_goals)

# save_graph(g, "bidirectional_hard_1_start_optimal_only_v3")
# save_graph(g, "solution_hard_puzzle_39_complete")
save_graph(g, "state_space_example_qual")

 