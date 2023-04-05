
board = load_data("ExploraTree")
arr = get_board_arr(board)
draw_board(arr)

begin
    # INITIALIZE
    forest = Vector{Dict{Int64, Vector{Int}}}()
    and_nodes_forest = Vector{Vector{Int}}()
    and_node_counter_forest = Vector{Dict{Int, Int}}()
    or_checkpoint = Vector{Int}()
    actions = Vector{Vector{Int}}()
    # RED CAR START
    init_tree = Dict{Int64, Vector{Int}}()
    red_car_block_init = reverse(blocked_by(arr, board.cars[9]))
    init_tree[9] = red_car_block_init

    and_nodes_init = [9]
    and_node_counter_init = Dict{Int, Int}()
    and_node_counter_init[9] = length(red_car_block)

    push!(forest, init_tree)
    push!(and_nodes_forest, and_nodes_init)
    push!(and_node_counter_forest, and_node_counter_init)
    push!(or_checkpoint, 9)

    max_iter = 6
    prev_car = 9
end
for iter in 1:max_iter
    # CHOP MOST PROMISING TREE
    tree = pop!(forest)
    and_nodes = pop!(and_nodes_forest)
    and_node_counter = pop!(and_node_counter_forest)
    prev_car = pop!(or_checkpoint)
    # CHOOSE MOST PROMISING FRUIT (GET LEAF CAR)
    children = tree[prev_car]
    # STORE AND NODES TO BACKTRACK TO
    if length(children) > 1
        # SELECT NEXT CHILD
        current_car = children[and_node_counter[prev_car]]
        # DO NOT REVISIT CHILD
        if and_node_counter[prev_car] == 1
            delete!(and_node_counter, prev_car)
            filter!(x->x!=prev_car, and_nodes)
        else
            and_node_counter[prev_car] -= 1
        end
    else
        # SELECT CHILD
        current_car = children[1]
    end
    # GET ALL SEEDS (GET CARS BLOCKING CURRENT STATE)
    moves, cars = moves_that_unblock(board.cars[prev_car], board.cars[current_car], arr)
    # SORT BY HOW MANY CARS ARE BLOCKING
    move_dict = SortedDict(cars .=> moves, Base.Order.Reverse)
    # RED CAR SPECIAL CASE 1
    if current_car == 9 && length(keys(move_dict)) == length(red_car_block)
        if length(children) > 1
            delete!(and_node_counter, prev_car)
            filter!(x->x!=prev_car, and_nodes)
        end
        # BACKTRACK TO MOST RECENT AND NODE
        prev_car = forest[end][and_nodes[end]][and_node_counter[and_nodes[end]]]
        #filter!(x->(first(x) ∈ and_nodes), tree)
        continue
    end
    # SPAWN NEW TREES (PLANT NEW SEEDS)
    for (n, new_cars) in enumerate(keys(move_dict))
        if length(new_cars) > 0
            # RED CAR SPECIAL CASE 2
            if current_car == 9 && issubset(Set(new_cars), Set(red_car_block))
                continue
            end
            # AND NODES
            if length(new_cars) > 1 && current_car ∉ and_nodes
                push!(and_nodes, current_car)
                and_node_counter[current_car] = length(new_cars)
            end
            # COPY OLD TREE
            new_tree = copy(tree)
            # APPEND NEW CHANGES
            new_tree[current_car] = reverse(new_cars)
            # ADD TREE TO FOREST
            push!(forest, new_tree)
            # NEXT ITERATION
            prev_car = current_car
        else
            # ADD POTENTIAL ACTION
            push!(actions, [current_car, move_dict[new_cars]])
            # # BACKTRACK TO MOST RECENT AND NODE
            # prev_car = and_nodes[end]
        end
    end
end
