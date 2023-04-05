
# function create_and_node(prev_car, cars)
#     return string(prev_car)*string(cars...)*"A"
# end

# function create_or_node(prev_car, cars)
#     return string(prev_car)*join([join(car) for car in unique(cars)], "_")*"O"
# end

function create_move_icon(move, board)
    car, m = move
    s = ""
    if board.cars[car].is_horizontal
        for i in 1:abs(m)
            if sign(m) == 1
                s *= "→"
            else
                s *= "←"
            end
        end
    else
        for i in 1:abs(m)
            if sign(m) == 1
                s *= "↓"
            else
                s *= "↑"
            end
        end
    end
    return s
end

function create_node_string(node)
    return string(node.value) *"-"* join([join(car) for car in node.children], "-")
end

struct Node
    value::Int
    children::Vector{Vector{Int}}
end

function Base.:(==)(n1::Node, n2::Node)
    n1.value == n2.value && n1.children == n2.children
end

board = load_data("ExploraTree")
board = load_data(prbs[end])
arr = get_board_arr(board)
draw_board(arr)
ROOT = 9

#make_move!(board, actions[1])

begin
    # INITIALIZE
    frontier = Vector{Tuple{Node, Int, Int}}()
    visited = Vector{Node}()
    actions = Vector{Tuple{Int, Int}}()
    draw_tree_nodes = DefaultOrderedDict{String, Vector{String}}([])
    # GET FIRST BLOCKED CARS
    red_car_block_init = reverse(blocked_by(arr, board.cars[ROOT]))
    # CREATE ROOT NODE
    start_node = Node(ROOT, [red_car_block_init])
    push!(visited, start_node)
    # INITIALIZE FRONTIER
    for i in eachindex(red_car_block_init)
        push!(frontier, (start_node, 1, i))
    end
end

max_iter = 100

for iter in 1:max_iter
    if isempty(frontier)
        println("REACHED THE END IN $(iter-1) ITERATIONS")
        break
    end
    # GET DFS NODE
    current_node, or_ind, and_ind = pop!(frontier)
    # SET NEW PARENT
    prev_car = current_node.value
    # SET NEXT TARGET
    current_car = current_node.children[or_ind][and_ind]
    # GET UNBLOCKING MOVES
    moves, next_cars = moves_that_unblock(board.cars[prev_car], board.cars[current_car], arr)
    # SORT
    #move_dict = SortedDict(next_cars .=> moves, Base.Order.Reverse)
    sorted_cars = reverse(unique(next_cars))
    new_node = Node(current_car, sorted_cars)
    # RED CAR SPECIAL CASE
    if current_car == ROOT
        if Node(current_car, [union(sorted_cars...)]) in visited
            continue
        end
    end
    # DO NOT EXPAND IF ALREADY VISITED
    if new_node in visited
        continue
    else
        push!(draw_tree_nodes[create_node_string(current_node)], create_node_string(new_node))
        push!(visited, new_node)
    end
    # EXTEND FRONTIER
    for i in eachindex(sorted_cars)
        if length(sorted_cars[i]) > 0
            for j in eachindex(sorted_cars[i])
                push!(frontier, (new_node, i, j))
            end
        else
            # IF UNBLOCKING MOVE IS VIABLE, ADD IT TO THE LIST
            for m in moves[isempty.(next_cars)]
                push!(actions, (current_car, m))
                push!(draw_tree_nodes[create_node_string(new_node)], "m"*create_move_icon((current_car, m), board))
            end
        end
    end
end
