
function create_move_icon(move, board)
    car, m = move
    s = string(car)
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

function create_node_string(node; rev=false, visited=false)
    s = ""
    if visited
        s *= "v"
    end
    if rev
        return s*string(node.value) *"_"* join([join(reverse(car)) for car in reverse(node.children)], "_")
    else
        return s*string(node.value) *"_"* join([join(car) for car in node.children], "_")
    end
end

struct Node
    value::Int
    children::Vector{Vector{Int}}
end

function Base.:(==)(n1::Node, n2::Node)
    n1.value == n2.value && n1.children == n2.children
end

function and_or_tree(board, arr; γ=0.0, DFS=true, max_iter=100)
    ROOT = length(unique(arr))-1
    begin
        # INITIALIZE
        #frontier = Vector{Tuple{Node, Int, Int}}()
        frontier = Vector{Tuple{Node, Int, Int, Tuple{Int, Int}}}()
        visited = Vector{Node}()
        timeline = Vector{Tuple{Node, Tuple{Int, Int}, Int}}()
        actions = Vector{Tuple{Int, Int}}()
        trials = Vector{Int}()
        draw_tree_nodes = DefaultOrderedDict{String, Vector{String}}([])
        # GET FIRST BLOCKED CARS
        if DFS
            red_car_block_init = reverse(blocked_by(arr, board.cars[ROOT]))
        else
            red_car_block_init = blocked_by(arr, board.cars[ROOT])
        end
        # GET POSSIBLE FIRST MOVES
        available_moves = []
        get_available_moves!(available_moves, arr, board.cars[ROOT])
        if length(available_moves) > 0
            available_moves_ = []
            for move in available_moves
                if move[2] > 0
                    push!(available_moves_, move)
                end
            end
            # CREATE ROOT NODE
            start_node = Node(ROOT, vcat([red_car_block_init], [[move[2]*1000] for move in reverse(available_moves_)]))
            for move in available_moves_
                push!(actions, (move[1], move[2]))
                push!(trials, 0)
                push!(draw_tree_nodes[create_node_string(start_node, rev=DFS)], "m"*create_move_icon((move[1], move[2]), board))
            end
        else
            start_node = Node(ROOT, [red_car_block_init])
        end
        push!(visited, start_node)
        # INITIALIZE FRONTIER
        for i in eachindex(red_car_block_init)
            if DFS
                push!(frontier, (start_node, 1, i, (ROOT, 5-board.cars[end].x)))
            else
                pushfirst!(frontier, (start_node, 1, i))
            end
        end
    end
    for iter in 1:max_iter
        if isempty(frontier) || rand() < γ
            #println("REACHED THE END IN $(iter-1) ITERATIONS")
            return actions, trials, visited, timeline, draw_tree_nodes
        end
        # GET DFS NODE
        current_node, or_ind, and_ind, prev_move = pop!(frontier)
        # SET NEW PARENT
        prev_car = current_node.value
        # SET NEXT TARGET
        current_car = current_node.children[or_ind][and_ind]
        push!(timeline, (current_node, prev_move, current_car))
        # GET UNBLOCKING MOVES
        moves, next_cars = moves_that_unblock(board.cars[prev_car], board.cars[current_car], arr)
        # SORT
        if !DFS
            reverse!.(next_cars)
        end
        if DFS
            sorted_cars = reverse(unique(next_cars))
        else
            sorted_cars = unique(next_cars)
        end
        # POSSIBLE MOVE SPECIAL CASE
        if any(isempty.(sorted_cars))
            if DFS
                new_node = Node(current_car, vcat(sorted_cars[.!(isempty.(sorted_cars))], [[aa] for aa in moves[isempty.(next_cars)]*1000]))
            else
                new_node = Node(current_car, vcat([[aa] for aa in reverse(moves[isempty.(next_cars)]*1000)], sorted_cars[.!(isempty.(sorted_cars))]))
            end
        else
            new_node = Node(current_car, sorted_cars)
        end
        # RED CAR SPECIAL CASE
        if current_car == ROOT && !any(isempty.(sorted_cars))
            check_car = Node(current_car, [union(sorted_cars...)])
            if check_car in visited
                if DFS
                    push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true, visited=true))
                else
                    draw_tree_list = draw_tree_nodes[create_node_string(current_node)]
                    idx = findlast(x->x[1]=='m', draw_tree_list)
                    if idx !== nothing
                        insert!(draw_tree_list, idx+1, create_node_string(new_node, visited=true))
                    else
                        push!(draw_tree_nodes[create_node_string(current_node)], create_node_string(new_node, visited=true))
                    end
                end
                push!(timeline, (new_node, prev_move, current_car))
                continue
            end
        end
        # DO NOT EXPAND IF ALREADY VISITED
        if new_node in visited
            if DFS
                push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true, visited=true))
            else
                draw_tree_list = draw_tree_nodes[create_node_string(current_node)]
                idx = findlast(x->x[1]=='m', draw_tree_list)
                if idx !== nothing
                    insert!(draw_tree_list, idx+1, create_node_string(new_node, visited=true))
                else
                    push!(draw_tree_nodes[create_node_string(current_node)], create_node_string(new_node, visited=true))
                end
            end
            push!(timeline, (new_node, prev_move, current_car))
            continue
        else
            if DFS
                push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true))
            else
                draw_tree_list = draw_tree_nodes[create_node_string(current_node)]
                idx = findlast(x->x[1]=='m', draw_tree_list)
                if idx !== nothing
                    insert!(draw_tree_list, idx+1, create_node_string(new_node))
                else
                    push!(draw_tree_nodes[create_node_string(current_node)], create_node_string(new_node))
                end
            end
            push!(visited, new_node)
        end
        # EXTEND FRONTIER
        for i in eachindex(sorted_cars)
            if length(sorted_cars[i]) > 0
                for j in eachindex(sorted_cars[i])
                    if DFS
                        push!(frontier, (new_node, i, j, (new_node.value, moves[findfirst(x->x==sorted_cars[i], next_cars)])))
                    else
                        pushfirst!(frontier, (new_node, i, j))
                    end
                end
            else
                # IF UNBLOCKING MOVE IS VIABLE, ADD IT TO THE LIST
                for m in moves[isempty.(next_cars)]
                    a = (current_car, m)
                    # if a in actions
                    #     push!(trials[findfirst(x->x==a, actions)], iter)
                    # else
                    if a ∉ actions
                        push!(actions, a)
                        push!(trials, iter)
                    end
                    push!(timeline, (new_node, a, current_car))
                    pushfirst!(draw_tree_nodes[create_node_string(new_node, rev=DFS)], "m"*create_move_icon((current_car, m), board))
                end
            end
        end
    end
    return throw(ErrorException("Could not find solution"))
end

function and_or_tree_move_probability(trials, γ)
    p = (1-γ) .^ trials
    return p / sum(p)
end

actions, trials, visited, timeline, draw_tree_nodes = and_or_tree(board, arr);
g = draw_tree(draw_tree_nodes)

board = load_data(prbs[1])
arr = get_board_arr(board)
draw_board(arr)
prev_move = (-1, 0)
action_traces = zeros(length(unique(arr))-1, 9)
τ = 0.8
β = 10
κ = 0.9
C = 0.5
γ = 0.1
λ = 0.0

arrs = [arr]

for i in 1:1000
    all_actions = get_all_available_moves(board, arr)
    actions, visited, draw_tree_nodes = and_or_tree(board, arr, γ=γ)
    if length(actions) == 0 || rand() < λ
        actions = all_actions
    end
    #g = draw_tree(draw_tree_nodes)
    counter = 1
    as = [action_traces[[a[1], a[2]+5]...] for a in actions]
    p = exp.(-β*as) / sum(exp.(-β*as))
    # display("===$(i)===")
    # display(action_traces)
    # display(actions)
    # display(p)

    move = wsample(actions, p)
    action_traces .*= τ
    action_traces[move[1], 5+move[2]] = 1
    action_traces[move[1], 5-move[2]] += κ*(1-action_traces[move[1], 5-move[2]])
    action_traces[move[1], :] .+= C*(1 .- action_traces[move[1], :])

    make_move!(board, move)
    arr = get_board_arr(board)
    push!(arrs, arr)
    #display(plot!(title=i))

    prev_move = move
end

anim = @animate for i ∈ 1:300
    draw_board(arrs[i])
    plot!(title=i-1, size=(200,230), dpi=300)
end
gif(anim, "anim_fps15.gif", fps = 40)



