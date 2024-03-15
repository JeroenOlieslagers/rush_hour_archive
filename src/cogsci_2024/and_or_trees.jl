include("rush_hour.jl")

function get_and_or_tree(board; backtracking=false, idv=false, max_iter=100)
    # AND-OR tree base node type
    # (car_id, (possible_moves,))
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    # (car_id, move_amount)
    a_type = Tuple{Int, Int}
    # AND node type
    # ((car_id, (possible_moves,)), depth)
    and_type = Tuple{s_type, Int}
    # OR node type
    # ((car_id, (possible_moves,)), (car_id, move_amount), depth)
    or_type = Tuple{s_type, a_type, Int}
    # keeps track of current AO nodes visited
    ### CHANGED
    visited = Vector{s_type}()
    #visited = Vector{a_type}()
    # keeps track of parents for backtracking
    parents_AND = DefaultDict{and_type, Vector{or_type}}([])
    parents_OR = DefaultDict{or_type, Vector{and_type}}([])
    parents_moves = DefaultDict{a_type, Vector{or_type}}([])
    # forward pass (children)
    AND = DefaultDict{and_type, Vector{or_type}}([])
    OR = DefaultDict{or_type, Vector{and_type}}([])
    # saves properties of nodes
    idv_AND = Dict{and_type, Matrix}()
    idv_OR = Dict{or_type, Matrix}()
    # initialize process
    arr = get_board_arr(board)
    # ultimate goal move
    m_init = 6 - (board.cars[end].x+1)
    root_move = (length(board.cars), m_init)
    # base for root AND & OR node
    ao_root = (length(board.cars), (m_init,))
    ### CHANGED
    push!(visited, ao_root)
    # for mm in 1:m_init
    #     push!(visited, (length(board.cars), mm))
    # end
    # Root AND
    AND_root = (ao_root, 1)
    push!(parents_AND[AND_root], ((0, (0,)), (0, 0), 0))
    # Root OR
    OR_root = (ao_root, root_move, 1)
    push!(parents_OR[OR_root], AND_root)
    push!(AND[AND_root], OR_root)
    idv_AND[AND_root] = [[0]'; [0]']'
    # Get first child nodes
    child_nodes = get_blocking_nodes(board, arr, root_move)
    # all possible moves from start position
    all_moves = Tuple.(get_all_available_moves(board, arr))
    # Tree with all info
    AND_OR_tree = [AND_root, AND, OR, parents_moves, idv_AND, idv_OR , parents_AND, parents_OR]
    # Expand tree and recurse
    forward!(OR_root, child_nodes, visited, AND_OR_tree, board, arr, 0, max_iter; backtracking=backtracking, idv=idv)
    return all_moves, AND_OR_tree
end
# board = load_data("chain_cycle");
# _, tree = get_and_or_tree(board);
# _, AND, OR, _, _, _, _, _ = tree;
# draw_ao_tree((AND, OR), board)
# draw_board(get_board_arr(board))

function forward!(prev_OR, child_nodes, visited, AND_OR_tree, board, arr, recursion_depth, max_iter; backtracking=false, idv=false)
    AND_root, AND, OR, parents_moves, idv_AND, idv_OR, parents_AND, parents_OR = AND_OR_tree
    if recursion_depth > max_iter
        throw(DomainError("max_iter depth reached"))
    end
    _, move, d = prev_OR

    # if move is unblocked, have reached end of chain
    if isempty(child_nodes)
        if idv
            if !haskey(idv_OR, prev_OR)
                idv_OR[prev_OR] = [[0]'; [0]']'
            end
        end
        if ((0, (0,)), d+1) ∉ OR[prev_OR]
            push!(OR[prev_OR], ((0, (0,)), d+1))
        end
        if prev_OR ∉ parents_moves[move]
            push!(parents_moves[move], prev_OR)
        end
        return nothing
    end
    # calculate features of AND node
    if idv
        a1 = collect(1:length(child_nodes))
        a2 = [length(c[2]) for c in child_nodes]
        if !haskey(idv_OR, prev_OR)
            idv_OR[prev_OR] = [a1'; a2']'
        end
    end
    # recurse all children
    for node in child_nodes
        and_node = (node, d+1)
        if and_node ∉ OR[prev_OR]
            push!(OR[prev_OR], and_node)
        end
        ### CHANGED
        if node in visited
            # and_cycle = ((-2, (-2,)), d+1)
            # if and_cycle ∉ OR[prev_OR]
            #     push!(OR[prev_OR], and_cycle)
            # end
            continue
        end
        ######
        if backtracking
            if prev_OR ∉ parents_AND[and_node]
                push!(parents_AND[and_node], prev_OR)
            end
        end
        # calculate features of OR node
        o1 = Int[]
        o2 = Int[]
        ls = Int[]
        childs = []
        for m in node[2]
            next_move = (node[1], m)
            child_nodes = get_blocking_nodes(board, arr, next_move)
            # move is impossible
            if child_nodes == [(0, (0,))]
                #dict[(-2, -2)] += p
                #return nothing
                continue
            end
            if idv
                push!(o1, length(child_nodes))
                push!(o2, move_blocks_red(board, next_move))
            end
            push!(ls, m)
            push!(childs, child_nodes)
        end
        if idv
            if !haskey(idv_AND, and_node)
                idv_AND[and_node] = [o1'; o2']'
            end
        end
        # loop over next set of OR nodes
        for (j, m) in enumerate(ls)
            next_move = (node[1], m)
            or_node = (node, next_move, d+1)
            if or_node ∉ AND[and_node]
                push!(AND[and_node], or_node)
            end
            if backtracking
                if and_node ∉ parents_OR[or_node]
                    push!(parents_OR[or_node], and_node)
                end
            end
            # we copy because we dont want the same nodes in a chain,
            # but across same chain (at different depths) we can have the same node repeat
            ### CHANGED
            cv = copy(visited)
            push!(cv, node)
            if next_move ∉ visited
                ########
                # cv = copy(visited)
                # for mm in 1:abs(m)
                #     push!(cv, (node[1], sign(m)*mm))
                # end
                ########
                forward!(or_node, childs[j], cv, AND_OR_tree, board, arr, recursion_depth + 1, max_iter; backtracking=backtracking, idv=idv)
            end
        end
    end
    return nothing
end

function get_blocking_nodes(board, arr, move)
    car_id, m = move
    # get all blocking cars
    cars = move_blocked_by(board.cars[car_id], m, arr)
    ls = []
    for car in cars
        # Get all OR nodes of next layer
        car1 = board.cars[car_id]
        car2 = board.cars[car]
        ms_new = unblocking_moves(car1, car2, arr; move_amount=m)
        # If no possible moves, end iteration for this move
        if length(ms_new) == 0
            return [(0, (0,))]
        end
        # new ao state
        push!(ls, (car, Tuple(ms_new)))
    end
    return ls
end

