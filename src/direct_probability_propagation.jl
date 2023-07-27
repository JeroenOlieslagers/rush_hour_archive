function get_move_probabilities(x, board; max_iter=1000)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # keeps track of current AO nodes visited
    current_visited = Vector{s_type}()
    # keeps track of probability over moves
    # also keeps track of probability of making random move
    dict = DefaultDict{a_type, Any}(0.0)
    # initialize process
    arr = get_board_arr(board)
    m_init = 6 - (board.cars[9].x+1)
    ao_root = (9, (m_init,))
    root_move = (9, m_init)
    push!(current_visited, ao_root)
    # fill all moves
    all_moves = Tuple.(get_all_available_moves(board, arr))
    for move in all_moves
        dict[move] = 0.0
    end
    # Expand tree and recurse
    propagate!(x, 1.0, root_move, current_visited, dict, board, arr, 0, max_iter)
    return dict
end

function propagate!(x, p, move, current_visited, dict, board, arr, recursion_depth, max_iter)
    if recursion_depth > max_iter
        throw(DomainError("max_iter depth reached"))
    end
    γ = x[1]
    #move = current_node[2]
    child_nodes = get_blocking_nodes(board, arr, move)
    # move is impossible
    if child_nodes == [(0, (0,))]
        dict[(-2, -2)] += p
        return nothing
    end
    # if move is unblocked, have reached end of chain
    if isempty(child_nodes)
        dict[move] += p
        return nothing
    end
    # assign probability to each child node being selected
    p_children = p / length(child_nodes)
    # recurse all children
    for node in child_nodes
        if node in current_visited
            dict[(-2, -2)] += p_children
            continue
        end
        # assign probability to each move being selected
        p_move = p_children / length(node[2])
        for m in node[2]
            next_move = (node[1], m)
            cv = copy(current_visited)
            push!(cv, node)
            p_stop = γ .* p_move
            dict[(-1, -1)] += p_stop
            p_continue = (1-γ) .* p_move
            propagate!(x, p_continue, next_move, cv, dict, board, arr, recursion_depth + 1, max_iter)
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

function process_dict(dict)
    moves = []
    ps = []
    for move in keys(dict)
        if move[1] > 0
            push!(moves, move)
            push!(ps, dict[move])
        end
    end
    # repeating because of cycle
    sp = sum(ps)
    p_cycle = dict[(-2, -2)]
    for i in eachindex(ps)
        ps[i] += p_cycle*(ps[i]/sp)
    end
    # random move in case of stopping early
    ps .+= dict[(-1, -1)]/length(ps)
    return moves, ps
end

function subject_nll(x, tot_moves)
    nll = 0
    γ = exp(-x[1])
    λ = x[2]
    for prb in keys(tot_moves)
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        for move in tot_moves_prb
            if move == (-1, 0)
                board = load_data(prb)
                continue
            end
            dict = get_move_probabilities([γ], board);
            all_moves, ps = process_dict(dict);
            ps = λ/length(ps) .+ (1-λ)*ps

            nll -= log(ps[findfirst(x->x==move, all_moves)])

            make_move!(board, move)
        end
    end
    return nll
end

function fit_subject(tot_moves, x0)
    res = optimize((x) -> subject_nll(x, tot_moves), [0.0, 0.0], [20.0, 1.0], x0, Fminbox(); autodiff=:forward)
    return Optim.minimizer(res), Optim.minimum(res)
end

function all_subjects_fit(all_subj_moves, x0)
    params = zeros(length(all_subj_moves), 2)
    fitness = 0
    for (n, subj) in ProgressBar(enumerate(keys(all_subj_moves)))
        x, nll = fit_subject(all_subj_moves[subj], x0)
        params[n, :] = x
        fitness += nll
    end
    return params, fitness
end

all_subj_moves = Dict{String, Dict{String, Vector{Tuple{Int64, Int64}}}}()
for subj in ProgressBar(subjs[1:2])
    _, tot_moves, _, _ = analyse_subject(data[subj]);
    all_subj_moves[subj] = tot_moves
end
x0 = [2.0, 0.1];
params, fitness = all_subjects_fit(all_subj_moves, x0)

nll = subject_fit(x0, tot_moves)

prb = prbs[sp[4]]
board = load_data(prb);
arr = get_board_arr(board)
dict = get_move_probabilities(x0, board);

moves, ps = process_dict(dict);



