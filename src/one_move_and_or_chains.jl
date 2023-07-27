function get_all_and_or_chains(board; max_iter=1000)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    ####### CHAINS APPROACH
    # # AND-OR chains (downward direction, i.e. children)
    # #chains = Vector{Vector{Tuple{s_type, a_type, Vector{s_type}}}}()
    # chains = Vector{Vector{Tuple{s_type, a_type}}}()
    # # individual chains
    # #current_chain = Vector{Tuple{s_type, a_type, Vector{s_type}}}()
    # current_chain = Vector{Tuple{s_type, a_type}}()
    # # keeps track of current AO nodes visited
    current_visited = Vector{s_type}()
    ####### AO TREE APPROACH
    # AND node type
    and_type = Tuple{s_type, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    AND = DefaultDict{or_type, Vector{and_type}}([])
    OR = DefaultDict{and_type, Vector{or_type}}([])
    # initialize process
    arr = get_board_arr(board)
    m_init = 6 - (board.cars[9].x+1)
    ao_root = (9, (m_init,))
    root = (9, m_init)
    #initial_nodes = get_blocking_nodes(board, arr, root)
    # push!(current_chain, (ao_root, root))#, initial_nodes))
    push!(current_visited, ao_root)
    and_node = (ao_root, 1)
    or_node = (ao_root, root, 1)
    push!(OR[and_node], or_node)
    # Expand tree and recurse
    #replan!(chains, current_chain, current_visited, board, arr, 0, max_iter)
    replan!(or_node, current_visited, AND, OR, board, arr, 0, max_iter)
    return (AND, OR)
end

function replan!(or_node, current_visited, AND, OR, board, arr, recursion_depth, max_iter)
    if recursion_depth > max_iter
        throw(DomainError("max_iter depth reached"))
    end
    s, move, d = or_node
    and_node = (s, d)
    child_nodes = get_blocking_nodes(board, arr, move)
    # move is impossible
    if child_nodes == [(0, (0,))]
        return nothing
    end
    # if move is unblocked, have reached end of chain
    if isempty(child_nodes)
        if or_node ∉ OR[and_node]
            push!(OR[and_node], or_node)
        end
        if ((0, (0,)), d+1) ∉ AND[or_node]
            push!(AND[or_node], ((0, (0,)), d+1))
        end
        return nothing
    end
    # recurse all children
    for node in child_nodes
        if node in current_visited
            continue
        end
        and_current = (node, d+1)
        if and_current ∉ AND[or_node]
            push!(AND[or_node], and_current)
        end
        for m in node[2]
            next_move = (node[1], m)
            next_or_node = (node, next_move, d+1)
            if next_or_node ∉ OR[and_current]
                push!(OR[and_current], next_or_node)
            end
            cv = copy(current_visited)
            push!(cv, node)
            replan!(next_or_node, cv, AND, OR, board, arr, recursion_depth + 1, max_iter)
        end
    end
    return nothing
end

function replan!(chains, current_chain, current_visited, board, arr, recursion_depth, max_iter)
    if recursion_depth > max_iter
        throw(DomainError("max_iter depth reached"))
    end
    move = current_chain[end][2]
    child_nodes = get_blocking_nodes(board, arr, move)
    # move is impossible
    if child_nodes == [(0, (0,))]
        return nothing
    end
    # if move is unblocked, have reached end of chain
    if isempty(child_nodes)
        push!(chains, current_chain)
        return nothing
    end
    # recurse all children
    for node in child_nodes
        if node in current_visited
            continue
        end
        for m in node[2]
            next_move = (node[1], m)
            next_node = (node, next_move)#, child_nodes)
            cc = copy(current_chain)
            push!(cc, next_node)
            cv = copy(current_visited)
            push!(cv, node)
            replan!(chains, cc, cv, board, arr, recursion_depth + 1, max_iter)
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
        #ms_new, _ = moves_that_unblock(car1, car2, arr; move_amount=m)
        # Remove impossible moves with cars in same row
        #ms_new = ms_new[(in).(ms_new, Ref(possible_moves(arr, car2)))]
        # If no possible moves, end iteration for this move
        if length(ms_new) == 0
            return [(0, (0,))]
        end
        # new ao state
        push!(ls, (car, Tuple(ms_new)))
    end
    return ls
end

function chains_to_ao(chains)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # AND node type
    and_type = Tuple{s_type, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    AND = DefaultDict{or_type, Vector{and_type}}([])
    OR = DefaultDict{and_type, Vector{or_type}}([])
    for chain in chains
        for (n, node) in enumerate(chain)
            AND_current = (node[1], n)
            OR_node = (node[1], node[2], n)
            AND_next = ()
            if n == length(chain)
                AND_next = ((0, (0,)), n+1)
            else
                AND_next = (chain[n+1][1], n+1)
            end
            if OR_node ∉ OR[AND_current]
                push!(OR[AND_current], OR_node)
            end
            if AND_next ∉ AND[OR_node]
                push!(AND[OR_node], AND_next)
            end
        end
    end
    return (AND, OR)
end


function filter_chains(chains, move)
    new_chains = []
    for chain in chains
        if chain[end][2] == move
            push!(new_chains, chain[1:end-1])
        end
    end
    return new_chains
end

function move_ps(chains, all_moves, heuristics, betas, λ)
    A = length(all_moves)
    N = length(chains)
    H = length(heuristics)
    xs = zeros(N, H)
    chain_first_moves = Vector{Tuple{Int, Int}}(undef, N)
    for (n, chain) in enumerate(chains)
        for (m, h) in enumerate(heuristics)
            xs[n, m] = h(chain)
        end
        chain_first_moves[n] = chain[end][2]
    end
    sm = exp.(xs * betas)
    p_chain = sm ./ sum(sm)
    ps = λ/A .+ (1-λ)*[sum(p_chain[findall(x->x==a, chain_first_moves)]) for a in all_moves]
    return ps
end

function propagate_ps(AO, all_moves, AND_root, γ, λ)
    function propagate!(ps, as, p, AND_current, AO)
        AND, OR = AO
        pp = (1-γ)*p
        ps .+= γ*p/length(ps)
        # ASSIGNING PROBABILITY TO OR NODES
        p_ors = ones(length(OR[AND_current]))/length(OR[AND_current])
        for (n, OR_node) in enumerate(OR[AND_current])
            p_or = pp * p_ors[n]
            # ASSIGNING PROBABILITY TO AND NODES
            p_ands = ones(length(AND[OR_node]))/length(AND[OR_node])
            for (m, AND_next) in enumerate(AND[OR_node])
                p_and = p_or * p_ands[m]
                if AND_next[1] == (0, (0,))
                    ps[findfirst(x->x==OR_node[2], as)] += p_and
                else
                    propagate!(ps, as, p_and, AND_next, AO)
                end
            end
        end
    end
    ps = zeros(length(all_moves))
    propagate!(ps, all_moves, 1.0, AND_root, AO)
    return λ/length(all_moves) .+ (1-λ)*ps
end

function loglik(all_moves, ps, subj_moves)
    ll = 0
    for move in subj_moves
        ll += log(ps[findfirst(x->x==move, all_moves)])
    end
    return ll
end


prb = prbs[sp[4]]
board = load_data(prb);
#make_move!(board, (3, 1))
arr = get_board_arr(board);
draw_board(arr)

chains = get_all_and_or_chains(board; max_iter=100);
all_moves = Tuple.(get_all_available_moves(board, arr))
heuristics = [length];
betas = [-2.0];
λ = 0.1;
ps = move_ps(chains, all_moves, heuristics, betas, λ)

prb_moves_all = [prb in keys(moves_all[subj]) ? moves_all[subj][prb] : [] for subj in subjs]
prb_moves_all = prb_moves_all[length.(prb_moves_all) .> 0]
prb_moves_all = [moves[2] for moves in prb_moves_all]

loglik(all_moves, ps, prb_moves_all)

function subject_fit(x, tot_moves)
    nll = 0
    γ = x[1]
    λ = x[2]
    for prb in keys(tot_moves)
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        println("==========")
        println(prb)
        for move in tot_moves_prb
            println(move)
            if move == (-1, 0)
                board = load_data(prb)
                continue
            end
            arr = get_board_arr(board)
            all_moves = Tuple.(get_all_available_moves(board, arr))
            m_init = 6 - (board.cars[9].x+1)
            ao_root = ((9, (m_init,)), 1)

            chains = get_all_and_or_chains(board)
            println("--")
            AO = chains_to_ao(chains);
            println("==")
            ps = propagate_ps(AO, all_moves, ao_root, γ, λ)
            nll -= log(ps[findfirst(x->x==move, all_moves)])

            make_move!(board, move)
        end
    end
    return nll
end
_, tot_moves, _, _ = analyse_subject(data[subjs[1]]);
nll = subject_fit(x0, tot_moves)

x0 = [0.1, 0.0];
chains_all = Dict{String, Vector{Any}}()
for prb in ProgressBar(prbs)
    board = load_data(prb);
    arr = get_board_arr(board);
    chains_all[prb] = get_all_and_or_chains(board)
end
res = optimize((x) -> subject_fit(x, moves_all, chains_all, prbs, heuristics), [0.0, -10.0], [1.0, 10.0], x0, Fminbox(); autodiff=:forward);
params = Optim.minimizer(res)
fitness = Optim.minimum(res)

prb = prbs[sp[4]]
prb = "prb79230_11"
prb = "prb33117_14"
board = load_data(prb);
for move in moves[2:end]
    make_move!(board, move)
end
arr = get_board_arr(board)
@time AO = get_all_and_or_chains(board);

# @time chains = get_all_and_or_chains(board);
# @time AO = chains_to_ao(chains);

g = draw_ao_tree(AO, board)

@time ps = propagate_ps(AO, all_moves, ((9, (3,)), 1), 0.1, 0.0)

ls = []
ms = []
for chain in new_chains
    push!(ls, length(chain))
    push!(ms, chain[end][2])
end

new_chains = chains;
new_chains = filter_chains(new_chains, (4, -3));
