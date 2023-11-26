include("basics_for_hpc.jl")
using StatsBase
#using Plots
using Distributions

function get_and_or_tree(board; backtracking=false, max_iter=100)
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
    visited = Vector{s_type}()
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
    ao_root = (length(board.cars), (m_init,))
    root_move = (length(board.cars), m_init)
    # all possible moves from start position
    all_moves = Tuple.(get_all_available_moves(board, arr))
    push!(visited, ao_root)
    AND_root = (ao_root, 1)
    OR_root = (ao_root, root_move, 1)
    push!(parents_AND[AND_root], ((0, (0,)), (0, 0), 0))
    push!(parents_OR[OR_root], AND_root)
    push!(AND[AND_root], OR_root)
    idv_AND[AND_root] = [[0]'; [0]']'
    child_nodes = get_blocking_nodes(board, arr, root_move)
    # Expand tree and recurse
    AND_OR_tree = [AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR]
    forward!(OR_root, child_nodes, visited, AND_OR_tree, board, arr, 0, max_iter; backtracking=backtracking)
    return all_moves, AND_OR_tree
end

function forward!(prev_OR, child_nodes, visited, AND_OR_tree, board, arr, recursion_depth, max_iter; backtracking=backtracking)
    AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR = AND_OR_tree
    if recursion_depth > max_iter
        throw(DomainError("max_iter depth reached"))
    end
    _, move, d = prev_OR

    # if move is unblocked, have reached end of chain
    if isempty(child_nodes)
        if !haskey(idv_OR, prev_OR)
            idv_OR[prev_OR] = [[0]'; [0]']'
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
    a1 = collect(1:length(child_nodes))
    a2 = [length(c[2]) for c in child_nodes]
    if !haskey(idv_OR, prev_OR)
        idv_OR[prev_OR] = [a1'; a2']'
    end
    # recurse all children
    for (i, node) in enumerate(child_nodes)
        and_node = (node, d+1)
        if and_node ∉ OR[prev_OR]
            push!(OR[prev_OR], and_node)
        end
        if node in visited
            # and_cycle = ((-2, (-2,)), d+1)
            # if and_cycle ∉ OR[prev_OR]
            #     push!(OR[prev_OR], and_cycle)
            # end
            continue
        end
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
            push!(o1, length(child_nodes))
            push!(o2, move_blocks_red(board, next_move))
            push!(ls, m)
            push!(childs, child_nodes)
        end
        if !haskey(idv_AND, and_node)
            idv_AND[and_node] = [o1'; o2']'
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
            cv = copy(visited)
            push!(cv, node)
            forward!(or_node, childs[j], cv, AND_OR_tree, board, arr, recursion_depth + 1, max_iter; backtracking=backtracking)
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

function propagate_ps(x, AND_OR_tree)
    AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR = AND_OR_tree
    function propagate!(x, p, visited, dict, AND_current, AND, OR, idv_AND, idv_OR)
        #γ = x[1]
        #γ = exp(-x[1])
        #γ = 0.01
        #γ = exp(-1.63144)
        γ = 0.0
        # β_AND1 = x[3]
        # β_AND2 = x[4]
        # β_OR1 = x[5]
        # β_OR2 = x[6]
        # ASSIGNING PROBABILITY TO AND NODES
        if !haskey(AND, AND_current)
            dict[((-2, (-2,)), (-2, -2), AND_current[2])] += p
            return nothing
        end
        mat_and = idv_AND[AND_current]
        #p_ands = size(mat_and)[1] == 1 ? p : p*softmax(mat_and, [β_AND1, β_AND2])
        p_ands = p*ones(size(mat_and)[1]) ./ size(mat_and)[1]
        for (n, OR_node) in enumerate(AND[AND_current])
            p_and = p_ands[n]
            if OR_node[1] in visited
                dict[((-2, (-2,)), (-2, -2), OR_node[3])] += p_and
                continue
            end
            # ASSIGNING PROBABILITY TO OR NODES
            mat_or = idv_OR[OR_node]
            #p_ors = size(mat_or)[1] == 1 ? p_and : p_and*softmax(mat_or, [β_OR1, β_OR2])
            p_ors = p_and*ones(size(mat_or)[1]) ./ size(mat_or)[1]
            for (m, AND_next) in enumerate(OR[OR_node])
                p_or = p_ors[m]
                if AND_next[1] == (0, (0,))
                    dict[OR_node] += p_or
                # elseif AND_next[1] == (-2, (-2,))
                #     dict[((-2, (-2,)), (-2, -2), OR_node[3])] += p_or
                else
                    pp = (1-γ)*p_or
                    dict[((-1, (-1,)), (-1, -1), OR_node[3])] += γ*p_or
                    cv = copy(visited)
                    push!(cv, OR_node[1])
                    propagate!(x, pp, cv, dict, AND_next, AND, OR, idv_AND, idv_OR)
                end
            end
        end
    end
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    dict = DefaultDict{or_type, Any}(0.0)
    # keeps track of current AO nodes visited
    visited = Set{s_type}()
    propagate!(x, 1.0, visited, dict, AND_root, AND, OR, idv_AND, idv_OR)
    return dict
end

function first_pass(tot_moves)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # AND node type
    and_type = Tuple{s_type, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    # forward pass
    dicts = Vector{Dict{or_type, Any}}()
    all_all_moves = Vector{Vector{a_type}}()
    # AND = DefaultDict{and_type, Vector{or_type}}
    # OR = DefaultDict{or_type, Vector{and_type}}
    # idv_AND = Dict{and_type, Matrix}
    # idv_OR = Dict{or_type, Matrix}
    # trees = Vector{Tuple{dict, and_type, AND, OR, idv_AND, idv_OR}}()
    trees = []
    moves = Vector{a_type}()

    dicts_prb = DefaultDict{String, Vector{Vector}}([])
    all_all_moves_prb = DefaultDict{String, Vector{Vector}}([])
    trees_prb = DefaultDict{String, Vector{Vector}}([])
    moves_prb = DefaultDict{String, Vector{Vector}}([])
    for prb in collect(keys(tot_moves))
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        restart_count = 0
        for move in tot_moves_prb
            if move == (-1, 0)
                board = load_data(prb)
                restart_count += 1
                push!(dicts_prb[prb], [])
                push!(all_all_moves_prb[prb], [])
                push!(trees_prb[prb], [])
                push!(moves_prb[prb], [])
                continue
            end
            arr = get_board_arr(board)
            if check_solved(arr)
                continue
            end
            all_moves, AND_OR_tree = get_and_or_tree(board; backtracking=true);
            push!(trees, AND_OR_tree)
            push!(trees_prb[prb][restart_count], AND_OR_tree)
            push!(all_all_moves, all_moves)
            push!(all_all_moves_prb[prb][restart_count], all_moves)
            dict = propagate_ps(zeros(8), AND_OR_tree)
            push!(dicts, dict)
            push!(dicts_prb[prb][restart_count], dict)
            push!(moves, move)
            push!(moves_prb[prb][restart_count], move)
            make_move!(board, move)
        end
    end
    tree_data = [trees, dicts, all_all_moves, moves]
    tree_data_prb = [trees_prb, dicts_prb, all_all_moves_prb, moves_prb]
    return tree_data, tree_data_prb
end

function get_Q(board, A, d_goals)
    Q = zeros(length(A))
    ns = zeros(BigInt, length(A))
    for (n, a) in enumerate(A)
        make_move!(board, a)
        arr_a = get_board_arr(board)
        s = board_to_int(arr_a, BigInt)
        # Ignore if move outside state space
        if s ∉ keys(d_goals)
            println("oops")
            undo_moves!(board, [a])
            return []
        end
        Q[n] = d_goals[s]
        ns[n] = s
        undo_moves!(board, [a])
    end
    return Q, ns
end

function get_QQs(tot_moves, tot_times, all_all_moves_prb, d_goals)
    QQs = DefaultDict{String, Vector{Vector}}([])
    RTs = DefaultDict{String, Vector{Vector}}([])
    d_goal_subj = DefaultDict{String, Vector{Vector}}([])
    states_subj = DefaultDict{String, Vector{Vector}}([])
    for prb in collect(keys(tot_moves))
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        tot_times_prb = tot_times[prb]
        restart_count = 0
        for (n, move) in enumerate(tot_moves_prb)
            if move == (-1, 0)
                board = load_data(prb)
                restart_count += 1
                push!(QQs[prb], [])
                push!(RTs[prb], [])
                push!(d_goal_subj[prb], [])
                push!(states_subj[prb], [])
                continue
            end
            arr = get_board_arr(board)
            s = board_to_int(arr, BigInt)
            if check_solved(arr)
                continue
            end
            all_moves = all_all_moves_prb[prb][restart_count][length(QQs[prb][restart_count])+1]
            #A = get_all_available_moves(board, arr)
            Q, ns = get_Q(board, all_moves, d_goals)
            push!(QQs[prb][restart_count], Q)
            push!(RTs[prb][restart_count], tot_times_prb[n] - tot_times_prb[n-1])
            push!(d_goal_subj[prb][restart_count], d_goals[s])
            push!(states_subj[prb][restart_count], s)
            make_move!(board, move)
        end
    end
    return QQs, RTs, d_goal_subj, states_subj
end

function process_dict(all_moves, dict, excl_moves, μ_same, μ_block, board)
    ps = []
    move_dict = Dict{Tuple{Int, Int}, Any}()
    # add all possible moves
    for move in all_moves
        move_dict[move] = 0.0
    end
    # random move
    move_dict[(-1, -1)] = 0.0
    # cycle
    move_dict[(-2, -2)] = 0.0
    # modulate probabilities by their depths
    p_excl = 0
    for or_node in keys(dict)
        #if or_node[2] == (-1, -1)
            #move_dict[or_node[2]] += dict[or_node]
        if or_node[2] ∉ excl_moves
            if !move_blocks_red(board, or_node[2])
                move_dict[or_node[2]] += dict[or_node]
            else
                move_dict[or_node[2]] += μ_block * dict[or_node]
                p_excl += (1-μ_block) * dict[or_node]
            end
        else
            if !move_blocks_red(board, or_node[2])
                move_dict[or_node[2]] += μ_same * dict[or_node]
                p_excl += (1-μ_same) * dict[or_node]
            else
                move_dict[or_node[2]] += μ_block * μ_same * dict[or_node]
                p_excl += (1-μ_block) * μ_same * dict[or_node]
                p_excl += (1-μ_same) * dict[or_node]
            end
        end
    end
    # CHANGED
    move_dict[(-2, -2)] += p_excl
    # vectorize move probabilities
    for move in all_moves
        push!(ps, move_dict[move])
    end
    # repeating because of cycle
    sp = sum(ps) + move_dict[(-1, -1)]
    if sp > 0
        p_cycle = move_dict[(-2, -2)]
        for i in eachindex(ps)
            ps[i] += p_cycle*(ps[i]/sp)
        end
        move_dict[(-1, -1)] += p_cycle*(move_dict[(-1, -1)]/sp) 
    else
        throw(DomainError("how is sum for cycle probability 0?"))
    end
    # random move in case of stopping early
    N = sum(ps .== 0)
    ps .+= move_dict[(-1, -1)]/length(ps)
    return ps, move_dict[(-1, -1)], N
end

function process_dict2(x, all_moves, dict)
    ps = []
    move_dict = Dict{Tuple{Int, Int}, Any}()
    # add all possible moves
    for move in all_moves
        move_dict[move] = 0.0
    end
    # random move
    move_dict[(-1, -1)] = 0.0
    # cycle
    move_dict[(-2, -2)] = 0.0
    # modulate probabilities by their depths
    for or_node in keys(dict)
        move = or_node[2]
        p = dict[or_node]
        d = or_node[3]
        #modulator = 1.0
        #modulator = 1 - cdf(Weibull(x[1], x[2]), d)
        #modulator = 1 - cdf(LogNormal(x[1], x[2]), d)
        #modulator = 1 - cdf(Chisq(x[1]), d)
        #modulator = 1 - cdf(Gamma(x[1], x[2]), d)
        #modulator = 1 - cdf(InverseGamma(x[1], x[2]), d)
        #modulator = 1 - cdf(truncated(Normal(x[1], x[2]), lower=0), d)
        #modulator = 1-bin_cdf(round(Int, x[1]), x[2], d)
        move_dict[move] += p#*modulator
        #move_dict[(-1, -1)] += p*(1-modulator)
    end
    # vectorize move probabilities
    for move in all_moves
        push!(ps, move_dict[move])
    end
    # repeating because of cycle
    sp = sum(ps) + move_dict[(-1, -1)]
    if sp > 0
        p_cycle = move_dict[(-2, -2)]
        for i in eachindex(ps)
            ps[i] += p_cycle*(ps[i]/sp)
        end
        move_dict[(-1, -1)] += p_cycle*(move_dict[(-1, -1)]/sp) 
    else
        throw(DomainError("how is sum for cycle probability 0?"))
    end
    # random move in case of stopping early
    ps .+= move_dict[(-1, -1)]/length(ps)

    # μ = x[end-1]
    # idx = Int[]
    # for n in eachindex(all_moves)
    #     if all_moves[n][1] == 9
    #         push!(idx, n)
    #     end
    # end
    # ps[idx] = μ/length(idx) .+ (1-μ)*ps[idx]
    λ = x[end]
    ps = λ/length(ps) .+ (1-λ)*ps
    return ps
end

function only_gamma_dict(dict, γ)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # AND node type
    and_type = Tuple{s_type, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    new_dict = Dict{or_type, Any}()
    factor = 1 / sum(values(dict) .> 0)
    for (k, v) in dict
        if v > 0
            new_dict[k] = v*(1-γ)^(k[3]-0)#
        end
    end
    new_dict[((-1, (-1,)), (-1, -1), 0)] = 1 - sum(values(new_dict))
    return new_dict
end

function logit_lookup_table(β, c; max_d=30)
    look_up_table = Dict{Int, Any}()
    for d in 0:max_d
        factor = 1 - (1 / (1 + exp(-β*(d-c))))
        if d == 0
            look_up_table[d] = factor
        else
            look_up_table[d] = factor * look_up_table[d-1]
        end
    end
    return look_up_table
end

#function logit_dict(dict, look_up_table)
function logit_dict(dict, β, c)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # AND node type
    and_type = Tuple{s_type, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    new_dict = Dict{or_type, Any}()
    for (k, v) in dict
        if v > 0
            new_dict[k] = v / (1 + exp(-β * (k[3] - c)))#look_up_table[k[3]]#(1-γ)^(k[3]-0)
        end
    end
    new_dict[((-1, (-1,)), (-1, -1), 0)] = round(1000000000*(1 - sum(values(new_dict))))/1000000000#1 - sum(values(new_dict))#
    return new_dict
end

function subject_nll(x, dicts, all_all_moves, moves)
    nll = 0
    for i in eachindex(moves)
        move = moves[i]
        dict = dicts[i]
        all_moves = all_all_moves[i]
        ps = process_dict2(x, all_moves, dict)
        p = ps[findfirst(x->x==move, all_moves)]
        nll -= log(p)
    end
    return nll
end

function simulate_gamma(x, dicts, all_all_moves; K=1)
    γ = x
    moves = []
    for k in 1:K
        for i in eachindex(dicts)
            dict = dicts[i]
            all_moves = all_all_moves[i]
            new_dict = only_gamma_dict(dict, γ)
            ps = process_dict(all_moves, new_dict)
            # idx = Int[]
            # for n in eachindex(all_moves)
            #     if all_moves[n][1] == 9
            #         push!(idx, n)
            #     end
            # end
            # if !isempty(idx)
            #     ps[idx] = sum(ps[idx])*μ/length(idx) .+ (1-μ)*ps[idx]
            # end
            #ps = λ/length(ps) .+ (1-λ)*ps
            push!(moves, wsample(all_moves, Float64.(ps)))
        end
    end
    return moves
end


function subject_nll_gamma(x, dicts, all_all_moves, moves, boards, tree, first_move)
    nll = 0
    #γ = x[1]
    #k = x[2]
    γ = x
    #γ = x[1]
    #μ_red = x[2]
    # μ_same = x[3]
    # μ_block = x[4]
    #λ = x[3]
    for i in eachindex(moves)
        move = moves[i]
        board = boards[i]
        #excl_moves = [i > 1 ? (moves[i-1][1], -moves[i-1][2]) : (0, 0)]
        excl_moves = i > 1 ? [(moves[i-1][1], j) for j in -4:4] : [(0, 0)]
        #excl_moves = [(0, 0)]
        dict = dicts[i]
        all_moves = all_all_moves[i]
        new_dict = only_gamma_dict(dict, γ)
        ps, _, _ = process_dict(all_moves, new_dict, excl_moves, 0.0, 1.0, board)

        # ps, _, _ = process_dict(all_moves, new_dict, excl_moves, μ_same, μ_block, board)
        # idx = Int[]
        # unblockable = false
        # move_over = 0
        # for n in eachindex(all_moves)
        #     if all_moves[n][1] == 9
        #         #push!(idx, n)
        #         if all_moves[n][2] > move_over
        #             move_over = all_moves[n][2]
        #             idx = [n]
        #         end
        #     end
        #     # if all_moves[n] == (9, 3)
        #     #     push!(idx, n)
        #     #     for k in keys(dict)
        #     #         if k[3] == 2 && k[2] != (-1, -1)
        #     #             unblockable = true
        #     #         end
        #     #     end
        #     # end
        # end
        # if !isempty(idx)# && !unblockable
        #     #ps[idx] = sum(ps[idx])*μ_red/length(idx) .+ (1-μ_red)*ps[idx]
        #     ps = μ_red*[ci in idx for (ci, _) in enumerate(ps)]/length(idx) .+ (1-μ_red)*ps
        # end
        if round(100000*sum(ps))/100000 != 1
            println("===============")
            println(sum(ps))
            println(ps)
            throw(DomainError("Not a valid probability distribution"))
        end
        #ps = λ/length(ps) .+ (1-λ)*ps

        # AND OR PARENTS SECTION
        # is_first = first_move[i]
        # if !is_first
        #     _, A, O, _, _, parents_moves, parents_AND, parents_OR = tree[i-1]
        #     prev_move = moves[i-1]
        #     if prev_move in keys(parents_moves)
        #         prev_parent_moves = []
        #         # looks at OR nodes corresponding to previous move
        #         for OR in parents_moves[prev_move]
        #             # parent AND nodes of those ORs
        #             for AND in parents_OR[OR]
        #                 # possible next moves after
        #                 for new_OR in parents_AND[AND]
        #                     possible_move = new_OR[2]
        #                     if possible_move ∉ prev_parent_moves
        #                         push!(prev_parent_moves, possible_move)
        #                     end
        #                 end
        #             end
        #         end
        #         pss = zeros(length(ps))
        #         for n in eachindex(all_moves)
        #             pss[n] = all_moves[n] in prev_parent_moves
        #         end
        #         if sum(pss) > 0
        #             pss ./= sum(pss)
        #             ps = (k .* pss) .+ ((1-k) .* ps)
        #         end
        #     end
        # end


        p = ps[findfirst(x->x==move, all_moves)]
        #p = λ/length(all_moves) + (1-λ)*p
        if p == 0
            println("===============")
            println("Zero probability move")
            println(i)
            continue
        end
        nll -= log(p)
    end
    return nll
end

function subject_nll_logit(x, dicts, all_all_moves, moves)
    nll = 0
    β = x[1]
    c = x[2]
    #lookup_table = logit_lookup_table(β, c)
    for i in eachindex(moves)
        move = moves[i]
        dict = dicts[i]
        all_moves = all_all_moves[i]
        #new_dict = logit_dict(dict, lookup_table)
        new_dict = logit_dict(dict, β, c)
        ps = process_dict(all_moves, new_dict)
        if round(100000*sum(ps))/100000 != 1
            println("===============")
            println(sum(ps))
            println(ps)
            println(idx)
            println("Not a valid probability distribution")
        end
        p = ps[findfirst(x->x==move, all_moves)]
        if p == 0
            nll += Inf
        else
            nll -= log(p)
        end
    end
    return nll
end

# function fsf(x)
#     return -cdf(truncated(Normal(4, 10), lower=0), x[1])
# end
# @btime optimize(fsf, [0.0], [10.0], [1.0], Fminbox(), Optim.Options(f_tol = 0.01); autodiff=:forward)

function get_all_subj_moves(data)
    all_subj_moves = Dict{String, Dict{String, Vector{Tuple{Int, Int}}}}()
    all_subj_states = Dict{String, Dict{String, Vector{BigInt}}}()
    all_subj_times = Dict{String, Dict{String, Vector{Int}}}()
    for subj in collect(keys(data))
        _, tot_moves, tot_states_visited, _, tot_times = analyse_subject(data[subj]);
        all_subj_moves[subj] = tot_moves
        all_subj_states[subj] = tot_states_visited
        all_subj_times[subj] = tot_times
    end
    return all_subj_moves, all_subj_states, all_subj_times
end

function fit_subject(tree_data, x0)
    tree, dicts, all_all_moves, moves = tree_data
    res = optimize((x) -> subject_nll(x, dicts, all_all_moves, moves), [0.0, 0.0, 0.0], [20.0, 10.01, 1.0], x0, Fminbox(), Optim.Options(f_tol = 0.1); autodiff=:forward)
    #res = optimize((x) -> subject_nll(x, dicts, all_all_moves, moves), [0.0, 0.0], [10.0, 1.0], x0, Fminbox(), Optim.Options(f_tol = 1.0); autodiff=:forward)
    return Optim.minimizer(res), Optim.minimum(res)
end

function all_subjects_fit(tree_datas, x0)
    params = zeros(length(tree_datas), length(x0))
    fitness = zeros(length(tree_datas))
    subjs = collect(keys(tree_datas))
    M = length(tree_datas)
    for m in ProgressBar(1:M)#Threads.@threads 
        x, nll = fit_subject(tree_datas[subjs[m]], x0)
        params[m, :] = x
        fitness[m] = nll
    end
    return params, fitness
end

function get_all_subjects_first_pass(all_subj_moves)
    tree_datas = Dict{String, Any}()
    tree_datas_prb = Dict{String, Any}()
    for subj in ProgressBar(keys(all_subj_moves))
        tree_data, tree_data_prb = first_pass(all_subj_moves[subj]);
        tree_datas[subj] = tree_data
        tree_datas_prb[subj] = tree_data_prb
    end
    return tree_datas, tree_datas_prb
end

function get_all_subjects_QQs(all_subj_moves, all_subj_times, tree_datas_prb)
    d_goals = load("data/processed_data/d_goals.jld2")["d_goals"];
    QQs_prb = Dict{String, Any}()
    RTs_prb = Dict{String, Any}()
    d_goal_prb = Dict{String, Any}()
    states_prb = Dict{String, Any}()
    for subj in ProgressBar(keys(all_subj_moves))
        QQs, RTs, d_goals_subj, states_subj = get_QQs(all_subj_moves[subj], all_subj_times[subj], tree_datas_prb[subj][3], d_goals);
        QQs_prb[subj] = QQs
        RTs_prb[subj] = RTs
        d_goal_prb[subj] = d_goals_subj
        states_prb[subj] = states_subj
    end
    return QQs_prb, RTs_prb, d_goal_prb, states_prb
end

function get_all_boards(visited_states)
    boards_all = Dict{String, Any}()
    boards_prb_all = Dict{String, Any}()
    is_first_move = Dict{String, Any}()
    for subj in ProgressBar(keys(visited_states))
        boards = []
        boards_prb = DefaultDict{String, Vector{Vector}}([])
        first_moves = []
        for prb in keys(visited_states[subj])
            for r in eachindex(visited_states[subj][prb])
                push!(boards_prb[prb], [])
                states = visited_states[subj][prb][r]
                for (n, state) in enumerate(states)
                    board = arr_to_board(int_to_arr(state))
                    push!(boards, board)
                    push!(boards_prb[prb][r], board)
                    if n == 1
                        push!(first_moves, true)
                    else
                        push!(first_moves, false)
                    end
                end
            end
        end
        boards_all[subj] = boards
        boards_prb_all[subj] = boards_prb
        is_first_move[subj] = first_moves
    end
    return boards_all, boards_prb_all, is_first_move
end

all_subj_moves, all_subj_states, all_subj_times = get_all_subj_moves(data);
tree_datas, tree_datas_prb = get_all_subjects_first_pass(all_subj_moves);
full_heur_dict_opt = load("data/processed_data/full_heur_dict_opt.jld2");
IDV = load("data/processed_data/IDV_OLD.jld2");
qqs, QQs, visited_states, neighbour_states = get_state_data(data, full_heur_dict_opt, heur=7);
boards, boards_prb, is_first_move = get_all_boards(visited_states);
QQs_prb, RTs_prb, d_goal_prb, states_prb = get_all_subjects_QQs(all_subj_moves, all_subj_times, tree_datas_prb);
#@save "data/processed_data/tree_datas_gamma=0.jld2" tree_datas
#save("data/processed_data/tree_datas_gamma=0.jld2", tree_datas)
#@load "data/processed_data/tree_datas_gamma=0.jld2" tree_datas
tree_data, tree_data_prb = first_pass(all_subj_moves[subjs[1]]);
tree, dicts, all_all_moves, moves = tree_data;
AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR = tree[1];
nd = only_gamma_dict(dict, 0.01)
pd1 = process_dict2([0], all_all_moves[1], nd)
subject_nll_logit
params1 = zeros(42)
fitness1 = zeros(42)
for m in ProgressBar(1:42)#Threads.@threads 
    tree_data = tree_datas[subjs[m]]
    tree, dicts, all_all_moves, moves = tree_data;
    res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves), 0.0, 1.0)
    #res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves), [0.0, 0.0], [1.0, 1.0], [0.12, 0.1], Fminbox(), Optim.Options(f_tol = 0.1); autodiff=:forward)
    params1[m] = Optim.minimizer(res)
    fitness1[m] = Optim.minimum(res)
end

#1 only gamma
#2 QQs
#3 eureka
#10 only gamma, no undo
#11 only gamma, no same car
#13 gamma, lambda, no same car
#13 gamma, mu, lambda, no same car
#14 gamma, new mu, lambda, no same car
#15 gamma, new mu, lambda
#16 gamma, specific (9, 3) mu, lambda
#17 gamma, specific (9, 3) mu, lambda, no same car
#18 gamma, new mu, lambda, no same car (cycle)
#19 gamma, new mu, no same car mu (cycle), lambda
#20 gamma, new mu, no same car mu (cycle), no block red mu (cycle), lambda
#21 gamma, new mu slide over, no same car mu (cycle), no block red mu (cycle), lambda
#22 gamma, k
#23 gamma, k, lambda
#24 gamma, k, no same car
#25
#26 gamma, new mu slide over, no same car (cycle)
#27 only gamma, no same car (no log for NLL)


params27 = zeros(42)
fitness27 = zeros(42)
Threads.@threads for m in ProgressBar(1:42)#
    tree_data = tree_datas[subjs[m]]
    boards_ = boards[subjs[m]]
    first_move = is_first_move[subjs[m]]
    tree, dicts, all_all_moves, moves = tree_data;
    #res = optimize((x) -> subject_nll_logit(x, dicts, all_all_moves, moves), [-10.0, -100.0], [10.0, 100.0], [0.1, 40.0], Fminbox(), Optim.Options(f_tol = 0.1); autodiff=:forward)
    #res = optimize((x) -> subject_nll_logit(x, dicts, all_all_moves, moves), [-0.5, 6.0], Optim.Options(f_tol = 0.1))#, [-2.0, 0.0], [0.0, 30.0], Fminbox(); autodiff=:forward)
    res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves, boards_, tree, first_move), 0.000001, 0.999999)
    #res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves, boards_, tree, first_move), [0.000001, 0.000001, 0.000001, 0.000001, 0.000001], [0.999999, 0.999999, 0.999999, 0.999999, 0.999999], [0.1, 0.1, 0.1, 0.1, 0.1], Fminbox(), Optim.Options(f_tol = 0.1); autodiff=:forward)
    #res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves, boards_, tree, first_move), [0.000001, 0.000001], [0.999999, 0.999999], [0.1, 0.1], Fminbox(), Optim.Options(f_tol = 0.1); autodiff=:forward)
    #P = MinimizationProblem((x) -> subject_nll_logit(x, dicts, all_all_moves, moves), [-2.0, 0.0], [0.0, 20.0])
    #res = multistart_minimization(TikTak(100), NLoptLocalMethod(NLopt.LN_BOBYQA), P)
    params27[m] = Optim.minimizer(res)
    fitness27[m] = Optim.minimum(res)
    #params9[m, :] = res.location
    #fitness9[m] = res.value
end

res1 = 0
res2 = 0
for m in ProgressBar(1:42)#Threads.@threads 
    tree_data = tree_datas[subjs[m]]
    tree, dicts, all_all_moves, moves = tree_data;
    res1 += subject_nll_gamma(0.12, dicts, all_all_moves, moves)
    res2 += subject_nll_logit([0.0326, 72], dicts, all_all_moves, moves)
end



params1_rec = zeros(42)
fitness1_rec = zeros(42)
for m in ProgressBar(1:42)
    tree_data = tree_datas[subjs[m]]
    tree, dicts, all_all_moves, moves = tree_data;
    sim_moves = simulate_gamma(params1[m], dicts, all_all_moves; K=10)
    res = optimize((x) -> subject_nll_gamma(x, repeat(dicts, 10), repeat(all_all_moves, 10), sim_moves), 0.0, 1.0)
    #res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves), [0.0, 0.0], [1.0, 1.0], [0.12, 0.1], Fminbox(), Optim.Options(f_tol = 0.1); autodiff=:forward)
    params1_rec[m] = Optim.minimizer(res)
    fitness1_rec[m] = Optim.minimum(res)
end

xxs = []
yys = []
for m in ProgressBar(1:42)
    ys = []
    xs = []
    tree_data = tree_datas[subjs[m]]
    tree, dicts, all_all_moves, moves = tree_data;
    for x in collect(0.01:0.01:1)
        push!(xs, x)
        push!(ys, subject_nll_gamma(x, dicts, all_all_moves, moves))
    end
    push!(xxs, xs)
    push!(yys, ys)
end
Y = zeros(42, 100)
for i in 1:42
    Y[i, :] = (yys[i] .- mean(yys[i])) ./ std(yys[i])
end

# nll=3013. eureka 3077
subject_nll([1.63144, 0, 0, 0, 0, 0, 0.0272332], dicts, all_all_moves, moves)

x0 = [8.0, 2.005, 0.2];
@time subject_nll(x0, dicts, all_all_moves, moves)
@time subject_nll1([1.63144, 0, 0, 0, 0, 0, 0.0272332], all_subj_moves[subjs[1]])
a, b = fit_subject(tree_data, x0)
params_weibull, fitness_weibull = all_subjects_fit(tree_datas, x0)

#all_moves, AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR = get_and_or_tree(load_data(prbs[sp[end-1]]));
all_moves, AND_OR_tree = get_and_or_tree(board);
dict = propagate_ps(zeros(7), AND_OR_tree)
p = process_dict2(x0, all_moves, dict)

##########

function bin_cdf(n, p, k)
    function bin_pdf(n, p, kk)
        #bin_coeff = factorial(n) / (factorial(kk)*factorial(n-kk))
        bin_coeff = binn(n, kk)
        return bin_coeff * p^kk * (1-p)^(n-kk)
    end
    res = 0
    for y in 0:k
        if y <= n
            res += bin_pdf(n, p, y)
        end
    end
    return res
end

Base.@assume_effects :terminates_locally function binn(n, k)
    n0, k0 = n, k
    k < 0 && return 0
    sgn = 1
    if n < 0
        n = -n + k - 1
        if isodd(k)
            sgn = -sgn
        end
    end
    k > n && return 0
    (k == 0 || k == n) && return sgn
    k == 1 && return sgn*n
    if k > (n>>1)
        k = (n - k)
    end
    x = nn = n - k + 1
    nn += 1
    rr = 2
    while rr <= k
        xt = div(x * nn, rr)
        x = xt
        x == xt || throw(OverflowError(LazyString("binomial(", n0, ", ", k0, " overflows")))
        rr += 1
        nn += 1
    end
    abs(x) * sign(sgn)
end

##########################

##########################


ds = []
ts = []
for subj in ProgressBar(subjs)
    tot_times = all_subj_times[subj]
    tot_moves = all_subj_moves[subj]
    for prb in collect(keys(tot_moves))
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        tot_times_prb = tot_times[prb]
        for (i, move) in enumerate(tot_moves_prb)
            if move == (-1, 0)
                board = load_data(prb)
                continue
            end
            t = tot_times_prb[i]
            all_moves, AND_OR_tree = get_and_or_tree(board)
            AND_root, AND, OR, idv_AND, idv_OR, parents_moves, _, _ = AND_OR_tree
            #push!(trees, (dict, AND_root, AND, OR, idv_AND, idv_OR, parents_moves))
            dict = propagate_ps(zeros(8), AND_root, AND, OR, idv_AND, idv_OR)
            Ds = []
            for or_node in keys(dict)
                if or_node[2] == move
                    push!(Ds, or_node[3])
                end
            end
            if !isempty(Ds)
                push!(ts, t)
                push!(ds, minimum(Ds))
            end
            break
            make_move!(board, move)
        end
    end
end



##########################

##########################

h = []
hh = []
hh_chance = []

hs = [[] for _ in 1:length(subjs)]
hhs = [[] for _ in 1:length(subjs)]
hhs_chance = [[] for _ in 1:length(subjs)]

chance_x = [0, 0, 0, 0, 0, 0, 1.0];

hm = []
hhm = [[] for _ in 1:1000]
hhm_chance = [[] for _ in 1:1000]

plot(layout=(7, 7), grid=false, size=(1400, 1400), yticks=nothing)
for m in ProgressBar(eachindex(subjs))
    #hm = []
    #hhm = []
    #hhm_chance = []
    trees, dicts, all_all_moves, moves = first_pass(all_subj_moves[subjs[m]]);
    x0 = zeros(7)
    x0[1] = params_fit[m, 1]
    x0[end] = params_fit[m, 2]
    for i in eachindex(moves)
        AND_root, AND, OR, idv_AND, idv_OR, parents_moves = trees[i]
        all_moves = all_all_moves[i]
        dict = dicts[i]
        tree_moves = collect(keys(parents_moves))
        mins = []
        for move in tree_moves
            min_d = 10000
            for parent in parents_moves[move]
                if parent[3] < min_d
                    min_d = parent[3]
                end
            end
            push!(mins, min_d)
        end
        ranks = denserank(mins)
        move = moves[i]
        if move in tree_moves
            push!(hs[m], ranks[findfirst(x->x==move, tree_moves)])
            #push!(hm, ranks[findfirst(x->x==move, tree_moves)])
        else
            push!(hs[m], 1000)
            #push!(hm, 1000)
        end
        #dict = propagate_ps(x0, copy(dict), AND_root, AND, OR, idv_AND, idv_OR)
        dict = propagate_ps(x0, AND_root, AND, OR, idv_AND, idv_OR)
        ps = process_dict2(x0, all_moves, dict)
        ps_chance = process_dict2(chance_x, all_moves, dict)
        for j in 1:1
            mmove = wsample(all_moves, ps)
            if mmove in tree_moves
                push!(hhs[m], ranks[findfirst(x->x==mmove, tree_moves)])
                #push!(hhm[j], ranks[findfirst(x->x==mmove, tree_moves)])
            else
                push!(hhs[m], 1000)
                #push!(hhm[j], 1000)
            end
        end
        for j in 1:1
            mmove = wsample(all_moves_chance, ps_chance)
            if mmove in tree_moves
                push!(hhs_chance[m], ranks[findfirst(x->x==mmove, tree_moves)])
                #push!(hhm_chance[j], ranks[findfirst(x->x==mmove, tree_moves)])
                #push!(hh_chance, ranks[findfirst(x->x==mmove, tree_moves)])
            else
                push!(hhs_chance[m], 1000)
                #push!(hhm_chance[j], 1000)
                #push!(hh_chance, 1000)
            end
        end
    end

    # cm1 = countmap(hm)
    # cm2 = countmap(hhm)
    # cm3 = countmap(hhm_chance)
    # x1 = collect(keys(cm1))
    # y1 = collect(values(cm1))
    # x2 = collect(keys(cm2))
    # y2 = collect(values(cm2))
    # x3 = collect(keys(cm3))
    # y3 = collect(values(cm3))

    # x1[x1 .== 1000] .= 7
    # x2[x2 .== 1000] .= 7
    # x3[x3 .== 1000] .= 7

    # bar!(x1, y1 ./ sum(y1), sp=m, xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Data", title=m)
    # bar!(x2, y2 ./ sum(y2), sp=m, xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Model", linecolor=:red, color=nothing)
    # bar!(x3, y3 ./ sum(y3), sp=m, xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Chance", linecolor=:black, color=nothing)
end

cm1 = countmap(hm)
cm2 = countmap(hhm[i])
cm3 = countmap(hhm_chance[i])
x1 = collect(keys(cm1))
y1 = collect(values(cm1)) ./ sum(values(cm1))
x2 = collect(keys(cm2))
y2 = collect(values(cm2))
x3 = collect(keys(cm3))
y3 = collect(values(cm3))

y1c = zeros(42, 7)
y2mc = zeros(1000, 7)
y3mc = zeros(1000, 7)
for i in 1:1000#length(subjs)
    #cm1 = countmap(hs[i])
    cm2 = countmap(hhm[i])
    cm3 = countmap(hhm_chance[i])
    #x1 = collect(keys(cm1))
    x2 = collect(keys(cm2))
    x3 = collect(keys(cm3))
    # for j in x1
    #     if j == 1000
    #         y1c[i, 7] = cm1[j]/sum(values(cm1))
    #     else
    #         y1c[i, j] = cm1[j]/sum(values(cm1))
    #     end
    # end
    for j in x2
        if j == 1000
            y2mc[i, 7] = cm2[j]/sum(values(cm2))
        else
            y2mc[i, j] = cm2[j]/sum(values(cm2))
        end
    end
    for j in x3
        if j == 1000
            y3mc[i, 7] = cm3[j]/sum(values(cm3))
        else
            y3mc[i, j] = cm3[j]/sum(values(cm3))
        end
    end
end

y1 = mean(y1c, dims=1)[1, :]
y2 = mean(y2c, dims=1)[1, :]
y3 = mean(y3c, dims=1)[1, :]
y1s = std(y1c, dims=1)[1, :]
y2s = std(y2c, dims=1)[1, :]
y3s = std(y3c, dims=1)[1, :]

x1[x1 .== 1000] .= 7
x2[x2 .== 1000] .= 7
x3[x3 .== 1000] .= 7

x1 = 1:7
x2 = 1:7
x3 = 1:7

plot(ylim=(0.0, 0.5), grid=false, ylabel="Proportion of all moves", xlabel="Depth of chosen move (rank)", dpi=300, fg_legend = :transparent, size=(400, 500))
#bar!(x1, y1, yerr=y1s ./ sqrt(length(subjs)), xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Data", linecolor=:match, color=:gray, fillstyle = :/, bar_width=0.4, markerstrokecolor=:gray)
bar!(x1, y1, xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Data", linecolor=:match, color=:gray, fillstyle = :/, bar_width=0.4, markerstrokecolor=:gray)
bar!(x3, y3, yerr=y3s, xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Chance", linecolor=:blue, color=nothing, legend=:top, bar_width=0.4, markerstrokecolor=:blue)
bar!(x2, y2, yerr=y2s, xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), label="Model", linecolor=:red, color=nothing, bar_width=0.4, markerstrokecolor=:red)



groupedbar([x1..., x2..., x3...], [y1..., y2..., y3...], yerr= [y1s..., y2s..., y3s...] ./ sqrt(length(subjs)), group=[["Data" for _ in 1:7]..., ["Model" for _ in 1:7]..., ["Chance" for _ in 1:7]...], grid=false, ylabel="Proportion of all moves", xlabel="Depth of chosen move (rank)", dpi=300, fg_legend = :transparent, xticks=(push!(collect(1:6), 7), push!(string.(1:6), "not in tree")), legend=:top)#, cmap=[[:black for _ in 1:7]..., [:cyan for _ in 1:7]...,[:red for _ in 1:7]...])