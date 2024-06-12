
function propagate_ps(x::Float64, AND_OR_tree)::Dict{and_type, Float64}
    function propagate!(x::Float64, p::Float64, train_of_thought, dict, AND_current::and_type, AND, OR)::Nothing#, idv_AND, idv_OR)
        γ = x#0.0
        # β_AND1 = x[3]
        # β_AND2 = x[4]
        # β_OR1 = x[5]
        # β_OR2 = x[6]
        # number of children of OR node
        N_or = length(AND[AND_current])
        # Rule 2: OR HEURISTICS
        #mat_and = idv_AND[AND_current]
        #p_ands = size(mat_and)[1] == 1 ? p : p*softmax(mat_and, [β_AND1, β_AND2])
        p_ors = p * ones(N_or) / N_or
        # propagate to AND nodes
        for (n, OR_node) in enumerate(AND[AND_current])
            p_or = p_ors[n]
            # CYCLE PROBABILITY
            if OR_node[2] in train_of_thought || OR_node ∉ keys(OR)
                dict[(OR_node[1], (-2, -2))] += (1-γ)*p_or
                dict[(OR_node[1], (-1, -1))] += γ*p_or
                continue
            end
            push!(train_of_thought, OR_node[2])
            # Rule 1a: don't stop
            pp = (1-γ)*p_or
            # Rule 1b: stop
            dict[(OR_node[1], (-1, -1))] += γ*p_or
            N_and = length(OR[OR_node])
            # Rule 3: AND HEURISTICS
            #mat_or = idv_OR[OR_node]
            #p_ors = size(mat_or)[1] == 1 ? p_and : p_and*softmax(mat_or, [β_OR1, β_OR2])
            p_ands = pp * ones(N_and) / N_and
            # propagate to AND nodes
            for (m, AND_next) in enumerate(OR[OR_node])
                p_and = p_ands[m]
                # leaf node
                if AND[AND_next][1][2][1] == 0
                    dict[AND_next] += p_and
                # elseif AND_next[1] == (-2, (-2,))
                #     dict[((-2, (-2,)), (-2, -2), OR_node[3])] += p_or
                else
                    # chain of thought
                    new_train_of_thought = copy(train_of_thought)
                    # push!(new_train_of_thought, OR_node[2])
                    # recurse
                    propagate!(x, p_and, new_train_of_thought, dict, AND_next, AND, OR)#, idv_AND, idv_OR)
                end
            end
        end
        return nothing
    end
    AND_root, AND, OR, parents_moves = AND_OR_tree#, idv_AND, idv_OR, parents_AND, parents_OR
    dict = DefaultDict{and_type, Float64}(0.0)
    # keeps track of current train of thought nodes visited
    train_of_thought = Vector{thought_type}()
    push!(train_of_thought, MVector{5, Int8}([AND_root[2][1], AND_root[2][2], 0, 0, 0]))
    propagate!(x, 1.0, train_of_thought, dict, AND_root, AND, OR)#, idv_AND, idv_OR)
    return Dict(dict)
end

# dict = propagate_ps(0.0, tree)


# s = load_data(prbs[41])
# all_moves = possible_moves(s, board_to_arr(s))
# a = get_and_or_tree(s; max_iter=100);
# draw_ao_tree(a[2], a[3], s)
# dict = propagate_ps(0.0, a);
# new_dict = apply_gamma(dict, 0.1);
# ps = process_dict(all_moves, new_dict, [(Int8(0), Int8(0))])

function apply_gamma(dict::Dict{and_type, Float64}, γ::Float64)::Dict{and_type, Any}
    # updated dict
    new_dict = Dict{and_type, Any}()
    for (k, v) in dict
        if v > 0
            new_dict[k] = v*(1-γ)^k[1]
        end
    end
    # probability of stopping
    new_dict[(0, (-1, -1))] = 1 - sum(values(new_dict))
    return new_dict
end

function process_dict(all_moves, dict, excl_moves::Vector{a_type})::Vector{Float64}
    # probability distribution over moves
    ps = Vector{Float64}()
    move_dict = Dict{a_type, Any}()
    # add all possible moves
    for move in all_moves
        if move == (0, 0)
            break
        end
        move_dict[move] = 0.0
    end
    # stopping
    move_dict[(-1, -1)] = 0.0
    # cycle
    move_dict[(-2, -2)] = 0.0
    # exclude certain moves (e.g. moving same car)
    p_excl = 0
    for and_node in keys(dict)
        if and_node[2] ∉ excl_moves
            move_dict[and_node[2]] += dict[and_node]
        else
            p_excl += dict[and_node]
        end
    end
    # If exclusion move is reached, treat as cycle
    move_dict[(-2, -2)] += p_excl
    # vectorize move probabilities
    for move in all_moves
        if move == (0, 0)
            break
        end
        push!(ps, move_dict[move])
    end
    # probabilities without cycle
    Z = sum(ps) + move_dict[(-1, -1)]
    # repeating because of cycle
    if Z > 0
        p_cycle = move_dict[(-2, -2)]
        # spread proportionally over all other options
        ps += ps * p_cycle/Z
        move_dict[(-1, -1)] += move_dict[(-1, -1)] * p_cycle/Z
    else # if only cycles are possible (very rare condition) spread over all
        ps .+= move_dict[(-2, -2)]/length(ps)
    end
    # spread stopping probability uniformly
    ps .+= move_dict[(-1, -1)]/length(ps)
    return ps
end

function calculate_features(s::s_type, blocked_cars::blocked_cars_type)::Matrix{Int}
    s_free, s_fixed = s
    arr = board_to_arr(s)
    m = 6 - (s_free[1]+s_fixed[1].len-1)
    move_blocked_by!(blocked_cars, (Int8(1), Int8(m)), s, arr)
    return [sum(blocked_cars .> 0) m]
end



function first_pass(df)
    stuff = Dict{String, Any}()
    trees = []
    dicts = Dict[]
    all_moves = Vector{a_type}[]
    neighs = Vector{Int32}[]
    features = Matrix{Float64}[]
    blocked_cars = zeros(blocked_cars_type)
    for row in ProgressBar(eachrow(df))
        if row.event != "move"
            push!(trees, [])
            push!(dicts, Dict())
            push!(all_moves, a_type[])
            push!(neighs, s_free_type[])
            push!(features, zeros(0, 0))
            continue
        end
        s = (row.s_free, row.s_fixed)
        arr = board_to_arr(s)
        tree = get_and_or_tree(s)
        dict = propagate_ps(0.0, tree)
        moves_ = possible_moves(s, arr)
        neigh = Int32[]
        moves = a_type[]
        fs = []
        for move in moves_
            if move[1] == 0
                break
            end
            sp = make_move(row.s_free, move)
            push!(neigh, board_to_int32(sp))
            push!(moves, move)
            push!(fs, calculate_features((sp, row.s_fixed), blocked_cars))
        end

        push!(trees, tree)
        push!(dicts, dict)
        push!(all_moves, moves)
        push!(neighs, neigh)
        push!(features, Float64.(reduce(vcat, fs)))
    end
    stuff["trees"] = trees
    stuff["dicts"] = dicts
    stuff["all_moves"] = all_moves
    stuff["neighs"] = neighs
    stuff["features"] = features
    return stuff
end



