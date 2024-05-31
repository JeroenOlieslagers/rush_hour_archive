
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
            if OR_node[2] in train_of_thought
                dict[(OR_node[1], (-2, -2))] += (1-γ)*p_or
                dict[(OR_node[1], (-1, -1))] += γ*p_or
                continue
            end
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
                    push!(new_train_of_thought, OR_node[2])
                    # recurse
                    propagate!(x, p_and, new_train_of_thought, dict, AND_next, AND, OR)#, idv_AND, idv_OR)
                end
            end
        end
        return nothing
    end
    OR_root, AND, OR, parents_moves = AND_OR_tree#, idv_AND, idv_OR, parents_AND, parents_OR
    dict = DefaultDict{and_type, Float64}(0.0)
    # keeps track of current train of thought nodes visited
    train_of_thought = Vector{thought_type}()
    propagate!(x, 1.0, train_of_thought, dict, OR_root, AND, OR)#, idv_AND, idv_OR)
    return Dict(dict)
end

s = load_data(prbs[41])
all_moves = possible_moves(s, board_to_arr(s))
a = get_and_or_tree(s; max_iter=100);
draw_ao_tree(a[2], a[3], s)
dict = propagate_ps(0.0, a);
new_dict = apply_gamma(dict, 0.1);
ps = process_dict(all_moves, new_dict, [(Int8(0), Int8(0))])

# TODO BELOW

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

function process_dict(all_moves::moves_type, dict::Dict{and_type, Any}, excl_moves::Vector{a_type})::Vector{Float64}
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

function first_pass(tot_moves)
    # AND-OR tree base node type
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
    states = Vector{BigInt}()
    boards = Vector{Board}()
    neighs = Vector{Vector{BigInt}}()
    first_moves = Vector{Bool}()

    # dicts_prb = DefaultDict{String, Vector{Vector}}([])
    # all_all_moves_prb = DefaultDict{String, Vector{Vector}}([])
    # trees_prb = DefaultDict{String, Vector{Vector}}([])
    # moves_prb = DefaultDict{String, Vector{Vector}}([])
    for prb in collect(keys(tot_moves))
        # init puzzle
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        restart_count = 0
        first_ = false
        for move in tot_moves_prb
            # if restart, reload and push new list for new attempt
            if move == (-1, 0)
                board = load_data(prb)
                restart_count += 1
                # push!(dicts_prb[prb], [])
                # push!(all_all_moves_prb[prb], [])
                # push!(trees_prb[prb], [])
                # push!(moves_prb[prb], [])
                push!(first_moves, true)
                first_ = true
                continue
            end
            arr = get_board_arr(board)
            # stop if solved
            if check_solved(arr)
                #continue
                break
            end
            # get AND/OR tree
            all_moves, AND_OR_tree = get_and_or_tree(board; backtracking=false, idv=false);
            # Propagate without stopping
            dict = propagate_ps(0, AND_OR_tree)
            # save
            push!(trees, AND_OR_tree[1:4])
            #push!(trees_prb[prb][restart_count], AND_OR_tree[1:4])
            push!(all_all_moves, all_moves)
            #push!(all_all_moves_prb[prb][restart_count], all_moves)
            push!(dicts, dict)
            #push!(dicts_prb[prb][restart_count], dict)
            push!(moves, move)
            #push!(moves_prb[prb][restart_count], move)
            push!(states, board_to_int(arr, BigInt))
            push!(boards, arr_to_board(arr))
            neighs_ = []
            for move_ in all_moves
                make_move!(board, move_)
                push!(neighs_, board_to_int(get_board_arr(board), BigInt))
                undo_moves!(board, [move_])
            end
            push!(neighs, neighs_)
            if first_
                first_ = false
            else
                push!(first_moves, false)
            end
            # next move
            make_move!(board, move)
        end
    end
    tree_data = [trees, dicts, all_all_moves, moves]
    #tree_data_prb = [trees_prb, dicts_prb, all_all_moves_prb, moves_prb]
    return tree_data, states, boards, neighs, first_moves#tree_data_prb
end


function get_all_subjects_first_pass(all_subj_moves)
    tree_datas = Dict{String, Any}()
    states = Dict{String, Any}()
    boards = Dict{String, Any}()
    neighs = Dict{String, Any}()
    first_moves = Dict{String, Any}()
    #tree_datas_prb = Dict{String, Any}()
    for subj in ProgressBar(keys(all_subj_moves))
        tree_data, states_, boards_, neighs_, first_moves_ = first_pass(all_subj_moves[subj]);#tree_data_prb
        tree_datas[subj] = tree_data
        states[subj] = states_
        boards[subj] = boards_
        neighs[subj] = neighs_
        first_moves[subj] = first_moves_
        #tree_datas_prb[subj] = tree_data_prb
    end
    return tree_datas, states, boards, neighs, first_moves#, tree_datas_prb
end

