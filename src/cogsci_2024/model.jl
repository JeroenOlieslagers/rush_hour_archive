using ProgressBars
using Optim
include("and_or_trees.jl")
include("pre_process.jl")

function propagate_ps(x, AND_OR_tree)
    AND_root, AND, OR, parents_moves = AND_OR_tree#, idv_AND, idv_OR, parents_AND, parents_OR
    function propagate!(x, p, visited, dict, AND_current, AND, OR)#, idv_AND, idv_OR)
        γ = x#0.0
        # β_AND1 = x[3]
        # β_AND2 = x[4]
        # β_OR1 = x[5]
        # β_OR2 = x[6]
        # CYCLE PROBABILITY
        if !haskey(AND, AND_current)
            dict[((-2, (-2,)), (-2, -2), AND_current[2])] += p
            return nothing
        end
        # number of children of AND node
        N_and = length(AND[AND_current])
        # Rule 2: AND HEURISTICS
        #mat_and = idv_AND[AND_current]
        #p_ands = size(mat_and)[1] == 1 ? p : p*softmax(mat_and, [β_AND1, β_AND2])
        p_ands = p * ones(N_and) / N_and
        # propagate to OR nodes
        for (n, OR_node) in enumerate(AND[AND_current])
            p_and = p_ands[n]
            # CYCLE PROBABILITY
            if OR_node[1] in visited || !haskey(OR, OR_node)
                dict[((-2, (-2,)), (-2, -2), OR_node[3])] += p_and
                continue
            end
            N_or = length(OR[OR_node])
            # Rule 3: OR HEURISTICS
            #mat_or = idv_OR[OR_node]
            #p_ors = size(mat_or)[1] == 1 ? p_and : p_and*softmax(mat_or, [β_OR1, β_OR2])
            p_ors = p_and * ones(N_or) / N_or
            # propagate to AND nodes
            for (m, AND_next) in enumerate(OR[OR_node])
                p_or = p_ors[m]
                # leaf node
                if AND_next[1] == (0, (0,))
                    dict[OR_node] += p_or
                # elseif AND_next[1] == (-2, (-2,))
                #     dict[((-2, (-2,)), (-2, -2), OR_node[3])] += p_or
                else
                    # Rule 1a: don't stop
                    pp = (1-γ)*p_or
                    # Rule 1b: stop
                    dict[((-1, (-1,)), (-1, -1), OR_node[3])] += γ*p_or
                    # chain of thought
                    cv = copy(visited)
                    push!(cv, OR_node[1])
                    # recurse
                    propagate!(x, pp, cv, dict, AND_next, AND, OR)#, idv_AND, idv_OR)
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
    propagate!(x, 1.0, visited, dict, AND_root, AND, OR)#, idv_AND, idv_OR)
    return dict
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

function apply_gamma(dict, γ)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    # updated dict
    new_dict = Dict{or_type, Any}()
    for (k, v) in dict
        if v > 0
            new_dict[k] = v*(1-γ)^(k[3]-0)
        end
    end
    # probability of stopping
    new_dict[((-1, (-1,)), (-1, -1), 0)] = 1 - sum(values(new_dict))
    return new_dict
end

function process_dict(all_moves, dict, excl_moves, μ_same, μ_block, board)
    # probability distribution over moves
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
    # exclude certain moves (e.g. undos or blocks red car)
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
    # If exclusion move is reached, treat as cycle
    move_dict[(-2, -2)] += p_excl
    # vectorize move probabilities
    for move in all_moves
        push!(ps, move_dict[move])
    end
    # probabilities without cycle
    sp = sum(ps) + move_dict[(-1, -1)]
    # repeating because of cycle
    if sp > 0
        p_cycle = move_dict[(-2, -2)]
        # for i in eachindex(ps)
        #     ps[i] += p_cycle*(ps[i]/sp)
        # end
        ps += p_cycle * ps/sp
        move_dict[(-1, -1)] += p_cycle * move_dict[(-1, -1)]/sp
    else
        throw(DomainError("how is sum for cycle probability 0?"))
    end
    # random move in case of stopping early
    N = sum(ps .== 0)
    ps .+= move_dict[(-1, -1)]/length(ps)
    return ps, move_dict[(-1, -1)], N
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
        # UNDOS
        #excl_moves = [i > 1 ? (moves[i-1][1], -moves[i-1][2]) : (0, 0)]
        # SAME CAR
        excl_moves = i > 1 ? [(moves[i-1][1], j) for j in -4:4] : [(0, 0)]
        # NONE
        #excl_moves = [(0, 0)]
        dict = dicts[i]
        all_moves = all_all_moves[i]
        new_dict = apply_gamma(dict, γ)
        ps, _, _ = process_dict(all_moves, new_dict, excl_moves, 0.0, 1.0, board)#μ_same, μ_block

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
        # end
        # if !isempty(idx)# && !unblockable
        #     #ps[idx] = sum(ps[idx])*μ_red/length(idx) .+ (1-μ_red)*ps[idx]
        #     ps = μ_red*[ci in idx for (ci, _) in enumerate(ps)]/length(idx) .+ (1-μ_red)*ps
        # end

        # LAPSE RATE (λ)
        #ps = λ/length(ps) .+ (1-λ)*ps

        # AND OR PARENTS SECTION (k)
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

        if round(100000*sum(ps))/100000 != 1
            println("========$(i)========")
            println(sum(ps))
            println(ps)
            throw(DomainError("Not a valid probability distribution"))
        end
        # Likelihood of subject's move under model
        p = ps[findfirst(x->x==move, all_moves)]
        if p == 0
            println("========$(i)========")
            println("Zero probability move")
            continue
        end
        nll -= log(p)
    end
    return nll
end

function subject_nll_eureka(x, states, neighs, all_all_moves, moves, d_goals)
    d, λ = x
    nll = 0
    for i in eachindex(states)
        move = moves[i]
        all_moves = all_all_moves[i]
        neigh = neighs[i]
        N = length(neigh)
        s = states[i]
        d_goal = d_goals[s]

        ps = zeros(N)
        if d_goal <= d
            opt_idx = [d_goals[s_p] < d_goal for s_p in neigh]
            ps[opt_idx] .= 1/sum(opt_idx)
        else
            ps = ones(N) ./ N
        end
        ps = λ/N .+ (1-λ) * ps

        if round(100000*sum(ps))/100000 != 1
            println("========$(i)========")
            println(sum(ps).value)
            println([pp.value for pp in ps])
            throw(DomainError("Not a valid probability distribution"))
        end
        # Likelihood of subject's move under model
        p = ps[findfirst(x->x==move, all_moves)]
        if p == 0
            println("========$(i)========")
            println("Zero probability move")
            continue
        end
        nll -= log(p)
    end
    return nll
end

function fit_eureka_model(tree_datas, states, neighs, d_goals)
    M = length(tree_datas)
    params = zeros(M, 2)
    fitness = zeros(M)
    Threads.@threads for m in ProgressBar(1:M)#
        tree_data = tree_datas[subjs[m]]
        states_subj = states[subjs[m]]
        neighs_subj = neighs[subjs[m]]
        tree, dicts, all_all_moves, moves = tree_data;
        res = optimize((x) -> subject_nll_eureka(x, states_subj, neighs_subj, all_all_moves, moves, d_goals), [0.0, 0.0], [25.0, 1.0], [10.0, 0.1], Fminbox(), Optim.Options(f_tol = 0.000001); autodiff=:forward)
        params[m, :] = Optim.minimizer(res)
        fitness[m] = Optim.minimum(res)
    end
    return params, fitness
end

function fit_model(tree_datas, boards)
    M = length(tree_datas)
    params = zeros(M)
    fitness = zeros(M)
    for m in ProgressBar(1:M)#Threads.@threads 
        tree_data = tree_datas[subjs[m]]
        tree, dicts, all_all_moves, moves = tree_data;
        board = boards[subjs[m]]
        res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves, board, nothing, nothing), 0.000001, 0.999999)
        #res = optimize((x) -> subject_nll_gamma(x, dicts, all_all_moves, moves), [0.0, 0.0], [1.0, 1.0], [0.12, 0.1], Fminbox(), Optim.Options(f_tol = 0.1); autodiff=:forward)
        params[m] = Optim.minimizer(res)
        fitness[m] = Optim.minimum(res)
    end
    return params, fitness
end

tree_datass, states, boards, neighs, first_moves = get_all_subjects_first_pass(all_subj_moves);

params_gamma_alt, fitness_gamma_alt = fit_model(tree_datass, boards);
# params_eureka, fitnesss_eureka = fit_eureka_model(tree_datas, states, neighs, d_goals);

# #save("tree_datas_red_actions.jld2", tree_datas)
tree_datas = load("tree_datas.jld2")
states = load("states.jld2")
boards = load("boards.jld2")
neighs = load("neighs.jld2")
first_moves = load("first_moves.jld2")
opt_act = load("optimal_act.jld2")
opt_act = merge(values(opt_act)...);

# # @save "params_model1.jld2" params
# # @save "fitness_model1.jld2" fitness

d_goals = load("d_goals.jld2")["d_goals"];
params_gamma = load("params_model1.jld2")["params"]
fitness_gamma = load("fitness_model1.jld2")["fitness"]

# trees = tree_datas[subjs[1]][1];
# moves = tree_datas[subjs[1]][4];
# boardss = boards[subjs[1]];
# ss = states[subjs[1]];
# ls = []
# ands = []
# ors = []
# for tree in trees
#     AND_root, AND, OR, parents_moves = tree
#     push!(ls, length(AND)+length(OR))
#     push!(ands, AND)
#     push!(ors, OR)
# end

# sp = sort(unique(ls));
# idx = findfirst(x->x==sp[40], ls);
# idx += 49#48

# begin
# idx += 1
# AND = ands[idx];
# OR = ors[idx];
# board = boardss[idx];
# s = ss[idx];
# opt = opt_act[s];
# parent_ORs = trees[idx][4];
# move = moves[idx];
# end

# _, tree = get_and_or_tree(board);
# _, AND, OR, _, _, _, _, _ = tree;

# all_moves = Tuple.(get_all_available_moves(board, get_board_arr(board)))
# dict = propagate_ps(0, tree)
# new_dict = apply_gamma(dict, 0.01)
# ps, _, _ = process_dict(all_moves, new_dict, [(0, 0)], 0.0, 1.0, board);


# draw_ao_tree((AND, OR), board; highlight_ORs=reduce(vcat, [parent_ORs[o] for o in opt]))
# #draw_ao_tree((AND, OR), board; highlight_ORs=parent_ORs[move])
# draw_board(get_board_arr(board))


