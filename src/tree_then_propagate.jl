include("basics_for_hpc.jl")
using StatsBase
#using Plots
using Distributions

function get_tree_idv(board; backtracking=false, max_iter=1000)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # keeps track of current AO nodes visited
    current_visited = Vector{s_type}()
    # AND node type
    and_type = Tuple{s_type, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    # keeps track of parents for backtracking
    parents_AND = DefaultDict{and_type, Vector{or_type}}([])
    parents_OR = DefaultDict{or_type, Vector{and_type}}([])
    parents_moves = DefaultDict{a_type, Vector{or_type}}([])
    # forward pass
    AND = DefaultDict{and_type, Vector{or_type}}([])
    OR = DefaultDict{or_type, Vector{and_type}}([])
    idv_AND = Dict{and_type, Matrix}()
    idv_OR = Dict{or_type, Matrix}()
    # initialize process
    arr = get_board_arr(board)
    m_init = 6 - (board.cars[9].x+1)
    ao_root = (9, (m_init,))
    root_move = (9, m_init)
    all_moves = Tuple.(get_all_available_moves(board, arr))
    push!(current_visited, ao_root)
    AND_root = (ao_root, 1)
    OR_root = (ao_root, root_move, 1)
    push!(parents_AND[AND_root], ((0, (0,)), (0, 0), 0))
    push!(parents_OR[OR_root], AND_root)
    child_nodes = get_blocking_nodes(board, arr, root_move)
    push!(AND[AND_root], OR_root)
    idv_AND[AND_root] = [[0]'; [0]']'
    # Expand tree and recurse
    forward!(OR_root, child_nodes, current_visited, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR, board, arr, 0, max_iter; backtracking=backtracking)
    return all_moves, AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR
end

function forward!(prev_OR, child_nodes, current_visited, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR, board, arr, recursion_depth, max_iter; backtracking=backtracking)
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
    # assign probability to each child node being selected
    #p_children = p / length(child_nodes)
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
        if node in current_visited
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
        # assign probability to each move being selected
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
            cv = copy(current_visited)
            push!(cv, node)
            forward!(or_node, childs[j], cv, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR, board, arr, recursion_depth + 1, max_iter; backtracking=backtracking)
        end
    end
    return nothing
end

function propagate_ps(x, AND_root, AND, OR, idv_AND, idv_OR)
    #x[1] = exp(-x[1])
    function propagate!(x, p, current_visited, dict, AND_current, AND, OR, idv_AND, idv_OR)
        γ = x[1]
        γ = exp(-x[1])
        #γ = 0.0
        β_AND1 = x[3]
        β_AND2 = x[4]
        β_OR1 = x[5]
        β_OR2 = x[6]
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
            if OR_node[1] in current_visited
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
                    cv = copy(current_visited)
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
    current_visited = Set{s_type}()
    propagate!(x, 1.0, current_visited, dict, AND_root, AND, OR, idv_AND, idv_OR)
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
    for prb in collect(keys(tot_moves))
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        for move in tot_moves_prb
            if move == (-1, 0)
                board = load_data(prb)
                continue
            end
            all_moves, AND_root, AND, OR, idv_AND, idv_OR, parents_moves, _, _ = get_tree_idv(board);
            push!(trees, (AND_root, AND, OR, idv_AND, idv_OR, parents_moves))
            dict = propagate_ps(zeros(8), AND_root, AND, OR, idv_AND, idv_OR)
            push!(dicts, dict)
            push!(all_all_moves, all_moves)
            push!(moves, move)
            make_move!(board, move)
        end
    end
    return trees, dicts, all_all_moves, moves
end

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

function process_dict2(x, all_moves, dict)
    moves = []
    ps = []
    move_dict = Dict{Tuple{Int, Int}, Any}()
    for move in all_moves
        move_dict[move] = 0.0
    end
    move_dict[(-1, -1)] = 0.0
    move_dict[(-2, -2)] = 0.0
    for or_node in keys(dict)
        move = or_node[2]
        p = dict[or_node]
        #d = or_node[3]
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
    for move in keys(move_dict)
        if move[1] > 0
            push!(moves, move)
            push!(ps, move_dict[move])
        end
    end
    # repeating because of cycle
    sp = sum(ps) + move_dict[(-1, -1)]
    if sp > 0
        p_cycle = move_dict[(-2, -2)]
        for i in eachindex(ps)
            ps[i] += p_cycle*(ps[i]/sp)
        end
        move_dict[(-1, -1)] += p_cycle*(move_dict[(-1, -1)]/sp) 
    end
    # random move in case of stopping early
    ps .+= move_dict[(-1, -1)]/length(ps)

    μ = x[end-1]
    λ = x[end]
    idx = Int[]
    for n in eachindex(moves)
        if moves[n][1] == 9
            push!(idx, n)
        end
    end
    ps[idx] = μ/length(idx) .+ (1-μ)*ps[idx]
    ps = λ/length(ps) .+ (1-λ)*ps
    return moves, ps
end

function subject_nll(x, dicts, all_all_moves, moves)
    nll = 0
    for i in eachindex(moves)
        move = moves[i]
        dict = dicts[i]
        all_moves = all_all_moves[i]
        all_moves, ps = process_dict2(x, all_moves, dict)
        p = ps[findfirst(x->x==move, all_moves)]
        nll -= log(p)
    end
    return nll
end

function fit_subject(tree, x0)
    dicts, all_all_moves, moves = tree
    #dicts, all_all_moves, moves = first_pass(tot_moves);
    res = optimize((x) -> subject_nll(x, dicts, all_all_moves, moves), [0.0, 0.0], [50.0, 50.0], x0, Fminbox(), Optim.Options(f_tol = 1.0); autodiff=:forward)
    return Optim.minimizer(res), Optim.minimum(res)
end

function all_subjects_fit(trees, x0)
    params = zeros(length(trees), length(x0))
    fitness = zeros(length(trees))
    subjs = collect(keys(trees))
    M = length(trees)
    Threads.@threads for m in ProgressBar(1:M)#
        x, nll = fit_subject(trees[subjs[m]], x0)
        params[m, :] = x
        fitness[m] = nll
    end
    return params, fitness
end

function get_all_subjects_first_pass(all_subj_moves)
    trees = Dict{String, Any}()
    for subj in ProgressBar(keys(all_subj_moves))
        trees[subj] = first_pass(all_subj_moves[subj]);
    end
    return trees
end


all_subj_moves, all_subj_times = get_all_subj_moves(data);
trees = get_all_subjects_first_pass(all_subj_moves);
dicts, all_all_moves, moves = first_pass(all_subj_moves[subjs[1]]);
x0 = [5.0, 2.5];
@time subject_nll(x0, dicts, all_all_moves, moves)
fit_subject(all_subj_moves[subjs[1]], x0)
params_weibull, fitness_weibull = all_subjects_fit(trees, x0)

#all_moves, AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR = get_tree_idv(load_data(prbs[sp[end-1]]));
all_moves, AND_root, AND, OR, idv_AND, idv_OR, parents_moves, parents_AND, parents_OR = get_tree_idv(board);
dict = propagate_ps(zeros(7), AND_root, AND, OR, idv_AND, idv_OR)
m, p = process_dict2(x0, all_moves, dict)


res = optimize((x) -> f(x), [0.0, 0.0], [10.0, 1.0], [2.0, 0.5], Fminbox(); autodiff=:forward)

function f(x)
    #tn = Binomial(round(x[1]), x[2])
    return cdf(truncated(Normal(x[1], x[2]), lower=0), 2)
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
            all_moves, AND_root, AND, OR, idv_AND, idv_OR, parents_moves, _, _ = get_tree_idv(board);
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
        all_moves, ps = process_dict2(x0, all_moves, dict)
        all_moves_chance, ps_chance = process_dict2(chance_x, all_moves, dict)
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