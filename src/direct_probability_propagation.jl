include("basics_for_hpc.jl")
#using StatsBase
#using Plots

function get_move_probabilities(x, board; backtracking=false, max_iter=1000)
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # keeps track of current AO nodes visited
    current_visited = Vector{s_type}()
    # keeps track of probability over moves
    # also keeps track of probability of making random move
    dict = DefaultDict{a_type, Any}(0.0)
    #dict = DefaultDict{a_type, Vector{Vector{Tuple{Tuple{Int, Vector, Vector}, Tuple{Int, Vector, Vector}}}}}([])
    #p_chain = Vector{Tuple{Tuple{Int, Vector, Vector}, Tuple{Int, Vector, Vector}}}()
    # AND node type
    and_type = Tuple{s_type, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    # keeps track of probability of every AND/OR state
    p_AND = DefaultDict{and_type, Any}(0.0)
    p_OR = DefaultDict{or_type, Any}(0.0)
    # keeps track of parents for backtracking
    parents_AND = DefaultDict{and_type, Vector{or_type}}([])
    parents_OR = DefaultDict{or_type, Vector{and_type}}([])
    parents_moves = DefaultDict{a_type, Vector{or_type}}([])
    # initialize process
    arr = get_board_arr(board)
    m_init = 6 - (board.cars[9].x+1)
    ao_root = (9, (m_init,))
    root_move = (9, m_init)
    push!(current_visited, ao_root)
    # fill all moves
    all_moves = Tuple.(get_all_available_moves(board, arr))
    for move in all_moves
        #dict[move] = []
        dict[move] = 0.0
    end
    AND_root = (ao_root, 0)
    OR_root = (ao_root, root_move, 0)
    p_AND[AND_root] += 1.0
    p_OR[OR_root] += 1.0
    push!(parents_AND[AND_root], ((0, (0,)), (0, 0), -1))
    push!(parents_OR[OR_root], AND_root)
    child_nodes = get_blocking_nodes(board, arr, root_move)
    # Expand tree and recurse
    propagate!(x, 1.0, OR_root, child_nodes, current_visited, dict, parents_moves, p_AND, p_OR, parents_AND, parents_OR, board, arr, 0, max_iter; backtracking=backtracking)
    return dict, p_AND, p_OR, parents_moves, parents_AND, parents_OR
end

function propagate!(x, p, prev_OR, child_nodes, current_visited, dict, parents_moves, p_AND, p_OR, parents_AND, parents_OR, board, arr, recursion_depth, max_iter; backtracking=false)
    if recursion_depth > max_iter
        throw(DomainError("max_iter depth reached"))
    end
    γ = exp(-x[1])
    β_AND1 = x[2]
    β_AND2 = x[3]
    β_OR1 = x[4]
    β_OR2 = x[5]
    _, move, d = prev_OR

    # if move is unblocked, have reached end of chain
    if isempty(child_nodes)
        dict[move] += p
        #push!(dict[move], p_chain)
        if prev_OR ∉ parents_moves[move]
            push!(parents_moves[move], prev_OR)
        end
        return nothing
    end
    # assign probability to each child node being selected
    #p_children = p / length(child_nodes)
    a1 = collect(1:length(child_nodes))
    a2 = [length(c[2]) for c in child_nodes]
    p_children = p * softmax([a1'; a2']', [β_AND1, β_AND2])
    # recurse all children
    for (i, node) in enumerate(child_nodes)
        if node in current_visited
            dict[(-2, -2)] += p_children[i]
            #pc = copy(p_chain)
            #push!(pc, ((i, a1, a2), (1, [1], [1])))
            #push!(dict[(-2, -2)], pc)
            continue
        end
        AND = (node, d+1)
        p_AND[AND] += p_children[i]
        if backtracking
            if prev_OR ∉ parents_AND[AND]
                push!(parents_AND[AND], prev_OR)
            end
        end
        # assign probability to each move being selected
        p_move = p_children / length(node[2])
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
        p_move = p_children[i] * softmax([o1'; o2']', [β_OR1, β_OR2])
        for (j, m) in enumerate(ls)
            next_move = (node[1], m)
            OR = (node, next_move, d+1)
            p_OR[OR] += p_move[j]
            if backtracking
                if AND ∉ parents_OR[OR]
                    push!(parents_OR[OR], AND)
                end
            end
            cv = copy(current_visited)
            push!(cv, node)
            p_stop = γ .* p_move[j]
            dict[(-1, -1)] += p_stop
            p_continue = (1-γ) .* p_move[j]
            #pc = copy(p_chain)
            #push!(pc, ((i, a1, a2), (j, o1, o2)))
            propagate!(x, p_continue, OR, childs[j], cv, dict, parents_moves, p_AND, p_OR, parents_AND, parents_OR, board, arr, recursion_depth + 1, max_iter; backtracking=backtracking)
        end
    end
    return nothing
end
@btime dict, _, _, _, _ = get_move_probabilities(x0, load_data(prbs[sp[end-1]]));

function softmax(idv, betas)
    sm = exp.(idv * betas)
    return sm ./ sum(sm)
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

function process_dict(dict, x)
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

function filter_prev_move(move, p_AND, p_OR, parents_moves, parents_AND, parents_OR)
    OR_nodes = parents_moves[move]
    AND_parents = []
    for OR_node in OR_nodes
        push!(AND_parents, parents_OR[OR_node]...)
    end
    OR_parents = []
    for AND_parent in unique(AND_parents)
        push!(OR_parents, parents_AND[AND_parent]...)
    end
    dict = DefaultDict{Tuple{Int, Int}, Float64}(0.0)
    for OR_parent in unique(OR_parents)
        dict[OR_parent[2]] += p_OR[OR_parent]
    end
    sm = sum(values(dict))
    return replace(kv -> kv[1] => kv[2]/sm, dict)
end

function simulate_model(x, tot_moves, optimal_a, IDV)
    nll = 0
    error_l_model = [Int[] for _ in 1:35]
    error_a_model = [Int[] for _ in 1:35]
    for prb in collect(keys(tot_moves))
        board = load_data(prb)
        arr = get_board_arr(board)
        s = board_to_int(arr, BigInt)
        tot_moves_prb = tot_moves[prb]
        opt_a = optimal_a[prb]
        idv = IDV[prb]
        for move in tot_moves_prb
            if move == (-1, 0)
                board = load_data(prb)
                arr = get_board_arr(board)
                s = board_to_int(arr, BigInt)
                continue
            end
            # Stop once puzzle is actually solved
            if idv[s][1] == 0
                break
            end

            dict = get_move_probabilities(x, board);
            all_moves, ps = process_dict(dict, x);

            nll -= log(ps[findfirst(x->x==move, all_moves)])

            ss = zeros(BigInt, length(all_moves))
            for (n, mov) in enumerate(all_moves)
                make_move!(board, mov)
                arr = get_board_arr(board)
                ss[n] = board_to_int(arr, BigInt)
                undo_moves!(board, [mov])
            end
            s_model = wsample(ss, ps)
            if s_model ∉ opt_a[s]
                push!(error_l_model[idv[s][1]], 1)
                push!(error_a_model[idv[s][2]], 1)
            else
                push!(error_l_model[idv[s][1]], 0)
                push!(error_a_model[idv[s][2]], 0)
            end

            make_move!(board, move)
            arr = get_board_arr(board)
            s = board_to_int(arr, BigInt)
        end
    end
    return error_l_model, error_a_model, nll
end

function subject_nll(x, tot_moves)
    nll = 0
    for prb in collect(keys(tot_moves))
        board = load_data(prb)
        tot_moves_prb = tot_moves[prb]
        for move in tot_moves_prb
            if move == (-1, 0)
                board = load_data(prb)
                continue
            end
            arr = get_board_arr(board)
            if check_solved(arr)
                continue
            end
            dict, _, _, _, _ = get_move_probabilities(x, board);
            all_moves, ps = process_dict(dict, x);

            nll -= log(ps[findfirst(x->x==move, all_moves)])

            make_move!(board, move)
        end
    end
    return nll
end

function fit_subject(tot_moves, x0)
    #res = optimize((x) -> subject_nll(x, tot_moves), [0.0, 0.0], [20.0, 1.0], x0, Fminbox(); autodiff=:forward)
    res = optimize((x) -> subject_nll(x, tot_moves), [0.0, -10, -10, -10, -10, 0.0, 0.0], [20.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0], x0, Fminbox(); autodiff=:forward, f_tol=1.0)#Optim.Options(f_tol = 1.0))
    return Optim.minimizer(res), Optim.minimum(res)
end

using PythonCall
pybads = pyimport("pybads")
BADS = pybads.BADS

bads_target = (x) -> subject_nll(x, all_subj_moves[subjs[1]])
lb = [0.0, -10, -10, -10, -10, 0.0, 0.0];
ub = [20.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0];
plb = [2.0, -5, -5, -5, -5, 0.05, 0.05];
pub = [5.0, 5, 5, 5, 5, 0.5, 0.5];
x0 = [2.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.2];
options = Dict("tolfun"=> 1.0, "max_fun_evals"=>1000);
bads = BADS(bads_target, x0, lb, ub, plb, pub, options=options)
@time res = bads.optimize();

x_min = res["x"]
fval = res["fval"]

println("BADS minimum at: x_min = $(x_min.flatten()), fval = $(fval)")
println("total f-count: $(res["func_count"]), time: $(res["total_time"]) s")

function all_subjects_fit(all_subj_moves, x0)
    params = zeros(length(all_subj_moves), length(x0))
    fitness = zeros(length(all_subj_moves))
    subjs = collect(keys(all_subj_moves))
    M = length(all_subj_moves)
    for m in ProgressBar(1:1)#Threads.@threads 
        x, nll = fit_subject(all_subj_moves[subjs[m]], x0)
        params[m, :] = x
        fitness[m] = nll
    end
    return params, fitness
end

function get_all_subj_moves(data)
    all_subj_moves = Dict{String, Dict{String, Vector{Tuple{Int, Int}}}}()
    all_subj_times = Dict{String, Dict{String, Vector{Int}}}()
    for subj in collect(keys(data))
        _, tot_moves, _, _, tot_times = analyse_subject(data[subj]);
        all_subj_moves[subj] = tot_moves
        all_subj_times[subj] = tot_times
    end
    return all_subj_moves, all_subj_times
end

board = load_data(prbs[sp[4]])
arr = get_board_arr(board)
x0 = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]

dict, p_AND, p_OR, parents_moves, parents_AND, parents_OR = get_move_probabilities(x0, board; backtracking=true);
dd = filter_prev_move((3, -1), p_AND, p_OR, parents_moves, parents_AND, parents_OR)

pilot_moves = [
    (2, -1),
    (7, 1),
    (1, -4),
    (5, -1),
    (3, 2)
]
prev_moves = []
for move in pilot_moves
    dict, p_AND, p_OR, parents_moves, parents_AND, parents_OR = get_move_probabilities(x0, board; backtracking=true);
    _, ps = process_dict(dict, x0)
    make_move!(board, move)
    pushfirst!(prev_moves, (p_AND, p_OR, parents_moves, parents_AND, parents_OR))
end

undo_moves!(board, pilot_moves)

#data = load("data/processed_data/filtered_data.jld2")["data"]
#subjs = collect(keys(data))

all_subj_moves, all_subj_times = get_all_subj_moves(data);
#x0 = [2.0, 0.1];
x0 = [2.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.2];
# params_fit = [1.63144  0.0272332;
# 4.1322   0.232287;
# 5.54945  0.376679;
# 1.78816  0.0420192;
# 20.0      0.28175;
# 3.43365  0.286021;
# 2.84836  0.212509;
# 4.61594  0.292724;
# 5.84368  0.291619;
# 2.28144  0.245883;
# 1.70216  0.180736;
# 2.0246   0.302909;
# 1.62686  0.287269;
# 2.17845  0.207132;
# 2.06946  1.09775e-9;
# 2.64261  0.0663281;
# 1.73288  0.0925714;
# 2.94924  0.226167;
# 2.38534  0.192865;
# 2.08849  0.0599586;
# 2.72966  0.303894;
# 2.552    0.163531;
# 1.90414  0.316244;
# 1.84811  1.07496e-10;
# 2.64932  0.0747666;
# 4.28873  0.450776;
# 3.56926  0.0790763;
# 2.86207  0.27396;
# 2.07517  0.152435;
# 2.17915  0.159951;
# 2.72093  0.103394;
# 2.31457  0.30568;
# 1.86464  0.0822624;
# 2.75314  0.0524247;
# 3.63077  0.305798;
# 2.45339  0.0499642;
# 3.03726  0.195869;
# 2.22496  0.181216;
# 2.37691  0.350596;
# 2.16347  0.260788;
# 1.95579  0.110893;
# 1.66287  0.00122012]

# f = zeros(42)
# error_l_model = Dict{String, Array{Array{Int, 1}, 1}}()
# error_a_model = Dict{String, Array{Array{Int, 1}, 1}}()
# Threads.@threads for n in ProgressBar(1:42)
#     #f[n] = subject_nll(params[n, :], all_subj_moves[subjs[n]])
#     error_l_model[subjs[n]], error_a_model[subjs[n]], f[n] = simulate_model(params[n, :], all_subj_moves[subjs[n]], optimal_a, IDV)
# end

# error_l_model_qb = quantile_binning(error_l_model);
# error_a_model_qb = quantile_binning(error_a_model);

# mu_hat_bar_l_model, error_bar_l_model = get_errorbar(error_l_model_qb);
# mu_hat_bar_a_model, error_bar_a_model = get_errorbar(error_a_model_qb);

# plot(layout=grid(2, 1, heights=[0.55, 0.45]), size=(900, 900), legend=:outertop, ylim=(0, 1), grid=false, background_color_legend = nothing, foreground_color_legend = nothing, dpi=300, legend_columns=1, 
# legendfont=font(18), 
# xtickfont=font(16), 
# ytickfont=font(16), 
# guidefont=font(32))
# plot!(av_bin_l, mu_hat_bar_l_model, ribbon=2*error_bar_l_model, sp=1, c=palette(:default)[6], lw=0.0, label=nothing)
# plot!(av_bin_l, mu_hat_bar_l_model, ribbon=2*error_bar_l_model, sp=1, c=palette(:default)[6], lw=0.0, label=nothing)
# plot!([], [], sp=1, c=palette(:default)[6], label="AND-OR tree model", lw=10)
# plot!(av_bin_a, mu_hat_bar_a_model, ribbon=2*error_bar_a_model, sp=2, c=palette(:default)[6], lw=0.0, label=nothing)
# plot!(av_bin_a, mu_hat_bar_a_model, ribbon=2*error_bar_a_model, sp=2, c=palette(:default)[6], lw=0.0, label=nothing)
# plot!([], [], sp=2, c=palette(:default)[6], label=nothing, lw=10)

# plot!(av_bin_l, mu_hat_bar_l_model1, alpha=1.0, ribbon=2*error_bar_l_model1, sp=1, c=palette(:default)[2], lw=0.0, label=nothing)
# plot!(av_bin_l, mu_hat_bar_l_model1, alpha=1.0, ribbon=2*error_bar_l_model1, sp=1, c=palette(:default)[2], lw=0.0, label=nothing)
# plot!([], [], sp=1, c=palette(:default)[2], label="Eureka model", lw=10)
# plot!(av_bin_a, mu_hat_bar_a_model1, ribbon=2*error_bar_a_model1, sp=2, c=palette(:default)[2], lw=0.0, label=nothing)
# plot!(av_bin_a, mu_hat_bar_a_model1, ribbon=2*error_bar_a_model1, sp=2, c=palette(:default)[2], lw=0.0, label=nothing)
# plot!([], [], sp=2, c=palette(:default)[2], label=nothing, lw=10)

# plot!([], [], yerr=[], sp=1, c=:black, label="Data")
# plot!(av_bin_l, mu_hat_bar_l_data, bottom_margin=-4Plots.mm, yerr=2*error_bar_l_data, sp=1, ms=10, l=nothing, c=:black, markershape=:none, label=nothing, xlabel=latexstring("d_\\textrm{goal}"), ylabel=latexstring("p(\\textrm{error})"), xticks=round.(av_bin_l, digits=1))
# plot!(av_bin_l, mu_hat_bar_l_data, yerr=2*error_bar_l_data, sp=1, c=:transparent, msw=1, label=nothing)
# plot!(av_bin_a, mu_hat_bar_a_data, yerr=2*error_bar_a_data, sp=2, ms=10, l=nothing, c=:black, label=nothing, xlabel=latexstring("n_A"), ylabel=latexstring("p(\\textrm{error})"), xticks=round.(av_bin_a, digits=1))
# plot!(av_bin_a, mu_hat_bar_a_data, yerr=2*error_bar_a_data, sp=2, c=:transparent, msw=1, label=nothing)

# plot!(av_bin_l, mu_hat_bar_l_model_f, ribbon=2*error_bar_l_model_f, sp=1, c=palette(:default)[4], lw=0.0, label=nothing)
# plot!(av_bin_l, mu_hat_bar_l_model_f, ribbon=2*error_bar_l_model_f, sp=1, c=palette(:default)[4], lw=0.0, label=nothing)
# plot!([], [], sp=1, c=palette(:default)[4], label="AND-OR tree model "*latexstring("\\gamma=e^{-10}, \\lambda=0"), lw=10)
# plot!(av_bin_a, mu_hat_bar_a_model_f, ribbon=2*error_bar_a_model_f, sp=2, c=palette(:default)[4], lw=0.0, label=nothing)
# plot!(av_bin_a, mu_hat_bar_a_model_f, ribbon=2*error_bar_a_model_f, sp=2, c=palette(:default)[4], lw=0.0, label=nothing)
# plot!([], [], sp=2, c=palette(:default)[4], label=nothing, lw=10)


params, fitness = all_subjects_fit(all_subj_moves, x0);
display(params)
println(sum(fitness))
@save "params.jld2" params

#@load "test.jld2" params
# nll = subject_fit(x0, tot_moves)

# prb = prbs[sp[4]]
# board = load_data(prb);
# arr = get_board_arr(board)
# dict = get_move_probabilities(x0, board);

# moves, ps = process_dict(dict);

#optimal_a = load("data/processed_data/optimal_a.jld2");
#IDV = load("data/processed_data/IDV.jld2");






# function process_p_chain_dict(x, dict)
#     γ = exp(-x[1])
#     β_AND1 = x[2]
#     β_AND2 = x[3]
#     β_OR1 = x[4]
#     β_OR2 = x[5]
#     p_dict = DefaultDict{Tuple{Int, Int}, Any}(0.0)
#     for move in keys(dict)
#         for chain in dict[move]
#             p = 1
#             for node in chain
#                 i = node[1][1]
#                 j = node[2][1]
#                 p_and = exp(β_AND1*node[1][2][i] + β_AND2*node[1][3][i])/sum(exp.(β_AND1*node[1][2] + β_AND2*node[1][3]))
#                 p_or = exp(β_OR1*node[2][2][j] + β_OR2*node[2][3][j])/sum(exp.(β_OR1*node[2][2] + β_OR2*node[2][3]))
#                 p *= p_and*p_or
#                 p *= (1-γ)
#             end
#             p_dict[move] += move == (-2, -2) ? p/(1-γ) : p
#         end
#     end
#     p_dict[(-1, -1)] = 1 - sum(values(p_dict))
#     return p_dict
# end

