include("engine.jl")

function sample_action(x::Vector{Float64}, Q::Vector{Float64})::Int
    # λ, logb = x
    # β = exp(logb)
    # p = lapsed_softmin(λ, β, Q)
    λ, d_fl = x
    d = Int(round(d_fl))
    p = lapsed_depth_limited_random(λ, d, Q)
    return wsample(p)
end

function get_Q(board, A, full_heur_dict; heur=4)
    Q = zeros(length(A))
    ns = zeros(BigInt, length(A))
    for (n, a) in enumerate(A)
        make_move!(board, a)
        arr_a = get_board_arr(board)
        s = board_to_int(arr_a, BigInt)
        # Ignore if move outside state space
        if string(s) ∉ keys(full_heur_dict)
            println("oops")
            undo_moves!(board, [a])
            return []
        end
        # Get specific heuristic value as Q value
        Q[n] = full_heur_dict[string(s)][heur]
        ns[n] = s
        # neighs[n] = s
        undo_moves!(board, [a])
    end
    return Q, ns
end

function generate_puzzle_trajectory(x, prb, full_heur_dict; heur=4, max_iter=1000)
    # Initialise outputs
    qs = Float64[]
    Qs = []
    visited = BigInt[]
    neighbours = []
    # Load puzzle
    board = load_data(prb)
    arr = get_board_arr(board)
    s_init = board_to_int(arr, BigInt)
    push!(visited, s_init)
    for i in 1:max_iter
        # Get all neighbours
        A = get_all_available_moves(board, arr)
        Q, ns = get_Q(board, A, full_heur_dict; heur=heur)
        # Sample action from model
        ind_a = sample_action(x, Q)
        move = A[ind_a]
        # Add to output
        push!(qs, Q[ind_a])
        push!(Qs, Q)
        push!(visited, ns[ind_a])
        push!(neighbours, ns)
        # Make move
        make_move!(board, move)
        arr = get_board_arr(board)
        # Generate states until puzzle is solved
        if check_solved(arr)
            break
        end
    end
    return qs, Qs, visited, neighbours
end

function generate_fake_data(M, prbs, x, R, full_heur_dict; heur=4)
    qqs = Dict{String, DefaultDict{String, Array{Array{Float64, 1}, 1}}}()
    QQs = Dict{String, DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}}()
    visited_states = Dict{String, DefaultDict{String, Array{Array{BigInt, 1}, 1}}}()
    neighbour_states = Dict{String, DefaultDict{String, Array{Array{Array{BigInt, 1}, 1}, 1}}}()
    cnt = 0
    for m in ProgressBar(1:M)
        subj_dict_qs = DefaultDict{String, Array{Array{Float64, 1}, 1}}([])
        subj_dict_Qs = DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}([])
        subj_dict_visited = DefaultDict{String, Array{Array{BigInt, 1}, 1}}([])
        subj_dict_neigh = DefaultDict{String, Array{Array{Array{BigInt, 1}, 1}, 1}}([])
        for prb in prbs
            for r in 1:R
                x_n = x#[x[1] + 0.1 + randn()*0.1, x[2] + 0.5 + randn()*0.2]
                qs, Qs, visited, neighbours = generate_puzzle_trajectory(x_n, prb, full_heur_dict; heur=heur);
                push!(subj_dict_qs[prb], qs)
                push!(subj_dict_Qs[prb], Qs)
                push!(subj_dict_visited[prb], visited)
                push!(subj_dict_neigh[prb], neighbours)
                cnt += length(qs)
            end
        end
        qqs[string(m)] = subj_dict_qs
        QQs[string(m)] = subj_dict_Qs
        visited_states[string(m)] = subj_dict_visited
        neighbour_states[string(m)] = subj_dict_neigh
    end
    return qqs, QQs, visited_states, neighbour_states, cnt
end

function fake_and_or_data(params, trials_all, actions_all, action_lengths_all, blockages_all, ungreen_all, diff_nodes_all, parents_all, move_parents_all, h_all, moves_all)
    M = length(moves_all)
    fake_moves_all = Dict{String, Dict{String, Vector{Tuple{Int, Int}}}}()
    for (m, subj) in enumerate(keys(moves_all))
        β₁, β₂, β₃, β₄, β₅, β₆, k, λ = params[m, :]
        fake_moves_subj = Dict{String, Vector{Tuple{Int, Int}}}()
        for prb in keys(moves_all[subj])
            fake_moves = Vector{Tuple{Int, Int}}()
            for n in eachindex(moves_all[subj][prb])
                trials = trials_all[subj][prb][n]
                actions = actions_all[subj][prb][n]
                action_lengths = action_lengths_all[subj][prb][n]
                blockages = blockages_all[subj][prb][n]
                ungreen = ungreen_all[subj][prb][n]
                diff_nodes = diff_nodes_all[subj][prb][n]
                parents = parents_all[subj][prb][n]
                move_parents = move_parents_all[subj][prb][n]
                h = h_all[subj][prb][n]
                p = bfs_prev_move_probability(trials, actions, action_lengths, blockages, ungreen, diff_nodes, parents, move_parents, h, β₁, β₂, β₃, β₄, β₅, β₆, k, λ)
                push!(fake_moves, wsample(actions, p))
            end
            fake_moves_subj[prb] = fake_moves
        end
        fake_moves_all[subjs[m]] = fake_moves_subj
    end
    return fake_moves_all
end