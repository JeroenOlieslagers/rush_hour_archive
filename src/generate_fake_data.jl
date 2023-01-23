include("engine.jl")

function sample_action(x::Vector{Float64}, Q::Vector{Float64})::Int
    λ, logb = x
    β = exp(logb)
    p = lapsed_softmin(λ, β, Q)
    return wsample(p)
end

function get_Q(board, A, full_heur_dict; heur=4)
    Q = zeros(length(A))
    for (n, a) in enumerate(A)
        make_move!(board, a)
        arr_a = get_board_arr(board)
        s = board_to_int(arr_a, BigInt)
        # Ignore if move outside state space
        if string(s) ∉ keys(full_heur_dict)
            undo_moves!(board, [a])
            return []
        end
        # Get specific heuristic value as Q value
        Q[n] = full_heur_dict[string(s)][heur]
        # neighs[n] = s
        undo_moves!(board, [a])
    end
    return Q
end

function generate_puzzle_trajectory(x, prb, full_heur_dict; max_iter=1000, heur=4)
    # Initialise outputs
    qs = Float64[]
    Qs = []
    # Load puzzle
    board = load_data(prb)
    arr = get_board_arr(board)
    for i in 1:max_iter
        # Get all neighbours
        A = get_all_available_moves(board, arr)
        Q = get_Q(board, A, full_heur_dict, heur=heur)
        # Sample action from model
        ind_a = sample_action(x, Q)
        move = A[ind_a]
        # Add to output
        push!(qs, Q[ind_a])
        push!(Qs, Q)
        # Make move
        make_move!(board, move)
        arr = get_board_arr(board)
        # Generate states until puzzle is solved
        if check_solved(arr)
            break
        end
    end
    return qs, Qs
end

function generate_fake_data(M, prbs, x, R)
    qqs = Dict{String, DefaultDict{String, Array{Array{Float64, 1}, 1}}}()
    QQs = Dict{String, DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}}()
    cnt = 0
    for m in ProgressBar(1:M)
        subj_dict_qs = DefaultDict{String, Array{Array{Float64, 1}, 1}}([])
        subj_dict_Qs = DefaultDict{String, Array{Array{Array{Float64, 1}, 1}, 1}}([])
        for prb in prbs
            for r in 1:R
                qs, Qs = generate_puzzle_trajectory(x, prb, full_heur_dict);
                push!(subj_dict_qs[prb], qs)
                push!(subj_dict_Qs[prb], Qs)
                cnt += length(qs)
            end
        end
        qqs[string(m)] = subj_dict_qs
        QQs[string(m)] = subj_dict_Qs
    end
    return qqs, QQs, cnt
end

