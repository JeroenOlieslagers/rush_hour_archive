using Polylogarithms

function get_QR(prb, states)
    t = typeof(first(states))
    s_free, s_fixed = load_data(prb)
    L = 9
    transient_states = Set{t}()
    absorbing_states = Set{t}()
    for s_int in states
        s = (int32_to_board(s_int, L), s_fixed)
        if check_solved(s)
            push!(absorbing_states, s_int)
        else
            push!(transient_states, s_int)
        end
    end
    N_t = length(transient_states)
    N_a = length(absorbing_states)+1
    state_to_idx = Dict{t, Int}()
    for (n, s_int) in enumerate(transient_states)
        state_to_idx[s_int] = n
    end
    for (n, s_int) in enumerate(absorbing_states)
        state_to_idx[s_int] = -n
    end
    Q = zeros(N_t, N_t)
    R = zeros(N_t, N_a)
    arr = zeros(Int8, 6, 6)
    moves = @MVector([(Int8(0), Int8(0)) for _ in 1:(4*L)])
    for s_int in transient_states
        id_from = state_to_idx[s_int]
        int32_to_board!(s_free, s_int, L)
        s = (s_free, s_fixed)
        board_to_arr!(arr, s)
        possible_moves!(moves, s, arr)
        N_n = 0
        s_free = s[1]
        for m in moves
            if m[1] == 0
                break
            end
            make_move!(s_free, m)
            sp = board_to_int32(s_free)
            id_to = state_to_idx[sp]
            if sp in absorbing_states
                R[id_from, -id_to] += 1
            else
                Q[id_from, id_to] += 1
            end
            N_n += 1
            undo_move!(s_free, m)
        end
        Q[id_from, :] ./= N_n
        R[id_from, :] ./= N_n
    end
    return Q, R, state_to_idx
end

function get_QR_dict(prbs)
    dict = Dict()
    for prb in ProgressBar(prbs)
        s = load_data(prb)
        visited = bfs(s[1], s[2]; d_goal=false)
        Q, R, state_to_idx = get_QR(prb, visited)
        dict[prb] = [sparse(Q), sparse(R), state_to_idx]
    end
    return dict
end

function get_mc_dict(df, params, dict)
    subjs = unique(df.subject)
    mc_dict = Dict{String, Dict{String, Vector{Float64}}}()
    for (m, subj) in ProgressBar(enumerate(subjs))
        subj_dict = Dict{String, Vector{Float64}}()
        log_gamma, k = params[m, :]
        γ = exp(-log_gamma)
        subj_df = df[df.subject .== subjs[m] .&& df.event .== "move", :];
        for prb in unique(subj_df.puzzle)
            Q, R, state_to_idx = dict[prb]
            F = apply_gamma(Q, R, γ)
            subj_dict[prb] = F
        end
        mc_dict[subj] = subj_dict
    end
    return mc_dict
end

function apply_gamma(Q::SparseMatrixCSC{Float64, Int64}, R::SparseMatrixCSC{Float64, Int64}, γ::Float64)::Vector{Float64}
    Rp = γ*ones(size(R, 1))
    B = (I - (1-γ)*Q)\Rp
    return B
end

function p_a(ks::Vector{Int}, ns::Vector{Int32}, F::Vector{Float64}, state_to_idx::Dict{Int32, Int64})::Vector{Vector{Float64}}
    m = length(ns)
    p_success = zeros(m)
    for (n, neigh) in enumerate(ns)
        idx = state_to_idx[neigh]
        if idx < 0
            p_success[n] = 1 / m
        else
            p_success[n] = (1 - F[state_to_idx[neigh]]) / m
        end
    end
    p_fail = 1 - sum(p_success)
    if p_fail == 1
        ps = [ones(m) ./ m for k in ks]
    else
        ps = [(p_success .* (1 - (p_fail .^ k))/(1 - p_fail)) .+ (p_fail .^ k/m) for k in ks]
    end
    return ps
end

function df_to_dict(subj_df)
    neighs_dict = DefaultDict{String, Vector{Vector{Int32}}}([])
    moves_dict = DefaultDict{String, Vector{a_type}}([])
    all_moves_dict = DefaultDict{String, Vector{Vector{a_type}}}([])
    s_dict = DefaultDict{String, Vector{s_type}}([])
    for row in eachrow(subj_df)
        prb = row.puzzle
        push!(neighs_dict[prb], row.neighs)
        push!(moves_dict[prb], row.move)
        push!(all_moves_dict[prb], row.all_moves)
        push!(s_dict[prb], (row.s_free, row.s_fixed))
    end
    return Dict(neighs_dict), Dict(moves_dict), Dict(all_moves_dict), Dict(s_dict)
end

function subj_nll_mc(params, neighs_dict::Dict, moves_dict::Dict, all_moves_dict::Dict, dict::Dict, ks::Vector{Int}; return_all=false)#::Float64
    log_gamma = params
    γ = exp(-log_gamma)
    nll = zeros(length(ks))
    for prb in keys(moves_dict)
        neighs = neighs_dict[prb]
        moves = moves_dict[prb]
        all_moves = all_moves_dict[prb]
        Q, R, state_to_idx = dict[prb]
        F = apply_gamma(Q, R, γ)
        for i in eachindex(moves)
            ps = p_a(ks, neighs[i], F, state_to_idx)
            idx = findfirst(x-> x == moves[i], all_moves[i])
            for (n, pps) in enumerate(ps)
                if moves[i] isa Vector
                    for m in moves[i]
                        p = pps[findfirst(x-> x == m, all_moves[i])]
                        nll[n] -= log(p)
                    end
                else
                    p = pps[idx]
                    nll[n] -= log(p)
                end
            end
        end
        break
    end
    if return_all
        return nll
    else
        return minimum(nll)
    end
end

function generate_data(params, neighs_dict, moves_dict, all_moves_dict, s_dict, dict; N=1)
    logγ, k = params
    γ = exp(-logγ)
    k = round(Int, k)
    new_moves_dict = Dict{String, Vector{Vector{a_type}}}()
    for prb in keys(moves_dict)
        neighs = neighs_dict[prb]
        all_moves = all_moves_dict[prb]
        s = s_dict[prb]
        #Q, R, state_to_idx = dict[prb]
        #F = apply_gamma(Q, R, γ)
        moves = Vector{Vector{a_type}}()
        for (n, i) in enumerate(eachindex(neighs))
            #ps = p_a(γ, [k], neighs[i], F, state_to_idx)[1]
            #push!(moves, wsample(all_moves[i], ps, N))
            push!(moves, [forward_search_simulation(γ, k, s[i], all_moves[i]) for _ in 1:N])
        end
        new_moves_dict[prb] = moves
    end
    return new_moves_dict
end

function forward_search_simulation(γ, k, s, all_moves)
    f_move = nothing
    for i in 1:k
        sp = (copy(s[1]), s[2])
        solved, first_move = rollout(γ, sp)
        if solved
            f_move = first_move
            break
        end
    end
    if f_move === nothing
        return rand(all_moves)
    else
        return f_move
    end
end

function rollout(γ, s; max_iter=100000)
    s_free, s_fixed = s
    first_move = nothing
    moves = @MVector([(Int8(0), Int8(0)) for _ in 1:(4*L)])
    arr = zeros(Int8, 6, 6)
    for n in 1:max_iter
        if rand() > γ || n == 1
            # Expand current node by getting all available moves
            board_to_arr!(arr, s)
            #possible_moves!(moves, s, arr)
            N_m = possible_moves_N!(moves, s, arr)
            # Randomly choose a move
            selected_move_idx = rand(1:N_m)
            move = moves[selected_move_idx]
            # Make move
            make_move!(s_free, move)
            s = (s_free, s_fixed)
            if n == 1
                first_move = move
            end
            if check_solved(s)
                return true, first_move
            end
        else
            return false, first_move
        end
    end
    return false, first_move
end

function parameter_recovery(df, dict; N=8)
    # this can be very expensive to run, so we will run it on the cluster to parallelize
    k_lim_log10 = 3
    true_params = zeros(N, 2)
    true_params[:, 1] = ones(N)*log(2) .+ rand(N)*5
    true_params[:, 2] = floor.(10 .^ (rand(N)*k_lim_log10))
    fitted_params = zeros(N, 2)

    subjs = unique(df.subject)
    ks = unique(Int.(floor.(10 .^ (range(0, k_lim_log10, 100)))))
    for n in 1:N#ProgressBar() #Threads.@threads 
        subj_df = df[df.subject .== rand(subjs) .&& df.event .== "move", :]
        neighs_dict, moves_dict, all_moves_dict, s_dict = df_to_dict(subj_df)
        params = true_params[n, :]
        println("$(n) - Generating data")
        new_moves_dict = generate_data(params, neighs_dict, moves_dict, all_moves_dict, s_dict, dict; N=round(Int, 10_0 / size(subj_df, 1)))
        println("$(n) - Fitting MC")
        target = (x) -> subj_nll_mc(x, neighs_dict, new_moves_dict, all_moves_dict, dict, ks)
        res = optimize(target, 0.0, 6.0, Brent(); show_trace=true, extended_trace=true, show_every=1)
        fitted_params[n, 1] = Optim.minimizer(res)
        nlls = subj_nll_mc(fitted_params[n, 1, 1], neighs_dict, new_moves_dict, all_moves_dict, dict, ks; return_all=true)
        fitted_params[n, 2] = ks[argmin(nlls)]
    end
    return true_params, fitted_params
end