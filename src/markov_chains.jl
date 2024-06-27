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
    for row in eachrow(subj_df)
        prb = row.puzzle
        push!(neighs_dict[prb], row.neighs)
        push!(moves_dict[prb], row.move)
        push!(all_moves_dict[prb], row.all_moves)
    end
    return Dict(neighs_dict), Dict(moves_dict), Dict(all_moves_dict)
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
                p = pps[idx]
                nll[n] -= log(p)
            end
        end
    end
    if return_all
        return nll
    else
        return minimum(nll)
    end
end
