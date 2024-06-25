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
    for (m, subj) in enumerate(subjs)
        subj_dict = Dict{String, Vector{Float64}}()
        log_gamma, k = params[m, :]
        γ = exp(-log_gamma)
        subj_df = df[df.subject .== subjs[1] .&& df.event .== "move", :];
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
    ps = [(p_success .* (1 - (p_fail .^ k))/(1 - p_fail)) .+ (p_fail .^ k/m) for k in ks]
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
                # for m in moves[i]
                #     p = pps[findfirst(x-> x == m, all_moves[i])]
                #     nll[n] -= log(p)
                # end
            end
        end
    end
    if return_all
        return nll
    else
        return minimum(nll)
    end
end

# function fit_subjects(df, dict)
#     subjs = unique(df.subject)
#     M = length(subjs)

#     params = zeros(M, 2)
#     fitness = zeros(M)
#     ks = unique(Int.(floor.(10 .^ (range(0, 10, 1000)))))
#     for m in 1:M
#         subj_df = df[df.subject .== subjs[m] .&& df.event .== "move", :]
#         neighs_dict, moves_dict, all_moves_dict = df_to_dict(subj_df)

#         target = (x) -> subj_nll(x, neighs_dict, moves_dict, all_moves_dict, dict, ks)
#         res = optimize(target, 0.0, 10.0, Brent();rel_tol=0.001, show_trace=true, extended_trace=true, show_every=1)
#         params[m, 1] = Optim.minimizer(res)
#         nlls = subj_nll(params[m, 1], neighs_dict, moves_dict, all_moves_dict, dict, ks; return_all=true)
#         params[m, 2] = ks[argmin(nlls)]
#         fitness[m] = nlls[argmin(nlls)]
#         break
#     end
#     return params, fitness
# end

# Q, R, state_to_idx = dict[prbs[1]];
# ns = neighs_dict[prbs[1]][1];
# @time F = apply_gamma(Q, R, exp(-0.1));
# @time a = p_a(exp(-2.0), ks, ns, F, state_to_idx);

# subj_df = df[df.subject .== subjs[1] .&& df.event .== "move", :];
# neighs_dict, moves_dict, all_moves_dict, s_dict = df_to_dict(subj_df);

# new_moves_dict = generate_data([3.0, 2000], neighs_dict, moves_dict, all_moves_dict, s_dict, dict; N=5)

# sum(length.(collect(values(moves_dict))))

# ks = unique(Int.(floor.(10 .^ (range(0, 8, 1000)))));
# lgs = range(log(2), 4, 10);

# landscape = zeros(length(lgs), length(ks));
# for i in ProgressBar(eachindex(lgs))
#     a = subj_nll(lgs[i], neighs_dict, new_moves_dict, all_moves_dict, dict, ks; return_all=true);
#     landscape[i, :] = a
# end
# heatmap(log10.(ks), lgs, landscape, cmap=cgrad(:nipy_spectral, rev=true), xlabel="log k", ylabel="-log gamma")#, xscale=:log10)

# fitted_params = []
# true_params = []
# for file in readdir("cluster_params/paramss")
#     if split(file, "_")[1] == "fitted"
#         push!(fitted_params, load("cluster_params/paramss/"*file)["data"][1, :, 1])
#     elseif split(file, "_")[1] == "true"
#         push!(true_params, load("cluster_params/paramss/"*file)["data"][1, :])
#     end
# end

# plot(layout=(1, 2), grid=false, aspect_ratio=1)
# #plot(layout=(1, 1), grid=false, aspect_ratio=1)
# for i in eachindex(true_params)
#     tp = true_params[i]
#     fp = fitted_params[i]
#     scatter!([tp[1]], [fp[1]], sp=1, c=:red, label=nothing, xlim=(0, 6.2), ylim=(0, 6.2))
#     scatter!(log10.([tp[2]]), log10.([fp[2]]), sp=2, c=:blue, label=nothing, xlim=(0, 6.2), ylim=(0, 6.2))
# end
# plot!([0, 6], [0, 6], sp=1, c=:black, label=nothing, title="log gamma")
# plot!([0, 6], [0, 6], sp=2, c=:black, label=nothing, title="log k")
# plot!(xlabel="true", ylabel="fitted")







