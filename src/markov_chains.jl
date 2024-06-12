

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
        state_to_idx[s_int] = n
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
                R[id_from, id_to] += 1
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

function apply_gamma(Q::SparseMatrixCSC{Float64, Int64}, R::SparseMatrixCSC{Float64, Int64}, γ::Float64)::Vector{Float64}
    #QQ = Q .* (1-γ)
    #RR = R .* (1-γ)
    #RR[:, end] .= γ
    Rp = γ*ones(size(R, 1))
    B = (I - (1-γ)*Q)\Rp
    #B = (I-QQ)\RR
    return B#sum(B[:, 1:end-1], dims=2), B[:, end]
end

function p_a(k::Int, ns::Vector{Int32}, F::Vector{Float64}, state_to_idx::Dict{Int32, Int64})::Vector{Float64}
    m = length(ns)
    p_success = zeros(m)
    #p_success = []
    for (n, neigh) in enumerate(ns)
        p_success[n] = (1 - F[state_to_idx[neigh]]) / m
        #push!(p_success, 1 - F[state_to_idx[neigh]])
    end
    p_fail = 1 - sum(p_success)
    p_success .+= (p_fail^k)/m
    return p_success ./ sum(p_success)
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

function subj_nll(params, neighs_dict::Dict, moves_dict::Dict, all_moves_dict::Dict, dict::Dict, k::Int)::Float64
    logγ = params
    #k = 50
    #logγ, k = params
    γ = exp(-logγ)
    # k = round(Int, k)
    nll = 0
    i = 0
    for prb in keys(moves_dict)#ProgressBar
        #prb_df = subj_df[subj_df.puzzle .== prb, :]
        neighs = neighs_dict[prb]
        moves = moves_dict[prb]
        all_moves = all_moves_dict[prb]
        Q, R, state_to_idx = dict[prb]
        F = apply_gamma(Q, R, γ)
        #for row in eachrow(prb_df)
        for i in eachindex(moves)
            ps = p_a(k, neighs[i], F, state_to_idx)
            p = ps[findfirst(x-> x == moves[i], all_moves[i])]
            nll -= log(p)
        end
        # if i > 0
        #     break
        # end
        i += 1
    end
    return nll
end

function fit_subjects(df, dict)
    lb = [0.0, 0.0]
    plb = [0.5, 1.0]
    ub = [20.0, 200.0]
    pub = [3.0, 100.0]
    x0 = [1.0, 50.0]

    subjs = unique(df.subject)
    M = length(subjs)
    N = length(x0)
    params = zeros(M, 2)
    fitness = zeros(M)
    ks = 2 .^ (1:2:11)
    #ks = Int.(sort(vcat(2 .^ (1:10), round.(2 .^ (1.5:10.5)))))
    for m in 1:M
        subj_df = df[df.subject .== subjs[m] .&& df.event .== "move", :]
        neighs_dict, moves_dict, all_moves_dict = df_to_dict(subj_df)
        # subj_nll(x0, subj_df, dict)
        # break
        # pybads = pyimport("pybads")
        # BADS = pybads.BADS
        # bads_target = (x) -> subj_nll(x, subj_df, dict, 0)
        # options = Dict("tolfun"=> 0.00001, "max_fun_evals"=>50, "display"=>"iter");
        # bads = BADS(bads_target, x0, lb, ub, plb, pub, options=options)
        # res = bads.optimize();
        # params[m, :] = pyconvert(Vector, res["x"])
        # fitness[m] = pyconvert(Float64, res["fval"])
        # target = (x) -> subj_nll(x, neighs_dict, moves_dict, all_moves_dict, dict)
        # res = optimize(target, lb, ub, x0, Fminbox(), Optim.Options(f_tol = 0.000001, f_calls_limit=100, show_trace=true, show_every=1); autodiff=:forward)
        # res = optimize(target, lb, ub, x0, Fminbox(), Optim.Options(f_tol = 0.000001, f_calls_limit=30, show_trace=true, extended_trace=true, show_every=1))
        gammas = zeros(length(ks))
        fs = zeros(length(ks))
        for (n, k) in enumerate(ks)
            target = (x) -> subj_nll(x, neighs_dict, moves_dict, all_moves_dict, dict, k)
            res = optimize(target, lb[1], ub[1], Brent(); rel_tol=0.01, show_trace=true, extended_trace=true, show_every=1)
            gammas[n] = Optim.minimizer(res)
            fs[n] = Optim.minimum(res)
        end
        return ks, gammas, fs
        ind = argmin(fs)
        params[m, 1] = gammas[ind]
        params[m, 2] = ks[ind]
        fitness[m] = fs[ind]
        #params[m, 1] = Optim.minimizer(res)
        #fitness[m] = Optim.minimum(res)
        break
    end
    return params, fitness
end

# subj_df = df[df.subject .== subjs[1] .&& df.event .== "move", :];
# neighs_dict, moves_dict, all_moves_dict = df_to_dict(subj_df);

# @time subj_nll(1.0, neighs_dict, moves_dict, all_moves_dict, dict, 20);


@time kss, gammass, fss = fit_subjects(df, dict);
@time params, fitness = fit_subjects(df, dict);

prb = "prb32795_7";
s = load_data(prb)
visited = bfs(s[1], s[2]; d_goal=false)

Q, R, state_to_idx = get_QR(prb, visited)
@time F = apply_gamma(Q, R, 0.1)

ps = p_a(1000, ns, F, state_to_idx)



