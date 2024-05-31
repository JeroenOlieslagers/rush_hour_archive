
function subject_nll_general(model, x, trees, dicts, states, boards, neighs, prev_moves, all_all_moves, moves, features, d_goals; high_res=false)
    if high_res
        nll = zeros(length(states))
    else
        nll = 0
    end
    for i in eachindex(states)#ProgressBar(
        tree = trees[i]
        dict = dicts[i]
        move = moves[i]
        all_moves = all_all_moves[i]
        s, board, neigh, prev_move, fs = states[i], boards[i], neighs[i], prev_moves[i], features[i]
        if sum(values(dict)) == 0
            println(board)
        end
        d_goal = d_goals[s]
        ps = model(x, tree, dict, all_moves, board, prev_move, neigh, d_goal, fs, d_goals)
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
            p = 0.00000001
        end
        if high_res
            nll[i] = -log(p)
        else
            nll -= log(p)
        end
    end
    return nll
end

function fit_model(model, lb, ub, x0, data_for_fitting, d_goals)#, plb, pub
    tree_datas, states, boards, neighs, prev_moves, features = data_for_fitting
    M = length(tree_datas)
    N = length(x0)
    params = zeros(M, N)
    fitness = zeros(M)
    #fitness = [[] for _ in 1:M]
    for m in ProgressBar(1:M)#Threads.@threads 
        tree_data = tree_datas[subjs[m]]
        states_subj = states[subjs[m]]
        boards_subj = boards[subjs[m]]
        neighs_subj = neighs[subjs[m]]
        prev_moves_subj = prev_moves[subjs[m]]
        features_subj = features[subjs[m]]
        trees, dicts, all_all_moves, moves = tree_data;
        if N == 1
            res = optimize((x) -> subject_nll_general(model, x, trees, dicts, states_subj, boards_subj, neighs_subj, prev_moves_subj, all_all_moves, moves, features_subj, d_goals), lb, ub)
            params[m, 1] = Optim.minimizer(res)
        else
            #bads_target = (x) -> subject_nll_general(model, x, trees, dicts, states_subj, boards_subj, neighs_subj, prev_moves_subj, all_all_moves, moves, features_subj, d_goals)
            #options = Dict("tolfun"=> 1, "max_fun_evals"=>50, "display"=>"iter");
            #bads = BADS(bads_target, x0, lb, ub, plb, pub, options=options)
            #res = bads.optimize();
            #params[m, :] = pyconvert(Vector, res["x"])
            #fitness[m] = pyconvert(Float64, res["fval"])
            res = optimize((x) -> subject_nll_general(model, x, trees, dicts, states_subj, boards_subj, neighs_subj, prev_moves_subj, all_all_moves, moves, features_subj, d_goals), lb, ub, x0, Fminbox(), Optim.Options(f_tol = 0.000001, f_calls_limit=100); autodiff=:forward)
            params[m, :] = Optim.minimizer(res)
        end
        #fitness[m] = subject_nll_general(model, Optim.minimizer(res), trees, dicts, states_subj, boards_subj, neighs_subj, prev_moves_subj, all_all_moves, moves, features_subj, d_goals; high_res=true)
        fitness[m] = Optim.minimum(res)
    end
    return params, fitness
end