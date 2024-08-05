
function subject_nll_general(model, x, df, d_goals_prbs)
    nll = 0
    if unique(df.event) != ["move"]
        throw(DomainError("More than just 'move' in events."))
    end
    for row in eachrow(df)
        ps = model(x, row, d_goals_prbs)
        if round(100000*sum(ps))/100000 != 1
            println("========$(i)========")
            println(sum(ps).value)
            println([pp.value for pp in ps])
            throw(DomainError("Not a valid probability distribution"))
        end
        # Likelihood of subject's move under model
        p = ps[findfirst(x-> x == row.move, row.all_moves)]
        if p == 0
            println("========$(i)========")
            println("Zero probability move")
            p = 0.00000001
        end
        nll -= log(p)
    end
    return nll
end

function fit_model(model, lb, ub, x0, df, d_goals_prbs, dict)
    subjs = unique(df.subject)
    M = length(subjs)
    N = length(x0)
    params = zeros(M, N)
    fitness = zeros(M)
    Threads.@threads for m in ProgressBar(1:M)#
        subj = subjs[m]
        df_subj = df[df.subject .== subj .&& df.event .== "move", :]
        if model == forward_search
            ks = unique(Int.(floor.(10 .^ (range(lb[2], ub[2], 1000)))))
            neighs_dict, moves_dict, all_moves_dict = df_to_dict(df_subj)

            target = (x) -> subj_nll_mc(x, neighs_dict, moves_dict, all_moves_dict, dict, ks)
            res = optimize(target, lb[1], ub[1], Brent(); rel_tol=0.001, show_trace=true, extended_trace=true, show_every=1)

            params[m, 1] = Optim.minimizer(res)
            nlls = subj_nll_mc(params[m, 1], neighs_dict, moves_dict, all_moves_dict, dict, ks; return_all=true)
            params[m, 2] = ks[argmin(nlls)]
            fitness[m] = nlls[argmin(nlls)]
        elseif N == 1
            res = optimize((x) -> subject_nll_general(model, x, df_subj, d_goals_prbs), lb, ub)
            params[m, 1] = Optim.minimizer(res)
            fitness[m] = Optim.minimum(res)
        else
            res = optimize((x) -> subject_nll_general(model, x, df_subj, d_goals_prbs), lb, ub, x0, Fminbox(), Optim.Options(f_tol = 0.000001, f_calls_limit=100); autodiff=:forward)
            params[m, :] = Optim.minimizer(res)
            fitness[m] = Optim.minimum(res)
        end
    end
    return params, fitness
end

function cross_validate(model, lb, ub, x0, df, d_goals_prbs, dict)
    subjs = unique(df.subject)
    M = length(subjs)
    N = length(x0)
    fitness = zeros(M)
    Threads.@threads for m in ProgressBar(1:M)#
        df_subj = df[df.subject .== subjs[m] .&& df.event .== "move", :]
        n = size(df_subj, 1)
        folds = collect(kfolds(shuffle(collect(1:n)), 5))
        for i in 1:5
            train, test = folds[i]
            df_train = df_subj[train, :]
            df_test = df_subj[test, :]
            if model == forward_search
                ks = unique(Int.(floor.(10 .^ (range(lb[2], ub[2], 1000)))))
                neighs_dict, moves_dict, all_moves_dict = df_to_dict(df_train)
    
                target = (x) -> subj_nll_mc(x, neighs_dict, moves_dict, all_moves_dict, dict, ks)
                res = optimize(target, lb[1], ub[1], Brent(); rel_tol=0.001, show_trace=true, extended_trace=true, show_every=1)
                log_gamma = Optim.minimizer(res)
                nlls_train = subj_nll_mc(log_gamma, neighs_dict, moves_dict, all_moves_dict, dict, ks; return_all=true)

                neighs_dict, moves_dict, all_moves_dict = df_to_dict(df_test)
                nlls_test = subj_nll_mc(log_gamma, neighs_dict, moves_dict, all_moves_dict, dict, ks; return_all=true)
                fitness[m] += nlls_test[argmin(nlls_train)]
            elseif N == 1
                res = optimize((x) -> subject_nll_general(model, x, df_train, d_goals_prbs), lb, ub)
                fitness[m] += subject_nll_general(model, Optim.minimizer(res), df_test, d_goals_prbs)
            else
                res = optimize((x) -> subject_nll_general(model, x, df_train, d_goals_prbs), lb, ub, x0, Fminbox(), Optim.Options(f_tol = 0.000001); autodiff=:forward)
                fitness[m] += subject_nll_general(model, Optim.minimizer(res), df_test, d_goals_prbs)
            end
        end
    end
    return fitness
end

function fit_all_models(df, d_goals_prbs, dict; m=1)
    models = [gamma_only_model, gamma_0_model, gamma_no_same_model, eureka_model, forward_search, opt_rand_model, hill_climbing_model, random_model]
    lbs = [0.000001, 0.000001, 0.000001, [0.0, 0.0], [0.0, 0.0], 0.000001, [-10.0, -10.0, -10.0], 0.0]
    ubs = [0.999999, 0.999999, 0.999999, [25.0, 1.0], [10.0, 10.0], 0.999999, [10.0, 10.0, 10.0], 1.0]
    x0s = [0.2, 0.2, 0.2, [10.0, 0.1], [2.0, 2.0], 0.2, [1.0, -1.0, -1.0, 1.0], 0.0]
    df_models = DataFrame(subject=String[], model=String[], nll=Float64[], cv_nll=Float64[])
    subjs = unique(df.subject)
    ps = []
    for i in eachindex(models)
        model = models[i]
        # The forward model is expensive to fit (and cross validate) so we will skip
        if model == forward_search
            continue
        end
        lb, ub, x0 = lbs[i], ubs[i], x0s[i]
        println("---fitting $(string(model)) ($(Threads.nthreads()) threads available) ---")
        p, f = fit_model(model, lb, ub, x0, df, d_goals_prbs, dict)
        println("---5-fold cross validation $(string(model)) ($(Threads.nthreads()) threads available) ---")
        cv = cross_validate(model, lb, ub, x0, df, d_goals_prbs, dict)
        for m in eachindex(f)
            push!(df_models, [subjs[m], string(model), f[m], cv[m]])
            push!(ps, p[m, :])
        end
    end
    return df_models, ps
end