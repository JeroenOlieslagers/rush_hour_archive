
function subject_nll_general(model, x, df, d_goals_prbs)
    nll = 0
    if unique(df.event) != ["move"]
        throw(DomainError("More than just 'move' in events."))
    end
    for row in eachrow(df)#ProgressBar(
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

function fit_model(model, lb, ub, x0, df, d_goals_prbs; plb=[], pub=[])
    subjs = unique(df.subject)
    M = length(subjs)
    N = length(x0)
    params = zeros(M, N)
    fitness = zeros(M)
    Threads.@threads for m in ProgressBar(1:M)#
        subj = subjs[m]
        df_subj = df[df.subject .== subj .&& df.event .== "move", :]
        if model == forward_search
            pybads = pyimport("pybads")
            BADS = pybads.BADS
            bads_target = (x) -> subject_nll_general(model, x, df_subj, d_goals_prbs)
            options = Dict("tolfun"=> 1, "max_fun_evals"=>50, "display"=>"iter");
            bads = BADS(bads_target, x0, lb, ub, plb, pub, options=options)
            res = bads.optimize();
            params[m, :] = pyconvert(Vector, res["x"])
            fitness[m] = pyconvert(Float64, res["fval"])
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

function cross_validate(model, lb, ub, x0, df, d_goals_prbs)
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
                pybads = pyimport("pybads")
                BADS = pybads.BADS
                bads_target = (x) -> subject_nll_general(model, x, df_subj, d_goals_prbs)
                options = Dict("tolfun"=> 1, "max_fun_evals"=>50, "display"=>"iter");
                bads = BADS(bads_target, x0, lb, ub, plb, pub, options=options)
                res = bads.optimize();
                fitness[m] += subject_nll_general(model, pyconvert(Vector, res["x"]), df_test, d_goals_prbs)
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

function fit_all_models(df, d_goals_prbs)
    models = [gamma_only_model, gamma_0_model, gamma_no_same_model, eureka_model, opt_rand_model, hill_climbing_model, random_model]
    lbs = [0.000001, 0.000001, 0.000001, [0.0, 0.0], 0.000001, [-10.0, -10.0, -10.0, 0.0], 0.0]
    ubs = [0.999999, 0.999999, 0.999999, [25.0, 1.0], 0.999999, [10.0, 10.0, 10.0, 20.0], 1.0]
    x0s = [0.2, 0.2, 0.2, [10.0, 0.1], 0.2, [1.0, -1.0, -1.0, 1.0], 0.0]
    df_models = DataFrame(subject=String[], model=String[], nll=Float64[], cv_nll=Float64[])
    subjs = unique(df.subject)
    ps = []
    for i in eachindex(models)
        model = models[i]
        lb, ub, x0 = lbs[i], ubs[i], x0s[i]
        println("---fitting $(string(model)) ($(Threads.nthreads()) threads available) ---")
        p, f = fit_model(model, lb, ub, x0, df, d_goals_prbs)
        println("---5-fold cross validation $(string(model)) ($(Threads.nthreads()) threads available) ---")
        cv = cross_validate(model, lb, ub, x0, df, d_goals_prbs)
        for m in eachindex(f)
            push!(df_models, [subjs[m], string(model), f[m], cv[m]])
            push!(ps, p[m, :])
        end
    end
    
    # df_subj = df[df.subject .== subjs[2] .&& df.event .== "move", :]
    # subject_nll_general(forward_search, [0.5, 50.0], df_subj, d_goals_prbs)
    #p, f = fit_model(forward_search, [0.0, 0.0], [5.0, 2000.0], [0.5, 50.0], df, d_goals_prbs; pub=[3.5, 1500.0], plb=[1.8, 500.0]);
    return df_models, ps
end

