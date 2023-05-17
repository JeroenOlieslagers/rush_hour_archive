using Distances

function create_move_icon(move, board)
    car, m = move
    s = string(car)
    if board.cars[car].is_horizontal
        for i in 1:abs(m)
            if sign(m) == 1
                s *= "→"
            else
                s *= "←"
            end
        end
    else
        for i in 1:abs(m)
            if sign(m) == 1
                s *= "↓"
            else
                s *= "↑"
            end
        end
    end
    return s
end

function create_node_string(node; rev=false, visited=false)
    s = ""
    if visited
        s *= "v"
    end
    if rev
        return s*string(node.value) *"_"* join([join(reverse(car)) for car in reverse(node.children)], "_")
    else
        return s*string(node.value) *"_"* join([join(car) for car in node.children], "_")
    end
end

struct Node
    value::Int
    children::Vector{Vector{Int}}
end

function Base.:(==)(n1::Node, n2::Node)
    n1.value == n2.value && n1.children == n2.children
end

function new_and_or_tree(board; max_iter=1000)
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    tree = Dict{s_type, Dict{Tuple{Int, Int}, Vector{s_type}}}()
    frontier = Vector{s_type}()
    visited = Vector{s_type}()
    parents_of_actions = Vector{s_type}()
    actions = Vector{Tuple{Int, Int}}()
    trials = Vector{Int}()
    parents = DefaultDict{s_type, Vector{Tuple{s_type, Tuple{Int, Int}}}}([])
    fs = Dict{s_type, Int}()
    hs = DefaultDict{s_type, Int}(0)

    arr = get_board_arr(board)
    m_init = 6 - (board.cars[9].x+1)
    root = (9, (m_init,))
    push!(frontier, root)
    fs[root] = 0

    for i in 0:max_iter-1
        if isempty(frontier)
            break
        end
        dict = Dict{Tuple{Int, Int}, Vector{s_type}}()
        # BFS change to pop! for DFS
        s_old = popfirst!(frontier)
        car_id, moves = s_old
        f_new = fs[s_old] + 1
        push!(visited, s_old)
        s_new_add = nothing
        for m in reverse(moves)
            move = (car_id, m)
            cars = move_blocked_by(board.cars[car_id], m, arr)
            if length(parents[s_old]) > 0
                parent_h = minimum([hs[parents[s_old][i][1]] for i in eachindex(parents[s_old])])
            else
                parent_h = 0
            end
            if length(cars) == 0 && move ∉ actions
                if parent_h == 0
                    hs[s_old] = 1
                else
                    hs[s_old] = parent_h + 1
                end
                push!(actions, move)
                push!(parents_of_actions, s_old)
                #push!(trials, i)
                # BFS trials
                push!(trials, f_new)
            else
                if parent_h > 0
                    hs[s_old] = parent_h + 1
                end
            end
            ls = Vector{s_type}()
            for car in cars
                moves_new, _ = moves_that_unblock(board.cars[car_id], board.cars[car], arr, move_amount=m)
                s_new = (car, Tuple(moves_new))
                if length(moves_new) == 0
                    push!(ls, s_new)
                    break
                end
                # 1 more lookahead
                # for mm in moves_new
                #     mmove = (car, mm)
                #     if isempty(move_blocked_by(board.cars[car], mm, arr)) && mmove ∉ actions
                #         push!(actions, mmove)
                #         push!(trials, i)
                #     end
                # end
                if (s_old, move) ∉ parents[s_new]#(s_new_add !== nothing || s_old in keys(tree)) && 
                    push!(parents[s_new], (s_old, move))
                end
                if s_new ∉ visited && s_new ∉ frontier 
                    s_new_add = s_new
                    if (s_old, move) ∉ parents[s_new]
                        push!(parents[s_new], (s_old, move))
                    end
                    push!(frontier, s_new)
                    fs[s_new] = f_new
                end
                push!(ls, s_new)
            end
            dict[move] = ls
        end
        if car_id == 9 && [sort(unique(reduce(vcat, collect(values(dict)))))] ∈ [sort.(unique(collect(values(a)))) for a in values(tree)]
            continue
        elseif length(reduce(vcat, collect(values(dict)))) == 0
            tree[s_old] = dict
        elseif unique(collect(values(dict))) ∉ [unique(collect(values(a))) for a in values(tree)]
            tree[s_old] = dict
        end
    end
    move_parents = DefaultDict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}([])
    h = DefaultDict{Tuple{Int, Int}, Vector{Int}}([])
    for a in actions
        for s_new in keys(parents)
            if s_new[1] == a[1] && a[2] in s_new[2]
                for i in eachindex(parents[s_new])
                    s_old, move = parents[s_new][i]
                    if s_old in keys(tree)
                        if length(tree[s_old][move]) > 1
                            ac, ss = find_children(tree, s_old, s_new)
                            push!(move_parents[a], ac...)
                            push!(h[a], [fs[si] for si in ss]...)
                        else
                            push!(move_parents[a], [parents[s_new][j][2] for j in eachindex(parents[s_new])]...)
                            push!(h[a], [-fs[parents[s_new][j][1]] for j in eachindex(parents[s_new])]...)
                        end
                    end
                end
            end
        end
        ind = unique(i -> move_parents[a][i], 1:length(move_parents[a]))
        move_parents[a] = move_parents[a][ind]
        h[a] = h[a][ind]
    end
    return tree, visited, actions, trials, parents_of_actions, move_parents, h
end

function find_children(tree, s, s_current; max_iter=1000)
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    frontier = Vector{s_type}()
    visited = Vector{s_type}()
    ss = Vector{s_type}()
    actions = Vector{Tuple{Int, Int}}()
    push!(frontier, s)
    for i in 1:max_iter
        if isempty(frontier)
            break
        end
        s_old = popfirst!(frontier)
        push!(visited, s_old)
        for move in keys(tree[s_old])
            if length(tree[s_old][move]) == 0 && move[1] != s_current[1] && move ∉ actions
                push!(actions, move)
                push!(ss, s_old)
            else
                for s_new in tree[s_old][move]
                    if s_new != s_current && s_new in keys(tree) && s_new ∉ visited && s_new ∉ frontier
                        push!(frontier, s_new)
                    end
                end
            end
        end
    end
    return actions, ss
end

function and_or_tree(board; γ=0.0, DFS=true, max_iter=100)
    arr = get_board_arr(board)
    ROOT = length(unique(arr))-1
    begin
        # INITIALIZE
        #frontier = Vector{Tuple{Node, Int, Int}}()
        frontier = Vector{Tuple{Node, Int, Int, Tuple{Int, Int}}}()
        visited = Vector{Node}()
        timeline = Vector{Tuple{Node, Tuple{Int, Int}, Int}}()
        actions = Vector{Tuple{Int, Int}}()
        trials = Vector{Int}()
        draw_tree_nodes = DefaultOrderedDict{String, Vector{String}}([])
        last_move_red = false
        last_or_node = -1
        last_prev_move = (-1, 0)
        last_current_car = -1
        iter = 0
        # GET FIRST BLOCKED CARS
        if DFS
            red_car_block_init = reverse(blocked_by(arr, board.cars[ROOT]))
        else
            red_car_block_init = blocked_by(arr, board.cars[ROOT])
        end
        # GET POSSIBLE FIRST MOVES
        available_moves = []
        get_available_moves!(available_moves, arr, board.cars[ROOT])
        if length(available_moves) > 0
            available_moves_ = []
            # for move in available_moves
            #     if move[2] > 0
            #         push!(available_moves_, move)
            #     end
            # end
            # CREATE ROOT NODE
            start_node = Node(ROOT, vcat([red_car_block_init], [[move[2]*1000] for move in reverse(available_moves_)]))
            for move in available_moves_
                push!(actions, (move[1], move[2]))
                push!(trials, 0)
                push!(draw_tree_nodes[create_node_string(start_node, rev=DFS)], "m"*create_move_icon((move[1], move[2]), board))
            end
        else
            start_node = Node(ROOT, [red_car_block_init])
        end
        push!(visited, start_node)
        # INITIALIZE FRONTIER
        for i in eachindex(red_car_block_init)
            if DFS
                push!(frontier, (start_node, 1, i, (ROOT, 5-board.cars[end].x)))
            else
                pushfirst!(frontier, (start_node, 1, i))
            end
        end
    end
    for _ in 1:max_iter
        if isempty(frontier) || rand() < γ
            #println("REACHED THE END IN $(iter-1) ITERATIONS")
            return actions, trials, visited, timeline, draw_tree_nodes
        end
        # GET DFS NODE
        current_node, or_ind, and_ind, prev_move = pop!(frontier)
        # SET NEW PARENT
        prev_car = current_node.value
        # SET NEXT TARGET
        current_car = current_node.children[or_ind][and_ind]
        push!(timeline, (current_node, prev_move, current_car))
        # GET UNBLOCKING MOVES
        moves, next_cars = moves_that_unblock(board.cars[prev_car], board.cars[current_car], arr, move_amount=prev_move[2])
        # SORT
        if !DFS
            reverse!.(next_cars)
        end
        if DFS
            sorted_cars = reverse(unique(next_cars))
        else
            sorted_cars = unique(next_cars)
        end
        # POSSIBLE MOVE SPECIAL CASE
        if any(isempty.(sorted_cars))
            if DFS
                new_node = Node(current_car, vcat(sorted_cars[.!(isempty.(sorted_cars))], [[aa] for aa in moves[isempty.(next_cars)]*1000]))
            else
                new_node = Node(current_car, vcat([[aa] for aa in reverse(moves[isempty.(next_cars)]*1000)], sorted_cars[.!(isempty.(sorted_cars))]))
            end
        else
            new_node = Node(current_car, sorted_cars)
        end
        # CHECK IF LAST MOVE WAS NOT ALLOWED AND IF IN SAME OR NODE
        # if last_move_red && last_prev_move[1] == prev_move[1]
        #     if last_or_node == or_ind
        #         push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true, visited=true))
        #         continue
        #     elseif last_or_node - 1 == or_ind && last_current_car == current_car
        #         push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true, visited=true))
        #         last_or_node = or_ind
        #         continue
        #     end
        # end
        iter += 1
        last_move_red = false
        # RED CAR SPECIAL CASE
        if current_car == ROOT && !any(isempty.(sorted_cars)) && length(sorted_cars) > 0
            check_car = Node(current_car, [union(sorted_cars...)])
            if check_car in visited
                if DFS
                    push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true, visited=true))
                else
                    draw_tree_list = draw_tree_nodes[create_node_string(current_node)]
                    idx = findlast(x->x[1]=='m', draw_tree_list)
                    if idx !== nothing
                        insert!(draw_tree_list, idx+1, create_node_string(new_node, visited=true))
                    else
                        push!(draw_tree_nodes[create_node_string(current_node)], create_node_string(new_node, visited=true))
                    end
                end
                push!(timeline, (new_node, prev_move, current_car))
                last_move_red = true
                last_or_node = or_ind
                last_prev_move = prev_move
                last_current_car = current_car
                continue
            end
        end
        # DO NOT EXPAND IF ALREADY VISITED
        if new_node in visited
            if DFS
                push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true, visited=true))
            else
                draw_tree_list = draw_tree_nodes[create_node_string(current_node)]
                idx = findlast(x->x[1]=='m', draw_tree_list)
                if idx !== nothing
                    insert!(draw_tree_list, idx+1, create_node_string(new_node, visited=true))
                else
                    push!(draw_tree_nodes[create_node_string(current_node)], create_node_string(new_node, visited=true))
                end
            end
            push!(timeline, (new_node, prev_move, current_car))
            last_move_red = true
            last_or_node = or_ind
            last_prev_move = prev_move
            last_current_car = current_car
            continue
        else
            if DFS
                push!(draw_tree_nodes[create_node_string(current_node, rev=true)], create_node_string(new_node, rev=true))
            else
                draw_tree_list = draw_tree_nodes[create_node_string(current_node)]
                idx = findlast(x->x[1]=='m', draw_tree_list)
                if idx !== nothing
                    insert!(draw_tree_list, idx+1, create_node_string(new_node))
                else
                    push!(draw_tree_nodes[create_node_string(current_node)], create_node_string(new_node))
                end
            end
            push!(visited, new_node)
        end
        # EXTEND FRONTIER
        for i in eachindex(sorted_cars)
            if length(sorted_cars[i]) > 0
                for j in eachindex(sorted_cars[i])
                    if DFS
                        push!(frontier, (new_node, i, j, (new_node.value, moves[findfirst(x->x==sorted_cars[i], next_cars)])))
                    else
                        pushfirst!(frontier, (new_node, i, j))
                    end
                end
            else
                # IF UNBLOCKING MOVE IS VIABLE, ADD IT TO THE LIST
                for m in moves[isempty.(next_cars)]
                    a = (current_car, m)
                    # if a in actions
                    #     push!(trials[findfirst(x->x==a, actions)], iter)
                    # else
                    if a ∉ actions
                        push!(actions, a)
                        push!(trials, iter)
                    end
                    push!(timeline, (new_node, a, current_car))
                    push!(draw_tree_nodes[create_node_string(new_node, rev=DFS)], "m"*create_move_icon((current_car, m), board))
                end
            end
        end
    end
    return throw(ErrorException("Could not find solution"))
end

function softmax(arr, β)
    ls = exp.(β*arr)
    return ls / sum(ls)
end

function softmax_over_same_parent_moves(parents, actions, p, β)
    for u in unique(parents)
        ind = findall(x->x==u, parents)
        L = length(ind)
        if L > 1
            pp = zeros(L)
            for i in eachindex(ind)
                pp[i] = abs(actions[ind[i]][2])
            end
            p[ind] = sum(p[ind])*softmax(pp, β)
        end
    end
    return p
end

function dfs_rollout_probability(trials, actions, γ::T, ϵ::T, β::T) where T
    N = length(trials)
    tr = trials[trials .< 1000000]
    maxx = maximum(tr)
    p_pass = ones(T, maxx+1)*(1-γ)
    p_pass[1] = 1
    p_pass[tr .+ 1] .*= ϵ
    cp = cumprod(p_pass)
    p_pass = 1 .- p_pass
    p = zeros(T, N)
    for i in 1:maxx
        if i < minimum(trials)
            p .+= p_pass[i+1]*cp[i] / N
        else
            p[trials .<= i] .+= p_pass[i+1]*cp[i] / sum(trials .<= i)
        end
    end
    p[trials .<= maxx] .+= cp[end] / sum(trials .<= maxx)
    p = softmax_over_same_parent_moves(parents, actions, p, β)
    # for u in unique(trials)
    #     ind = findall(x->x==u, trials)
    #     L = length(ind)
    #     if L > 1
    #         pp = zeros(L)
    #         for i in eachindex(ind)
    #             pp[i] = abs(actions[ind[i]][2])
    #         end
    #         p[ind] = sum(p[ind])*softmax(pp, β)
    #     end
    # end
    return p
end

function bfs_prev_move_probability(trials, actions, parents, move_parents, h, β₁::T, β₂::T, β₃::T, k::T, λ::T) where T
    # Completing the distributions
    p1 = zeros(T, length(actions))
    p2 = zeros(T, length(actions))
    # Probability distribution for replanning
    p1[trials .< 1000000] = softmax(trials[trials .< 1000000], β₁)
    # Probability distribution for using old plan
    pp = softmax(h, β₂)
    for i in eachindex(actions)
        if actions[i] in move_parents
            for j in eachindex(move_parents)
                if actions[i] == move_parents[j]
                    p2[i] = pp[j]
                end
            end
        end
    end
    # Mixture model
    p = k*p1 + (1-k)*p2
    # Editing moves with same parent
    p = softmax_over_same_parent_moves(parents, actions, p, β₃)
    return λ/length(p) .+ (1-λ)*p
end

function probability_over_moves(trials, actions, γ, ϵ, β, λ)
    p = dfs_rollout_probability(trials, actions, γ, ϵ, β)
    p = λ/length(p) .+ (1-λ)*p
    return p
end

function analyze_all_trees(trials_all, actions_all, parents_all, move_parents_all, h_all, moves_all, opt_all, d_goal_all; time_limit=100)
    M = length(moves_all)
    params = zeros(M, 5)
    #params = zeros(M, 4)
    #params = zeros(M, 2)
    fitness = zeros(M)
    Threads.@threads for m in ProgressBar(1:M) 
        x0 = [-1.0, -1.0, 1.0, 0.5, 0.2]
        #x0 = [0.1, 2.0, 0.9, 0.0]
        #x0 = [rand()*0.5, rand()*4, 0.8 + rand()*0.2, (rand()*5) - 2.5]
        #x0 = [0.3, 8]
        trials_subj = trials_all[subjs[m]]
        actions_subj = actions_all[subjs[m]]
        parents_subj = parents_all[subjs[m]]
        move_parents_subj = move_parents_all[subjs[m]]
        h_subj = h_all[subjs[m]]
        moves_subj = moves_all[subjs[m]]
        opt_subj = opt_all[subjs[m]]
        d_goal_subj = d_goal_all[subjs[m]]
        res = optimize((x) -> fit_subj(x, trials_subj, actions_subj, parents_subj, move_parents_subj, h_subj, moves_subj, opt_subj, d_goal_subj), [-10.0, -10.0, -10.0, 0.0, 0.0], [10.0, 10.0, 10.0, 1.0, 1.0], x0, Fminbox(), Optim.Options(time_limit=time_limit), autodiff=:forward)
        #res = optimize((x) -> fit_subj(x, trials_subj, actions_subj, parents_subj, move_parents_subj, h_subj, moves_subj, opt_subj, d_goal_subj), [0.0, 0.0, 0.0, -10], [1.0, 10.0, 1.0, 10], x0, Fminbox(), Optim.Options(time_limit=time_limit), autodiff=:forward)
        #res = optimize((x) -> fit_subj(x, trials_subj, actions_subj, moves_subj, opt_subj, d_goal_subj), [0.0, 0.0,], [1.0, 20.0], x0, Fminbox(), Optim.Options(time_limit=time_limit), autodiff=:forward)
        params[m, :] = Optim.minimizer(res)
        fitness[m] = Optim.minimum(res)
    end
    return fitness, params
end

function eureka(actions, opt, d_goal, d, λ)
    if d_goal <= d
        p = zeros(length(actions))
        for i in eachindex(actions)
            if actions[i] in opt
                p[i] = 1
            end
        end
    else
        p = ones(length(actions))
    end
    p ./= sum(p)
    return λ/length(p) .+ (1-λ)*p
end

function fit_subj(x, trials_subj, actions_subj, parents_subj, move_parents_subj, h_subj, moves_subj, opt_subj, d_goal_subj)
    β₁, β₂, β₃, k, λ = x
    # λ, logγ, ϵ, β = x
    # γ = exp(-logγ)
    # λ, d_fl = x
    # d = Int(round(d_fl))
    nll = 0
    for prb in keys(moves_subj)
        trials_prb = trials_subj[prb]
        actions_prb = actions_subj[prb]
        parents_prb = parents_subj[prb]
        move_parents_prb = move_parents_subj[prb]
        h_prb = h_subj[prb]
        moves_prb = moves_subj[prb]
        opt_prb = opt_subj[prb]
        d_goal_prb = d_goal_subj[prb]
        for n in eachindex(moves_prb)
            trials = trials_prb[n]
            actions = actions_prb[n]
            parents = parents_prb[n]
            move_parents = move_parents_prb[n]
            h = h_prb[n]
            m = moves_prb[n]
            opt = opt_prb[n]
            d_goal = d_goal_prb[n]
            p = bfs_prev_move_probability(trials, actions, parents, move_parents, h, β₁, β₂, β₃, k, λ)
            #p = probability_over_moves(trials, actions, γ, ϵ, β, λ)
            #p = eureka(actions, opt, d_goal, d, λ)
            nll -= log(p[findfirst(x->x==m, actions)])
        end
    end
    return nll
end

function get_errors(params, params_a, states_all, trials_all, actions_all, moves_all, opt_all, d_goal_all, n_a_all)
    error_l_model = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_model = Dict{String, Array{Array{Int, 1}, 1}}()
    error_l_model_a = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_model_a = Dict{String, Array{Array{Int, 1}, 1}}()
    error_l_data = Dict{String, Array{Array{Int, 1}, 1}}()
    error_a_data = Dict{String, Array{Array{Int, 1}, 1}}()

    ps, pps, acs, ops, movs, js, ss = [], [], [], [], [], [], []
    for (m, subj) in enumerate(keys(trials_all))
        x = params[m, :]
        λ, logγ, ϵ, β = x
        γ = exp(-logγ)

        xx = params_a[m, :]
        λ2, d_fl = xx
        d = Int(round(d_fl))
        error_l_model[subj] = [[] for _ in 1:35]
        error_a_model[subj] = [[] for _ in 1:35]
        error_l_model_a[subj] = [[] for _ in 1:35]
        error_a_model_a[subj] = [[] for _ in 1:35]
        error_l_data[subj] = [[] for _ in 1:35]
        error_a_data[subj] = [[] for _ in 1:35]
        for prb in keys(trials_all[subj])
            for i in eachindex(trials_all[subj][prb])
                state = states_all[subj][prb][i]
                trials = trials_all[subj][prb][i]
                actions = actions_all[subj][prb][i]
                move = moves_all[subj][prb][i]
                opt = opt_all[subj][prb][i]
                d_goal = d_goal_all[subj][prb][i]
                n_a = n_a_all[subj][prb][i]
                # ss = sum(trials .== 1000000)
                # ss = ss > 0 ? ss : length(trials)
                p = probability_over_moves(trials, actions, γ, ϵ, 0.0, λ)
                model_a = wsample(actions, p)
                if model_a ∉ opt
                    # if d_goal == 2
                    #     println(subj, prb, i)
                    #     println(move)
                    #     println(actions)
                    #     println(p)
                    #     println(model_a)
                    #     println(trials)
                    #     return 0,0
                    # end
                    push!(error_l_model[subj][d_goal], 1)
                    push!(error_a_model[subj][n_a], 1)
                else
                    push!(error_l_model[subj][d_goal], 0)
                    push!(error_a_model[subj][n_a], 0)
                end
                if move ∉ opt
                    push!(error_l_data[subj][d_goal], 1)
                    push!(error_a_data[subj][n_a], 1)
                else
                    push!(error_l_data[subj][d_goal], 0)
                    push!(error_a_data[subj][n_a], 0)
                end
                
                pp = eureka(actions, opt, d_goal, d, λ2)
                model_aa = wsample(actions, pp)
                if model_aa ∉ opt
                    push!(error_l_model_a[subj][d_goal], 1)
                    push!(error_a_model_a[subj][n_a], 1)
                else
                    push!(error_l_model_a[subj][d_goal], 0)
                    push!(error_a_model_a[subj][n_a], 0)
                end

                push!(ps, p)
                push!(pps, pp)
                push!(acs, actions)
                push!(ops, opt)
                push!(movs, move)
                push!(js, js_divergence(p, pp))
                push!(ss, state)
            end
        end
    end
    return ps, pps, acs, ops, movs, js, ss
    #return error_l_model, error_a_model, error_l_model_a, error_a_model_a, error_l_data, error_a_data
end

function get_and_or_tree_data(data)
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # Initialize
    states_all = Dict{String, Dict{String, Vector{BigInt}}}()
    trials_all = Dict{String, Dict{String, Vector{Vector{Int}}}}()
    actions_all = Dict{String, Dict{String, Vector{Vector{Tuple{Int, Int}}}}}()
    parents_all = Dict{String, Dict{String, Vector{Vector{s_type}}}}()
    move_parents_all = Dict{String, Dict{String, Vector{Vector{Tuple{Int, Int}}}}}()
    h_all = Dict{String, Dict{String, Vector{Vector{Int}}}}()
    moves_all = Dict{String, Dict{String, Vector{Tuple{Int, Int}}}}()
    opt_all = Dict{String, Dict{String, Vector{Vector{Tuple{Int, Int}}}}}()
    d_goal_all = Dict{String, Dict{String, Vector{Int}}}()
    n_a_all = Dict{String, Dict{String, Vector{Int}}}()
    white_size_all = Dict{String, Dict{String, Vector{Int}}}()
    red_size_all = Dict{String, Dict{String, Vector{Int}}}()
    optimal_act = load("data/processed_data/optimal_act.jld2")
    IDV = load("data/processed_data/IDV_OLD.jld2")
    for subj in ProgressBar(keys(data))
        # Initialize
        states_all_prb = Dict{String, Vector{BigInt}}()
        trials_all_prb = Dict{String, Vector{Vector{Int}}}()
        actions_all_prb = Dict{String, Vector{Vector{Tuple{Int, Int}}}}()
        parents_all_prb = Dict{String, Vector{Vector{s_type}}}()
        move_parents_all_prb = Dict{String, Vector{Vector{Tuple{Int, Int}}}}()
        h_all_prb = Dict{String, Vector{Vector{Int}}}()
        moves_all_prb = Dict{String, Vector{Tuple{Int, Int}}}()
        opt_all_prb = Dict{String, Vector{Vector{Tuple{Int, Int}}}}()
        d_goal_all_prb = Dict{String, Vector{Int}}()
        n_a_all_prb = Dict{String, Vector{Int}}()
        white_size_all_prb = Dict{String, Vector{Int}}()
        red_size_all_prb = Dict{String, Vector{Int}}()
        # Get all moves from data
        tot_arrs, tot_move_tuples, tot_states_visited, attempts = analyse_subject(data[subj])
        for prb in keys(tot_move_tuples)
            # Initialize
            states = Vector{BigInt}()
            trials_state = Vector{Vector{Int}}()
            actions_state = Vector{Vector{Tuple{Int, Int}}}()
            parents_state = Vector{Vector{s_type}}()
            move_parents_state = Vector{Vector{Tuple{Int, Int}}}()
            h_state = Vector{Vector{Int}}()
            moves_state = Vector{Tuple{Int, Int}}()
            opt_state = Vector{Vector{Tuple{Int, Int}}}()
            d_goal = Vector{Int}()
            n_a = Vector{Int}()
            white_size = Vector{Int}()
            red_size = Vector{Int}()
            push!(h_state, [])
            push!(move_parents_state, [])
            # Initial load
            board = load_data(prb)
            for move in tot_move_tuples[prb]
                # Load board if restart / start
                if move == (-1, 0)
                    board = load_data(prb)
                    continue
                end
                # Get AND/OR tree
                #actions, trials, visited, timeline, draw_tree_nodes = and_or_tree(board)
                tree, visited, actions, trials, parents, move_parents, h = new_and_or_tree(board)
                arr = get_board_arr(board)
                # Incorporate other moves
                all_moves = get_all_available_moves(board, arr)
                for movee in all_moves
                    mm = Tuple(movee)
                    if mm ∉ actions
                        push!(actions, mm)
                        push!(trials, 1000000)
                    end
                end
                d_goal_s = IDV[prb][board_to_int(arr, BigInt)][1]
                n_a_s = IDV[prb][board_to_int(arr, BigInt)][2]
                if d_goal_s == 0
                    break
                end
                push!(states, board_to_int(arr, BigInt))
                push!(trials_state, trials)
                push!(actions_state, actions)
                push!(parents_state, parents)
                if Tuple(move) in keys(move_parents)
                    push!(h_state, h[Tuple(move)])
                    push!(move_parents_state, move_parents[Tuple(move)])
                else
                    push!(h_state, [])
                    push!(move_parents_state, [])
                end
                push!(moves_state, Tuple(move))
                push!(opt_state, optimal_act[prb][board_to_int(arr, BigInt)])
                push!(d_goal, d_goal_s)
                push!(n_a, n_a_s)
                push!(white_size, length(visited))
                push!(red_size, length(reduce(vcat, values(draw_tree_nodes))))
                # Next state
                make_move!(board, move)
            end
            pop!(h_state)
            pop!(move_parents_state)
            states_all_prb[prb] = states
            trials_all_prb[prb] = trials_state
            actions_all_prb[prb] = actions_state
            parents_all_prb[prb] = parents_state
            move_parents_all_prb[prb] = move_parents_state
            h_all_prb[prb] = h_state
            moves_all_prb[prb] = moves_state
            opt_all_prb[prb] = opt_state
            d_goal_all_prb[prb] = d_goal
            n_a_all_prb[prb] = n_a
            white_size_all_prb[prb] = white_size
            red_size_all_prb[prb] = red_size
        end
        states_all[subj] = states_all_prb
        trials_all[subj] = trials_all_prb
        actions_all[subj] = actions_all_prb
        parents_all[subj] = parents_all_prb
        move_parents_all[subj] = move_parents_all_prb
        h_all[subj] = h_all_prb
        moves_all[subj] = moves_all_prb
        opt_all[subj] = opt_all_prb
        d_goal_all[subj] = d_goal_all_prb
        n_a_all[subj] = n_a_all_prb
        white_size_all[subj] = white_size_all_prb
        red_size_all[subj] = red_size_all_prb
    end
    return states_all, trials_all, actions_all, parents_all, move_parents_all, h_all, moves_all, opt_all, d_goal_all, n_a_all, white_size_all, red_size_all
end

function dict_of_dicts_to_arr(dict)
    ls = []
    for k in keys(dict)
        push!(ls, reduce(vcat, values(dict[k]))...)
    end
    return ls
end

n_a = dict_of_dicts_to_arr(n_a_all);
d_goal = dict_of_dicts_to_arr(d_goal_all);
white_size = dict_of_dicts_to_arr(white_size_all);
red_size = dict_of_dicts_to_arr(red_size_all);

histogram2d(d_goal, white_size, bins=((1:maximum(d_goal)) .+ 0.5, (2:maximum(white_size)) .+ 0.5), ylim=(1, 20), xlim=(0, 35), xlabel=latexstring("d_\\textrm{goal}"), ylabel="AND-OR tree size (white nodes)", size=(400,300), dpi=300, xticks=[0, 10, 20, 30], grid=false, colorbar_title="\nHistogram frequency", right_margin = 5Plots.mm)

board = load_data("example_and_or")

actions, trials, visited, timeline, draw_tree_nodes = and_or_tree(board)
g = draw_tree(draw_tree_nodes)
probability_over_moves(trials, actions, 0.1, 1.0, 0.0, 0.0)


states_all, trials_all, actions_all, parents_all, move_parents_all, h_all, moves_all, opt_all, d_goal_all, n_a_all, white_size_all, red_size_all = get_and_or_tree_data(data);
fitness7, params7 = analyze_all_trees(trials_all, actions_all, parents_all, move_parents_all, h_all, moves_all, opt_all, d_goal_all);
error_l_model1, error_a_model1, error_l_model, error_a_model, error_l_data, error_a_data = get_errors(params3, params, states_all, trials_all, actions_all, moves_all, opt_all, d_goal_all, n_a_all);

begin
error_l_model_qb1 = quantile_binning(error_l_model);
error_a_model_qb1 = quantile_binning(error_a_model);
mu_hat_bar_l_model1, error_bar_l_model1 = get_errorbar(error_l_model_qb1);
mu_hat_bar_a_model1, error_bar_a_model1 = get_errorbar(error_a_model_qb1);

error_l_model_qb2 = quantile_binning(error_l_model1);
error_a_model_qb2 = quantile_binning(error_a_model1);
mu_hat_bar_l_model2, error_bar_l_model2 = get_errorbar(error_l_model_qb2);
mu_hat_bar_a_model2, error_bar_a_model2 = get_errorbar(error_a_model_qb2);

error_l_data_qb = quantile_binning(error_l_data);
error_a_data_qb = quantile_binning(error_a_data);
mu_hat_bar_l_data, error_bar_l_data = get_errorbar(error_l_data_qb);
mu_hat_bar_a_data, error_bar_a_data = get_errorbar(error_a_data_qb);

plot(layout=grid(2, 1, heights=[0.55, 0.45]), size=(900, 900), legend=:outertop, ylim=(0, 1), grid=false, background_color_legend = nothing, foreground_color_legend = nothing, dpi=300, legend_columns=3, 
legendfont=font(18), 
xtickfont=font(16), 
ytickfont=font(16), 
guidefont=font(32))
plot!(av_bin_l, mu_hat_bar_l_model1, alpha=1.0, ribbon=2*error_bar_l_model1, sp=1, c=palette(:default)[2], lw=0.0, label=nothing)
plot!(av_bin_l, mu_hat_bar_l_model1, alpha=1.0, ribbon=2*error_bar_l_model1, sp=1, c=palette(:default)[2], lw=0.0, label=nothing)
plot!([], [], sp=1, c=palette(:default)[2], label="Eureka model", lw=10)
plot!(av_bin_a, mu_hat_bar_a_model1, ribbon=2*error_bar_a_model1, sp=2, c=palette(:default)[2], lw=0.0, label=nothing)
plot!(av_bin_a, mu_hat_bar_a_model1, ribbon=2*error_bar_a_model1, sp=2, c=palette(:default)[2], lw=0.0, label=nothing)
plot!([], [], sp=2, c=palette(:default)[2], label=nothing, lw=10)

plot!(av_bin_l, mu_hat_bar_l_model2, alpha=1.0, ribbon=2*error_bar_l_model2, sp=1, c=palette(:default)[3], lw=0.0, label=nothing)
plot!(av_bin_l, mu_hat_bar_l_model2, alpha=1.0, ribbon=2*error_bar_l_model2, sp=1, c=palette(:default)[3], lw=0.0, label=nothing)
plot!([], [], sp=1, c=palette(:default)[3], label="AND-OR model", lw=10)
plot!(av_bin_a, mu_hat_bar_a_model2, ribbon=2*error_bar_a_model2, sp=2, c=palette(:default)[3], lw=0.0, label=nothing)
plot!(av_bin_a, mu_hat_bar_a_model2, ribbon=2*error_bar_a_model2, sp=2, c=palette(:default)[3], lw=0.0, label=nothing)
plot!([], [], sp=2, c=palette(:default)[3], label=nothing, lw=10)

plot!([], [], yerr=[], sp=1, c=:black, label="Data")
plot!(av_bin_l, mu_hat_bar_l_data, bottom_margin=-4Plots.mm, yerr=2*error_bar_l_data, sp=1, ms=10, l=nothing, c=:black, markershape=:none, label=nothing, xlabel=latexstring("d_\\textrm{goal}"), ylabel=latexstring("p(\\textrm{error})"), xticks=round.(av_bin_l, digits=1))
plot!(av_bin_l, mu_hat_bar_l_data, yerr=2*error_bar_l_data, sp=1, c=:transparent, msw=1, label=nothing)
plot!(av_bin_a, mu_hat_bar_a_data, yerr=2*error_bar_a_data, sp=2, ms=10, l=nothing, c=:black, label=nothing, xlabel=latexstring("n_A"), ylabel=latexstring("p(\\textrm{error})"), xticks=round.(av_bin_a, digits=1))
plot!(av_bin_a, mu_hat_bar_a_data, yerr=2*error_bar_a_data, sp=2, c=:transparent, msw=1, label=nothing)
end

actions, trials, visited, timeline, draw_tree_nodes = and_or_tree(board);
g = draw_tree(draw_tree_nodes)

p = dfs_rollout_probability(trials, 0.5, 0.5)
a = wsample(actions, p)
make_move!(board, a)
arr = get_board_arr(board)
draw_board(arr)

plot(layout=(2, 1), size=(800, 800), dpi=300, xticks=1:length(trials), grid=false, foreground_color_legend=:white, legend_columns=3,
legendfont=font(18), xtickfont=font(16), ytickfont=font(16), guidefont=font(32), link=:x)
bar!(trials, sp=1, ylabel="Expansions", yticks=[0, 5, 10], label=nothing)
gammas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
cs = palette(:matter, 8)
λ = 0.0
for (n, γ) in enumerate(gammas)
    plot!(dfs_rollout_probability(trials, γ), sp=2, label=γ, xlabel="Move index", c=cs[n+1], lw=3, legendtitle=latexstring("\\gamma"), ylabel=latexstring("p(\\textrm{move})"), legendtitlefontsize=32)
end
plot!()


board = load_data(prbs[1])
arr = get_board_arr(board)
draw_board(arr)
prev_move = (-1, 0)
action_traces = zeros(length(unique(arr))-1, 9)
τ = 0.8
β = 10
κ = 0.9
C = 0.5
γ = 0.1
λ = 0.0

arrs = [arr]

for i in 1:1000
    all_actions = get_all_available_moves(board, arr)
    actions, visited, draw_tree_nodes = and_or_tree(board, arr, γ=γ)
    if length(actions) == 0 || rand() < λ
        actions = all_actions
    end
    #g = draw_tree(draw_tree_nodes)
    counter = 1
    as = [action_traces[[a[1], a[2]+5]...] for a in actions]
    p = exp.(-β*as) / sum(exp.(-β*as))
    # display("===$(i)===")
    # display(action_traces)
    # display(actions)
    # display(p)

    move = wsample(actions, p)
    action_traces .*= τ
    action_traces[move[1], 5+move[2]] = 1
    action_traces[move[1], 5-move[2]] += κ*(1-action_traces[move[1], 5-move[2]])
    action_traces[move[1], :] .+= C*(1 .- action_traces[move[1], :])

    make_move!(board, move)
    arr = get_board_arr(board)
    push!(arrs, arr)
    #display(plot!(title=i))

    prev_move = move
end

anim = @animate for i ∈ 1:300
    draw_board(arrs[i])
    plot!(title=i-1, size=(200,230), dpi=300)
end
gif(anim, "anim_fps15.gif", fps = 40)



