function random_model(params, row, d_goals)
    N = length(row.neighs)
    return ones(N) / N
end

function optimal_model(params, row, d_goals)
    N = length(row.neighs)
    ps = zeros(N)
    opt_idx = [d_goals[row.puzzle][s_p] < row.d_goal for s_p in row.neighs]
    ps[opt_idx] .= 1/sum(opt_idx)
    return ps
end

function hill_climbing_model(params, row, d_goals)
    b0, b1, b2, β = params
    #b0, b1, b2, b3, β = params
    f1 = row.features[:, 1]
    f2 = row.features[:, 2]
    #f3 = fs[:, 3]
    ex = exp.((b0 .+ (b1 .* f1) .+ (b2 .* f2)))# ./ β)
    #ex = exp.((b0 .+ (b1 .* f1) .+ (b2 .* f2) .+ (b3 .* f3)) ./ β)
    return ex ./ sum(ex)
end

function forward_search(params, row, d_goals)
    function rollout(s_free::s_free_type, s_fixed::s_fixed_type, γ::Float64; max_iter=10000)
        poss_moves = @MVector([(Int8(0), Int8(0)) for _ in 1:(4*L)])
        first_move = nothing
        prev_move = (Int8(0), Int8(0))
        for n in 1:max_iter
            if rand() > γ || n == 1
                s = (s_free, s_fixed)
                arr = board_to_arr(s)
                # Check if complete
                if check_solved(s)
                    return true, first_move
                end
                # Expand current node by getting all available moves
                possible_moves!(poss_moves, s, arr)
                N_moves = 0
                for move in poss_moves
                    if move[1] == 0
                        break
                    end
                    N_moves += 1
                end
                #available_moves = get_all_available_moves(board, arr)
                # filter!(e -> e[1] != prev_move[1], poss_moves)
                # if isempty(available_moves)
                #     available_moves = get_all_available_moves(board, arr)
                # end
                # Randomly choose a move
                selected_move_idx = rand(1:N_moves)
                selected_move = poss_moves[selected_move_idx]
                for i in 1:1000
                    if N_moves > 1 && selected_move == prev_move
                        selected_move_idx = rand(1:N_moves)
                        selected_move = poss_moves[selected_move_idx]
                    else
                        break
                    end
                end
                # Make move
                make_move!(s_free, selected_move)
                if n == 1
                    first_move = selected_move
                end
                prev_move = selected_move#[selected_move[1], -selected_move[2]]
            else
                return false, first_move
            end
        end
        return false, first_move
    end
    logγ, k = params
    γ = exp(-logγ)
    k = round(Int, k)
    N = length(row.neighs)
    ps = zeros(N)
    N_REPEATS = 1000
    s_free_init = row.s_free
    s_fixed = row.s_fixed
    for n in 1:N_REPEATS
        f_move = nothing
        for i in 1:k
            solved, first_move = rollout(copy(s_free_init), s_fixed, γ)
            if solved
                f_move = first_move
                break
            end
        end
        if f_move === nothing
            ps .+= 1/N
        else
            move_index = findfirst(x->x==Tuple(f_move), row.all_moves)
            ps[move_index] += 1
        end
    end
    return ps ./= N_REPEATS
end
# ff = (x) -> forward_search(x, tree_datas[subjs[3]][1][138], tree_datas[subjs[3]][2][138],  tree_datas[subjs[3]][3][138], boards[subjs[3]][138], prev_moves[subjs[3]][138], neighs[subjs[3]][138], 3, features[subjs[3]][138], d_goals)[findfirst(x->x==tree_datas[subjs[3]][4][138], tree_datas[subjs[3]][3][138])]
# a=[[ff([ii, jj]) for ii in 1:0.5:4] for jj in 100:100:2000]
# ff([1.0, 1000])
# options = Dict("tolfun"=> 0.01, "max_fun_evals"=>100, "display"=>"iter");
# bads = BADS(ff, [2.0, 100.0], [0.0, 0.0], [10.0, 2000.0], [1.0, 50.0], [5.0, 1000.0], options=options)
# ress = bads.optimize();

function gamma_only_model(params, row, d_goals)
    γ = params
    same_car_moves = row.prev_move == (0, 0) ? [(Int8(0), Int8(0))] : [(row.prev_move[1], Int8(j)) for j in -4:4]
    # modulate depths by (1-γ)^d
    new_dict = apply_gamma(row.dict, γ)
    # turn dict into probability over actions
    ps = process_dict(row.all_moves, new_dict, same_car_moves)
    return ps
end

function gamma_0_model(params, row, d_goals)
    λ = params
    same_car_moves = row.prev_move == (0, 0) ? [(Int8(0), Int8(0))] : [(row.prev_move[1], Int8(j)) for j in -4:4]
    # turn dict into probability over actions
    ps = process_dict(row.all_moves, row.dict, same_car_moves)
    ps = λ/length(ps) .+ (1-λ) * ps
    return ps
end

function gamma_no_same_model(params, row, d_goals)
    γ = params
    same_car_moves = [(Int8(0), Int8(0))]
    # modulate depths by (1-γ)^d
    new_dict = apply_gamma(row.dict, γ)
    # turn dict into probability over actions
    ps = process_dict(row.all_moves, new_dict, same_car_moves)
    return ps
end

function eureka_model(params, row, d_goals)
    d, λ = params
    N = length(row.neighs)
    ps = zeros(N)
    if row.d_goal <= d
        opt_idx = [d_goals[row.puzzle][s_p] < row.d_goal for s_p in row.neighs]
        ps[opt_idx] .= 1/sum(opt_idx)
    else
        ps = ones(N) ./ N
    end
    ps = λ/N .+ (1-λ) * ps
    return ps
end

function opt_rand_model(params, row, d_goals)
    λ = params
    N = length(row.neighs)
    ps = zeros(N)
    opt_idx = [d_goals[row.puzzle][s_p] < row.d_goal for s_p in row.neighs]
    ps[opt_idx] .= 1/sum(opt_idx)
    ps = λ/N .+ (1-λ) * ps
    return ps
end