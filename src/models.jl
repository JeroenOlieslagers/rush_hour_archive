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

function forward_search(params, row, d_goals, F, state_to_idx)
    log_gamma, k = params
    return p_a([k], row.neighs, F, state_to_idx)[1]
end