function simulate_game(params, board; max_iters=1000)
    γ = params
    states = []
    move = [0, 0]
    for iter in 1:max_iters
        arr = get_board_arr(board)
        s = board_to_int(arr, BigInt)
        push!(states, s)
        if check_solved(arr)
            return length(states)
        end
        all_moves, AND_OR_tree = get_and_or_tree(board)
        _, AND, OR, idv_AND, idv_OR, parents_moves, _, _ = AND_OR_tree
        dict = propagate_ps(zeros(8), AND_OR_tree)

        new_dict = only_gamma_dict(dict, γ)
        same_car_moves = iter > 1 ? [(move[1], j) for j in -4:4] : [(0, 0)]

        ps, dropout, N = process_dict(all_moves, new_dict, same_car_moves, 0.0, 1.0, board)
        move = wsample(collect(all_moves), Float64.(ps))
        make_move!(board, move)
    end
    return max_iters
    throw(DomainError("Iteration limit reached"))
end

γ = 0.000001
nns = []
for prb in ProgressBar(prbs)
    board = load_data(prb)
    push!(nns, simulate_game(γ, board))
end

nnss = [[] for _ in 1:70]
for subj in subjs
    for prb in keys(all_subj_moves[subj])
        push!(nnss[findfirst(x->x==prb, prbs)], length(all_subj_moves[subj][prb]))
    end
end

for nn in ns
    nn[findall(x->x==10000, nn)] .= 1000
end

plot([median(nn) for nn in ns], label="AND/OR Model", grid=false, xlabel="Puzzle index (higher is harder)", ylabel="Moves to solve puzzle")
plot!([median(nn) for nn in nnss], label="Subjects")

# for prb in ProgressBar(prbs)
#     dummy = []
#     for _ in 1:10
#         board = load_data(prb)
#         push!(dummy, simulate_game(γ, board))
#     end
#     push!(ns, dummy)
# end