
tree_sizes = []
dg = []

for subj in subjs
    for i in eachindex(states[subj])
        _, and, or, _  = tree_datas[subj][1][i]
        if length(and) + length(or) > 40
            continue
        end
        if d_goals[states[subj][i]] < 10
            continue
        end
        if d_goals[states[subj][i]] > 15
            continue
        end
        push!(tree_sizes, length(and) + length(or))
        push!(dg, d_goals[states[subj][i]])
        if length(tree_sizes) == 4484#45038
            println(subj)
            println(i)
        end
    end
end

subj = "A18T3WK7J16C1B:3IJXV6UZ1YR1KY4G6BFEG1DXR47RIC"
ii = 384

d_goals[states[subj][ii]]
_, and, or, po  = tree_datas[subj][1][ii+7]
subj_moves = tree_datas[subj][4][ii+7:ii+20]
AO3 = (and, or)
board3 = deepcopy(boards[subj][ii+7])
draw_board(get_board_arr(board3))
opt_move = tree_datas[subj][3][ii][4]
highlight = po[opt_move]
g = draw_ao_tree(AO3, board3; highlight_ORs=highlight)

board3 = load_data(prb)
move = (9, 3)
make_move!(board3, move)
_, and, or, po = get_and_or_tree(board3)[2][1:4]
AO3 = (and, or)
highlight = po[move]
g = draw_ao_tree(AO3, board3; highlight_ORs=highlight)


prb = prbs[41]
all_moves = []
for subj in subjs
    if prb in keys(all_subj_moves[subj])
        push!(all_moves, all_subj_moves[subj][prb])
    end
end

board3 = load_data(prb)
all_moves_list = all_moves[end-2][2:end]
all_states_list = [board_to_int(get_board_arr(board3), BigInt)]
for m in all_moves_list
    make_move!(board3, m)
    push!(all_states_list, board_to_int(get_board_arr(board3), BigInt))
end
all_states = Dict(all_states_list[1:end-1] .=> [[] for _ in 1:12])
for i in eachindex(all_moves)
    for m in all_moves[i]
        if m == (-1, 0)
            board3 = load_data(prb)
            continue
        end
        si = board_to_int(get_board_arr(board3), BigInt)
        if si in keys(all_states)
            push!(all_states[si], m)
        end
        make_move!(board3, m)
    end
end

for state in all_states_list[1:end-1]
    board = arr_to_board(int_to_arr(state))
    _, and, or, po = get_and_or_tree(board)[2][1:4]
    AO3 = (and, or)
    highlight = reduce(vcat, [po[move] for move in all_states[state]])
    g = draw_ao_tree(AO3, board; highlight_ORs=highlight)
    draw_board(get_board_arr(board))
    display(g)
    break
end