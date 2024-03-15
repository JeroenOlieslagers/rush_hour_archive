
n = 9
m = 2
states = visited_states[subjs[n]][prbs[m]][1][1:end-1]
moves = moves_all[subjs[n]][prbs[m]]

for (i, ss) in enumerate(states)
    move_icon = "m"*create_move_icon(moves[i], board)
    arr = int_to_arr(ss)
    board = arr_to_board(arr)
    actions, trials, visited, timeline, draw_tree_nodes = and_or_tree(board)
    g = draw_tree(draw_tree_nodes, highlight_node=move_icon)
    if i < 10
        nn = "00"*string(i)
    elseif i < 100
        nn = "0"*string(i)
    else
        nn = string(i)
    end
    FileIO.save("subj"*string(n)*"_prb"*string(m)*"_"*nn*".svg", g)
    draw_board(arr)
    savefig("subj"*string(n)*"_prb"*string(m)*"_"*nn*".png")
end

function many_move_hist(prbs)
    N = length(prbs)
    plot(layout=grid(N, 6, widths=[0.7/3, 0.3/3, 0.7/3, 0.3/3, 0.7/3, 0.3/3]), size=(1800,250*N), grid=false, dpi=300, legend=false)
    for (n, prb) in enumerate(prbs)
        board = load_data(prb)
        arr = get_board_arr(board)
        movez = []
        moves = []
        for i in 1:42
            if prb in keys(moves_all[subjs[i]])
                mm = moves_all[subjs[i]][prb][1]
                push!(movez, create_move_icon(mm, board))
                push!(moves, mm)
            end
        end
        hist = sort(countmap(movez))
        k = collect(keys(hist))
        v = collect(values(hist))
        bar!(k, v, sp=1 + (n-1)*6, label=false, left_margin=1.1*N*Plots.mm)
        draw_board(arr, sp=2 + (n-1)*6)
        plot!(sp=2 + (n-1)*6, legend=false, xticks=[], yticks=[], title="prb $(n)")

        hist = sort(countmap(moves))
        k = collect(keys(hist))
        v = collect(values(hist))
        prev_move = k[argmax(v)]
        movez = []
        moves = []
        for i in 1:42
            if prb in keys(moves_all[subjs[i]])
                mm = moves_all[subjs[i]][prb]
                if mm[1] == prev_move
                    push!(movez, create_move_icon(mm[2], board))
                    push!(moves, mm[2])
                end
            end
        end
        hist = sort(countmap(movez))
        k = collect(keys(hist))
        v = collect(values(hist))
        bar!(k, v, sp=3 + (n-1)*6, label=false, left_margin=1.1*N*Plots.mm, yticks=[])
        make_move!(board, prev_move)
        arr = get_board_arr(board)
        draw_board(arr, sp=4 + (n-1)*6)
        plot!(sp=4 + (n-1)*6, legend=false, xticks=[], yticks=[], title="prb $(n)")

        hist = sort(countmap(moves))
        k = collect(keys(hist))
        v = collect(values(hist))
        prev_prev_move = k[argmax(v)]
        movez = []
        moves = []
        for i in 1:42
            if prb in keys(moves_all[subjs[i]])
                mm = moves_all[subjs[i]][prb]
                if mm[1] == prev_move && mm[2] == prev_prev_move
                    push!(movez, create_move_icon(mm[3], board))
                    push!(moves, mm[3])
                end
            end
        end
        hist = sort(countmap(movez))
        k = collect(keys(hist))
        v = collect(values(hist))
        bar!(k, v, sp=5 + (n-1)*6, label=false, left_margin=1.1*N*Plots.mm, yticks=[])
        make_move!(board, prev_prev_move)
        arr = get_board_arr(board)
        draw_board(arr, sp=6 + (n-1)*6)
        plot!(sp=6 + (n-1)*6, legend=false, xticks=[], yticks=[], title="prb $(n)")
    end
    display(plot!())
end

function move_hist(p1, p2, ps_prev, actions, move, opt, s, subind, inds)
    N = length(subind)
    plot(layout=grid(N, 2, widths=[0.7, 0.3]), size=(600,200*N), grid=false, dpi=300, legend=false, foreground_color_legend = nothing, yticks=[])
    for i in 1:N
        arr = int_to_arr(s[inds[subind[i]]])
        board = arr_to_board(arr)
        movez = []
        optz = []
        for mm in actions[inds[subind[i]]]
            push!(movez, create_move_icon(mm, board))
        end
        for mm in opt[inds[subind[i]]]
            push!(optz, create_move_icon(mm, board))
        end
        if length(p1) > 0
            bar!(movez, p1[inds[subind[i]]], sp=1 + (i-1)*2, alpha=0.5, label="AND/OR", legend=true, xticks=:all)
        end
        if length(p2) > 0
            bar!(movez, p2[inds[subind[i]]], sp=1 + (i-1)*2, alpha=0.5, label="AND/OR+", legend=true)
        end
        bar!(optz, ones(length(optz)) ./ length(optz), alpha=0.5, sp=1 + (i-1)*2, label="Optimal", bar_width=0.5, legend=true)
        bar!([create_move_icon(move[inds[subind[i]]], board)], [1], sp=1 + (i-1)*2, label="Human", bar_width=0.2, c=:black, legend=true)
        if length(ps_prev) > 0
            bar!(movez, ps_prev[inds[subind[i]]], sp=1 + (i-1)*2, alpha=0.2, label="Same car", legend=true)
        end
        draw_board(arr, sp=2 + (i-1)*2)
        plot!(sp=2 + (i-1)*2, legend=false, xticks=[], yticks=[])
    end
    display(plot!())
end

ps, pps, ppps, pppps, acs, blcks, ungs, diffns, ops, op_inds, movs, mov_parents, mov_inds, js, ss, all_prbs, all_subjs = get_errors(paramss, params3, states_all, trials_all, actions_all, action_lengths_all, blockages_all, ungreen_all, diff_nodes_all, ac_count_all, parents_all, move_parents_all, h_all, moves_all, opt_all, d_goal_all, n_a_all, ds=dss[5]);


indss = reverse(sortperm(js))
ind = inds[1]
extra_plan = [1, 3, 7, 8, 10, 18]
move_9 = [2, 4]
good = [9, 11, 12]
bad = [15, 20]

move_hist(ps, pps, acs, movs, ops, ss, [1, 2, 3, 4, 5], indss)


ll1, ll2, ll3, cor1, cor2 = 0, 0, 0, 0, 0
for i in eachindex(ps)
    N = length(ps[i])
    ll1 += ps[i][mov_inds[i]]
    ll2 += pps[i][mov_inds[i]]
    ll3 += 1/N
    if wsample(1:N, ps[i]) in op_inds[i]
        if wsample(1:N, pps[i]) ∉ op_inds[i]
            cor2 += 1
        end
    end
end


# many_move_hist(prbs[1:18]);
# many_move_hist(prbs[19:36]);
# many_move_hist(prbs[37:53]);
# many_move_hist(prbs[54:end]);

