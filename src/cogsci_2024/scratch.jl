
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

for (n, state) in enumerate(all_states_list[1:end-1])
    board = arr_to_board(int_to_arr(state))
    _, and, or, po = get_and_or_tree(board)[2][1:4]
    AO3 = (and, or)
    highlight = reduce(vcat, [po[move] for move in all_states[state] if move in keys(po)])
    outside_moves = unique([move for move in all_states[state] if move ∉ keys(po)])
    heatmap_moves = countmap(all_states[state])
    g = draw_ao_tree(AO3, board; highlight_ORs=highlight, outside_moves=outside_moves, heatmap_moves=heatmap_moves)
    #println(outside_moves)
    draw_board(get_board_arr(board))
    # savefig("board_"*string(n)*".svg")
    # display(g)
    # save_graph(g, "frame_"*string(n))
end

ds = []
for prb in [prbs[15], prbs[20], prbs[48], prbs[69]]
#for prb in [prbs[15], prbs[18], prbs[41], prbs[69]]
    dds = []
    for subj in subjs
        dd = []
        if prb in keys(all_subj_states[subj])
            for state in all_subj_states[subj][prb]
                push!(dd, d_goals[state])
                if d_goals[state] == 0
                    break
                end
            end
            push!(dds, dd)
        end
    end
    push!(ds, dds)
end

plot(size=(350, 450), layout=grid(2, 2), grid=false, dpi=300, xflip=false,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(12, "helvetica"), 
        ytickfont=font(12, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=3Plots.mm, top_margin=0Plots.mm, bottom_margin=3Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, link=:y)
xticks = [[0, 5, 10, 15, 20], [0, 10, 20, 30], [0, 20, 40, 60, 80], [0, 50, 100, 150]]
yticks = [[0, 5, 10, 15], [], [0, 5, 10, 15], []]
titles = ["Length 5", "Length 9", "Length 12", "Length 14"]
ylabels = ["Distance to goal" "" "Distance to goal" ""]
xlabels = ["" "" "Move number" "Move number"]
for i in 1:4
    for d in ds[i]
        plot!(d, label=nothing, sp=i, c=:black, alpha=0.2, xticks=xticks[i], yticks=yticks[i], xlim=(0, Inf), ylim=(0, 17), ylabel=ylabels[i], xlabel=xlabels[i], title=titles[i])
    end
end
plot!()

board = load_data(prbs[41])


subj_prbs = []
for subj in subjs
    push!(subj_prbs, collect(keys(all_subj_states[subj])))
end

prb_ds = Dict{String, Vector{Int}}()
for prb in ProgressBar(prbs)
    aa = zeros(Int, length(a[prb]))
    for (n, s) in enumerate(keys(a[prb]))
        aa[n] = d_goals[s]
    end
    prb_ds[prb] = aa
end

hist_all = [[] for _ in 1:20];
hist_opt = [[] for _ in 1:20];
for aa in subj_prbs
    T = 0
    To = 0
    for i in 1:20
        push!(hist_all[i], 0)
        push!(hist_opt[i], 0)
    end
    for prb in aa
        for i in 1:19
            L = length(findall(x->x==i, prb_ds[prb]))
            T += L
            hist_all[i][end] += L
        end
        L = length(findall(x->x>=20, prb_ds[prb]))
        T += L
        hist_all[20][end] += L
        for i in 1:parse(Int, split(prb, "_")[end])
            To += 1
            hist_opt[i][end] += 1
        end
    end
    for i in 1:20
        hist_all[i][end] /= T
        hist_opt[i][end] /= To
    end
end

plot(1:20, mean.(hist_all), ribbon=2*std.(hist_all) / sqrt(42), label=nothing, c=palette(:default)[1], l=nothing)

histogram(all_ds[all_ds .> 0], bins=20, normalize=true, alpha=0.5)
bar!(mean.(hist_all), alpha=0.5)
plot!(collect(1:20), mean.(hist_opt), ribbon=std.(hist_opt)/sqrt(42), c=:red, label="Optimal", linetype=:steppost)

new_features = Dict{String, Vector{Matrix}}()
for subj in ProgressBar(subjs)
    nf = []
    for i in eachindex(boards[subj])
        board = boards[subj][i]
        moves = tree_datas[subj][3][i]
        fs = features[subj][i]
        ls = zeros(length(moves))
        for (n, move) in enumerate(moves)
            make_move!(board, move)
            _, tree = get_and_or_tree(board)
            ls[n] = length(tree[2]) + length(tree[3])
            undo_moves!(board, [move])
        end
        push!(nf, hcat(fs, ls))
    end
    new_features[subj] = nf
end

ls = []
for subj in subjs
    for f in new_features[subj]
        push!(ls, f[:, 3]...)
    end
end

using JSON
for prb in prbs
    d = JSON.parsefile("data/raw_data/problems/" * prb * ".json")
    open("prbs/"*prb*".json", "w") do f
        write(f, JSON.json(d))
     end
    #break
end

dd = JSON.parsefile("test.json")


numbers = 0:79
for number in numbers
    println("Cluster id: $(number), animal: $(number ÷ 20), iteration: $(number % 20)")
end


V = initialize_value_map(S)
for (m, n) in outer_loop #loop in order
    subsets = randomly_split(S[(m, n)])
    Threads.@threads for subset in subsets
        while !converged
            for s in subset
                V[s] = update_value(s, V)
            end
        end
    end
end

using BenchmarkTools
using ProgressBars
a=collect(UInt32, 1:275784032)
V = Dict{UInt32, Float64}()
for b in ProgressBar(a)
    V[b] = 0.0
end

for b in ProgressBar(a)
    V[b] = rand() + (V[b]+1)
end
