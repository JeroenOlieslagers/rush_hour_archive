include("engine.jl")
include("solvers.jl")
include("plot_graph.jl")
using Plots
using Colors


PROBLEM_DIR = "data/problems"
jsons = readdir(PROBLEM_DIR)[1:5084];

function heuristic_analysis(jsons, heurs)
    hs = [Array{Array{Int, 1}, 1}() for _ in 1:6]
    for (n, js) in enumerate(jsons)
        board = load_data(PROBLEM_DIR*"/"*js);
        arr, moves, exp = a_star(board);
        for (m, heur) in enumerate(heurs)
            push!(hs[m], calculate_heur(board, moves, heur))
        end
        if length(hs[6][end]) > 5
            if hs[6][end][end-5] > 5
                println(js)
            end
        end
        if n % 100 == 0
            println(n)
        end
    end
    return hs
end

function plot_heurs(hs, heurs)
    ps = []
    for (n, heur) in enumerate(heurs)
        if n == 6
            p = plot(legend=false, title=string(heur), grid=false, left_margin = 10Plots.mm, ylabel="Heuristic value", xlabel="Moves from goal")
        else
            p = plot(legend=false, title=string(heur), grid=false, left_margin = 10Plots.mm, ylabel="Heuristic value")
        end
        push!(ps, p)
        for h in hs[n]
            plot!(-length(h)+1:0, h, linealpha=0.05, color="black", legend = false)
        end
    end
    plot(ps..., layout = (6, 1), size=(1000,1500))
end

function solution_analysis(jsons)
    heurs = [zer, red_distance, red_pos, calculate_blocked_moves, mag_size_layers, mag_size_nodes]
    ts = [Array{Float64, 1}() for _ in 1:6]
    exps = [Array{Int, 1}() for _ in 1:6]
    for (n, js) in enumerate(jsons)
        for (m, heur) in enumerate(heurs)
            board = load_data(PROBLEM_DIR*"/"*js);
            data = @timed a_star(board, heur);
            push!(ts[m], data.time)
            arr, moves, exp = data.value
            push!(exps[m], exp)
        end
        if n % 10 == 0
            println(n)
        end
    end
    return ts, exps
end

for i in 1:length(arrs)
    anim = @animate for arr ∈ arrs[i]
        draw_board(arr)
    end
    gif(anim, "anim_" * string(i) * ".gif", fps = 1)
end

#ts, exps = solution_analysis(jsons)

heurs = [red_distance, calculate_blocked_moves, mag_size_layers, mag_size_nodes, multi_mag_size_layers, multi_mag_size_nodes]

hs = heuristic_analysis(jsons, heurs);

plot_heurs(hs, heurs)

savefig("multi_mag_heuristic_analysis.png")

heur = hs[5]

for (n, h) in enumerate(heur)
    if length(h) > 7
        if h[end-6] > 7
            println(h, n)
        end
    end
end

arrs = []
for move in pm
    make_move!(board, move)
    push!(arrs, get_board_arr(board))
end
anim = @animate for arr ∈ arrs
    draw_board(arr)
end
gif(anim, "hard_puzzle_40_optimal.gif", fps = 5)
