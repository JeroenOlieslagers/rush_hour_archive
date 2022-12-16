include("engine.jl")
include("solvers.jl")
include("data_analysis.jl")


"""
    softmax(β, a, A)

``\\frac{e^{βa}}{\\sum_{a_i∈A}e^{βa_i}}``
"""
function softmax(β, a, A)
    return exp(β*a)/sum(exp.(β*A))
end

function log_lik(β, λ, tot_move_tuples, full_heur_dict, heur=4)
    qs = []
    Qs = []
    for prb in keys(tot_move_tuples)
        board = load_data(prb)
        arr = get_board_arr(board)
        for move in tot_move_tuples[prb]
            if move == (-1, 0)
                board = load_data(prb)
                arr = get_board_arr(board)
                continue
            end
            A = get_all_available_moves(board, arr)
            Q = zeros(length(A))

            for (n, a) in enumerate(A)
                make_move!(board, a)
                arr_a = get_board_arr(board)
                s = board_to_int(arr_a, BigInt)
                if !(string(s) in keys(full_heur_dict))
                    undo_moves!(board, [a])
                    break
                end
                Q[n] = full_heur_dict[string(s)][heur]
                undo_moves!(board, [a])
            end

            make_move!(board, move)
            arr = get_board_arr(board)
            s = board_to_int(arr, BigInt)
            if string(s) in keys(full_heur_dict)
                push!(qs, full_heur_dict[string(s)][heur])
                push!(Qs, Q)
                #res += log(λ/length(A) +(1-λ)*softmax(β, -q, -Q))
            end
        end
    end
    return qs, Qs
end

data = load("data/processed_data/filtered_data.jld2")["data"];
subjs = collect(keys(data));

full_heur_dict = load("data/processed_data/full_heur_dict.jld2")

qqs = []
QQs = []
for (m, subj) in enumerate(subjs)
    tot_arrs, tot_move_tuples, tot_states_visited, attempts = analyse_subject(data[subj]);
    qs, Qs = log_lik(0.1, 0.0, tot_move_tuples, full_heur_dict)
    push!(qqs, qs)
    push!(QQs, Qs)
end


N = 100

mll = zeros(length(subjs))
bts = zeros(length(subjs))
lmds = zeros(length(subjs))

betas = 10 .^(range(-0.3,stop=0.7,length=N))
lambdas = range(0,stop=0.5,length=N)

for (m, subj) in ProgressBar(enumerate(subjs))
    qs = qqs[m]
    Qs = QQs[m]
    probs = zeros(N, N)
    for i in eachindex(betas)
        for j in eachindex(lambdas)
            for k in eachindex(Qs)
                probs[i, j] += log(lambdas[j]/length(Qs[k]) + (1 - lambdas[j])*softmax(betas[i], -qs[k], -Qs[k]))
            end
        end
    end
    ma = argmax(probs)
    mll[m] = probs[ma]
    bts[m] = betas[ma[1]]
    lmds[m] = lambdas[ma[2]]
    #plot!(lambdas, probs[m, :], c=:black, label=nothing)
end

plot()
names = ["red_dist", "mag_nodes", "min_forest_nodes", "mean_leave_one_out",  "rand_leave_one_out", "max_leave_one_out"]
for i in 1:6
    if i == 5
        continue
    end
    m = argmax(probs[i, :])
    plot!([betas[m], betas[m]], [-120000, maximum(probs[i, :])], label=nothing, c=:black)
    plot!(betas, probs[i, :], xscale=:log10, label=names[i], legend=:bottomleft)
end
plot!(xlabel=latexstring("\\beta"), ylabel="Log likelihood")

subj = "ACN0C9HPYDTED:36DSNE9QZ66UIWW6W1JAY7NSUULJOD";
prb = "prb10206_7";

tot_arrs, tot_move_tuples, tot_states_visited, attempts = analyse_subject(data[subj]);
movess = tot_move_tuples[prb]
board = load_data(prb)


for move in movess[2:end]
    make_move!(board, move)
    arr = get_board_arr(board)
    display(arr)
end