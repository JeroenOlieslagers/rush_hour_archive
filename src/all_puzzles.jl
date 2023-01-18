include("engine.jl")

"""
    board = line_to_board(line)

Takes in a string where capital letters are cars and o are empty spaces. A is player
"""
function line_to_board(line)
    cars = Car[]
    for (n, char) in enumerate(reverse(sort(unique(line)))[2:end])
        index = findfirst(char, line)-1
        len = count(i->(i==char), line)
        is_horizontal = line[index+2] == char
        push!(cars, Car((index%6)+1, (index÷6)+1, len, n, is_horizontal))
    end
    return Board(cars, "")
end

#board = line_to_board("FoHBBBFoHIoKGAAIoKGCCJoLoooJoLDDEEEL") # SHORTEST
#board = line_to_board("ooJBBMCCJoLMoIAALMHIoKDDHEEKoooFFGGo") # LONGEST
board = line_to_board("GooJBBGoIJCCHoIAALHoooKLHDDDKMEEEFFM") # MEDIUM 800



lines = readlines("analysis/processed_data/no_wall.txt")
opts = zeros(length(lines))
sss = zeros(length(lines))
state_space_features = Dict{BigInt, Vector{Float64}}()
shuffled_lines = shuffle(lines)
for i in ProgressBar(1:length(lines))
    opt_len, line, full_ss = split(lines[i], " ")
    board = line_to_board(line)
    arr = get_board_arr(board)
    prb = board_to_int(arr, BigInt)
    state_space_features[prb] = get_state_space_features(board)
    opts[i] = parse(Int, opt_len)
    sss[i] = parse(Int, full_ss)
end

prbs = collect(keys(state_space_features))
X = zeros(length(prbs), 10)
for (i, prb) in enumerate(prbs)
    for j in 1:10
        X[i, j] = state_space_features[prb][j]
    end
end
XX = zeros(size(X))
for i in 1:10
    XX[:, i] = X[:, i] / std(X[:, i])
end

stat = scatter_corr_plot(X)
plot(layout=grid(5, 2), size=(800, 1000), dpi=200)
for i in 1:10
    xl = nms[i]
    if i in [1, 5, 6, 7, 8, 9, 10]
        histogram!(log10.(X[:, i]), sp=i, bins=50, label=nothing, xlabel="log( "*xl*")")
    else
        histogram!(X[:, i], sp=i, bins=50, label=nothing, xlabel=xl)
    end
end
plot!()