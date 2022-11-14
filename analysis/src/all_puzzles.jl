include("rushhour.jl")

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





lines = readlines("analysis/processed_data/no_wall.txt")
opts = zeros(length(lines))
sss = zeros(length(lines))
for i in eachindex(lines)
    opt_len, line, full_ss = split(lines[i], " ")
    #board = line_to_board(line)
    opts[i] = parse(Int, opt_len)
    sss[i] = parse(Int, full_ss)
end