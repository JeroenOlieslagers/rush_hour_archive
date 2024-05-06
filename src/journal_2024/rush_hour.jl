using JSON
using DataStructures

####### RUSH HOUR ENGINE

"""
    Car(x, y, len, id, is_horizontal)

Mutable class to store basic car properties
# Arguments
- `x::Integer`:         x position of car.
- `y::Integer`:         y position of car.
- `len::Integer`:       how long the car is.
- `id::Integer`:        car identifier.
- `x::Boolean`:         whether car is horizontal or not
"""
mutable struct Car
    x::Int
    y::Int
    len::Int
    id::Int
    is_horizontal::Bool
end

"""
    Board{cars, id}

Holds array of cars and problem ID of current board
"""
struct Board
    """
    Holds list of cars and problem ID
    """
    cars::Array{Car}
    id::String
end


function convert_dict_to_car(dict, id)
    """
    Instantiates car objects
    """
    x = dict["position"] % 6
    y = dict["position"] ÷ 6
    return Car(1 + x, 1 + y, dict["length"], id, dict["orientation"] == "horizontal")
end

function load_data(js)
    """
    Returns Board class from json file
    """
    if length(js) > 1
        data = JSON.parsefile("data/raw_data/problems/" * js * ".json")
    else
        data = JSON.parsefile("data/raw_data/problems/hard_puzzle_40.json")#prb19369_9.json")#prb39654_9.json")#prb3092_9.json")#prb11306_13.json")#prb10546_6.json")
    end
    size = length(data["cars"])
    cars = Array{Car}(undef, size)
    for car in data["cars"]
        id = car["id"] != "r" ? parse(Int, car["id"]) + 1 : size
        # List list of cars is in order (first element is car with id=1, etc)
        cars[id] = convert_dict_to_car(car, id)
    end
    return Board(cars, data["id"])
end

function make_move!(board, move)
    """
    Moves car in board move=[car id, move amount (±)]
    """
    id, m = move
    car = board.cars[id]
    if car.is_horizontal
        car.x += m
    else
        car.y += m
    end
    return nothing
end

function undo_moves!(board, moves)
    """
    Undoes list of moves
    """
    for move in reverse(moves)
        make_move!(board, [move[1], -move[2]])
    end
    return nothing
end

function get_1d_arr(arr, car)
    """
    Helper function that extracts row or column of current car
    """
    if car.is_horizontal
        row = @view arr[car.y, :]
    else
        row = @view arr[:, car.x]
    end
    return row
end

function get_all_moves(arr, car)
    """
    Get all moves, regardless of whether they are blocked by a car
    """
    # Get row/col
    row = get_1d_arr(arr, car)
    # Get x move if car is horizontal, y else
    moves = car.is_horizontal ? findall(!=(car.id), row) .- car.x : findall(!=(car.id), row) .- car.y
    moves[moves .> 0] .-= car.len - 1
    return moves
end

function move_blocked_by(car, m, arr)
    """
    Returns car indices that are blocked by move m from car
    """
    # Get row/col
    row = get_1d_arr(arr, car)
    # Get car position in row
    pos = car.is_horizontal ? car.x : car.y
    # Get blocked arr elements
    blockages = Vector{Int}()
    if m > 0
        for i in pos+car.len:pos+car.len-1+m
            c = row[i]
            if c != 0 && c ∉ blockages
                push!(blockages, c)
            end
        end
        #blockages = row[pos+car.len:pos+car.len-1+m]
    else
        for i in pos+m:pos-1
            c = row[i]
            if c != 0 && c ∉ blockages
                pushfirst!(blockages, c)
            end
        end
        #blockages = reverse(row[pos+m:pos-1])
    end
    #deleteat!(blockages, findall(x->x == 0, blockages))
    return blockages#unique(blockages)
end

function move_blocks_red(board, move)
    c, m = move
    if c < 1
        return false
    end
    car = board.cars[c]
    red_car = board.cars[end]
    if !car.is_horizontal
        if red_car.x < car.x
            if car.y+m <= red_car.y && car.y+m+car.len-1 >= red_car.y
                return true
            end
        end
    end
    return false
end

function possible_moves(arr, car)
    # Get row/col
    row = get_1d_arr(arr, car)
    # Get car position in row
    pos = car.is_horizontal ? car.x : car.y
    lb = 0
    ub = 6
    visited = Int[]
    for c in row
        if c != 0 && c != car.id
            if c ∉ visited
                push!(visited, c)
                cnt = count(x->x==c, row)
                if cnt > 1
                    fst = findfirst(x->x==c, row)
                    if fst < pos
                        lb += cnt
                    else
                        ub -= cnt
                    end
                end
            end
        end
    end
    if lb == 0 && ub == 6
        return nothing
    else
        moves = collect(lb-(pos-1):ub-(pos-1+car.len))
        deleteat!(moves, moves .== 0)
        return moves
    end
end

function unblocking_moves(car1, car2, arr; move_amount=nothing)
    moves = Vector{Int}()
    # Get car position in row
    pos2 = car2.is_horizontal ? car2.x : car2.y
    # squares behind and in front of car
    b = pos2-1
    f = 6-(pos2-1+car2.len)
    # Only check possible moves
    poss_moves = possible_moves(arr, car2)
    # If both cars are aligned
    if !(car1.is_horizontal ⊻ car2.is_horizontal)
        pos1 = car1.is_horizontal ? car1.x : car1.y
        # backward moves
        if move_amount < 0
            if (pos2+car2.len-1) - (pos1+move_amount) < b
                for i in (pos2+car2.len-1) - (pos1+move_amount) + 1:b
                    if poss_moves === nothing
                        pushfirst!(moves, -i)
                    elseif -i in poss_moves
                        pushfirst!(moves, -i)
                    end
                end
            end
        else # forward moves
            if (pos1+car1.len-1+move_amount) - pos2 < f
                for i in (pos1+car1.len-1+move_amount) - pos2 + 1:f
                    if poss_moves === nothing
                        push!(moves, i)
                    elseif i in poss_moves
                        push!(moves, i)
                    end
                end
            end
        end
    else
        pos1_perp = car1.is_horizontal ? car1.y : car1.x
        # backward moves
        if (pos2+car2.len-1) - pos1_perp < b
            for i in (pos2+car2.len-1) - pos1_perp + 1:b
                if poss_moves === nothing
                    pushfirst!(moves, -i)
                elseif -i in poss_moves
                    pushfirst!(moves, -i)
                end
            end
        end
        # forward moves
        if pos1_perp - pos2 < f
            for i in pos1_perp - pos2 + 1:f
                if poss_moves === nothing
                    push!(moves, i)
                elseif i in poss_moves
                    push!(moves, i)
                end
            end
        end
    end
    return moves
end

function get_all_available_moves(board::Board, arr)
    """
    Get list of all available moves on board
    """
    # Initalise available move array
    available_moves = Array{Array{Int, 1}, 1}()
    for car in board.cars
        get_available_moves!(available_moves, arr, car)
    end
    return available_moves
end

function get_available_moves!(available_moves, arr, car)
    """
    Calculates all available moves of given car
    """
    # Initialise
    #available_moves = Array{Array{Int, 1}, 1}()
    # Get row/col
    row = get_1d_arr(arr, car)
    # Get id
    id = car.id
    # Split row/col
    idx = findfirst(x -> x == id, row)
    # Get backward moves
    b = reverse(@view row[1:idx-1])
    for (m, pos) in enumerate(b)
        if pos == 0
            push!(available_moves, [id, -m])
        else
            break
        end
    end
    # Get forward moves
    f = @view row[idx+car.len:end]
    for (m, pos) in enumerate(f)
        if pos == 0
            push!(available_moves, [id, m])
        else
            break
        end
    end
    return nothing
    #return available_moves
end

function get_move_amount(piece, target, board::Board)
    """
    From piece ID and target location, get move amount
    """
    # Get car object
    car = board.cars[piece]
    if car.is_horizontal
        return 1 + (target % 6) - car.x
    else
        return 1 + (target ÷ 6) - car.y
    end
end

function check_solved(arr)
    """
    Checks if board is solved, returns true if so, false else
    """
    # Get row with red car on it
    row = arr[3, :]
    # Split row/col
    idx = findfirst(x -> x == maximum(arr), row)
    # Get forward moves
    f = row[idx+2:end]
    # Check if there is anything blocking red car
    if sum(f) == 0 || length(f) == 0
        return true
    else
        return false
    end
end

function get_board_arr(board::Board)::Matrix{Int}
    """
    Return 6x6 array with car id as integer values
    """
    # Initialise with zeros
    arr = zeros(Int, 6, 6)
    for car in board.cars
        # Fill in array
        if car.is_horizontal
            for l in 1:car.len
                if car.y < 1 || car.x+l-1 > 6
                    println(board)
                end
                arr[car.y, car.x+l-1] = car.id
            end
        else
            for l in 1:car.len
                if car.y+l-1 < 1 || car.x > 6
                    println(board)
                end
                arr[car.y+l-1, car.x] = car.id
            end
        end
    end
    return arr
end

function board_to_int(arr, T)
    """
    Returns flattened version of array as string
    """
    return parse(T, string(vec(arr)...))
end

"""
    int_to_arr(int)

Turn int into board arr
"""
function int_to_arr(int)
    # Convert to string
    ints_str = string(int)
    extra = 36 - length(string(int))
    # Add extra zeros
    ints_str = "0"^extra * ints_str
    # Reshape arr
    arr = reshape([parse(Int, i) for i in ints_str], 6, 6)
    return arr
end

"""
    arr_to_board(arr)

Turn arr into board
"""
function arr_to_board(arr)
    # Get unique cars
    us = sort(unique(arr))[2:end]
    # Initialise array of Car
    cars = Array{Car}(undef, length(us))
    for u in us
        yx = findfirst(isequal(u), arr)
        y = yx[1]
        x = yx[2]
        l = count(isequal(u), arr)
        if x == 6
            cars[u] = Car(x, y, l, u, false)
        else
            cars[u] = Car(x, y, l, u, arr[y, x+1] == u)
        end
    end
    return Board(cars, "_")
end

"""
    red_distance(board, arr)

Calculate how many vehicles block the red car
"""
function red_distance(arr)
    # Get row with red car on it
    row = arr[3, :]
    # Split row/col
    idx = findfirst(x -> x == maximum(arr), row)
    # Get forward moves
    f = row[idx+2:end]
    # Return unique cars
    u = unique(f)
    # Remove zeros
    deleteat!(u, u .== 0)
    return length(u)
end

"""
    red_pos(arr)

Calculate how close the red car is to the exit
"""
function red_pos(arr)
    # Get row with red car on it
    row = arr[3, :]
    # Find position of red car
    idx = findfirst(x -> x == maximum(arr), row) + 1
    return 6 - idx
end
