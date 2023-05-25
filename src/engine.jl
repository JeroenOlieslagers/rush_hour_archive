using JSON
using DataStructures


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

function get_board_arr!(arr, board::Board)
    """
    Return 6x6 array with car id as integer values
    """
    # Initialise with zeros
    #arr = zeros(Int, 6, 6)
    arr .= 0
    for car in board.cars
        # Fill in array
        if car.is_horizontal
            for l in 1:car.len
                arr[car.y, car.x+l-1] = car.id
            end
        else
            for l in 1:car.len
                arr[car.y+l-1, car.x] = car.id
            end
        end
    end
    return nothing
    #return arr
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
                arr[car.y, car.x+l-1] = car.id
            end
        else
            for l in 1:car.len
                arr[car.y+l-1, car.x] = car.id
            end
        end
    end
    return arr
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

function get_all_available_moves!(available_moves, board::Board, arr)
    """
    Get list of all available moves on board
    """
    # Initalise available move array
    empty!(available_moves)
    for car in board.cars
        get_available_moves!(available_moves, arr, car)
    end
    return nothing
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

function get_available_moves(arr, car)
    """
    Calculates all available moves of given car
    """
    # Initialise
    available_moves = Array{Array{Int, 1}, 1}()
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

function get_type(arr)
    """
    Return datatype necessary to store int representation of board
    """
    str = string(vcat(arr...)...)
    return length(str) > 36 ? BigInt : Int128 
end

function board_to_int(arr, T)
    """
    Returns flattened version of array as string
    """
    return parse(T, string(vcat(arr...)...))
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
    if unique(f) == [0] || length(f) == 0
        return true
    else
        return false
    end
end

function blocked_by(arr, car)
    """
    Calculates which cars are blocked by given car
    """
    # Get row/col
    row = get_1d_arr(arr, car)
    # Get id
    id = car.id
    # Only register cars to the right of red car as blockages
    if id == maximum(arr)
        idx = findfirst(x -> x == id, row)
        row = @view row[idx+2:end]
    end
    # Get other cars on same row/column
    uniq = unique(row)
    return [car_id for car_id in uniq if !(car_id in [0, id])]
end

function blocked_by_amount(arr, car)
    """
    Calculates which cars are blocked by given car and how far they are
    """
    blockages = Vector{Tuple{Int, Int}}()
    # Get row/col
    row = get_1d_arr(arr, car)
    # Get id
    id = car.id
    # Split row/col
    idx = findfirst(x -> x == id, row)
    if id != 9
        # Get backward moves
        b = reverse(@view row[1:idx-1])
        for (m, pos) in enumerate(b)
            if pos != 0
                push!(blockages, (pos, m))
            end
        end
    end
    # Get forward moves
    f = @view row[idx+car.len:end]
    for (m, pos) in enumerate(f)
        if pos != 0
            push!(blockages, (pos, m))
        end
    end
    return blockages
end

function get_mag(board::Board, arr)
    """
    Calculates macro-action graph (MAG)
    """
    # Initialisations
    visited_nodes = Array{Int, 1}()
    # Top node is red car
    current_nodes = [maximum(arr)]
    mag = Array{Dict{Int, Array{Int, 1}}, 1}()
    # Loop over a maximum of number of cars layers
    for i in 1:length(board.cars)
        # Fill in layer
        mag_level = Dict{Int, Array}()
        # Only have one copy of each car as node
        visited_nodes = vcat(visited_nodes, current_nodes)
        # Nodes for next layer not already visited
        nodes = []
        for current_node in current_nodes
            # Get edges
            next_nodes = blocked_by(arr, board.cars[current_node])
            for next_node in next_nodes
                # Select unvisited nodes
                if !(next_node in visited_nodes)
                    push!(nodes, next_node)
                end
            end
            mag_level[current_node] = next_nodes
        end
        current_nodes = nodes
        push!(mag, mag_level)
        # Stop if layer does not have any downstream nodes
        if length(current_nodes) == 0
            break
        end
    end
    return mag
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
    if m > 0
        blockages = reverse(row[pos:pos+m+car.len-1])
    else
        blockages = row[pos+m:pos+car.len-1]
    end
    deleteat!(blockages, findall(x->x in [0, car.id], blockages))
    return unique(blockages)
end

function possible_moves(arr, car)
    """
    Returns car indices that are blocked by move m from car
    """
    # Get row/col
    row = get_1d_arr(arr, car)
    # Get car position in row
    pos = car.is_horizontal ? car.x : car.y
    lb = 0
    ub = 6
    for c in unique(row)
        if c ∉ [0, car.id]
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
    moves = collect(lb-(pos-1):ub-(pos-1+car.len))
    deleteat!(moves, moves .== 0)
    return moves
end

function moves_that_unblock(car1, car2, arr; move_amount=nothing)
    """
    Returns list of all moves by car2 that move it out of car1's way
    Also returns a list of cars blocked by each of those moves
    """
    # Initialise
    # Dictionary that sorts moves with fewest blockages
    possible_moves = OrderedDict{Int, Int}()
    # Dictionary that maps moves to cars it blocks
    blockages = Dict{Int, Array{Int, 1}}()
    # Get all moves by car2
    moves = reverse(get_all_moves(arr, car2))
    moves = moves[sortperm(abs.(moves), alg=MergeSort)]
    # Get position depending on orientation
    pos2 = car2.is_horizontal ? car2.x : car2.y
    # If both cars are aligned
    if !(car1.is_horizontal ⊻ car2.is_horizontal)
        pos1 = car1.is_horizontal ? car1.x : car1.y
        for move in moves
            if (pos2+move >= pos1+move_amount+car1.len) && move > 0 && move_amount > 0
                blocks = move_blocked_by(car2, move, arr)
                push!(possible_moves, move=>length(blocks))
                push!(blockages, move=>blocks)
            elseif (pos2+move <= pos1+move_amount-car2.len) && move < 0 && move_amount < 0
                blocks = move_blocked_by(car2, move, arr)
                push!(possible_moves, move=>length(blocks))
                push!(blockages, move=>blocks)
            end
        end
    else
        # Check whether move unblocks car
        pos1 = car1.is_horizontal ? car1.y : car1.x
        for move in moves
            if move+pos2+car2.len-1 < pos1 || move+pos2 > pos1
                blocks = move_blocked_by(car2, move, arr)
                push!(possible_moves, move=>length(blocks))
                push!(blockages, move=>blocks)
            end
        end
    end
    # Sort blockages and returns only move keys
    constrained_moves = collect(keys(sort(possible_moves, byvalue=true)))
    # Return sorted blockages
    return constrained_moves, [blockages[i] for i in constrained_moves]
end

function get_constrained_mag(board::Board, arr)
    """
    Get MAG where only blockages that are 'relevant' (i.e. prevent red car from moving)
    are considered instead of every blockage
    """
    # Get initial blockages
    visited_nodes = Set{Int}()
    # Top node is red car
    current_nodes = [maximum(arr)]
    # Get initial blockages
    next_nodes = [blocked_by(arr, board.cars[current_nodes[1]])]
    # Initialise mag
    mag = Array{Dict{Int, Array{Int, 1}}, 1}()
    # Push first blockage by red car onto mag
    push!(mag, Dict(current_nodes[1] => next_nodes[1]))
    push!(visited_nodes, current_nodes[1])
    if length(next_nodes[1]) > 0
        push!(visited_nodes, next_nodes[1]...)
    end
    for i in 1:length(board.cars)
        children = Array{Int, 1}()
        # Fill in layer
        mag_level = Dict{Int, Array}()
        for prev_node in keys(last(mag))
            for current_node in last(mag)[prev_node]
                # Get children nodes
                moves, blockages = moves_that_unblock(board.cars[prev_node], board.cars[current_node], arr)
                if length(blockages) > 0
                    # Get car with fewest blockages
                    blocks = first(blockages)
                    # Do not include blocks that are in visited nodes
                    true_blocks = Array{Int, 1}()
                    for block in blocks
                        if !(block in visited_nodes)
                            push!(true_blocks, block)
                        end
                    end
                    # Extend current mag level
                    mag_level[current_node] = true_blocks
                    push!(children, true_blocks...)
                end
            end
        end
        # Add to visited nodes
        if length(children) > 0
            push!(visited_nodes, children...)
        end
        if length(mag_level) > 0
            push!(mag, mag_level)
        end
        # End if no more child nodes
        if length(vcat(values(mag_level)...)) == 0
            break
        end
    end
    return mag
end

function get_multiple_mags(board::Board, arr)
    """
    Return list of MAGs for moves which have same number of blockages
    """
    # Get initial blockages
    visited_nodes_arr = Array{Set{Int}, 1}()
    # Top node is red car
    current_nodes = [maximum(arr)]
    # Get initial blockages
    next_nodes = [blocked_by(arr, board.cars[current_nodes[1]])]
    # Initialise mag
    mags = Array{Array{Dict{Int, Array{Int, 1}}, 1}, 1}()
    # Array keeping track whether MAG is finished
    is_finished = [false]
    # Push first blockage by red car onto mag
    mag = Array{Dict{Int, Array{Int, 1}}, 1}()
    visited_nodes = Set{Int}()
    push!(mag, Dict(current_nodes[1] => next_nodes[1]))
    push!(visited_nodes, current_nodes[1])
    if length(next_nodes[1]) > 0
        push!(visited_nodes, next_nodes[1]...)
    end
    # Create first mag
    push!(mags, mag)
    push!(visited_nodes_arr, visited_nodes)
    for i in 1:length(board.cars)
        # Loop over all mags
        for n in 1:length(mags)
            # Set mag
            mag = mags[n]
            # Fill in layer
            mag_levels = Array{Dict{Int, Array}, 1}()
            first_mag_level = Dict{Int, Array}()
            # Select visited nodes from this MAG
            visited_nodes = visited_nodes_arr[n]
            children = Array{Int, 1}()
            if !is_finished[n]
                for prev_node in keys(last(mag))
                    for current_node in last(mag)[prev_node]
                        # Get children nodes
                        moves, blockages = moves_that_unblock(board.cars[prev_node], board.cars[current_node], arr)
                        if length(blockages) == 0
                            blockages = first(visited_nodes)
                        end
                        for (m, blockage) in enumerate(unique(blockages))
                            # Do not include blocks that are in visited nodes
                            true_blocks = Array{Int, 1}()
                            for block in blockage
                                if !(block in visited_nodes)
                                    push!(true_blocks, block)
                                end
                            end
                            if m == 1
                                if length(mag_levels) == 0
                                    # Extend current mag level
                                    first_mag_level[current_node] = true_blocks
                                    push!(mag_levels, first_mag_level)
                                else
                                    for (w, mag_level) in enumerate(mag_levels)
                                        mag_level[current_node] = true_blocks
                                    end
                                end
                                # if length(blockage) == 0
                                #     break
                                # end
                            else
                                for w in 1:length(mag_levels)
                                    dummy = copy(mag_levels[w])
                                    dummy[current_node] = true_blocks
                                    if !(dummy in mag_levels)
                                        push!(mag_levels, dummy)
                                    end
                                end
                            end
                            push!(children, true_blocks...)
                        end
                    end
                end
                # Add to visited nodes
                # if length(children) > 0
                #     push!(visited_nodes, children...)
                # end
                if length(mag_levels) == 0
                    break
                end
                dummy_mag = copy(mag)
                for (m, mag_level) in enumerate(mag_levels)
                    # End if no more child nodes
                    if length(vcat(values(mag_level)...)) == 0
                        is_finished[n] = true
                    end
                    if length(mag_level) > 0
                        # Newly visited nodes to be added to visited list
                        new_nodes = unique(reduce(vcat, collect(values(mag_level))))
                        if m == 1
                            push!(mag, mag_level)
                            if length(new_nodes) > 0
                                push!(visited_nodes, new_nodes...)
                            end
                        else
                            dummy = copy(dummy_mag)
                            push!(dummy, mag_level)
                            push!(mags, dummy)
                            dummy = copy(visited_nodes)
                            if length(new_nodes) > 0
                                push!(dummy, new_nodes...)
                            end
                            push!(visited_nodes_arr, dummy)
                            push!(is_finished, length(vcat(values(mag_level)...)) == 0)
                        end
                    end
                end
            end
        end
    end
    return mags
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