
function load_data(prb::String)::s_type
    data = JSON.parsefile("data/raw_data/problems/" * prb * ".json")
    size = length(data["cars"])
    s_free = zeros(Int8, size)
    s_fixed = Array{Car}(undef, size)
    for car in data["cars"]
        id = car["id"] != "r" ? parse(Int, car["id"]) + 2 : 1
        x = 1 + car["position"] % 6
        y = 1 + car["position"] ÷ 6
        dir = car["orientation"] == "horizontal" ? :x : :y
        len = car["length"]
        if dir == :x
            s_free[id] = x
            s_fixed[id] = Car(y, dir, len)
        else
            s_free[id] = y
            s_fixed[id] = Car(x, dir, len)
        end
    end
    return s_free, Tuple(s_fixed)
end

function load_data(prb::String15)::s_type
    data = JSON.parsefile("data/raw_data/problems/" * prb * ".json")
    size = length(data["cars"])
    s_free = zeros(Int8, size)
    s_fixed = Array{Car}(undef, size)
    for car in data["cars"]
        id = car["id"] != "r" ? parse(Int, car["id"]) + 2 : 1
        x = 1 + car["position"] % 6
        y = 1 + car["position"] ÷ 6
        dir = car["orientation"] == "horizontal" ? :x : :y
        len = car["length"]
        if dir == :x
            s_free[id] = x
            s_fixed[id] = Car(y, dir, len)
        else
            s_free[id] = y
            s_fixed[id] = Car(x, dir, len)
        end
    end
    return s_free, Tuple(s_fixed)
end

function make_move(s_free::s_free_type, a::a_type)::s_free_type
    s_free_copy = copy(s_free)
    s_free_copy[a[1]] += a[2]
    return s_free_copy
end

function make_move!(s_free::s_free_type, a::a_type)::Nothing
    s_free[a[1]] += a[2]
    return nothing
end

function undo_move!(s_free::s_free_type, a::a_type)::Nothing
    make_move!(s_free, (a[1], -a[2]))
    return nothing
end

function make_move!(s::s_type, arr::arr_type, a::a_type)::Nothing
    s_free, s_fixed = s
    id = a[1]
    if s_fixed[id].dir == :x
        for i in 1:s_fixed[id].len
            arr[s_fixed[id].dim2, s_free[id]+i-1] = 0
        end
    elseif s_fixed[id].dir == :y
        for i in 1:s_fixed[id].len
            arr[s_free[id]+i-1, s_fixed[id].dim2] = 0
        end
    end
    s_free[id] += a[2]
    if s_fixed[id].dir == :x
        for i in 1:s_fixed[id].len
            arr[s_fixed[id].dim2, s_free[id]+i-1] = id
        end
    elseif s_fixed[id].dir == :y
        for i in 1:s_fixed[id].len
            arr[s_free[id]+i-1, s_fixed[id].dim2] = id
        end
    end
    return nothing
end

function undo_move!(s::s_type, arr::arr_type, a::a_type)::Nothing
    make_move!(s, arr, (a[1], -a[2]))
    return nothing
end

function possible_moves!(moves::moves_type, s::s_type, arr::arr_type)::Nothing
    fill!(moves, (0, 0))
    s_free, s_fixed = s
    iter = 0
    for id in eachindex(s_free)
        for dir in (-1, 1)
            for amount in 1:4
                dim2 = s_fixed[id].dim2
                dim1 = s_free[id] + (dir == 1 ? s_fixed[id].len-1 : 0) + dir*amount
                if (dim1 < 1) || (dim1 > 6)
                    break
                end
                if s_fixed[id].dir == :x
                    if arr[dim2, dim1] != 0
                        break
                    end
                elseif s_fixed[id].dir == :y
                    if arr[dim1, dim2] != 0
                        break
                    end
                end
                iter += 1
                moves[iter] = (id, dir*amount)
            end
        end
    end
    return nothing
end
#moves = @MVector([(Int8(0), Int8(0)) for _ in 1:(4*L)])

function possible_moves_N!(moves::moves_type, s::s_type, arr::arr_type)::Int
    fill!(moves, (0, 0))
    s_free, s_fixed = s
    iter = 0
    for id in eachindex(s_free)
        for dir in (-1, 1)
            for amount in 1:4
                dim2 = s_fixed[id].dim2
                dim1 = s_free[id] + (dir == 1 ? s_fixed[id].len-1 : 0) + dir*amount
                if (dim1 < 1) || (dim1 > 6)
                    break
                end
                if s_fixed[id].dir == :x
                    if arr[dim2, dim1] != 0
                        break
                    end
                elseif s_fixed[id].dir == :y
                    if arr[dim1, dim2] != 0
                        break
                    end
                end
                iter += 1
                moves[iter] = (id, dir*amount)
            end
        end
    end
    return iter
end

function possible_moves(s::s_type, arr::arr_type)::moves_type
    moves = @MVector([(Int8(0), Int8(0)) for _ in 1:(4*L)])
    s_free, s_fixed = s
    iter = 0
    for id in eachindex(s_free)
        for dir in (-1, 1)
            for amount in 1:4
                dim2 = s_fixed[id].dim2
                dim1 = s_free[id] + (dir == 1 ? s_fixed[id].len-1 : 0) + dir*amount
                if (dim1 < 1) || (dim1 > 6)
                    break
                end
                if s_fixed[id].dir == :x
                    if arr[dim2, dim1] != 0
                        break
                    end
                elseif s_fixed[id].dir == :y
                    if arr[dim1, dim2] != 0
                        break
                    end
                end
                iter += 1
                moves[iter] = (id, dir*amount)
            end
        end
    end
    return moves
end

function overlap(s1::Tuple{Int8, Car}, s2::Tuple{Int8, Car})::Bool
    dim11, car1 = s1
    dim12, car2 = s2
    if car1.dir == car2.dir 
        if car1.dim2 == car2.dim2  #dont need to check for this really
            return dim12 - car1.len < dim11 < dim12 + car2.len
        else
            return false
        end
    else
        return (dim11 + car1.len > car2.dim2 >= dim11) &&
               (dim12 + car2.len > car1.dim2 >= dim12)
    end 
end

function check_solved(s::s_type)::Bool
    s_free, s_fixed = s
    red_car_dim1 = s_free[1]
    red_car = s_fixed[1]
    # Construct ghost car, which will determine whether path to goal is open
    # 6 = board_dim[red_car.dir==:x ? 1 : 2]
    g_car = Car(red_car.dim2, red_car.dir, 6 - (red_car_dim1+red_car.len-1))
    g = (red_car_dim1+red_car.len, g_car)
    for id in eachindex(s_free)
        # Do not consider cars which are in the same direction as the red car
        # NOTE: this requires that there is no car lying between the red car
        # and the goal in the same direction as the red car, which would make
        # the puzzle impossible to solve.
        if s_fixed[id].dir != g_car.dir
            if overlap((s_free[id], s_fixed[id]), g)
                return false
            end
        end
    end
    return true
end

function move_blocked_by!(blocked_cars::blocked_cars_type, a::a_type, s::s_type, arr::arr_type)::Nothing
    fill!(blocked_cars, 0)
    iter = 0
    prev_car = 0
    s_free, s_fixed = s
    id, m = a
    for i in 1:abs(m)
        if s_fixed[id].dir == :x
            y = s_fixed[id].dim2
            if m > 0
                x = s_free[id]+s_fixed[id].len-1+i
            else
                x = s_free[id]-i
            end
        else
            x = s_fixed[id].dim2
            if m > 0
                y = s_free[id]+s_fixed[id].len-1+i
            else
                y = s_free[id]-i
            end
        end
        car = arr[y, x]
        if car == 0
            continue
        end
        if car == prev_car
            continue
        end
        iter += 1
        blocked_cars[iter] = car
        prev_car = car
    end
    return nothing
end
#blocked_cars = zeros(blocked_cars_type)

function unblocking_moves!(move_amounts::move_amounts_type, a::a_type, id2::Int8, s::s_type)::Nothing
    # ASSUMPTIONS:
    # car `id2` blocks the move `a`
    fill!(move_amounts, 0)
    iter = 0
    s_free, s_fixed = s
    id1, m = a
    # squares behind and in front of car
    b = s_free[id2]-1
    f = 6 - (s_free[id2] + s_fixed[id2].len-1)
    # This prevents impossible moves from being suggested
    for id in eachindex(s_free)
        if s_fixed[id].dim2 == s_fixed[id2].dim2
            if s_fixed[id].dir == s_fixed[id2].dir
                if id != id2
                    if s_free[id] > s_free[id2]
                        f -= s_fixed[id].len
                    else 
                        b -= s_fixed[id].len
                    end
                end
            end
        end
    end
    # If both cars are aligned
    if s_fixed[id1].dir == s_fixed[id2].dir
        if m < 0
            # check if enough squares behind
            if (s_free[id2]+s_fixed[id2].len-1) - (s_free[id1]+m) < b
                for i in (s_free[id2]+s_fixed[id2].len-1) - (s_free[id1]+m) + 1:b
                    iter += 1
                    move_amounts[iter] = -i
                end
            end
        else
            # check if enough squares in front
            if (s_free[id1]+s_fixed[id1].len-1+m) - s_free[id2] < f
                for i in (s_free[id1]+s_fixed[id1].len-1+m) - s_free[id2] + 1:f
                    iter += 1
                    move_amounts[iter] = i
                end
            end
        end
    else # If cars are not aligned, just need to check how far to move above/below car
        # check if enough squares behind
        if s_free[id2] + s_fixed[id2].len-1-s_fixed[id1].dim2  < b
            for i in s_free[id2] + s_fixed[id2].len-1-s_fixed[id1].dim2 + 1:b
                iter += 1
                move_amounts[iter] = -i
            end
        end
        # check if enough squares in front
        if s_fixed[id1].dim2-s_free[id2] < f
            for i in s_fixed[id1].dim2-s_free[id2]+1:f
                iter += 1
                move_amounts[iter] = i
            end
        end
    end
    return nothing
end
# move_amounts = zeros(move_amounts_type)
