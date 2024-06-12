struct Car
    dim2::Int8
    dir::Symbol
    len::Int8
end
# FOR FIXED SIZE RUSH HOUR PUZZLES
L = 9
######
s_free_type = MVector{L, Int8}#Vector{Int8}#
s_fixed_type = NTuple{L, Car}#Tuple{Vararg{Car, T}} where T# 
s_type = Tuple{s_free_type, s_fixed_type}
a_type = Tuple{Int8, Int8}
arr_type = Matrix{Int8}#MMatrix{6, 6, Int8}#

moves_type = MVector{4*L, a_type}
blocked_cars_type = MVector{4, Int8}
move_amounts_type = MVector{4, Int8}
blocking_nodes_type = Matrix{Int8}

thought_type = MVector{5, Int8}#(car, moves that unblock)
or_type = Tuple{Int8, thought_type}# depth
and_type = Tuple{Int8, a_type}# depth, move

function board_to_arr!(arr::arr_type, s::s_type)::Nothing
    s_free, s_fixed = s
    fill!(arr, 0)
    for id in eachindex(s_free)
        car = s_fixed[id]
        dim1 = s_free[id]
        if car.dir == :x
            for i in 0:car.len-1
                arr[car.dim2, dim1+i] = id
            end
        else
            for i in 0:car.len-1
                arr[dim1+i, car.dim2] = id
            end
        end
    end
    return nothing
end

function board_to_arr(s::s_type)::arr_type
    s_free, s_fixed = s
    #arr = zeros(arr_type)
    arr = zeros(6, 6)
    for id in eachindex(s_free)
        car = s_fixed[id]
        dim1 = s_free[id]
        if car.dir == :x
            for i in 0:car.len-1
                arr[car.dim2, dim1+i] = id
            end
        else
            for i in 0:car.len-1
                arr[dim1+i, car.dim2] = id
            end
        end
    end
    return arr
end

function arr_to_board(arr::arr_type)::s_type
    # Get unique cars
    us = sort(unique(arr))[2:end]
    s_free = zeros(s_free_type)
    s_fixed = Array{Car}(undef, length(us))
    for u in us
        yx = findfirst(isequal(u), arr)
        y = yx[1]
        x = yx[2]
        dir = arr[y, x+1] == u ? :x : :y
        len = count(isequal(u), arr)
        if dir == :x
            s_free[u] = x
            s_fixed[u] = Car(y, dir, len)
        else
            s_free[u] = y
            s_fixed[u] = Car(x, dir, len)
        end
    end
    return (s_free, Tuple(s_fixed))
end

function arr_to_bigint(arr::arr_type)::BigInt
    return parse(BigInt, string(vec(arr)...))
end

function bigint_to_arr(bigint::BigInt)::arr_type
    # Convert to string
    ints_str = string(bigint)
    extra = 36 - length(string(bigint))
    # Add extra zeros
    ints_str = "0"^extra * ints_str
    # Reshape arr
    arr = reshape([parse(Int64, i) for i in ints_str], 6, 6)
    return arr
end

function board_to_int32(s_free::s_free_type)::Int32
    state = Int32(0)
    if length(s_free) > 13
        throw(ErrorException("Cant convert to 32 bit with length > 13"))
    end
    for (n, i) in enumerate(s_free)
        state += (i-1)*5^(n-1)
    end
    return state
end

function board_to_int32(s::s_type)::Tuple{Int64, Int32}
    s_free, s_fixed = s
    problem = Int64(0)
    state = Int32(0)
    if length(s_free) > 13
        throw(ErrorException("Cant convert to 32 bit with length > 13"))
    end
    for (n, i) in enumerate(s_fixed)
        problem += (i.dim2-1)*6^(n-1)
        problem += (i.dir == :x)*2^(34+n-1)
        problem += (i.len == 2)*2^(48+n-1)
    end
    for (n, i) in enumerate(s_free)
        state += (i-1)*5^(n-1)
    end
    return problem, state
end

function int32_to_board(int::Int32, L::Int64)::s_free_type
    s_free = zeros(s_free_type)
    for i in 1:L
        s_free[i] = (int ÷ 5^(i-1)) % 5 + 1
    end
    return s_free
end

function int32_to_board!(s_free::s_free_type, int::Int32, L::Int64)::Nothing
    for i in 1:L
        s_free[i] = (int ÷ 5^(i-1)) % 5 + 1
    end
    return nothing
end

function board_to_int64(s::s_type)::Tuple{Int64, Int64}
    s_free, s_fixed = s
    problem = Int64(0)
    state = Int64(0)
    for (n, i) in enumerate(s_fixed)
        problem += (i.dim2-1)*6^(n-1)
        problem += (i.dir == :x)*2^(34+n-1)
        problem += (i.len == 2)*2^(48+n-1)
    end
    for (n, i) in enumerate(s_free)
        state += (i-1)*5^(n-1)
    end
    return problem, state
end

function int64_to_board(int::Int64)::s_free_type
    return
end