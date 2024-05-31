
function bfs(s::s_type; max_iter=1000000)
    s_free, s_fixed = s
    L = length(s_free)
    problem, s_init = board_to_int32(s)
    arr = board_to_arr(s)
    visited = Set{Int32}()
    frontier = Vector{Int32}()
    push!(visited, s_init)
    pushfirst!(frontier, s_init)
    moves = MVector{4*length(s_free)}([(Int8(0), Int8(0)) for _ in 1:(4*length(s_free))])
    for _ in 1:max_iter
        if isempty(frontier)
            break
        end
        s_int = pop!(frontier)
        int32_to_board!(s_free, s_int, L)
        s = (s_free, s_fixed)
        if check_solved(s)
            continue
        end
        board_to_arr!(arr, s)
        possible_moves!(moves, s, arr)
        for move in moves
            if move == (0, 0)
                break
            end
            s_free = s[1]
            make_move!(s_free, move)
            s_int_new = board_to_int32(s_free)
            if s_int_new ∉ visited
                push!(visited, s_int_new)
                pushfirst!(frontier, s_int_new)
            end
            undo_move!(s_free, move)
        end
    end
    int32_to_board!(s_free, s_init, L)
    return visited
end

# s = load_data(prbs[1])
# @btime a=bfs(s)

# a = [length(bfs(load_data(prbs[i]))) for i in 1:70]
# b = [length(state_spaces_prb[prbs[i]]) for i in 1:70]
