
function bfs(s_free::s_free_type, s_fixed::s_fixed_type; max_iter=1000000, d_goal=true)
    s = (s_free, s_fixed)
    L = length(s_free)
    problem, s_init = board_to_int32(s)
    arr = board_to_arr(s)
    visited = Set{Int32}()
    frontier = Vector{Int32}()
    distances = Dict{Int32, Int8}()
    push!(visited, s_init)
    pushfirst!(frontier, s_init)
    distances[s_init[1]] = 0
    moves = MVector{4*length(s_free)}([(Int8(0), Int8(0)) for _ in 1:(4*length(s_free))])
    for _ in 1:max_iter
        if isempty(frontier)
            break
        end
        s_int = pop!(frontier)
        d = distances[s_int]
        int32_to_board!(s_free, s_int, L)
        s = (s_free, s_fixed)
        if check_solved(s)
            if d_goal
                return d
            else
                continue
            end
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
                distances[s_int_new] = d+1
            end
            undo_move!(s_free, move)
        end
    end
    int32_to_board!(s_free, s_init, L)
    return visited
end

function get_d_goals(prbs)
    d_goals_prbs = Dict{String, Dict{Int32, Int8}}()
    L = 9;
    for prb in ProgressBar(prbs)
        s_free, s_fixed = load_data(prb)
        d_goals_prb = Dict{Int32, Int8}()
        ss = bfs(s_free, s_fixed; d_goal=false)
        for s in ss
            int32_to_board!(s_free, s, L)
            d_goals_prb[s] = bfs(s_free, s_fixed)
        end
        d_goals_prbs[prb] = d_goals_prb
    end
    save("data/processed_data/d_goals_prbs.jld2", d_goals_prbs)
end

