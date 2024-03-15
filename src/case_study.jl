subj = subjs[20]
prb = prbs[1]
r = 1
board = load_data(prb)

tree_, _, _, moves_ = tree_datas_prb[subj];
tree = tree_[prb][r];
moves = moves_[prb][r]

s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
a_type = Tuple{Int, Int}
and_type = Tuple{s_type, Int}
or_type = Tuple{s_type, a_type, Int}
nA = DefaultDict{and_type, Vector{or_type}}([])
nO = DefaultDict{or_type, Vector{and_type}}([])
visited_ORs = Vector{or_type}()

_, sol, _ = a_star(board)
sol = sol[1:end-1]

for i in eachindex(sol)
    #move = moves[i]
    #root_AND, A, O, _, _, parents_moves, parents_AND, parents_OR = tree[i];

    move = Tuple(sol[i])
    _, tre = get_and_or_tree(board; backtracking=true)
    root_AND, A, O, _, _, parents_moves, parents_AND, parents_OR = tre;
    make_move!(board, move)
    if move in keys(parents_moves)
        move_ORs = parents_moves[move]
        if isempty(move_ORs)
            throw(DomainError("weird empty list"))
        end
        start_OR = nothing
        min_d = 1000000
        for OR in values(parents_moves[move])
            if OR[3] < min_d
                start_OR = OR
                min_d = OR[3]
            end
        end
        if start_OR === nothing
            throw(DomainError("weird OR selection"))
        end
        push!(visited_ORs, start_OR)
        next_ORs, nA, nO = backtrack_chain(nA, nO, start_OR, parents_AND, parents_OR, root_AND);
    else
        start_AND = ((move[1], (move[2],)), -1)
        start_OR = ((move[1], (move[2],)), move, -1)
        push!(visited_ORs, start_OR)
        nA[start_AND] = [start_OR]
        nO[start_OR] = [((0, (0,)), -1)]
    end
end
g = draw_ao_tree((nA, nO), board; highlight_ORs=visited_ORs)
draw_board(get_board_arr(board))

# ====== FIRST MOVE ==========
i = 1
move = moves[i]
root_AND, A, O, _, _, parents_moves, parents_AND, parents_OR = tree[i];
move_ORs = parents_moves[move]
start_OR = move_ORs[1]

next_ORs, nA, nO = backtrack_chain(start_OR, parents_AND, parents_OR, root_AND);

g = draw_ao_tree((A, O), board; highlight_ORs=move_ORs)
g = draw_ao_tree((nA, nO), board)

# ====== SECOND MOVE ==========
i = 2
move = moves[i]
root_AND, A, O, _, _, parents_moves, parents_AND, parents_OR = tree[i];
move_ORs = parents_moves[move]
start_OR = move_ORs[1]

next_ORs, nA, nO = backtrack_chain(start_OR, parents_AND, parents_OR, root_AND; chain_AND=nA, chain_OR=nO);

g = draw_ao_tree((A, O), board; highlight_ORs=move_ORs)
g = draw_ao_tree((nA, nO), board)

# ====== THIRD MOVE ==========
i = 3
move = moves[i]
root_AND, A, O, _, _, parents_moves, parents_AND, parents_OR = tree[i];
move_ORs = parents_moves[move]
start_OR = move_ORs[1]

next_ORs, nA, nO = backtrack_chain(start_OR, parents_AND, parents_OR, root_AND; chain_AND=nA, chain_OR=nO);

g = draw_ao_tree((A, O), board; highlight_ORs=move_ORs)
g = draw_ao_tree((nA, nO), board)

# ====== FOURTH MOVE ==========
i = 4
move = moves[i]
root_AND, A, O, _, _, parents_moves, parents_AND, parents_OR = tree[i];
move_ORs = parents_moves[move]
start_OR = move_ORs[1]

next_ORs, nA, nO = backtrack_chain(start_OR, parents_AND, parents_OR, root_AND; chain_AND=nA, chain_OR=nO);

g = draw_ao_tree((A, O), board; highlight_ORs=move_ORs)
g = draw_ao_tree((nA, nO), board)

function backtrack_chain(chain_AND, chain_OR, start_OR, parents_AND, parents_OR, root_AND; max_iter=100)
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    a_type = Tuple{Int, Int}
    and_type = Tuple{s_type, Int}
    or_type = Tuple{s_type, a_type, Int}
    frontier = Vector{or_type}()
    visited = Set{or_type}()
    pushfirst!(frontier, start_OR)
    push!(visited, start_OR)
    push!(chain_OR[start_OR], ((0, (0,)), -1))
    next_moves = Vector{or_type}()
    for AND in parents_OR[start_OR]
        for OR in parents_AND[AND]
            if OR ∉ next_moves
                push!(next_moves, OR)
            end
        end
    end
    for i in 1:max_iter
        if isempty(frontier)
            return next_moves, chain_AND, chain_OR
        end
        next_OR = pop!(frontier)
        for AND in parents_OR[next_OR]
            if next_OR ∉ chain_AND[AND]
                push!(chain_AND[AND], next_OR)
            end
            if AND == root_AND
                continue
            end
            for OR in parents_AND[AND]
                if AND ∉ chain_OR[OR]
                    push!(chain_OR[OR], AND)
                end
                if OR ∉ visited
                    pushfirst!(frontier, OR)
                    push!(visited, OR)
                end
            end
        end
    end
    throw(DomainError("max_iter depth reached"))
end


