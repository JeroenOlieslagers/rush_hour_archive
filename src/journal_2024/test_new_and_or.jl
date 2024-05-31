function propagate_ps(x, AND_OR_tree)
    AND_root, AND, OR, parents_moves = AND_OR_tree#, idv_AND, idv_OR, parents_AND, parents_OR
    function propagate!(x, p, visited, dict, AND_current, AND, OR)#, idv_AND, idv_OR)
        γ = x#0.0
        # β_AND1 = x[3]
        # β_AND2 = x[4]
        # β_OR1 = x[5]
        # β_OR2 = x[6]
        # CYCLE PROBABILITY
        if !haskey(AND, AND_current)
            dict[((-2, (-2,)), (-2, -2), AND_current[2])] += p
            return nothing
        end
        # number of children of AND node
        N_and = length(AND[AND_current])
        # Rule 2: AND HEURISTICS
        #mat_and = idv_AND[AND_current]
        #p_ands = size(mat_and)[1] == 1 ? p : p*softmax(mat_and, [β_AND1, β_AND2])
        p_ands = p * ones(N_and) / N_and
        # propagate to OR nodes
        for (n, OR_node) in enumerate(AND[AND_current])
            p_and = p_ands[n]
            # CYCLE PROBABILITY
            ## CHANGED
            if OR_node[1] in visited || !haskey(OR, OR_node)
                dict[((-2, (-2,)), (-2, -2), OR_node[3])] += p_and
                continue
            end
            N_or = length(OR[OR_node])
            # Rule 3: OR HEURISTICS
            #mat_or = idv_OR[OR_node]
            #p_ors = size(mat_or)[1] == 1 ? p_and : p_and*softmax(mat_or, [β_OR1, β_OR2])
            p_ors = p_and * ones(N_or) / N_or
            # propagate to AND nodes
            for (m, AND_next) in enumerate(OR[OR_node])
                p_or = p_ors[m]
                # leaf node
                if AND_next[1] == (0, (0,))
                    dict[OR_node] += p_or
                # elseif AND_next[1] == (-2, (-2,))
                #     dict[((-2, (-2,)), (-2, -2), OR_node[3])] += p_or
                else
                    # Rule 1a: don't stop
                    pp = (1-γ)*p_or
                    # Rule 1b: stop
                    dict[((-1, (-1,)), (-1, -1), OR_node[3])] += γ*p_or
                    # chain of thought
                    ## CHANGED
                    cv = copy(visited)
                    push!(cv, OR_node[1])
                    # recurse
                    propagate!(x, pp, visited, dict, AND_next, AND, OR)#, idv_AND, idv_OR)
                end
            end
        end
    end
    # AND-OR tree node type
    s_type = Tuple{Int, Tuple{Vararg{Int, T}} where T}
    # action type
    a_type = Tuple{Int, Int}
    # OR node type
    or_type = Tuple{s_type, a_type, Int}
    dict = DefaultDict{or_type, Any}(0.0)
    # keeps track of current AO nodes visited
    visited = Set{s_type}()
    propagate!(x, 1.0, visited, dict, AND_root, AND, OR)#, idv_AND, idv_OR)
    return dict
end

b = arr_to_board(int_to_arr(709022709010066610044880333000005550))

board = load_data(prbs[41]);
make_move!(board, (6, -1))
board = load_data(prbs[68]);
board = load_data(prbs[1]);
_, tree = get_and_or_tree(board);
s, AND, OR, p, _, _, _, _ = tree;
draw_ao_tree((AND, OR), board)
draw_ao_tree((a, o), b)
draw_board(get_board_arr(b))

tree = (s, AND, OR, p);
dict = propagate_ps(0, tree)

b = []
for subj in subjs
    for i in eachindex(tree_datass[subj][1])
        if length(tree_datass[subj][1][i][2])+length(tree_datass[subj][1][i][3]) > 500
            b = boards[subj][i]
            break
        end
    end
end

