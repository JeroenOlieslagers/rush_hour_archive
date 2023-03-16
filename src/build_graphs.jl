include("solvers.jl")

board = load_data("hard_puzzle_11");
arr = get_board_arr(board)
start = board_to_int(arr, BigInt)

tree, seen, stat, dict, parents, children, solutions = bfs_path_counters(board; traverse_full=true, all_parents=true);


