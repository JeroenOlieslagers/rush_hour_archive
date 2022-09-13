import numpy as np
from sklearn import neighbors
from rushhour import load_board, Board
import time
from tqdm import tqdm
from sys import getsizeof
import os
import json
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from heapq import heappop, heappush,heapify


def breadth_first_search(board, deep=False):
    """
    Returns solved board which contains list of all moves leading to optimal solution
    Solver uses breath first search which implements a queue to keep track of which 
    nodes to expand.
    Setting deep=True does not stop the search when a solution is found, but searches
    full tree of all possible states
    """
    # Set of all boards that have been visited so far
    visited_boards = set()
    board.get_available_moves()
    visited_boards.add(board.return_arr_as_int())
    # Queue that shows which boards to visit next
    # Structure of queue elements consists of lists 
    # [moves to get to current node, possible moves from current node].
    queue = deque([[[{0: 0}], [m]] for m in board.available_moves])
    print([m for m in board.available_moves])
    # Number of unique solutions the search reaches
    solutions = 0
    # Optimal number of moves
    optimal_len = None
    # Iterate over 50k iters instead of until queue is empty to avoid really long loops
    expansions = 0
    for i in tqdm(range(200000)):
        board.get_available_moves()
        # Stopping criterion
        if len(queue) == 0:
            break
        # Get next node to visit in tree
        moves = queue.pop()
        # Get moves to get to this node
        past_moves = moves[0]
        # Get adjacent nodes to current node
        current_moves = moves[1]
        # Move to current node
        for move in past_moves:
            board.move(move)
        # Loop over adjacent nodes
        for move in current_moves:
            board.move(move)
            # Check whether the newly visited node is new, if not, skip
            if board.return_arr_as_int() not in visited_boards:
                # Check if board is solved
                if board.solved:
                    if deep:
                        solutions += 1
                        if solutions == 1:
                            optimal_len = len(board.moves)+1
                    else:
                        print(expansions)
                        return board
                # Add newly visited node to list of visited nodes
                visited_boards.add(board.return_arr_as_int())
                # Update queue with new move list to get to adjacent node as well 
                # as it's adjacent nodes
                expansions += 1
                queue.extendleft([[past_moves+[move], board.available_moves]])
            # Go back to current node
            board.undo_moves(1)
        # Go back to source node
        board.undo_moves(len(past_moves))
    # Print info
    # print("Number of visited states: "+str(len(visited_boards)))
    # print("Number of unique solutions: "+str(solutions))
    return optimal_len, len(visited_boards)


class moves_list(list):
    """
    Custom class to allow pushing lists of dicts onto heap without having to compare
    dicts (no built-in __lt__ functionality in dict)
    """
    def __lt__(self, other):
        return True


def a_star(board, h=lambda x: 0, graph_search=False):
    """
    Implementation of A* algorithm with zero heuristic as default heuristic function.
    The search is a tree-search, not a graph-search, which means visited nodes can be
    revisited but the required heuristic function to ensure optimality only needs to be
    admissible, and not monotonic (as would be required for graph search).
    """
    # Initialise all three "lists" (four if graph_search=True)
    start = board.return_arr_as_int()
    # g_scores is a map that gives cheapest cost from start to node currently known
    # (initialised with infinity as default value)
    g_scores = defaultdict(lambda: np.Inf)
    g_scores[start] = 0
    # f_scores (= g_scores[n] + h(n))is a map that gives current best guess as to
    # how cheap a path could be from start to finish if it goes through node n.
    # (initialised with infinity as default value)
    f_scores = defaultdict(lambda: np.Inf)
    f_scores[start] = h(start)
    # open_set is priority queue that will indicate which node to expand next
    open_set = []
    heappush(open_set, (f_scores[start], []))
    # Fifth list for graph_search keeps track of all visited nodes
    closed_set = set()
    closed_set.add(start)
    # count node expansions to prevent infinite loop
    expansions = 0
    while len(open_set) > 0:
        # Exit if too many expansions occured
        if expansions > 100000:
            raise ValueError("Expansion limit reached")
        # Get state with lowest f score
        f, past_moves = heappop(open_set)
        # Move to current node
        if len(past_moves) > 0:
            for move in past_moves:
                board.move(move)
        # Get current board int-string
        current = board.return_arr_as_int()
        # Check if solved
        if board.solved:
            return board
        # Expand current node by getting all available moves
        board.get_available_moves()
        available_moves = board.available_moves
        # Increment expansion counter
        expansions += 1
        for move in available_moves:
            # Perform available move
            board.move(move)
            # Get board int-string
            neighbor = board.return_arr_as_int()
            if graph_search:
                # Ignore if neighbor already visited
                if neighbor in closed_set:
                    # Undo move
                    board.undo_moves(1)
                    continue
                # Add to list if not yet visited
                else:
                    closed_set.add(neighbor)
            # Calculates g score of neighbor
            tentative_score = g_scores[current] + 1
            # If this is the fastest way to get to neighbor, update its costs
            if tentative_score < g_scores[neighbor]:
                # Update score
                g_scores[neighbor] = tentative_score
                f_score = tentative_score + h(neighbor)
                f_scores[neighbor] = f_score
                # If neighbor is not yet queued to be visited, queue it
                if neighbor not in open_set:
                    # Get all moves leading to current neighbor
                    total_moves = moves_list(past_moves + [move])
                    # Push node onto heap
                    heappush(open_set, (f_score, total_moves))
            # Undo move
            board.undo_moves(1)
        # Undo rest of moves back to initial board
        board.undo_moves(len(past_moves))
        if expansions % 1000 == 0:
            print(expansions, len(open_set))
    
        

            

        


if __name__ == "__main__":
    PROBLEM_DIR = 'data/problems'
    PDF_DIR = 'figures/individual_move_graphs'

    with open(PROBLEM_DIR+'/prb11_6.json', 'r') as filename:
        data = json.load(filename)
        board = Board()
        board.load_data(data)
    solution = breadth_first_search(board, deep=True)
    print(board.arr)
    # start = time.time()
    # solution = a_star(board)
    # print(time.time()-start)
    # print(len(solution.moves))


    # jsons = np.sort(os.listdir(PROBLEM_DIR))
    # problem_data = []
    # start = time.time()
    # for n, js in enumerate(jsons[::-1]):
    #     print(n)
    #     if n == 500:
    #         break
    #     with open(PROBLEM_DIR+'/'+js, 'r') as filename:
    #         data = json.load(filename)
    #         board = Board()
    #         board.load_data(data)
    #         opt_len, no_states = breadth_first_search(board, deep=True)
    #         #solution = breadth_first_search(board)
    #         #problem_data.append([js, solution])
    # print(time.time() - start)
    # problem_data = np.array(problem_data)
    # with open('problem_opt_boards.npy', 'wb') as f:
    #     np.save(f, problem_data)

    # with open('problem_opt_boards.npy', 'rb') as f:
    #     problem_data = np.load(f, allow_pickle=True)
    # data = problem_data[:, 1:].astype(int)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()

            
    # llens1 = []
    # llens2 = []
    # for prob, solution in problem_data:
    #     lens1 = []
    #     lens2 = []
    #     #solution.undo_moves(1)
    #     for i in range(len(solution.moves)):
    #         solution.get_mag()
    #         ll1 = len(solution.mag)
    #         ll2 = [0]*8
    #         for i in range(8):
    #             for n, level in enumerate(solution.mag):
    #                 ll2[i] += len(level)*(i-n+1)
    #         lens1.append(ll1)
    #         lens2.append(ll2)
    #         solution.undo_moves(1)
        
    #     llens1.append(lens1[::-1])
    #     llens2.append(np.array(lens2[::-1]))

    # for ll in llens2:
    #     plt.plot(-np.arange(len(ll[:, -1]))[::-1], ll[:, -1], c='k', alpha=0.05)
    # plt.show()
