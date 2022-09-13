from matplotlib.style import available
import numpy as np
import json
import time
import os
import graphviz
from PyPDF2 import PdfMerger
import matplotlib.pyplot as plt
from sys import getsizeof


class Car:
    def __init__(self, board, pos, length, orientation, id):
        # Parent board class holding all cars
        self.board = board
        # Position in x and y from 0-6 (pos is int from 0-35)
        self.x = pos % 6
        self.y = pos // 6
        # Length of car, either 2 or 3
        self.length = length
        # 'vertical' or 'horizontal'
        self.orientation = orientation
        # ID of car (9 = red car = player), (no ID=0 because arr is filled with 0s)
        self.id = id
        # List of all previous moves this car has done
        #self.past_moves = []
        # List of possible moves (list of ints)
        self.available_moves = []
        # Which car id is this car blocked by
        self.blocked_by = []
    
    def move(self, m):
        """
        Moves this car by m places (positive is down and right, negative is up and left)
        """
        if self.orientation == 'horizontal':
            self.x += m
        else:
            self.y += m
        #self.past_moves.append(m)
        # Updates car attributes
        self.board.update()
    
    def update(self, careful=False):
        """
        Updates the available moves and car ids this car is blocked by using parent Board
        """
        self.calculate_blocked_by()
        if careful:
            self.calculate_available_moves()

    def calculate_available_moves(self):
        """
        Calculates all possible moves (integers) for this car
        """
        # Reset available moves
        self.available_moves = []
        # Gets relevant row or column
        arr = self.get_1d_arr()
        # Split array into forward and backward moves
        # Get first occurence of this car
        idx = np.where(arr == self.id)[0][0]
        # Get backward moves
        b = np.flip(arr[:idx])
        for n, pos in enumerate(b):
            if pos == 0:
                self.available_moves.append(-(n+1))
            else:
                break
        # Get forward moves
        f = arr[idx+self.length:]
        for n, pos in enumerate(f):
            if pos == 0:
                self.available_moves.append(n+1)
            else:
                break
        #np.sort(self.available_moves)


    def calculate_blocked_by(self, careful=False):
        """
        Calculates which cars (IDs) this car is blocked by
        """
        # Get row/column of car
        arr = self.get_1d_arr()
        # Only register cars to the right of red car as blockages
        if self.id == 9:
            idx = np.where(arr == self.id)[0][0]
            arr = arr[idx+self.length:]
            if not arr.any():
                self.board.solved = True
            else:
                self.board.solved = False
        # Get other cars on same row/column
        if careful:
            uniq = np.unique(arr)
            self.blocked_by = [car_id for car_id in uniq if car_id not in [0, self.id]]

    def get_1d_arr(self):
        """
        Helper function that extracts row or column of current car
        """
        if self.orientation == 'horizontal':
            arr = self.board.arr[self.y, :]
        else:
            arr = self.board.arr[:, self.x]
        return arr
        

class Board:
    def __init__(self):
        self.arr = np.zeros((6, 6)).astype(int)
        # List of Car classes
        self.cars = []
        # List of all moves (as dicts)
        self.moves = []
        # List of all possible moves
        self.available_moves = []
        # Is true if solved
        self.solved = False
        # Macro action graph (MAG) array for current board state
        # Constructed as matrix where M_ij = car i is blocked by car j
        #self.mag_arr = np.zeros((9, 9))
        # MAG
        #self.mag = []
        # Optimal solution length of current board status
        #self.current_solution_length = None

    def __eq__(self, other):
        return np.array_equal(self.arr, other.arr)

    def load_data(self, data):
        """
        Loads json data into Car objects and updates class
        """
        # Initialise empty car list to order by ID
        self.cars = [None] * len(data['cars'])
        for car in data['cars']:
            # ID conversion
            id = int(car['id']) + 1 if car['id'] != 'r' else 9
            # Instantiate Car objects
            self.cars[id-1] = Car(self, car['position'], car['length'], car['orientation'], id)
        self.update()
        self.problem_id = data['id']
    
    def update(self, careful=False):
        """
        Update all cars' available moves and blockages
        """
        self.update_arr()
        if careful:
            for car in self.cars:
                car.update()
            self.get_available_moves()
        else:
            self.cars[-1].update()
        # for car in self.cars:
        #     car.update()
        #TODO: SOLVER
        #self.get_mag_arr()
        #self.get_mag()
        #self.current_solution_length = None

    def get_available_moves(self):
        """
        Get list of all available moves of board
        """
        self.available_moves = []
        for car in self.cars:
            car.calculate_available_moves()
            for move in car.available_moves:
                self.available_moves.append({car.id: move})

    def update_arr(self):
        """
        Updates array of all car positions
        """
        self.arr = np.zeros((6, 6)).astype(int)
        for car in self.cars:
            if car.orientation == 'horizontal':
                self.arr[car.y, car.x:car.x+car.length] = car.id
            else:
                self.arr[car.y:car.y+car.length, car.x] = car.id

    def move(self, move_dict, careful=False):
        """
        Moves a piece by m units (positive is down and right, negative is up and left)
        """
        car_id, m = list(*move_dict.items())
        if car_id == 0:
            return
        car = self.cars[car_id-1]

        # Check if move is legal, else raise error
        if careful:
            if car.available_moves:
                if m in car.available_moves:
                    # Commit move
                    car.move(m)
                    # Register move
                    #self.moves.append({"id": car_id, "move": m})
                    self.moves.append({car_id: m})
                else:
                    raise ValueError("Move not available, the possible moves are:"+str(car.available_moves))
            else:
                raise ValueError("This car has no available moves")
        else:
            # Move car without checking if moves are legal
            car.move(m)
            self.moves.append({car_id: m})

    
    def undo_moves(self, n):
        """
        Undoes the last n moves and removes them from the list of moves
        """
        last_moves = self.moves[-n:][::-1]
        for move in last_moves:
            car_id, m = list(*move.items())
            # Undo last move
            self.move({car_id: -m})
            # Remove undo move and previous move
            self.moves.pop()
            self.moves.pop()

    def return_arr_as_int(self):
        """
        Converts board state into integer for easier storage
        """
        str_arr = self.arr.flatten().astype(str).tolist()
        return int(''.join(str_arr))

    def get_mag_arr(self):
        """
        Calculates MAG array
        """
        self.mag_arr = np.zeros((9, 9)).astype(int)
        for n, car in enumerate(self.cars):
            if car.blocked_by:
                self.mag_arr[n, np.array(car.blocked_by)-1] = 1

    def get_mag(self):
        """
        Calculates MAG (including layers)
        """
        # Initialisations
        visited_nodes = []
        # Top node is red car
        current_nodes = [9]
        mag = []
        # Loop over a maximum of 9 layers
        for i in range(len(self.cars)):
            next_nodes = []
            mag_level = []
            # Loop over nodes in current layer
            for node in current_nodes:
                # Only have one copy of each car as node
                visited_nodes.append(node)
                car = self.cars[node-1]
                # Get nodes for next layer
                next_nodes += [d for d in car.blocked_by if d not in visited_nodes+next_nodes]
                # Set edges
                mag_level.append({car.id: car.blocked_by})
            current_nodes = next_nodes
            mag.append(mag_level)
            # Stop if layer does not have any downstream nodes
            if len(current_nodes) == 0:
                break
        self.mag = mag

    def visualise_graph(self):
        # Add zeros to order correctly
        if len(self.moves) < 10:
            s = '0' + str(len(self.moves))
        else:
            s = str(len(self.moves))
        # Initialise and name graph
        g = graphviz.Digraph(self.problem_id + '_move_' + s)
        g.node('dummy', label='Move '+str(len(self.moves)), shape='box')
        g.edge('dummy', '9', style='invis')
        # Get MAG
        mag = self.mag
        visited_nodes = []
        # Loop over MAG and create a cluster for each layer
        for n, layer in enumerate(mag):
            with g.subgraph(name='cluster_'+str(n)) as c:
                # Remove cluster box
                c.attr(peripheries='0')
                # Create nodes in each layer
                for node in layer:
                    source = next(iter(node))
                    # Updated visited nodes
                    visited_nodes.append(source)
                    if source == 9:
                        c.node('9', label='R', style='filled', fillcolor='#ff0000')
                    else:
                        c.node(str(source))
                    # Create edges between layers
                    for target in node[source][::-1]:
                        if target in visited_nodes:
                            g.edge(str(source), str(target), constraint='false')
                        else:
                            g.edge(str(source), str(target))
        # Render graph
        g.render(directory='figures/individual_move_graphs')
        

def load_board():
    PROBLEM_DIR = 'data/problems'
    PDF_DIR = 'figures/individual_move_graphs'
    jsons = os.listdir(PROBLEM_DIR)
    board = Board()
    with open(PROBLEM_DIR+'/prb11_6.json', 'r') as filename:
        data = json.load(filename)
        board.load_data(data)
    return board


if __name__ == "__main__":
    board = load_board()
    board.get_available_moves()
    print(board.available_moves)
    # board.get_mag()
    # print(board.arr)
    # print(board.mag)
    # for level in board.mag[1:]:
    #     print(level, len(level))
    # board.visualise_graph()

    # board.move({9: 1})
    # print(board.arr)

    # board.move({6: -3})
    # print(board.arr)

    # board.move({3: -2})
    # print(board.arr)

    # board.move({1: 3})
    # print(board.arr)

    # board.move({2: 2})
    # print(board.arr)

    # board.move({7: 2})
    # print(board.arr)
    # print(board.solved)

    # board.undo_moves(3)
    # print(board.arr)
    # print(board.solved)
    # board.visualise_graph()

    # board.move(5, -3)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(2, -3)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(8, 3)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(7, -2)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(4, 3)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(1, 2)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(9, 2)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(3, 3)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(9, -2)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(1, -2)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(5, 2)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(6, -4)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(7, -1)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(1, 2)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(2, -1)
    # print(board.arr)
    # board.visualise_graph()

    # board.move(5, -2)
    # print(board.arr)
    # board.visualise_graph()
    
    # for n, j in enumerate(jsons):
    #     with open(PROBLEM_DIR+'/'+j, 'r') as filename:
    #         data = json.load(filename)
    #         board.load_data(data)
    #     # if n > 3:
    #     #     break
    #     board.visualise_graph()
    # merger = PdfMerger()
    # pdfs = np.sort(os.listdir(PDF_DIR))
    # for pdf in pdfs:
    #     if pdf.endswith('pdf') and pdf[:5] == 'prb70':
    #         merger.append(PDF_DIR+'/'+pdf)

    # merger.write("moves_prob_70.pdf")
    # merger.close()