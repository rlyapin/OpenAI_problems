# The following script implements tetris environemnt to feed to RL algorithms

import numpy as np

# Specifying the list of tetrominos (as in wikipedia)
# I: xxxx J: xxx L: xxx O: xx S:  xx T: xxx Z: xx
#              x    x      xx    xx      x      xx
TETROMINO_LETTERS = ["I", "J", "L", "O", "S", "T", "Z"]

# Each figure is represented as a numpy matrix where each row holds coordinates of building blocks
# The following dict specifies the spawning coordinates of figures (assuming 10x20 tetris grid) 
SPAWN_DICT = {"I": np.array([[3, 19], [4, 19], [5, 19], [6, 19]]),
              "J": np.array([[3, 19], [4, 19], [5, 19], [5, 18]]),
              "L": np.array([[3, 18], [3, 19], [4, 19], [5, 19]]),
              "O": np.array([[4, 19], [5, 19], [5, 18], [4, 18]]),
              "S": np.array([[3, 18], [4, 18], [4, 19], [5, 19]]),
              "T": np.array([[3, 19], [4, 19], [5, 19], [4, 18]]),
              "Z": np.array([[3, 19], [4, 19], [4, 18], [5, 18]])}

# During the game the agent has the option to rotate the falling figure
# The details of how that could work are gven here: http://tetris.wikia.com/wiki/SRS
# Such rotationss can be respesented with matrix additions
# Each figure in general has several of them that sum up to zero (returning to original position)
ROTATION_DICT = {}
ROTATION_DICT["I"] = [np.array([[1, 2], [0, 1], [-1, 0], [-2, -1]]),
                      np.array([[2, -1], [1, 0], [0, 1], [-1, 2]]),
                      np.array([[-1, -2], [0, -1], [1, 0], [2, 1]]),
                      np.array([[-2, 1], [-1, 0], [0, -1], [1, -2]])]


ROTATION_DICT["J"] = [np.array([[1, 1], [0, 0], [-1, -1], [-2, 0]]),
                      np.array([[1, -1], [0, 0], [-1, 1], [0, 2]]),
                      np.array([[-1, -1], [0, 0], [1, 1], [2, 0]]),
                      np.array([[-1, 1], [0, 0], [1, -1], [0, -2]])]

ROTATION_DICT["L"] = [np.array([[0, 2], [1, 1], [0, 0], [-1, -1]]),
                      np.array([[2, 0], [1, -1], [0, 0], [-1, 1]]),
                      np.array([[0, -2], [-1, -1], [0, 0], [1, 1]]),
                      np.array([[-2, 0], [-1, 1], [0, 0], [1, -1]])]

ROTATION_DICT["O"] = [np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                      np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                      np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                      np.array([[0, 0], [0, 0], [0, 0], [0, 0]])]

ROTATION_DICT["S"] = [np.array([[0, 2], [-1, 1], [0, 0], [-1, -1]]),
                      np.array([[2, 0], [1, 1], [0, 0], [-1, 1]]),
                      np.array([[0, -2], [1, -1], [0, 0], [1, 1]]),
                      np.array([[-2, 0], [-1, -1], [0, 0], [1, -1]])]

ROTATION_DICT["T"] = [np.array([[1, 1], [0, 0], [-1, -1], [-1, 1]]),
                      np.array([[1, -1], [0, 0], [-1, 1], [1, 1]]),
                      np.array([[-1, -1], [0, 0], [1, 1], [1, -1]]),
                      np.array([[-1, 1], [0, 0], [1, -1], [-1, -1]])]

ROTATION_DICT["Z"] = [np.array([[1, 1], [0, 0], [-1, 1], [-2, 0]]),
                      np.array([[1, -1], [0, 0], [1, 1], [0, 2]]),
                      np.array([[-1, -1], [0, 0], [1, -1], [2, 0]]),
                      np.array([[-1, 1], [0, 0], [-1, -1], [0, -2]])]

# Defining a class for tetrominos (instance of the class would correspond to active falling pieces)
class Tetromino:
    # The base class holds the details of tetrominos (coordinates of constructing blocks)
    # Additionally it handles the manipulations with tetrominos (rotations and movements) 
    def __init__(self, letter):
        # Recording the tetromino class, fetching coordinates and assigning orientation (wrt clockwise rotation)
        self.letter = letter
        self.blocks = SPAWN_DICT[self.letter]
        self.orientation = 0

    def move_left(self):
        if min(self.blocks[:, 0]) > 0:
            self.blocks += np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]])

    def move_right(self):
        if max(self.blocks[:, 0]) < 9:
            self.blocks += np.array([[1, 0], [1, 0], [1, 0], [1, 0]])

    def move_down(self):
        self.blocks += np.array([[0, -1], [0, -1], [0, -1], [0, -1]])

    def move_up(self):
        self.blocks += np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        
    def rotate_clockwise(self):
        self.blocks += ROTATION_DICT[self.letter][self.orientation]
        self.orientation = (self.orientation + 1) % 4

    def rotate_anticlockwise(self):
        self.orientation = (self.orientation - 1) % 4
        self.blocks -= ROTATION_DICT[self.letter][self.orientation]






