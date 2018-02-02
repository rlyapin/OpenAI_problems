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
        self.blocks = np.array(SPAWN_DICT[self.letter])
        self.orientation = 0

    def do_nothing(self):
        pass

    def move_left(self):
        self.blocks += np.array([[-1, 0], [-1, 0], [-1, 0], [-1, 0]])

    def move_right(self):
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

    def reset(self, letter):
        self.letter = letter
        self.blocks = np.array(SPAWN_DICT[self.letter])
        self.orientation = 0


class TetrisEnv:
    # The base class implementing tetris RL environment
    # By analogy with OpenAI gym, it should have reset() and step() methods
    # The way this environment is supposed to work:
    # After the enironment is reset the player sees [10, 20] grid filled with 0-1 values
    # Is it assumed 0 values are empty, 1 values are filled with fallen or falling blocks
    # Each steps contains player making an action and falling block moving one row down
    # Possible moves: do nothing, move left, move right, rotate clockwise
    # Player gets rewards for each filled row
    # The mechanics of the game is supposed to mimic original tetris with a hard cap on the number of total steps
    def __init__(self, max_steps):
        # Defining tetris internals: note that I separate falling block from the rest of the field
        self.iter = 0
        self.max_steps = max_steps
        self.action_map = {0: Tetromino.do_nothing, 
                           1: Tetromino.move_left,
                           2: Tetromino.move_right,
                           3: Tetromino.rotate_clockwise}
        self.reverse_action_map = {0: Tetromino.do_nothing, 
                                   1: Tetromino.move_right,
                                   2: Tetromino.move_left,
                                   3: Tetromino.rotate_anticlockwise}
        self.falling_tetromino = Tetromino(np.random.choice(TETROMINO_LETTERS))
        self.field = np.zeros((10, 20))
        self.done = False
        self.info = ""

    def show_frame(self):
        # Showing current status of the game
        # As falling tetronimo and the rest of the screen are separated I need to impose one on another
        # Additionally I transpose and reflect the "image" to make it more similar to actual game
        output_frame = np.array(self.field)
        for block in self.falling_tetromino.blocks:
            if block[1] < 20:
                output_frame[block[0], block[1]] = 1
        return np.transpose(output_frame[:, ::-1])

    def reset(self):
        # Reseting the environment
        # In order to mimic OpenAI gym environments I also return the first frame of the new game
        self.iter = 0
        self.falling_tetromino = Tetromino(np.random.choice(TETROMINO_LETTERS))
        self.field = np.zeros((10, 20))
        self.done = False
        return self.show_frame()

    def check_feasibility(self):
        # The method that checks whether falling tetromino performed a valid move 
        # Used to confirm agent actions and decide whether to freeze the tetromino if it reached floor
         # Checking if tetromino is inside the playing field      
        for block in self.falling_tetromino.blocks:
            if block[0] < 0:
                return False
            if block[0] > 9:
                return False
            if block[1] < 0:
                return False
        # Checking the intersections with the frozen blocks
        for block in self.falling_tetromino.blocks:
            if block[1] < 20:
                if self.field[block[0], block[1]] == 1:
                    return False
        # No collisions so far so the move is valid
        return True

    def clear_filled_rows(self):
        # Function to confirm reward and get rid of filled rows
        filled_indices = np.nonzero(np.sum(self.field, axis=0) == 10)[0]
        remaining_indices = np.nonzero(np.sum(self.field, axis=0) < 10)[0]
        reward = len(filled_indices)
        if reward > 0:
            self.field = np.hstack([self.field[:, remaining_indices], np.zeros((10, reward))])
        return reward

    def step(self, action):
        # The heart of the environment
        # In order to mimic OpenAI gym environments I return (state, reward, done) info
        # Possible actions:
        # 0 - do nothing
        # 1 - move left
        # 2 - move right
        # 3 - rotate clockwise
        if self.done:
            final_frame = self.show_frame()
            return final_frame, 0, self.done, self.info
        else:
            # Updating number of iterations and stopping the game if necessary
            self.iter += 1
            if self.iter == self.max_steps:
                self.done = True

            # Executing players agent and reverting it if it is not valid
            self.action_map[action](self.falling_tetromino)
            if self.check_feasibility() == False:
                self.reverse_action_map[action](self.falling_tetromino)

            # Executing environment action: trying to move falling tetromino down one row
            # If it is valid no other actions are required
            self.falling_tetromino.move_down()
            if self.check_feasibility() == True:
                return_frame = self.show_frame()
                return return_frame, 0, self.done, self.info

            # If it is invalid that means the floor or other block below are reached:
            # In this case I reverse the down move and add the tetromino to the playing field
            else:
                self.falling_tetromino.move_up()
                for block in self.falling_tetromino.blocks:
                    if block[1] < 20:
                        self.field[block[0], block[1]] = 1  


                # Then I need to check whether I can erase some filled rows (and get rewards)        
                reward = self.clear_filled_rows()    

                # After that I introduce new falling piece and check for possible game over
                self.falling_tetromino = Tetromino(np.random.choice(TETROMINO_LETTERS))
                if self.check_feasibility() == False:
                    self.done = True

                return_frame = self.show_frame()
                return return_frame, reward, self.done, self.info

# Small script to test game execution
# import time
# test = TetrisEnv(1000)
# for _ in range(1000):
#     test.step(np.random.choice(range(4)))
#     print test.show_frame()
#     time.sleep(0.1)
#     if test.done:
#         break

