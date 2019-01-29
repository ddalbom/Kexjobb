# Reinforcement learning
import numpy as np
import random as rnd

state = np.zeros((4, 4), dtype = int)
ACTIONS = ['right','left','up','down'] # available actions


# Does not use RL at the moment
class Robot:
    """Implements test robot for RL algorithm. """
    def __init__(self, xpos = 0, ypos = 0):
        self.x = xpos
        self.y = ypos


def move_robot(robot):
    """Moves the robot according to the given action."""
    global state
    x = robot.x
    y = robot.y
    action = choose_action(state)
    if action == 'right':
        if state[0][3] == 1 or state [1][3] == 1 or state[2][3] == 1 or state [3][3] == 1: # Check if we are at the right side of the matrix
            pass
        else:
            state[x][y] = 0
            y += 1
    elif action == 'left':
        if state[0][0] == 1 or state [1][0] == 1 or state[2][0] == 1 or state [3][0] == 1: # Checks if we are at the left side of matrix
            pass
        else:
            state[x][y] = 0
            y = y - 1
    elif action == 'up':
        if state[0][0] == 1 or state [0][1] == 1 or state[0][2] == 1 or state [0][3] == 1: # Checks if we are at the top of the matrix
            pass
        else:
            state[x][y] = 0
            x = x - 1
    elif action == 'down':
        if state[3][0] == 1 or state [3][1] == 1 or state[3][2] == 1 or state [3][3] == 1: # Checks if we are at the bottom of the matrix
            pass
        else:
            state[x][y] = 0
            x += 1
    robot.x = x
    robot.y = y
    state[x][y] = 1

def choose_action(state): # Given a state, what is the probability of performing action a?
    """Defines policy to follow."""
    if state[-1][-1] != 1: # Robot is not at end point
        action = np.random.choice(ACTIONS) # Choose a random action
    return action

r1 = Robot()
state[0][0] = 1 # Our initial state
count = 0
print(state)

while state[-1][-1] != 1:
    move_robot(r1) # Moves the robot randomly one step
    print(state)  # Prints out the current state of the system
    print()
    count += 1  # Counts how many steps is needed

print('number of steps: ',count)
