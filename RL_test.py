# Reinforcement learning
import numpy as np
import random as rnd
import tensorflow as tf

grid_size = 3
domain = np.zeros((grid_size, grid_size), dtype = int)

ACTIONS = ['Right', 'Left', 'Up', 'Down']

# Does not use RL at the moment

class Robot:
    """Implements test robot for RL algorithm. """
    def __init__(self, xpos = 0, ypos = 0):
        self.x = xpos
        self.y = ypos
        # self.carry ... # Maybe = True if it carries load?
        # self.vx/vy # Velocity maybe?

def move_robot(robot):
    """Moves the robot according to the given action."""
    global domain
    x = robot.x
    y = robot.y
    domain[x][y] = 0
    p = [0.25, 0.25, 0.25, 0.25]
    action = choose_action(domain, p)
    if action == 'Right':
        if x + 1 >= grid_size:
            pass
        else:
            x += 1
    elif action == 'Left':
        if x - 1 < 0:
            pass
        else:
            x -= 1
    elif action == 'Up':
        if y + 1 >= grid_size:
            pass
        else:
            y += 1
    elif action == 'Down':
        if y - 1 < 0:
            pass
        else:
            y -= 1
    x = x % grid_size
    y = y % grid_size
    robot.x = x
    robot.y = y
    domain[x][y] = 1

def choose_action(state, prob): # Given a state, what is the probability of performing action a?
    """Defines policy to follow."""
    if state[-1][-1] != 1: # Robot is not at end point
        action = np.random.choice(ACTIONS, p = prob) # Choose an action at random
    return action

r1 = Robot() # Create a test robot
count = 0 # Counts the number of steps taken

while domain[2][2] != 1: # Unless
    move_robot(r1)
    print(domain)
    print([r1.x, r1.y])
    count += 1

print('Count = ', count)
