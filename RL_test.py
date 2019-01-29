# Reinforcement learning
import numpy as np
import random as rnd
import tensorflow as tf

domain = np.zeros((3, 3), dtype = int)

# Does not use RL at the moment

class Robot:
    """Implements test robot for RL algorithm. """
    def __init__(self, xpos = 0, ypos = 0):
        self.x = xpos
        self.y = ypos

def move_robot(robot):
    """Moves the robot according to the given action."""
    global domain
    x = robot.x
    y = robot.y
    domain[x][y] = 0
    move = policy(domain)
    if move == 'Right':
        x += 1
    elif move == 'Left':
        x -= 1
    elif move == 'Up':
        y += 1
    elif move == 'Down':
        y -= 1
    robot.x = x
    robot.y = y
    domain[x][y] = 1

def policy(state): # Given a state, what is the probability of performing action a?
    """Defines policy to follow."""
    if state[2][2] != 1: # Robot is not at end point
        r = rnd.random()
        if r <= 0.24:
            action = 'Right'
        elif r <= 0.50 and r > 0.24:
            action = 'Left'
        elif r <= 0.74 and r > 0.50:
            action = 'Up'
        elif r > 0.74:
            action = 'Down'
    else:
        pass

    return action

r1 = Robot()
while domain[2][2] != 1:
    move_robot(r1)
    print(domain)
