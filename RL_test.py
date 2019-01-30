# Reinforcement learning
import numpy as np
import random as rnd
import tensorflow as tf

grid_size = 4
ACTIONS = ['Right', 'Left', 'Up', 'Down']
# Initialize Action-Value function Q(s, a)
# Q as 'cube' tensor or matrix maybe?
Q = np.random.rand(grid_size, grid_size, len(ACTIONS))
Q[-1][-1] = [0, 0, 0, 0]
eps = 0.1
gamma = 0.7
alpha = 1 # Step size, does not need to be 1

# Does use RL at the moment

class Robot:
    """Implements test robot for RL algorithm. """
    def __init__(self, row = 0, col = 0):
        self.m = row
        self.n = col
        # self.carry ... # Maybe = True if it carries load?
        # self.vx/vy # Velocity maybe?
        # Should maybe have the Q-function encoded

def move_robot(robot, state):
    """Moves the robot according to the given action."""
    global Q
    m = robot.m
    n = robot.n
    p = []
    for i in range(len(ACTIONS)):
        p.append(eps/4)
    Qmax = max(Q[m][n])
    for i in range(len(p)):
        if Q[m][n][i] == Qmax:
            p[i] = 1 - eps + eps/4
            break # Use if number of episodes is large
    cur_state = state
    cur_state[m][n] = 0
    action = choose_action(state, p)
    if action == 'Right':
        if n + 1 >= grid_size or cur_state[m][n+1] == 2:
            pass
        else:
            n += 1
        a = 0

    elif action == 'Left':
        if n - 1 < 0 or cur_state[m][n-1] == 2:
            pass
        else:
            n -= 1
        a = 1

    elif action == 'Up':
        if m - 1 < 0 or cur_state[m-1][n] == 2:
            pass
        else:
            m -= 1
        a = 2

    elif action == 'Down':
        if m + 1 >= grid_size or cur_state[m+1][n] == 2:
            pass
        else:
            m += 1
        a = 3

    m = m % grid_size
    n = n % grid_size
    robot.m = m
    robot.n = n
    cur_state[m][n] = 1
    return cur_state, a

def choose_action(state, prob): # Given a state and a probability distribution, chooses an action!
    """Defines policy to follow."""
    action = np.random.choice(ACTIONS, p = prob) # Chooses an action at random
    return action

def episode():
    """Simulation of one episode."""
    global Q
    # Initialize S
    S = np.zeros((grid_size, grid_size), dtype = int) # Initializes the state, S
    S[0][0] = 1 # Initializes position of robot
    S[0][2] = 2 # Obstacle 1
    S[2][1] = 2 # Obstacle 2
    r1 = Robot()
    count = 0
    while S[-1][-1] != 1:
        m_old = r1.m
        n_old = r1.n
        S_new, action_number = move_robot(r1, S)
        m_new = r1.m
        n_new = r1.n
        R = -1
        Q[m_old][n_old][action_number] += alpha*(R + gamma*max(Q[m_new][n_new] - Q[m_old][n_old][action_number]))
        S = S_new
        # print(S)
        # print()
        count += 1
    return count

def simulation():
    """Iterates through all episodes."""
    for i in range(100):
        nsteps = episode()
        print("End of episode!")
        print(nsteps)

simulation()
