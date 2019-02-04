# RL algorithm with obstacles

# Reinforcement learning
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import tensorflow as tf

grid_size = 4
m_A = 0
n_A = 0
m_B = grid_size - 1
n_B = grid_size - 1
ACTIONS = ['Right', 'Left', 'Up', 'Down']
# Initialize Action-Value function Q(s, a)
# Q as 'cube' tensor or matrix maybe?
Q = np.random.rand(grid_size, grid_size, len(ACTIONS)) # Q-function for moving from A to B
# Q[m_B][n_B] = [0, 0, 0, 0] # Should not be zero??
Q2 = np.random.rand(grid_size, grid_size, len(ACTIONS)) # Q-function for moving from B to A
Q2[m_A][n_A] = [0, 0, 0, 0] # Should be zero??
eps = 0.1
gamma = 0.7
alpha = 1 # Step size, does not need to be 1


class Robot:
    """Implements test robot for RL algorithm. """
    def __init__(self, row = m_A, col = n_A, cargo = False):
        self.m = row
        self.n = col
        self.carry = cargo
        # self.vx/vy # Velocity maybe?
        # Should maybe have the Q-function encoded

def move_robot(robot, state):
    """Moves the robot according to the given action."""
    global Q, Q2
    m = robot.m
    n = robot.n
    p = []
    for i in range(len(ACTIONS)):
        p.append(eps/4)
    if robot.carry is False: # If the robot is moving from A to B
        Qmax = max(Q[m][n])
        for i in range(len(p)):
            if Q[m][n][i] == Qmax:
                p[i] = 1 - eps + eps/4
                break # Use if number of episodes is large
    elif robot.carry is True: # If robot is moving from B to A
        Qmax = max(Q2[m][n])
        for i in range(len(p)):
            if Q2[m][n][i] == Qmax:
                p[i] = 1 - eps + eps/4
                break # Use if number of episodes is large # Randomize which 'path' to choose
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
    # if state[-1][-1] != 1: # Robot is not at end point
    action = np.random.choice(ACTIONS, p = prob) # Chooses an action at random
    return action

def episode():
    """Simulation of one episode."""
    global Q, Q2
    # Initialize S
    S = np.zeros((grid_size, grid_size), dtype = int) # Initializes the state, S
    S[m_A][n_A] = 1 # Initializes position of robot
    S[0][2] = 2
    S[2][1] = 2
    S[2][2] = 2
    r1 = Robot()
    print(S)
    print()
    count = 0
    while r1.carry is False:
        m_old = r1.m
        n_old = r1.n
        S_new, action_number = move_robot(r1, S)
        m_new = r1.m
        n_new = r1.n
        if m_new != m_B or n_new != n_B:
            R = -1
        else:
            R = 5
            r1.carry = True # Picks up cargo
        Q[m_old][n_old][action_number] += alpha*(R + gamma*max(Q[m_new][n_new] - Q[m_old][n_old][action_number]))
        S = S_new
        print(S)
        print()
        count += 1
        # print(r1.carry)
    while r1.carry is True:
        m_old = r1.m
        n_old = r1.n
        S_new, action_number = move_robot(r1, S)
        m_new = r1.m
        n_new = r1.n
        if m_new != m_A or n_new != n_A:
            R = -1
        else:
            R = 5
            r1.carry = False # Drops off cargo
        Q2[m_old][n_old][action_number] += alpha*(R + gamma*max(Q2[m_new][n_new] - Q2[m_old][n_old][action_number]))
        S = S_new
        print(S)
        print()
        count += 1
        # print(r1.carry)

    return count

nepisodes = []
step_list = []

def simulation():
    """Iterates through all episodes."""
    for i in range(200):
        nsteps = episode()
        nepisodes.append(i+1)
        step_list.append(nsteps)
        print("End of episode!")
        print(nsteps)

simulation()

plt.plot(nepisodes, step_list, '.')
plt.show()
