# Reinforcement learning
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

grid_size = 4
ACTIONS = ['Right', 'Left', 'Up', 'Down']
# Initialize Action-Value function Q(s, a)
# Q as 'cube' tensor or matrix maybe?
Q = np.random.rand(grid_size, grid_size, len(ACTIONS))
Q[-1][-1] = [0, 0, 0, 0]
eps = 0.1 # Greedy policy
gamma = 0.9 # Discount factor
alpha = 1 # Step size, does not need to be 1

# Does use RL at the moment

# Used to plot the time evolution of the system
step_list = []
episode_list = []

class Robot:
    """Implements test robot for RL algorithm. """
    def __init__(self, row = 0, col = 0):
        self.m = row
        self.n = col
        # self.carry ... # Maybe = True if it carries load?
        # Should maybe have the Q-function encoded

def move_robot(robot, state):
    """Moves the robot according to the given action."""
    global Q
    m = robot.m
    n = robot.n
    p = []   # Probability for each action in current state
    for i in range(len(ACTIONS)):
        p.append(eps/4)
    Qmax = max(Q[m][n])  # Picks the action with highest Q-value in our current state
    for i in range(len(p)):
        if Q[m][n][i] == Qmax:
            p[i] = 1 - eps + eps/4
            break # Use when two actions got the same Q-value
    cur_state = state
    cur_state[m][n] = 0 # leaves or current state before making a step
    action = choose_action(state, p) # Choose an random action weighted with the actions probabilities
    if action == 'Right':
        if n + 1 >= grid_size or cur_state[m][n+1] == 2:  # Checks if there are a wall or obstacle to the right
            pass
        elif cur_state[m][n+1] == 4 and cur_state[-1][-1] != 0: # Need to have picked up the object from A before dropping it on B
            pass
        else:
            n += 1 # make a step to the right
        a = 0  # 0 represents Right

    elif action == 'Left':
        if n - 1 < 0 or cur_state[m][n-1] == 2: # Check if there are a wall or obstacle to the left
            pass
        elif cur_state[m][n-1] == 4 and cur_state[-1][-1] != 0: # Need to have picked up the object from A before dropping it on B
            pass
        else:
            n -= 1 # make a step to the left
        a = 1 # 1 represents Left

    elif action == 'Up':
        if m - 1 < 0 or cur_state[m-1][n] == 2: # Check if there are a wall or ostacle above the agent
            pass
        elif cur_state[m-1][n] == 4 and cur_state[-1][-1] != 0: # Need to have picked up the object from A before dropping it on B
            pass
        else:
            m -= 1 # make a step up
        a = 2 # 2 represents up

    elif action == 'Down':
        if m + 1 >= grid_size or cur_state[m+1][n] == 2: # Check if ther are a wall or ostacle under the agent
            pass
        elif cur_state[m+1][n] == 4 and cur_state[-1][-1] != 0: # Need to have picked up the object from A before dropping it on B
            pass
        else:
            m += 1 # Make a step to down
        a = 3 # 4 represents down
    #print("Action taken :",action) # Used for troubleshooting
    robot.m = m # Updates robot
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
    S[0][2] = 2 # Initializes position of obstacle 1
    S[2][1] = 2 # Initializes position of obstacle 2
    S[-1][-1] = 3 # Point A
    S[2][0] = 4 # Point B
    r1 = Robot()
    count = 0
    while S[-1][-1] == 3 or S[2][0] == 4: # Take a new step as long as the agent haven't made it from A to B
        m_old = r1.m # Saves old state
        n_old = r1.n
        S_new, action_number = move_robot(r1, S)
        m_new = r1.m # New state
        n_new = r1.n
        R = -1  # -1 for moving to each state
        Q[m_old][n_old][action_number] += alpha*(R + gamma*max(Q[m_new][n_new] - Q[m_old][n_old][action_number])) # Q-learning algorithm, assigns a Q-value for the action we took
        S = S_new # Updates state
        #print(S) # Used to check what path the robot is taking
        #print()
        count += 1
    return count

def simulation():
    """Iterates through all episodes."""
    for i in range(100):
        nsteps = episode()
        print("End of episode!")
        print("number of steps :",nsteps)
        step_list.append(nsteps)
        episode_list.append(i+1)


simulation()

plt.plot(episode_list,step_list)
plt.show()
