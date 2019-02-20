# Two agents
# Reinforcement learning
# No deep learning

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy

print('Running')

grid_size = 4
m_A = 0 # Start coordinate
n_A = 0 # End coordinate
m_B = grid_size - 1 # End coordinate
n_B = grid_size - 1 # End coordinate
ACTIONS = ['Right', 'Left', 'Up', 'Down']
eps = 0.1
gamma = 0.7
alpha = 1

class State:
    """Defines the current state of the agent."""
    def __init__(self, grid, pos):
        self.grid = grid
        self.pos = pos

    def __eq__(self, other):
        """Override the default Equals behaviour."""
        return np.all(self.grid == other.grid) and np.all(self.pos == other.pos)

    def __ne__(self, other):
        """Override the default Unequal behaviour."""
        return np.all(self.grid != other.grid) or np.all(self.pos != other.pos)

    def __hash__(self):
        return hash(str(self.grid) + str(self.pos))


class Robot:
    """Implements agent. """
    def __init__(self, start, end, cargo = False):
        self.m = start[0] # Robot position in grid (row)
        self.n = start[1] # Robot position in grid (col)
        self.start = start
        self.end = end
        self.carry = cargo # True if robot carries cargo, False if not
        self.Q = dict()
        self.Q2 = dict()

    def move_robot(self, state):
        """Moves the robot according to the given action."""
        m = self.m # Current row
        n = self.n # Current col
        if [m, n] == [self.end]:
            pass
        else:
            p = [] # Probability distribution
            for i in range(len(ACTIONS)):
                p.append(eps/4)
            Qmax = max(self.Q[state])
            for i in range(len(p)):
                if self.Q[state][i] == Qmax:
                    p[i] = 1 - eps + eps/4
                    break # Use if number of episodes is large
            cur_env = deepcopy(state.grid)
            cur_env[m][n] = 0
            action = choose_action(p)
            if action == 'Right':
                if n + 1 >= grid_size or cur_env[m][n+1] == 1:
                    Rew = -5 # Reward -5 if we move into wall or another agent
                else:
                    n += 1
                    Rew = -1 # Reward -1 otherwise
                a = 0 # Action number
            elif action == 'Left':
                if n - 1 < 0 or cur_env[m][n-1] == 1:
                    Rew = -5
                else:
                    n -= 1
                    Rew = -1
                a = 1
            elif action == 'Up':
                if m - 1 < 0 or cur_env[m-1][n] == 1:
                    Rew = -5
                else:
                    m -= 1
                    Rew = -1
                a = 2
            elif action == 'Down':
                if m + 1 >= grid_size or cur_env[m+1][n] == 1:
                    Rew = -5
                else:
                    m += 1
                    Rew = -1
                a = 3
            m = m % grid_size
            n = n % grid_size
            self.m = m
            self.n = n
            cur_env[m][n] = 1
            new_state = State(cur_env, [m, n])
            if new_state not in self.Q:
                self.Q[new_state] = np.random.rand(len(ACTIONS))
            return new_state, a, Rew

def choose_action(prob): # Given a probability distribution, chooses an action!
    """Defines policy to follow."""
    action = np.random.choice(ACTIONS, p = prob) # Chooses an action at random
    return action

def epi(robots):
    """Simulation of one episode for multiple robots."""
    # Initialize E
    E = np.zeros((grid_size, grid_size), dtype = int) # environment, i.e. the grid
    for robot in robots:
        robot.m = robot.start[0]
        robot.n = robot.start[1]
        E[robot.m][robot.n] = 1
        robot.carry = False

    counts = [0 for count in range(len(robots))] # Steps for each robot to reach destination
    rewards = [0 for rew in range(len(robots))] # Rewards for each robot
    help_list = [robot.carry for robot in robots]
    while not np.all(help_list):
        help_ind = 0
        for robot in robots:
            S = State(E, [robot.m, robot.n])
            if S not in robot.Q:
                robot.Q[S] = np.random.rand(len(ACTIONS))
            S_new, action_number, R = robot.move_robot(S)
            if robot.carry is False:
                counts[help_ind] += 1
            if robot.m != robot.end[0] or robot.n != robot.end[1]: # Should give reward in move function
                # rewards[help_ind] += R
                pass
            else:
                R = 10
                robot.carry = True
                # rewards[help_ind] += R # Change this
            robot.Q[S][action_number] += alpha*(R + gamma*max(robot.Q[S_new]) - robot.Q[S][action_number]) # Update Q-function
            S = S_new
            E = S.grid # Update environment
            print(E)
            print()
            help_list[help_ind] = robot.carry
            help_ind += 1

    return counts, rewards

def episode(robot):
    """Simulation of one episode."""
    # Initialize position of robot
    Rtot = 0
    robot.m = m_A
    robot.n = n_A
    robot.carry = False
    # Initialize E, S
    E = np.zeros((grid_size, grid_size), dtype = int) # Initializes the environment, E
    E[robot.m][robot.n] = 1 # Initializes position of robot
    S = State(E) # Initializes state of robot
    robot.Q[S] = np.random.rand(len(ACTIONS))
    count = 0

    while robot.carry is False:
        S_new, action_number = robot.move_robot(S)
        m_new = robot.m
        n_new = robot.n
        if m_new != m_B or n_new != n_B:
            R = -1
        else:
            R = 10
            robot.carry = True # Picks up cargo
        robot.Q[S][action_number] += alpha*(R + gamma*max(robot.Q[S_new]) - robot.Q[S][action_number])
        S = S_new
        count += 1
        Rtot += R

    robot.Q2[S] = np.random.rand(len(ACTIONS))
    while robot.carry is True:
        S_new, action_number = robot.move_robot(S)
        m_new = robot.m
        n_new = robot.n
        if m_new != m_B or n_new != n_B:
            R = -1
        else:
            R = 10
            robot.carry = False # Drops off cargo
        robot.Q2[S][action_number] += alpha*(R + gamma*max(robot.Q2[S_new]) - robot.Q2[S][action_number])
        S = S_new
        count += 1
        Rtot += R

    return count, Rtot

nepisodes = []
step_list1 = []
step_list2 = []
step_list3 = []
rew_list1 = []
rew_list2 = []

def simulation():
    """Iterates through all episodes."""
    # Initialize robots
    r1 = Robot(start = [0, 0], end = [grid_size-1, grid_size-1])
    r2 = Robot(start = [0, grid_size-1], end = [grid_size-1, 0])
    # r3 = Robot(start = [0, grid_size//2], end = [grid_size-1, grid_size//2])
    for i in range(1000):
        nsteps, rtot = epi([r1, r2])
        nepisodes.append(i+1)
        step_list1.append(nsteps[0])
        step_list2.append(nsteps[1])
        # step_list3.append(nsteps[2])
        # rew_list1.append(rtot[0])
        # rew_list2.append(rtot[1])
        print('End of episode! Number of steps: ', nsteps)

simulation()


plt.plot(nepisodes, step_list1, '.b')
plt.plot(nepisodes, step_list2, '.r')
# plt.plot(nepisodes, step_list3, '.g')
plt.show()
# plt.plot(nepisodes, rew_list1, '-b')
# plt.plot(nepisodes, rew_list2, '-r')
# plt.show()
