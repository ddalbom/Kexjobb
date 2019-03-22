
# Reinforcement learning
# Multiple agents
# No deep learning

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy

print('Running')

grid_size = 4
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
    def __init__(self, dropoff, pickup, cargo = False):
        self.m = dropoff[0] # Robot position in grid (row)
        self.n = dropoff[1] # Robot position in grid (col)
        self.dropoff = dropoff # Drop off location of cargo
        self.pickup = pickup # Position of cargo to be picked up
        self.carry = cargo # True if robot carries cargo, False if not
        self.Q = dict() # Q-function for going to cargo
        self.Q2 = dict() # Q-function for bringing cargo to drop off location

    def move_robot(self, state):
        """Moves the robot according to the given action."""
        m = self.m # Current row
        n = self.n # Current col
        if [m, n] == self.dropoff and self.carry:
            # Robot has performed its duty
            step = 0 # Does not take new step
            back = True
            return state, None, None, step, back

        else:
            cur_env = deepcopy(state.grid)
            cur_env[m][n] = 0
            action = self.choose_action(state)

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
            self.m = m # Update position of robot
            self.n = n # Update position of robot
            cur_env[m][n] = 1 # Update grid
            new_state = State(cur_env, [m, n]) # Set new state

            if [m, n] == self.pickup and not self.carry: # If pickup point is reached
                self.carry = True # Pick up cargo
                self.Q2[state] = np.random.rand(len(ACTIONS))
                # Reward for picking up cargo?

            if [m, n] == self.dropoff and self.carry: # If dropoff point is reached
                Rew = 10
                self.Q2[new_state] = [0, 0, 0, 0] # Terminal state is reached

            if not self.carry:
                if new_state not in self.Q:
                    self.Q[new_state] = np.random.rand(len(ACTIONS))
            elif self.carry:
                if new_state not in self.Q2:
                    self.Q2[new_state] = np.random.rand(len(ACTIONS))

            step = 1 # Takes new step
            back = False

            return new_state, a, Rew, step, back

    def choose_action(self, state): # Given a state, chooses an action
        """Defines behavior policy as epsilon-greedy. Given a state, chooses an action."""
        prob = [] # Probability distribution
        for i in range(len(ACTIONS)):
            prob.append(eps/4)
        if not self.carry:
            Qmax = max(self.Q[state])
            for i in range(len(prob)):
                if self.Q[state][i] == Qmax:
                    prob[i] = 1 - eps + eps/4
                    break # Use if number of episodes is large # NOTE: Always picks first maximum, if two paths are eqaul ...
        elif self.carry:
            Qmax = max(self.Q2[state])
            for i in range(len(prob)):
                if self.Q2[state][i] == Qmax:
                    prob[i] = 1 - eps + eps/4
                    break # Use if number of episodes is large
        action = np.random.choice(ACTIONS, p = prob) # Chooses an action at random
        return action

def iterate(robots, E):
    """Performs one iteration, i.e. simulation of one time step."""
    terminal_list = []
    rewards = []
    steps = []
    for robot in robots:
        S = State(E, [robot.m, robot.n])
        if not robot.carry:
            if S not in robot.Q:
                robot.Q[S] = np.random.rand(len(ACTIONS))
        elif robot.carry:
            if S not in robot.Q2:
                robot.Q2[S] = np.random.rand(len(ACTIONS))

        S_new, action_number, R, step, back = robot.move_robot(S) # Moves robot

        if not back:
            if not robot.carry:
                robot.Q[S][action_number] += alpha*(R + gamma*max(robot.Q[S_new]) - robot.Q[S][action_number]) # Update Q-function
            elif robot.carry:
                robot.Q2[S][action_number] += alpha*(R + gamma*max(robot.Q2[S_new]) - robot.Q2[S][action_number]) # Update Q-function

        S = S_new
        E = S.grid # Update environment

        terminal_list.append(back)
        steps.append(step)

        print(E) # Show grid
        print()

    terminal = np.all(terminal_list)

    return E, terminal, steps

def episode(robots):
    """Simulation of one episode for multiple robots. Back and forth."""
    # Initialize E
    E = np.zeros((grid_size, grid_size), dtype = int) # Environment, i.e. the grid
    for robot in robots:
        robot.m = robot.dropoff[0] # Initialize robot position
        robot.n = robot.dropoff[1] # Initialize robot position
        E[robot.m][robot.n] = 1  # Initialize robot position on grid
        robot.carry = False # Initialize robot carry, i.e. the robot does NOT carry cargo

    counts = [0 for count in range(len(robots))] # Steps for each robot to reach destination
    rewards = [0 for rew in range(len(robots))] # Rewards for each robot
    terminal = False

    while not terminal: # While agents have not reached their terminal state
        E_new, term, steps = iterate(robots, E)
        E = E_new
        for i in range(len(counts)):
            counts[i] += steps[i]
        terminal = term

    return counts
    # return rewards

def simulation():
    """Iterates through all episodes."""
    nepisodes = []
    # Initialize robots
    robots = [] # List containing all robots
    r1 = Robot(dropoff = [0, 0], pickup = [grid_size-1, grid_size-1]) # Create robot
    r2 = Robot(dropoff = [0, grid_size-1], pickup = [grid_size-1, 0]) # Create robot
    robots.append(r1)
    robots.append(r2)
    steps_list = [[] for robot in range(len(robots))] # Store steps taken each episode for all robots
    rew_list = [[[] for robot in range(len(robots))]]
    for i in range(1000): # Choose number of episodes to run
        nsteps = episode(robots) # Number of steps and total reward
        nepisodes.append(i+1) # Episode number
        print('End of episode! Number of steps: ', nsteps)
        print('End of episode!')
        for j in range(len(nsteps)):
            steps_list[j].append(nsteps[j])
    for list in steps_list:
        plt.plot(nepisodes, list, '.')
    plt.show()


simulation() # Run one simulation
