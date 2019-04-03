import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns

grid_size = 5
ACTIONS = ['Right', 'Left', 'Up', 'Down']
eps = 0.1
gamma = 0.9
alpha = 1
episode_list = []

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


class Agent:
    """Implements agent. """
    def __init__(self, start, end, nr):
        self.m = start[0]
        self.n = start[1]
        self.start = start
        self.end = end
        self.nr = nr
        self.steps = 0
        self.reward = 0
        self.Q = dict() # Q-function

    def move_agent(self, state):
        """Moves the robot according to the given action."""
        m = self.m # Current row
        n = self.n # Current col

        cur_env = deepcopy(state.grid)
        cur_env[m][n] = 0
        action = self.choose_action(state)

        if action == 'Right':
            if n + 1 >= grid_size or cur_env[m][n+1] != 0:
                Rew = -5 # Reward -5 if we move into wall or another agent
            else:
                n += 1
                Rew = -1 # Reward -1 otherwise
            a = 0 # Action number
        elif action == 'Left':
            if n - 1 < 0 or cur_env[m][n-1] != 0:
                Rew = -5
            else:
                n -= 1
                Rew = -1
            a = 1
        elif action == 'Up':
            if m - 1 < 0 or cur_env[m-1][n] != 0:
                Rew = -5
            else:
                m -= 1
                Rew = -1
            a = 2
        elif action == 'Down':
            if m + 1 >= grid_size or cur_env[m+1][n] != 0:
                Rew = -5
            else:
                m += 1
                Rew = -1
            a = 3

        self.m = m # Update position of robot
        self.n = n # Update position of robot
        cur_env[m][n] = 1 # Update grid
        new_state = State(cur_env, [m, n]) # Set new state
        terminal = False

        if [m, n] == self.end:
            Rew = 10
            terminal = True

        if new_state not in self.Q:
            self.Q[new_state] = np.random.rand(len(ACTIONS))

        return new_state, a, Rew, terminal

    def choose_action(self, state): # Given a state, chooses an action
        """Defines behavior policy as epsilon-greedy. Given a state, chooses an action."""
        prob = [] # Probability distribution
        for i in range(len(ACTIONS)):
            prob.append(eps/4)
            Qmax = max(self.Q[state])
            for i in range(len(prob)):
                if self.Q[state][i] == Qmax:
                    prob[i] = 1 - eps + eps/4
                    break # Use if number of episodes is large # NOTE: Always picks first maximum, if two paths are eqaul ...
        action = np.random.choice(ACTIONS, p = prob) # Chooses an action at random
        return action

def iterate(agents, E):
    """Performs one iteration, i.e. simulation of one time step."""
    terminal_list = []
    for agent in agents:
        S = State(E, [agent.m, agent.n])
        if [agent.m,agent.n] != agent.end:
            if S not in agent.Q:
                agent.Q[S] = np.random.rand(len(ACTIONS))

            S_new, action_number, r, terminal = agent.move_agent(S) # Moves agent

            agent.Q[S][action_number] += alpha*(r + gamma*max(agent.Q[S_new]) - agent.Q[S][action_number]) # Update Q-function

            agent.steps += 1
            print('steps for agent {}: '.format(agent.nr),agent.steps)

            agent.reward += r

            E = S_new.grid # Update environment

            terminal_list.append(terminal)

            show(E) # Show grid

    terminal = np.all(terminal_list)

    return E, terminal

def episode(agents):
    """Simulation of one episode for multiple robots. Back and forth."""
    # Initialize E
    E = np.zeros((grid_size, grid_size), dtype = int) # Environment, i.e. the grid
    for agent in agents:
        agent.m = agent.start[0] # Initialize robot position
        agent.n = agent.start[1] # Initialize robot position
        E[agent.m][agent.n] = 1  # Initialize robot position on grid
        agent.steps = 0
        agent.reward = 0

    show(E)

    terminal = False

    while not terminal: # While agents have not reached their terminal state
        E_new, term = iterate(agents, E)
        E = E_new
        terminal = term


def simulation():
    """Iterates through all episodes."""
    # Initialize robots
    agents = [] # List containing all agents
    a1 = Agent(start = [0, 0], end = [grid_size-1, grid_size-1], nr=1) # Create agent 1
    a2 = Agent(start = [0, grid_size-1], end = [grid_size-1, 0], nr=2) # Create agent 2
    agents.append(a1)
    agents.append(a2)
    reward_list = [[] for i in range(len(agents))]

    for i in range(100): # Choose number of episodes to run
        episode(agents) #
        episode_list.append(i+1) # Episode number
        #print('End of episode! Number of steps: ', nsteps)
        #print('End of episode!')
        print('End of episode ', i+1)
        for agent in agents:
            reward_list[agent.nr-1].append(agent.reward)


    '''Plots the results'''
    plt.clf() # Removes animation window
    sns.set()
    agent_nr = 1
    for list in reward_list:
        plt.plot(episode_list, list, label = 'Agent {}'.format(agent_nr))
        agent_nr += 1
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.legend()
    plt.show()

def show(E):
    plt.grid('on')
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, 10, 1))
    ax.set_yticks(np.arange(0.5, 10, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(E, cmap='binary')
    ax.set_title("Episode {}".format(len(episode_list)+1))
    plt.pause(0.01)
    plt.clf()

simulation() # Run one simulation
