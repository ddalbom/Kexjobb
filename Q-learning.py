import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns



grid_size = 10
ACTIONS = ['Right', 'Left', 'Up', 'Down']
gamma = 0.9
alpha = 1


nsimulations = 10 # Not in use yet
nepisodes = 10000


initial_exploration_rate = 0.9
exploration_decay_rate = 0.995
minimum_exploration_rate = 0.01


r_step = -0.1
r_collide = -2
r_goal = 5

episode_list = [i+1 for i in range(nepisodes)]

show_grid = False # True = show animation, False = dont show

number_of_agents = 2 # Use this somehow

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
        self.eps = initial_exploration_rate
        self.eps_decay = exploration_decay_rate
        self.eps_min = minimum_exploration_rate
        self.reward = 0
        self.reward_list = []
        self.collision_counter = 0
        self.collision_list = []
        self.steps = 0
        self.step_list = []
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
                Rew = r_collide
                self.collision_counter += 1
                #print('number of collisions for agent {} :'.format(self.nr),self.collide_counter)
            else:
                n += 1
                Rew = r_step
            a = 0 # Action number
        elif action == 'Left':
            if n - 1 < 0 or cur_env[m][n-1] != 0:
                Rew = r_collide
                self.collision_counter += 1
                #print('number of collisions for agent {} :'.format(self.nr),self.collide_counter)
            else:
                n -= 1
                Rew = r_step
            a = 1
        elif action == 'Up':
            if m - 1 < 0 or cur_env[m-1][n] != 0:
                Rew = r_collide
                self.collision_counter += 1
                #print('number of collisions for agent {} :'.format(self.nr),self.collide_counter)
            else:
                m -= 1
                Rew = r_step
            a = 2
        elif action == 'Down':
            if m + 1 >= grid_size or cur_env[m+1][n] != 0:
                Rew = r_collide
                self.collision_counter += 1
                #print('number of collisions for agent {} :'.format(self.nr),self.collide_counter)
            else:
                m += 1
                Rew = r_step
            a = 3

        self.m = m # Update position of robot
        self.n = n # Update position of robot
        cur_env[m][n] = 1 # Update grid
        new_state = State(cur_env, [m, n]) # Set new state
        terminal = False

        if [m, n] == self.end:
            Rew = r_goal
            terminal = True

        if new_state not in self.Q:
            self.Q[new_state] = np.random.rand(len(ACTIONS))

        return new_state, a, Rew, terminal

    def choose_action(self, state): # Given a state, chooses an action
        """Defines behavior policy as epsilon-greedy. Given a state, chooses an action."""
        prob = [] # Probability distribution
        for i in range(len(ACTIONS)):
            prob.append(self.eps/4)
            Qmax = max(self.Q[state])
            for i in range(len(prob)):
                if self.Q[state][i] == Qmax:
                    prob[i] = 1 - self.eps + self.eps/4
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
            #print('steps for agent {}: '.format(agent.nr),agent.steps)

            if agent.eps > agent.eps_min:
                agent.eps *= agent.eps_decay

            agent.reward += r

            E = S_new.grid # Update environment

            terminal_list.append(terminal)
            if show_grid == True:
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
        agent.collision_counter=0


    '''Enivornment for the 4x4 grid '''
    #E[1][1] = 2

    '''Environment for the 10x10 grid'''
    E[0][3] = 2
    E[1][3] = 2
    E[0][6] = 2
    E[1][6] = 2
    E[2][6] = 2
    E[7][3] = 2
    E[8][3] = 2
    E[9][3] = 2
    E[8][6] = 2
    E[9][6] = 2

    if show_grid == True:
        show(E)

    terminal = False

    while not terminal: # While agents have not reached their terminal state
        E_new, term = iterate(agents, E)
        E = E_new
        terminal = term

    '''Saves reward,collisions and number of steps'''
    for agent in agents:
        agent.reward_list.append(agent.reward)
        agent.collision_list.append(agent.collision_counter)
        agent.step_list.append(agent.steps)


def simulation():
    """Iterates through all episodes."""
    # Initialize robots
    agents = [] # List containing all agents
    a1 = Agent(start = [0, 0], end = [grid_size-1, grid_size-1], nr=1) # Create agent 1
    #a2 = Agent(start = [0, grid_size-1], end = [grid_size-1, 0], nr=2) # Create agent 2
    #a3 = Agent(start = [grid_size-1, 0], end = [0, grid_size-1], nr=3) # Create agent 3
    #a4 = Agent(start = [grid_size-1, grid_size-1], end = [0, 0], nr=4) # Create agent 4

    #for i in range(nagents):
        #agents.append(a)

    agents.append(a1)
    #agents.append(a2)
    #agents.append(a3)
    #agents.append(a4)


    for i in range(nepisodes): # Choose number of episodes to run
        episode(agents)
        print('End of episode ', i+1)

    '''Plots the cumulative reward for each episode'''
    plt.clf() # Removes animation window
    sns.set()
    plt.rcParams['font.size'] = 40
    for agent in agents:
        plt.plot(episode_list, agent.reward_list, label = 'Agent {}'.format(agent.nr))
    plt.xlabel('Episodes',size=25)
    plt.ylabel('Cumulative reward',size=25)
    plt.legend()
    plt.show()

    '''Plots the number of collisions for each episode'''
    for agent in agents:
        plt.plot(episode_list, agent.collision_list, label = 'Agent {}'.format(agent.nr))
    plt.xlabel('Episodes',size=25)
    plt.ylabel('Collisions',size=25)
    plt.legend()
    plt.show()

    '''Plots the number of steps for each episode'''
    for agent in agents:
        plt.plot(episode_list, agent.step_list, label = 'Agent {}'.format(agent.nr))
    plt.xlabel('Episodes',size=25)
    plt.ylabel('Steps',size=25)
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
    #ax.set_title("Episode {}".format(len(episode_list)))
    plt.pause(0.01)
    plt.clf()


'''WIP'''
def run_simulations(nagents):
    for i in range(nsimulations):
        reward_list = simulation()


simulation()
