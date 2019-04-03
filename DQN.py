# Deep learning for multiple agents
# Deep Q network
# Lets go!

# Add in obstacles

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D
from keras.optimizers import Adam, RMSprop
from collections import deque
from copy import deepcopy
import seaborn as sns


grid_size = 5
ACTIONS = ['Right', 'Left', 'Up', 'Down']
gamma = 0.9 # Discount factor
alpha = 0.01 # Learning rate
batch_size = 32 # Size of training sample

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
    """Defines the agent."""
    def __init__(self, start, end, nr):
        self.m = start[0]
        self.n = start[1]
        self.start = start
        self.end = end
        self.nr = nr
        self.steps = 0
        self.reward = 0
        self.memory  = deque(maxlen=2000) # Memory for storing experiences, maximum capacity 2000
        self.model = self.build_model() # Create a neural network for each agent
        self.eps = 0.9 # 90% at start, decrease after episode instead of after step
        self.eps_min = 0.01
        self.eps_decay = 0.995

# implement CNN
# hubber loss

    def build_model(self): # Pass state_size and action_size
        """Create neural network of agent."""
        model = Sequential()

        '''Dense layer'''
        model.add(Dense(24, input_dim = grid_size**2+2, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(len(ACTIONS), activation = 'linear'))
        model.compile(loss = 'mse', optimizer = RMSprop(lr = alpha))

        '''Convolutional layer WIP'''
        #model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), activation='relu',input_shape= grid_size**2+2))

        return model

    def train_network(self, batch):
        """Train the neural network using sampled batch of experiences from replay memory."""
        for exp in batch:
            S = exp[0]
            S = process_state(S)
            action_number = exp[1]
            r = exp[2]
            S_new = exp[3]
            S_new = process_state(S_new)
            terminal = exp[4]

            if not terminal: # If agent is not at its final destination
                target = (r + gamma*np.amax(self.model.predict(S_new)[0]))
            else:
                target = r
            target_f = self.model.predict(S)
            target_f[0][action_number] = target # Update something???
            self.model.fit(S, target_f, epochs=1, verbose=0) # Train network

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            print('Exploration rate for agent {}'.format(self.nr),self.eps)


    def move_agent(self, state):
        """Move agent one step."""
        m = self.m
        n = self.n

        cur_env = deepcopy(state.grid)
        cur_env[m, n] = 0
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

        self.m = m # Update position of agent
        self.n = n # Update position of agent
        cur_env[m][n] = 1 # Update grid
        new_state = State(cur_env, [m, n]) # Set new state
        terminal = False

        if [m, n] == self.end:
            Rew = 10
            terminal = True

        return new_state, a, Rew, terminal

    def choose_action(self, state):
        """Choose an action depending on the given state. """
        prob = [] # Probability distribution
        for i in range(len(ACTIONS)):
            prob.append(self.eps/4)
        Q_func = self.model.predict(process_state(state))
        Q_vals = Q_func[0]
        Qmax = np.amax(Q_vals)
        for i in range(len(prob)):
            if Q_vals[i] == Qmax:
                prob[i] = 1 - self.eps + self.eps/4
                break
        action = np.random.choice(ACTIONS, p = prob)
        return action

def iterate(agents, E):
    """Performs one iteration, i.e. simulation of one time step."""
    terminal_list = []
    for agent in agents:
        S = State(E, [agent.m, agent.n])
        if [agent.m,agent.n] != agent.end:

            S_new, action_number, r, terminal = agent.move_agent(S) # Moves agent

            agent.steps += 1
            print('steps for agent {}: '.format(agent.nr),agent.steps)

            agent.reward += r

            e = (S, action_number, r, S_new, terminal)
            agent.memory.append(e)

            E = S_new.grid # Update environment

            if terminal == True:
                print('end of episode for agent {}'.format(agent.nr))

            if len(agent.memory) > batch_size:
                batch = sample_batch(agent.memory, batch_size)
                agent.train_network(batch)
            terminal_list.append(terminal)
            print()

    # Use if you dont want animation for the first episodes (can take a lot of time)
    # if len(episode_list) >= 10:
    #     show(E)

    # Always displays the enivironment
    show(E)

    terminal = np.all(terminal_list) # Only returns true when all agents has reached their goal

    return E, terminal


def episode(agents):
    """Simulation of one episode for multiple agents."""
    # Initialize E
    E = np.zeros((grid_size, grid_size), dtype = int) # Environment, i.e. the grid
    for agent in agents:
        agent.m = agent.start[0] # Initialize robot position
        agent.n = agent.start[1] # Initialize robot position
        E[agent.m][agent.n] = 1  # Initialize robot position on grid
        agent.steps = 0 # Initialize number of steps taken during episode
        agent.reward = 0

    # Adds some obtacles
    # E[1][1] = 2
    # E[2][1] = 2
    # E[2][3] = 2
    # E[3][3] = 2

    # Use if you dont want animation for the first episodes (can take a lot of time for large grids and many agents)
    # if len(episode_list) >= 10:
    #     show(E)

    # Always displays the enivironment
    show(E)

    terminal = False # Terminal state has not been reached

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
    #a3 = Agent(start = [grid_size-1, 0], end = [0, grid_size-1], nr=3) # Create agent 3
    #a4 = Agent(start = [grid_size-1, grid_size-1], end = [0, 0], nr=4) # Create agent 4
    agents.append(a1)
    agents.append(a2)
    #agents.append(a3)
    #agents.append(a4)

    reward_list = [[] for i in range(len(agents))]

    for i in range(10): # Choose number of episodes to run
        episode(agents) # Run one episode
        episode_list.append(i+1) # Episode number
        print('End of episode ', i+1)
        print()
        for agent in agents:
            reward_list[agent.nr-1].append(agent.reward)

    # Use if you want to decrease epsilon after episode instead of after a step
        # for agent in agents:
        #     if agent.eps > agent.eps_min:
        #         agent.eps *= agent.eps_decay



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

def process_state(state):
    """Pre-process state by converting it into an array which can be passed to network."""
    grid = state.grid
    pos = state.pos
    reshaped_grid = np.reshape(grid,(1, grid_size*grid_size)) # Only use squared for square matrices
    reshaped_grid = reshaped_grid[0]
    processed_state = np.concatenate((pos, reshaped_grid)) # Position of the agent + the environent == current state
    processed_state = np.array([processed_state])

    return processed_state

# Displays the environment
# OpengymAI might have some tools for visualizing the environment
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

def sample_batch(memory, n):
    """Samples n episodes from replay memory."""
    batch = rnd.sample(memory, n) # List containing tuples
    return batch

simulation() # Run the simulation
