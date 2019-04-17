# Deep learning for multiple agents
# Deep Q network
# Lets go!

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque
from copy import deepcopy

grid_size = 6
ACTIONS = ['Right', 'Left', 'Up', 'Down']
eps = 1 # Initial exploration rate
eps_min = 0.01 # Minimum exploration rate
eps_decay = 0.995 # Decrease chance of exploration by this factor after each "training session"
gamma = 0.9 # Discount factor
alpha = 0.001 # Learning rate
batch_size = 32

env_list = []

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

def process_state(state):
    """Pre-process state by converting it into an array which can be passed to network."""
    grid = state.grid
    pos = state.pos
    reshaped_grid = np.reshape(grid,(1, grid_size*grid_size)) # Only use squared for square matrices
    reshaped_grid = reshaped_grid[0]
    processed_state = np.concatenate((pos, reshaped_grid))
    processed_state = np.array([processed_state])
    # processed_state.reshape(1, 1, grid_size*grid_size+2, 1)
    #print(processed_state.shape)

    return processed_state

def sample_batch(memory, n):
    """Samples n episodes from replay memory."""
    batch = rnd.sample(memory, n) # List containing tuples
    return batch

class Agent:
    """Defines the agent."""
    global eps, eps_min, eps_decay
    def __init__(self, start, end):
        self.m = start[0]
        self.n = start[1]
        self.start = start
        self.end = end
        self.steps = 0
        self.reward = 0
        self.epsilon = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.memory  = deque(maxlen=2000) # Memory for storing experiences, maximum capacity 2000
        self.policy = self.build_dense() # Create a dense policy network for each agent
        self.target = deepcopy(self.policy) # Set to target network as copy of policy network; its weights are updated every nth iteration
        # self.policy = self.build_cnn() # Fix this maybe

    def build_dense(self): # Pass state_size and action_size
        """Create a dense neural network."""
        model = Sequential()
        model.add(Dense(24, input_dim = grid_size*grid_size+2, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(len(ACTIONS), activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = alpha))

        return model

    def build_cnn(self):
        """Create a convolutional neural network."""
        model = Sequential()
        model.add(Conv2D(24, (1, 3), activation = 'relu', input_shape = (1, grid_size*grid_size+2, 1)))
        model.add(Conv2D(24, (1, 3), activation = 'relu', input_shape = (1, grid_size*grid_size+2, 1)))
        model.add(Flatten())
        model.add(Dense(len(ACTIONS), activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = alpha))

        return model

    def update_target_network(self):
        """Update raget network using the weights of policy network."""
        self.target.set_weights(self.policy.get_weights()) # Update weights of target network with weights of policy network

    def train_network(self, batch):
        """Train the neural network using sampled batch of experiences from replay memory."""
        global eps, eps_min, eps_decay
        for exp in batch:
            S = exp[0]
            S = process_state(S)
            action_number = exp[1]
            r = exp[2]
            S_new = exp[3]
            S_new = process_state(S_new)
            terminal = exp[4]

            if not terminal: # If agent is not at its final destination
                target = (r + gamma*np.amax(self.target.predict(S_new)[0]))
            else:
                target = r
            target_f = self.policy.predict(S)

            target_f[0][action_number] = target # Update something???
            self.policy.fit(S, target_f, epochs=1, verbose=0) # Train network # Verbose - makes training line?
        if self.epsilon > self.eps_min: # Decrease exploration rate
            self.epsilon *= self.eps_decay

    def move_agent(self, state):
        """Move agent one step."""
        m = self.m
        n = self.n

        cur_env = deepcopy(state.grid)
        cur_env[m, n] = 0
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
        self.m = m # Update position of agent
        self.n = n # Update position of agent
        cur_env[m][n] = 1 # Update grid
        new_state = State(cur_env, [m, n]) # Set new state
        terminal = False

        if [m, n] == self.end:
            Rew = 10
            terminal = True
            self.carry = True

        return new_state, a, Rew, terminal

    def choose_action(self, state):
        """Choose an action depending on the given state. """
        prob = [] # Probability distribution
        for i in range(len(ACTIONS)):
            prob.append(self.epsilon/4)
        Q_func = self.policy.predict(process_state(state))
        Q_vals = Q_func[0] # Ugly fix :(
        # print('Q-values are: ', Q_vals)
        Qmax = np.amax(Q_vals)
        for i in range(len(prob)):
            if Q_vals[i] == Qmax:
                prob[i] = 1 - self.epsilon + self.epsilon/4
                break
        action = np.random.choice(ACTIONS, p = prob)
        return action

def iterate(agents, E, t):
    """Performs one iteration, i.e. simulation of one time step."""
    global batch_size
    print('I am inside one time step!')
    terminal_list = []
    for agent in agents:
        S = State(E, [agent.m, agent.n])
        if not agent.carry:

            S_new, action_number, r, terminal = agent.move_agent(S) # Moves agent

            agent.steps += 1
            agent.reward += r

            e = (S, action_number, r, S_new, terminal)
            agent.memory.append(e)

            E = S_new.grid # Update environment

            if len(agent.memory) > batch_size:
                batch = sample_batch(agent.memory, batch_size)
                agent.train_network(batch)

            terminal_list.append(terminal)

            if t % 50 == 0:
                agent.update_target_network()

        # print(E) # Show grid
        # print()

    terminal = np.all(terminal_list) # True if all agents have reached destination
    env_list.append(E)

    return E, terminal

def episode(agents, t):
    """Simulation of one episode for multiple agents."""
    # Initialize E
    print('I am inside the episode')
    E = np.zeros((grid_size, grid_size), dtype = int) # Environment, i.e. the grid
    env_list.append(E)
    for agent in agents:
        agent.m = agent.start[0] # Initialize robot position
        agent.n = agent.start[1] # Initialize robot position
        E[agent.m][agent.n] = 1  # Initialize robot position on grid
        agent.carry = False # Initialize robot carry, i.e. the robot does NOT carry cargo
        agent.steps = 0 # Initialize number of steps taken during episode
        agent.reward = 0 # Do not reset if we should plot accumulated over all episodes

    terminal = False # Terminal state has not been reached

    while not terminal: # While agents have not reached their terminal state
        E, terminal = iterate(agents, E, t)
        t += 1

    print('I am done with the episode')

    return t


def simulation(nepisodes):
    """Iterates through all episodes."""
    # Initialize robots
    print('I am inside the simulation')
    agents = [] # List containing all robots
    a1 = Agent(start = [0, 0], end = [grid_size-1, grid_size-1]) # Create agent 1
    a2 = Agent(start = [0, grid_size-1], end = [grid_size-1, 0]) # Create agent 2
    a3 = Agent(start = [grid_size-1, 0], end = [0, grid_size-1]) # Create agent 3
    a4 = Agent(start = [grid_size-1, grid_size-1], end = [0, 0]) # Create agent 4
    agents.append(a1)
    # agents.append(a2)
    # agents.append(a3)
    # agents.append(a4)

    episode_list = []
    steps_list = [[] for i in range(len(agents))]
    reward_list = [[] for i in range(len(agents))]

    t = 0 # Set time to zero
    for i in range(nepisodes): # Choose number of episodes to run
        print('I am inside the loop')
        t = episode(agents, t) # Run one episode
        # print(t)

        episode_list.append(i+1) # Episode number
        print('End of episode ', i+1)
        agent_index = 0
        for agent in agents:
            steps_list[agent_index].append(agent.steps)
            reward_list[agent_index].append(agent.reward)
            agent_index += 1

    return steps_list, reward_list

    # for list in steps_list:
    #     plt.plot(episode_list, list, '.', label = 'Number of steps taken for each episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Number of steps')
    # plt.legend()
    # plt.show()
    # for list in reward_list:
    #     plt.plot(episode_list, list, label = 'Total reward of each episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Total reward')
    # plt.legend()
    # plt.show()

def animate(frames):
    """Animate movement on grid."""
    plt.grid('on')
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, 10, 1))
    ax.set_yticks(np.arange(0.5, 10, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(len(env_list)):
        ax.imshow(env_list[i],cmap='binary')
        ax.set_title("Steps {}".format(i+1))
        plt.pause(0.05)


# simulation(200) # Run the simulation
# print(len(env_list))
# animate(env_list)

def run_simulations():
    """Run multiple simulations."""
    nepisodes = 20 # Number of episodes in each simulation
    nsims = 10 # Number of simulations to run
    nagents = 4

    total_rewards = [[] for i in range(nagents)] # Average rewards over all simulations
    total_steps = [[] for i in range(nagents)] # Average steps over all simulations
    episode_list = [i+1 for i in range(nepisodes)]

    print('I am here')

    for sim in range(nsims):
         steps_list, reward_list = simulation(nepisodes)
         for agent in range(nagents):
             for ep in range(nepisodes):
                 total_rewards[agent][ep] += reward_list[agent][ep]
                 total_steps[agent][ep] += steps_list[agent][ep]

    ave_rew = np.array(total_rewards, dtype = 'f')/nsims
    ave_steps = np.array(total_steps, dtype = 'f')/nsims

    for list in ave_steps:
        plt.plot(episode_list, list, '.', label = 'Number of steps taken for each episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of steps')
    plt.legend()
    plt.show()
    for list in ave_rew:
        plt.plot(episode_list, list, label = 'Total reward of each episode')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.legend()
    plt.show()

run_simulations()
