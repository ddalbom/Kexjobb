# Deep learning for multiple agents
# Deep Q network
# Lets go!

import numpy as np
import random as rnd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque
from copy import deepcopy
import seaborn as sns
import pickle

grid_size = 10
ACTIONS = ['Right', 'Left', 'Up', 'Down']
eps = 0.9 # Initial exploration rate
eps_min = 0.001 # Minimum exploration rate
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
    def __init__(self, start, end, nr):
        self.m = start[0]
        self.n = start[1]
        self.start = start
        self.end = end
        self.steps = 0
        self.reward = 0
        self.collisions = 0
        self.nr = nr
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
        model.compile(loss = 'mse', optimizer = RMSprop(lr = alpha))

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

    def train_network(self, batch, episode_nr):
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
        if self.epsilon > self.eps_min and episode_nr > 10:
            self.epsilon *= self.eps_decay # Decrease exploration rate

    def move_agent(self, state):
        """Move agent one step."""
        m = self.m
        n = self.n

        cur_env = deepcopy(state.grid)
        cur_env[m, n] = 0
        action = self.choose_action(state)

        if action == 'Right':
            if n + 1 >= grid_size or cur_env[m][n+1] != 0:
                Rew = -2 # Reward -5 if we move into wall or another agent
                self.collisions += 1
            else:
                n += 1
                Rew = -0.1 # Reward -1 otherwise
            a = 0 # Action number
        elif action == 'Left':
            if n - 1 < 0 or cur_env[m][n-1] != 0:
                Rew = -2
                self.collisions += 1
            else:
                n -= 1
                Rew = -0.1
            a = 1
        elif action == 'Up':
            if m - 1 < 0 or cur_env[m-1][n] != 0:
                Rew = -2
                self.collisions += 1
            else:
                m -= 1
                Rew = -0.1
            a = 2
        elif action == 'Down':
            if m + 1 >= grid_size or cur_env[m+1][n] != 0:
                Rew = -2
                self.collisions += 1
            else:
                m += 1
                Rew = -0.1
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
        Q_vals = Q_func[0]
        max_index = []
        Qmax = np.amax(Q_vals)
        for i in range(len(prob)):
            if Q_vals[i] == Qmax:
                # max_index.append(i)
                prob[i] = 1 - self.epsilon + self.epsilon/4
                break
        # ind = np.random.choice(max_index)
        # prob[ind] = 1 - self.epsilon + self.epsilon/4
        action = np.random.choice(ACTIONS, p = prob)
        return action

    def load_policy(self, name):
        """Load policy weights."""
        self.policy.load_weights(name)

    def save_policy(self, name):
        """Save policy weights."""
        self.policy.save_weights(name)

    def load_target(self, name):
        """Load target weights."""
        self.target.load_weights(name)

    def save_target(self, name):
        """Save policy weights."""
        self.target.save_weights(name)

def iterate(agents, E, t, episode_nr):
    """Performs one iteration, i.e. simulation of one time step."""
    global batch_size
    # print('I am inside one time step!')
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
                agent.train_network(batch, episode_nr)

            terminal_list.append(terminal)

            if t % 50 == 0: # Choose time interval for updating target network weights
                agent.update_target_network()

        # print(E) # Show grid
        # print()

    # if episode_nr > 5:
        # show_grid(E, episode_nr)

    # show_grid(E, episode_nr)

    terminal = np.all(terminal_list) # True if all agents have reached destination
    env_list.append(E)

    return E, terminal


def episode(agents, t, episode_nr):
    """Simulation of one episode for multiple agents."""
    # Initialize E
    E = np.zeros((grid_size, grid_size), dtype = int) # Environment, i.e. the grid

    E[0][3] = 2 # Use for 10x10 grid
    E[1][3] = 2
    E[0][6] = 2
    E[1][6] = 2
    E[2][6] = 2
    E[7][3] = 2
    E[8][3] = 2
    E[9][3] = 2
    E[8][6] = 2
    E[9][6] = 2

    env_list.append(E)

    for agent in agents:
        agent.m = agent.start[0] # Initialize robot position
        agent.n = agent.start[1] # Initialize robot position
        E[agent.m][agent.n] = 1  # Initialize robot position on grid
        agent.carry = False # Initialize robot carry, i.e. the robot does NOT carry cargo
        agent.steps = 0 # Initialize number of steps taken during episode
        agent.reward = 0 # Do not reset if we should plot accumulated over all episodes
        agent.collisions = 0 # Reset number of collisions
        if episode_nr % 100 == 0:
            agent.save_target('target_weights_{}.h5'.format(agent.nr))
            agent.save_policy('policy_weights_{}.h5'.format(agent.nr))

    terminal = False # Terminal state has not been reached

    while not terminal: # While agents have not reached their terminal state
        E, terminal = iterate(agents, E, t, episode_nr)
        t += 1

    return t


def simulation(nepisodes):
    """Iterate through all episodes."""
    # Initialize robots
    # print('I am inside the simulation')
    agents = [] # List containing all robots
    a1 = Agent(start = [0, 0], end = [grid_size-1, grid_size-1], nr = 1) # Create agent 1
    a2 = Agent(start = [0, grid_size-1], end = [grid_size-1, 0], nr = 2) # Create agent 2
    a3 = Agent(start = [grid_size-1, 0], end = [0, grid_size-1], nr = 3) # Create agent 3
    a4 = Agent(start = [grid_size-1, grid_size-1], end = [0, 0], nr = 4) # Create agent 4
    agents.append(a1)
    agents.append(a2)
    agents.append(a3)
    agents.append(a4)

    # for agent in agents:
    #     agent.load_target('target_weights_{}.h5'.format(agent.nr))
    #     agent.load_policy('policy_weights_{}.h5'.format(agent.nr))
    #     print('loaded')

    steps_list = [[] for i in range(len(agents))]
    reward_list = [[] for i in range(len(agents))]
    cumulative_rewards = [[] for i in range(len(agents))]
    collisions_list = [[] for i in range(len(agents))]

    t = 0 # Set time to zero
    for i in range(nepisodes):
        t = episode(agents, t, i+1) # Run one episode

        print('End of episode ', i+1)
        agent_index = 0
        for agent in agents:
            steps_list[agent_index].append(agent.steps)
            reward_list[agent_index].append(agent.reward)
            collisions_list[agent_index].append(agent.collisions)
            if i == 0:
                cumulative_rewards[agent_index].append(agent.reward)
            else:
                cumulative_rewards[agent_index].append(agent.reward + cumulative_rewards[agent_index][i-1])
            agent_index += 1

        if i % 1000 == 0:
            with open('reward_4_agents_{}'.format(i),'wb') as f:
                pickle.dump(reward_list,f)

            with open('steps_4_agents_{}'.format(i), 'wb') as f:
                pickle.dump(steps_list, f)

            with open('cols_4_agents_{}'.format(i), 'wb') as f:
                pickle.dump(collisions_list, f)


    return steps_list, reward_list, collisions_list, cumulative_rewards

def show_grid(frame, episode_nr):
    """Animate movement on grid as simulation runs."""
    plt.grid('on')
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, 10, 1))
    ax.set_yticks(np.arange(0.5, 10, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(frame, cmap='binary')
    ax.set_title("Episode {}".format(episode_nr))
    plt.pause(0.01)
    plt.clf()

def animate(frames):
    """Animate movement on grid after performing the simulation."""
    plt.grid('on')
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, 10, 1))
    ax.set_yticks(np.arange(0.5, 10, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(len(env_list)):
        ax.imshow(env_list[i],cmap='binary')
        plt.pause(0.05)

def run_simulations():
    """Run multiple simulations."""
    nepisodes = 10000 # Number of episodes in each simulation
    nsims = 1 # Number of simulations to run
    nagents = 4 # Need to set the agents in simulation() as well

    # Total summed over all simulations; before averaging
    total_rewards = [[0 for j in range(nepisodes)] for i in range(nagents)]
    total_cum_rew = [[0 for j in range(nepisodes)] for i in range(nagents)]
    total_steps = [[0 for j in range(nepisodes)] for i in range(nagents)]
    total_collisions = [[0 for j in range(nepisodes)] for i in range(nagents)]
    episode_list = [i+1 for i in range(nepisodes)]

    for sim in range(nsims):
         print('Simulation', sim+1)
         steps_list, reward_list, collisions_list, cumulative_rewards = simulation(nepisodes)
         for agent in range(nagents):
             for ep in range(nepisodes):
                 total_rewards[agent][ep] += reward_list[agent][ep]
                 total_steps[agent][ep] += steps_list[agent][ep]
                 total_collisions[agent][ep] += collisions_list[agent][ep]
                 total_cum_rew[agent][ep] += cumulative_rewards[agent][ep]

    ave_rew = np.array(total_rewards, dtype = 'f')/nsims
    ave_steps = np.array(total_steps, dtype = 'f')/nsims
    ave_cols = np.array(total_collisions, dtype = 'f')/nsims
    ave_cum_rew = np.array(total_collisions, dtype = 'f')/nsims

    # print('Print totals over all episodes')
    # print('Reward: ', total_rewards)
    # print('Cumulative reward: ', total_cum_rew)
    # print('Steps: ', total_steps)
    # print('collisions: ', total_collisions)
    # print()
    # print('Averages')
    # print('Reward: ', ave_rew)
    # print('Cumulative reward: ', ave_cum_rew)
    # print('Steps: ', ave_steps)
    # print('Collisions: ', ave_cols)

    with open('average_reward_4_agents_10000_{}'.format(nsims),'wb') as f:
        pickle.dump(ave_rew,f)

    with open('average_steps_4_agents_10000_{}'.format(nsims), 'wb') as f:
        pickle.dump(ave_steps, f)

    with open('average_cols_4_agents_10000_{}'.format(nsims), 'wb') as f:
        pickle.dump(ave_cols, f)

    with open('average_cum_rew_4_agents_10000_{}'.format(nsims), 'wb') as f:
        pickle.dump(ave_cum_rew, f)


    sns.set()
    matplotlib.rc('xtick', labelsize=25)
    matplotlib.rc('ytick', labelsize=25)
    plt.rc('axes', titlesize=40)
    plt.rc('axes', labelsize=40)
    plt.rc('legend', fontsize=20)

    k = 1
    for list in ave_steps:
        plt.plot(episode_list, list, '.', label = 'Agent {}'.format(k))
        k +=1
    plt.xlabel('Episode')
    plt.ylabel('Number of steps')
    plt.legend()
    plt.show()
    k = 1
    for list in ave_rew:
        plt.plot(episode_list, list, label = 'Agent {}'.format(k))
        k += 1
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    k = 1
    for list in ave_cols:
        plt.plot(episode_list, list, '.', label = 'Agent {}'.format(k))
        k += 1
    plt.xlabel('Episode')
    plt.ylabel('Number of collisions')
    plt.legend()
    plt.show()
    k = 1
    for list in ave_cum_rew:
        plt.plot(episode_list, list, label = 'Agent {}'.format(k))
        k += 1
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.legend()
    plt.show()


run_simulations() # Run multiple simulations and average

# simulation(100) # Run one simulation
# animate(env_list) # Animate single simulation
