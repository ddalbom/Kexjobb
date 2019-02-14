# Reinforcement learning

# Adding a decreasing epsilon over time could fix the convergence 

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

grid_size = 4
ACTIONS = ['Right', 'Left', 'Up', 'Down']
gamma = 0.9 # Discount factor
alpha = 1 # Step size, does not need to be 1
eps = 0.1 # Greedy policy


'''Q-tables for all agents'''
# Initialize Action-Value function Q(s, a) for agent 1
Q1 = np.random.rand((grid_size)**2*((grid_size)**2-1), len(ACTIONS))
Q1[-1] = [0, 0, 0, 0]

# Initialize Action-Value function Q(s, a) for agent 2
Q2 = np.random.rand((grid_size)**2*((grid_size)**2-1), len(ACTIONS))
Q2[-1] = [0, 0, 0, 0]


"""Starting position and goal"""
# Starting position for agent 1
m_A = 0
n_A = 0

# Goal for agent 1
m_B = grid_size - 1
n_B = grid_size - 1

# Starting position for agent 2
m_C = 0
n_C = grid_size - 1

# Goal for agent 2
m_D = grid_size - 1
n_D = 0


'''lists used for plotting results'''
agent1_step_list = []
agent2_step_list = []
reward_list1 = []
reward_list2 = []
nepisodes = []


''' Rewards '''
goal_reward = 10 - 1 # Reward for reachcing goal 10 minus 1 for new state transition
new_state_reward = -1 # Reward for new state transitioon
colliding_reward = -2 # Reward for colliding with agent/wall


class Agent:
    """Implements test agent for RL algorithm. """
    def __init__(self, row, col):
        self.m = row
        self.n = col

def move_agent(agent, state, Q, agent_nr, state_number):
    """Moves the robot according to the given action."""
    m = agent.m
    n = agent.n

    state[m][n] = 0  # Remove our agent from its position
    action = choose_action(state, Q, state_number) # Choose an random action weighted with the actions probabilities
    if action == 'Right':
        if n + 1 >= grid_size or state[m][n+1] != 0:  # Checks if there is a wall to the right or position is occupied
            pass
        else:
            n += 1 # make a step to the right
        a = 0  # 0 represents Right

    elif action == 'Left':
        if n - 1 < 0 or state[m][n-1] != 0: # Check if there is a wall to the left or position is occupied
            pass
        else:
            n -= 1 # make a step to the left
        a = 1 # 1 represents left

    elif action == 'Up':
        if m - 1 < 0 or state[m-1][n] != 0: # Check if there is a wall above the agent or position is occupied
            pass
        else:
            m -= 1 # make a step up
        a = 2 # 2 represents up

    elif action == 'Down':
        if m + 1 >= grid_size or state[m+1][n] != 0: # Check if there is a wall under the agent or position is occupied
            pass
        else:
            m += 1 # Make a step down
        a = 3 # 4 represents down

    print("Action taken by agent ",agent_nr,': ',  action)

    # Updates the position of the agent
    agent.m = m
    agent.n = n
    state[m][n] = agent_nr # Updates our agents position to the new position
    return state, a

def choose_action(state, Q, state_number): # Given a state and a probability distribution, chooses an action!
    prob = []   # Probability for each action in current state
    for i in range(len(ACTIONS)):
        prob.append(eps/4)
    Qmax = max(Q[state_number])  # Picks the action with highest Q-value in our current state
    for i in range(len(prob)):
        if Q[state_number][i] == Qmax:
            prob[i] = 1 - eps + eps/4
            break # Use when two actions got the same Q-value
    action = np.random.choice(ACTIONS, p=prob) # Chooses an action at random
    return action

def episode():
    """Simulation of one episode."""
    global Q1, Q2
    # Initialize S
    S = np.zeros((grid_size, grid_size), dtype = int) # Initializes the environment
    a1 = Agent(row=m_A,col=n_A) # Creates agent 1
    a2 = Agent(row=m_C,col=n_C) # Creates agent 2
    S[a1.m][a1.n] = 1 # Initializes position of agent 1
    S[a2.m][a2.n] = 2 # Initializes position of agent 2
    R1tot = 0 # Computes the total reward for agent 1
    R2tot = 0 # Computes the total reward for agent 2

    print(S)
    print()


    count1 = 0 # Counts the step for agent 1
    count2 = 0 # Counts the step for agent 2

    while S[m_B][n_B] != 1 or S[m_D][n_D] != 2: # runs the episode for as long as both the agents haven't reached their goal

        # Agent 1
        if S[m_B][n_B] != 1:
            m_old = a1.m # Saves old position
            n_old = a1.n # Saves old position

            state_number = state_to_number(S)
            S_new, action_number = move_agent(a1, S, Q1, 1, state_number) # Agent 1 performs a move
            new_state_number = state_to_number(S_new)

            m_new = a1.m # new position
            n_new = a1.n # new position

            '''Gives the agent an reward'''
            if m_new == m_B and n_new == n_B: # Reaching goal = 10 points
                R1 = goal_reward
            elif m_new == m_old and n_new == n_old: # Colliding with agent or wall = -2 points
                R1 = colliding_reward
            else:
                R1 = new_state_reward # Transition to new state = -1

            '''Q-learning algorithm, assigns a Q-value for the action we took'''
            Q1[state_number][action_number] += alpha*(R1 + gamma*max(Q1[new_state_number]) - Q1[state_number][action_number])

            '''Updates the state of the environment'''
            S = S_new
            count1 += 1

        else: # Stop updating agents reward when goal is reached
            R1 = 0

        # Agent 2
        if S[m_B][n_D] != 2: # If agent 2 hasn't reached it's goal
            m_old = a2.m
            n_old = a2.n

            state_number = state_to_number(S)
            S_new, action_number = move_agent(a2, S, Q2, 2, state_number) # Agent 2 performs a move
            m_new = a2.m # New state
            n_new = a2.n

            '''Gives the agent an reward'''
            if m_new == m_D and n_new == n_D:
                R2 = goal_reward
            elif m_new == m_old and n_new == n_old:
                R2 = colliding_reward
            else:
                R2 = new_state_reward

            new_state_number = state_to_number(S_new)

            '''Q-learning algorithm, assigns a Q-value for the action we took'''
            Q2[state_number][action_number] += alpha*(R2 + gamma*max(Q2[new_state_number]) - Q2[state_number][action_number])

            '''Updates the state of the environment'''
            S = S_new # Updates state
            count2 += 1
        else:
            R2 = 0

        print(S)
        print()
        R1tot += R1
        R2tot += R2
    return count1, count2, R1tot, R2tot

# Converts current state to a UNIQUE number
def state_to_number(S):
    count1 = find_agent_1(S)
    count2 = find_agent_2(S)
    return ((grid_size*grid_size-1)*count1 + count2 - 1)

# Help function
def find_agent_1(S):
    count = 0
    for i in range(4):
        for j in range(4):
            if S[i][j] == 1:
                return count
            else:
                count += 1
# Help function
def find_agent_2(S):
    count = 0
    for i in range(4):
        for j in range(4):
            if S[i][j] == 2:
                return count
            else:
                count += 1


def simulation():
    """Iterates through all episodes."""
    for i in range(1000):
        steps1, steps2, R1tot, R2tot = episode()
        print("End of episode!")
        print("number of steps for agent1 :",steps1)
        print("number of steps for agent2 :",steps2)
        print("Total reward for agent 1 :",R1tot)
        print("Total reward for agent 2 :",R2tot)

        agent1_step_list.append(steps1)
        agent2_step_list.append(steps2)
        reward_list1.append(R1tot)
        reward_list2.append(R2tot)
        nepisodes.append(i+1)


simulation()

# Plots the steps for both agents
#plt.plot(nepisodes,agent1_step_list)
#plt.plot(nepisodes,agent2_step_list)

plt.plot(nepisodes,reward_list1)
plt.plot(nepisodes,reward_list2)

plt.show()
