import numpy as np
import random

# Define environment (4x4 grid)
num_states = 16 # 4x4 grid 
num_actions = 4 # up, down, left, right
q_table = np.zeros((num_states, num_actions))

# Define parameters
alpha = 0.1 # Learning rate
gamma = 0.9 # discount factor - how much weight future awards
epsilon = 0.2 # exploration rate (random action)
num_episodes = 1000

# Define simple reward structure
rewards = np.zeros(num_states)
rewards[15] = 1 # Goal state with a reward

# Function to determine the next state based on the action
def get_next_state(state, action):
    if action == 0 and state >= 4:
        return state - 4 # up
    elif action == 1 and (state - 1) % 4 != 0:
        return state + 1 # right
    elif action == 2 and state < 12:
        return state + 4 # down
    elif action == 3 and state % 4 != 0:
        return state - 1 # left
    else:
        return state

# Q-learning algorithm
for episode in range(num_episodes):
    state = random.randint(0, num_states - 1) # start from random state 
    while state != 15: # Loop until reaching goal
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions-1)
        else:
            action = np.argmax(q_table[state])
        
        next_state = get_next_state(state, action)
        reward = rewards[next_state]
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q-learning update rule
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state