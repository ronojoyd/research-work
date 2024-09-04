import random
import numpy as np


def learn(Q, epsilon, state_size, last_state, current_state, reward, lr, gamma):

    if last_state is not None and current_state is not None:
        Q[last_state, current_state] = \
            Q[last_state, current_state] + lr * (reward + gamma * np.max(Q[current_state, :]) - Q[last_state, current_state])

    # Explore: select a random action
    if current_state is None or random.uniform(0, 1) < epsilon:
        next_state = random.choice([i for i in range(state_size)])
    else:
        # Exploit: select the action with max value (future reward)
        next_state = np.argmax(Q[current_state])

    return Q, current_state, next_state


# Set the percent you want to explore
epsilon = 0.8
decay = 0.999

state_size = 4
lr = 0.3
gamma = 0.8

current_state = None
last_state = None

# Initialize q-table values to 0
Q = np.zeros((state_size, state_size))

for i in range(10000):

    print (Q)
    if current_state is None:
        reward = random.randint(1, 20)
    else:
        reward = (current_state + 1) * 10

    print (i, epsilon, last_state, current_state, reward)
    print ('\n')

    Q, last_state, current_state = learn(Q, epsilon, state_size, last_state, current_state, reward, lr, gamma)
    epsilon *= decay


