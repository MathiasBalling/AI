# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Q-learning
#
# In this notebook, you will implement Q-learning as described in [Sutton and Barto's book, Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html). We will use the grid ```World``` class from the previous lectures.

# %% [markdown]
# ### Install dependencies

# %% colab={"base_uri": "https://localhost:8080/"}
# ! pip install numpy pandas tqdm

# %% [markdown]
# ### Imports

# %%
import numpy as np
import random
import sys  # We use sys to get the max value of a float
from IPython.display import display
import pandas as pd  # We only use pandas for displaying tables nicely
from tqdm import tqdm

pd.options.display.float_format = "{:,.3f}".format

# %% [markdown]
# ### ```World``` class and globals
#
# The ```World``` is a grid represented as a two-dimensional array of characters where each character can represent free space, an obstacle, or a terminal. Each non-obstacle cell is associated with a reward that an agent gets for moving to that cell (can be 0). The size of the world is _width_ $\times$ _height_ characters.
#
# A _state_ is a tuple $(x,y)$.
#
# An empty world is created in the ```__init__``` method. Obstacles, rewards and terminals can then be added with ```add_obstacle``` and ```add_reward```.
#
# To calculate the next state of an agent (that is, an agent is in some state $s = (x,y)$ and performs and action, $a$), ```get_next_state()```should be called.

# %%
# Globals:
ACTIONS = ("up", "down", "left", "right")

# Rewards, terminals and obstacles are characters:
REWARDS = {" ": 0, ".": 0.1, "+": 10, "-": -10}
TERMINALS = ("+", "-")  # Note a terminal should also have a reward assigned
OBSTACLES = "#"

# Discount factor
gamma = 1

# The probability of a random move:
rand_move_probability = 0


class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Create an empty world where the agent can move to all cells
        self.grid = np.full((width, height), " ", dtype="U1")

    def add_obstacle(self, start_x, start_y, end_x=None, end_y=None):
        """
        Create an obstacle in either a single cell or rectangle.
        """
        if end_x == None:
            end_x = start_x
        if end_y == None:
            end_y = start_y

        self.grid[start_x : end_x + 1, start_y : end_y + 1] = OBSTACLES[0]

    def add_reward(self, x, y, reward):
        assert reward in REWARDS, f"{reward} not in {REWARDS}"
        self.grid[x, y] = reward

    def add_terminal(self, x, y, terminal):
        assert terminal in TERMINALS, f"{terminal} not in {TERMINALS}"
        self.grid[x, y] = terminal

    def is_obstacle(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        else:
            return self.grid[x, y] in OBSTACLES

    def is_terminal(self, x, y):
        return self.grid[x, y] in TERMINALS

    def get_reward(self, x, y):
        """
        Return the reward associated with a given location
        """
        return REWARDS[self.grid[x, y]]

    def get_next_state(self, current_state, action):
        """
        Get the next state given a current state and an action. The outcome can be
        stochastic  where rand_move_probability determines the probability of
        ignoring the action and performing a random move.
        """
        assert action in ACTIONS, f"Unknown acion {action} must be one of {ACTIONS}"

        x, y = current_state

        # If our current state is a terminal, there is no next state
        if self.grid[x, y] in TERMINALS:
            return None

        # Check of a random action should be performed:
        if np.random.rand() < rand_move_probability:
            action = np.random.choice(ACTIONS)

        if action == "up":
            y -= 1
        elif action == "down":
            y += 1
        elif action == "left":
            x -= 1
        elif action == "right":
            x += 1

        # If the next state is an obstacle, stay in the current state
        return (x, y) if not self.is_obstacle(x, y) else current_state


# %% [markdown]
# ## A simple world

# %%
world = World(4, 6)

# Since we only focus on episodic tasks, we must have a terminal state that the
# agent eventually reaches
world.add_terminal(3, 5, "+")

print(world.grid.T)

# %% [markdown]
# ## Exercise: Q-learning
#
# Implement and test Q-learning. You should be able to base much of your code on your implementation of SARSA. Since Q-learning is an off-policy method, we can use whatever behavior policy we want during training, but the choice of behavioral policy still manners so it is a good idea to balance exploration and exploitation. During testing, we can then use the learnt policy (the target policy).
#
# As for the behavior policy, you can use an simple $\epsilon$-greedy policy, but you can also experiment with alternatives, for instance, optimistic initial values.
#
# See page 131 in [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html) for the Q-learning algorithm.
#


# %%
def greedy_policy(Q, state, epsilon):
    a = {action: epsilon / len(ACTIONS) for action in ACTIONS}

    _, max_value = max(Q[state].items(), key=lambda x: x[1])
    # Count cells that have the same max value
    max_value_count = sum(Q[state][action] == max_value for action in ACTIONS)
    for action in ACTIONS:
        if Q[state][action] == max_value:
            a[action] += (1 - epsilon) / max_value_count

    return a


def q_learning(world: World, start_state, policy, Q, gamma=0.9, alpha=0.1, epsilon=0.1):
    current_state = start_state
    while not world.is_terminal(*current_state):
        # Choose the next action based on the epsilon-greedy policy
        possible_actions = policy(Q, current_state, epsilon)

        # Initialize the starting action
        current_action = random.choices(
            population=list(possible_actions.keys()),
            weights=list(possible_actions.values()),
            k=1,
        )[0]

        # Get the next state and reward
        next_state = world.get_next_state(current_state, current_action)
        reward = world.get_reward(*next_state)

        # Find the best next action in the next state
        best_next_action_value = max(Q[next_state].values())

        # Update the Q-table
        Q[current_state][current_action] += alpha * (
            reward + gamma * best_next_action_value - Q[current_state][current_action]
        )

        # Update the state
        current_state = next_state

    # Return the Q-table after training to be used as a policy
    return Q


# %%
ACTIONS = ("up", "down", "left", "right")
world = World(3, 3)
world.add_terminal(1, 2, "+")
display(pd.DataFrame(world.grid.T))

Q = {}
for x in range(world.width):
    for y in range(world.height):
        Q[(x, y)] = {action: 0.0 for action in ACTIONS}

nb_episodes = 10000
for i in range(nb_episodes):
    Q = q_learning(
        world=world,
        start_state=(0, 0),
        policy=greedy_policy,
        Q=Q,
        gamma=0.9,
        alpha=0.1,
        epsilon=0.1,
    )

final_policy = np.full((world.width, world.height), "          ")
for i in range(world.width):
    for j in range(world.height):
        if world.is_terminal(i, j):
            final_policy[(i, j)] = "termnal"
        elif world.is_obstacle(i, j):
            final_policy[(i, j)] = "#"
        else:
            final_policy[(i, j)] = max(Q[(i, j)], key=Q[(i, j)].get)
display(pd.DataFrame(final_policy.T))

# %% [markdown]
# ## Exercise: Compare Q-learning and SARSA
#
# Setup experiments to compare the performance of Q-learning and SARSA. You can use different ```Worlds``` and test different parameter setting, e.g. for $\alpha$ and $\epsilon$.

# %%
### TODO: Implement your code here

# %% [markdown]
# ## Optional exercise: Maximization Bias and Double Learning
#
# Below is an implementation of the task shown in Example 6.7 on page 134 in [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html). There are two states, ```A``` and ```B``` where the agent can perform actions, and a terminal state ```T```. ```A``` and ```B``` have different actions available:
#
# * ```A``` has ```left``` (to ```B```) and ```right``` to the terminal state
# * ```B``` has a larger number of actions all leading to a terminal state.
#

# %%
# States "A" and "B" have actions while "T" is a terminal state.
STATES = ("A", "B", "T")


class Example67MDP:
    def __init__(self, number_of_B_actions):
        """
        Create an example and set the number of outgoing actions for state "B"
        (in the book, they do not give a specific number, but merely write that
        from "B" there "are many possible actions all of which cause immediate
        termination with a reward drawn from a normal distribution with mean
        -0.1 and variance 1. So, feel free to play with different number of
        actions in state B)

        """
        self.number_of_B_actions = number_of_B_actions

    def get_actions(self, state):
        """
        Returns the set of actions availabe in a given state (a tuple
        with strings).
        """
        assert state in STATES, f"State must be one of {STATES}, not {state}"
        if state == "A":
            return ("left", "right")
        if state == "B":
            return tuple(f"{i}" for i in range(self.number_of_B_actions))
        if state == "T":
            return tuple("N")

    def get_next_state_and_reward(self, state, action):
        """
        Get the next state and reward given a current state and an action
        """
        assert state in STATES, f"Unknown state: {state}"
        assert action in self.get_actions(
            state
        ), f"Unknown action {action} for state {state}"

        if state == "T":
            raise Exception("The terminal state has no actions and no next state")

        if state == "A":
            if action == "right":
                return "T", 0
            if action == "left":
                return "B", 0

        if state == "B":
            return "T", np.random.normal(loc=-0.1)

    def is_terminal(self, state):
        assert state in STATES, f"Unknown state: {state}"
        return state == "T"


# %% [markdown]
# Implement Double Q-learning (see page 136 in [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)) and test it on the ```Example67MDP``` above. Notice, that the number of actions differs between the two states ($\mathcal{A}$(```"A"```) $\neq \mathcal{A}$(```"B"```)), which you have to take into account in your Q-tables. See the code for ```Example67MDP``` above: you can get the set of actions available in a given state by calling ```get_actions(...)``` with the state as argument.
#
# Compare action-value estimates for ```"left"``` and ```"right"``` in state ```"A"```  at different times during learning when using double-Q learning and when using normal Q-learning.

# %%
# Create an instance of Example 6.7 with 10 actions in B
# example = Example67MDP(10)
#
# gamma = 1
# alpha = 0.05
#
# # Create two Q-tables (feel free to use your own representation):
# Q1 = [[0 for _ in range(len(example.get_actions(state)))] for state in STATES]
# Q2 = [[0 for _ in range(len(example.get_actions(state)))] for state in STATES]

# Uncomment to disable double-Q-learning:
# Q2 = Q1

# You can use the below policy method if you use the Q1 and Q2 as defined above.
# If you have done your own representation, you probably have to modify or
# rewrite the function below:


# def e_greedy_dql_policy(state):
#     global example
#     actions = {
#         a: epsilon / len(example.get_actions(state)) for a in example.get_actions(state)
#     }
#     # Do a Q1 + Q2 to do epsilon greedy based on both tables:
#     Q = [sum(x) for x in zip(Q1[STATES.index(state)], Q2[STATES.index(state)])]
#     actions[example.get_actions(state)[np.argmax(Q)]] = (
#         1 - epsilon + epsilon / len(example.get_actions(state))
#     )
#     return actions


# %%
def double_q_learning(
    world: World, start_state, policy, Q1, Q2, gamma=0.9, alpha=0.1, epsilon=0.1
):
    current_state = start_state
    while not world.is_terminal(*current_state):
        # Choose the next action based on the epsilon-greedy policy
        possible_actions = policy(Q1, Q2, current_state, epsilon)

        # Initialize the starting action
        current_action = random.choices(
            population=list(possible_actions.keys()),
            weights=list(possible_actions.values()),
            k=1,
        )[0]

        # Get the next state and reward
        next_state = world.get_next_state(current_state, current_action)
        reward = world.get_reward(*next_state)

        if np.random.rand() < 0.5:
            # Find best action from next state using Q1
            q1_best_action = max(Q1[next_state], key=Q1[next_state].get)
            # Update the Q1-table
            Q1[current_state][current_action] += alpha * (
                reward
                + gamma * Q2[next_state][q1_best_action]
                - Q1[current_state][current_action]
            )
        else:
            # Find best action from next state using Q1
            q2_best_action = max(Q2[next_state], key=Q2[next_state].get)
            # Update the Q2-table
            Q2[current_state][current_action] += alpha * (
                reward
                + gamma * Q1[next_state][q2_best_action]
                - Q2[current_state][current_action]
            )

        # Update the state
        current_state = next_state

    # Return the Q-table after training to be used as a policy
    return Q1, Q2


# %%
def greedy_policy_double(Q1, Q2, state, epsilon):
    a = {action: epsilon / len(ACTIONS) for action in ACTIONS}

    # Combine Q1 and Q2 values
    combined_Q = {action: Q1[state][action] + Q2[state][action] for action in ACTIONS}

    _, max_value = max(combined_Q.items(), key=lambda x: x[1])
    # Count cells that have the same max value
    max_value_count = sum(combined_Q[action] == max_value for action in ACTIONS)
    for action in ACTIONS:
        if combined_Q[action] == max_value:
            a[action] += (1 - epsilon) / max_value_count

    return a


# %%
ACTIONS = ("up", "down", "left", "right")
world = World(6, 6)
world.add_terminal(5, 5, "+")
display(pd.DataFrame(world.grid.T))

Q1, Q2 = {}, {}
for x in range(world.width):
    for y in range(world.height):
        Q1[(x, y)] = {action: 0.0 for action in ACTIONS}
        Q2[(x, y)] = {action: 0.0 for action in ACTIONS}

nb_episodes = 100000
for i in tqdm(range(nb_episodes)):
    Q1, Q2 = double_q_learning(
        world=world,
        start_state=(0, 0),
        policy=greedy_policy_double,
        Q1=Q1,
        Q2=Q2,
        gamma=1,
        alpha=0.1,
        epsilon=0.2,
    )

final_policy = np.full((world.width, world.height), "          ")
for i in range(world.width):
    for j in range(world.height):
        if world.is_terminal(i, j):
            final_policy[(i, j)] = "termnal"
        elif world.is_obstacle(i, j):
            final_policy[(i, j)] = "#"
        else:
            combined_Q = {
                action: Q1[(i, j)].get(action, 0) + Q2[(i, j)].get(action, 0)
                for action in Q1[(i, j)]
            }
            final_policy[(i, j)] = max(combined_Q, key=combined_Q.get)
            # print(f"{Q1[(i, j)]}{Q2[(i,j)]}{max(combined_Q,key=combined_Q.get)}")

display(pd.DataFrame(final_policy.T))
