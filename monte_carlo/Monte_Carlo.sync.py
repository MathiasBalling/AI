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
# # Monte Carlo approaches to prediction and control
#
# In this notebook, you will implement the Monte Carlo approaches to prediction and control described in [Sutton and Barto's book, Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html). We will use the grid ```World``` class from the previous lecture, but now without relying on knowledge of the task dynamics, that is, without relying on knowledge about transition probabilities.

# %% [markdown]
# ### Install dependencies

# %% colab={"base_uri": "https://localhost:8080/"}
# ! pip install numpy pandas

# %% [markdown]
# ### Imports

# %%
import numpy as np
import random
import sys  # We use sys to get the max value of a float
import pandas as pd  # We only use pandas for displaying tables nicely
from IPython.display import display
from pandas.core.arrays.arrow.accessors import ListAccessor

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
#
# __Note that ```get_state_transition_probabilities``` has been removed and an agent must now rely on experience interacting with a world to learn.__

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
# ## Basic example: Generating episodes
#
# An episode is the series of states, actions and rewards reflecting an agent's experience interacting with the environment. An episode starts with an agent being placed at some initial state and continues till the agent reaches a terminal state.  To generate episodes, we first need a world and a policy:
#

# %% colab={"base_uri": "https://localhost:8080/"}
world = World(2, 3)

# Since we only focus on episodic tasks, we must have a terminal state that the
# agent eventually reaches
world.add_terminal(1, 2, "+")


def equiprobable_random_policy(x, y):
    return {k: 1 / len(ACTIONS) for k in ACTIONS}


print(world.grid.T)


# %% [markdown]
# To generate an episode, we need to provide a ```World```, a policy, and a start state.
#
# In each step, we do the following:
# 1. perform one of the actions (weighted random) returned by the policy for the giving state
# 2. get the reward and add a new entry to the episode $[S_t, A_t, R_{t+1}]$
# 3. move the agent to the next state
#
# When a terminal state is reached, we return all the $[[S_0, A_0, R_1], ..., [S_{T}, A_T, R_{T+1}]]$ observed in the episode.


# %%
def generate_episode(world: World, policy, start_state):
    current_state = start_state
    episode = []
    while not world.is_terminal(*current_state):
        # Get the possible actions and their probabilities that our policy says
        # that the agent should perform in the current state:
        possible_actions = policy(*current_state)

        # Pick a weighted random action:
        action = random.choices(
            population=list(possible_actions.keys()),
            weights=possible_actions.values(),
            k=1,
        )

        # Get the next state from the world
        next_state = world.get_next_state(current_state, action[0])

        # Get the reward for performing the action
        reward = world.get_reward(*next_state)

        # Save the state, action and reward for this time step in our episode
        episode.append([current_state, action[0], reward])

        # Move the agent to the new state
        current_state = next_state

    return episode


# %% [markdown]
# Now, we can try to generate a couple of episodes and print the result:

# %% colab={"base_uri": "https://localhost:8080/"}
for i in range(5):
    print(f"Episode {i}:")
    episode = generate_episode(world, equiprobable_random_policy, (0, 0))
    print(pd.DataFrame(episode, columns=["State", "Action", "Reward"]), end="\n\n")

# %% [markdown]
# ### Exercise: Implement Monte Carlo-based prediction for state values
#
# You should implement first-visit MC prediction for estimating $V≈v_\pi$. See page 92 of [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html).
#


# %%
def mc_fv_prediction(world: World, policy, nb_episodes, gamma):
    V = np.zeros((world.width, world.height))
    Returns = {}
    # Initialize returns as lists
    for i in range(world.width):
        for j in range(world.height):
            Returns[i, j] = []

    for _ in range(nb_episodes):
        # Generate a random start state
        start_state = (np.random.randint(world.width), np.random.randint(world.height))
        episode = generate_episode(world, policy, start_state)
        G = 0
        for t, (state, _, reward) in enumerate(reversed(episode)):
            idx = len(episode) - t - 1
            G = gamma * G + reward
            if state not in [s for s, _, _ in episode[:idx]]:
                Returns[state].append(G)
                V[state] = np.average(Returns[state])

    return V


# %% [markdown]
# First, try your algorithm on the small $2\times3$ world above using an equiprobable policy and $\gamma = 0.9$. Depending on the number of episodes you use, you should get close to the true values:
#
# <table class="dataframe" border="1">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>0</th>
#       <th>1</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>3.283</td>
#       <td>3.616</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>4.409</td>
#       <td>5.556</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>6.349</td>
#       <td>0.000</td>
#     </tr>
#   </tbody>
# </table>
#
#
#

# %%
gamma = 0.9
world = World(2, 3)
world.add_terminal(1, 2, "+")
V = mc_fv_prediction(world, equiprobable_random_policy, 10000, gamma)
display(pd.DataFrame(V.T))

# %% [markdown]
# Try to run your MC prediction code on worlds of different sizes (be careful not to make your world too large or you should have multiple terminals that an agent is likely to hit, otherwise it may take too long). You can try to change the policy as well, but rememeber that the agent **must** eventually reach a terminal state under any policy that you try.

# %%
gamma = 0.9
world = World(7, 5)
world.add_terminal(1, 2, "+")
world.add_terminal(2, 4, "+")
V = mc_fv_prediction(world, equiprobable_random_policy, 10000, gamma)
display(pd.DataFrame(world.grid.T))
display(pd.DataFrame(V.T))

# %% [markdown]
# ### Exercise: Implement Monte Carlo-based prediction for state-action values
#
# There is one more step that has to be in place before we can start to optimize a policy: estimating state-action values, $q_\pi(s,a)$, based on experience. Above where we estimated $v_\pi$, we only needed to keep track of the average return observed for _each state_. However, in order to estimate state-action values, we need to compute the average return observed for _each state-action_ pair.
#
# That is, for every state $(0,0), (0,1), (0,2)...$ we need to compute different estimates for the four actions ```[ "up", "down", "left", "right" ]```


# %%
def mc_fv_state_action(world, policy, nb_episodes, gamma=0.9):
    Q = {}
    returns = {}

    # Initialize Q, policy, and returns
    for i in range(world.width):
        for j in range(world.height):
            Q[(i, j)] = {action: 0.0 for action in ACTIONS}
            returns[(i, j)] = {action: [] for action in ACTIONS}

    for _ in range(nb_episodes):
        # Exploring start - random initial state and action
        start_state = (0, 0)
        episode = generate_episode(world, policy, start_state)

        G = 0
        # Iterate forward through the episode
        for t, (state, action, reward) in enumerate(reversed(episode)):
            idx = len(episode) - t - 1
            G = gamma * G + reward

            # Check if the (state, action) pair has been seen before in the episode
            if (state, action) not in [(s, a) for s, a, _ in episode[:idx]]:
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])

    return Q


# %% [markdown]
# Try to experiment with your implementation by running it on different world sizes (be careful not to make your world too large or you should have multiple terminals that an agent is likely to hit, otherwise it may take too long), and try to experiment with different numbers of episodes:

# %%
gamma = 0.9
world = World(2, 3)
world.add_terminal(1, 2, "+")
Q = mc_fv_state_action(world, equiprobable_random_policy, 10000, gamma)
display(pd.DataFrame(Q))

# %% [markdown]
# ### Exercise: Implement on-policy Monte Carlo-based control with an $\epsilon$-soft policy
#
# You are now ready to implement MC-based control (see page 101 of [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html) for the algorithm).
#
# In your implementation, you need to update the state-action estimates like in the exercise above, but now, you also need implement an $ϵ$-soft policy that you can modify. How could you do that?
#
# _Hint_: You can either represent your policy explicitly. That is, for each state $(x,y)$ you have a ```dict``` with actions and their probabilities which you then update each time you step through an episode. When the policy is called, it then just returns the ```dict``` with action probablities corresponding to the current state.
#
# Alternatively, you can compute the action probabilities when your policy is called based on the current action-values estimates.


# %%
def mc_epsillon_soft(world, nb_episodes, gamma=0.9, epsilon=0.1):
    policy = {}
    Q = {}
    returns = {}

    # Initialize Q, policy, and returns
    for i in range(world.width):
        for j in range(world.height):
            Q[(i, j)] = {action: 0.0 for action in ACTIONS}
            policy[(i, j)] = {action: 1 / len(ACTIONS) for action in ACTIONS}
            returns[(i, j)] = {action: [] for action in ACTIONS}

    for _ in range(nb_episodes):
        start_state = (0, 0)
        episode = generate_episode(world, lambda x, y: policy[(x, y)], start_state)

        G = 0
        # Iterate forward through the episode
        for t, (state, action, reward) in enumerate(reversed(episode)):
            idx = len(episode) - t - 1
            G = gamma * G + reward
            # Check if the (state, action) pair has been seen before in the episode
            if (state, action) not in [(s, a) for s, a, _ in episode[:idx]]:
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])

                # Update policy to be greedy
                best_action = max(Q[state], key=Q[state].get)
                for a in ACTIONS:
                    policy[state][a] = (
                        1 - epsilon + epsilon / abs(policy[state][a])
                        if a == best_action
                        else epsilon / abs(policy[state][a])
                    )

    return Q, policy


# %% [markdown]
# Try to experiment with your implementation by running it on different world sizes (be careful not to make your world too large or you should have multiple terminals that an agent is likely to hit, otherwise it may take too long), try to experiment with different numbers of episodes, and different values of epsilon:

# %%
world = World(3, 3)
world.add_terminal(0, 0, "+")
Q, policy = mc_epsillon_soft(world, 10000)

display(pd.DataFrame(world.grid.T))
display(pd.DataFrame(policy).T)
display(pd.DataFrame(Q).T)

# %% [markdown]
# ### Optional exercise
#
# Try to implement exploring starts (see page 99 of [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html) for the algorithm). It should be straightforward and only require minimal changes to the code for the exercise above.


# %%
def mc_exploring_starts(world, nb_episodes, gamma=0.9):
    policy = {}
    Q = {}
    returns = {}

    # Initialize Q, policy, and returns
    for i in range(world.width):
        for j in range(world.height):
            Q[(i, j)] = {action: 0.0 for action in ACTIONS}
            policy[(i, j)] = {action: 1 / len(ACTIONS) for action in ACTIONS}
            returns[(i, j)] = {action: [] for action in ACTIONS}

    for _ in range(nb_episodes):
        # Exploring start - random initial state and action
        start_state = (
            random.randint(0, world.width - 1),
            random.randint(0, world.height - 1),
        )
        # Do not start in a terminal state or an obstacle
        while world.is_terminal(*start_state) or world.is_obstacle(*start_state):
            start_state = (
                random.randint(0, world.width - 1),
                random.randint(0, world.height - 1),
            )
        episode = generate_episode(world, lambda x, y: policy[(x, y)], start_state)

        G = 0
        # Iterate forward through the episode
        for t, (state, action, reward) in enumerate(reversed(episode)):
            idx = len(episode) - t - 1
            G = gamma * G + reward
            # Check if the (state, action) pair has been seen before in the episode
            if (state, action) not in [(s, a) for s, a, _ in episode[:idx]]:
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])

                # Update policy to be greedy
                best_action = max(Q[state], key=Q[state].get)
                for a in ACTIONS:
                    policy[state][a] = 1 if a == best_action else 0

    return Q, policy


# %%
gamma = 0.9
world = World(5, 5)
world.add_terminal(0, 0, "+")
world.add_terminal(4, 3, "+")
Q, policy = mc_exploring_starts(world, 10000, gamma)

display(pd.DataFrame(world.grid.T))
display(pd.DataFrame(policy).T)
display(pd.DataFrame(Q).T)
