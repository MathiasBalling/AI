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
# # Temporal difference prediction and control
#
# In this notebook, you will implement temporal difference approaches to prediction and control described in [Sutton and Barto's book, Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html). We will use the grid ```World``` class from the previous lectures.


# %% [markdown]
# ### Imports

# %%
import numpy as np
import random
import sys  # We use sys to get the max value of a float
import pandas as pd  # We only use pandas for displaying tables nicely
from IPython.display import display

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
        # Create an empty windy grid
        self.windy_grid = {}
        self.initialize_windy_grid()

    def initialize_windy_grid(self):
        for x in range(self.width):
            for y in range(self.height):
                self.windy_grid[(x, y)] = (" ", 0)

    def add_obstacle(self, start_x, start_y, end_x=None, end_y=None):
        """
        Create an obstacle in either a single cell or rectangle.
        """
        if end_x == None:
            end_x = start_x
        if end_y == None:
            end_y = start_y

        self.grid[start_x : end_x + 1, start_y : end_y + 1] = OBSTACLES[0]

    def add_wind(self, x, y, wind_dir, wind_strength):
        """
        Add wind to a cell.
        x, y: coordinates of the cell
        wind_dir: one of the four cardinal directions
        wind_strength: the strength of the wind
        """
        self.windy_grid[(x, y)] = (wind_dir, wind_strength)

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
        elif action == "up-left":
            x -= 1
            y -= 1
        elif action == "up-right":
            x += 1
            y -= 1
        elif action == "down-left":
            x -= 1
            y += 1
        elif action == "down-right":
            x += 1
            y += 1

        # If the next state is an obstacle, stay in the current state
        return (x, y) if not self.is_obstacle(x, y) else current_state

    def get_next_state_windy(self, current_state, action):
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

        if self.windy_grid[x, y][0] == "up":
            y -= self.windy_grid[x, y][1]
        elif self.windy_grid[x, y][0] == "down":
            y += self.windy_grid[x, y][1]
        elif self.windy_grid[x, y][0] == "left":
            x -= self.windy_grid[x, y][1]
        elif self.windy_grid[x, y][0] == "right":
            x += self.windy_grid[x, y][1]

        if action == "up":
            y -= 1
        elif action == "down":
            y += 1
        elif action == "left":
            x -= 1
        elif action == "right":
            x += 1
        elif action == "up-left":
            x -= 1
            y -= 1
        elif action == "up-right":
            x += 1
            y -= 1
        elif action == "down-left":
            x -= 1
            y += 1
        elif action == "down-right":
            x += 1
            y += 1

        # Limit values to be within the grid
        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)

        # If the next state is an obstacle, stay in the current state
        return (x, y) if not self.is_obstacle(x, y) else current_state


# %% [markdown]
# ## A simple world and a simple policy

# %%
ACTIONS = ("up", "down", "left", "right")
world = World(2, 3)

# Since we only focus on episodic tasks, we must have a terminal state that the
# agent eventually reaches
world.add_terminal(1, 2, "+")


def equiprobable_random_policy(x, y):
    return {k: 1 / len(ACTIONS) for k in ACTIONS}


print(world.grid.T)

# %% [markdown]
# ## Exercise: TD prediction
#
# You should implement TD prediction for estimating $V≈v_\pi$. See page 120 of [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html).
#
#
# To implement TD prediction, the agent has to interact with the world for a certain number of episodes. However, unlike in the Monte Carlo case, we do not rely on complete sample runs, but instead update estimates (for prediction and control) and the policy (for control only) each time step in an episode.
#

# %% [markdown]
# Below, you can see the code for running an episode, with a TODO where you have to add your code for prediction. Also, play with the parameters ```alpha``` and ```EPISODES```, you will typically need a lot more than 10 episodes for an agent to learn anything.

# %%
# Global variable to keep track of current estimates
V = {}

# Our step size / learing rate
alpha = 0.05

# Discount factor
gamma = 0.9

# Episodes to run
EPISODES = 10


def TD_prediction_run_episode(world, policy, start_state):
    current_state = start_state
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

        if current_state not in V:
            V[current_state] = 0
        if next_state not in V:
            V[current_state] = 0

        print(
            f"Current state (S) = {current_state}, next_state S' = {next_state}, reward = {reward}"
        )

        # Move the agent to the new state
        current_state = next_state


for episode in range(EPISODES):
    print(f"Episode {episode + 1 }/{EPISODES}:")
    TD_prediction_run_episode(world, equiprobable_random_policy, (0, 0))


# %% [markdown]
# ## Exercise: SARSA
#
# Implement and test SARSA with an $\epsilon$-greedy policy. See page 130 of [Introduction to Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html) on different worlds. Make sure that it is easy to show a learnt policy (most probable action in each state).
#


# %%
def greedy_policy(Q, state, epsilon):
    """
    Selects an action using an epsilon-greedy policy.

    Args:
        Q (dict): The Q-table containing state-action values.
        state (tuple): The current state.
        epsilon (float): The probability of selecting a random action (exploration).

    Returns:
        action (str): The action selected.
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        if state not in Q:
            Q[state] = {action: 0.0 for action in ACTIONS}
        return max(Q[state], key=Q[state].get)


def SARSA(world: World, n_episodes, gamma=0.9, alpha=0.1, epsilon=0.1, use_wind=False):
    """
    Implements the SARSA algorithm for on-policy TD control.

    Args:
        world (World): The environment in which the agent operates.
        n_episodes (int): The number of episodes to run the algorithm.
        gamma (float): The discount factor.
        alpha (float): The learning rate.
        epsilon (float): The probability of selecting a random action (exploration).

    Returns:
        Q (dict): The Q-table containing state-action values.
    """
    Q = {}

    for _ in range(n_episodes):
        # Initialize the starting state
        start_state = (
            random.randint(0, world.width - 1),
            random.randint(0, world.height - 1),
        )
        # Make sure the starting state is not a terminal or obstacle
        while world.is_terminal(*start_state) or world.is_obstacle(*start_state):
            start_state = (
                random.randint(0, world.width - 1),
                random.randint(0, world.height - 1),
            )

        # Initialize the starting action
        current_action = greedy_policy(Q, start_state, epsilon)

        current_state = start_state
        while not world.is_terminal(*current_state):
            # Get the next state, action, and reward
            if use_wind:
                next_state = world.get_next_state_windy(current_state, current_action)
            else:
                next_state = world.get_next_state(current_state, current_action)

            next_action = greedy_policy(Q, next_state, epsilon)
            reward = world.get_reward(*next_state)

            # Helper to initialize Q-values for new states and actions
            if current_state not in Q:
                Q[current_state] = {action: 0.0 for action in ACTIONS}
            if next_state not in Q:
                Q[next_state] = {action: 0.0 for action in ACTIONS}

            # Update the Q-value for the current state-action pair
            Q[current_state][current_action] += alpha * (
                reward
                + gamma * Q[next_state][next_action]
                - Q[current_state][current_action]
            )

            # Move to the next state and action
            current_action = next_action
            current_state = next_state

    return Q


# %%
ACTIONS = ("up", "down", "left", "right")
world = World(3, 3)
world.add_terminal(1, 2, "+")
display(pd.DataFrame(world.grid.T))

Q = SARSA(world, 50000)
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
# ## Exercise: Windy Gridworld
#
# Implement the Windy Gridworld (Example 6.5 on page 130 in the book) and test your SARSA implementation on the Windy Gridworld, first with the four actions (```up, down, left, right```) that move the agent in the cardinal directions, and then with King's moves as described in Exercise 6.9. How long does it take to learn a good policy for different values of $\alpha$ and $\epsilon$?


# %% [markdown]
# ### Without King's Moves (Windy Gridworld)
# %%
ACTIONS = ("up", "down", "left", "right")
world = World(width=5, height=5)

# Add wind to the grid in x=2 with strength 1
for i in range(world.height):
    world.add_wind(2, i, "up", 1)

world.add_terminal(4, 4, "+")

display(pd.DataFrame(world.grid.T))

Q = SARSA(world, 30000, use_wind=True)
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
# ### With King's Moves (Windy Gridworld)

# %%
ACTIONS = (
    "up",
    "down",
    "left",
    "right",
    "up-left",
    "up-right",
    "down-left",
    "down-right",
)
Q = SARSA(world, 30000, use_wind=True)
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
