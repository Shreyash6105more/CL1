import numpy as np
import matplotlib.pyplot as plt
import random

maze = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [1, 1, 0, 0]
])

start = (0, 0)
goal = (3, 3)

rows, cols = maze.shape

actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
arrows = ['↑', '→', '↓', '←']

Q = np.zeros((rows * cols, 4))

alpha = 0.1
gamma = 0.9
epsilon = 1.0

episodes = 1000
rewards = []

def get_state(pos):
    return pos[0] * cols + pos[1]

def move(pos, action):
    r = pos[0] + actions[action][0]
    c = pos[1] + actions[action][1]

    if r < 0 or r >= rows or c < 0 or c >= cols or maze[r, c] == 1:
        return pos, -5, False

    new_pos = (r, c)

    if new_pos == goal:
        return new_pos, 50, True

    return new_pos, -1, False

for ep in range(episodes):
    pos = start
    total_reward = 0

    for step in range(100):
        state = get_state(pos)

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        new_pos, reward, done = move(pos, action)
        next_state = get_state(new_pos)

        target = reward if done else reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (target - Q[state, action])

        pos = new_pos
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)

    epsilon = max(0.05, epsilon * 0.995)

plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-learning Training")
plt.show()

print("Learned Maze Policy:")

for r in range(rows):
    for c in range(cols):
        if maze[r, c] == 1:
            print("#", end=" ")
        elif (r, c) == goal:
            print("G", end=" ")
        else:
            state = get_state((r, c))
            print(arrows[np.argmax(Q[state])], end=" ")
    print()

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports `numpy` for matrix/Q-table operations, `matplotlib` for plotting learning curves, and `random` for exploration.
# - Defines a 4x4 grid `maze` (`0` = open path, `1` = wall), `start = (0, 0)`, `goal = (3, 3)`, and 4 movement directions (up, right, down, left).
# - Initializes a 16x4 `Q` table (`rows * cols` states × `4` actions) with zeros, along with learning rate (`alpha`), discount factor (`gamma`), and exploration rate (`epsilon`).
# - `get_state(pos)` maps a 2D `(row, col)` coordinate into a unique 1D state index (`0` to `15`).
# - `move(pos, action)` simulates a step: hitting a wall/boundary stays in place with `-5` penalty; reaching `goal` gives `+50` reward and ends the episode (`done=True`); normal steps cost `-1` to encourage shortest paths.
# - Training loop runs 1,000 episodes (max 100 steps each) using an **epsilon-greedy** strategy to balance random exploration and exploiting the best learned Q-value (`np.argmax(Q[state])`).
# - Updates `Q[state, action]` using the Bellman Q-learning update rule, then decays `epsilon` by `0.995` (down to a minimum of `0.05`) after each episode.
# - Plots total reward per episode to visualize convergence, then prints the optimal learned policy grid using directional arrows, `#` for walls, and `G` for the goal.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is Reinforcement Learning (RL), and how does Q-Learning work?
# A1: RL is a learning paradigm where an agent learns optimal actions by interacting with an environment and receiving rewards/penalties. Q-Learning is a model-free RL algorithm that learns the quality (`Q-value`) of taking each action in every state.
#
# Q2: What is a Q-Table, and why is its shape `(16, 4)` here?
# A2: It is a lookup table storing expected cumulative rewards for state-action pairs. The 4×4 maze has 16 possible cell positions (states) and 4 possible moves (actions) per cell.
#
# Q3: What do the hyperparameters `alpha`, `gamma`, and `epsilon` represent?
# A3: `alpha` (0.1) is the learning rate (how fast new info overrides old Q-values); `gamma` (0.9) is the discount factor (importance of future rewards vs. immediate rewards); `epsilon` (1.0) is the exploration probability.
#
# Q4: Explain the Epsilon-Greedy strategy and why `epsilon` is decayed (`epsilon * 0.995`).
# A4: With probability `epsilon`, the agent explores a random action; otherwise it exploits the highest Q-value action. Decaying `epsilon` lets the agent explore heavily early on and exploit its learned optimal path later.
#
# Q5: Why does a valid normal step return a `-1` reward instead of `0`?
# A5: A small negative step penalty encourages the agent to reach the goal in the fewest possible steps rather than wandering aimlessly.
#
# Q6: What is the Bellman Equation used for Q-Learning update in this code?
# A6: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$, which updates the current Q-value toward the immediate reward plus the discounted maximum future Q-value.
#
# Q7: Why is Q-Learning called an "off-policy" algorithm?
# A7: Because it updates Q-values assuming the greedy optimal action (`np.max(Q[next_state])`) will be taken in the next state, even while following an exploratory epsilon-greedy policy during training.max(Q[next_state])`) will be taken in the next state, even while following an exploratory epsilon-greedy policy during training.7: Because it updates Q-values assuming the greedy optimal action (`np.max(Q[next_state])`) will be taken in the next state, even while following an exploratory epsilon-greedy policy during training.: Because it updates Q-values assuming the greedy optimal action (`np.max(Q[next_state])`) will be taken in the next state, even while following an exploratory epsilon-greedy policy during 