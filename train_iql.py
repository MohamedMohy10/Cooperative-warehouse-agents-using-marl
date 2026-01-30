import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from warehouse_env import WarehouseEnv
from pettingzoo import ParallelEnv

# -----------------------------
# Hyperparameters
# -----------------------------
EPISODES = 2000
ALPHA = 0.1        # learning rate
GAMMA = 0.95       # discount factor
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# -----------------------------
# Helper functions
# -----------------------------
def obs_to_state(obs):
    """
    Convert observation dict to a hashable state.
    """
    grid = obs["grid"].flatten()
    carrying = obs["carrying"]
    return tuple(grid.tolist() + [carrying])

def epsilon_greedy(Q, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

# -----------------------------
# Training Loop
# -----------------------------
env = WarehouseEnv(grid_size=7, n_agents=2, n_items=3)

# Q-tables (one per agent)
Q = {
    agent: defaultdict(lambda: np.zeros(env.action_spaces[agent].n))
    for agent in env.agents
}

episode_rewards = []

for episode in range(EPISODES):
    observations, _ = env.reset()
    states = {
        agent: obs_to_state(observations[agent])
        for agent in env.agents
    }

    total_reward = 0
    done = False

    while not done:
        # Select actions
        actions = {}
        for agent in env.agents:
            actions[agent] = epsilon_greedy(
                Q[agent],
                states[agent],
                env.action_spaces[agent].n,
                EPSILON,
            )

        next_obs, rewards, terminations, truncations, _ = env.step(actions)

        next_states = {
            agent: obs_to_state(next_obs[agent])
            for agent in env.agents
        }

        # Q-learning update
        for agent in env.agents:
            a = actions[agent]
            r = rewards[agent]
            s = states[agent]
            s_next = next_states[agent]

            Q[agent][s][a] += ALPHA * (
                r + GAMMA * np.max(Q[agent][s_next]) - Q[agent][s][a]
            )

            total_reward += r

        states = next_states
        done = all(terminations.values())

    episode_rewards.append(total_reward)
    np.save("iql_rewards.npy", episode_rewards)

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

# -----------------------------
# Save Q-tables for later evaluation
# -----------------------------
Q_to_save = {
    agent: dict(Q[agent])   # remove lambda
    for agent in Q
}
with open("iql_qtables.pkl", "wb") as f:
    pickle.dump(Q_to_save, f)
print("Q-tables saved to 'iql_qtables.pkl'")

# -----------------------------
# Plot Learning Curve
# -----------------------------
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Team Reward")
plt.title("Independent Q-Learning: Team Reward per Episode")
plt.grid()
plt.show()
