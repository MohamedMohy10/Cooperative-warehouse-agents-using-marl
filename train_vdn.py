import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warehouse_env import WarehouseEnv

# ======================================================
# Hyperparameters
# ======================================================
EPISODES = 2000
GAMMA = 0.99
LR = 1e-3
EPSILON = 0.2
TARGET_UPDATE = 200
SAVE_PATH = "vdn_policy.pt"

# ======================================================
# Q-Network
# ======================================================
class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ======================================================
# Observation processing
# ======================================================
def flatten_obs(obs):
    grid = obs["grid"].flatten()
    carrying = np.array([obs["carrying"]], dtype=np.float32)
    return np.concatenate([grid, carrying])

# ======================================================
# Save policy
# ======================================================
def save_vdn_policy(q_nets, path):
    torch.save(
        {agent: q_nets[agent].state_dict() for agent in q_nets},
        path
    )
    print(f"✅ VDN policy saved to {path}")

# ======================================================
# Training
# ======================================================
def train_vdn(env):
    agents = env.agents
    obs_size = (2 * env.obs_radius + 1) ** 2 + 1
    n_actions = env.action_spaces[agents[0]].n

    q_nets = {agent: QNetwork(obs_size, n_actions) for agent in agents}
    target_nets = {agent: QNetwork(obs_size, n_actions) for agent in agents}

    for agent in agents:
        target_nets[agent].load_state_dict(q_nets[agent].state_dict())

    optimizers = {
        agent: torch.optim.Adam(q_nets[agent].parameters(), lr=LR)
        for agent in agents
    }

    episode_rewards = []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = {}
            obs_tensors = {}

            # Action selection
            for agent in agents:
                o = flatten_obs(obs[agent])
                obs_tensors[agent] = torch.tensor(o, dtype=torch.float32)

                if np.random.rand() < EPSILON:
                    actions[agent] = np.random.randint(n_actions)
                else:
                    q_vals = q_nets[agent](obs_tensors[agent])
                    actions[agent] = torch.argmax(q_vals).item()

            next_obs, rewards, terms, truncs, _ = env.step(actions)
            team_reward = sum(rewards.values())
            total_reward += team_reward

            # -------- VDN UPDATE --------
            q_sum = 0
            target_q_sum = 0

            for agent in agents:
                q_vals = q_nets[agent](obs_tensors[agent])
                q_sum += q_vals[actions[agent]]

                next_o = flatten_obs(next_obs[agent])
                next_o = torch.tensor(next_o, dtype=torch.float32)
                target_q_sum += torch.max(target_nets[agent](next_o))

            td_target = team_reward + GAMMA * target_q_sum
            loss = (q_sum - td_target.detach()) ** 2

            for opt in optimizers.values():
                opt.zero_grad()
            loss.backward()
            for opt in optimizers.values():
                opt.step()

            obs = next_obs
            done = all(terms.values())

        episode_rewards.append(total_reward)
        np.save("vdn_rewards.npy", episode_rewards)

        if ep % 100 == 0:
            print(f"Episode {ep}, Team Reward: {total_reward:.2f}")

        if ep % TARGET_UPDATE == 0:
            for agent in agents:
                target_nets[agent].load_state_dict(q_nets[agent].state_dict())

    save_vdn_policy(q_nets, SAVE_PATH)
    return episode_rewards

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    env = WarehouseEnv(
        grid_size=7,
        n_agents=2,
        n_items=3,
        max_steps=100
    )

    train_vdn(env)
    print("Training finished.")
