import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from warehouse_env import WarehouseEnv

# ======================================================
# Hyperparameters
# ======================================================
EPISODES = 2000
GAMMA = 0.99
LR = 1e-3
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
SAVE_PATH = "qmix_policy.pt"

# ======================================================
# Utility: Global State
# ======================================================
def get_global_state(env):
    grid = env.grid.flatten()
    agent_info = []
    for agent in env.agents:
        x, y = env.agent_positions[agent]
        carrying = int(env.carrying[agent])
        agent_info.extend([x, y, carrying])
    return np.concatenate([grid, np.array(agent_info)], axis=0)

# ======================================================
# Agent Q Network
# ======================================================
class AgentQNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ======================================================
# QMIX Mixing Network
# ======================================================
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super().__init__()
        self.hyper_w = nn.Linear(state_dim, n_agents)
        self.hyper_b = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, state):
        w = torch.abs(self.hyper_w(state))  # monotonicity
        b = self.hyper_b(state)
        q_tot = torch.sum(agent_qs * w, dim=1, keepdim=True) + b
        return q_tot

# ======================================================
# Save policy
# ======================================================
def save_qmix_policy(agent_nets, mixer, path):
    torch.save(
        {
            "agents": [net.state_dict() for net in agent_nets],
            "mixer": mixer.state_dict(),
        },
        path
    )
    print(f"✅ QMIX policy saved to {path}")

# ======================================================
# Training
# ======================================================
def train_qmix():
    env = WarehouseEnv(
        grid_size=7,
        n_agents=2,
        n_items=3,
        max_steps=100
    )

    n_agents = len(env.agents)
    n_actions = env.action_spaces[env.agents[0]].n

    obs_dim = np.prod(env.observation_spaces[env.agents[0]]["grid"].shape) + 1
    state_dim = env.grid.size + n_agents * 3

    agents = [AgentQNet(obs_dim, n_actions) for _ in range(n_agents)]
    mixer = MixingNetwork(n_agents, state_dim)

    params = []
    for net in agents:
        params += list(net.parameters())
    params += list(mixer.parameters())

    optimizer = optim.Adam(params, lr=LR)
    loss_fn = nn.MSELoss()

    rewards_log = []

    epsilon = EPSILON

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = {}
            obs_tensors = []

            # ----- Action selection -----
            for i, agent in enumerate(env.agents):
                grid = obs[agent]["grid"].flatten()
                carry = np.array([obs[agent]["carrying"]], dtype=np.float32)
                o = np.concatenate([grid, carry])
                o_t = torch.tensor(o, dtype=torch.float32)

                if np.random.rand() < epsilon:
                    act = env.action_spaces[agent].sample()
                else:
                    with torch.no_grad():
                        act = agents[i](o_t).argmax().item()

                actions[agent] = act
                obs_tensors.append(o_t)

            # ----- Step environment -----
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            done = all(terms.values())
            team_reward = sum(rewards.values())
            total_reward += team_reward

            # ----- Compute Qs -----
            agent_qs = []
            next_agent_qs = []

            for i, agent in enumerate(env.agents):
                q = agents[i](obs_tensors[i])[actions[agent]]
                agent_qs.append(q)

                grid2 = next_obs[agent]["grid"].flatten()
                carry2 = np.array([next_obs[agent]["carrying"]], dtype=np.float32)
                o2 = np.concatenate([grid2, carry2])
                o2_t = torch.tensor(o2, dtype=torch.float32)

                with torch.no_grad():
                    q_next = agents[i](o2_t).max()
                next_agent_qs.append(q_next)

            agent_qs = torch.stack(agent_qs).unsqueeze(0)
            next_agent_qs = torch.stack(next_agent_qs).unsqueeze(0)

            state = torch.tensor(
                get_global_state(env),
                dtype=torch.float32
            ).unsqueeze(0)

            q_tot = mixer(agent_qs, state)
            q_tot_next = mixer(next_agent_qs, state)

            target = team_reward + GAMMA * q_tot_next
            loss = loss_fn(q_tot, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

        rewards_log.append(total_reward)
        np.save("qmix_rewards.npy", rewards_log)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if ep % 100 == 0:
            print(f"Episode {ep}, Team Reward: {total_reward:.2f}")

    save_qmix_policy(agents, mixer, SAVE_PATH)
    print("QMIX training finished.")

# ======================================================
# Run
# ======================================================
if __name__ == "__main__":
    train_qmix()
