import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from warehouse_env import WarehouseEnv

# =========================================================
# Evaluation settings
# =========================================================
EVAL_EPISODES = 50
EVAL_EPS = 0.05   # IMPORTANT FIX: small exploration during evaluation

# =========================================================
# Helper functions
# =========================================================
def flatten_obs(obs):
    grid = obs["grid"].flatten()
    carrying = np.array([obs["carrying"]])
    return np.concatenate([grid, carrying])

def obs_to_state(obs):
    grid = obs["grid"].flatten()
    carrying = obs["carrying"]
    return tuple(grid.tolist() + [carrying])

# =========================================================
# IQL Evaluation
# =========================================================
def evaluate_iql(env, Q, episodes=EVAL_EPISODES):
    total_rewards = []
    items_delivered_list = []
    collisions_list = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        ep_reward = 0.0
        ep_items = 0
        ep_collisions = 0

        while not done:
            actions = {}
            for agent in env.agents:
                state = obs_to_state(obs[agent])

                if np.random.rand() < EVAL_EPS:
                    actions[agent] = env.action_spaces[agent].sample()
                else:
                    actions[agent] = int(np.argmax(Q[state]))

            next_obs, rewards, terms, truncs, info = env.step(actions)

            ep_reward += sum(rewards.values())
            ep_items += sum(agent_info["items_delivered"] for agent_info in info.values())
            ep_collisions += sum(agent_info["collisions"] for agent_info in info.values())

            obs = next_obs
            done = all(terms.values())

        total_rewards.append(ep_reward)
        items_delivered_list.append(ep_items)
        collisions_list.append(ep_collisions)

    return total_rewards, items_delivered_list, collisions_list

# =========================================================
# VDN Evaluation
# =========================================================
class QNetwork(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def evaluate_vdn(env, policy_path, episodes=EVAL_EPISODES):
    agents = env.agents
    obs_size = (2 * env.obs_radius + 1) ** 2 + 1
    n_actions = env.action_spaces[agents[0]].n

    checkpoint = torch.load(policy_path, map_location="cpu")

    q_nets = {agent: QNetwork(obs_size, n_actions) for agent in agents}
    for i, agent in enumerate(agents):
        q_nets[agent].load_state_dict(checkpoint[f"agent_{i}"])
        q_nets[agent].eval()

    rewards_log = []
    items_delivered_list = []
    collisions_list = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        ep_reward = 0.0
        ep_items = 0
        ep_collisions = 0

        while not done:
            actions = {}
            for agent in agents:
                o = flatten_obs(obs[agent])
                o_t = torch.tensor(o, dtype=torch.float32)

                if np.random.rand() < EVAL_EPS:
                    actions[agent] = env.action_spaces[agent].sample()
                else:
                    with torch.no_grad():
                        actions[agent] = int(torch.argmax(q_nets[agent](o_t)))

            next_obs, rewards, terms, truncs, info = env.step(actions)

            ep_reward += sum(rewards.values())
            ep_items += sum(agent_info["items_delivered"] for agent_info in info.values())
            ep_collisions += sum(agent_info["collisions"] for agent_info in info.values())

            obs = next_obs
            done = all(terms.values())

        rewards_log.append(ep_reward)
        items_delivered_list.append(ep_items)
        collisions_list.append(ep_collisions)

    return rewards_log, items_delivered_list, collisions_list

# =========================================================
# QMIX Evaluation
# =========================================================
class AgentQNet(torch.nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class MixingNetwork(torch.nn.Module):
    def __init__(self, n_agents, state_dim):
        super().__init__()
        self.hyper_w = torch.nn.Linear(state_dim, n_agents)
        self.hyper_b = torch.nn.Linear(state_dim, 1)

    def forward(self, agent_qs, state):
        w = torch.abs(self.hyper_w(state))
        b = self.hyper_b(state)
        q_tot = torch.sum(agent_qs * w, dim=1, keepdim=True) + b
        return q_tot

def get_global_state(env):
    grid = env.grid.flatten()
    agent_info = []
    for agent in env.agents:
        x, y = env.agent_positions[agent]
        carrying = int(env.carrying[agent])
        agent_info.extend([x, y, carrying])
    return np.concatenate([grid, np.array(agent_info)], axis=0)

def evaluate_qmix(env, policy_path, episodes=EVAL_EPISODES):
    checkpoint = torch.load(policy_path, map_location="cpu")

    n_agents = len(env.agents)
    obs_dim = np.prod(env.observation_spaces[env.agents[0]]["grid"].shape) + 1
    state_dim = env.grid.size + n_agents * 3
    n_actions = env.action_spaces[env.agents[0]].n

    agents = [AgentQNet(obs_dim, n_actions) for _ in range(n_agents)]
    mixer = MixingNetwork(n_agents, state_dim)

    for i in range(n_agents):
        agents[i].load_state_dict(checkpoint["agents"][i])
        agents[i].eval()

    mixer.load_state_dict(checkpoint["mixer"])
    mixer.eval()

    rewards_log = []
    items_delivered_list = []
    collisions_list = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        ep_reward = 0.0
        ep_items = 0
        ep_collisions = 0

        while not done:
            actions = {}

            for i, agent in enumerate(env.agents):
                grid = obs[agent]["grid"].flatten()
                carry = np.array([obs[agent]["carrying"]])
                o = torch.tensor(np.concatenate([grid, carry]), dtype=torch.float32)

                q_values = agents[i](o)

                if np.random.rand() < EVAL_EPS:
                    actions[agent] = env.action_spaces[agent].sample()
                else:
                    actions[agent] = int(torch.argmax(q_values))

            next_obs, rewards, terms, truncs, info = env.step(actions)

            ep_reward += sum(rewards.values())
            ep_items += sum(agent_info["items_delivered"] for agent_info in info.values())
            ep_collisions += sum(agent_info["collisions"] for agent_info in info.values())

            obs = next_obs
            done = all(terms.values())

        rewards_log.append(ep_reward)
        items_delivered_list.append(ep_items)
        collisions_list.append(ep_collisions)

    return rewards_log, items_delivered_list, collisions_list

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    env = WarehouseEnv(grid_size=7, n_agents=2, n_items=3, max_steps=100)

    # -------- Load IQL --------
    with open("iql.pkl", "rb") as f:
        Q_iql_loaded = pickle.load(f)

    Q_iql = defaultdict(
        lambda: np.zeros(env.action_spaces[env.agents[0]].n, dtype=float),
        Q_iql_loaded
    )

    # -------- Evaluate --------
    iql_results = evaluate_iql(env, Q_iql)
    vdn_results = evaluate_vdn(env, "vdn_policy.pt")
    qmix_results = evaluate_qmix(env, "qmix_policy.pt")

    results = {
        "IQL": iql_results,
        "VDN": vdn_results,
        "QMIX": qmix_results
    }

    # -------- Summary --------
    print("\nEvaluation Summary:")
    print(f"{'Algorithm':<10} | {'Avg Reward':<12} | {'Avg Items':<10} | {'Avg Collisions':<15} | {'Coop Score':<12}")
    print("-" * 70)

    for name, (rewards, items, collisions) in results.items():
        avg_reward = np.mean(rewards)
        avg_items = np.mean(items)
        avg_collisions = np.mean(collisions)
        coop_score = (avg_items / (avg_collisions + 1)) * 100

        print(f"{name:<10} | {avg_reward:<12.2f} | {avg_items:<10.2f} | {avg_collisions:<15.2f} | {coop_score:<12.2f}")

    # -------- Plot --------
    plt.figure(figsize=(10, 6))
    for name, (rewards, _, _) in results.items():
        plt.plot(rewards, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Team Reward")
    plt.title("IQL vs VDN vs QMIX (Evaluation)")
    plt.legend()
    plt.grid(True)
    plt.show()
