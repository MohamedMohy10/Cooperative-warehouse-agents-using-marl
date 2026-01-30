from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class WarehouseEnv(ParallelEnv):
    metadata = {"name": "warehouse_v0"}

    def __init__(self, grid_size=7, n_agents=2, n_items=3, obs_radius=2, max_steps=100):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_items = n_items
        self.obs_radius = obs_radius
        self.max_steps = max_steps

        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]

        # Grid encoding
        self.EMPTY = 0
        self.OBSTACLE = 1
        self.ITEM = 2
        self.DEPOT = 3
        self.AGENT = 4

        # Actions
        # 0: up, 1: down, 2: left, 3: right, 4: pick, 5: drop, 6: noop
        self.action_spaces = {
            agent: spaces.Discrete(7) for agent in self.agents
        }

        obs_size = 2 * obs_radius + 1
        self.observation_spaces = {
            agent: spaces.Dict({
                "grid": spaces.Box(
                    low=0,
                    high=4,
                    shape=(obs_size, obs_size),
                    dtype=np.int32,
                ),
                "carrying": spaces.Discrete(2),
            })
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.agents = self.possible_agents[:]

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place depot at center
        self.depot_pos = (self.grid_size // 2, self.grid_size // 2)
        self.grid[self.depot_pos] = self.DEPOT

        # Place items randomly
        self.items = set()
        while len(self.items) < self.n_items:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if self.grid[pos] == self.EMPTY:
                self.items.add(pos)
                self.grid[pos] = self.ITEM

        # Place agents
        self.agent_positions = {}
        self.carrying = {agent: False for agent in self.agents}
        for agent in self.agents:
            while True:
                pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
                if self.grid[pos] == self.EMPTY:
                    self.agent_positions[agent] = pos
                    break

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        items_delivered_this_step = 0
        collisions_this_step = 0
        self.steps += 1
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # Save previous positions for reward shaping
        prev_positions = self.agent_positions.copy()

        # Resolve movements
        desired_positions = {}
        for agent, action in actions.items():
            x, y = self.agent_positions[agent]
            if action == 0:
                desired_positions[agent] = (x - 1, y)
            elif action == 1:
                desired_positions[agent] = (x + 1, y)
            elif action == 2:
                desired_positions[agent] = (x, y - 1)
            elif action == 3:
                desired_positions[agent] = (x, y + 1)
            else:
                desired_positions[agent] = (x, y)

        # Collision handling
        occupied = set(self.agent_positions.values())
        for agent, new_pos in desired_positions.items():
            if (
                0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size
                and new_pos not in occupied
            ):
                self.agent_positions[agent] = new_pos
            else:
                rewards[agent] -= 0.1  # collision penalty
                collisions_this_step += 1

        # Handle pick/drop
        for agent, action in actions.items():
            pos = self.agent_positions[agent]
            if action == 4 and not self.carrying[agent] and pos in self.items:
                self.carrying[agent] = True
                self.items.remove(pos)
                self.grid[pos] = self.EMPTY
                rewards[agent] += 10.0 # reward for picking up
            elif action == 4 and not self.carrying[agent] and pos not in self.items:
                rewards[agent] -= 1.0  # penalty for failed pick
            elif action == 5 and self.carrying[agent] and pos == self.depot_pos:
                self.carrying[agent] = False
                rewards[agent] += 40.0 # reward for successful drop
                items_delivered_this_step += 1
                for a in rewards:
                    rewards[a] += 60.0  # shared reward
                # Reward shaping
            if self.carrying[agent]:
                rewards[agent] += 1 # carrying reward
                rewards[agent] -= 0.05 # small penalty for time carrying
                # Reward for moving closer to depot
                old_dist = abs(prev_positions[agent][0] - self.depot_pos[0]) + abs(prev_positions[agent][1] - self.depot_pos[1])
                new_dist = abs(self.agent_positions[agent][0] - self.depot_pos[0]) + abs(self.agent_positions[agent][1] - self.depot_pos[1])
                rewards[agent] += 0.5 * (old_dist - new_dist)  # positive if closer
            else:
                # Reward for moving closer to nearest item
                if self.items:
                    distances_prev = [abs(prev_positions[agent][0] - ix) + abs(prev_positions[agent][1] - iy) for (ix, iy) in self.items]
                    distances_new = [abs(self.agent_positions[agent][0] - ix) + abs(self.agent_positions[agent][1] - iy) for (ix, iy) in self.items]
                    reward_shape = max([d_prev - d_new for d_prev, d_new in zip(distances_prev, distances_new)])
                    rewards[agent] += 0.3 * reward_shape
        if self.steps >= self.max_steps:
            for agent in terminations:
                terminations[agent] = True

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        for agent in infos:
            infos[agent]["items_delivered"] = items_delivered_this_step
            infos[agent]["collisions"] = collisions_this_step
        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        x, y = self.agent_positions[agent]
        r = self.obs_radius
        obs = np.ones((2 * r + 1, 2 * r + 1), dtype=np.int32)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = x + dx, y + dy
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    obs[dx + r, dy + r] = self.grid[gx, gy]

        return {
            "grid": obs,
            "carrying": int(self.carrying[agent]),
        }
    def render(self):
        """Clean ASCII render of the warehouse environment."""
        grid = self.grid.copy()
        for agent, pos in self.agent_positions.items():
            grid[pos] = self.AGENT

        symbol_map = {
            self.EMPTY: ".",   # empty
            self.ITEM: "I",    # item
            self.DEPOT: "D",   # depot
            self.AGENT: "A",   # agent
            self.OBSTACLE: "#",  # defined for future use
        }

        print("\nWarehouse State:")
        for i in range(self.grid_size):
            row = ""
            for j in range(self.grid_size):
                row += symbol_map.get(grid[i, j], "?") + " "
            print(row)
        print()
    def render_matplotlib(self):
        """Clean matplotlib visualization of the warehouse."""
        grid = self.grid.copy()
        for agent, pos in self.agent_positions.items():
            grid[pos] = self.AGENT


        plt.figure(figsize=(4, 4))
        plt.imshow(grid)
        plt.xticks(range(self.grid_size))
        plt.yticks(range(self.grid_size))
        plt.grid(True)
        plt.title("Cooperative Warehouse Environment")
        plt.show()


if __name__ == "__main__":
    env = WarehouseEnv(grid_size=7, n_agents=2, n_items=3)

    observations, infos = env.reset()
    print("Initial observations:")
    for agent, obs in observations.items():
        print(agent, obs)

    for step in range(5):
        actions = {
            agent: env.action_spaces[agent].sample()
            for agent in env.agents
        }

        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

        print(f"\nStep {step}")
        print("Actions:", actions)
        print("Rewards:", rewards)

        if all(terminations.values()):
            break
