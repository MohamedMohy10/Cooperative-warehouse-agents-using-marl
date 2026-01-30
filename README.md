# Cooperative Warehouse Robots for Resource Collection using MARL

This project investigates **cooperative multi-agent reinforcement learning (MARL)** in a simplified warehouse environment. Multiple autonomous agents must coordinate under **partial observability** to collect distributed items and deliver them to a central depot while avoiding collisions.

The goal is to analyze how different MARL paradigms handle **coordination, credit assignment, and learning stability** under identical environmental conditions.

---

## Overview

- Environment: Grid-based warehouse
- Task: Collect items and deliver them to a depot
- Agents: Multiple homogeneous robots
- Objective: Maximize shared team reward through cooperation

---

## Environment Design

- Grid size: **7 × 7**
- Agents: **2**
- Items: **3**
- Depot: **1 fixed location**
- Episode length: **100 steps**
- Observability: **Partial (local window per agent)**

### Observations
Each agent observes:
- A local grid window centered on its position
- Nearby items, depot, and other agents
- Binary indicator of whether it is carrying an item

### Actions
Discrete action space:
- Move up / down / left / right
- Pick up item
- Drop item at depot
- No-op

Invalid actions (collisions, illegal pickup/drop) result in penalties.

---

## Reward Design

The reward function combines **shared task rewards** with **dense shaping**:

- **Shared team reward** when any agent delivers an item
- Positive shaping rewards:
  - Moving toward items (when not carrying)
  - Moving toward depot (when carrying)
  - Successful item pickup
- Penalties:
  - Collisions
  - Invalid actions
---

## MARL Algorithms

All methods use **decentralized execution**.

### Independent Q-Learning (IQL)
- Each agent learns its own Q-function independently
- Treats other agents as part of the environment
- Stable learning but limited coordination

### Value Decomposition Networks (VDN)
- Centralized training with additive value decomposition:
  \[
  Q_{tot}(s, a) = \sum_i Q_i(o_i, a_i)
  \]
- Encourages cooperation via shared value optimization
- Higher performance but increased variance

### QMIX
- Uses a learned mixing network conditioned on the global state
- Enforces monotonicity between individual and global Q-values
- Higher representational capacity but sensitive to reward design

---

## Implementation

- Language: **Python**
- Framework: **PettingZoo (parallel API)**
- Deep learning: **PyTorch**

Key components:
- Custom warehouse environment
- Separate training scripts for each algorithm
- Evaluation pipeline with quantitative metrics
- ASCII-based visualization for debugging and analysis

---

## Evaluation

### Metrics
- **Team reward per episode**
- **Evaluation reward**
- **Learning stability**
- **Emergent cooperative behavior**

---

## Key Findings

- Complex MARL algorithms do **not automatically outperform** simpler methods
- Dense reward shaping favors decentralized or lightly centralized learning
- Centralized value decomposition methods are **highly sensitive to reward design**
- Cooperative behavior can emerge even without explicit coordination mechanisms

---

Future extensions:
- More agents and items
- Sparse or delayed rewards
- Dynamic obstacles
- Inter-agent communication
- Role specialization

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
