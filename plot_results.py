import numpy as np
import matplotlib.pyplot as plt

iql = np.load("iql_rewards.npy")
vdn = np.load("vdn_rewards.npy")
qmix = np.load("qmix_rewards.npy")

def smooth(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode="valid")

plt.figure(figsize=(7,4))
plt.plot(smooth(iql), label="Independent Q-Learning")
plt.plot(smooth(vdn), label="VDN (Cooperative)")
plt.plot(smooth(qmix), label="QMIX (Cooperative)")
plt.xlabel("Episode")
plt.ylabel("Team Reward")
plt.legend()
plt.title("IQL vs VDN vs QMIX Performance")
plt.grid(True)
plt.show()
