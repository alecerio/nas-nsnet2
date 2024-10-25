import matplotlib.pyplot as plt
import numpy as np

generations = 50

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

generations = [0, 9, 24, 49]

for i in generations:
    pareto_front = np.loadtxt(f'pareto_gen_{i}.csv', delimiter=',', skiprows=1)  # Skip header
    ax.scatter(1+pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], label=f'Gen {i}', alpha=0.3)
ax.set_title("Pareto Front Evolution")
ax.set_xlabel("PESQ normalized")
ax.set_ylabel("Inference Time normalized")
ax.set_zlabel("Memory Footprint normalized")
ax.legend()
plt.show()

plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'y']
for i, color in zip(generations, colors):
    pareto_front = np.loadtxt(f'pareto_gen_{i}.csv', delimiter=',', skiprows=1) 
    pesq_metric = 0.5 + (1+pareto_front[:, 0]) * 4
    inference_time_metric = 0.005 + pareto_front[:, 1] * 0.003
    memory_footprint_metric = 8*93 + pareto_front[:, 2] * (32*93 - 8*93)
    plt.scatter(pesq_metric, inference_time_metric, c=color, label=f'Gen {i}', alpha=0.6)
plt.title("2D Pareto Front Evolution")
plt.xlabel("PESQ")
plt.ylabel("Inference Time")
plt.legend()
plt.show()