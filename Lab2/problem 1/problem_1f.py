import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load model
try:
    model = torch.load('neural-network-1.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)

# set parameters
y_min = 0.0
y_max = 1.5
w_min = -np.pi
w_max = np.pi
steps = 1000

ys = np.linspace(y_min, y_max, steps, endpoint=True)
ws = np.linspace(w_min, w_max, steps, endpoint=True)

# loop through each and collect results
max_qs = []
argmax_qs = []
for y in tqdm(ys, desc="Processing y-values"):
    for w in tqdm(ws, desc="Processing w-values", leave=False):
        state = [0, y, 0, 0, w, 0, 0, 0]
        q_values = model(torch.tensor(state, dtype=torch.float32))
        max_q, argmax_q = torch.max(q_values, dim=0)
        max_qs.append(max_q.item())
        argmax_qs.append(argmax_q.item())
# place in arrays for easy manipulation
max_qs = np.array(max_qs).reshape(len(ys), len(ws))
argmax_qs = np.array(argmax_qs).reshape(len(ys), len(ws))

# 3D Plotting
Y, W = np.meshgrid(ys, ws, indexing='ij')

# Max Q-Values
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(Y, W, max_qs, cmap='viridis', edgecolor='none')
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$\omega$')
ax.set_zlabel(r'$\mathrm{max}_a Q \left(s\left(y,\omega\right), a\right)$')
ax.set_title(r'Maximum Q-Value for states varying with height ($y$) and angle ($\omega$)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

fig.savefig('outputs/problem_1f/3d_max.png', bbox_inches='tight')

# Heatmap for max_qs
fig, ax = plt.subplots(figsize=(8, 6))

c = ax.imshow(max_qs, cmap='viridis', origin='lower', extent=[w_min, w_max, y_min, y_max], aspect='auto')

fig.colorbar(c, ax=ax, label=r'$\mathrm{max}_a Q \left(s\left(y,\omega\right), a\right)$')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$y$')
ax.set_title(r'Maximum Q-Value for states varying with height ($y$) and angle ($\omega$)')

fig.savefig('outputs/problem_1f/2d_max.png', bbox_inches='tight')

# Argmax Q-Values
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(Y, W, argmax_qs, cmap='viridis', edgecolor='none')
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$\omega$')
ax.set_zlabel(r'$\mathrm{argmax}_a Q \left(s\left(y,\omega\right), a\right)$')
ax.set_title(r'Action with maximum Q-Value for states varying with height ($y$) and angle ($\omega$)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

fig.savefig('outputs/problem_1f/3d_argmax.png', bbox_inches='tight')

# Heatmap for argmax_qs
fig, ax = plt.subplots(figsize=(8, 6))

cmap = ListedColormap(['red', 'blue', 'green', 'orange'])
c = ax.imshow(argmax_qs, cmap=cmap, origin='lower', extent=[w_min, w_max, y_min, y_max], aspect='auto')

fig.colorbar(c, ax=ax, label=r'$\mathrm{argmax}_a Q \left(s\left(y,\omega\right), a\right)$')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$y$')
ax.set_title(r'Action with maximum Q-Value for states varying with height ($y$) and angle ($\omega$)')

fig.savefig('outputs/problem_1f/2d_argmax.png', bbox_inches='tight')

plt.show()