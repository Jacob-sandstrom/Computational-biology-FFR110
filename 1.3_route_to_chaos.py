# %%

import numpy as np
import matplotlib.pyplot as plt



def model(N, R, a):
    return R*N*np.exp(-a*N)

R_values = np.arange(1, 30, 0.1)
N0 = 900
a = 0.01

t_max = 300


# %%
# a

for R in R_values:
    N_vals = np.zeros(t_max)
    N_vals[0] = N0
    for t in range(t_max-1):
        N_vals[t+1] = model(N_vals[t], R, a)
    plt.plot(np.full(100, R), N_vals[-100:], linestyle="none", marker='.', markersize=0.2, color='black')

plt.xlabel('R')
plt.ylabel(r'Population Size $\eta$')
plt.show()

# %%
# b

R_values = [5, 10, 23, 14]
t_max = 41

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, R in enumerate(R_values):
    N_vals = np.zeros(t_max)
    N_vals[0] = N0
    for t in range(t_max-1):
        N_vals[t+1] = model(N_vals[t], R, a)
    axes[i].plot(range(t_max), N_vals,label=f'R={R}', marker='.')

    axes[i].set_xlabel(r'$\tau$')
    axes[i].set_ylabel(r'Population Size $\eta$')
    axes[i].legend()
plt.show()
# %%




# %%
# d
R_values = np.arange(14.2, 14.8, 0.0001)
N0 = 900
a = 0.01

t_max = 1000

for R in R_values:
    N_vals = np.zeros(t_max)
    N_vals[0] = N0
    for t in range(t_max-1):
        N_vals[t+1] = model(N_vals[t], R, a)
    plt.plot(np.full(100, R), N_vals[-100:], linestyle="none", marker='.', markersize=0.2, color='black')

plt.xlabel('R')
plt.ylabel(r'Population Size $\eta$')
plt.ylim(80,150)
plt.show()

 # %%
