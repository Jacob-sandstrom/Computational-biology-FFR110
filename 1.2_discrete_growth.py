# %%
import numpy as np
import matplotlib.pyplot as plt

#%%
# d
def linear_stability_growth_unstable(r,v0,t_max):
    return [(1+r)**t*v0 for t in range(t_max)]

def growth_model(N, r, K, b, t_max):
    return (r+1)*N/(1+(N/K)**b)


N0_vals = [1,2,3,10]
t_max = 100
K = 1000
r = 0.1
b = 1

for N0 in N0_vals:
    v_vals = linear_stability_growth_unstable(r, N0, t_max)
    plt.plot(range(t_max),v_vals,label=f'N_0={N0} linear approx')

    N_vals = np.zeros(t_max)
    N_vals[0] = N0
    for t in range(t_max-1):
        N_vals[t+1] = growth_model(N_vals[t], r, K, b, t_max)
    plt.plot(range(t_max),N_vals,label=f'N_0={N0}, iterated map')


plt.xlabel('Time (t)')
plt.ylabel('Population Size N')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


#%%
# f

def linear_stability_growth_stable(r,b,v0,t_max):
    return [(1-r*b/(1+r))**t*v0 for t in range(t_max)]


del_N0_vals = [-10,-3,-2,-1,1,2,3,10]
# del_N0_vals = [-10]
t_max = 50
K = 1000
r = 0.1
b = 1


fig, axes = plt.subplots(4, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, del_N0 in enumerate(del_N0_vals):
    ax = axes[idx]
    N0 = K*r**(1/b) + del_N0
    v_vals = linear_stability_growth_stable(r, b, del_N0, t_max)
    ax.plot(range(t_max), [v+K*r**(1/b) for v in v_vals], label=f'N_0={del_N0} linear approx')

    N_vals = np.zeros(t_max)
    N_vals[0] = N0
    for t in range(t_max-1):
        N_vals[t+1] = growth_model(N_vals[t], r, K, b, t_max)
    ax.plot(range(t_max), N_vals, label=f'N_0={N0}, iterated map')
    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Population Size N')
    ax.legend()

plt.tight_layout()
plt.show()
# %%
