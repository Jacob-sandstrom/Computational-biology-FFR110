# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# def N(t):


def allee(t, dt, N_vals, N0, r, T, K, A):
    if t < T:
        return r*N_vals[int(t/dt)]*(1-N0/K)*(1-N_vals[int(t/dt)]/K)
    
    return r*N_vals[int(t/dt)]*(1-N_vals[int((t-T)/dt)]/K)*(N_vals[int(t/dt)]/A-1)

T_range = np.linspace(0.1, 5, 50)
A = 20
K = 100
r = 0.1
N0 = 50

t_max = 500
dt = 0.01



# %%
# a
T_range = [0.1, 3, 4]

t_max = 500
dt = 0.01

N_vals = np.zeros(int(t_max/dt))
N_vals[0] = N0

for T in T_range:
    for i in range(len(N_vals)-1):
        t = i * dt
        N_vals[i+1] = N_vals[i] + allee(t, dt, N_vals, N0, r, T, K, A) * dt

    plt.plot(np.arange(0, t_max, dt), N_vals, label=f'T={T}')
    plt.xlabel('Time (t)')
    plt.ylabel('Population Size N(t)')
    plt.title(f'Population Dynamics for Allee Effect with Time Delay {T}')
    plt.show()
    # break


# %%
# b
t_max = 200
dt = 0.01

N_vals = np.zeros(int(t_max/dt))
N_vals[0] = N0
T_range = np.linspace(0.91, 1, 10)
last_dNdt = 0

for T in T_range:
    occilations = 0
    for i in range(len(N_vals)-1):
        t = i * dt
        dNdt = allee(t, dt, N_vals, N0, r, T, K, A) * dt
        N_vals[i+1] = N_vals[i] + dNdt

        if t > T and np.sign(dNdt) != np.sign(last_dNdt):
            occilations += 1
            # print(f'T={T}, t={t}, N={N_vals[i]:.2f}')

        last_dNdt = dNdt

    # plt.plot(np.arange(0, t_max, dt), N_vals, label=f'T={T}')
    # plt.xlabel('Time (t)')
    # plt.ylabel('Population Size N(t)')
    # plt.title(f'Population Dynamics for Allee Effect with Time Delay {T}')
    # plt.ylim(90, 120)
    # plt.show()
    print(f'T={T}, Occilations: {occilations}')


# %%
# c
t_max = 10000
dt = 0.01

N_vals = np.zeros(int(t_max/dt))
N_vals[0] = N0
T_range = np.linspace(3.86, 4, 15)

dNdt_values = []

for T in T_range:
    for i in range(len(N_vals)-1):
        t = i * dt
        dNdt = allee(t, dt, N_vals, N0, r, T, K, A) * dt
        N_vals[i+1] = N_vals[i] + dNdt

        dNdt_values.append(dNdt)

    print(f'T={T}, Max dN/dt in the last 200 steps: {np.max(dNdt_values[-200:]):.5f}')

# %%
