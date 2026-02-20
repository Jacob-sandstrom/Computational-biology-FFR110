# %%
import numpy as np
import matplotlib.pyplot as plt
import time

def model(u_val, rho, q):
    h = 1
    del_u_del_tau = np.zeros(u_val.shape)
    for i in range(len(u_val)):
        if i == 0 or i == len(u_val)-1:
            del_u_del_tau[i] = 0
        else:
            del_u_del_tau[i] = rho * u_val[i] * (1 - u_val[i]/q) - (u_val[i]/(1+u_val[i])) + (u_val[i+h] + u_val[i-h] - 2*u_val[i])/h**2
    return del_u_del_tau


L = 100
rho = 0.5
q = 8
d_tau = 0.01

steady_state0 = 0
steady_state1 = (q-1)/2+np.sqrt((q-1)**2/4-(1/rho-1)*q)
steady_state2 = (q-1)/2-np.sqrt((q-1)**2/4-(1/rho-1)*q)

print(steady_state1)
print(steady_state2)

#%%
# i
xi0 = 20
u0 = steady_state1

fix, ax = plt.subplots(1, 2, figsize=(10, 6))

tau_max = 200
u_val = np.array([u0/(1+np.exp(xi-xi0)) for xi in range(L)])
for step in range(int(tau_max/d_tau)+1):
    del_u_del_tau = model(u_val, rho, q)
    u_val += del_u_del_tau * d_tau
    u_val[0] = u_val[1]
    # u_val[np.where(u_val < 0)] = 0

    if step == int(tau_max/4/d_tau):
        measure_0 = (np.abs(u_val - u0/2)).argmin()
        print(f"middle of wave front is xi={measure_0} at tau={step*d_tau:.2f}")
    if step == int(3*tau_max/4/d_tau):
        measure_1 = (np.abs(u_val - u0/2)).argmin()
        print(f"middle of wave front is xi={measure_1} at tau={step*d_tau:.2f}")
        

c = (measure_1 - measure_0) / (tau_max/2)
print(f'c = {c:.2f}')

ax[0].plot(u_val)
ax[0].set_title(f'Wave at tau = {step*d_tau:.2f}')
ax[0].set_xlabel(r'$\xi$')
ax[0].set_ylabel(r'$u(\xi, \tau)$')

v = np.gradient(u_val)
ax[1].plot(u_val, v)
ax[1].set_xlabel(r'$u(\xi, \tau)$')
ax[1].set_ylabel(r'$\frac{\partial u}{\partial \xi}$')

ax[1].scatter(steady_state0, 0, color='red', label='Steady State 0')
ax[1].scatter(steady_state1, 0, color='blue', label='Steady State 1')
ax[1].scatter(steady_state2, 0, color='green', label='Steady State 2')
ax[1].legend()

plt.tight_layout()
plt.show()


# %%
# ii

xi0 = 50
u0 = steady_state2

fix, ax = plt.subplots(1, 2, figsize=(10, 6))

tau_max = 40
u_val = np.array([u0/(1+np.exp(xi-xi0)) for xi in range(L)])
for step in range(int(tau_max/d_tau)+1):
    del_u_del_tau = model(u_val, rho, q)
    u_val += del_u_del_tau * d_tau
    u_val[0] = u_val[1]
    # u_val[np.where(u_val < 0)] = 0
    # plt.plot(u_val)
    # plt.title(f'Wave at tau = {step*d_tau:.2f}')
    # if step % 100 == 0:
    #     plt.pause(0.01)

    if step == int(tau_max/4/d_tau):
        measure_0 = (np.abs(u_val - u0/2)).argmin()
        print(f"middle of wave front is xi={measure_0} at tau={step*d_tau:.2f}")
    if step == int(3*tau_max/4/d_tau):
        measure_1 = (np.abs(u_val - u0/2)).argmin()
        print(f"middle of wave front is xi={measure_1} at tau={step*d_tau:.2f}")
        

c = (measure_1 - measure_0) / (tau_max/2)
print(f'c = {c:.2f}')

ax[0].plot(u_val)
ax[0].set_title(f'Wave at tau = {step*d_tau:.2f}')
ax[0].set_xlabel(r'$\xi$')
ax[0].set_ylabel(r'$u(\xi, \tau)$')

v = np.gradient(u_val)
ax[1].plot(u_val, v)
ax[1].set_xlabel(r'$u(\xi, \tau)$')
ax[1].set_ylabel(r'$\frac{\partial u}{\partial \xi}$')

ax[1].scatter(steady_state0, 0, color='red', label='Steady State 0')
ax[1].scatter(steady_state1, 0, color='blue', label='Steady State 1')
ax[1].scatter(steady_state2, 0, color='green', label='Steady State 2')
ax[1].legend()

plt.tight_layout()
plt.show()


# %%
# iii


xi0 = 50
u0 = 1.1*steady_state2

fix, ax = plt.subplots(1, 2, figsize=(10, 6))


tau_max = 200
u_val = np.array([u0/(1+np.exp(xi-xi0)) for xi in range(L)])
for step in range(int(tau_max/d_tau)+1):
    del_u_del_tau = model(u_val, rho, q)
    u_val += del_u_del_tau * d_tau
    u_val[0] = u_val[1]
    # u_val[np.where(u_val < 0)] = 0



    if step == int(tau_max/4/d_tau):
        measure_0 = (np.abs(u_val - u0/2)).argmin()
        print(f"middle of wave front is xi={measure_0} at tau={step*d_tau:.2f}")
    if step == int(3*tau_max/4/d_tau):
        measure_1 = (np.abs(u_val - u0/2)).argmin()
        print(f"middle of wave front is xi={measure_1} at tau={step*d_tau:.2f}")
        

c = (measure_1 - measure_0) / (tau_max/2)
print(f'c = {c:.2f}')

# u_val = u_val[1:]

ax[0].plot(u_val)
ax[0].set_title(f'Wave at tau = {step*d_tau:.2f}')
ax[0].set_xlabel(r'$\xi$')
ax[0].set_ylabel(r'$u(\xi, \tau)$')

v = np.gradient(u_val)
print(v)
ax[1].plot(u_val, v)
ax[1].set_xlabel(r'$u(\xi, \tau)$')
ax[1].set_ylabel(r'$\frac{\partial u}{\partial \xi}$')

ax[1].scatter(steady_state0, 0, color='red', label='Steady State 0')
ax[1].scatter(steady_state1, 0, color='blue', label='Steady State 1')
ax[1].scatter(steady_state2, 0, color='green', label='Steady State 2')
ax[1].legend()

# plt.plot(u_val)
plt.tight_layout()
plt.show()


# %%
# c

