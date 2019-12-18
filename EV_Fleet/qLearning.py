"""Main Script"""
import matplotlib.pyplot as plt
import numpy as np

import EV_Fleet.ev_fleet_model as fleet

n_ev = 1
min_e, max_e, t_time = fleet.initialise_fleet(n_ev)
U = np.linspace(0, 1, 11)
X = np.arange(0, int(round(max(max_e)[0]))+1, 0.5)
Q = [np.zeros((len(X), len(U))) for i in range(len(t_time)+1)]
epsilon = 0.5
alpha = 1
gamma = 1

for epoch in range(10000):
    state_track = []
    x_k = 0
    r_cum = 0
    for i, t in enumerate(t_time):
        x_index = np.where(round(x_k*2)/2 == X)[0][0]
        if np.random.random() > epsilon:
            u_k = U[np.argmin(Q[i][x_index, :]).item()]
        else:
            u_k = np.random.choice(U)         # kW of charging power drawn
        u_index = np.where(u_k == U)[0][0]
        x_k1, r_k = fleet.environment(t, x_k, u_k*3.3, [min_e[i], max_e[i]])
        r_cum += r_k
        a = Q[i][x_index][u_index]
        b = np.min(Q[i + 1][np.where(round(x_k1.item()*2)/2 == X)[0][0], :])
        Q[i][x_index][u_index] = a + alpha * (r_k + gamma * b - a)
        x_k = x_k1.item()
        state_track.append(x_k1)
    print("Cost =", r_cum)
    if epoch%500 == 0:
        epsilon = epsilon*0.5

plt.plot(t_time, min_e, label='Minimum Energy', linestyle='--')
plt.plot(t_time, max_e, label='Maximum Energy', linestyle='--')
plt.plot(t_time, state_track, label='Energy')
plt.legend()
plt.show()
