"""Main Script"""
import ev_fleet_model as fleet
import matplotlib.pyplot as plt
import mip as mip
import numpy as np


def export_output():
    return S, R_min, R_mean, R_max, min_e, max_e, t_time, price, mip_opt, value


n_ev = 10
charger = 3.3  # kW
min_e, max_e, t_time, price = fleet.initialise_fleet(n_ev)
X = np.arange(0, int(round(max(max_e)[0])) + 1, 0.5)
U = np.linspace(0, 1, 11)
alpha = 1
gamma = 1
state_track = []
R = []
rew = []
S = np.linspace(0, 8000, 21, True)

for k in range(8):
    Q = [np.zeros((len(X), len(U))) for i in range(len(t_time) + 1)]
    epsilon = 0.5
    for epoch in range(8001):
        state_track = []
        x_k = 0
        r_cum = 0
        for i, t in enumerate(t_time):
            x_index = np.where(round(x_k) == X)[0][0]
            if np.random.random() > epsilon:
                u_k = U[np.argmin(Q[i][x_index, :]).item()]
            else:
                u_k = np.random.choice(U)  # kW of charging power drawn
            u_index = np.where(u_k == U)[0][0]
            x_k1, r_k = fleet.environment(t, x_k, u_k * n_ev * charger, [min_e[i], max_e[i]])
            r_cum += r_k
            a = Q[i][x_index][u_index]
            b = np.min(Q[i + 1][np.where(round(x_k1.item()) == X)[0][0], :])
            Q[i][x_index][u_index] = a + alpha * (r_k + gamma * b - a)
            x_k = x_k1.item()
            state_track.append(x_k1)
        # print("Cost =", r_cum)
        rew.append(r_cum)
        if epoch % 700 == 0:
            epsilon = epsilon * 0.6
    R.append(np.asarray(rew).take(np.int32(S)))
    rew = []

R = np.vstack(R)
R_max = np.amax(R, axis=0)
R_min = np.amin(R, axis=0)
R_mean = np.mean(R, axis=0)

mip_opt, value = mip.solve_mip(n_ev, charger, price, min_e, max_e)

fig_1 = plt.figure()

ax_3 = fig_1.add_subplot(311)
ax_3.plot(t_time, price, label='Spot Price')
ax_3.set_xlabel('Hour')
ax_3.set_ylabel('Price')
ax_3.legend()

ax_1 = fig_1.add_subplot(312)
ax_1.plot(t_time, min_e, label='Minimum Energy', linestyle='--')
ax_1.plot(t_time, max_e, label='Maximum Energy', linestyle='--')
ax_1.plot(t_time, state_track, label='Energy')
ax_1.plot(t_time, np.cumsum(mip_opt), label='MIP Optimal', linestyle='-.')
ax_1.set_xlabel('Hour')
ax_1.set_ylabel('State of Energy')
ax_1.legend()

ax_2 = fig_1.add_subplot(313)
ax_2.plot(S, R_mean, label='Q-learning')
ax_2.axhline(y=value, label='Optimal', color='tab:red', linestyle='--')
ax_2.set_xlabel('Samples')
ax_2.set_ylabel('Cost')
ax_2.fill_between(S, R_min, R_max, color='k', alpha=0.1)
ax_2.legend()

fig_1.show()

print("Cost value - Q Learning = " + str(R_mean[-1]))
print("Cost value - MIP = " + str(value))
