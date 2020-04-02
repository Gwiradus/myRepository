"""Main Script"""
import ev_fleet_model as fleet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qLearning_fittedQ as qLearn
from sklearn.neural_network import MLPRegressor


def export_output():
    return np.linspace(1, days, days), rew


"Initial conditions"
n_ev = 10
charger = 3.3  # kW
# min_e, max_e, t_time = fleet.initialise_fleet(n_ev)
S, R_min, R_mean, R_max, min_e, max_e, t_time, price, mip_opt, value = qLearn.export_output()
hours = len(t_time)
days = 150
X = np.arange(0, int(round(max(max_e)[0])) + 1, 1)
U = np.linspace(0, 1, 11)
epsilon = 1
gamma = 1
mul = 0.1
r_cum = 0

x_T = np.empty((0, hours))
x_t = np.zeros(hours)
x_T1 = np.empty((0, hours))
x_t1 = np.zeros(hours)
u_T = np.empty((0, hours))
u_t = np.zeros(hours)
r_T = np.empty((0, hours))
r_t = np.zeros(hours)
q_T = np.empty((0, hours))
y_t = np.zeros(hours)
rew = []

neural_nets = []
model = []
for i in range(hours):
    inputs = pd.DataFrame({'X': [x_t[i]], 'U': [u_t[i]], 't': [i]})
    outputs = y_t[i]
    model = MLPRegressor(activation='tanh', hidden_layer_sizes=(6,), max_iter=10000, solver='lbfgs', warm_start='true')
    model.fit(inputs, [outputs])
    neural_nets.append(model)

for d in range(days):
    x_k = 0
    r_cum = 0
    for i, t in enumerate(t_time):
        x_t[i] = x_k
        model = neural_nets[i]
        if np.random.random() > epsilon:
            a = []
            for u in U:
                a.append(model.predict(pd.DataFrame({'X': [x_t[i]], 'U': [u], 't': [i]})))
            u_t[i] = U[np.random.choice(np.where(a == np.asarray(a).min())[0])]
        else:
            u_t[i] = np.random.choice(U)  # % of charging power drawn

        x_t1[i], r_t[i] = fleet.environment(t, x_t[i], u_t[i] * n_ev * charger, [min_e[i], max_e[i]])
        r_cum += r_t[i]
        x_k = x_t1[i]

    x_T = np.vstack((x_T, x_t))
    u_T = np.vstack((u_T, u_t))
    x_T1 = np.vstack((x_T1, x_t1))
    r_T = np.vstack((r_T, r_t))
    rew.append(r_cum)
    print("Day = ", d)
    print("Cost =", r_cum)
    print("Epsilon = ", epsilon)

    q_T = np.zeros((x_T.shape[0], hours))
    for i in reversed(range(hours)):
        if i < hours - 1:
            model_t1 = neural_nets[(i + 1)]
            for j in range(x_T.shape[0]):
                b = []
                for u in U:
                    b.append(model_t1.predict(pd.DataFrame({'X': [x_T1[j][i]], 'U': [u], 't': [i + 1]})))
                q_T[j, i] = r_T[j][i] + gamma * np.min(b)
        else:
            for j in range(x_T.shape[0]):
                q_T[j, i] = r_T[j][i]

        t_vec = np.empty(x_T.shape[0])
        t_vec.fill(i)
        model = neural_nets[i]
        inputs = pd.DataFrame({'X': x_T[:, i], 'U': u_T[:, i], 't': t_vec})
        outputs = q_T[:, i]
        model.fit(inputs, outputs)
        neural_nets[i] = model

    if d % 50 == 49:
        epsilon = epsilon * mul

fig_1 = plt.figure()

ax_3 = fig_1.add_subplot(311)
ax_3.plot(t_time, price, label='Spot Price')
ax_3.set_xlabel('Hour')
ax_3.set_ylabel('Price')
ax_3.legend()

ax_1 = fig_1.add_subplot(312)
ax_1.plot(t_time, min_e, label='Minimum Energy', linestyle='--')
ax_1.plot(t_time, max_e, label='Maximum Energy', linestyle='--')
ax_1.plot(t_time, x_T1[-1, :], label='Energy')
ax_1.plot(t_time, np.cumsum(mip_opt), label='MIP Optimal', linestyle='-.')
ax_1.set_xlabel('Hour')
ax_1.set_ylabel('State of Energy')
ax_1.legend()

ax_2 = fig_1.add_subplot(313)
ax_2.plot(S, R_mean, label='Q-learning')
ax_2.plot(np.linspace(1, days, days), rew, label='Fitted Q-iteration', color='tab:green')
ax_2.axhline(y=value, label='Optimal', color='tab:red', linestyle='--')
ax_2.set_xlabel('Samples')
ax_2.set_ylabel('Cost')
ax_2.fill_between(S, R_min, R_max, color='k', alpha=0.1)
ax_2.legend()

fig_1.show()

print("Cost value - Q Learning = " + str(R_mean[-1]))
print("Cost value - Fitted Q-iteration = " + str(r_cum))
print("Cost value - MIP = " + str(value))
