"""Main Script"""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor

import ev_fleet_model as fleet
import qLearning_fittedQ as qLearn

random.seed(0)

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
mul = 0.96
Q = []  # Q table

for hour in reversed(t_time):  # Build Q table
    q = np.zeros((len(X), len(U)))
    for num_s, x_t in enumerate(X):
        for num_a, u_t in enumerate(U):
            x_t1, r = fleet.environment(hour, x_t, u_t * n_ev * charger, [min_e[hour - 6], max_e[hour - 6]])
            q[num_s, num_a] = r
            if hour != 18:
                q[num_s, num_a] += gamma * np.min(Q[len(Q) - 1][np.where(np.round(x_t1) == X)[0][0], :])
    Q.append(q)

Q.reverse()  # Sort Q in the right order

x_T = np.empty((0, hours))
x_t = np.zeros(hours)
x_T1 = np.empty((0, hours))
x_t1 = np.zeros(hours)
u_T = np.empty((0, hours))
u_t = np.zeros(hours)
r_T = np.empty((0, hours))
r_t = np.zeros(hours)
q_T = np.empty((0, hours))
q_t = np.zeros(hours)
rew = []
r_cum = 0

neural_nets = []
model = []
for i in range(hours):
    inputs = pd.DataFrame({'X': [x_t[i]], 'U': [u_t[i]], 't': [i + 7]})
    outputs = q_t[i]
    model = MLPRegressor(activation='relu', hidden_layer_sizes=(12, 12), max_iter=100000, solver='lbfgs')
    model.fit(inputs, [outputs])
    neural_nets.append(model)

"Building the set of four tuples x_t, u_t, r_t, x_t+1"
for d in range(days):
    x_k = 0
    r_cum = 0
    for i, t in enumerate(t_time):
        x_t[i] = x_k
        model = neural_nets[i]
        if np.random.random() > epsilon:
            a = []
            for u in U:
                a.append(model.predict(pd.DataFrame({'X': [x_t[i]], 'U': [u], 't': [t]})))
            u_t[i] = U[np.random.choice(np.where(a == min(np.asarray(a)))[0])]
        else:
            u_t[i] = np.random.choice(U)  # % of charging power drawn

        x_t1[i], r_t[i] = fleet.environment(t, x_t[i], u_t[i] * n_ev * charger, [min_e[i + 1], max_e[i + 1]])
        r_cum += r_t[i]
        x_k = x_t1[i]

    x_T = np.vstack((x_T, x_t))
    u_T = np.vstack((u_T, u_t))
    r_T = np.vstack((r_T, r_t))
    x_T1 = np.vstack((x_T1, x_t1))
    rew.append(r_cum)
    print("Day = ", d)
    print("Cost =", r_cum)
    print("Epsilon = ", epsilon)

    "Building the training set"
    q_T = np.zeros((x_T.shape[0], hours))
    for i in reversed(range(hours)):
        if i < hours - 1:
            model_t1 = neural_nets[(i + 1)]
            for j in range(x_T.shape[0]):
                b = []
                for u in U:
                    b.append(model_t1.predict(pd.DataFrame({'X': [x_T1[j][i]], 'U': [u], 't': [i + 1 + 7]})))
                q_T[j, i] = r_T[j][i] + gamma * np.min(b)
        else:
            for j in range(x_T.shape[0]):
                q_T[j, i] = r_T[j][i]

        "Use the regression algorithm to induce from the training set the function Q(x,u)"
        t_vec = np.empty(x_T.shape[0])
        t_vec.fill(i + 7)
        inputs = pd.DataFrame({'X': x_T[:, i], 'U': u_T[:, i], 't': t_vec})
        outputs = q_T[:, i]
        neural_nets[i].fit(inputs, outputs)
    epsilon = epsilon * mul

fig_1 = plt.figure()

ax_3 = fig_1.add_subplot(311)
ax_3.plot(t_time, price, label='Spot Price')
ax_3.set_xlabel('Hour')
ax_3.set_ylabel('Price')
ax_3.legend()

ax_1 = fig_1.add_subplot(312)
ax_1.plot(t_time, min_e[:-1], label='Minimum Energy', linestyle='--')
ax_1.plot(t_time, max_e[:-1], label='Maximum Energy', linestyle='--')
ax_1.plot(t_time, x_t1, label='Energy')
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
print("Error = " + str(100 * (r_cum - value) / value) + "%")

for time in t_time:
    Q_tree_app = np.zeros((len(X), len(U)))
    for xi, x in enumerate(X):
        for ui, u in enumerate(U):
            Q_tree_app[xi][ui] = neural_nets[(time - 7)].predict(pd.DataFrame({'X': [x], 'U': [u], 't': [time]}))

    fig_2 = plt.figure(num='Hour ' + str(time))
    ax = Axes3D(fig_2)
    Xs, Us = np.meshgrid(X, U)
    ax.scatter(Xs, Us, [*zip(*Q[11])], c='r', marker='o', label='Q-function benchmark')
    # ax.scatter(Xs, Us, [*zip(*Q_tree)], c='b', marker='o', label='Q-function approximation')
    ax.scatter(Xs, Us, [*zip(*Q_tree_app)], c='g', marker='o', label='Q-function approximation')
    ax.set_xlabel('State space (kWh)')
    ax.set_ylabel('Action space (kWp)')
    ax.legend()
    ax.view_init(30, -60)
    fig_2.show()
