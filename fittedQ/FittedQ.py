import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


def next_state(x, u):  # Current state x, performed action u
    x += 10 * u
    return np.clip(x, 0, 100)  # Next state x


def reward(p, x, u):  # Current price p, current state x, performed action u
    return p * (x - next_state(x, u))  # Immediate reward


days = 60
hours = 24
T = np.linspace(0, 23, hours).astype(int)  # Time t (hours)
X = np.linspace(0, 100, 11)  # State space X (SOC)
U = [-1, 0, 1]  # Action space U (-1: discharge, 0: do nothing, 1: charge)
P = 3 - np.sin(0.5 * T)  # Prices
gamma = 1  # Discount factor
epsilon = 0.85

trees = []
model = []
rew = []

# Day 1
x_0 = 50
x_t = np.zeros(hours)
x1_t = np.zeros(hours)
u_t = np.zeros(hours)
r_t = np.zeros(hours)
y_t = np.zeros(hours)
x_t[0] = x_0
r_cum = 0

for t in T:
    u_t[t] = np.random.choice(U)
    x1_t[t] = next_state(x_t[t], u_t[t])
    r_t[t] = reward(P[t], x_t[t], u_t[t])
    r_cum += r_t[t]
    y_t[t] = r_t[t]
    if t < 23:
        x_t[t+1] = next_state(x_t[t], u_t[t])
        x1_t[t] = x_t[t+1]
    else:
        x_0 = next_state(x_t[t], u_t[t])

x_T = x_t.copy()
x1_T = x1_t.copy()
u_T = u_t.copy()
r_T = r_t.copy()
y_T = y_t.copy()
rew.append(r_cum)

for t in T:
    inputs = pd.DataFrame({'X': [x_T[t]], 'U': [u_T[t]]})
    outputs = y_T[t]
    model = ExtraTreesRegressor(n_estimators=50)
    model.fit(inputs, [outputs])
    trees.append(model)

# Next days
for d in range(days-1):
    x_t[0] = 50
    r_cum = 0
    for t in T:
        model = trees[t]
        if np.random.random() > epsilon:
            a = []
            for u in U:
                a.append(model.predict(pd.DataFrame({'X': [x_t[t]], 'U': [u]})))
            u_t[t] = U[np.argmax(a).item()]
        else:
            u_t[t] = np.random.choice(U)

        if t < hours-1:
            x_t[t + 1] = next_state(x_t[t], u_t[t])
            x1_t[t] = x_t[t + 1]
        else:
            x_0 = next_state(x_t[t], u_t[t])

        r_t[t] = reward(P[t], x_t[t], u_t[t])
        r_cum += r_t[t]

    x_T = np.vstack((x_T, x_t))
    u_T = np.vstack((u_T, u_t))
    x1_T = np.vstack((x1_T, x1_t))
    r_T = np.vstack((r_T, r_t))
    rew.append(r_cum)

    y_T = np.zeros((x_T.shape[0], hours))
    for i in range(x_T.shape[0]):
        for j in range(hours):
            model_1 = trees[(j + 1) % 24]
            b = []
            for u in U:
                b.append(model_1.predict(pd.DataFrame({'X': [x1_T[i][j]], 'U': [u]})))
            y_T[i, j] = r_T[i][j] + gamma * np.max(b)

    for t in T:
        model = ExtraTreesRegressor(n_estimators=50)
        inputs = pd.DataFrame({'X': x_T[:, t], 'U': u_T[:, t]})
        outputs = y_T[:, t]
        model.fit(inputs, outputs)
        trees[t] = model

    epsilon = epsilon*.997

    if d % 15 == 0:
        epsilon = epsilon/3

    print('---')
    print('Day ' + str(d+1))
    print(trees[0].predict(pd.DataFrame({'X': [50], 'U': [-1]})))
    print(trees[0].predict(pd.DataFrame({'X': [50], 'U': [0]})))
    print(trees[0].predict(pd.DataFrame({'X': [50], 'U': [1]})))

# Plot 1
fig, (ax1, ax2) = plt.subplots(2)                           # Plot
fig.suptitle('Electricity price and battery SOC')
ax1.plot(T, P)
ax1.set_ylabel('Price')
ax2.plot(T, x_T[-1, :], 'tab:red')
ax2.set_ylabel('SOC')

# Plot 2
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111)
ax_2.plot(np.linspace(0, days-1, days), rew)
ax_2.set_xlabel('Samples')
ax_2.set_ylabel('Reward function')
plt.show()