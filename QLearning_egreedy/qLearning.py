import matplotlib.pyplot as plt
import numpy as np


def next_state(x, u):                                       # Current state x, performed action u
    x += 10 * u
    return np.clip(x, 0, 100)                               # Next state x


def reward(p, x, u):                                        # Current price p, current state x, performed action u
    return p * (x - next_state(x, u))                       # Immediate reward


T = np.linspace(1, 24, 24).astype(int)                      # Time t (hours)
X = np.linspace(0, 100, 11)                                 # States x (SOC)
U = [-1, 0, 1]                                              # Actions u (-1: discharge, 0: do nothing, 1: charge)
P = 3 + np.sin(0.5*T)                                       # Prices
gamma = 1                                                   # Discount factor
SOC = []
R = []
rew = []

for k in range(10):
    epsilon = 0.25                                          # Exploration schedule
    alpha = 1                                               # Learning rate schedule
    Q = [np.zeros((len(X), len(U))) for i in range(len(T) + 1)]  # Q table
    for j in range(20):
        for i in range(500):
            SOC = [50]                                          # Random initial state x_0
            r_cum = 0
            for num_h, hour in enumerate(T):
                x_index = np.where(SOC[num_h] == X)[0][0]
                if np.random.random() > epsilon:
                    u_k = np.int32(U[np.argmax(Q[num_h][x_index, :]).item()])
                else:
                    u_k = np.random.choice(U)
                u_index = np.where(u_k == U)[0][0]
                x_next = next_state(SOC[num_h], u_k)
                SOC.append(x_next)
                r = reward(P[num_h], SOC[num_h], u_k)
                r_cum += r
                a = Q[num_h][x_index][u_index]
                b = np.max(Q[num_h+1][np.where(x_next == X)[0][0], :])
                Q[num_h][x_index][u_index] = a + alpha*(r + gamma*b - a)
            rew.append(r_cum)
        epsilon = 0.5 * epsilon
        alpha = 0.75 * alpha
    R.append(np.asarray(rew).take(np.int32(np.linspace(0, 9000, 91, endpoint=True))))
    rew = []

SOC.pop(len(T))                                             # Delete random initial state x_k-1
SOC = np.asarray(SOC)
fig, (ax1, ax2) = plt.subplots(2)                           # Plot
fig.suptitle('Electricity price and battery SOC')
ax1.plot(T, P)
ax2.plot(T, SOC, 'tab:red')
plt.plot(rew)
