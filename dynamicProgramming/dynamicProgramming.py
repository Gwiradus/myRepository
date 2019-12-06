import matplotlib.pyplot as plt
import numpy as np


def next_state(x, u):                               # Current state x, performed action u
    x += 10 * u
    return np.clip(x, 0, 100)                       # Next state x


def reward(p, x, u):                                # Current price p, current state x, performed action u
    return p * (x - next_state(x, u))               # Immediate reward


T = np.linspace(1, 24, 24).astype(int)              # Time t (hours)
X = np.linspace(0, 100, 11)                         # States x (SOC)
U = [-1, 0, 1]                                      # Actions u (-1: discharge, 0: do nothing, 1: charge)
P = 3 + np.sin(0.5*T)                               # Prices
gamma = 1                                           # Discount factor
Q = []                                              # Q table

for hour in reversed(T):                            # Build Q table
    q = np.zeros((len(X), len(U)))
    for num_s, state in enumerate(X):
        for num_a, action in enumerate(U):
            q[num_s, num_a] = reward(P[hour - 1], state, action)
            if hour != 24:
                q[num_s, num_a] += gamma * np.max(Q[len(Q) - 1][np.where(next_state(state, action) == X)[0][0], :])
    Q.append(q)

Q.reverse()                                         # Sort Q in the right order
SOC = [np.random.choice(X)]                         # Random initial state x_k-1
profit = 0                                          # Initial profit
for num_q, q_tables in enumerate(Q):
    soc = SOC[len(SOC) - 1]
    i = np.where(soc == X)[0][0]
    u_0 = np.argmax(q_tables[i, :])
    SOC.append(next_state(soc, U[u_0.item()]))
    profit += reward(P[num_q], soc, U[u_0.item()])  # Profit
SOC.pop(0)                                          # Delete random initial state x_k-1
SOC = np.asarray(SOC)

fig, (ax1, ax2) = plt.subplots(2)                   # Plot
fig.suptitle('Electricity price and battery SOC')
ax1.plot(T, P)
ax2.plot(T, SOC, 'tab:red')

print("Total profit is: " + str(profit))
