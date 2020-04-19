"""Reinforcement Learning problem - Event-triggered EV Charging Fleet"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import batchGenerator
import mipSolver
from rlAgent import RLAgent

np.random.seed(0)

"Initial conditions"
n_ev = 10
actions = np.linspace(0, 1, 11)
myAgent = RLAgent({"actions": actions, "epsilon": 1, "discount": 1.0})
time = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
batch_t = []
batch_t1 = []
batch_xt = []
batch_at = []
batch_rt1 = []
batch_xt1 = []
Q = []
G = []

# Create batch
episodes = 1
myFleets = batchGenerator.generate_batch(n_ev, episodes)
myFleet = None
events = None
iterations = 100

for i in range(iterations):
    g = []
    print("Day " + str(i + 1) + "/" + str(iterations) + ", \n"
                                                        "Generating the batch...")
    for ep, aFleet in enumerate(myFleets):
        myFleet = deepcopy(aFleet)
        events = time
        for next_event in events:
            batch_t.append(myFleet.current_time)
            batch_xt.append(myFleet.current_state)
            current_action = myAgent.agent_step(myFleet.current_time, myFleet.current_state)
            batch_at.append(current_action)
            myFleet.env_step(current_action, next_event)
            batch_rt1.append(myFleet.reward_obs_term[0])
            batch_xt1.append(myFleet.reward_obs_term[1])
            batch_t1.append(myFleet.current_time)
        g.append(sum(batch_rt1[-len(events):]))

    G.append(g)
    print("Cost =", g[-1])
    print("Epsilon = ", myAgent.epsilon)
    print("Walk =", batch_xt[-len(events):])

    # Building the training set.
    q = np.zeros(len(batch_t))
    print("Building the training set...")
    for b_t, next_time in enumerate(batch_t1):
        if (b_t + 1) % len(batch_t) != 0:
            greedy = []
            for action in actions:
                greedy.append(myAgent.q.predict(
                    pd.DataFrame({'X': [batch_xt1[b_t]], 'U': [action], 't': [next_time]})))
            q[b_t] = batch_rt1[b_t] + myAgent.discount * np.min(greedy)
        else:
            q[b_t] = batch_rt1[b_t]
    Q.append(q)

    # Use the regression algorithm to induce from the training set the function Q(x,u)
    print("Training the approximator...")
    inputs = pd.DataFrame({'X': batch_xt, 'U': batch_at, 't': batch_t})
    outputs = q
    myAgent.q.fit(inputs, outputs)
    myAgent.epsilon = myAgent.epsilon * .96
    print("Done!\n---*---*---")

time.insert(0, 7)
events = time
mip_opt, value = mipSolver.mip_solver(myFleet.connected_evs, events[:-1], myFleet.delta, myFleet.charging_power,
                                      myFleet.spot_price, myFleet.min_energy, myFleet.max_energy)
mip_opt = np.insert(mip_opt, 0, 0)

path = batch_xt[-(len(events) - 1):]
path.append(batch_xt[-1])
myFleet.min_energy.insert(0, 0)
myFleet.max_energy.insert(0, 0)
spot_price = []
for t in time:
    spot_price.append(25 + 8 * np.sin(12 * np.pi * int(t) / 22))

fig_1 = plt.figure()
ax_3 = fig_1.add_subplot(311)
ax_3.plot(time, spot_price, label='Spot Price')
ax_3.set_xlabel('Hour')
ax_3.set_ylabel('Price (¤/kWh)')
ax_3.legend()

ax_1 = fig_1.add_subplot(312)
ax_1.plot(events, myFleet.min_energy, label='Minimum Energy', linestyle='--')
ax_1.plot(events, myFleet.max_energy, label='Maximum Energy', linestyle='--')
ax_1.plot(events, path, label='Fitted-Q iteration')
ax_1.plot(events, mip_opt, label='MIP Optimal', linestyle='-.')
ax_1.set_xlabel('Hour')
ax_1.set_ylabel('State of Energy (kWh)')
ax_1.legend()

ax_2 = fig_1.add_subplot(313)
# ax_2.plot(S, R_mean, label='Q-learning')
ax_2.plot(np.linspace(1, iterations, iterations), G, label='Fitted-Q iteration', color='tab:green')
ax_2.axhline(y=value, label='Optimal', color='tab:red', linestyle='--')
ax_2.set_xlabel('Days')
ax_2.set_ylabel('Cost (¤)')
ax_2.set_ylim(1800, 3200)
# ax_2.fill_between(S, R_min, R_max, color='k', alpha=0.1)
ax_2.legend()
fig_1.show()

# Saving the objects:
# with open('fqiResults.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([myFleets, myAgent, G, Q, batch_t, batch_xt, batch_at, batch_rt1, batch_t1, batch_xt1, fig_1], f)
