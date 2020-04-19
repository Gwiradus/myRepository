import numpy as np
import pulp as p


def mip_solver(number_ev, events, delta_t, charging_power, price, min_e, max_e):

    # Input variables
    actions = []  # Action vector
    energy = []  # Energy vector

    # Create a LP minimization problem
    lp_prob = p.LpProblem('Problem', p.LpMinimize)

    # Create problem variables
    for event in events:
        actions.append(p.LpVariable(("u" + str(event)), lowBound=0))  # Create variables ut >= 0

    for ix, action in enumerate(actions):
        energy.append(number_ev[ix] * delta_t[ix] * charging_power * action)
    cum_energy = np.cumsum(energy)

    # Objective Function
    lp_prob += np.sum([x*y for x, y in zip(energy, price)])

    # Constraints:
    for ix, limit in enumerate(min_e):
        lp_prob += cum_energy[ix] >= limit

    for ix, limit in enumerate(max_e):
        lp_prob += cum_energy[ix] <= limit

    for action in actions:
        lp_prob += action <= 1

    # # Display the problem
    # print(lp_prob)

    lp_prob.solve()  # Solver

    mip_opt = []
    for action in actions:
        mip_opt.append(p.value(action))

    results = []    # Starting from t1
    for ix, action in enumerate(mip_opt):
        results.append(number_ev[ix] * delta_t[ix] * charging_power * action)

    path = np.cumsum(results)   # Starting from t1
    value = p.value(lp_prob.objective)
    return path, value
