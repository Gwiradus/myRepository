"""EV Fleet Model"""

import numpy as np
np.random.seed(0)

"""Helper Functions"""


def inverse_normal_cdf(mean, sd):
    """Inverse of normal cumulative distribution function"""
    return np.random.normal(mean, sd, 1)


def truncate_normal(mean, sd, min_value, max_value, size):
    """Function that gives a list of normally distributed random numbers with given mean, standard deviation and
    max-min """
    y_list = np.zeros((size, 1), dtype='float')
    for i in range(size):
        y = np.random.normal(mean, sd, 1)
        y_list[i] = np.clip(y, min_value, max_value)
    return y_list


def ev_single_boundary(time, time_vector, energy_req, power_max=3.3):
    """Function to find the max min energy boundaries of a single EV"""
    arrive_time = time_vector[0]
    depart_time = time_vector[1]
    if time < arrive_time:
        return [0, 0]
    elif arrive_time <= time <= depart_time:
        e_min = max(energy_req - power_max * (depart_time - time), 0)
        e_max = min(energy_req, power_max * (time - arrive_time))
        return [e_min, e_max]
    else:
        return [energy_req, energy_req]


def ev_fleet_boundary(time, arrive_vector, depart_vector, energy_req_vector, number_of_evs, power_max=3.3):
    """Function to find the max min energy boundaries of a fleet of EVs"""
    e_max = 0
    e_min = 0
    for ev in range(number_of_evs):
        energy_vector = ev_single_boundary(time, [arrive_vector[ev], depart_vector[ev]],
                                           energy_req_vector[ev], power_max)
        e_min += energy_vector[0]
        e_max += energy_vector[1]

    return [e_min, e_max]


def initialise_fleet(number_of_evs):
    """Function to initialise fleet parameters like arrival/departure times, distance covered etc."""
    t_min = 7
    t_max = 18
    arrive_time = truncate_normal(8, 0.5, t_min, 9, number_of_evs)
    depart_time = truncate_normal(17, 0.5, 16, t_max, number_of_evs)
    energy_req = np.array([(inverse_normal_cdf(70, 20) * 0.174) for _ in range(number_of_evs)])
    p_max = 3.3  # kW
    e_max = []
    e_min = []
    time = []
    price = []
    for t in range(t_min, t_max+1):
        energy = ev_fleet_boundary(t, arrive_time, depart_time, energy_req, number_of_evs, p_max)
        e_min.append(energy[0])
        e_max.append(energy[1])
        time.append(t)
        price.append(spot_price(t))
    # e_min[len(e_min) - 1] = e_min[len(e_min) - 3]
    # e_min[len(e_min) - 2] = e_min[len(e_min) - 3]
    e_min.append(e_min[len(e_min)-1])
    e_max.append(e_max[len(e_max)-1])
    return e_min, e_max, time, price


"""MDP Functions"""


def spot_price(time):
    """Function that returns the day ahead price of that hour"""
    return 25 + 8 * np.sin(12 * np.pi * time/22)


def transition_function(current_state, action, constraints):
    """Transition Function that gives the next state depending on current state and action"""
    delta_t = 1
    next_state = current_state + action * delta_t
    return np.clip(next_state, constraints[0], constraints[1])


def reward_function(price, current_state, next_state):
    """Reward function that gives the reward for each hour, based on current state and next state reached"""
    return price * (next_state - current_state)


def environment(time, current_state, action, constraint):
    """Environment function that gives the next state and reward based on current state and action"""
    price = spot_price(time)
    next_state = transition_function(current_state, action, constraint)
    reward = reward_function(price, current_state, next_state)

    return next_state, reward
