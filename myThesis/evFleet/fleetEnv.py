"""Environment - EV Fleet Model"""
import numpy as np
from scipy.stats import truncnorm

np.random.seed(0)


class FleetEnvironment:
    """Implements the environment for the EV Fleet problem."""

    def __init__(self, ev_number, t_opening_hours=(7.0, 18.0), ad_means_sd=(8, 17, 1.5), charging_power=3.3):
        """Initialization for the environment.

        Args:
            ev_number: Number of simulated electric vehicles.
            t_opening_hours: Opening and closing hours of the EV charging parking lot.
            ad_means_sd: Mean and standard deviation of arrival and departure times, respectively.
            charging_power: Power rating of the EV chargers [kW].
        """
        self.ev_number = ev_number
        self.t_opening_hours = t_opening_hours
        self.ad_means_sd = ad_means_sd
        self.charging_power = charging_power

        self.start_state = 0.0
        self.min_energy = []
        self.max_energy = []
        self.current_energy = []
        self.spot_price = []
        self.connected_evs = []
        self.delta = []

        self.current_time = None
        self.arrival_times = None
        self.departure_times = None
        self.required_energy = None
        self.current_evs = None
        self.current_state = None
        self.current_action = None
        self.initialize_fleet()

        reward = 0.0
        soc = 0.0
        termination = False
        self.reward_obs_term = [reward, soc, termination]

    def initialize_fleet(self, distance_mean=70, distance_sd=20, fuel_economy=0.174):
        """Setup for the environment called when the class is created. Initializes the fleet parameters.

        Args:
            distance_mean: Mean of the distance driven by the vehicles [km].
            distance_sd: Standard deviation of the distance driven by the vehicles [km].
            fuel_economy: Fuel economy of the vehicles [kWH/km].
        """
        t_open = self.t_opening_hours[0]
        t_close = self.t_opening_hours[1]
        arrivals_mean = self.ad_means_sd[0]
        departures_mean = self.ad_means_sd[1]
        sd = self.ad_means_sd[2]
        self.arrival_times = np.around(truncnorm((t_open - arrivals_mean) / sd, (12 - arrivals_mean) / sd,
                                                 loc=arrivals_mean, scale=sd).rvs(size=self.ev_number), 2)
        self.departure_times = np.around(truncnorm((12 - departures_mean) / sd, (t_close - departures_mean) / sd,
                                                   loc=departures_mean, scale=sd).rvs(size=self.ev_number), 2)
        travelled_distance = np.random.normal(distance_mean, distance_sd, self.ev_number)
        self.required_energy = travelled_distance * fuel_economy
        self.current_time = t_open
        self.current_evs = np.zeros(self.ev_number)
        self.current_state = self.start_state

    def get_reward_state(self, time):
        """Gets the reward and new state after taking the previous action.

        Args:
            time: The new event time.
        """
        is_terminal = False
        old_soc = self.current_state
        old_time = self.current_time
        self.current_time = time

        # update current_state with the action
        delta_t = self.current_time - old_time
        self.define_boundaries()
        self.current_state = round(
            old_soc + delta_t * sum(self.current_evs) * self.charging_power * self.current_action, 2)
        self.current_state = np.clip(self.current_state, self.min_energy[-1], self.max_energy[-1])
        self.current_energy.append(self.current_state)
        self.connected_evs.append(sum(self.current_evs))
        self.delta.append(delta_t)

        # calculate reward
        reward = spot_price(old_time) * (self.current_state - old_soc)
        self.spot_price.append(spot_price(old_time))

        # terminate if goal is reached
        if self.current_time == self.t_opening_hours[1]:
            is_terminal = True

        self.reward_obs_term = [reward, self.current_state, is_terminal]

    def env_step(self, action, next_time):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent.
            next_time: The time of the next event.
        """

        self.current_action = action
        self.get_reward_state(next_time)

    def define_boundaries(self):
        """Function to find the energy boundaries of an episode."""
        energy = self.ev_fleet_boundary()
        self.min_energy.append(energy[0])
        self.max_energy.append(energy[1])

    def ev_fleet_boundary(self):
        """Function to find the energy boundaries of the fleet of EVs.

        Returns:
            [int, int]: A list with the lower and upper energy boundaries for the fleet of EVs.
        """
        e_min = 0
        e_max = 0
        for ev in range(self.ev_number):
            if self.arrival_times[ev] < self.current_time:
                self.current_evs[ev] = 1
            if self.departure_times[ev] < self.current_time:
                self.current_evs[ev] = 0
            energy_vector = self.ev_single_boundary([self.arrival_times[ev], self.departure_times[ev]],
                                                    self.required_energy[ev])
            e_min += energy_vector[0]
            e_max += energy_vector[1]
        return [e_min, e_max]

    def ev_single_boundary(self, time_vector, required_energy):
        """Function to find the energy boundaries of a single EV.

        Returns:
            [int, int]: A list with the lower and upper energy boundaries for a single EV.
        """
        time = self.current_time
        arrival_time = time_vector[0]
        departure_time = time_vector[1]
        if time < arrival_time:
            return [0, 0]
        elif arrival_time <= time <= departure_time:
            e_min = max(required_energy - self.charging_power * (departure_time - time), 0)
            e_max = min(required_energy, self.charging_power * (time - arrival_time))
            return np.around([e_min, e_max], 2)
        else:
            return np.around([required_energy, required_energy], 2)


def spot_price(time):
    """Function that returns the day ahead price of that hour.

    Returns:
        int: The electricity price for calculating the reward during a step.
    """
    return 25 + 8 * np.sin(12 * np.pi * int(time) / 22)
