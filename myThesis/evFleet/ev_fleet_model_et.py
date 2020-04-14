"""EV Fleet Model"""
import numpy as np
from scipy.stats import truncnorm

np.random.seed(0)


class FleetEnvironment:
    """Implements the environment for the EV Fleet problem."""

    def __init__(self, ev_number, t_opening_hours=(7, 18), ad_means_sd=(8, 17, 1.5), charging_power=3.3):
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
        self.start_state = 0
        self.time = 0
        self.min_energy = []
        self.max_energy = []
        self.current_energy = []

        self.arrival_times = None
        self.departure_times = None
        self.required_energy = None
        self.current_state = None
        self.initialize_fleet()

        reward = 0.0
        soc = 0
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

        self.arrival_times = truncnorm((t_open - arrivals_mean) / sd, (12 - arrivals_mean) / sd,
                                       loc=arrivals_mean, scale=sd).rvs(size=self.ev_number)
        self.departure_times = truncnorm((12 - departures_mean) / sd, (t_close - departures_mean) / sd,
                                         loc=departures_mean, scale=sd).rvs(size=self.ev_number)

        travelled_distance = np.random.normal(distance_mean, distance_sd, self.ev_number)
        self.required_energy = travelled_distance * fuel_economy
        self.current_state = self.start_state

    def env_step(self, time, action):
        """A step taken by the environment.

        Args:
            time: The new event time.
            action: The action taken by the agent.

        Returns:
            (float, state, Boolean): A tuple of the reward, state observation, and boolean indicating if it's terminal.
        """

        is_terminal = False
        old_soc = self.current_state
        old_time = self.time
        self.time = time

        # update current_state with the action
        old_soc += (self.time - old_time) * self.ev_number * self.charging_power * action
        self.define_boundaries()
        self.current_state = np.clip(old_soc, self.min_energy[-1], self.max_energy[-1])
        self.current_energy.append(self.current_state)
        # calculate reward
        reward = self.spot_price()*(self.current_state - old_soc)

        if self.time == self.t_opening_hours[1]:  # terminate if goal is reached
            is_terminal = True

        self.reward_obs_term = [reward, self.current_state, is_terminal]

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
        time = self.time
        arrival_time = time_vector[0]
        departure_time = time_vector[1]
        if time < arrival_time:
            return [0, 0]
        elif arrival_time <= time <= departure_time:
            e_min = max(required_energy - self.charging_power * (departure_time - time), 0)
            e_max = min(required_energy, self.charging_power * (time - arrival_time))
            return [e_min, e_max]
        else:
            return [required_energy, required_energy]

    def spot_price(self):
        """Function that returns the day ahead price of that hour.

        Returns:
            int: The electricity price for calculating the reward during a step.
        """
        return 25 + 8 * np.sin(12 * np.pi * int(self.time) / 22)
