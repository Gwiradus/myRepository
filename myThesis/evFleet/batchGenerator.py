"""Environment - Batch samples generator"""
import numpy as np

from fleetEnv import FleetEnvironment

np.random.seed(0)


def generate_batch(n_ev=10, n_episodes=50):
    """Generates a batch of EV arrivals and departures sample.

    Args:
        n_ev: Number of electric vehicles.
        n_episodes: Number of episodes in the batch.
    """
    fleets = []
    for n in range(n_episodes):
        fleets.append(FleetEnvironment(n_ev))
    return fleets
