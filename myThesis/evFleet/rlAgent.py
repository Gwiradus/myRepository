"""Agent - FQI RL"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
np.random.seed(0)


class RLAgent:

    def __init__(self, agent_info):
        """Initialization for the agent.

        Args:
        agent_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            actions (numpy array): Environment actions,
            epsilon (float): The epsilon parameter for exploration,
            discount (float): The discount factor,
        }
        """
        self.actions = None
        self.states_set = None
        self.actions_set = None
        self.times_set = None
        self.q_set = None
        self.epsilon = None
        self.discount = None
        self.q = None

        self.agent_init(agent_info)

    def agent_init(self, agent_info):
        """Setup for the agent.

        Args:
        agent_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            actions (numpy array): Environment actions,
            epsilon (float): The epsilon parameter for exploration,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_info.
        self.actions = agent_info["actions"]
        self.epsilon = agent_info["epsilon"]
        self.discount = agent_info["discount"]
        self.states_set = []
        self.actions_set = []
        self.times_set = []
        self.q_set = []

        # Create the action-value function approximator.
        self.q = ExtraTreesRegressor(n_estimators=100)
        inputs = pd.DataFrame({'X': [0.0], 'U': [0.0], 't': [7.0]})
        outputs = [0.0]
        self.q.fit(inputs, outputs)

    def agent_step(self, time, state):
        """A step taken by the agent.
        Args:
            time (float): the time of the last action taken
            state (float): the state from the environment's step based on where the agent ended up after the
            last step.
        Returns:
            action (int): the action the agent is taking.
        """

        # Choose action using epsilon greedy.
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            a = []
            for act in self.actions:
                a.append(self.q.predict(pd.DataFrame({'X': [state], 'U': [act], 't': [time]})))
            action = self.actions[np.random.choice(np.where(a == min(np.asarray(a)))[0])]

        self.actions_set.append(action)
        return action

    def agent_train(self, time, reward, state, isterminal):
        """Training the agent.
        Args:
            time (float): the time of the last action taken.
            reward (float): reward after the last action taken.
            state (float): the state from the environment's step based on where the agent ended up after the
            last step.
            isterminal (bool): flag when an episode is finished.
        Returns:
            action (int): the action the agent is taking.
        """

        # Building the training set.
        if not isterminal:
            b = []
            for action in self.actions:
                b.append(self.q.predict(pd.DataFrame({'X': [state], 'U': [action], 't': [time]})))
            output = reward + self.discount * np.min(b)
        else:
            output = reward

        self.states_set.append(state)
        self.times_set.append(time)
        self.q_set.append(output)

        # Use the regression algorithm to induce from the training set the function Q(x,u)
        inputs = pd.DataFrame({'X': self.states_set, 'U': self.actions_set, 't': self.times_set})
        outputs = self.q_set
        self.q.fit(inputs, outputs)
