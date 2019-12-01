import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


class NeuronLayer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.rand(n_inputs, n_outputs)
        self.bias = np.random.rand(n_outputs, 1)
        self.z_value = None
        self.inputs = None

    def set_weights_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate_z(self, inputs):
        self.inputs = inputs
        self.z_value = np.dot(self.inputs, self.weights) + self.bias.T
        return self.z_value


class NeuralNetwork:
    def __init__(self, x, y, learn_rate=0.025, n_neurons_hl=32, n_hl=1):
        self.inputs = x
        self.y = y
        self.outputs = np.zeros(self.y.shape)
        self.learn_rate = learn_rate
        self.n_neurons_hl = n_neurons_hl  # Number of neurons in hidden layers
        self.n_hl = n_hl  # Number of hidden layers
        self.layers = None
        self.create_network()

    def create_network(self):
        # create the first layer, hidden layers, and hidden-to-output layer
        self.layers = [NeuronLayer(self.inputs.shape[1], self.n_neurons_hl)]
        self.layers += [NeuronLayer(self.n_neurons_hl, self.n_neurons_hl) for _ in range(0, self.n_hl - 1)]
        self.layers += [NeuronLayer(self.n_neurons_hl, self.outputs.shape[1])]

    def feedforward(self, x):
        self.inputs = x
        # calculate the first layer, hidden layers, and hidden-to-output layer
        self.layers[0].z_value = tanh(self.layers[0].calculate_z(self.inputs))
        for j in range(0, self.n_hl - 1):
            self.layers[j].z_value = tanh(self.layers[j].calculate_z(self.inputs))
        self.layers[self.n_hl].z_value = identity(
            self.layers[self.n_hl].calculate_z(self.layers[self.n_hl - 1].z_value))
        self.outputs = self.layers[self.n_hl].z_value

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights and biases
        dcost_dao = 2 / np.shape(self.outputs)[0] * (self.outputs - self.y)
        dao_dzo = identity_derivative(self.layers[self.n_hl].calculate_z(self.layers[0].z_value))
        dzo_dwo = self.layers[0].z_value
        d_w2 = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)
        d_b2 = np.sum(dcost_dao * dao_dzo, axis=0)

        dcost_dzo = dcost_dao * dao_dzo
        dzo_dah = self.layers[self.n_hl].weights
        dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
        dah_dzh = tanh_derivative(self.layers[0].calculate_z(self.inputs))
        dzh_dwh = self.inputs
        d_w1 = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
        d_b1 = np.sum(dah_dzh * dcost_dah, axis=0)

        # update the weights and biases with the derivative of the loss function
        self.layers[0].set_weights_bias(self.layers[0].weights - self.learn_rate * d_w1,
                                        self.layers[0].bias - self.learn_rate * d_b1.reshape(self.layers[0].bias.shape))
        self.layers[1].set_weights_bias(self.layers[1].weights - self.learn_rate * d_w2,
                                        self.layers[1].bias - self.learn_rate * d_b2.reshape(self.layers[1].bias.shape))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def identity(x):
    return x


def identity_derivative(x):
    return 1


min_samples = 20
max_samples = 2000
iterations = 15
samples = np.linspace(min_samples, max_samples, iterations)
mse_1 = np.zeros(iterations)
mse_2 = np.zeros(iterations)

df_train = pd.read_csv('fx_train.csv')
df_test = pd.read_csv('fx_test.csv').truncate(after=25)

for i in range(iterations):
    # Training Set
    df_train_i = df_train.truncate(after=round(samples[i]))
    X = df_train_i[['x_1', 'x_2']].to_numpy()
    Y = df_train_i['y'].to_numpy().reshape(df_train_i.shape[0], 1)

    # Scikit Model
    model_1 = ExtraTreesRegressor(n_estimators=50)
    model_1.fit(X, Y)

    # My Model
    nn = NeuralNetwork(X, Y)
    
    for nn_train in range(25000):
        nn.feedforward(X)
        nn.backprop()

    # Testing Set
    X_test = df_test[['x_1', 'x_2']].to_numpy()
    Y_test = df_test['y'].to_numpy().reshape(df_test.shape[0], 1)

    # Predictions
    pred_test_1 = model_1.predict(X_test)
    nn.feedforward(X_test)
    pred_test_2 = nn.outputs

    # Indicators
    mse_1[i] = mean_squared_error(Y_test, pred_test_1)
    mse_2[i] = mean_squared_error(Y_test, pred_test_2)
    print(mse_1[i])
    print(mse_2[i])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_test['x_1'], df_test['x_2'], Y_test, c='r', marker='o')
ax.scatter(df_test['x_1'], df_test['x_2'], pred_test_2, c='b', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# Plot 2
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111)
ax_2.scatter(samples, mse_1, c='r', marker='o')
ax_2.scatter(samples, mse_2, c='b', marker='o')
ax_2.set_xlabel('Samples')
ax_2.set_ylabel('Mean Squared Error')
plt.show()
