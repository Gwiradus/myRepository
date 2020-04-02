import extraTree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

min_samples = 20
max_samples = 2000
iterations = 15
samples = np.linspace(min_samples, max_samples, iterations)
mse_1 = np.zeros(iterations)
mse_2 = np.zeros(iterations)

df_train = pd.read_csv('fx_train.csv')
df_test = pd.read_csv('fx_test.csv').truncate(after=25)

y_test = None
avg_test_2 = None

for i in range(iterations):
    # Training Set
    df_train_i = df_train.truncate(after=round(samples[i]))
    X_train = df_train_i[['x_1', 'x_2']]
    y_train = df_train_i['y']

    # Scikit Model
    model_1 = ExtraTreesRegressor(n_estimators=50)
    model_1.fit(X_train, y_train)

    # My Model
    Ntr = 50
    trees = []
    for j in range(Ntr):
        trees.append(extraTree.MyExtraTreesRegression().fit(X_train, y_train, ktr=X_train.shape[1]))

    # Testing Set
    X_test = df_test[['x_1', 'x_2']]
    y_test = df_test['y']

    # Predictions
    pred_test_1 = model_1.predict(X_test)
    pred_test_2 = 0
    for tree in trees:
        pred_test_2 += tree.predict(X_test)
    avg_test_2 = pred_test_2 / Ntr

    # Indicators
    mse_1[i] = mean_squared_error(y_test, pred_test_1)
    mse_2[i] = mean_squared_error(y_test, avg_test_2)

# Plot MSE
fig = plt.figure()
ax_2 = fig.add_subplot(111)
ax_2.scatter(samples, mse_1, c='r', marker='o')
ax_2.scatter(samples, mse_2, c='b', marker='o')
ax_2.set_xlabel('Samples')
ax_2.set_ylabel('Mean Squared Error')
plt.show()
