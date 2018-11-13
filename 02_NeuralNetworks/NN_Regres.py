
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


INPUT_DIMENSION = 0
DATA_FILE = ''

#Función de normalización para atributos con distintos rangos
def norm(X):
    X_norm = X.copy()
    mu = []
    sigma = []

    for j in range(len(X[0])):
        mu.append(np.mean(X[:,j]))
        sigma.append(np.std(X[:,j]))
        for i in range(len(X)):
            X_norm[i][j] = (X[i][j] - mu[j])/sigma[j]
    return X_norm

#Carga de datos (una sola salida)
def load_data():
    data = pd.read_csv(DATA_FILE).values
    x = data[:, 0:INPUT_DIMENSION]
    y = data[:, INPUT_DIMENSION]
    return x, y


def baseline_model():
    model = Sequential()
    model.add(Dense(INPUT_DIMENSION, input_dim=INPUT_DIMENSION, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)

    X, Y = load_data()
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
