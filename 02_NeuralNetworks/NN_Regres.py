
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_DIMENSION = 0

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
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':   
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)
    INPUT_DIMENSION = X_train.shape[1]
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=3, verbose=0)
    estimator.fit(np.array(X_train),np.array(y_train))
    results = cross_val_score(estimator, X_test, y_test, cv = 5)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
