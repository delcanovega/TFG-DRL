
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

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

def baseline_model():
    model = Sequential()
    model.add(Dense(INPUT_DIMENSION, input_dim=INPUT_DIMENSION, kernel_initializer='normal', activation='relu'))
    model.add(Dense(26, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam' )
    return model



if __name__ == '__main__':   
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)
    INPUT_DIMENSION = X_train.shape[1]
    
    modelo = baseline_model()
    #si pones X_train y X_test la funcion de aprendizaje  se suaviza y queda mas bonita  
    results = modelo.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=1)
    
    s1 = results.history["loss"]
    s2 = results.history["val_loss"]

    plt.plot(s1, "r")#error cometido en el entrenamiento
    plt.plot(s2)#error cometido durante la validacion

    plt.show()

    print("Results: %.2f MSE" % (results.history["loss"][-1]))
