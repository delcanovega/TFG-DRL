
import numpy as np
from scipy.io import loadmat

from keras.models import  Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold
from collections import deque

from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def baseline_model():
    ### Neural Network's architecture
    model = Sequential()
    model.add(Dense(20, input_dim=400))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(40))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss='squared_hinge', optimizer='Adam', metrics=['accuracy'])
    return model


def load_data():
    data = loadmat('ex3data1.mat')
    y = data['y']
    x = data['X']
    return x, y


if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)

    X, Y =load_data()
    dummy_y = np_utils.to_categorical(Y)

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=1)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

