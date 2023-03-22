import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import csv
import numpy as np


class RNN:
    def __init__(self, x_file = 'xtrain.csv', y_file = 'ytrain.csv' ,weights="./rnn_weights"):
        self.x_file = x_file
        self.y_file = y_file
        self.weights = weights
        self.x_train = []
        self.y_train = []
        self.init_train()
        self.model = self.init_model()
        # self.load_weights()

    def init_train(self):
        self.x_train = list(self.x_train)
        self.y_train = list(self.y_train)
        with open(self.x_file, newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                tmp = []
                for i in row:
                    tmp.append(float(i))
                self.x_train.append(list(tmp))
        t = []
        xt = []
        for i in range(0,len(self.x_train)):
            t.append(self.x_train[i])
            if(i%5 == 4):
                xt.append(t)
                t = []
        self.x_train = np.array(xt[1:-1])
        with open(self.y_file, newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                tmp = []
                for i in row:
                    tmp.append(float(i))
                self.y_train.append(list(tmp))
        self.y_train = np.array(self.y_train)
        t = []
        xt = []
        for i in range(0,len(self.y_train)):
            t.append(self.y_train[i])
            if(i%5 == 4):
                xt.append(t)
                t = []
        self.y_train = np.array(xt[1:-1])

    def init_model(self):
        model = Sequential()

        # IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
        model.add(LSTM(256, input_shape=self.x_train.shape[1:], activation='relu',return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(256, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='sigmoid'))

        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        # Compile model
        model.compile(
            loss='mse',
            optimizer=opt,
            metrics=['accuracy'],
        )
        self.model = model
        return model

    def load_weights(self):
        self.model.load_weights(self.weights)

    def save_weights(self):
        self.model.save_weights(self.weights)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train,epochs=100)

    def predict(self, x):
        p = self.model.predict(x)
        return p

r = RNN()
# r.train_model()
# sample = r.x_train[0]
# y = r.model.predict(sample)
# print("prediction ", y)
# print("actual ", r.y_train[0])
#
# sample = r.x_train[-1]
# y = r.model.predict(sample)
# print("prediction ", y)
# print("actual ", r.y_train[-1])
