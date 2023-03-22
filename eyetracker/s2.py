import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import csv
import numpy as np
import matplotlib.pyplot as plt

class SEQ:
    def __init__(self, weights = "./seq2", x_file="xtrain.csv", y_file="ytrain.csv"):
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
        self.x_train = np.array(self.x_train)
        with open(self.y_file, newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                tmp = []
                for i in row:
                    tmp.append(float(i))
                self.y_train.append(list(tmp))
        self.y_train = np.array(self.y_train)

    def init_model(self):
        model = Sequential()
        model.add(Dense(256,input_dim=142,activation='relu'))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(2,activation='sigmoid'))
        model.compile(loss='MSE')
        return model

    def load_weights(self):
        self.model.load_weights(self.weights)

    def save_weights(self):
        self.model.save_weights(self.weights)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, validation_split=.2,batch_size = 20,epochs=1000)

    def predict(self, x):
        p = self.model.predict(x)
        return p

seq = SEQ()
