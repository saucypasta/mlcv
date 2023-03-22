import pandas as pd
import numpy as np
import random as rd
import math
import os
import tensorflow.keras as k
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Input

#input = [dl, dr, dt]
#output = [d, dtheta]
training_data = list()
output = list()
test_data = list()
test_output = list()
for i in range(0, 250):
    dl = rd.randrange(-100,100)
    dr = rd.randrange(-100,100)
    model_dtheta = (dr - dl)
    actual_dtheta = model_dtheta - model_dtheta * .08 - .002 * model_dtheta**2
    model_d = (dl + dr ) / 2 #theoretical change in distance
    actual_d = model_d - model_dtheta *.05 - .001 * model_dtheta**2

    training_data.append(np.array([dl, dr]))
    output.append(np.array([actual_d, actual_dtheta]))

for i in range(0, 250):
    dl = rd.randrange(-100,100)
    dr = rd.randrange(-100,100)
    model_dtheta = (dr - dl)
    actual_dtheta = model_dtheta - model_dtheta * .08 - .002 * model_dtheta**2
    model_d = (dl + dr ) / 2 #theoretical change in distance
    actual_d = model_d - model_dtheta *.05 - .001 * model_dtheta**2

    test_data.append(np.array([dl, dr]))
    test_output.append(np.array([actual_d, actual_dtheta]))




def init_network():
    model = Sequential()
    model.add(Dense(256,input_dim=2,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(2,activation='linear'))
    model.compile(loss='MSE')
    return model

dir = os.getcwd()

# training_data = list()
# training_data.append(np.array(range(1,5)))
# training_data.append(np.array(range(5,9)))
# outs = np.array([np.array([1,2,3]),np.array([10,11,12])])
model = init_network()
model.load_weights("./weights")
model.fit(np.array(training_data),np.array(output), validation_split=.1 ,epochs=1000)
predict = model.predict(np.array(test_data))
model.save_weights("./weights")
exit(0)


#
