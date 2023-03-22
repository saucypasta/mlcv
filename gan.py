import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape,Input#, CuDNNLSTM
import numpy as np
import random as ra


class model:
    def __init__(self, name):
        self.dbg = False
        self.model_name = name

    def load_model(self, directory = "./"):
        self.model.load_weights(directory + self.model_name)

    def save_model(self, directory = "./"):
        self.model.save_weights(directory + self.model_name)

    def toggle_dbg():
        self.dbg = not self.dbg

class digit_model(model):
    def __init__(self, name="numbers2"):
        self.model = self.init_model()
        self.model_name = name
        mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
        (self.x_train, self.y_train),(self.x_test, self.y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test
        # self.y_train = self.convert_mnist_output(self.y_train)
        # self.y_test = self.convert_mnist_output(self.y_test)
        self.x_train = self.x_train/255.0
        self.x_test = self.x_test/255.0


    def init_model(self):
        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        model = Sequential()
        model.add(LSTM(128, input_shape=((28,28)), return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(128))
        model.add(Dropout(0.1))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(11, activation='softmax'))
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'],
        )
        return model

    def convert_mnist_output(self, y_data):
        y = np.zeros(shape=(np.size(y_data),11))
        for row_num,col in enumerate(y_data):
            y[row_num][col] = 1
        return y

    def train_on_batch(self, data_in, data_out,batch_size=45):
        indeces = np.random.choice(np.shape(self.x_train)[0],batch_size)
        x_data = [self.x_train[i] for i in indeces]
        y_data = self.y_train[indeces]
        self.di = data_in
        self.do = data_out
        self.xd = x_data
        self.yd = y_data
        x_data = np.concatenate((x_data, np.array(data_in)))
        y_data = np.concatenate((y_data, np.array(data_out)))

        history = self.model.fit(x_data,y_data, validation_data=(x_data, y_data))
        return history.history['val_accuracy'][0]

    def train_on_mnist(self):
        self.model.fit(self.x_train,
                  self.y_train,
                  epochs=2,
                  validation_data=(self.x_test, self.y_test))

    def validate(self, data_in, data_out):
        yTrue = []
        for i in range(len(data_in[0])):
            row = np.zeros(11)
            row[-1] = 1
            yTrue.append(row)

        model_out = self.model.predict(data_in)
        loss = tf.reduce_mean(tf.keras.losses.MSE(model_out,yTrue), 0)
        print("\n\nDigit only noise classification loss: ", loss)
        model_out = self.model.predict(self.x_test)
        print("\n\nDigit no noise classification: ", loss)
        # self.normal_eval = self.model.evaluate(self.x_test, self.y_test)


class trickster(model):
    def __init__(self, name="faked",mem_size = 10000, input_shape=624, digit_type=9):
        self.input_shape = input_shape
        self.model_name = name
        self.model = self.init_model()
        self.memory = []
        self.mem_size = 10000
        self.digit_type = digit_type
        tf.keras.backend.set_floatx('float64')

    def init_model(self):
        trickster_in = Input(shape=self.input_shape)
        trickster = Dense(784,activation='relu')(trickster_in)
        trickster = Dense(784,activation='relu')(trickster)
        trickster = Dense(28*28,activation='sigmoid')(trickster)
        trickster = Reshape((28,28))(trickster)
        trickster = Model(inputs=[trickster_in],outputs=[trickster])
        return trickster

    #digit_type = -2 = make random data, digit_type = -1 = make non random(ensures each digit is made),
    # digit_type > -1 = use only one digit
    def generate_data(self, size):
        trickster_input = []
        # yTrue = []
        yTrue = np.zeros(shape=(size,11))
        yTrue[:,-1] = 1
        for i in range(0, size):
            digit = self.digit_type
            if self.digit_type == -2:
                digit = ra.randrange(0,11)
            elif self.digit_type == -1:
                digit = i % 11
            trickster_input.append([digit])
            row = np.zeros(11)
            row[digit] = 1
            # yTrue.append(row)
        return (np.array(trickster_input), yTrue)

    def train_trickster(self, other_model, batch_size=32,flag=False):
        (trickster_input, yTrue) = self.generate_data(batch_size)
        # yTrue = np.zeros(shape=(batch_size,11))
        # yTrue[:,10] = 1
        # trickster_input = self.generate_data(self)
        self.yt = yTrue
        self.tin = trickster_input


        # trickster_input = np.ones(shape=(batch_size, self.input_shape)) * self.trick_num

        Adam = tf.keras.optimizers.Adam()
        with tf.GradientTape(watch_accessed_variables = False) as g:
            g.watch(self.model.trainable_weights)
            trickster_out = self.model(trickster_input)
            model_out = other_model.model(trickster_out)
            # trick_norms = -tf.math.log(tf.reduce_sum(tf.math.square(model_out[:,0:9]),axis=1))
            # trick_fool = 0#-tf.math.log(model_out[:,10])
            # trickster_loss = (tf.reduce_mean(trick_norms + trick_fool,axis=0))
            # trickster_loss = tf.keras.losses.MSE(model_out,np.array([yTrue]))
            # print("Rickste loss: ", trickster_loss)
            # trickster_loss = tf.keras.losses.MSE(model_out,np.array([yTrue]))
            # print("Rickste loss: ", trickster_loss)
            trickster_fake_loss = tf.reduce_mean(-tf.math.log(1-model_out[:,-1]))
            # highest_probs = [np.max(model_out[i]) for i in range(len(model_out))]
            desired_probs = [model_out[i][trickster_input[i][0]] for i in range(len(model_out))]
            # print("Desired probs: ", np.array(desired_probs))
            # print("Model out: ", model_out)
            # trickster_tricked_loss = tf.reduce_mean(-tf.math.log(highest_prob))
            trickster_tricked_loss = tf.reduce_mean(-tf.math.log(desired_probs))
            trickster_loss = trickster_fake_loss + trickster_tricked_loss
            # print("Log loss: ",trickster_loss)
            # trickster_loss = tf.keras.losses.MSE(model_out,np.array([yTrue]))
            # print("Rickste loss: ", trickster_loss)
            # print("Model Out: ", model_out[:,-1])
            trickster_grads = g.gradient(trickster_loss,self.model.trainable_weights)

        Adam.apply_gradients(zip(trickster_grads, self.model.trainable_weights))
        self.remember(trickster_out)
        self.loss = trickster_loss
        self.out = model_out
        if(flag):
            print("faked probabilities:", tf.reduce_mean(model_out[:,-1]))
            print("Desired digit probability: ", tf.reduce_mean(desired_probs))
            print("Trickster Loss: ", trickster_loss)
            #print("Trickster Norm: ", trick_norms)
            #print("Trickster Fool: ", trick_fool)
        return trickster_loss

    def remember(self,outputs):
        self.memory.extend(outputs)

        if(len(self.memory) > self.mem_size):
            start_index = len(self.memory) - self.mem_size
            self.memory = self.memory[start_index:]


    def predict(self, input):
        return self.trickster(input)

    def validate(self, model):
        (trickster_input, yTrue) = self.generate_data(1)
        trick_out = self.model.predict(trickster_input)
        model_out = model.model.predict(trick_out)
        MSE = tf.keras.losses.MSE(model_out,yTrue)
        loss = tf.reduce_mean(MSE,0)
        self.model_out = model_out
        self.loss = loss
        return loss








#
#
#
#
# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
#
# # Compile model
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=opt,
#     metrics=['accuracy'],
# )
#
# model.load_weights("./numbers")
#
# trickster_input = np.ones(10)
# trickster_in = Input(shape=np.shape(trickster_input))
# trickster = Dense(512,activation='relu')(trickster_in)
# trickster = Dense(512,activation='relu')(trickster)
# trickster = Dense(28*28,activation='sigmoid')(trickster)
# trickster = Reshape((28,28))(trickster)
# trickster = Model(inputs=[trickster_in],outputs=[trickster])
#
#
# def train_trickster(model,trickster,num_epochs=1000,batch_size=32,trick_num=9):
#     yTrue = np.zeros(10)
#     yTrue[trick_num] = 1
#     trickster_input = np.ones(10)
#     Adam = tf.keras.optimizers.Adam()
#     for i in range(num_epochs):
#         if(i%100 == 0):
#             trickster.save_weights("./trickster")
#             print("model saved!!!!")
#         trickster_input = np.random.uniform(size=(32,10))
#         with tf.GradientTape(watch_accessed_variables = False) as g:
#             g.watch(trickster.trainable_weights)
#             trickster_out = trickster(trickster_input)
#             model_out = model(trickster_out)
#             trickster_loss = tf.keras.losses.MSE(model_out,np.array([yTrue]))
#             trickster_grads = g.gradient(trickster_loss,trickster.trainable_weights)
#         print(model_out[:,9])
#
#         Adam.apply_gradients(zip(trickster_grads,trickster.trainable_weights))
#
#
# '''
# def update_trickster_on_batch(model,trickster,):
#     with tf.gradientTape(watch_accessed_variables = False) as g:
#         g.watch(trickster.trainable_weights)
#         trickster_out = trickster.predict()
#         model_out = model.predict(trickster_out)'''
# def show_img(trickster_input):
#
#     out = trickster.predict(trickster_input)
#     img = Image.fromarray(np.uint8(out[0] * 255) , 'L')
#     img.show()
# #train_trickster(model,trickster)
# #trickster.save_weights("./trickster")
# trickster_input = np.random.uniform(size=(1,10))
# trickster.load_weights("./trickster")
# show_img(trickster_input)
