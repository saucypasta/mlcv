from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')

class NN:
	def __init__(self, state_size, action_size, hidden_layers, reg_const, learning_rate, model_file):
		self.state_size = state_size
		self.action_size = action_size
		self.hidden_layers = hidden_layers
		self.reg_const = reg_const
		self.learning_rate = learning_rate
		self.model = self.build_model()
		self.model_file = model_file

	def save(self):
		self.model.save(self.model_file)

	def load(self):
		self.model.load(self.model_file)

	def predict(self,state):
		return self.model(state)

	def fit(self,state,value,action,epochs=1):
		self.model.fit(state,np.array([value,action]),epochs=epochs)

	def build_model(self):
		#copied and slightly changed from
		#https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/model.py

		main_input = Input(shape = self.state_size, name = 'main_input')

		x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

		if len(self.hidden_layers) > 1:
			for h in self.hidden_layers[1:]:
				x = self.residual_layer(x, h['filters'], h['kernel_size'])

		vh = self.value_head(x)
		ph = self.policy_head(x)

		model = Model(inputs=[main_input], outputs=[vh, ph])
		model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': tf.nn.softmax_cross_entropy_with_logits},
			optimizer=Adam(learning_rate=self.learning_rate),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
			)

		return model

	def conv_layer(self, x, filters, kernel_size):
		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=-1,dtype='float32')(x)
		x = LeakyReLU()(x)

		return (x)

	def residual_layer(self, input_block, filters, kernel_size):

		x = self.conv_layer(input_block, filters, kernel_size)

		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=-1,dtype='float32')(x)

		x = add([input_block, x])

		x = LeakyReLU()(x)

		return (x)

	def policy_head(self, x):

		x = Conv2D(
		filters = 32
		, kernel_size = (1,1)
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=-1,dtype='float32')(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			self.action_size
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'policy_head'
			)(x)

		return (x)

	def value_head(self, x):

		x = Conv2D(
		filters = 32
		, kernel_size = (1,1)
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)


		x = BatchNormalization(axis=-1,dtype='float32')(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			40
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			)(x)

		x = LeakyReLU()(x)

		x = Dense(
			1
			, use_bias=False
			, activation='tanh'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = 'value_head'
			)(x)

		return (x)

