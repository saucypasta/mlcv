from net import NN
import config
import numpy as np
import time

state_size = config.STATE_SIZE
action_size = config.ACTION_SIZE
hidden_layers = config.HIDDEN_CNN_LAYERS
reg_const = config.REG_CONST
learning_rate = config.LEARNING_RATE
model_file = "./models/test_model"

start = time.time()
net = NN(state_size,action_size,hidden_layers,reg_const,learning_rate,model_file)
end = time.time()
print("model instantiation took",end-start,"seconds")

for i in range(10,61,10):
	start = time.time()
	test_val,test_act = net.predict(np.random.uniform(size = (i,4,4,8)))
	end = time.time()
	print("model pass for batch of size ",i,"took ",end-start,"seconds")