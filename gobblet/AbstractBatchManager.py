from waiting import wait
import numpy as np
import time
import logging

import threading


class Abstract_Batch_Manager():
	def __init__(self,net,batch_size):
		self.net = net
		self.batch_size = batch_size
		self.batch_done = False
		self.hold = False

		self.batch = []
		self.prediction = []

	def make_pred(self,batch):
		return self.net.predict(np.array(self.batch))

	def request_prediction(self,state):
		wait(lambda:self.not_holding())

		self.batch.append(state)
		idx = len(self.batch)-1

		if len(self.batch) == self.batch_size:
			self.prediction = self.run_batch(self.batch)
			self.batch_done = True
			self.hold = True
			self.finished_waiting = 1
			self.total_batch_size = len(self.batch)
			self.batch = []


		else:
			wait(lambda:self.batch_ready(),waiting_for="batch to be full")
			self.finished_waiting += 1

		if self.finished_waiting == self.total_batch_size:
			self.batch_done = False
			self.hold = False

		return self.prediction[idx]
		
	def not_holding(self):
		return not self.hold

	def batch_ready(self):
		return self.batch_done


# b = Abstract_Batch_Manager(None,5)

# def thread_function(req):
# 	print(b.request_prediction(req))

# for i in range(5):
# 	x = threading.Thread(target=thread_function, args=[i])
# 	x.start()
# 	time.sleep(1)



