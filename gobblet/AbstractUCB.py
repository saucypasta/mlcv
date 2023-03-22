import numpy as np
import config


class UCB_Node():

	def __init__(self,parent,parent_action_idx,state,GameManager):
		self.parent = parent  #parent node
		self.parent_action_idx = parent_action_idx #index of parent action
		self.state = state

		self.GameManager = GameManager #follows AbstractGameManager class
		#self.BatchManager = BatchManager #follows AbstractBatchManager class

		self.action_size = config.ACTION_SIZE
		self.c_puct = config.CPUCT
		self.epsilon = config.EPSILON
		self.alpha = config.ALPHA 

		self.N_s = 1 #total states explored (plus one)
		self.N_s_a = np.zeros(action_size) #num nodes explored stemming from each state,action pair

		self.N_s_a_t = np.zeros(action_size) #num terminal nodes reached from each state_action pair
		self.Z = 0 #Z = sum of win/tie/loss stat from all terminal nodes stemming from self

		self.v_loss = np.zeros(action_size) #virtual losses returned by each action 

		self.children = [None for _ in range(self.action_size)]

		#Q = sum of values predicted by net of all nodes stemming from self
		self.Q = np.zeros(action_size)

	def add_dirichlet_noise(self,epsilon,alpha):
		#adds dirichlet noise only if UCB_Node is root node
		if self.parent == None:
			pass #add noise


	def fix_actions(self):
		#set probability of illegal actions to zero
		pass

	def select_leaf(self):
		#calculate UCB and determine best action
		#self.v_loss[best_action] += 1

		#if never taken
			#initialize child node corresponding to best action
			#set self.children[action_idx] to child
			#return child 
		#elif not terminal node
			#child.select_leaf()

		pass

	def expand(self,pred_v,pred_action):
		#expand is run after caller has formed and processed batch

		self.pred_v,self.pred_action = pred_v,pred_action

		self.fix_actions(GameManager)

		self.backup()


	def backup(self):
		# recursively adds self.pred_v to parent.Q[self.parent_action_idx] until root is reached
			#negative for opposite team
		#recursively subtracts 1 from parent.v_loss[self.parent_action_idx] until root

		#terminal,w = self.terminal_node()
		#if terminal
			#w = 1 if win, -1 if loss, 0 if tie
			# recursively adds w to parent.Z until root is reached
			# recursively adds 1 to parent.N_s_a_t[self.parent_action_idx] until root is reached
		pass

	def terminal_node(self):
		#returns True,w is state a terminal state
		#w = 1 if win, -1 if loss
		return self.GameManager.is_terminal()

	def valid_action(self,action_idx):
		return self.GameManager.valid_action(self.state,action_idx)

	def calculate_U(self):
		return self.c_puct*self.pred_action*np.sqrt(self.N_s)/(self.N_s_a+np.ones(action_size))

	def calculate_Q(self):
		#average predicted value
		#behave as if value of -1 was returned for every virtual loss
		self.vQ = self.Q - self.v_loss

		return np.nan_to_num(self.vQ/(self.N_s_a + self.v_loss))

	def calculate_Z(self):
		#average terminal state value
		return np.nan_to_num(self.Z/self.N_s_a_t)

	def calculate_UCB(self):
		return self.calculate_Q() + self.calculate_U()

	def request_prediction(self,state,BatchManager):
		#call some manager that waits for many requests before doing batch prediction
		pred_v,pred_action = BatchManager.request_prediction(state)

		return pred_v, pred_action
		