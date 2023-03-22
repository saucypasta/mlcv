class Abstract_Game_Manager():
	def initialize_game(self):
		#return state of initialized game
		pass

	def is_valid(self,state,action):
		pass

	def is_terminal(self,state):
		#returns True,w is state a terminal state
		#w = 1 if win, -1 if loss
		pass

	def take_action(self,curr_state,action):
		#returns the new state following the action taken from curr_state
		pass

	def visualize_game(self,state):
		#visualizes game through terminal or something fancier
		pass

	def process_user_action(self,action):
		#returns index of user action
		pass

	def state_rules(self):
		#prints out how the user can interact to play the game
		pass