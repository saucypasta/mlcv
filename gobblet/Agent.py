import config
from UCB import UCB_node

class Agent():
	def __init__(self,net,GameManager):
		self.temp = config.TURNS_UNTIL_TAU0

		self.training_epochs = config.EPOCHS 
		self.mini_batch_size = config.BATCH_SIZE 

		self.games_per_epoch = config.EPISODES
		self.mcts_per_game = config.MCTS_SIM
		self.mcts_batch_size = config.MCTS_BATCH_SIZE 

		#memory buffer params
		self.window_low = config.WINDOW_LOW
		self.window_high = config.WINDOW_HIGH 
		self.window = self.window_low
		self.mem = []

		self.net = net
		self.GameManager = GameManager

	def play_optimally(self,state):
		pass

	def self_play_turn(self,root_nodes):
		batch = []
		leaves = []
		mcts_rollouts_done = 0

		while self.mcts_rollouts_done < self.mcts_per_game:
			for root in root_nodes:
				for _ in range(self.mcts_batch_size):
					leaf = root.select_leaf()
					batch.append(leaf.state)
					leaves.append(leaf)
				mcts_rollouts_done += self.mcts_batch_size
			batch = np.array(batch)
			preds = self.net.predict(batch)

			for i,leaf in enumerate(leaves):
				leaf.expand(preds[i][0],preds[i][1])

		return root_nodes


	def self_play_game(self):
		init_state = self.GameManager.initialize_game()
		pred_v,pred_action = net.predict(init_state)

		for game in range(self.games_per_epoch):
			UCB = UCB_node(None,None,init_state,GameManager)
			UCB.expand(pred_v,pred_action)
			UCB.add_dirichlet_noise()
			root_nodes.append(UCB)

		terminal = np.zeros(self.games_per_epoch)
		turn_count = 0
		while not np.all(terminal):
			searched_roots = self.self_play_turn(root_nodes)
			new_roots = [root.select_action(turn_count > self.temp) for root in searched_roots]
			for root in new_roots:
				root.parent = None
				if turn_count < self.temp:
					root.add_dirichlet_noise()







	def train_on_memories(self):
		pass