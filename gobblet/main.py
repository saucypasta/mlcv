import config
from net import NN
from UCB import UCB
from GameManagers import GobletGameManager

def train_model(GameManager,CPU):

	training_epochs = config.TOTAL_TRAINING_EPOCHS

	for i in range(training_epochs):
		CPU.self_play(games_per_epoch,mcts_per_game,mcts_batch_size)


def run_model(GameManager,CPU,run):
	game_state = GameManager.initialize_game()
	GameManager.state_rules()
	GameManager.visualize_game()

	if run == 1:
		user_turn = True
	elif run == 2:
		user_turn = False
	else:
		user_turn = random.choice([True,False])

	while not done:
		if(user_turn):
			action = input("Select Action")
			action_idx = GameManager.process_user_action(action)
			done,w = GameManager.is_terminal(game_state)
			
		else:
			action_idx = CPU.play(state)

			done,w = GameManager.is_terminal(game_state)
			w = -w

		user_turn = not user_turn
		game_state = GameManager.take_action(game_state,action_idx)
		GameManager.visualize_game()
		if done:
			if w == 1:
				print("You Win")
			elif w == -1:
				print("You Lose")
			else:
				print("You Tied")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dest',dest = 'model_dest', type = str, default = './models/',help=
    	"default = ./models")
    parser.add_argument('-game',dest = 'game', type = str, default = 'gobblet',help = 
    	"default combines with model_dest to create model save/load location = ./models/gobblet")
    parser.add_argument('-verbose', dest='verbose',action='store_true',default=False)
    parser.add_argument('-load_model', dest='load_model',action='store_true',default=False)
    parser.add_argument('-train',dest = "train",action = 'store_true',default=False)
    parser.add_argument('-run',dest = "run",type = int, default = 0,help = 
    	"1 to go first, 2 to go second, otherwise it's random")

    return parser.parse_args()

def main(args):
	args = parse_arguments()
	state_size = config.STATE_SIZE 
	action_size = config.ACTION_SIZE 
	hidden_layers = config.HIDDEN_CNN_LAYERS 
	reg_const = config.REG_CONST 
	learning_rate = config.LEARNING_RATE 

	model_file = args.model_dest + args.game

	net = NN(state_size, action_size, hidden_layers, reg_const, learning_rate, model_file)

	if args.game == "gobblet":
		GameManager = GobletGameManager()

	if args.load_model:
		net.load()

	CPU = Agent(net,GameManager)

	if args.train:
		train_model(GameManager,CPU)
	if args.run:
		run_model(GameManager,CPU,args.run)
		

		

if __name__ == '__main__':
	main()