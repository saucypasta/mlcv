#### SELF PLAY
EPISODES = 20 #number of games to play per training epoch
MCTS_SIMS = 1000 #rollouts on every turn
MCTS_BATCH_SIZE = 50
TURNS_UNTIL_TAU0 = 20 # turn on which it starts playing deterministically
CPUCT = 2 #constant for tuning UCB = Q + CPUCT*U
EPSILON = 0.2 
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 2
REG_CONST = 0.0001 #regularization for convolutional layers
LEARNING_RATE = 0.001 #learning rate for Adam Optimizer
TRAINING_LOOPS = 10 #num minibatches to train on per self-play epoch
#WINDOW_SIZE = 4 means to remember last 4 training epochs when updating net
WINDOW_SIZE_LOW = 4 
WINDOW_SIZE_HIGH = 20

#Shapes of convolutional and Resnet layers
HIDDEN_CNN_LAYERS = [{'filters':128, 'kernel_size': (3,3)} for _ in range(20)]

####TRAINING DURATION
TOTAL_TRAINING_EPOCHS = 1000

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3


#### GAME PARAMS
#any square on the board can have any combination of 8 types of pieces on it
STATE_SIZE = (4,4,8)
#Move a piece from any square to any other square, or move a piece from your hand to anywhere on the board
ACTION_SIZE = 16*15+4*16