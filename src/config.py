# Hyperparameters for the RL agent
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
MEMORY_CAPACITY = 10000
TRAIN_START = 1000
TRAIN_FREQ = 1

# Model architecture parameters
STATE_SIZE = 138  # Size of the state representation
ACTION_SIZE = 4   # Number of discrete actions (e.g., check, call, raise, fold)

# File paths
MODEL_SAVE_PATH = "models/model.h5"