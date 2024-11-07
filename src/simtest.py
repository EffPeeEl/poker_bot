import numpy as np
import matplotlib.pyplot as plt

from rl_bot import RLBot
import bots
from game import TexasHoldem, GameState

STATE_SIZE = 138 
ACTION_SIZE = 4  


rl_agent = RLBot("RLAgent", 1000, state_size=STATE_SIZE, action_size=ACTION_SIZE)

players = [
    bots.RandomBot("RandomBot1", 1000),
    bots.StrategicBot("StrategicBot", 1000, aggression_level=0.5, tightness_level=0.5, bluff_frequency=0.2),
    bots.StrategicBot("StrategicTight", 1000, aggression_level=0.6, tightness_level=0.7, bluff_frequency=0.3),
    rl_agent,
    bots.StrategicBot("StrategicLoose", 1000, aggression_level=0.6, tightness_level=0.2, bluff_frequency=0.3),
    bots.StrategicBot("StrategicAggro", 1000, aggression_level=0.8, tightness_level=0.5, bluff_frequency=0.65)
]


game = TexasHoldem(small_blind=5, big_blind=10, player_list=players)

model_load_path = 'models\\model.h5'
rl_agent.load_model(model_load_path)
print(f"Model loaded from {model_load_path}")

num_hands = 100

total_rewards = []
hands_won = 0
bluffs = 0
actions_count = {'fold': 0, 'call': 0, 'raise': 0, 'check': 0}
bluff_actions = []

for hand in range(num_hands):

    game.reset()
    print(f"Hand #: {hand}")

    game.play()
    
    final_game_state = GameState.from_game(game)
    
    final_chips = rl_agent.chips
    
    reward = final_chips - 1000 
    
    total_rewards.append(reward)
    
    if rl_agent in game.winners:
        hands_won += 1
        print(f"Hands won: {hands_won}")
    
    
    ##TODO: Make this more extensive, bluffs are not really like this
    hand_actions = rl_agent.get_hand_actions()
    for action in hand_actions:
        action_type = action.action_type
        if action_type in actions_count:
            actions_count[action_type] += 1
        
        if action_type == 'raise':
            hand_strength = rl_agent.get_hand_strength_at_action(action)
            if hand_strength is not None and hand_strength < 0.3:
                bluffs += 1
                bluff_actions.append((hand, action))
    
    rl_agent.end_hand(final_game_state)

average_reward = np.mean(total_rewards)

win_rate = hands_won / num_hands

bluff_frequency = bluffs / num_hands

# # total_actions = sum(actions_count.values())
# action_distribution = {action: count / total_actions for action, count in actions_count.items()}

print(f"Average Reward per Hand: {average_reward:.2f}")
print(f"Win Rate: {win_rate * 100:.2f}%")
print(f"Bluff Frequency: {bluff_frequency * 100:.2f}%")
print("Action Distribution:")
# for action, frequency in action_distribution.items():
#     print(f"  {action}: {frequency * 100:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(total_rewards)
plt.title('Total Rewards Over Hands')
plt.xlabel('Hand Number')
plt.ylabel('Reward')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(total_rewards, bins=30)
plt.title('Histogram of Rewards')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# labels = list(action_distribution.keys())
# sizes = list(action_distribution.values())

# plt.figure(figsize=(6, 6))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
# plt.title('Action Distribution')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()

cumulative_bluffs = np.cumsum([1 if hand in [b[0] for b in bluff_actions] else 0 for hand in range(num_hands)])

plt.figure(figsize=(12, 6))
plt.plot(cumulative_bluffs)
plt.title('Cumulative Bluffs Over Hands')
plt.xlabel('Hand Number')
plt.ylabel('Cumulative Bluffs')
plt.grid(True)
plt.show()
