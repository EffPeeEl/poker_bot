import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import pandas as pd
import seaborn as sns
from collections import Counter

from rl_bot import RLBot
import bots  
from game import TexasHoldem, GameState
from poker_serializer import PokerSerializer

logging.basicConfig(level=logging.INFO, format='%(message)s')

def convert_numpy_types(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

class Simulation:
    def __init__(self, game, number_of_games=10, iterations=1000, starting_chips=1000) -> None:
        self.game = game
        self.iterations = iterations
        self.number_of_games = number_of_games
        self.starting_chips = starting_chips

    def run_simulation(self):
        hands = []
        rounds_per_game = self.iterations // self.number_of_games

        # Loop over each "game"
        for i in range(self.number_of_games):
            # Reset each player's chip count and bet at the start of the game
            for player in self.game.players:
                player.chips = self.starting_chips
                player.bet = 0

            for j in range(rounds_per_game):
                print(f"Playing game [{i+1}/{self.number_of_games}]: Round {j+1} out of {rounds_per_game}")
                self.game.play()
                game_state = GameState.from_game(self.game)
                hands.append(PokerSerializer.game_state_to_dict(game_state))
                # Clear console (this works on Windows; use 'clear' for Linux/Mac if needed)
                os.system('cls')
       
        with open('hands.json', 'w') as f:
            json.dump(hands, f, indent=4)
        
        return hands
    
    def display_stats(self):
        """
        Loads the simulated hands data from the JSON file and generates visualizations:
        1. Number of Hands Won per Player
        2. Chip Counts Over Time per Player
        3. Card Win Rates
        """
        try:
            with open('hands.json', 'r') as f:
                hands_data = json.load(f)
            print("Loaded 'hands.json' successfully.")
        except FileNotFoundError:
            print("Error: 'hands.json' not found. Please run the simulation first.")
            return
        
        win_counts = Counter()
        chip_history = {}  
        card_counter = Counter()
        player_names = set()
        
        for hand_index, hand in enumerate(hands_data):
            players = hand.get('players', [])
            for player in players:
                name = player['name']
                chips = player['chips']
                player_names.add(name)
                if name not in chip_history:
                    chip_history[name] = []
                chip_history[name].append(chips)
            
            winners = hand.get('winners', [])
            for winner in winners:
                win_counts[winner] += 1
                for player in players:
                    if player['name'] == winner:
                        hand_cards = [card['rank'] for card in player['hand']]
                        card_counter.update(hand_cards)
                        break

        chip_df = pd.DataFrame(chip_history)
        chip_df.index.name = 'Hand Number'
        chip_df.reset_index(inplace=True)
        chip_df['Hand Number'] += 1  
    
        win_counts_df = pd.DataFrame(win_counts.items(), columns=['Player', 'Hands Won'])
        win_counts_df = win_counts_df.sort_values(by='Hands Won', ascending=False)
    
        cards_df = pd.DataFrame(card_counter.items(), columns=['Card', 'Wins'])
        cards_df = cards_df.sort_values(by='Wins', ascending=False)
    
        sns.set(style="whitegrid")
    
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Player', y='Hands Won', data=win_counts_df, palette='viridis')
        plt.title('Number of Hands Won per Player')
        plt.xlabel('Player')
        plt.ylabel('Hands Won')
        plt.tight_layout()
        plt.show()
    
        plt.figure(figsize=(12, 6))
        for player in player_names:
            plt.plot(chip_df['Hand Number'], chip_df[player], label=player)
        plt.title('Chip Counts Over Time')
        plt.xlabel('Hand Number')
        plt.ylabel('Chips')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Card', y='Wins', data=cards_df, palette='coolwarm')
        plt.title('Card Ranks in Winning Hands')
        plt.xlabel('Card Rank')
        plt.ylabel('Number of Wins')
        plt.tight_layout()
        plt.show()
    
        print("Visualizations generated successfully.")

def simulate_instance(sim_config):
    """
    Runs a simulation instance.
    
    sim_config is a dictionary containing:
      - iterations: number of rounds for this instance.
      - starting_chips: initial chip count for each player.
    """
    rl_agent1 = RLBot("RLA V1", sim_config["starting_chips"], state_size=138, action_size=4)
    rl_agent1.load_model("weights/model_v1.h5")
    
    rl_agent1.epsilon = 0.0

    rl_agent2 = RLBot("RLA V2", sim_config["starting_chips"], state_size=138, action_size=4)
    rl_agent2.load_model("weights/model_v2.h5")
    rl_agent2.epsilon = 0.0

    strategicAggro = bots.StrategicBot("StrategicAggro", sim_config["starting_chips"], 
                                       aggression_level=0.8, tightness_level=0.4, bluff_frequency=0.4)
    strategicTight = bots.StrategicBot("StrategicTight", sim_config["starting_chips"], 
                                       aggression_level=0.3, tightness_level=0.7, bluff_frequency=0.1)

    players = [rl_agent1, strategicAggro, rl_agent2, strategicTight]

    game = TexasHoldem(small_blind=5, big_blind=10, player_list=players)

    from simulation import Simulation  
    sim = Simulation(game, number_of_games=1, iterations=sim_config["iterations"], starting_chips=sim_config["starting_chips"])
    
    hands = sim.run_simulation()
    return hands


import concurrent.futures
import json

if __name__ == "__main__":
    # Configuration for each simulation instance.
    sim_config = {
        "iterations": 100,         # Number of rounds per instance
        "starting_chips": 1000     # Initial chip count for players
    }
    
    num_instances = 4  # For example, run 4 simulation instances concurrently.
    all_hands = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the simulation function to a list of configurations.
        results = executor.map(simulate_instance, [sim_config] * num_instances)
    
    # Combine results from all instances.
    for hands in results:
        all_hands.extend(hands)
    
    # Save combined results to a file.
    with open("all_hands.json", "w") as f:
         json.dump(all_hands, f, indent=4)
    
    print("Simulation complete. Results saved to 'all_hands.json'.")


# if __name__ == "__main__":
#     rl_agent1 = RLBot("RLA V1", 1000, state_size=138, action_size=4)
#     rl_agent1.load_model(r"weights/model_v1.h5")
    
    
#     rl_agent2 = RLBot("RLA V2", 1000, state_size=138, action_size=4)
#     rl_agent2.load_model(r"weights/model_v2.h5")

#     ##we have already loaded the weights
#     rl_agent1.epsilon = 0.0
#     rl_agent2.epsilon = 0.0
                             
#     strategicAggro = bots.StrategicBot("StrategicAggro", 1000, 
#                                        aggression_level=0.8, tightness_level=0.4, bluff_frequency=0.4)
#     strategicTight = bots.StrategicBot("StrategicTight", 1000, 
#                                        aggression_level=0.3, tightness_level=0.7, bluff_frequency=0.1)
    
#     players = [
#         rl_agent1,
#         strategicAggro,
#         rl_agent2,
#         strategicTight,    
        
#     ]
    
#     game = TexasHoldem(small_blind=5, big_blind=10, player_list=players)

#     sim = Simulation(game, number_of_games=10, iterations=1000, starting_chips=1000)
#     sim.run_simulation()
#     sim.display_stats()
