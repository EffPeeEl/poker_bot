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
    def __init__(self, game, iterations=1000) -> None:
        self.game = game
        self.iterations = iterations

    def run_simulation(self):
        hands = []

        for i in range(self.iterations):
            self.game.play()
            game_state = GameState.from_game(self.game)
            hands.append(PokerSerializer.game_state_to_dict(game_state))
            print(f"Game {i+1} out of {self.iterations} finished")
            
       
        with open('hands.json', 'w') as f:
            hands 
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

if __name__ == "__main__":
    rl_agent = RLBot("rla", 1000, state_size=138, action_size=4)
    rl_agent.load_model(r'models\model.h5')  
    
    players = [
        bots.RandomBot("Random", 1000),
        bots.LearningBot("Learning", 1000),
        bots.StrategicBot("Strategic", 1000, aggression_level=0.5, tightness_level=0.5, bluff_frequency=0.2),
        bots.StrategicBot("StrategicTight", 1000, aggression_level=0.3, tightness_level=0.7, bluff_frequency=0.1),
        bots.StrategicBot("StrategicAggro", 1000, aggression_level=0.8, tightness_level=0.4, bluff_frequency=0.4),
        rl_agent
    ]

    game = TexasHoldem(small_blind=5, big_blind=10, player_list=players)

    sim = Simulation(game, iterations=1000)
    sim.run_simulation()
    sim.display_stats()
