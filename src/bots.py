from player import Player
from card_utils import Deck
from action import Action
import random
import itertools
import math

class GTOBot(Player):
    def __init__(self, name, chips, aggression_rate=0.7):
        super().__init__(name, chips)
        self.aggression_rate = aggression_rate
        self.player_mapping = {}
        self.history = []

    def add_to_mapping(self, game_state):
        self.player_mapping

    def get_action(self, game_state, action_validator) -> Action:
        self.history.append(game_state)
        self.add_to_mapping(game_state)

        if self.chips == 0:
            return Action(self, "fold")

        legal_moves = self.get_legal_moves(game_state, action_validator)

        if legal_moves[2] and random.random() < self.aggression_rate:
            raise_amount = min(self.chips, self.min_raise)
            return Action(self, "raise", raise_amount)
        elif legal_moves[1]:
            call_amount = action_validator.get_call_amount(self, game_state)
            return Action(self, "call", call_amount)
        elif legal_moves[0]:
            return Action(self, "check")
        else:
            return Action(self, "fold")


class ConservativeBot(Player):
    def __init__(self, name, chips):
        super().__init__(name, chips)
    
    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        legal_moves = self.get_legal_moves(game_state, action_validator)
        
        if any(card.rank in ['A', 'K', 'Q', 'J', 'T'] for card in self.hand):
            if legal_moves[1]:
                call_amount = action_validator.get_call_amount(self, game_state)
                return Action(self, "call", call_amount)
            elif legal_moves[0]:
                return Action(self, "check")
        return Action(self, "fold")
        

class AggressiveBot(Player):
    def __init__(self, name, chips, aggression_rate=0.7):
        super().__init__(name, chips)
        self.aggression_rate = aggression_rate
    
    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        legal_moves = self.get_legal_moves(game_state, action_validator)
        
        if legal_moves[2] and random.random() < self.aggression_rate:
            raise_amount = min(self.chips, self.min_raise)
            return Action(self, "raise", raise_amount)
        elif legal_moves[1]:
            call_amount = action_validator.get_call_amount(self, game_state)
            return Action(self, "call", call_amount)
        elif legal_moves[0]:
            return Action(self, "check")
        else:
            return Action(self, "fold")


class RandomBot(Player):
    def __init__(self, name, chips):
        super().__init__(name, chips)

    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        legal_moves = self.get_legal_moves(game_state, action_validator)
        self.set_raise_range(game_state, action_validator)
        
        actions = []
        if legal_moves[0]:
            actions.append("check")
        if legal_moves[1]:
            actions.append("call")
        if legal_moves[2]:
            actions.append("raise")
        actions.append("fold") 
        
        action_type = random.choice(actions)
        if action_type == "call":
            call_amount = action_validator.get_call_amount(self, game_state)
            return Action(self, "call", call_amount)
        elif action_type == "raise":
            min_raise = math.floor(self.min_raise)
            max_raise = math.floor(self.max_raise)
            if min_raise >= max_raise:
                raise_amount = min(self.chips, min_raise)
            else:
                raise_amount = random.randint(min_raise, max_raise)
                raise_amount = min(self.chips, raise_amount)
            return Action(self, "raise", raise_amount)
        elif action_type == "check":
            return Action(self, "check")
        else:
            return Action(self, "fold")
            

class TightBot(Player):
    def __init__(self, name, chips):
        super().__init__(name, chips)
    
    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        legal_moves = self.get_legal_moves(game_state, action_validator)
        
        if any(card.rank in ['A', 'K', 'Q', 'J', 'T'] for card in self.hand) or self.hand[0].rank == self.hand[1].rank:
            if legal_moves[1]:
                call_amount = action_validator.get_call_amount(self, game_state)
                return Action(self, "call", call_amount)
            elif legal_moves[0]:
                return Action(self, "check")
        return Action(self, "fold")
        

class LooseBot(Player):
    def __init__(self, name, chips):
        super().__init__(name, chips)
    
    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        legal_moves = self.get_legal_moves(game_state, action_validator)
        
        if random.random() < 0.9: 
            if legal_moves[1] and random.random() < 0.5:
                call_amount = action_validator.get_call_amount(self, game_state)
                return Action(self, "call", call_amount)
            elif legal_moves[0]:
                return Action(self, "check")
        return Action(self, "fold")
    
    
class StrategicBot(Player):
    def __init__(self, name, chips, aggression_level=0.5, tightness_level=0.5, bluff_frequency=0.1):
        super().__init__(name, chips)
        self.aggression_level = aggression_level
        self.tightness_level = tightness_level  
        self.bluff_frequency = bluff_frequency  
    
    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        legal_moves = self.get_legal_moves(game_state, action_validator)
        self.set_raise_range(game_state, action_validator)
        
        hand_strength = self.evaluate_hand_strength()
        
        threshold = self.tightness_level * 0.5 + 0.25  # Maps tightness_level to a threshold between 0.25 and 0.75
        
        if random.random() < self.bluff_frequency:
            hand_strength = 1  # Pretend to have the best hand
        
        if hand_strength >= threshold:
            # play aggressively
            if legal_moves[2] and random.random() < self.aggression_level:
                raise_amount = self.calculate_raise_amount(game_state, action_validator)
                return Action(self, "raise", raise_amount)
            elif legal_moves[1]:
                call_amount = action_validator.get_call_amount(self, game_state)
                return Action(self, "call", call_amount)
            elif legal_moves[0]:
                return Action(self, "check")
        else:
            # play conservatively
            if legal_moves[0]:
                return Action(self, "check")
            else:
                return Action(self, "fold")
            
        return Action(self, "fold")
    
    def evaluate_hand_strength(self):
        # hand strength evaluation (0 to 1)
        high_ranks = ['A', 'K', 'Q', 'J', 'T']
        pair = self.hand[0].rank == self.hand[1].rank
        high_cards = sum(1 for card in self.hand if card.rank in high_ranks)
        
        strength = 0
        if pair:
            strength = 0.8
        elif high_cards == 2:
            strength = 0.6
        elif high_cards == 1:
            strength = 0.4
        else:
            strength = 0.2
        return strength
    
    def calculate_raise_amount(self, game_state, action_validator):
        min_raise, max_raise = self.min_raise, self.max_raise
        raise_amount = min_raise + (max_raise - min_raise) * self.aggression_level
        raise_amount = min(raise_amount, self.chips)
        
        return raise_amount

class LearningBot(Player):
    def __init__(self, name, chips):
        super().__init__(name, chips)
        self.opponent_stats = {}  
        self.history = []         
    
    def update_opponent_stats(self, game_state):
        for action in game_state.actions_this_betting_round:
            player = action.player
            if player == self:
                continue
            if player not in self.opponent_stats:
                self.opponent_stats[player] = {'aggressive_actions': 0, 'total_actions': 0}
            if action.action_type in ['raise', 'bet']:
                self.opponent_stats[player]['aggressive_actions'] += 1
            self.opponent_stats[player]['total_actions'] += 1
    
    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        self.update_opponent_stats(game_state)
        legal_moves = self.get_legal_moves(game_state, action_validator)
        self.set_raise_range(game_state, action_validator)
        

        total_aggression = 0
        for stats in self.opponent_stats.values():
            if stats['total_actions'] > 0:
                aggression = stats['aggressive_actions'] / stats['total_actions']
                total_aggression += aggression
        average_aggression = total_aggression / len(self.opponent_stats) if self.opponent_stats else 0.5
        

        if average_aggression > 0.6:
            # Opponents are aggressive - play tighter
            tightness = 0.7
        else:
            tightness = 0.3
        
        hand_strength = self.evaluate_hand_strength()
        
        if hand_strength > tightness:
            if legal_moves[2]:
                raise_amount = self.calculate_raise_amount(game_state, action_validator)
                return Action(self, "raise", raise_amount)
            elif legal_moves[1]:
                call_amount = action_validator.get_call_amount(self, game_state)
                return Action(self, "call", call_amount)
            elif legal_moves[0]:
                return Action(self, "check")
        else:
            if legal_moves[0]:
                return Action(self, "check")
            else:
                return Action(self, "fold")
    
    def evaluate_hand_strength(self):
        
        high_ranks = ['A', 'K', 'Q', 'J', 'T']
        pair = self.hand[0].rank == self.hand[1].rank
        suited = self.hand[0].suit == self.hand[1].suit
        high_cards = sum(1 for card in self.hand if card.rank in high_ranks)
        
        strength = 0
        if pair and high_cards == 2:
            strength = 0.9
        elif pair:
            strength = 0.7
        elif suited and high_cards == 2:
            strength = 0.6
        elif high_cards == 2:
            strength = 0.5
        elif high_cards == 1:
            strength = 0.3
        else:
            strength = 0.1
        return strength
    
    def calculate_raise_amount(self, game_state, action_validator):
      
        min_raise, max_raise = self.min_raise, self.max_raise
        raise_amount = min_raise + (max_raise - min_raise) / 2
        raise_amount = min(raise_amount, self.chips)
        return raise_amount


class MonteCarloBot(Player):
    def __init__(self, name, chips, simulations=100):
        super().__init__(name, chips)
        self.simulations = simulations
    
    def get_action(self, game_state, action_validator) -> Action:
        if self.chips == 0:
            return Action(self, "fold")
        
        legal_moves = self.get_legal_moves(game_state, action_validator)
        self.set_raise_range(game_state, action_validator)
        
        win_probability = self.estimate_win_probability(game_state)
        
        if win_probability > 0.6:
            if legal_moves[2]:
                raise_amount = self.calculate_raise_amount(win_probability)
                return Action(self, "raise", raise_amount)
            elif legal_moves[1]:
                call_amount = action_validator.get_call_amount(self, game_state)
                return Action(self, "call", call_amount)
            else:
                return Action(self, "check")
        elif win_probability > 0.3:
            if legal_moves[1]:
                call_amount = action_validator.get_call_amount(self, game_state)
                return Action(self, "call", call_amount)
            else:
                return Action(self, "check")
        else:
            if legal_moves[0]:
                return Action(self, "check")
            else:
                return Action(self, "fold")
    
    def estimate_win_probability(self, game_state):
        import copy
        wins = 0
        total = self.simulations
        deck = copy.deepcopy(Deck())
        deck.remove_cards(self.hand + game_state.community_cards)
        
        for _ in range(self.simulations):
            simulation_deck = copy.deepcopy(deck)
            simulation_community = copy.deepcopy(game_state.community_cards)
            while len(simulation_community) < 5:
                simulation_community.append(simulation_deck.deal_card())
            opponent_hand = simulation_deck.deal_hand(2)
            my_best_hand = max(itertools.combinations(self.hand + simulation_community, 5), key=lambda x: self.game.evaluate_hand(x))
            opponent_best_hand = max(itertools.combinations(opponent_hand + simulation_community, 5), key=lambda x: self.game.evaluate_hand(x))
            my_score = self.game.evaluate_hand(my_best_hand)
            opponent_score = self.game.evaluate_hand(opponent_best_hand)
            if my_score > opponent_score:
                wins += 1
        return wins / total
    
    def calculate_raise_amount(self, win_probability):
        min_raise, max_raise = self.min_raise, self.max_raise
        raise_amount = min_raise + (max_raise - min_raise) * (win_probability - 0.6) / 0.4
        raise_amount = min(raise_amount, self.chips)
        return raise_amount
