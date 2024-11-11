# #     print(f"Score: {score}")
from card_utils import Card
from action import Action
from player import Player

class GameState:
    def __init__(self, community_cards: list[Card], 
                 pot_size: float,  
                 actions_full_round: list[Action], 
                 players: list[Player], 
                 players_in_hand: list[Player], 
                 actions_this_betting_round: list[Action],
                 big_blind: float,
                 winners: list[Player] = None 
                ):
        self.positions = ["sb", "bb", "utg", "hijack", "cut-off", "button"][:len(players)]
        self.positions[-1] = "button"

        self.players = players
        self.players_in_hand = players_in_hand
        self.position_map = {player.name: pos for player, pos in zip(players_in_hand, self.positions)}       

        self.community_cards = community_cards
        self.pot_size = pot_size
        self.big_blind = big_blind 

        self.actions_full_round = actions_full_round
        self.actions_this_betting_round = actions_this_betting_round
        self.winners = winners  

        amounts = [action.amount for action in self.actions_this_betting_round if action.action_type in ['call', 'raise']]

        if not amounts:
            highest_bet = 0
            second_highest_bet = 0
        else:
            highest_bet = max(amounts)
            if len(amounts) > 1:
                amounts.sort(reverse=True)
                second_highest_bet = amounts[1]
            else:
                second_highest_bet = 0

        self.min_raise = highest_bet - second_highest_bet
        self.current_bet = self.calculate_current_bet()
        self.last_raise_amount = self.calculate_last_raise_amount()    
        
        
    def calculate_current_bet(self):
        return max([player.bet for player in self.players_in_hand], default=0)
    def calculate_last_raise_amount(self):
        last_raise_actions = [action for action in self.actions_this_betting_round if action.action_type == 'raise']
        if len(last_raise_actions) >= 1:
            last_raise = last_raise_actions[-1]
            previous_bet = 0
            if len(last_raise_actions) >= 2:
                previous_raise = last_raise_actions[-2]
                previous_bet = previous_raise.amount
            elif self.actions_this_betting_round[0].action_type == 'bet':
                previous_bet = self.actions_this_betting_round[0].amount
            else:
                previous_bet = self.big_blind
            return last_raise.amount - previous_bet
        else:
            return 0   
    
    
    @staticmethod
    def from_game(game):
        if game.end_pot:
            pot = game.end_pot
        else:
            pot = game.pot
        return GameState(
            game.community_cards, 
            pot, 
            game.actions_full_round, 
            game.players, 
            game.players_in_hand, 
            game.actions_this_betting_round, 
            game.big_blind,
            game.winners 
        )
    def to_json(self):
        import json
        
        state_json = {
            "community_cards": [str(card) for card in self.community_cards],
            "pot_size": self.pot_size,
            "actions_this_betting_round" : [str(action) for action in self.actions_this_betting_round],
            "players_in_hand": [player for player in self.players_in_hand]
        }
    
        return state_json
    
    
    def whos_in_action(self):
        last_action = self.actions_this_betting_round[-1]
        if last_action.action_type == "post":
            return self.players_in_hand[2]  ##person after bb
    
        
        last_player_name = last_action.player.name
        last_player_position = self.position_map[last_player_name]
        
        current_index = self.positions.index(last_player_position)
        num_players = len(self.positions)
    
        for i in range(1, num_players):
            next_index = (current_index + i) % num_players
            next_position = self.positions[next_index]
            next_player = next(player for player in self.players_in_hand if self.position_map[player.name] == next_position)
            
            if self.players_in_hand.__contains__(next_player):
                return next_player
        
        return None


    @staticmethod
    def from_json(json_string: str):
        import json
        data = json.loads(json_string)
        return GameState(data['community_cards'], data['pot_size'], data['current_round_actions'], data['players_in_hand'])
    

    def __str__(self) -> str:
        import json
        
        return json.dumps(self)  
    
    def __repr__(self) -> str:
        return self.__str__(self)
        

        