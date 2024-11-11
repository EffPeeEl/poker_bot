from action import Action
from game_state import GameState

# class ActionValidator:

#     def __init__(self) -> None:
#         pass


#     ##not neccessary to print anything, printing should be on a case by case
#     @staticmethod
#     def is_eligible_action(action: Action, game_state : GameState) -> bool:
#         highest_bet = max([player.bet for player in game_state.players_in_hand], default=0)
#         call_amount = highest_bet - action.player.bet
#         #print(f"Call Amount: {call_amount}, Highest Bet: {highest_bet}")
        
#         if action.action_type == "fold":
#             return True
        
#         elif action.action_type == "call":
#             return action.player.chips >= call_amount 
        
#         elif action.action_type == "raise":
#             min_raise = max(highest_bet * 2, highest_bet)
#             total_bet = action.player.bet + action.amount
#             #print(f"Total Bet: {total_bet}, Min Raise: {min_raise}, Chips: {action.player.chips}")
#             return action.player.chips >= action.amount and total_bet >= min_raise
        
#         elif action.action_type == "check":
#             if call_amount == 0:
#                 return True
#             else:
#                 #print(f"Invalid action {action} (call amount: {call_amount})")
#                 return False
        
#         elif action.action_type == "post":
#             return action.player.chips >= action.amount
        
#         #print(f"INVALID ACTION [{action}] (highest bet: {highest_bet})")
#         return False
    
#     @staticmethod 
#     def get_call_amount(player, game_state : GameState) -> float:
#         highest_bet = max([player.bet for player in game_state.players_in_hand], default=0)
#         call_amount = highest_bet - player.bet
#         return call_amount 
    
    
#     @staticmethod
#     def get_raise_range(game_state : GameState, player) -> tuple[int, int]:

        
#         highest_bet = max([player.bet for player in game_state.players_in_hand], default=0)
#         min_raise = max(highest_bet * 2, highest_bet)
#         max_raise = player.chips + player.bet
#         return (min_raise, max_raise)

class ActionValidator:

    @staticmethod
    def is_eligible_action(action: Action, game_state: GameState) -> bool:
        if action is None:
            return False
        
        current_bet = game_state.current_bet
        last_raise_amount = game_state.last_raise_amount
        player_bet = action.player.bet
        call_amount = current_bet - player_bet

        if action.action_type == "fold":
            return True

        elif action.action_type == "call":
            return action.player.chips >= call_amount and call_amount > 0

        elif action.action_type == "raise":
            min_raise_amount = last_raise_amount if last_raise_amount > 0 else game_state.big_blind
            amount_to_put_in = call_amount + min_raise_amount
            raise_amount = action.amount
            has_enough_chips = action.player.chips >= raise_amount
            is_valid_raise = raise_amount >= amount_to_put_in
            return is_valid_raise and has_enough_chips

        elif action.action_type == "check":
            return call_amount == 0

        elif action.action_type == "post":
            return action.player.chips >= action.amount

        return False

    @staticmethod
    def get_call_amount(player, game_state: GameState) -> float:
        current_bet = game_state.current_bet
        call_amount = current_bet - player.bet
        return max(call_amount, 0)

    @staticmethod    
    def get_raise_range(game_state: GameState, player) -> tuple[int, int]:
        current_bet = game_state.current_bet
        last_raise_amount = game_state.last_raise_amount
        min_raise_amount = last_raise_amount if last_raise_amount > 0 else game_state.big_blind
        call_amount = current_bet - player.bet
        min_raise = call_amount + min_raise_amount
        max_raise = player.chips
        return (min_raise, max_raise)