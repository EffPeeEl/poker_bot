
from action import Action


class Player:
    def __init__(self, name : str, chips: float):
        self.name = name
        self.chips = chips
        self.hand = []
        self.bet = 0
        self.min_raise = 0
        self.max_raise = 0
        
    
    def receive_cards(self, cards : list):
        self.hand = cards

    def edit_name(self, name:str):
        self.name = name
    
    def set_best_hand(self, best_hand):
        self.best_hand = best_hand
    
    def get_action(self, action_validator) -> Action:
        print(f"{self.name}, what is your action?")
        x =  input().split()
        action_type = x[0]
        amount = 0
        if len(x) > 1:
            amount = x[1]
        
        
        return Action(self, action_type, amount)   
    
    def __str__(self) -> str:
        return self.name
    
    
    
    
    
    def set_raise_range(self, game_state, action_validator):
        range = action_validator.get_raise_range(game_state, self)
        self.min_raise = range[0]
        self.max_raise = range[1]
        
    
    ###this is very ugly, should not be needed to send the game in but this is to not have circular imports
    #TODO refactor this whole thing
    def get_legal_moves(self, game_state, action_validator) -> list[int]:
        self.set_raise_range(game_state, action_validator)
        legal_moves = [0, 0, 0, 0]  # [check, call, raise, fold]

        legal_moves[0] = 1 if action_validator.is_eligible_action(Action(self, "check", 0), game_state) else 0
        legal_moves[1] = 1 if action_validator.is_eligible_action(
            Action(self, "call", action_validator.get_call_amount(self, game_state)), game_state) else 0
        legal_moves[2] = 1 if action_validator.is_eligible_action(
            Action(self, "raise", self.min_raise), game_state) else 0
        legal_moves[3] = 1  # Players can always choose to fold, maybe change this to ensure play quality but nvm

        return legal_moves