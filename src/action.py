

class Action:
    def __init__(self, player , action : str , amount: int = 0):
        self.player  = player
        self.action_type : str = action
        self.amount : int = amount

    def __str__(self):
        return f"{self.player}: [{self.action_type.upper()}] {self.amount}"
   
    def __repr__(self):
        return self.__str__()