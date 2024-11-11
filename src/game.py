from player import Player
from action import Action
from card_utils import Deck, Card
import itertools
import random
from game_state import GameState
from poker_serializer import PokerSerializer
from action_validator import ActionValidator

from collections import Counter


class TexasHoldem:

    def __init__(self, 
                 player_list : list[Player] = [], 
                 num_human_players_to_add : int = 0, 
                 starting_chips : float = 1000, 
                 deck : Deck = Deck(), 
                 small_blind: float = 5, 
                 big_blind : float = 10, 
                 ante : float= 0):
        
        
        self.reset_stacks = True
        self.winners = []
        self.players = [Player(f"Player {i + 1}", starting_chips) for i in range(num_human_players_to_add)]
    
        self.players += player_list
        self.players_in_hand :list[Player] = self.players.copy()
        
        self.deck : Deck  = deck
        self.pot :float = 0
        
        self.end_pot : float = 0
        
        self.community_cards = []

        self.actions_this_betting_round = []
        self.actions_full_round = []

        self.is_game_on = True

        self.small_blind : float = small_blind
        self.big_blind : float = big_blind
        self.ante : float = ante

    def last_action(self) -> Action:
        return self.actions_this_betting_round[-1] if self.actions_this_betting_round else None

    def add_player(self, player: Player = None ) -> None:
        
        if player is None:
            return
        self.players.append(player)
        self.players_in_hand.append(player)

    def deal_hands(self) -> None:
        for player in self.players:
            player.receive_cards(self.deck.deal_hand(2))
    

    def deal_flop(self) -> None:
        self.community_cards = self.deck.deal_hand(3)
    
    def deal_turn(self) -> None:
        self.community_cards.append(self.deck.deal_card())
    
    def deal_river(self) -> None:
        self.community_cards.append(self.deck.deal_card())



    def determine_winner(self) :
        for player in self.players_in_hand:
            player.set_best_hand(max(itertools.combinations(self.community_cards + player.hand, 5), key=lambda x: self.evaluate_hand(x)))

        best_hands = [player.best_hand for player in self.players_in_hand]
        winning_index = max(range(len(best_hands)), key=lambda x: self.evaluate_hand(best_hands[x]))
        

        ##need logic for split pots
        winning_players : list[Player] = [] 
        winning_hand_value = self.evaluate_hand(best_hands[winning_index])
        
        for i, hand in enumerate(best_hands):
            if self.evaluate_hand(hand) == winning_hand_value:
                winning_players.append(self.players_in_hand[i])
        
        return winning_players
    



    def handle_action(self, action : Action) -> None:
        
        if action.action_type == "fold":
            self.players_in_hand.remove(action.player)
            if len(self.players_in_hand) == 1:
                self.betting_round_over = True
        elif action.action_type in ["call", "raise", 'post']:
            if action.player.chips >= action.amount:
                action.player.chips -= action.amount  
                action.player.bet += action.amount
                self.pot += action.amount
            else:
                raise ValueError(f"Player {action.player.name} does not have enough chips.")
        elif action.action_type == "check": 
            pass

        
        else:
            raise ValueError(f"Player {action.player.name}: {action.action_type} :\n [FULL ACTIONS]: \n {self.actions_full_round} )")
         # print(action)

        self.actions_this_betting_round.append(action)
        self.actions_full_round.append(action)
    

    ## could do calculation on all eligble hands and have them stored, might be faster
    # should be approx 7k hands, should actually do this pre-calc in some kind of dict
    def evaluate_hand(self, five_card_hand) -> int:
    
        rank_map = {r: i for i, r in enumerate('23456789TJQKA', 2)}
        rank_values = [rank_map[card.rank] for card in five_card_hand]
        rank_values.sort(reverse=True)  
        
        rank_counts = Counter(rank_values)
        counts = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))  # sort by count descending, then rank descending

        suits = [card.suit for card in five_card_hand]
        is_flush = len(set(suits)) == 1

        is_straight = False
        if len(rank_counts) == 5:
            if rank_values[0] - rank_values[-1] == 4:
                is_straight = True
            elif rank_values == [14, 5, 4, 3, 2]:  #ace straight
                is_straight = True
                rank_values = [5, 4, 3, 2, 1]  # adjust for scoring


        #kinda ghetto but should work
        HAND_RANKS = {
            'HIGH_CARD': 0,
            'ONE_PAIR': 1 * 10**10,
            'TWO_PAIR': 2 * 10**10,
            'THREE_OF_A_KIND': 3 * 10**10,
            'STRAIGHT': 4 * 10**10,
            'FLUSH': 5 * 10**10,
            'FULL_HOUSE': 6 * 10**10,
            'FOUR_OF_A_KIND': 7 * 10**10,
            'STRAIGHT_FLUSH': 8 * 10**10,
            'ROYAL_FLUSH': 9 * 10**10,
        }

        score = 0

    

        if is_flush and is_straight:
            if rank_values[0] == 14:
                score += HAND_RANKS['ROYAL_FLUSH']
            else:
                score += HAND_RANKS['STRAIGHT_FLUSH'] + rank_values[0] * 10**8
        elif counts[0][1] == 4:
            score += HAND_RANKS['FOUR_OF_A_KIND']
            score += counts[0][0] * 10**8  
            score += counts[1][0] * 10**6  
        elif counts[0][1] == 3 and counts[1][1] == 2:
            score += HAND_RANKS['FULL_HOUSE']
            score += counts[0][0] * 10**8  
            score += counts[1][0] * 10**6  
        elif is_flush:
            score += HAND_RANKS['FLUSH']
            for i, rank in enumerate(rank_values):
                score += rank * 10**(8 - i * 2)
        elif is_straight:
            score += HAND_RANKS['STRAIGHT']
            score += rank_values[0] * 10**8  # highest card in straight, only matters due to nature of straight
        elif counts[0][1] == 3:
            score += HAND_RANKS['THREE_OF_A_KIND']
            score += counts[0][0] * 10**8
            kickers = [rank for rank in rank_values if rank != counts[0][0]]
            for i, rank in enumerate(kickers):
                score += rank * 10**(6 - i * 2)
        elif counts[0][1] == 2 and counts[1][1] == 2:
            score += HAND_RANKS['TWO_PAIR']
            score += counts[0][0] * 10**8
            score += counts[1][0] * 10**6
            kicker = [rank for rank in rank_values if rank != counts[0][0] and rank != counts[1][0]][0]
            score += kicker * 10**4  
        elif counts[0][1] == 2:
            score += HAND_RANKS['ONE_PAIR']
            score += counts[0][0] * 10**8
            kickers = [rank for rank in rank_values if rank != counts[0][0]]
            for i, rank in enumerate(kickers):
                score += rank * 10**(6 - i * 2)
        else:
            score += HAND_RANKS['HIGH_CARD']
            for i, rank in enumerate(rank_values):
                score += rank * 10**(8 - i * 2)

        return score

    def take_blinds(self) -> bool:
        if(len(self.players_in_hand) < 3):
             # print("Not enough players to play")
            return False
        blind_action = Action(self.players_in_hand[0], "post", self.small_blind)
        valid_action = ActionValidator.is_eligible_action(blind_action, GameState.from_game(self))
        if not valid_action:
             # print("Invalid small blind action")
            return False
        
        self.handle_action(blind_action)

        blind_action = Action(self.players_in_hand[1], "post", self.big_blind)
        valid_action = ActionValidator.is_eligible_action(blind_action, GameState.from_game(self))
        if not valid_action:
             # print("Invalid big blind action")
            return False
        self.handle_action(blind_action)
        
        for player in self.players_in_hand:
            valid_action = ActionValidator.is_eligible_action(Action(player, "post", self.ante), GameState.from_game(self))
            if not valid_action:
                 # print(f"Invalid ante action: {player}")
                return False
            self.handle_action(Action(player, "post", self.ante))
    
    def reset(self):
        self.community_cards = []
        self.pot = 0
        self.actions_this_betting_round = []
        self.actions_full_round = []
        self.players_in_hand = self.players.copy()
        self.deck.reset()
        self.end_pot = 0
        
        for player in self.players:
            player.bet = 0
            
        
            
    def play(self):
            
        if self.reset_stacks:
            for p in self.players:
                if p.chips < 1000:
                    p.chips = 1000

        self.deck.reset()
        self.reset()
        self.take_blinds()

        self.deal_hands()
        self.betting_round(start_index=2)

        if len(self.players_in_hand) > 1:
            self.deal_flop()
            self.betting_round()

        if len(self.players_in_hand) > 1:
            self.deal_turn()
            self.betting_round()

        if len(self.players_in_hand) > 1:
            self.deal_river()
            self.betting_round()

        winning_players = []
        if len(self.players_in_hand) > 1:
            winning_players = self.determine_winner()
        elif len(self.players_in_hand) == 1:
            winning_players.append(self.players_in_hand[0])
        else:
            # All players have folded, which shouldn't happen
            print("Error: No players left in hand. No winners.")
            self.winners = [] 
            return  

        # print("Winner(s):", ', '.join([player.name for player in winning_players]))
        self.winners = winning_players 

        self.payout(winning_players)
        self.end_pot = self.pot

        self.players_in_hand = self.players.copy()
        self.board_rotation()
        
        
        final_game_state = GameState.from_game(self)

        for player in self.players:
            if hasattr(player, 'end_hand'):
                player.end_hand(final_game_state)
        

    def play_loop(self):

        while self.is_game_on:
            self.play()
           
            if input("Play another hand? (y/n): ").lower() != 'y':
                self.is_game_on = False

    
    def payout(self, players : list[Player]) -> None:
        
        for p in players:
            p.chips += self.pot / len(players)
            
        
        self.pot = 0

    ## this makes big blinds always at index 1, and small blinds at index 0, first aggressor is at index 2 if there are more than 2 players
    #TODO, fix more so works heads up (less than 3 players)
    def board_rotation(self) -> None:
        self.players = self.players[1:] + self.players[:1]



    def parse_action(self,action_str : str, player) -> Action:
        action_str = action_str.lower()
        action_parts = action_str.split()
        action_type = action_parts[0]
        amount = int(action_parts[1]) if len(action_parts) > 1 else 0
        
        highest_bet = max([player.bet for player in self.players_in_hand], default=0)
        call_amount = highest_bet - player.bet

        ## unsure if should include post here since it´s not a standard action, maybe break this whole func out to a parser class
        if action_type not in ["fold", "call", "raise", "check", "post"]:
            raise ValueError("Invalid action")
        
        return Action(player, action_type, amount)



    # def betting_round(self, start_index=0):

        
    #     active_players = self.players_in_hand.copy()[start_index:] + self.players_in_hand.copy()[:start_index]
    #     current_bet = 0
    #     last_aggressor = None
    #     players_acted = 0
    #     total_players = len(active_players)

    #     while players_acted < total_players:
    #         for player in active_players[:]:
    #             if player == last_aggressor and players_acted >= total_players:
    #                 return 

    #             valid_action = False
    #             while not valid_action:
                    
    #                 action = player.get_action(GameState.from_game(self), ActionValidator())
    #                 valid_action = ActionValidator.is_eligible_action(action, GameState.from_game(self))
    #                 if not valid_action:
    #                     print(f"INVALID ACTION: {ActionValidator.is_eligible_action(action, GameState.from_game(self))}")

    #             self.handle_action(action)

    #             if action.action_type == "fold":
    #                 active_players.remove(player)
    #                 total_players -= 1
    #                 if total_players == 1:
    #                     return 
    #             elif action.action_type == "raise":
    #                 current_bet = player.bet
    #                 last_aggressor = player
    #                 players_acted = 1  
    #             elif action.action_type == "call":
    #                 players_acted += 1
    #             elif action.action_type == "check":
    #                 if current_bet == 0:
    #                     players_acted += 1

    #      # print("The betting round is over.")
        
    #     for player in self.players_in_hand:
    #         player.bet = 0


    def betting_round(self, start_index=0):
        active_players = self.players_in_hand.copy()
        bets = {player: 0 for player in active_players}
        current_bet = 0

        action_order = active_players[start_index:] + active_players[:start_index]
        players_to_act = action_order.copy()

        self.betting_round_over = False 

        while True:
            if not players_to_act or self.betting_round_over:
                break

            player = players_to_act.pop(0)

            if player not in self.players_in_hand:
                continue

            if player.chips == 0:
                continue

            valid_action = False
            while not valid_action:
                game_state = GameState.from_game(self)
                action = player.get_action(game_state, ActionValidator())
                if action is None:
                    print(f"Error: {player.name}'s get_action method returned None. Defaulting to fold.")
                    action = Action(player, "fold")

                valid_action = ActionValidator.is_eligible_action(action, game_state)
                if not valid_action:
                    print(f"INVALID ACTION by {player.name}: {action}")
                    action = Action(player, "fold")
                    valid_action = True  

            self.handle_action(action)

            if self.betting_round_over:
                break

            if action.action_type == "fold":
                # for x in self.players_in_hand:
                #     print(x)
                # # print(self.players_in_hand)
                # print(f"Inserted: {player}")

                # self.players_in_hand.remove(player)
                
                bets.pop(player, None)
                if not players_to_act:
                    break

            elif action.action_type == "call":
                call_amount = current_bet - bets[player]
                bets[player] += call_amount

            elif action.action_type == "check":
                pass

            elif action.action_type == "raise":
                raise_amount = action.amount
                bets[player] += raise_amount
                current_bet = bets[player]
                # reset players_to_act to others who need to respond to the raise
                players_to_act = [p for p in self.players_in_hand if p != player and p.chips > 0 and bets[p] < current_bet]

            else:
                raise ValueError("Unknown action type")

            if not players_to_act:
                if all(bets[p] == current_bet or p.chips == 0 for p in self.players_in_hand):
                    break
                else:
                    players_to_act = [p for p in self.players_in_hand if bets[p] < current_bet and p.chips > 0]

        for player in self.players_in_hand:
            player.bet = 0

    def generate_test_hands() -> dict[str, list[Card]]:
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['hearts', 'diamonds', 'clubs', 'spades']

        def create_card(rank, suit):
            return Card(rank, suit)

        test_hands = {
        "Royal Flush": [create_card(rank, 'hearts') for rank in ['T', 'J', 'Q', 'K', 'A']],
        "Straight Flush": [create_card(rank, 'spades') for rank in ['5', '6', '7', '8', '9']],
        "Four of a Kind": [create_card('A', suit) for suit in suits[:4]] + [create_card('K', 'hearts')],
        "Full House": [create_card('K', suit) for suit in suits[:3]] + [create_card('Q', suit) for suit in suits[:2]],
        "Flush": [create_card(rank, 'diamonds') for rank in ['2', '5', '7', 'J', 'A']],
        "Straight": [create_card('6', 'hearts'), create_card('7', 'diamonds'), create_card('8', 'clubs'), create_card('9', 'spades'), create_card('T', 'hearts')],
        "Three of a Kind": [create_card('J', suit) for suit in suits[:3]] + [create_card('9', 'hearts'), create_card('7', 'spades')],
        "Two Pair": [create_card('T', 'hearts'), create_card('T', 'diamonds'), create_card('8', 'clubs'), create_card('8', 'spades'), create_card('A', 'hearts')],
        "One Pair": [create_card('5', 'hearts'), create_card('5', 'spades'), create_card('7', 'diamonds'), create_card('J', 'clubs'), create_card('A', 'hearts')],
        "High Card": [create_card('2', 'hearts'), create_card('7', 'diamonds'), create_card('T', 'clubs'), create_card('J', 'spades'), create_card('A', 'hearts')]
    }

        for i in range(5):
            random_hand = [create_card(random.choice(ranks), random.choice(suits)) for _ in range(5)]
            test_hands[f"Random Hand {i+1}"] = random_hand

        return test_hands


        

    # def get_legal_moves(self, state : GameState = None) -> list[int]:
    
    #     if state is None:
    #         state = GameState.game_state_from_game(self)
        
    #     person_in_action = state.whos_in_action()

    #     legal_moves = [0, 0, 0, 0] 

    #     legal_moves[0] = 1 if TexasHoldem.is_eligible_action(TexasHoldem.parse_action("check", person_in_action), person_in_action) else 0
    #     legal_moves[1] = 1 if TexasHoldem.is_eligible_action(TexasHoldem.parse_action("call",  person_in_action), person_in_action) else 0 
    #     legal_moves[2] = 1 if TexasHoldem.is_eligible_action(TexasHoldem.parse_action("raise", person_in_action), person_in_action) else 0
    #     legal_moves[3] = not legal_moves[0] #don´t want to fold where you can check, sure is legal but is not +ev-play
            
    #     return legal_moves


# test_hands = TexasHoldem.generate_test_hands()
# game = TexasHoldem(2, 100)
# for hand_name, hand in test_hands.items():
#      # print(f"\n{hand_name}:")
#      # print(", ".join(str(card) for card in hand))
#     score = TexasHoldem.evaluate_hand(deck= game.deck,five_card_hand=hand)




# game = TexasHoldem(4)

# game.take_blinds()
# game.deal_hands()
# from poker_serializer import PokerSerializer
# game.pot = 1000
# game.community_cards = [Card('A', 'hearts'), Card('K', 'hearts'), Card('Q', 'hearts')]
# state = GameState.from_game(game)

#  # print(PokerSerializer.game_state_to_json(state))

# game.play()
