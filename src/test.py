
import unittest
from card_utils import Card
from game import TexasHoldem
from action import Action
from  player import Player

class TestEvaluateHandInteger(unittest.TestCase):
    def setUp(self):
        self.game = TexasHoldem()

    def test_high_card(self):
        hand = [
            Card('A', 'spades'),
            Card('K', 'diamonds'),
            Card('Q', 'clubs'),
            Card('J', 'hearts'),
            Card('9', 'spades')
        ]
        score = self.game.evaluate_hand(hand)
        expected_score = (
            0 +
            14 * 10**8 +
            13 * 10**6 +
            12 * 10**4 +
            11 * 10**2 +
            9 * 10**0
        )
        self.assertEqual(score, expected_score)

    def test_one_pair(self):
        hand = [
            Card('K', 'hearts'),
            Card('K', 'spades'),
            Card('A', 'diamonds'),
            Card('Q', 'clubs'),
            Card('J', 'hearts')
        ]
        score = self.game.evaluate_hand(hand)
        expected_score = (
            1 * 10**10 +
            13 * 10**8 +  # Pair of Kings
            14 * 10**6 +  # Kicker A
            12 * 10**4 +  # Kicker Q
            11 * 10**2    # Kicker J
        )
        self.assertEqual(score, expected_score)

    def test_straight_flush(self):
        hand = [
            Card('9', 'hearts'),
            Card('8', 'hearts'),
            Card('7', 'hearts'),
            Card('6', 'hearts'),
            Card('5', 'hearts')
        ]
        score = self.game.evaluate_hand(hand)
        expected_score = (
            8 * 10**10 +
            9 * 10**8  # Highest card in straight flush
        )
        self.assertEqual(score, expected_score)

    # Add more tests for other hand types...

class TestBettingFunctionalities(unittest.TestCase):
    def setUp(self):
        self.game = TexasHoldem()
        self.player1 = Player("Alice", 1000)
        self.player2 = Player("Bob", 1000)
        self.game.players_in_hand = [self.player1, self.player2]
        self.game.pot = 0
        self.game.actions_full_round = []
        self.game.actions_this_betting_round = []
    
    def test_betting(self):
        # Player1 calls 10
        action1 = Action(self.player1, "call", 10)
        self.game.handle_action(action1)
        self.assertEqual(self.player1.chips, 990)
        self.assertEqual(self.player1.bet, 10)
        self.assertEqual(self.game.pot, 10)
        self.assertIn(action1, self.game.actions_full_round)
        self.assertIn(action1, self.game.actions_this_betting_round)
    
    def test_overbetting(self):
        # Player1 tries to raise 2000, which exceeds their chips
        action = Action(self.player1, "raise", 2000)
        with self.assertRaises(ValueError):
            self.game.handle_action(action)
    
    def test_folding(self):
        # Player2 folds
        action = Action(self.player2, "fold", 0)
        self.game.handle_action(action)
        self.assertNotIn(self.player2, self.game.players_in_hand)
        self.assertEqual(self.game.pot, 0)  # Assuming folding doesn't add to the pot
        self.assertIn(action, self.game.actions_full_round)
        self.assertIn(action, self.game.actions_this_betting_round)
    
    def test_checking(self):
        # Both players check
        action1 = Action(self.player1, "check", 0)
        action2 = Action(self.player2, "check", 0)
        self.game.handle_action(action1)
        self.game.handle_action(action2)
        self.assertEqual(self.player1.chips, 1000)
        self.assertEqual(self.player2.chips, 1000)
        self.assertEqual(self.game.pot, 0)
    
    def test_raise_and_call(self):
        # Player1 raises 50
        action1 = Action(self.player1, "raise", 50)
        self.game.handle_action(action1)
        self.assertEqual(self.player1.chips, 950)
        self.assertEqual(self.player1.bet, 50)
        self.assertEqual(self.game.pot, 50)
        
        # Player2 calls 50
        action2 = Action(self.player2, "call", 50)
        self.game.handle_action(action2)
        self.assertEqual(self.player2.chips, 950)
        self.assertEqual(self.player2.bet, 50)
        self.assertEqual(self.game.pot, 100)
    
    def test_multiple_bets(self):
        # Player1 raises 50
        action1 = Action(self.player1, "raise", 50)
        self.game.handle_action(action1)
        
        # Player2 raises 100
        action2 = Action(self.player2, "raise", 100)
        self.game.handle_action(action2)
        
        # Player1 calls 50
        action3 = Action(self.player1, "call", 50)
        self.game.handle_action(action3)
        
        self.assertEqual(self.player1.chips, 900)
        self.assertEqual(self.player1.bet, 100)
        self.assertEqual(self.player2.chips, 900)
        self.assertEqual(self.player2.bet, 100)
        self.assertEqual(self.game.pot, 200)


class TestFullGameFlow(unittest.TestCase):
    def setUp(self):
        # Initialize game with two players
        self.player1 = Player("Alice", 1000)
        self.player2 = Player("Bob", 1000)
        self.game = TexasHoldem(player_list=[self.player1, self.player2])
        self.game.players_in_hand = [self.player1, self.player2]
        self.game.pot = 0
        self.game.actions_full_round = []
        self.game.actions_this_betting_round = []
        self.game.deck.reset()
    
    def test_full_game_hand(self):
        # Pre-flop: Player1 calls 10, Player2 raises 20, Player1 calls 10
        action1 = Action(self.player1, "call", 10)
        self.game.handle_action(action1)
        action2 = Action(self.player2, "raise", 20)
        self.game.handle_action(action2)
        action3 = Action(self.player1, "call", 10)
        self.game.handle_action(action3)
        
        # Verify pot and player chips after betting
        self.assertEqual(self.game.pot, 40)
        self.assertEqual(self.player1.chips, 980)
        self.assertEqual(self.player2.chips, 980)
        self.assertEqual(self.player1.bet, 20)
        self.assertEqual(self.player2.bet, 20)
        
        # Simulate showdown with Player1 having a better hand
        self.game.community_cards = [
            Card('2', 'hearts'),
            Card('5', 'clubs'),
            Card('9', 'spades'),
            Card('J', 'hearts'),
            Card('K', 'diamonds')
        ]
        self.player1.hand = [
            Card('A', 'spades'),
            Card('K', 'clubs')  # Pair of Kings with Ace kicker
        ]
        self.player2.hand = [
            Card('Q', 'clubs'),
            Card('J', 'clubs')  # Pair of Jacks with Queen kicker
        ]
        
        winning_players = self.game.determine_winner()
        self.assertEqual(len(winning_players), 1)
        self.assertEqual(winning_players[0].name, "Alice")
        
        # Payout
        self.game.payout(winning_players)
        
        # Verify chips after payout
        self.assertEqual(self.player1.chips, 1020)  # 980 + 40
        self.assertEqual(self.player2.chips, 980)   # No change, as chips were already deducted
        self.assertEqual(self.game.pot, 0)


if __name__ == '__main__':
    unittest.main()
