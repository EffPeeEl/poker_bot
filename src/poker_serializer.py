import logging
from card_utils import Card
from action import Action
from game_state import GameState
from player import Player

class PokerSerializer:
    @staticmethod
    def card_to_dict(card: Card) -> dict:
        return {"rank": card.rank, "suit": card.suit}

    @staticmethod
    def card_from_dict(card: dict) -> Card:
        return Card(card["rank"], card["suit"])

    @staticmethod
    def action_to_dict(action: Action) -> dict:
        return {
            "player": str(action.player),
            "action_type": action.action_type,
            "amount": action.amount
        }

    @staticmethod
    def action_from_dict(action: dict) -> Action:
        return Action(action["player"], action["action_type"], action["amount"])

    @staticmethod
    def player_to_dict(player: Player) -> dict:
        return {
            "name": player.name,
            "chips": player.chips,
            "hand": [PokerSerializer.card_to_dict(card) for card in player.hand],
            "bet": player.bet
        }

    @staticmethod
    def player_from_dict(player: dict) -> Player:
        dict_player = Player(player["name"], player["chips"])
        dict_player.hand = [PokerSerializer.card_from_dict(card) for card in player["hand"]]
        dict_player.bet = player["bet"]
        return dict_player

    @staticmethod
    def game_state_to_dict(game_state: GameState) -> dict:
        try:
            game_dict = {
                'players': [PokerSerializer.player_to_dict(player) for player in game_state.players],
                'community_cards': [PokerSerializer.card_to_dict(card) for card in game_state.community_cards],
                'pot_size': float(game_state.pot_size),
                'actions_full_round': [PokerSerializer.action_to_dict(action) for action in game_state.actions_full_round],
                'actions_this_betting_round': [PokerSerializer.action_to_dict(action) for action in game_state.actions_this_betting_round],
                'big_blind': game_state.big_blind,
                'winners': [winner.name for winner in game_state.winners]
            }
            return game_dict
        except Exception as e:
            logging.error(f"Error serializing GameState: {e}")
            return {}

    @staticmethod
    def game_state_from_dict(game_dict: dict) -> GameState:
        players = [PokerSerializer.player_from_dict(player_dict) for player_dict in game_dict["players"]]
        community_cards = [PokerSerializer.card_from_dict(card_dict) for card_dict in game_dict["community_cards"]]
        actions_full_round = [PokerSerializer.action_from_dict(action_dict) for action_dict in game_dict["actions_full_round"]]
        actions_this_betting_round = [PokerSerializer.action_from_dict(action_dict) for action_dict in game_dict["actions_this_betting_round"]]
        winners = [next(player for player in players if player.name == winner_name) for winner_name in game_dict["winners"]]

        return GameState(
            community_cards,
            game_dict["pot_size"],
            actions_full_round,
            players,
            [player for player in players if player in game_dict["players_in_hand"]],
            actions_this_betting_round,
            game_dict["big_blind"],
            winners
        )

    @staticmethod
    def game_state_to_json(game_state: GameState) -> str:
        import json
        return json.dumps(PokerSerializer.game_state_to_dict(game_state), indent=4)

    @staticmethod
    def game_state_from_json(json_string: str) -> GameState:
        import json
        game_dict = json.loads(json_string)
        return PokerSerializer.game_state_from_dict(game_dict)