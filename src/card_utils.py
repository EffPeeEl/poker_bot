import random



class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit


    def __str__(self):
        return self.rank + self.suit[0]
    def __repr__(self):
        return self.__str__()

class Deck:
    def __init__(self):
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.suits = ['hearts', 'diamonds', 'clubs', 'spades']
        self.reset()
        
    
    def reset(self):
        self.cards = [Card(rank, suit) for rank in self.ranks for suit in self.suits]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()

    def deal_hand(self, num_cards):
        return [self.deal_card() for _ in range(num_cards)]

    def remove_cards(self, hand):
        for x in hand:
            self.cards.remove(x)
