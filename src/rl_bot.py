import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from player import Player
from action import Action
from action_validator import ActionValidator
from models.dueling_model import build_dueling_model
import config

class RLBot(Player):
    """Everything is normalized by BB and rewarded by Chip difference"""
    def __init__(self, name, chips, state_size, action_size):
        super().__init__(name, chips)
        
        
        self.state_size = config.STATE_SIZE #maybe just remove
        self.action_size = config.ACTION_SIZE #maybe just remove
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.memory_capacity = config.MEMORY_CAPACITY
        self.train_start = config.TRAIN_START
        self.train_freq = config.TRAIN_FREQ
        
        self.previous_raise_amount = None


        ##Load / Build model sec
        # self.model = self.build_continous_model()
        # self.target_model = self.build_continous_model()
        # self.update_target_model()
        self.model = build_dueling_model(self.state_size, self.action_size, self.learning_rate)
        self.target_model = build_dueling_model(self.state_size, self.action_size, self.learning_rate)
        self.update_target_model()

        #end sec

        self.hand_actions = []
        self.hand_strengths = []
        

        self.previous_state = None
        self.previous_action = None
        self.previous_chips = chips

        self.hand_counter = 0

    def build_continous_model(self):

        input_layer = layers.Input(shape=(self.state_size,))
        dense1 = layers.Dense(128, activation='relu')(input_layer)
        dense2 = layers.Dense(128, activation='relu')(dense1)

        # categorical
        action_output = layers.Dense(self.action_size, activation='linear', name='action_output')(dense2)

        # continous amount
        raise_output = layers.Dense(1, activation='linear', name='raise_output')(dense2)

        model = tf.keras.Model(inputs=input_layer, outputs=[action_output, raise_output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={'action_output': 'mse', 'raise_output': 'mse'}
        )
        return model


    # TODO: Add experimential depth & size. 
    # Continous might not be best way and you just might want to add some actions like below
        # Action Indices:
        # 0: Check
        # 1: Call
        # 2: Raise Min
        # 3: Raise Half Pot
        # 4: Raise Pot
        # 5: Raise All-in
        # 6: Fold
    # Or something like that.
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, raise_amount, reward, next_state, done):
        self.memory.append((state, action, raise_amount, reward, next_state, done))
        if raise_amount is None:
            raise_amount = 0
        self.memory.append((state, action, raise_amount, reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)
            
            
            
    def act(self, state, legal_actions, min_raise, max_raise):
        if np.random.rand() <= self.epsilon:
            # explore: select a random legal action
            action_idx = random.choice(legal_actions)
            if action_idx == self.action_types.index("raise"):
                raise_amount = random.uniform(min_raise, max_raise)
            else:
                raise_amount = None
        else:
            # exploit: select the best action based on Q-values
            action_values, raise_value = self.model.predict(np.array([state]))
            action_values = action_values[0]
            raise_value = raise_value[0][0]

            # mask illegal actions
            masked_q_values = np.full(self.action_size, -np.inf)
            masked_q_values[legal_actions] = action_values[legal_actions]
            action_idx = np.argmax(masked_q_values)

            # for raise action, use the predicted raise amount
            if action_idx == self.action_types.index("raise"):
                raise_amount = np.clip(raise_value, min_raise, max_raise)
            else:
                raise_amount = None
        return action_idx, raise_amount


    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states = np.array([sample[0] for sample in minibatch])
        actions = [sample[1] for sample in minibatch]
        raise_amounts = [sample[2] if sample[2] is not None else 0 for sample in minibatch]
        rewards = [sample[3] for sample in minibatch]
        next_states = np.array([sample[4] for sample in minibatch])
        dones = [sample[5] for sample in minibatch]

        target_action_values, target_raise_values = self.target_model.predict(next_states)
        target_action_values = target_action_values
        target_raise_values = target_raise_values.flatten()
        
        # predictions = self.target_model.predict(next_states)
  

        target_actions, target_raises = self.model.predict(states)
        target_raises = target_raises.flatten()

        for i in range(len(minibatch)):
            action_idx = actions[i]
            if dones[i]:
                target_actions[i][action_idx] = rewards[i]
                if action_idx == self.action_types.index("raise"):
                    target_raises[i] = raise_amounts[i]
            else:
                target_actions[i][action_idx] = rewards[i] + self.gamma * np.amax(target_action_values[i])
                if action_idx == self.action_types.index("raise"):
                    target_raises[i] = rewards[i] + self.gamma * target_raise_values[i]

        self.model.fit(
            states,
            {'action_output': target_actions, 'raise_output': target_raises},
            epochs=1,
            verbose=0
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_hand_actions(self):
        return self.hand_actions
    def get_hand_strength_at_action(self, action):
        try:
            idx = self.hand_actions.index(action)
            return self.hand_strengths[idx]
        except ValueError:
            return None

    def get_action(self, game_state, action_validator) -> Action:
        state = self.encode_state(game_state)
        legal_moves = self.get_legal_moves(game_state, action_validator)
        legal_actions = [idx for idx, is_legal in enumerate(legal_moves) if is_legal]
        min_raise, max_raise = action_validator.get_raise_range(game_state, self)

        action_idx, raise_amount = self.act(state, legal_actions, min_raise, max_raise)
        action = self.decode_action(action_idx, raise_amount, game_state, action_validator)

        self.hand_actions.append(action)
        current_hand_strength = self.estimate_hand_strength(game_state)
        self.hand_strengths.append(current_hand_strength)
    
    
        if self.previous_state is not None and self.previous_action is not None:
            reward = self.chips - self.previous_chips
            done = False
            self.remember(self.previous_state, 
                          self.previous_action, 
                          self.previous_raise_amount, 
                          reward, 
                          state, 
                          done)

        self.previous_state = state
        self.previous_action = action_idx
        self.previous_raise_amount = raise_amount 
        self.previous_chips = self.chips

        return action


    def end_hand(self, final_game_state):
        if self.previous_state is not None and self.previous_action is not None:
            next_state = self.encode_state(final_game_state)
            reward = self.chips - self.previous_chips  
            done = True
            self.remember(
                self.previous_state,
                self.previous_action,
                self.previous_raise_amount,
                reward,
                next_state,
                done
            )

        self.previous_state = None
        self.previous_action = None
        self.previous_raise_amount = None
        self.previous_chips = self.chips

        self.hand_counter += 1
        if self.hand_counter % self.train_freq == 0:
            self.replay()
            self.update_target_model()

               
        self.hand_actions = []
        self.hand_strengths = []


    def encode_state(self, game_state):
        state = []

        ## use big blind to normalize
        big_blind = game_state.big_blind  


        state.append(self.chips / big_blind)  

        state.append(game_state.pot_size / big_blind) 

        state.append(len(game_state.players_in_hand) / len(game_state.players))

       
        positions = ["sb", "bb", "utg", "hijack", "cut-off", "button"]
        position_vector = [0] * len(positions)
        player_position = game_state.position_map[self.name]
        if player_position in positions:
            position_index = positions.index(player_position)
            position_vector[position_index] = 1
        state.extend(position_vector)

        # Current btc
        call_amount = ActionValidator.get_call_amount(self, game_state)
        state.append(call_amount / big_blind)


        # Encode the last N actions
        # TODO: I think I want to do a weighing here, weighing the biggest threat action of some kind. 
        # Might be worth exploring
        # Also might want to want to do log-weighing across prior hands
        N = 5
        action_types = ["fold", "call", "raise", "check", "post"]
        action_vector = []
        recent_actions = game_state.actions_this_betting_round[-N:]
        for action in recent_actions:
            action_encoding = [0] * len(action_types)
            if action.action_type in action_types:
                action_index = action_types.index(action.action_type)
                action_encoding[action_index] = 1
            action_vector.extend(action_encoding)
            
        # pad zeros if fewer than N actions
        action_vector += [0] * ((N - len(recent_actions)) * len(action_types))
        state.extend(action_vector)

        # (one-hot encoding for each rank)
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                    '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        hand_rank_vector = [0] * 13  
        for card in self.hand:
            rank_index = rank_map[card.rank]
            hand_rank_vector[rank_index] = 1 
        state.extend(hand_rank_vector)


        #(one-hot encoding for each suit)
        suit_map = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}
        hand_suit_vector = [0] * 4 
        for card in self.hand:
            suit_index = suit_map[card.suit]
            hand_suit_vector[suit_index] = 1  
        state.extend(hand_suit_vector)


        community_cards = game_state.community_cards
        for i in range(5):
            if i < len(community_cards):
                card = community_cards[i]
              
                rank_vector = [0] * 13
                rank_index = rank_map[card.rank]
                rank_vector[rank_index] = 1
                state.extend(rank_vector)
                
                suit_vector = [0] * 4
                suit_index = suit_map[card.suit]
                suit_vector[suit_index] = 1
                state.extend(suit_vector)
            else:
                state.extend([0]*13)  
                state.extend([0]*4)   

  
       
        hand_strength = self.estimate_hand_strength(game_state)
        state.append(hand_strength)  # should be between 0 and 1

        return np.array(state)


        # #  counting the number of high cards (Jacks or higher)
        # high_ranks = ['J', 'Q', 'K', 'A']
        # high_cards = sum(1 for card in self.hand if card.rank in high_ranks)
        # state.append(high_cards / 2)  # Normalize to [0,1]
        # # legal_moves = self.get_legal_moves(game_state, ActionValidator())
        # # state.extend(legal_moves)  # Assuming legal_moves is a list of 0s and 1s

        # return np.array(state)

    def estimate_hand_strength(self, game_state):
        from itertools import combinations

        total_cards = self.hand + game_state.community_cards

        if len(total_cards) < 5:
            return self.evaluate_preflop_strength()
        
        possible_hands = combinations(total_cards, 5)

        best_hand_rank = 0  

        for hand in possible_hands:
          
            hand_rank = self.evaluate_hand_rank(hand)
            if hand_rank > best_hand_rank:
                best_hand_rank = hand_rank

        # Normalize hand strength between 0 and 1
        # 9 = sf
        normalized_strength = best_hand_rank / 9.0

        return normalized_strength


    #TODO: Make this use use game class's method of evaluating
    def evaluate_hand_rank(self, hand):
        # hand is a list of 5 Card objects
        ranks = [card.rank for card in hand]
        suits = [card.suit for card in hand]

        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        rank_numbers = [rank_map[rank] for rank in ranks]
        rank_counts = {}
        for rank in rank_numbers:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        is_flush = len(set(suits)) == 1

        sorted_ranks = sorted(set(rank_numbers))
        is_straight = False
        if len(sorted_ranks) == 5:
            if sorted_ranks[-1] - sorted_ranks[0] == 4:
                is_straight = True
            elif sorted_ranks == [2, 3, 4, 5, 14]:
                is_straight = True

        if is_straight and is_flush:
            return 9  

        if 4 in rank_counts.values():
            return 8 
        
        if sorted(rank_counts.values()) == [2, 3]:
            return 7  

        if is_flush:
            return 6  

        if is_straight:
            return 5  

        if 3 in rank_counts.values():
            return 4 
        
        
        if list(rank_counts.values()).count(2) == 2:
            return 3
        
        
        if 2 in rank_counts.values():
            return 2  
        
        return 1  


    def evaluate_preflop_strength(self):
        
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        card1, card2 = self.hand
        rank1 = rank_map[card1.rank]
        rank2 = rank_map[card2.rank]

        is_pair = rank1 == rank2

        is_suited = card1.suit == card2.suit

        rank_diff = abs(rank1 - rank2)

        high_cards = sum(1 for rank in [rank1, rank2] if rank >= 11)

        strength = 0.1
        
        if is_pair:
            strength += 0.4

        if high_cards == 2:
            strength += 0.3
        elif high_cards == 1:
            strength += 0.2

        if is_suited:
            strength += 0.1

        if rank_diff == 1:
            strength += 0.1 
        elif rank_diff == 0:
            strength += 0.0  
            

        # strength should be between 0 and 1
        strength = min(strength, 1.0)

        return strength

    def decode_action(self, action_idx, raise_amount, game_state, action_validator):
        action_type = self.action_types[action_idx]
        if action_type == "call":
            amount = action_validator.get_call_amount(self, game_state)
            return Action(self, "call", amount)
        elif action_type == "raise":
            amount = raise_amount
            return Action(self, "raise", amount)
        elif action_type == "check":
            return Action(self, "check")
        else:
            return Action(self, "fold")

        
        
    def save_model(self, filepath):
        self.model.save(filepath)

    #don't compile model while loading
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath, compile=False)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={'action_output': 'mean_squared_error', 'raise_output': 'mean_squared_error'}
        )

        self.update_target_model()
        print(f"Model loaded and recompiled from {filepath}")