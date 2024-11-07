# Poker Bot Simulation with Reinforcement Learning

Side project created to act as a playground for RL and other poker strategies. 


## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running Simulations](#running-simulations)
  - [Training the RL Bot](#training-the-rl-bot)
- [Customization](#customization)

---

## Project Overview

This project aims to create an environment where different poker-playing bots can compete against each other in Texas Hold'em:

- **Game Logic**: Manages the flow of the poker game, including dealing cards, managing bets, and determining winners.
- **Bots**: Different AI strategies implemented as bots, such as random playing bots, strategic bots, and an RL bot.
- **Reinforcement Learning Bot**: An RL agent that uses a Deep Q-Network (DQN) to learn optimal strategies over time.
- **Simulation**: Runs multiple iterations of games to observe bot performance and train the RL bot.

---


## BB RL Agent playing hands against BB bots
<img src="imgs\Figure_1.png" alt="rla_chips" width="750"/>

## Features

- **Reinforcement Learning Integration**: The RL bot learns from experiences using TensorFlow.
- **Customizable Parameters**: Adjust game settings, bot strategies, and training parameters.
- **Statistical Analysis**: Collect and display statistics on bot performance over simulations.
- **Model Saving and Loading**: Save trained models for future use and continue training from saved states.

---

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/effpeeel/poker-bot-simulation.git
   cd poker-bot-simulation
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```


---

## Dependencies

The project requires the following Python packages:

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook (for `model_builder.ipynb`)
- Additional standard libraries (`random`, `itertools`, etc.)

Install the dependencies using pip:

```bash
pip install tensorflow numpy matplotlib jupyter
```

---

## Project Structure

The project directory contains the following files and directories:

- **`action.py`**: Defines the `Action` class representing player actions.
- **`action_validator.py`**: Contains the `ActionValidator` class for validating actions.
- **`bots/`**: Directory containing different bot implementations:
  - **`random_bot.py`**: A bot that makes random decisions.
  - **`strategic_bot.py`**: A bot that uses a predefined strategy.
- **`deck.py`**: Defines the `Deck` class for managing the deck of cards.
- **`game.py`**: Contains the `TexasHoldem` class that manages the game logic.
- **`game_state.py`**: Defines the `GameState` class for representing the current state of the game.
- **`player.py`**: Defines the `Player` class and serves as a base class for all bots.
- **`rl_bot.py`**: Contains the `RLBot` class, which implements the reinforcement learning agent.
- **`simulation.py`**: Script to run simulations of games between bots.
- **`model_builder.ipynb`**: Jupyter Notebook for building and training the RL model.
- **`README.md`**: Project documentation (this file).
- **`requirements.txt`**: Lists project dependencies.

---

## Usage

### Running Simulations

To run a simulation of the poker game with the bots, execute the `simulation.py` script:

```bash
python simulation.py
```

**Example `simulation.py` Content**:

```python
from game import TexasHoldem
from rl_bot import RLBot
import bots

if __name__ == "__main__":
    # Standard for this game
    STATE_SIZE = 138  
    ACTION_SIZE = 4   # ["check", "call", "raise", "fold"]

    players = [
        bots.RandomBot("RandomBot1", 1000),
        bots.StrategicBot("StrategicBot", 1000, aggression_level=0.5, tightness_level=0.5, bluff_frequency=0.2),
        RLBot("RLBot", 1000, state_size=STATE_SIZE, action_size=ACTION_SIZE),
    ]

    game = TexasHoldem(small_blind=5, big_blind=10, player_list=players)

    for episode in range(100):  # # of hands to play
        game.play()
        print(f"Game {episode + 1} finished")
```

### Training the RL Bot

To train the RL bot using the provided Jupyter Notebook:

1. **Open the Notebook**

   ```bash
   jupyter notebook model_builder.ipynb
   ```

2. **Run the Cells**

   - The notebook guides you through setting up the environment, defining the RL agent, and training it.
   - Adjust parameters as needed (e.g., number of episodes, learning rate).

3. **Save the Trained Model**

   - After training, the model is saved using `rl_agent.save_model('rl_agent_model.h5')`.
   - You can load this model later using `rl_agent.load_model('rl_agent_model.h5')`.

---

## Customization

You can customize various aspects of the simulation and bots:

- **Game Settings**

  - Adjust the small and big blind amounts.
  - Change the number of players.

- **Bot Strategies**

  - Modify existing bots or create new ones by extending the `Player` class.
  - Adjust parameters like aggression level, tightness level, and bluff frequency for pre made `StrategicBot`.

- **RL Bot Parameters**

  - Modify neural network architecture in `RLBot.build_model()`.
  - Adjust training parameters like learning rate, epsilon decay, and batch size.
  - Enhance the `encode_state` method to include more features.

---
