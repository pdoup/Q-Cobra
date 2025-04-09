# Q-Cobra

## Advanced DQN Agent for Snake 🐍
### A Deep Reinforcement Learning Implementation

## Overview 📜

This repository contains the implementation of an intelligent agent designed to master the classic game of Snake. Utilizing advanced Deep Reinforcement Learning (DRL) techniques based on Deep Q-Networks (DQN), this project aims to develop a high-performing agent.

## Key Features ✨

* **Classic Snake Environment:** 🎮 A clean implementation of the Snake game logic.
* **Deep Q-Network (DQN) Foundation:** 🧠 Utilizes neural networks to approximate the action-value function (Q-function).
* **Advanced DQN Enhancements:** Incorporates modern techniques for improved performance and stability:
    * ✨ **Dueling Network Architecture:** Separates the estimation of state value and action advantages.
    * ✨ **Prioritized Experience Replay (PER):** Focuses learning on more significant experiences by replaying transitions with higher temporal-difference (TD) errors more frequently.
    * ✨ **Multi-step Learning:** Improves credit assignment by calculating Q-value targets based on rewards accumulated over multiple future steps.
    * ✨ **Double DQN:** Mitigates the overestimation bias common in standard Q-learning by decoupling action selection and evaluation.
* **Modular Code Structure:** 🏗️ Well-organized Python modules for different components of the project.
* **Training Utilities:** Includes logging 📈 and plotting 📊 functionalities to monitor training progress.
* **Configuration Management:** ⚙️ Training parameters managed via a YAML configuration file.

## Technical Concepts 🔬

This agent builds upon the foundational Deep Q-Network algorithm with several key enhancements:

1.  **Deep Q-Networks (DQN):** At its core, the agent uses a deep neural network to estimate the expected cumulative discounted reward (Q-value) for taking each possible action in a given state. It learns by minimizing the difference between predicted Q-values and target Q-values derived from interactions stored in a replay buffer.

2.  **Double DQN:** To combat the tendency of standard DQN to overestimate Q-values, this technique uses the main network to select the *best* action for the next state, but uses the separate *target network* to evaluate the Q-value of that selected action when calculating the learning target.

3.  **Dueling DQN Architecture:** Instead of directly outputting Q-values, the network head splits into two streams: one estimating the *state value* V(s) (how good it is to be in a state) and the other estimating the *advantage* A(s, a) for each action (how much better taking action 'a' is compared to other actions in state 's'). These are combined (Q = V + (A - mean(A))) to form the final Q-values, often leading to better policy evaluation.

4.  **Prioritized Experience Replay (PER):** This technique improves learning efficiency by sampling transitions from the replay buffer non-uniformly. Transitions that lead to a large prediction error (TD error) are considered more "surprising" or informative and are thus sampled more often. To correct for the bias introduced by this non-uniform sampling, updates are weighted using Importance Sampling (IS) weights.

5.  **Multi-step Learning:** Rather than relying solely on the immediate reward and the estimated value of the very next state (1-step return), multi-step learning calculates target values using the discounted sum of rewards over 'n' future steps, plus the discounted estimated value of the state reached after those 'n' steps. This can accelerate the propagation of reward information through the agent's value estimates.

## Installation 🛠️

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/pdoup/Q-Cobra.git
    cd Q-Cobra
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:** Create a `requirements.txt` file with the following content:

    ```txt
    # requirements.txt
    torch
    numpy
    tqdm
    matplotlib
    PyYAML
    ```

    Then, install the requirements:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

The primary interaction with the project is through the `main.py` script for training and evaluation, and `play_game.py` for visual playback.

### Configuration

Training and evaluation behaviour is controlled by a YAML configuration file (default: `config.yaml`). Create a `config.yaml` file in the root directory with parameters like these:

```yaml
# config.yaml (Example)

# Training parameters
episodes: 2000         # Maximum number of training episodes
patience: 300          # Episodes to wait for improvement before early stopping
save_path: "snake_dqn_advanced_best.pth" # Filename for the best saved model (saved in ./models/)
plot_filename: "training_progress_advanced.png" # Filename for the training plot (saved in ./plots/)

# Evaluation parameters (run after training or if load_model_path is set)
evaluation_games: 50   # Number of games to run for evaluation

# Model Loading (Optional)
# Set this to skip training and only evaluate/use a pre-trained model
load_model_path: null  # Example: "models/snake_dqn_advanced_best.pth"
```

### Training

To start a new training session using the parameters defined in `config.yaml`:

```bash
python main.py
```

Alternatively, specify a different configuration file:
```bash
python main.py --config path/to/your_custom_config.yaml
```

### During Training:

* Progress will be displayed in the terminal via `tqdm`.
* Logs will be written to `snake_dqn_training.log` and printed to the console.
* The model with the best average score (over the last 100 episodes) will be saved to the `models/` directory (e.g., `models/snake_dqn_advanced_best.pth`).
* Training will stop early if no improvement is seen for the specified `patience` number of episodes.

### After Training:

* A plot visualizing training progress (scores, average scores, loss) will be saved to the `plots/` directory (e.g., `plots/training_progress_advanced.png`).
* An evaluation phase will automatically run using the best saved model (or the model specified in `load_model_path`).

### Evaluation Only

To evaluate a pre-trained model without running training:
1. Modify your `config.yaml` file, setting `load_model_path` to the path of the desired model file (e.g., `load_model_path: "models/snake_dqn_advanced_best.pth"`).
2. Run the main script:
   ```bash
   python main.py # Or python main.py --config your_eval_config.yaml
   ```
 The script will skip training, load the specified model, and run the evaluation phase, printing the results.

 ### Playing (Visual Playback) ▶️

 To watch a trained agent play the game in your terminal:
 ```bash
python play_game.py [OPTIONS]
```

Common Options:

* `--model-path`: Specify the path to the model file to load.
    * Example: `python play_game.py --model-path models/snake_dqn_advanced_best.pth` (Default: `snake_dqn_best.pth` - _Note: You might need to adjust this default or the filename based on your training output)_
* `--num-games`: Set the number of games to play sequentially.
    * Example: `python play_game.py --num-games 3` (Default: 5)
* `--delay`: Adjust the rendering speed (delay between steps in seconds). Lower is faster.
    * Example: `python play_game.py --delay 0.05` (Default: 0.01)

The script will render the game board in the terminal and print statistics after the requested number of games are completed.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
