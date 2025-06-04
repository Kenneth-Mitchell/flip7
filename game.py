import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

class Flip7CumulativeEnv:
    def __init__(self, target_score=200, seed=None):
        self.target_score = target_score
        self.total_score = 0
        self.steps = 0
        self.rounds = 0
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.reset_round()

    def _build_deck(self):
        self.deck = []
        for value in range(1, 13):  # 1 to 12
            self.deck.extend([value] * value)
        self.deck.append(0)  # one 0 card
        random.shuffle(self.deck)

    def reset_round(self):
        self._build_deck()
        self.flipped = []
        self.round_score = 0
        self.round_done = False
        self.busted = False

    def reset(self):
        self.total_score = 0
        self.steps = 0
        self.rounds = 0
        self.reset_round()
        return self._get_obs()

    def _get_obs(self):
        return {
            "flipped": list(self.flipped),
            "round_score": self.round_score,
            "total_score": self.total_score,
            "steps": self.steps,
            "rounds": self.rounds,
            "cards_remaining": len(self.deck),
            "round_done": self.round_done,
            "busted": self.busted,
        }

    def step(self, action):
        if self.total_score >= self.target_score:
            return self._get_obs(), 0, True, {}

        self.steps += 1

        if self.round_done:
            self.reset_round()

        if action == 0:  # stay
            self.round_score = sum(self.flipped)
            self.total_score += self.round_score
            self.round_done = True
            self.rounds += 1
            reward = self.round_score
            done = self.total_score >= self.target_score
            return self._get_obs(), reward, done, {}

        elif action == 1:  # hit
            if not self.deck:
                self._build_deck()

            card = self.deck.pop()

            if card in self.flipped:
                self.busted = True
                self.round_score = 0
                self.round_done = True
                self.rounds += 1
                return self._get_obs(), 0, False, {}
            else:
                self.flipped.append(card)
                if len(self.flipped) == 7:
                    self.round_score = sum(self.flipped) + 15
                    self.total_score += self.round_score
                    self.round_done = True
                    self.rounds += 1
                    reward = self.round_score
                    done = self.total_score >= self.target_score
                    return self._get_obs(), reward, done, {}
                else:
                    return self._get_obs(), 0, False, {}

        else:
            raise ValueError("Action must be 0 (stay) or 1 (hit)")

    def render(self):
        print(f"Flipped: {self.flipped} | Round Score: {self.round_score} | "
              f"Total Score: {self.total_score} | Steps: {self.steps} | Round Done: {self.round_done}")


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.995, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Larger memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        # Convert to numpy arrays first to avoid the warning
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select action, target network to evaluate
        next_actions = self.q_network(next_states).max(1)[1]
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.huber_loss(current_q_values.squeeze(), target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def calculate_bust_probability(flipped_cards, cards_seen_this_round=None):
    """Calculate probability of busting if we hit"""
    if len(flipped_cards) == 0:
        return 0.0
    
    # Count cards that would cause a bust
    bust_count = 0
    total_remaining = 0
    
    # Count each card type in the full deck
    card_counts = {0: 1}  # One zero card
    for i in range(1, 13):
        card_counts[i] = i
    
    # Subtract cards already flipped
    for card in flipped_cards:
        if card in card_counts:
            card_counts[card] -= 1
    
    # Count remaining cards and bust cards
    for card, count in card_counts.items():
        total_remaining += max(0, count)
        if card in flipped_cards:
            bust_count += max(0, count)
    
    if total_remaining <= 0:
        return 1.0
    
    return bust_count / total_remaining


def calculate_expected_value(flipped_cards):
    """Calculate expected value of hitting"""
    if len(flipped_cards) >= 7:
        return 0
    
    bust_prob = calculate_bust_probability(flipped_cards)
    
    # If we would bust, expected value is 0
    if bust_prob >= 1.0:
        return 0
    
    # Simple heuristic: higher value for safer plays
    current_sum = sum(flipped_cards)
    cards_needed = 7 - len(flipped_cards)
    
    if cards_needed == 1:
        return (current_sum + 15) * (1 - bust_prob)  # Perfect flip7 bonus
    
    # Estimate value based on current sum and safety
    safety_factor = 1 - bust_prob
    progress_value = current_sum * 0.1  # Small reward for progress
    
    return progress_value * safety_factor


def state_to_vector(obs):
    """Enhanced state representation"""
    flipped_cards = obs["flipped"]
    
    # Binary representation of flipped cards (0-12)
    flipped_vector = [0] * 13
    for card in flipped_cards:
        flipped_vector[card] = 1
    
    # Card count features
    card_counts = [0] * 13
    for card in flipped_cards:
        card_counts[card] += 1
    
    # Game state features
    num_flipped = len(flipped_cards)
    current_sum = sum(flipped_cards)
    bust_prob = calculate_bust_probability(flipped_cards)
    expected_value = calculate_expected_value(flipped_cards)
    
    # Progress features
    progress_features = [
        num_flipped / 7.0,  # How close to 7 cards
        current_sum / 50.0,  # Normalized current sum (rough max ~50)
        obs["total_score"] / obs.get("target_score", 200),  # Progress to goal
        obs["rounds"] / 100.0,  # Normalized rounds
        bust_prob,  # Risk assessment
        expected_value / 10.0,  # Normalized expected value
        float(obs["round_done"]),
        float(obs["busted"]),
    ]
    
    # Combine all features
    all_features = flipped_vector + progress_features
    return np.array(all_features, dtype=np.float32)


def calculate_reward(obs, action, next_obs, done):
    """Reward function focused on minimizing rounds to reach exactly 200"""
    
    # Game completion reward
    if done and next_obs["total_score"] >= 200:
        # Base reward for completion
        base_reward = 100
        
        # Penalty for taking too many rounds (linear penalty)
        rounds_penalty = next_obs["rounds"] * 2
        
        # Penalty for overshooting the target
        overshoot = max(0, next_obs["total_score"] - 200)
        overshoot_penalty = overshoot * 0.5
        
        total_reward = base_reward - rounds_penalty - overshoot_penalty
        return total_reward
    
    # Intermediate rewards during gameplay
    if next_obs["busted"]:
        # Penalty for busting (wastes a round)
        return -5
    
    elif next_obs["round_done"] and not next_obs["busted"]:
        # Reward for successful round, scaled by how much it helps reach 200
        remaining_to_target = max(0, 200 - obs["total_score"])
        round_contribution = min(next_obs["round_score"], remaining_to_target)
        
        # Higher reward for rounds that get us closer to exactly 200
        if remaining_to_target > 0:
            efficiency = round_contribution / remaining_to_target
            return efficiency * 10
        else:
            # We're already at/past target, small penalty for continuing
            return -1
    
    # Small penalty for each action to encourage efficiency
    return -0.05


def train_agent(episodes=10000, target_score=200):
    env = Flip7CumulativeEnv(target_score=target_score)
    state_size = 21  # Enhanced features
    action_size = 2
    
    agent = DQNAgent(state_size, action_size, lr=0.0005, gamma=0.95, epsilon_min=0.05)  # Lower lr, gamma, higher min epsilon
    scores = []
    rounds_history = []
    total_scores = []  # Track actual game scores
    
    for episode in range(episodes):
        obs = env.reset()
        state = state_to_vector(obs)
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_obs, env_reward, done, _ = env.step(action)
            next_state = state_to_vector(next_obs)
            
            # Use enhanced reward function
            reward = calculate_reward(obs, action, next_obs, done)
            
            agent.remember(state, action, reward, next_state, done)
            
            obs = next_obs
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Train more frequently with larger batches
        if len(agent.memory) > 2000:  # Start training later
            for _ in range(4):  # Multiple training steps per episode
                agent.replay(batch_size=32)  # Smaller batch size
        
        # Update target network more frequently for this simpler reward structure
        if episode % 50 == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        rounds_history.append(next_obs["rounds"])
        total_scores.append(next_obs["total_score"])  # Track final scores
        
        if episode % 1000 == 0:
            avg_rounds = np.mean(rounds_history[-100:]) if len(rounds_history) >= 100 else np.mean(rounds_history)
            avg_total_score = np.mean(total_scores[-100:]) if len(total_scores) >= 100 else np.mean(total_scores)
            avg_reward = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Avg Rounds: {avg_rounds:.2f}, Avg Final Score: {avg_total_score:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, rounds_history, total_scores


def test_agent(agent, episodes=100, target_score=200):
    env = Flip7CumulativeEnv(target_score=target_score)
    agent.epsilon = 0  # No exploration during testing
    
    rounds_taken = []
    
    for episode in range(episodes):
        obs = env.reset()
        state = state_to_vector(obs)
        
        while True:
            action = agent.act(state)
            obs, reward, done, _ = env.step(action)
            state = state_to_vector(obs)
            
            if done:
                rounds_taken.append(obs["rounds"])
                break
    
    return rounds_taken


def compare_strategies(episodes=1000, target_score=200):
    """Compare RL agent vs enhanced heuristic strategies"""
    
    # Test enhanced heuristic based on bust probability
    env = Flip7CumulativeEnv(target_score=target_score)
    heuristic_rounds = []
    
    for _ in range(episodes):
        obs = env.reset()
        while True:
            if obs["round_done"]:
                action = 1  # hit to start new round
            else:
                bust_prob = calculate_bust_probability(obs["flipped"])
                # Dynamic threshold based on current sum and cards flipped
                if len(obs["flipped"]) <= 2:
                    action = 1  # Always hit with 2 or fewer cards
                elif len(obs["flipped"]) >= 6:
                    action = 0  # Stay when close to 7
                elif bust_prob < 0.3:
                    action = 1  # Hit if low bust probability
                elif sum(obs["flipped"]) >= 25:
                    action = 0  # Stay with decent sum
                else:
                    action = 1  # Default to hit
            
            obs, _, done, _ = env.step(action)
            if done:
                heuristic_rounds.append(obs["rounds"])
                break
    
    print(f"Enhanced Heuristic - Avg Rounds: {np.mean(heuristic_rounds):.2f} ± {np.std(heuristic_rounds):.2f}")
    return heuristic_rounds


if __name__ == "__main__":
    print("Training Enhanced DQN Agent for Flip7...")
    
    # Train the agent
    agent, scores, rounds_history, total_scores = train_agent(episodes=10000)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_rounds = test_agent(agent, episodes=1000)
    
    # Test final scores to verify they're close to 200
    env = Flip7CumulativeEnv(target_score=200)
    agent.epsilon = 0
    test_final_scores = []
    
    for _ in range(100):
        obs = env.reset()
        while True:
            state = state_to_vector(obs)
            action = agent.act(state)
            obs, _, done, _ = env.step(action)
            if done:
                test_final_scores.append(obs["total_score"])
                break
    
    print(f"Trained Agent - Avg Rounds: {np.mean(test_rounds):.2f} ± {np.std(test_rounds):.2f}")
    print(f"Trained Agent - Avg Final Score: {np.mean(test_final_scores):.2f} ± {np.std(test_final_scores):.2f}")
    
    # Compare with heuristic
    print("\nComparing with enhanced heuristic...")
    heuristic_rounds = compare_strategies()
    
    # Plot results
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(np.convolve(rounds_history, np.ones(100)/100, mode='valid'))
    plt.title('Training Progress: Rounds to Complete')
    plt.xlabel('Episode')
    plt.ylabel('Rounds (100-episode avg)')
    
    plt.subplot(1, 4, 2)
    plt.plot(np.convolve(total_scores, np.ones(100)/100, mode='valid'))
    plt.title('Training Progress: Final Scores')
    plt.xlabel('Episode')
    plt.ylabel('Final Score (100-episode avg)')
    plt.axhline(y=200, color='r', linestyle='--', label='Target Score')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.hist(test_rounds, bins=20, alpha=0.7, label='RL Agent')
    plt.hist(heuristic_rounds, bins=20, alpha=0.7, label='Enhanced Heuristic')
    plt.title('Rounds Distribution')
    plt.xlabel('Rounds to Complete')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 4, 4)
    plt.hist(test_final_scores, bins=20, alpha=0.7)
    plt.title('Final Scores Distribution (RL Agent)')
    plt.xlabel('Final Score')
    plt.ylabel('Frequency')
    plt.axvline(x=200, color='r', linestyle='--', label='Target Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'enhanced_flip7_dqn_model.pth')
    print("\nModel saved as 'enhanced_flip7_dqn_model.pth'")
    
    # Demo a game with the trained agent
    print("\nDemo game with trained agent:")
    env = Flip7CumulativeEnv(target_score=200)
    obs = env.reset()
    agent.epsilon = 0  # No exploration
    
    round_num = 1
    print(f"\n=== Round {round_num} ===")
    
    while True:
        if not obs['round_done']:
            state = state_to_vector(obs)
            action = agent.act(state)
            action_name = "stay" if action == 0 else "hit"
            
            bust_prob = calculate_bust_probability(obs['flipped'])
            print(f"Flipped: {obs['flipped']}, Sum: {sum(obs['flipped'])}, Bust Prob: {bust_prob:.2f}, Action: {action_name}")
            
            obs, reward, done, _ = env.step(action)
            
            if obs['round_done']:
                if obs['busted']:
                    print(f"  -> BUSTED! Round score: 0")
                else:
                    print(f"  -> Round score: {obs['round_score']}")
                
                if done:
                    break
                else:
                    round_num += 1
                    print(f"\n=== Round {round_num} ===")
        else:
            obs, reward, done, _ = env.step(1)  # Hit to start new round
            if done:
                break
    
    print(f"\nGame completed in {obs['rounds']} rounds with total score {obs['total_score']}")
    print(f"Target was 200, achieved {obs['total_score']} (overshoot: {obs['total_score'] - 200})")