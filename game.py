import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

class MultiAgentFlip7Env:
    def __init__(self, num_agents=5, target_score=200, seed=None):
        self.num_agents = num_agents
        self.target_score = target_score
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Game state
        self.agent_scores = [0] * num_agents
        self.agent_round_scores = [0] * num_agents
        self.agent_flipped = [[] for _ in range(num_agents)]
        self.agent_busted = [False] * num_agents
        self.agent_round_done = [False] * num_agents
        
        self.current_agent = 0
        self.game_steps = 0
        self.round_number = 0
        self.final_round = False
        self.game_over = False
        
        # Shared deck for all agents in a round
        self.deck = []
        self.reset()

    def _build_deck(self):
        self.deck = []
        for value in range(1, 13):  # 1 to 12
            self.deck.extend([value] * value)
        self.deck.append(0)  # one 0 card
        random.shuffle(self.deck)

    def reset(self):
        self.agent_scores = [0] * self.num_agents
        self.agent_round_scores = [0] * self.num_agents
        self.agent_flipped = [[] for _ in range(self.num_agents)]
        self.agent_busted = [False] * self.num_agents
        self.agent_round_done = [False] * self.num_agents
        
        self.current_agent = 0
        self.game_steps = 0
        self.round_number = 0
        self.final_round = False
        self.game_over = False
        
        self._build_deck()
        return self._get_obs_all()

    def _get_obs(self, agent_id):
        """Get observation for a specific agent"""
        return {
            "agent_id": agent_id,
            "flipped": list(self.agent_flipped[agent_id]),
            "round_score": self.agent_round_scores[agent_id],
            "total_score": self.agent_scores[agent_id],
            "round_done": self.agent_round_done[agent_id],
            "busted": self.agent_busted[agent_id],
            "cards_remaining": len(self.deck),
            "current_agent": self.current_agent,
            "round_number": self.round_number,
            "final_round": self.final_round,
            "game_over": self.game_over,
            # Opponent information
            "opponent_scores": [self.agent_scores[i] for i in range(self.num_agents) if i != agent_id],
            "opponent_round_scores": [self.agent_round_scores[i] for i in range(self.num_agents) if i != agent_id],
            "max_opponent_score": max([self.agent_scores[i] for i in range(self.num_agents) if i != agent_id]) if self.num_agents > 1 else 0,
            "my_rank": sorted(self.agent_scores, reverse=True).index(self.agent_scores[agent_id]) + 1,
            "agents_at_target": sum(1 for score in self.agent_scores if score >= self.target_score)
        }

    def _get_obs_all(self):
        """Get observations for all agents"""
        return [self._get_obs(i) for i in range(self.num_agents)]

    def _start_new_round(self):
        """Start a new round for all agents"""
        self.round_number += 1
        self._build_deck()
        
        # Reset round state for all agents
        self.agent_flipped = [[] for _ in range(self.num_agents)]
        self.agent_round_scores = [0] * self.num_agents
        self.agent_busted = [False] * self.num_agents
        self.agent_round_done = [False] * self.num_agents
        self.current_agent = 0

    def step(self, actions):
        """
        Step function that takes actions for all agents
        actions: list of actions for each agent (0=stay, 1=hit)
        """
        if self.game_over:
            return self._get_obs_all(), [0] * self.num_agents, True, {}

        # Check if we need to start a new round
        if all(self.agent_round_done):
            # Add scores from completed round
            for i in range(self.num_agents):
                if not self.agent_busted[i]:
                    self.agent_scores[i] += self.agent_round_scores[i]
            
            # Check if someone reached target score (trigger final round)
            if not self.final_round and any(score >= self.target_score for score in self.agent_scores):
                self.final_round = True
            
            # If we just finished the final round, end the game
            if self.final_round:
                self.game_over = True
                return self._get_obs_all(), self._calculate_final_rewards(), True, {}
            
            self._start_new_round()

        rewards = [0] * self.num_agents
        
        # Process actions for each agent
        for agent_id in range(self.num_agents):
            if not self.agent_round_done[agent_id]:
                reward = self._process_agent_action(agent_id, actions[agent_id])
                rewards[agent_id] = reward

        self.game_steps += 1
        
        # Check if game should end
        done = self.game_over or (self.final_round and all(self.agent_round_done))
        if done and not self.game_over:
            self.game_over = True
            final_rewards = self._calculate_final_rewards()
            rewards = [r + fr for r, fr in zip(rewards, final_rewards)]
        
        return self._get_obs_all(), rewards, done, {}

    def _process_agent_action(self, agent_id, action):
        """Process action for a single agent"""
        if action == 0:  # stay
            self.agent_round_scores[agent_id] = sum(self.agent_flipped[agent_id])
            self.agent_round_done[agent_id] = True
            return self.agent_round_scores[agent_id] * 0.1  # Small reward for scoring
            
        elif action == 1:  # hit
            if not self.deck:
                self._build_deck()

            card = self.deck.pop()

            if card in self.agent_flipped[agent_id]:
                # Bust
                self.agent_busted[agent_id] = True
                self.agent_round_scores[agent_id] = 0
                self.agent_round_done[agent_id] = True
                return -5  # Penalty for busting
            else:
                self.agent_flipped[agent_id].append(card)
                if len(self.agent_flipped[agent_id]) == 7:
                    # Perfect flip7
                    bonus = 15
                    self.agent_round_scores[agent_id] = sum(self.agent_flipped[agent_id]) + bonus
                    self.agent_round_done[agent_id] = True
                    return self.agent_round_scores[agent_id] * 0.1 + 10  # Bonus for perfect flip7
                else:
                    return 0.1  # Small reward for successful hit
        
        return 0

    def _calculate_final_rewards(self):
        """Calculate final rewards based on ranking"""
        # Sort agents by score (descending)
        agent_scores_with_ids = [(self.agent_scores[i], i) for i in range(self.num_agents)]
        agent_scores_with_ids.sort(reverse=True, key=lambda x: x[0])
        
        rewards = [0] * self.num_agents
        
        # Reward structure: 1st gets 100, 2nd gets 50, 3rd gets 20, others get penalty
        reward_structure = [100, 50, 20, -10, -20]
        
        for rank, (score, agent_id) in enumerate(agent_scores_with_ids):
            base_reward = reward_structure[rank] if rank < len(reward_structure) else -30
            
            # Bonus for reaching target
            target_bonus = 20 if score >= self.target_score else 0
            
            # Penalty for being far behind
            max_score = agent_scores_with_ids[0][0]
            distance_penalty = max(0, (max_score - score) * 0.1)
            
            rewards[agent_id] = base_reward + target_bonus - distance_penalty
        
        return rewards

    def get_rankings(self):
        """Get current rankings of all agents"""
        agent_scores_with_ids = [(self.agent_scores[i], i) for i in range(self.num_agents)]
        agent_scores_with_ids.sort(reverse=True, key=lambda x: x[0])
        return agent_scores_with_ids


class CompetitiveDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(CompetitiveDQN, self).__init__()
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


class CompetitiveDQNAgent:
    def __init__(self, agent_id, state_size, action_size, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.agent_id = agent_id
        self.agent_type = "DQN"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        
        # Neural networks
        self.q_network = CompetitiveDQN(state_size, action_size)
        self.target_network = CompetitiveDQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, obs=None):
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN
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


class HeuristicAgent:
    def __init__(self, agent_id, strategy="conservative"):
        self.agent_id = agent_id
        self.strategy = strategy
        self.agent_type = f"Heuristic_{strategy}"
        self.epsilon = 0  # No exploration for heuristics
        
    def act(self, state, obs):
        """
        Different heuristic strategies:
        - conservative: Play it safe, avoid high bust probabilities
        - aggressive: Take more risks to get higher scores
        - adaptive: Change strategy based on competitive position
        """
        flipped_cards = obs["flipped"]
        bust_prob = calculate_bust_probability(flipped_cards)
        current_sum = sum(flipped_cards)
        num_flipped = len(flipped_cards)
        
        # Get competitive context
        my_score = obs["total_score"]
        max_opponent_score = obs["max_opponent_score"]
        my_rank = obs["my_rank"]
        final_round = obs["final_round"]
        
        if self.strategy == "conservative":
            return self._conservative_strategy(bust_prob, current_sum, num_flipped)
        elif self.strategy == "aggressive":
            return self._aggressive_strategy(bust_prob, current_sum, num_flipped)
        elif self.strategy == "adaptive":
            return self._adaptive_strategy(bust_prob, current_sum, num_flipped, 
                                         my_score, max_opponent_score, my_rank, final_round)
        else:
            return self._conservative_strategy(bust_prob, current_sum, num_flipped)
    
    def _conservative_strategy(self, bust_prob, current_sum, num_flipped):
        """Conservative: Minimize risk, stay early with decent scores"""
        if num_flipped == 0:
            return 1  # Always hit first card
        elif num_flipped >= 6:
            return 0  # Stay when close to 7
        elif bust_prob > 0.25:
            return 0  # Stay if bust probability too high
        elif current_sum >= 20:
            return 0  # Stay with decent sum
        else:
            return 1  # Hit otherwise
    
    def _aggressive_strategy(self, bust_prob, current_sum, num_flipped):
        """Aggressive: Take more risks for higher scores"""
        if num_flipped == 0:
            return 1  # Always hit first card
        elif num_flipped >= 6:
            return 0  # Must stay at 6 cards
        elif bust_prob > 0.4:  # Higher risk tolerance
            return 0  # Only stay if very risky
        elif current_sum >= 35:  # Aim for higher scores
            return 0  # Stay with high sum
        else:
            return 1  # Hit otherwise
    
    def _adaptive_strategy(self, bust_prob, current_sum, num_flipped, 
                          my_score, max_opponent_score, my_rank, final_round):
        """Adaptive: Change strategy based on competitive position"""
        if num_flipped == 0:
            return 1  # Always hit first card
        elif num_flipped >= 6:
            return 0  # Must stay at 6 cards
        
        # Determine if we need to take risks
        score_gap = max_opponent_score - my_score
        behind = score_gap > 10  # Significantly behind
        way_behind = score_gap > 30  # Way behind
        leading = my_rank == 1
        
        # Adjust risk tolerance based on position
        if way_behind or (final_round and behind):
            # Take big risks when desperate
            risk_threshold = 0.5
            sum_threshold = 40
        elif behind:
            # Take moderate risks when behind
            risk_threshold = 0.35
            sum_threshold = 30
        elif leading:
            # Play safe when leading
            risk_threshold = 0.2
            sum_threshold = 15
        else:
            # Balanced play when in middle
            risk_threshold = 0.3
            sum_threshold = 25
        
        if bust_prob > risk_threshold:
            return 0  # Stay if too risky for our position
        elif current_sum >= sum_threshold:
            return 0  # Stay with adequate sum for our position
        else:
            return 1  # Hit otherwise
    
    def remember(self, state, action, reward, next_state, done):
        pass  # Heuristic agents don't learn
    
    def replay(self, batch_size=64):
        pass  # Heuristic agents don't train
    
    def update_target_network(self):
        pass  # Heuristic agents don't have networks


def calculate_bust_probability(flipped_cards, cards_seen_this_round=None):
    """Calculate probability of busting if we hit"""
    if len(flipped_cards) == 0:
        return 0.0
    
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


def state_to_vector(obs, num_agents=5):
    """Enhanced state representation for competitive play"""
    flipped_cards = obs["flipped"]
    
    # Binary representation of flipped cards (0-12)
    flipped_vector = [0] * 13
    for card in flipped_cards:
        if 0 <= card <= 12:  # Safety check
            flipped_vector[card] = 1
    
    # Game state features
    num_flipped = len(flipped_cards)
    current_sum = sum(flipped_cards)
    bust_prob = calculate_bust_probability(flipped_cards)
    
    # Competitive features
    my_score = obs["total_score"]
    max_opponent_score = obs["max_opponent_score"]
    my_rank = obs["my_rank"]
    agents_at_target = obs["agents_at_target"]
    
    # Strategic features
    score_gap = max_opponent_score - my_score  # How far behind/ahead we are
    relative_position = (num_agents - my_rank + 1) / num_agents  # 1.0 = first place, 0.2 = last
    urgency = float(obs["final_round"] or agents_at_target > 0)  # Need to take risks?
    
    # Progress features
    progress_features = [
        num_flipped / 7.0,  # How close to 7 cards
        current_sum / 50.0,  # Normalized current sum
        my_score / 200.0,  # Progress to target
        max_opponent_score / 200.0,  # Leading opponent progress
        score_gap / 200.0,  # Normalized score gap (can be negative)
        relative_position,  # Current ranking position
        urgency,  # Game urgency
        bust_prob,  # Risk assessment
        float(obs["round_done"]),
        float(obs["busted"]),
        float(obs["final_round"]),
        obs["round_number"] / 100.0,  # Normalized round number
    ]
    
    # Combine all features
    all_features = flipped_vector + progress_features
    return np.array(all_features, dtype=np.float32)


def train_competitive_agents(episodes=5000, num_agents=5, target_score=200):
    env = MultiAgentFlip7Env(num_agents=num_agents, target_score=target_score)
    state_size = 25  # Enhanced competitive features
    action_size = 2
    
    # Create mixed agents: 3 DQN agents + 2 heuristic agents
    agents = []
    # DQN agents
    for i in range(3):
        agents.append(CompetitiveDQNAgent(i, state_size, action_size, lr=0.0005, gamma=0.95))
    
    # Heuristic agents
    agents.append(HeuristicAgent(3, strategy="conservative"))
    agents.append(HeuristicAgent(4, strategy="adaptive"))
    
    # Tracking
    episode_rewards = [[] for _ in range(num_agents)]
    win_counts = [0] * num_agents
    avg_scores = [[] for _ in range(num_agents)]
    agent_types = [agent.agent_type for agent in agents]
    
    for episode in range(episodes):
        obs_list = env.reset()
        states = [state_to_vector(obs, num_agents) for obs in obs_list]
        total_rewards = [0] * num_agents
        step_count = 0
        max_steps = 1000  # Prevent infinite loops
        
        while step_count < max_steps:
            # Get actions from all agents
            actions = []
            for i in range(num_agents):
                if agents[i].agent_type.startswith("Heuristic"):
                    actions.append(agents[i].act(states[i], obs_list[i]))
                else:
                    actions.append(agents[i].act(states[i]))
            
            # Environment step
            next_obs_list, rewards, done, _ = env.step(actions)
            next_states = [state_to_vector(obs, num_agents) for obs in next_obs_list]
            
            # Store experiences and update total rewards (only for DQN agents)
            for i in range(num_agents):
                if agents[i].agent_type == "DQN":
                    agents[i].remember(states[i], actions[i], rewards[i], next_states[i], done)
                total_rewards[i] += rewards[i]
            
            states = next_states
            obs_list = next_obs_list
            step_count += 1
            
            if done:
                break
        
        # Training (only DQN agents)
        for agent in agents:
            if agent.agent_type == "DQN" and len(agent.memory) > 1000:
                for _ in range(2):  # Multiple training steps
                    agent.replay(batch_size=32)
        
        # Update target networks (only DQN agents)
        if episode % 100 == 0:
            for agent in agents:
                if agent.agent_type == "DQN":
                    agent.update_target_network()
        
        # Track statistics
        rankings = env.get_rankings()
        winner_id = rankings[0][1]
        win_counts[winner_id] += 1
        
        for i in range(num_agents):
            episode_rewards[i].append(total_rewards[i])
            avg_scores[i].append(obs_list[i]["total_score"])
        
        # Progress reporting
        if episode % 500 == 0:
            print(f"\nEpisode {episode}:")
            for i in range(num_agents):
                avg_reward = np.mean(episode_rewards[i][-100:]) if len(episode_rewards[i]) >= 100 else np.mean(episode_rewards[i]) if episode_rewards[i] else 0
                avg_score = np.mean(avg_scores[i][-100:]) if len(avg_scores[i]) >= 100 else np.mean(avg_scores[i]) if avg_scores[i] else 0
                win_rate = win_counts[i] / (episode + 1) * 100
                epsilon_str = f", Epsilon: {agents[i].epsilon:.3f}" if agents[i].agent_type == "DQN" else ""
                print(f"  Agent {i} ({agent_types[i]}): Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.2f}, Win Rate: {win_rate:.1f}%{epsilon_str}")
    
    return agents, episode_rewards, win_counts, avg_scores, agent_types


def test_competitive_agents(agents, episodes=100, num_agents=5, target_score=200):
    env = MultiAgentFlip7Env(num_agents=num_agents, target_score=target_score)
    
    # Set DQN agents to no exploration
    for agent in agents:
        if agent.agent_type == "DQN":
            agent.epsilon = 0
    
    win_counts = [0] * num_agents
    final_scores = [[] for _ in range(num_agents)]
    
    for episode in range(episodes):
        obs_list = env.reset()
        step_count = 0
        max_steps = 1000
        
        while step_count < max_steps:
            states = [state_to_vector(obs, num_agents) for obs in obs_list]
            actions = []
            for i in range(num_agents):
                if agents[i].agent_type.startswith("Heuristic"):
                    actions.append(agents[i].act(states[i], obs_list[i]))
                else:
                    actions.append(agents[i].act(states[i]))
            
            obs_list, _, done, _ = env.step(actions)
            step_count += 1
            
            if done:
                rankings = env.get_rankings()
                winner_id = rankings[0][1]
                win_counts[winner_id] += 1
                
                for i in range(num_agents):
                    final_scores[i].append(obs_list[i]["total_score"])
                break
    
    return win_counts, final_scores


def demo_competitive_game(agents, target_score=200):
    """Demo a single competitive game"""
    env = MultiAgentFlip7Env(num_agents=len(agents), target_score=target_score)
    
    # Set DQN agents to no exploration
    for agent in agents:
        if agent.agent_type == "DQN":
            agent.epsilon = 0
    
    obs_list = env.reset()
    round_num = 1
    step_count = 0
    max_steps = 1000
    
    print(f"=== COMPETITIVE FLIP7 DEMO ===")
    print(f"Target Score: {target_score}")
    print(f"Agents: {[f'Agent {i} ({agent.agent_type})' for i, agent in enumerate(agents)]}")
    
    while step_count < max_steps:
        # Check if we're starting a new round
        if all(obs["round_done"] for obs in obs_list) and not env.game_over:
            print(f"\n--- Round {round_num} ---")
            # Show current standings
            current_scores = [obs["total_score"] for obs in obs_list]
            print("Current Scores:", current_scores)
            round_num += 1
        
        states = [state_to_vector(obs, len(agents)) for obs in obs_list]
        actions = []
        for i in range(len(agents)):
            if agents[i].agent_type.startswith("Heuristic"):
                actions.append(agents[i].act(states[i], obs_list[i]))
            else:
                actions.append(agents[i].act(states[i]))
        
        # Show actions for agents still playing in current round
        active_agents = [i for i, obs in enumerate(obs_list) if not obs["round_done"]]
        if active_agents:
            for i in active_agents:
                obs = obs_list[i]
                action = actions[i]
                action_name = "stay" if action == 0 else "hit"
                bust_prob = calculate_bust_probability(obs["flipped"])
                print(f"Agent {i} ({agents[i].agent_type}): Flipped {obs['flipped']}, Sum: {sum(obs['flipped']) if obs['flipped'] else 0}, Bust Prob: {bust_prob:.2f}, Action: {action_name}")
        
        obs_list, rewards, done, _ = env.step(actions)
        step_count += 1
        
        # Show immediate results for actions taken
        for i, obs in enumerate(obs_list):
            if obs["busted"] and i in active_agents:
                print(f"Agent {i}: BUSTED!")
            elif obs["round_done"] and obs["round_score"] > 0 and i in active_agents:
                print(f"Agent {i}: Round score: {obs['round_score']}")
        
        if done:
            break
    
    print(f"\n=== FINAL RESULTS ===")
    rankings = env.get_rankings()
    for rank, (score, agent_id) in enumerate(rankings, 1):
        print(f"{rank}. Agent {agent_id} ({agents[agent_id].agent_type}): {score} points")
    
    return rankings


def analyze_agent_performance(win_counts, final_scores, agent_types, episodes):
    """Analyze and display agent performance statistics"""
    print("\n=== PERFORMANCE ANALYSIS ===")
    
    # Win rate analysis
    win_rates = [count / episodes for count in win_counts]
    for i, (win_rate, agent_type) in enumerate(zip(win_rates, agent_types)):
        print(f"Agent {i} ({agent_type}): {win_rate:.3f} win rate ({win_counts[i]}/{episodes} wins)")
    
    # Score statistics
    print("\n=== SCORE STATISTICS ===")
    for i, (scores, agent_type) in enumerate(zip(final_scores, agent_types)):
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            max_score = max(scores)
            min_score = min(scores)
            print(f"Agent {i} ({agent_type}):")
            print(f"  Average: {avg_score:.2f} ± {std_score:.2f}")
            print(f"  Range: {min_score} - {max_score}")
    
    return win_rates


def plot_training_results(episode_rewards, win_counts, avg_scores, agent_types):
    """Plot comprehensive training results"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode rewards over time
    ax1.set_title("Training Rewards Over Time")
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, rewards in enumerate(episode_rewards):
        if len(rewards) > 0:
            # Smooth the rewards using moving average
            window_size = min(100, len(rewards) // 10)
            if window_size > 1:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(rewards)), smoothed, 
                        color=colors[i % len(colors)], label=f'Agent {i} ({agent_types[i]})')
            else:
                ax1.plot(rewards, color=colors[i % len(colors)], label=f'Agent {i} ({agent_types[i]})')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Win distribution
    ax2.set_title("Win Distribution")
    agent_labels = [f'Agent {i}\n({agent_types[i]})' for i in range(len(win_counts))]
    bars = ax2.bar(range(len(win_counts)), win_counts, color=colors[:len(win_counts)])
    ax2.set_xlabel("Agent")
    ax2.set_ylabel("Number of Wins")
    ax2.set_xticks(range(len(win_counts)))
    ax2.set_xticklabels(agent_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, win_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # Plot 3: Average scores over time
    ax3.set_title("Average Scores Over Time")
    for i, scores in enumerate(avg_scores):
        if len(scores) > 0:
            ax3.plot(scores, color=colors[i % len(colors)], label=f'Agent {i} ({agent_types[i]})')
    ax3.set_xlabel("Episode (x100)")
    ax3.set_ylabel("Average Score")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Win rate comparison
    ax4.set_title("Win Rate Comparison")
    total_episodes = sum(win_counts)
    win_rates = [count / total_episodes for count in win_counts] if total_episodes > 0 else [0] * len(win_counts)
    bars = ax4.bar(range(len(win_rates)), win_rates, color=colors[:len(win_rates)])
    ax4.set_xlabel("Agent")
    ax4.set_ylabel("Win Rate")
    ax4.set_xticks(range(len(win_rates)))
    ax4.set_xticklabels(agent_labels, rotation=45, ha='right')
    ax4.set_ylim(0, max(win_rates) * 1.1 if win_rates else 1)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def save_agents(agents, filename="flip7_agents.pkl"):
    """Save trained agents to file"""
    import pickle
    agent_data = []
    for agent in agents:
        if hasattr(agent, 'q_network'):  # DQN agent
            agent_data.append({
                'type': agent.agent_type,
                'state_dict': agent.q_network.state_dict(),
                'epsilon': agent.epsilon
            })
        else:  # Heuristic agent
            agent_data.append({
                'type': agent.agent_type
            })
    
    with open(filename, 'wb') as f:
        pickle.dump(agent_data, f)
    print(f"Agents saved to {filename}")


def run_tournament(agents, rounds=10, target_score=200):
    """Run a tournament between all agents"""
    print(f"\n=== TOURNAMENT ({rounds} rounds) ===")
    
    tournament_wins = [0] * len(agents)
    all_rankings = []
    
    for round_num in range(rounds):
        print(f"\nRound {round_num + 1}/{rounds}")
        rankings = demo_competitive_game(agents, target_score)
        all_rankings.append(rankings)
        
        # Award points based on ranking (1st place gets most points)
        winner_id = rankings[0][1]
        tournament_wins[winner_id] += 1
    
    print(f"\n=== TOURNAMENT RESULTS ===")
    for i, wins in enumerate(tournament_wins):
        win_rate = wins / rounds
        print(f"Agent {i} ({agents[i].agent_type}): {wins}/{rounds} wins ({win_rate:.1%})")
    
    return tournament_wins, all_rankings


if __name__ == "__main__":
    print("Training Competitive Flip7 Agents...")
    
    # Train agents
    agents, episode_rewards, win_counts, avg_scores, agent_types = train_competitive_agents(episodes=3000)
    
    print("\nTesting trained agents...")
    test_wins, test_scores = test_competitive_agents(agents, episodes=500)
    
    # Results analysis
    print(f"\nTraining Results (Win Rates):")
    training_win_rates = analyze_agent_performance(win_counts, [], agent_types, sum(win_counts))
    
    print(f"\nTesting Results (Win Rates):")
    testing_win_rates = analyze_agent_performance(test_wins, test_scores, agent_types, 500)
    
    # Plot results
    plot_training_results(episode_rewards, test_wins, avg_scores, agent_types)
    
    # Save trained agents
    save_agents(agents, "trained_flip7_agents.pkl")
    
    # Run a demonstration tournament
    print("\n" + "="*50)
    print("DEMONSTRATION TOURNAMENT")
    print("="*50)
    tournament_wins, tournament_rankings = run_tournament(agents, rounds=5, target_score=200)
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Best performing agent in training: Agent {np.argmax(win_counts)} ({agent_types[np.argmax(win_counts)]})")
    print(f"Best performing agent in testing: Agent {np.argmax(test_wins)} ({agent_types[np.argmax(test_wins)]})")
    print(f"Most consistent performer: Agent {np.argmin([np.std(scores) for scores in test_scores if len(scores) > 0])} (lowest score variance)")
    
    # Performance comparison table
    print(f"\nPerformance Comparison:")
    print(f"{'Agent':<15} {'Type':<15} {'Train Win%':<12} {'Test Win%':<12} {'Avg Score':<12}")
    print("-" * 70)
    for i in range(len(agents)):
        train_wr = training_win_rates[i] if i < len(training_win_rates) else 0
        test_wr = testing_win_rates[i] if i < len(testing_win_rates) else 0
        avg_score = np.mean(test_scores[i]) if len(test_scores[i]) > 0 else 0
        print(f"Agent {i:<9} {agent_types[i]:<15} {train_wr:<11.1%} {test_wr:<11.1%} {avg_score:<11.1f}")