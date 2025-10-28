#!/usr/bin/env python3
"""
WARHAMMER: THE OLD WORLD - AI STRATEGIC LEARNING SYSTEM
======================================================
Reinforcement Learning Agent for discovering optimal battle strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nuln_vs_trolls import NulnVsTrollBattle
from old_world_battle import OldWorldUnit, UnitType, FormationType

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class WarhammerBattleEnvironment:
    """Simplified battle environment for RL training"""
    
    def __init__(self):
        self.battle = None
        self.initial_state = None
        self.max_turns = 6
        self.current_turn = 1
        self.action_history = []
        
        # Action names for UI display
        self.action_names = [
            "Move North", "Move NE", "Move East", "Move SE", 
            "Move South", "Move SW", "Move West", "Move NW",
            "Artillery Strike", "Cavalry Charge", "Defensive Formation", 
            "Flanking Maneuver", "Mass Shooting"
        ]
        
    def reset(self):
        """Reset environment for new episode"""
        self.battle = NulnVsTrollBattle()
        armies = self.battle.create_armies()
        for unit in armies:
            self.battle.add_unit(unit)
        
        self.current_turn = 1
        self.action_history = []
        self.initial_state = self._get_state()
        return self.initial_state
    
    def _get_state(self) -> np.ndarray:
        """Convert battle state to neural network input"""
        state_features = []
        
        # Global battle info
        state_features.extend([
            self.current_turn / 6.0,  # Normalized turn
            len([u for u in self.battle.units if u.player == 1 and u.is_alive()]) / 13,  # Nuln units alive ratio
            len([u for u in self.battle.units if u.player == 2 and u.is_alive()]) / 5,   # Troll units alive ratio
        ])
        
        # Nuln army state (Player 1) - 13 units
        nuln_units = [u for u in self.battle.units if u.player == 1]
        for i in range(13):  # Fixed size for consistent input
            if i < len(nuln_units):
                unit = nuln_units[i]
                state_features.extend([
                    unit.x / 72.0,  # Normalized position
                    unit.y / 48.0,
                    unit.models / unit.max_models if unit.max_models > 0 else 0,  # Health ratio
                    1.0 if unit.is_alive() else 0.0,
                    unit.weapon_range / 48.0 if unit.weapon_range > 0 else 0.0,
                    1.0 if unit.has_charged else 0.0,
                ])
            else:
                state_features.extend([0.0] * 6)  # Padding for missing units
        
        # Enemy army state (Player 2) - 5 units
        troll_units = [u for u in self.battle.units if u.player == 2]
        for i in range(5):  # Fixed size
            if i < len(troll_units):
                unit = troll_units[i]
                state_features.extend([
                    unit.x / 72.0,
                    unit.y / 48.0,
                    unit.models / unit.max_models if unit.max_models > 0 else 0,
                    1.0 if unit.is_alive() else 0.0,
                ])
            else:
                state_features.extend([0.0] * 4)  # Padding
        
        return np.array(state_features, dtype=np.float32)
    
    def _get_available_actions(self) -> List[int]:
        """Get list of valid actions for current state"""
        actions = []
        
        # Movement actions for each alive Nuln unit (0-12)
        nuln_units = [u for u in self.battle.units if u.player == 1 and u.is_alive()]
        for i, unit in enumerate(nuln_units[:13]):  # Max 13 units
            if unit.is_alive():
                actions.extend([i * 10 + j for j in range(8)])  # 8 movement directions per unit
        
        # Shooting target selection (130-142)
        shooting_units = [u for u in nuln_units if u.weapon_range > 0]
        enemy_units = [u for u in self.battle.units if u.player == 2 and u.is_alive()]
        for i, shooter in enumerate(shooting_units[:5]):  # Max 5 shooting units
            for j, target in enumerate(enemy_units[:5]):  # Max 5 targets
                actions.append(130 + i * 5 + j)
        
        # Formation changes (200-212)
        for i, unit in enumerate(nuln_units[:13]):
            if unit.unit_type == UnitType.INFANTRY:
                actions.extend([200 + i * 3 + j for j in range(3)])  # 3 formations
        
        # Special tactics (300-310)
        actions.extend([300, 301, 302, 303, 304])  # Defensive, Aggressive, Flanking, etc.
        
        return actions if actions else [0]  # Always have at least one action
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return new state, reward, done, info"""
        reward = 0.0
        info = {}
        
        # Execute the action
        reward += self._execute_action(action)
        
        # Run enemy turn (simplified AI)
        self._run_enemy_turn()
        
        # Check if battle is over
        done = self._is_battle_over()
        
        # Calculate reward
        if not done:
            reward += self._calculate_positional_reward()
        else:
            reward += self._calculate_final_reward()
        
        # Get new state
        new_state = self._get_state()
        
        # Advance turn
        self.current_turn += 0.5  # Half turn increments
        
        info = {
            'turn': self.current_turn,
            'action_executed': action,
            'nuln_units_alive': len([u for u in self.battle.units if u.player == 1 and u.is_alive()]),
            'troll_units_alive': len([u for u in self.battle.units if u.player == 2 and u.is_alive()])
        }
        
        return new_state, reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return immediate reward"""
        reward = 0.0
        
        # Simple action space: 0-7 movement directions, 8-12 special tactics
        if action < 8:  # Movement actions
            reward += self._move_army_formation(action)
        elif action < 13:  # Special tactics
            reward += self._execute_special_tactic(action - 8)
        
        return reward
    
    def _move_army_formation(self, direction: int) -> float:
        """Move entire army in formation"""
        directions = [(0, 2), (1, 1), (2, 0), (1, -1), (0, -2), (-1, -1), (-2, 0), (-1, 1)]
        
        if direction < len(directions):
            dx, dy = directions[direction]
            nuln_units = [u for u in self.battle.units if u.player == 1 and u.is_alive()]
            
            for unit in nuln_units:
                old_x, old_y = unit.x, unit.y
                unit.x = max(0, min(72, unit.x + dx))
                unit.y = max(0, min(48, unit.y + dy))
                
            return 1.0  # Small reward for coordinated movement
        
        return 0.0
    
    def _execute_special_tactic(self, tactic: int) -> float:
        """Execute special battlefield tactic"""
        nuln_units = [u for u in self.battle.units if u.player == 1 and u.is_alive()]
        enemy_units = [u for u in self.battle.units if u.player == 2 and u.is_alive()]
        
        if tactic == 0:  # Concentrated fire
            artillery = [u for u in nuln_units if u.unit_type == UnitType.ARTILLERY]
            casualties = 0
            for gun in artillery:
                if enemy_units and random.random() < 0.3:
                    target = random.choice(enemy_units)
                    target.models = max(0, target.models - 1)
                    target.update_formation()
                    casualties += 1
            return casualties * 5.0
            
        elif tactic == 1:  # Cavalry charge
            cavalry = [u for u in nuln_units if u.unit_type == UnitType.CAVALRY]
            for rider in cavalry:
                rider.has_charged = True
            return len(cavalry) * 3.0
            
        elif tactic == 2:  # Defensive formation
            infantry = [u for u in nuln_units if u.unit_type == UnitType.INFANTRY]
            for unit in infantry:
                unit.formation = FormationType.DEEP
                unit.update_formation()
            return 2.0
            
        elif tactic == 3:  # Flanking maneuver
            mobile_units = [u for u in nuln_units if u.unit_type in [UnitType.CAVALRY, UnitType.CHARACTER]]
            for unit in mobile_units:
                unit.x = min(72, unit.x + 8)  # Move right for flanking
            return 4.0
            
        elif tactic == 4:  # Mass shooting
            shooters = [u for u in nuln_units if u.weapon_range > 0]
            hits = 0
            for shooter in shooters:
                if enemy_units and random.random() < 0.4:
                    hits += 1
            return hits * 2.0
        
        return 0.0
    
    def _run_enemy_turn(self):
        """Simple AI for enemy turn"""
        troll_units = [u for u in self.battle.units if u.player == 2 and u.is_alive()]
        nuln_units = [u for u in self.battle.units if u.player == 1 and u.is_alive()]
        
        for unit in troll_units:
            if nuln_units:
                # Move towards closest enemy
                closest_enemy = min(nuln_units, key=lambda e: self.battle.distance(unit, e))
                distance = self.battle.distance(unit, closest_enemy)
                
                if distance > 3:  # Move closer
                    dx = closest_enemy.x - unit.x
                    dy = closest_enemy.y - unit.y
                    move_dist = min(unit.movement, distance - 1)
                    
                    if distance > 0:
                        unit.x += (dx / distance) * move_dist
                        unit.y += (dy / distance) * move_dist
                
                # Simple combat if in range
                if distance <= 3:
                    if random.random() < 0.3:  # 30% chance to cause casualty
                        closest_enemy.models = max(0, closest_enemy.models - 1)
                        closest_enemy.update_formation()
    
    def _calculate_positional_reward(self) -> float:
        """Calculate reward for current positioning"""
        reward = 0.0
        
        nuln_units = [u for u in self.battle.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.battle.units if u.player == 2 and u.is_alive()]
        
        # Reward for unit preservation
        nuln_models = sum(u.models for u in nuln_units)
        troll_models = sum(u.models for u in troll_units)
        reward += (nuln_models - troll_models) * 0.1
        
        return reward
    
    def _calculate_final_reward(self) -> float:
        """Calculate final battle reward"""
        if hasattr(self.battle, 'calculate_victory_points'):
            vp_nuln, vp_troll = self.battle.calculate_victory_points()
            if vp_nuln > vp_troll:
                return 100.0 + (vp_nuln - vp_troll)  # Victory bonus
            elif vp_troll > vp_nuln:
                return -50.0 - (vp_troll - vp_nuln)  # Defeat penalty
            else:
                return 25.0  # Draw bonus
        return 0.0
    
    def _is_battle_over(self) -> bool:
        """Check if battle is finished"""
        nuln_alive = any(u.player == 1 and u.is_alive() for u in self.battle.units)
        troll_alive = any(u.player == 2 and u.is_alive() for u in self.battle.units)
        
        return not nuln_alive or not troll_alive or self.current_turn >= 6


class DQNNetwork(nn.Module):
    """Deep Q-Network for learning battle strategies"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class WarhammerAIAgent:
    """Reinforcement Learning Agent for Warhammer: The Old World"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Training metrics
        self.scores = []
        self.victories = 0
        self.defeats = 0
        self.draws = 0
        
        # Strategy analysis
        self.successful_strategies = []
        self.failed_strategies = []
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def analyze_strategy(self, episode_actions, final_reward):
        """Analyze strategies for pattern recognition"""
        strategy_pattern = {
            'actions': episode_actions[:10],  # First 10 actions
            'reward': final_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        if final_reward > 50:  # Victory
            self.successful_strategies.append(strategy_pattern)
            self.victories += 1
        elif final_reward < -25:  # Defeat
            self.failed_strategies.append(strategy_pattern)
            self.defeats += 1
        else:  # Draw
            self.draws += 1
    
    def get_strategy_insights(self) -> Dict[str, Any]:
        """Extract insights from learned strategies"""
        insights = {
            'total_battles': len(self.scores),
            'win_rate': self.victories / max(1, len(self.scores)),
            'average_score': np.mean(self.scores) if self.scores else 0,
            'victories': self.victories,
            'defeats': self.defeats,
            'draws': self.draws,
            'epsilon': self.epsilon
        }
        
        # Analyze successful patterns
        if self.successful_strategies:
            common_tactics = {}
            for strategy in self.successful_strategies[-20:]:  # Last 20 successful
                for action in strategy['actions'][:5]:  # Opening moves
                    tactic = self._categorize_action(action)
                    common_tactics[tactic] = common_tactics.get(tactic, 0) + 1
            
            insights['successful_tactics'] = dict(sorted(
                common_tactics.items(), key=lambda x: x[1], reverse=True
            ))
        
        return insights
    
    def _categorize_action(self, action: int) -> str:
        """Categorize action for pattern analysis"""
        if action < 8:
            directions = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]
            return f"Move {directions[action]}"
        else:
            tactics = ["Artillery Strike", "Cavalry Charge", "Defensive Formation", "Flanking", "Mass Shooting"]
            return tactics[min(action - 8, len(tactics) - 1)]
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': self.scores,
            'successful_strategies': self.successful_strategies,
            'victories': self.victories,
            'defeats': self.defeats,
            'draws': self.draws
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.scores = checkpoint['scores']
        self.successful_strategies = checkpoint['successful_strategies']
        self.victories = checkpoint['victories']
        self.defeats = checkpoint['defeats']
        self.draws = checkpoint['draws']


def train_agent(episodes: int = 500):
    """Train the AI agent"""
    print("🤖 WARHAMMER: THE OLD WORLD - AI TRAINING SYSTEM")
    print("=" * 60)
    print("🧠 Training strategic AI agent...")
    print(f"📊 Episodes: {episodes}")
    print()
    
    # Initialize environment and agent
    env = WarhammerBattleEnvironment()
    state_size = len(env.reset())
    action_size = 13  # 8 movement + 5 tactics
    agent = WarhammerAIAgent(state_size, action_size)
    
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_actions = []
        step_count = 0
        
        while True:
            action = agent.act(state)
            episode_actions.append(action)
            
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done or step_count > 50:  # Max steps per episode
                break
        
        scores.append(total_reward)
        agent.scores.append(total_reward)
        agent.analyze_strategy(episode_actions, total_reward)
        
        # Train agent
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        # Update target network periodically
        if episode % 50 == 0:
            agent.update_target_network()
        
        # Progress reporting
        if episode % 25 == 0:
            avg_score = np.mean(scores[-25:]) if len(scores) >= 25 else np.mean(scores)
            insights = agent.get_strategy_insights()
            print(f"Episode {episode:3d} | Avg Score: {avg_score:6.1f} | Win Rate: {insights['win_rate']:.2%} | ε: {agent.epsilon:.3f}")
            
            if episode % 100 == 0 and insights.get('successful_tactics'):
                print(f"  Top Strategy: {list(insights['successful_tactics'].keys())[0]}")
    
    # Save trained model
    agent.save_model('warhammer_ai_model.pth')
    
    # Final analysis
    print("\n🎯 TRAINING COMPLETE!")
    print("=" * 40)
    insights = agent.get_strategy_insights()
    print(f"📈 Final Win Rate: {insights['win_rate']:.2%}")
    print(f"📊 Average Score: {insights['average_score']:.1f}")
    print(f"🏆 Total Victories: {insights['victories']}")
    print(f"💀 Total Defeats: {insights['defeats']}")
    print(f"🤝 Draws: {insights['draws']}")
    print()
    
    if insights.get('successful_tactics'):
        print("🧠 DISCOVERED STRATEGIES:")
        for tactic, count in list(insights['successful_tactics'].items())[:5]:
            print(f"  • {tactic}: {count} successful uses")
    
    return agent


def test_trained_agent(model_path: str = 'warhammer_ai_model.pth', test_battles: int = 10):
    """Test the trained agent"""
    print("🎮 TESTING TRAINED AI AGENT")
    print("=" * 40)
    
    env = WarhammerBattleEnvironment()
    state_size = len(env.reset())
    action_size = 13
    agent = WarhammerAIAgent(state_size, action_size)
    
    try:
        agent.load_model(model_path)
        print(f"✅ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model file {model_path} not found. Training new agent...")
        agent = train_agent(200)
    
    # Test battles
    agent.epsilon = 0.0  # No exploration during testing
    wins = 0
    total_rewards = []
    
    for battle in range(test_battles):
        state = env.reset()
        total_reward = 0
        moves = []
        
        step_count = 0
        while step_count < 20:  # Max 20 moves per test
            action = agent.act(state)
            moves.append(agent._categorize_action(action))
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        total_rewards.append(total_reward)
        if total_reward > 50:  # Victory threshold
            wins += 1
            print(f"Battle {battle + 1:2d}: VICTORY   (Score: {total_reward:6.1f}) - Strategy: {', '.join(moves[:3])}")
        else:
            print(f"Battle {battle + 1:2d}: Defeat   (Score: {total_reward:6.1f}) - Strategy: {', '.join(moves[:3])}")
    
    print(f"\n🏆 TEST RESULTS:")
    print(f"   Win Rate: {wins}/{test_battles} ({wins/test_battles:.1%})")
    print(f"   Avg Score: {np.mean(total_rewards):.1f}")
    print(f"   Best Score: {max(total_rewards):.1f}")
    
    return agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Warhammer: The Old World AI Training')
    parser.add_argument('--train', action='store_true', help='Train new agent')
    parser.add_argument('--test', action='store_true', help='Test trained agent')
    parser.add_argument('--episodes', type=int, default=300, help='Training episodes')
    parser.add_argument('--battles', type=int, default=10, help='Test battles')
    
    args = parser.parse_args()
    
    if args.train:
        train_agent(args.episodes)
    elif args.test:
        test_trained_agent(test_battles=args.battles)
    else:
        # Default: train then test
        print("🚀 STARTING AI TRAINING & TESTING SEQUENCE")
        print("=" * 50)
        agent = train_agent(args.episodes)
        print("\n" + "="*60)
        test_trained_agent(test_battles=args.battles) 