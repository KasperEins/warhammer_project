#!/usr/bin/env python3
"""
TROLL ARMY AI TRAINER
====================
Train an AI agent to command the Troll Horde army
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from warhammer_ai_agent import DQNNetwork
import time

class TrollBattleEnvironment:
    """Battle environment for training Troll army AI"""
    
    def __init__(self):
        # Troll army composition (Player 2)
        self.troll_units = {
            'Bigboss on Boar Chariot': {'models': 1, 'strength': 8, 'toughness': 6, 'points': 180},
            'Goblin Boss': {'models': 1, 'strength': 5, 'toughness': 4, 'points': 85},
            'Orc Shaman': {'models': 1, 'strength': 4, 'toughness': 4, 'points': 120},
            'Night Goblin Shaman': {'models': 1, 'strength': 3, 'toughness': 3, 'points': 90},
            'Stone Trolls': {'models': 3, 'strength': 6, 'toughness': 5, 'points': 195},
            'River Trolls': {'models': 3, 'strength': 5, 'toughness': 4, 'points': 165},
            'Common Trolls': {'models': 6, 'strength': 5, 'toughness': 4, 'points': 210},
            'Goblin Wolf Riders': {'models': 10, 'strength': 3, 'toughness': 3, 'points': 140},
            'Orc Boar Boyz': {'models': 8, 'strength': 4, 'toughness': 4, 'points': 168},
            'Night Goblins': {'models': 40, 'strength': 3, 'toughness': 3, 'points': 280},
            'Orc Boyz': {'models': 30, 'strength': 4, 'toughness': 4, 'points': 330}
        }
        
        # Enemy (Nuln) units to fight against
        self.enemy_units = {
            'Great Cannons': {'models': 2, 'strength': 10, 'toughness': 7, 'points': 180},
            'Helblaster Volley Guns': {'models': 3, 'strength': 5, 'toughness': 7, 'points': 210},
            'Nuln State Troops': {'models': 40, 'strength': 4, 'toughness': 3, 'points': 280},
            'Outriders': {'models': 10, 'strength': 4, 'toughness': 3, 'points': 140},
            'Engineers': {'models': 2, 'strength': 3, 'toughness': 3, 'points': 120}
        }
        
        self.max_turns = 6
        self.reset()
    
    def reset(self):
        """Reset battle state"""
        self.turn = 1
        self.troll_health = 100
        self.enemy_health = 100
        self.troll_position = np.array([20.0, 50.0])  # Starting on left side
        self.enemy_position = np.array([80.0, 50.0])  # Enemy on right side
        self.battle_momentum = 0  # Positive = Troll advantage, negative = Enemy advantage
        
        # Troll special abilities
        self.regeneration_available = True
        self.stupidity_risk = 0.1  # Trolls can be stupid
        self.fear_aura = True
        self.stone_skin_active = False
        
        return self._get_state()
    
    def _get_state(self):
        """Get current battle state for AI"""
        state = np.array([
            self.troll_health / 100.0,
            self.enemy_health / 100.0,
            self.turn / 6.0,
            self.battle_momentum / 50.0,
            self.troll_position[0] / 100.0,
            self.troll_position[1] / 100.0,
            self.enemy_position[0] / 100.0,
            self.enemy_position[1] / 100.0,
            float(self.regeneration_available),
            self.stupidity_risk,
            float(self.fear_aura),
            float(self.stone_skin_active),
            np.linalg.norm(self.troll_position - self.enemy_position) / 100.0  # Distance to enemy
        ])
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        reward = 0
        
        # Troll action mapping
        if action == 0:  # Troll Charge
            reward += self._troll_charge()
        elif action == 1:  # Regeneration
            reward += self._regeneration()
        elif action == 2:  # Stone Skin Defense
            reward += self._stone_skin()
        elif action == 3:  # Fear Roar
            reward += self._fear_roar()
        elif action == 4:  # Smash Attack
            reward += self._smash_attack()
        elif action == 5:  # Goblin Swarm
            reward += self._goblin_swarm()
        elif action == 6:  # Boar Cavalry Charge
            reward += self._boar_charge()
        elif action == 7:  # Magic Attack
            reward += self._magic_attack()
        elif action == 8:  # Move Forward
            reward += self._move_forward()
        elif action == 9:  # Move Back
            reward += self._move_back()
        elif action == 10: # Flank Left
            reward += self._flank_left()
        elif action == 11: # Flank Right
            reward += self._flank_right()
        elif action == 12: # Hold Position
            reward += self._hold_position()
        
        # Enemy responds (simplified)
        enemy_damage = self._enemy_response()
        self.troll_health -= enemy_damage
        reward -= enemy_damage * 0.5
        
        # Check for victory conditions
        done = False
        if self.troll_health <= 0:
            reward -= 100  # Large penalty for losing
            done = True
        elif self.enemy_health <= 0:
            reward += 200  # Large reward for winning
            done = True
        elif self.turn >= self.max_turns:
            # Victory points calculation
            if self.troll_health > self.enemy_health:
                reward += 100
            done = True
        
        self.turn += 1
        return self._get_state(), reward, done, {}
    
    def _troll_charge(self):
        """Devastating troll charge attack"""
        if np.linalg.norm(self.troll_position - self.enemy_position) < 30:
            damage = random.randint(15, 25)
            self.enemy_health -= damage
            self.battle_momentum += 15
            self.troll_position += (self.enemy_position - self.troll_position) * 0.3
            return damage
        return -5  # Too far to charge
    
    def _regeneration(self):
        """Troll regeneration ability"""
        if self.regeneration_available and self.troll_health < 80:
            heal = random.randint(10, 20)
            self.troll_health = min(100, self.troll_health + heal)
            self.regeneration_available = False
            return heal
        return -2
    
    def _stone_skin(self):
        """Stone troll defensive ability"""
        self.stone_skin_active = True
        self.battle_momentum += 5
        return 8
    
    def _fear_roar(self):
        """Terrifying roar to demoralize enemies"""
        if self.fear_aura:
            fear_effect = random.randint(5, 15)
            self.battle_momentum += fear_effect
            self.fear_aura = False  # One use per battle
            return fear_effect
        return -3
    
    def _smash_attack(self):
        """Powerful smashing attack"""
        damage = random.randint(8, 18)
        self.enemy_health -= damage
        self.battle_momentum += 5
        return damage
    
    def _goblin_swarm(self):
        """Overwhelm with goblin numbers"""
        swarm_damage = random.randint(12, 20)
        self.enemy_health -= swarm_damage
        self.battle_momentum += 8
        return swarm_damage
    
    def _boar_charge(self):
        """Orc boar cavalry charge"""
        charge_damage = random.randint(10, 16)
        self.enemy_health -= charge_damage
        self.battle_momentum += 6
        return charge_damage
    
    def _magic_attack(self):
        """Shaman magic attack"""
        magic_damage = random.randint(6, 14)
        self.enemy_health -= magic_damage
        self.battle_momentum += 4
        return magic_damage
    
    def _move_forward(self):
        """Advance position"""
        self.troll_position[0] += 5
        self.battle_momentum += 2
        return 3
    
    def _move_back(self):
        """Retreat position"""
        self.troll_position[0] -= 5
        self.battle_momentum -= 2
        return 1
    
    def _flank_left(self):
        """Flank to the left"""
        self.troll_position[1] -= 8
        self.battle_momentum += 4
        return 5
    
    def _flank_right(self):
        """Flank to the right"""
        self.troll_position[1] += 8
        self.battle_momentum += 4
        return 5
    
    def _hold_position(self):
        """Hold defensive position"""
        self.battle_momentum += 1
        return 2
    
    def _enemy_response(self):
        """Simplified enemy AI response"""
        # Nuln artillery and shooting
        if random.random() < 0.7:  # 70% chance of artillery
            base_damage = random.randint(8, 15)
            # Reduced damage if stone skin active
            if self.stone_skin_active:
                base_damage = int(base_damage * 0.6)
                self.stone_skin_active = False
            return base_damage
        else:
            return random.randint(3, 8)

class TrollAIAgent:
    """AI Agent specifically for commanding Troll armies"""
    
    def __init__(self, state_size=13, action_size=13):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Troll-specific action names
        self.action_names = [
            "Troll Charge", "Regeneration", "Stone Skin", "Fear Roar",
            "Smash Attack", "Goblin Swarm", "Boar Charge", "Magic Attack",
            "Move Forward", "Move Back", "Flank Left", "Flank Right", "Hold Position"
        ]
        
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filename):
        """Save the trained model"""
        torch.save(self.q_network.state_dict(), filename)
    
    def load_model(self, filename):
        """Load a trained model"""
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.epsilon = self.epsilon_min  # Use trained model

def train_troll_ai(episodes=5000):
    """Train the Troll AI agent"""
    print("🧌 TRAINING TROLL ARMY AI COMMANDER")
    print("=" * 50)
    
    env = TrollBattleEnvironment()
    agent = TrollAIAgent()
    
    scores = []
    wins = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_score = 0
        
        for step in range(20):  # Max 20 actions per battle
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            total_score += reward
            state = next_state
            
            if done:
                if reward > 0:  # Won the battle
                    wins += 1
                break
        
        agent.replay()
        
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        
        scores.append(total_score)
        
        # Progress reporting
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            win_rate = (wins / (episode + 1)) * 100
            print(f"Episode {episode:4d} | Avg Score: {avg_score:6.1f} | Win Rate: {win_rate:5.2f}% | ε: {agent.epsilon:.3f}")
            
            # Show favorite strategy
            if episode > 0 and episode % 500 == 0:
                strategy_counts = {}
                test_state = env.reset()
                for _ in range(100):
                    action = agent.act(test_state)
                    strategy = agent.action_names[action]
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                top_strategy = max(strategy_counts, key=strategy_counts.get)
                print(f"  🧌 Top Troll Strategy: {top_strategy}")
    
    # Final statistics
    final_win_rate = (wins / episodes) * 100
    final_avg_score = np.mean(scores[-1000:]) if len(scores) >= 1000 else np.mean(scores)
    
    print("\n🎯 TROLL TRAINING COMPLETE!")
    print("=" * 50)
    print(f"📈 Final Win Rate: {final_win_rate:.2f}%")
    print(f"📊 Average Score: {final_avg_score:.1f}")
    print(f"🏆 Total Victories: {wins}")
    print(f"💀 Total Defeats: {episodes - wins}")
    
    # Save the trained model
    agent.save_model('troll_ai_model.pth')
    print(f"💾 Troll AI model saved as 'troll_ai_model.pth'")
    
    return agent, scores

if __name__ == "__main__":
    trained_agent, training_scores = train_troll_ai(5000)
    print("\n🧌 Troll AI is ready for battle!") 