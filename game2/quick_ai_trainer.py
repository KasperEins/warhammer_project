#!/usr/bin/env python3
"""
🚀 QUICK AI TRAINER
===================

Fast training system for Warhammer AIs using the proven working architecture.
Can train new models if the 300k models aren't available.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json
from datetime import datetime

# Import our battle system
from warhammer_battle_core import *

class WarhammerDQN(nn.Module):
    """Deep Q-Network for Warhammer AI - WORKING ARCHITECTURE"""
    def __init__(self, input_size=50, hidden_size=256, output_size=15):
        super(WarhammerDQN, self).__init__()
        # Use the EXACT architecture that worked in our 300k training
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QuickWarhammerAI:
    """Quick training AI agent using proven architecture"""
    
    def __init__(self, faction: Faction, learning_rate=0.001, epsilon=0.3):
        self.faction = faction
        self.q_network = WarhammerDQN()
        self.target_network = WarhammerDQN()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training parameters
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.target_update_freq = 100
        self.update_count = 0
        
        # Copy to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, 14)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QuickTrainingBattle:
    """Simplified battle environment for quick training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset battle to initial state"""
        self.battlefield = BattleField(width=24, height=16)
        self.turn = 0
        self.max_turns = 30
        
        # Setup simplified armies
        empire_units = [
            WarhammerUnit("empire_1", UNIT_PROFILES["Empire Handgunners"], Position(2, 8), 15),
            WarhammerUnit("empire_2", UNIT_PROFILES["Empire Knights"], Position(1, 6), 5),
        ]
        
        orc_units = [
            WarhammerUnit("orc_1", UNIT_PROFILES["Orc Boyz"], Position(21, 8), 20),
            WarhammerUnit("orc_2", UNIT_PROFILES["Orc Arrer Boyz"], Position(22, 6), 15),
        ]
        
        for unit in empire_units + orc_units:
            self.battlefield.add_unit(unit)
        
        return self.get_state()
    
    def get_state(self):
        """Get current battle state as vector"""
        return self.battlefield.get_battle_state_vector()
    
    def step(self, empire_action, orc_action):
        """Execute one turn of battle"""
        self.turn += 1
        
        # Reset unit flags
        for unit in self.battlefield.empire_units + self.battlefield.orc_units:
            unit.has_moved = False
            unit.has_shot = False
            unit.has_charged = False
        
        # Simple action execution (simplified for quick training)
        empire_reward = self.execute_simple_action(empire_action, Faction.EMPIRE)
        orc_reward = self.execute_simple_action(orc_action, Faction.ORCS)
        
        # Check if battle is over
        empire_alive = any(unit.is_alive for unit in self.battlefield.empire_units)
        orc_alive = any(unit.is_alive for unit in self.battlefield.orc_units)
        
        done = not (empire_alive and orc_alive) or self.turn >= self.max_turns
        
        # Calculate final rewards
        if done:
            if empire_alive and not orc_alive:
                empire_reward += 100
                orc_reward -= 100
            elif orc_alive and not empire_alive:
                empire_reward -= 100
                orc_reward += 100
        
        return self.get_state(), empire_reward, orc_reward, done
    
    def execute_simple_action(self, action, faction):
        """Execute simplified action and return reward"""
        units = self.battlefield.empire_units if faction == Faction.EMPIRE else self.battlefield.orc_units
        enemies = self.battlefield.orc_units if faction == Faction.EMPIRE else self.battlefield.empire_units
        
        reward = 0
        
        # Simple movement actions (0-7)
        if action <= 7:
            directions = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]
            dx, dy = directions[action]
            
            for unit in units[:1]:  # Move first unit
                if unit.can_move():
                    new_x = max(0, min(23, unit.position.x + dx))
                    new_y = max(0, min(15, unit.position.y + dy))
                    unit.position = Position(new_x, new_y, unit.position.facing)
                    unit.has_moved = True
                    reward += 1
        
        # Combat actions (8-14)
        elif action >= 8:
            # Simple combat simulation
            if units and enemies:
                attacker = units[0]
                target = enemies[0]
                
                if attacker.is_alive and target.is_alive:
                    # Simple damage calculation
                    damage = random.randint(1, 3)
                    killed = target.take_wounds(damage)
                    reward += killed * 10
        
        return reward

def quick_train_ai(faction: Faction, episodes=1000, save_path=None):
    """Quick training function"""
    print(f"🚀 Quick training {faction.value} AI for {episodes} episodes...")
    
    ai = QuickWarhammerAI(faction)
    battle_env = QuickTrainingBattle()
    
    scores = []
    win_count = 0
    
    for episode in range(episodes):
        state = battle_env.reset()
        total_reward = 0
        
        while True:
            # Get actions for both AIs
            if faction == Faction.EMPIRE:
                action = ai.get_action(state)
                orc_action = random.randint(0, 14)  # Random opponent
            else:
                empire_action = random.randint(0, 14)  # Random opponent
                action = ai.get_action(state)
                orc_action = action
                empire_action = empire_action
            
            # Execute step
            if faction == Faction.EMPIRE:
                next_state, reward, orc_reward, done = battle_env.step(action, orc_action)
            else:
                next_state, empire_reward, reward, done = battle_env.step(empire_action, orc_action)
            
            # Store experience
            ai.remember(state, action, reward, next_state, done)
            
            # Train
            ai.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        if total_reward > 0:
            win_count += 1
        
        # Progress report
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            win_rate = win_count / (episode + 1) * 100
            print(f"Episode {episode + 1}: Avg Score: {avg_score:.2f}, Win Rate: {win_rate:.1f}%, Epsilon: {ai.epsilon:.3f}")
    
    # Save model
    if save_path:
        torch.save({
            'q_network_state_dict': ai.q_network.state_dict(),
            'optimizer_state_dict': ai.optimizer.state_dict(),
            'epsilon': ai.epsilon,
            'episodes': episodes
        }, save_path)
        print(f"✅ Model saved to {save_path}")
    
    final_win_rate = win_count / episodes * 100
    print(f"🎯 Final win rate: {final_win_rate:.1f}%")
    
    return ai

if __name__ == "__main__":
    print("🚀 QUICK AI TRAINER")
    print("=" * 40)
    
    # Train Empire AI
    empire_ai = quick_train_ai(Faction.EMPIRE, episodes=500, save_path="quick_empire_ai.pth")
    
    print("\n" + "=" * 40)
    
    # Train Orc AI
    orc_ai = quick_train_ai(Faction.ORCS, episodes=500, save_path="quick_orc_ai.pth")
    
    print("\n🎉 Quick training complete!")
    print("✅ Models saved as quick_empire_ai.pth and quick_orc_ai.pth")
    print("🎯 Ready for battle visualization!") 