#!/usr/bin/env python3
"""
ANTI-ARTILLERY AI TRAINER
=========================
Specialized training for Orc AI to counter Empire artillery tactics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
from datetime import datetime

class AntiArtilleryEnvironment:
    """Specialized environment for training anti-artillery tactics"""
    
    def __init__(self):
        # Empire Artillery threat levels
        self.artillery_units = [
            {'name': 'Great Cannon', 'range': 48, 'strength': 10, 'threat': 0.9},
            {'name': 'Helblaster Volley Gun', 'range': 24, 'strength': 5, 'threat': 0.8},
            {'name': 'War Wagon', 'range': 20, 'strength': 4, 'threat': 0.6}
        ]
        
        # Orc counters
        self.orc_strategies = {
            'Fast Flanking': {'speed_bonus': 2, 'cover_bonus': 0.3, 'success_rate': 0.0},
            'Magic Disruption': {'range': 24, 'disruption': 0.4, 'success_rate': 0.0},
            'Rapid Advance': {'speed_bonus': 1.5, 'charge_bonus': 0.2, 'success_rate': 0.0},
            'Troll Regeneration': {'toughness_bonus': 2, 'regen': 0.3, 'success_rate': 0.0},
            'Goblin Distraction': {'decoy_factor': 0.5, 'redirection': 0.4, 'success_rate': 0.0},
            'Underground Advance': {'stealth': 0.8, 'surprise': 0.6, 'success_rate': 0.0},
            'Flying Attack': {'aerial': True, 'speed': 3, 'success_rate': 0.0},
            'Magic Resistance': {'ward_save': 0.25, 'spell_immunity': 0.3, 'success_rate': 0.0}
        }
        
        self.reset()
    
    def reset(self):
        """Reset battle scenario"""
        # Random artillery deployment
        self.artillery_positions = []
        for i, unit in enumerate(self.artillery_units):
            x = random.randint(10, 25)  # Empire deployment zone
            y = random.randint(5, 20)
            self.artillery_positions.append({'x': x, 'y': y, 'unit': unit, 'active': True})
        
        # Orc army starting positions
        self.orc_positions = []
        for i in range(8):  # 8 Orc units
            x = random.randint(50, 70)  # Orc deployment zone
            y = random.randint(5, 20)
            self.orc_positions.append({
                'x': x, 'y': y, 'alive': True, 'casualties': 0, 'in_combat': False
            })
        
        self.turn = 1
        self.artillery_shots_fired = 0
        self.orcs_in_combat = 0
        self.magic_disruptions = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current battle state"""
        state = []
        
        # Turn and general info
        state.extend([
            self.turn / 6.0,  # Normalized turn
            self.artillery_shots_fired / 20.0,  # Artillery activity
            self.orcs_in_combat / 8.0,  # Orcs engaged
            self.magic_disruptions / 10.0  # Magic activity
        ])
        
        # Artillery threats (3 units max)
        for i in range(3):
            if i < len(self.artillery_positions):
                art = self.artillery_positions[i]
                state.extend([
                    art['x'] / 72.0,
                    art['y'] / 48.0,
                    1.0 if art['active'] else 0.0,
                    art['unit']['threat']
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])
        
        # Orc units (8 units)
        for i in range(8):
            if i < len(self.orc_positions):
                orc = self.orc_positions[i]
                state.extend([
                    orc['x'] / 72.0,
                    orc['y'] / 48.0,
                    1.0 if orc['alive'] else 0.0,
                    orc['casualties'] / 10.0,
                    1.0 if orc['in_combat'] else 0.0
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute action and advance battle"""
        reward = 0
        strategy_names = list(self.orc_strategies.keys())
        
        if action < len(strategy_names):
            strategy = strategy_names[action]
            reward = self._execute_strategy(strategy)
        
        # Artillery shooting phase
        artillery_damage = self._artillery_phase()
        reward -= artillery_damage * 10  # Penalty for taking casualties
        
        # Check victory conditions
        done = False
        orcs_alive = sum(1 for orc in self.orc_positions if orc['alive'])
        artillery_alive = sum(1 for art in self.artillery_positions if art['active'])
        
        if orcs_alive == 0:
            reward = -100  # Defeat
            done = True
        elif artillery_alive == 0 or self.orcs_in_combat >= 5:
            reward = 100   # Victory - neutralized artillery or engaged in melee
            done = True
        elif self.turn >= 6:
            # Evaluate position
            if self.orcs_in_combat >= 3:
                reward = 50  # Partial victory
            else:
                reward = -50  # Failure to engage
            done = True
        
        self.turn += 1
        next_state = self._get_state()
        
        return next_state, reward, done, {'strategy': strategy_names[action] if action < len(strategy_names) else 'Unknown'}
    
    def _execute_strategy(self, strategy):
        """Execute Orc strategy and return immediate reward"""
        reward = 0
        
        if strategy == 'Fast Flanking':
            # Move units to flanking positions, avoid direct fire
            for orc in self.orc_positions[:4]:  # Half the army flanks
                if orc['alive']:
                    # Move to sides to avoid artillery
                    if orc['y'] > 12:
                        orc['y'] = max(orc['y'] - 5, 2)  # North flank
                    else:
                        orc['y'] = min(orc['y'] + 5, 22) # South flank
                    orc['x'] = max(orc['x'] - 8, 0)  # Advance
                    reward += 5
        
        elif strategy == 'Magic Disruption':
            # Attempt to disrupt enemy artillery with magic
            if random.random() < 0.6:  # 60% success chance
                disrupted_artillery = random.choice([art for art in self.artillery_positions if art['active']])
                disrupted_artillery['active'] = False
                self.magic_disruptions += 1
                reward += 20
                print(f"🪄 Magic disrupts {disrupted_artillery['unit']['name']}!")
        
        elif strategy == 'Rapid Advance':
            # All units charge forward at maximum speed
            for orc in self.orc_positions:
                if orc['alive']:
                    orc['x'] = max(orc['x'] - 12, 0)  # Fast advance
                    if orc['x'] <= 15:  # Reached combat
                        orc['in_combat'] = True
                        self.orcs_in_combat += 1
                        reward += 15
        
        elif strategy == 'Troll Regeneration':
            # Trolls regenerate and become tougher
            for orc in self.orc_positions[:3]:  # First 3 are trolls
                if orc['alive'] and orc['casualties'] > 0:
                    orc['casualties'] = max(0, orc['casualties'] - 2)
                    reward += 3
        
        elif strategy == 'Goblin Distraction':
            # Small units create distractions
            for art in self.artillery_positions:
                if random.random() < 0.3:  # 30% chance to be distracted
                    art['active'] = False
                    reward += 8
        
        elif strategy == 'Underground Advance':
            # Some units advance unseen
            for orc in self.orc_positions[4:]:  # Back half uses tunnels
                if orc['alive'] and random.random() < 0.7:
                    orc['x'] = max(orc['x'] - 15, 0)  # Surprise advance
                    reward += 10
        
        elif strategy == 'Flying Attack':
            # Aerial units bypass ground defenses
            if self.orc_positions[1]['alive']:  # Wyvern rider
                self.orc_positions[1]['x'] = 5  # Direct to enemy lines
                self.orc_positions[1]['in_combat'] = True
                self.orcs_in_combat += 1
                reward += 25
        
        elif strategy == 'Magic Resistance':
            # Increase resilience against shooting
            for orc in self.orc_positions:
                orc['ward_save'] = 0.25  # 25% ward save
                reward += 2
        
        # Update strategy success rates
        self.orc_strategies[strategy]['success_rate'] += reward / 100.0
        
        return reward
    
    def _artillery_phase(self):
        """Execute Empire artillery shooting"""
        casualties = 0
        
        for art_pos in self.artillery_positions:
            if not art_pos['active']:
                continue
                
            art = art_pos['unit']
            
            # Find targets in range
            targets = []
            for i, orc in enumerate(self.orc_positions):
                if not orc['alive'] or orc['in_combat']:
                    continue
                    
                distance = math.sqrt((art_pos['x'] - orc['x'])**2 + (art_pos['y'] - orc['y'])**2)
                if distance <= art['range']:
                    targets.append((i, orc, distance))
            
            if targets:
                # Target closest unit
                target_idx, target_orc, distance = min(targets, key=lambda x: x[2])
                
                # Apply casualties based on weapon strength and distance
                hit_chance = max(0.2, min(0.8, art['strength'] / 10.0 - distance / art['range']))
                ward_save = target_orc.get('ward_save', 0.0)
                
                if random.random() < hit_chance * (1 - ward_save):
                    damage = random.randint(1, art['strength'] // 2)
                    target_orc['casualties'] += damage
                    casualties += damage
                    self.artillery_shots_fired += 1
                    
                    if target_orc['casualties'] >= 8:  # Unit destroyed
                        target_orc['alive'] = False
                        print(f"💥 {art['name']} destroys Orc unit!")
        
        return casualties

class DQNNetwork(nn.Module):
    """Deep Q-Network for anti-artillery tactics"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AntiArtilleryAI:
    """AI Agent specialized in countering artillery"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_network()
        
        # Strategy tracking
        self.successful_strategies = []
        self.failed_strategies = []
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
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
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'successful_strategies': self.successful_strategies,
            'failed_strategies': self.failed_strategies
        }, filepath)

def train_anti_artillery_ai(episodes=3000):
    """Train AI to counter artillery tactics"""
    print("🧌 TRAINING ANTI-ARTILLERY ORC AI")
    print("="*50)
    print("🎯 Mission: Neutralize Empire Artillery")
    print("🏹 Counter-strategies:")
    print("   • Fast Flanking")
    print("   • Magic Disruption") 
    print("   • Rapid Advance")
    print("   • Troll Regeneration")
    print("   • Goblin Distraction")
    print("   • Underground Advance")
    print("   • Flying Attack")
    print("   • Magic Resistance")
    print()
    
    env = AntiArtilleryEnvironment()
    state = env.reset()
    state_size = len(state)
    action_size = 8  # 8 anti-artillery strategies
    
    agent = AntiArtilleryAI(state_size, action_size)
    
    scores = []
    victories = 0
    strategy_success = {name: 0 for name in env.orc_strategies.keys()}
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_strategies = []
        
        for step in range(20):  # Max 20 actions per battle
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            episode_strategies.append(info.get('strategy', 'Unknown'))
            
            total_reward += reward
            state = next_state
            
            if done:
                if reward > 0:  # Victory conditions
                    victories += 1
                    # Track successful strategies
                    for strategy in episode_strategies:
                        if strategy in strategy_success:
                            strategy_success[strategy] += 1
                break
        
        agent.replay()
        scores.append(total_reward)
        
        # Update target network
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Progress reporting
        if episode % 200 == 0:
            elapsed = time.time() - start_time
            win_rate = (victories / (episode + 1)) * 100
            avg_score = np.mean(scores[-200:]) if len(scores) >= 200 else np.mean(scores)
            
            print(f"Episode {episode:4d} | Win Rate: {win_rate:5.1f}% | Avg Score: {avg_score:6.1f} | ε: {agent.epsilon:.3f}")
            
            if episode > 0:
                # Show most successful counter-strategy
                best_strategy = max(strategy_success, key=strategy_success.get)
                print(f"  🏆 Best Counter: {best_strategy} ({strategy_success[best_strategy]} victories)")
                print(f"  ⏱️  Training Time: {elapsed/60:.1f} minutes")
    
    # Final results
    final_win_rate = (victories / episodes) * 100
    final_score = np.mean(scores[-500:]) if len(scores) >= 500 else np.mean(scores)
    
    print("\n🎯 ANTI-ARTILLERY TRAINING COMPLETE!")
    print("="*50)
    print(f"📈 Final Win Rate: {final_win_rate:.2f}%")
    print(f"📊 Average Score: {final_score:.1f}")
    print(f"🏆 Total Victories: {victories}")
    print(f"💀 Total Defeats: {episodes - victories}")
    print(f"⏱️  Total Training Time: {(time.time() - start_time)/60:.1f} minutes")
    print()
    
    print("🧠 DISCOVERED ANTI-ARTILLERY TACTICS:")
    sorted_strategies = sorted(strategy_success.items(), key=lambda x: x[1], reverse=True)
    for strategy, count in sorted_strategies[:5]:
        success_rate = (count / victories) * 100 if victories > 0 else 0
        print(f"  • {strategy}: {count} victories ({success_rate:.1f}% of wins)")
    
    # Save the improved model
    agent.save_model('orc_anti_artillery_ai.pth')
    print(f"\n💾 Enhanced Orc AI saved as 'orc_anti_artillery_ai.pth'")
    
    return agent

if __name__ == "__main__":
    print("🔥 STARTING ANTI-ARTILLERY AI TRAINING")
    print("🎯 Teaching Orcs to counter Empire gunlines...")
    print()
    
    # Train the specialized anti-artillery AI
    trained_agent = train_anti_artillery_ai(3000)
    
    print("\n🧌 Orcs now know how to fight artillery!")
    print("💪 Ready to face the Empire gunline!") 