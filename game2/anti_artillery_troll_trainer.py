#!/usr/bin/env python3
"""
ANTI-ARTILLERY TROLL AI TRAINER
===============================
Specialized training to counter Empire artillery tactics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque
from warhammer_ai_agent import DQNNetwork

class AntiArtilleryBattleEnvironment:
    """Training environment specifically for countering artillery-heavy armies"""
    
    def __init__(self):
        # Empire Artillery setup (the enemy)
        self.empire_artillery = [
            {'name': 'Great Cannon', 'range': 48, 'strength': 10, 'models': 4, 'active': True},
            {'name': 'Great Cannon', 'range': 48, 'strength': 10, 'models': 4, 'active': True},
            {'name': 'Helblaster', 'range': 24, 'strength': 5, 'models': 3, 'active': True},
            {'name': 'Helblaster', 'range': 24, 'strength': 5, 'models': 3, 'active': True},
            {'name': 'Helblaster', 'range': 24, 'strength': 5, 'models': 3, 'active': True},
            {'name': 'War Wagon', 'range': 20, 'strength': 4, 'models': 4, 'active': True}
        ]
        
        # Enhanced action space for anti-artillery tactics
        self.anti_artillery_actions = [
            "Fast Flanking March",      # 0 - Move fast to avoid fire lanes
            "Underground Advance",      # 1 - Tunneling goblins
            "Magic Disruption",         # 2 - Weirdnob disrupts artillery
            "Flying Assault",           # 3 - Wyvern direct attack
            "Troll Regeneration",       # 4 - Heal casualties
            "Decoy Swarm",             # 5 - Small units distract
            "Shield Wall Advance",      # 6 - Protected advance
            "Night Attack",            # 7 - Stealth approach
            "Magic Resistance",        # 8 - Ward saves
            "Massed Charge",          # 9 - All-out assault
            "Artillery Hunt",         # 10 - Target enemy guns
            "Smoke Screen",           # 11 - Block line of sight
            "Terror Tactics"          # 12 - Psychological warfare
        ]
        
        self.reset()
    
    def reset(self):
        """Reset battle with artillery-focused scenario"""
        # Empire deployment (artillery line)
        self.empire_units = []
        for i, art in enumerate(self.empire_artillery):
            x = random.randint(5, 20)  # Back line deployment
            y = 12 + (i * 4)  # Spread out to avoid counter-battery
            self.empire_units.append({
                'x': x, 'y': y, 'name': art['name'],
                'range': art['range'], 'strength': art['strength'],
                'models': art['models'], 'active': True,
                'shots_fired': 0, 'hits_scored': 0
            })
        
        # Orc deployment (attacking force)
        self.orc_units = [
            {'name': 'Orc Bigboss on Boar Chariot', 'x': 60, 'y': 12, 'models': 3, 'alive': True, 'in_combat': False},
            {'name': 'Orc Warboss on Wyvern', 'x': 65, 'y': 15, 'models': 1, 'alive': True, 'in_combat': False},
            {'name': 'Orc Weirdnob Wizard', 'x': 55, 'y': 10, 'models': 1, 'alive': True, 'in_combat': False},
            {'name': '8 Common Trolls', 'x': 58, 'y': 8, 'models': 8, 'alive': True, 'in_combat': False},
            {'name': '27 Orc Boys with Warbows', 'x': 62, 'y': 20, 'models': 27, 'alive': True, 'in_combat': False},
            {'name': '4 River Trolls', 'x': 60, 'y': 25, 'models': 4, 'alive': True, 'in_combat': False},
            {'name': '4 River Trolls', 'x': 58, 'y': 5, 'models': 4, 'alive': True, 'in_combat': False},
            {'name': '4 River Trolls', 'x': 65, 'y': 18, 'models': 4, 'alive': True, 'in_combat': False}
        ]
        
        self.turn = 1
        self.total_casualties = 0
        self.artillery_neutralized = 0
        self.orcs_in_combat = 0
        self.magic_cast = 0
        self.successful_tactics = []
        
        return self._get_state()
    
    def _get_state(self):
        """Get detailed state for anti-artillery training"""
        state = []
        
        # Battle overview
        state.extend([
            self.turn / 6.0,  # Turn progression
            self.total_casualties / 50.0,  # Casualty rate
            self.artillery_neutralized / 6.0,  # Artillery destroyed
            self.orcs_in_combat / 8.0,  # Units in melee
            self.magic_cast / 10.0  # Magic usage
        ])
        
        # Empire artillery status (6 units)
        for i in range(6):
            if i < len(self.empire_units):
                unit = self.empire_units[i]
                state.extend([
                    unit['x'] / 72.0,
                    unit['y'] / 48.0,
                    1.0 if unit['active'] else 0.0,
                    unit['shots_fired'] / 20.0,
                    unit['strength'] / 10.0
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Orc army status (8 units)
        for i in range(8):
            if i < len(self.orc_units):
                unit = self.orc_units[i]
                state.extend([
                    unit['x'] / 72.0,
                    unit['y'] / 48.0,
                    1.0 if unit['alive'] else 0.0,
                    unit['models'] / 30.0,  # Normalized unit size
                    1.0 if unit['in_combat'] else 0.0
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute anti-artillery action"""
        reward = 0
        info = {'action': 'none', 'result': 'failed'}
        
        if action < len(self.anti_artillery_actions):
            action_name = self.anti_artillery_actions[action]
            reward, result = self._execute_anti_artillery_action(action)
            info = {'action': action_name, 'result': result}
        
        # Empire artillery response
        artillery_casualties = self._empire_artillery_phase()
        reward -= artillery_casualties * 15  # Heavy penalty for losses
        
        # Check battle outcome
        done = False
        orcs_alive = sum(1 for orc in self.orc_units if orc['alive'])
        artillery_active = sum(1 for art in self.empire_units if art['active'])
        
        # Victory conditions
        if artillery_active == 0:
            reward += 200  # Major victory - artillery destroyed
            info['result'] = 'VICTORY - Artillery Neutralized!'
            done = True
        elif self.orcs_in_combat >= 5:
            reward += 150  # Victory - engaged in melee
            info['result'] = 'VICTORY - Melee Engaged!'
            done = True
        elif orcs_alive <= 2:
            reward -= 200  # Defeat - army destroyed
            info['result'] = 'DEFEAT - Army Destroyed'
            done = True
        elif self.turn >= 6:
            # Evaluate final position
            if self.orcs_in_combat >= 3:
                reward += 100  # Partial success
                info['result'] = 'Partial Victory'
            elif artillery_active <= 3:
                reward += 50   # Damaged artillery
                info['result'] = 'Artillery Damaged'
            else:
                reward -= 100  # Failed to advance
                info['result'] = 'Failed Advance'
            done = True
        
        self.turn += 1
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _execute_anti_artillery_action(self, action):
        """Execute specific anti-artillery tactics"""
        action_name = self.anti_artillery_actions[action]
        reward = 0
        result = "failed"
        
        if action_name == "Fast Flanking March":
            # Move units to flanks to avoid artillery
            for i, orc in enumerate(self.orc_units[:4]):  # First half flanks
                if orc['alive']:
                    old_y = orc['y']
                    if old_y > 15:
                        orc['y'] = min(orc['y'] + 8, 45)  # North flank
                    else:
                        orc['y'] = max(orc['y'] - 8, 3)   # South flank
                    orc['x'] = max(orc['x'] - 6, 0)  # Advance while flanking
                    reward += 8
            result = "flanking"
            
        elif action_name == "Underground Advance":
            # Goblins tunnel under artillery
            if random.random() < 0.7:  # 70% success
                for orc in self.orc_units[4:]:  # Goblin units
                    if orc['alive']:
                        orc['x'] = max(orc['x'] - 15, 5)  # Surprise advance
                        reward += 12
                result = "tunneling"
                
        elif action_name == "Magic Disruption":
            # Weirdnob disrupts artillery
            if self.orc_units[2]['alive'] and random.random() < 0.6:
                target_artillery = random.choice([art for art in self.empire_units if art['active']])
                target_artillery['active'] = False
                self.artillery_neutralized += 1
                self.magic_cast += 1
                reward += 30
                result = f"magic disrupts {target_artillery['name']}"
                
        elif action_name == "Flying Assault":
            # Wyvern attacks artillery directly
            if self.orc_units[1]['alive']:  # Wyvern rider
                self.orc_units[1]['x'] = 15  # Direct assault
                self.orc_units[1]['in_combat'] = True
                self.orcs_in_combat += 1
                # Chance to destroy artillery piece
                if random.random() < 0.4:
                    target = random.choice([art for art in self.empire_units if art['active']])
                    target['active'] = False
                    self.artillery_neutralized += 1
                    reward += 40
                result = "aerial assault"
                
        elif action_name == "Troll Regeneration":
            # Trolls regenerate casualties
            for i in [0, 3, 5, 6, 7]:  # Troll units
                if i < len(self.orc_units) and self.orc_units[i]['alive']:
                    if self.orc_units[i]['models'] < 8:  # Damaged
                        self.orc_units[i]['models'] = min(self.orc_units[i]['models'] + 2, 8)
                        reward += 5
            result = "regeneration"
            
        elif action_name == "Decoy Swarm":
            # Small units create distractions
            for art in self.empire_units:
                if random.random() < 0.25:  # 25% chance per gun
                    art['shots_fired'] += 1  # Wasted shot
                    reward += 3
            result = "distraction"
            
        elif action_name == "Shield Wall Advance":
            # Protected advance formation
            for orc in self.orc_units:
                if orc['alive']:
                    orc['x'] = max(orc['x'] - 4, 0)  # Slow but protected advance
                    # Increased protection against shooting
                    orc['protection'] = 0.3
                    reward += 4
            result = "protected advance"
            
        elif action_name == "Massed Charge":
            # All units charge at once
            for orc in self.orc_units:
                if orc['alive']:
                    orc['x'] = max(orc['x'] - 12, 0)  # Fast advance
                    if orc['x'] <= 20:  # Reached combat
                        orc['in_combat'] = True
                        self.orcs_in_combat += 1
                        reward += 15
            result = "mass charge"
            
        elif action_name == "Artillery Hunt":
            # Specifically target enemy artillery
            if random.random() < 0.5:
                target = random.choice([art for art in self.empire_units if art['active']])
                target['active'] = False
                self.artillery_neutralized += 1
                reward += 35
                result = f"destroyed {target['name']}"
        
        return reward, result
    
    def _empire_artillery_phase(self):
        """Empire shoots at advancing Orcs"""
        total_casualties = 0
        
        for artillery in self.empire_units:
            if not artillery['active']:
                continue
                
            # Find targets in range
            targets = []
            for orc in self.orc_units:
                if not orc['alive'] or orc['in_combat']:
                    continue
                    
                distance = abs(artillery['x'] - orc['x']) + abs(artillery['y'] - orc['y'])
                if distance <= artillery['range']:
                    targets.append(orc)
            
            if targets:
                target = random.choice(targets)
                protection = target.get('protection', 0.0)
                
                # Artillery effectiveness
                if artillery['name'] == 'Great Cannon':
                    if random.random() < (0.8 - protection):
                        casualties = random.randint(3, 8)
                        target['models'] = max(0, target['models'] - casualties)
                        total_casualties += casualties
                        artillery['shots_fired'] += 1
                        
                elif 'Helblaster' in artillery['name']:
                    if random.random() < (0.6 - protection):
                        casualties = random.randint(2, 5)
                        target['models'] = max(0, target['models'] - casualties)
                        total_casualties += casualties
                        artillery['shots_fired'] += 1
                        
                elif artillery['name'] == 'War Wagon':
                    if random.random() < (0.5 - protection):
                        casualties = random.randint(1, 3)
                        target['models'] = max(0, target['models'] - casualties)
                        total_casualties += casualties
                        artillery['shots_fired'] += 1
                
                # Check if unit is destroyed
                if target['models'] <= 0:
                    target['alive'] = False
        
        self.total_casualties += total_casualties
        return total_casualties

class AntiArtilleryAI:
    """AI specifically trained to counter artillery"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks optimized for anti-artillery
        self.q_network = DQNNetwork(state_size, action_size, 512).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, 512).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        
        self.update_target_network()
        
        # Track successful tactics
        self.tactic_success = {}
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):
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
            'epsilon': self.epsilon,
            'tactic_success': self.tactic_success
        }, filepath)

def train_anti_artillery_ai(episodes=2000):
    """Train AI to specifically counter artillery tactics"""
    print("🏹 ANTI-ARTILLERY ORC AI TRAINING")
    print("="*60)
    print("Mission: Neutralize Empire Artillery")
    print("Enemy: 2 Great Cannons, 3 Helblasters, 1 War Wagon")
    print("Tactics: Fast Flanking, Magic Disruption, Flying Assault, etc.")
    print()
    
    env = AntiArtilleryBattleEnvironment()
    state = env.reset()
    state_size = len(state)
    action_size = 13  # 13 anti-artillery tactics
    
    agent = AntiArtilleryAI(state_size, action_size)
    
    scores = []
    victories = 0
    tactic_usage = {name: 0 for name in env.anti_artillery_actions}
    tactic_success = {name: 0 for name in env.anti_artillery_actions}
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_tactics = []
        
        for step in range(15):  # Max 15 actions per battle
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            # Track tactics
            if info['action'] != 'none':
                tactic_usage[info['action']] += 1
                episode_tactics.append(info['action'])
                if reward > 0:
                    tactic_success[info['action']] += 1
            
            total_reward += reward
            state = next_state
            
            if done:
                if reward > 0:  # Victory
                    victories += 1
                    # Reward successful tactics used this episode
                    for tactic in episode_tactics:
                        agent.tactic_success[tactic] = agent.tactic_success.get(tactic, 0) + 1
                break
        
        agent.replay()
        scores.append(total_reward)
        
        # Update target network
        if episode % 50 == 0:
            agent.update_target_network()
        
        # Progress reporting
        if episode % 100 == 0:
            elapsed = time.time() - start_time
            win_rate = (victories / (episode + 1)) * 100
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            
            print(f"Episode {episode:4d} | Win Rate: {win_rate:5.1f}% | Avg Score: {avg_score:6.1f} | ε: {agent.epsilon:.3f}")
            
            if episode > 0:
                # Show most effective tactic
                effective_tactics = [(name, tactic_success[name] / max(1, tactic_usage[name])) 
                                   for name in env.anti_artillery_actions]
                effective_tactics.sort(key=lambda x: x[1], reverse=True)
                
                if effective_tactics[0][1] > 0:
                    print(f"  🎯 Best Tactic: {effective_tactics[0][0]} ({effective_tactics[0][1]:.1%} success)")
                print(f"  ⏱️  Training Time: {elapsed/60:.1f} minutes")
    
    # Final analysis
    final_win_rate = (victories / episodes) * 100
    final_score = np.mean(scores[-200:]) if len(scores) >= 200 else np.mean(scores)
    
    print("\n🎯 ANTI-ARTILLERY TRAINING COMPLETE!")
    print("="*60)
    print(f"📈 Final Win Rate: {final_win_rate:.2f}%")
    print(f"📊 Average Score: {final_score:.1f}")
    print(f"🏆 Victories: {victories}/{episodes}")
    print(f"⏱️  Total Training Time: {(time.time() - start_time)/60:.1f} minutes")
    print()
    
    print("🧠 MOST EFFECTIVE ANTI-ARTILLERY TACTICS:")
    effective_tactics = [(name, tactic_success[name] / max(1, tactic_usage[name])) 
                       for name in env.anti_artillery_actions if tactic_usage[name] > 0]
    effective_tactics.sort(key=lambda x: x[1], reverse=True)
    
    for i, (tactic, success_rate) in enumerate(effective_tactics[:8]):
        print(f"  {i+1}. {tactic}: {success_rate:.1%} success ({tactic_success[tactic]} wins)")
    
    # Save the specialized anti-artillery model
    agent.save_model('orc_anti_artillery_ai.pth')
    print(f"\n💾 Anti-Artillery Orc AI saved as 'orc_anti_artillery_ai.pth'")
    
    return agent

if __name__ == "__main__":
    print("🔥 STARTING SPECIALIZED ANTI-ARTILLERY TRAINING")
    print("Teaching Orcs advanced tactics against Empire gunlines...")
    print()
    
    # Train the anti-artillery specialist
    trained_agent = train_anti_artillery_ai(2000)
    
    print("\n🧌 Orcs are now artillery specialists!")
    print("💪 Ready to crush the Empire gunline!") 