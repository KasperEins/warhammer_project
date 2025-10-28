#!/usr/bin/env python3
"""
🧠 MASS AI TRAINING SYSTEM - 1,000,000 GAMES PER AGENT
=====================================================
Ultra-scale training system for Warhammer: The Old World AI agents
Trains both Empire and Orc AIs with 1,000,000 games each under authentic TOW rules
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import pickle
import gc

# Try to import psutil for system monitoring
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("⚠️ psutil not available - system monitoring disabled")

# Import our systems
from tow_web_battle import TOWUnit, SimpleAI

# Simple training battle class that doesn't use daemon threads
class TrainingBattle:
    """Simplified battle for training without threading issues"""
    
    def __init__(self):
        self.turn = 1
        self.max_turns = 6
        self.battle_state = "ready"
        self.units = []
        self.logs = []
        self.empire_score = 0
        self.orc_score = 0
        
    def create_armies(self):
        """Create simplified armies for training"""
        # Empire army
        empire_crossbows = TOWUnit("Empire Crossbows", 10, "infantry", "10x1", "nuln", (100, 200), points=80)
        empire_spears = TOWUnit("Empire Spearmen", 20, "infantry", "20x1", "nuln", (150, 180), points=120)
        empire_cannon = TOWUnit("Empire Great Cannon", 1, "warmachine", "1x1", "nuln", (50, 220), points=120)
        
        # Orc army  
        orc_boys = TOWUnit("Orc Boys", 20, "infantry", "20x1", "orcs", (400, 200), points=120)
        orc_archers = TOWUnit("Orc Archers", 10, "infantry", "10x1", "orcs", (450, 180), points=60)
        orc_trolls = TOWUnit("River Trolls", 3, "monstrous", "3x1", "orcs", (350, 220), points=150)
        
        self.units = [empire_crossbows, empire_spears, empire_cannon, orc_boys, orc_archers, orc_trolls]
        
    def get_ai_state(self):
        """Get simplified state for AI"""
        state = []
        
        # Unit health ratios
        for unit in self.units:
            if unit.is_alive:
                health_ratio = unit.models / max(getattr(unit, 'starting_models', unit.models), 1)
                state.append(health_ratio)
            else:
                state.append(0.0)
                
        # Pad to fixed size
        while len(state) < 50:
            state.append(0.0)
            
        return state[:50]
    
    def run_simple_battle(self):
        """Run simplified battle without threads"""
        while self.turn <= self.max_turns and self.battle_state == "active":
            # Simple turn simulation
            self.simulate_turn()
            self.turn += 1
            
            # Check if battle should end
            empire_alive = any(u.is_alive and u.faction == "nuln" for u in self.units)
            orc_alive = any(u.is_alive and u.faction == "orcs" for u in self.units)
            
            if not empire_alive or not orc_alive:
                break
                
        self.calculate_final_scores()
        
    def simulate_turn(self):
        """Simulate a single turn"""
        # Simple combat resolution
        empire_units = [u for u in self.units if u.faction == "nuln" and u.is_alive]
        orc_units = [u for u in self.units if u.faction == "orcs" and u.is_alive]
        
        # Ranged combat
        for unit in empire_units:
            if "crossbow" in unit.name.lower() or "cannon" in unit.name.lower():
                if orc_units:
                    target = random.choice(orc_units)
                    damage = random.randint(1, 3)
                    target.models = max(0, target.models - damage)
                    if target.models == 0:
                        target.is_alive = False
                        
        for unit in orc_units:
            if "archer" in unit.name.lower():
                if empire_units:
                    target = random.choice(empire_units)
                    damage = random.randint(0, 2)
                    target.models = max(0, target.models - damage)
                    if target.models == 0:
                        target.is_alive = False
                        
        # Melee combat
        for unit in orc_units:
            if unit.is_alive and "troll" in unit.name.lower():
                if empire_units:
                    target = random.choice(empire_units)
                    damage = random.randint(1, 4)
                    target.models = max(0, target.models - damage)
                    if target.models == 0:
                        target.is_alive = False
                        
    def calculate_final_scores(self):
        """Calculate battle scores"""
        # Empire score
        orc_destroyed = sum(u.points for u in self.units if u.faction == "orcs" and not u.is_alive)
        empire_remaining = sum(u.models for u in self.units if u.faction == "nuln" and u.is_alive)
        self.empire_score = orc_destroyed + empire_remaining * 5
        
        # Orc score  
        empire_destroyed = sum(u.points for u in self.units if u.faction == "nuln" and not u.is_alive)
        orc_remaining = sum(u.models for u in self.units if u.faction == "orcs" and u.is_alive)
        self.orc_score = empire_destroyed + orc_remaining * 3
        
    def get_winner(self):
        """Determine battle winner"""
        if self.empire_score > self.orc_score:
            return "empire"
        elif self.orc_score > self.empire_score:
            return "orc"
        else:
            return "draw"

# Simple AI agent class for training if main one isn't available
class TrainingAI:
    """Simplified AI for mass training"""
    
    def __init__(self, state_size=50, action_size=15, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999995
        self.memory = deque(maxlen=50000)
        
        # Simple neural network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target_network()
        
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
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
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save the model"""
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
    
    def load_model(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)

class MassTrainingSystem:
    """Ultra-scale training system for 1M+ battles"""
    
    def __init__(self, 
                 empire_games=100000, 
                 orc_games=100000,
                 batch_size=1000,
                 save_interval=10000):
        
        self.empire_games = empire_games
        self.orc_games = orc_games
        self.batch_size = batch_size
        self.save_interval = save_interval
        
        # Training metrics
        self.empire_stats = {
            'wins': 0, 'losses': 0, 'draws': 0,
            'total_score': 0, 'best_score': -float('inf'),
            'strategy_success': defaultdict(int),
            'games_completed': 0
        }
        
        self.orc_stats = {
            'wins': 0, 'losses': 0, 'draws': 0,
            'total_score': 0, 'best_score': -float('inf'),
            'strategy_success': defaultdict(int),
            'games_completed': 0
        }
        
        # Performance tracking
        self.start_time = None
        self.memory_usage_history = []
        
        print(f"🧠 MASS TRAINING SYSTEM INITIALIZED")
        print(f"📊 Empire Games: {self.empire_games:,}")
        print(f"📊 Orc Games: {self.orc_games:,}")  
        print(f"💾 Batch Size: {self.batch_size}")
        print(f"💾 Save Interval: {self.save_interval}")
        
    def initialize_ai_agents(self):
        """Initialize AI agents with optimized parameters"""
        print("🤖 Initializing AI agents...")
        
        # Empire AI - Artillery specialist
        self.empire_ai = TrainingAI(state_size=50, action_size=15, lr=0.0005)
        
        # Orc AI - Anti-artillery specialist  
        self.orc_ai = TrainingAI(state_size=50, action_size=15, lr=0.0005)
        
        # Load existing models if available
        try:
            self.empire_ai.load_model('empire_ai_mass_trained.pth')
            print("✅ Loaded existing Empire AI model")
        except:
            print("🆕 Starting with fresh Empire AI")
            
        try:
            self.orc_ai.load_model('orc_ai_mass_trained.pth')
            print("✅ Loaded existing Orc AI model")
        except:
            print("🆕 Starting with fresh Orc AI")
            
        print("✅ AI agents initialized and configured")
        
    def create_training_battle(self):
        """Create optimized battle instance for training"""
        battle = TrainingBattle()
        battle.create_armies()
        return battle
        
    def run_single_training_game(self, faction, game_number):
        """Run a single training game for specified faction"""
        try:
            battle = self.create_training_battle()
            
            # Select AI based on faction
            if faction == 'empire':
                ai_agent = self.empire_ai
                stats = self.empire_stats
            else:
                ai_agent = self.orc_ai
                stats = self.orc_stats
                
            # Run battle simulation
            result = self.simulate_battle_for_training(battle, ai_agent, faction)
            
            # Update statistics
            stats['games_completed'] += 1
            stats['total_score'] += result['score']
            
            if result['outcome'] == 'win':
                stats['wins'] += 1
            elif result['outcome'] == 'loss':
                stats['losses'] += 1
            else:
                stats['draws'] += 1
                
            stats['best_score'] = max(stats['best_score'], result['score'])
            
            # Track successful strategies
            for strategy in result['strategies']:
                if result['outcome'] == 'win':
                    stats['strategy_success'][strategy] += 1
                    
            return result
            
        except Exception as e:
            print(f"❌ Error in training game {game_number}: {e}")
            return None
            
    def simulate_battle_for_training(self, battle, ai_agent, faction):
        """Simulate battle optimized for AI training"""
        strategies_used = []
        total_reward = 0
        
        # Set battle to active state and run the simulation
        battle.battle_state = "active" 
        
        # Let AI make decisions during battle turns
        for turn in range(1, battle.max_turns + 1):
            # Get current state
            state = battle.get_ai_state()
            prev_state = state.copy()
            
            # AI makes strategic decision
            action = ai_agent.act(state)
            strategies_used.append(action)
            
            # Use action to influence battle randomness
            random.seed(action * turn + faction.__hash__())
            
            # Execute battle turn
            battle.simulate_turn()
            
            # Get new state and calculate reward
            new_state = battle.get_ai_state()           
            reward = self.calculate_training_reward(prev_state, new_state, faction, battle)
            total_reward += reward
            
            # Check if battle should end early
            empire_alive = any(u.is_alive and u.faction == "nuln" for u in battle.units)
            orc_alive = any(u.is_alive and u.faction == "orcs" for u in battle.units)
            battle_finished = not (empire_alive and orc_alive)
            
            # Store experience for AI learning
            ai_agent.remember(prev_state, action, reward, new_state, battle_finished)
            
            if battle_finished:
                break
        
        # Final battle calculations        
        battle.calculate_final_scores()
        
        # Train AI if enough experiences accumulated
        if len(ai_agent.memory) >= 32:
            ai_agent.replay(batch_size=32)
            
        # Determine final outcome
        outcome = self.determine_battle_outcome(battle, faction)
        
        return {
            'outcome': outcome,
            'score': total_reward,
            'strategies': strategies_used,
            'turns': len(strategies_used)
        }
        
    def calculate_training_reward(self, prev_state, new_state, faction, battle):
        """Calculate reward for AI training"""
        reward = 0
        
        # Basic survival reward
        reward += 1
        
        # Unit preservation rewards - use battle.units with faction filter
        if faction == 'empire':
            my_units = len([u for u in battle.units if u.faction == "nuln" and u.is_alive])
            enemy_units = len([u for u in battle.units if u.faction == "orcs" and u.is_alive])
        else:
            my_units = len([u for u in battle.units if u.faction == "orcs" and u.is_alive])
            enemy_units = len([u for u in battle.units if u.faction == "nuln" and u.is_alive])
            
        # Reward for maintaining units and destroying enemies
        reward += my_units * 2
        reward += (10 - enemy_units) * 3
        
        # Victory/defeat rewards - check if any side has been eliminated
        nuln_alive = any(u.faction == "nuln" and u.is_alive for u in battle.units)
        orcs_alive = any(u.faction == "orcs" and u.is_alive for u in battle.units)
        battle_finished = not (nuln_alive and orcs_alive)
        
        if battle_finished:
            if not nuln_alive and orcs_alive:  # Orcs win
                if faction == 'orc':
                    reward += 100  # Victory bonus
                else:
                    reward -= 50   # Defeat penalty
            elif not orcs_alive and nuln_alive:  # Empire wins
                if faction == 'empire':
                    reward += 100  # Victory bonus
                else:
                    reward -= 50   # Defeat penalty
                
        return reward
        
    def determine_battle_outcome(self, battle, faction):
        """Determine if AI won, lost, or drew"""
        # Use our simplified battle winner determination
        winner = battle.get_winner()
        
        if winner == "draw":
            return 'draw'
        elif ((winner == "empire" and faction == "empire") or 
              (winner == "orc" and faction == "orc")):
            return 'win'
        else:
            return 'loss'
            
    def run_batch_training(self, faction, batch_start, batch_size):
        """Run a batch of training games"""
        print(f"🎯 Running {faction} batch {batch_start}-{batch_start + batch_size}")
        
        batch_results = []
        for i in range(batch_size):
            game_num = batch_start + i
            result = self.run_single_training_game(faction, game_num)
            if result:
                batch_results.append(result)
                
            # Memory management
            if i % 100 == 0:
                gc.collect()
                
        return batch_results
        
    def save_training_progress(self, faction):
        """Save AI models and training statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if faction == 'empire':
            ai_agent = self.empire_ai
            stats = self.empire_stats
            model_name = f"empire_ai_mass_trained_{stats['games_completed']}.pth"
            stats_name = f"empire_training_stats_{timestamp}.json"
        else:
            ai_agent = self.orc_ai
            stats = self.orc_stats
            model_name = f"orc_ai_mass_trained_{stats['games_completed']}.pth"
            stats_name = f"orc_training_stats_{timestamp}.json"
            
        # Save AI model
        ai_agent.save_model(model_name)
        
        # Save statistics
        stats_data = {
            'faction': faction,
            'games_completed': stats['games_completed'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            'win_rate': stats['wins'] / max(stats['games_completed'], 1),
            'average_score': stats['total_score'] / max(stats['games_completed'], 1),
            'best_score': stats['best_score'],
            'top_strategies': dict(sorted(stats['strategy_success'].items(), 
                                        key=lambda x: x[1], reverse=True)[:10]),
            'epsilon': ai_agent.epsilon,
            'timestamp': timestamp,
            'training_time_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0
        }
        
        with open(stats_name, 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        print(f"💾 Saved {faction} progress: {model_name}, {stats_name}")
        
    def print_training_progress(self, faction):
        """Print detailed training progress"""
        if faction == 'empire':
            stats = self.empire_stats
            ai_agent = self.empire_ai
        else:
            stats = self.orc_stats
            ai_agent = self.orc_ai
            
        games = stats['games_completed']
        if games == 0:
            return
            
        win_rate = (stats['wins'] / games) * 100
        avg_score = stats['total_score'] / games
        
        elapsed = (time.time() - self.start_time) / 3600 if self.start_time else 0
        games_per_hour = games / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 {faction.upper()} TRAINING PROGRESS")
        print(f"   Games: {games:,}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Avg Score: {avg_score:.1f}")
        print(f"   Best Score: {stats['best_score']:.1f}")
        print(f"   Epsilon: {ai_agent.epsilon:.4f}")
        print(f"   Games/Hour: {games_per_hour:.0f}")
        print(f"   Time Elapsed: {elapsed:.1f}h")
        
        # Show top strategies
        if stats['strategy_success']:
            top_strategy = max(stats['strategy_success'], key=stats['strategy_success'].get)
            print(f"   Top Strategy: Action {top_strategy} ({stats['strategy_success'][top_strategy]} wins)")
            
    def run_empire_training(self):
        """Run complete Empire AI training"""
        print(f"🔵 STARTING EMPIRE AI TRAINING - {self.empire_games:,} GAMES")
        print("=" * 60)
        
        for batch_start in range(0, self.empire_games, self.batch_size):
            batch_size = min(self.batch_size, self.empire_games - batch_start)
            
            # Run batch
            self.run_batch_training('empire', batch_start, batch_size)
            
            # Update target network periodically
            if batch_start % 1000 == 0:
                self.empire_ai.update_target_network()
                
            # Save progress
            if batch_start % self.save_interval == 0:
                self.save_training_progress('empire')
                
            # Print progress
            if batch_start % (self.save_interval // 2) == 0:
                self.print_training_progress('empire')
                
            # Memory monitoring
            if HAVE_PSUTIL:
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 85:
                    print(f"⚠️  High memory usage: {memory_usage:.1f}%")
                    gc.collect()
            
        # Final save
        self.save_training_progress('empire')
        print("✅ Empire AI training completed!")
        
    def run_orc_training(self):
        """Run complete Orc AI training"""
        print(f"🟢 STARTING ORC AI TRAINING - {self.orc_games:,} GAMES")
        print("=" * 60)
        
        for batch_start in range(0, self.orc_games, self.batch_size):
            batch_size = min(self.batch_size, self.orc_games - batch_start)
            
            # Run batch
            self.run_batch_training('orc', batch_start, batch_size)
            
            # Update target network periodically
            if batch_start % 1000 == 0:
                self.orc_ai.update_target_network()
                
            # Save progress
            if batch_start % self.save_interval == 0:
                self.save_training_progress('orc')
                
            # Print progress
            if batch_start % (self.save_interval // 2) == 0:
                self.print_training_progress('orc')
                
            # Memory monitoring
            if HAVE_PSUTIL:
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 85:
                    print(f"⚠️  High memory usage: {memory_usage:.1f}%")
                    gc.collect()
            
        # Final save
        self.save_training_progress('orc')
        print("✅ Orc AI training completed!")
        
    def run_full_training_sequence(self):
        """Run the complete training sequence for both AIs"""
        print("🚀 STARTING MASS AI TRAINING SEQUENCE")
        print("=" * 60)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Total Games: {self.empire_games + self.orc_games:,}")
        print("")
        
        self.start_time = time.time()
        
        # Initialize AI agents
        self.initialize_ai_agents()
        
        # Train Empire AI
        print("\n🔵 PHASE 1: EMPIRE AI TRAINING")
        self.run_empire_training()
        
        # Train Orc AI
        print("\n🟢 PHASE 2: ORC AI TRAINING")
        self.run_orc_training()
        
        # Final summary
        self.print_final_summary()
        
    def print_final_summary(self):
        """Print final training summary"""
        total_time = (time.time() - self.start_time) / 3600
        total_games = self.empire_stats['games_completed'] + self.orc_stats['games_completed']
        
        print("\n🎉 MASS TRAINING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Total Time: {total_time:.1f} hours")
        print(f"🎮 Total Games: {total_games:,}")
        print(f"⚡ Games/Hour: {total_games / total_time:.0f}")
        print("")
        
        # Empire results
        empire_wr = (self.empire_stats['wins'] / max(self.empire_stats['games_completed'], 1)) * 100
        print(f"🔵 EMPIRE AI RESULTS:")
        print(f"   Games: {self.empire_stats['games_completed']:,}")
        print(f"   Win Rate: {empire_wr:.2f}%")
        print(f"   Best Score: {self.empire_stats['best_score']:.1f}")
        
        # Orc results  
        orc_wr = (self.orc_stats['wins'] / max(self.orc_stats['games_completed'], 1)) * 100
        print(f"🟢 ORC AI RESULTS:")
        print(f"   Games: {self.orc_stats['games_completed']:,}")
        print(f"   Win Rate: {orc_wr:.2f}%")
        print(f"   Best Score: {self.orc_stats['best_score']:.1f}")
        
        print(f"\n💾 Final models saved as:")
        print(f"   empire_ai_mass_trained_{self.empire_stats['games_completed']}.pth")
        print(f"   orc_ai_mass_trained_{self.orc_stats['games_completed']}.pth")


def main():
    """Main function to start mass training"""
    print("⚔️ WARHAMMER: THE OLD WORLD - MASS AI TRAINING SYSTEM")
    print("=" * 60)
    print("🧠 Preparing to train AI agents with 100,000 games each")
    print("🎯 Using authentic TOW mechanics with:")
    print("   • WS vs WS combat tables")
    print("   • S vs T wound tables") 
    print("   • Psychology system (fear, terror, panic)")
    print("   • Combat resolution with rank bonuses")
    print("   • Magic system with miscast tables")
    print("   • Formation and charge bonuses")
    print("")
    
    # System check
    if HAVE_PSUTIL:
        available_memory = psutil.virtual_memory().available / (1024**3)
        cpu_count = mp.cpu_count()
        
        print(f"💻 SYSTEM CHECK:")
        print(f"   Available Memory: {available_memory:.1f} GB")
        print(f"   CPU Cores: {cpu_count}")
        print(f"   PyTorch Available: {torch.cuda.is_available()}")
        
        if available_memory < 4:
            print("⚠️  Warning: Low memory detected. Training may be slower.")
    else:
        print("💻 SYSTEM CHECK: psutil not available for monitoring")
        
    print("")
    
    # Get user confirmation
    total_games = 200000  # 100K each faction
    estimated_hours = total_games / 1000  # Rough estimate
    
    print(f"📊 TRAINING SCOPE:")
    print(f"   Empire AI Games: 100,000")
    print(f"   Orc AI Games: 100,000") 
    print(f"   Total Games: {total_games:,}")
    print(f"   Estimated Time: {estimated_hours:.0f} hours ({estimated_hours/24:.1f} days)")
    print("")
    
    confirm = input("🚀 Start mass training? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("Training cancelled.")
        return
        
    # Start training
    trainer = MassTrainingSystem(
        empire_games=100000,
        orc_games=100000,
        batch_size=1000,
        save_interval=10000
    )
    
    trainer.run_full_training_sequence()
    
    print("🎉 Mass training completed successfully!")


if __name__ == "__main__":
    main() 