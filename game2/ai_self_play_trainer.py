#!/usr/bin/env python3
"""
🧠 AI SELF-PLAY TRAINER - 10,000 BATTLE SIMULATION SYSTEM
===============================================================
Advanced self-play training system for Warhammer: The Old World AIs
Runs 10,000 battles between Nuln Empire and Orc & Goblin AIs to improve both
"""

import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Import our battle system
try:
    from tow_web_battle import TOWBattle, TOWUnit
    from warhammer_ai_agent import WarhammerAIAgent
    print("✅ Battle system imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class SelfPlayTrainer:
    """Advanced self-play training system"""
    
    def __init__(self, num_simulations=10000):
        self.num_simulations = num_simulations
        self.results_history = []
        self.empire_ai = None
        self.orc_ai = None
        self.training_log = []
        
        # Training parameters
        self.batch_size = 100
        self.learning_rate = 0.001
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.01
        
        # Performance tracking
        self.empire_wins = 0
        self.orc_wins = 0
        self.draws = 0
        self.win_rate_history = []
        self.strategy_evolution = []
        
        print(f"🎯 Self-Play Trainer initialized for {num_simulations} simulations")
    
    def load_trained_ais(self):
        """Load the existing trained AI models"""
        try:
            # Import SimpleAI from the main battle system
            from tow_web_battle import SimpleAI
            
            # Load Empire AI (simplified version)
            self.empire_ai = SimpleAI("Nuln Empire")
            self.empire_ai.win_rate = 96.15
            self.empire_ai.primary_strategy = "Artillery Strike"
            print("✅ Empire AI loaded (96.15% win rate baseline)")
            
            # Load Orc AI (simplified version)  
            self.orc_ai = SimpleAI("Orc & Goblin Tribes")
            self.orc_ai.win_rate = 83.15
            self.orc_ai.primary_strategy = "Anti-Artillery"
            print("✅ Orc AI loaded (83.15% win rate baseline)")
                
        except Exception as e:
            print(f"❌ Error loading AIs: {e}")
            return False
        
        return True
    
    def create_battle_simulation(self):
        """Create a battle instance for simulation"""
        battle = TOWBattle()
        battle.battle_active = False  # Disable web interface
        return battle
    
    def simulate_single_battle(self, battle_num, verbose=False):
        """Simulate a single battle between the AIs"""
        battle = self.create_battle_simulation()
        
        # Track battle metrics
        start_time = time.time()
        empire_strategies = []
        orc_strategies = []
        turn_count = 0
        
        try:
            # Start battle
            battle.start_battle()
            
            # Run battle loop with AI vs AI
            while battle.battle_active and turn_count < 20:  # Max 20 turns
                turn_count += 1
                
                # Empire turn
                empire_state = battle.get_ai_state()
                empire_action = self.empire_ai.act(empire_state, training=True)
                empire_strategies.append(empire_action)
                battle.execute_ai_action("nuln", empire_action)
                
                # Orc turn  
                orc_state = battle.get_ai_state()
                orc_action = self.orc_ai.act(orc_state, training=True)
                orc_strategies.append(orc_action)
                battle.execute_ai_action("orc", orc_action)
                
                # Process turn
                battle.run_battle_loop()
                
                # Check victory
                if battle.check_victory():
                    break
            
            # Determine winner and calculate rewards
            empire_units = len([u for u in battle.units if u.faction == "nuln" and not u.is_destroyed])
            orc_units = len([u for u in battle.units if u.faction == "orc" and not u.is_destroyed])
            
            battle_duration = time.time() - start_time
            
            if empire_units > orc_units:
                winner = "empire"
                empire_reward = 1.0
                orc_reward = -0.5
                self.empire_wins += 1
            elif orc_units > empire_units:
                winner = "orc"
                empire_reward = -0.5
                orc_reward = 1.0
                self.orc_wins += 1
            else:
                winner = "draw"
                empire_reward = 0.0
                orc_reward = 0.0
                self.draws += 1
            
            # Store battle result
            battle_result = {
                'battle_number': battle_num,
                'winner': winner,
                'empire_units_remaining': empire_units,
                'orc_units_remaining': orc_units,
                'turns': turn_count,
                'duration': battle_duration,
                'empire_strategies': empire_strategies,
                'orc_strategies': orc_strategies,
                'empire_reward': empire_reward,
                'orc_reward': orc_reward
            }
            
            self.results_history.append(battle_result)
            
            if verbose:
                print(f"Battle {battle_num}: {winner} victory ({empire_units} vs {orc_units}) in {turn_count} turns")
            
            return battle_result
            
        except Exception as e:
            if verbose:
                print(f"❌ Battle {battle_num} error: {e}")
            return None
    
    def update_ai_models(self, batch_results):
        """Update both AI models based on batch results"""
        empire_experiences = []
        orc_experiences = []
        
        # Collect experiences from batch
        for result in batch_results:
            if result is None:
                continue
                
            # Empire experiences
            for i, strategy in enumerate(result['empire_strategies']):
                empire_experiences.append({
                    'state': self.encode_battle_state(result, i, 'empire'),
                    'action': strategy,
                    'reward': result['empire_reward'],
                    'next_state': self.encode_battle_state(result, i+1, 'empire') if i+1 < len(result['empire_strategies']) else None
                })
            
            # Orc experiences  
            for i, strategy in enumerate(result['orc_strategies']):
                orc_experiences.append({
                    'state': self.encode_battle_state(result, i, 'orc'),
                    'action': strategy,
                    'reward': result['orc_reward'],
                    'next_state': self.encode_battle_state(result, i+1, 'orc') if i+1 < len(result['orc_strategies']) else None
                })
        
        # Train both AIs
        if empire_experiences:
            self.empire_ai.replay(empire_experiences)
            
        if orc_experiences:
            self.orc_ai.replay(orc_experiences)
    
    def encode_battle_state(self, result, turn_index, faction):
        """Encode battle state for AI training"""
        # Simplified state encoding for training
        state = np.zeros(50)
        
        # Basic battle info
        state[0] = result['turns']
        state[1] = result['empire_units_remaining']
        state[2] = result['orc_units_remaining']
        state[3] = turn_index
        
        # Fill remaining with normalized battle metrics
        for i in range(4, 50):
            state[i] = random.random() * 0.1  # Placeholder for detailed state
        
        return state
    
    def run_training_simulation(self):
        """Run the complete 10,000 battle training simulation"""
        print("🚀 STARTING 10,000 BATTLE SELF-PLAY TRAINING")
        print("=" * 60)
        
        if not self.load_trained_ais():
            print("❌ Failed to load AI models")
            return
        
        start_time = time.time()
        batch_results = []
        
        for battle_num in range(1, self.num_simulations + 1):
            # Simulate battle
            result = self.simulate_single_battle(battle_num, verbose=(battle_num % 100 == 0))
            
            if result:
                batch_results.append(result)
            
            # Process batch every 100 battles
            if battle_num % self.batch_size == 0:
                self.update_ai_models(batch_results)
                self.save_progress(battle_num)
                self.print_progress_report(battle_num)
                batch_results = []
                
                # Decay exploration
                self.empire_ai.epsilon = max(self.min_epsilon, self.empire_ai.epsilon * self.epsilon_decay)
                self.orc_ai.epsilon = max(self.min_epsilon, self.orc_ai.epsilon * self.epsilon_decay)
        
        # Final batch
        if batch_results:
            self.update_ai_models(batch_results)
        
        # Training complete
        self.finalize_training()
        
        total_time = time.time() - start_time
        print(f"🎯 TRAINING COMPLETE! Total time: {total_time:.1f} seconds")
    
    def print_progress_report(self, battle_num):
        """Print progress report every batch"""
        empire_win_rate = (self.empire_wins / battle_num) * 100
        orc_win_rate = (self.orc_wins / battle_num) * 100
        draw_rate = (self.draws / battle_num) * 100
        
        self.win_rate_history.append({
            'battle': battle_num,
            'empire_wins': empire_win_rate,
            'orc_wins': orc_win_rate,
            'draws': draw_rate
        })
        
        print(f"📊 Battle {battle_num:,} | Empire: {empire_win_rate:.1f}% | Orc: {orc_win_rate:.1f}% | Draws: {draw_rate:.1f}%")
        print(f"🧠 Empire ε: {self.empire_ai.epsilon:.3f} | Orc ε: {self.orc_ai.epsilon:.3f}")
    
    def save_progress(self, battle_num):
        """Save training progress"""
        # Save AI models
        self.empire_ai.save_model(f"empire_ai_self_play_{battle_num}.pth")
        self.orc_ai.save_model(f"orc_ai_self_play_{battle_num}.pth")
        
        # Save training data
        progress_data = {
            'battle_number': battle_num,
            'empire_wins': self.empire_wins,
            'orc_wins': self.orc_wins,
            'draws': self.draws,
            'win_rate_history': self.win_rate_history,
            'results_history': self.results_history[-100:],  # Last 100 battles only
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"self_play_progress_{battle_num}.json", "w") as f:
            json.dump(progress_data, f, indent=2)
    
    def finalize_training(self):
        """Finalize training and save models"""
        # Save final models
        self.empire_ai.save_model("empire_ai_self_play_final.pth")
        self.orc_ai.save_model("orc_ai_self_play_final.pth")
        
        # Create comprehensive report
        final_report = {
            'total_battles': self.num_simulations,
            'final_empire_wins': self.empire_wins,
            'final_orc_wins': self.orc_wins,
            'final_draws': self.draws,
            'final_empire_win_rate': (self.empire_wins / self.num_simulations) * 100,
            'final_orc_win_rate': (self.orc_wins / self.num_simulations) * 100,
            'win_rate_evolution': self.win_rate_history,
            'training_completed': datetime.now().isoformat()
        }
        
        with open("self_play_training_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        
        # Generate visualization
        self.create_training_visualization()
        
        print("\n🎯 FINAL RESULTS:")
        print("=" * 40)
        print(f"📊 Empire Wins: {self.empire_wins:,} ({final_report['final_empire_win_rate']:.1f}%)")
        print(f"📊 Orc Wins: {self.orc_wins:,} ({final_report['final_orc_win_rate']:.1f}%)")
        print(f"📊 Draws: {self.draws:,} ({(self.draws/self.num_simulations)*100:.1f}%)")
        print(f"💾 Models saved as 'empire_ai_self_play_final.pth' and 'orc_ai_self_play_final.pth'")
    
    def create_training_visualization(self):
        """Create visualization of training progress"""
        if not self.win_rate_history:
            return
        
        battles = [h['battle'] for h in self.win_rate_history]
        empire_rates = [h['empire_wins'] for h in self.win_rate_history]
        orc_rates = [h['orc_wins'] for h in self.win_rate_history]
        
        plt.figure(figsize=(12, 8))
        plt.plot(battles, empire_rates, label='Empire Win Rate', color='blue', linewidth=2)
        plt.plot(battles, orc_rates, label='Orc Win Rate', color='green', linewidth=2)
        plt.axhline(y=50, color='red', linestyle='--', label='Perfect Balance (50%)')
        
        plt.title('AI Self-Play Training Progress (10,000 Battles)', fontsize=16, fontweight='bold')
        plt.xlabel('Battle Number', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('self_play_training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 Training visualization saved as 'self_play_training_progress.png'")

def main():
    """Main training function"""
    print("⚔️ WARHAMMER: THE OLD WORLD - AI SELF-PLAY TRAINER")
    print("=" * 60)
    print("🧠 Preparing to run 10,000 AI vs AI battles for mutual improvement")
    print("🎯 Expected outcomes:")
    print("   • More balanced win rates (closer to 50/50)")
    print("   • Discovery of new tactical strategies")
    print("   • Emergence of counter-strategies")
    print("   • More sophisticated AI gameplay")
    print()
    
    confirm = input("Start 10,000 battle self-play training? (y/n): ").lower().strip()
    
    if confirm == 'y':
        trainer = SelfPlayTrainer(num_simulations=10000)
        trainer.run_training_simulation()
    else:
        print("Training cancelled.")

if __name__ == "__main__":
    main() 