#!/usr/bin/env python3
"""
🔧 FIXED IMPROVED ORC TRAINING SYSTEM
====================================
Properly compatible with original architecture + enhanced learning
"""

from mass_training_system import MassTrainingSystem, TrainingBattle
from warhammer_ai_agent import WarhammerAIAgent, DQNNetwork
import time
import torch
import random
import numpy as np
from collections import deque

class FixedImprovedAI(WarhammerAIAgent):
    """Enhanced AI using ORIGINAL architecture with improved learning"""
    
    def __init__(self, state_size=50, action_size=15, lr=0.002):
        # Initialize with original architecture
        super().__init__(state_size, action_size, lr)
        
        # Enhanced learning parameters
        self.epsilon = 0.4  # Higher exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999  # Slower decay for longer exploration
        self.learning_rate = lr
        
        # Enhanced memory
        self.memory = deque(maxlen=50000)  # Larger memory buffer
        
        # Training stats
        self.victories = 0
        self.defeats = 0
        self.draws = 0
        self.successful_strategies = []
        
        print(f"✅ Fixed AI initialized with original architecture")
        print(f"   • Epsilon: {self.epsilon}")
        print(f"   • Learning rate: {lr}")
        print(f"   • Memory size: {self.memory.maxlen}")
    
    def replay(self, batch_size=32):
        """
        CRITICAL FIX: Override replay to accept batch_size parameter
        that MassTrainingSystem passes, but use original replay logic
        """
        # Call the original replay method (which doesn't take parameters)
        super().replay()
    
    def save_model_with_memory(self, filepath):
        """Save model with experience memory using ORIGINAL format"""
        # Convert deque to list for serialization
        memory_list = list(self.memory)
        
        # Use ORIGINAL checkpoint format but add memory
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),  # ORIGINAL key name
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': getattr(self, 'scores', []),
            'successful_strategies': getattr(self, 'successful_strategies', []),
            'victories': getattr(self, 'victories', 0),
            'defeats': getattr(self, 'defeats', 0),
            'draws': getattr(self, 'draws', 0),
            # Enhanced additions
            'memory': memory_list,
            'memory_maxlen': self.memory.maxlen
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Saved compatible model with {len(memory_list)} experiences")
    
    def load_model_with_memory(self, filepath):
        """Load model with memory using ORIGINAL format"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load using original method
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        
        # Load original attributes
        self.scores = checkpoint.get('scores', [])
        self.successful_strategies = checkpoint.get('successful_strategies', [])
        self.victories = checkpoint.get('victories', 0)
        self.defeats = checkpoint.get('defeats', 0)
        self.draws = checkpoint.get('draws', 0)
        
        # Restore enhanced memory if available
        if 'memory' in checkpoint and checkpoint['memory']:
            self.memory.clear()
            for experience in checkpoint['memory']:
                self.memory.append(experience)
            print(f"🧠 Restored {len(self.memory)} experiences from memory")
        else:
            print("🆕 Starting with empty memory")

    def save_enhanced_model(self, filepath):
        """Save model with enhanced data"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': getattr(self, 'scores', []),
            'successful_strategies': self.successful_strategies,
            'victories': self.victories,
            'defeats': self.defeats,
            'draws': self.draws,
            'memory': list(self.memory),
            'memory_maxlen': self.memory.maxlen
        }
        torch.save(checkpoint, filepath)
        print(f"💾 Enhanced model saved: {filepath}")
        print(f"   Memory: {len(self.memory):,} experiences")
        print(f"   Battle record: {self.victories}W-{self.defeats}L-{self.draws}D")

class FixedTrainingBattle(TrainingBattle):
    """Enhanced battle that works with WarhammerAIAgent"""
    
    def run_battle(self, ai1, ai2, faction1='empire', faction2='orc'):
        """Run battle compatible with WarhammerAIAgent"""
        self.create_armies()
        
        total_reward_1 = 0
        total_reward_2 = 0
        turn = 0
        max_turns = 20
        
        prev_state_1 = self.get_ai_state()
        prev_state_2 = self.get_ai_state()
        
        while turn < max_turns and self.battle_state == "active":
            # AI 1 turn
            if hasattr(ai1, 'act'):  # WarhammerAIAgent
                action1 = ai1.act(prev_state_1)
            else:  # Fallback
                action1 = random.randint(0, 14)
            
            # AI 2 turn  
            if hasattr(ai2, 'act'):
                action2 = ai2.act(prev_state_2)
            else:
                action2 = random.randint(0, 14)
            
            # Simulate turn
            self.simulate_turn()
            turn += 1
            
            # Get new state and calculate rewards
            new_state = self.get_ai_state()
            
            reward1 = self.calculate_reward(prev_state_1, new_state, faction1)
            reward2 = self.calculate_reward(prev_state_2, new_state, faction2)
            
            total_reward_1 += reward1
            total_reward_2 += reward2
            
            # Store experience for WarhammerAIAgent
            done = (turn >= max_turns or self.battle_state != "active")
            
            if hasattr(ai1, 'remember'):
                ai1.remember(prev_state_1, action1, reward1, new_state, done)
            if hasattr(ai2, 'remember'):
                ai2.remember(prev_state_2, action2, reward2, new_state, done)
            
            prev_state_1 = new_state
            prev_state_2 = new_state
            
            # Check if battle should end
            empire_alive = any(u.is_alive and u.faction == "nuln" for u in self.units)
            orc_alive = any(u.is_alive and u.faction == "orcs" for u in self.units)
            
            if not empire_alive or not orc_alive:
                break
        
        self.calculate_final_scores()
        winner = self.get_winner()
        
        # Train the AIs if they support it (fixed - no batch_size parameter)
        if hasattr(ai1, 'replay') and len(ai1.memory) > ai1.batch_size:
            ai1.replay()
        if hasattr(ai2, 'replay') and len(ai2.memory) > ai2.batch_size:
            ai2.replay()
        
        return winner
    
    def calculate_reward(self, prev_state, new_state, faction):
        """Calculate training reward"""
        reward = 0.0
        
        if faction == 'empire':
            # Empire rewards
            empire_units = [u for u in self.units if u.faction == "nuln" and u.is_alive]
            orc_units = [u for u in self.units if u.faction == "orcs" and u.is_alive]
            
            reward += len(empire_units) * 2  # Survival bonus
            reward -= len(orc_units) * 1  # Enemy penalty
            
        elif faction == 'orc':
            # Orc rewards
            empire_units = [u for u in self.units if u.faction == "nuln" and u.is_alive]
            orc_units = [u for u in self.units if u.faction == "orcs" and u.is_alive]
            
            reward += len(orc_units) * 2  # Survival bonus
            reward -= len(empire_units) * 1  # Enemy penalty
        
        # Final battle bonus
        if self.battle_state != "active":
            winner = self.get_winner()
            if (faction == 'empire' and winner == 'empire') or \
               (faction == 'orc' and winner == 'orc'):
                reward += 50  # Victory bonus
            elif winner == 'draw':
                reward += 10  # Draw bonus
            else:
                reward -= 25  # Defeat penalty
        
        return reward

class FixedMassTraining(MassTrainingSystem):
    """Fixed training system with original architecture"""
    
    def __init__(self, orc_games=50000, batch_size=500, save_interval=2500):
        super().__init__(empire_games=0, orc_games=orc_games, 
                        batch_size=batch_size, save_interval=save_interval)
        
    def initialize_ai_agents(self):
        """Initialize compatible enhanced AI agents"""
        print("🤖 Initializing FIXED IMPROVED AI agents...")
        
        # Use enhanced AI with ORIGINAL architecture
        self.orc_ai = FixedImprovedAI(state_size=50, action_size=15, lr=0.002)
        
        # Try to load existing compatible model
        try:
            self.orc_ai.load_model_with_memory('orc_ai_fixed_improved.pth')
            print("✅ Loaded existing Fixed Improved Orc AI with memory")
        except:
            # Try to load from original format
            try:
                self.orc_ai.load_model('warhammer_ai_model.pth')
                print("✅ Loaded base Orc AI and enhanced it")
            except:
                print("🆕 Starting with fresh Fixed Improved Orc AI")
            
        print("✅ Fixed enhanced AI agents ready!")
        
    def create_training_battle(self):
        """Create enhanced training battle"""
        return FixedTrainingBattle()
        
    def save_training_progress(self, faction):
        """Enhanced save with memory preservation"""
        if faction == 'orc':
            # Save with memory preservation
            model_name = f"orc_ai_fixed_{self.orc_stats['games_completed']}.pth"
            self.orc_ai.save_model_with_memory(model_name)
            
            # Also save as main model
            self.orc_ai.save_model_with_memory('orc_ai_fixed_improved.pth')
            
            print(f"💾 Fixed compatible save: {model_name}")

def run_fixed_training():
    """Run fixed improved Orc training"""
    print("🔧 FIXED IMPROVED ORC TRAINING SYSTEM")
    print("=" * 45)
    print("✅ Features:")
    print("   • Original DQNNetwork architecture (compatible!)")
    print("   • Memory preservation between sessions")
    print("   • Higher exploration rate (epsilon=0.4)")
    print("   • Better learning rate (0.002)")
    print("   • Slower epsilon decay")
    print("   • Larger memory buffer (50,000)")
    print()
    
    # Ask for number of games
    try:
        games = int(input("How many games to train? (e.g. 50000): ") or "50000")
    except ValueError:
        games = 50000
    
    print(f"🎯 Training {games:,} games with fixed improved system")
    confirm = input("🚀 Start fixed training? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print("\n🟢 STARTING FIXED IMPROVED ORC TRAINING...")
    print("=" * 45)
    
    # Create fixed trainer
    trainer = FixedMassTraining(
        orc_games=games,
        batch_size=500,
        save_interval=2500
    )
    
    start_time = time.time()
    
    # Initialize enhanced but compatible AI
    trainer.initialize_ai_agents()
    
    print(f"🧠 Starting epsilon: {trainer.orc_ai.epsilon:.6f}")
    print(f"📝 Memory size: {len(trainer.orc_ai.memory)}")
    print(f"🏗️ Architecture: Original DQNNetwork (compatible)")
    print()
    
    # Run training with enhanced monitoring
    print("🟢 RUNNING FIXED ENHANCED TRAINING...")
    trainer.run_orc_training()
    
    # Calculate time
    total_time = (time.time() - start_time) / 3600
    
    print(f"\n🎉 FIXED IMPROVED TRAINING COMPLETE!")
    print("=" * 45)
    print(f"⏱️  Total Time: {total_time:.2f} hours")
    print(f"🎮 Games Completed: {trainer.orc_stats['games_completed']:,}")
    
    # Show results
    if trainer.orc_stats['games_completed'] > 0:
        win_rate = (trainer.orc_stats['wins'] / trainer.orc_stats['games_completed']) * 100
        print(f"🏆 Final Win Rate: {win_rate:.2f}%")
        print(f"⭐ Best Score: {trainer.orc_stats['best_score']:.1f}")
        print(f"🧠 Final Epsilon: {trainer.orc_ai.epsilon:.6f}")
        print(f"📝 Final Memory Size: {len(trainer.orc_ai.memory)}")
    
    print(f"\n💾 Compatible model saved: orc_ai_fixed_{trainer.orc_stats['games_completed']}.pth")
    print("✅ NOW COMPATIBLE WITH ORIGINAL BATTLE SYSTEM!")

if __name__ == "__main__":
    run_fixed_training() 