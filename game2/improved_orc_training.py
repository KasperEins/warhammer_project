#!/usr/bin/env python3
"""
🚀 IMPROVED ORC TRAINING SYSTEM
==============================
Fixed training with proper memory preservation and enhanced learning
"""

from mass_training_system import MassTrainingSystem, TrainingAI
import time
import torch
import pickle
import os

class ImprovedTrainingAI(TrainingAI):
    """Enhanced AI with better learning and memory preservation"""
    
    def __init__(self, state_size=50, action_size=15, lr=0.001):
        super().__init__(state_size, action_size, lr)
        # Start with higher epsilon for better exploration
        self.epsilon = 0.4
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999  # Slower decay
        
    def save_model_with_memory(self, filepath):
        """Save model including experience memory"""
        # Convert deque to list for serialization
        memory_list = list(self.memory)
        
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': memory_list,
            'memory_maxlen': self.memory.maxlen
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Saved model with {len(memory_list)} experiences")
    
    def load_model_with_memory(self, filepath):
        """Load model including experience memory"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load neural networks
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        
        # Restore memory if available
        if 'memory' in checkpoint and checkpoint['memory']:
            self.memory.clear()
            for experience in checkpoint['memory']:
                self.memory.append(experience)
            print(f"🧠 Restored {len(self.memory)} experiences from memory")
        else:
            print("🆕 Starting with empty memory")

class ImprovedMassTraining(MassTrainingSystem):
    """Enhanced training system with better learning"""
    
    def __init__(self, orc_games=50000, batch_size=500, save_interval=2500):
        super().__init__(empire_games=0, orc_games=orc_games, 
                        batch_size=batch_size, save_interval=save_interval)
        
    def initialize_ai_agents(self):
        """Initialize enhanced AI agents"""
        print("🤖 Initializing IMPROVED AI agents...")
        
        # Use improved AI
        self.orc_ai = ImprovedTrainingAI(state_size=50, action_size=15, lr=0.002)
        
        # Try to load existing model with memory
        try:
            self.orc_ai.load_model_with_memory('orc_ai_improved.pth')
            print("✅ Loaded existing Improved Orc AI with memory")
        except:
            print("🆕 Starting with fresh Improved Orc AI")
            
        print("✅ Enhanced AI agents ready!")
        
    def save_training_progress(self, faction):
        """Enhanced save with memory preservation"""
        if faction == 'orc':
            # Save with memory preservation
            model_name = f"orc_ai_improved_{self.orc_stats['games_completed']}.pth"
            self.orc_ai.save_model_with_memory(model_name)
            
            # Also save as main model
            self.orc_ai.save_model_with_memory('orc_ai_improved.pth')
            
            print(f"💾 Enhanced save: {model_name}")

def run_improved_training():
    """Run improved Orc training with better learning"""
    print("🚀 IMPROVED ORC TRAINING SYSTEM")
    print("=" * 45)
    print("🔧 Enhanced with:")
    print("   • Memory preservation between sessions")
    print("   • Higher exploration rate (epsilon=0.4)")
    print("   • Better learning rate (0.002)")
    print("   • Slower epsilon decay")
    print()
    
    # Ask for number of games
    try:
        games = int(input("How many games to train? (e.g. 50000): ") or "50000")
    except ValueError:
        games = 50000
    
    print(f"🎯 Training {games:,} games with improved system")
    confirm = input("🚀 Start improved training? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print("\n🟢 STARTING IMPROVED ORC TRAINING...")
    print("=" * 45)
    
    # Create improved trainer
    trainer = ImprovedMassTraining(
        orc_games=games,
        batch_size=500,  # Smaller batches for better learning
        save_interval=2500  # More frequent saves
    )
    
    start_time = time.time()
    
    # Initialize enhanced AI
    trainer.initialize_ai_agents()
    
    print(f"🧠 Starting epsilon: {trainer.orc_ai.epsilon:.6f}")
    print(f"📝 Memory size: {len(trainer.orc_ai.memory)}")
    print()
    
    # Run training with enhanced monitoring
    print("🟢 RUNNING ENHANCED TRAINING...")
    trainer.run_orc_training()
    
    # Calculate time
    total_time = (time.time() - start_time) / 3600
    
    print(f"\n🎉 IMPROVED TRAINING COMPLETE!")
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
    
    print(f"\n💾 Enhanced model saved: orc_ai_improved_{trainer.orc_stats['games_completed']}.pth")
    print("🚀 Ready for continued improved training!")

if __name__ == "__main__":
    run_improved_training() 