#!/usr/bin/env python3
"""
🏛️ MASSIVE DUAL ARMY TRAINING SYSTEM
=====================================
Train both Empire and Orc armies with 300,000 simulations each
Based on the proven working simple_working_training.py approach
"""

from fixed_improved_training import FixedImprovedAI
from mass_training_system import TrainingBattle
import time
import random
import os

class MassiveDualTrainer:
    """Massive training system for both armies"""
    
    def __init__(self):
        self.empire_games = 300000
        self.orc_games = 300000
        self.save_interval = 5000  # Save every 5K games
        self.progress_interval = 1000  # Progress update every 1K games
        
        print("🏛️ MASSIVE DUAL ARMY TRAINING SYSTEM")
        print("=" * 50)
        print(f"📊 Empire simulations: {self.empire_games:,}")
        print(f"📊 Orc simulations: {self.orc_games:,}")
        print(f"💾 Save interval: {self.save_interval:,} games")
        
    def create_ai_pair(self):
        """Create both Empire and Orc AIs"""
        print("\n🤖 Creating AI commanders...")
        
        # Empire AI - defensive/artillery specialist
        empire_ai = FixedImprovedAI(state_size=50, action_size=15, lr=0.001)
        empire_ai.epsilon = 0.3  # Moderate exploration for empire
        empire_ai.epsilon_decay = 0.99995  # Slower decay for long training
        
        # Orc AI - aggressive specialist  
        orc_ai = FixedImprovedAI(state_size=50, action_size=15, lr=0.002)
        orc_ai.epsilon = 0.4  # Higher exploration for orcs
        orc_ai.epsilon_decay = 0.99994  # Slightly faster decay
        
        # Try to load existing models
        try:
            print("🔍 Looking for existing Empire model...")
            empire_checkpoint = torch.load('empire_massive_300k.pth', map_location='cpu', weights_only=False)
            empire_ai.q_network.load_state_dict(empire_checkpoint['q_network_state_dict'])
            empire_ai.target_network.load_state_dict(empire_checkpoint['target_network_state_dict'])
            empire_ai.epsilon = empire_checkpoint.get('epsilon', 0.3)
            empire_ai.victories = empire_checkpoint.get('victories', 0)
            empire_ai.defeats = empire_checkpoint.get('defeats', 0)
            if 'memory' in empire_checkpoint:
                empire_ai.memory.extend(empire_checkpoint['memory'])
            print(f"✅ Loaded existing Empire AI (ε={empire_ai.epsilon:.3f}, {len(empire_ai.memory)} exp)")
        except:
            print("🆕 Starting fresh Empire AI")
            
        try:
            print("🔍 Looking for existing Orc model...")
            orc_checkpoint = torch.load('orc_massive_300k.pth', map_location='cpu', weights_only=False)
            orc_ai.q_network.load_state_dict(orc_checkpoint['q_network_state_dict'])
            orc_ai.target_network.load_state_dict(orc_checkpoint['target_network_state_dict'])
            orc_ai.epsilon = orc_checkpoint.get('epsilon', 0.4)
            orc_ai.victories = orc_checkpoint.get('victories', 0)
            orc_ai.defeats = orc_checkpoint.get('defeats', 0)
            if 'memory' in orc_checkpoint:
                orc_ai.memory.extend(orc_checkpoint['memory'])
            print(f"✅ Loaded existing Orc AI (ε={orc_ai.epsilon:.3f}, {len(orc_ai.memory)} exp)")
        except:
            print("🆕 Starting fresh Orc AI")
        
        print(f"\n🏛️ Empire AI: ε={empire_ai.epsilon:.3f}, memory={len(empire_ai.memory):,}")
        print(f"🟢 Orc AI: ε={orc_ai.epsilon:.3f}, memory={len(orc_ai.memory):,}")
        
        return empire_ai, orc_ai
    
    def run_training_battle(self, empire_ai, orc_ai, primary_faction, game_num):
        """Run a single training battle"""
        battle = TrainingBattle()
        battle.create_armies()
        
        # Track both AIs but focus training on primary faction
        empire_reward = 0
        orc_reward = 0
        empire_states = []
        orc_states = []
        empire_actions = []
        orc_actions = []
        
        initial_state = battle.get_ai_state()
        
        # Battle simulation
        for turn in range(25):  # Slightly longer battles for better learning
            prev_state = battle.get_ai_state()
            
            # Both AIs make decisions
            empire_action = empire_ai.act(prev_state)
            orc_action = orc_ai.act(prev_state)
            
            # Store states and actions
            empire_states.append(prev_state.copy())
            orc_states.append(prev_state.copy()) 
            empire_actions.append(empire_action)
            orc_actions.append(orc_action)
            
            # Use primary faction's action to influence battle
            if primary_faction == 'empire':
                random.seed(empire_action * turn + game_num)
            else:
                random.seed(orc_action * turn + game_num)
            
            # Execute battle turn
            battle.simulate_turn()
            new_state = battle.get_ai_state()
            
            # Calculate rewards for both factions
            empire_alive = sum(1 for u in battle.units if u.faction == "nuln" and u.is_alive)
            orc_alive = sum(1 for u in battle.units if u.faction == "orcs" and u.is_alive)
            
            # Empire rewards
            empire_turn_reward = empire_alive * 3 - orc_alive * 2 + 1
            if orc_alive == 0 and empire_alive > 0:
                empire_turn_reward += 50  # Victory bonus
            elif empire_alive == 0:
                empire_turn_reward -= 30  # Defeat penalty
                
            # Orc rewards (opposite perspective)
            orc_turn_reward = orc_alive * 3 - empire_alive * 2 + 1
            if empire_alive == 0 and orc_alive > 0:
                orc_turn_reward += 50  # Victory bonus
            elif orc_alive == 0:
                orc_turn_reward -= 30  # Defeat penalty
            
            empire_reward += empire_turn_reward
            orc_reward += orc_turn_reward
            
            # Check battle end
            battle_done = empire_alive == 0 or orc_alive == 0
            
            # Store experiences (only for states we have)
            if len(empire_states) >= 2:
                empire_ai.remember(empire_states[-2], empire_actions[-2], 
                                 empire_turn_reward, prev_state, battle_done)
            if len(orc_states) >= 2:
                orc_ai.remember(orc_states[-2], orc_actions[-2], 
                               orc_turn_reward, prev_state, battle_done)
            
            if battle_done:
                break
        
        # Final battle resolution
        battle.calculate_final_scores()
        winner = battle.get_winner()
        
        # Update statistics
        if winner == 'empire':
            empire_ai.victories += 1
            orc_ai.defeats += 1
        elif winner == 'orc':
            orc_ai.victories += 1
            empire_ai.defeats += 1
        else:
            empire_ai.draws += 1
            orc_ai.draws += 1
        
        return winner, empire_reward, orc_reward
    
    def train_empire(self, empire_ai, orc_ai):
        """Train Empire AI for 300K games"""
        print(f"\n🏛️ TRAINING EMPIRE AI - 300,000 SIMULATIONS")
        print("=" * 55)
        
        start_time = time.time()
        empire_wins = 0
        
        for game in range(self.empire_games):
            winner, empire_reward, orc_reward = self.run_training_battle(
                empire_ai, orc_ai, 'empire', game)
            
            if winner == 'empire':
                empire_wins += 1
            
            # Train Empire AI periodically
            if len(empire_ai.memory) >= 64 and game % 10 == 0:
                empire_ai.replay()
            
            # Decay epsilon
            if empire_ai.epsilon > empire_ai.epsilon_min:
                empire_ai.epsilon *= empire_ai.epsilon_decay
            
            # Progress updates
            if (game + 1) % self.progress_interval == 0:
                win_rate = (empire_wins / (game + 1)) * 100
                elapsed = time.time() - start_time
                games_per_sec = (game + 1) / elapsed
                eta_sec = (self.empire_games - game - 1) / games_per_sec if games_per_sec > 0 else 0
                eta_min = eta_sec / 60
                
                print(f"  Empire {game+1:6,}/300,000: {win_rate:5.1f}% win rate, "
                      f"ε={empire_ai.epsilon:.4f}, mem={len(empire_ai.memory):,}, "
                      f"ETA: {eta_min:.1f}m")
            
            # Save checkpoints
            if (game + 1) % self.save_interval == 0:
                empire_ai.save_enhanced_model(f"empire_massive_{game+1}.pth")
                print(f"    💾 Empire checkpoint saved at {game+1:,} games")
        
        # Final save
        empire_ai.save_enhanced_model("empire_massive_300k_final.pth")
        final_rate = (empire_wins / self.empire_games) * 100
        print(f"\n🏛️ EMPIRE TRAINING COMPLETE!")
        print(f"   Final win rate: {final_rate:.2f}%")
        print(f"   Final epsilon: {empire_ai.epsilon:.6f}")
        print(f"   Memory size: {len(empire_ai.memory):,}")
        
    def train_orc(self, empire_ai, orc_ai):
        """Train Orc AI for 300K games"""
        print(f"\n🟢 TRAINING ORC AI - 300,000 SIMULATIONS") 
        print("=" * 50)
        
        start_time = time.time()
        orc_wins = 0
        
        for game in range(self.orc_games):
            winner, empire_reward, orc_reward = self.run_training_battle(
                empire_ai, orc_ai, 'orc', game)
            
            if winner == 'orc':
                orc_wins += 1
            
            # Train Orc AI periodically
            if len(orc_ai.memory) >= 64 and game % 10 == 0:
                orc_ai.replay()
            
            # Decay epsilon
            if orc_ai.epsilon > orc_ai.epsilon_min:
                orc_ai.epsilon *= orc_ai.epsilon_decay
            
            # Progress updates
            if (game + 1) % self.progress_interval == 0:
                win_rate = (orc_wins / (game + 1)) * 100
                elapsed = time.time() - start_time
                games_per_sec = (game + 1) / elapsed
                eta_sec = (self.orc_games - game - 1) / games_per_sec if games_per_sec > 0 else 0
                eta_min = eta_sec / 60
                
                print(f"  Orc {game+1:6,}/300,000: {win_rate:5.1f}% win rate, "
                      f"ε={orc_ai.epsilon:.4f}, mem={len(orc_ai.memory):,}, "
                      f"ETA: {eta_min:.1f}m")
            
            # Save checkpoints
            if (game + 1) % self.save_interval == 0:
                orc_ai.save_enhanced_model(f"orc_massive_{game+1}.pth")
                print(f"    💾 Orc checkpoint saved at {game+1:,} games")
        
        # Final save
        orc_ai.save_enhanced_model("orc_massive_300k_final.pth")
        final_rate = (orc_wins / self.orc_games) * 100
        print(f"\n🟢 ORC TRAINING COMPLETE!")
        print(f"   Final win rate: {final_rate:.2f}%")
        print(f"   Final epsilon: {orc_ai.epsilon:.6f}")
        print(f"   Memory size: {len(orc_ai.memory):,}")
    
    def run_massive_training(self):
        """Run the complete massive training sequence"""
        print(f"\n🚀 STARTING MASSIVE DUAL TRAINING")
        print(f"Total simulations: {(self.empire_games + self.orc_games):,}")
        
        overall_start = time.time()
        
        # Create AIs
        empire_ai, orc_ai = self.create_ai_pair()
        
        # Train Empire first
        self.train_empire(empire_ai, orc_ai)
        
        # Train Orc second  
        self.train_orc(empire_ai, orc_ai)
        
        # Final summary
        total_time = time.time() - overall_start
        total_games = self.empire_games + self.orc_games
        games_per_hour = (total_games / total_time) * 3600
        
        print(f"\n🎉 MASSIVE TRAINING COMPLETE!")
        print("=" * 40)
        print(f"📊 Total games: {total_games:,}")
        print(f"⏱️ Total time: {total_time/3600:.2f} hours")
        print(f"🚀 Speed: {games_per_hour:,.0f} games/hour")
        print(f"🏛️ Empire AI: {empire_ai.victories:,}W-{empire_ai.defeats:,}L")
        print(f"🟢 Orc AI: {orc_ai.victories:,}W-{orc_ai.defeats:,}L")
        print(f"💾 Models saved as: empire_massive_300k_final.pth, orc_massive_300k_final.pth")

def main():
    """Main training execution"""
    import torch  # Import here to avoid issues
    
    trainer = MassiveDualTrainer()
    trainer.run_massive_training()

if __name__ == "__main__":
    main() 