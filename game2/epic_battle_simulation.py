#!/usr/bin/env python3
"""
Epic Battle Simulation - Trained AI Agents Face Off!
Empire vs Orc armies with fully trained neural networks (300k games each)
"""

import torch
import numpy as np
import random
import time
from warhammer_ai_agent import WarhammerAIAgent
from mass_training_system import TrainingBattle

class EpicBattleSimulator:
    """Runs epic battles between trained AI agents"""
    
    def __init__(self):
        self.empire_agent = None
        self.orc_agent = None
        self.battle = None
        
    def load_trained_agents(self):
        """Load the fully trained 300k game agents"""
        print("🏰 Loading Empire AI (Defensive Specialist)...")
        self.empire_agent = WarhammerAIAgent(
            state_size=50,
            action_size=15,
            lr=0.001
        )
        self.empire_agent.epsilon = 0.0  # Pure exploitation - no exploration
        
        try:
            # Load the final trained Empire model
            checkpoint = torch.load('empire_massive_300k_final.pth', map_location='cpu', weights_only=False)
            if 'q_network_state_dict' in checkpoint:
                self.empire_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            else:
                self.empire_agent.q_network.load_state_dict(checkpoint)
            print("✅ Empire AI loaded successfully!")
        except FileNotFoundError:
            print("❌ Empire final model not found, trying alternative...")
            try:
                checkpoint = torch.load('empire_massive_300000.pth', map_location='cpu', weights_only=False)
                if 'q_network_state_dict' in checkpoint:
                    self.empire_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                else:
                    self.empire_agent.q_network.load_state_dict(checkpoint)
                print("✅ Empire AI loaded from 300k checkpoint!")
            except FileNotFoundError:
                print("❌ No Empire model found!")
                return False
        
        print("🗡️ Loading Orc AI (Aggressive Specialist)...")
        self.orc_agent = WarhammerAIAgent(
            state_size=50,
            action_size=15,
            lr=0.002
        )
        self.orc_agent.epsilon = 0.0  # Pure exploitation
        
        try:
            # Load the final trained Orc model
            checkpoint = torch.load('orc_massive_300k_final.pth', map_location='cpu', weights_only=False)
            if 'q_network_state_dict' in checkpoint:
                self.orc_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            else:
                self.orc_agent.q_network.load_state_dict(checkpoint)
            print("✅ Orc AI loaded successfully!")
        except FileNotFoundError:
            print("❌ Orc final model not found, trying alternative...")
            try:
                checkpoint = torch.load('orc_massive_300000.pth', map_location='cpu', weights_only=False)
                if 'q_network_state_dict' in checkpoint:
                    self.orc_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                else:
                    self.orc_agent.q_network.load_state_dict(checkpoint)
                print("✅ Orc AI loaded from 300k checkpoint!")
            except FileNotFoundError:
                print("❌ No Orc model found!")
                return False
        
        return True
    
    def display_battle_header(self, battle_num):
        """Display epic battle header"""
        print("\n" + "=" * 70)
        print(f"🌟 EPIC BATTLE #{battle_num} 🌟")
        print("⚔️  Empire (300k games trained) vs Orc (300k games trained) ⚔️")
        print("=" * 70)
        
    def get_faction_color(self, faction):
        """Get color coding for factions"""
        if faction == "Empire":
            return "🔵"
        else:
            return "🟢"
    
    def describe_action(self, action, faction):
        """Convert action number to descriptive text"""
        action_descriptions = [
            "advances northward", "moves northeast", "advances eastward", "moves southeast",
            "retreats southward", "moves southwest", "advances westward", "moves northwest", 
            "unleashes artillery barrage", "orders cavalry charge", "forms defensive line",
            "executes flanking maneuver", "commands mass volley"
        ]
        
        if 0 <= action < len(action_descriptions):
            return f"{self.get_faction_color(faction)} {faction} {action_descriptions[action]}"
        else:
            return f"{self.get_faction_color(faction)} {faction} takes tactical action {action}"
    
    def run_single_battle(self, battle_num, show_details=True):
        """Run a single battle between the AIs"""
        self.display_battle_header(battle_num)
        
        # Create new battle
        self.battle = TrainingBattle()
        self.battle.create_armies()
        
        empire_agent = self.empire_agent
        orc_agent = self.orc_agent
        
        turn = 1
        max_turns = 20  # Extended for epic battles
        
        while turn <= max_turns:
            if show_details:
                print(f"\n🔄 Turn {turn}")
                print("-" * 30)
            
            # Get current state
            state = self.battle.get_ai_state()
            
            # Empire's turn
            empire_action = empire_agent.act(state)
            if show_details:
                print(f"   {self.describe_action(empire_action, 'Empire')}")
            
            # Orc's turn
            orc_action = orc_agent.act(state)
            if show_details:
                print(f"   {self.describe_action(orc_action, 'Orc')}")
            
            # Use actions to influence battle randomness
            random.seed(empire_action * turn + orc_action * turn * 2)
            
            # Execute battle turn
            self.battle.simulate_turn()
            
            # Check if battle is over
            empire_alive = sum(1 for u in self.battle.units if u.faction == "nuln" and u.is_alive)
            orc_alive = sum(1 for u in self.battle.units if u.faction == "orcs" and u.is_alive)
            
            if empire_alive == 0 and orc_alive > 0:
                if show_details:
                    print(f"\n🏆 Orc achieves decisive victory on turn {turn}!")
                return "Orc", turn
            elif orc_alive == 0 and empire_alive > 0:
                if show_details:
                    print(f"\n🏆 Empire achieves decisive victory on turn {turn}!")
                return "Empire", turn
            elif empire_alive == 0 and orc_alive == 0:
                if show_details:
                    print(f"\n⚖️ Mutual annihilation on turn {turn}")
                return "Draw", turn
            
            turn += 1
            
            # Small delay for dramatic effect
            if show_details:
                time.sleep(0.5)
        
        # Battle went to maximum turns - determine winner by remaining forces
        if show_details:
            print(f"\n⏰ Battle duration limit reached after {max_turns} turns!")
        
        self.battle.calculate_final_scores()
        winner = self.battle.get_winner()
        
        if winner == 'empire':
            if show_details:
                print("🏆 Empire wins by tactical superiority!")
            return "Empire", max_turns
        elif winner == 'orc':
            if show_details:
                print("🏆 Orc wins by battlefield dominance!")
            return "Orc", max_turns
        else:
            if show_details:
                print("⚖️ Draw - Both armies prove equally matched!")
            return "Draw", max_turns
    
    def run_battle_series(self, num_battles=5):
        """Run a series of battles and track results"""
        print(f"\n🎯 LAUNCHING {num_battles} BATTLE CAMPAIGN")
        print("=" * 50)
        
        results = {"Empire": 0, "Orc": 0, "Draw": 0}
        battle_details = []
        
        for i in range(num_battles):
            winner, turns = self.run_single_battle(i + 1, show_details=(i == 0))  # Show details for first battle only
            results[winner] += 1
            battle_details.append((winner, turns))
            
            if i > 0:  # Brief summary for battles 2+
                print(f"⚔️ Battle {i + 1}: {winner} wins in {turns} turns")
        
        # Final results
        print("\n" + "=" * 50)
        print("🏁 CAMPAIGN RESULTS")
        print("=" * 50)
        
        empire_pct = (results["Empire"] / num_battles) * 100
        orc_pct = (results["Orc"] / num_battles) * 100
        draw_pct = (results["Draw"] / num_battles) * 100
        
        print(f"🔵 Empire victories: {results['Empire']}/{num_battles} ({empire_pct:.1f}%)")
        print(f"🟢 Orc victories:    {results['Orc']}/{num_battles} ({orc_pct:.1f}%)")
        print(f"⚖️ Draws:           {results['Draw']}/{num_battles} ({draw_pct:.1f}%)")
        
        avg_turns = sum(turns for _, turns in battle_details) / len(battle_details)
        print(f"⏱️ Average battle duration: {avg_turns:.1f} turns")
        
        # Determine overall campaign winner
        if results["Empire"] > results["Orc"]:
            print(f"\n🎊 EMPIRE WINS THE CAMPAIGN! 🎊")
            print("The disciplined forces of the Empire prove superior through tactical excellence!")
        elif results["Orc"] > results["Empire"]:
            print(f"\n🎊 ORC WAAAGH! VICTORIOUS! 🎊") 
            print("The brutal Orc war machine overwhelms all opposition!")
        else:
            print(f"\n🤝 CAMPAIGN ENDS IN STALEMATE!")
            print("Both armies prove equally matched - neither can claim dominance!")
        
        return results, battle_details

def main():
    """Main battle simulation function"""
    print("⚔️ WARHAMMER AI: EPIC BATTLE SIMULATOR ⚔️")
    print("Loading 300,000-game trained neural networks...")
    print("=" * 60)
    
    simulator = EpicBattleSimulator()
    
    # Load trained models
    if not simulator.load_trained_agents():
        print("❌ Failed to load trained models. Make sure the .pth files are present.")
        return
    
    print(f"\n🧠 Empire AI: Defensive specialist (ε=0.001 lr, 300k games)")
    print(f"🧠 Orc AI: Aggressive specialist (ε=0.002 lr, 300k games)")
    print(f"⚡ Both AIs in pure exploitation mode (ε=0.0)")
    
    # Run battle campaign
    try:
        results, details = simulator.run_battle_series(num_battles=5)
        
        print(f"\n🎮 Battle simulation complete!")
        print(f"📊 AI performance comparison after 300,000 training games each")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Battle simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")

if __name__ == "__main__":
    main() 