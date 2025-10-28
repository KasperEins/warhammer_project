#!/usr/bin/env python3
"""
AI VS AI EPIC BATTLE SYSTEM
============================
Watch trained AI armies clash in automated warfare!
Nuln Army AI (96.15% win rate) vs Troll Horde AI (99.10% win rate)
"""

import time
import random
import numpy as np
import os
from warhammer_ai_agent import WarhammerAIAgent, WarhammerBattleEnvironment
from troll_ai_trainer import TrollAIAgent, TrollBattleEnvironment

class AIvAIBattlefield:
    """Advanced battlefield for AI vs AI combat"""
    
    def __init__(self):
        # Initialize both AI agents
        self.nuln_ai = None
        self.troll_ai = None
        self.load_ai_agents()
        
        # Battle state
        self.turn = 1
        self.max_turns = 10
        self.nuln_health = 100
        self.troll_health = 100
        self.nuln_position = np.array([20.0, 50.0])
        self.troll_position = np.array([80.0, 50.0])
        self.battle_log = []
        
        # Army compositions
        self.nuln_army = {
            'General von Löwenhacke': {'health': 100, 'strength': 8},
            'Great Cannons': {'health': 100, 'strength': 10},
            'Helblaster Volley Guns': {'health': 100, 'strength': 7},
            'State Troops': {'health': 100, 'strength': 5},
            'Outriders': {'health': 100, 'strength': 6}
        }
        
        self.troll_army = {
            'Bigboss on Boar Chariot': {'health': 100, 'strength': 8},
            'Stone Trolls': {'health': 100, 'strength': 9},
            'River Trolls': {'health': 100, 'strength': 7},
            'Goblin Wolf Riders': {'health': 100, 'strength': 4},
            'Orc Boyz': {'health': 100, 'strength': 6}
        }
    
    def load_ai_agents(self):
        """Load both trained AI agents"""
        print("🤖 Loading AI Commanders...")
        
        # Load Nuln AI
        try:
            nuln_env = WarhammerBattleEnvironment()
            state = nuln_env.reset()
            self.nuln_ai = WarhammerAIAgent(state_size=len(state), action_size=13)
            self.nuln_ai.load_model('warhammer_ai_model.pth')
            print("✅ Nuln AI loaded (96.15% win rate)")
        except Exception as e:
            print(f"❌ Error loading Nuln AI: {e}")
            return False
            
        # Load Troll AI
        try:
            self.troll_ai = TrollAIAgent(state_size=13, action_size=13)
            self.troll_ai.load_model('troll_ai_model.pth')
            print("✅ Troll AI loaded (99.10% win rate)")
        except Exception as e:
            print(f"❌ Error loading Troll AI: {e}")
            return False
            
        return True
    
    def get_nuln_state(self):
        """Get battle state from Nuln perspective using original format"""
        state_features = []
        
        # Global battle info (3 features)
        state_features.extend([
            self.turn / self.max_turns,  # Normalized turn
            self.nuln_health / 100.0,    # Nuln units alive ratio
            self.troll_health / 100.0,   # Troll units alive ratio
        ])
        
        # Nuln army state (13 units × 6 features = 78 features)
        for i in range(13):
            if i == 0:  # General
                state_features.extend([
                    self.nuln_position[0] / 72.0,
                    self.nuln_position[1] / 48.0,
                    self.nuln_health / 100.0,
                    1.0 if self.nuln_health > 0 else 0.0,
                    0.0,  # weapon range
                    0.0,  # has charged
                ])
            elif i == 1:  # Artillery
                state_features.extend([
                    (self.nuln_position[0] - 5) / 72.0,
                    self.nuln_position[1] / 48.0,
                    self.nuln_health / 100.0,
                    1.0 if self.nuln_health > 0 else 0.0,
                    48.0 / 48.0,  # long range
                    0.0,
                ])
            elif i < 5:  # Other key units
                state_features.extend([
                    (self.nuln_position[0] + random.uniform(-3, 3)) / 72.0,
                    (self.nuln_position[1] + random.uniform(-2, 2)) / 48.0,
                    self.nuln_health / 100.0,
                    1.0 if self.nuln_health > 0 else 0.0,
                    random.uniform(0.0, 0.5),  # weapon range
                    0.0,
                ])
            else:  # Padding units
                state_features.extend([0.0] * 6)
        
        # Enemy army state (5 units × 4 features = 20 features)
        for i in range(5):
            if i == 0:  # Boss
                state_features.extend([
                    self.troll_position[0] / 72.0,
                    self.troll_position[1] / 48.0,
                    self.troll_health / 100.0,
                    1.0 if self.troll_health > 0 else 0.0,
                ])
            elif i < 3:  # Trolls and cavalry
                state_features.extend([
                    (self.troll_position[0] + random.uniform(-3, 3)) / 72.0,
                    (self.troll_position[1] + random.uniform(-2, 2)) / 48.0,
                    self.troll_health / 100.0,
                    1.0 if self.troll_health > 0 else 0.0,
                ])
            else:  # Padding
                state_features.extend([0.0] * 4)
        
        return np.array(state_features, dtype=np.float32)
    
    def get_troll_state(self):
        """Get battle state from Troll perspective"""
        distance = np.linalg.norm(self.troll_position - self.nuln_position)
        return np.array([
            self.troll_health / 100.0,
            self.nuln_health / 100.0,
            self.turn / self.max_turns,
            (50.0 - abs(self.troll_health - self.nuln_health)) / 50.0,  # Battle momentum
            self.troll_position[0] / 100.0,
            self.troll_position[1] / 100.0,
            self.nuln_position[0] / 100.0,
            self.nuln_position[1] / 100.0,
            1.0,  # Regeneration available
            0.1,  # Stupidity risk
            1.0,  # Fear aura
            0.0,  # Stone skin active
            distance / 100.0
        ])
    
    def execute_nuln_action(self, action):
        """Execute Nuln AI action and return damage dealt"""
        nuln_actions = [
            "Move North", "Move NE", "Move East", "Move SE", 
            "Move South", "Move SW", "Move West", "Move NW",
            "Artillery Strike", "Cavalry Charge", "Defensive Formation", 
            "Flanking Maneuver", "Mass Shooting"
        ]
        
        action_name = nuln_actions[action]
        damage = 0
        
        if action_name == "Artillery Strike":
            damage = random.randint(18, 28)
            self.battle_log.append(f"🎯 Nuln Artillery Strike deals {damage} damage!")
        elif action_name == "Mass Shooting":
            damage = random.randint(12, 20)
            self.battle_log.append(f"🏹 Nuln Mass Shooting deals {damage} damage!")
        elif action_name == "Cavalry Charge":
            damage = random.randint(15, 22)
            self.battle_log.append(f"🐎 Nuln Cavalry Charge deals {damage} damage!")
        elif "Move" in action_name:
            damage = random.randint(3, 8)
            self.battle_log.append(f"🏃 Nuln {action_name} and attacks for {damage} damage")
        else:
            damage = random.randint(8, 15)
            self.battle_log.append(f"⚔️ Nuln {action_name} deals {damage} damage")
        
        return damage
    
    def execute_troll_action(self, action):
        """Execute Troll AI action and return damage dealt"""
        troll_actions = [
            "Troll Charge", "Regeneration", "Stone Skin", "Fear Roar",
            "Smash Attack", "Goblin Swarm", "Boar Charge", "Magic Attack",
            "Move Forward", "Move Back", "Flank Left", "Flank Right", "Hold Position"
        ]
        
        action_name = troll_actions[action]
        damage = 0
        
        if action_name == "Goblin Swarm":
            damage = random.randint(16, 26)
            self.battle_log.append(f"🧌 Troll Goblin Swarm deals {damage} damage!")
        elif action_name == "Troll Charge":
            damage = random.randint(20, 30)
            self.battle_log.append(f"💥 Devastating Troll Charge deals {damage} damage!")
        elif action_name == "Smash Attack":
            damage = random.randint(12, 22)
            self.battle_log.append(f"🔨 Troll Smash Attack deals {damage} damage!")
        elif action_name == "Regeneration":
            heal = random.randint(8, 15)
            self.troll_health = min(100, self.troll_health + heal)
            self.battle_log.append(f"💚 Trolls regenerate {heal} health!")
            return 0
        elif action_name == "Fear Roar":
            damage = random.randint(8, 18)
            self.battle_log.append(f"😱 Fear Roar terrorizes enemies for {damage} damage!")
        else:
            damage = random.randint(6, 14)
            self.battle_log.append(f"⚔️ Troll {action_name} deals {damage} damage")
        
        return damage
    
    def print_battlefield(self, nuln_action, troll_action):
        """Print visual battlefield with AI actions"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("⚔️" * 70)
        print("🤖 AI vs AI EPIC BATTLE - TURN", self.turn)
        print("⚔️" * 70)
        print()
        
        print("🏰 BATTLEFIELD:")
        print("┌" + "─" * 68 + "┐")
        
        # Action names
        nuln_actions = [
            "Move North", "Move NE", "Move East", "Move SE", 
            "Move South", "Move SW", "Move West", "Move NW",
            "Artillery Strike", "Cavalry Charge", "Defensive Formation", 
            "Flanking Maneuver", "Mass Shooting"
        ]
        
        troll_actions = [
            "Troll Charge", "Regeneration", "Stone Skin", "Fear Roar",
            "Smash Attack", "Goblin Swarm", "Boar Charge", "Magic Attack",
            "Move Forward", "Move Back", "Flank Left", "Flank Right", "Hold Position"
        ]
        
        nuln_action_name = nuln_actions[nuln_action] if nuln_action < len(nuln_actions) else "Unknown"
        troll_action_name = troll_actions[troll_action] if troll_action < len(troll_actions) else "Unknown"
        
        # Create battlefield visualization
        for i in range(15):
            row = "│"
            for j in range(68):
                if i == 7 and j == 10:  # Nuln position
                    row += "🔵"
                elif i == 7 and j == 58:  # Troll position  
                    row += "🔴"
                elif i == 5 and j == 20 and "Artillery" in nuln_action_name:
                    row += "💥"
                elif i == 9 and j == 48 and "Goblin" in troll_action_name:
                    row += "🧌"
                elif i == 6 and j == 35 and "Charge" in troll_action_name:
                    row += "⚡"
                elif i == 8 and j == 25 and "Shooting" in nuln_action_name:
                    row += "🏹"
                else:
                    row += " "
            row += "│"
            print(row)
        
        print("└" + "─" * 68 + "┘")
        print()
        
        # Army status
        print("📊 ARMY STATUS:")
        nuln_bar = "█" * (self.nuln_health // 5) + "░" * ((100 - self.nuln_health) // 5)
        troll_bar = "█" * (self.troll_health // 5) + "░" * ((100 - self.troll_health) // 5)
        
        print(f"🔵 Army of Nuln: {nuln_bar} {self.nuln_health}%")
        print(f"🔴 Troll Horde:  {troll_bar} {self.troll_health}%")
        print()
        
        # AI decisions
        print("🧠 AI DECISIONS:")
        print(f"🔵 Nuln AI:  {nuln_action_name}")
        print(f"🔴 Troll AI: {troll_action_name}")
        print()
        
        # Battle log (last 3 events)
        print("📜 BATTLE LOG:")
        for event in self.battle_log[-3:]:
            print(f"   {event}")
        print()
    
    def run_ai_battle(self):
        """Run complete AI vs AI battle"""
        print("🤖 INITIALIZING AI vs AI WARFARE...")
        print("🔵 Nuln AI (96.15% win rate) vs 🔴 Troll AI (99.10% win rate)")
        print()
        
        if not self.nuln_ai or not self.troll_ai:
            print("❌ Failed to load AI agents!")
            return
        
        print("⚡ BATTLE COMMENCING!")
        time.sleep(2)
        
        while self.turn <= self.max_turns and self.nuln_health > 0 and self.troll_health > 0:
            # Get AI states
            nuln_state = self.get_nuln_state()
            troll_state = self.get_troll_state()
            
            # AI make decisions
            nuln_action = self.nuln_ai.act(nuln_state)
            troll_action = self.troll_ai.act(troll_state)
            
            # Display battlefield
            self.print_battlefield(nuln_action, troll_action)
            
            # Execute actions
            nuln_damage = self.execute_nuln_action(nuln_action)
            troll_damage = self.execute_troll_action(troll_action)
            
            # Apply damage
            self.troll_health = max(0, self.troll_health - nuln_damage)
            self.nuln_health = max(0, self.nuln_health - troll_damage)
            
            # Check victory conditions
            if self.nuln_health <= 0:
                print("🏆 TROLL HORDE VICTORY!")
                print("🧌 The Trolls have crushed the Empire forces!")
                break
            elif self.troll_health <= 0:
                print("🏆 ARMY OF NULN VICTORY!")
                print("🔵 Superior tactics and artillery prevail!")
                break
            
            self.turn += 1
            time.sleep(2.5)  # Dramatic pause
        
        # Final battle report
        if self.nuln_health > 0 and self.troll_health > 0:
            if self.nuln_health > self.troll_health:
                print("🏆 NULN VICTORY BY POINTS!")
            elif self.troll_health > self.nuln_health:
                print("🏆 TROLL VICTORY BY POINTS!")
            else:
                print("🤝 EPIC DRAW!")
        
        print("\n⚔️ BATTLE STATISTICS:")
        print(f"   Turns Fought: {self.turn - 1}")
        print(f"   Nuln Final Health: {self.nuln_health}%")
        print(f"   Troll Final Health: {self.troll_health}%")
        print(f"   Total Battle Events: {len(self.battle_log)}")

def main():
    """Launch AI vs AI battle"""
    battlefield = AIvAIBattlefield()
    battlefield.run_ai_battle()

if __name__ == "__main__":
    main() 