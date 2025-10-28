#!/usr/bin/env python3
"""
SIMPLE VISUAL AI vs AI BATTLE
=============================
Simplified version that focuses on showing units clearly
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# Import our AI systems
try:
    from warhammer_ai_agent import WarhammerAIAgent, WarhammerBattleEnvironment
    from troll_ai_trainer import TrollAIAgent, TrollBattleEnvironment
    HAVE_AI = True
except ImportError as e:
    print(f"AI import error: {e}")
    HAVE_AI = False

class SimpleUnit:
    def __init__(self, name, x, y, player, color, health=100, strength=5):
        self.name = name
        self.x = x
        self.y = y
        self.player = player
        self.color = color
        self.health = health
        self.max_health = health
        self.strength = strength
        self.size = 4  # Visual size
        
    def is_alive(self):
        return self.health > 0
        
    def take_damage(self, damage):
        self.health = max(0, self.health - damage)
    
    def move_towards(self, target_x, target_y, speed=2):
        """Move unit towards target position"""
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > speed:
            self.x += (dx / distance) * speed
            self.y += (dy / distance) * speed
        else:
            self.x = target_x
            self.y = target_y

class SimpleVisualBattle:
    def __init__(self):
        # Setup matplotlib - use interactive mode
        plt.ion()
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        
        # Load AI if available
        self.nuln_ai = None
        self.troll_ai = None
        if HAVE_AI:
            self.load_ai_agents()
        
        # Battle state
        self.turn = 1
        self.max_turns = 12
        self.battle_log = []
        self.units = []
        
        # Create armies
        self.create_simple_armies()
        
    def load_ai_agents(self):
        """Load both trained AI agents"""
        print("🤖 Loading AI Commanders...")
        
        try:
            nuln_env = WarhammerBattleEnvironment()
            state = nuln_env.reset()
            self.nuln_ai = WarhammerAIAgent(state_size=len(state), action_size=13)
            self.nuln_ai.load_model('warhammer_ai_model.pth')
            print("✅ Nuln AI loaded (96.15% win rate)")
        except Exception as e:
            print(f"❌ Error loading Nuln AI: {e}")
            
        try:
            self.troll_ai = TrollAIAgent(state_size=13, action_size=13)
            self.troll_ai.load_model('troll_ai_model.pth')
            print("✅ Troll AI loaded (99.10% win rate)")
        except Exception as e:
            print(f"❌ Error loading Troll AI: {e}")
    
    def create_simple_armies(self):
        """Create simple, clearly visible armies"""
        self.units = []
        
        # NULN ARMY (Left side - Blue)
        nuln_units = [
            SimpleUnit("General Löwenhacke", 15, 30, 1, 'blue', 120, 8),
            SimpleUnit("State Troops", 20, 35, 1, 'lightblue', 100, 5),
            SimpleUnit("Great Cannons", 10, 25, 1, 'navy', 80, 10),
            SimpleUnit("Handgunners", 15, 40, 1, 'cyan', 90, 6),
            SimpleUnit("Empire Knights", 25, 20, 1, 'purple', 110, 7),
        ]
        
        # TROLL HORDE (Right side - Red/Green)
        troll_units = [
            SimpleUnit("Bigboss Chariot", 65, 30, 2, 'darkgreen', 130, 8),
            SimpleUnit("Stone Trolls", 60, 35, 2, 'gray', 150, 9),
            SimpleUnit("River Trolls", 70, 25, 2, 'green', 120, 7),
            SimpleUnit("Goblin Riders", 55, 40, 2, 'orange', 80, 4),
            SimpleUnit("Orc Boyz", 65, 20, 2, 'red', 100, 6),
        ]
        
        self.units.extend(nuln_units + troll_units)
        print(f"🏛️ Created {len(nuln_units)} Nuln units vs {len(troll_units)} Troll units")
        
        # Print army details
        print("\n📋 ARMY ROSTER:")
        print("🔵 NULN FORCES:")
        for unit in nuln_units:
            print(f"  • {unit.name} - HP: {unit.health}, STR: {unit.strength}")
        print("🔴 TROLL HORDE:")
        for unit in troll_units:
            print(f"  • {unit.name} - HP: {unit.health}, STR: {unit.strength}")
    
    def draw_complete_battlefield(self):
        """Draw both terrain and units on the same figure"""
        # Clear and setup the axes
        self.ax.clear()
        self.ax.set_xlim(0, 80)
        self.ax.set_ylim(0, 60)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0a0a0a')  # Dark background
        
        # Draw terrain features
        # Forest
        forest = patches.Rectangle((5, 45), 20, 10, color='darkgreen', alpha=0.3)
        self.ax.add_patch(forest)
        self.ax.text(15, 50, 'FOREST', ha='center', va='center', 
                    fontsize=10, color='green', weight='bold')
        
        # River
        self.ax.plot([0, 80], [30, 30], color='blue', linewidth=6, alpha=0.5)
        self.ax.text(40, 32, 'RIVER', ha='center', va='center', 
                    fontsize=8, color='lightblue', weight='bold')
        
        # Hills
        hill = patches.Circle((65, 50), 8, color='saddlebrown', alpha=0.3)
        self.ax.add_patch(hill)
        self.ax.text(65, 50, 'HILL', ha='center', va='center', 
                    fontsize=8, color='brown', weight='bold')
        
        # Battlefield grid
        for x in range(0, 81, 20):
            self.ax.axvline(x, color='gray', alpha=0.2, linewidth=0.5)
        for y in range(0, 61, 20):
            self.ax.axhline(y, color='gray', alpha=0.2, linewidth=0.5)
        
        # Draw all units on the same plot
        units_drawn = 0
        for unit in self.units:
            if unit.is_alive():
                # Unit circle (bigger and clearer)
                circle = patches.Circle((unit.x, unit.y), unit.size, 
                                      facecolor=unit.color, alpha=0.8,
                                      edgecolor='white', linewidth=2)
                self.ax.add_patch(circle)
                
                # Health percentage
                health_pct = int((unit.health / unit.max_health) * 100)
                
                # Unit label
                self.ax.text(unit.x, unit.y, f"P{unit.player}", 
                           ha='center', va='center', fontweight='bold', 
                           color='white', fontsize=8)
                
                # Name and health above unit
                info_text = f"{unit.name}\n{health_pct}% HP"
                self.ax.text(unit.x, unit.y + 7, info_text, 
                           ha='center', va='bottom', fontsize=8, 
                           color=unit.color, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='black', alpha=0.7))
                
                # Health bar
                bar_width = 6
                bar_height = 1
                health_ratio = unit.health / unit.max_health
                
                # Background bar
                health_bg = patches.Rectangle((unit.x - bar_width/2, unit.y - 8), 
                                            bar_width, bar_height, 
                                            facecolor='darkred', alpha=0.5)
                self.ax.add_patch(health_bg)
                
                # Health bar
                health_bar = patches.Rectangle((unit.x - bar_width/2, unit.y - 8), 
                                             bar_width * health_ratio, bar_height,
                                             facecolor='green' if health_ratio > 0.5 else 'yellow' if health_ratio > 0.2 else 'red',
                                             alpha=0.8)
                self.ax.add_patch(health_bar)
                
                units_drawn += 1
        
        # Title and info
        self.ax.set_title('⚔️ AI vs AI: EPIC WARHAMMER BATTLE ⚔️', 
                         fontsize=18, fontweight='bold', color='gold', pad=15)
        
        # Army status
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        nuln_health = sum(u.health for u in nuln_units)
        troll_health = sum(u.health for u in troll_units)
        
        status = f"""TURN {self.turn}/{self.max_turns}
AI vs AI BATTLE

🔵 NULN ARMY: {len(nuln_units)} units
   Total Health: {nuln_health:.0f}

🔴 TROLL HORDE: {len(troll_units)} units  
   Total Health: {troll_health:.0f}"""
        
        self.ax.text(0.02, 0.98, status, transform=self.ax.transAxes,
                    fontsize=11, va='top', fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='darkblue', alpha=0.8))
        
        # Battle log
        if self.battle_log:
            log_text = "BATTLE LOG:\n" + "\n".join(self.battle_log[-4:])
            self.ax.text(0.98, 0.02, log_text, transform=self.ax.transAxes,
                        fontsize=9, va='bottom', ha='right', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='darkred', alpha=0.8))
        
        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        print(f"🎨 Drew {units_drawn} units with terrain on single battlefield")
        
        # Force display update
        plt.draw()
        plt.pause(0.5)  # Pause to see the battle
    
    def get_ai_states(self):
        """Get states for both AIs"""
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        # Simple states
        nuln_health = sum(u.health for u in nuln_units) / max(1, len(nuln_units))
        troll_health = sum(u.health for u in troll_units) / max(1, len(troll_units))
        
        # For Nuln AI (complex state)
        nuln_state = [
            self.turn / self.max_turns,
            nuln_health / 100.0,
            troll_health / 100.0,
        ]
        # Pad to expected size (101 features)
        while len(nuln_state) < 101:
            nuln_state.append(0.0)
        
        # For Troll AI (simple state - 13 features)
        troll_state = [
            troll_health / 100.0,
            nuln_health / 100.0,
            self.turn / self.max_turns,
            abs(troll_health - nuln_health) / 100.0,
            50.0 / 80.0,  # center positions
            30.0 / 60.0,
            30.0 / 80.0,
            30.0 / 60.0,
            1.0,  # regeneration
            0.1,  # stupidity
            1.0,  # fear
            0.0,  # stone skin
            40.0 / 100.0  # distance
        ]
        
        return np.array(nuln_state[:101], dtype=np.float32), np.array(troll_state, dtype=np.float32)
    
    def execute_turn(self):
        """Execute one turn of AI vs AI combat"""
        print(f"\n⚔️ TURN {self.turn} - AI DECISION PHASE")
        
        if self.nuln_ai and self.troll_ai:
            # Get AI states
            nuln_state, troll_state = self.get_ai_states()
            
            # Get AI actions
            nuln_action = self.nuln_ai.act(nuln_state)
            troll_action = self.troll_ai.act(troll_state)
            
            print(f"🔵 Nuln AI chooses action {nuln_action}")
            print(f"🔴 Troll AI chooses action {troll_action}")
            
            # Execute actions (movement + combat)
            self.execute_nuln_action(nuln_action)
            self.execute_troll_action(troll_action)
            
        else:
            # Random actions if no AI
            self.move_units_randomly()
            self.battle_log.append("🤖 Random AI actions (agents not loaded)")
        
        # Apply combat damage after movement
        nuln_damage = random.randint(15, 30)
        troll_damage = random.randint(15, 30)
        
        self.apply_damage(1, troll_damage)  # Trolls damage Nuln
        self.apply_damage(2, nuln_damage)  # Nuln damages Trolls
        
        print(f"💥 Damage dealt - Nuln: {nuln_damage}, Trolls: {troll_damage}")
    
    def execute_nuln_action(self, action):
        """Execute Nuln action with movement and return damage"""
        actions = ["Move North", "Move NE", "Move East", "Move SE", 
                  "Move South", "Move SW", "Move West", "Move NW",
                  "Artillery Strike", "Cavalry Charge", "Defensive Formation", 
                  "Flanking Maneuver", "Mass Shooting"]
        
        action_name = actions[action]
        
        # Move Nuln units based on action
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        
        if "Move North" in action_name:
            for unit in nuln_units:
                unit.move_towards(unit.x, min(55, unit.y + 5), 3)
        elif "Move East" in action_name:
            for unit in nuln_units:
                unit.move_towards(min(75, unit.x + 5), unit.y, 3)
        elif "Move South" in action_name:
            for unit in nuln_units:
                unit.move_towards(unit.x, max(5, unit.y - 5), 3)
        elif "Move West" in action_name:
            for unit in nuln_units:
                unit.move_towards(max(5, unit.x - 5), unit.y, 3)
        elif "Cavalry Charge" in action_name:
            # Knights charge forward
            for unit in nuln_units:
                if "Knight" in unit.name:
                    unit.move_towards(min(75, unit.x + 8), unit.y, 5)
                else:
                    unit.move_towards(min(75, unit.x + 4), unit.y, 3)
        elif "Flanking Maneuver" in action_name:
            # Units spread out and advance
            for i, unit in enumerate(nuln_units):
                target_y = unit.y + (i - 2) * 3  # Spread vertically
                unit.move_towards(min(75, unit.x + 3), target_y, 3)
        
        # Combat damage based on action
        if action_name == "Artillery Strike":
            damage = random.randint(25, 40)
            self.battle_log.append(f"💥 Nuln Artillery Strike! {damage} damage")
        elif action_name == "Mass Shooting":
            damage = random.randint(20, 30)
            self.battle_log.append(f"🏹 Nuln Mass Shooting! {damage} damage")
        elif action_name == "Cavalry Charge":
            damage = random.randint(22, 35)
            self.battle_log.append(f"🐎 Nuln Cavalry Charge! {damage} damage")
        else:
            damage = random.randint(15, 25)
            self.battle_log.append(f"⚔️ Nuln {action_name}! {damage} damage")
        
        return damage
    
    def execute_troll_action(self, action):
        """Execute Troll action with movement and return damage"""
        actions = ["Troll Charge", "Regeneration", "Stone Skin", "Fear Roar",
                  "Smash Attack", "Goblin Swarm", "Boar Charge", "Magic Attack",
                  "Move Forward", "Move Back", "Flank Left", "Flank Right", "Hold Position"]
        
        action_name = actions[action]
        
        # Move Troll units based on action
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        if "Move Forward" in action_name or "Charge" in action_name:
            for unit in troll_units:
                unit.move_towards(max(5, unit.x - 5), unit.y, 4)
        elif "Move Back" in action_name:
            for unit in troll_units:
                unit.move_towards(min(75, unit.x + 3), unit.y, 2)
        elif "Flank Left" in action_name:
            for unit in troll_units:
                unit.move_towards(max(5, unit.x - 3), min(55, unit.y + 4), 3)
        elif "Flank Right" in action_name:
            for unit in troll_units:
                unit.move_towards(max(5, unit.x - 3), max(5, unit.y - 4), 3)
        elif "Goblin Swarm" in action_name:
            # Goblins rush forward
            for unit in troll_units:
                if "Goblin" in unit.name:
                    unit.move_towards(max(5, unit.x - 8), unit.y, 6)
                else:
                    unit.move_towards(max(5, unit.x - 4), unit.y, 3)
        
        # Combat damage and effects
        if action_name == "Goblin Swarm":
            damage = random.randint(28, 42)
            self.battle_log.append(f"🏹 Goblin Swarm overwhelms! {damage} damage")
        elif action_name == "Troll Charge":
            damage = random.randint(30, 45)
            self.battle_log.append(f"💪 Massive Troll Charge! {damage} damage")
        elif action_name == "Regeneration":
            # Heal some units
            for unit in troll_units[:2]:
                heal = min(20, unit.max_health - unit.health)
                unit.health += heal
            damage = random.randint(12, 20)
            self.battle_log.append(f"🩹 Trolls regenerate and attack! {damage} damage")
        else:
            damage = random.randint(18, 32)
            self.battle_log.append(f"⚔️ Troll {action_name}! {damage} damage")
        
        return damage
    
    def move_units_randomly(self):
        """Random unit movement when AI not loaded"""
        for unit in self.units:
            if unit.is_alive():
                # Random small movements
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                unit.x = max(5, min(75, unit.x + dx))
                unit.y = max(5, min(55, unit.y + dy))
    
    def apply_damage(self, target_player, total_damage):
        """Apply damage to target army"""
        target_units = [u for u in self.units if u.player == target_player and u.is_alive()]
        if not target_units:
            return
        
        # Distribute damage
        damage_per_unit = total_damage / len(target_units)
        
        for unit in target_units:
            unit.take_damage(damage_per_unit)
            if unit.health <= 0:
                print(f"💀 {unit.name} has been slain!")
    
    def check_victory(self):
        """Check battle outcome"""
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        if not nuln_units:
            return "🟢 TROLL HORDE VICTORY! Greenskins dominate the battlefield!"
        elif not troll_units:
            return "🔵 NULN ARMY VICTORY! The Empire prevails!"
        elif self.turn >= self.max_turns:
            nuln_health = sum(u.health for u in nuln_units)
            troll_health = sum(u.health for u in troll_units)
            
            if nuln_health > troll_health:
                return "🔵 NULN VICTORY BY ATTRITION! Strategic superiority!"
            elif troll_health > nuln_health:
                return "🟢 TROLL VICTORY BY BRUTALITY! Raw power wins!"
            else:
                return "⚖️ EPIC STALEMATE! Both armies fought to exhaustion!"
        
        return None
    
    def run_battle(self):
        """Run the complete battle"""
        print("🏛️ SIMPLE VISUAL AI vs AI BATTLE!")
        print("=" * 50)
        
        # Initial display
        self.draw_complete_battlefield()
        
        while self.turn <= self.max_turns:
            print(f"\n{'='*20} TURN {self.turn} {'='*20}")
            
            # Execute turn (movement + combat)
            self.execute_turn()
            
            # Update display with new positions
            self.draw_complete_battlefield()
            
            # Check victory
            result = self.check_victory()
            if result:
                print(f"\n🏆 {result}")
                print("=" * 50)
                
                # Final stats
                nuln_alive = len([u for u in self.units if u.player == 1 and u.is_alive()])
                troll_alive = len([u for u in self.units if u.player == 2 and u.is_alive()])
                
                print(f"Final Status:")
                print(f"🔵 Nuln survivors: {nuln_alive}/5")
                print(f"🔴 Troll survivors: {troll_alive}/5")
                print(f"⚔️ Battle lasted {self.turn} turns")
                
                # Keep display open
                plt.ioff()
                plt.show()
                return
            
            self.turn += 1
            time.sleep(1.0)  # Pause between turns

def main():
    print("⚔️ SIMPLE VISUAL AI vs AI BATTLE")
    print("=" * 35)
    print("Watch trained AI armies clash in real-time!")
    print("Units will be clearly visible as colored circles")
    print()
    
    battle = SimpleVisualBattle()
    battle.run_battle()

if __name__ == "__main__":
    main() 