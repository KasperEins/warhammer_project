#!/usr/bin/env python3
"""
AUTHENTIC WARHAMMER: THE OLD WORLD VISUAL BATTLE
=================================================
Proper ranked formations with regiment blocks
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import math

# Import our AI systems
try:
    from warhammer_ai_agent import WarhammerAIAgent, WarhammerBattleEnvironment
    from troll_ai_trainer import TrollAIAgent, TrollBattleEnvironment
    HAVE_AI = True
except ImportError as e:
    print(f"AI import error: {e}")
    HAVE_AI = False

class TOWUnit:
    def __init__(self, name, x, y, player, color, models=20, formation="deep"):
        self.name = name
        self.center_x = x
        self.center_y = y
        self.player = player
        self.color = color
        self.models = models
        self.max_models = models
        self.formation = formation  # "deep", "wide", "skirmish"
        self.facing = 0 if player == 1 else 180  # degrees
        self.width = 0
        self.height = 0
        self.calculate_formation()
        
    def calculate_formation(self):
        """Calculate unit dimensions based on formation and model count"""
        if self.formation == "deep":
            # 5 wide, multiple ranks
            self.width = 5
            self.height = max(1, self.models // 5)
        elif self.formation == "wide":
            # 10 wide, fewer ranks
            self.width = min(10, self.models)
            self.height = max(1, self.models // 10)
        elif self.formation == "skirmish":
            # Loose formation
            self.width = min(8, self.models)
            self.height = max(1, self.models // 8)
        
        # Minimum dimensions
        self.width = max(2, self.width)
        self.height = max(1, self.height)
        
    def is_alive(self):
        return self.models > 0
        
    def take_casualties(self, casualties):
        """Remove models from the unit"""
        self.models = max(0, self.models - casualties)
        if self.models > 0:
            self.calculate_formation()
            
    def get_bounds(self):
        """Get unit rectangle bounds"""
        half_w = self.width / 2
        half_h = self.height / 2
        return (self.center_x - half_w, self.center_y - half_h,
                self.width, self.height)
    
    def move_towards(self, target_x, target_y, speed=3):
        """Move unit formation towards target"""
        dx = target_x - self.center_x
        dy = target_y - self.center_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > speed:
            self.center_x += (dx / distance) * speed
            self.center_y += (dy / distance) * speed
            # Update facing
            self.facing = math.degrees(math.atan2(dy, dx))
        else:
            self.center_x = target_x
            self.center_y = target_y

class AuthenticTOWBattle:
    def __init__(self):
        # Setup matplotlib
        plt.ion()
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        
        # Battle parameters
        self.battlefield_width = 120
        self.battlefield_height = 80
        self.turn = 1
        self.max_turns = 10
        self.battle_log = []
        
        # Load AI
        self.nuln_ai = None
        self.troll_ai = None
        if HAVE_AI:
            self.load_ai_agents()
        
        # Create authentic TOW armies
        self.units = []
        self.create_tow_armies()
        
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
    
    def create_tow_armies(self):
        """Create authentic The Old World armies with proper formations"""
        print("🏛️ DEPLOYING THE OLD WORLD ARMIES")
        print("=" * 50)
        
        # NULN EMPIRE ARMY (Left deployment zone)
        empire_units = [
            # Infantry Regiment
            TOWUnit("State Troops Regiment", 25, 40, 1, 'lightblue', 25, "deep"),
            # Detachment 
            TOWUnit("Handgunners Detachment", 20, 50, 1, 'cyan', 10, "wide"),
            # War Machine
            TOWUnit("Great Cannon", 15, 35, 1, 'navy', 3, "skirmish"),
            # Cavalry
            TOWUnit("Empire Knights", 30, 25, 1, 'purple', 8, "wide"),
            # Character
            TOWUnit("General von Löwenhacke", 25, 30, 1, 'gold', 1, "skirmish"),
        ]
        
        # GREENSKIN HORDE (Right deployment zone)
        greenskin_units = [
            # Big Unit
            TOWUnit("Orc Boyz Mob", 95, 40, 2, 'green', 30, "deep"),
            # Monsters
            TOWUnit("Troll Pack", 90, 50, 2, 'gray', 6, "wide"),
            # Fast Cavalry  
            TOWUnit("Goblin Wolf Riders", 85, 25, 2, 'orange', 10, "wide"),
            # Chariot
            TOWUnit("Bigboss Chariot", 100, 35, 2, 'darkgreen', 1, "skirmish"),
            # Shooters
            TOWUnit("Goblin Archers", 90, 60, 2, 'yellow', 15, "wide"),
        ]
        
        self.units = empire_units + greenskin_units
        
        print("🔵 NULN EMPIRE DEPLOYMENT:")
        for unit in empire_units:
            print(f"  • {unit.name}: {unit.models} models in {unit.formation} formation")
        
        print("🟢 GREENSKIN HORDE DEPLOYMENT:")
        for unit in greenskin_units:
            print(f"  • {unit.name}: {unit.models} models in {unit.formation} formation")
    
    def draw_battlefield(self):
        """Draw The Old World battlefield with terrain and units"""
        # Clear and setup
        self.ax.clear()
        self.ax.set_xlim(0, self.battlefield_width)
        self.ax.set_ylim(0, self.battlefield_height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a3d1a')  # Dark green battlefield
        
        # Battlefield grid (for reference)
        for x in range(0, self.battlefield_width+1, 20):
            self.ax.axvline(x, color='darkgreen', alpha=0.3, linewidth=0.5)
        for y in range(0, self.battlefield_height+1, 20):
            self.ax.axhline(y, color='darkgreen', alpha=0.3, linewidth=0.5)
        
        # Terrain features
        self.draw_terrain()
        
        # Deployment zones
        # Empire zone (left)
        empire_zone = patches.Rectangle((0, 0), 40, self.battlefield_height, 
                                      fill=False, edgecolor='lightblue', 
                                      linewidth=2, linestyle='--', alpha=0.5)
        self.ax.add_patch(empire_zone)
        self.ax.text(20, 5, 'EMPIRE DEPLOYMENT', ha='center', va='center',
                    fontsize=10, color='lightblue', weight='bold', alpha=0.7)
        
        # Greenskin zone (right)
        greenskin_zone = patches.Rectangle((80, 0), 40, self.battlefield_height,
                                         fill=False, edgecolor='green',
                                         linewidth=2, linestyle='--', alpha=0.5)
        self.ax.add_patch(greenskin_zone)
        self.ax.text(100, 5, 'GREENSKIN DEPLOYMENT', ha='center', va='center',
                    fontsize=10, color='green', weight='bold', alpha=0.7)
        
        # Draw all units as proper formations
        self.draw_tow_units()
        
        # Battle info
        self.draw_battle_info()
        
        # Title
        self.ax.set_title('⚔️ WARHAMMER: THE OLD WORLD - EPIC BATTLE ⚔️', 
                         fontsize=20, fontweight='bold', color='gold', pad=20)
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        plt.draw()
        plt.pause(0.8)
    
    def draw_terrain(self):
        """Draw authentic Old World terrain"""
        # Forest
        forest = patches.Ellipse((30, 65), 20, 12, color='darkgreen', alpha=0.6)
        self.ax.add_patch(forest)
        self.ax.text(30, 65, 'DARKWOOD\nFOREST', ha='center', va='center',
                    fontsize=9, color='white', weight='bold')
        
        # Hill
        hill = patches.Circle((90, 20), 12, color='saddlebrown', alpha=0.4)
        self.ax.add_patch(hill)
        self.ax.text(90, 20, 'WATCHTOWER\nHILL', ha='center', va='center',
                    fontsize=8, color='white', weight='bold')
        
        # River
        self.ax.plot([0, self.battlefield_width], [40, 40], 
                    color='blue', linewidth=8, alpha=0.6)
        self.ax.text(60, 42, 'RIVER REIK', ha='center', va='center',
                    fontsize=10, color='lightblue', weight='bold')
        
        # Bridge
        bridge = patches.Rectangle((55, 38), 10, 4, color='brown', alpha=0.8)
        self.ax.add_patch(bridge)
        self.ax.text(60, 40, 'BRIDGE', ha='center', va='center',
                    fontsize=7, color='white', weight='bold')
        
        # Ruins
        ruins = patches.Rectangle((70, 55), 8, 8, fill=False, 
                                edgecolor='gray', linewidth=2)
        self.ax.add_patch(ruins)
        self.ax.text(74, 59, 'ANCIENT\nRUINS', ha='center', va='center',
                    fontsize=7, color='gray', weight='bold')
    
    def draw_tow_units(self):
        """Draw units as proper The Old World regiment blocks"""
        units_drawn = 0
        
        for unit in self.units:
            if unit.is_alive():
                # Get unit bounds
                x, y, width, height = unit.get_bounds()
                
                # Main formation rectangle
                if unit.formation == "skirmish":
                    # Skirmishers - loose formation
                    for i in range(unit.models):
                        offset_x = (i % 4) * 1.5 - 2.5
                        offset_y = (i // 4) * 1.5 - 1
                        model_x = unit.center_x + offset_x
                        model_y = unit.center_y + offset_y
                        
                        model = patches.Circle((model_x, model_y), 0.3,
                                             facecolor=unit.color, alpha=0.8,
                                             edgecolor='white', linewidth=1)
                        self.ax.add_patch(model)
                else:
                    # Ranked formation - solid rectangle
                    formation_rect = patches.Rectangle((x, y), width, height,
                                                     facecolor=unit.color, alpha=0.7,
                                                     edgecolor='white', linewidth=2)
                    self.ax.add_patch(formation_rect)
                    
                    # Add rank markers to show individual models
                    for rank in range(int(height)):
                        for file in range(int(width)):
                            model_num = rank * int(width) + file + 1
                            if model_num <= unit.models:
                                model_x = x + file + 0.5
                                model_y = y + rank + 0.5
                                
                                # Individual model marker
                                self.ax.plot(model_x, model_y, 'o', 
                                           color='white', markersize=2, alpha=0.8)
                
                # Unit label with proper TOW info
                label_text = f"{unit.name}\n{unit.models}/{unit.max_models} models\n{unit.formation.upper()}"
                
                # Calculate label position (above unit)
                label_y = y + height + 2
                
                self.ax.text(unit.center_x, label_y, label_text,
                           ha='center', va='bottom', fontsize=8,
                           color=unit.color, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='black', alpha=0.8))
                
                # Facing arrow for formations
                if unit.formation != "skirmish":
                    arrow_length = max(width, height) * 0.7
                    arrow_x = arrow_length * math.cos(math.radians(unit.facing))
                    arrow_y = arrow_length * math.sin(math.radians(unit.facing))
                    
                    self.ax.arrow(unit.center_x, unit.center_y,
                                arrow_x, arrow_y,
                                head_width=1, head_length=1,
                                fc='white', ec='white', alpha=0.6)
                
                units_drawn += 1
        
        print(f"🎨 Drew {units_drawn} TOW regiment formations")
    
    def draw_battle_info(self):
        """Draw battle information panel"""
        # Count armies
        empire_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        greenskin_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        empire_models = sum(u.models for u in empire_units)
        greenskin_models = sum(u.models for u in greenskin_units)
        
        # Battle status
        status_text = f"""TURN {self.turn}/{self.max_turns}
THE OLD WORLD BATTLE

🔵 EMPIRE ARMY
   Units: {len(empire_units)}
   Models: {empire_models}

🟢 GREENSKIN HORDE  
   Units: {len(greenskin_units)}
   Models: {greenskin_models}"""
        
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes,
                    fontsize=12, va='top', fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='darkblue', alpha=0.9))
        
        # Battle log
        if self.battle_log:
            log_text = "BATTLE LOG:\n" + "\n".join(self.battle_log[-3:])
            self.ax.text(0.98, 0.02, log_text, transform=self.ax.transAxes,
                        fontsize=10, va='bottom', ha='right', color='white',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='darkred', alpha=0.9))
    
    def execute_battle_turn(self):
        """Execute one turn of The Old World battle"""
        print(f"\n⚔️ TURN {self.turn} - THE OLD WORLD")
        print("=" * 40)
        
        # Movement Phase
        print("📍 MOVEMENT PHASE")
        self.movement_phase()
        
        # Shooting Phase  
        print("🏹 SHOOTING PHASE")
        self.shooting_phase()
        
        # Combat Phase
        print("⚔️ COMBAT PHASE")
        self.combat_phase()
        
    def movement_phase(self):
        """The Old World movement phase with AI decisions"""
        if self.nuln_ai and self.troll_ai:
            # Get AI decisions
            empire_state, greenskin_state = self.get_ai_states()
            empire_action = self.nuln_ai.act(empire_state)
            greenskin_action = self.troll_ai.act(greenskin_state)
            
            print(f"  🔵 Empire AI: Action {empire_action}")
            print(f"  🟢 Greenskin AI: Action {greenskin_action}")
            
            # Execute AI movements
            self.execute_empire_movement(empire_action)
            self.execute_greenskin_movement(greenskin_action)
        else:
            # Default movement towards center
            self.default_movement()
    
    def execute_empire_movement(self, action):
        """Execute Empire movement based on AI decision"""
        empire_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        
        actions = ["Advance", "Hold", "Refuse Flank", "Wheel", 
                  "Reform Wide", "Reform Deep", "March", "Charge"]
        
        action_name = actions[min(action, len(actions)-1)]
        
        if action_name == "Advance":
            for unit in empire_units:
                unit.move_towards(unit.center_x + 8, unit.center_y, 6)
        elif action_name == "Reform Wide":
            for unit in empire_units:
                if unit.formation == "deep":
                    unit.formation = "wide"
                    unit.calculate_formation()
        elif action_name == "Charge":
            # Find nearest enemy and charge
            for unit in empire_units:
                if unit.name == "Empire Knights":
                    unit.move_towards(unit.center_x + 15, unit.center_y, 12)
        
        self.battle_log.append(f"🔵 Empire: {action_name}")
    
    def execute_greenskin_movement(self, action):
        """Execute Greenskin movement based on AI decision"""
        greenskin_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        actions = ["Waaagh!", "Hold", "Mob Up", "Stomp Forward",
                  "Troll Charge", "Wolf Rider Flank", "Goblin Swarm", "Advance"]
        
        action_name = actions[min(action, len(actions)-1)]
        
        if action_name == "Waaagh!":
            for unit in greenskin_units:
                unit.move_towards(unit.center_x - 10, unit.center_y, 8)
        elif action_name == "Troll Charge":
            for unit in greenskin_units:
                if "Troll" in unit.name:
                    unit.move_towards(unit.center_x - 15, unit.center_y, 10)
        elif action_name == "Wolf Rider Flank":
            for unit in greenskin_units:
                if "Wolf" in unit.name:
                    unit.move_towards(unit.center_x - 12, unit.center_y + 15, 14)
        
        self.battle_log.append(f"🟢 Greenskins: {action_name}")
    
    def default_movement(self):
        """Default movement when AI not available"""
        for unit in self.units:
            if unit.is_alive():
                if unit.player == 1:  # Empire advances
                    unit.move_towards(unit.center_x + 4, unit.center_y, 4)
                else:  # Greenskins advance
                    unit.move_towards(unit.center_x - 4, unit.center_y, 4)
    
    def shooting_phase(self):
        """The Old World shooting phase"""
        shooters = [u for u in self.units if u.is_alive() and 
                   ("Handgunners" in u.name or "Archers" in u.name or "Cannon" in u.name)]
        
        for shooter in shooters:
            # Find nearest enemy
            enemies = [u for u in self.units if u.player != shooter.player and u.is_alive()]
            if enemies:
                target = min(enemies, key=lambda e: 
                           math.sqrt((e.center_x - shooter.center_x)**2 + 
                                   (e.center_y - shooter.center_y)**2))
                
                # Calculate shots and hits
                if "Cannon" in shooter.name:
                    casualties = random.randint(2, 6)
                    self.battle_log.append(f"💥 {shooter.name} fires at {target.name}!")
                else:
                    casualties = random.randint(1, 3)
                    self.battle_log.append(f"🏹 {shooter.name} shoots {target.name}")
                
                target.take_casualties(casualties)
                if target.models <= 0:
                    self.battle_log.append(f"💀 {target.name} destroyed!")
    
    def combat_phase(self):
        """The Old World combat phase"""
        # Simple combat - units in close proximity fight
        for unit in self.units:
            if unit.is_alive():
                enemies = [u for u in self.units if u.player != unit.player and u.is_alive()]
                for enemy in enemies:
                    distance = math.sqrt((enemy.center_x - unit.center_x)**2 + 
                                       (enemy.center_y - unit.center_y)**2)
                    
                    if distance < 8:  # In combat
                        # Both units take casualties
                        unit_casualties = random.randint(1, 3)
                        enemy_casualties = random.randint(1, 3)
                        
                        unit.take_casualties(enemy_casualties)
                        enemy.take_casualties(unit_casualties)
                        
                        self.battle_log.append(f"⚔️ {unit.name} fights {enemy.name}")
                        break
    
    def get_ai_states(self):
        """Get AI states for decision making"""
        empire_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        greenskin_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        empire_strength = sum(u.models for u in empire_units)
        greenskin_strength = sum(u.models for u in greenskin_units)
        
        # Simple states for AI
        empire_state = np.array([self.turn/self.max_turns, empire_strength/100.0, greenskin_strength/100.0] + [0.0]*98, dtype=np.float32)
        greenskin_state = np.array([greenskin_strength/100.0, empire_strength/100.0, self.turn/self.max_turns] + [0.0]*10, dtype=np.float32)
        
        return empire_state, greenskin_state
    
    def check_victory(self):
        """Check battle outcome"""
        empire_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        greenskin_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        if not empire_units:
            return "🟢 GREENSKIN VICTORY! The Waaagh! overwhelms the Empire!"
        elif not greenskin_units:
            return "🔵 EMPIRE VICTORY! Order triumphs over Chaos!"
        elif self.turn >= self.max_turns:
            empire_models = sum(u.models for u in empire_units)
            greenskin_models = sum(u.models for u in greenskin_units)
            
            if empire_models > greenskin_models:
                return "🔵 EMPIRE VICTORY! Strategic superiority!"
            elif greenskin_models > empire_models:
                return "🟢 GREENSKIN VICTORY! Brutal dominance!"
            else:
                return "⚖️ DRAW! Both armies fought to exhaustion!"
        
        return None
    
    def run_battle(self):
        """Run the complete The Old World battle"""
        print("🏛️ WARHAMMER: THE OLD WORLD - EPIC BATTLE!")
        print("=" * 55)
        print("Watch authentic regiment formations clash!")
        print()
        
        # Initial deployment
        self.draw_battlefield()
        
        while self.turn <= self.max_turns:
            print(f"\n{'='*25} TURN {self.turn} {'='*25}")
            
            # Execute battle turn
            self.execute_battle_turn()
            
            # Update display
            self.draw_battlefield()
            
            # Check victory
            result = self.check_victory()
            if result:
                print(f"\n🏆 {result}")
                print("=" * 55)
                
                # Final army status
                empire_survivors = len([u for u in self.units if u.player == 1 and u.is_alive()])
                greenskin_survivors = len([u for u in self.units if u.player == 2 and u.is_alive()])
                
                print(f"Final Status:")
                print(f"🔵 Empire units remaining: {empire_survivors}/5")
                print(f"🟢 Greenskin units remaining: {greenskin_survivors}/5")
                print(f"⚔️ Battle lasted {self.turn} turns")
                
                # Keep display open
                plt.ioff()
                plt.show()
                return
            
            self.turn += 1
            time.sleep(2.0)  # Pause to appreciate the formations

def main():
    print("⚔️ WARHAMMER: THE OLD WORLD - AUTHENTIC BATTLE")
    print("=" * 50)
    print("Experience proper ranked formations!")
    print("• Regiment blocks with individual models")
    print("• Authentic TOW movement and phases")
    print("• AI-controlled army commanders")
    print()
    
    battle = AuthenticTOWBattle()
    battle.run_battle()

if __name__ == "__main__":
    main() 