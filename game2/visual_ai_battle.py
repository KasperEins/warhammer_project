#!/usr/bin/env python3
"""
VISUAL AI VS AI BATTLE SYSTEM
=============================
Watch trained AI armies clash in epic visual warfare!
Nuln Army AI (96.15% win rate) vs Troll Horde AI (99.10% win rate)

Features:
- Real-time terrain visualization
- Authentic unit formations and ranks
- AI-controlled strategic decision making
- Live battle updates and commentary
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

# Import our AI systems
try:
    from warhammer_ai_agent import WarhammerAIAgent, WarhammerBattleEnvironment
    from troll_ai_trainer import TrollAIAgent, TrollBattleEnvironment
except ImportError as e:
    print(f"AI import error: {e}")

class UnitType(Enum):
    INFANTRY = "Infantry"
    CAVALRY = "Cavalry" 
    ARTILLERY = "Artillery"
    MONSTER = "Monster"
    CHARACTER = "Character"

class FormationType(Enum):
    DEEP = "Deep Formation"
    WIDE = "Wide Formation"
    SKIRMISH = "Skirmish"

@dataclass
class VisualUnit:
    name: str
    x: float
    y: float
    facing: float
    models: int
    max_models: int
    unit_type: UnitType
    formation: FormationType
    width: int
    depth: int
    health: float
    max_health: float
    strength: int
    player: int
    color: str
    weapon_range: int = 0
    
    def __post_init__(self):
        self._id = id(self)
        self.update_formation()
    
    def __hash__(self):
        return self._id
    
    def update_formation(self):
        """Update formation based on current models"""
        if self.models <= 0:
            self.width = 0
            self.depth = 0
            return
            
        if self.formation == FormationType.DEEP:
            self.width = min(5, self.models)
            self.depth = math.ceil(self.models / self.width)
        elif self.formation == FormationType.WIDE:
            self.width = min(8, self.models)
            self.depth = math.ceil(self.models / self.width)
        else:  # SKIRMISH
            self.width = min(4, self.models)
            self.depth = math.ceil(self.models / self.width)
    
    def is_alive(self):
        return self.models > 0 and self.health > 0
    
    def get_formation_points(self) -> List[Tuple[float, float]]:
        """Get formation rectangle corners"""
        if not self.is_alive():
            return []
        
        front_width = self.width * 1.2
        depth_size = self.depth * 1.2
        
        angle_rad = math.radians(self.facing)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        corners = [
            (-front_width/2, -depth_size/2),
            (front_width/2, -depth_size/2),
            (front_width/2, depth_size/2),
            (-front_width/2, depth_size/2)
        ]
        
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            rotated_corners.append((rx, ry))
        
        return rotated_corners

class VisualAIBattle:
    """Visual AI vs AI Battle System with terrain and formations"""
    
    def __init__(self):
        # Initialize matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.setup_battlefield()
        
        # Initialize AI agents
        self.nuln_ai = None
        self.troll_ai = None
        self.load_ai_agents()
        
        # Battle state
        self.turn = 1
        self.max_turns = 15
        self.battle_log = []
        self.units = []
        
        # Create armies
        self.create_armies()
        
        # Battle phase tracking
        self.phase = "Deployment"
        self.battle_active = True
        
    def setup_battlefield(self):
        """Create the Old World battlefield with terrain"""
        self.ax.clear()
        self.ax.set_xlim(0, 80)
        self.ax.set_ylim(0, 60)
        self.ax.set_aspect('equal')
        
        # Battlefield background
        self.ax.set_facecolor('#228B22')  # Forest green
        
        # Add terrain features
        # Forest (left side)
        forest = patches.Rectangle((2, 45), 15, 12, color='#006400', alpha=0.7)
        self.ax.add_patch(forest)
        self.ax.text(9.5, 51, '🌲 DARKWOOD FOREST 🌲', ha='center', va='center', 
                    fontsize=12, color='white', weight='bold')
        
        # River running through middle
        river_points = [(0, 30), (25, 28), (55, 32), (80, 30)]
        river_x = [p[0] for p in river_points]
        river_y = [p[1] for p in river_points]
        
        for i in range(len(river_x)-1):
            for offset in [-1.5, -0.5, 0.5, 1.5]:
                self.ax.plot([river_x[i], river_x[i+1]], 
                           [river_y[i]+offset, river_y[i+1]+offset], 
                           color='#4169E1', linewidth=3, alpha=0.7)
        
        # Bridge
        bridge = patches.Rectangle((38, 28), 4, 4, color='#8B4513', alpha=0.9)
        self.ax.add_patch(bridge)
        self.ax.text(40, 30, 'BRIDGE', ha='center', va='center', 
                    fontsize=8, color='white', weight='bold')
        
        # Hills (right side)
        hill = patches.Circle((65, 15), 8, color='#8B7355', alpha=0.6)
        self.ax.add_patch(hill)
        self.ax.text(65, 15, '⛰️ WATCHTOWER\nHILL ⛰️', ha='center', va='center', 
                    fontsize=10, color='white', weight='bold')
        
        # Ruins (center)
        ruins = patches.Rectangle((35, 10), 8, 6, color='#696969', alpha=0.8)
        self.ax.add_patch(ruins)
        self.ax.text(39, 13, '🏛️ ANCIENT\nRUINS 🏛️', ha='center', va='center', 
                    fontsize=10, color='white', weight='bold')
        
        self.ax.set_title('⚔️ AI vs AI: WARHAMMER THE OLD WORLD ⚔️', 
                         fontsize=24, fontweight='bold', color='gold', pad=20)
    
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
            
        # Load Troll AI
        try:
            self.troll_ai = TrollAIAgent(state_size=13, action_size=13)
            self.troll_ai.load_model('troll_ai_model.pth')
            print("✅ Troll AI loaded (99.10% win rate)")
        except Exception as e:
            print(f"❌ Error loading Troll AI: {e}")
    
    def create_armies(self):
        """Create visual armies for both sides"""
        self.units.clear()
        
        # NULN ARMY (Player 1 - Blue)
        # General von Löwenhacke
        general = VisualUnit(
            name="General von Löwenhacke", x=15, y=20, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            health=100, max_health=100, strength=8, player=1, color='darkblue'
        )
        
        # State Troops
        state_troops = VisualUnit(
            name="State Troops", x=20, y=25, facing=90,
            models=20, max_models=20, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=5, depth=4,
            health=100, max_health=100, strength=5, player=1, color='blue'
        )
        
        # Great Cannons
        cannons = VisualUnit(
            name="Great Cannons", x=10, y=15, facing=90,
            models=2, max_models=2, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=2, depth=1,
            health=100, max_health=100, strength=10, player=1, color='navy', weapon_range=48
        )
        
        # Helblaster Volley Guns
        volley_guns = VisualUnit(
            name="Helblaster Volley Guns", x=12, y=35, facing=90,
            models=2, max_models=2, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=2, depth=1,
            health=100, max_health=100, strength=7, player=1, color='lightblue', weapon_range=24
        )
        
        # Outriders
        outriders = VisualUnit(
            name="Outriders", x=25, y=12, facing=90,
            models=6, max_models=6, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=3, depth=2,
            health=100, max_health=100, strength=6, player=1, color='purple'
        )
        
        self.units.extend([general, state_troops, cannons, volley_guns, outriders])
        
        # TROLL HORDE (Player 2 - Red/Green)
        # Bigboss on Boar Chariot
        bigboss = VisualUnit(
            name="Bigboss on Boar Chariot", x=65, y=20, facing=270,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            health=100, max_health=100, strength=8, player=2, color='darkgreen'
        )
        
        # Stone Trolls
        stone_trolls = VisualUnit(
            name="Stone Trolls", x=60, y=25, facing=270,
            models=3, max_models=3, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=3, depth=1,
            health=100, max_health=100, strength=9, player=2, color='gray'
        )
        
        # River Trolls
        river_trolls = VisualUnit(
            name="River Trolls", x=62, y=35, facing=270,
            models=3, max_models=3, unit_type=UnitType.MONSTER,
            formation=FormationType.WIDE, width=3, depth=1,
            health=100, max_health=100, strength=7, player=2, color='darkgreen'
        )
        
        # Goblin Wolf Riders
        wolf_riders = VisualUnit(
            name="Goblin Wolf Riders", x=70, y=12, facing=270,
            models=8, max_models=8, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=4, depth=2,
            health=100, max_health=100, strength=4, player=2, color='orange'
        )
        
        # Orc Boyz
        orc_boyz = VisualUnit(
            name="Orc Boyz", x=58, y=15, facing=270,
            models=15, max_models=15, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=5, depth=3,
            health=100, max_health=100, strength=6, player=2, color='red'
        )
        
        self.units.extend([bigboss, stone_trolls, river_trolls, wolf_riders, orc_boyz])
        
        print(f"🏛️ Armies deployed: {len([u for u in self.units if u.player == 1])} Nuln units vs {len([u for u in self.units if u.player == 2])} Troll units")
    
    def draw_units(self):
        """Draw all units with formations and health"""
        for unit in self.units:
            if unit.is_alive():
                corners = unit.get_formation_points()
                if not corners:
                    continue
                
                # Formation rectangle with health-based transparency
                health_alpha = max(0.3, unit.health / unit.max_health)
                formation = patches.Polygon(corners, closed=True, 
                                          facecolor=unit.color, alpha=health_alpha,
                                          edgecolor='white', linewidth=2)
                self.ax.add_patch(formation)
                
                # Individual models
                models_drawn = 0
                for rank in range(unit.depth):
                    for file in range(unit.width):
                        if models_drawn >= unit.models:
                            break
                        
                        local_x = (file - unit.width/2 + 0.5) * 1.2
                        local_y = (rank - unit.depth/2 + 0.5) * 1.2
                        
                        angle_rad = math.radians(unit.facing)
                        model_x = local_x * math.cos(angle_rad) - local_y * math.sin(angle_rad) + unit.x
                        model_y = local_x * math.sin(angle_rad) + local_y * math.cos(angle_rad) + unit.y
                        
                        # Unit type specific visualization
                        if unit.unit_type == UnitType.CHARACTER:
                            marker, size = '*', 80
                        elif unit.unit_type == UnitType.MONSTER:
                            marker, size = 'o', 60
                        elif unit.unit_type == UnitType.CAVALRY:
                            marker, size = 's', 45
                        elif unit.unit_type == UnitType.ARTILLERY:
                            marker, size = '^', 50
                        else:  # INFANTRY
                            marker, size = 'o', 25
                        
                        self.ax.scatter(model_x, model_y, s=size, c=unit.color,
                                      marker=marker, edgecolors='white', linewidth=1, 
                                      alpha=health_alpha)
                        
                        models_drawn += 1
                    if models_drawn >= unit.models:
                        break
                
                # Unit label with health bar
                banner_x = unit.x
                banner_y = unit.y + unit.depth + 2
                
                health_percent = int((unit.health / unit.max_health) * 100)
                health_color = 'green' if health_percent > 60 else 'yellow' if health_percent > 30 else 'red'
                
                unit_info = f"{unit.name}\n{unit.models}/{unit.max_models} models\n{health_percent}% HP"
                
                bbox_color = 'lightblue' if unit.player == 1 else 'lightcoral'
                self.ax.text(banner_x, banner_y, unit_info, ha='center', va='bottom',
                           fontsize=8, fontweight='bold', color='black',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.9))
                
                # Health bar
                bar_width = 4
                bar_height = 0.5
                health_bar_bg = patches.Rectangle((banner_x - bar_width/2, banner_y + 3), 
                                                bar_width, bar_height, 
                                                facecolor='black', alpha=0.7)
                self.ax.add_patch(health_bar_bg)
                
                health_bar = patches.Rectangle((banner_x - bar_width/2, banner_y + 3), 
                                             bar_width * (unit.health / unit.max_health), bar_height,
                                             facecolor=health_color, alpha=0.9)
                self.ax.add_patch(health_bar)
                
                # Weapon range circle
                if unit.weapon_range > 0:
                    range_circle = patches.Circle((unit.x, unit.y), unit.weapon_range,
                                                fill=False, color=unit.color, 
                                                alpha=0.2, linestyle=':', linewidth=1)
                    self.ax.add_patch(range_circle)
                
                # Facing arrow
                arrow_length = max(unit.width, unit.depth) + 1
                arrow_x = unit.x + arrow_length * math.cos(math.radians(unit.facing))
                arrow_y = unit.y + arrow_length * math.sin(math.radians(unit.facing))
                
                self.ax.annotate('', xy=(arrow_x, arrow_y), xytext=(unit.x, unit.y),
                               arrowprops=dict(arrowstyle='->', color='yellow', 
                                             linewidth=2, alpha=0.8))
    
    def get_nuln_state(self):
        """Get battle state for Nuln AI"""
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        total_nuln_health = sum(u.health for u in nuln_units) / max(1, len(nuln_units))
        total_troll_health = sum(u.health for u in troll_units) / max(1, len(troll_units))
        
        state_features = []
        
        # Global battle info (3 features)
        state_features.extend([
            self.turn / self.max_turns,
            total_nuln_health / 100.0,
            total_troll_health / 100.0,
        ])
        
        # Nuln army state (simplified to match expected dimensions)
        for i in range(78):  # Pad to expected 78 features for army state
            if i < len(nuln_units) * 6:
                unit_idx = i // 6
                feature_idx = i % 6
                if unit_idx < len(nuln_units):
                    unit = nuln_units[unit_idx]
                    if feature_idx == 0:
                        state_features.append(unit.x / 80.0)
                    elif feature_idx == 1:
                        state_features.append(unit.y / 60.0)
                    elif feature_idx == 2:
                        state_features.append(unit.health / 100.0)
                    elif feature_idx == 3:
                        state_features.append(1.0 if unit.is_alive() else 0.0)
                    elif feature_idx == 4:
                        state_features.append(unit.weapon_range / 48.0)
                    else:
                        state_features.append(0.0)
                else:
                    state_features.append(0.0)
            else:
                state_features.append(0.0)
        
        # Enemy army state (20 features)
        for i in range(20):
            if i < len(troll_units) * 4:
                unit_idx = i // 4
                feature_idx = i % 4
                if unit_idx < len(troll_units):
                    unit = troll_units[unit_idx]
                    if feature_idx == 0:
                        state_features.append(unit.x / 80.0)
                    elif feature_idx == 1:
                        state_features.append(unit.y / 60.0)
                    elif feature_idx == 2:
                        state_features.append(unit.health / 100.0)
                    else:
                        state_features.append(1.0 if unit.is_alive() else 0.0)
                else:
                    state_features.append(0.0)
            else:
                state_features.append(0.0)
        
        return np.array(state_features, dtype=np.float32)
    
    def get_troll_state(self):
        """Get battle state for Troll AI"""
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        total_nuln_health = sum(u.health for u in nuln_units) / max(1, len(nuln_units))
        total_troll_health = sum(u.health for u in troll_units) / max(1, len(troll_units))
        
        # Calculate center positions
        if nuln_units:
            nuln_center_x = sum(u.x for u in nuln_units) / len(nuln_units)
            nuln_center_y = sum(u.y for u in nuln_units) / len(nuln_units)
        else:
            nuln_center_x = nuln_center_y = 0
        
        if troll_units:
            troll_center_x = sum(u.x for u in troll_units) / len(troll_units)
            troll_center_y = sum(u.y for u in troll_units) / len(troll_units)
        else:
            troll_center_x = troll_center_y = 0
        
        distance = math.sqrt((troll_center_x - nuln_center_x)**2 + (troll_center_y - nuln_center_y)**2)
        
        return np.array([
            total_troll_health / 100.0,
            total_nuln_health / 100.0,
            self.turn / self.max_turns,
            (50.0 - abs(total_troll_health - total_nuln_health)) / 50.0,
            troll_center_x / 80.0,
            troll_center_y / 60.0,
            nuln_center_x / 80.0,
            nuln_center_y / 60.0,
            1.0,  # Regeneration available
            0.1,  # Stupidity risk
            1.0,  # Fear aura
            0.0,  # Stone skin active
            distance / 100.0
        ])
    
    def execute_ai_actions(self):
        """Get actions from both AIs and execute them"""
        if not self.nuln_ai or not self.troll_ai:
            return
        
        # Get AI decisions
        nuln_state = self.get_nuln_state()
        troll_state = self.get_troll_state()
        
        nuln_action = self.nuln_ai.act(nuln_state)
        troll_action = self.troll_ai.act(troll_state)
        
        # Execute actions
        nuln_damage = self.execute_nuln_action(nuln_action)
        troll_damage = self.execute_troll_action(troll_action)
        
        # Apply damage
        self.apply_damage_to_army(2, nuln_damage)  # Nuln damages Trolls
        self.apply_damage_to_army(1, troll_damage)  # Trolls damage Nuln
        
        return nuln_action, troll_action
    
    def execute_nuln_action(self, action):
        """Execute Nuln AI action"""
        nuln_actions = [
            "Move North", "Move NE", "Move East", "Move SE", 
            "Move South", "Move SW", "Move West", "Move NW",
            "Artillery Strike", "Cavalry Charge", "Defensive Formation", 
            "Flanking Maneuver", "Mass Shooting"
        ]
        
        action_name = nuln_actions[action]
        damage = 0
        
        # Move units slightly based on action
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        
        if "Move" in action_name:
            move_x, move_y = 0, 0
            if "North" in action_name: move_y = 2
            if "South" in action_name: move_y = -2
            if "East" in action_name: move_x = 2
            if "West" in action_name: move_x = -2
            
            for unit in nuln_units:
                unit.x = max(5, min(75, unit.x + move_x))
                unit.y = max(5, min(55, unit.y + move_y))
            
            damage = random.randint(8, 15)
            self.battle_log.append(f"🔵 Nuln forces {action_name} and engage for {damage} damage")
        
        elif action_name == "Artillery Strike":
            damage = random.randint(22, 35)
            self.battle_log.append(f"💥 Nuln Artillery Strike devastates enemy for {damage} damage!")
        
        elif action_name == "Mass Shooting":
            damage = random.randint(15, 25)
            self.battle_log.append(f"🏹 Nuln Mass Shooting volleys deal {damage} damage!")
        
        elif action_name == "Cavalry Charge":
            damage = random.randint(18, 28)
            self.battle_log.append(f"🐎 Nuln Cavalry Charge crashes home for {damage} damage!")
        
        else:
            damage = random.randint(12, 20)
            self.battle_log.append(f"⚔️ Nuln {action_name} inflicts {damage} damage")
        
        return damage
    
    def execute_troll_action(self, action):
        """Execute Troll AI action"""
        troll_actions = [
            "Troll Charge", "Regeneration", "Stone Skin", "Fear Roar",
            "Smash Attack", "Goblin Swarm", "Boar Charge", "Magic Attack",
            "Move Forward", "Move Back", "Flank Left", "Flank Right", "Hold Position"
        ]
        
        action_name = troll_actions[action]
        damage = 0
        
        # Move units slightly based on action
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        if "Move" in action_name or "Flank" in action_name:
            move_x, move_y = 0, 0
            if "Forward" in action_name: move_x = -2
            if "Back" in action_name: move_x = 2
            if "Left" in action_name: move_y = 2
            if "Right" in action_name: move_y = -2
            
            for unit in troll_units:
                unit.x = max(5, min(75, unit.x + move_x))
                unit.y = max(5, min(55, unit.y + move_y))
            
            damage = random.randint(10, 18)
            self.battle_log.append(f"🔴 Troll Horde {action_name} and attacks for {damage} damage")
        
        elif action_name == "Goblin Swarm":
            damage = random.randint(20, 30)
            self.battle_log.append(f"🏹 Goblin Swarm overwhelms the enemy for {damage} damage!")
        
        elif action_name == "Troll Charge":
            damage = random.randint(25, 35)
            self.battle_log.append(f"💪 Massive Troll Charge smashes for {damage} damage!")
        
        elif action_name == "Regeneration":
            # Heal some units
            for unit in troll_units[:2]:
                heal_amount = min(15, unit.max_health - unit.health)
                unit.health += heal_amount
            damage = random.randint(8, 15)
            self.battle_log.append(f"🩹 Trolls regenerate wounds and attack for {damage} damage")
        
        else:
            damage = random.randint(12, 22)
            self.battle_log.append(f"⚔️ Troll {action_name} deals {damage} damage")
        
        return damage
    
    def apply_damage_to_army(self, target_player, damage):
        """Apply damage to target army"""
        target_units = [u for u in self.units if u.player == target_player and u.is_alive()]
        if not target_units:
            return
        
        # Distribute damage across units
        damage_per_unit = damage / len(target_units)
        
        for unit in target_units:
            unit.health = max(0, unit.health - damage_per_unit)
            
            # Remove models based on health loss
            health_ratio = unit.health / unit.max_health
            unit.models = max(0, int(unit.max_models * health_ratio))
            unit.update_formation()
    
    def update_display(self):
        """Update the visual display"""
        self.setup_battlefield()
        self.draw_units()
        
        # Army status
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        nuln_models = sum(u.models for u in nuln_units)
        troll_models = sum(u.models for u in troll_units)
        nuln_health = sum(u.health for u in nuln_units) / max(1, len(nuln_units))
        troll_health = sum(u.health for u in troll_units) / max(1, len(troll_units))
        
        status = f"""TURN {self.turn} - {self.phase}
AI vs AI EPIC BATTLE

🔵 NULN ARMY (96.15% AI): {len(nuln_units)} units, {nuln_models} models
   Average Health: {nuln_health:.1f}%

🔴 TROLL HORDE (99.10% AI): {len(troll_units)} units, {troll_models} models
   Average Health: {troll_health:.1f}%"""
        
        self.ax.text(0.02, 0.98, status, transform=self.ax.transAxes,
                    fontsize=12, va='top', fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='navy', alpha=0.9))
        
        # Battle log
        if self.battle_log:
            log_text = "⚔️ BATTLE LOG ⚔️\n" + "\n".join(self.battle_log[-5:])
            self.ax.text(0.98, 0.02, log_text, transform=self.ax.transAxes,
                        fontsize=10, va='bottom', ha='right', color='white',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='darkgreen', alpha=0.9))
        
        plt.draw()
        plt.pause(0.1)
    
    def check_victory(self):
        """Check if battle is over"""
        nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        if not nuln_units:
            return "Troll Horde Victory! The greenskins have triumphed!"
        elif not troll_units:
            return "Nuln Army Victory! The Empire stands strong!"
        elif self.turn >= self.max_turns:
            nuln_strength = sum(u.health * u.models for u in nuln_units)
            troll_strength = sum(u.health * u.models for u in troll_units)
            
            if nuln_strength > troll_strength:
                return "Nuln Victory by Points! Strategic supremacy achieved!"
            elif troll_strength > nuln_strength:
                return "Troll Victory by Points! Brutal dominance displayed!"
            else:
                return "Epic Draw! Both armies fought to exhaustion!"
        
        return None
    
    def run_battle(self):
        """Run the complete AI vs AI battle"""
        print("🏛️ VISUAL AI vs AI BATTLE COMMENCING!")
        print("=" * 60)
        
        self.phase = "Battle Begins"
        self.update_display()
        time.sleep(2)
        
        while self.battle_active and self.turn <= self.max_turns:
            print(f"\n⚔️ TURN {self.turn} - AI COMMAND PHASE")
            
            self.phase = f"Turn {self.turn} - AI Commands"
            
            # Execute AI actions
            nuln_action, troll_action = self.execute_ai_actions()
            
            self.update_display()
            
            # Check for victory
            victory_result = self.check_victory()
            if victory_result:
                self.phase = victory_result
                self.update_display()
                
                print(f"\n🏆 {victory_result}")
                print("=" * 60)
                
                # Final statistics
                nuln_units = [u for u in self.units if u.player == 1 and u.is_alive()]
                troll_units = [u for u in self.units if u.player == 2 and u.is_alive()]
                
                print(f"Final Army Status:")
                print(f"🔵 Nuln: {len(nuln_units)} units surviving")
                print(f"🔴 Trolls: {len(troll_units)} units surviving")
                print(f"⚔️ Battle Duration: {self.turn} turns")
                print(f"📜 Battle Events: {len(self.battle_log)} recorded")
                
                break
            
            self.turn += 1
            time.sleep(1.5)  # Pause between turns for viewing
        
        # Keep display open
        print("\n📺 Battle visualization complete. Close the window to exit.")
        plt.ioff()
        plt.show()

def main():
    """Launch the Visual AI vs AI Battle System"""
    print("⚔️ VISUAL AI vs AI BATTLE SYSTEM")
    print("=" * 40)
    print("Preparing epic AI warfare with terrain visualization...")
    print("Watch as trained AI commanders clash in real-time!")
    print()
    
    battle = VisualAIBattle()
    battle.run_battle()

if __name__ == "__main__":
    main() 