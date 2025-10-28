#!/usr/bin/env python3
"""
Warhammer: The Old World - Epic Battle System
Ranked formations, cavalry charges, artillery, and massive battles
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math
import time
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

print("⚔️ WARHAMMER: THE OLD WORLD - EPIC BATTLE SYSTEM")
print("=" * 55)

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
class OldWorldUnit:
    name: str
    x: float
    y: float
    facing: float  # Direction facing (degrees)
    models: int
    max_models: int
    unit_type: UnitType
    formation: FormationType
    width: int  # Models wide
    depth: int  # Ranks deep
    
    # Stats
    movement: int
    weapon_skill: int
    ballistic_skill: int
    strength: int
    toughness: int
    wounds: int
    attacks: int
    leadership: int
    armor_save: int
    
    # Battle state
    player: int
    color: str
    weapon_range: int = 0
    has_charged: bool = False
    is_fleeing: bool = False
    
    # Additional properties
    has_standard: bool = False
    has_musician: bool = False
    armor_piercing: bool = False
    lance_formation: bool = False
    frenzy: bool = False
    fast_cavalry: bool = False
    fear: bool = False
    immune_to_fear: bool = False
    stubborn: bool = False
    terror: bool = False
    regeneration: bool = False
    stupidity: bool = False
    
    # Points cost for victory calculations
    points_cost: int = 0
    starting_models: int = 0  # Track original size for VP calculation
    
    def __post_init__(self):
        # Create unique ID for hashing FIRST
        self._id = id(self)
        self.starting_models = self.models  # Track original size
        self.update_formation()
    
    def __hash__(self):
        return self._id
    
    def __eq__(self, other):
        if isinstance(other, OldWorldUnit):
            return self._id == other._id
        return False
    
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
            self.width = min(10, self.models)
            self.depth = math.ceil(self.models / self.width)
        else:  # SKIRMISH
            self.width = min(6, self.models)
            self.depth = math.ceil(self.models / self.width)
    
    def is_alive(self):
        return self.models > 0 and not self.is_fleeing
    
    def get_formation_points(self) -> List[Tuple[float, float]]:
        """Get formation rectangle corners"""
        if not self.is_alive():
            return []
        
        front_width = self.width * 1.0
        depth_size = self.depth * 1.0
        
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

class OldWorldBattle:
    def __init__(self):
        self.width = 72  # 6 feet
        self.height = 48  # 4 feet
        self.units = []
        self.turn = 1
        self.phase = "Start"
        self.current_player = 1
        self.battle_events = []
        
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(18, 12))
        self.setup_battlefield()
    
    def setup_battlefield(self):
        """Create Old World battlefield"""
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#2a4d1a')
        
        # Distance grid
        for i in range(0, self.width, 6):
            self.ax.axvline(i, color='#4a6d2a', alpha=0.4, linewidth=1)
            if i > 0:
                self.ax.text(i, 2, f'{i}"', ha='center', va='bottom', 
                           fontsize=8, color='lightgreen', alpha=0.7)
        
        for i in range(0, self.height, 6):
            self.ax.axhline(i, color='#4a6d2a', alpha=0.4, linewidth=1)
            if i > 0:
                self.ax.text(2, i, f'{i}"', ha='left', va='center', 
                           fontsize=8, color='lightgreen', alpha=0.7)
        
        # Terrain
        # Ancient Forest
        forest = patches.Rectangle((15, 8), 18, 12, color='#0d3300', alpha=0.8)
        self.ax.add_patch(forest)
        tree_positions = [(18, 12), (22, 15), (26, 10), (30, 16)]
        for tx, ty in tree_positions:
            tree = patches.Circle((tx, ty), 1.5, color='#228B22', alpha=0.9)
            self.ax.add_patch(tree)
        self.ax.text(24, 14, 'DARKWOOD FOREST', ha='center', va='center', 
                    fontsize=11, color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='darkgreen', alpha=0.8))
        
        # Hill
        hill = patches.Ellipse((55, 30), 16, 10, color='#8B4513', alpha=0.7)
        self.ax.add_patch(hill)
        self.ax.text(55, 30, 'VALIANT HILL', ha='center', va='center', 
                    fontsize=10, color='white', weight='bold')
        
        # Road
        road = patches.Rectangle((0, 22), self.width, 4, color='#8B7355', alpha=0.6)
        self.ax.add_patch(road)
        self.ax.text(36, 24, 'THE GREAT ROAD', ha='center', va='center', 
                    fontsize=9, color='gold', weight='bold')
        
        # Ruins
        ruins = patches.Rectangle((8, 35), 4, 6, color='#696969', alpha=0.8)
        self.ax.add_patch(ruins)
        self.ax.text(10, 38, 'ANCIENT\nRUINS', ha='center', va='center', 
                    fontsize=8, color='white', weight='bold')
        
        self.ax.set_title('⚔️ WARHAMMER: THE OLD WORLD ⚔️', 
                         fontsize=20, fontweight='bold', color='gold', pad=20)
    
    def add_unit(self, unit: OldWorldUnit):
        self.units.append(unit)
    
    def distance(self, unit1: OldWorldUnit, unit2: OldWorldUnit) -> float:
        return math.sqrt((unit1.x - unit2.x)**2 + (unit1.y - unit2.y)**2)
    
    def draw_units(self):
        """Draw units with formations"""
        for unit in self.units:
            if unit.is_alive():
                corners = unit.get_formation_points()
                if not corners:
                    continue
                
                # Formation rectangle
                formation = patches.Polygon(corners, closed=True, 
                                          facecolor=unit.color, alpha=0.6,
                                          edgecolor='white', linewidth=2)
                self.ax.add_patch(formation)
                
                # Individual models
                models_drawn = 0
                for rank in range(unit.depth):
                    for file in range(unit.width):
                        if models_drawn >= unit.models:
                            break
                        
                        local_x = (file - unit.width/2 + 0.5) * 1.0
                        local_y = (rank - unit.depth/2 + 0.5) * 1.0
                        
                        angle_rad = math.radians(unit.facing)
                        model_x = local_x * math.cos(angle_rad) - local_y * math.sin(angle_rad) + unit.x
                        model_y = local_x * math.sin(angle_rad) + local_y * math.cos(angle_rad) + unit.y
                        
                        model_size = 25 if unit.unit_type == UnitType.INFANTRY else 40
                        marker = 'o' if unit.unit_type == UnitType.INFANTRY else 's'
                        
                        self.ax.scatter(model_x, model_y, s=model_size, c=unit.color,
                                      marker=marker, edgecolors='white', linewidth=1, alpha=0.9)
                        
                        models_drawn += 1
                    if models_drawn >= unit.models:
                        break
                
                # Unit label
                banner_x = unit.x
                banner_y = unit.y + unit.depth/2 + 2
                
                formation_symbol = {"Deep Formation": "█", "Wide Formation": "▬", 
                                  "Skirmish": "▪"}[unit.formation.value]
                
                unit_info = f"{unit.name}\n{unit.models}/{unit.max_models}\n{formation_symbol}"
                
                bbox_color = 'lightblue' if unit.player == 1 else 'lightcoral'
                self.ax.text(banner_x, banner_y, unit_info, ha='center', va='bottom',
                           fontsize=8, fontweight='bold', color='black',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.9))
                
                # Range circle
                if unit.weapon_range > 0:
                    range_circle = patches.Circle((unit.x, unit.y), unit.weapon_range,
                                                fill=False, color=unit.color, 
                                                alpha=0.3, linestyle=':', linewidth=2)
                    self.ax.add_patch(range_circle)
                
                # Facing arrow
                arrow_length = max(unit.width, unit.depth) / 2 + 1
                arrow_x = unit.x + arrow_length * math.cos(math.radians(unit.facing))
                arrow_y = unit.y + arrow_length * math.sin(math.radians(unit.facing))
                
                self.ax.annotate('', xy=(arrow_x, arrow_y), xytext=(unit.x, unit.y),
                               arrowprops=dict(arrowstyle='->', color='yellow', 
                                             linewidth=2, alpha=0.8))
    
    def update_display(self):
        """Update display"""
        self.setup_battlefield()
        self.draw_units()
        
        # Status
        p1_units = [u for u in self.units if u.player == 1 and u.is_alive()]
        p2_units = [u for u in self.units if u.player == 2 and u.is_alive()]
        
        p1_models = sum(u.models for u in p1_units)
        p2_models = sum(u.models for u in p2_units)
        
        status = f"""TURN {self.turn} - {self.phase}
Player {self.current_player} Active

🔵 EMPIRE: {len(p1_units)} units, {p1_models} models
🔴 ORCS: {len(p2_units)} units, {p2_models} models"""
        
        self.ax.text(0.02, 0.98, status, transform=self.ax.transAxes,
                    fontsize=11, va='top', fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='navy', alpha=0.9))
        
        # Battle log
        if self.battle_events:
            log_text = "BATTLE LOG:\n" + "\n".join(self.battle_events[-4:])
            self.ax.text(0.98, 0.02, log_text, transform=self.ax.transAxes,
                        fontsize=9, va='bottom', ha='right', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='darkgreen', alpha=0.9))
        
        plt.draw()
        plt.pause(0.05)
    
    def create_armies(self):
        """Create Old World armies with standards and musicians"""
        armies = []
        
        # EMPIRE (Player 1)
        halberdiers = OldWorldUnit(
            name="Empire Halberdiers", x=15, y=35, facing=90,
            models=20, max_models=20, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=5, depth=4,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=5, player=1, color='blue'
        )
        halberdiers.has_standard = True  # Add standard bearer
        halberdiers.has_musician = True  # Add musician
        
        handgunners = OldWorldUnit(
            name="Handgunners", x=10, y=25, facing=90,
            models=15, max_models=15, unit_type=UnitType.INFANTRY,
            formation=FormationType.WIDE, width=10, depth=2,
            movement=4, weapon_skill=3, ballistic_skill=4, strength=4,  # S4 handguns
            toughness=3, wounds=1, attacks=1, leadership=7,
            armor_save=5, player=1, color='lightblue', weapon_range=24
        )
        handgunners.armor_piercing = True  # Handguns ignore armor
        
        cannon = OldWorldUnit(
            name="Great Cannon", x=5, y=20, facing=90,
            models=3, max_models=3, unit_type=UnitType.ARTILLERY,
            formation=FormationType.SKIRMISH, width=3, depth=1,
            movement=0, weapon_skill=0, ballistic_skill=4, strength=10,
            toughness=7, wounds=3, attacks=0, leadership=7,
            armor_save=6, player=1, color='navy', weapon_range=48
        )
        cannon.armor_piercing = True  # Cannons ignore armor
        
        knights = OldWorldUnit(
            name="Empire Knights", x=20, y=12, facing=90,
            models=6, max_models=6, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=3, depth=2,
            movement=8, weapon_skill=4, ballistic_skill=3, strength=4,  # S4 on charge
            toughness=3, wounds=1, attacks=1, leadership=8,
            armor_save=3, player=1, color='purple'
        )
        knights.has_standard = True
        knights.has_musician = True
        knights.lance_formation = True  # Special cavalry rule
        
        armies.extend([halberdiers, handgunners, cannon, knights])
        
        # ORCS (Player 2)
        orc_boyz = OldWorldUnit(
            name="Orc Boyz", x=55, y=35, facing=270,
            models=25, max_models=25, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=5, depth=5,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, attacks=1, leadership=7,
            armor_save=6, player=2, color='red'
        )
        orc_boyz.has_standard = True
        orc_boyz.has_musician = True
        orc_boyz.frenzy = True  # Orcs get Frenzy
        
        orc_archers = OldWorldUnit(
            name="Orc Archers", x=60, y=25, facing=270,
            models=15, max_models=15, unit_type=UnitType.INFANTRY,
            formation=FormationType.WIDE, width=8, depth=2,
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=4, wounds=1, attacks=1, leadership=6,
            armor_save=7, player=2, color='darkred', weapon_range=18
        )
        
        wolf_riders = OldWorldUnit(
            name="Wolf Riders", x=65, y=15, facing=270,
            models=6, max_models=6, unit_type=UnitType.CAVALRY,
            formation=FormationType.WIDE, width=3, depth=2,
            movement=9, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, attacks=1, leadership=6,
            armor_save=6, player=2, color='orange'
        )
        wolf_riders.fast_cavalry = True  # Special rule
        
        # Add some terrifying creatures to the Orc army
        black_orcs = OldWorldUnit(
            name="Black Orcs", x=50, y=30, facing=270,
            models=12, max_models=12, unit_type=UnitType.INFANTRY,
            formation=FormationType.DEEP, width=4, depth=3,
            movement=4, weapon_skill=4, ballistic_skill=3, strength=4,
            toughness=4, wounds=1, attacks=1, leadership=8,
            armor_save=4, player=2, color='darkred'
        )
        black_orcs.has_standard = True
        black_orcs.fear = True  # Black Orcs cause Fear
        black_orcs.immune_to_fear = True  # They're immune to psychology
        black_orcs.stubborn = True  # Never use negative modifiers for break tests
        
        troll = OldWorldUnit(
            name="River Troll", x=45, y=20, facing=270,
            models=1, max_models=1, unit_type=UnitType.MONSTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=6, weapon_skill=3, ballistic_skill=1, strength=5,
            toughness=4, wounds=3, attacks=3, leadership=4,
            armor_save=5, player=2, color='green'
        )
        troll.fear = True  # Trolls cause Fear
        troll.terror = True  # And Terror to smaller creatures
        troll.regeneration = True  # Trolls regenerate wounds
        troll.stupidity = True  # But they're stupid
        
        armies.extend([orc_boyz, orc_archers, wolf_riders, black_orcs, troll])
        
        # Add some Empire heroes
        empire_captain = OldWorldUnit(
            name="Empire Captain", x=18, y=30, facing=90,
            models=1, max_models=1, unit_type=UnitType.CHARACTER,
            formation=FormationType.SKIRMISH, width=1, depth=1,
            movement=4, weapon_skill=5, ballistic_skill=4, strength=4,
            toughness=4, wounds=2, attacks=3, leadership=9,
            armor_save=3, player=1, color='gold'
        )
        empire_captain.immune_to_fear = True  # Heroes are fearless
        empire_captain.inspiring_presence = True  # Can rally nearby units
        
        armies.append(empire_captain)
        
        return armies
    
    def run_turn(self):
        """Run one battle turn"""
        print(f"\nTURN {self.turn} - PLAYER {self.current_player}")
        
        # Movement
        self.phase = "Movement"
        print("Movement Phase")
        self.execute_movement()
        self.update_display()
        time.sleep(1)
        
        # Shooting
        self.phase = "Shooting"
        print("Shooting Phase")
        self.execute_shooting()
        self.update_display()
        time.sleep(1)
        
        # Charges
        self.phase = "Charges"
        print("Charge Phase")
        self.execute_charges()
        self.update_display()
        time.sleep(1)
        
        # Combat
        self.phase = "Combat"
        print("Combat Phase")
        self.execute_combat()
        self.update_display()
        
        # Next player
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
            self.turn += 1
        
        return self.check_victory()
    
    def execute_movement(self):
        """Handle movement phase"""
        active_units = [u for u in self.units if u.player == self.current_player and u.is_alive()]
        
        for unit in active_units:
            if unit.unit_type == UnitType.CAVALRY:
                # Cavalry seeks flanks
                enemies = [u for u in self.units if u.player != self.current_player and u.is_alive()]
                if enemies:
                    target = min(enemies, key=lambda e: self.distance(unit, e))
                    if self.current_player == 1:
                        unit.x = min(unit.x + unit.movement, target.x - 6)
                    else:
                        unit.x = max(unit.x - unit.movement, target.x + 6)
                        
            elif unit.weapon_range > 0:
                # Ranged units seek position
                if self.current_player == 1:
                    unit.x = max(5, unit.x - 2)
                else:
                    unit.x = min(self.width - 5, unit.x + 2)
                    
            else:
                # Infantry advances
                if self.current_player == 1:
                    unit.x = unit.x + 3
                else:
                    unit.x = unit.x - 3
            
            unit.update_formation()
            print(f"  {unit.name} moves")
    
    def execute_shooting(self):
        """Handle shooting phase"""
        shooters = [u for u in self.units if u.player == self.current_player 
                   and u.is_alive() and u.weapon_range > 0]
        
        print(f"  Found {len(shooters)} shooting units for Player {self.current_player}")
        
        for shooter in shooters:
            enemies = [u for u in self.units if u.player != self.current_player and u.is_alive()]
            if not enemies:
                continue
                
            in_range = [e for e in enemies if self.distance(shooter, e) <= shooter.weapon_range]
            print(f"  {shooter.name} (range {shooter.weapon_range}): {len(in_range)} targets in range")
            
            if not in_range:
                # Print distances for debugging
                closest_enemy = min(enemies, key=lambda e: self.distance(shooter, e))
                dist = self.distance(shooter, closest_enemy)
                print(f"    Closest target {closest_enemy.name} at {dist:.1f}\" (out of range)")
                continue
                
            target = min(in_range, key=lambda e: self.distance(shooter, e))
            
            # Shooting mechanics
            shots = shooter.models if shooter.unit_type != UnitType.ARTILLERY else 1
            wounds = 0
            
            for _ in range(shots):
                if random.randint(1, 6) >= 4:  # Hit
                    if random.randint(1, 6) >= 4:  # Wound
                        if random.randint(1, 6) < target.armor_save:  # Failed save
                            wounds += 1
            
            if wounds > 0:
                target.models = max(0, target.models - wounds)
                target.update_formation()
                
                # Visual effect
                self.ax.plot([shooter.x, target.x], [shooter.y, target.y], 
                           'orange', linewidth=3, alpha=0.8)
                plt.pause(0.2)
                
                event = f"{shooter.name} → {target.name}: {wounds} casualties"
                self.battle_events.append(event)
                print(f"  {event}")
            else:
                print(f"  {shooter.name} shoots at {target.name} but misses!")
    
    def execute_charges(self):
        """Handle charge phase"""
        chargers = [u for u in self.units if u.player == self.current_player 
                   and u.is_alive() and u.unit_type == UnitType.CAVALRY]
        
        for charger in chargers:
            enemies = [u for u in self.units if u.player != self.current_player and u.is_alive()]
            if not enemies:
                continue
                
            charge_range = charger.movement * 2
            in_range = [e for e in enemies if self.distance(charger, e) <= charge_range]
            
            if in_range:
                target = min(in_range, key=lambda e: self.distance(charger, e))
                print(f"  {charger.name} charges {target.name}!")
                
                # Move into contact
                charger.x = target.x
                charger.y = target.y - 3
                charger.has_charged = True
                
                # Visual effect
                self.ax.plot([charger.x, target.x], [charger.y, target.y], 
                           'red', linewidth=5, alpha=0.8)
                plt.pause(0.2)
    
    def execute_combat(self):
        """Handle combat phase"""
        combat_pairs = []
        processed_ids = set()
        
        try:
            for unit in self.units:
                if not unit.is_alive() or unit._id in processed_ids:
                    continue
                    
                enemies = [u for u in self.units if u.player != unit.player 
                          and u.is_alive() and self.distance(unit, u) <= 3]
                
                for enemy in enemies:
                    if enemy._id not in processed_ids:
                        combat_pairs.append((unit, enemy))
                        processed_ids.add(unit._id)
                        processed_ids.add(enemy._id)
                        break
        except Exception as e:
            print(f"Combat error: {e}")
            print(f"Unit types: {[type(u) for u in self.units[:3]]}")
            return
        
        for unit1, unit2 in combat_pairs:
            print(f"  {unit1.name} vs {unit2.name}")
            
            # Simple combat resolution
            attacks1 = unit1.attacks * min(unit1.models, unit1.width)
            attacks2 = unit2.attacks * min(unit2.models, unit2.width)
            
            casualties1 = 0
            casualties2 = 0
            
            for _ in range(attacks2):
                if random.randint(1, 6) >= 4:  # Hit
                    if random.randint(1, 6) >= 4:  # Wound
                        casualties1 += 1
                        
            for _ in range(attacks1):
                if random.randint(1, 6) >= 4:  # Hit
                    if random.randint(1, 6) >= 4:  # Wound
                        casualties2 += 1
            
            unit1.models = max(0, unit1.models - casualties1)
            unit2.models = max(0, unit2.models - casualties2)
            
            unit1.update_formation()
            unit2.update_formation()
            
            if casualties1 > 0 or casualties2 > 0:
                event = f"Combat: {unit1.name} -{casualties1}, {unit2.name} -{casualties2}"
                self.battle_events.append(event)
                print(f"    {event}")
    
    def check_victory(self) -> bool:
        """Check victory conditions using authentic Old World rules"""
        p1_alive = any(u.player == 1 and u.is_alive() for u in self.units)
        p2_alive = any(u.player == 2 and u.is_alive() for u in self.units)
        
        # Immediate victory - army eliminated
        if not p1_alive:
            print("\n🏆 TROLL HORDE VICTORY! (Army Eliminated)")
            return False
        elif not p2_alive:
            print("\n🏆 ARMY OF NULN VICTORY! (Army Eliminated)")
            return False
        
        # Game ends after 6 turns - count victory points
        if self.turn > 6:
            vp_p1, vp_p2 = self.calculate_victory_points()
            print(f"\n⏰ BATTLE ENDS AFTER 6 TURNS!")
            print(f"📊 VICTORY POINTS:")
            print(f"   Army of Nuln: {vp_p1} VP")
            print(f"   Troll Horde: {vp_p2} VP")
            
            if vp_p1 > vp_p2:
                print(f"\n🏆 ARMY OF NULN VICTORY! (+{vp_p1 - vp_p2} VP)")
            elif vp_p2 > vp_p1:
                print(f"\n🏆 TROLL HORDE VICTORY! (+{vp_p2 - vp_p1} VP)")
            else:
                print(f"\n🤝 HONORABLE DRAW! (Tied at {vp_p1} VP each)")
            return False
        
        return True
    
    def calculate_victory_points(self) -> tuple[int, int]:
        """Calculate victory points earned by each player"""
        vp_p1 = 0  # Points earned by Player 1 (Nuln)
        vp_p2 = 0  # Points earned by Player 2 (Trolls)
        
        for unit in self.units:
            if unit.points_cost == 0:
                continue  # Skip units without point values
                
            if unit.player == 1:  # Nuln unit
                # Player 2 (Trolls) gets VP for casualties inflicted
                casualties = unit.starting_models - unit.models
                if casualties > 0:
                    # Proportional points for partial destruction
                    casualty_percentage = casualties / unit.starting_models
                    if casualty_percentage >= 1.0:  # Completely destroyed
                        vp_p2 += unit.points_cost
                    elif casualty_percentage >= 0.5:  # Half or more destroyed
                        vp_p2 += unit.points_cost // 2
                    # Less than half = no VP (Old World rule)
                        
            elif unit.player == 2:  # Troll unit
                # Player 1 (Nuln) gets VP for casualties inflicted
                casualties = unit.starting_models - unit.models
                if casualties > 0:
                    casualty_percentage = casualties / unit.starting_models
                    if casualty_percentage >= 1.0:  # Completely destroyed
                        vp_p1 += unit.points_cost
                    elif casualty_percentage >= 0.5:  # Half or more destroyed
                        vp_p1 += unit.points_cost // 2
        
        return vp_p1, vp_p2
    
    def run_battle(self):
        """Run complete battle"""
        print("Setting up Old World battle...")
        
        armies = self.create_armies()
        for unit in armies:
            self.add_unit(unit)
        
        print("BATTLE BEGINS!")
        self.update_display()
        
        battle_continues = True
        while battle_continues:
            try:
                battle_continues = self.run_turn()
                if battle_continues:
                    import time
                    time.sleep(1.5)  # Auto-advance after 1.5 seconds
                    print("⚡ ADVANCING TO NEXT TURN...")
                    print("="*50)
            except KeyboardInterrupt:
                print("\nBattle ended!")
                break
            except Exception as e:
                print(f"Battle error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        plt.show()

def main():
    """Launch Old World battle"""
    print("Launching Warhammer: The Old World battle...")
    print("Features:")
    print("• Ranked infantry formations")
    print("• Cavalry charges and flanking")
    print("• Artillery bombardments")
    print("• Proper game phases")
    print("• Authentic Old World scale")
    print()
    
    try:
        battle = OldWorldBattle()
        battle.run_battle()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nOld World battle complete!")

if __name__ == "__main__":
    main() 