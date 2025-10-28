#!/usr/bin/env python3
"""
🏛️ WARHAMMER BATTLE CORE ENGINE
===============================

Complete battle engine for AI-commanded Warhammer visualization.
"""

import numpy as np
import random
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import uuid
from datetime import datetime

# =============================================================================
# WARHAMMER UNIT SYSTEM
# =============================================================================

class UnitType(Enum):
    INFANTRY = "Infantry"
    CAVALRY = "Cavalry" 
    ARTILLERY = "Artillery"
    MONSTER = "Monster"
    HERO = "Hero"

class Faction(Enum):
    EMPIRE = "Empire"
    ORCS = "Orcs & Goblins"

@dataclass
class UnitProfile:
    name: str
    unit_type: UnitType
    faction: Faction
    movement: int
    weapon_skill: int  
    ballistic_skill: int
    strength: int
    toughness: int
    wounds: int
    initiative: int
    attacks: int
    leadership: int
    armor_save: int
    weapons: List[str]
    armor: List[str]
    special_rules: List[str]
    base_size: int
    max_unit_size: int
    points_per_model: int

# Unit profiles
UNIT_PROFILES = {
    "Empire Handgunners": UnitProfile(
        name="Empire Handgunners", unit_type=UnitType.INFANTRY, faction=Faction.EMPIRE,
        movement=4, weapon_skill=3, ballistic_skill=4, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=7, armor_save=6,
        weapons=["Handgun"], armor=["Light Armor"], special_rules=["Stand and Shoot"],
        base_size=10, max_unit_size=30, points_per_model=12
    ),
    "Empire Knights": UnitProfile(
        name="Empire Knights", unit_type=UnitType.CAVALRY, faction=Faction.EMPIRE,
        movement=8, weapon_skill=4, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=8, armor_save=3,
        weapons=["Lance", "Hand Weapon"], armor=["Heavy Armor", "Shield"],
        special_rules=["Cavalry", "Charge Bonus"], base_size=5, max_unit_size=15,
        points_per_model=26
    ),
    "Empire Cannon": UnitProfile(
        name="Empire Cannon", unit_type=UnitType.ARTILLERY, faction=Faction.EMPIRE,
        movement=0, weapon_skill=3, ballistic_skill=4, strength=7, toughness=4,
        wounds=2, initiative=3, attacks=1, leadership=7, armor_save=7,
        weapons=["Cannon"], armor=[], special_rules=["Artillery", "Crew"],
        base_size=1, max_unit_size=1, points_per_model=120
    ),
    "Orc Boyz": UnitProfile(
        name="Orc Boyz", unit_type=UnitType.INFANTRY, faction=Faction.ORCS,
        movement=4, weapon_skill=3, ballistic_skill=3, strength=3, toughness=4,
        wounds=1, initiative=2, attacks=1, leadership=7, armor_save=6,
        weapons=["Hand Weapon"], armor=["Light Armor"], special_rules=["Animosity"],
        base_size=10, max_unit_size=30, points_per_model=6
    ),
    "Orc Arrer Boyz": UnitProfile(
        name="Orc Arrer Boyz", unit_type=UnitType.INFANTRY, faction=Faction.ORCS,
        movement=4, weapon_skill=3, ballistic_skill=3, strength=3, toughness=4,
        wounds=1, initiative=2, attacks=1, leadership=7, armor_save=7,
        weapons=["Bow", "Hand Weapon"], armor=[], special_rules=["Animosity"],
        base_size=10, max_unit_size=30, points_per_model=6
    ),
    "Orc Wolf Riders": UnitProfile(
        name="Orc Wolf Riders", unit_type=UnitType.CAVALRY, faction=Faction.ORCS,
        movement=9, weapon_skill=3, ballistic_skill=3, strength=3, toughness=4,
        wounds=1, initiative=2, attacks=1, leadership=7, armor_save=6,
        weapons=["Spear", "Bow"], armor=["Light Armor"],
        special_rules=["Fast Cavalry", "Animosity"], base_size=5, max_unit_size=15,
        points_per_model=14
    )
}

@dataclass 
class Position:
    x: int
    y: int
    facing: int = 0
    
    def distance_to(self, other: 'Position') -> int:
        return (abs(self.x - other.x) + abs(self.x + self.y - other.x - other.y) + abs(self.y - other.y)) // 2

@dataclass
class WarhammerUnit:
    id: str
    profile: UnitProfile
    position: Position
    current_models: int
    wounds_taken: List[int] = None
    status_effects: List[str] = None
    has_moved: bool = False
    has_shot: bool = False
    has_charged: bool = False
    is_engaged: bool = False
    
    def __post_init__(self):
        if self.wounds_taken is None:
            self.wounds_taken = [0] * self.current_models
        if self.status_effects is None:
            self.status_effects = []
    
    @property
    def is_alive(self) -> bool:
        return self.current_models > 0
    
    @property
    def total_wounds_remaining(self) -> int:
        return sum(max(0, self.profile.wounds - wounds) for wounds in self.wounds_taken[:self.current_models])
    
    def take_wounds(self, wounds: int) -> int:
        models_killed = 0
        wounds_left = wounds
        
        for i in range(self.current_models):
            if wounds_left <= 0:
                break
            wounds_this_model = min(wounds_left, self.profile.wounds - self.wounds_taken[i])
            self.wounds_taken[i] += wounds_this_model
            wounds_left -= wounds_this_model
            
            if self.wounds_taken[i] >= self.profile.wounds:
                models_killed += 1
        
        self.current_models -= models_killed
        return models_killed
    
    def can_move(self) -> bool:
        return not self.has_moved and not self.is_engaged and self.is_alive
    
    def can_shoot(self) -> bool:
        return not self.has_shot and not self.is_engaged and self.is_alive and self.profile.ballistic_skill > 0

# =============================================================================
# BATTLEFIELD SYSTEM
# =============================================================================

class BattleField:
    def __init__(self, width: int = 24, height: int = 16):
        self.width = width
        self.height = height
        self.empire_units: List[WarhammerUnit] = []
        self.orc_units: List[WarhammerUnit] = []
        
    def add_unit(self, unit: WarhammerUnit):
        if unit.profile.faction == Faction.EMPIRE:
            self.empire_units.append(unit)
        else:
            self.orc_units.append(unit)
    
    def get_unit_at(self, position: Position) -> Optional[WarhammerUnit]:
        for unit in self.empire_units + self.orc_units:
            if unit.position.x == position.x and unit.position.y == position.y and unit.is_alive:
                return unit
        return None
    
    def is_valid_position(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height
    
    def get_battle_state_vector(self) -> np.ndarray:
        state = np.zeros(50)
        
        # Empire units (first 25 elements)
        for i, unit in enumerate(self.empire_units[:5]):
            if unit.is_alive:
                base_idx = i * 5
                state[base_idx] = unit.position.x / self.width
                state[base_idx + 1] = unit.position.y / self.height
                state[base_idx + 2] = unit.current_models / unit.profile.max_unit_size
                state[base_idx + 3] = unit.total_wounds_remaining / (unit.current_models * unit.profile.wounds) if unit.current_models > 0 else 0
                state[base_idx + 4] = 1.0 if unit.can_move() else 0.0
        
        # Orc units (next 25 elements)
        for i, unit in enumerate(self.orc_units[:5]):
            if unit.is_alive:
                base_idx = 25 + i * 5  
                state[base_idx] = unit.position.x / self.width
                state[base_idx + 1] = unit.position.y / self.height
                state[base_idx + 2] = unit.current_models / unit.profile.max_unit_size
                state[base_idx + 3] = unit.total_wounds_remaining / (unit.current_models * unit.profile.wounds) if unit.current_models > 0 else 0
                state[base_idx + 4] = 1.0 if unit.can_move() else 0.0
        
        return state

# =============================================================================
# AI DECISION TRANSLATOR
# =============================================================================

class WarhammerCommand:
    def __init__(self, unit_id: str):
        self.unit_id = unit_id
        self.timestamp = datetime.now()

class MoveCommand(WarhammerCommand):
    def __init__(self, unit_id: str, target_position: Position):
        super().__init__(unit_id)
        self.target_position = target_position
        self.command_type = "MOVE"

class ShootCommand(WarhammerCommand):
    def __init__(self, unit_id: str, target_unit_id: str):
        super().__init__(unit_id)
        self.target_unit_id = target_unit_id
        self.command_type = "SHOOT"

class ChargeCommand(WarhammerCommand):
    def __init__(self, unit_id: str, target_unit_id: str):
        super().__init__(unit_id)
        self.target_unit_id = target_unit_id
        self.command_type = "CHARGE"

class AIDecisionTranslator:
    def __init__(self, battlefield: BattleField):
        self.battlefield = battlefield
        self.action_names = [
            "Move North", "Move South", "Move East", "Move West",
            "Move NE", "Move NW", "Move SE", "Move SW", 
            "Cavalry Charge", "Artillery Strike", "Defensive Formation",
            "Magic Attack", "Mass Shooting", "Special Tactic A", "Special Tactic B"
        ]
        
        self.hex_directions = {
            0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0),
            4: (1, -1), 5: (-1, -1), 6: (1, 1), 7: (-1, 1)
        }
    
    def translate_ai_decision(self, action: int, q_value: float, faction: Faction) -> List[WarhammerCommand]:
        commands = []
        units = self.battlefield.empire_units if faction == Faction.EMPIRE else self.battlefield.orc_units
        enemy_units = self.battlefield.orc_units if faction == Faction.EMPIRE else self.battlefield.empire_units
        
        if not units:
            return commands
        
        if action <= 7:  # Movement actions
            commands.extend(self._translate_movement(action, units, q_value))
        elif action == 8:  # Cavalry Charge
            commands.extend(self._translate_cavalry_charge(units, enemy_units, q_value))
        elif action == 9:  # Artillery Strike
            commands.extend(self._translate_artillery_strike(units, enemy_units, q_value))
        elif action == 12:  # Mass Shooting
            commands.extend(self._translate_mass_shooting(units, enemy_units, q_value))
        elif action >= 13:  # Special Tactics
            commands.extend(self._translate_special_tactics(units, enemy_units, q_value))
        
        return commands
    
    def _translate_movement(self, direction: int, units: List[WarhammerUnit], q_value: float) -> List[WarhammerCommand]:
        commands = []
        dx, dy = self.hex_directions[direction]
        moveable_units = [u for u in units if u.can_move()]
        num_units_to_move = min(len(moveable_units), max(1, int(abs(q_value) / 5)))
        
        for unit in moveable_units[:num_units_to_move]:
            new_x = unit.position.x + dx
            new_y = unit.position.y + dy
            new_pos = Position(new_x, new_y, unit.position.facing)
            
            if self.battlefield.is_valid_position(new_pos) and not self.battlefield.get_unit_at(new_pos):
                commands.append(MoveCommand(unit.id, new_pos))
        
        return commands
    
    def _translate_cavalry_charge(self, units: List[WarhammerUnit], enemies: List[WarhammerUnit], q_value: float) -> List[WarhammerCommand]:
        commands = []
        cavalry_units = [u for u in units if u.profile.unit_type == UnitType.CAVALRY and u.can_move()]
        
        if not cavalry_units or not enemies:
            return commands
        
        for cavalry in cavalry_units:
            alive_enemies = [e for e in enemies if e.is_alive]
            if alive_enemies:
                closest_enemy = min(alive_enemies, key=lambda e: cavalry.position.distance_to(e.position))
                if cavalry.position.distance_to(closest_enemy.position) <= cavalry.profile.movement + 6:
                    commands.append(ChargeCommand(cavalry.id, closest_enemy.id))
        
        return commands
    
    def _translate_artillery_strike(self, units: List[WarhammerUnit], enemies: List[WarhammerUnit], q_value: float) -> List[WarhammerCommand]:
        commands = []
        artillery_units = [u for u in units if u.profile.unit_type == UnitType.ARTILLERY and u.can_shoot()]
        
        if not artillery_units or not enemies:
            return commands
        
        for artillery in artillery_units:
            alive_enemies = [e for e in enemies if e.is_alive]
            if alive_enemies:
                target = max(alive_enemies, key=lambda e: e.current_models)
                commands.append(ShootCommand(artillery.id, target.id))
        
        return commands
    
    def _translate_mass_shooting(self, units: List[WarhammerUnit], enemies: List[WarhammerUnit], q_value: float) -> List[WarhammerCommand]:
        commands = []
        shooting_units = [u for u in units if u.can_shoot() and u.profile.ballistic_skill > 0]
        
        if not shooting_units or not enemies:
            return commands
        
        for shooter in shooting_units:
            alive_enemies = [e for e in enemies if e.is_alive]
            if alive_enemies:
                target = min(alive_enemies, key=lambda e: shooter.position.distance_to(e.position))
                commands.append(ShootCommand(shooter.id, target.id))
        
        return commands
    
    def _translate_special_tactics(self, units: List[WarhammerUnit], enemies: List[WarhammerUnit], q_value: float) -> List[WarhammerCommand]:
        commands = []
        
        if enemies and units:
            for unit in units[:2]:
                if unit.can_move():
                    alive_enemies = [e for e in enemies if e.is_alive]
                    if alive_enemies:
                        closest_enemy = min(alive_enemies, key=lambda e: unit.position.distance_to(e.position))
                        dx = 1 if closest_enemy.position.x > unit.position.x else -1 if closest_enemy.position.x < unit.position.x else 0
                        dy = 1 if closest_enemy.position.y > unit.position.y else -1 if closest_enemy.position.y < unit.position.y else 0
                        
                        new_pos = Position(unit.position.x + dx, unit.position.y + dy, unit.position.facing)
                        if self.battlefield.is_valid_position(new_pos) and not self.battlefield.get_unit_at(new_pos):
                            commands.append(MoveCommand(unit.id, new_pos))
        
        return commands

if __name__ == "__main__":
    print("🏛️ WARHAMMER BATTLE CORE ENGINE")
    print("=" * 40)
    print("✅ Unit system loaded!")
    print("✅ Battlefield mechanics ready!")
    print("✅ AI decision translator initialized!")
    print("\n🎯 Ready for 300k-trained AI battles!") 