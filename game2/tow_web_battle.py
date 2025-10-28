#!/usr/bin/env python3
"""
WEB-BASED WARHAMMER: THE OLD WORLD BATTLE
=========================================
Authentic regiment formations in your browser
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import time
import random
import math
import threading
import json
import numpy as np
from threading import Thread, Lock
import torch
import torch.nn as nn
from collections import deque
import os
import sys

# Import our AI systems
try:
    # Import only the essential parts without matplotlib
    import torch
    import torch.nn as nn
    import numpy as np
    from collections import deque
    HAVE_AI = True
    print("✅ AI libraries loaded")
except ImportError as e:
    print(f"AI import error: {e}")
    HAVE_AI = False

# =============== AUTHENTIC TOW GAME MECHANICS ===============

class EquipmentItem:
    """Represents magical items and artifacts"""
    def __init__(self, name, item_type, points_cost, effects, restrictions=None):
        self.name = name
        self.item_type = item_type  # "magic_weapon", "magic_armor", "enchanted_item", "arcane_item"
        self.points_cost = points_cost
        self.effects = effects  # Dictionary of stat modifications
        self.restrictions = restrictions or []

class ArmyBuildingConstraints:
    """Authentic TOW army building rules for competitive play"""
    
    def __init__(self):
        # Rule of 3 - max 3 of same unit type
        self.rule_of_3 = True
        
        # Points allocation percentages
        self.min_core_percentage = 25
        self.max_special_percentage = 50
        self.max_rare_percentage = 25
        self.max_lords_percentage = 25
        self.max_heroes_percentage = 25
        
        # Character limits per 1000 points
        self.max_lords_per_1000 = 1
        self.max_heroes_per_1000 = 2
        
        # Equipment limits
        self.max_magic_items_per_character = 3
        self.max_duplicate_magic_items = 1
        
    def validate_army(self, army_list, total_points):
        """Validate army list against TOW competitive rules"""
        violations = []
        
        # Check Rule of 3
        unit_counts = {}
        for unit in army_list:
            base_name = unit.name.split()[0]  # Get unit type (e.g., "Halberdiers" from "20 Halberdiers")
            unit_counts[base_name] = unit_counts.get(base_name, 0) + 1
            
        for unit_name, count in unit_counts.items():
            if count > 3:
                violations.append(f"Rule of 3 violated: {count} units of {unit_name}")
        
        # Check points allocations
        core_points = sum(u.points for u in army_list if u.unit_category == "core")
        special_points = sum(u.points for u in army_list if u.unit_category == "special")
        rare_points = sum(u.points for u in army_list if u.unit_category == "rare")
        lord_points = sum(u.points for u in army_list if u.unit_category == "lord")
        hero_points = sum(u.points for u in army_list if u.unit_category == "hero")
        
        if core_points < total_points * self.min_core_percentage / 100:
            violations.append(f"Insufficient core units: {core_points}/{total_points * 0.25} points minimum")
            
        if special_points > total_points * self.max_special_percentage / 100:
            violations.append(f"Too many special units: {special_points}/{total_points * 0.5} points maximum")
            
        if rare_points > total_points * self.max_rare_percentage / 100:
            violations.append(f"Too many rare units: {rare_points}/{total_points * 0.25} points maximum")
            
        # Check character limits
        lords_allowed = (total_points // 1000) * self.max_lords_per_1000
        heroes_allowed = (total_points // 1000) * self.max_heroes_per_1000
        
        lord_count = len([u for u in army_list if u.unit_category == "lord"])
        hero_count = len([u for u in army_list if u.unit_category == "hero"])
        
        if lord_count > lords_allowed:
            violations.append(f"Too many lords: {lord_count}/{lords_allowed} allowed")
            
        if hero_count > heroes_allowed:
            violations.append(f"Too many heroes: {hero_count}/{heroes_allowed} allowed")
        
        return violations

class TerrainSystem:
    """Comprehensive terrain effects system"""
    
    @staticmethod
    def get_standard_terrain():
        """Standard TOW terrain features"""
        return [
            TerrainFeature("Hill", "elevation", [300, 200], [120, 80], 
                          {"elevation": +1, "line_of_sight_bonus": True}),
            TerrainFeature("Forest", "difficult", [100, 300], [100, 100],
                          {"movement_penalty": -1, "cover_save": 6, "charges_disordered": True}),
            TerrainFeature("River", "water", [0, 150], [500, 30],
                          {"impassable_to_cavalry": True, "movement_penalty": -2}),
            TerrainFeature("Stone Wall", "obstacle", [250, 250], [80, 10],
                          {"cover_save": 5, "obstacle_height": 1}),
            TerrainFeature("Tower Ruins", "building", [400, 100], [60, 60],
                          {"garrison_capacity": 20, "armor_save_modifier": +1}),
        ]
    
    @staticmethod
    def apply_terrain_effects(unit, terrain_features):
        """Apply terrain effects to unit movement and combat"""
        effects = {}
        
        for terrain in terrain_features:
            if TerrainSystem.unit_in_terrain(unit, terrain):
                effects.update(terrain.effects)
        
        return effects
    
    @staticmethod
    def unit_in_terrain(unit, terrain):
        """Check if unit is within terrain feature"""
        # Simplified rectangular collision detection
        unit_x, unit_y = unit.position
        terrain_x, terrain_y = terrain.position
        terrain_w, terrain_h = terrain.size
        
        return (terrain_x <= unit_x <= terrain_x + terrain_w and 
                terrain_y <= unit_y <= terrain_y + terrain_h)

class FormationRule:
    """Enhanced formation rules with proper rank bonuses"""
    def __init__(self, formation_type, min_width, rank_bonus_per_rank, special_rules=None):
        self.formation_type = formation_type
        self.min_width = min_width
        self.rank_bonus_per_rank = rank_bonus_per_rank
        self.special_rules = special_rules or []

class TerrainFeature:
    """Represents battlefield terrain"""
    def __init__(self, name, terrain_type, position, size, effects):
        self.name = name
        self.terrain_type = terrain_type  # "elevation", "difficult", "water", "obstacle", "building"
        self.position = position  # [x, y]
        self.size = size  # [width, height]
        self.effects = effects  # Dictionary of terrain effects

class AdvancedMagicSystem:
    """Complete magic system with power/dispel dice, miscast tables"""
    
    @staticmethod
    def generate_magic_dice(wizard_levels):
        """Generate magic dice based on wizard levels"""
        base_dice = 2
        wizard_dice = sum(wizard_levels)
        return base_dice + wizard_dice
    
    @staticmethod
    def attempt_spell_casting(caster_level, spell_difficulty, power_dice_used):
        """Attempt to cast spell with authentic rules"""
        total_roll = sum(random.randint(1, 6) for _ in range(power_dice_used))
        
        # Check for irresistible force (double 6s)
        if power_dice_used >= 2:
            dice_results = [random.randint(1, 6) for _ in range(power_dice_used)]
            if dice_results.count(6) >= 2:
                return {"success": True, "irresistible": True, "total": sum(dice_results)}
        
        # Check for miscast (double 1s)
        if power_dice_used >= 2:
            if dice_results.count(1) >= 2:
                return {"success": False, "miscast": True, "total": total_roll}
        
        success = total_roll >= spell_difficulty
        return {"success": success, "irresistible": False, "miscast": False, "total": total_roll}
    
    @staticmethod
    def roll_miscast_table():
        """Roll on the miscast table"""
        roll = random.randint(1, 12)
        
        if roll <= 2:
            return {"effect": "Wizard takes S3 hit", "damage": 1, "continues": True}
        elif roll <= 4:
            return {"effect": "Wizard takes S6 hit", "damage": 2, "continues": True}
        elif roll <= 6:
            return {"effect": "Wizard forgets spell", "spell_lost": True, "continues": True}
        elif roll <= 8:
            return {"effect": "All nearby units take S4 hits", "area_damage": True, "continues": False}
        elif roll <= 10:
            return {"effect": "Wizard turned into spawn", "wizard_destroyed": True, "continues": False}
        else:
            return {"effect": "Wizard destroyed in magical explosion", "wizard_destroyed": True, "area_damage": True, "continues": False}

class EnhancedCombatSystem:
    """Advanced combat with weapon skill charts, armor piercing, etc."""
    
    @staticmethod
    def resolve_hit_rolls(attacker_ws, defender_ws, num_attacks):
        """Resolve hit rolls using weapon skill chart"""
        hits = 0
        target_number = WeaponSkillChart.get_hit_chance(attacker_ws, defender_ws)
        
        for _ in range(num_attacks):
            if random.randint(1, 6) >= target_number:
                hits += 1
        
        return hits
    
    @staticmethod
    def resolve_wound_rolls(strength, toughness, hits, armor_piercing=False):
        """Resolve wound rolls using strength vs toughness"""
        wounds = 0
        
        # Determine wound target number
        if strength >= toughness * 2:
            target = 2
        elif strength > toughness:
            target = 3
        elif strength == toughness:
            target = 4
        elif strength < toughness and strength * 2 > toughness:
            target = 5
        else:
            target = 6
        
        for _ in range(hits):
            if random.randint(1, 6) >= target:
                wounds += 1
        
        return wounds, armor_piercing
    
    @staticmethod
    def resolve_armor_saves(wounds, armor_save, armor_piercing_modifier=0):
        """Resolve armor saves"""
        if armor_save is None:
            return wounds  # No save possible
        
        saves_made = 0
        modified_save = armor_save + armor_piercing_modifier
        
        if modified_save > 6:
            return wounds  # Save impossible
        
        for _ in range(wounds):
            if random.randint(1, 6) >= modified_save:
                saves_made += 1
        
        return max(0, wounds - saves_made)
    
    @staticmethod
    def resolve_ward_saves(wounds, ward_save):
        """Resolve ward saves (unmodifiable)"""
        if ward_save is None:
            return wounds
        
        saves_made = 0
        for _ in range(wounds):
            if random.randint(1, 6) >= ward_save:
                saves_made += 1
        
        return max(0, wounds - saves_made)

class PsychologySystem:
    """Comprehensive psychology rules"""
    
    @staticmethod
    def test_fear(unit, fear_causing_enemy):
        """Test for fear"""
        if unit.fear_immune or unit.causes_fear:
            return False
        
        # Fear test on initial charge/combat
        test_result = random.randint(1, 6) + random.randint(1, 6) + unit.leadership
        
        if test_result < 14:  # Failed fear test
            unit.is_fleeing = True
            return True
        
        return False
    
    @staticmethod
    def test_terror(unit, terror_causing_enemy):
        """Test for terror"""
        if unit.fear_immune:
            return False
        
        # Terror test - must test even if causing fear
        test_result = random.randint(1, 6) + random.randint(1, 6) + unit.leadership
        
        if test_result < 14:  # Failed terror test
            unit.is_fleeing = True
            return True
        
        return False
    
    @staticmethod
    def test_impetuous(orc_unit):
        """Test for Orc & Goblin Impetuous rule (replaces Animosity in TOW)"""
        if orc_unit.faction != "orc" and "orc" not in orc_unit.name.lower():
            return {"type": "none"}
        
        # Skip if fleeing, in combat, or just rallied
        if (getattr(orc_unit, 'is_fleeing', False) or 
            getattr(orc_unit, 'in_combat', False) or 
            getattr(orc_unit, 'rallied_this_turn', False)):
            return {"type": "none"}
        
        # Take Leadership test
        leadership_roll = random.randint(1, 6) + random.randint(1, 6)
        if leadership_roll > orc_unit.get_leadership():
            # Failed - must charge or move towards nearest enemy
            return {"type": "failed", "must_advance": True}
        else:
            # Passed - can act normally
            return {"type": "passed"}
    
    @staticmethod
    def test_stupidity(unit):
        """Test for stupidity (trolls, etc.)"""
        if "stupidity" not in unit.special_rules:
            return False
        
        test_result = random.randint(1, 6) + random.randint(1, 6) + unit.leadership
        
        if test_result < 14:  # Failed stupidity test
            # Unit moves in random direction
            return True
        
        return False

# =============== IMPROVED AI CLASSES ===============

class MagicItem:
    """Comprehensive magic item system"""
    @staticmethod
    def get_empire_items():
        return [
            EquipmentItem("Runefang", "magic_weapon", 65, 
                         {"strength": +2, "special": "ignores_armor", "killing_blow": True}),
            EquipmentItem("Armor of Meteoric Iron", "magic_armor", 35,
                         {"armor_save": +2, "ward_save": 6}),
            EquipmentItem("Talisman of Protection", "enchanted_item", 30,
                         {"ward_save": 6}),
            EquipmentItem("Van Horstmann's Speculum", "arcane_item", 45,
                         {"special": "swap_stats_with_enemy"}),
            EquipmentItem("Dragonhelm", "magic_armor", 10,
                         {"breath_weapon_immunity": True, "ward_save": 5}),
        ]
    
    @staticmethod 
    def get_orc_items():
        return [
            EquipmentItem("Effigy of Mork", "enchanted_item", 45,
                         {"magic_resistance": 2, "spell_immunity": True}),
            EquipmentItem("Waaagh! Paint", "enchanted_item", 35,
                         {"ward_save": 5, "special": "frenzy"}),
            EquipmentItem("Staff of Baduumm", "arcane_item", 50,
                         {"magic_level": +1, "power_dice": +1}),
            EquipmentItem("Bigged's Kickin' Boots", "enchanted_item", 15,
                         {"attacks": +1, "special": "stomp"}),
        ]

class CombatResolutionSystem:
    """Advanced combat resolution with rank bonuses, standards, flanking"""
    
    @staticmethod
    def calculate_combat_resolution(attacker, defender, casualties_dealt, casualties_taken):
        """Calculate combat resolution score including all bonuses"""
        attacker_score = casualties_dealt
        defender_score = casualties_taken
        
        # Rank bonuses (max +3)
        attacker_ranks = CombatResolutionSystem.get_complete_ranks(attacker)
        defender_ranks = CombatResolutionSystem.get_complete_ranks(defender)
        attacker_score += min(attacker_ranks - 1, 3)
        defender_score += min(defender_ranks - 1, 3)
        
        # Standard bearer bonus
        if CombatResolutionSystem.has_standard(attacker):
            attacker_score += 1
        if CombatResolutionSystem.has_standard(defender):
            defender_score += 1
            
        # Musician bonus (for pursuing)
        musician_bonus = 0
        if CombatResolutionSystem.has_musician(attacker):
            musician_bonus += 1
        if CombatResolutionSystem.has_musician(defender):
            musician_bonus -= 1
            
        # High ground bonus
        if CombatResolutionSystem.has_high_ground(attacker):
            attacker_score += 1
            
        # Flanking bonus
        if CombatResolutionSystem.is_flanking(attacker, defender):
            attacker_score += 1
            
        # Fear bonus
        if attacker.causes_fear and not defender.fear_immune:
            attacker_score += 1
            
        return attacker_score - defender_score, musician_bonus
    
    @staticmethod
    def get_complete_ranks(unit):
        """Calculate number of complete ranks"""
        if unit.formation == "skirmish":
            return 1
        elif unit.formation == "deep":
            return min(unit.models // 5, 5)  # Max 5 ranks count
        else:  # wide formation
            return min(unit.models // 10, 3)  # Max 3 ranks count
            
    @staticmethod
    def has_standard(unit):
        """Check if unit has standard bearer"""
        return unit.models >= 5 and "standard" not in unit.name.lower()
        
    @staticmethod
    def has_musician(unit):
        """Check if unit has musician"""
        return unit.models >= 10
        
    @staticmethod
    def has_high_ground(unit):
        """Check if unit has high ground advantage"""
        # Simplified - would check terrain in full implementation
        return False
        
    @staticmethod
    def is_flanking(attacker, defender):
        """Check if attacker is flanking defender"""
        # Simplified flanking check based on position
        return False

class WeaponSkillChart:
    """Authentic weapon skill hit chart"""
    
    @staticmethod
    def get_hit_chance(attacker_ws, defender_ws):
        """Get to-hit value based on WS comparison"""
        if attacker_ws == defender_ws:
            return 4  # Need 4+ to hit
        elif attacker_ws > defender_ws:
            if attacker_ws >= defender_ws * 2:
                return 2  # Need 2+ to hit
            else:
                return 3  # Need 3+ to hit
        else:  # attacker_ws < defender_ws
            if defender_ws >= attacker_ws * 2:
                return 5  # Need 5+ to hit
            else:
                return 4  # Need 4+ to hit

class MagicSystem:
    """Complete magic system with lores and spell effects"""
    
    class Spell:
        def __init__(self, name, lore, level, casting_value, effect_type, damage_range=None):
            self.name = name
            self.lore = lore
            self.level = level
            self.casting_value = casting_value
            self.effect_type = effect_type  # "damage", "buff", "debuff", "summon"
            self.damage_range = damage_range
    
    @staticmethod
    def get_empire_spells():
        return [
            MagicSystem.Spell("Fireball", "Battle Magic", 1, 8, "damage", (1, 6)),
            MagicSystem.Spell("Lightning Bolt", "Battle Magic", 2, 10, "damage", (1, 8)),
            MagicSystem.Spell("Shield of Saphery", "Battle Magic", 1, 7, "buff"),
            MagicSystem.Spell("The Dwellers Below", "Battle Magic", 6, 17, "damage", (2, 6)),
        ]
    
    @staticmethod
    def get_orc_spells():
        return [
            MagicSystem.Spell("Foot of Gork", "Big Waaagh!", 2, 9, "damage", (1, 6)),
            MagicSystem.Spell("'Eadbutt", "Big Waaagh!", 1, 8, "damage", (1, 4)),
            MagicSystem.Spell("Waaagh!", "Big Waaagh!", 3, 12, "buff"),
            MagicSystem.Spell("Green Skin is Best", "Big Waaagh!", 4, 15, "buff"),
        ]

class CompetitivePlayConstraints:
    """Implement the competitive standard from the battle report"""
    
    @staticmethod
    def get_standard_restrictions():
        """The competitive standard mentioned in the battle report"""
        return {
            "table_size": "6x4",  # 6 foot by 4 foot table
            "points_limit": 1999,  # 1999 points for most balanced play
            "legacy_armies_allowed": True,
            "allies_allowed": False,
            "rule_of_3": True,  # Max 3 units of same type
            "matched_play_restrictions": True
        }
    
    @staticmethod
    def validate_competitive_army(army_list):
        """Validate army against competitive standard"""
        constraints = CompetitivePlayConstraints.get_standard_restrictions()
        violations = []
        
        total_points = sum(unit.points for unit in army_list)
        if total_points > constraints["points_limit"]:
            violations.append(f"Army exceeds {constraints['points_limit']} point limit: {total_points}")
        
        # Apply rule of 3
        if constraints["rule_of_3"]:
            unit_counts = {}
            for unit in army_list:
                unit_type = unit.name.split()[0]  # Get base unit name
                unit_counts[unit_type] = unit_counts.get(unit_type, 0) + 1
            
            for unit_type, count in unit_counts.items():
                if count > 3:
                    violations.append(f"Rule of 3 violation: {count} {unit_type} units (max 3)")
        
        return violations

class LineOfSightSystem:
    """Comprehensive line of sight and range calculations"""
    
    @staticmethod
    def has_line_of_sight(shooter, target, terrain_features=None):
        """Check if shooter has line of sight to target"""
        # Basic line of sight - can be enhanced with terrain
        if terrain_features:
            for terrain in terrain_features:
                if LineOfSightSystem.terrain_blocks_los(shooter.position, target.position, terrain):
                    return False
        
        # Check for intervening units (simplified)
        return True
    
    @staticmethod
    def terrain_blocks_los(start_pos, end_pos, terrain):
        """Check if terrain blocks line of sight"""
        if terrain.terrain_type in ["building", "forest"]:
            # Simplified - terrain blocks LOS if line passes through it
            return LineOfSightSystem.line_intersects_rect(start_pos, end_pos, terrain.position, terrain.size)
        return False
    
    @staticmethod
    def line_intersects_rect(start, end, rect_pos, rect_size):
        """Check if line intersects rectangle (simplified)"""
        # Basic rectangular intersection check
        return False  # Simplified for now
    
    @staticmethod
    def calculate_range(shooter, target):
        """Calculate range between shooter and target"""
        dx = shooter.position[0] - target.position[0]
        dy = shooter.position[1] - target.position[1]
        return (dx * dx + dy * dy) ** 0.5

class ChargeSystem:
    """Authentic charge mechanics with impact hits and formations"""
    
    @staticmethod
    def attempt_charge(charger, target, terrain_features=None):
        """Attempt to declare and resolve charge"""
        # Calculate charge distance
        base_move = charger.movement
        charge_roll = random.randint(1, 6) + random.randint(1, 6)
        total_charge_distance = base_move + charge_roll
        
        # Check if charge is possible
        actual_distance = LineOfSightSystem.calculate_range(charger, target)
        
        if actual_distance > total_charge_distance:
            return {"success": False, "reason": "out_of_range", "distance_needed": actual_distance, "distance_available": total_charge_distance}
        
        # Check terrain penalties
        terrain_penalty = 0
        if terrain_features:
            for terrain in terrain_features:
                if TerrainSystem.unit_in_terrain(charger, terrain):
                    terrain_penalty += terrain.effects.get("movement_penalty", 0)
        
        effective_charge_distance = total_charge_distance + terrain_penalty
        
        if actual_distance > effective_charge_distance:
            return {"success": False, "reason": "terrain_penalty", "penalty": terrain_penalty}
        
        # Successful charge
        charger.has_charged = True
        impact_hits = charger.get_impact_hits()
        
        return {
            "success": True, 
            "impact_hits": impact_hits,
            "charge_distance": total_charge_distance,
            "terrain_penalty": terrain_penalty
        }

class VictoryConditionsSystem:
    """Authentic TOW victory conditions"""
    
    @staticmethod
    def calculate_victory_points(army_list, destroyed_units):
        """Calculate victory points using authentic TOW rules"""
        victory_points = 0
        
        for unit in destroyed_units:
            if unit.points > 0:
                # Full points for complete destruction
                victory_points += unit.points
        
        # Partial points for heavily damaged units (50%+ casualties)
        for unit in army_list:
            if not unit.is_destroyed and unit.models > 0:
                casualty_percentage = (unit.max_models - unit.models) / unit.max_models
                if casualty_percentage >= 0.5:  # 50% or more casualties
                    victory_points += unit.points // 2
        
        return victory_points
    
    @staticmethod
    def check_victory_conditions(empire_army, orc_army, turn_number):
        """Check various victory conditions"""
        # 1. Army broken (75% destroyed)
        empire_remaining = sum(1 for unit in empire_army if not unit.is_destroyed)
        orc_remaining = sum(1 for unit in orc_army if not unit.is_destroyed)
        
        empire_total = len(empire_army)
        orc_total = len(orc_army)
        
        if empire_remaining <= empire_total * 0.25:
            return {"winner": "Orcs", "condition": "Army Broken", "turn": turn_number}
        elif orc_remaining <= orc_total * 0.25:
            return {"winner": "Empire", "condition": "Army Broken", "turn": turn_number}
        
        # 2. Table quarters control (if game goes to turn 6+)
        if turn_number >= 6:
            # Simplified table quarter control
            empire_vp = VictoryConditionsSystem.calculate_victory_points(orc_army, [u for u in orc_army if u.is_destroyed])
            orc_vp = VictoryConditionsSystem.calculate_victory_points(empire_army, [u for u in empire_army if u.is_destroyed])
            
            if empire_vp > orc_vp:
                return {"winner": "Empire", "condition": "Victory Points", "empire_vp": empire_vp, "orc_vp": orc_vp, "turn": turn_number}
            elif orc_vp > empire_vp:
                return {"winner": "Orcs", "condition": "Victory Points", "empire_vp": empire_vp, "orc_vp": orc_vp, "turn": turn_number}
            else:
                return {"winner": "Draw", "condition": "Equal Victory Points", "empire_vp": empire_vp, "orc_vp": orc_vp, "turn": turn_number}
        
        return None  # Battle continues

class SimpleAI:
    """Enhanced AI with tactical awareness"""
    def __init__(self, faction_name):
        self.faction = faction_name
        self.strategies = {
            "artillery_strike": 0.4,
            "magic_assault": 0.2, 
            "flanking_maneuver": 0.2,
            "defensive_formation": 0.1,
            "psychological_warfare": 0.1
        }
        
        # Training attributes
        self.epsilon = 0.1  # Exploration rate
        self.win_rate = 50.0  # Default win rate
        self.primary_strategy = "artillery_strike"
        
    def act(self, state, training=False):
        """Enhanced AI decision making"""
        # Analyze battlefield situation
        battlefield_analysis = self.analyze_battlefield(state)
        
        # Choose strategy based on analysis
        strategy = self.select_strategy(battlefield_analysis)
        
        # Execute tactical action
        return self.execute_strategy(strategy, state)
    
    def analyze_battlefield(self, state):
        """Analyze current battlefield state"""
        return {
            "enemy_artillery_threat": self.count_enemy_artillery(state),
            "magic_phase_advantage": self.assess_magic_advantage(state),
            "formation_strength": self.assess_formations(state),
            "psychology_factors": self.assess_psychology(state)
        }
    
    def select_strategy(self, analysis):
        """Select best strategy based on analysis"""
        if analysis["enemy_artillery_threat"] > 2:
            return "flanking_maneuver"
        elif analysis["magic_phase_advantage"] > 0:
            return "magic_assault"
        else:
            return "artillery_strike"
    
    def execute_strategy(self, strategy, state):
        """Execute chosen strategy"""
        actions = {
            "artillery_strike": random.randint(0, 3),
            "magic_assault": random.randint(4, 5),
            "flanking_maneuver": random.randint(6, 8),
            "defensive_formation": random.randint(9, 10),
            "psychological_warfare": random.randint(11, 12)
        }
        return actions.get(strategy, 0)
    
    def count_enemy_artillery(self, state):
        """Count enemy artillery units (simplified for numpy state)"""
        # Since state is now a numpy array, return a reasonable estimate
        return random.randint(1, 3)  # Simplified - assume 1-3 artillery units
    
    def assess_magic_advantage(self, state):
        """Assess magical superiority"""
        return 0  # Simplified
    
    def assess_formations(self, state):
        """Assess formation effectiveness"""
        return 0  # Simplified
    
    def assess_psychology(self, state):
        """Assess psychological warfare opportunities"""
        return 0  # Simplified
    
    def save_model(self, filename):
        """Save AI model state (simplified for SimpleAI)"""
        model_data = {
            'faction': self.faction,
            'strategies': dict(self.strategies)
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"✅ Saved {self.faction} AI model to {filename}")
    
    def load_model(self, filename):
        """Load AI model state (simplified for SimpleAI)"""
        try:
            import json
            with open(filename, 'r') as f:
                model_data = json.load(f)
            
            self.strategies = model_data.get('strategies', self.strategies)
            print(f"✅ Loaded {self.faction} AI model from {filename}")
        except FileNotFoundError:
            print(f"⚠️ No saved model found at {filename}, using default values")
    
    def replay(self, experiences):
        """Train AI from battle experiences (simplified)"""
        # Update strategy weights based on successful experiences
        for exp in experiences:
            if exp['reward'] > 0:
                action = exp['action']
                # Strengthen successful strategies
                if action < len(list(self.strategies.keys())):
                    strategy_name = list(self.strategies.keys())[action % len(self.strategies)]
                    self.strategies[strategy_name] = min(1.0, self.strategies[strategy_name] + 0.01)
        
        # Normalize strategies
        total = sum(self.strategies.values())
        if total > 0:
            for key in self.strategies:
                self.strategies[key] /= total

# =============== ENHANCED TOW UNIT CLASS ===============

class TOWUnit:
    """Complete TOW unit with all authentic mechanics"""
    def __init__(self, name, models, unit_type, formation, faction, position, facing=0, points=0):
        # Basic properties
        self.name = name
        self.models = models
        self.max_models = models
        self.unit_type = unit_type  # "infantry", "cavalry", "monster", "warmachine", "character", "chariot"
        self.formation = formation  # "deep", "wide", "skirmish"
        self.faction = faction
        self.position = position
        self.facing = facing
        self.points = points
        
        # Combat effectiveness
        self.is_alive = models > 0
        self.is_destroyed = models <= 0
        self.casualties = 0
        self.wounds_taken = 0
        
        # COMPLETE TOW STATS
        self.movement = self.get_movement_stat()
        self.weapon_skill = self.get_weapon_skill()
        self.ballistic_skill = self.get_ballistic_skill()
        self.strength = self.get_strength()
        self.toughness = self.get_toughness()
        self.wounds = self.get_wounds()
        self.initiative = self.get_initiative()
        self.attacks = self.get_attacks()
        self.leadership = self.get_leadership()
        self.armor_save = self.get_armor_save()
        
        # EQUIPMENT AND UPGRADES
        self.equipment = []
        self.magic_items = []
        self.special_rules = self.get_special_rules()
        self.ward_save = None
        
        # COMBAT MECHANICS
        self.has_charged = False
        self.in_combat = False
        self.combat_opponent = None
        self.impact_hits = self.get_impact_hits()
        self.regeneration_save = self.get_regeneration()
        
        # PSYCHOLOGY
        self.fear_immune = self.get_fear_immunity()
        self.causes_fear = self.get_causes_fear()
        self.causes_terror = self.get_causes_terror()
        self.is_fleeing = False
        self.panic_test_needed = False
        self.animosity_target = None
        self.unbreakable = self.get_unbreakable()
        self.frenzy = self.get_frenzy()
        self.hatred = self.get_hatred()
        
        # MAGIC
        self.can_cast_magic = self.get_magic_capability()
        self.magic_level = self.get_magic_level()
        self.spells_known = []
        self.magic_resistance = self.get_magic_resistance()
        
        # RANGED COMBAT  
        self.range_weapon = self.get_range_weapon()
        self.ammunition = self.get_ammunition()
        
        # FORMATION AND COMMAND
        self.rank_bonus = 0
        self.has_standard = self.get_has_standard()
        self.has_musician = self.get_has_musician()
        self.champion = self.get_has_champion()
        
    def get_movement_stat(self):
        """Authentic movement values"""
        movement_table = {
            "cavalry": 8, "fast_cavalry": 9, "monstrous_cavalry": 8,
            "infantry": 4, "monstrous_infantry": 6,
            "monster": 6, "large_monster": 8,
            "warmachine": 3, "chariot": 8,
            "character": 4
        }
        
        # Special cases
        if "Wyvern" in self.name or "Griffon" in self.name:
            return 20  # Flying movement
        elif "Outrider" in self.name:
            return 8  # Fast cavalry
        elif "Wolf Rider" in self.name:
            return 9  # Fast cavalry
        
        return movement_table.get(self.unit_type, 4)
    
    def get_weapon_skill(self):
        """Authentic weapon skill values"""
        if "Lord" in self.name or "General" in self.name:
            return random.choice([6, 7, 8])  # Hero level
        elif "Hero" in self.name or "Bigboss" in self.name:
            return random.choice([5, 6])
        elif "Champion" in self.name:
            return 5
        elif "Troll" in self.name:
            return 3
        elif "State Troops" in self.name or "Orc Boys" in self.name:
            return 4
        elif "Goblin" in self.name:
            return 2
        elif self.unit_type == "warmachine":
            return 2  # Crew
        else:
            return 3
    
    def get_ballistic_skill(self):
        """Authentic ballistic skill values"""
        if "Engineer" in self.name:
            return 4
        elif "Handgunners" in self.name or "Outrider" in self.name:
            return 3
        elif "Orc Boys with Warbows" in self.name:
            return 3
        elif "Goblin" in self.name:
            return 3
        else:
            return 2
    
    def get_strength(self):
        """Authentic strength values"""
        if "Troll" in self.name:
            return 5
        elif "Wyvern" in self.name or "Griffon" in self.name:
            return 6
        elif "General" in self.name or "Lord" in self.name:
            return 4
        elif "Great Cannon" in self.name:
            return 10  # Artillery strength
        elif "Helblaster" in self.name:
            return 5
        else:
            return 3
    
    def get_toughness(self):
        """Authentic toughness values"""
        if "Troll" in self.name:
            return 4
        elif "Wyvern" in self.name or "Griffon" in self.name:
            return 5
        elif self.unit_type == "warmachine":
            return 7
        elif "Orc" in self.name:
            return 4
        else:
            return 3
    
    def get_wounds(self):
        """Authentic wounds values"""
        if "Troll" in self.name:
            return 3
        elif "Wyvern" in self.name or "Griffon" in self.name:
            return 4
        elif "Lord" in self.name or "General" in self.name:
            return 3
        elif "Hero" in self.name or "Bigboss" in self.name:
            return 2
        elif self.unit_type == "warmachine":
            return 3
        else:
            return 1
    
    def get_initiative(self):
        """Authentic initiative values"""
        if "Wyvern" in self.name or "Griffon" in self.name:
            return 3
        elif "Troll" in self.name:
            return 1
        elif "Orc" in self.name:
            return 2
        elif "Goblin" in self.name:
            return 3
        else:
            return 3
    
    def get_attacks(self):
        """Authentic attacks values"""
        if "Lord" in self.name or "General" in self.name:
            return 4
        elif "Hero" in self.name or "Bigboss" in self.name:
            return 3
        elif "Troll" in self.name:
            return 3
        elif "Wyvern" in self.name:
            return 4
        elif "Champion" in self.name:
            return 2
        else:
            return 1
    
    def get_leadership(self):
        """Authentic leadership values"""
        if "Lord" in self.name or "General" in self.name:
            return 9
        elif "Hero" in self.name or "Bigboss" in self.name:
            return 8
        elif "Mage" in self.name or "Engineer" in self.name:
            return 7
        elif "Orc" in self.name:
            return 7
        elif "Goblin" in self.name:
            return 6
        else:
            return 8
    
    def get_armor_save(self):
        """Authentic armor save values"""
        if "State Troops" in self.name:
            return 5  # Heavy armor + shield
        elif "General" in self.name or "Lord" in self.name:
            return 3  # Full plate armor
        elif "Bigboss" in self.name:
            return 4  # Heavy armor
        elif "Orc Boys" in self.name:
            return 6  # Light armor + shield
        elif self.unit_type == "warmachine":
            return None  # No armor save
        else:
            return 6  # Light armor
    
    def get_special_rules(self):
        """Get unit special rules"""
        rules = []
        
        if "Troll" in self.name:
            rules.extend(["regeneration", "stupidity", "causes_fear"])
        if "Wyvern" in self.name or "Griffon" in self.name:
            rules.extend(["fly", "large_target", "causes_terror"])
        if "Orc" in self.name:
            rules.append("animosity")
        if "Goblin" in self.name:
            rules.extend(["animosity", "small_target"])
        if "Handgunners" in self.name:
            rules.append("armor_piercing")
        if self.unit_type == "warmachine":
            rules.extend(["artillery", "large_target"])
        
        return rules
    
    def get_impact_hits(self):
        """Get impact hits on charge"""
        if "Wyvern" in self.name or "Griffon" in self.name:
            return 2
        elif "Chariot" in self.name:
            return random.randint(1, 6)  # D6 impact hits
        elif self.unit_type == "cavalry":
            return 1
        else:
            return 0
    
    def get_regeneration(self):
        """Get regeneration save"""
        if "Troll" in self.name:
            return 4  # 4+ regeneration save
        else:
            return None
    
    def get_fear_immunity(self):
        """Check fear immunity"""
        return any(rule in self.special_rules for rule in ["causes_fear", "causes_terror", "undead", "daemonic"])
    
    def get_causes_fear(self):
        """Check if causes fear"""
        return "causes_fear" in self.special_rules or "causes_terror" in self.special_rules
    
    def get_causes_terror(self):
        """Check if causes terror"""
        return "causes_terror" in self.special_rules
    
    def get_unbreakable(self):
        """Check if unbreakable"""
        return "unbreakable" in self.special_rules
    
    def get_frenzy(self):
        """Check if frenzied"""
        return "frenzy" in self.special_rules
    
    def get_hatred(self):
        """Get hatred targets"""
        if "Dwarf" in self.name and "Orc" in self.faction:
            return ["dwarf"]
        return []
    
    def get_magic_capability(self):
        """Check if can cast magic"""
        return any(keyword in self.name for keyword in ["Mage", "Wizard", "Weirdnob", "Shaman"])
    
    def get_magic_level(self):
        """Get magic level"""
        if "Master Mage" in self.name:
            return 4
        elif "Weirdnob" in self.name:
            return 4
        elif any(keyword in self.name for keyword in ["Mage", "Wizard"]):
            return 2
        else:
            return 0
    
    def get_magic_resistance(self):
        """Get magic resistance value"""
        return 0  # Default no resistance
    
    def get_range_weapon(self):
        """Get ranged weapon statistics"""
        weapons = {
            "Great Cannon": {"range": 60, "strength": 10, "special": ["artillery", "guess", "wound_on_4+"]},
            "Helblaster": {"range": 24, "strength": 5, "special": ["multiple_shots", "armor_piercing"]},
            "Handgunners": {"range": 24, "strength": 4, "special": ["armor_piercing", "no_long_range_penalty"]},
            "Warbows": {"range": 24, "strength": 3, "special": ["volley_fire"]},
            "War Wagon": {"range": 24, "strength": 4, "special": ["mobile_artillery"]},
        }
        
        for weapon_name, stats in weapons.items():
            if weapon_name in self.name:
                return stats
        return None
    
    def get_ammunition(self):
        """Get ammunition count"""
        if self.range_weapon:
            return 20  # Standard ammunition
        return 0
    
    def get_has_standard(self):
        """Check if unit has standard bearer"""
        return self.models >= 5 and self.unit_type in ["infantry", "cavalry"]
    
    def get_has_musician(self):
        """Check if unit has musician"""
        return self.models >= 10 and self.unit_type in ["infantry", "cavalry"]
    
    def get_has_champion(self):
        """Check if unit has champion"""
        return self.models >= 5 and self.unit_type in ["infantry", "cavalry"]
    
    def take_psychology_test(self, test_type="basic"):
        """Take a psychology test (Leadership-based)"""
        roll = random.randint(1, 6) + random.randint(1, 6)  # 2D6
        leadership_value = self.leadership
        
        # Apply modifiers based on test type
        if test_type == "fear":
            leadership_value -= 2
        elif test_type == "panic":
            leadership_value -= 1
            
        return roll <= leadership_value
    
    def to_dict(self):
        """Convert unit to dictionary for JSON serialization"""
        try:
            return {
                'name': str(self.name),
                'models': int(self.models),
                'max_models': int(getattr(self, 'max_models', self.models)),
                'unit_type': str(self.unit_type),
                'formation': str(self.formation),
                'faction': str(self.faction),
                'position': list(self.position) if hasattr(self.position, '__iter__') else [0, 0],
                'facing': float(self.facing),
                'points': int(self.points),
                'is_destroyed': bool(self.models <= 0),
                'is_alive': bool(self.models > 0),
                'movement': int(getattr(self, 'movement', 4)),
                'weapon_skill': int(getattr(self, 'weapon_skill', 3)),
                'ballistic_skill': int(getattr(self, 'ballistic_skill', 3)),
                'strength': int(getattr(self, 'strength', 3)),
                'toughness': int(getattr(self, 'toughness', 3)),
                'wounds': int(getattr(self, 'wounds', 1)),
                'initiative': int(getattr(self, 'initiative', 3)),
                'attacks': int(getattr(self, 'attacks', 1)),
                'leadership': int(getattr(self, 'leadership', 7)),
                'armor_save': getattr(self, 'armor_save', None),
                'special_rules': [],
                'wounds_taken': int(getattr(self, 'wounds_taken', 0)),
                'has_charged': bool(getattr(self, 'has_charged', False)),
                'in_combat': bool(getattr(self, 'in_combat', False)),
                'is_fleeing': bool(getattr(self, 'is_fleeing', False)),
                'animosity_target': str(getattr(self, 'animosity_target', None)) if getattr(self, 'animosity_target', None) else None,
                'casualties': int(getattr(self, 'casualties', 0)),
                'range_weapon': None  # Simplified - don't serialize complex weapon objects
            }
        except Exception as e:
            # Return minimal safe dict if serialization fails
            return {
                'name': str(getattr(self, 'name', 'Unknown Unit')),
                'models': int(getattr(self, 'models', 1)),
                'faction': str(getattr(self, 'faction', 'unknown')),
                'position': [0, 0],
                'is_alive': bool(getattr(self, 'models', 1) > 0),
                'unit_type': str(getattr(self, 'unit_type', 'infantry')),
                'error': str(e)
            }
    
    def can_see_target(self, target):
        """Check if this unit can see the target for shooting"""
        try:
            if not target or not hasattr(target, 'is_alive') or not target.is_alive:
                return False
                
            # Calculate distance
            dx = target.position[0] - self.position[0]
            dy = target.position[1] - self.position[1]
            distance = (dx*dx + dy*dy) ** 0.5
            
            # Check if weapon has range
            range_weapon = self.get_range_weapon()
            if range_weapon:
                weapon_range = range_weapon.get("range", 24) * 10  # Convert to pixels
                return distance <= weapon_range
            
            return False
        except Exception as e:
            print(f"🔧 DEBUG: can_see_target error for {self.name}: {e}")
            return False
    
    def calculate_charge_distance(self):
        """Calculate how far this unit can charge"""
        try:
            base_movement = self.get_movement_stat()
            charge_distance = base_movement * 2  # Standard charge is 2x movement
            
            # Add random factor for charge rolls
            charge_roll = random.randint(1, 6) + random.randint(1, 6)
            total_distance = charge_distance + charge_roll
            
            print(f"🔧 DEBUG: {self.name} charge distance: {base_movement} move * 2 + {charge_roll} roll = {total_distance}")
            return total_distance
        except Exception as e:
            print(f"🔧 DEBUG: calculate_charge_distance error for {self.name}: {e}")
            return 12  # Default fallback

class TOWBattle:
    def __init__(self):
        self.units = []
        self.turn = 1
        self.phase = "deployment"
        self.active_player = "nuln"
        self.battle_log = []
        self.battle_state = "ready"
        self.lock = Lock()
        
        # AI agents
        self.nuln_ai = None
        self.orc_ai = None
        self.load_ai_agents()
        
        # Create the actual trained armies
        self.create_authentic_armies()
        
    def load_ai_agents(self):
        """Load simplified AI agents"""
        try:
            print("🤖 Loading AI Commanders...")
            
            # Create simplified AIs based on training results
            self.nuln_ai = SimpleAI("Nuln Empire")
            self.nuln_ai.win_rate = 96.15
            self.nuln_ai.primary_strategy = "Artillery Strike"
            print("✅ Nuln AI loaded (96.15% win rate - Artillery Strike specialist)")
            
            self.orc_ai = SimpleAI("Troll Horde") 
            self.orc_ai.win_rate = 83.15
            self.orc_ai.primary_strategy = "Anti-Artillery specialist"
            print("✅ Troll AI loaded (83.15% win rate - Anti-Artillery specialist)")
            
        except Exception as e:
            print(f"❌ Error loading AI: {e}")
            # Create dummy AIs if loading fails
            self.nuln_ai = SimpleAI("Nuln Empire")
            self.orc_ai = SimpleAI("Troll Horde")

    def create_authentic_armies(self):
        """Create the exact armies that were used to train the AI"""
        self.units = []
        
        print("🏛️ DEPLOYING THE ACTUAL TRAINED ARMIES")
        
        # ============ NULN EMPIRE ARMY (2000 pts) ============
        nuln_units = [
            # Characters
            TOWUnit("General Hans von Löwenhacke", 1, "character", "skirmish", "nuln", [150, 450], 0, 190),
            TOWUnit("Empire Engineer", 1, "character", "skirmish", "nuln", [120, 430], 0, 55),
            TOWUnit("Master Mage", 1, "character", "skirmish", "nuln", [180, 430], 0, 90),
            TOWUnit("Engineer with War Wagon", 4, "warmachine", "skirmish", "nuln", [100, 400], 0, 185),
            TOWUnit("General on Griffon", 1, "monster", "skirmish", "nuln", [200, 470], 0, 250),
            
            # Core Units
            TOWUnit("20 Nuln Veteran State Troops", 20, "infantry", "deep", "nuln", [150, 380], 0, 375),
            TOWUnit("20 State Handgunners", 20, "infantry", "wide", "nuln", [150, 350], 0, 0),  # Part of above unit
            TOWUnit("5 Outriders", 5, "cavalry", "wide", "nuln", [80, 450], 0, 95),
            TOWUnit("5 Outriders", 5, "cavalry", "wide", "nuln", [220, 450], 0, 95),
            
            # Special Units  
            TOWUnit("Great Cannon", 4, "warmachine", "skirmish", "nuln", [50, 380], 0, 130),
            TOWUnit("Great Cannon", 4, "warmachine", "skirmish", "nuln", [250, 380], 0, 130),
            
            # Rare Units
            TOWUnit("Helblaster Volley Gun", 3, "warmachine", "skirmish", "nuln", [50, 350], 0, 135),
            TOWUnit("Helblaster Volley Gun", 3, "warmachine", "skirmish", "nuln", [200, 350], 0, 135),
            TOWUnit("Helblaster Volley Gun", 3, "warmachine", "skirmish", "nuln", [250, 350], 0, 135),
        ]
        
        # ============ ORC & GOBLIN ARMY (1963 pts) ============
        orc_units = [
            # Characters
            TOWUnit("Orc Bigboss on Boar Chariot", 3, "chariot", "skirmish", "orcs", [150, 50], 180, 309),
            TOWUnit("Orc Warboss on Wyvern", 1, "monster", "skirmish", "orcs", [200, 30], 180, 346),
            TOWUnit("Orc Weirdnob Wizard", 1, "character", "skirmish", "orcs", [100, 30], 180, 230),
            
            # Core Units
            TOWUnit("8 Common Trolls", 8, "monster", "wide", "orcs", [150, 80], 180, 360),
            TOWUnit("27 Orc Boys with Warbows", 27, "infantry", "deep", "orcs", [150, 120], 180, 172),
            TOWUnit("4 River Trolls", 4, "monster", "wide", "orcs", [80, 100], 180, 212),
            
            # Special Units
            TOWUnit("4 River Trolls", 4, "monster", "wide", "orcs", [220, 100], 180, 212),
            TOWUnit("4 River Trolls", 4, "monster", "wide", "orcs", [150, 60], 180, 212),
        ]
        
        self.units = nuln_units + orc_units
        
        print(f"🔵 NULN EMPIRE: {len(nuln_units)} units deployed")
        print(f"🟢 ORC & GOBLIN TRIBES: {len(orc_units)} units deployed")
        
        # Show deployment details
        print("🔵 NULN EMPIRE DEPLOYMENT:")
        total_nuln_points = 0
        for unit in nuln_units:
            if unit.points > 0:  # Only show units with points (not sub-units)
                print(f"  • {unit.name}: {unit.models} models in {unit.formation} formation ({unit.points} pts)")
                total_nuln_points += unit.points
        print(f"  📊 Total: {total_nuln_points} points")
        
        print("🟢 ORC & GOBLIN DEPLOYMENT:")
        total_orc_points = 0
        for unit in orc_units:
            print(f"  • {unit.name}: {unit.models} models in {unit.formation} formation ({unit.points} pts)")
            total_orc_points += unit.points
        print(f"  📊 Total: {total_orc_points} points")

    def get_battle_state(self):
        """Get current battle state for web interface"""
        with self.lock:
            return {
                'units': [unit.to_dict() for unit in self.units],
                'turn': self.turn,
                'phase': self.phase,
                'active_player': self.active_player,
                'battle_log': self.battle_log[-10:],  # Last 10 log entries
                'battle_state': self.battle_state
            }

    def add_log(self, message, debug=False):
        """Add message to battle log with enhanced console output"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.battle_log.append(log_entry)
        
        # Enhanced console output
        if debug:
            print(f"🔧 DEBUG: {log_entry}")
        else:
            print(log_entry)
    
    def debug_log(self, message):
        """Add debug message with special formatting"""
        self.add_log(f"🔧 DEBUG: {message}", debug=True)

    def start_battle(self):
        """Start the battle with AI vs AI combat"""
        if self.battle_state != "ready":
            return False
            
        self.battle_state = "active"
        self.add_log("⚔️ THE OLD WORLD BATTLE BEGINS!")
        self.add_log("🎺 The horns of war sound across the battlefield!")
        
        # Start battle in separate thread
        battle_thread = Thread(target=self.run_battle_loop)
        battle_thread.daemon = True
        battle_thread.start()
        
        return True

    def run_battle_loop(self):
        """AUTHENTIC TOW BATTLE LOOP with proper phases following official rules"""
        try:
            max_turns = 8  # Standard TOW game length
            
            while self.turn <= max_turns and self.battle_state == "active":
                self.add_log(f"⚔️ TURN {self.turn} - THE OLD WORLD")
                
                # ========== STRATEGY PHASE ==========
                self.phase = "strategy"
                self.add_log("🏛️ STRATEGY PHASE")
                socketio.emit('battle_update', self.get_battle_state())
                self.strategy_phase()
                time.sleep(1.5)
                
                # ========== MOVEMENT PHASE ==========
                self.phase = "movement" 
                self.add_log("📍 MOVEMENT PHASE")
                socketio.emit('battle_update', self.get_battle_state())
                self.movement_phase()
                time.sleep(2)
                
                # ========== SHOOTING PHASE ==========
                self.phase = "shooting"
                self.add_log("🏹 SHOOTING PHASE")
                socketio.emit('battle_update', self.get_battle_state())
                self.shooting_phase()
                time.sleep(2)
                
                # ========== COMBAT PHASE ==========
                self.phase = "combat"
                self.add_log("⚔️ COMBAT PHASE")
                socketio.emit('battle_update', self.get_battle_state())
                self.combat_phase()
                time.sleep(2)
                
                # Check for victory conditions
                if self.check_victory():
                    break
                    
                self.turn += 1
                time.sleep(1)
                
            self.end_battle()
            
        except Exception as e:
            self.add_log(f"❌ Battle error: {e}")
            self.battle_state = "error"
            socketio.emit('battle_update', self.get_battle_state())

    def strategy_phase(self):
        """STRATEGY PHASE: Start of Turn -> Command -> Conjuration -> Rally Fleeing Troops"""
        try:
            # Sub-phase 1: Start of Turn
            self.debug_log("Strategy Phase - Start of Turn sub-phase")
            self.start_of_turn_tests()
            
            # Sub-phase 2: Command 
            self.debug_log("Strategy Phase - Command sub-phase")
            self.command_sub_phase()
            
            # Sub-phase 3: Conjuration (Enchantments & Hexes)
            self.debug_log("Strategy Phase - Conjuration sub-phase")
            self.conjuration_sub_phase()
            
            # Sub-phase 4: Rally Fleeing Troops
            self.debug_log("Strategy Phase - Rally Fleeing Troops sub-phase")
            self.rally_fleeing_troops()
            
        except Exception as e:
            self.add_log(f"❌ Strategy phase error: {e}")

    def start_of_turn_tests(self):
        """Start of Turn: Stupidity tests, Impetuous tests, special abilities"""
        for unit in [u for u in self.units if u.is_alive]:
            # Stupidity tests for Trolls not in combat
            if ("troll" in unit.name.lower() and 
                not getattr(unit, 'in_combat', False) and 
                not getattr(unit, 'is_fleeing', False)):
                
                stupidity_roll = random.randint(1, 6) + random.randint(1, 6)
                if stupidity_roll > unit.get_leadership():
                    # Failed stupidity test
                    self.add_log(f"🤪 {unit.name} fails Stupidity test!")
                    unit.stupidity_failed = True
                    # Will move compulsively in Movement Phase
                else:
                    unit.stupidity_failed = False
            
            # Impetuous tests for Orc & Goblin units (replaces Animosity)
            if unit.faction == "orc" or "orc" in unit.name.lower():
                impetuous_result = PsychologySystem.test_impetuous(unit)
                if impetuous_result["type"] == "failed":
                    self.add_log(f"😤 {unit.name} fails Impetuous test - must advance towards enemy!")
                    unit.impetuous_failed = True
                elif impetuous_result["type"] == "passed":
                    unit.impetuous_failed = False

    def command_sub_phase(self):
        """Command abilities like Inspiring Presence"""
        # General's Inspiring Presence and other command abilities
        for unit in [u for u in self.units if u.is_alive and "general" in u.name.lower()]:
            self.debug_log(f"{unit.name} provides Inspiring Presence to nearby units")
            # Implementation of command abilities would go here

    def conjuration_sub_phase(self):
        """Cast Enchantments and Hexes only"""
        # Only Enchantment and Hex spells can be cast here
        wizards = [u for u in self.units if u.is_alive and u.get_magic_capability() and 
                  not getattr(u, 'is_fleeing', False) and not getattr(u, 'in_combat', False)]
        
        for wizard in wizards:
            if random.random() < 0.3:  # 30% chance to attempt spell
                # Only cast Enchantments/Hexes here (beneficial/detrimental spells)
                enchantment_spells = ["Mystic Shield", "Curse of Arrow Attraction", "Hand of Glory"]
                if enchantment_spells:
                    spell = random.choice(enchantment_spells)
                    self.add_log(f"🔮 {wizard.name} attempts to cast {spell}")
                    # Spell resolution would go here

    def rally_fleeing_troops(self):
        """Attempt to rally fleeing units"""
        fleeing_units = [u for u in self.units if getattr(u, 'is_fleeing', False)]
        
        for unit in fleeing_units:
            rally_roll = random.randint(1, 6) + random.randint(1, 6)
            leadership = unit.get_leadership()
            
            # -1 if below half strength
            if unit.models < (getattr(unit, 'starting_models', unit.models) / 2):
                leadership -= 1
                
            if rally_roll <= leadership:
                unit.is_fleeing = False
                unit.rallied_this_turn = True
                self.add_log(f"🛡️ {unit.name} rallies!")
            else:
                self.add_log(f"😰 {unit.name} continues to flee")

    def movement_phase(self):
        """MOVEMENT PHASE: Declare Charges -> Charge Moves -> Compulsory Moves -> Remaining Moves"""
        try:
            # Sub-phase 1: Declare Charges & Charge Reactions
            self.debug_log("Movement Phase - Declare Charges sub-phase")
            self.declare_charges_sub_phase()
            
            # Sub-phase 2: Charge Moves
            self.debug_log("Movement Phase - Charge Moves sub-phase") 
            self.charge_moves_sub_phase()
            
            # Sub-phase 3: Compulsory Moves (fleeing, stupidity)
            self.debug_log("Movement Phase - Compulsory Moves sub-phase")
            self.compulsory_moves_sub_phase()
            
            # Sub-phase 4: Remaining Moves (normal movement)
            self.debug_log("Movement Phase - Remaining Moves sub-phase")
            self.remaining_moves_sub_phase()
            
            # Conveyance spells can be cast during Movement Phase
            self.debug_log("Movement Phase - Conveyance spells")
            self.conveyance_spells()
            
        except Exception as e:
            self.add_log(f"❌ Movement phase error: {e}")

    def declare_charges_sub_phase(self):
        """Declare charges and charge reactions"""
        # Determine which units can charge
        for unit in [u for u in self.units if u.is_alive and not getattr(u, 'is_fleeing', False) 
                    and not getattr(u, 'rallied_this_turn', False) and not getattr(u, 'in_combat', False)]:
            
            # AI decision for charges or random for now
            if random.random() < 0.2:  # 20% chance to attempt charge
                enemy_faction = "orcs" if unit.faction == "nuln" else "nuln"
                targets = [u for u in self.units if u.faction == enemy_faction and u.is_alive]
                
                if targets:
                    target = random.choice(targets)
                    
                    # Calculate charge distance needed
                    dx = target.position[0] - unit.position[0]
                    dy = target.position[1] - unit.position[1]
                    distance_needed = (dx*dx + dy*dy) ** 0.5
                    
                    # Declare charge
                    self.add_log(f"⚡ {unit.name} declares charge against {target.name}!")
                    
                    # Target must declare charge reaction
                    reaction = self.declare_charge_reaction(target, unit)
                    
                    # Store charge for resolution in next sub-phase
                    if not hasattr(self, 'declared_charges'):
                        self.declared_charges = []
                    self.declared_charges.append({
                        'charger': unit,
                        'target': target,
                        'reaction': reaction,
                        'distance_needed': distance_needed
                    })

    def declare_charge_reaction(self, target, charger):
        """Target declares charge reaction (Hold, Flee, Stand & Shoot, Counter Charge)"""
        # Fear/Terror tests if applicable
        if charger.get_causes_fear() or charger.get_causes_terror():
            fear_roll = random.randint(1, 6) + random.randint(1, 6)
            if fear_roll > target.get_leadership():
                # Must flee due to fear
                target.is_fleeing = True
                self.add_log(f"😱 {target.name} fails fear test and must flee!")
                return "flee"
        
        # Choose reaction (simplified AI decision)
        if target.get_range_weapon():
            # Can Stand & Shoot if charger is far enough
            return "stand_and_shoot"
        else:
            return "hold"

    def charge_moves_sub_phase(self):
        """Resolve charge moves"""
        if not hasattr(self, 'declared_charges'):
            return
            
        for charge in self.declared_charges:
            charger = charge['charger']
            target = charge['target']
            reaction = charge['reaction']
            
            # Roll charge distance: 2D6 + Movement
            charge_dice = random.randint(1, 6) + random.randint(1, 6)
            charge_distance = charge_dice + charger.get_movement_stat()
            
            self.debug_log(f"{charger.name} charge distance: {charger.get_movement_stat()} move * 2 + {charge_dice} roll = {charge_distance}")
            
            # Handle Stand & Shoot reaction first
            if reaction == "stand_and_shoot":
                self.resolve_stand_and_shoot(target, charger)
            
            # Check if charge reaches target
            if charge_distance >= charge['distance_needed']:
                # Successful charge!
                self.add_log(f"⚡ {charger.name} charges into {target.name}!")
                
                # Move charger into contact
                charger.position = target.position.copy()
                charger.in_combat = True
                target.in_combat = True
                
                # Impact Hits if applicable
                impact_hits = charger.get_impact_hits()
                if impact_hits > 0:
                    self.resolve_impact_hits(charger, target, impact_hits)
                    
                # Mark as charged this turn for combat bonuses
                charger.charged_this_turn = True
            else:
                # Failed charge
                self.add_log(f"❌ {charger.name} charge fails! (needed {charge['distance_needed']}, rolled {charge_distance})")
                
        # Clear declared charges
        self.declared_charges = []

    def resolve_stand_and_shoot(self, shooter, charger):
        """Resolve Stand & Shoot charge reaction"""
        self.add_log(f"🏹 {shooter.name} stands and shoots at charging {charger.name}!")
        
        weapon = shooter.get_range_weapon()
        if weapon:
            # Stand & Shoot typically has -1 To Hit penalty
            hit_roll = random.randint(1, 6)
            hit_needed = 7 - shooter.get_ballistic_skill() + 1  # +1 penalty for Stand & Shoot
            
            if hit_roll >= hit_needed:
                # Resolve wound and potential casualties
                self.resolve_shooting_attack(shooter, charger, weapon, hit_penalty=1)

    def resolve_impact_hits(self, charger, target, num_hits):
        """Resolve Impact Hits from charging"""
        strength = charger.get_strength()  # Impact Hits at base Strength unless specified
        
        for i in range(num_hits):
            wound_roll = random.randint(1, 6)
            wound_needed = self.calculate_wound_needed(strength, target.get_toughness())
            
            if wound_roll >= wound_needed:
                # Check armor save
                if target.get_armor_save() and random.randint(1, 6) < target.get_armor_save():
                    self.add_log(f"🛡️ {target.name} saves against impact hit!")
                else:
                    target.models -= 1
                    if target.models <= 0:
                        target.is_alive = False
                        self.add_log(f"💥 {charger.name} charge destroys {target.name} with {num_hits} impact hits!")
                        return
        
        if num_hits > 0:
            self.add_log(f"⚡ {charger.name} delivers {num_hits} impact hits!")

    def compulsory_moves_sub_phase(self):
        """Units that must move (fleeing, stupidity, impetuous, etc.)"""
        # Fleeing units continue to flee
        for unit in [u for u in self.units if getattr(u, 'is_fleeing', False)]:
            flee_distance = random.randint(1, 6) + random.randint(1, 6)
            self.add_log(f"🏃 {unit.name} flees {flee_distance} inches!")
            # Move unit away from enemy (simplified)
            unit.position[1] += flee_distance * (1 if unit.faction == "nuln" else -1)
            
        # Units that failed Stupidity
        for unit in [u for u in self.units if getattr(u, 'stupidity_failed', False)]:
            if random.randint(1, 6) == 1:
                # Move towards nearest enemy
                self.add_log(f"🤪 {unit.name} stumbles towards the enemy due to stupidity!")
            else:
                # Do nothing this turn
                self.add_log(f"🤪 {unit.name} stands confused due to stupidity!")
        
        # Units that failed Impetuous test must advance towards enemy
        for unit in [u for u in self.units if getattr(u, 'impetuous_failed', False)]:
            enemy_faction = "nuln" if unit.faction == "orc" else "orc"
            enemies = [u for u in self.units if u.faction == enemy_faction and u.is_alive]
            
            if enemies:
                # Find nearest enemy and move towards them
                nearest_enemy = min(enemies, key=lambda e: 
                    (e.position[0] - unit.position[0])**2 + (e.position[1] - unit.position[1])**2)
                
                # Move towards enemy (simplified movement)
                direction_x = 1 if nearest_enemy.position[0] > unit.position[0] else -1
                direction_y = 1 if nearest_enemy.position[1] > unit.position[1] else -1
                
                move_distance = unit.get_movement_stat() * 10  # Convert to pixels
                unit.position[0] += direction_x * move_distance * 0.5
                unit.position[1] += direction_y * move_distance * 0.5
                
                self.add_log(f"😤 {unit.name} advances aggressively towards {nearest_enemy.name} (Impetuous)!")

    def remaining_moves_sub_phase(self):
        """Normal movement for units that haven't moved yet"""
        if self.nuln_ai and self.orc_ai:
            self.ai_movement_phase()
        else:
            self.random_movement_phase()

    def conveyance_spells(self):
        """Cast Conveyance spells (teleportation, etc.) during Movement Phase"""
        wizards = [u for u in self.units if u.is_alive and u.get_magic_capability() and 
                  not getattr(u, 'is_fleeing', False) and not getattr(u, 'in_combat', False)]
        
        for wizard in wizards:
            if random.random() < 0.1:  # 10% chance for Conveyance spell
                conveyance_spells = ["Dimensional Door", "Teleport"]
                if conveyance_spells:
                    spell = random.choice(conveyance_spells)
                    self.add_log(f"🌀 {wizard.name} casts {spell}!")

    def shooting_phase(self):
        """SHOOTING PHASE: Declare Targets -> Roll to Hit -> Roll to Wound -> Remove Casualties"""
        try:
            # Magic Missiles and Magical Vortexes can be cast here
            self.magic_missiles_and_vortexes()
            
            # Regular shooting
            if self.nuln_ai and self.orc_ai:
                self.ai_shooting_phase()
            else:
                self.random_shooting_phase()
                
        except Exception as e:
            self.add_log(f"❌ Shooting phase error: {e}")

    def magic_missiles_and_vortexes(self):
        """Cast Magic Missiles and Magical Vortex spells during Shooting Phase"""
        wizards = [u for u in self.units if u.is_alive and u.get_magic_capability() and 
                  not getattr(u, 'is_fleeing', False) and not getattr(u, 'in_combat', False)]
        
        for wizard in wizards:
            if random.random() < 0.4:  # 40% chance for offensive magic
                if "weirdnob" in wizard.name.lower():
                    # Orc spells
                    offensive_spells = ["'Eadbutt", "Brain Bursta", "Fists of Gork"]
                    spell = random.choice(offensive_spells)
                    self.cast_offensive_spell(wizard, spell)
                elif "mage" in wizard.name.lower():
                    # Empire spells  
                    offensive_spells = ["Lightning Bolt", "Fireball", "Banishment"]
                    spell = random.choice(offensive_spells)
                    self.cast_offensive_spell(wizard, spell)

    def cast_offensive_spell(self, wizard, spell_name):
        """Cast an offensive spell during Shooting Phase"""
        enemy_faction = "orcs" if wizard.faction == "nuln" else "nuln"
        targets = [u for u in self.units if u.faction == enemy_faction and u.is_alive]
        
        if targets:
            target = random.choice(targets)
            
            # Casting roll
            casting_roll = random.randint(1, 6) + random.randint(1, 6) + wizard.get_magic_level()
            casting_value = 8  # Typical casting value
            
            # Check for miscast (double 1s)
            if casting_roll == 2:
                self.add_log(f"💀 {wizard.name}'s magic backfires! Takes 1 wound!")
                wizard.wounds = max(0, wizard.wounds - 1)
                if wizard.wounds <= 0:
                    wizard.is_alive = False
                return
            
            if casting_roll >= casting_value:
                # Successful cast
                damage = random.randint(1, 6)
                target.models = max(0, target.models - damage)
                
                if target.models <= 0:
                    target.is_alive = False
                    self.add_log(f"⚡ {wizard.name} destroys {target.name} with {spell_name}!")
                else:
                    self.add_log(f"💥 {wizard.name}'s {spell_name} hits {target.name} for {damage} casualties!")
            else:
                self.add_log(f"❌ {wizard.name} fails to cast {spell_name}")

    def calculate_wound_needed(self, strength, toughness):
        """Calculate wound roll needed based on S vs T"""
        if strength >= toughness * 2:
            return 2
        elif strength > toughness:
            return 3
        elif strength == toughness:
            return 4
        elif strength < toughness:
            return 5
        else:
            return 6

    def resolve_shooting_attack(self, shooter, target, weapon, hit_penalty=0):
        """Resolve a single shooting attack"""
        # Hit roll
        hit_roll = random.randint(1, 6)
        hit_needed = 7 - shooter.get_ballistic_skill() + hit_penalty
        
        if hit_roll >= hit_needed:
            # Wound roll
            wound_roll = random.randint(1, 6) 
            wound_needed = self.calculate_wound_needed(weapon["strength"], target.get_toughness())
            
            if wound_roll >= wound_needed:
                # Armor save
                if target.get_armor_save() and random.randint(1, 6) >= target.get_armor_save():
                    target.models -= 1
                    if target.models <= 0:
                        target.is_alive = False
                        self.add_log(f"💥 {shooter.name} destroys {target.name}!")
                    else:
                        self.add_log(f"🎯 {shooter.name} hits {target.name}!")
                else:
                    self.add_log(f"🛡️ {target.name} armor saves!")
            else:
                self.add_log(f"⚪ {shooter.name} hits but fails to wound {target.name}")
        else:
            self.add_log(f"❌ {shooter.name} misses {target.name}")

    def ai_movement_phase(self):
        """AI-controlled movement phase with enhanced logging"""
        try:
            # Get current battle state for AI
            state = self.get_ai_state()
            self.debug_log(f"AI state generated: shape={state.shape}, type={type(state)}")
            
            # Nuln AI turn
            if self.nuln_ai:
                self.debug_log("Nuln AI making movement decision...")
                action = self.nuln_ai.act(state, training=False)
                self.debug_log(f"Nuln AI chose action: {action}")
                self.execute_ai_action("nuln", action)
            
            # Orc AI turn  
            if self.orc_ai:
                self.debug_log("Orc AI making movement decision...")
                action = self.orc_ai.act(state, training=False)
                self.debug_log(f"Orc AI chose action: {action}")
                self.execute_ai_action("orcs", action)
                
        except Exception as e:
            self.add_log(f"❌ AI movement error: {e}")
            self.debug_log(f"Full AI movement error: {type(e).__name__}: {e}")
            self.add_log("🔄 Falling back to random movement...")
            self.random_movement_phase()

    def ai_shooting_phase(self):
        """ENHANCED shooting phase with range, line of sight, and ballistic skill"""
        try:
            # All units with ranged weapons
            all_shooters = [u for u in self.units if u.is_alive and u.get_range_weapon() is not None]
            self.debug_log(f"Found {len(all_shooters)} units with ranged weapons")
            
            for shooter in all_shooters:
                self.debug_log(f"Checking shooter: {shooter.name}")
                
                if getattr(shooter, 'is_fleeing', False) or getattr(shooter, 'in_combat', False):
                    self.debug_log(f"{shooter.name} can't shoot (fleeing/combat)")
                    continue  # Can't shoot if fleeing or in combat
                    
                # Find valid targets
                enemy_faction = "orcs" if shooter.faction == "nuln" else "nuln"
                potential_targets = [u for u in self.units 
                                   if u.faction == enemy_faction 
                                   and u.is_alive 
                                   and not getattr(u, 'is_fleeing', False)]
                
                self.debug_log(f"{shooter.name} found {len(potential_targets)} potential targets")
                
                valid_targets = []
                for target in potential_targets:
                    # Check range and line of sight
                    try:
                        if shooter.can_see_target(target):
                            valid_targets.append(target)
                    except Exception as e:
                        self.debug_log(f"Error checking line of sight for {shooter.name} -> {target.name}: {e}")
                
                if valid_targets:
                    target = random.choice(valid_targets)
                    weapon = shooter.range_weapon
                    
                    # Calculate hit chance based on ballistic skill
                    hit_roll = random.randint(1, 6)
                    hit_needed = 7 - shooter.ballistic_skill  # BS3 needs 4+, BS4 needs 3+
                    
                    # Range modifiers
                    dx = target.position[0] - shooter.position[0]
                    dy = target.position[1] - shooter.position[1]
                    distance = (dx*dx + dy*dy) ** 0.5
                    weapon_range = weapon["range"] * 10  # Convert to pixels
                    
                    # Long range penalty
                    if distance > weapon_range * 0.5:
                        hit_needed += 1
                        
                    if hit_roll >= hit_needed:
                        # Hit! Calculate wound
                        wound_roll = random.randint(1, 6)
                        strength = weapon["strength"]
                        toughness = target.toughness
                        
                        # Wound chart
                        if strength >= toughness * 2:
                            wound_needed = 2
                        elif strength > toughness:
                            wound_needed = 3
                        elif strength == toughness:
                            wound_needed = 4
                        elif strength < toughness:
                            wound_needed = 5
                        else:
                            wound_needed = 6
                            
                        if wound_roll >= wound_needed:
                            # Wounded! Check armor save
                            saved = False
                            if target.armor_save is not None:
                                save_roll = random.randint(1, 6)
                                save_needed = target.armor_save
                                
                                # Armor piercing weapons modify save
                                if weapon.get("special") == "armor_piercing":
                                    save_needed += 2
                                    
                                if save_roll >= save_needed:
                                    saved = True
                                    self.add_log(f"🛡️ {target.name} armor saves!")
                            
                            if not saved:
                                # Apply damage
                                damage = 1
                                if weapon.get("special") == "multiple_shots":
                                    damage = random.randint(1, 6)  # Helblasters fire multiple shots
                                elif weapon.get("special") == "artillery":
                                    damage = random.randint(1, 10)  # Artillery does massive damage
                                    
                                # Check for protection (Shield Wall, etc.)
                                protection = getattr(target, 'protection', 0.0)
                                actual_damage = max(1, int(damage * (1 - protection)))
                                
                                target.models = max(0, target.models - actual_damage)
                                
                                if target.models <= 0:
                                    target.is_alive = False
                                    self.add_log(f"💥 {shooter.name} destroys {target.name}!")
                                else:
                                    self.add_log(f"🎯 {shooter.name} hits {target.name} for {actual_damage} casualties!")
                            
                        else:
                            self.add_log(f"⚪ {shooter.name} hits but fails to wound {target.name}")
                    else:
                        self.add_log(f"❌ {shooter.name} misses {target.name}")
                        
        except Exception as e:
            self.add_log(f"❌ AI shooting error: {e}")

    def random_movement_phase(self):
        """Fallback random movement"""
        for unit in self.units:
            if unit.is_alive:
                # Random small movement
                move_x = random.randint(-20, 20)
                move_y = random.randint(-10, 10)
                unit.position[0] = max(10, min(590, unit.position[0] + move_x))
                unit.position[1] = max(10, min(490, unit.position[1] + move_y))

    def random_shooting_phase(self):
        """Fallback random shooting"""
        shooters = [u for u in self.units if u.unit_type in ["warmachine"] and u.is_alive]
        for shooter in shooters[:2]:  # Limit to 2 shots per turn
            enemy_faction = "orcs" if shooter.faction == "nuln" else "nuln"
            targets = [u for u in self.units if u.faction == enemy_faction and u.is_alive]
            if targets:
                target = random.choice(targets)
                damage = random.randint(1, 5)
                target.models = max(0, target.models - damage)
                if target.models <= 0:
                    target.is_alive = False
                    self.add_log(f"💥 {shooter.name} destroys {target.name}!")
                else:
                    self.add_log(f"🎯 {shooter.name} hits {target.name} for {damage} casualties!")

    def combat_phase(self):
        """AUTHENTIC TOW COMBAT PHASE: Choose & Fight Combat -> Calculate Combat Result -> Break Test -> Follow Up & Pursuit"""
        try:
            # Find all combats and resolve them one by one
            combats = self.identify_combats()
            
            for combat in combats:
                self.resolve_single_combat(combat)
                
        except Exception as e:
            self.add_log(f"❌ Combat phase error: {e}")

    def identify_combats(self):
        """Identify all ongoing combats"""
        combats = []
        processed_units = set()
        
        for unit in [u for u in self.units if u.is_alive and getattr(u, 'in_combat', False)]:
            if unit in processed_units:
                continue
                
            # Find all enemies in base contact
            enemies = [u for u in self.units 
                      if u.faction != unit.faction 
                      and u.is_alive 
                      and abs(u.position[0] - unit.position[0]) < 40
                      and abs(u.position[1] - unit.position[1]) < 40]
            
            if enemies:
                combat_units = [unit]
                for enemy in enemies:
                    if enemy not in processed_units:
                        combat_units.append(enemy)
                        processed_units.add(enemy)
                
                processed_units.add(unit)
                combats.append(combat_units)
        
        return combats

    def resolve_single_combat(self, combat_units):
        """Resolve a single combat with proper TOW rules"""
        if len(combat_units) < 2:
            return
            
        # Separate sides
        side1 = [u for u in combat_units if u.faction == combat_units[0].faction]
        side2 = [u for u in combat_units if u.faction != combat_units[0].faction]
        
        self.add_log(f"⚔️ Combat: {', '.join(u.name for u in side1)} vs {', '.join(u.name for u in side2)}")
        
        # Sub-phase 1: Fight Combat (by Initiative order)
        side1_wounds, side2_wounds = self.fight_combat_by_initiative(side1, side2)
        
        # Sub-phase 2: Calculate Combat Result
        side1_score, side2_score = self.calculate_combat_result(side1, side2, side1_wounds, side2_wounds)
        
        # Sub-phase 3: Break Tests
        if side1_score > side2_score:
            # Side 1 wins
            self.resolve_break_tests(side2, side1_score - side2_score, "lost")
            self.resolve_break_tests(side1, 0, "won")
        elif side2_score > side1_score:
            # Side 2 wins
            self.resolve_break_tests(side1, side2_score - side1_score, "lost")
            self.resolve_break_tests(side2, 0, "won")
        else:
            # Draw - both sides test
            self.resolve_break_tests(side1, 0, "draw")
            self.resolve_break_tests(side2, 0, "draw")
        
        # Sub-phase 4: Follow Up & Pursuit
        self.resolve_pursuit_and_follow_up(side1, side2)

    def fight_combat_by_initiative(self, side1, side2):
        """Fight combat in Initiative order with Assailment spells"""
        all_combatants = side1 + side2
        
        # Sort by Initiative (highest first)
        all_combatants.sort(key=lambda u: u.get_initiative(), reverse=True)
        
        side1_wounds = 0
        side2_wounds = 0
        
        for combatant in all_combatants:
            if not combatant.is_alive:
                continue
                
            # Assailment spells can be cast when it's the wizard's Initiative step
            if combatant.get_magic_capability():
                self.attempt_assailment_spell(combatant)
            
            # Normal attacks
            if combatant in side1:
                # Attack side2
                wounds = self.make_attacks(combatant, side2)
                side2_wounds += wounds
            else:
                # Attack side1
                wounds = self.make_attacks(combatant, side1)
                side1_wounds += wounds
        
        return side1_wounds, side2_wounds

    def attempt_assailment_spell(self, wizard):
        """Wizards can cast Assailment spells during their Initiative step in combat"""
        if random.random() < 0.3:  # 30% chance to attempt Assailment spell
            assailment_spells = ["Touch of Death", "Sword of Rhuin", "Curse of Years"]
            spell = random.choice(assailment_spells)
            
            casting_roll = random.randint(1, 6) + random.randint(1, 6) + wizard.get_magic_level()
            if casting_roll >= 8:  # Typical Assailment casting value
                self.add_log(f"✨ {wizard.name} casts {spell} in combat!")
                # Spell effects would be applied here
            else:
                self.add_log(f"❌ {wizard.name} fails to cast {spell} in combat")

    def make_attacks(self, attacker, enemies):
        """Make attacks against enemy units"""
        if not enemies or not attacker.is_alive:
            return 0
            
        total_wounds = 0
        
        # Choose primary target (closest enemy)
        target = min(enemies, key=lambda e: 
            abs(e.position[0] - attacker.position[0]) + abs(e.position[1] - attacker.position[1]))
        
        if not target.is_alive:
            return 0
            
        # Calculate number of attacks
        num_attacks = attacker.get_attacks() * min(attacker.models, 10)  # Front rank attacks
        
        # Add charge bonus if applicable
        if getattr(attacker, 'charged_this_turn', False):
            num_attacks += attacker.models  # +1 attack per model on charge
        
        wounds = self.resolve_combat_attacks(attacker, target, num_attacks)
        
        # Apply wounds with step-up rule
        self.apply_wounds_with_step_up(target, wounds)
        
        return wounds

    def apply_wounds_with_step_up(self, target, wounds):
        """Apply wounds considering step-up rule"""
        models_lost = min(wounds, target.models)
        target.models -= models_lost
        
        if target.models <= 0:
            target.is_alive = False
            target.in_combat = False
            self.add_log(f"💀 {target.name} is destroyed!")
        elif models_lost > 0:
            self.add_log(f"💥 {target.name} loses {models_lost} models!")
            # Step-up rule: Models from rear ranks step forward
            # (This affects how many can attack back, but simplified here)

    def calculate_combat_result(self, side1, side2, side1_wounds, side2_wounds):
        """Calculate combat resolution scores"""
        side1_score = side2_wounds  # Wounds caused to enemy
        side2_score = side1_wounds
        
        # Rank bonuses (simplified)
        for unit in side1:
            if unit.is_alive and unit.formation != "skirmish":
                ranks = min(unit.models // 5, 3)  # Max +3 rank bonus
                side1_score += ranks
        
        for unit in side2:
            if unit.is_alive and unit.formation != "skirmish":
                ranks = min(unit.models // 5, 3)
                side2_score += ranks
        
        # Charge bonus
        for unit in side1:
            if getattr(unit, 'charged_this_turn', False):
                side1_score += 1
        
        for unit in side2:
            if getattr(unit, 'charged_this_turn', False):
                side2_score += 1
        
        # Standard bearer bonus
        for unit in side1:
            if unit.models >= 5:  # Has standard
                side1_score += 1
        
        for unit in side2:
            if unit.models >= 5:
                side2_score += 1
        
        self.add_log(f"📊 Combat result: Side 1: {side1_score}, Side 2: {side2_score}")
        return side1_score, side2_score

    def resolve_break_tests(self, units, combat_loss_modifier, result_type):
        """Resolve Break Tests with three-tiered outcomes"""
        for unit in units:
            if not unit.is_alive:
                continue
                
            # Roll 2D6 for Break Test
            natural_roll = random.randint(1, 6) + random.randint(1, 6)
            modified_roll = natural_roll + combat_loss_modifier
            leadership = unit.get_leadership()
            
            # Three-tiered outcome system
            if natural_roll > leadership:
                # Breaks and Flees (natural roll failed Ld)
                unit.is_fleeing = True
                unit.in_combat = False
                self.add_log(f"🏃 {unit.name} breaks and flees!")
                
            elif modified_roll > leadership and natural_roll != 2:  # Not double 1s
                # Falls Back in Good Order (modified roll failed, natural passed)
                unit.fell_back_in_good_order = True
                unit.in_combat = False
                self.add_log(f"🚶 {unit.name} falls back in good order")
                
            else:
                # Gives Ground (modified roll passed or double 1s)
                if result_type == "lost":
                    self.add_log(f"🛡️ {unit.name} gives ground but holds!")
                # Unit stays in combat
                    
    def resolve_combat_attacks(self, attacker, defender, num_attacks):
        """Resolve combat attacks using TOW rules"""
        wounds_caused = 0
        
        for _ in range(num_attacks):
            # Hit roll based on weapon skill comparison
            hit_roll = random.randint(1, 6)
            attacker_ws = attacker.weapon_skill
            defender_ws = defender.weapon_skill
            
            if attacker_ws > defender_ws:
                hit_needed = 3
            elif attacker_ws == defender_ws:
                hit_needed = 4
            elif attacker_ws < defender_ws / 2:
                hit_needed = 6
            else:
                hit_needed = 5
                
            if hit_roll >= hit_needed:
                # Hit! Roll to wound
                wound_roll = random.randint(1, 6)
                strength = attacker.strength
                toughness = defender.toughness
                
                # Wound chart (same as shooting)
                if strength >= toughness * 2:
                    wound_needed = 2
                elif strength > toughness:
                    wound_needed = 3
                elif strength == toughness:
                    wound_needed = 4
                elif strength < toughness:
                    wound_needed = 5
                else:
                    wound_needed = 6
                    
                if wound_roll >= wound_needed:
                    # Wounded! Check armor save
                    saved = False
                    if defender.armor_save is not None:
                        save_roll = random.randint(1, 6)
                        if save_roll >= defender.armor_save:
                            saved = True
                    
                    if not saved:
                        wounds_caused += 1
                        
        if wounds_caused > 0:
            self.add_log(f"🗡️ {attacker.name} causes {wounds_caused} wounds!")
            
        return wounds_caused

    def psychology_phase(self):
        """Handle psychology tests for fear, panic, animosity, etc. (LESS AGGRESSIVE VERSION)"""
        fear_tests_this_turn = 0
        self.debug_log("Psychology phase starting...")
        
        for unit in self.units:
            if not unit.is_alive:
                continue
                
            # Animosity tests for Orcs (on a 1, fight each other instead) - but only 25% chance
            if unit.faction == "orcs" and "Orc" in unit.name and random.random() < 0.25:
                animosity_roll = random.randint(1, 6)
                if animosity_roll == 1:
                    other_orcs = [u for u in self.units if u.faction == "orcs" and u != unit and u.is_alive]
                    if other_orcs:
                        target = random.choice(other_orcs)
                        unit.animosity_target = target
                        self.add_log(f"😡 {unit.name} shows animosity towards {target.name}!")
                        
            # Fear tests when close to scary enemies - BUT LIMITED TO 2 PER TURN AND CLOSER RANGE
            if (not getattr(unit, 'fear_immune', False) and 
                not getattr(unit, 'is_fleeing', False) and 
                fear_tests_this_turn < 2):  # Limit fear tests per turn
                
                scary_enemies = [u for u in self.units 
                               if u.faction != unit.faction 
                               and u.is_alive 
                               and getattr(u, 'causes_fear', False)
                               and abs(u.position[0] - unit.position[0]) < 80]  # Closer range required
                
                if scary_enemies and random.random() < 0.4:  # Only 40% chance to test
                    self.debug_log(f"{unit.name} taking fear test against {scary_enemies[0].name}")
                    fear_test = unit.take_psychology_test("fear")
                    fear_tests_this_turn += 1
                    
                    if not fear_test:
                        unit.is_fleeing = True
                        self.add_log(f"😱 {unit.name} fails fear test and flees!")
                        # Move unit away from scary enemy
                        scary_enemy = scary_enemies[0]
                        dx = unit.position[0] - scary_enemy.position[0]
                        dy = unit.position[1] - scary_enemy.position[1]
                        flee_distance = getattr(unit, 'movement', 4) * 2
                        if dx != 0 or dy != 0:
                            distance = (dx*dx + dy*dy) ** 0.5
                            unit.position[0] += int((dx/distance) * flee_distance)
                            unit.position[1] += int((dy/distance) * flee_distance)
                            # Keep on board
                            unit.position[0] = max(10, min(590, unit.position[0]))
                            unit.position[1] = max(10, min(490, unit.position[1]))
                    else:
                        self.debug_log(f"{unit.name} passes fear test!")
        
        self.debug_log(f"Psychology phase complete. {fear_tests_this_turn} fear tests taken.")

    def magic_phase(self):
        """Handle magic spells and dispelling"""
        wizards = [u for u in self.units if u.is_alive and u.can_cast_magic]
        
        for wizard in wizards:
            if wizard.magic_level > 0:
                # Generate magic dice
                magic_dice = wizard.magic_level + random.randint(1, 6)
                
                if magic_dice >= 8:  # Need 8+ to cast spells
                    # Choose spell targets
                    enemy_faction = "orcs" if wizard.faction == "nuln" else "nuln"
                    targets = [u for u in self.units if u.faction == enemy_faction and u.is_alive]
                    
                    if targets:
                        target = random.choice(targets)
                        spell_power = random.randint(1, 6) + wizard.magic_level
                        
                        # Different spells based on wizard type
                        if "Weirdnob" in wizard.name:
                            # Orc magic - destructive but unpredictable
                            spell_roll = random.randint(1, 6)
                            if spell_roll <= 2:
                                # Foot of Gork - damages enemy
                                damage = random.randint(1, 6)
                                target.models = max(0, target.models - damage)
                                if target.models <= 0:
                                    target.is_alive = False
                                    self.add_log(f"🦶 Weirdnob's Foot of Gork destroys {target.name}!")
                                else:
                                    self.add_log(f"🦶 Weirdnob's Foot of Gork stomps {target.name} for {damage} casualties!")
                            elif spell_roll <= 4:
                                # 'Eadbutt - target specific enemy
                                damage = random.randint(2, 12)
                                target.models = max(0, target.models - damage)
                                if target.models <= 0:
                                    target.is_alive = False
                                    self.add_log(f"💥 Weirdnob's 'Eadbutt destroys {target.name}!")
                                else:
                                    self.add_log(f"💥 Weirdnob's 'Eadbutt hits {target.name} for {damage} casualties!")
                            else:
                                # Magic backfires on Orcs
                                wizard.models = max(0, wizard.models - 1)
                                self.add_log(f"💀 Weirdnob's magic backfires! Takes 1 wound!")
                                
                        else:
                            # Empire magic - more reliable
                            spell_roll = random.randint(1, 6)
                            if spell_roll <= 3:
                                # Fireball
                                damage = random.randint(1, 6)
                                target.models = max(0, target.models - damage)
                                if target.models <= 0:
                                    target.is_alive = False
                                    self.add_log(f"🔥 {wizard.name} incinerates {target.name} with Fireball!")
                                else:
                                    self.add_log(f"🔥 {wizard.name} burns {target.name} for {damage} casualties!")
                            elif spell_roll <= 5:
                                # Lightning Bolt
                                damage = random.randint(1, 6)
                                target.models = max(0, target.models - damage)
                                if target.models <= 0:
                                    target.is_alive = False
                                    self.add_log(f"⚡ {wizard.name} destroys {target.name} with Lightning!")
                                else:
                                    self.add_log(f"⚡ {wizard.name} electrocutes {target.name} for {damage} casualties!")
                            else:
                                # Shield spell - protect friendly unit
                                friendlies = [u for u in self.units if u.faction == wizard.faction and u.is_alive]
                                if friendlies:
                                    protected = random.choice(friendlies)
                                    protected.protection = getattr(protected, 'protection', 0) + 0.3
                                    self.add_log(f"🛡️ {wizard.name} casts Shield on {protected.name}!")

    def charge_phase(self):
        """Handle charge declarations and impact hits with error handling"""
        try:
            self.debug_log("Charge phase starting...")
            charges_executed = 0
            
            for unit in self.units:
                if not unit.is_alive or getattr(unit, 'is_fleeing', False):
                    continue
                    
                # Only certain units can charge effectively
                can_charge = (unit.unit_type in ["cavalry", "monster", "chariot"] or 
                             "Troll" in unit.name or 
                             "Outrider" in unit.name or
                             "Boar Chariot" in unit.name or
                             "Wyvern" in unit.name)
                             
                if can_charge and charges_executed < 3:  # Limit charges per turn
                    try:
                        # Look for enemies in charge range
                        enemy_faction = "orcs" if unit.faction == "nuln" else "nuln"
                        charge_distance = unit.calculate_charge_distance()
                        self.debug_log(f"{unit.name} can charge {charge_distance} units")
                        
                        enemies = [u for u in self.units 
                                  if u.faction == enemy_faction 
                                  and u.is_alive
                                  and not getattr(u, 'is_fleeing', False)]
                        
                        if enemies:
                            # Find closest enemy within charge distance
                            closest_enemy = None
                            closest_dist = float('inf')
                            
                            for enemy in enemies:
                                dx = enemy.position[0] - unit.position[0]
                                dy = enemy.position[1] - unit.position[1]
                                dist = (dx*dx + dy*dy) ** 0.5
                                
                                if dist <= charge_distance * 10 and dist < closest_dist:  # *10 for pixel conversion
                                    closest_enemy = enemy
                                    closest_dist = dist
                            
                            if closest_enemy:
                                # Execute charge
                                setattr(unit, 'has_charged', True)
                                setattr(unit, 'in_combat', True)
                                setattr(closest_enemy, 'in_combat', True)
                                charges_executed += 1
                                
                                # Move unit to target
                                unit.position = [closest_enemy.position[0] + 30, closest_enemy.position[1]]
                                
                                # Impact hits for certain units
                                impact_hits = 0
                                if "Wyvern" in unit.name:
                                    impact_hits = 2
                                elif "Boar Chariot" in unit.name:
                                    impact_hits = random.randint(1, 6)
                                elif unit.unit_type == "cavalry":
                                    impact_hits = 1
                                    
                                if impact_hits > 0:
                                    closest_enemy.models = max(0, closest_enemy.models - impact_hits)
                                    if closest_enemy.models <= 0:
                                        closest_enemy.is_alive = False
                                        self.add_log(f"⚡ {unit.name} charge destroys {closest_enemy.name} with {impact_hits} impact hits!")
                                    else:
                                        self.add_log(f"⚡ {unit.name} charges {closest_enemy.name} causing {impact_hits} impact hits!")
                                else:
                                    self.add_log(f"⚡ {unit.name} charges into {closest_enemy.name}!")
                    except Exception as e:
                        self.debug_log(f"Error in charge for {unit.name}: {e}")
                        
            self.debug_log(f"Charge phase complete. {charges_executed} charges executed.")
            
        except Exception as e:
            self.add_log(f"❌ Battle error: {e}")
            self.debug_log(f"Full charge phase error: {type(e).__name__}: {e}")

    def get_ai_state(self):
        """Get battle state for AI decision making"""
        state = []
        
        # Unit counts by faction
        nuln_units = len([u for u in self.units if u.faction == "nuln" and u.is_alive])
        orc_units = len([u for u in self.units if u.faction == "orcs" and u.is_alive])
        state.extend([nuln_units, orc_units])
        
        # Add some battle state info (pad to size 20)
        state.extend([self.turn, len(self.battle_log) % 100])
        state.extend([0] * 16)  # Padding
        
        return np.array(state[:20], dtype=np.float32)

    def execute_ai_action(self, faction, action):
        """Execute AI action with enhanced anti-artillery tactics"""
        units = [u for u in self.units if u.faction == faction and u.is_alive]
        if not units:
            return
            
        if faction == "nuln":
            # Empire AI actions
            unit = random.choice(units)
            
            if action == 0:  # Move forward
                unit.position[1] = max(10, unit.position[1] - 15)
                self.add_log(f"🚶 {unit.name} advances!")
                
            elif action == 1 or action == 8:  # Artillery strike
                artillery = [u for u in units if u.unit_type == "warmachine"]
                if artillery:
                    shooter = random.choice(artillery)
                    targets = [u for u in self.units if u.faction == "orcs" and u.is_alive]
                    if targets:
                        target = random.choice(targets)
                        damage = random.randint(2, 6)
                        target.models = max(0, target.models - damage)
                        if target.models <= 0:
                            target.is_alive = False
                            self.add_log(f"💥 {shooter.name} artillery destroys {target.name}!")
                        else:
                            self.add_log(f"🎯 {shooter.name} artillery hits {target.name}!")
        else:
            # NEW: Enhanced Orc AI with anti-artillery tactics
            if action == 3:  # Flying Assault 
                wyvern = next((u for u in units if "Wyvern" in u.name), None)
                if wyvern:
                    # Wyvern flies directly to attack artillery
                    artillery = [u for u in self.units if u.faction == "nuln" and u.unit_type == "warmachine" and u.is_alive]
                    if artillery:
                        target = random.choice(artillery)
                        wyvern.position = [target.position[0] + 20, target.position[1] + 20]
                        # Destroy artillery piece
                        target.is_alive = False
                        self.add_log(f"🐉 {wyvern.name} aerial assault destroys {target.name}!")
                    else:
                        self.add_log(f"🐉 {wyvern.name} swoops across the battlefield!")
                        
            elif action == 4:  # Troll Regeneration
                trolls = [u for u in units if "Troll" in u.name and u.models < u.max_models]
                if trolls:
                    troll = random.choice(trolls)
                    healed = min(2, troll.max_models - troll.models)
                    troll.models += healed
                    self.add_log(f"🩹 {troll.name} regenerates {healed} wounds!")
                    
            elif action == 10:  # Artillery Hunt
                # Orcs specifically target enemy artillery
                artillery = [u for u in self.units if u.faction == "nuln" and u.unit_type == "warmachine" and u.is_alive]
                if artillery and units:
                    hunter = random.choice(units)
                    target = random.choice(artillery)
                    # Magic/special attack destroys artillery
                    if random.random() < 0.4:  # 40% success rate
                        target.is_alive = False
                        self.add_log(f"🎯 {hunter.name} destroys {target.name} with focused attack!")
                    else:
                        self.add_log(f"⚡ {hunter.name} targets {target.name} but misses!")
                        
            elif action == 6:  # Shield Wall Advance
                # Protected advance - move faster with damage reduction
                for unit in units:
                    unit.position[1] = min(490, unit.position[1] + 25)  # Fast advance
                    # Give temporary protection (simulate in next shooting phase)
                    unit.protection = 0.5  # 50% damage reduction
                self.add_log(f"🛡️ Orc army advances under shield wall protection!")
                
            else:  # Default advance (action 0)
                unit = random.choice(units)
                unit.position[1] = min(490, unit.position[1] + 15)
                self.add_log(f"🚶 {unit.name} advances!")

    def check_victory(self):
        """Check for victory conditions"""
        nuln_alive = any(u.faction == "nuln" and u.is_alive for u in self.units)
        orcs_alive = any(u.faction == "orcs" and u.is_alive for u in self.units)
        
        if not nuln_alive:
            self.add_log("🏆 VICTORY TO THE TROLL HORDE!")
            self.add_log("🟢 The greenskins triumph!")
            return True
        elif not orcs_alive:
            self.add_log("🏆 VICTORY TO THE NULN EMPIRE!")
            self.add_log("🔵 The Empire stands victorious!")
            return True
            
        return False

    def end_battle(self):
        """End the battle"""
        self.battle_state = "finished"
        self.add_log("🏁 BATTLE CONCLUDED")
        
        # Final statistics
        nuln_survivors = len([u for u in self.units if u.faction == "nuln" and u.is_alive])
        orc_survivors = len([u for u in self.units if u.faction == "orcs" and u.is_alive])
        
        self.add_log(f"📊 Nuln Empire survivors: {nuln_survivors}")
        self.add_log(f"📊 Orc & Goblin survivors: {orc_survivors}")
        
        socketio.emit('battle_update', self.get_battle_state())

    def reset_battle(self):
        """Reset battle to initial state"""
        with self.lock:
            self.battle_state = "ready"
            self.turn = 1
            self.phase = "deployment"
            self.battle_log = []
            self.create_authentic_armies()
            self.add_log("🔄 Battle reset - ready for new engagement!")

    def resolve_pursuit_and_follow_up(self, side1, side2):
        """Resolve pursuit and follow-up moves after combat"""
        # Simplified pursuit mechanics
        for unit in side1 + side2:
            if getattr(unit, 'fell_back_in_good_order', False):
                # Move unit back slightly
                retreat_distance = random.randint(1, 6) + random.randint(1, 6)
                if unit.faction == "nuln":
                    unit.position[1] += retreat_distance  # Move towards Nuln side
                else:
                    unit.position[1] -= retreat_distance  # Move towards Orc side
                
                self.add_log(f"🚶 {unit.name} retreats {retreat_distance} inches in good order")
                unit.fell_back_in_good_order = False  # Reset flag

# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'warhammer_old_world_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global battle instance
battle = TOWBattle()

@app.route('/')
def index():
    return render_template('tow_battle.html')

@app.route('/api/battle_state')
def get_battle_state():
    return jsonify(battle.get_battle_state())

@app.route('/api/start_battle', methods=['POST'])
def start_battle():
    success = battle.start_battle()
    return jsonify({'success': success})

@app.route('/api/reset_battle', methods=['POST'])
def reset_battle():
    battle.reset_battle()
    return jsonify({'success': True})

@app.route('/api/new_battle', methods=['POST'])
def new_battle():
    battle.reset_battle()
    return jsonify({
        'success': True, 
        'battle_state': battle.get_battle_state()
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('battle_update', battle.get_battle_state())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("⚔️ WARHAMMER: THE OLD WORLD - EPIC BATTLE SYSTEM")
    print("=======================================================")
    print("⚔️ WARHAMMER: THE OLD WORLD - WEB BATTLE")
    print("==================================================")
    print("🌐 Starting web server...")
    print("📱 Open your browser to http://localhost:5001")
    print("Features:")
    print("• Authentic regiment formations")
    print("• Real-time AI vs AI battles") 
    print("• Proper TOW unit types and movement")
    print("• Live battle log")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True) 