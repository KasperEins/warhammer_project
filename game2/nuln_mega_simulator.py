#!/usr/bin/env python3
"""
🔥 ARMY OF NULN MEGA SIMULATOR - 300,000 BATTLES
Test Army of Nuln against all enemy types with faction bonuses
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

@dataclass
class NulnUnit:
    name: str
    category: str  # character, core, special, rare
    base_cost: int
    cost_per_model: int
    min_size: int
    max_size: int
    effectiveness: float
    battlefield_role: str
    special_rules: List[str]
    upgrades: Dict[str, int]
    nuln_bonus: float  # Faction-specific bonus multiplier

@dataclass
class NulnArmyComposition:
    units: List[Tuple[str, int, List[str]]]  # (unit_name, size, upgrades)
    total_points: int
    effectiveness_score: float
    synergy_bonus: float
    nuln_engineering_bonus: float  # Special Nuln bonus

@dataclass
class EnemyArmy:
    name: str
    faction: str
    units: List[Dict]
    total_points: int
    strengths: List[str]
    weaknesses: List[str]
    difficulty: float

@dataclass
class BattleResult:
    nuln_army: str
    enemy_army: str
    nuln_victory: bool
    victory_margin: float
    battle_duration: int
    nuln_casualties: float
    enemy_casualties: float

class ArmyOfNulnSimulator:
    """Army of Nuln specialized simulator with faction bonuses"""
    
    def __init__(self):
        self.nuln_units = self._create_nuln_database()
        self.enemy_armies = self._create_enemy_database()
        
    def _create_nuln_database(self) -> Dict[str, NulnUnit]:
        """Create Army of Nuln specialized unit database"""
        units = {}
        
        # CHARACTERS - Enhanced engineering leaders
        units["Nuln General"] = NulnUnit(
            name="Nuln General",
            category="character",
            base_cost=80,
            cost_per_model=0,
            min_size=1,
            max_size=1,
            effectiveness=8.5,
            battlefield_role="Leadership",
            special_rules=["Master Engineer", "Gunpowder Expertise"],
            upgrades={
                "Heavy Armour": 4,
                "Shield": 2,
                "Warhorse": 18,
                "Imperial Banner": 25,
                "Engineering Tools": 15
            },
            nuln_bonus=1.2  # 20% bonus for engineering leadership
        )
        
        units["Master Engineer"] = NulnUnit(
            name="Master Engineer",
            category="character",
            base_cost=65,
            cost_per_model=0,
            min_size=1,
            max_size=1,
            effectiveness=7.5,
            battlefield_role="Artillery Support",
            special_rules=["Artillery Expert", "Repair", "Engineering Genius"],
            upgrades={
                "Hochland Long Rifle": 20,
                "Repeater Handgun": 15,
                "Engineering Kit": 10
            },
            nuln_bonus=1.4  # 40% bonus for artillery support
        )
        
        units["Nuln Wizard"] = NulnUnit(
            name="Nuln Wizard",
            category="character",
            base_cost=45,
            cost_per_model=0,
            min_size=1,
            max_size=1,
            effectiveness=6.8,
            battlefield_role="Magic",
            special_rules=["Metal Magic", "Engineering Affinity"],
            upgrades={
                "Level 2": 35,
                "Dispel Scroll": 25,
                "Power Stone": 15
            },
            nuln_bonus=1.1  # 10% bonus for metal magic
        )
        
        # CORE UNITS - Infantry backbone
        units["Nuln Spearmen"] = NulnUnit(
            name="Nuln Spearmen",
            category="core",
            base_cost=0,
            cost_per_model=4,
            min_size=10,
            max_size=30,
            effectiveness=5.5,
            battlefield_role="Infantry Block",
            special_rules=["Disciplined", "City Watch"],
            upgrades={
                "Light Armour": 1,
                "Shields": 1,
                "Champion": 10,
                "Standard Bearer": 10,
                "Musician": 5
            },
            nuln_bonus=1.0  # Standard bonus
        )
        
        units["Nuln Handgunners"] = NulnUnit(
            name="Nuln Handgunners",
            category="core",
            base_cost=0,
            cost_per_model=9,
            min_size=10,
            max_size=20,
            effectiveness=7.2,
            battlefield_role="Ranged",
            special_rules=["Gunpowder Weapons", "Armor Piercing", "Nuln Craftsmanship"],
            upgrades={
                "Champion": 10,
                "Musician": 5,
                "Marksman's Kit": 15
            },
            nuln_bonus=1.3  # 30% bonus for superior gunpowder
        )
        
        units["Nuln Crossbowmen"] = NulnUnit(
            name="Nuln Crossbowmen", 
            category="core",
            base_cost=0,
            cost_per_model=8,
            min_size=10,
            max_size=20,
            effectiveness=6.8,
            battlefield_role="Ranged",
            special_rules=["Crossbows", "Move and Shoot"],
            upgrades={
                "Champion": 10,
                "Pavise": 2,
                "Musician": 5
            },
            nuln_bonus=1.15  # 15% bonus for quality
        )
        
        # SPECIAL UNITS - Nuln's pride
        units["Nuln Ironsides"] = NulnUnit(
            name="Nuln Ironsides",
            category="special",
            base_cost=0,
            cost_per_model=12,
            min_size=8,
            max_size=16,
            effectiveness=8.5,
            battlefield_role="Elite Infantry",
            special_rules=["Heavy Armour", "Handguns", "Veteran", "Nuln Elite"],
            upgrades={
                "Champion": 12,
                "Standard Bearer": 12,
                "Musician": 6,
                "Superior Gunpowder": 20
            },
            nuln_bonus=1.4  # 40% bonus for elite gunners
        )
        
        units["Outriders"] = NulnUnit(
            name="Outriders",
            category="special",
            base_cost=0,
            cost_per_model=18,
            min_size=3,
            max_size=10,
            effectiveness=7.8,
            battlefield_role="Fast Cavalry",
            special_rules=["Fast Cavalry", "Repeater Handguns", "Scout"],
            upgrades={
                "Champion": 18,
                "Musician": 9,
                "Grenade Launching Blunderbuss": 25
            },
            nuln_bonus=1.25  # 25% bonus for mobility + firepower
        )
        
        units["Engineers"] = NulnUnit(
            name="Engineers",
            category="special",
            base_cost=0,
            cost_per_model=8,
            min_size=5,
            max_size=10,
            effectiveness=6.5,
            battlefield_role="Support",
            special_rules=["Engineer", "Repair", "Hochland Long Rifles"],
            upgrades={
                "Champion": 10,
                "Blasting Charges": 15,
                "Superior Tools": 20
            },
            nuln_bonus=1.3  # 30% bonus for engineering expertise
        )
        
        # RARE UNITS - Artillery supremacy
        units["Great Cannon"] = NulnUnit(
            name="Great Cannon",
            category="rare",
            base_cost=100,
            cost_per_model=0,
            min_size=1,
            max_size=1,
            effectiveness=9.2,
            battlefield_role="Artillery",
            special_rules=["Artillery", "Armor Piercing", "Nuln Forged"],
            upgrades={
                "Extra Crew": 10,
                "Superior Gunpowder": 15
            },
            nuln_bonus=1.5  # 50% bonus for Nuln cannons
        )
        
        units["Helblaster Volley Gun"] = NulnUnit(
            name="Helblaster Volley Gun",
            category="rare",
            base_cost=110,
            cost_per_model=0,
            min_size=1,
            max_size=1,
            effectiveness=8.8,
            battlefield_role="Artillery",
            special_rules=["Artillery", "Multiple Shots", "Nuln Innovation"],
            upgrades={
                "Extra Crew": 10,
                "Improved Mechanisms": 20
            },
            nuln_bonus=1.6  # 60% bonus for advanced engineering
        )
        
        units["Mortar"] = NulnUnit(
            name="Mortar",
            category="rare",
            base_cost=75,
            cost_per_model=0,
            min_size=1,
            max_size=1,
            effectiveness=8.0,
            battlefield_role="Artillery",
            special_rules=["Artillery", "Indirect Fire", "Nuln Expertise"],
            upgrades={
                "Extra Crew": 8,
                "Explosive Shells": 15
            },
            nuln_bonus=1.45  # 45% bonus for siege weapons
        )
        
        units["Steam Tank"] = NulnUnit(
            name="Steam Tank",
            category="rare",
            base_cost=250,
            cost_per_model=0,
            min_size=1,
            max_size=1,
            effectiveness=9.8,
            battlefield_role="Behemoth",
            special_rules=["Steam Tank", "Terror", "Steam Cannon", "Nuln Masterwork"],
            upgrades={
                "Improved Boiler": 25,
                "Reinforced Armor": 30
            },
            nuln_bonus=1.8  # 80% bonus for ultimate engineering
        )
        
        return units
    
    def _create_enemy_database(self) -> List[EnemyArmy]:
        """Create enemy army database (same as before but enhanced)"""
        enemies = []
        
        # ORCS & GOBLINS
        enemies.append(EnemyArmy(
            name="Orc Horde",
            faction="Orcs & Goblins",
            units=[
                {"name": "Orc Warboss", "count": 1, "points": 120},
                {"name": "Orc Boys", "count": 30, "points": 180},
                {"name": "Orc Boys", "count": 25, "points": 150},
                {"name": "Goblin Archers", "count": 20, "points": 100},
                {"name": "Orc Boar Boyz", "count": 5, "points": 110},
                {"name": "Rock Lobber", "count": 1, "points": 80}
            ],
            total_points=740,
            strengths=["Numbers", "Melee Combat", "Cheap Units"],
            weaknesses=["Low Leadership", "Poor Shooting", "Animosity"],
            difficulty=6.5
        ))
        
        # DWARFS
        enemies.append(EnemyArmy(
            name="Dwarf Gunline",
            faction="Dwarfs",
            units=[
                {"name": "Dwarf Lord", "count": 1, "points": 140},
                {"name": "Dwarf Warriors", "count": 20, "points": 160},
                {"name": "Dwarf Quarrellers", "count": 15, "points": 165},
                {"name": "Dwarf Thunderers", "count": 10, "points": 130},
                {"name": "Dwarf Cannon", "count": 1, "points": 120},
                {"name": "Organ Gun", "count": 1, "points": 120}
            ],
            total_points=735,
            strengths=["Heavy Armor", "Superior Shooting", "High Leadership"],
            weaknesses=["Slow Movement", "Expensive", "Low Model Count"],
            difficulty=8.5
        ))
        
        # HIGH ELVES
        enemies.append(EnemyArmy(
            name="High Elf Magic Heavy",
            faction="High Elves",
            units=[
                {"name": "Archmage", "count": 1, "points": 185},
                {"name": "Elf Mage", "count": 1, "points": 100},
                {"name": "Elf Spearmen", "count": 16, "points": 160},
                {"name": "Elf Archers", "count": 10, "points": 110},
                {"name": "Phoenix Guard", "count": 10, "points": 150},
                {"name": "Eagle Claw Bolt Thrower", "count": 1, "points": 70}
            ],
            total_points=775,
            strengths=["Devastating Magic", "Ward Saves", "High Leadership"],
            weaknesses=["Magic Dependent", "Low Model Count", "Expensive"],
            difficulty=9.5
        ))
        
        # CHAOS
        enemies.append(EnemyArmy(
            name="Chaos Warriors",
            faction="Warriors of Chaos",
            units=[
                {"name": "Chaos Lord", "count": 1, "points": 150},
                {"name": "Chaos Warriors", "count": 12, "points": 216},
                {"name": "Chaos Warriors", "count": 10, "points": 180},
                {"name": "Chaos Knights", "count": 4, "points": 160},
                {"name": "Chaos Hounds", "count": 10, "points": 60}
            ],
            total_points=766,
            strengths=["Heavy Armor", "High Toughness", "Fear"],
            weaknesses=["Very Expensive", "Low Numbers", "Slow"],
            difficulty=9.5
        ))
        
        # BRETONNIANS
        enemies.append(EnemyArmy(
            name="Bretonnian Lance",
            faction="Bretonnia",
            units=[
                {"name": "Bretonnian Lord", "count": 1, "points": 140},
                {"name": "Knights of the Realm", "count": 8, "points": 232},
                {"name": "Knights Errant", "count": 6, "points": 138},
                {"name": "Men-at-Arms", "count": 20, "points": 100},
                {"name": "Peasant Bowmen", "count": 15, "points": 90},
                {"name": "Trebuchet", "count": 1, "points": 90}
            ],
            total_points=790,
            strengths=["Heavy Cavalry", "Devastating Charge", "High Leadership"],
            weaknesses=["Expensive", "Peasant Troops", "Limited Flexibility"],
            difficulty=8.5
        ))
        
        return enemies
    
    def generate_nuln_armies(self, count: int = 25) -> List[NulnArmyComposition]:
        """Generate Army of Nuln compositions with faction bonuses"""
        armies = []
        
        # Define Army of Nuln templates
        templates = [
            # Template 1: Artillery Park
            {
                "units": [
                    ("Nuln General", 1, ["Heavy Armour", "Engineering Tools"]),
                    ("Master Engineer", 1, ["Engineering Kit"]),
                    ("Nuln Spearmen", 16, ["Light Armour", "Shields", "Champion"]),
                    ("Nuln Handgunners", 10, ["Champion"]),
                    ("Great Cannon", 1, ["Superior Gunpowder"]),
                    ("Helblaster Volley Gun", 1, ["Improved Mechanisms"]),
                    ("Mortar", 1, ["Explosive Shells"])
                ]
            },
            # Template 2: Ironsides Elite
            {
                "units": [
                    ("Nuln General", 1, ["Heavy Armour", "Imperial Banner"]),
                    ("Nuln Spearmen", 20, ["Light Armour", "Shields", "Champion", "Standard Bearer"]),
                    ("Nuln Ironsides", 12, ["Champion", "Standard Bearer", "Superior Gunpowder"]),
                    ("Nuln Handgunners", 10, ["Champion", "Marksman's Kit"]),
                    ("Great Cannon", 1, ["Extra Crew"])
                ]
            },
            # Template 3: Mobile Gunline
            {
                "units": [
                    ("Nuln General", 1, ["Heavy Armour", "Warhorse"]),
                    ("Master Engineer", 1, ["Hochland Long Rifle"]),
                    ("Nuln Handgunners", 15, ["Champion", "Musician"]),
                    ("Nuln Crossbowmen", 10, ["Champion", "Pavise"]),
                    ("Outriders", 5, ["Champion", "Musician"]),
                    ("Engineers", 6, ["Champion", "Superior Tools"]),
                    ("Helblaster Volley Gun", 1, [])
                ]
            },
            # Template 4: Steam Tank Focus
            {
                "units": [
                    ("Master Engineer", 1, ["Repeater Handgun"]),
                    ("Nuln Spearmen", 12, ["Light Armour", "Champion"]),
                    ("Nuln Handgunners", 8, ["Champion"]),
                    ("Steam Tank", 1, ["Improved Boiler", "Reinforced Armor"])
                ]
            },
            # Template 5: Engineering Corps
            {
                "units": [
                    ("Nuln General", 1, ["Heavy Armour", "Engineering Tools"]),
                    ("Master Engineer", 1, ["Engineering Kit"]),
                    ("Nuln Wizard", 1, ["Level 2", "Dispel Scroll"]),
                    ("Nuln Spearmen", 18, ["Light Armour", "Shields", "Champion"]),
                    ("Engineers", 8, ["Champion", "Blasting Charges"]),
                    ("Great Cannon", 1, ["Superior Gunpowder"])
                ]
            }
        ]
        
        for i in range(count):
            # Use templates and variations
            template = templates[i % len(templates)]
            army = self._build_nuln_army(template)
            if army and army.total_points <= 750:
                armies.append(army)
        
        return armies
    
    def _build_nuln_army(self, template: Dict) -> NulnArmyComposition:
        """Build a Nuln army from template"""
        units = []
        total_points = 0
        total_effectiveness = 0
        engineering_bonus = 0
        
        for unit_name, size, upgrades in template["units"]:
            if unit_name not in self.nuln_units:
                continue
            
            unit = self.nuln_units[unit_name]
            
            # Calculate cost
            unit_cost = unit.base_cost + (unit.cost_per_model * size)
            for upgrade in upgrades:
                if upgrade in unit.upgrades:
                    unit_cost += unit.upgrades[upgrade]
            
            total_points += unit_cost
            
            # Calculate effectiveness with Nuln bonuses
            base_effectiveness = unit.effectiveness * size * unit.nuln_bonus
            total_effectiveness += base_effectiveness
            
            # Engineering bonus for artillery and specialists
            if unit.battlefield_role in ['Artillery', 'Artillery Support']:
                engineering_bonus += unit.nuln_bonus * 2
            
            units.append((unit_name, size, upgrades))
        
        # Synergy bonuses for Army of Nuln
        synergy_bonus = 0
        artillery_count = sum(1 for unit_name, _, _ in units 
                            if self.nuln_units[unit_name].battlefield_role == 'Artillery')
        if artillery_count >= 2:
            synergy_bonus += 5  # Multiple artillery pieces
        
        gunpowder_count = sum(1 for unit_name, _, _ in units 
                            if 'Gunpowder' in self.nuln_units[unit_name].special_rules)
        if gunpowder_count >= 3:
            synergy_bonus += 3  # Gunpowder synergy
        
        return NulnArmyComposition(
            units=units,
            total_points=total_points,
            effectiveness_score=total_effectiveness / 10,
            synergy_bonus=synergy_bonus,
            nuln_engineering_bonus=engineering_bonus
        )
    
    def simulate_battle(self, nuln_army: NulnArmyComposition, enemy_army: EnemyArmy) -> BattleResult:
        """Simulate battle with Nuln faction bonuses"""
        
        # Calculate Nuln army strength with faction bonuses
        nuln_strength = self._calculate_nuln_strength(nuln_army)
        enemy_strength = self._calculate_enemy_strength(enemy_army)
        
        # Apply tactical modifiers
        nuln_modifier = self._get_nuln_tactical_modifier(nuln_army, enemy_army)
        enemy_modifier = self._get_enemy_tactical_modifier(enemy_army, nuln_army)
        
        # Randomness
        nuln_dice = random.uniform(0.7, 1.3)
        enemy_dice = random.uniform(0.7, 1.3)
        
        # Final strength calculation
        nuln_final = nuln_strength * nuln_modifier * nuln_dice
        enemy_final = enemy_strength * enemy_modifier * enemy_dice
        
        # Determine outcome
        nuln_victory = nuln_final > enemy_final
        if nuln_victory:
            victory_margin = (nuln_final - enemy_final) / enemy_final
        else:
            victory_margin = (enemy_final - nuln_final) / nuln_final
        
        return BattleResult(
            nuln_army=f"Nuln Army {id(nuln_army)}",
            enemy_army=enemy_army.name,
            nuln_victory=nuln_victory,
            victory_margin=victory_margin,
            battle_duration=random.randint(4, 8),
            nuln_casualties=self._calculate_nuln_casualties(nuln_army, enemy_army, nuln_victory),
            enemy_casualties=self._calculate_enemy_casualties(enemy_army, nuln_army, not nuln_victory)
        )
    
    def _calculate_nuln_strength(self, army: NulnArmyComposition) -> float:
        """Calculate Nuln army strength with faction bonuses"""
        total_strength = 0.0
        
        for unit_name, size, upgrades in army.units:
            unit = self.nuln_units[unit_name]
            
            # Base strength with Nuln bonus
            unit_strength = unit.effectiveness * size * unit.nuln_bonus
            
            # Upgrade bonuses
            for upgrade in upgrades:
                if upgrade in unit.upgrades:
                    unit_strength *= 1.08  # 8% per upgrade
            
            # Role-based bonuses enhanced for Nuln
            if unit.battlefield_role == 'Artillery':
                unit_strength *= 1.4  # 40% bonus for Nuln artillery supremacy
            elif unit.battlefield_role == 'Ranged':
                unit_strength *= 1.25  # 25% bonus for superior gunpowder
            elif unit.battlefield_role == 'Elite Infantry':
                unit_strength *= 1.2  # 20% bonus for professional troops
            
            total_strength += unit_strength
        
        # Add Nuln-specific bonuses
        total_strength += army.synergy_bonus * 15
        total_strength += army.nuln_engineering_bonus * 5
        
        return total_strength
    
    def _calculate_enemy_strength(self, enemy: EnemyArmy) -> float:
        """Calculate enemy strength"""
        base_strength = enemy.difficulty * 50
        for unit in enemy.units:
            base_strength += unit["points"] * 0.1
        return base_strength
    
    def _get_nuln_tactical_modifier(self, nuln_army: NulnArmyComposition, enemy: EnemyArmy) -> float:
        """Get Nuln tactical bonuses vs enemy"""
        modifier = 1.0
        
        nuln_roles = [self.nuln_units[unit[0]].battlefield_role for unit in nuln_army.units]
        
        # Artillery dominance vs all targets
        artillery_count = sum(1 for role in nuln_roles if role == 'Artillery')
        if artillery_count >= 2:
            modifier += 0.4  # Massive artillery advantage
            if "Numbers" in enemy.strengths:
                modifier += 0.3  # Extra vs hordes
            if "Heavy Armor" in enemy.strengths:
                modifier += 0.25  # Extra vs armor
        
        # Gunpowder superiority
        ranged_count = sum(1 for role in nuln_roles if role == 'Ranged')
        if ranged_count >= 2:
            modifier += 0.3  # Superior firepower
        
        # Engineering advantage
        if nuln_army.nuln_engineering_bonus > 10:
            modifier += 0.2  # Engineering mastery
        
        # Anti-magic with engineering
        if "Magic" in enemy.strengths:
            modifier += 0.15  # Engineering counters magic
        
        return max(0.6, min(2.5, modifier))
    
    def _get_enemy_tactical_modifier(self, enemy: EnemyArmy, nuln_army: NulnArmyComposition) -> float:
        """Get enemy tactical modifier vs Nuln"""
        modifier = 1.0
        
        # Reduced effectiveness vs Nuln's superior technology
        if "Superior Shooting" in enemy.strengths:
            modifier -= 0.1  # Nuln has better guns
        
        if "Heavy Armor" in enemy.strengths:
            artillery_count = sum(1 for unit_name, _, _ in nuln_army.units 
                                if self.nuln_units[unit_name].battlefield_role == 'Artillery')
            if artillery_count >= 2:
                modifier -= 0.2  # Artillery negates armor
        
        return max(0.5, min(2.0, modifier))
    
    def _calculate_nuln_casualties(self, nuln_army: NulnArmyComposition, enemy: EnemyArmy, victory: bool) -> float:
        """Calculate Nuln casualties"""
        base_casualties = 0.25 if victory else 0.65  # Lower due to superior equipment
        
        if "Numbers" in enemy.strengths:
            base_casualties += 0.08
        if "Heavy Armor" in enemy.strengths:
            base_casualties += 0.1
        
        return min(1.0, base_casualties + random.uniform(-0.08, 0.08))
    
    def _calculate_enemy_casualties(self, enemy: EnemyArmy, nuln_army: NulnArmyComposition, victory: bool) -> float:
        """Calculate enemy casualties vs Nuln"""
        base_casualties = 0.35 if victory else 0.75  # Higher due to Nuln firepower
        
        # Nuln causes heavy casualties
        artillery_count = sum(1 for unit_name, _, _ in nuln_army.units 
                            if self.nuln_units[unit_name].battlefield_role == 'Artillery')
        base_casualties += artillery_count * 0.15  # Artillery devastation
        
        return min(1.0, base_casualties + random.uniform(-0.1, 0.1))
    
    def run_nuln_mega_simulation(self, num_battles: int = 300000):
        """Run mega simulation for Army of Nuln"""
        print(f"🔥 ARMY OF NULN MEGA SIMULATOR")
        print(f"=" * 80)
        print(f"🎯 Running {num_battles:,} battles...")
        print(f"⚔️ Testing against {len(self.enemy_armies)} enemy army types")
        print(f"🔧 With full Nuln faction bonuses and restrictions!")
        print()
        
        # Generate Nuln armies
        print("🔥 Generating Army of Nuln variants...")
        nuln_armies = self.generate_nuln_armies(25)
        print(f"✅ Generated {len(nuln_armies)} Army of Nuln variants")
        print()
        
        # Run simulation
        battles_per_combo = num_battles // (len(nuln_armies) * len(self.enemy_armies))
        results = {}
        total_battles = 0
        start_time = time.time()
        
        for i, nuln_army in enumerate(nuln_armies):
            army_results = []
            
            print(f"🔥 Testing Nuln Army #{i+1}: {nuln_army.effectiveness_score:.1f} effectiveness")
            print(f"   Engineering Bonus: {nuln_army.nuln_engineering_bonus:.1f}")
            
            for enemy in self.enemy_armies:
                enemy_results = []
                
                for battle in range(battles_per_combo):
                    result = self.simulate_battle(nuln_army, enemy)
                    enemy_results.append(result)
                    total_battles += 1
                    
                    if total_battles % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = total_battles / elapsed
                        print(f"   ⚡ {total_battles:,} battles completed ({rate:.0f} battles/sec)")
                
                # Calculate stats
                victories = sum(1 for r in enemy_results if r.nuln_victory)
                win_rate = victories / len(enemy_results)
                
                army_results.append({
                    'enemy': enemy.name,
                    'battles': len(enemy_results),
                    'victories': victories,
                    'win_rate': win_rate,
                    'enemy_difficulty': enemy.difficulty
                })
            
            # Overall performance
            total_victories = sum(r['victories'] for r in army_results)
            total_army_battles = sum(r['battles'] for r in army_results)
            overall_win_rate = total_victories / total_army_battles
            
            weighted_score = sum(r['win_rate'] * r['enemy_difficulty'] for r in army_results) / sum(r['enemy_difficulty'] for r in army_results)
            
            results[f"Nuln_Army_{i+1}"] = {
                'army_composition': nuln_army,
                'enemy_results': army_results,
                'total_battles': total_army_battles,
                'total_victories': total_victories,
                'overall_win_rate': overall_win_rate,
                'weighted_score': weighted_score
            }
            
            print(f"   📈 Win Rate: {overall_win_rate:.1%} | Weighted Score: {weighted_score:.3f}")
            print()
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Nuln simulation completed in {elapsed_time:.1f} seconds")
        print(f"🎯 Total battles: {total_battles:,}")
        print(f"⚡ Average rate: {total_battles/elapsed_time:.0f} battles/second")
        
        return self._analyze_nuln_results(results)
    
    def _analyze_nuln_results(self, results: Dict):
        """Analyze Army of Nuln results"""
        print(f"\n🔥 ARMY OF NULN ANALYSIS")
        print(f"=" * 80)
        
        sorted_armies = sorted(results.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        
        print(f"🏆 TOP PERFORMING NULN ARMIES")
        print(f"-" * 50)
        
        for rank, (army_name, data) in enumerate(sorted_armies[:5], 1):
            army = data['army_composition']
            print(f"\n#{rank}. {army_name}")
            print(f"   Weighted Score: {data['weighted_score']:.3f}")
            print(f"   Overall Win Rate: {data['overall_win_rate']:.1%}")
            print(f"   Engineering Bonus: {army.nuln_engineering_bonus:.1f}")
            print(f"   Army Points: {army.total_points}")
            
            print(f"   Units:")
            for unit_name, size, upgrades in army.units:
                upgrade_str = f" ({', '.join(upgrades)})" if upgrades else ""
                print(f"     - {size}x {unit_name}{upgrade_str}")
        
        print(f"\n🎯 ENEMY MATCHUP ANALYSIS - BEST NULN ARMY")
        print(f"-" * 50)
        
        best_army = sorted_armies[0][1]
        for enemy_result in best_army['enemy_results']:
            print(f"   vs {enemy_result['enemy']}: {enemy_result['win_rate']:.1%}")
        
        return sorted_armies[0][1]['army_composition']

def main():
    """Run Army of Nuln mega simulation"""
    simulator = ArmyOfNulnSimulator()
    
    print("🔥 ARMY OF NULN MEGA SIMULATOR STARTING...")
    print("⚔️ Superior engineering meets battlefield supremacy!")
    print()
    
    best_nuln_army = simulator.run_nuln_mega_simulation(300000)
    
    print(f"\n🏆 ULTIMATE ARMY OF NULN 750PT ARMY!")
    print(f"🔥 The pinnacle of Imperial engineering warfare!")

if __name__ == '__main__':
    main() 