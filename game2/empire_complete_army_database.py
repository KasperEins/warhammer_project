#!/usr/bin/env python3
"""
🏛️ COMPLETE EMPIRE ARMY DATABASE
Exact stats, points costs, and equipment options for AI use
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class UnitCategory(Enum):
    CHARACTER = "Character"
    CORE = "Core"
    SPECIAL = "Special"
    RARE = "Rare"
    MERCENARY = "Mercenary"

class TroopType(Enum):
    INFANTRY = "Infantry"
    CAVALRY = "Cavalry"
    MONSTROUS_INFANTRY = "Monstrous Infantry"
    WAR_MACHINE = "War Machine"
    BEHEMOTH = "Behemoth"

@dataclass
class UnitStats:
    movement: int
    weapon_skill: int
    ballistic_skill: int
    strength: int
    toughness: int
    wounds: int
    initiative: int
    attacks: int
    leadership: int

@dataclass
class EquipmentOption:
    name: str
    cost: int
    description: str
    per_model: bool = False

@dataclass
class EmpireUnit:
    name: str
    category: UnitCategory
    troop_type: TroopType
    stats: UnitStats
    base_cost: int
    cost_per_model: int
    min_size: int
    max_size: int
    equipment_options: Dict[str, EquipmentOption]
    special_rules: List[str]
    default_equipment: List[str]
    restrictions: List[str] = None

def create_empire_army_database() -> Dict[str, EmpireUnit]:
    """Complete Empire army database with exact points and stats"""
    
    units = {}
    
    # ===========================================
    # CHARACTERS
    # ===========================================
    
    units["General of the Empire"] = EmpireUnit(
        name="General of the Empire",
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=6, ballistic_skill=5, strength=4,
            toughness=4, wounds=3, initiative=5, attacks=3, leadership=9
        ),
        base_cost=80,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={
            "Additional Hand Weapon": EquipmentOption("Additional Hand Weapon", 3, "+1 Attack"),
            "Great Weapon": EquipmentOption("Great Weapon", 6, "+2 Strength, Strike Last"),
            "Lance": EquipmentOption("Lance", 4, "+2 Strength on charge if mounted"),
            "Heavy Armour": EquipmentOption("Heavy Armour", 6, "4+ Armour Save"),
            "Shield": EquipmentOption("Shield", 3, "+1 Armour Save"),
            "Warhorse": EquipmentOption("Warhorse", 18, "Move 8, +1 Strength on charge"),
            "Barded Warhorse": EquipmentOption("Barded Warhorse", 24, "Move 8, +1 Strength, 2+ save"),
            "Magic Weapon": EquipmentOption("Magic Weapon", 25, "Magic weapon allowance"),
            "Magic Armour": EquipmentOption("Magic Armour", 25, "Magic armour allowance"),
            "Enchanted Item": EquipmentOption("Enchanted Item", 15, "Enchanted item allowance"),
            "Arcane Item": EquipmentOption("Arcane Item", 25, "Arcane item allowance"),
        },
        special_rules=["General", "Inspiring Presence"],
        default_equipment=["Hand Weapon", "Full Plate Armour"]
    )
    
    units["Captain of the Empire"] = EmpireUnit(
        name="Captain of the Empire",
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=5, ballistic_skill=5, strength=4,
            toughness=4, wounds=2, initiative=4, attacks=2, leadership=8
        ),
        base_cost=50,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={
            "Additional Hand Weapon": EquipmentOption("Additional Hand Weapon", 3, "+1 Attack"),
            "Great Weapon": EquipmentOption("Great Weapon", 6, "+2 Strength, Strike Last"),
            "Lance": EquipmentOption("Lance", 4, "+2 Strength on charge if mounted"),
            "Heavy Armour": EquipmentOption("Heavy Armour", 6, "4+ Armour Save"),
            "Shield": EquipmentOption("Shield", 3, "+1 Armour Save"),
            "Warhorse": EquipmentOption("Warhorse", 18, "Move 8, +1 Strength on charge"),
            "Battle Standard Bearer": EquipmentOption("Battle Standard Bearer", 25, "Army re-rolls break tests"),
            "Magic Weapon": EquipmentOption("Magic Weapon", 25, "Magic weapon allowance"),
            "Magic Armour": EquipmentOption("Magic Armour", 25, "Magic armour allowance"),
            "Enchanted Item": EquipmentOption("Enchanted Item", 15, "Enchanted item allowance"),
        },
        special_rules=["Inspiring Presence"],
        default_equipment=["Hand Weapon", "Light Armour"]
    )
    
    units["Wizard"] = EmpireUnit(
        name="Wizard",
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=2, initiative=3, attacks=1, leadership=7
        ),
        base_cost=65,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={
            "Level 2": EquipmentOption("Level 2", 35, "Wizard Level 2"),
            "Level 3": EquipmentOption("Level 3", 70, "Wizard Level 3"),
            "Level 4": EquipmentOption("Level 4", 105, "Wizard Level 4"),
            "Dispel Scroll": EquipmentOption("Dispel Scroll", 25, "Auto-dispel one spell"),
            "Power Stone": EquipmentOption("Power Stone", 20, "+2 Power Dice"),
            "Scroll of Shielding": EquipmentOption("Scroll of Shielding", 15, "5+ Ward Save vs spells"),
            "Warhorse": EquipmentOption("Warhorse", 18, "Move 8"),
            "Magic Weapon": EquipmentOption("Magic Weapon", 25, "Magic weapon allowance"),
            "Magic Armour": EquipmentOption("Magic Armour", 25, "Magic armour allowance"),
            "Arcane Item": EquipmentOption("Arcane Item", 50, "Arcane item allowance"),
        },
        special_rules=["Wizard (Level 1)"],
        default_equipment=["Hand Weapon"]
    )
    
    units["Empire Engineer"] = EmpireUnit(
        name="Empire Engineer",
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=3, ballistic_skill=4, strength=3,
            toughness=3, wounds=2, initiative=3, attacks=1, leadership=7
        ),
        base_cost=45,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={
            "Repeater Handgun": EquipmentOption("Repeater Handgun", 10, "Multiple shots"),
            "Hochland Long Rifle": EquipmentOption("Hochland Long Rifle", 20, "S4, Move or Fire"),
            "Light Armour": EquipmentOption("Light Armour", 3, "6+ Armour Save"),
        },
        special_rules=["Engineer"],
        default_equipment=["Hand Weapon"]
    )
    
    # ===========================================
    # CORE UNITS
    # ===========================================
    
    units["State Troops"] = EmpireUnit(
        name="State Troops",
        category=UnitCategory.CORE,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7
        ),
        base_cost=0,
        cost_per_model=5,
        min_size=10, max_size=40,
        equipment_options={
            "Spears": EquipmentOption("Spears", 1, "Fight in extra rank", True),
            "Light Armour": EquipmentOption("Light Armour", 1, "6+ Armour Save", True),
            "Shields": EquipmentOption("Shields", 1, "+1 Armour Save vs shooting", True),
            "Champion": EquipmentOption("Champion", 10, "+1 WS, +1 LD"),
            "Standard Bearer": EquipmentOption("Standard Bearer", 10, "Unit Standard"),
            "Musician": EquipmentOption("Musician", 5, "+1 to combat resolution"),
        },
        special_rules=["State Troops"],
        default_equipment=["Hand Weapon"]
    )
    
    units["Spearmen"] = EmpireUnit(
        name="Spearmen",
        category=UnitCategory.CORE,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7
        ),
        base_cost=0,
        cost_per_model=4,
        min_size=10, max_size=40,
        equipment_options={
            "Light Armour": EquipmentOption("Light Armour", 1, "6+ Armour Save", True),
            "Shields": EquipmentOption("Shields", 1, "+1 Armour Save vs shooting", True),
            "Champion": EquipmentOption("Champion", 10, "+1 WS, +1 LD"),
            "Standard Bearer": EquipmentOption("Standard Bearer", 10, "Unit Standard"),
            "Musician": EquipmentOption("Musician", 5, "+1 to combat resolution"),
        },
        special_rules=["State Troops", "Fight in Extra Rank"],
        default_equipment=["Spear"]
    )
    
    units["Crossbowmen"] = EmpireUnit(
        name="Crossbowmen",
        category=UnitCategory.CORE,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7
        ),
        base_cost=0,
        cost_per_model=9,
        min_size=10, max_size=20,
        equipment_options={
            "Champion": EquipmentOption("Champion", 10, "+1 WS, +1 LD"),
            "Standard Bearer": EquipmentOption("Standard Bearer", 10, "Unit Standard"),
            "Musician": EquipmentOption("Musician", 5, "+1 to combat resolution"),
        },
        special_rules=["State Troops"],
        default_equipment=["Crossbow", "Hand Weapon"]
    )
    
    units["Handgunners"] = EmpireUnit(
        name="Handgunners",
        category=UnitCategory.CORE,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7
        ),
        base_cost=0,
        cost_per_model=10,
        min_size=10, max_size=20,
        equipment_options={
            "Hochland Long Rifle": EquipmentOption("Hochland Long Rifle", 20, "S4, Move or Fire (one model)"),
            "Champion": EquipmentOption("Champion", 10, "+1 WS, +1 LD"),
            "Standard Bearer": EquipmentOption("Standard Bearer", 10, "Unit Standard"),
            "Musician": EquipmentOption("Musician", 5, "+1 to combat resolution"),
        },
        special_rules=["State Troops"],
        default_equipment=["Handgun", "Hand Weapon"]
    )
    
    units["Free Company"] = EmpireUnit(
        name="Free Company",
        category=UnitCategory.CORE,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=6
        ),
        base_cost=0,
        cost_per_model=5,
        min_size=10, max_size=30,
        equipment_options={
            "Champion": EquipmentOption("Champion", 10, "+1 WS, +1 LD"),
            "Standard Bearer": EquipmentOption("Standard Bearer", 10, "Unit Standard"),
            "Musician": EquipmentOption("Musician", 5, "+1 to combat resolution"),
        },
        special_rules=["Militia", "Skirmishers"],
        default_equipment=["Hand Weapon", "Additional Hand Weapon"]
    )
    
    # ===========================================
    # SPECIAL UNITS
    # ===========================================
    
    units["Greatswords"] = EmpireUnit(
        name="Greatswords",
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.INFANTRY,
        stats=UnitStats(
            movement=4, weapon_skill=4, ballistic_skill=3, strength=4,
            toughness=3, wounds=1, initiative=4, attacks=1, leadership=8
        ),
        base_cost=0,
        cost_per_model=10,
        min_size=10, max_size=25,
        equipment_options={
            "Champion": EquipmentOption("Champion", 15, "+1 WS, +1 LD"),
            "Standard Bearer": EquipmentOption("Standard Bearer", 15, "Unit Standard"),
            "Musician": EquipmentOption("Musician", 10, "+1 to combat resolution"),
        },
        special_rules=["Stubborn"],
        default_equipment=["Great Weapon", "Full Plate Armour"]
    )
    
    units["Knights"] = EmpireUnit(
        name="Knights",
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.CAVALRY,
        stats=UnitStats(
            movement=7, weapon_skill=4, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=8
        ),
        base_cost=0,
        cost_per_model=22,
        min_size=5, max_size=12,
        equipment_options={
            "Inner Circle": EquipmentOption("Inner Circle", 3, "+1 WS", True),
            "Champion": EquipmentOption("Champion", 20, "+1 WS, +1 LD"),
            "Standard Bearer": EquipmentOption("Standard Bearer", 20, "Unit Standard"),
            "Musician": EquipmentOption("Musician", 10, "+1 to combat resolution"),
        },
        special_rules=["Heavy Cavalry"],
        default_equipment=["Lance", "Heavy Armour", "Shield", "Barded Warhorse"]
    )
    
    units["Pistoliers"] = EmpireUnit(
        name="Pistoliers",
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.CAVALRY,
        stats=UnitStats(
            movement=8, weapon_skill=3, ballistic_skill=3, strength=3,
            toughness=3, wounds=1, initiative=3, attacks=1, leadership=7
        ),
        base_cost=0,
        cost_per_model=16,
        min_size=5, max_size=10,
        equipment_options={
            "Repeater Pistol": EquipmentOption("Repeater Pistol", 3, "Multiple shots", True),
            "Champion": EquipmentOption("Champion", 12, "+1 WS, +1 LD"),
            "Musician": EquipmentOption("Musician", 8, "+1 to combat resolution"),
        },
        special_rules=["Fast Cavalry"],
        default_equipment=["Pistol", "Hand Weapon", "Light Armour", "Warhorse"]
    )
    
    units["Great Cannon"] = EmpireUnit(
        name="Great Cannon",
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.WAR_MACHINE,
        stats=UnitStats(
            movement=0, weapon_skill=0, ballistic_skill=0, strength=0,
            toughness=6, wounds=3, initiative=0, attacks=0, leadership=7
        ),
        base_cost=100,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={},
        special_rules=["War Machine", "Cannon", "Guess Range"],
        default_equipment=["3 Crew"]
    )
    
    units["Mortar"] = EmpireUnit(
        name="Mortar",
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.WAR_MACHINE,
        stats=UnitStats(
            movement=0, weapon_skill=0, ballistic_skill=0, strength=0,
            toughness=5, wounds=3, initiative=0, attacks=0, leadership=7
        ),
        base_cost=75,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={},
        special_rules=["War Machine", "Stone Thrower", "Guess Range"],
        default_equipment=["3 Crew"]
    )
    
    # ===========================================
    # RARE UNITS
    # ===========================================
    
    units["Steam Tank"] = EmpireUnit(
        name="Steam Tank",
        category=UnitCategory.RARE,
        troop_type=TroopType.BEHEMOTH,
        stats=UnitStats(
            movement=4, weapon_skill=0, ballistic_skill=0, strength=6,
            toughness=7, wounds=10, initiative=0, attacks=999, leadership=8
        ),
        base_cost=300,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={},
        special_rules=["Terror", "Large Target", "Steam Points", "Unbreakable"],
        default_equipment=["Steam Cannon", "Engineer"],
        restrictions=["0-1 per 1000 points"]
    )
    
    units["Helblaster Volley Gun"] = EmpireUnit(
        name="Helblaster Volley Gun",
        category=UnitCategory.RARE,
        troop_type=TroopType.WAR_MACHINE,
        stats=UnitStats(
            movement=0, weapon_skill=0, ballistic_skill=0, strength=0,
            toughness=6, wounds=3, initiative=0, attacks=0, leadership=7
        ),
        base_cost=110,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={},
        special_rules=["War Machine", "Volley Gun", "Multiple Shots"],
        default_equipment=["3 Crew"]
    )
    
    units["Empire War Wagon"] = EmpireUnit(
        name="Empire War Wagon",
        category=UnitCategory.RARE,
        troop_type=TroopType.WAR_MACHINE,
        stats=UnitStats(
            movement=6, weapon_skill=0, ballistic_skill=0, strength=0,
            toughness=5, wounds=4, initiative=0, attacks=0, leadership=7
        ),
        base_cost=120,
        cost_per_model=0,
        min_size=1, max_size=1,
        equipment_options={
            "Mortar": EquipmentOption("Mortar", 25, "Stone Thrower"),
            "Volley Gun": EquipmentOption("Volley Gun", 30, "Multiple Shots"),
        },
        special_rules=["War Machine", "Mobile"],
        default_equipment=["4 Crew", "Crossbows"]
    )
    
    return units

def display_army_database():
    """Display the complete army database"""
    units = create_empire_army_database()
    
    print("🏛️ COMPLETE EMPIRE ARMY DATABASE")
    print("=" * 80)
    print()
    
    categories = [UnitCategory.CHARACTER, UnitCategory.CORE, UnitCategory.SPECIAL, UnitCategory.RARE]
    
    for category in categories:
        print(f"📋 {category.value.upper()} UNITS")
        print("-" * 50)
        
        category_units = [unit for unit in units.values() if unit.category == category]
        
        for unit in category_units:
            print(f"\n• {unit.name}")
            print(f"  Type: {unit.troop_type.value}")
            
            stats = unit.stats
            print(f"  Stats: M{stats.movement} WS{stats.weapon_skill} BS{stats.ballistic_skill} " +
                  f"S{stats.strength} T{stats.toughness} W{stats.wounds} " +
                  f"I{stats.initiative} A{stats.attacks} Ld{stats.leadership}")
            
            if unit.base_cost > 0 and unit.cost_per_model > 0:
                print(f"  Cost: {unit.base_cost} + {unit.cost_per_model} per model")
            elif unit.base_cost > 0:
                print(f"  Cost: {unit.base_cost} points")
            else:
                print(f"  Cost: {unit.cost_per_model} per model")
            
            print(f"  Unit Size: {unit.min_size}-{unit.max_size}")
            print(f"  Equipment: {', '.join(unit.default_equipment)}")
            print(f"  Special Rules: {', '.join(unit.special_rules)}")
            
            if unit.equipment_options:
                print("  Equipment Options:")
                for name, option in unit.equipment_options.items():
                    cost_type = "per model" if option.per_model else "per unit"
                    print(f"    - {name}: {option.cost} pts {cost_type} ({option.description})")
            
            if unit.restrictions:
                print(f"  Restrictions: {', '.join(unit.restrictions)}")
        
        print()

def main():
    """Run the army database display"""
    display_army_database()

if __name__ == '__main__':
    main() 