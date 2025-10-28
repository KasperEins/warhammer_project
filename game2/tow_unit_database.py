#!/usr/bin/env python3
"""
🏛️ WARHAMMER: THE OLD WORLD - COMPREHENSIVE UNIT DATABASE
========================================================

Complete unit statistics, costs, and rules for:
- Orc & Goblin Tribes Grand Army
- City-State of Nuln Army of Infamy

Built from official TOW sources for authentic army building AI.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import json

class Faction(Enum):
    ORC_GOBLIN_TRIBES = "Orc & Goblin Tribes"
    CITY_STATE_NULN = "City-State of Nuln"

class UnitCategory(Enum):
    CHARACTER = "Character"
    CORE = "Core"
    SPECIAL = "Special"
    RARE = "Rare"
    MERCENARY = "Mercenary"

class TroopType(Enum):
    REGULAR_INFANTRY = "Regular Infantry"
    HEAVY_INFANTRY = "Heavy Infantry"
    LIGHT_CAVALRY = "Light Cavalry"
    HEAVY_CAVALRY = "Heavy Cavalry"
    MONSTROUS_INFANTRY = "Monstrous Infantry"
    MONSTROUS_CAVALRY = "Monstrous Cavalry"
    LIGHT_CHARIOT = "Light Chariot"
    HEAVY_CHARIOT = "Heavy Chariot"
    WAR_MACHINE = "War Machine"
    BEHEMOTH = "Behemoth"
    SWARM = "Swarm"

@dataclass
class WeaponOption:
    """Weapon or equipment option"""
    name: str
    cost_per_model: float
    special_rules: List[str] = field(default_factory=list)
    replaces: Optional[str] = None  # What it replaces

@dataclass
class UnitUpgrade:
    """Unit upgrade option"""
    name: str
    cost_per_model: float = 0
    cost_per_unit: float = 0
    description: str = ""
    special_rules: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)

@dataclass
class TOWUnit:
    """Complete TOW unit with all statistics and options"""
    # Basic info
    name: str
    faction: Faction
    category: UnitCategory
    troop_type: TroopType
    
    # Core statistics
    movement: int
    weapon_skill: int
    ballistic_skill: int
    strength: int
    toughness: int
    wounds: int
    initiative: int
    attacks: int
    leadership: int
    
    # Organization
    points_per_model: float
    min_unit_size: int
    max_unit_size: int
    
    # Special rules and equipment
    special_rules: List[str] = field(default_factory=list)
    default_equipment: List[str] = field(default_factory=list)
    
    # Options
    weapon_options: List[WeaponOption] = field(default_factory=list)
    upgrade_options: List[UnitUpgrade] = field(default_factory=list)
    
    # Command options
    can_have_champion: bool = False
    champion_cost: float = 0
    champion_name: str = ""
    can_have_standard: bool = False
    standard_cost: float = 0
    can_have_musician: bool = False
    musician_cost: float = 0
    
    # Restrictions
    max_per_army: int = -1  # -1 = unlimited
    requires_character: Optional[str] = None
    
    def total_cost(self, unit_size: int, equipment_choices: List[str] = None, 
                   upgrades: List[str] = None, include_command: bool = False) -> float:
        """Calculate total unit cost with options"""
        base_cost = self.points_per_model * unit_size
        
        # Add equipment costs
        if equipment_choices:
            for choice in equipment_choices:
                for weapon in self.weapon_options:
                    if weapon.name == choice:
                        base_cost += weapon.cost_per_model * unit_size
                        break
        
        # Add upgrade costs
        if upgrades:
            for upgrade_name in upgrades:
                for upgrade in self.upgrade_options:
                    if upgrade.name == upgrade_name:
                        base_cost += upgrade.cost_per_model * unit_size + upgrade.cost_per_unit
                        break
        
        # Add command costs
        if include_command:
            if self.can_have_champion:
                base_cost += self.champion_cost
            if self.can_have_standard:
                base_cost += self.standard_cost
            if self.can_have_musician:
                base_cost += self.musician_cost
        
        return base_cost

# =============================================================================
# ORC & GOBLIN TRIBES UNIT DATABASE
# =============================================================================

def create_orc_goblin_database() -> Dict[str, TOWUnit]:
    """Create complete Orc & Goblin Tribes unit database"""
    units = {}
    
    # CHARACTERS
    units["Black Orc Warboss"] = TOWUnit(
        name="Black Orc Warboss",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=4, weapon_skill=7, ballistic_skill=3, strength=5, toughness=5,
        wounds=3, initiative=6, attacks=4, leadership=9,
        points_per_model=135,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Choppas", "Quell Impetuosity", "Immune to Psychology"],
        default_equipment=["Choppa", "Heavy Armour"],
        weapon_options=[
            WeaponOption("Great Weapon", 6, ["Two-Handed", "Armour Bane (1)"]),
            WeaponOption("Shield", 2, ["Parry (6+)"]),
        ]
    )
    
    units["Black Orc Bigboss"] = TOWUnit(
        name="Black Orc Bigboss",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=4, weapon_skill=6, ballistic_skill=3, strength=4, toughness=5,
        wounds=2, initiative=5, attacks=3, leadership=8,
        points_per_model=75,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Choppas", "Quell Impetuosity"],
        default_equipment=["Choppa", "Heavy Armour"],
        can_have_standard=True, standard_cost=25  # Battle Standard Bearer
    )
    
    units["Orc Warboss"] = TOWUnit(
        name="Orc Warboss",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=4, weapon_skill=6, ballistic_skill=3, strength=5, toughness=5,
        wounds=3, initiative=5, attacks=4, leadership=8,
        points_per_model=110,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Choppas", "Impetuous"],
        default_equipment=["Choppa", "Light Armour"]
    )
    
    units["Orc Bigboss"] = TOWUnit(
        name="Orc Bigboss",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=4, weapon_skill=5, ballistic_skill=3, strength=4, toughness=5,
        wounds=2, initiative=4, attacks=3, leadership=7,
        points_per_model=55,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Choppas", "Impetuous"],
        default_equipment=["Choppa", "Light Armour"]
    )
    
    units["Orc Weirdnob"] = TOWUnit(
        name="Orc Weirdnob",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=4, weapon_skill=3, ballistic_skill=3, strength=4, toughness=5,
        wounds=3, initiative=3, attacks=1, leadership=7,
        points_per_model=115,  # Level 3 Wizard
        min_unit_size=1, max_unit_size=1,
        special_rules=["Wizard (Level 3)", "Choppas", "Lore of Da Big Waaagh!"],
        default_equipment=["Choppa"]
    )
    
    units["Goblin Warboss"] = TOWUnit(
        name="Goblin Warboss",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=5, ballistic_skill=4, strength=4, toughness=4,
        wounds=3, initiative=5, attacks=4, leadership=7,
        points_per_model=70,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Fear of Elves"],
        default_equipment=["Choppa", "Light Armour"]
    )
    
    # CORE UNITS
    units["Orc Mob"] = TOWUnit(
        name="Orc Mob",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CORE,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=3, ballistic_skill=3, strength=3, toughness=4,
        wounds=1, initiative=2, attacks=1, leadership=7,
        points_per_model=7,
        min_unit_size=10, max_unit_size=40,
        special_rules=["Choppas", "Impetuous", "Warband"],
        default_equipment=["Choppa"],
        weapon_options=[
            WeaponOption("Shield", 1, ["Parry (6+)"]),
            WeaponOption("Spear", 1, ["Fight in Extra Rank"]),
            WeaponOption("Additional Hand Weapon", 1, ["+1 Attack"])
        ],
        upgrade_options=[
            UnitUpgrade("Big 'Uns", 2, 0, "Bigger, stronger Orcs", 
                       ["Strength +1", "Armour Bane (1)"], ["0-1 per army"]),
            UnitUpgrade("Savage Orcs", 2, 0, "Frenzied tattooed warriors",
                       ["Frenzy", "Warpaint (6+ Ward)"], ["Cannot take armour"])
        ],
        can_have_champion=True, champion_cost=7, champion_name="Orc Boss",
        can_have_standard=True, standard_cost=7,
        can_have_musician=True, musician_cost=5
    )
    
    units["Goblin Mob"] = TOWUnit(
        name="Goblin Mob",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CORE,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=2, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=2, attacks=1, leadership=5,
        points_per_model=4,
        min_unit_size=10, max_unit_size=40,
        special_rules=["Fear of Elves", "Warband"],
        default_equipment=["Choppa"],
        weapon_options=[
            WeaponOption("Shield", 0.5, ["Parry (6+)"]),
            WeaponOption("Shortbow", 0, ["Range 16\""], "Choppa/Shield")
        ],
        can_have_champion=True, champion_cost=5, champion_name="Goblin Boss",
        can_have_standard=True, standard_cost=5,
        can_have_musician=True, musician_cost=5
    )
    
    units["Goblin Wolf Riders"] = TOWUnit(
        name="Goblin Wolf Riders",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CORE,
        troop_type=TroopType.LIGHT_CAVALRY,
        movement=9, weapon_skill=2, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=6,
        points_per_model=10,
        min_unit_size=5, max_unit_size=20,
        special_rules=["Fear of Elves", "Fire & Flee", "Chariot Runners"],
        default_equipment=["Choppa", "Giant Wolf"],
        weapon_options=[
            WeaponOption("Shield", 1, ["Parry (6+)"]),
            WeaponOption("Spear", 1, ["Fight in Extra Rank"])
        ],
        can_have_champion=True, champion_cost=6, champion_name="Wolf Rider Boss",
        can_have_standard=True, standard_cost=6,
        can_have_musician=True, musician_cost=6
    )
    
    units["Night Goblin Mob"] = TOWUnit(
        name="Night Goblin Mob",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.CORE,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=2, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=5,
        points_per_model=5,
        min_unit_size=10, max_unit_size=40,
        special_rules=["Fear of Elves", "Warband"],
        default_equipment=["Choppa"],
        weapon_options=[
            WeaponOption("Shield", 0.5, ["Parry (6+)"]),
            WeaponOption("Shortbow", 0, ["Range 16\""], "Choppa"),
            WeaponOption("Spear", 1, ["Fight in Extra Rank"])
        ],
        upgrade_options=[
            UnitUpgrade("Netters", 0, 15, "3 models with nets", 
                       ["Enemy unit -1 Strength in combat"], ["3 models only"]),
            UnitUpgrade("Fanatics", 0, 25, "Hidden fanatics", 
                       ["Secret deployment", "2D6 Impact Hits"], ["Max 3, 1 per 10 models"])
        ],
        can_have_champion=True, champion_cost=5, champion_name="Night Goblin Boss",
        can_have_standard=True, standard_cost=5,
        can_have_musician=True, musician_cost=5,
        requires_character="Night Goblin Chief or Shaman"
    )
    
    # SPECIAL UNITS
    units["Black Orc Mob"] = TOWUnit(
        name="Black Orc Mob",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=4, weapon_skill=4, ballistic_skill=3, strength=4, toughness=4,
        wounds=1, initiative=3, attacks=1, leadership=8,
        points_per_model=12,
        min_unit_size=10, max_unit_size=30,
        special_rules=["Choppas", "Immune to Psychology", "Ignore Panic"],
        default_equipment=["Choppa", "Heavy Armour"],
        weapon_options=[
            WeaponOption("Shield", 1, ["Parry (6+)"]),
            WeaponOption("Great Weapon", 0, ["Two-Handed", "Armour Bane (1)"], "Choppa")
        ],
        can_have_champion=True, champion_cost=10, champion_name="Black Orc Boss",
        can_have_standard=True, standard_cost=10,
        can_have_musician=True, musician_cost=10
    )
    
    units["Common Troll Mob"] = TOWUnit(
        name="Common Troll Mob",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.MONSTROUS_INFANTRY,
        movement=6, weapon_skill=3, ballistic_skill=1, strength=5, toughness=4,
        wounds=3, initiative=2, attacks=3, leadership=4,
        points_per_model=45,
        min_unit_size=3, max_unit_size=9,
        special_rules=["Stupidity", "Regeneration (5+)", "Troll Vomit", "Fear"],
        default_equipment=["Claws and Fangs"]
    )
    
    units["Orc Boar Boys"] = TOWUnit(
        name="Orc Boar Boys",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.HEAVY_CAVALRY,
        movement=7, weapon_skill=3, ballistic_skill=3, strength=3, toughness=4,
        wounds=1, initiative=2, attacks=1, leadership=7,
        points_per_model=19,
        min_unit_size=5, max_unit_size=20,
        special_rules=["Choppas", "Impetuous", "Furious Charge", "Tusker Charge"],
        default_equipment=["Choppa", "Light Armour", "War Boar"],
        weapon_options=[
            WeaponOption("Shield", 2, ["Parry (6+)"]),
            WeaponOption("Spear", 2, ["Fight in Extra Rank"])
        ],
        upgrade_options=[
            UnitUpgrade("Big 'Uns", 3, 0, "Bigger, stronger Orc Boar Boys", 
                       ["Strength +1", "Armour Bane (1)"], ["0-1 per army"])
        ],
        can_have_champion=True, champion_cost=10, champion_name="Boar Boy Boss",
        can_have_standard=True, standard_cost=10,
        can_have_musician=True, musician_cost=10
    )
    
    # RARE UNITS
    units["Giant"] = TOWUnit(
        name="Giant",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.RARE,
        troop_type=TroopType.BEHEMOTH,
        movement=6, weapon_skill=3, ballistic_skill=1, strength=6, toughness=6,
        wounds=6, initiative=2, attacks=999, leadership=10,  # Special attack table
        points_per_model=175,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Terror", "Large Target", "Stubborn", "Pick Attack Table"],
        default_equipment=["Massive Clubs and Fists"]
    )
    
    units["Arachnarok Spider"] = TOWUnit(
        name="Arachnarok Spider",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.RARE,
        troop_type=TroopType.BEHEMOTH,
        movement=7, weapon_skill=4, ballistic_skill=0, strength=5, toughness=6,
        wounds=7, initiative=4, attacks=6, leadership=7,  # Crew leadership
        points_per_model=230,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Terror", "Large Target", "Poisoned Attacks", "Spider Shrine"],
        default_equipment=["Venomous Fangs", "8 Goblin Crew"]
    )
    
    units["Goblin Rock Lobber"] = TOWUnit(
        name="Goblin Rock Lobber",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.RARE,
        troop_type=TroopType.WAR_MACHINE,
        movement=0, weapon_skill=0, ballistic_skill=0, strength=0, toughness=6,
        wounds=4, initiative=0, attacks=0, leadership=4,  # Crew leadership
        points_per_model=75,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Stone Thrower", "Guess Range"],
        default_equipment=["4 Goblin Crew"]
    )
    
    # MERCENARIES
    units["Badlands Ogre Bulls"] = TOWUnit(
        name="Badlands Ogre Bulls",
        faction=Faction.ORC_GOBLIN_TRIBES,
        category=UnitCategory.MERCENARY,
        troop_type=TroopType.MONSTROUS_INFANTRY,
        movement=6, weapon_skill=3, ballistic_skill=2, strength=4, toughness=4,
        wounds=3, initiative=2, attacks=3, leadership=7,
        points_per_model=33,
        min_unit_size=3, max_unit_size=9,
        special_rules=["Fear"],
        default_equipment=["Clubs"],
        weapon_options=[
            WeaponOption("Ironfists", 3, ["Parry (6+)", "Armour Bane (1)"]),
            WeaponOption("Additional Hand Weapons", 3, ["+1 Attack"]),
            WeaponOption("Light Armour", 3, ["6+ Save"]),
            WeaponOption("Great Weapons", 6, ["Two-Handed", "Armour Bane (1)"])
        ],
        can_have_champion=True, champion_cost=7, champion_name="Ogre Crusher"
    )
    
    return units

# =============================================================================
# CITY-STATE OF NULN UNIT DATABASE
# =============================================================================

def create_nuln_database() -> Dict[str, TOWUnit]:
    """Create complete City-State of Nuln unit database"""
    units = {}
    
    # CHARACTERS
    units["General of the Empire"] = TOWUnit(
        name="General of the Empire",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=6, ballistic_skill=5, strength=4, toughness=4,
        wounds=3, initiative=5, attacks=3, leadership=9,
        points_per_model=85,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Inspiring Presence"],
        default_equipment=["Hand Weapon", "Full Plate Armour"]
    )
    
    units["Captain of the Empire"] = TOWUnit(
        name="Captain of the Empire",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=5, ballistic_skill=5, strength=4, toughness=4,
        wounds=2, initiative=4, attacks=2, leadership=8,
        points_per_model=45,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Inspiring Presence"],
        default_equipment=["Hand Weapon", "Light Armour"],
        can_have_standard=True, standard_cost=25  # Battle Standard Bearer
    )
    
    units["Empire Engineer"] = TOWUnit(
        name="Empire Engineer",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=3, ballistic_skill=4, strength=3, toughness=3,
        wounds=2, initiative=3, attacks=1, leadership=7,
        points_per_model=45,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Artillery Master", "Big Guns Know No Fear"],
        default_equipment=["Hand Weapon"],
        can_have_standard=True, standard_cost=25  # Battle Standard Bearer for Nuln
    )
    
    units["Master Mage"] = TOWUnit(
        name="Master Mage",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CHARACTER,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=3, ballistic_skill=3, strength=3, toughness=3,
        wounds=2, initiative=3, attacks=1, leadership=8,
        points_per_model=60,  # Level 1 base
        min_unit_size=1, max_unit_size=1,
        special_rules=["Wizard (Level 1)"],
        default_equipment=["Hand Weapon"]
        # Level upgrades: +35 pts per level
    )
    
    # CORE UNITS
    units["Nuln State Troops"] = TOWUnit(
        name="Nuln State Troops",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CORE,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=3, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=7,
        points_per_model=5,
        min_unit_size=10, max_unit_size=40,
        special_rules=["State Troops"],
        default_equipment=["Hand Weapon", "Light Armour"],
        weapon_options=[
            WeaponOption("Halberd", 1, ["Fight in Extra Rank", "Armour Bane (1)"]),
            WeaponOption("Handgun", 1, ["Range 24\"", "Armour Bane (2)", "Move or Fire"]),
            WeaponOption("Shield", 0.5, ["Parry (6+)"])
        ],
        can_have_champion=True, champion_cost=5, champion_name="Sergeant",
        can_have_standard=True, standard_cost=5,
        can_have_musician=True, musician_cost=5
    )
    
    units["Nuln Veteran State Troops"] = TOWUnit(
        name="Nuln Veteran State Troops",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CORE,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=4, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=7,
        points_per_model=7,
        min_unit_size=10, max_unit_size=30,
        special_rules=["State Troops", "Veteran"],
        default_equipment=["Hand Weapon", "Light Armour"],
        weapon_options=[
            WeaponOption("Halberd", 1, ["Fight in Extra Rank", "Armour Bane (1)"]),
            WeaponOption("Handgun", 1, ["Range 24\"", "Armour Bane (2)", "Move or Fire"])
        ],
        can_have_champion=True, champion_cost=5, champion_name="Sergeant",
        can_have_standard=True, standard_cost=5,
        can_have_musician=True, musician_cost=5
    )
    
    units["Nuln Veteran Outriders"] = TOWUnit(
        name="Nuln Veteran Outriders",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CORE,
        troop_type=TroopType.LIGHT_CAVALRY,
        movement=8, weapon_skill=3, ballistic_skill=4, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=7,
        points_per_model=19,
        min_unit_size=5, max_unit_size=10,
        special_rules=["Fast Cavalry", "Veteran Outriders"],  # No Ponderous rule
        default_equipment=["Repeater Handgun", "Light Armour", "Empire Warhorse"],
        can_have_champion=True, champion_cost=6, champion_name="Sharpshooter",
        can_have_musician=True, musician_cost=6
    )
    
    units["Free Company Militia"] = TOWUnit(
        name="Free Company Militia",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.CORE,
        troop_type=TroopType.REGULAR_INFANTRY,
        movement=4, weapon_skill=3, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=6,
        points_per_model=5,
        min_unit_size=10, max_unit_size=30,
        special_rules=["Militia"],
        default_equipment=["Hand Weapon"],
        weapon_options=[
            WeaponOption("Additional Hand Weapon", 1, ["+1 Attack"]),
            WeaponOption("Mixed Weapons", 0, ["Includes throwing weapons"])
        ],
        can_have_champion=True, champion_cost=5, champion_name="Militia Leader",
        can_have_standard=True, standard_cost=5,
        can_have_musician=True, musician_cost=5
    )
    
    # SPECIAL UNITS
    units["Empire Greatswords"] = TOWUnit(
        name="Empire Greatswords",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=4, weapon_skill=4, ballistic_skill=3, strength=4, toughness=3,
        wounds=1, initiative=4, attacks=1, leadership=8,
        points_per_model=12,
        min_unit_size=10, max_unit_size=30,
        special_rules=["Stubborn"],
        default_equipment=["Great Weapon", "Full Plate Armour"],
        can_have_champion=True, champion_cost=10, champion_name="Count's Champion",
        can_have_standard=True, standard_cost=10,
        can_have_musician=True, musician_cost=10
    )
    
    units["Empire Knights"] = TOWUnit(
        name="Empire Knights",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.HEAVY_CAVALRY,
        movement=7, weapon_skill=4, ballistic_skill=3, strength=3, toughness=3,
        wounds=1, initiative=3, attacks=1, leadership=8,
        points_per_model=20,
        min_unit_size=5, max_unit_size=15,
        special_rules=["Devastating Charge"],
        default_equipment=["Lance", "Heavy Armour", "Shield", "Barded Warhorse"],
        can_have_champion=True, champion_cost=10, champion_name="Preceptor",
        can_have_standard=True, standard_cost=10,
        can_have_musician=True, musician_cost=10
    )
    
    units["Great Cannon"] = TOWUnit(
        name="Great Cannon",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.WAR_MACHINE,
        movement=0, weapon_skill=0, ballistic_skill=0, strength=0, toughness=6,
        wounds=3, initiative=0, attacks=0, leadership=7,
        points_per_model=125,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Cannon", "Guess Range", "Multiple Wounds (D6)"],
        default_equipment=["3 Gun Crew"],
        upgrade_options=[
            UnitUpgrade("Gun Limbers", 0, 10, "Enhanced mobility and reliability",
                       ["Vanguard", "Veteran"], [])
        ]
    )
    
    units["Mortar"] = TOWUnit(
        name="Mortar",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.SPECIAL,
        troop_type=TroopType.WAR_MACHINE,
        movement=0, weapon_skill=0, ballistic_skill=0, strength=0, toughness=5,
        wounds=3, initiative=0, attacks=0, leadership=7,
        points_per_model=95,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Stone Thrower", "Guess Range", "No Line of Sight"],
        default_equipment=["3 Gun Crew"],
        upgrade_options=[
            UnitUpgrade("Gun Limbers", 0, 10, "Enhanced mobility and reliability",
                       ["Vanguard", "Veteran"], [])
        ]
    )
    
    # RARE UNITS
    units["Steam Tank"] = TOWUnit(
        name="Steam Tank",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.RARE,
        troop_type=TroopType.BEHEMOTH,
        movement=4, weapon_skill=0, ballistic_skill=0, strength=6, toughness=7,
        wounds=10, initiative=0, attacks=999, leadership=8,  # Special attack table
        points_per_model=285,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Terror", "Large Target", "Steam Points", "Unbreakable"],
        default_equipment=["Steam Cannon", "Engineer Commander"],
        max_per_army=1
    )
    
    units["Helblaster Volley Gun"] = TOWUnit(
        name="Helblaster Volley Gun",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.RARE,
        troop_type=TroopType.WAR_MACHINE,
        movement=0, weapon_skill=0, ballistic_skill=0, strength=0, toughness=6,
        wounds=3, initiative=0, attacks=0, leadership=7,
        points_per_model=120,
        min_unit_size=1, max_unit_size=1,
        special_rules=["Volley Gun", "Multiple Shots", "Misfire"],
        default_equipment=["3 Gun Crew"],
        upgrade_options=[
            UnitUpgrade("Gun Limbers", 0, 15, "Enhanced mobility and reliability",
                       ["Vanguard", "Veteran"], [])
        ]
    )
    
    # MERCENARIES
    units["Imperial Ogres"] = TOWUnit(
        name="Imperial Ogres",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.MERCENARY,
        troop_type=TroopType.MONSTROUS_INFANTRY,
        movement=6, weapon_skill=3, ballistic_skill=2, strength=4, toughness=4,
        wounds=3, initiative=2, attacks=3, leadership=7,
        points_per_model=38,
        min_unit_size=3, max_unit_size=6,
        special_rules=["Fear"],
        default_equipment=["Clubs"],
        weapon_options=[
            WeaponOption("Ironfists", 4, ["Parry (6+)", "Armour Bane (1)"]),
            WeaponOption("Light Armour", 4, ["6+ Save"]),
            WeaponOption("Ogre Pistols", 5, ["Range 6\"", "Armour Bane (1)"]),
            WeaponOption("Light Cannon", 20, ["Range 24\"", "Cannon"], "1 per unit")
        ],
        can_have_champion=True, champion_cost=10, champion_name="Ogre Captain"
    )
    
    units["Imperial Dwarfs"] = TOWUnit(
        name="Imperial Dwarfs",
        faction=Faction.CITY_STATE_NULN,
        category=UnitCategory.MERCENARY,
        troop_type=TroopType.HEAVY_INFANTRY,
        movement=3, weapon_skill=4, ballistic_skill=3, strength=3, toughness=4,
        wounds=1, initiative=2, attacks=1, leadership=9,
        points_per_model=9,
        min_unit_size=10, max_unit_size=30,
        special_rules=["Resolute", "Hatred (Orcs & Goblins)"],
        default_equipment=["Hand Weapon", "Heavy Armour", "Shield"],
        weapon_options=[
            WeaponOption("Great Weapon", 2, ["Two-Handed", "Armour Bane (1)"], "Shield")
        ],
        can_have_champion=True, champion_cost=7, champion_name="Dwarf Veteran",
        can_have_standard=True, standard_cost=7,
        can_have_musician=True, musician_cost=7
    )
    
    return units

# =============================================================================
# COMBINED DATABASE
# =============================================================================

def get_all_units() -> Dict[str, TOWUnit]:
    """Get combined database of all TOW units"""
    units = {}
    units.update(create_orc_goblin_database())
    units.update(create_nuln_database())
    return units

def get_faction_units(faction: Faction) -> Dict[str, TOWUnit]:
    """Get units for specific faction"""
    all_units = get_all_units()
    return {name: unit for name, unit in all_units.items() if unit.faction == faction}

def save_database_to_json(filename: str = "tow_unit_database.json"):
    """Save unit database to JSON file"""
    all_units = get_all_units()
    serializable_units = {}
    
    for name, unit in all_units.items():
        # Convert dataclass to dict for JSON serialization
        unit_dict = {
            'name': unit.name,
            'faction': unit.faction.value,
            'category': unit.category.value,
            'troop_type': unit.troop_type.value,
            'stats': {
                'movement': unit.movement,
                'weapon_skill': unit.weapon_skill,
                'ballistic_skill': unit.ballistic_skill,
                'strength': unit.strength,
                'toughness': unit.toughness,
                'wounds': unit.wounds,
                'initiative': unit.initiative,
                'attacks': unit.attacks,
                'leadership': unit.leadership
            },
            'points_per_model': unit.points_per_model,
            'min_unit_size': unit.min_unit_size,
            'max_unit_size': unit.max_unit_size,
            'special_rules': unit.special_rules,
            'default_equipment': unit.default_equipment
        }
        serializable_units[name] = unit_dict
    
    with open(filename, 'w') as f:
        json.dump(serializable_units, f, indent=2)

if __name__ == "__main__":
    print("🏛️ WARHAMMER: THE OLD WORLD UNIT DATABASE")
    print("=" * 60)
    
    all_units = get_all_units()
    orc_units = get_faction_units(Faction.ORC_GOBLIN_TRIBES)
    nuln_units = get_faction_units(Faction.CITY_STATE_NULN)
    
    print(f"✅ Total units loaded: {len(all_units)}")
    print(f"⚔️ Orc & Goblin Tribes units: {len(orc_units)}")
    print(f"🏰 City-State of Nuln units: {len(nuln_units)}")
    
    # Show unit breakdown by category
    for faction in [Faction.ORC_GOBLIN_TRIBES, Faction.CITY_STATE_NULN]:
        faction_units = get_faction_units(faction)
        print(f"\n📊 {faction.value} breakdown:")
        
        for category in UnitCategory:
            category_units = [u for u in faction_units.values() if u.category == category]
            if category_units:
                print(f"   {category.value}: {len(category_units)} units")
    
    # Save to JSON for AI training
    save_database_to_json()
    print(f"\n💾 Database saved to tow_unit_database.json")
    print("🤖 Ready for AI army building and training!") 