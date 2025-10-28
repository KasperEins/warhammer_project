#!/usr/bin/env python3
"""
🏛️ WARHAMMER: THE OLD WORLD - INTELLIGENT ARMY BUILDER
====================================================

Advanced army building system that can:
- Construct valid 2000 point armies
- Enforce TOW army composition rules
- Generate random armies for AI training
- Optimize army builds through genetic algorithms
- Validate army legality

Built for AI agents to learn optimal army compositions.
"""

import random
import copy
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from tow_unit_database import Faction, UnitCategory, TroopType, TOWUnit, get_faction_units, get_all_units

@dataclass
class ArmyUnit:
    """A unit instance in an army with specific equipment and size"""
    unit_profile: TOWUnit
    unit_size: int
    equipment_choices: List[str] = field(default_factory=list)
    upgrade_choices: List[str] = field(default_factory=list)
    has_champion: bool = False
    has_standard: bool = False
    has_musician: bool = False
    
    @property
    def total_cost(self) -> float:
        """Calculate total unit cost"""
        return self.unit_profile.total_cost(
            self.unit_size, 
            self.equipment_choices, 
            self.upgrade_choices,
            self.has_champion or self.has_standard or self.has_musician
        )
    
    @property
    def display_name(self) -> str:
        """Get display name with equipment"""
        name = f"{self.unit_profile.name} ({self.unit_size})"
        if self.equipment_choices:
            name += f" [{', '.join(self.equipment_choices)}]"
        if self.upgrade_choices:
            name += f" <{', '.join(self.upgrade_choices)}>"
        return name

@dataclass
class TOWArmy:
    """Complete TOW army with validation"""
    faction: Faction
    units: List[ArmyUnit] = field(default_factory=list)
    points_limit: int = 2000
    name: str = ""
    
    @property
    def total_points(self) -> float:
        """Calculate total army points"""
        return sum(unit.total_cost for unit in self.units)
    
    @property
    def remaining_points(self) -> float:
        """Points remaining in budget"""
        return self.points_limit - self.total_points
    
    @property
    def is_valid(self) -> bool:
        """Check if army meets all requirements"""
        return self.validate_army()[0]
    
    def validate_army(self) -> Tuple[bool, List[str]]:
        """Validate army composition and return errors"""
        errors = []
        
        # Check points limit
        if self.total_points > self.points_limit:
            errors.append(f"Army exceeds points limit: {self.total_points}/{self.points_limit}")
        
        # Check minimum core requirements (25% of army)
        core_points = sum(unit.total_cost for unit in self.units 
                         if unit.unit_profile.category == UnitCategory.CORE)
        min_core = self.points_limit * 0.25
        if core_points < min_core:
            errors.append(f"Insufficient Core units: {core_points}/{min_core} points required")
        
        # Check character limits (25% of army)
        character_points = sum(unit.total_cost for unit in self.units 
                              if unit.unit_profile.category == UnitCategory.CHARACTER)
        max_characters = self.points_limit * 0.25
        if character_points > max_characters:
            errors.append(f"Too many Characters: {character_points}/{max_characters} points maximum")
        
        # Check special/rare limits (50% each)
        special_points = sum(unit.total_cost for unit in self.units 
                            if unit.unit_profile.category == UnitCategory.SPECIAL)
        rare_points = sum(unit.total_cost for unit in self.units 
                         if unit.unit_profile.category == UnitCategory.RARE)
        
        max_special_rare = self.points_limit * 0.5
        if special_points > max_special_rare:
            errors.append(f"Too many Special units: {special_points}/{max_special_rare} points maximum")
        if rare_points > max_special_rare:
            errors.append(f"Too many Rare units: {rare_points}/{max_special_rare} points maximum")
        
        # Check unit-specific limits
        unit_counts = {}
        for unit in self.units:
            unit_name = unit.unit_profile.name
            unit_counts[unit_name] = unit_counts.get(unit_name, 0) + 1
            
            if (unit.unit_profile.max_per_army > 0 and 
                unit_counts[unit_name] > unit.unit_profile.max_per_army):
                errors.append(f"Too many {unit_name}: {unit_counts[unit_name]}/{unit.unit_profile.max_per_army} maximum")
        
        # Check faction-specific requirements
        if self.faction == Faction.CITY_STATE_NULN:
            # Must include Nuln State Troops or Nuln Veteran State Troops
            has_nuln_troops = any(unit.unit_profile.name in ["Nuln State Troops", "Nuln Veteran State Troops"] 
                                 for unit in self.units)
            if not has_nuln_troops:
                errors.append("Nuln armies must include Nuln State Troops or Nuln Veteran State Troops")
        
        return len(errors) == 0, errors
    
    def add_unit(self, army_unit: ArmyUnit) -> bool:
        """Add unit to army if valid"""
        test_army = copy.deepcopy(self)
        test_army.units.append(army_unit)
        
        if test_army.total_points <= self.points_limit:
            self.units.append(army_unit)
            return True
        return False
    
    def remove_unit(self, index: int) -> bool:
        """Remove unit from army"""
        if 0 <= index < len(self.units):
            self.units.pop(index)
            return True
        return False
    
    def get_army_summary(self) -> str:
        """Get detailed army summary"""
        summary = f"🏛️ {self.faction.value} Army ({self.total_points:.0f}/{self.points_limit} pts)\n"
        summary += "=" * 60 + "\n"
        
        # Group by category
        by_category = {}
        for unit in self.units:
            category = unit.unit_profile.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(unit)
        
        for category in [UnitCategory.CHARACTER, UnitCategory.CORE, UnitCategory.SPECIAL, UnitCategory.RARE, UnitCategory.MERCENARY]:
            if category in by_category:
                category_units = by_category[category]
                category_points = sum(unit.total_cost for unit in category_units)
                
                summary += f"\n{category.value} ({category_points:.0f} pts):\n"
                for unit in category_units:
                    summary += f"  • {unit.display_name} - {unit.total_cost:.0f} pts\n"
        
        # Validation
        is_valid, errors = self.validate_army()
        summary += f"\nArmy Status: {'✅ VALID' if is_valid else '❌ INVALID'}\n"
        if errors:
            summary += "Issues:\n"
            for error in errors:
                summary += f"  - {error}\n"
        
        return summary

class TOWArmyBuilder:
    """Intelligent army builder for TOW"""
    
    def __init__(self, faction: Faction):
        self.faction = faction
        self.unit_database = get_faction_units(faction)
        self.army_templates = self._create_army_templates()
    
    def _create_army_templates(self) -> List[Dict]:
        """Create army composition templates"""
        if self.faction == Faction.ORC_GOBLIN_TRIBES:
            return [
                {
                    "name": "Orc Horde",
                    "core_focus": ["Orc Mob", "Goblin Mob"],
                    "special_focus": ["Black Orc Mob", "Orc Boar Boys"],
                    "rare_focus": ["Giant", "Goblin Rock Lobber"],
                    "character_types": ["Orc Warboss", "Orc Bigboss"]
                },
                {
                    "name": "Night Goblin Swarm", 
                    "core_focus": ["Night Goblin Mob", "Goblin Wolf Riders"],
                    "special_focus": ["Common Troll Mob"],
                    "rare_focus": ["Arachnarok Spider", "Mangler Squigs"],
                    "character_types": ["Goblin Warboss", "Night Goblin Bigboss"]
                },
                {
                    "name": "Elite Black Orc Force",
                    "core_focus": ["Orc Mob"],
                    "special_focus": ["Black Orc Mob", "Orc Boar Boys"],
                    "rare_focus": ["Giant"],
                    "character_types": ["Black Orc Warboss", "Black Orc Bigboss"]
                }
            ]
        else:  # Nuln
            return [
                {
                    "name": "Artillery Battery",
                    "core_focus": ["Nuln State Troops", "Nuln Veteran Outriders"],
                    "special_focus": ["Great Cannon", "Mortar"],
                    "rare_focus": ["Helblaster Volley Gun", "Steam Tank"],
                    "character_types": ["Empire Engineer", "Captain of the Empire"]
                },
                {
                    "name": "Combined Arms",
                    "core_focus": ["Nuln Veteran State Troops", "Free Company Militia"],
                    "special_focus": ["Empire Knights", "Empire Greatswords"],
                    "rare_focus": ["Great Cannon"],
                    "character_types": ["General of the Empire", "Captain of the Empire"]
                },
                {
                    "name": "Gunpowder Elite",
                    "core_focus": ["Nuln State Troops", "Nuln Veteran State Troops"],
                    "special_focus": ["Great Cannon", "Mortar"],
                    "rare_focus": ["Helblaster Volley Gun", "Helstorm Rocket Battery"],
                    "character_types": ["Empire Engineer", "Master Mage"]
                }
            ]
    
    def generate_random_army(self, points_limit: int = 2000, template: Optional[str] = None) -> TOWArmy:
        """Generate a random but viable army"""
        army = TOWArmy(self.faction, points_limit=points_limit)
        
        # Choose template
        if template:
            army_template = next((t for t in self.army_templates if t["name"] == template), None)
        else:
            army_template = random.choice(self.army_templates)
        
        army.name = army_template["name"] if army_template else "Random Army"
        
        # Add mandatory general
        general_options = [unit for unit in self.unit_database.values() 
                          if unit.category == UnitCategory.CHARACTER and unit.leadership >= 8]
        if general_options:
            general = random.choice(general_options)
            army_unit = ArmyUnit(general, 1)
            army.add_unit(army_unit)
        
        # Build army using template preferences
        attempts = 0
        while army.remaining_points > 50 and attempts < 100:
            attempts += 1
            
            # Determine what category to add based on current composition
            core_points = sum(u.total_cost for u in army.units if u.unit_profile.category == UnitCategory.CORE)
            min_core_needed = points_limit * 0.25 - core_points
            
            if min_core_needed > 0:
                # Need more core units
                self._add_template_unit(army, army_template, "core_focus", min_core_needed)
            else:
                # Can add other types
                category_weights = {
                    "core_focus": 30,
                    "special_focus": 40,
                    "rare_focus": 20,
                    "character_types": 10
                }
                
                category = random.choices(
                    list(category_weights.keys()),
                    weights=list(category_weights.values())
                )[0]
                
                self._add_template_unit(army, army_template, category, army.remaining_points)
        
        return army
    
    def _add_template_unit(self, army: TOWArmy, template: Dict, category_key: str, max_points: float):
        """Add a unit from template category"""
        if category_key not in template:
            return
        
        preferred_units = template[category_key]
        available_units = [unit for name, unit in self.unit_database.items() 
                          if name in preferred_units and unit.points_per_model * unit.min_unit_size <= max_points]
        
        if not available_units:
            # Fallback to any units in the right category
            if category_key == "character_types":
                available_units = [unit for unit in self.unit_database.values() 
                                 if unit.category == UnitCategory.CHARACTER and 
                                 unit.points_per_model <= max_points]
            elif category_key == "core_focus":
                available_units = [unit for unit in self.unit_database.values() 
                                 if unit.category == UnitCategory.CORE and 
                                 unit.points_per_model * unit.min_unit_size <= max_points]
            elif category_key == "special_focus":
                available_units = [unit for unit in self.unit_database.values() 
                                 if unit.category == UnitCategory.SPECIAL and 
                                 unit.points_per_model * unit.min_unit_size <= max_points]
            elif category_key == "rare_focus":
                available_units = [unit for unit in self.unit_database.values() 
                                 if unit.category == UnitCategory.RARE and 
                                 unit.points_per_model * unit.min_unit_size <= max_points]
        
        if available_units:
            chosen_unit = random.choice(available_units)
            
            # Determine unit size
            max_affordable = int(max_points // chosen_unit.points_per_model)
            unit_size = min(max_affordable, chosen_unit.max_unit_size)
            unit_size = max(unit_size, chosen_unit.min_unit_size)
            
            if unit_size >= chosen_unit.min_unit_size:
                army_unit = ArmyUnit(chosen_unit, unit_size)
                
                # Add some equipment randomly
                if chosen_unit.weapon_options and random.random() < 0.3:
                    option = random.choice(chosen_unit.weapon_options)
                    if (army_unit.total_cost + option.cost_per_model * unit_size <= max_points):
                        army_unit.equipment_choices.append(option.name)
                
                # Add command randomly
                if chosen_unit.can_have_champion and random.random() < 0.5:
                    if army_unit.total_cost + chosen_unit.champion_cost <= max_points:
                        army_unit.has_champion = True
                
                army.add_unit(army_unit)
    
    def optimize_army(self, base_army: TOWArmy, generations: int = 50) -> TOWArmy:
        """Use genetic algorithm to optimize army composition"""
        population_size = 20
        mutation_rate = 0.3
        
        # Create initial population
        population = [base_army]
        for _ in range(population_size - 1):
            mutated = self._mutate_army(copy.deepcopy(base_army))
            population.append(mutated)
        
        for generation in range(generations):
            # Evaluate fitness (simplified - could be based on battle results)
            fitness_scores = [self._evaluate_army_fitness(army) for army in population]
            
            # Select best armies for breeding
            sorted_armies = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            elite = [army for army, _ in sorted_armies[:population_size//2]]
            
            # Create next generation
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover_armies(parent1, parent2)
                
                if random.random() < mutation_rate:
                    child = self._mutate_army(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best army
        final_fitness = [self._evaluate_army_fitness(army) for army in population]
        best_army = population[final_fitness.index(max(final_fitness))]
        return best_army
    
    def _evaluate_army_fitness(self, army: TOWArmy) -> float:
        """Evaluate army fitness (simplified heuristic)"""
        score = 0.0
        
        # Penalty for invalid armies
        is_valid, _ = army.validate_army()
        if not is_valid:
            return -1000
        
        # Reward for using points efficiently
        efficiency = army.total_points / army.points_limit
        score += efficiency * 100
        
        # Reward for balanced composition
        core_ratio = sum(u.total_cost for u in army.units if u.unit_profile.category == UnitCategory.CORE) / army.total_points
        special_ratio = sum(u.total_cost for u in army.units if u.unit_profile.category == UnitCategory.SPECIAL) / army.total_points
        
        # Ideal ratios (rough heuristic)
        ideal_core = 0.4
        ideal_special = 0.3
        
        score += 50 - abs(core_ratio - ideal_core) * 200
        score += 50 - abs(special_ratio - ideal_special) * 200
        
        # Reward for unit diversity
        unique_units = len(set(unit.unit_profile.name for unit in army.units))
        score += unique_units * 10
        
        return score
    
    def _mutate_army(self, army: TOWArmy) -> TOWArmy:
        """Mutate army by changing units/equipment"""
        mutated = copy.deepcopy(army)
        
        mutation_type = random.choice(["add_unit", "remove_unit", "change_size", "change_equipment"])
        
        if mutation_type == "add_unit" and len(mutated.units) < 15:
            # Try to add a random unit
            available_units = list(self.unit_database.values())
            for _ in range(10):  # Try 10 times
                unit = random.choice(available_units)
                army_unit = ArmyUnit(unit, unit.min_unit_size)
                if mutated.add_unit(army_unit):
                    break
        
        elif mutation_type == "remove_unit" and len(mutated.units) > 1:
            # Remove random unit (not the general)
            non_general_units = [i for i, unit in enumerate(mutated.units) 
                               if unit.unit_profile.leadership < 9]
            if non_general_units:
                mutated.remove_unit(random.choice(non_general_units))
        
        elif mutation_type == "change_size" and mutated.units:
            # Change unit size
            unit = random.choice(mutated.units)
            if unit.unit_profile.max_unit_size > unit.unit_profile.min_unit_size:
                new_size = random.randint(unit.unit_profile.min_unit_size, 
                                        min(unit.unit_profile.max_unit_size, 
                                            unit.unit_size + 5))
                old_cost = unit.total_cost
                unit.unit_size = new_size
                if mutated.total_points <= mutated.points_limit:
                    pass  # Keep change
                else:
                    unit.unit_size = old_cost  # Revert
        
        return mutated
    
    def _crossover_armies(self, parent1: TOWArmy, parent2: TOWArmy) -> TOWArmy:
        """Create child army from two parents"""
        child = TOWArmy(self.faction, points_limit=parent1.points_limit)
        
        # Combine units from both parents
        all_units = parent1.units + parent2.units
        random.shuffle(all_units)
        
        for unit in all_units:
            if child.remaining_points >= unit.total_cost:
                child.add_unit(copy.deepcopy(unit))
        
        return child

def create_sample_armies():
    """Create sample armies for testing"""
    print("🏛️ Creating Sample TOW Armies")
    print("=" * 50)
    
    # Orc army
    orc_builder = TOWArmyBuilder(Faction.ORC_GOBLIN_TRIBES)
    orc_army = orc_builder.generate_random_army()
    print(orc_army.get_army_summary())
    
    print("\n" + "="*50 + "\n")
    
    # Nuln army
    nuln_builder = TOWArmyBuilder(Faction.CITY_STATE_NULN)
    nuln_army = nuln_builder.generate_random_army()
    print(nuln_army.get_army_summary())
    
    return orc_army, nuln_army

if __name__ == "__main__":
    create_sample_armies() 