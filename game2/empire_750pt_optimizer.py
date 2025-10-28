#!/usr/bin/env python3
"""
🏛️ EMPIRE 750-POINT TOURNAMENT OPTIMIZER
Advanced army composition analysis with all restrictions
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools

@dataclass
class Unit:
    name: str
    category: str  # 'core', 'special', 'rare', 'character'
    base_cost: int
    min_size: int
    max_size: int
    cost_per_model: int
    upgrades: Dict[str, int]
    special_rules: List[str]
    battlefield_role: str
    effectiveness: float  # 1-10 rating
    restrictions: List[str] = None

@dataclass
class ArmyComposition:
    units: List[Tuple[str, int, List[str]]]  # (unit_name, size, upgrades)
    total_points: int
    core_points: int
    special_points: int
    rare_points: int
    character_points: int
    effectiveness_score: float
    synergy_bonus: float

class Empire750Optimizer:
    """Optimize Empire armies for 750-point tournaments"""
    
    def __init__(self):
        self.max_points = 750
        self.max_character = 188  # 25%
        self.max_core_unit = 263  # 35%
        self.max_special_unit = 225  # 30%
        self.max_rare_unit = 188  # 25%
        self.max_0_per_1000 = 2
        
        self.empire_units = self._create_empire_database()
    
    def _create_empire_database(self):
        """Complete Empire unit database with costs and stats"""
        units = {
            # CHARACTERS
            'General of the Empire': Unit(
                name='General of the Empire',
                category='character',
                base_cost=80,
                min_size=1, max_size=1,
                cost_per_model=0,
                upgrades={
                    'Heavy Armour': 6,
                    'Shield': 3,
                    'Great Weapon': 6,
                    'Additional Hand Weapon': 3,
                    'Warhorse': 18,
                    'Magic Weapon': 25,
                    'Magic Armour': 25
                },
                special_rules=['Leadership 9', 'General'],
                battlefield_role='Leadership',
                effectiveness=8.5
            ),
            
            'Captain of the Empire': Unit(
                name='Captain of the Empire',
                category='character',
                base_cost=50,
                min_size=1, max_size=1,
                cost_per_model=0,
                upgrades={
                    'Heavy Armour': 6,
                    'Shield': 3,
                    'Great Weapon': 6,
                    'Warhorse': 18,
                    'Magic Weapon': 25
                },
                special_rules=['Leadership 8'],
                battlefield_role='Leadership',
                effectiveness=7.0
            ),
            
            'Wizard': Unit(
                name='Wizard',
                category='character',
                base_cost=65,
                min_size=1, max_size=1,
                cost_per_model=0,
                upgrades={
                    'Level 2': 35,
                    'Dispel Scroll': 25,
                    'Power Stone': 20
                },
                special_rules=['Magic User'],
                battlefield_role='Magic',
                effectiveness=8.0
            ),
            
            # CORE UNITS
            'State Troops': Unit(
                name='State Troops',
                category='core',
                base_cost=0,
                min_size=10, max_size=40,
                cost_per_model=5,
                upgrades={
                    'Spears': 1,
                    'Light Armour': 1,
                    'Shields': 1,
                    'Champion': 10,
                    'Standard Bearer': 10,
                    'Musician': 5
                },
                special_rules=['State Troops'],
                battlefield_role='Infantry Block',
                effectiveness=7.5
            ),
            
            'Spearmen': Unit(
                name='Spearmen',
                category='core',
                base_cost=0,
                min_size=10, max_size=40,
                cost_per_model=4,
                upgrades={
                    'Light Armour': 1,
                    'Shields': 1,
                    'Champion': 10,
                    'Standard Bearer': 10,
                    'Musician': 5
                },
                special_rules=['Spears', 'State Troops'],
                battlefield_role='Infantry Block',
                effectiveness=7.0
            ),
            
            'Crossbowmen': Unit(
                name='Crossbowmen',
                category='core',
                base_cost=0,
                min_size=10, max_size=20,
                cost_per_model=9,
                upgrades={
                    'Champion': 10,
                    'Standard Bearer': 10,
                    'Musician': 5
                },
                special_rules=['Crossbows', 'State Troops'],
                battlefield_role='Ranged',
                effectiveness=8.0
            ),
            
            'Handgunners': Unit(
                name='Handgunners',
                category='core',
                base_cost=0,
                min_size=10, max_size=20,
                cost_per_model=10,
                upgrades={
                    'Champion': 10,
                    'Standard Bearer': 10,
                    'Musician': 5,
                    'Hochland Long Rifle': 20
                },
                special_rules=['Handguns', 'State Troops'],
                battlefield_role='Ranged',
                effectiveness=8.5
            ),
            
            # SPECIAL UNITS
            'Greatswords': Unit(
                name='Greatswords',
                category='special',
                base_cost=0,
                min_size=10, max_size=25,
                cost_per_model=10,
                upgrades={
                    'Champion': 15,
                    'Standard Bearer': 15,
                    'Musician': 10
                },
                special_rules=['Great Weapons', 'Stubborn'],
                battlefield_role='Elite Infantry',
                effectiveness=8.5
            ),
            
            'Knights': Unit(
                name='Knights',
                category='special',
                base_cost=0,
                min_size=5, max_size=12,
                cost_per_model=22,
                upgrades={
                    'Inner Circle': 3,
                    'Champion': 20,
                    'Standard Bearer': 20,
                    'Musician': 10
                },
                special_rules=['Heavy Cavalry', 'Lances'],
                battlefield_role='Heavy Cavalry',
                effectiveness=9.0
            ),
            
            'Pistoliers': Unit(
                name='Pistoliers',
                category='special',
                base_cost=0,
                min_size=5, max_size=10,
                cost_per_model=16,
                upgrades={
                    'Champion': 12,
                    'Musician': 8,
                    'Repeater Pistol': 3
                },
                special_rules=['Fast Cavalry', 'Pistols'],
                battlefield_role='Fast Cavalry',
                effectiveness=7.5
            ),
            
            'Great Cannon': Unit(
                name='Great Cannon',
                category='special',
                base_cost=100,
                min_size=1, max_size=1,
                cost_per_model=0,
                upgrades={},
                special_rules=['War Machine', 'Artillery'],
                battlefield_role='Artillery',
                effectiveness=8.0
            ),
            
            # RARE UNITS
            'Steam Tank': Unit(
                name='Steam Tank',
                category='rare',
                base_cost=300,
                min_size=1, max_size=1,
                cost_per_model=0,
                upgrades={},
                special_rules=['Steam Points', 'Terror', 'Unbreakable'],
                battlefield_role='Monster',
                effectiveness=9.5,
                restrictions=['0-1 per 1000 points']
            ),
            
            'Helblaster Volley Gun': Unit(
                name='Helblaster Volley Gun',
                category='rare',
                base_cost=110,
                min_size=1, max_size=1,
                cost_per_model=0,
                upgrades={},
                special_rules=['War Machine', 'Multiple Shots'],
                battlefield_role='Artillery',
                effectiveness=8.5
            )
        }
        
        return units
    
    def calculate_unit_cost(self, unit_name: str, size: int, upgrades: List[str]) -> int:
        """Calculate total cost for a unit with upgrades"""
        unit = self.empire_units[unit_name]
        cost = unit.base_cost + (unit.cost_per_model * size)
        
        for upgrade in upgrades:
            if upgrade in unit.upgrades:
                if upgrade in ['Level 2']:  # Per-model upgrades for characters
                    cost += unit.upgrades[upgrade]
                elif upgrade in ['Champion', 'Standard Bearer', 'Musician']:  # Fixed upgrades
                    cost += unit.upgrades[upgrade]
                else:  # Per-model upgrades
                    cost += unit.upgrades[upgrade] * size
        
        return cost
    
    def is_valid_army(self, composition: List[Tuple[str, int, List[str]]]) -> Tuple[bool, str]:
        """Check if army composition is valid under tournament rules"""
        total_points = 0
        core_points = 0
        special_points = 0
        rare_points = 0
        character_points = 0
        restricted_count = 0
        
        has_general = False
        
        for unit_name, size, upgrades in composition:
            unit = self.empire_units[unit_name]
            unit_cost = self.calculate_unit_cost(unit_name, size, upgrades)
            
            total_points += unit_cost
            
            # Check category limits
            if unit.category == 'core':
                core_points += unit_cost
                if unit_cost > self.max_core_unit:
                    return False, f"{unit_name} exceeds core unit limit ({unit_cost} > {self.max_core_unit})"
            elif unit.category == 'special':
                special_points += unit_cost
                if unit_cost > self.max_special_unit:
                    return False, f"{unit_name} exceeds special unit limit ({unit_cost} > {self.max_special_unit})"
            elif unit.category == 'rare':
                rare_points += unit_cost
                if unit_cost > self.max_rare_unit:
                    return False, f"{unit_name} exceeds rare unit limit ({unit_cost} > {self.max_rare_unit})"
            elif unit.category == 'character':
                character_points += unit_cost
                if unit_cost > self.max_character:
                    return False, f"{unit_name} exceeds character limit ({unit_cost} > {self.max_character})"
                if unit_name == 'General of the Empire':
                    has_general = True
            
            # Check 0-X per 1000 restrictions
            if unit.restrictions and '0-1 per 1000 points' in unit.restrictions:
                restricted_count += 1
        
        # Validation checks
        if total_points > self.max_points:
            return False, f"Army exceeds point limit ({total_points} > {self.max_points})"
        
        if not has_general:
            return False, "Army must include a General"
        
        if core_points < total_points * 0.25:
            return False, f"Insufficient core units ({core_points} < {total_points * 0.25})"
        
        if restricted_count > self.max_0_per_1000:
            return False, f"Too many 0-X per 1000 units ({restricted_count} > {self.max_0_per_1000})"
        
        return True, "Valid army composition"
    
    def calculate_effectiveness(self, composition: List[Tuple[str, int, List[str]]]) -> float:
        """Calculate army effectiveness score"""
        total_effectiveness = 0
        synergy_bonus = 0
        
        unit_roles = []
        
        for unit_name, size, upgrades in composition:
            unit = self.empire_units[unit_name]
            
            # Base effectiveness
            effectiveness = unit.effectiveness
            
            # Size bonus for infantry blocks
            if unit.battlefield_role == 'Infantry Block' and size >= 20:
                effectiveness += 0.5
            if size >= 30:
                effectiveness += 0.5
                
            # Upgrade bonuses
            if 'Champion' in upgrades:
                effectiveness += 0.3
            if 'Standard Bearer' in upgrades:
                effectiveness += 0.2
            if 'Musician' in upgrades:
                effectiveness += 0.2
            
            total_effectiveness += effectiveness
            unit_roles.append(unit.battlefield_role)
        
        # Synergy bonuses
        role_count = {}
        for role in unit_roles:
            role_count[role] = role_count.get(role, 0) + 1
        
        # Combined arms bonus
        if 'Infantry Block' in role_count and 'Ranged' in role_count:
            synergy_bonus += 1.0
        if 'Leadership' in role_count and 'Infantry Block' in role_count:
            synergy_bonus += 0.5
        if 'Heavy Cavalry' in role_count and 'Infantry Block' in role_count:
            synergy_bonus += 0.8
        if 'Artillery' in role_count:
            synergy_bonus += 0.5
        
        return total_effectiveness + synergy_bonus
    
    def generate_optimal_armies(self, num_armies: int = 5) -> List[ArmyComposition]:
        """Generate optimal army compositions"""
        print('🏛️ EMPIRE 750-POINT TOURNAMENT OPTIMIZER')
        print('=' * 60)
        print('⚔️ Analyzing optimal army compositions...')
        print()
        
        # Core army templates - FIXED for tournament restrictions
        templates = [
            # Template 1: Balanced Combined Arms (FIXED)
            [
                ('General of the Empire', 1, ['Heavy Armour', 'Shield']),
                ('State Troops', 23, ['Spears', 'Light Armour', 'Shields', 'Champion', 'Standard Bearer', 'Musician']),
                ('Handgunners', 12, ['Champion']),
                ('Knights', 5, ['Champion', 'Standard Bearer']),
                ('Great Cannon', 1, [])
            ],
            
            # Template 2: Infantry Horde (FIXED)
            [
                ('General of the Empire', 1, ['Heavy Armour']),
                ('State Troops', 25, ['Spears', 'Light Armour', 'Shields', 'Champion', 'Standard Bearer']),
                ('Spearmen', 22, ['Light Armour', 'Shields', 'Champion', 'Standard Bearer']),
                ('Crossbowmen', 12, ['Champion']),
                ('Greatswords', 10, ['Champion'])
            ],
            
            # Template 3: Elite Strike Force (FIXED)
            [
                ('General of the Empire', 1, ['Heavy Armour', 'Magic Weapon']),
                ('State Troops', 18, ['Spears', 'Light Armour', 'Champion', 'Standard Bearer']),
                ('Knights', 6, ['Champion', 'Standard Bearer']),
                ('Greatswords', 15, ['Champion', 'Standard Bearer', 'Musician']),
                ('Crossbowmen', 10, ['Champion'])
            ],
            
            # Template 4: Artillery Support (FIXED)
            [
                ('General of the Empire', 1, ['Heavy Armour', 'Shield']),
                ('Wizard', 1, ['Level 2', 'Dispel Scroll']),
                ('State Troops', 20, ['Spears', 'Light Armour', 'Champion', 'Standard Bearer']),
                ('Crossbowmen', 10, ['Champion']),
                ('Great Cannon', 1, []),
                ('Helblaster Volley Gun', 1, [])
            ],
            
            # Template 5: Mobile Force (Original - worked)
            [
                ('General of the Empire', 1, ['Heavy Armour', 'Warhorse']),
                ('State Troops', 20, ['Spears', 'Light Armour', 'Champion', 'Standard Bearer']),
                ('Knights', 6, ['Champion', 'Standard Bearer']),
                ('Pistoliers', 8, ['Champion', 'Musician', 'Repeater Pistol']),
                ('Handgunners', 12, ['Champion', 'Musician'])
            ],
            
            # Template 6: Defensive Castle
            [
                ('General of the Empire', 1, ['Heavy Armour', 'Magic Armour']),
                ('State Troops', 30, ['Spears', 'Light Armour', 'Shields', 'Champion', 'Standard Bearer', 'Musician']),
                ('Crossbowmen', 15, ['Champion', 'Standard Bearer', 'Musician']),
                ('Handgunners', 10, ['Hochland Long Rifle']),
                ('Great Cannon', 1, [])
            ],
            
            # Template 7: Combined Magic Support
            [
                ('General of the Empire', 1, ['Heavy Armour']),
                ('Wizard', 1, ['Level 2', 'Power Stone']),
                ('State Troops', 25, ['Spears', 'Light Armour', 'Champion', 'Standard Bearer']),
                ('Knights', 5, ['Champion']),
                ('Greatswords', 12, ['Champion', 'Standard Bearer']),
                ('Crossbowmen', 10, [])
            ],
            
            # Template 8: Elite Compact Force
            [
                ('General of the Empire', 1, ['Heavy Armour', 'Shield', 'Magic Weapon']),
                ('State Troops', 20, ['Spears', 'Light Armour', 'Shields', 'Champion', 'Standard Bearer']),
                ('Greatswords', 18, ['Champion', 'Standard Bearer', 'Musician']),
                ('Handgunners', 15, ['Champion', 'Musician']),
                ('Knights', 5, ['Champion'])
            ]
        ]
        
        valid_armies = []
        
        for i, template in enumerate(templates, 1):
            is_valid, message = self.is_valid_army(template)
            
            if is_valid:
                total_cost = sum(self.calculate_unit_cost(name, size, upgrades) 
                               for name, size, upgrades in template)
                effectiveness = self.calculate_effectiveness(template)
                
                # Calculate category breakdowns
                core_points = sum(self.calculate_unit_cost(name, size, upgrades) 
                                for name, size, upgrades in template 
                                if self.empire_units[name].category == 'core')
                special_points = sum(self.calculate_unit_cost(name, size, upgrades) 
                                   for name, size, upgrades in template 
                                   if self.empire_units[name].category == 'special')
                rare_points = sum(self.calculate_unit_cost(name, size, upgrades) 
                                for name, size, upgrades in template 
                                if self.empire_units[name].category == 'rare')
                character_points = sum(self.calculate_unit_cost(name, size, upgrades) 
                                     for name, size, upgrades in template 
                                     if self.empire_units[name].category == 'character')
                
                army = ArmyComposition(
                    units=template,
                    total_points=total_cost,
                    core_points=core_points,
                    special_points=special_points,
                    rare_points=rare_points,
                    character_points=character_points,
                    effectiveness_score=effectiveness,
                    synergy_bonus=0
                )
                
                valid_armies.append(army)
                
                print(f'✅ Template {i}: {message} ({total_cost} points, {effectiveness:.1f} effectiveness)')
            else:
                print(f'❌ Template {i}: {message}')
        
        # Sort by effectiveness
        valid_armies.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        return valid_armies[:num_armies]
    
    def display_army(self, army: ArmyComposition, rank: int):
        """Display detailed army composition"""
        print(f'\n🏆 RANK {rank} ARMY COMPOSITION')
        print('=' * 45)
        print(f'📊 Total Points: {army.total_points}/750')
        print(f'⚡ Effectiveness Score: {army.effectiveness_score:.1f}')
        print()
        
        print('🏛️ ARMY LIST:')
        print('-' * 25)
        
        category_totals = {'character': 0, 'core': 0, 'special': 0, 'rare': 0}
        
        for unit_name, size, upgrades in army.units:
            unit = self.empire_units[unit_name]
            cost = self.calculate_unit_cost(unit_name, size, upgrades)
            category_totals[unit.category] += cost
            
            print(f'• {unit_name}', end='')
            if size > 1:
                print(f' ({size} models)', end='')
            
            if upgrades:
                print(f' with {", ".join(upgrades)}', end='')
            
            print(f' - {cost} points')
        
        print()
        print('📋 CATEGORY BREAKDOWN:')
        print(f'👑 Characters: {category_totals["character"]} points ({category_totals["character"]/750*100:.1f}%)')
        print(f'⚔️ Core: {category_totals["core"]} points ({category_totals["core"]/750*100:.1f}%)')
        print(f'🛡️ Special: {category_totals["special"]} points ({category_totals["special"]/750*100:.1f}%)')
        print(f'🔥 Rare: {category_totals["rare"]} points ({category_totals["rare"]/750*100:.1f}%)')
        
        # Tactical analysis
        print()
        print('🎯 TACTICAL ANALYSIS:')
        roles = []
        for unit_name, _, _ in army.units:
            roles.append(self.empire_units[unit_name].battlefield_role)
        
        role_count = {}
        for role in roles:
            role_count[role] = role_count.get(role, 0) + 1
        
        for role, count in role_count.items():
            print(f'• {role}: {count} unit(s)')
        
        print()

def main():
    """Run the Empire 750pt optimizer"""
    optimizer = Empire750Optimizer()
    optimal_armies = optimizer.generate_optimal_armies(3)
    
    print(f'\n🏛️ TOP 3 OPTIMAL EMPIRE ARMIES (750 POINTS)')
    print('=' * 60)
    
    for i, army in enumerate(optimal_armies, 1):
        optimizer.display_army(army, i)
    
    print('\n✅ OPTIMIZATION COMPLETE!')
    print('🎯 These armies are optimized for tournament play!')

if __name__ == '__main__':
    main() 