#!/usr/bin/env python3

import sys
from authentic_nuln_simulator import NulnArmyBuilder, generate_army_templates, generate_enemy_armies

def test_effectiveness():
    """Test effectiveness calculations to debug battle simulation"""
    print("🔍 BATTLE EFFECTIVENESS DEBUG TEST")
    print("="*50)
    
    builder = NulnArmyBuilder()
    templates = generate_army_templates()
    enemy_armies = generate_enemy_armies()
    
    # Test Nuln army effectiveness
    nuln_army = templates[0]  # First template
    nuln_effectiveness = builder.calculate_effectiveness(nuln_army)
    
    print(f"🏛️ NULN ARMY EFFECTIVENESS:")
    print(f"   Army: {nuln_army}")
    print(f"   Total Effectiveness: {nuln_effectiveness:.2f}")
    print()
    
    # Test enemy army effectiveness calculations
    print(f"⚔️ ENEMY ARMY EFFECTIVENESS:")
    
    # Sample enemy army
    test_enemy = enemy_armies["Orc Horde"]
    print(f"   Army: {test_enemy}")
    
    # Calculate enemy effectiveness using same logic as simulate_battle
    enemy_effectiveness_map = {
        "Big Boss": 3.5, "Warlord": 3.5, "Lord": 4.0, "Prince": 4.5, "Noble": 3.0, "Thane": 3.5,
        "Oldblood": 4.0, "Highborn": 3.8, "Warboss": 3.2,
        "Warriors": 2.0, "Boyz": 1.8, "Clanrats": 1.5, "Spearmen": 2.2, "Men-at-Arms": 1.8,
        "Skeleton": 1.3, "Zombies": 1.0, "Temple Guard": 2.8, "White Lions": 2.5, "Hammerers": 2.4,
        "Knights": 3.2, "Silver Helms": 2.8, "Black Knights": 3.0, "Boar Boyz": 2.5,
        "Archers": 1.8, "Crossbowmen": 1.9, "Quarrellers": 2.0, "Glade Guard": 2.1,
        "Cannon": 3.0, "Organ Gun": 2.8, "Rock Lobber": 2.5, "Trebuchet": 2.7,
        "Bolt Thrower": 2.3, "Ratling Gun": 2.2, "Warpfire": 2.0
    }
    
    effectiveness2 = 0
    for unit in test_enemy:
        unit_name = unit.split('(')[0].strip()
        
        if '(' in unit and ')' in unit:
            try:
                count = int(unit.split('(')[1].split(')')[0])
            except:
                count = 1
        else:
            count = 1
        
        unit_effectiveness = 1.5  # Default
        for key, value in enemy_effectiveness_map.items():
            if key.lower() in unit_name.lower():
                unit_effectiveness = value
                break
        
        unit_total = unit_effectiveness * (count ** 0.8)
        effectiveness2 += unit_total
        print(f"      {unit}: base {unit_effectiveness:.1f} × {count}^0.8 = {unit_total:.2f}")
    
    print(f"   Subtotal: {effectiveness2:.2f}")
    
    # Apply bonuses
    original = effectiveness2
    if "Cannon" in str(test_enemy) or "Gun" in str(test_enemy):
        effectiveness2 *= 1.15
        print(f"   Artillery bonus: {original:.2f} → {effectiveness2:.2f}")
    
    print(f"   Final Enemy Effectiveness: {effectiveness2:.2f}")
    print()
    
    print(f"📊 COMPARISON:")
    print(f"   Nuln Army: {nuln_effectiveness:.2f}")
    print(f"   Enemy Army: {effectiveness2:.2f}")
    print(f"   Ratio: {nuln_effectiveness/effectiveness2:.2f}:1")
    
    if nuln_effectiveness > effectiveness2:
        print(f"   ✅ Nuln should win most battles")
    else:
        print(f"   ❌ Enemy should win most battles")

if __name__ == "__main__":
    test_effectiveness() 