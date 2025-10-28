#!/usr/bin/env python3
"""
Simple test of the Warhammer: The Old World battle system
Tests the core mechanics without web interface
"""

import sys
import json
from tow_web_battle import TOWBattle

def test_battle_system():
    """Test the core battle system"""
    print("🧪 TESTING WARHAMMER: THE OLD WORLD BATTLE SYSTEM")
    print("=" * 60)
    
    # Create battle instance
    battle = TOWBattle()
    
    # Test battle state serialization
    print("📊 Testing battle state serialization...")
    try:
        state = battle.get_battle_state()
        json_str = json.dumps(state, indent=2)
        print("✅ Battle state serialization: SUCCESS")
        print(f"   Units: {len(state['units'])}")
        print(f"   Turn: {state['turn']}")
        print(f"   Phase: {state['phase']}")
    except Exception as e:
        print(f"❌ Battle state serialization: FAILED - {e}")
        return False
    
    # Test unit methods
    print("\n🔧 Testing unit methods...")
    test_unit = battle.units[0] if battle.units else None
    if test_unit:
        try:
            # Test missing methods
            charge_dist = test_unit.calculate_charge_distance()
            print(f"✅ calculate_charge_distance: {charge_dist}")
            
            # Test can_see_target with a dummy target
            if len(battle.units) > 1:
                target = battle.units[1]
                can_see = test_unit.can_see_target(target)
                print(f"✅ can_see_target: {can_see}")
            
            # Test to_dict
            unit_dict = test_unit.to_dict()
            json.dumps(unit_dict)  # Test JSON serialization
            print("✅ Unit to_dict serialization: SUCCESS")
            
        except Exception as e:
            print(f"❌ Unit methods: FAILED - {e}")
            return False
    
    # Test AI state
    print("\n🤖 Testing AI state...")
    try:
        ai_state = battle.get_ai_state()
        print(f"✅ AI state generation: SUCCESS (shape: {ai_state.shape})")
    except Exception as e:
        print(f"❌ AI state generation: FAILED - {e}")
        return False
    
    # Test battle phases (without threading)
    print("\n⚔️ Testing battle phases...")
    try:
        # Test psychology phase
        battle.psychology_phase()
        print("✅ Psychology phase: SUCCESS")
        
        # Test movement phase
        battle.ai_movement_phase()
        print("✅ Movement phase: SUCCESS")
        
        # Test shooting phase
        battle.ai_shooting_phase()
        print("✅ Shooting phase: SUCCESS")
        
        # Test charge phase
        battle.charge_phase()
        print("✅ Charge phase: SUCCESS")
        
    except Exception as e:
        print(f"❌ Battle phases: FAILED - {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("The battle system is working correctly.")
    return True

if __name__ == "__main__":
    success = test_battle_system()
    sys.exit(0 if success else 1) 