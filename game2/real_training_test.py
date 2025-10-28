#!/usr/bin/env python3
"""
Real training test with actual battles to verify replay fix
"""

from fixed_improved_training import FixedImprovedAI, FixedTrainingBattle
import time

def real_training_test():
    """Test real training with actual battles"""
    print("🎯 REAL TRAINING TEST")
    print("=" * 30)
    
    # Create AIs
    print("Creating training AIs...")
    orc_ai = FixedImprovedAI(state_size=50, action_size=15)
    empire_ai = FixedImprovedAI(state_size=50, action_size=15)
    empire_ai.epsilon = 0.1  # Lower exploration for empire
    
    print(f"Orc AI epsilon: {orc_ai.epsilon}")
    print(f"Empire AI epsilon: {empire_ai.epsilon}")
    
    # Run some training battles
    print("\n🔥 Running training battles...")
    battle = FixedTrainingBattle()
    
    wins = 0
    for i in range(10):  # Small test
        print(f"  Battle {i+1}/10...", end=" ")
        
        try:
            result = battle.run_battle(orc_ai, empire_ai, 'orc', 'empire')
            winner = result.get('winner', 'draw')
            
            if winner == 'orc':
                wins += 1
                print("🟢 ORC WIN")
            elif winner == 'empire':
                print("🔵 Empire win")  
            else:
                print("⚪ Draw")
                
            # Test replay after each battle
            if len(orc_ai.memory) > 32:
                print(f"    Training Orc AI (memory: {len(orc_ai.memory)})...", end=" ")
                orc_ai.replay(batch_size=32)  # This should work now!
                print("✅")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    
    win_rate = (wins / 10) * 100
    print(f"\n📊 Results:")
    print(f"   Orc wins: {wins}/10 ({win_rate:.1f}%)")
    print(f"   Orc memory: {len(orc_ai.memory)} experiences")
    print(f"   Empire memory: {len(empire_ai.memory)} experiences")
    
    # Test saving
    print(f"\n💾 Testing save...")
    try:
        orc_ai.save_enhanced_model("test_trained_orc.pth")
        print("✅ Save successful!")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False
    
    print(f"\n🎉 REAL TRAINING TEST PASSED!")
    return True

if __name__ == "__main__":
    real_training_test() 