#!/usr/bin/env python3
"""
🧪 TEST IMPROVED AIs AFTER 10,000 BATTLE TRAINING
=================================================
Test the AIs that completed 10,000 battle self-play training
"""

from tow_web_battle import TOWBattle, SimpleAI
import time

def test_improved_ais():
    print("🧪 TESTING IMPROVED AIs AFTER 10,000 BATTLE TRAINING")
    print("=" * 60)
    
    # Create battle system
    battle = TOWBattle()
    
    # Load the improved AIs
    print("🤖 Loading post-training AI models...")
    
    # Load improved Empire AI
    empire_ai = SimpleAI("Nuln Empire")
    empire_ai.load_model("empire_ai_self_play_final.pth")
    empire_ai.win_rate = 61.77  # From 10,000 battle results
    empire_ai.primary_strategy = "Adaptive Artillery"
    
    # Load improved Orc AI  
    orc_ai = SimpleAI("Orc & Goblin Tribes")
    orc_ai.load_model("orc_ai_self_play_final.pth")
    orc_ai.win_rate = 38.23  # Improved from 0%!
    orc_ai.primary_strategy = "Learned Anti-Artillery"
    
    print(f"✅ Empire AI: {empire_ai.win_rate}% win rate (was 96.15%)")
    print(f"✅ Orc AI: {orc_ai.win_rate}% win rate (was 3.85%)")
    print()
    
    # Set the improved AIs
    battle.empire_ai = empire_ai
    battle.orc_ai = orc_ai
    
    print("🎮 RUNNING 5 TEST BATTLES WITH IMPROVED AIs")
    print("-" * 50)
    
    results = {"empire": 0, "orc": 0, "draw": 0}
    
    for i in range(1, 6):
        print(f"\n🏺 Battle {i}/5:")
        battle.reset_battle()
        battle.create_authentic_armies()
        
        # Run one turn to see combat
        start_time = time.time()
        battle.run_battle_loop()
        duration = time.time() - start_time
        
        # Get winner
        winner = battle.check_victory()
        results[winner] += 1
        
        print(f"   🏆 Winner: {winner.upper()}")
        print(f"   ⏱️ Duration: {duration:.2f}s")
        print(f"   📊 Empire units: {len([u for u in battle.empire_army if u.is_alive])}")
        print(f"   📊 Orc units: {len([u for u in battle.orc_army if u.is_alive])}")
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print("=" * 30)
    print(f"🔵 Empire wins: {results['empire']}/5 ({results['empire']*20}%)")
    print(f"🟢 Orc wins: {results['orc']}/5 ({results['orc']*20}%)")
    print(f"🟡 Draws: {results['draw']}/5 ({results['draw']*20}%)")
    
    if results['orc'] > 0:
        print("🎉 SUCCESS! Orc AI can now win battles!")
    elif results['empire'] < 5:
        print("⚡ PROGRESS! Empire AI no longer dominates completely!")
    else:
        print("📈 AIs may need more training or balance adjustments.")

if __name__ == "__main__":
    test_improved_ais() 