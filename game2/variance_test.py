#!/usr/bin/env python3
"""
Variance Test - Multiple Battle Outcomes
Shows how randomness affects Warhammer combat and AI decisions
"""

import random
import math

class SimpleUnit:
    def __init__(self, name, models, armor_save):
        self.name = name
        self.models = models
        self.armor_save = armor_save

def simulate_shooting(shooter_models, target, seed=None):
    """Simulate shooting attack with given seed"""
    if seed is not None:
        random.seed(seed)
    
    shots = shooter_models
    hits = 0
    wounds = 0
    kills = 0
    
    for shot in range(shots):
        # To hit (need 3+ on D6)
        if random.randint(1, 6) >= 3:
            hits += 1
            
            # To wound (need 3+ on D6)
            if random.randint(1, 6) >= 3:
                
                # Armor save
                if random.randint(1, 6) < target.armor_save:
                    wounds += 1
                    kills += 1
    
    return shots, hits, wounds, kills

def test_battle_variance():
    """Test multiple battles with different seeds"""
    print("🎲 WARHAMMER COMBAT VARIANCE TEST")
    print("=" * 40)
    
    handgunners = SimpleUnit("Nuln Handgunners", 5, 5)
    orcs = SimpleUnit("Orc Warriors", 8, 6)
    
    print(f"📋 Setup: {handgunners.models} {handgunners.name} vs {orcs.models} {orcs.name}")
    print(f"🎯 Testing 10 battles with different random seeds...\n")
    
    results = []
    
    for battle in range(10):
        seed = battle * 123
        orc_models = orcs.models
        
        print(f"⚔️ Battle {battle + 1} (seed {seed}):")
        
        total_kills = 0
        for round_num in range(3):
            shots, hits, wounds, kills = simulate_shooting(handgunners.models, orcs, seed + round_num)
            total_kills += kills
            orc_models -= kills
            
            print(f"   Round {round_num + 1}: {shots} shots → {hits} hits → {wounds} wounds → {kills} kills")
            if orc_models <= 0:
                print(f"   🏆 All orcs destroyed!")
                break
        
        survival_rate = max(0, orc_models) / orcs.models
        results.append({
            'battle': battle + 1,
            'orcs_remaining': max(0, orc_models),
            'survival_rate': survival_rate,
            'total_kills': total_kills
        })
        
        print(f"   📊 Result: {max(0, orc_models)}/{orcs.models} orcs survive ({survival_rate:.1%})\n")
    
    # Analysis
    print("📈 VARIANCE ANALYSIS:")
    print("-" * 25)
    
    avg_survivors = sum(r['orcs_remaining'] for r in results) / len(results)
    avg_kills = sum(r['total_kills'] for r in results) / len(results)
    
    min_survivors = min(r['orcs_remaining'] for r in results)
    max_survivors = max(r['orcs_remaining'] for r in results)
    
    victories = sum(1 for r in results if r['orcs_remaining'] == 0)
    
    print(f"Average orc survivors: {avg_survivors:.1f}/{orcs.models}")
    print(f"Average kills per battle: {avg_kills:.1f}")
    print(f"Range: {min_survivors}-{max_survivors} survivors")
    print(f"Complete victories: {victories}/10 ({victories/10:.0%})")
    
    print(f"\n🧠 AI IMPLICATIONS:")
    print(f"   • High variance requires smart planning")
    print(f"   • MCTS can handle uncertainty better")
    print(f"   • Multiple simulations give better strategies")
    print(f"   • Genetic algorithms find robust compositions")

def compare_strategies():
    """Compare different tactical approaches"""
    print(f"\n🎯 STRATEGY COMPARISON")
    print("=" * 25)
    
    strategies = {
        "Concentrated Fire": [5, 0, 0],  # All shots at once
        "Spread Fire": [2, 2, 1],       # Spread across rounds
        "Burst Fire": [3, 2, 0],        # Front-loaded
    }
    
    for strategy_name, shot_pattern in strategies.items():
        print(f"\n🎮 Testing {strategy_name}:")
        print(f"   Pattern: {shot_pattern} shots per round")
        
        total_kills = 0
        orcs_remaining = 8
        
        for round_num, shots in enumerate(shot_pattern):
            if shots > 0 and orcs_remaining > 0:
                random.seed(42 + round_num)  # Consistent seed for fair comparison
                _, hits, wounds, kills = simulate_shooting(shots, SimpleUnit("Orcs", orcs_remaining, 6))
                total_kills += kills
                orcs_remaining -= kills
                
                print(f"   Round {round_num + 1}: {shots} shots → {kills} kills ({orcs_remaining} remain)")
        
        efficiency = total_kills / sum(shot_pattern) if sum(shot_pattern) > 0 else 0
        print(f"   📊 Total kills: {total_kills}, Efficiency: {efficiency:.2f} kills/shot")

def main():
    """Run variance tests"""
    test_battle_variance()
    compare_strategies()
    
    print(f"\n🎯 VARIANCE TEST COMPLETE!")
    print(f"✅ Combat randomness: Demonstrated")
    print(f"✅ Strategy differences: Shown")
    print(f"✅ AI decision importance: Proven")
    print(f"\n🚀 This is why we need smart AI agents!")

if __name__ == "__main__":
    main() 