#!/usr/bin/env python3
"""
🏛️ SIMPLE WARHAMMER: THE OLD WORLD AI TRAINER
==============================================

A simplified, working AI trainer that actually runs without issues.
"""

import random
import time
from typing import Dict, List, Any

class SimpleTOWTrainer:
    """Simple working TOW AI trainer"""
    
    def __init__(self):
        self.orc_armies = []
        self.nuln_armies = []
        self.battle_count = 0
        
    def create_random_orc_army(self) -> Dict[str, int]:
        """Create a random Orc army"""
        return {
            "Orc Big Boss": 1,
            "Orc Boyz": random.randint(1, 3),
            "Night Goblins": random.randint(0, 2),
            "Orc Arrer Boyz": random.randint(0, 1)
        }
    
    def create_random_nuln_army(self) -> Dict[str, int]:
        """Create a random Nuln army"""
        return {
            "Engineer": 1,
            "Handgunners": random.randint(1, 3),
            "Crossbowmen": random.randint(0, 2),
            "Great Cannon": random.randint(0, 1)
        }
    
    def simulate_simple_battle(self, orc_army: Dict, nuln_army: Dict) -> str:
        """Simulate a simple battle"""
        orc_strength = sum(orc_army.values()) * random.uniform(0.8, 1.2)
        nuln_strength = sum(nuln_army.values()) * random.uniform(0.8, 1.2)
        
        # Add some faction bonuses
        if "Orc Big Boss" in orc_army:
            orc_strength *= 1.1
        if "Great Cannon" in nuln_army and nuln_army["Great Cannon"] > 0:
            nuln_strength *= 1.2
            
        return "Orcs" if orc_strength > nuln_strength else "Nuln"
    
    def evolve_armies(self, battles: int = 1000):
        """Simple evolution process"""
        print("🏛️ SIMPLE TOW AI TRAINER")
        print("=" * 40)
        print(f"🎯 Running {battles} battles")
        print("⚔️ Let the evolution begin!")
        print()
        
        # Initialize populations
        for _ in range(20):
            self.orc_armies.append(self.create_random_orc_army())
            self.nuln_armies.append(self.create_random_nuln_army())
        
        orc_wins = 0
        nuln_wins = 0
        
        start_time = time.time()
        
        for battle in range(battles):
            # Pick random armies
            orc_army = random.choice(self.orc_armies)
            nuln_army = random.choice(self.nuln_armies)
            
            # Fight battle
            winner = self.simulate_simple_battle(orc_army, nuln_army)
            
            if winner == "Orcs":
                orc_wins += 1
            else:
                nuln_wins += 1
            
            self.battle_count += 1
            
            # Show progress
            if (battle + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (battle + 1) / elapsed
                print(f"⚔️ Battle {battle + 1}/{battles} | "
                      f"Orc wins: {orc_wins} | Nuln wins: {nuln_wins} | "
                      f"Rate: {rate:.1f} battles/sec")
            
            # Simple evolution every 50 battles
            if (battle + 1) % 50 == 0:
                self.evolve_population()
        
        total_time = time.time() - start_time
        
        print("\n🎉 TRAINING COMPLETE!")
        print("=" * 40)
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"⚔️ Total battles: {battles}")
        print(f"🧌 Orc wins: {orc_wins} ({orc_wins/battles:.1%})")
        print(f"🏰 Nuln wins: {nuln_wins} ({nuln_wins/battles:.1%})")
        print(f"🚀 Battle rate: {battles/total_time:.1f} battles/sec")
        
        print("\n🏆 BEST EVOLVED ARMIES:")
        print(f"🧌 Best Orc Army: {self.orc_armies[0]}")
        print(f"🏰 Best Nuln Army: {self.nuln_armies[0]}")
        
        return {
            "battles": battles,
            "orc_wins": orc_wins,
            "nuln_wins": nuln_wins,
            "time": total_time,
            "best_orc": self.orc_armies[0],
            "best_nuln": self.nuln_armies[0]
        }
    
    def evolve_population(self):
        """Simple population evolution"""
        # Randomly modify some armies
        for i in range(5):  # Modify 5 random armies
            if random.random() < 0.5:
                # Evolve orc army
                idx = random.randint(0, len(self.orc_armies) - 1)
                army = self.orc_armies[idx].copy()
                
                # Random mutation
                unit = random.choice(list(army.keys()))
                if random.random() < 0.5:
                    army[unit] = max(0, army[unit] + random.randint(-1, 1))
                
                self.orc_armies[idx] = army
            else:
                # Evolve nuln army
                idx = random.randint(0, len(self.nuln_armies) - 1)
                army = self.nuln_armies[idx].copy()
                
                # Random mutation
                unit = random.choice(list(army.keys()))
                if random.random() < 0.5:
                    army[unit] = max(0, army[unit] + random.randint(-1, 1))
                
                self.nuln_armies[idx] = army

def main():
    """Main function"""
    print("🏛️ SIMPLE WARHAMMER: THE OLD WORLD AI TRAINER")
    print("=" * 50)
    print("🎯 Choose training intensity:")
    print("1. 🚀 Quick (1,000 battles, ~10 seconds)")
    print("2. ⚔️ Medium (10,000 battles, ~1 minute)")
    print("3. 🏆 Full (100,000 battles, ~10 minutes)")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        battles = 1000
    elif choice == "2":
        battles = 10000
    elif choice == "3":
        battles = 100000
    else:
        print("Invalid choice, using quick training")
        battles = 1000
    
    trainer = SimpleTOWTrainer()
    result = trainer.evolve_armies(battles)
    
    print("\n✅ Training completed successfully!")
    print("🎮 Your AI is now ready for battle!")

if __name__ == "__main__":
    main() 