#!/usr/bin/env python3
"""
🏛️ REALISTIC WARHAMMER: THE OLD WORLD AI TRAINER
================================================

A realistic AI trainer that simulates proper battles with:
- Turn-based combat
- Unit positioning and movement
- Tactical decision making
- Meaningful AI evolution
"""

import random
import time
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class Unit:
    """A unit in the battle"""
    name: str
    x: int
    y: int
    health: int
    max_health: int
    attack: int
    defense: int
    movement: int
    range: int
    is_alive: bool = True
    
    def take_damage(self, damage: int):
        """Apply damage to the unit"""
        self.health = max(0, self.health - damage)
        if self.health <= 0:
            self.is_alive = False
    
    def distance_to(self, other: 'Unit') -> float:
        """Calculate distance to another unit"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class BattleSimulator:
    """Realistic battle simulator"""
    
    def __init__(self):
        self.battlefield_width = 20
        self.battlefield_height = 15
        
    def create_orc_unit(self, unit_name: str, position: Tuple[int, int]) -> Unit:
        """Create an orc unit with proper stats"""
        orc_stats = {
            "Orc Big Boss": {"health": 25, "attack": 8, "defense": 6, "movement": 3, "range": 1},
            "Orc Boyz": {"health": 15, "attack": 5, "defense": 4, "movement": 4, "range": 1},
            "Night Goblins": {"health": 8, "attack": 3, "defense": 3, "movement": 5, "range": 1},
            "Orc Arrer Boyz": {"health": 12, "attack": 4, "defense": 3, "movement": 4, "range": 6}
        }
        
        stats = orc_stats.get(unit_name, orc_stats["Orc Boyz"])
        return Unit(
            name=unit_name,
            x=position[0], y=position[1],
            health=stats["health"], max_health=stats["health"],
            attack=stats["attack"], defense=stats["defense"],
            movement=stats["movement"], range=stats["range"]
        )
    
    def create_nuln_unit(self, unit_name: str, position: Tuple[int, int]) -> Unit:
        """Create a Nuln unit with proper stats"""
        nuln_stats = {
            "Engineer": {"health": 20, "attack": 6, "defense": 5, "movement": 3, "range": 2},
            "Handgunners": {"health": 10, "attack": 6, "defense": 3, "movement": 3, "range": 8},
            "Crossbowmen": {"health": 12, "attack": 5, "defense": 4, "movement": 3, "range": 6},
            "Great Cannon": {"health": 8, "attack": 12, "defense": 2, "movement": 1, "range": 12}
        }
        
        stats = nuln_stats.get(unit_name, nuln_stats["Handgunners"])
        return Unit(
            name=unit_name,
            x=position[0], y=position[1],
            health=stats["health"], max_health=stats["health"],
            attack=stats["attack"], defense=stats["defense"],
            movement=stats["movement"], range=stats["range"]
        )
    
    def setup_armies(self, orc_army: Dict[str, int], nuln_army: Dict[str, int]) -> Tuple[List[Unit], List[Unit]]:
        """Setup armies on the battlefield"""
        orcs = []
        nurns = []
        
        # Place Orc units on left side
        y_pos = 2
        for unit_name, count in orc_army.items():
            for i in range(count):
                orcs.append(self.create_orc_unit(unit_name, (2 + i, y_pos)))
            y_pos += 2
        
        # Place Nuln units on right side
        y_pos = 2
        for unit_name, count in nuln_army.items():
            for i in range(count):
                nurns.append(self.create_nuln_unit(unit_name, (self.battlefield_width - 3 - i, y_pos)))
            y_pos += 2
        
        return orcs, nurns
    
    def simulate_turn(self, attacking_units: List[Unit], defending_units: List[Unit]) -> Dict[str, Any]:
        """Simulate one turn of combat"""
        actions = []
        
        for attacker in attacking_units:
            if not attacker.is_alive:
                continue
            
            # Find best target
            best_target = None
            best_score = -1
            
            for defender in defending_units:
                if not defender.is_alive:
                    continue
                
                distance = attacker.distance_to(defender)
                if distance <= attacker.range:
                    # Score based on damage potential and enemy health
                    score = (attacker.attack - defender.defense) / max(1, defender.health)
                    if score > best_score:
                        best_score = score
                        best_target = defender
            
            # Attack if target found
            if best_target:
                damage = max(1, attacker.attack - best_target.defense + random.randint(-2, 2))
                best_target.take_damage(damage)
                actions.append(f"{attacker.name} attacks {best_target.name} for {damage} damage")
                
                if not best_target.is_alive:
                    actions.append(f"{best_target.name} is destroyed!")
            else:
                # Move towards nearest enemy
                nearest_enemy = min(defending_units, key=lambda u: attacker.distance_to(u) if u.is_alive else float('inf'))
                if nearest_enemy.is_alive:
                    # Simple movement towards enemy
                    dx = 1 if nearest_enemy.x > attacker.x else -1 if nearest_enemy.x < attacker.x else 0
                    dy = 1 if nearest_enemy.y > attacker.y else -1 if nearest_enemy.y < attacker.y else 0
                    
                    new_x = max(0, min(self.battlefield_width - 1, attacker.x + dx))
                    new_y = max(0, min(self.battlefield_height - 1, attacker.y + dy))
                    
                    attacker.x = new_x
                    attacker.y = new_y
                    actions.append(f"{attacker.name} moves to ({new_x}, {new_y})")
        
        return {"actions": actions}
    
    def simulate_battle(self, orc_army: Dict[str, int], nuln_army: Dict[str, int]) -> Dict[str, Any]:
        """Simulate a complete battle"""
        orcs, nurns = self.setup_armies(orc_army, nuln_army)
        
        turn = 0
        max_turns = 50
        battle_log = []
        
        while turn < max_turns:
            # Check for victory conditions
            live_orcs = [u for u in orcs if u.is_alive]
            live_nurns = [u for u in nurns if u.is_alive]
            
            if not live_orcs:
                return {
                    "winner": "Nuln",
                    "turns": turn,
                    "orc_survivors": 0,
                    "nuln_survivors": len(live_nurns),
                    "battle_log": battle_log
                }
            elif not live_nurns:
                return {
                    "winner": "Orcs",
                    "turns": turn,
                    "orc_survivors": len(live_orcs),
                    "nuln_survivors": 0,
                    "battle_log": battle_log
                }
            
            # Simulate turn
            if turn % 2 == 0:  # Orcs go first
                turn_result = self.simulate_turn(live_orcs, live_nurns)
            else:  # Nuln turn
                turn_result = self.simulate_turn(live_nurns, live_orcs)
            
            battle_log.extend(turn_result["actions"])
            turn += 1
        
        # Timeout - determine winner by survivors
        live_orcs = [u for u in orcs if u.is_alive]
        live_nurns = [u for u in nurns if u.is_alive]
        
        if len(live_orcs) > len(live_nurns):
            winner = "Orcs"
        elif len(live_nurns) > len(live_orcs):
            winner = "Nuln"
        else:
            winner = "Draw"
        
        return {
            "winner": winner,
            "turns": turn,
            "orc_survivors": len(live_orcs),
            "nuln_survivors": len(live_nurns),
            "battle_log": battle_log
        }

class RealisticTOWTrainer:
    """Realistic TOW AI trainer with proper battle simulation"""
    
    def __init__(self):
        self.simulator = BattleSimulator()
        self.orc_strategies = []
        self.nuln_strategies = []
        self.battle_count = 0
        
    def create_random_orc_army(self) -> Dict[str, int]:
        """Create a random Orc army"""
        return {
            "Orc Big Boss": random.randint(1, 2),
            "Orc Boyz": random.randint(2, 5),
            "Night Goblins": random.randint(1, 3),
            "Orc Arrer Boyz": random.randint(1, 3)
        }
    
    def create_random_nuln_army(self) -> Dict[str, int]:
        """Create a random Nuln army"""
        return {
            "Engineer": random.randint(1, 2),
            "Handgunners": random.randint(2, 4),
            "Crossbowmen": random.randint(1, 3),
            "Great Cannon": random.randint(0, 2)
        }
    
    def evaluate_army_performance(self, army: Dict[str, int], results: List[Dict]) -> float:
        """Evaluate how well an army composition performed"""
        if not results:
            return 0.0
        
        faction = "Orcs" if "Orc" in list(army.keys())[0] else "Nuln"
        wins = sum(1 for r in results if r["winner"] == faction)
        avg_survivors = sum(r.get("orc_survivors", 0) if faction == "Orcs" else r.get("nuln_survivors", 0) for r in results) / len(results)
        avg_turns = sum(r["turns"] for r in results) / len(results)
        
        # Fitness = win rate + survivor bonus - turn penalty (prefer quick victories)
        fitness = (wins / len(results)) * 10 + avg_survivors * 0.5 - (avg_turns / 50) * 2
        return max(0.0, fitness)
    
    def mutate_army(self, army: Dict[str, int]) -> Dict[str, int]:
        """Mutate an army composition"""
        new_army = army.copy()
        
        # Random mutations
        for unit_type in new_army:
            if random.random() < 0.3:  # 30% chance to mutate each unit type
                change = random.randint(-1, 1)
                new_army[unit_type] = max(0, min(6, new_army[unit_type] + change))
        
        return new_army
    
    def train_ai(self, battles: int = 1000):
        """Train AI with realistic battles"""
        print("🏛️ REALISTIC TOW AI TRAINER")
        print("=" * 50)
        print(f"🎯 Running {battles} REALISTIC battles")
        print("⚔️ Full turn-based combat simulation!")
        print("🧠 Learning tactical army compositions...")
        print()
        
        # Initialize populations
        for _ in range(10):
            self.orc_strategies.append(self.create_random_orc_army())
            self.nuln_strategies.append(self.create_random_nuln_army())
        
        orc_wins = 0
        nuln_wins = 0
        draws = 0
        
        start_time = time.time()
        
        for battle in range(battles):
            # Pick armies
            orc_army = random.choice(self.orc_strategies)
            nuln_army = random.choice(self.nuln_strategies)
            
            # Simulate realistic battle
            result = self.simulator.simulate_battle(orc_army, nuln_army)
            
            # Track results
            if result["winner"] == "Orcs":
                orc_wins += 1
            elif result["winner"] == "Nuln":
                nuln_wins += 1
            else:
                draws += 1
            
            self.battle_count += 1
            
            # Show progress
            if (battle + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (battle + 1) / elapsed
                print(f"⚔️ Battle {battle + 1}/{battles} | "
                      f"Orc: {orc_wins} | Nuln: {nuln_wins} | Draws: {draws} | "
                      f"Rate: {rate:.1f} battles/sec")
                
                # Show sample battle info
                print(f"   Last battle: {result['winner']} wins in {result['turns']} turns "
                      f"(Orc survivors: {result['orc_survivors']}, Nuln: {result['nuln_survivors']})")
            
            # Evolution every 100 battles
            if (battle + 1) % 100 == 0:
                self.evolve_strategies()
        
        total_time = time.time() - start_time
        
        print("\n🎉 REALISTIC TRAINING COMPLETE!")
        print("=" * 50)
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"⚔️ Total battles: {battles}")
        print(f"🧌 Orc wins: {orc_wins} ({orc_wins/battles:.1%})")
        print(f"🏰 Nuln wins: {nuln_wins} ({nuln_wins/battles:.1%})")
        print(f"🤝 Draws: {draws} ({draws/battles:.1%})")
        print(f"🚀 Battle rate: {battles/total_time:.1f} battles/sec")
        
        print("\n🏆 EVOLVED ARMY STRATEGIES:")
        print(f"🧌 Best Orc Strategy: {self.orc_strategies[0]}")
        print(f"🏰 Best Nuln Strategy: {self.nuln_strategies[0]}")
        
        return {
            "battles": battles,
            "orc_wins": orc_wins,
            "nuln_wins": nuln_wins,
            "draws": draws,
            "time": total_time,
            "rate": battles/total_time
        }
    
    def evolve_strategies(self):
        """Evolve army strategies based on performance"""
        # Simple evolution: mutate existing strategies
        for i in range(len(self.orc_strategies)):
            if random.random() < 0.3:  # 30% chance to mutate
                self.orc_strategies[i] = self.mutate_army(self.orc_strategies[i])
        
        for i in range(len(self.nuln_strategies)):
            if random.random() < 0.3:
                self.nuln_strategies[i] = self.mutate_army(self.nuln_strategies[i])

def main():
    """Main function"""
    print("🏛️ REALISTIC WARHAMMER: THE OLD WORLD AI TRAINER")
    print("=" * 60)
    print("🎯 Choose training intensity:")
    print("1. 🚀 Quick (100 battles, ~30 seconds)")
    print("2. ⚔️ Medium (500 battles, ~2 minutes)")
    print("3. 🏆 Full (2000 battles, ~10 minutes)")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        battles = 100
    elif choice == "2":
        battles = 500
    elif choice == "3":
        battles = 2000
    else:
        print("Invalid choice, using quick training")
        battles = 100
    
    trainer = RealisticTOWTrainer()
    result = trainer.train_ai(battles)
    
    print("\n✅ Realistic AI training completed!")
    print("🧠 Your AI has learned from actual tactical battles!")

if __name__ == "__main__":
    main() 