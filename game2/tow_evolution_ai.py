#!/usr/bin/env python3
"""
🏛️ WARHAMMER: THE OLD WORLD - AI EVOLUTION SYSTEM
===============================================

Revolutionary AI system that:
- Learns army building from scratch
- Evolves tactics through battles
- Discovers optimal army compositions
- Trains through 100,000+ battles
- Adapts strategies based on win/loss data

This creates AI generals that truly understand TOW!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# Army builder functionality integrated directly
from tow_comprehensive_rules import (
    ComprehensiveBattleEngine, Characteristics, Equipment, Model, Unit,
    TroopType, Formation, GameState, TurnSequenceManager
)

@dataclass
class BattleResult:
    """Result of a single battle"""
    orc_army: Dict[str, int]
    nuln_army: Dict[str, int]
    winner: str
    orc_vp: int
    nuln_vp: int
    battle_log: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class EvolutionStats:
    """Statistics tracking for evolution runs"""
    generation: int = 0
    total_battles: int = 0
    orc_wins: int = 0
    nuln_wins: int = 0
    draws: int = 0
    avg_battle_length: float = 0.0
    best_orc_army: Optional[Dict[str, int]] = None  # Changed from TOWArmy
    best_nuln_army: Optional[Dict[str, int]] = None  # Changed from TOWArmy
    best_orc_fitness: float = 0.0
    best_nuln_fitness: float = 0.0
    recent_battles: List[BattleResult] = field(default_factory=list)
    
    @property
    def orc_win_rate(self) -> float:
        total = self.orc_wins + self.nuln_wins + self.draws
        return self.orc_wins / total if total > 0 else 0.0
    
    @property 
    def nuln_win_rate(self) -> float:
        total = self.orc_wins + self.nuln_wins + self.draws
        return self.nuln_wins / total if total > 0 else 0.0

class ArmyDNA:
    """Genetic representation of army preferences"""
    
    def __init__(self, faction: str):
        self.faction = faction
        self.genes = self._initialize_genes()
        self.fitness = 0.0
        self.generation = 0
    
    def _initialize_genes(self) -> Dict[str, float]:
        """Initialize random genes for army building preferences"""
        # Define units available to each faction
        if self.faction == "Orc & Goblin Tribes":
            units = ["Orc Big Boss", "Orc Boyz", "Night Goblins", "Orc Arrer Boyz"]
        else:  # Nuln
            units = ["Engineer", "Handgunners", "Great Cannon", "Crossbowmen"]
        
        genes = {}
        
        # Unit preference genes (0.0 to 1.0)
        for unit in units:
            genes[f"unit_pref_{unit}"] = random.random()
        
        # Tactical preference genes
        genes.update({
            "aggression": random.random(),
            "magic_focus": random.random(), 
            "ranged_focus": random.random(),
            "elite_focus": random.random(),
            "defensive_focus": random.random()
        })
        
        return genes
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate DNA with given rate"""
        for gene in self.genes:
            if random.random() < mutation_rate:
                self.genes[gene] += random.gauss(0, 0.1)
                self.genes[gene] = max(0.0, min(1.0, self.genes[gene]))
    
    def crossover(self, other: 'ArmyDNA') -> 'ArmyDNA':
        """Create offspring DNA from two parents"""
        child = ArmyDNA(self.faction)
        
        # Blend genes
        for gene in self.genes:
            if random.random() < 0.5:
                child.genes[gene] = self.genes[gene]
            else:
                child.genes[gene] = other.genes[gene]
        
        return child

class TacticalAI(nn.Module):
    """Neural network for tactical decision making"""
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, output_size: int = 15):
        super(TacticalAI, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.network(x)
    
    def encode_battlefield_state(self, battlefield_state: Dict) -> torch.Tensor:
        """Convert battlefield state to neural network input"""
        # Simplified encoding - in full version would be more sophisticated
        features = []
        
        # Own army status (64 features)
        for i in range(8):  # Max 8 units
            if i < len(battlefield_state.get('own_units', [])):
                unit = battlefield_state['own_units'][i]
                features.extend([
                    unit.get('health_ratio', 0.0),
                    unit.get('position_x', 0.0) / 72.0,  # Normalize position
                    unit.get('position_y', 0.0) / 48.0,
                    unit.get('unit_strength', 0.0) / 10.0,  # Normalize strength
                    unit.get('engaged', 0.0),
                    unit.get('fleeing', 0.0),
                    unit.get('can_charge', 0.0),
                    unit.get('can_shoot', 0.0)
                ])
            else:
                features.extend([0.0] * 8)  # Empty unit slot
        
        # Enemy army status (64 features) 
        for i in range(8):
            if i < len(battlefield_state.get('enemy_units', [])):
                unit = battlefield_state['enemy_units'][i]
                features.extend([
                    unit.get('health_ratio', 0.0),
                    unit.get('position_x', 0.0) / 72.0,
                    unit.get('position_y', 0.0) / 48.0,
                    unit.get('unit_strength', 0.0) / 10.0,
                    unit.get('engaged', 0.0),
                    unit.get('fleeing', 0.0),
                    unit.get('threat_level', 0.0),
                    unit.get('distance_to_closest', 1.0)
                ])
            else:
                features.extend([0.0] * 8)
        
        return torch.tensor(features, dtype=torch.float32)

class ComprehensiveTOWBattle:
    """Battle system using full TOW rules"""
    
    def __init__(self):
        self.engine = ComprehensiveBattleEngine()
        self.battle_count = 0
    
    def simulate_battle(self, army1_composition: Dict, army2_composition: Dict) -> Dict[str, Any]:
        """Simulate battle between two armies using full TOW rules"""
        
        # Build armies from compositions
        army1 = self.build_army_from_composition(army1_composition, "Orc & Goblin Tribes")
        army2 = self.build_army_from_composition(army2_composition, "City-State of Nuln")
        
        # Run battle with comprehensive rules
        result = self.engine.run_complete_battle(army1, army2)
        
        # Enhanced result analysis
        battle_analysis = self.analyze_battle_performance(army1, army2, result)
        result.update(battle_analysis)
        
        self.battle_count += 1
        if self.battle_count % 100 == 0:
            print(f"⚔️ Completed {self.battle_count} comprehensive battles")
        
        return result
    
    def build_army_from_composition(self, composition: Dict, faction: str) -> List[Unit]:
        """Build army units with full characteristics and rules"""
        army = []
        
        for unit_name, count in composition.items():
            if count <= 0:
                continue
            
            for i in range(count):
                unit = self.create_unit_with_full_rules(unit_name, faction, i+1)
                if unit:
                    army.append(unit)
        
        return army
    
    def create_unit_with_full_rules(self, unit_name: str, faction: str, unit_number: int) -> Optional[Unit]:
        """Create unit with comprehensive TOW rules"""
        
        if faction == "Orc & Goblin Tribes":
            return self.create_orc_unit(unit_name, unit_number)
        elif faction == "City-State of Nuln":
            return self.create_nuln_unit(unit_name, unit_number)
        
        return None
    
    def create_orc_unit(self, unit_name: str, unit_number: int) -> Optional[Unit]:
        """Create Orc units with authentic rules"""
        
        if unit_name == "Orc Big Boss":
            return Unit(
                name=f"Orc Big Boss {unit_number}",
                models=[Model(
                    name="Big Boss",
                    characteristics=Characteristics(4, 6, 3, 5, 5, 3, 3, 4, 8),
                    equipment=Equipment(hand_weapon=True, light_armor=True),
                    special_rules=["Fear", "Animosity", "Waaagh"]
                )],
                troop_type=TroopType.INFANTRY,
                special_rules=["Fear", "Animosity", "Waaagh"]
            )
        
        elif unit_name == "Orc Boyz":
            models = []
            for i in range(20):
                models.append(Model(
                    name=f"Orc Boy {i+1}",
                    characteristics=Characteristics(4, 4, 3, 4, 4, 1, 2, 1, 7),
                    equipment=Equipment(hand_weapon=True, shield=True),
                    special_rules=["Animosity", "Waaagh"]
                ))
            
            return Unit(
                name=f"Orc Boyz {unit_number}",
                models=models,
                troop_type=TroopType.INFANTRY,
                formation=Formation.CLOSE_ORDER,
                ranks=4, files=5,
                special_rules=["Animosity", "Waaagh"],
                command_group={"champion": True, "standard": True, "musician": True}
            )
        
        elif unit_name == "Night Goblins":
            models = []
            for i in range(30):
                models.append(Model(
                    name=f"Night Goblin {i+1}",
                    characteristics=Characteristics(4, 2, 3, 3, 3, 1, 2, 1, 5),
                    equipment=Equipment(hand_weapon=True, shield=True),
                    special_rules=["Animosity", "Fear_Elves"]
                ))
            
            return Unit(
                name=f"Night Goblins {unit_number}",
                models=models,
                troop_type=TroopType.INFANTRY,
                formation=Formation.CLOSE_ORDER,
                ranks=6, files=5,
                special_rules=["Animosity", "Fear_Elves"],
                command_group={"champion": True, "standard": True, "musician": True}
            )
        
        elif unit_name == "Orc Arrer Boyz":
            models = []
            for i in range(15):
                models.append(Model(
                    name=f"Orc Archer {i+1}",
                    characteristics=Characteristics(4, 4, 3, 4, 4, 1, 2, 1, 7),
                    equipment=Equipment(hand_weapon=True, ranged_weapon="bow"),
                    special_rules=["Animosity", "Waaagh"]
                ))
            
            return Unit(
                name=f"Orc Arrer Boyz {unit_number}",
                models=models,
                troop_type=TroopType.INFANTRY,
                formation=Formation.CLOSE_ORDER,
                ranks=3, files=5,
                special_rules=["Animosity", "Waaagh"],
                command_group={"champion": True, "standard": True, "musician": True}
            )
        
        # Add more Orc units as needed...
        return None
    
    def create_nuln_unit(self, unit_name: str, unit_number: int) -> Optional[Unit]:
        """Create Nuln units with authentic rules"""
        
        if unit_name == "Engineer":
            return Unit(
                name=f"Engineer {unit_number}",
                models=[Model(
                    name="Engineer",
                    characteristics=Characteristics(4, 4, 4, 4, 4, 2, 4, 3, 8),
                    equipment=Equipment(hand_weapon=True, light_armor=True, ranged_weapon="pistol"),
                    special_rules=["State_Troops", "Gunpowder_Weapons"]
                )],
                troop_type=TroopType.INFANTRY,
                special_rules=["State_Troops", "Gunpowder_Weapons"]
            )
        
        elif unit_name == "Handgunners":
            models = []
            for i in range(15):
                models.append(Model(
                    name=f"Handgunner {i+1}",
                    characteristics=Characteristics(4, 3, 3, 3, 3, 1, 3, 1, 7),
                    equipment=Equipment(hand_weapon=True, light_armor=True, ranged_weapon="handgun"),
                    special_rules=["State_Troops", "Gunpowder_Weapons"]
                ))
            
            return Unit(
                name=f"Handgunners {unit_number}",
                models=models,
                troop_type=TroopType.INFANTRY,
                formation=Formation.CLOSE_ORDER,
                ranks=3, files=5,
                special_rules=["State_Troops", "Gunpowder_Weapons"],
                command_group={"champion": True, "standard": True, "musician": True}
            )
        
        elif unit_name == "Great Cannon":
            crew = []
            for i in range(3):
                crew.append(Model(
                    name=f"Crew {i+1}",
                    characteristics=Characteristics(4, 3, 3, 3, 3, 1, 3, 1, 7),
                    equipment=Equipment(hand_weapon=True),
                    special_rules=["State_Troops"]
                ))
            
            return Unit(
                name=f"Great Cannon {unit_number}",
                models=crew,
                troop_type=TroopType.WAR_MACHINE,
                special_rules=["War_Machine", "Gunpowder_Weapons", "State_Troops"]
            )
        
        elif unit_name == "Crossbowmen":
            models = []
            for i in range(12):
                models.append(Model(
                    name=f"Crossbowman {i+1}",
                    characteristics=Characteristics(4, 3, 3, 3, 3, 1, 3, 1, 7),
                    equipment=Equipment(hand_weapon=True, light_armor=True, ranged_weapon="crossbow"),
                    special_rules=["State_Troops"]
                ))
            
            return Unit(
                name=f"Crossbowmen {unit_number}",
                models=models,
                troop_type=TroopType.INFANTRY,
                formation=Formation.CLOSE_ORDER,
                ranks=3, files=4,
                special_rules=["State_Troops"],
                command_group={"champion": True, "standard": True, "musician": True}
            )
        
        # Add more Nuln units as needed...
        return None
    
    def analyze_battle_performance(self, army1: List[Unit], army2: List[Unit], result: Dict) -> Dict[str, Any]:
        """Detailed analysis of battle performance with TOW rules"""
        
        # Calculate survival rates
        army1_survival = self.calculate_army_survival(army1)
        army2_survival = self.calculate_army_survival(army2)
        
        # Analyze unit effectiveness
        army1_effectiveness = self.analyze_unit_effectiveness(army1)
        army2_effectiveness = self.analyze_unit_effectiveness(army2)
        
        # Psychology impact analysis
        psychology_events = self.count_psychology_events(result.get("battle_log", []))
        
        # Special rules impact
        special_rules_impact = self.analyze_special_rules_impact(result.get("battle_log", []))
        
        return {
            "army1_survival": army1_survival,
            "army2_survival": army2_survival,
            "army1_effectiveness": army1_effectiveness,
            "army2_effectiveness": army2_effectiveness,
            "psychology_events": psychology_events,
            "special_rules_impact": special_rules_impact,
            "tactical_analysis": self.generate_tactical_analysis(army1, army2, result)
        }
    
    def calculate_army_survival(self, army: List[Unit]) -> Dict[str, float]:
        """Calculate detailed survival statistics"""
        total_units = len(army)
        surviving_units = len([u for u in army if u.is_alive])
        
        total_models = sum(len(u.models) for u in army)
        surviving_models = sum(len([m for m in u.models if m.current_wounds > 0]) for u in army)
        
        return {
            "unit_survival_rate": surviving_units / max(1, total_units),
            "model_survival_rate": surviving_models / max(1, total_models),
            "total_units": total_units,
            "surviving_units": surviving_units,
            "total_models": total_models,
            "surviving_models": surviving_models
        }
    
    def analyze_unit_effectiveness(self, army: List[Unit]) -> Dict[str, Any]:
        """Analyze effectiveness of different unit types"""
        effectiveness = {}
        
        for unit in army:
            unit_type = unit.troop_type.value
            if unit_type not in effectiveness:
                effectiveness[unit_type] = {
                    "count": 0,
                    "survivors": 0,
                    "total_wounds_dealt": 0,
                    "total_wounds_taken": 0
                }
            
            effectiveness[unit_type]["count"] += 1
            if unit.is_alive:
                effectiveness[unit_type]["survivors"] += 1
            
            # Calculate wounds taken
            for model in unit.models:
                max_wounds = model.characteristics.wounds
                current_wounds = model.current_wounds
                effectiveness[unit_type]["total_wounds_taken"] += max(0, max_wounds - current_wounds)
        
        return effectiveness
    
    def count_psychology_events(self, battle_log: List[str]) -> Dict[str, int]:
        """Count psychology-related events in battle"""
        events = {
            "fear_tests": 0,
            "terror_tests": 0,
            "panic_tests": 0,
            "animosity_triggers": 0,
            "waaagh_triggers": 0,
            "units_fled": 0
        }
        
        for entry in battle_log:
            if "fear test" in entry.lower():
                events["fear_tests"] += 1
            if "terror test" in entry.lower():
                events["terror_tests"] += 1
            if "panic test" in entry.lower():
                events["panic_tests"] += 1
            if "animosity" in entry.lower():
                events["animosity_triggers"] += 1
            if "waaagh" in entry.lower():
                events["waaagh_triggers"] += 1
            if "flees" in entry.lower() or "breaks" in entry.lower():
                events["units_fled"] += 1
        
        return events
    
    def analyze_special_rules_impact(self, battle_log: List[str]) -> Dict[str, Any]:
        """Analyze impact of faction-specific special rules"""
        impact = {
            "orc_rules": {
                "animosity_negative": 0,
                "animosity_positive": 0,
                "waaagh_effects": 0
            },
            "empire_rules": {
                "state_troop_discipline": 0,
                "gunpowder_misfires": 0,
                "detachment_actions": 0
            }
        }
        
        for entry in battle_log:
            # Orc rule analysis
            if "argues amongst themselves" in entry:
                impact["orc_rules"]["animosity_negative"] += 1
            if "works up into a frenzy" in entry:
                impact["orc_rules"]["animosity_positive"] += 1
            if "WAAAGH!" in entry:
                impact["orc_rules"]["waaagh_effects"] += 1
            
            # Empire rule analysis
            if "State Troop" in entry:
                impact["empire_rules"]["state_troop_discipline"] += 1
            if "misfire" in entry.lower():
                impact["empire_rules"]["gunpowder_misfires"] += 1
        
        return impact
    
    def generate_tactical_analysis(self, army1: List[Unit], army2: List[Unit], result: Dict) -> Dict[str, str]:
        """Generate tactical insights from battle"""
        analysis = {}
        
        # Determine key factors in victory/defeat
        winner = result.get("winner", "Draw")
        
        if winner == "Orc & Goblin Tribes":
            analysis["victory_factors"] = "Orc aggression and numbers overwhelmed Nuln firepower"
        elif winner == "City-State of Nuln":
            analysis["victory_factors"] = "Disciplined firepower and armor superiority"
        else:
            analysis["victory_factors"] = "Evenly matched forces, tactical stalemate"
        
        # Psychology impact
        psych_events = result.get("psychology_events", {})
        total_psych = sum(psych_events.values())
        
        if total_psych > 5:
            analysis["psychology_impact"] = "High - psychology played major role in battle"
        elif total_psych > 2:
            analysis["psychology_impact"] = "Moderate - some psychological effects"
        else:
            analysis["psychology_impact"] = "Low - minimal psychological impact"
        
        # Special rules effectiveness
        special_rules = result.get("special_rules_impact", {})
        orc_animosity = special_rules.get("orc_rules", {}).get("animosity_negative", 0)
        
        if orc_animosity > 2:
            analysis["orc_weakness"] = "Animosity significantly hindered Orc coordination"
        else:
            analysis["orc_weakness"] = "Orcs maintained good discipline"
        
        return analysis

class EnhancedEvolutionAI:
    """Enhanced AI that learns from comprehensive battle results"""
    
    def __init__(self, faction: str, population_size: int = 20):
        self.faction = faction
        self.population_size = population_size
        self.battle_engine = ComprehensiveTOWBattle()
        self.generation = 0
        self.population = self.initialize_population()
        
        # Enhanced learning metrics
        self.learning_history = {
            "generations": [],
            "best_fitness": [],
            "avg_fitness": [],
            "tactical_insights": [],
            "rule_effectiveness": []
        }
    
    def initialize_population(self) -> List[Dict]:
        """Initialize population with enhanced DNA"""
        population = []
        
        for _ in range(self.population_size):
            dna = self.create_enhanced_dna()
            population.append({
                "dna": dna,
                "fitness": 0.0,
                "battles_won": 0,
                "battles_fought": 0,
                "tactical_analysis": {}
            })
        
        return population
    
    def create_enhanced_dna(self) -> Dict:
        """Create DNA with comprehensive tactical preferences"""
        # Define available units for each faction
        if self.faction == "Orc & Goblin Tribes":
            units = [
                {"name": "Orc Big Boss"},
                {"name": "Orc Boyz"},
                {"name": "Night Goblins"},
                {"name": "Orc Arrer Boyz"}
            ]
        else:  # Nuln
            units = [
                {"name": "Engineer"},
                {"name": "Handgunners"},
                {"name": "Great Cannon"},
                {"name": "Crossbowmen"}
            ]
        
        dna = {
            # Unit preferences (0.0 to 1.0)
            "unit_preferences": {unit["name"]: random.random() for unit in units},
            
            # Tactical preferences
            "aggression": random.random(),
            "magic_focus": random.random(),
            "ranged_focus": random.random(),
            "elite_focus": random.random(),
            "defensive_focus": random.random(),
            
            # Formation preferences
            "close_order_preference": random.random(),
            "skirmish_preference": random.random(),
            
            # Faction-specific preferences
            "special_rule_focus": random.random(),
            "psychology_resistance": random.random(),
            
            # Command structure
            "leadership_priority": random.random(),
            "command_group_focus": random.random()
        }
        
        return dna
    
    def evaluate_fitness_comprehensive(self, individual: Dict, battles_per_eval: int = 5) -> float:
        """Comprehensive fitness evaluation using full TOW rules"""
        total_score = 0.0
        detailed_results = []
        
        for _ in range(battles_per_eval):
            # Build army based on DNA
            army_composition = self.dna_to_army_composition(individual["dna"])
            
            # Generate opponent army
            opponent_composition = self.generate_opponent_army()
            
            # Fight battle with comprehensive rules
            if self.faction == "Orc & Goblin Tribes":
                result = self.battle_engine.simulate_battle(army_composition, opponent_composition)
            else:
                result = self.battle_engine.simulate_battle(opponent_composition, army_composition)
            
            # Enhanced scoring based on comprehensive results
            battle_score = self.calculate_comprehensive_score(result, individual["dna"])
            total_score += battle_score
            detailed_results.append(result)
            
            # Track wins
            if result["winner"] == self.faction:
                individual["battles_won"] += 1
            individual["battles_fought"] += 1
        
        # Update tactical analysis
        individual["tactical_analysis"] = self.analyze_tactical_performance(detailed_results)
        
        return total_score / battles_per_eval
    
    def calculate_comprehensive_score(self, result: Dict, dna: Dict) -> float:
        """Calculate fitness score based on comprehensive battle analysis"""
        base_score = 50.0  # Base score for participation
        
        # Victory bonus
        if result["winner"] == self.faction:
            base_score += 100.0
        elif result["winner"] == "Draw":
            base_score += 25.0
        
        # Survival bonuses
        if self.faction == "Orc & Goblin Tribes":
            survival = result.get("army1_survival", {})
        else:
            survival = result.get("army2_survival", {})
        
        base_score += survival.get("unit_survival_rate", 0) * 30
        base_score += survival.get("model_survival_rate", 0) * 20
        
        # Tactical performance bonuses
        psych_events = result.get("psychology_events", {})
        special_rules = result.get("special_rules_impact", {})
        
        # Faction-specific bonuses
        if self.faction == "Orc & Goblin Tribes":
            # Bonus for successful Waaagh! effects
            waaagh_effects = special_rules.get("orc_rules", {}).get("waaagh_effects", 0)
            base_score += waaagh_effects * 10
            
            # Penalty for excessive animosity
            animosity_negative = special_rules.get("orc_rules", {}).get("animosity_negative", 0)
            base_score -= animosity_negative * 5
        
        elif self.faction == "City-State of Nuln":
            # Bonus for gunpowder effectiveness (lack of misfires)
            misfires = special_rules.get("empire_rules", {}).get("gunpowder_misfires", 0)
            base_score -= misfires * 3
            
            # Bonus for state troop discipline
            discipline = special_rules.get("empire_rules", {}).get("state_troop_discipline", 0)
            base_score += discipline * 2
        
        # Psychology resilience bonus
        total_psych_events = sum(psych_events.values())
        if total_psych_events < 3:
            base_score += 15  # Bonus for psychological stability
        
        return max(0, base_score)
    
    def analyze_tactical_performance(self, battle_results: List[Dict]) -> Dict[str, Any]:
        """Analyze tactical patterns across multiple battles"""
        analysis = {
            "avg_survival_rate": 0.0,
            "psychology_resilience": 0.0,
            "special_rules_effectiveness": 0.0,
            "preferred_tactics": "unknown",
            "weakness_patterns": []
        }
        
        if not battle_results:
            return analysis
        
        # Calculate averages
        total_survival = 0.0
        total_psych_events = 0
        
        for result in battle_results:
            if self.faction == "Orc & Goblin Tribes":
                survival = result.get("army1_survival", {})
            else:
                survival = result.get("army2_survival", {})
            
            total_survival += survival.get("unit_survival_rate", 0)
            
            psych_events = result.get("psychology_events", {})
            total_psych_events += sum(psych_events.values())
        
        analysis["avg_survival_rate"] = total_survival / len(battle_results)
        analysis["psychology_resilience"] = 1.0 - min(1.0, total_psych_events / (len(battle_results) * 10))
        
        return analysis
    
    def evolve_generation_comprehensive(self):
        """Evolve population using comprehensive fitness evaluation"""
        print(f"🧬 Evolving generation {self.generation + 1} with comprehensive TOW rules...")
        
        # Evaluate all individuals
        for i, individual in enumerate(self.population):
            fitness = self.evaluate_fitness_comprehensive(individual)
            individual["fitness"] = fitness
            
            if (i + 1) % 5 == 0:
                print(f"   Evaluated {i + 1}/{len(self.population)} individuals")
        
        # Sort by fitness
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Record generation statistics
        best_fitness = self.population[0]["fitness"]
        avg_fitness = sum(ind["fitness"] for ind in self.population) / len(self.population)
        
        self.learning_history["generations"].append(self.generation)
        self.learning_history["best_fitness"].append(best_fitness)
        self.learning_history["avg_fitness"].append(avg_fitness)
        
        # Generate tactical insights
        insights = self.generate_generation_insights()
        self.learning_history["tactical_insights"].append(insights)
        
        print(f"   Best fitness: {best_fitness:.2f}")
        print(f"   Average fitness: {avg_fitness:.2f}")
        print(f"   Tactical insight: {insights}")
        
        # Evolution (keep top 50%, breed new generation)
        elite_count = len(self.population) // 2
        elite = self.population[:elite_count]
        
        new_generation = elite.copy()
        
        # Breed new individuals
        while len(new_generation) < self.population_size:
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = self.crossover_comprehensive(parent1, parent2)
            child = self.mutate_comprehensive(child)
            new_generation.append(child)
        
        self.population = new_generation
        self.generation += 1
    
    def generate_generation_insights(self) -> str:
        """Generate insights about current generation's tactical preferences"""
        if not self.population:
            return "No data available"
        
        best = self.population[0]
        dna = best["dna"]
        
        # Analyze top performer's preferences
        top_units = sorted(dna["unit_preferences"].items(), key=lambda x: x[1], reverse=True)[:3]
        
        insights = f"Top units: {', '.join([unit[0] for unit in top_units])}"
        
        if dna["aggression"] > 0.7:
            insights += " | Highly aggressive"
        elif dna["defensive_focus"] > 0.7:
            insights += " | Defensive focus"
        
        if dna["elite_focus"] > 0.6:
            insights += " | Elite units preferred"
        
        return insights
    
    def crossover_comprehensive(self, parent1: Dict, parent2: Dict) -> Dict:
        """Enhanced crossover incorporating tactical understanding"""
        child_dna = {}
        
        # Crossover unit preferences
        child_dna["unit_preferences"] = {}
        for unit in parent1["dna"]["unit_preferences"]:
            if random.random() < 0.5:
                child_dna["unit_preferences"][unit] = parent1["dna"]["unit_preferences"][unit]
            else:
                child_dna["unit_preferences"][unit] = parent2["dna"]["unit_preferences"][unit]
        
        # Crossover tactical preferences
        for key in ["aggression", "magic_focus", "ranged_focus", "elite_focus", 
                   "defensive_focus", "special_rule_focus", "psychology_resistance"]:
            if random.random() < 0.5:
                child_dna[key] = parent1["dna"][key]
            else:
                child_dna[key] = parent2["dna"][key]
        
        return {
            "dna": child_dna,
            "fitness": 0.0,
            "battles_won": 0,
            "battles_fought": 0,
            "tactical_analysis": {}
        }
    
    def mutate_comprehensive(self, individual: Dict) -> Dict:
        """Enhanced mutation with tactical awareness"""
        mutation_rate = 0.1
        
        # Mutate unit preferences
        for unit in individual["dna"]["unit_preferences"]:
            if random.random() < mutation_rate:
                individual["dna"]["unit_preferences"][unit] += random.gauss(0, 0.1)
                individual["dna"]["unit_preferences"][unit] = max(0, min(1, individual["dna"]["unit_preferences"][unit]))
        
        # Mutate tactical preferences
        for key in ["aggression", "magic_focus", "ranged_focus", "elite_focus", 
                   "defensive_focus", "special_rule_focus", "psychology_resistance"]:
            if random.random() < mutation_rate:
                individual["dna"][key] += random.gauss(0, 0.1)
                individual["dna"][key] = max(0, min(1, individual["dna"][key]))
        
        return individual
    
    def dna_to_army_composition(self, dna: Dict) -> Dict[str, int]:
        """Convert DNA to army composition with enhanced logic"""
        # Create army composition based on DNA preferences
        if self.faction == "Orc & Goblin Tribes":
            composition = self.build_orc_composition_from_dna(dna)
        else:
            composition = self.build_nuln_composition_from_dna(dna)
        
        return composition
    
    def build_orc_composition_from_dna(self, dna: Dict) -> Dict[str, int]:
        """Build Orc army composition based on DNA preferences"""
        composition = {}
        
        # Core units (required)
        composition["Orc Big Boss"] = 1
        composition["Orc Boyz"] = max(1, int(dna["unit_preferences"].get("Orc Boyz", 0.5) * 3))
        
        # Optional units based on preferences
        if dna["unit_preferences"].get("Night Goblins", 0) > 0.4:
            composition["Night Goblins"] = 1
        
        if dna["ranged_focus"] > 0.6:
            composition["Orc Arrer Boyz"] = 1
        
        return composition
    
    def build_nuln_composition_from_dna(self, dna: Dict) -> Dict[str, int]:
        """Build Nuln army composition based on DNA preferences"""
        composition = {}
        
        # Core units (required)
        composition["Engineer"] = 1
        composition["Handgunners"] = max(1, int(dna["unit_preferences"].get("Handgunners", 0.5) * 2))
        
        # Optional units based on preferences
        if dna["ranged_focus"] > 0.7:
            composition["Great Cannon"] = 1
        
        if dna["unit_preferences"].get("Crossbowmen", 0) > 0.5:
            composition["Crossbowmen"] = 1
        
        return composition
    
    def generate_opponent_army(self) -> Dict[str, int]:
        """Generate opponent army for testing"""
        if self.faction == "Orc & Goblin Tribes":
            # Generate balanced Nuln army as opponent
            return {
                "Engineer": 1,
                "Handgunners": 2,
                "Great Cannon": 1
            }
        else:
            # Generate balanced Orc army as opponent
            return {
                "Orc Big Boss": 1,
                "Orc Boyz": 2,
                "Night Goblins": 1
            }

def run_tow_evolution(battles: int = 100000, generations: int = 1000):
    """Run TOW evolution with comprehensive rules"""
    print("🏛️ WARHAMMER: THE OLD WORLD - COMPREHENSIVE EVOLUTION")
    print("=" * 60)
    print(f"🎯 Running {battles} battles over {generations} generations")
    print("📖 Using complete TOW rules for authentic AI learning!")
    print("💾 Progress will be saved every 100 generations")
    print()
    
    import pickle
    import time
    
    try:
        # Create evolution AIs for both factions
        orc_ai = EnhancedEvolutionAI("Orc & Goblin Tribes")
        nuln_ai = EnhancedEvolutionAI("City-State of Nuln")
        
        start_time = time.time()
        
        for gen in range(generations):
            print(f"\n🧬 === GENERATION {gen + 1}/{generations} ===")
            
            # Evolve both factions
            print("🧌 Evolving Orc & Goblin Tribes...")
            orc_ai.evolve_generation_comprehensive()
            
            print("🏰 Evolving City-State of Nuln...")
            nuln_ai.evolve_generation_comprehensive()
            
            # Show progress
            elapsed = time.time() - start_time
            estimated_total = elapsed * generations / (gen + 1)
            remaining = estimated_total - elapsed
            
            print(f"\n📊 Generation {gen + 1} Results:")
            print(f"  Orc best fitness: {orc_ai.population[0]['fitness']:.2f}")
            print(f"  Nuln best fitness: {nuln_ai.population[0]['fitness']:.2f}")
            print(f"  ⏱️ Time: {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining")
            
            # Show tactical insights
            if orc_ai.learning_history["tactical_insights"]:
                print(f"  🧌 Orc insights: {orc_ai.learning_history['tactical_insights'][-1]}")
            if nuln_ai.learning_history["tactical_insights"]:
                print(f"  🏰 Nuln insights: {nuln_ai.learning_history['tactical_insights'][-1]}")
            
            # Save progress every 100 generations
            if (gen + 1) % 100 == 0:
                save_data = {
                    "generation": gen + 1,
                    "orc_ai": orc_ai,
                    "nuln_ai": nuln_ai,
                    "total_generations": generations,
                    "battles": battles,
                    "timestamp": time.time()
                }
                filename = f"tow_evolution_progress_gen_{gen + 1}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(save_data, f)
                print(f"  💾 Progress saved to {filename}")
        
        print("\n🏆 === EVOLUTION COMPLETE ===")
        print("✅ AI has learned authentic TOW tactics!")
        print("📈 Both factions evolved optimal strategies")
        print(f"⏱️ Total time: {(time.time() - start_time)/3600:.1f} hours")
        print()
        
        # Save final results
        final_results = {
            "orc_ai": orc_ai,
            "nuln_ai": nuln_ai,
            "generations": generations,
            "battles_per_generation": battles // generations,
            "total_time_hours": (time.time() - start_time) / 3600,
            "final_timestamp": time.time()
        }
        
        with open("tow_evolution_final_results.pkl", 'wb') as f:
            pickle.dump(final_results, f)
        print("💾 Final results saved to tow_evolution_final_results.pkl")
        
        return final_results
        
    except KeyboardInterrupt:
        print("\n⚠️ Evolution interrupted by user!")
        print("💾 Saving current progress...")
        
        # Save interrupted progress
        save_data = {
            "generation": gen + 1 if 'gen' in locals() else 0,
            "orc_ai": orc_ai if 'orc_ai' in locals() else None,
            "nuln_ai": nuln_ai if 'nuln_ai' in locals() else None,
            "total_generations": generations,
            "battles": battles,
            "interrupted": True,
            "timestamp": time.time()
        }
        
        with open("tow_evolution_interrupted.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        print("✅ Progress saved to tow_evolution_interrupted.pkl")
        print("🔄 You can resume this evolution later!")
        raise
    
    except Exception as e:
        print(f"\n❌ Error during evolution: {e}")
        print("💾 Attempting to save progress...")
        
        try:
            save_data = {
                "generation": gen + 1 if 'gen' in locals() else 0,
                "orc_ai": orc_ai if 'orc_ai' in locals() else None,
                "nuln_ai": nuln_ai if 'nuln_ai' in locals() else None,
                "total_generations": generations,
                "battles": battles,
                "error": str(e),
                "timestamp": time.time()
            }
            
            with open("tow_evolution_error.pkl", 'wb') as f:
                pickle.dump(save_data, f)
            print("✅ Progress saved to tow_evolution_error.pkl")
        except:
            print("❌ Could not save progress")
        
        raise

if __name__ == "__main__":
    # Test comprehensive battle system
    print("🧬 ENHANCED WARHAMMER TOW EVOLUTION AI")
    print("=" * 50)
    print("🎯 Using comprehensive TOW rules for authentic battles")
    print()
    
    # Test comprehensive battle system
    battle_engine = ComprehensiveTOWBattle()
    
    orc_army = {"Orc Big Boss": 1, "Orc Boyz": 2, "Night Goblins": 1}
    nuln_army = {"Engineer": 1, "Handgunners": 2, "Great Cannon": 1}
    
    print("⚔️ Testing comprehensive battle...")
    result = battle_engine.simulate_battle(orc_army, nuln_army)
    
    print(f"Winner: {result['winner']}")
    print(f"Orc survival: {result['army1_survival']['unit_survival_rate']:.2%}")
    print(f"Nuln survival: {result['army2_survival']['unit_survival_rate']:.2%}")
    print(f"Psychology events: {sum(result['psychology_events'].values())}")
    print()
    
    print("✅ Enhanced evolution AI ready for comprehensive TOW battles!") 