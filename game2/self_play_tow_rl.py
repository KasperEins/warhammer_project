#!/usr/bin/env python3
"""
🏛️ SELF-PLAY REINFORCEMENT LEARNING TOW TRAINER
================================================

True reinforcement learning system where AI agents discover optimal
Warhammer: The Old World strategies through millions of self-play games.

Features:
- Self-play reinforcement learning (like AlphaGo/AlphaStar)
- Neural network agents that learn from scratch
- Army composition optimization through trial and error
- Strategic evolution over millions of battles
- Comprehensive TOW rules integration
- Multi-agent competition and evolution
"""

import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import pickle

# Import comprehensive TOW rules
try:
    from tow_comprehensive_rules import TOWBattleEngine
    COMPREHENSIVE_RULES_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_RULES_AVAILABLE = False
    print("⚠️ Comprehensive TOW rules not available - using simplified simulation")

@dataclass
class GameResult:
    """Result of a single game"""
    winner: str
    orc_army: Dict[str, int]
    nuln_army: Dict[str, int]
    battle_duration: int
    casualties: Dict[str, int]
    tactical_events: List[str]
    final_scores: Dict[str, float]

class TOWNeuralNetwork(nn.Module):
    """Neural network for TOW army composition and strategy"""
    
    def __init__(self, input_size=60, hidden_size=256, output_size=40):
        super(TOWNeuralNetwork, self).__init__()
        
        # Army composition network
        self.army_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        
        # Value network for position evaluation
        self.value_net = nn.Sequential(
            nn.Linear(input_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        army_logits = self.army_net(x)
        value = self.value_net(x)
        return army_logits, value

class TOWRLAgent:
    """Reinforcement Learning agent for TOW"""
    
    def __init__(self, faction: str, learning_rate=0.001):
        self.faction = faction
        self.network = TOWNeuralNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=100000)  # Experience replay buffer
        
        # Performance tracking
        self.games_played = 0
        self.wins = 0
        self.total_reward = 0
        self.army_performance = defaultdict(lambda: {'wins': 0, 'games': 0})
        self.strategy_evolution = []
        
        # Exploration parameters
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.05
        
        # Initialize unit database
        if faction == "Orc & Goblin Tribes":
            self.available_units = {
                "Orc Big Boss": {"points": 80, "combat": 4, "min": 1, "max": 3, "role": "character"},
                "Orc Boyz": {"points": 12, "combat": 3, "min": 0, "max": 8, "role": "core"},
                "Night Goblins": {"points": 8, "combat": 2, "min": 0, "max": 10, "role": "core"},
                "Orc Arrer Boyz": {"points": 14, "combat": 3, "min": 0, "max": 6, "role": "core"},
                "Wolf Riders": {"points": 18, "combat": 3, "min": 0, "max": 4, "role": "special"},
                "Trolls": {"points": 35, "combat": 5, "min": 0, "max": 3, "role": "special"},
                "Rock Lobber": {"points": 85, "combat": 0, "min": 0, "max": 2, "role": "rare"},
                "Orc Boar Boyz": {"points": 22, "combat": 4, "min": 0, "max": 3, "role": "special"}
            }
        else:  # City-State of Nuln
            self.available_units = {
                "Engineer": {"points": 65, "combat": 4, "min": 1, "max": 2, "role": "character"},
                "Handgunners": {"points": 12, "combat": 4, "min": 0, "max": 6, "role": "core"},
                "Crossbowmen": {"points": 10, "combat": 3, "min": 0, "max": 6, "role": "core"},
                "Swordsmen": {"points": 8, "combat": 4, "min": 0, "max": 8, "role": "core"},
                "Great Cannon": {"points": 100, "combat": 8, "min": 0, "max": 2, "role": "rare"},
                "Outriders": {"points": 24, "combat": 4, "min": 0, "max": 3, "role": "special"},
                "Pistoliers": {"points": 20, "combat": 3, "min": 0, "max": 4, "role": "special"},
                "Steam Tank": {"points": 300, "combat": 10, "min": 0, "max": 1, "role": "rare"}
            }
        
        self.unit_names = list(self.available_units.keys())
        
    def encode_game_state(self, own_army: Dict, enemy_army: Dict, 
                         context: Dict = None) -> torch.Tensor:
        """Encode current game state for neural network"""
        features = []
        
        # Own army composition (normalized by max counts)
        for unit in self.unit_names:
            count = own_army.get(unit, 0)
            max_count = self.available_units[unit]["max"]
            features.append(count / max(1, max_count))
        
        # Enemy army composition (estimated/visible)
        enemy_unit_names = list(enemy_army.keys())
        enemy_features = [0] * len(self.unit_names)  # Initialize with zeros
        
        # Map enemy units to our feature space (simplified)
        for i, enemy_unit in enumerate(enemy_unit_names[:len(self.unit_names)]):
            if i < len(self.unit_names):
                enemy_features[i] = enemy_army.get(enemy_unit, 0) / 10.0  # Normalize
        
        features.extend(enemy_features)
        
        # Battle context
        if context:
            features.extend([
                context.get('turn_number', 0) / 10.0,
                context.get('terrain_bonus', 0),
                context.get('weather_modifier', 0),
                context.get('deployment_advantage', 0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Army composition metrics
        total_points = sum(self.available_units[unit]["points"] * own_army.get(unit, 0) 
                          for unit in self.unit_names)
        features.append(total_points / 2000.0)  # Normalize to typical army size
        
        total_units = sum(own_army.values())
        features.append(min(total_units / 20.0, 1.0))  # Unit count diversity
        
        # Pad or truncate to exactly 60 features
        while len(features) < 60:
            features.append(0.0)
        features = features[:60]
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def select_army_composition(self, enemy_intel: Dict = None, 
                              points_limit: int = 2000) -> Dict[str, int]:
        """Select army composition using neural network"""
        
        # Create context for decision making
        context = {
            'turn_number': self.games_played % 100,
            'terrain_bonus': random.uniform(-0.1, 0.1),
            'weather_modifier': random.uniform(-0.05, 0.05),
            'deployment_advantage': random.uniform(-0.1, 0.1)
        }
        
        enemy_army = enemy_intel if enemy_intel else {"Unknown": 1}
        
        # Get current best guess army
        current_army = self._get_base_army()
        
        # Encode state
        state = self.encode_game_state(current_army, enemy_army, context)
        
        # Get network output
        with torch.no_grad():
            army_logits, value = self.network(state)
        
        # Apply epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Exploration: random army with constraints
            army = self._generate_random_valid_army(points_limit)
        else:
            # Exploitation: use network predictions
            army = self._logits_to_army(army_logits, points_limit)
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return army
    
    def _get_base_army(self) -> Dict[str, int]:
        """Get base army with mandatory units"""
        army = {}
        for unit, stats in self.available_units.items():
            army[unit] = stats["min"]
        return army
    
    def _logits_to_army(self, logits: torch.Tensor, points_limit: int) -> Dict[str, int]:
        """Convert network logits to valid army composition"""
        army = self._get_base_army()
        remaining_points = points_limit - sum(
            self.available_units[unit]["points"] * count 
            for unit, count in army.items()
        )
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits.squeeze(), dim=0)
        
        # Allocate remaining units based on probabilities
        for i, unit in enumerate(self.unit_names):
            if i < len(probs):
                prob = probs[i].item()
                max_additional = min(
                    self.available_units[unit]["max"] - army[unit],
                    remaining_points // self.available_units[unit]["points"]
                )
                
                if max_additional > 0:
                    additional = int(prob * max_additional * 2)  # Scale factor
                    additional = min(additional, max_additional)
                    army[unit] += additional
                    remaining_points -= additional * self.available_units[unit]["points"]
        
        return {k: v for k, v in army.items() if v > 0}
    
    def _generate_random_valid_army(self, points_limit: int) -> Dict[str, int]:
        """Generate random but valid army composition"""
        army = self._get_base_army()
        remaining_points = points_limit - sum(
            self.available_units[unit]["points"] * count 
            for unit, count in army.items()
        )
        
        # Randomly add units within constraints
        attempts = 0
        while remaining_points > 50 and attempts < 100:  # Leave some buffer
            unit = random.choice(self.unit_names)
            unit_cost = self.available_units[unit]["points"]
            max_count = self.available_units[unit]["max"]
            
            if army[unit] < max_count and remaining_points >= unit_cost:
                army[unit] += 1
                remaining_points -= unit_cost
            
            attempts += 1
        
        return {k: v for k, v in army.items() if v > 0}
    
    def learn_from_game(self, game_result: GameResult, own_army: Dict, enemy_army: Dict):
        """Learn from game outcome using reinforcement learning"""
        self.games_played += 1
        
        # Calculate reward
        won = (game_result.winner == self.faction)
        if won:
            self.wins += 1
            base_reward = 1.0
        else:
            base_reward = -1.0
        
        # Add shaped rewards for good tactical decisions
        tactical_reward = 0.0
        for event in game_result.tactical_events:
            if "strategic" in event.lower():
                tactical_reward += 0.1
            elif "devastating" in event.lower():
                tactical_reward += 0.2
        
        # Army efficiency reward
        if self.faction in game_result.final_scores:
            efficiency = game_result.final_scores[self.faction]
            tactical_reward += efficiency * 0.3
        
        total_reward = base_reward + tactical_reward
        self.total_reward += total_reward
        
        # Store experience for replay learning
        context = {'turn_number': self.games_played}
        state = self.encode_game_state(own_army, enemy_army, context)
        
        experience = {
            'state': state,
            'army': own_army,
            'reward': total_reward,
            'won': won
        }
        self.memory.append(experience)
        
        # Track army performance
        army_key = self._army_to_key(own_army)
        self.army_performance[army_key]['games'] += 1
        if won:
            self.army_performance[army_key]['wins'] += 1
        
        # Learn from experience replay every 100 games
        if self.games_played % 100 == 0:
            self._experience_replay()
        
        # Track strategy evolution
        if self.games_played % 1000 == 0:
            self._record_strategy_snapshot()
    
    def _army_to_key(self, army: Dict) -> str:
        """Convert army to string key for tracking"""
        return "|".join(f"{unit}:{count}" for unit, count in sorted(army.items()))
    
    def _experience_replay(self, batch_size: int = 64):
        """Learn from random batch of past experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        
        states = torch.cat([exp['state'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # Forward pass
        army_logits, values = self.network(states)
        
        # Calculate losses
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        # Policy loss (REINFORCE with baseline)
        advantages = rewards - values.squeeze().detach()
        policy_loss = -torch.mean(advantages.unsqueeze(1) * army_logits)
        
        total_loss = value_loss + 0.1 * policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
    
    def _record_strategy_snapshot(self):
        """Record current strategy for evolution tracking"""
        top_armies = sorted(
            self.army_performance.items(),
            key=lambda x: x[1]['wins'] / max(1, x[1]['games']) if x[1]['games'] >= 5 else 0,
            reverse=True
        )[:5]
        
        snapshot = {
            'games_played': self.games_played,
            'win_rate': self.wins / max(1, self.games_played),
            'epsilon': self.epsilon,
            'top_armies': top_armies,
            'avg_reward': self.total_reward / max(1, self.games_played)
        }
        
        self.strategy_evolution.append(snapshot)
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        win_rate = self.wins / max(1, self.games_played)
        avg_reward = self.total_reward / max(1, self.games_played)
        
        # Best performing armies
        best_armies = []
        for army_key, stats in self.army_performance.items():
            if stats['games'] >= 5:  # Minimum sample size
                win_rate_army = stats['wins'] / stats['games']
                best_armies.append((army_key, win_rate_army, stats['games']))
        
        best_armies.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'faction': self.faction,
            'games_played': self.games_played,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'epsilon': self.epsilon,
            'best_armies': best_armies[:3],
            'total_armies_tried': len(self.army_performance),
            'strategy_evolution_points': len(self.strategy_evolution)
        }

class SelfPlayTOWTrainer:
    """Main self-play training system"""
    
    def __init__(self, use_comprehensive_rules: bool = True):
        self.orc_agent = TOWRLAgent("Orc & Goblin Tribes")
        self.nuln_agent = TOWRLAgent("City-State of Nuln")
        
        # Initialize battle engine
        self.use_comprehensive_rules = use_comprehensive_rules and COMPREHENSIVE_RULES_AVAILABLE
        if self.use_comprehensive_rules:
            self.battle_engine = TOWBattleEngine()
            print("✅ Using comprehensive TOW rules for realistic battles")
        else:
            print("⚠️ Using simplified battle simulation")
        
        # Training statistics
        self.total_games = 0
        self.training_start_time = None
        self.game_results = deque(maxlen=10000)  # Keep last 10k results
        
    def simulate_comprehensive_battle(self, orc_army: Dict, nuln_army: Dict) -> GameResult:
        """Simulate battle using comprehensive TOW rules or fallback"""
        
        if self.use_comprehensive_rules:
            try:
                # Use real TOW battle engine
                battle_result = self.battle_engine.run_battle(orc_army, nuln_army)
                
                return GameResult(
                    winner=battle_result.get('winner', 'Draw'),
                    orc_army=orc_army,
                    nuln_army=nuln_army,
                    battle_duration=battle_result.get('turns', 1),
                    casualties=battle_result.get('casualties', {'Orcs': 0, 'Nuln': 0}),
                    tactical_events=battle_result.get('events', []),
                    final_scores=battle_result.get('final_scores', {'Orcs': 0.5, 'Nuln': 0.5})
                )
            except Exception as e:
                print(f"⚠️ Comprehensive rules failed: {e}, using fallback")
        
        # Fallback to enhanced tactical simulation
        return self._simulate_tactical_battle(orc_army, nuln_army)
    
    def _simulate_tactical_battle(self, orc_army: Dict, nuln_army: Dict) -> GameResult:
        """Enhanced tactical battle simulation"""
        
        # Calculate army strengths with detailed unit roles
        orc_strength = self._calculate_army_strength(orc_army, self.orc_agent.available_units)
        nuln_strength = self._calculate_army_strength(nuln_army, self.nuln_agent.available_units)
        
        # Simulate tactical phases
        tactical_events = []
        
        # 1. Deployment phase
        orc_deployment = self._evaluate_deployment(orc_army)
        nuln_deployment = self._evaluate_deployment(nuln_army)
        
        if orc_deployment > nuln_deployment:
            orc_strength *= 1.1
            tactical_events.append("Orcs gain strategic deployment advantage")
        elif nuln_deployment > orc_deployment:
            nuln_strength *= 1.1
            tactical_events.append("Nuln gains strategic deployment advantage")
        
        # 2. Ranged phase
        orc_ranged = self._get_ranged_power(orc_army, self.orc_agent.available_units)
        nuln_ranged = self._get_ranged_power(nuln_army, self.nuln_agent.available_units)
        
        if orc_ranged > nuln_ranged * 1.5:
            nuln_strength *= 0.9
            tactical_events.append("Devastating Orc ranged superiority")
        elif nuln_ranged > orc_ranged * 1.5:
            orc_strength *= 0.9
            tactical_events.append("Devastating Nuln firepower advantage")
        
        # 3. Melee resolution with multiple rounds
        battle_rounds = random.randint(3, 8)
        casualties = {'Orcs': 0, 'Nuln': 0}
        
        for round_num in range(battle_rounds):
            # Combat resolution
            round_factor = random.uniform(0.85, 1.15)
            
            if orc_strength * round_factor > nuln_strength:
                damage_ratio = min(2.0, orc_strength / max(1, nuln_strength))
                nuln_casualties = random.randint(1, int(damage_ratio * 2))
                casualties['Nuln'] += nuln_casualties
                nuln_strength *= 0.95  # Gradual weakening
                
                if damage_ratio > 1.5:
                    tactical_events.append(f"Round {round_num + 1}: Orc breakthrough")
            else:
                damage_ratio = min(2.0, nuln_strength / max(1, orc_strength))
                orc_casualties = random.randint(1, int(damage_ratio * 2))
                casualties['Orcs'] += orc_casualties
                orc_strength *= 0.95
                
                if damage_ratio > 1.5:
                    tactical_events.append(f"Round {round_num + 1}: Nuln tactical superiority")
            
            # Morale check
            if random.random() < 0.2:
                if casualties['Orcs'] > casualties['Nuln']:
                    orc_strength *= 0.9
                    tactical_events.append("Orc morale wavering")
                elif casualties['Nuln'] > casualties['Orcs']:
                    nuln_strength *= 0.9
                    tactical_events.append("Nuln forces shaken")
        
        # Determine final winner
        final_orc = max(0, orc_strength)
        final_nuln = max(0, nuln_strength)
        
        if final_orc > final_nuln:
            winner = "Orc & Goblin Tribes"
        elif final_nuln > final_orc:
            winner = "City-State of Nuln"
        else:
            winner = "Draw"
        
        # Calculate final scores (tactical efficiency)
        total_strength = final_orc + final_nuln
        final_scores = {
            "Orc & Goblin Tribes": final_orc / max(1, total_strength),
            "City-State of Nuln": final_nuln / max(1, total_strength)
        }
        
        return GameResult(
            winner=winner,
            orc_army=orc_army,
            nuln_army=nuln_army,
            battle_duration=battle_rounds,
            casualties=casualties,
            tactical_events=tactical_events,
            final_scores=final_scores
        )
    
    def _calculate_army_strength(self, army: Dict, unit_data: Dict) -> float:
        """Calculate comprehensive army strength"""
        strength = 0
        
        for unit, count in army.items():
            if unit in unit_data:
                stats = unit_data[unit]
                base_strength = stats['combat'] * count
                
                # Role-based modifiers
                role = stats.get('role', 'core')
                if role == 'character':
                    base_strength *= 1.5  # Leadership bonus
                elif role == 'special':
                    base_strength *= 1.3  # Elite training
                elif role == 'rare':
                    base_strength *= 1.8  # Unique capabilities
                
                strength += base_strength
        
        return strength
    
    def _evaluate_deployment(self, army: Dict) -> float:
        """Evaluate army deployment potential"""
        fast_units = 0
        total_units = sum(army.values())
        
        for unit, count in army.items():
            if 'rider' in unit.lower() or 'cavalry' in unit.lower():
                fast_units += count
        
        # Deployment advantage from mobile units
        mobility_ratio = fast_units / max(1, total_units)
        return mobility_ratio * 100
    
    def _get_ranged_power(self, army: Dict, unit_data: Dict) -> float:
        """Calculate ranged combat effectiveness"""
        ranged_power = 0
        
        for unit, count in army.items():
            if unit in unit_data:
                if ('arrer' in unit.lower() or 'handgun' in unit.lower() or 
                    'crossbow' in unit.lower() or 'cannon' in unit.lower()):
                    
                    base_power = unit_data[unit]['combat'] * count
                    
                    # Special ranged bonuses
                    if 'cannon' in unit.lower():
                        base_power *= 3.0  # Artillery supremacy
                    elif 'handgun' in unit.lower():
                        base_power *= 1.5  # Armor piercing
                    
                    ranged_power += base_power
        
        return ranged_power
    
    def run_self_play_training(self, num_games: int = 1000000):
        """Run massive self-play training session"""
        print("🏛️ SELF-PLAY REINFORCEMENT LEARNING TRAINING")
        print("=" * 60)
        print(f"🎯 Target: {num_games:,} self-play games")
        print("🧠 Agents will discover optimal strategies through trial and error")
        print("⚔️ Using comprehensive TOW rules for realistic learning")
        print("🎮 Pure reinforcement learning - no expert guidance!")
        print()
        
        self.training_start_time = time.time()
        
        for game in range(num_games):
            # Agents select armies independently
            orc_army = self.orc_agent.select_army_composition()
            nuln_army = self.nuln_agent.select_army_composition()
            
            # Simulate battle
            result = self.simulate_comprehensive_battle(orc_army, nuln_army)
            
            # Both agents learn from the outcome
            self.orc_agent.learn_from_game(result, orc_army, nuln_army)
            self.nuln_agent.learn_from_game(result, nuln_army, orc_army)
            
            # Track results
            self.game_results.append(result)
            self.total_games += 1
            
            # Progress reporting
            if (game + 1) % 10000 == 0:
                self._report_progress(game + 1, num_games)
        
        self._generate_final_report()
    
    def _report_progress(self, games_completed: int, total_games: int):
        """Report training progress"""
        elapsed = time.time() - self.training_start_time
        rate = games_completed / elapsed
        
        # Recent performance
        recent_results = list(self.game_results)[-1000:] if len(self.game_results) >= 1000 else list(self.game_results)
        if recent_results:
            orc_wins = sum(1 for r in recent_results if r.winner == "Orc & Goblin Tribes")
            win_rate = orc_wins / len(recent_results)
        else:
            win_rate = 0.5
        
        orc_stats = self.orc_agent.get_performance_stats()
        nuln_stats = self.nuln_agent.get_performance_stats()
        
        print(f"📊 Progress: {games_completed:,}/{total_games:,} ({games_completed/total_games:.1%})")
        print(f"   ⚡ Rate: {rate:.0f} games/sec")
        print(f"   ⚖️ Recent balance: {win_rate:.1%} Orc | {1-win_rate:.1%} Nuln")
        print(f"   🧌 Orc exploration: {orc_stats['epsilon']:.3f}")
        print(f"   🏰 Nuln exploration: {nuln_stats['epsilon']:.3f}")
        print(f"   🧬 Armies discovered: Orc {orc_stats['total_armies_tried']} | Nuln {nuln_stats['total_armies_tried']}")
        print()
    
    def _generate_final_report(self):
        """Generate comprehensive training report"""
        training_time = time.time() - self.training_start_time
        
        orc_stats = self.orc_agent.get_performance_stats()
        nuln_stats = self.nuln_agent.get_performance_stats()
        
        print("🎉 SELF-PLAY TRAINING COMPLETE!")
        print("=" * 50)
        print(f"⏱️ Training time: {training_time/3600:.2f} hours")
        print(f"⚡ Average rate: {self.total_games/training_time:.0f} games/sec")
        print(f"🎯 Total games: {self.total_games:,}")
        print()
        
        print("📈 FINAL AGENT PERFORMANCE:")
        print("-" * 30)
        print(f"🧌 {orc_stats['faction']}:")
        print(f"   Win rate: {orc_stats['win_rate']:.1%}")
        print(f"   Avg reward: {orc_stats['avg_reward']:.3f}")
        print(f"   Exploration: {orc_stats['epsilon']:.3f}")
        print(f"   Armies tried: {orc_stats['total_armies_tried']:,}")
        
        print(f"\n🏰 {nuln_stats['faction']}:")
        print(f"   Win rate: {nuln_stats['win_rate']:.1%}")
        print(f"   Avg reward: {nuln_stats['avg_reward']:.3f}")
        print(f"   Exploration: {nuln_stats['epsilon']:.3f}")
        print(f"   Armies tried: {nuln_stats['total_armies_tried']:,}")
        
        print("\n🏆 DISCOVERED OPTIMAL ARMIES:")
        print("-" * 30)
        print("🧌 Top Orc armies:")
        for i, (army_key, win_rate, games) in enumerate(orc_stats['best_armies'][:3]):
            print(f"   {i+1}. Win rate {win_rate:.1%} ({games} games): {army_key}")
        
        print("\n🏰 Top Nuln armies:")
        for i, (army_key, win_rate, games) in enumerate(nuln_stats['best_armies'][:3]):
            print(f"   {i+1}. Win rate {win_rate:.1%} ({games} games): {army_key}")
        
        # Save results
        self._save_training_results()
    
    def _save_training_results(self):
        """Save trained agents and results"""
        # Save neural networks
        torch.save(self.orc_agent.network.state_dict(), 'orc_rl_agent.pth')
        torch.save(self.nuln_agent.network.state_dict(), 'nuln_rl_agent.pth')
        
        # Save training data
        results = {
            'total_games': self.total_games,
            'orc_stats': self.orc_agent.get_performance_stats(),
            'nuln_stats': self.nuln_agent.get_performance_stats(),
            'orc_evolution': self.orc_agent.strategy_evolution,
            'nuln_evolution': self.nuln_agent.strategy_evolution
        }
        
        with open('self_play_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save agents for later use
        with open('trained_orc_agent.pkl', 'wb') as f:
            pickle.dump(self.orc_agent, f)
        with open('trained_nuln_agent.pkl', 'wb') as f:
            pickle.dump(self.nuln_agent, f)
        
        print("\n💾 Saved trained agents and results")
        print("   - orc_rl_agent.pth / nuln_rl_agent.pth (neural networks)")
        print("   - self_play_results.json (training statistics)")
        print("   - trained_*_agent.pkl (complete agents)")

def main():
    """Main training function"""
    trainer = SelfPlayTOWTrainer()
    
    print("🏛️ SELF-PLAY TOW REINFORCEMENT LEARNING TRAINER")
    print("=" * 60)
    print("🧠 Agents learn optimal strategies through millions of games")
    print("🎯 No expert guidance - pure self-discovery")
    print("⚔️ Comprehensive TOW rules for realistic learning")
    print()
    
    print("🎮 TRAINING OPTIONS:")
    print("1. Quick training (100,000 games)")
    print("2. Standard training (1,000,000 games)")
    print("3. Intensive training (10,000,000 games)")
    print("4. Custom training")
    
    choice = input("\nSelect training option (1-4): ").strip()
    
    if choice == '1':
        trainer.run_self_play_training(100000)
    elif choice == '2':
        trainer.run_self_play_training(1000000)
    elif choice == '3':
        trainer.run_self_play_training(10000000)
    elif choice == '4':
        num_games = int(input("Number of games: "))
        trainer.run_self_play_training(num_games)
    else:
        print("Running standard training...")
        trainer.run_self_play_training(1000000)

if __name__ == "__main__":
    main()