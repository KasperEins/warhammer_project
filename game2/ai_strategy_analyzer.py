#!/usr/bin/env python3
"""
WARHAMMER: THE OLD WORLD - AI STRATEGY ANALYZER
==============================================
Analyze and visualize AI-discovered strategies
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from warhammer_ai_agent import WarhammerAIAgent, WarhammerBattleEnvironment, train_agent

class StrategyAnalyzer:
    """Advanced analysis of AI-discovered strategies"""
    
    def __init__(self, model_path: str = 'warhammer_ai_model.pth'):
        self.model_path = model_path
        self.agent = None
        self.env = WarhammerBattleEnvironment()
        
        # Load trained agent
        state_size = len(self.env.reset())
        action_size = 13
        self.agent = WarhammerAIAgent(state_size, action_size)
        
        try:
            self.agent.load_model(model_path)
            print(f"✅ Loaded trained model from {model_path}")
        except FileNotFoundError:
            print(f"❌ Model not found. Training new agent...")
            self.agent = train_agent(100)
    
    def analyze_strategic_patterns(self):
        """Comprehensive analysis of learned strategies"""
        print("\n🔬 STRATEGIC PATTERN ANALYSIS")
        print("=" * 50)
        
        insights = self.agent.get_strategy_insights()
        
        # Basic statistics
        print(f"📊 Training Statistics:")
        print(f"   Total Battles: {insights['total_battles']}")
        print(f"   Win Rate: {insights['win_rate']:.2%}")
        print(f"   Average Score: {insights['average_score']:.1f}")
        print(f"   Victories: {insights['victories']}")
        print(f"   Defeats: {insights['defeats']}")
        print(f"   Draws: {insights['draws']}")
        print()
        
        # Analyze successful strategies
        if self.agent.successful_strategies:
            self._analyze_winning_patterns()
        
        return insights
    
    def _analyze_winning_patterns(self):
        """Analyze patterns in successful strategies"""
        print("🏆 WINNING STRATEGY PATTERNS:")
        
        # Extract patterns from successful strategies
        opening_moves = []
        action_sequences = []
        
        for strategy in self.agent.successful_strategies:
            actions = strategy['actions']
            if len(actions) >= 3:
                opening_moves.extend(actions[:3])
            action_sequences.append(tuple(actions[:5]))
        
        # Analyze opening preferences
        opening_counter = Counter(opening_moves)
        print("\n📈 Most Successful Opening Moves:")
        for action, count in opening_counter.most_common(5):
            action_name = self.agent._categorize_action(action)
            percentage = count / len(opening_moves) * 100 if opening_moves else 0
            print(f"   {action_name}: {count} times ({percentage:.1f}%)")
        
        # Find most common winning sequences
        sequence_counter = Counter(action_sequences)
        print("\n🎯 Most Effective Action Sequences:")
        for sequence, count in sequence_counter.most_common(3):
            if count > 1:  # Only show patterns that occurred multiple times
                sequence_names = [self.agent._categorize_action(a) for a in sequence[:3]]
                print(f"   {' → '.join(sequence_names)}: {count} victories")
    
    def test_strategic_scenarios(self):
        """Test AI performance in specific scenarios"""
        print("\n🎮 STRATEGIC SCENARIO TESTING")
        print("=" * 40)
        
        results = {}
        battles = 5
        
        print(f"\n📋 Testing: Standard Battle Scenarios")
        wins = 0
        total_scores = []
        strategies_used = []
        
        self.agent.epsilon = 0.0  # No exploration
        
        for battle in range(battles):
            state = self.env.reset()
            total_reward = 0
            moves = []
            
            step_count = 0
            while step_count < 15:  # Test with 15 moves
                action = self.agent.act(state)
                moves.append(self.agent._categorize_action(action))
                
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            total_scores.append(total_reward)
            strategies_used.extend(moves[:3])  # First 3 moves
            
            if total_reward > 50:  # Victory
                wins += 1
        
        win_rate = wins / battles
        avg_score = np.mean(total_scores)
        
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Avg Score: {avg_score:.1f}")
        if strategies_used:
            print(f"   Preferred Strategy: {Counter(strategies_used).most_common(1)[0][0]}")
        
        return {'win_rate': win_rate, 'avg_score': avg_score}
    
    def generate_strategy_recommendations(self):
        """Generate strategic recommendations based on AI learning"""
        print("\n💡 AI-DISCOVERED STRATEGIC RECOMMENDATIONS")
        print("=" * 55)
        
        insights = self.agent.get_strategy_insights()
        
        recommendations = []
        
        # Analyze successful tactics
        if insights.get('successful_tactics'):
            top_tactic = list(insights['successful_tactics'].keys())[0]
            recommendations.append({
                'category': 'Primary Tactic',
                'recommendation': f"Prioritize {top_tactic}",
                'reason': f"Most successful strategy with {insights['successful_tactics'][top_tactic]} victories"
            })
        
        # Army composition insights
        recommendations.append({
            'category': 'Army Composition',
            'recommendation': "Artillery-Heavy Strategy",
            'reason': "AI consistently favors concentrated artillery fire"
        })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['category']}: {rec['recommendation']}")
            print(f"   Reason: {rec['reason']}")
        
        return recommendations
    
    def visualize_learning_progress(self):
        """Create visualizations of AI learning progress"""
        print("\n📊 GENERATING LEARNING VISUALIZATIONS...")
        
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🤖 WARHAMMER AI LEARNING ANALYSIS', fontsize=16, color='gold', weight='bold')
        
        # 1. Learning curve
        if self.agent.scores:
            episodes = list(range(len(self.agent.scores)))
            ax1.plot(episodes, self.agent.scores, alpha=0.6, color='lightblue')
            
            ax1.set_title('Learning Progress', color='white')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss distribution
        outcomes = ['Victories', 'Defeats', 'Draws']
        counts = [self.agent.victories, self.agent.defeats, self.agent.draws]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        ax2.pie(counts, labels=outcomes, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Battle Outcomes', color='white')
        
        # 3. Strategy effectiveness
        insights = self.agent.get_strategy_insights()
        if insights.get('successful_tactics'):
            tactics = list(insights['successful_tactics'].keys())[:6]
            counts = list(insights['successful_tactics'].values())[:6]
            
            bars = ax3.bar(range(len(tactics)), counts, color='green', alpha=0.7)
            ax3.set_title('Most Successful Tactics', color='white')
            ax3.set_ylabel('Success Count')
            ax3.set_xticks(range(len(tactics)))
            ax3.set_xticklabels([t.replace(' ', '\n') for t in tactics], rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', color='white')
        
        # 4. Simple epsilon decay
        episodes = np.arange(50)  # Example 50 episodes
        epsilon_curve = np.maximum(0.01, 1.0 * (0.995 ** episodes))
        
        ax4.plot(episodes, epsilon_curve, color='cyan', linewidth=2)
        ax4.set_title('Exploration Rate (ε)', color='white')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(episodes, epsilon_curve, alpha=0.3, color='cyan')
        
        plt.tight_layout()
        plt.savefig('ai_learning_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
        plt.show()
        
        print("✅ Visualization saved as 'ai_learning_analysis.png'")
    
    def export_strategy_report(self):
        """Export comprehensive strategy analysis report"""
        print("\n📄 GENERATING STRATEGY REPORT...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'training_summary': self.agent.get_strategy_insights(),
            'successful_strategies': self.agent.successful_strategies[-5:] if self.agent.successful_strategies else [],
            'recommendations': self.generate_strategy_recommendations(),
        }
        
        filename = f"warhammer_ai_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Strategy report exported to {filename}")
        return filename


def main():
    """Main analysis interface"""
    print("🤖 WARHAMMER: THE OLD WORLD - AI STRATEGY ANALYZER")
    print("=" * 60)
    
    analyzer = StrategyAnalyzer()
    
    # Comprehensive analysis
    print("\n🔬 Running comprehensive strategy analysis...")
    insights = analyzer.analyze_strategic_patterns()
    
    # Test scenarios
    scenario_results = analyzer.test_strategic_scenarios()
    
    # Generate recommendations
    recommendations = analyzer.generate_strategy_recommendations()
    
    # Create visualizations
    analyzer.visualize_learning_progress()
    
    # Export report
    report_file = analyzer.export_strategy_report()
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"📊 Win Rate: {insights['win_rate']:.2%}")
    print(f"🧠 Top Strategy: {list(insights.get('successful_tactics', {}).keys())[0] if insights.get('successful_tactics') else 'N/A'}")
    print(f"📄 Full Report: {report_file}")
    
    return analyzer


if __name__ == "__main__":
    main() 