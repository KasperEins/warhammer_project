#!/usr/bin/env python3
"""
WARHAMMER: THE OLD WORLD - STRATEGY EVOLUTION TRACKER
===================================================
Advanced analysis of AI strategy evolution during training
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple

class StrategyEvolutionTracker:
    """Track and analyze how AI strategies evolve during training"""
    
    def __init__(self, model_path='warhammer_ai_model.pth', log_path='training_log.jsonl'):
        self.model_path = model_path
        self.log_path = log_path
        self.action_names = [
            'Move North', 'Move South', 'Move East', 'Move West',
            'Move NE', 'Move NW', 'Move SE', 'Move SW',
            'Artillery Strike', 'Cavalry Charge', 'Defensive Formation',
            'Flanking', 'Mass Shooting'
        ]
        
    def load_training_data(self):
        """Load training data from model checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            return checkpoint
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None
    
    def analyze_strategy_evolution(self, checkpoint):
        """Analyze how strategies evolved over training"""
        if not checkpoint:
            return None
            
        # Extract data
        episodes = len(checkpoint.get('scores', []))
        strategies = checkpoint.get('successful_strategies', [])
        
        # Extract action sequences from strategy dictionaries
        all_actions = []
        strategy_patterns = []
        for strategy in strategies:
            if isinstance(strategy, dict) and 'actions' in strategy:
                all_actions.extend(strategy['actions'])
                # Convert action sequence to string pattern for analysis
                pattern = '-'.join([self.action_names[a] if a < len(self.action_names) else f'Action{a}' 
                                  for a in strategy['actions'][:3]])  # First 3 actions as pattern
                strategy_patterns.append(pattern)
        
        analysis = {
            'total_episodes': episodes,
            'total_strategies': len(strategies),
            'strategy_frequency': Counter(strategy_patterns),
            'action_evolution': self._analyze_action_evolution(all_actions),
            'q_value_convergence': {},  # Not available in current checkpoint
            'learning_phases': self._identify_learning_phases(checkpoint),
            'strategic_preferences': self._analyze_strategic_preferences(all_actions),
        }
        
        return analysis
    
    def _analyze_action_evolution(self, action_history):
        """Analyze how action preferences changed over time"""
        if not action_history:
            return {}
            
        # Divide training into phases
        phases = ['Early (0-25%)', 'Mid-Early (25-50%)', 'Mid-Late (50-75%)', 'Late (75-100%)']
        phase_size = len(action_history) // 4
        
        phase_analysis = {}
        for i, phase in enumerate(phases):
            start_idx = i * phase_size
            end_idx = (i + 1) * phase_size if i < 3 else len(action_history)
            phase_actions = action_history[start_idx:end_idx]
            
            if phase_actions:
                action_counts = Counter(phase_actions)
                total_actions = sum(action_counts.values())
                
                phase_analysis[phase] = {
                    'action_distribution': {
                        self.action_names[action]: count / total_actions 
                        for action, count in action_counts.items() 
                        if action < len(self.action_names)
                    },
                    'most_used_action': self.action_names[max(action_counts.keys())] if action_counts else 'None',
                    'action_diversity': len(action_counts) / len(self.action_names),
                    'total_actions': total_actions
                }
        
        return phase_analysis
    
    def _analyze_q_value_convergence(self, q_values_history):
        """Analyze how Q-values converged during training"""
        if not q_values_history:
            return {}
            
        # Calculate Q-value statistics over time
        convergence_data = {
            'mean_q_values': [],
            'max_q_values': [],
            'q_value_variance': [],
            'action_preferences': []
        }
        
        window_size = max(1, len(q_values_history) // 100)  # 100 data points
        
        for i in range(0, len(q_values_history), window_size):
            window = q_values_history[i:i+window_size]
            if window:
                window_array = np.array(window)
                convergence_data['mean_q_values'].append(np.mean(window_array))
                convergence_data['max_q_values'].append(np.max(window_array))
                convergence_data['q_value_variance'].append(np.var(window_array))
                convergence_data['action_preferences'].append(np.argmax(np.mean(window_array, axis=0)))
        
        return convergence_data
    
    def _identify_learning_phases(self, checkpoint):
        """Identify distinct learning phases"""
        scores = checkpoint.get('scores', [])
        if len(scores) < 100:
            return {}
            
        # Calculate moving averages
        window = 100
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        
        # Identify phase transitions (significant score improvements)
        phase_transitions = []
        improvement_threshold = 50  # Score improvement threshold
        
        for i in range(1, len(moving_avg)):
            if moving_avg[i] - moving_avg[i-1] > improvement_threshold:
                phase_transitions.append(i + window - 1)  # Adjust for convolution offset
        
        phases = {
            'exploration_phase': {
                'episodes': list(range(0, min(1000, len(scores)))),
                'avg_score': np.mean(scores[:min(1000, len(scores))]),
                'description': 'Initial random exploration and basic learning'
            }
        }
        
        if phase_transitions:
            phases['breakthrough_phase'] = {
                'episodes': list(range(phase_transitions[0], min(phase_transitions[0] + 1000, len(scores)))),
                'avg_score': np.mean(scores[phase_transitions[0]:min(phase_transitions[0] + 1000, len(scores))]),
                'description': 'First major strategic breakthrough'
            }
            
            if len(phase_transitions) > 1:
                phases['mastery_phase'] = {
                    'episodes': list(range(phase_transitions[-1], len(scores))),
                    'avg_score': np.mean(scores[phase_transitions[-1]:]),
                    'description': 'Strategic refinement and mastery'
                }
        
        return phases
    
    def _analyze_strategic_preferences(self, all_actions):
        """Analyze strategic preference patterns"""
        if not all_actions:
            return {}
            
        # Count individual actions
        action_analysis = Counter(all_actions)
        total_actions = len(all_actions)
        
        # Convert action indices to names
        action_names_count = {}
        for action_idx, count in action_analysis.items():
            if action_idx < len(self.action_names):
                action_names_count[self.action_names[action_idx]] = count
        
        # Categorize strategies
        strategy_categories = {
            'Aggressive': ['Artillery Strike', 'Cavalry Charge', 'Mass Shooting'],
            'Defensive': ['Defensive Formation'],
            'Tactical': ['Flanking'],
            'Positional': ['Move North', 'Move South', 'Move East', 'Move West', 
                          'Move NE', 'Move NW', 'Move SE', 'Move SW']
        }
        
        category_preferences = {}
        for category, actions in strategy_categories.items():
            category_count = sum(action_names_count.get(action, 0) for action in actions)
            category_preferences[category] = {
                'count': category_count,
                'percentage': category_count / total_actions if total_actions > 0 else 0,
                'actions': {action: action_names_count.get(action, 0) for action in actions}
            }
        
        return {
            'strategy_distribution': action_names_count,
            'category_preferences': category_preferences,
            'most_preferred_strategy': max(action_names_count.keys(), key=action_names_count.get) if action_names_count else 'None',
            'strategy_diversity': len(action_names_count) / len(self.action_names)
        }
    
    def generate_evolution_visualizations(self, analysis):
        """Generate comprehensive visualization of strategy evolution"""
        if not analysis:
            return
            
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Strategy Distribution Over Time
        ax1 = plt.subplot(3, 3, 1)
        if 'action_evolution' in analysis:
            phases = list(analysis['action_evolution'].keys())
            artillery_usage = [analysis['action_evolution'][phase]['action_distribution'].get('Artillery Strike', 0) 
                              for phase in phases]
            cavalry_usage = [analysis['action_evolution'][phase]['action_distribution'].get('Cavalry Charge', 0) 
                            for phase in phases]
            defensive_usage = [analysis['action_evolution'][phase]['action_distribution'].get('Defensive Formation', 0) 
                              for phase in phases]
            
            x = np.arange(len(phases))
            width = 0.25
            
            ax1.bar(x - width, artillery_usage, width, label='Artillery Strike', color='red', alpha=0.8)
            ax1.bar(x, cavalry_usage, width, label='Cavalry Charge', color='blue', alpha=0.8)
            ax1.bar(x + width, defensive_usage, width, label='Defensive Formation', color='green', alpha=0.8)
            
            ax1.set_xlabel('Training Phase')
            ax1.set_ylabel('Usage Frequency')
            ax1.set_title('Strategic Preference Evolution', color='white', fontsize=12, weight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(phases, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Action Diversity Over Time
        ax2 = plt.subplot(3, 3, 2)
        if 'action_evolution' in analysis:
            diversity_scores = [analysis['action_evolution'][phase]['action_diversity'] 
                               for phase in phases]
            ax2.plot(phases, diversity_scores, marker='o', linewidth=3, color='cyan')
            ax2.set_xlabel('Training Phase')
            ax2.set_ylabel('Action Diversity Score')
            ax2.set_title('Tactical Diversity Evolution', color='white', fontsize=12, weight='bold')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Q-Value Convergence
        ax3 = plt.subplot(3, 3, 3)
        if 'q_value_convergence' in analysis and analysis['q_value_convergence']:
            conv_data = analysis['q_value_convergence']
            if conv_data['mean_q_values']:
                episodes_range = np.linspace(0, analysis['total_episodes'], len(conv_data['mean_q_values']))
                ax3.plot(episodes_range, conv_data['mean_q_values'], color='orange', linewidth=2, label='Mean Q-Value')
                ax3.plot(episodes_range, conv_data['max_q_values'], color='yellow', linewidth=2, label='Max Q-Value')
                ax3.set_xlabel('Episodes')
                ax3.set_ylabel('Q-Value')
                ax3.set_title('Q-Value Convergence', color='white', fontsize=12, weight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Strategy Category Preferences
        ax4 = plt.subplot(3, 3, 4)
        if 'strategic_preferences' in analysis and 'category_preferences' in analysis['strategic_preferences']:
            categories = list(analysis['strategic_preferences']['category_preferences'].keys())
            percentages = [analysis['strategic_preferences']['category_preferences'][cat]['percentage'] * 100 
                          for cat in categories]
            colors = ['red', 'green', 'blue', 'purple']
            
            wedges, texts, autotexts = ax4.pie(percentages, labels=categories, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            ax4.set_title('Strategic Category Distribution', color='white', fontsize=12, weight='bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
        
        # 5. Most Used Actions
        ax5 = plt.subplot(3, 3, 5)
        if 'strategic_preferences' in analysis and 'strategy_distribution' in analysis['strategic_preferences']:
            strategy_dist = analysis['strategic_preferences']['strategy_distribution']
            # Get top 8 strategies
            top_strategies = dict(sorted(strategy_dist.items(), key=lambda x: x[1], reverse=True)[:8])
            
            actions = list(top_strategies.keys())
            counts = list(top_strategies.values())
            
            bars = ax5.barh(actions, counts, color='gold', alpha=0.8)
            ax5.set_xlabel('Usage Count')
            ax5.set_title('Most Used Actions', color='white', fontsize=12, weight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax5.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                        str(count), ha='left', va='center', color='white', weight='bold')
        
        # 6. Learning Phase Analysis
        ax6 = plt.subplot(3, 3, 6)
        if 'learning_phases' in analysis:
            phases_data = analysis['learning_phases']
            phase_names = list(phases_data.keys())
            avg_scores = [phases_data[phase]['avg_score'] for phase in phase_names]
            
            bars = ax6.bar(phase_names, avg_scores, color=['red', 'orange', 'green'][:len(phase_names)], alpha=0.8)
            ax6.set_xlabel('Learning Phase')
            ax6.set_ylabel('Average Score')
            ax6.set_title('Learning Phase Performance', color='white', fontsize=12, weight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, avg_scores):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(min(avg_scores))*0.05, 
                        f'{score:.0f}', ha='center', va='bottom', color='white', weight='bold')
        
        # 7. Training Progress Summary
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        summary_text = f"""
🤖 AI TRAINING EVOLUTION SUMMARY

📊 Total Episodes: {analysis['total_episodes']:,}
🎯 Strategies Learned: {analysis['total_strategies']}
🏆 Most Preferred: {analysis['strategic_preferences'].get('most_preferred_strategy', 'N/A')}
🎲 Strategy Diversity: {analysis['strategic_preferences'].get('strategy_diversity', 0):.2f}

🔥 Top Strategic Categories:
"""
        
        if 'strategic_preferences' in analysis and 'category_preferences' in analysis['strategic_preferences']:
            cat_prefs = analysis['strategic_preferences']['category_preferences']
            sorted_cats = sorted(cat_prefs.items(), key=lambda x: x[1]['percentage'], reverse=True)
            for cat, data in sorted_cats[:3]:
                summary_text += f"   {cat}: {data['percentage']:.1%}\n"
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11, 
                verticalalignment='top', color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        # 8. Action Preference Heatmap
        ax8 = plt.subplot(3, 3, 8)
        if 'action_evolution' in analysis:
            # Create heatmap data
            phase_names = list(analysis['action_evolution'].keys())
            action_matrix = []
            
            for phase in phase_names:
                phase_actions = analysis['action_evolution'][phase]['action_distribution']
                action_row = [phase_actions.get(action, 0) for action in self.action_names[:8]]  # Top 8 actions
                action_matrix.append(action_row)
            
            if action_matrix:
                im = ax8.imshow(action_matrix, cmap='hot', aspect='auto', interpolation='nearest')
                ax8.set_xticks(range(8))
                ax8.set_xticklabels(self.action_names[:8], rotation=45, ha='right')
                ax8.set_yticks(range(len(phase_names)))
                ax8.set_yticklabels(phase_names)
                ax8.set_title('Action Usage Heatmap', color='white', fontsize=12, weight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax8)
                cbar.set_label('Usage Frequency', color='white')
        
        # 9. Strategy Effectiveness Timeline
        ax9 = plt.subplot(3, 3, 9)
        if 'strategic_preferences' in analysis:
            # Create effectiveness metric (usage count as proxy)
            strategy_dist = analysis['strategic_preferences']['strategy_distribution']
            top_5_strategies = dict(sorted(strategy_dist.items(), key=lambda x: x[1], reverse=True)[:5])
            
            strategies = list(top_5_strategies.keys())
            effectiveness = list(top_5_strategies.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
            
            bars = ax9.bar(range(len(strategies)), effectiveness, color=colors, alpha=0.8)
            ax9.set_xlabel('Strategy Rank')
            ax9.set_ylabel('Usage Count')
            ax9.set_title('Top 5 Strategy Effectiveness', color='white', fontsize=12, weight='bold')
            ax9.set_xticks(range(len(strategies)))
            ax9.set_xticklabels([f"#{i+1}" for i in range(len(strategies))])
            ax9.grid(True, alpha=0.3, axis='y')
            
            # Add strategy names as legend
            legend_labels = [f"#{i+1}: {strategy}" for i, strategy in enumerate(strategies)]
            ax9.legend(bars, legend_labels, loc='upper right', fontsize=8)
        
        plt.suptitle(f'🧠 WARHAMMER AI STRATEGY EVOLUTION ANALYSIS - {analysis["total_episodes"]:,} Episodes', 
                    fontsize=16, color='gold', weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.savefig('strategy_evolution_analysis.png', dpi=150, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
        plt.close()
        
        print(f"📊 Strategy evolution analysis saved: strategy_evolution_analysis.png")
    
    def generate_detailed_report(self, analysis):
        """Generate detailed text report of strategy evolution"""
        if not analysis:
            return
            
        report = f"""
🤖 WARHAMMER: THE OLD WORLD - AI STRATEGY EVOLUTION REPORT
{'='*65}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 TRAINING OVERVIEW
{'='*20}
Total Episodes Completed: {analysis['total_episodes']:,}
Total Strategies Learned: {analysis['total_strategies']}
Strategy Diversity Score: {analysis['strategic_preferences'].get('strategy_diversity', 0):.3f}

🎯 STRATEGIC PREFERENCES
{'='*25}
Most Preferred Strategy: {analysis['strategic_preferences'].get('most_preferred_strategy', 'N/A')}

Strategy Category Breakdown:
"""
        
        if 'strategic_preferences' in analysis and 'category_preferences' in analysis['strategic_preferences']:
            for category, data in analysis['strategic_preferences']['category_preferences'].items():
                report += f"  {category:12}: {data['percentage']:6.1%} ({data['count']:,} uses)\n"
        
        report += f"""
🔄 ACTION EVOLUTION
{'='*18}
"""
        
        if 'action_evolution' in analysis:
            for phase, data in analysis['action_evolution'].items():
                report += f"\n{phase}:\n"
                report += f"  Most Used Action: {data['most_used_action']}\n"
                report += f"  Action Diversity: {data['action_diversity']:.3f}\n"
                report += f"  Total Actions: {data['total_actions']:,}\n"
                
                # Top 3 actions in this phase
                sorted_actions = sorted(data['action_distribution'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
                report += "  Top Actions:\n"
                for action, freq in sorted_actions:
                    report += f"    {action:15}: {freq:6.1%}\n"
        
        report += f"""
🧠 LEARNING PHASES
{'='*17}
"""
        
        if 'learning_phases' in analysis:
            for phase_name, phase_data in analysis['learning_phases'].items():
                report += f"\n{phase_name.title()}:\n"
                report += f"  Episodes: {len(phase_data['episodes']):,}\n"
                report += f"  Average Score: {phase_data['avg_score']:.1f}\n"
                report += f"  Description: {phase_data['description']}\n"
        
        report += f"""
📈 KEY INSIGHTS
{'='*14}
"""
        
        # Generate insights based on the data
        insights = []
        
        if 'strategic_preferences' in analysis:
            cat_prefs = analysis['strategic_preferences']['category_preferences']
            if cat_prefs.get('Aggressive', {}).get('percentage', 0) > 0.5:
                insights.append("🔥 AI developed a highly aggressive playstyle, favoring direct combat")
            elif cat_prefs.get('Defensive', {}).get('percentage', 0) > 0.3:
                insights.append("🛡️ AI learned to prioritize defensive positioning and formation")
            
            strategy_dist = analysis['strategic_preferences']['strategy_distribution']
            if strategy_dist.get('Artillery Strike', 0) > strategy_dist.get('Cavalry Charge', 0):
                insights.append("🎯 Artillery strikes emerged as the preferred tactical choice")
            
            diversity = analysis['strategic_preferences'].get('strategy_diversity', 0)
            if diversity > 0.7:
                insights.append("🎲 AI maintained high strategic diversity throughout training")
            elif diversity < 0.3:
                insights.append("🎯 AI converged to specialized, focused strategies")
        
        if 'action_evolution' in analysis:
            phases = list(analysis['action_evolution'].keys())
            if len(phases) >= 2:
                early_artillery = analysis['action_evolution'][phases[0]]['action_distribution'].get('Artillery Strike', 0)
                late_artillery = analysis['action_evolution'][phases[-1]]['action_distribution'].get('Artillery Strike', 0)
                
                if late_artillery > early_artillery + 0.2:
                    insights.append("📈 Artillery usage increased significantly over training")
                elif early_artillery > late_artillery + 0.2:
                    insights.append("📉 AI moved away from artillery toward other tactics")
        
        if not insights:
            insights.append("🤔 AI is still in early learning phases, patterns emerging")
        
        for insight in insights:
            report += f"  {insight}\n"
        
        report += f"""
{'='*65}
End of Report
"""
        
        # Save report
        with open('strategy_evolution_report.txt', 'w') as f:
            f.write(report)
        
        print(f"📄 Detailed report saved: strategy_evolution_report.txt")
        return report
    
    def run_analysis(self):
        """Run complete strategy evolution analysis"""
        print("🔍 ANALYZING AI STRATEGY EVOLUTION")
        print("=" * 40)
        
        # Load training data
        print("📊 Loading training data...")
        checkpoint = self.load_training_data()
        
        if not checkpoint:
            print("❌ No training data found. Make sure training is running.")
            return None
        
        # Analyze evolution
        print("🧠 Analyzing strategy evolution...")
        analysis = self.analyze_strategy_evolution(checkpoint)
        
        if not analysis:
            print("❌ Failed to analyze strategy evolution.")
            return None
        
        # Generate visualizations
        print("📊 Generating evolution visualizations...")
        self.generate_evolution_visualizations(analysis)
        
        # Generate detailed report
        print("📄 Generating detailed report...")
        report = self.generate_detailed_report(analysis)
        
        print("\n✅ Strategy evolution analysis complete!")
        print(f"📊 Visualizations: strategy_evolution_analysis.png")
        print(f"📄 Report: strategy_evolution_report.txt")
        
        return analysis


def main():
    """Main analysis interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze AI Strategy Evolution')
    parser.add_argument('--model-path', default='warhammer_ai_model.pth', help='Path to model file')
    parser.add_argument('--log-path', default='training_log.jsonl', help='Path to training log')
    
    args = parser.parse_args()
    
    tracker = StrategyEvolutionTracker(
        model_path=args.model_path,
        log_path=args.log_path
    )
    
    analysis = tracker.run_analysis()
    
    if analysis:
        print(f"\n🎯 QUICK SUMMARY:")
        print(f"   Episodes: {analysis['total_episodes']:,}")
        print(f"   Strategies: {analysis['total_strategies']}")
        print(f"   Most Preferred: {analysis['strategic_preferences'].get('most_preferred_strategy', 'N/A')}")


if __name__ == "__main__":
    main() 