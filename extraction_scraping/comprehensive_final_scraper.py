#!/usr/bin/env python3
"""
Final comprehensive scraper for Warhammer The Old World data from New Recruit.
This script creates a complete summary of all data we can extract.
"""

import json
import pandas as pd
from datetime import datetime
import os

def load_all_data():
    """Load all data files we've created."""
    data_summary = {
        'timestamp': datetime.now().isoformat(),
        'files_found': [],
        'players': [],
        'army_compositions': [],
        'analysis': {},
        'next_steps': []
    }
    
    # Check for player data
    if os.path.exists('tow_players_found.json'):
        try:
            with open('tow_players_found.json', 'r', encoding='utf-8') as f:
                players = json.load(f)
            data_summary['players'] = players
            data_summary['files_found'].append('tow_players_found.json')
            print(f"✅ Loaded {len(players)} Warhammer The Old World players")
        except Exception as e:
            print(f"❌ Error loading player data: {e}")
    
    # Check for army composition data
    army_files = [
        'warhammer_tow_army_compositions.json',
        'selenium_tow_army_compositions.json'
    ]
    
    for filename in army_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    army_data = json.load(f)
                data_summary['army_compositions'].extend(army_data)
                data_summary['files_found'].append(filename)
                print(f"✅ Loaded {len(army_data)} army compositions from {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    # Check for analysis files
    analysis_files = ['army_composition_analysis.json', 'known_list_analysis.json']
    for filename in analysis_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                data_summary['analysis'][filename] = analysis
                data_summary['files_found'].append(filename)
                print(f"✅ Loaded analysis from {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    return data_summary

def analyze_player_data(players):
    """Analyze the player data we've collected."""
    if not players:
        return {}
    
    analysis = {
        'total_players': len(players),
        'sample_player_ids': [p['player_id'] for p in players[:5]],
        'sample_player_names': [p['player_name'] for p in players[:5]],
        'url_patterns': list(set([p['profile_url'].split('?')[0] for p in players[:10]]))
    }
    
    return analysis

def create_army_analysis(army_compositions):
    """Analyze army composition data."""
    if not army_compositions:
        return {
            'total_compositions': 0,
            'message': 'No army compositions found'
        }
    
    # Faction analysis
    faction_counts = {}
    point_totals = []
    
    for army in army_compositions:
        faction = army.get('faction', 'Unknown')
        faction_counts[faction] = faction_counts.get(faction, 0) + 1
        
        points = army.get('total_points', 0)
        if points > 0:
            point_totals.append(points)
    
    analysis = {
        'total_compositions': len(army_compositions),
        'factions_found': faction_counts,
        'point_statistics': {
            'total_lists_with_points': len(point_totals),
            'average_points': sum(point_totals) / len(point_totals) if point_totals else 0,
            'min_points': min(point_totals) if point_totals else 0,
            'max_points': max(point_totals) if point_totals else 0
        }
    }
    
    return analysis

def generate_recommendations(data_summary):
    """Generate recommendations for next steps."""
    recommendations = []
    
    players = data_summary['players']
    army_compositions = data_summary['army_compositions']
    
    if players and len(players) > 0:
        recommendations.append({
            'category': 'Player Data',
            'status': 'SUCCESS',
            'description': f'Successfully extracted {len(players)} Warhammer The Old World players',
            'next_steps': [
                'Players can be used for targeted army list extraction',
                'Consider implementing match history extraction',
                'Focus on top-ranked players for best army compositions'
            ]
        })
    else:
        recommendations.append({
            'category': 'Player Data',
            'status': 'FAILED',
            'description': 'No player data found',
            'next_steps': [
                'Re-run run_scraper_tow.py to get player list',
                'Check if New Recruit site structure has changed'
            ]
        })
    
    if army_compositions and len(army_compositions) > 0:
        recommendations.append({
            'category': 'Army Compositions',
            'status': 'SUCCESS',
            'description': f'Successfully extracted {len(army_compositions)} army compositions',
            'next_steps': [
                'Analyze faction balance and popular builds',
                'Extract detailed unit compositions',
                'Correlate with win/loss records'
            ]
        })
    else:
        recommendations.append({
            'category': 'Army Compositions',
            'status': 'NEEDS_WORK',
            'description': 'No army compositions extracted yet',
            'next_steps': [
                'Player profiles appear to use dynamic content - need Selenium',
                'Try alternative approaches like direct tournament list URLs',
                'Consider extracting from tournament reports instead',
                'Focus on players with recent activity'
            ]
        })
    
    # Alternative data sources
    recommendations.append({
        'category': 'Alternative Sources',
        'status': 'SUGGESTION',
        'description': 'Consider additional data sources',
        'next_steps': [
            'Tournament reports from Woehammer, Bell of Lost Souls',
            'Reddit communities for The Old World',
            'BGG (BoardGameGeek) army list databases',
            'Official Games Workshop tournament results'
        ]
    })
    
    return recommendations

def create_comprehensive_report():
    """Create a comprehensive report of all extraction efforts."""
    
    print("📊 COMPREHENSIVE WARHAMMER THE OLD WORLD DATA SUMMARY")
    print("=" * 70)
    
    # Load all data
    data_summary = load_all_data()
    
    # Analyze what we have
    print(f"\n📁 FILES FOUND: {len(data_summary['files_found'])}")
    for filename in data_summary['files_found']:
        print(f"  ✅ {filename}")
    
    # Player analysis
    player_analysis = analyze_player_data(data_summary['players'])
    print(f"\n👥 PLAYER DATA ANALYSIS:")
    if player_analysis:
        print(f"  Total players: {player_analysis['total_players']}")
        print(f"  Sample names: {', '.join(player_analysis['sample_player_names'])}")
        print(f"  URL pattern: {player_analysis['url_patterns'][0] if player_analysis['url_patterns'] else 'None'}")
    else:
        print("  ❌ No player data available")
    
    # Army composition analysis
    army_analysis = create_army_analysis(data_summary['army_compositions'])
    print(f"\n⚔️ ARMY COMPOSITION ANALYSIS:")
    print(f"  Total compositions: {army_analysis['total_compositions']}")
    if army_analysis['total_compositions'] > 0:
        print(f"  Factions found: {army_analysis['factions_found']}")
        stats = army_analysis['point_statistics']
        print(f"  Average points: {stats['average_points']:.0f}")
        print(f"  Point range: {stats['min_points']} - {stats['max_points']}")
    
    # Generate recommendations
    recommendations = generate_recommendations(data_summary)
    print(f"\n💡 RECOMMENDATIONS & NEXT STEPS:")
    for rec in recommendations:
        status_emoji = "✅" if rec['status'] == 'SUCCESS' else "⚠️" if rec['status'] == 'NEEDS_WORK' else "💡"
        print(f"\n{status_emoji} {rec['category']}: {rec['description']}")
        for step in rec['next_steps']:
            print(f"     • {step}")
    
    # Save comprehensive report
    data_summary['player_analysis'] = player_analysis
    data_summary['army_analysis'] = army_analysis
    data_summary['recommendations'] = recommendations
    
    report_filename = 'comprehensive_tow_data_report.json'
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(data_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 COMPREHENSIVE REPORT SAVED TO: {report_filename}")
    
    # Create CSV summary for easy viewing
    if data_summary['players']:
        df_players = pd.DataFrame(data_summary['players'])
        df_players.to_csv('tow_players_summary.csv', index=False)
        print(f"📊 Player summary CSV: tow_players_summary.csv")
    
    if data_summary['army_compositions']:
        # Flatten army composition data for CSV
        army_flat = []
        for army in data_summary['army_compositions']:
            flat_record = {
                'player_name': army.get('player_name', ''),
                'faction': army.get('faction', ''),
                'total_points': army.get('total_points', 0),
                'list_url': army.get('list_url', ''),
                'old_world_score': army.get('old_world_score', 0)
            }
            army_flat.append(flat_record)
        
        df_armies = pd.DataFrame(army_flat)
        df_armies.to_csv('tow_army_compositions_summary.csv', index=False)
        print(f"⚔️ Army compositions CSV: tow_army_compositions_summary.csv")
    
    # Summary statistics
    print(f"\n📈 FINAL STATISTICS:")
    print(f"  Players discovered: {len(data_summary['players'])}")
    print(f"  Army compositions: {len(data_summary['army_compositions'])}")
    print(f"  Files created: {len(data_summary['files_found'])}")
    
    return data_summary

def main():
    """Main execution function."""
    print("🏛️ WARHAMMER THE OLD WORLD - FINAL DATA COMPILATION")
    print("=" * 60)
    print("This script compiles all data extracted from New Recruit")
    print("and provides a comprehensive summary and recommendations.")
    print()
    
    report = create_comprehensive_report()
    
    print(f"\n🎯 MISSION STATUS:")
    if report['players'] and len(report['players']) > 0:
        print("✅ Successfully discovered Warhammer The Old World players")
    if report['army_compositions'] and len(report['army_compositions']) > 0:
        print("✅ Successfully extracted army compositions")
        print("🎉 MISSION ACCOMPLISHED!")
    else:
        print("🔄 Player discovery successful, army extraction needs more work")
        print("💡 See recommendations for next steps")

if __name__ == "__main__":
    main() 