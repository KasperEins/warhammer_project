#!/usr/bin/env python3
"""
Extract army lists and match data from Warhammer The Old World players.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import time
from urllib.parse import urljoin, urlparse
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.newrecruit.eu"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def load_tow_players():
    """Load the Warhammer The Old World players from the JSON file."""
    try:
        with open('tow_players_found.json', 'r', encoding='utf-8') as f:
            players = json.load(f)
        print(f"📊 Loaded {len(players)} Warhammer The Old World players")
        return players
    except FileNotFoundError:
        print("❌ tow_players_found.json not found. Please run run_scraper_tow.py first.")
        return []

def extract_player_match_data(player_url, session=None, use_selenium=False):
    """Extract match history and army lists from a player's profile."""
    
    if use_selenium:
        return extract_player_data_selenium(player_url)
    else:
        return extract_player_data_requests(player_url, session)

def extract_player_data_requests(player_url, session=None):
    """Extract player data using requests (may not work with dynamic content)."""
    if session is None:
        session = requests.Session()
    
    try:
        time.sleep(1)  # Rate limiting
        response = session.get(player_url, headers=HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for match history
        match_links = soup.find_all('a', string=re.compile(r'Match Details'))
        army_list_links = soup.find_all('a', href=re.compile(r'/app/list/'))
        
        print(f"  📊 Found {len(match_links)} match details, {len(army_list_links)} army lists")
        
        return {
            'match_links': [urljoin(player_url, link.get('href', '')) for link in match_links],
            'army_list_links': [urljoin(player_url, link.get('href', '')) for link in army_list_links]
        }
        
    except Exception as e:
        print(f"  ❌ Error extracting data from {player_url}: {e}")
        return {'match_links': [], 'army_list_links': []}

def extract_player_data_selenium(player_url):
    """Extract player data using Selenium for dynamic content."""
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(player_url)
        time.sleep(3)  # Wait for dynamic content
        
        # Look for match history section
        match_elements = driver.find_elements(By.PARTIAL_LINK_TEXT, "Match Details")
        army_list_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/app/list/']")
        
        match_links = [elem.get_attribute('href') for elem in match_elements if elem.get_attribute('href')]
        army_list_links = [elem.get_attribute('href') for elem in army_list_elements if elem.get_attribute('href')]
        
        print(f"  📊 Found {len(match_links)} match details, {len(army_list_links)} army lists")
        
        return {
            'match_links': match_links,
            'army_list_links': army_list_links
        }
        
    except Exception as e:
        print(f"  ❌ Error with Selenium: {e}")
        return {'match_links': [], 'army_list_links': []}
    finally:
        driver.quit()

def parse_army_list(list_url, session=None):
    """Parse an army list page to extract composition data."""
    if session is None:
        session = requests.Session()
    
    try:
        time.sleep(1)  # Rate limiting
        response = session.get(list_url, headers=HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract basic list info
        list_data = {
            'list_url': list_url,
            'list_id': list_url.split('/')[-1] if '/' in list_url else '',
            'title': '',
            'faction': '',
            'total_points': 0,
            'units': [],
            'raw_text': soup.get_text()
        }
        
        # Extract title
        title_element = soup.find('title')
        if title_element:
            list_data['title'] = title_element.get_text().strip()
        
        # Try to extract faction from title or content
        page_text = soup.get_text()
        
        # Common The Old World factions
        factions = [
            'Empire of Man', 'Empire', 'High Elves', 'Dark Elves', 'Wood Elves',
            'Dwarfs', 'Orcs & Goblins', 'Warriors of Chaos', 'Daemons of Chaos',
            'Beastmen', 'Vampire Counts', 'Tomb Kings', 'Lizardmen', 'Skaven',
            'Bretonnians', 'Ogre Kingdoms', 'Chaos Dwarfs'
        ]
        
        for faction in factions:
            if faction.lower() in page_text.lower():
                list_data['faction'] = faction
                break
        
        # Extract point values
        points_matches = re.findall(r'\[(\d+)pts\]', page_text)
        if points_matches:
            # Try to find the total (usually the largest number)
            point_values = [int(p) for p in points_matches]
            list_data['total_points'] = max(point_values) if point_values else 0
        
        # Store unit information (basic extraction)
        list_data['units'] = points_matches  # For now, just store point values
        
        return list_data
        
    except Exception as e:
        print(f"  ❌ Error parsing army list {list_url}: {e}")
        return None

def extract_army_compositions(max_players=20, max_lists_per_player=5):
    """Main function to extract army compositions from players."""
    
    print("=" * 70)
    print("🏛️  EXTRACTING WARHAMMER THE OLD WORLD ARMY COMPOSITIONS")
    print("=" * 70)
    
    # Load players
    players = load_tow_players()
    if not players:
        return []
    
    session = requests.Session()
    all_army_data = []
    
    print(f"\n📊 Processing {min(max_players, len(players))} players...")
    
    for i, player in enumerate(players[:max_players]):
        print(f"\n👤 {i+1}/{min(max_players, len(players))}: {player['player_name']}")
        print(f"   Profile: {player['profile_url']}")
        
        # Extract match data from player profile
        player_data = extract_player_data_requests(player['profile_url'], session)
        
        # Process army lists
        army_lists = player_data.get('army_list_links', [])
        processed_lists = 0
        
        for list_url in army_lists[:max_lists_per_player]:
            if processed_lists >= max_lists_per_player:
                break
                
            print(f"    📋 Parsing army list: {list_url}")
            list_data = parse_army_list(list_url, session)
            
            if list_data:
                # Add player context
                list_data.update({
                    'player_name': player['player_name'],
                    'player_id': player['player_id'],
                    'player_profile': player['profile_url']
                })
                all_army_data.append(list_data)
                processed_lists += 1
                print(f"      ✅ Faction: {list_data.get('faction', 'Unknown')}, Points: {list_data.get('total_points', 0)}")
            else:
                print(f"      ❌ Failed to parse list")
        
        print(f"    📊 Extracted {processed_lists} army lists from {player['player_name']}")
    
    print(f"\n✅ Extraction complete! Found {len(all_army_data)} army lists total")
    
    # Save the results
    output_file = 'warhammer_tow_army_compositions.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_army_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved army composition data to {output_file}")
    
    # Create summary analysis
    analyze_army_data(all_army_data)
    
    return all_army_data

def analyze_army_data(army_data):
    """Analyze the extracted army data."""
    if not army_data:
        print("No army data to analyze")
        return
    
    print(f"\n📊 ARMY COMPOSITION ANALYSIS")
    print("=" * 50)
    
    # Faction distribution
    faction_counts = {}
    point_ranges = {}
    
    for army in army_data:
        faction = army.get('faction', 'Unknown')
        points = army.get('total_points', 0)
        
        faction_counts[faction] = faction_counts.get(faction, 0) + 1
        
        if faction not in point_ranges:
            point_ranges[faction] = []
        point_ranges[faction].append(points)
    
    print(f"📋 Faction Distribution:")
    for faction, count in sorted(faction_counts.items(), key=lambda x: x[1], reverse=True):
        avg_points = sum(point_ranges[faction]) / len(point_ranges[faction]) if point_ranges[faction] else 0
        print(f"  {faction:<20}: {count:3d} lists (avg {avg_points:.0f} pts)")
    
    # Save analysis
    analysis = {
        'total_lists': len(army_data),
        'faction_distribution': faction_counts,
        'average_points_by_faction': {
            faction: sum(points) / len(points) if points else 0 
            for faction, points in point_ranges.items()
        }
    }
    
    with open('army_composition_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved analysis to army_composition_analysis.json")

def main():
    """Main function."""
    # Extract army compositions from first 20 players
    army_data = extract_army_compositions(max_players=20, max_lists_per_player=3)
    
    if army_data:
        print(f"\n🎉 Successfully extracted {len(army_data)} army compositions!")
        print("📁 Files created:")
        print("  - warhammer_tow_army_compositions.json (raw army data)")
        print("  - army_composition_analysis.json (analysis summary)")
    else:
        print("\n❌ No army data extracted")

if __name__ == "__main__":
    main() 