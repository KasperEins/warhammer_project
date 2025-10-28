#!/usr/bin/env python3
"""
Test script to parse a known army list URL.
"""

import requests
from bs4 import BeautifulSoup
import re
import json

def test_known_army_list():
    """Test parsing a known army list URL."""
    
    # From our research, we know this URL exists
    known_url = "https://www.newrecruit.eu/app/list/Wfirs"  # Empire list from Woehammer tournament
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"🔍 Testing known army list: {known_url}")
    
    try:
        response = requests.get(known_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(f"✅ Successfully fetched list (status: {response.status_code})")
        print(f"📄 Content length: {len(response.text)} characters")
        
        # Extract title
        title = soup.find('title')
        if title:
            print(f"📋 Title: {title.get_text()}")
        
        # Get page text for analysis
        page_text = soup.get_text()
        
        # Look for faction indicators
        factions = ['Empire', 'High Elves', 'Orcs', 'Goblins', 'Dwarfs', 'Chaos']
        found_factions = []
        for faction in factions:
            if faction.lower() in page_text.lower():
                found_factions.append(faction)
        
        if found_factions:
            print(f"🏛️ Detected factions: {', '.join(found_factions)}")
        
        # Look for point values
        points_matches = re.findall(r'\\[(\\d+)pts\\]', page_text)
        if points_matches:
            point_values = [int(p) for p in points_matches]
            print(f"🎯 Found {len(points_matches)} point values: {points_matches[:10]}...")  # First 10
            print(f"📊 Point range: {min(point_values)} - {max(point_values)} pts")
        
        # Look for unit names (common patterns)
        unit_patterns = [
            r'(\\d+)\\s+(\\w+\\s+\\w+)',  # "10 State Troops"
            r'(\\w+\\s+\\w+)\\s+\\[(\\d+)pts\\]',  # "State Troops [100pts]"
        ]
        
        units_found = []
        for pattern in unit_patterns:
            matches = re.findall(pattern, page_text)
            if matches:
                units_found.extend(matches[:5])  # First 5 matches
        
        if units_found:
            print(f"⚔️ Sample units found: {units_found}")
        
        # Check if this looks like Old World content
        old_world_indicators = ['old world', 'empire', 'warhammer fantasy', 'state troops', 'detachment']
        old_world_score = sum(1 for indicator in old_world_indicators if indicator in page_text.lower())
        print(f"🏛️ Old World content score: {old_world_score}/5")
        
        # Save for manual inspection
        with open('known_list_sample.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"💾 Saved full HTML to known_list_sample.html")
        
        # Extract key information
        list_data = {
            'url': known_url,
            'title': title.get_text() if title else '',
            'detected_factions': found_factions,
            'point_values': points_matches,
            'sample_units': units_found,
            'old_world_score': old_world_score,
            'content_length': len(page_text)
        }
        
        with open('known_list_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(list_data, f, indent=2, ensure_ascii=False)
        print(f"📊 Saved analysis to known_list_analysis.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fetching known list: {e}")
        return False

def test_player_profile():
    """Test accessing a player profile directly."""
    
    # Use one of the player URLs we found
    player_url = "https://www.newrecruit.eu/app/Profile?id=65c1ed233771bd78230a7539"  # Arvid_dc
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"\\n🔍 Testing player profile: {player_url}")
    
    try:
        response = requests.get(player_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(f"✅ Successfully fetched profile (status: {response.status_code})")
        print(f"📄 Content length: {len(response.text)} characters")
        
        # Look for dynamic content indicators
        page_text = soup.get_text()
        
        # Check for Vue.js or dynamic content
        if 'vue' in page_text.lower() or len(page_text) < 500:
            print("⚠️ This appears to be a dynamic/Vue.js page")
        
        # Look for any list links in the HTML
        list_links = soup.find_all('a', href=re.compile(r'/app/list/'))
        print(f"📋 Found {len(list_links)} army list links in HTML")
        
        if list_links:
            for i, link in enumerate(list_links[:3]):  # First 3
                href = link.get('href', '')
                text = link.get_text(strip=True)
                print(f"  {i+1}. {text} -> {href}")
        
        # Save profile HTML for inspection
        with open('player_profile_sample.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"💾 Saved profile HTML to player_profile_sample.html")
        
        return len(list_links) > 0
        
    except Exception as e:
        print(f"❌ Error fetching player profile: {e}")
        return False

def main():
    """Test both known list and player profile."""
    print("🧪 TESTING NEW RECRUIT DATA EXTRACTION")
    print("=" * 50)
    
    # Test known army list
    list_success = test_known_army_list()
    
    # Test player profile
    profile_success = test_player_profile()
    
    print(f"\\n📊 TEST RESULTS:")
    print(f"  Known army list parsing: {'✅ Success' if list_success else '❌ Failed'}")
    print(f"  Player profile access: {'✅ Success' if profile_success else '❌ Failed'}")
    
    if list_success:
        print("\\n💡 Army list parsing works! We can extract data from direct list URLs.")
    
    if not profile_success:
        print("\\n💡 Player profiles need Selenium due to dynamic content loading.")

if __name__ == "__main__":
    main() 