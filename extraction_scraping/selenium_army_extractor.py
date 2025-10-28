#!/usr/bin/env python3
"""
Selenium-based army composition extractor for Warhammer The Old World.
"""

import json
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def setup_driver(headless=True):
    """Set up Chrome driver with appropriate options."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def extract_army_lists_from_player(driver, player_url, max_lists=5):
    """Extract army list URLs from a player's profile using Selenium."""
    
    try:
        print(f"    🌐 Loading player profile: {player_url}")
        driver.get(player_url)
        
        # Wait for page to load
        wait = WebDriverWait(driver, 15)
        time.sleep(3)  # Extra wait for dynamic content
        
        # Look for army list links
        army_list_links = []
        
        # Try multiple selectors for army list links
        selectors = [
            "a[href*='/app/list/']",
            "a[href*='list']",
            ".army-list-link",
            ".list-link"
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"    ✅ Found {len(elements)} elements with selector: {selector}")
                    for element in elements[:max_lists]:
                        href = element.get_attribute('href')
                        text = element.text.strip()
                        if href and '/app/list/' in href:
                            army_list_links.append({
                                'url': href,
                                'text': text,
                                'list_id': href.split('/')[-1] if '/' in href else ''
                            })
                    break
            except:
                continue
        
        # If no direct list links, look for match details that might contain lists
        if not army_list_links:
            print("    🔍 No direct army list links found, looking for match details...")
            try:
                # Look for match history or game details
                match_elements = driver.find_elements(By.PARTIAL_LINK_TEXT, "Match")
                if not match_elements:
                    match_elements = driver.find_elements(By.PARTIAL_LINK_TEXT, "Details")
                
                print(f"    📋 Found {len(match_elements)} potential match elements")
                
                # Click on a few match details to find army lists
                for i, element in enumerate(match_elements[:3]):
                    try:
                        print(f"    🔗 Clicking match detail {i+1}")
                        element.click()
                        time.sleep(2)
                        
                        # Look for army list links in the match details
                        new_list_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/app/list/']")
                        for list_elem in new_list_elements:
                            href = list_elem.get_attribute('href')
                            if href and href not in [link['url'] for link in army_list_links]:
                                army_list_links.append({
                                    'url': href,
                                    'text': list_elem.text.strip(),
                                    'list_id': href.split('/')[-1] if '/' in href else ''
                                })
                        
                        # Go back to continue looking
                        driver.back()
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"    ⚠️ Error clicking match detail {i+1}: {e}")
                        continue
                        
            except Exception as e:
                print(f"    ❌ Error looking for match details: {e}")
        
        print(f"    📊 Found {len(army_list_links)} army list links")
        return army_list_links
        
    except Exception as e:
        print(f"    ❌ Error extracting from player profile: {e}")
        return []

def parse_army_list_with_selenium(driver, list_url):
    """Parse an army list using Selenium to handle dynamic content."""
    
    try:
        print(f"      🌐 Loading army list: {list_url}")
        driver.get(list_url)
        
        # Wait for content to load
        time.sleep(5)  # Give time for army list to fully load
        
        # Extract page content
        page_source = driver.page_source
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Extract basic information
        list_data = {
            'list_url': list_url,
            'list_id': list_url.split('/')[-1] if '/' in list_url else '',
            'title': driver.title,
            'faction': '',
            'total_points': 0,
            'units': [],
            'raw_text': page_text[:1000]  # First 1000 chars for debugging
        }
        
        # Try to extract faction
        factions = [
            'Empire of Man', 'Empire', 'High Elves', 'Dark Elves', 'Wood Elves',
            'Dwarfs', 'Orcs & Goblins', 'Warriors of Chaos', 'Daemons of Chaos',
            'Beastmen', 'Vampire Counts', 'Tomb Kings', 'Lizardmen', 'Skaven',
            'Bretonnians', 'Ogre Kingdoms', 'Chaos Dwarfs'
        ]
        
        for faction in factions:
            if faction.lower() in page_text.lower():
                list_data['faction'] = faction
                print(f"      🏛️ Detected faction: {faction}")
                break
        
        # Extract point values
        points_matches = re.findall(r'\\[(\\d+)\\s*pts?\\]', page_text, re.IGNORECASE)
        if points_matches:
            point_values = [int(p) for p in points_matches]
            list_data['total_points'] = max(point_values) if point_values else 0
            list_data['units'] = points_matches  # Store all point values for analysis
            print(f"      🎯 Found {len(points_matches)} point values, max: {list_data['total_points']}")
        
        # Try to extract unit information more specifically
        try:
            # Look for table rows or structured unit data
            unit_elements = driver.find_elements(By.CSS_SELECTOR, "tr, .unit, .army-unit")
            unit_count = len([elem for elem in unit_elements if elem.text.strip()])
            if unit_count > 0:
                print(f"      ⚔️ Found {unit_count} potential unit elements")
            
        except:
            pass
        
        # Check if this looks like a valid Old World list
        old_world_indicators = ['old world', 'empire', 'state troops', 'detachment', 'character', 'regiment']
        old_world_score = sum(1 for indicator in old_world_indicators if indicator in page_text.lower())
        list_data['old_world_score'] = old_world_score
        
        print(f"      📊 Old World content score: {old_world_score}/6")
        
        return list_data if old_world_score > 0 else None
        
    except Exception as e:
        print(f"      ❌ Error parsing army list: {e}")
        return None

def extract_army_compositions_selenium(max_players=10, max_lists_per_player=3):
    """Main function to extract army compositions using Selenium."""
    
    print("=" * 80)
    print("🏛️  SELENIUM-BASED WARHAMMER THE OLD WORLD ARMY EXTRACTION")
    print("=" * 80)
    
    # Load players
    try:
        with open('tow_players_found.json', 'r', encoding='utf-8') as f:
            players = json.load(f)
        print(f"📊 Loaded {len(players)} Warhammer The Old World players")
    except FileNotFoundError:
        print("❌ tow_players_found.json not found. Please run run_scraper_tow.py first.")
        return []
    
    all_army_data = []
    driver = setup_driver(headless=False)  # Show browser for debugging
    
    try:
        print(f"\\n📊 Processing {min(max_players, len(players))} players...")
        
        for i, player in enumerate(players[:max_players]):
            print(f"\\n👤 {i+1}/{min(max_players, len(players))}: {player['player_name']}")
            
            # Extract army list URLs from player profile
            army_list_links = extract_army_lists_from_player(
                driver, player['profile_url'], max_lists_per_player
            )
            
            if not army_list_links:
                print(f"    ⚠️ No army lists found for {player['player_name']}")
                continue
            
            # Parse each army list
            player_lists_processed = 0
            for list_info in army_list_links[:max_lists_per_player]:
                list_data = parse_army_list_with_selenium(driver, list_info['url'])
                
                if list_data:
                    # Add player context
                    list_data.update({
                        'player_name': player['player_name'],
                        'player_id': player['player_id'],
                        'player_profile': player['profile_url'],
                        'list_text_preview': list_info.get('text', '')
                    })
                    all_army_data.append(list_data)
                    player_lists_processed += 1
                    print(f"      ✅ Successfully extracted list #{player_lists_processed}")
                else:
                    print(f"      ❌ Failed to extract valid Old World data")
            
            print(f"    📊 Extracted {player_lists_processed} valid lists from {player['player_name']}")
    
    finally:
        driver.quit()
        print("🔚 Browser closed")
    
    print(f"\\n✅ Extraction complete! Found {len(all_army_data)} valid army lists")
    
    if all_army_data:
        # Save results
        output_file = 'selenium_tow_army_compositions.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_army_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved army composition data to {output_file}")
        
        # Quick analysis
        faction_counts = {}
        for army in all_army_data:
            faction = army.get('faction', 'Unknown')
            faction_counts[faction] = faction_counts.get(faction, 0) + 1
        
        print(f"\\n📊 FACTION DISTRIBUTION:")
        for faction, count in sorted(faction_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {faction:<20}: {count:3d} lists")
        
        return all_army_data
    else:
        print("\\n❌ No valid army compositions extracted")
        return []

def main():
    """Main execution function."""
    army_data = extract_army_compositions_selenium(max_players=5, max_lists_per_player=2)
    
    if army_data:
        print(f"\\n🎉 Successfully extracted {len(army_data)} Warhammer The Old World army compositions!")
        print("\\n📁 Output file: selenium_tow_army_compositions.json")
    else:
        print("\\n💡 Consider:")
        print("  - Trying with headless=False to debug browser interactions")
        print("  - Increasing wait times for slower connections")
        print("  - Checking if the site structure has changed")

if __name__ == "__main__":
    main() 