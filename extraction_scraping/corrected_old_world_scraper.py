#!/usr/bin/env python3
"""
CORRECTED OLD WORLD SCRAPER - Extracts players AFTER selecting Old World
"""

import json
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def setup_driver(headless=False):
    """Set up Chrome driver with appropriate options."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def handle_cookie_consent(driver):
    """Handle the FC (Full Cookie) consent dialog."""
    
    print("    🍪 Checking for cookie consent dialog...")
    
    try:
        consent_selectors = [
            ".fc-consent-root",
            ".fc-primary-button", 
            ".fc-button",
            "[class*='fc-button']",
            "[class*='consent']"
        ]
        
        for selector in consent_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"    ✅ Found consent element: {selector}")
                    element = elements[0]
                    driver.execute_script("arguments[0].scrollIntoView();", element)
                    time.sleep(1)
                    
                    try:
                        element.click()
                    except:
                        driver.execute_script("arguments[0].click();", element)
                    
                    time.sleep(3)
                    print("    ✅ Clicked consent button")
                    return True
            except:
                continue
        
        print("    ⚠️ No consent dialog found")
        return True
        
    except Exception as e:
        print(f"    ❌ Error handling consent: {e}")
        return True

def select_old_world_and_extract_players(driver):
    """Select Old World from dropdown AND extract players from that specific ladder."""
    
    print("🎯 Going to ladder and selecting Warhammer The Old World...")
    
    try:
        # Go to ladder page
        driver.get("https://www.newrecruit.eu/ladder")
        time.sleep(5)
        
        # Handle cookie consent
        handle_cookie_consent(driver)
        
        # Find and select game system dropdown
        dropdown_selectors = ["select", "[data-v-5065dd38]", ".form-select"]
        
        dropdown_found = False
        for selector in dropdown_selectors:
            try:
                dropdowns = driver.find_elements(By.CSS_SELECTOR, selector)
                for dropdown in dropdowns:
                    select = Select(dropdown)
                    options = select.options
                    
                    # Look for Old World option
                    for i, option in enumerate(options):
                        option_text = option.text.strip()
                        if 'old world' in option_text.lower():
                            print(f"✅ Found: '{option_text}' at index {i}")
                            select.select_by_index(i)
                            time.sleep(5)  # Wait for content to load
                            print(f"✅ Selected Warhammer The Old World!")
                            dropdown_found = True
                            break
                    
                    if dropdown_found:
                        break
                        
            except Exception as e:
                continue
        
        if not dropdown_found:
            print("❌ Could not select Old World")
            return []
        
        # NOW extract players from the Old World ladder
        print("🔍 Extracting players from Old World ladder...")
        
        # Wait a bit more for the ladder to fully load
        time.sleep(3)
        
        # Look for player profile links after Old World is selected
        player_patterns = [
            "a[href*='/app/Profile?id=']",
            "a[class*='blue'][href*='Profile']",
            "a[href*='Profile']"
        ]
        
        old_world_players = []
        for pattern in player_patterns:
            try:
                player_links = driver.find_elements(By.CSS_SELECTOR, pattern)
                print(f"    Found {len(player_links)} potential players with pattern: {pattern}")
                
                for link in player_links:
                    href = link.get_attribute('href')
                    text = link.text.strip()
                    
                    if href and text and 'Profile?id=' in href:
                        # Extract player ID from URL
                        try:
                            player_id = href.split('id=')[1].split('&')[0]
                            if player_id and len(player_id) > 10:  # Valid ID format
                                old_world_players.append({
                                    'player_id': player_id,
                                    'player_name': text,
                                    'profile_url': href
                                })
                        except:
                            continue
                
                if old_world_players:
                    break  # Found players with this pattern
                    
            except Exception as e:
                continue
        
        # Remove duplicates
        unique_players = []
        seen_ids = set()
        for player in old_world_players:
            if player['player_id'] not in seen_ids:
                unique_players.append(player)
                seen_ids.add(player['player_id'])
        
        print(f"✅ Extracted {len(unique_players)} unique Old World players!")
        
        # Save the Old World players for future use
        if unique_players:
            with open('fresh_old_world_players.json', 'w', encoding='utf-8') as f:
                json.dump(unique_players, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved Old World players to fresh_old_world_players.json")
        
        return unique_players
        
    except Exception as e:
        print(f"❌ Error selecting Old World and extracting players: {e}")
        return []

def extract_match_data_from_old_world_player(driver, player_url, max_attempts=3):
    """Extract match data from a verified Old World player."""
    
    print(f"    🌐 Visiting Old World player: {player_url}")
    
    try:
        driver.get(player_url)
        time.sleep(5)
        
        # Handle cookie consent on this page
        handle_cookie_consent(driver)
        
        print("    🔍 Looking for Match History section...")
        
        # Look for Match History
        match_history_patterns = [
            "//*[contains(text(), 'Match History')]",
            "//h3[contains(text(), 'Match History')]",
            "//*[@class and contains(@class, 'arrowTitle')]"
        ]
        
        match_history_element = None
        match_count = 0
        
        for pattern in match_history_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                for element in elements:
                    element_text = element.text
                    if 'Match History' in element_text:
                        match_history_element = element
                        
                        # Extract match count from text like "Match History (5)"
                        count_match = re.search(r'\((\d+)\)', element_text)
                        if count_match:
                            match_count = int(count_match.group(1))
                            print(f"    ✅ Found Match History with {match_count} matches")
                        else:
                            print(f"    ✅ Found Match History: {element_text}")
                        break
                        
                if match_history_element:
                    break
            except:
                continue
        
        if not match_history_element:
            print("    ❌ No Match History found")
            return {'matches': 0, 'army_lists': []}
        
        if match_count == 0:
            print("    ⚠️ Player has 0 matches - skipping")
            return {'matches': 0, 'army_lists': []}
        
        # Try to expand Match History for players with matches
        print(f"    🔍 Expanding Match History ({match_count} matches)...")
        for attempt in range(max_attempts):
            try:
                handle_cookie_consent(driver)
                
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", match_history_element)
                time.sleep(2)
                
                try:
                    match_history_element.click()
                except:
                    driver.execute_script("arguments[0].click();", match_history_element)
                
                time.sleep(4)
                print(f"    ✅ Expanded Match History (attempt {attempt + 1})")
                break
                
            except Exception as e:
                print(f"    ⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)
        
        # Look for army list links
        print("    🔍 Looking for army list links...")
        
        army_list_patterns = [
            "//a[contains(@href, '/app/list/')]",
            "//img[@src='/assets/icons/eye.png']",
            "//*[contains(@onclick, 'list')]"
        ]
        
        army_lists_found = []
        for pattern in army_list_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                if elements:
                    print(f"    ✅ Found {len(elements)} potential army lists with: {pattern}")
                    
                    for element in elements:
                        try:
                            href = element.get_attribute('href')
                            onclick = element.get_attribute('onclick')
                            
                            list_url = None
                            if href and '/app/list/' in href:
                                list_url = href
                            elif onclick and 'list' in onclick.lower():
                                url_match = re.search(r'/app/list/[^"\']+', onclick)
                                if url_match:
                                    list_url = f"https://www.newrecruit.eu{url_match.group()}"
                            
                            if list_url:
                                list_id = list_url.split('/')[-1]
                                army_lists_found.append({
                                    'url': list_url,
                                    'list_id': list_id,
                                    'found_via': pattern,
                                    'player_url': player_url
                                })
                        except:
                            continue
            except:
                continue
        
        # Remove duplicates
        unique_lists = []
        seen_ids = set()
        for army_list in army_lists_found:
            if army_list['list_id'] not in seen_ids:
                unique_lists.append(army_list)
                seen_ids.add(army_list['list_id'])
        
        print(f"    📊 Found {len(unique_lists)} unique army lists")
        
        return {
            'matches': match_count,
            'army_lists': unique_lists
        }
        
    except Exception as e:
        print(f"    ❌ Error extracting from player: {e}")
        return {'matches': 0, 'army_lists': []}

def main():
    """Main function that properly extracts Old World players and their army lists."""
    
    print("=" * 80)
    print("🏛️  CORRECTED OLD WORLD ARMY EXTRACTION")
    print("🔧  Extracts players AFTER selecting Old World (not from pre-saved data)")
    print("=" * 80)
    
    driver = setup_driver(headless=False)
    all_army_lists = []
    
    try:
        # Step 1: Select Old World AND extract players from that ladder
        old_world_players = select_old_world_and_extract_players(driver)
        
        if not old_world_players:
            print("❌ No Old World players found!")
            return
        
        print(f"\n📊 Found {len(old_world_players)} Old World players")
        
        # Step 2: Process players to find ones with matches
        max_players_to_check = 10  # Check more players to find ones with matches
        players_with_matches = 0
        
        print(f"\n🔍 Checking {max_players_to_check} players for match data...")
        
        for i, player in enumerate(old_world_players[:max_players_to_check]):
            print(f"\n👤 {i+1}/{max_players_to_check}: {player['player_name']}")
            
            match_data = extract_match_data_from_old_world_player(
                driver, player['profile_url']
            )
            
            if match_data['matches'] > 0:
                players_with_matches += 1
                print(f"    ✅ Player has {match_data['matches']} matches!")
                
                for army_list in match_data['army_lists']:
                    army_list.update({
                        'player_name': player['player_name'],
                        'player_id': player['player_id'],
                        'match_count': match_data['matches']
                    })
                    all_army_lists.append(army_list)
            else:
                print(f"    ⚠️ Player has {match_data['matches']} matches - skipping")
    
    finally:
        driver.quit()
        print("🔚 Browser closed")
    
    # Save results
    if all_army_lists:
        output_file = 'corrected_old_world_army_lists.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_army_lists, f, indent=2, ensure_ascii=False)
        
        unique_list_ids = set(army_list['list_id'] for army_list in all_army_lists)
        
        print(f"\n✅ SUCCESS! Results saved to {output_file}")
        print(f"📊 Statistics:")
        print(f"  - Players with matches found: {players_with_matches}")
        print(f"  - Total army lists found: {len(all_army_lists)}")
        print(f"  - Unique army lists: {len(unique_list_ids)}")
        
        if unique_list_ids:
            print(f"\n📋 Sample army list URLs:")
            for i, list_id in enumerate(list(unique_list_ids)[:5]):
                print(f"  {i+1}. https://www.newrecruit.eu/app/list/{list_id}")
    else:
        print(f"\n⚠️ No army lists found. Found {players_with_matches} players with matches out of {max_players_to_check} checked.")

if __name__ == "__main__":
    main() 