#!/usr/bin/env python3
"""
Find Active Old World Players - Search for players with match history
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
    """Handle cookie consent dialog."""
    
    try:
        consent_selectors = [".fc-consent-root", ".fc-primary-button", ".fc-button"]
        
        for selector in consent_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    element = elements[0]
                    driver.execute_script("arguments[0].scrollIntoView();", element)
                    time.sleep(1)
                    
                    try:
                        element.click()
                    except:
                        driver.execute_script("arguments[0].click();", element)
                    
                    time.sleep(2)
                    return True
            except:
                continue
        
        return True
        
    except Exception as e:
        return True

def check_player_for_matches(driver, player):
    """Quickly check if a player has matches."""
    
    try:
        driver.get(player['profile_url'])
        time.sleep(3)
        
        handle_cookie_consent(driver)
        
        # Look for Match History text
        page_text = driver.page_source
        
        # Find Match History with number
        match_history_regex = r'Match History.*?\((\d+)\)'
        match = re.search(match_history_regex, page_text)
        
        if match:
            match_count = int(match.group(1))
            return match_count
        else:
            return 0
            
    except Exception as e:
        return 0

def find_active_old_world_players():
    """Search through Old World players to find ones with matches."""
    
    print("🔍 Loading Old World players...")
    
    # Load the fresh Old World players we just extracted
    try:
        with open('fresh_old_world_players.json', 'r', encoding='utf-8') as f:
            all_players = json.load(f)
        print(f"📊 Loaded {len(all_players)} Old World players")
    except FileNotFoundError:
        print("❌ Please run corrected_old_world_scraper.py first to get fresh player data")
        return
    
    driver = setup_driver(headless=False)
    players_with_matches = []
    
    try:
        print(f"🔍 Checking first 100 Old World players sequentially...")
        print(f"Looking for players with > 0 matches...")
        
        # Check first 100 players sequentially 
        check_count = 100
        
        for i in range(min(check_count, len(all_players))):
            player = all_players[i]
            
            print(f"\r👤 Player #{i+1}/{check_count}: {player['player_name']:<25}", end='', flush=True)
            
            match_count = check_player_for_matches(driver, player)
            
            if match_count > 0:
                player['match_count'] = match_count
                players_with_matches.append(player)
                print(f"\n✅ FOUND: Player #{i+1} - {player['player_name']} has {match_count} matches!")
            
            # Small delay to be respectful
            time.sleep(1)
    
    finally:
        driver.quit()
        print("\n🔚 Browser closed")
    
    print(f"\n📊 RESULTS:")
    print(f"  - Players checked: {check_count}")
    print(f"  - Players with matches: {len(players_with_matches)}")
    
    if players_with_matches:
        # Save players with matches
        output_file = 'active_old_world_players.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(players_with_matches, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved active players to {output_file}")
        print(f"\n🎯 Active Old World players found:")
        for player in players_with_matches:
            print(f"  - {player['player_name']}: {player['match_count']} matches")
    else:
        print(f"\n⚠️ No active players found in first {check_count} players")
        print(f"This suggests the Old World ladder is very new or inactive")
        
        # Alternative suggestions
        print(f"\n💡 ALTERNATIVE APPROACHES:")
        print(f"1. Try other game systems that might have army list data")
        print(f"2. Look for tournament/event sections")
        print(f"3. Search for known army list URLs directly")

if __name__ == "__main__":
    find_active_old_world_players() 