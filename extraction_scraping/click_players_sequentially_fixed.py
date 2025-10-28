#!/usr/bin/env python3
"""
Click Old World Players Sequentially - Direct clicking from ladder table
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

def select_old_world_ladder(driver, navigate_to_page=True):
    """Navigate to ladder page and select Old World."""
    
    if navigate_to_page:
        print("🎯 Going to ladder and selecting Warhammer The Old World...")
        # Go to ladder page
        driver.get("https://www.newrecruit.eu/ladder")
        time.sleep(5)
        # Handle cookie consent
        handle_cookie_consent(driver)
    else:
        print("    🎯 Re-selecting Warhammer The Old World from dropdown...")
    
    try:
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
                            if navigate_to_page:
                                print(f"✅ Found: '{option_text}' at index {i}")
                            select.select_by_index(i)
                            time.sleep(3)  # Wait for content to load
                            if navigate_to_page:
                                print(f"✅ Selected Warhammer The Old World!")
                            else:
                                print(f"    ✅ Re-selected Warhammer The Old World!")
                            dropdown_found = True
                            break
                    
                    if dropdown_found:
                        break
                        
            except Exception as e:
                continue
        
        return dropdown_found
        
    except Exception as e:
        if navigate_to_page:
            print(f"❌ Error selecting Old World: {e}")
        return False

def check_player_match_history(driver):
    """Check current player page for match history."""
    
    try:
        # Handle cookie consent on player page
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

def click_players_sequentially():
    """Click on Old World ladder players sequentially."""
    
    print("=" * 80)
    print("🏛️  CLICKING OLD WORLD PLAYERS SEQUENTIALLY")
    print("🖱️  Direct clicking from ladder table")
    print("🔧  Re-selects Old World after each player check")
    print("=" * 80)
    
    driver = setup_driver(headless=False)
    players_with_matches = []
    
    try:
        # Step 1: Get to Old World ladder
        if not select_old_world_ladder(driver):
            print("❌ Failed to select Old World ladder")
            return
        
        print("\n🔍 Finding player links in the ladder table...")
        
        # Debug: Check what's actually on the page
        time.sleep(3)
        player_link_selectors = [
            "a.blue[href*='Profile?id=']",
            "a[class*='blue'][href*='Profile']", 
            "a[href*='/app/Profile?id=']",
            "a[href*='Profile']"
        ]
        
        initial_player_links = []
        for selector in player_link_selectors:
            try:
                links = driver.find_elements(By.CSS_SELECTOR, selector)
                if links:
                    initial_player_links = links
                    print(f"✅ Found {len(initial_player_links)} player links with: {selector}")
                    break
            except:
                continue
        
        if not initial_player_links:
            print("❌ No player links found! Let's debug...")
            # Save page source for debugging
            with open('debug_ladder_page.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print("💾 Saved page source to debug_ladder_page.html")
            return
        
        # Step 2: Click on each player sequentially
        max_players_to_check = min(20, len(initial_player_links))  # Don't exceed available players
        print(f"\n🖱️ Clicking first {max_players_to_check} players sequentially...")
        print("    (Re-selecting Old World after each player)")
        
        for i in range(max_players_to_check):
            try:
                print(f"\n👤 Player #{i+1}/{max_players_to_check}")
                
                # Find player links on current page
                player_link_selectors = [
                    "a.blue[href*='Profile?id=']",
                    "a[class*='blue'][href*='Profile']",
                    "a[href*='/app/Profile?id=']"
                ]
                
                current_player_links = []
                for selector in player_link_selectors:
                    try:
                        links = driver.find_elements(By.CSS_SELECTOR, selector)
                        if links:
                            current_player_links = links
                            break
                    except:
                        continue
                
                if i >= len(current_player_links):
                    print(f"    ⚠️ Not enough players found (only {len(current_player_links)}), stopping")
                    break
                
                player_link = current_player_links[i]
                player_name = player_link.text.strip()
                
                print(f"    👤 {player_name}")
                print(f"    🖱️ Clicking on player link...")
                
                # Click on the player
                driver.execute_script("arguments[0].scrollIntoView();", player_link)
                time.sleep(1)
                
                try:
                    player_link.click()
                except:
                    driver.execute_script("arguments[0].click();", player_link)
                
                time.sleep(4)
                
                # Check match history on player page
                match_count = check_player_match_history(driver)
                print(f"    📊 Match History: {match_count} matches")
                
                if match_count > 0:
                    players_with_matches.append({
                        'player_name': player_name,
                        'match_count': match_count,
                        'position': i + 1
                    })
                    print(f"    🎉 FOUND ACTIVE PLAYER: {player_name} has {match_count} matches!")
                
                # Go back to ladder
                print(f"    ⬅️ Going back to ladder...")
                driver.back()
                time.sleep(3)
                
                # ALWAYS re-select Old World after going back (page resets to default)
                success = False
                for attempt in range(3):  # Try up to 3 times
                    if select_old_world_ladder(driver, navigate_to_page=False):
                        success = True
                        break
                    else:
                        print(f"    ⚠️ Re-selection attempt {attempt + 1} failed, waiting...")
                        time.sleep(2)
                
                if not success:
                    print("    ❌ Failed to re-select Old World after 3 attempts, trying full refresh...")
                    driver.get("https://www.newrecruit.eu/ladder")
                    time.sleep(3)
                    select_old_world_ladder(driver, navigate_to_page=True)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"    ❌ Error with player #{i+1}: {e}")
                # Try to get back to ladder and re-select Old World
                try:
                    print("    🔄 Trying to recover...")
                    driver.get("https://www.newrecruit.eu/ladder")
                    time.sleep(3)
                    select_old_world_ladder(driver, navigate_to_page=True)
                    time.sleep(2)
                except:
                    print("    ❌ Failed to recover, stopping")
                    break
                continue
    
    finally:
        driver.quit()
        print("\n🔚 Browser closed")
    
    # Results
    print(f"\n📊 FINAL RESULTS:")
    print(f"  - Players checked: {max_players_to_check}")
    print(f"  - Players with matches: {len(players_with_matches)}")
    
    if players_with_matches:
        output_file = 'active_old_world_players_from_clicking.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(players_with_matches, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved active players to {output_file}")
        print(f"\n🎯 Active Old World players found:")
        for player in players_with_matches:
            print(f"  #{player['position']}: {player['player_name']} - {player['match_count']} matches")
    else:
        print(f"\n⚠️ No players with matches found in the checked players")
        print(f"Old World ladder appears to be very new with minimal activity")

if __name__ == "__main__":
    click_players_sequentially() 