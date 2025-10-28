#!/usr/bin/env python3
"""
FINAL WORKING SCRAPER - Handles cookie consent dialog that's blocking interactions
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
    """Handle the FC (Full Cookie) consent dialog that blocks interactions."""
    
    print("    🍪 Checking for cookie consent dialog...")
    
    try:
        # Look for the FC consent dialog
        consent_selectors = [
            ".fc-consent-root",
            ".fc-primary-button", 
            ".fc-button",
            "[class*='fc-button']",
            "[class*='consent']",
            "button:contains('Accept')",
            "button:contains('Agree')",
            "button:contains('Allow')"
        ]
        
        for selector in consent_selectors:
            try:
                if "contains" in selector:
                    text = selector.split("'")[1]
                    elements = driver.find_elements(By.XPATH, f"//button[contains(text(), '{text}')]")
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements:
                    print(f"    ✅ Found consent element: {selector}")
                    
                    # Try to click the first element
                    element = elements[0]
                    driver.execute_script("arguments[0].scrollIntoView();", element)
                    time.sleep(1)
                    
                    # Try different click methods
                    try:
                        element.click()
                    except:
                        try:
                            driver.execute_script("arguments[0].click();", element)
                        except:
                            continue
                    
                    time.sleep(3)
                    print("    ✅ Clicked consent button")
                    return True
                    
            except Exception as e:
                continue
        
        print("    ⚠️ No consent dialog found")
        return True  # Continue anyway
        
    except Exception as e:
        print(f"    ❌ Error handling consent: {e}")
        return True  # Continue anyway

def select_old_world_game_system(driver):
    """Select Warhammer The Old World from the dropdown."""
    
    print("🎯 Selecting Warhammer The Old World...")
    
    try:
        # Go to ladder page
        driver.get("https://www.newrecruit.eu/ladder")
        time.sleep(5)
        
        # Handle cookie consent first
        handle_cookie_consent(driver)
        
        # Find and select game system dropdown
        dropdown_selectors = ["select", "[data-v-5065dd38]", ".form-select"]
        
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
                            time.sleep(5)
                            print(f"✅ Selected Warhammer The Old World!")
                            return True
                            
            except Exception as e:
                continue
        
        print("❌ Could not select Old World")
        return False
        
    except Exception as e:
        print(f"❌ Error selecting Old World: {e}")
        return False

def extract_army_lists_from_old_world_player(driver, player_url, max_attempts=3):
    """Extract army lists from a Warhammer The Old World player profile."""
    
    print(f"    🌐 Visiting: {player_url}")
    
    try:
        # Ensure Old World context and navigate
        if not select_old_world_game_system(driver):
            return []
        
        driver.get(player_url)
        time.sleep(5)
        
        # Handle cookie consent on this page too
        handle_cookie_consent(driver)
        
        print("    🔍 Looking for Match History section...")
        
        # Look for Match History
        match_history_patterns = [
            "//*[contains(text(), 'Match History')]",
            "//h3[contains(text(), 'Match History')]",
            "//*[@class and contains(@class, 'arrowTitle')]"
        ]
        
        match_history_element = None
        for pattern in match_history_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                if elements:
                    match_history_element = elements[0]
                    print(f"    ✅ Found Match History: {pattern}")
                    break
            except:
                continue
        
        if not match_history_element:
            print("    ❌ No Match History found")
            return []
        
        # Try to expand Match History
        print("    🔍 Attempting to expand Match History...")
        for attempt in range(max_attempts):
            try:
                # Handle any remaining consent dialogs
                handle_cookie_consent(driver)
                
                # Scroll to and click match history
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", match_history_element)
                time.sleep(2)
                
                # Try clicking
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
                else:
                    print("    ❌ Could not expand Match History")
        
        # Save debug HTML
        expanded_html = driver.page_source
        debug_filename = f"final_debug_{player_url.split('=')[-1]}.html"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            f.write(expanded_html)
        print(f"    💾 Saved debug HTML: {debug_filename}")
        
        # Look for army list links after expansion
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
                            # Get href or extract from onclick
                            href = element.get_attribute('href')
                            onclick = element.get_attribute('onclick')
                            
                            list_url = None
                            if href and '/app/list/' in href:
                                list_url = href
                            elif onclick and 'list' in onclick.lower():
                                # Extract list URL from onclick
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
        return unique_lists
        
    except Exception as e:
        print(f"    ❌ Error extracting from player: {e}")
        return []

def main():
    """Main function to extract army lists from Warhammer The Old World players."""
    
    print("=" * 80)
    print("🏛️  FINAL WORKING OLD WORLD ARMY EXTRACTION")
    print("🍪  Handles cookie consent dialog that blocks interactions")
    print("=" * 80)
    
    # Load players
    try:
        with open('tow_players_found.json', 'r', encoding='utf-8') as f:
            players = json.load(f)
        print(f"📊 Loaded {len(players)} The Old World players")
    except FileNotFoundError:
        print("❌ tow_players_found.json not found. Please run run_scraper_tow.py first.")
        return
    
    driver = setup_driver(headless=False)
    all_army_lists = []
    
    try:
        # Test with a few players
        max_players = 3
        
        print(f"\n📊 Processing {max_players} Old World players...")
        
        for i, player in enumerate(players[:max_players]):
            print(f"\n👤 {i+1}/{max_players}: {player['player_name']}")
            
            army_lists = extract_army_lists_from_old_world_player(
                driver, player['profile_url']
            )
            
            for army_list in army_lists:
                army_list.update({
                    'player_name': player['player_name'],
                    'player_id': player['player_id']
                })
                all_army_lists.append(army_list)
    
    finally:
        driver.quit()
        print("🔚 Browser closed")
    
    # Save results
    if all_army_lists:
        output_file = 'final_tow_army_lists.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_army_lists, f, indent=2, ensure_ascii=False)
        
        unique_list_ids = set(army_list['list_id'] for army_list in all_army_lists)
        
        print(f"\n✅ SUCCESS! Results saved to {output_file}")
        print(f"📊 Statistics:")
        print(f"  - Total army lists found: {len(all_army_lists)}")
        print(f"  - Unique army lists: {len(unique_list_ids)}")
        
        if unique_list_ids:
            print(f"\n📋 Sample army list URLs:")
            for i, list_id in enumerate(list(unique_list_ids)[:5]):
                print(f"  {i+1}. https://www.newrecruit.eu/app/list/{list_id}")
    else:
        print("\n❌ No army lists found")

if __name__ == "__main__":
    main() 