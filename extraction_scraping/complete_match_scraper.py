#!/usr/bin/env python3
"""
Complete scraper that handles consent/theme selection, then extracts army lists from Match History.
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
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

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

def handle_initial_consent_and_theme(driver):
    """Handle any consent popups and theme selection at the beginning."""
    
    print("🔍 Checking for consent popup and theme selection...")
    
    try:
        # Wait a bit for any popups to appear
        time.sleep(3)
        
        # Look for common consent/cookie popup patterns
        consent_selectors = [
            "button:contains('Accept')",
            "button:contains('Consent')", 
            "button:contains('Agree')",
            "button:contains('OK')",
            "button:contains('Continue')",
            ".consent-button",
            ".cookie-accept",
            ".accept-cookies",
            "[data-testid='consent']",
            "[class*='consent']",
            "[class*='cookie']"
        ]
        
        # Try to find and click consent buttons
        for selector in consent_selectors:
            try:
                if "contains" in selector:
                    # Use XPath for text-based selectors
                    text = selector.split("'")[1]
                    elements = driver.find_elements(By.XPATH, f"//button[contains(text(), '{text}')]")
                else:
                    # Use CSS selector
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements:
                    print(f"✅ Found consent button with selector: {selector}")
                    elements[0].click()
                    time.sleep(2)
                    print("✅ Clicked consent button")
                    break
            except Exception as e:
                continue
        
        # Look for theme selection
        print("🔍 Looking for theme selection...")
        theme_selectors = [
            "button:contains('Dark')",
            "button:contains('Light')", 
            "button:contains('Theme')",
            ".theme-selector",
            ".theme-button",
            "[class*='theme']"
        ]
        
        for selector in theme_selectors:
            try:
                if "contains" in selector:
                    text = selector.split("'")[1]
                    elements = driver.find_elements(By.XPATH, f"//button[contains(text(), '{text}')]")
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements:
                    print(f"✅ Found theme button with selector: {selector}")
                    elements[0].click()
                    time.sleep(2)
                    print("✅ Selected theme")
                    break
            except Exception as e:
                continue
        
        # Additional wait to ensure page is ready
        time.sleep(2)
        print("✅ Initial setup complete")
        
    except Exception as e:
        print(f"⚠️ Error during initial setup: {e}")

def ensure_old_world_game_system(driver):
    """Navigate to ladder and ensure 'Warhammer The Old World' is selected."""
    
    print("🎯 Navigating to ladder and selecting Warhammer The Old World...")
    
    try:
        # Go to the ladder page
        driver.get("https://www.newrecruit.eu/ladder")
        time.sleep(5)
        
        # Handle consent again if needed
        handle_initial_consent_and_theme(driver)
        
        # Look for game system dropdown
        print("🔍 Looking for game system dropdown...")
        dropdown_selectors = [
            "select",
            "[data-v-5065dd38]",  # From user's HTML
            ".select",
            ".dropdown",
            "select[name*='game']",
            "select[name*='system']"
        ]
        
        dropdown_found = False
        for selector in dropdown_selectors:
            try:
                dropdowns = driver.find_elements(By.CSS_SELECTOR, selector)
                for dropdown in dropdowns:
                    # Check if this dropdown has Old World option
                    options = dropdown.find_elements(By.TAG_NAME, "option")
                    for option in options:
                        option_text = option.text.strip()
                        if 'old world' in option_text.lower():
                            print(f"✅ Found Warhammer The Old World option: '{option_text}'")
                            
                            # Select this option
                            from selenium.webdriver.support.ui import Select
                            select = Select(dropdown)
                            select.select_by_visible_text(option_text)
                            print(f"✅ Selected: {option_text}")
                            
                            time.sleep(3)  # Wait for page to update
                            dropdown_found = True
                            break
                    
                    if dropdown_found:
                        break
                        
                if dropdown_found:
                    break
                    
            except Exception as e:
                continue
        
        if not dropdown_found:
            print("⚠️ Could not find or select Warhammer The Old World from dropdown")
            
            # Try alternative approach - look for any Old World reference on page
            old_world_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Old World')]")
            if old_world_elements:
                print(f"✅ Found {len(old_world_elements)} Old World references on page")
                for element in old_world_elements[:3]:
                    print(f"  - {element.text.strip()}")
            else:
                print("❌ No Old World references found on page")
        
        return dropdown_found
        
    except Exception as e:
        print(f"❌ Error selecting game system: {e}")
        return False

def extract_match_data_from_player(driver, player_url, max_matches=10):
    """Extract match data by clicking through Match History and Match Details."""
    
    try:
        print(f"    🌐 Loading player profile: {player_url}")
        driver.get(player_url)
        
        # Handle consent/theme if needed on this page
        handle_initial_consent_and_theme(driver)
        
        time.sleep(5)  # Wait for page to load
        
        # Look for Match History section
        print("    🔍 Looking for Match History section...")
        
        match_history_element = None
        
        # Try multiple approaches to find Match History
        match_history_patterns = [
            "//*[contains(text(), 'Match History')]",
            "//*[contains(text(), 'match history')]", 
            "//*[contains(text(), 'MATCH HISTORY')]",
            "//h3[contains(text(), 'Match History')]",
            "//*[@class and contains(@class, 'match') and contains(@class, 'history')]",
            "//*[@class and contains(@class, 'arrowTitle')]"
        ]
        
        for pattern in match_history_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                if elements:
                    print(f"    ✅ Found Match History with pattern: {pattern}")
                    match_history_element = elements[0]
                    break
            except:
                continue
        
        if match_history_element:
            print(f"    🎯 Found Match History element")
            
            # Try to click on Match History to expand it
            try:
                # Scroll to element
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", match_history_element)
                time.sleep(2)
                
                # Try different click methods
                try:
                    match_history_element.click()
                    print("    ✅ Clicked Match History (direct click)")
                except:
                    try:
                        driver.execute_script("arguments[0].click();", match_history_element)
                        print("    ✅ Clicked Match History (JavaScript click)")
                    except:
                        # Try clicking parent element
                        parent = match_history_element.find_element(By.XPATH, "./..")
                        parent.click()
                        print("    ✅ Clicked Match History (parent click)")
                
                time.sleep(3)  # Wait for expansion
                
            except Exception as e:
                print(f"    ⚠️ Could not click Match History: {e}")
        else:
            print("    ⚠️ Match History section not found")
        
        # Look for Match Details links with multiple approaches
        print("    🔍 Looking for Match Details links...")
        
        detail_elements = []
        
        # Try multiple selectors for Match Details
        detail_patterns = [
            "//a[contains(text(), 'Match Details')]",
            "//a[contains(text(), 'Details')]", 
            "//a[contains(text(), 'match details')]",
            "//button[contains(text(), 'Details')]",
            "//*[contains(text(), 'Details') and (@onclick or @href)]",
            "//a[contains(@class, 'match')]",
            "//a[contains(@class, 'detail')]",
            # Additional patterns for Old World specific elements
            "//a[contains(@href, 'match')]",
            "//a[contains(@href, 'battle')]",
            "//a[contains(@href, 'game')]",
            "//*[@onclick and contains(@onclick, 'match')]",
            "//*[@onclick and contains(@onclick, 'game')]"
        ]
        
        for pattern in detail_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                if elements:
                    detail_elements.extend(elements)
                    print(f"    ✅ Found {len(elements)} elements with pattern: {pattern}")
                    # Debug: Show first few element texts
                    for i, elem in enumerate(elements[:3]):
                        elem_text = elem.text.strip()[:50]
                        elem_href = elem.get_attribute('href') or elem.get_attribute('onclick') or 'no-link'
                        print(f"      [{i+1}] '{elem_text}' -> {elem_href}")
            except:
                continue
        
        # Remove duplicates
        unique_elements = []
        seen_elements = set()
        for element in detail_elements:
            element_id = id(element)
            if element_id not in seen_elements:
                unique_elements.append(element)
                seen_elements.add(element_id)
        
        detail_elements = unique_elements
        print(f"    📊 Found {len(detail_elements)} unique Match Details elements")
        
        # If still no elements, try looking for clickable match entries
        if not detail_elements:
            print("    🔍 No Match Details found, looking for clickable match entries...")
            
            # Look for match entries by date patterns or player names
            match_entry_patterns = [
                "//*[contains(text(), '2025-') or contains(text(), '2024-')]",  # Date patterns
                "//tr[contains(@class, 'match')]",
                "//div[contains(@class, 'match')]",
                "//*[@onclick and (contains(text(), 'vs') or contains(text(), 'paradowski'))]"
            ]
            
            for pattern in match_entry_patterns:
                try:
                    elements = driver.find_elements(By.XPATH, pattern)
                    if elements:
                        detail_elements.extend(elements[:5])  # Take first 5
                        print(f"    ✅ Found {len(elements)} match entries with pattern: {pattern}")
                        break
                except:
                    continue
        
        all_match_data = []
        
        # Process each match detail (limit to max_matches)
        for i, element in enumerate(detail_elements[:max_matches]):
            try:
                print(f"    🔗 Processing match element {i+1}/{len(detail_elements[:max_matches])}")
                
                # Scroll to element and click
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
                time.sleep(1)
                
                # Try clicking the element
                try:
                    element.click()
                    print(f"      ✅ Clicked match element {i+1}")
                except:
                    try:
                        driver.execute_script("arguments[0].click();", element)
                        print(f"      ✅ JavaScript clicked match element {i+1}")
                    except Exception as click_error:
                        print(f"      ❌ Could not click match element {i+1}: {click_error}")
                        continue
                
                time.sleep(4)  # Wait for match details to load
                
                # Extract army list data from current page
                match_data = extract_army_lists_from_current_page(driver)
                if match_data:
                    match_data['match_index'] = i
                    match_data['element_text'] = element.text.strip()[:100]  # First 100 chars
                    all_match_data.append(match_data)
                    print(f"      ✅ Extracted data from match {i+1}: {len(match_data.get('army_lists', []))} army lists")
                else:
                    print(f"      ❌ No army list data found in match {i+1}")
                
                # Go back to profile
                try:
                    driver.back()
                    time.sleep(3)
                    
                    # Re-expand Match History if needed
                    if i < len(detail_elements[:max_matches]) - 1:
                        try:
                            mh_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Match History')]")
                            if mh_elements:
                                mh_elements[0].click()
                                time.sleep(2)
                        except:
                            pass
                except Exception as back_error:
                    print(f"      ⚠️ Error going back: {back_error}")
                
            except Exception as e:
                print(f"    ❌ Error processing match {i+1}: {e}")
                continue
        
        print(f"    📊 Successfully extracted data from {len(all_match_data)} matches")
        return all_match_data
        
    except Exception as e:
        print(f"    ❌ Error extracting match data: {e}")
        return []

def extract_army_lists_from_current_page(driver):
    """Extract army list URLs from the current page (could be match detail or expanded match)."""
    
    try:
        match_data = {
            'army_lists': [],
            'match_info': {},
            'result': None,
            'page_url': driver.current_url
        }
        
        # Look for army list links with multiple approaches
        army_list_selectors = [
            "a[href*='/app/list/']",
            "a[href*='list']",
            "*[href*='/app/list/']",
            "img[src*='eye.png']",  # Eye icons from screenshots
            ".army-list-link",
            "*[onclick*='list']"
        ]
        
        army_list_elements = []
        for selector in army_list_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                army_list_elements.extend(elements)
            except:
                continue
        
        print(f"      📋 Found {len(army_list_elements)} potential army list elements")
        
        # Extract army list URLs
        for element in army_list_elements:
            try:
                href = element.get_attribute('href')
                onclick = element.get_attribute('onclick')
                
                # Extract URL from href or onclick
                list_url = None
                if href and '/app/list/' in href:
                    list_url = href
                elif onclick and 'list' in onclick:
                    # Try to extract URL from onclick
                    url_match = re.search(r'/app/list/[^"\']+', onclick)
                    if url_match:
                        list_url = f"https://www.newrecruit.eu{url_match.group()}"
                
                if list_url:
                    list_id = list_url.split('/')[-1]
                    text = element.text.strip() or element.get_attribute('alt') or element.get_attribute('title') or 'Army List'
                    
                    match_data['army_lists'].append({
                        'url': list_url,
                        'text': text,
                        'list_id': list_id
                    })
            except Exception as e:
                continue
        
        # Try to extract match result information
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
            if 'win' in page_text or 'victory' in page_text:
                match_data['result'] = 'win'
            elif 'loss' in page_text or 'defeat' in page_text:
                match_data['result'] = 'loss'
            elif 'draw' in page_text or 'tie' in page_text:
                match_data['result'] = 'draw'
        except:
            pass
        
        # Store page title and URL for debugging
        try:
            match_data['match_info']['page_title'] = driver.title
            match_data['match_info']['current_url'] = driver.current_url
        except:
            pass
        
        return match_data if match_data['army_lists'] else None
        
    except Exception as e:
        print(f"      ❌ Error extracting from current page: {e}")
        return None

def complete_army_extraction(max_players=3, max_matches_per_player=5):
    """Main function with complete consent handling and army extraction."""
    
    print("=" * 80)
    print("🏛️  COMPLETE WARHAMMER THE OLD WORLD ARMY EXTRACTION")
    print("🔗  Handles consent → Game System Selection → Match History → Army Lists")
    print("=" * 80)
    
    # Load players
    try:
        with open('tow_players_found.json', 'r', encoding='utf-8') as f:
            players = json.load(f)
        print(f"📊 Loaded {len(players)} Warhammer The Old World players")
        
        # Show some example player names to verify we have Old World players
        print("📋 Sample players:")
        for i, player in enumerate(players[:5]):
            print(f"  {i+1}. {player['player_name']} (ID: {player['player_id']})")
            
    except FileNotFoundError:
        print("❌ tow_players_found.json not found. Please run run_scraper_tow.py first.")
        return []
    
    all_extracted_data = []
    driver = setup_driver(headless=False)  # Show browser for debugging
    
    try:
        # Handle initial setup and ensure we're on the right game system
        print("\n🚀 Initial setup - handling consent and selecting game system...")
        driver.get("https://www.newrecruit.eu")
        handle_initial_consent_and_theme(driver)
        
        # Navigate to ladder and select Warhammer The Old World
        old_world_selected = ensure_old_world_game_system(driver)
        if not old_world_selected:
            print("⚠️ Warning: Could not confirm Warhammer The Old World is selected")
            print("⚠️ Proceeding anyway since we have Old World players from previous extraction")
        
        print(f"\n📊 Processing {min(max_players, len(players))} Warhammer The Old World players...")
        
        for i, player in enumerate(players[:max_players]):
            print(f"\n👤 {i+1}/{min(max_players, len(players))}: {player['player_name']}")
            
            # Extract match data from player profile
            match_data_list = extract_match_data_from_player(
                driver, player['profile_url'], max_matches_per_player
            )
            
            if match_data_list:
                for match_data in match_data_list:
                    # Add player context
                    match_data.update({
                        'player_name': player['player_name'],
                        'player_id': player['player_id'],
                        'player_profile': player['profile_url']
                    })
                    all_extracted_data.append(match_data)
                
                print(f"    ✅ Extracted {len(match_data_list)} matches from {player['player_name']}")
            else:
                print(f"    ⚠️ No match data found for {player['player_name']}")
    
    finally:
        driver.quit()
        print("🔚 Browser closed")
    
    print(f"\n✅ Extraction complete! Found {len(all_extracted_data)} matches with army data")
    
    if all_extracted_data:
        # Save the results
        output_file = 'complete_match_army_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_extracted_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved match and army data to {output_file}")
        
        # Quick statistics
        total_army_lists = sum(len(match.get('army_lists', [])) for match in all_extracted_data)
        unique_list_ids = set()
        for match in all_extracted_data:
            for army_list in match.get('army_lists', []):
                unique_list_ids.add(army_list.get('list_id', ''))
        
        print(f"\n📊 EXTRACTION STATISTICS:")
        print(f"  Total matches processed: {len(all_extracted_data)}")
        print(f"  Total army lists found: {total_army_lists}")
        print(f"  Unique army lists: {len(unique_list_ids)}")
        
        # Show some sample army list URLs
        if unique_list_ids:
            print(f"\n📋 Sample army list URLs:")
            for i, list_id in enumerate(list(unique_list_ids)[:5]):
                print(f"  {i+1}. https://www.newrecruit.eu/app/list/{list_id}")
        
        return all_extracted_data
    else:
        print("\n❌ No match data extracted")
        return []

def main():
    """Main execution function."""
    print("🎯 COMPLETE MATCH HISTORY SCRAPER WITH CONSENT HANDLING")
    print("This scraper will:")
    print("1. Handle initial consent and theme selection")
    print("2. Navigate to player profiles")
    print("3. Click on Match History section")
    print("4. Click on individual Match Details or match entries")
    print("5. Extract army list URLs")
    print("6. Save all extracted data")
    print()
    
    army_data = complete_army_extraction(max_players=3, max_matches_per_player=5)
    
    if army_data:
        print(f"\n🎉 SUCCESS! Extracted army data from {len(army_data)} matches!")
        print("\n📁 Files created:")
        print("  - complete_match_army_data.json (complete match and army list data)")
    else:
        print("\n💡 Debugging suggestions:")
        print("  - Check browser window to see what's happening")
        print("  - Verify consent popups are being handled")
        print("  - Ensure Match History sections are expanding")
        print("  - Check if army list URLs are being extracted correctly")

if __name__ == "__main__":
    main() 