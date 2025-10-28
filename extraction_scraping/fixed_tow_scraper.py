#!/usr/bin/env python3
"""
Fixed scraper that maintains Warhammer The Old World context throughout the entire scraping process.
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

def handle_consent_and_select_old_world(driver):
    """Handle consent AND ensure Warhammer The Old World is selected."""
    
    print("🔍 Handling consent and selecting Warhammer The Old World...")
    
    try:
        # First, go to the ladder page
        driver.get("https://www.newrecruit.eu/ladder")
        time.sleep(5)
        
        # Handle any consent popups
        consent_selectors = [
            "[class*='consent']",
            "[class*='cookie']",
            "button:contains('Accept')",
            "button:contains('Agree')"
        ]
        
        for selector in consent_selectors:
            try:
                if "contains" in selector:
                    text = selector.split("'")[1]
                    elements = driver.find_elements(By.XPATH, f"//button[contains(text(), '{text}')]")
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements:
                    elements[0].click()
                    time.sleep(2)
                    print("✅ Handled consent popup")
                    break
            except:
                continue
        
        # Find and select "Warhammer The Old World" dropdown
        print("🎯 Looking for game system dropdown...")
        
        dropdown_selectors = ["select", "[data-v-5065dd38]", ".form-select"]
        
        old_world_selected = False
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
                            time.sleep(5)  # Wait for page to update
                            old_world_selected = True
                            print(f"✅ Selected Warhammer The Old World!")
                            break
                    
                    if old_world_selected:
                        break
                        
                if old_world_selected:
                    break
                    
            except Exception as e:
                continue
        
        if not old_world_selected:
            print("❌ Could not select Warhammer The Old World!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error in consent/selection: {e}")
        return False

def ensure_old_world_context_before_profile(driver, profile_url):
    """Ensure we're in Old World context before visiting a profile."""
    
    try:
        # Always go to ladder first to ensure Old World is selected
        print("    🎯 Ensuring Old World context...")
        driver.get("https://www.newrecruit.eu/ladder")
        time.sleep(3)
        
        # Re-select Old World if needed
        try:
            dropdown = driver.find_element(By.CSS_SELECTOR, "select")
            select = Select(dropdown)
            current_selection = select.first_selected_option.text
            
            if 'old world' not in current_selection.lower():
                print("    ⚠️ Old World not selected, re-selecting...")
                
                for option in select.options:
                    if 'old world' in option.text.lower():
                        select.select_by_visible_text(option.text)
                        time.sleep(3)
                        print("    ✅ Re-selected Old World")
                        break
            else:
                print(f"    ✅ Old World already selected: {current_selection}")
                
        except Exception as e:
            print(f"    ⚠️ Could not verify/re-select Old World: {e}")
        
        # Now navigate to the profile URL
        print(f"    🌐 Navigating to profile with Old World context...")
        driver.get(profile_url)
        time.sleep(5)
        
        # Verify we're seeing Old World content
        page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
        if any(term in page_text for term in ['old world', 'empire', 'bretonnian', 'dwarf', 'orc']):
            print("    ✅ Old World content detected on profile page")
            return True
        else:
            print("    ⚠️ Old World content not clearly detected")
            return True  # Continue anyway
            
    except Exception as e:
        print(f"    ❌ Error ensuring Old World context: {e}")
        return False

def extract_match_data_from_tow_player(driver, player_url, max_matches=5):
    """Extract match data specifically for Old World players."""
    
    try:
        # Ensure Old World context before visiting profile
        if not ensure_old_world_context_before_profile(driver, player_url):
            return []
        
        print("    🔍 Looking for Match History section...")
        
        # Look for Match History with various patterns
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
                    print(f"    ✅ Found Match History with pattern: {pattern}")
                    break
            except:
                continue
        
        if not match_history_element:
            print("    ❌ No Match History section found")
            return []
        
        # Click to expand Match History
        try:
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", match_history_element)
            time.sleep(2)
            match_history_element.click()
            time.sleep(3)
            print("    ✅ Expanded Match History")
        except Exception as e:
            print(f"    ⚠️ Could not expand Match History: {e}")
        
        # Save page source for debugging
        expanded_html = driver.page_source
        debug_filename = f"debug_expanded_match_history_{player_url.split('=')[-1]}.html"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            f.write(expanded_html)
        print(f"    💾 Saved expanded HTML to {debug_filename}")
        
        # Look for match entries after expansion
        print("    🔍 Looking for individual match entries...")
        
        # Try multiple approaches to find clickable match elements
        match_entry_patterns = [
            "//a[contains(text(), 'Match Details')]",
            "//a[contains(text(), 'Details')]",
            "//button[contains(text(), 'Details')]",
            "//tr[contains(@class, 'match')]",
            "//div[contains(@class, 'match')]",
            "//*[contains(@onclick, 'match')]",
            "//*[contains(@onclick, 'game')]",
            "//a[contains(@href, 'match')]",
            "//a[contains(@href, 'game')]",
            # Look for any clickable elements in the expanded area
            "//a[@href and not(contains(@href, 'Profile'))]",
            "//*[@onclick and not(contains(@onclick, 'undefined'))]"
        ]
        
        all_potential_elements = []
        for pattern in match_entry_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                if elements:
                    print(f"    ✅ Found {len(elements)} elements with pattern: {pattern}")
                    
                    # Show details of first few elements
                    for i, elem in enumerate(elements[:3]):
                        elem_text = elem.text.strip()[:50] if elem.text else "No text"
                        elem_href = elem.get_attribute('href') or elem.get_attribute('onclick') or "No link"
                        elem_tag = elem.tag_name
                        print(f"      [{i+1}] <{elem_tag}> '{elem_text}' -> {elem_href}")
                    
                    all_potential_elements.extend(elements)
            except Exception as e:
                continue
        
        print(f"    📊 Total potential match elements found: {len(all_potential_elements)}")
        
        # Try clicking on the most promising elements
        match_data = []
        processed_count = 0
        
        for i, element in enumerate(all_potential_elements[:max_matches]):
            if processed_count >= max_matches:
                break
                
            try:
                print(f"    🔗 Attempting to click element {i+1}: {element.text.strip()[:30]}")
                
                # Scroll to element
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
                time.sleep(1)
                
                # Get current URL to check if we navigate
                current_url = driver.current_url
                
                # Try clicking
                element.click()
                time.sleep(4)
                
                # Check if we navigated to a different page
                new_url = driver.current_url
                if new_url != current_url:
                    print(f"      ✅ Navigation successful! New URL: {new_url}")
                    
                    # Look for army lists on this page
                    army_lists = extract_army_lists_from_page(driver)
                    if army_lists:
                        match_data.append({
                            'match_url': new_url,
                            'army_lists': army_lists,
                            'element_clicked': element.text.strip()[:50]
                        })
                        processed_count += 1
                        print(f"      ✅ Found {len(army_lists)} army lists!")
                    else:
                        print(f"      ❌ No army lists found on this page")
                    
                    # Go back
                    driver.back()
                    time.sleep(3)
                    
                    # Re-expand match history if needed
                    try:
                        mh = driver.find_elements(By.XPATH, "//*[contains(text(), 'Match History')]")
                        if mh:
                            mh[0].click()
                            time.sleep(2)
                    except:
                        pass
                        
                else:
                    print(f"      ❌ No navigation occurred")
                    
            except Exception as e:
                print(f"      ❌ Error clicking element {i+1}: {e}")
                continue
        
        print(f"    📊 Successfully extracted {len(match_data)} matches with army data")
        return match_data
        
    except Exception as e:
        print(f"    ❌ Error extracting match data: {e}")
        return []

def extract_army_lists_from_page(driver):
    """Extract army list URLs from current page."""
    
    try:
        army_lists = []
        
        # Look for army list links
        list_selectors = [
            "a[href*='/app/list/']",
            "img[src*='eye.png']",  # Eye icons
            "*[onclick*='list']"
        ]
        
        for selector in list_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    href = element.get_attribute('href')
                    onclick = element.get_attribute('onclick')
                    
                    list_url = None
                    if href and '/app/list/' in href:
                        list_url = href
                    elif onclick and 'list' in onclick:
                        # Extract URL from onclick
                        url_match = re.search(r'/app/list/[^"\']+', onclick)
                        if url_match:
                            list_url = f"https://www.newrecruit.eu{url_match.group()}"
                    
                    if list_url:
                        list_id = list_url.split('/')[-1]
                        army_lists.append({
                            'url': list_url,
                            'list_id': list_id,
                            'found_via': selector
                        })
            except:
                continue
        
        # Remove duplicates
        seen_ids = set()
        unique_lists = []
        for army_list in army_lists:
            if army_list['list_id'] not in seen_ids:
                unique_lists.append(army_list)
                seen_ids.add(army_list['list_id'])
        
        return unique_lists
        
    except Exception as e:
        print(f"      ❌ Error extracting army lists: {e}")
        return []

def main():
    """Main function to extract army lists from Warhammer The Old World players."""
    
    print("=" * 80)
    print("🏛️  FIXED WARHAMMER THE OLD WORLD ARMY EXTRACTION")
    print("🔗  Maintains Old World context throughout the entire process")
    print("=" * 80)
    
    # Load The Old World players
    try:
        with open('tow_players_found.json', 'r', encoding='utf-8') as f:
            players = json.load(f)
        print(f"📊 Loaded {len(players)} The Old World players")
    except FileNotFoundError:
        print("❌ tow_players_found.json not found. Please run run_scraper_tow.py first.")
        return
    
    driver = setup_driver(headless=False)
    all_match_data = []
    
    try:
        # Initial setup - handle consent and select Old World
        if not handle_consent_and_select_old_world(driver):
            print("❌ Failed to setup Old World context")
            return
        
        # Process a few players for testing
        max_players = 3
        max_matches_per_player = 3
        
        print(f"\n📊 Processing {max_players} Old World players (max {max_matches_per_player} matches each)...")
        
        for i, player in enumerate(players[:max_players]):
            print(f"\n👤 {i+1}/{max_players}: {player['player_name']}")
            
            match_data = extract_match_data_from_tow_player(
                driver, player['profile_url'], max_matches_per_player
            )
            
            for match in match_data:
                match.update({
                    'player_name': player['player_name'],
                    'player_id': player['player_id'],
                    'player_profile': player['profile_url']
                })
                all_match_data.append(match)
    
    finally:
        driver.quit()
        print("🔚 Browser closed")
    
    # Save results
    if all_match_data:
        output_file = 'fixed_tow_army_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_match_data, f, indent=2, ensure_ascii=False)
        
        total_lists = sum(len(match['army_lists']) for match in all_match_data)
        unique_list_ids = set()
        for match in all_match_data:
            for army_list in match['army_lists']:
                unique_list_ids.add(army_list['list_id'])
        
        print(f"\n✅ SUCCESS! Results saved to {output_file}")
        print(f"📊 Statistics:")
        print(f"  - Matches processed: {len(all_match_data)}")
        print(f"  - Total army lists: {total_lists}")
        print(f"  - Unique army lists: {len(unique_list_ids)}")
        
        if unique_list_ids:
            print(f"\n📋 Sample army list URLs:")
            for i, list_id in enumerate(list(unique_list_ids)[:5]):
                print(f"  {i+1}. https://www.newrecruit.eu/app/list/{list_id}")
    else:
        print("\n❌ No army data extracted")
    
    # Save results
    if all_match_data:
        output_file = 'fixed_tow_army_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_match_data, f, indent=2, ensure_ascii=False)
        
        total_lists = sum(len(match['army_lists']) for match in all_match_data)
        unique_list_ids = set()
        for match in all_match_data:
            for army_list in match['army_lists']:
                unique_list_ids.add(army_list['list_id'])
        
        print(f"\n✅ SUCCESS! Results saved to {output_file}")
        print(f"📊 Statistics:")
        print(f"  - Matches processed: {len(all_match_data)}")
        print(f"  - Total army lists: {total_lists}")
        print(f"  - Unique army lists: {len(unique_list_ids)}")
        
        if unique_list_ids:
            print(f"\n📋 Sample army list URLs:")
            for i, list_id in enumerate(list(unique_list_ids)[:5]):
                print(f"  {i+1}. https://www.newrecruit.eu/app/list/{list_id}")
    else:
        print("\n❌ No army data extracted")

if __name__ == "__main__":
    main() 