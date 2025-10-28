#!/usr/bin/env python3
"""
Enhanced scraper that navigates Match History and Match Details to extract army lists.
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
from selenium.webdriver.common.action_chains import ActionChains

def setup_driver(headless=False):
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

def extract_match_data_from_player(driver, player_url, max_matches=10):
    """Extract match data by clicking through Match History and Match Details."""
    
    try:
        print(f"    🌐 Loading player profile: {player_url}")
        driver.get(player_url)
        time.sleep(5)  # Wait for page to load
        
        # Look for Match History section
        print("    🔍 Looking for Match History section...")
        
        match_history_element = None
        
        # Try to find Match History by text content
        try:
            all_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Match History')]")
            if all_elements:
                print(f"    ✅ Found {len(all_elements)} elements containing 'Match History'")
                match_history_element = all_elements[0]
        except Exception as e:
            print(f"    ⚠️ Error finding Match History: {e}")
        
        if match_history_element:
            print(f"    🎯 Found Match History element")
            
            # Try to click on Match History to expand it
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", match_history_element)
                time.sleep(1)
                match_history_element.click()
                print("    ✅ Clicked on Match History section")
                time.sleep(3)  # Wait for expansion
            except Exception as e:
                print(f"    ⚠️ Could not click Match History: {e}")
                # Try JavaScript click
                driver.execute_script("arguments[0].click();", match_history_element)
                time.sleep(3)
        
        # Look for Match Details links
        print("    🔍 Looking for Match Details links...")
        
        detail_elements = driver.find_elements(By.XPATH, "//a[contains(text(), 'Match Details')]")
        if not detail_elements:
            detail_elements = driver.find_elements(By.XPATH, "//a[contains(text(), 'Details')]")
        
        print(f"    ✅ Found {len(detail_elements)} Match Details elements")
        
        all_match_data = []
        
        # Process each match detail (limit to max_matches)
        for i, element in enumerate(detail_elements[:max_matches]):
            try:
                print(f"    🔗 Processing Match Details {i+1}/{len(detail_elements[:max_matches])}")
                
                # Scroll and click
                driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(1)
                element.click()
                time.sleep(3)  # Wait for match details to load
                
                # Extract army list data from match detail page
                match_data = extract_army_lists_from_match_detail(driver)
                if match_data:
                    match_data['match_index'] = i
                    all_match_data.append(match_data)
                    print(f"      ✅ Extracted data from match {i+1}")
                else:
                    print(f"      ❌ No data extracted from match {i+1}")
                
                # Go back to profile
                driver.back()
                time.sleep(2)
                
                # Re-click Match History if needed
                if i < len(detail_elements[:max_matches]) - 1:
                    try:
                        mh_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Match History')]")
                        if mh_elements:
                            mh_elements[0].click()
                            time.sleep(1)
                    except:
                        pass
                
            except Exception as e:
                print(f"    ❌ Error processing match {i+1}: {e}")
                try:
                    driver.back()
                    time.sleep(1)
                except:
                    pass
                continue
        
        print(f"    📊 Successfully extracted data from {len(all_match_data)} matches")
        return all_match_data
        
    except Exception as e:
        print(f"    ❌ Error extracting match data: {e}")
        return []

def extract_army_lists_from_match_detail(driver):
    """Extract army list URLs from a match detail page."""
    
    try:
        match_data = {
            'army_lists': [],
            'match_info': {},
            'result': None
        }
        
        # Look for army list links
        army_list_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/app/list/']")
        print(f"      📋 Found {len(army_list_elements)} army list links")
        
        for element in army_list_elements:
            href = element.get_attribute('href')
            text = element.text.strip()
            if href and '/app/list/' in href:
                list_id = href.split('/')[-1]
                match_data['army_lists'].append({
                    'url': href,
                    'text': text,
                    'list_id': list_id
                })
        
        # Try to extract match result
        page_text = driver.find_element(By.TAG_NAME, "body").text
        if 'win' in page_text.lower() or 'victory' in page_text.lower():
            match_data['result'] = 'win'
        elif 'loss' in page_text.lower() or 'defeat' in page_text.lower():
            match_data['result'] = 'loss'
        
        return match_data if match_data['army_lists'] else None
        
    except Exception as e:
        print(f"      ❌ Error extracting from match detail: {e}")
        return None

def enhanced_army_extraction(max_players=3, max_matches_per_player=5):
    """Main function using enhanced match history navigation."""
    
    print("=" * 80)
    print("🏛️  ENHANCED WARHAMMER THE OLD WORLD ARMY EXTRACTION")
    print("🔗  Clicking through Match History -> Match Details")
    print("=" * 80)
    
    # Load players
    try:
        with open('tow_players_found.json', 'r', encoding='utf-8') as f:
            players = json.load(f)
        print(f"📊 Loaded {len(players)} Warhammer The Old World players")
    except FileNotFoundError:
        print("❌ tow_players_found.json not found. Please run run_scraper_tow.py first.")
        return []
    
    all_extracted_data = []
    driver = setup_driver(headless=False)  # Show browser
    
    try:
        print(f"\n📊 Processing {min(max_players, len(players))} players...")
        
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
        output_file = 'enhanced_match_army_data.json'
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
        print(f"  Total matches: {len(all_extracted_data)}")
        print(f"  Total army lists found: {total_army_lists}")
        print(f"  Unique army lists: {len(unique_list_ids)}")
        
        return all_extracted_data
    else:
        print("\n❌ No match data extracted")
        return []

def main():
    """Main execution function."""
    print("🎯 ENHANCED MATCH HISTORY SCRAPER")
    print("This scraper will:")
    print("1. Navigate to player profiles")
    print("2. Click on Match History section")
    print("3. Click on individual Match Details")
    print("4. Extract army list URLs from match details")
    print("5. Save all the data")
    print()
    
    army_data = enhanced_army_extraction(max_players=3, max_matches_per_player=5)
    
    if army_data:
        print(f"\n🎉 SUCCESS! Extracted {len(army_data)} matches with army data!")
        print("\n📁 Files created:")
        print("  - enhanced_match_army_data.json (match data with army list URLs)")
    else:
        print("\n💡 Tips for debugging:")
        print("  - Check if Match History section is visible")
        print("  - Verify Match Details links are clickable")
        print("  - Ensure army list URLs are correctly extracted")

if __name__ == "__main__":
    main() 