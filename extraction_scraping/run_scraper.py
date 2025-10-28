#!/usr/bin/env python3
"""
Run the New Recruit scraper for Warhammer The Old World army compositions.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import time
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.newrecruit.eu"
LADDER_URL = f"{BASE_URL}/ladder"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

REQUEST_DELAY = 1
TOW_GAME_SYSTEM = "Warhammer The Old World"

def extract_id_from_url(url, param_name='id'):
    """Extract ID parameter from URL."""
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        return query_params.get(param_name, [None])[0]
    except:
        return None

def get_players_with_selenium(headless=True, timeout=15):
    """Use Selenium to interact with the dynamic ladder page."""
    
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait, Select
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        print("🚀 Starting Selenium browser automation...")
        
        # Set up Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        
        # Create driver
        print("📥 Installing/setting up Chrome driver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            print(f"🌐 Opening New Recruit ladder page: {LADDER_URL}")
            driver.get(LADDER_URL)
            
            # Wait for page to load
            wait = WebDriverWait(driver, timeout)
            print("⏳ Waiting for page to load...")
            
            # Give extra time for dynamic content
            time.sleep(3)
            
            # Look for the game system dropdown
            print("🔍 Looking for game system dropdown...")
            try:
                # Try multiple selectors to find the dropdown
                dropdown_selectors = [
                    "select",  # Generic select element
                    "[data-v-5065dd38]",  # From user's HTML snippet
                    ".form-select",
                    ".dropdown-select",
                    "select[name*='game']",
                    "select[name*='system']"
                ]
                
                dropdown = None
                for selector in dropdown_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            dropdown = elements[0]
                            print(f"✅ Found dropdown with selector: {selector}")
                            break
                    except:
                        continue
                
                if dropdown:
                    # Try to select "Warhammer The Old World"
                    select = Select(dropdown)
                    
                    # Print all available options
                    options = select.options
                    print(f"📋 Found {len(options)} dropdown options:")
                    found_tow = False
                    for i, option in enumerate(options):
                        option_text = option.get_attribute('textContent') or option.text
                        option_value = option.get_attribute('value')
                        print(f"  {i}: '{option_text}' (value: {option_value})")
                        
                        # Try to select Old World option
                        if 'old world' in option_text.lower() or 'warhammer' in option_text.lower():
                            print(f"🎯 Selecting: {option_text}")
                            try:
                                select.select_by_visible_text(option_text)
                                found_tow = True
                                time.sleep(3)  # Wait for content to load
                                break
                            except:
                                try:
                                    select.select_by_value(option_value)
                                    found_tow = True
                                    time.sleep(3)
                                    break
                                except:
                                    continue
                    
                    if not found_tow:
                        print("⚠️ Could not find Warhammer The Old World option, proceeding with current selection")
                else:
                    print("⚠️ No dropdown found, proceeding to look for players")
                
                # Now look for player links
                print("🔍 Looking for player profile links...")
                
                # Try multiple patterns for player links
                player_link_selectors = [
                    "a[href*='Profile']",
                    "a[href*='/app/Profile']",
                    "a.blue[href*='id=']",
                    ".player-link",
                    ".ladder-player"
                ]
                
                players = []
                for selector in player_link_selectors:
                    try:
                        player_links = driver.find_elements(By.CSS_SELECTOR, selector)
                        if player_links:
                            print(f"✅ Found {len(player_links)} links with selector: {selector}")
                            
                            for link in player_links:
                                try:
                                    href = link.get_attribute('href')
                                    text = link.text.strip()
                                    if href and text and 'id=' in href:
                                        player_id = extract_id_from_url(href)
                                        if player_id:
                                            players.append({
                                                'player_id': player_id,
                                                'player_name': text,
                                                'profile_url': href
                                            })
                                except:
                                    continue
                            break  # If we found players with this selector, don't try others
                    except:
                        continue
                
                print(f"✅ Found {len(players)} unique players!")
                
                # Remove duplicates
                unique_players = []
                seen_ids = set()
                for player in players:
                    if player['player_id'] not in seen_ids:
                        unique_players.append(player)
                        seen_ids.add(player['player_id'])
                
                return unique_players
                
            except Exception as e:
                print(f"❌ Error finding dropdown or players: {e}")
                
                # Fallback: Save page source for debugging
                try:
                    with open('selenium_page_source.html', 'w', encoding='utf-8') as f:
                        f.write(driver.page_source)
                    print("💾 Saved page source to selenium_page_source.html for debugging")
                except:
                    pass
                
                return []
        
        finally:
            driver.quit()
            print("🔚 Browser closed")
            
    except ImportError as e:
        print(f"❌ Selenium not available: {e}")
        print("Install with: pip install selenium webdriver-manager")
        return []
    except Exception as e:
        print(f"❌ Selenium error: {e}")
        return []

def main():
    """Main function to run the scraper."""
    print("=" * 60)
    print("🏛️  WARHAMMER THE OLD WORLD ARMY LIST SCRAPER")
    print("=" * 60)
    print()
    
    # Run the Selenium scraper
    players = get_players_with_selenium(headless=False, timeout=20)
    
    if players:
        print(f"\n✅ SUCCESS! Found {len(players)} players")
        print("\n📋 Player list:")
        for i, player in enumerate(players[:10]):  # Show first 10
            print(f"  {i+1:2d}. {player['player_name']:<20} (ID: {player['player_id']})")
        
        if len(players) > 10:
            print(f"     ... and {len(players) - 10} more players")
        
        # Save initial results
        with open('players_found.json', 'w', encoding='utf-8') as f:
            json.dump(players, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved player data to players_found.json")
        
        return players
    else:
        print("\n❌ No players found. Check the debug output above.")
        return []

if __name__ == "__main__":
    main() 