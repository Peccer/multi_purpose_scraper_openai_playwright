import json
import re
from bs4 import BeautifulSoup
import requests
from playwright.async_api import async_playwright  # <-- Use the async API
import os
from openai import AsyncOpenAI
import subprocess
import tempfile
# from loguru import logger
from typing import Dict, Any, List, Optional

async def scrape_webpage(url: str, client: AsyncOpenAI) -> Optional[str]:
    """
    Scrape a webpage and return the HTML content using async Playwright.
    Falls back to requests if Playwright fails.
    """
    print(f"[WEB SCRAPER] Starting to scrape URL (async) with Playwright: {url}")

    custom_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/108.0.0.0 Safari/537.36"
    )

    # Additional headers that can help
    # (some sites check for these, but it depends on the site)
    custom_headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.9"
        )
        # Add more if necessary for your target site
    }

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()

            # Create a new browser context with a custom user-agent
            context = await browser.new_context(
                user_agent=custom_user_agent,
                # Optionally set default extra HTTP headers at the context level
                extra_http_headers=custom_headers
            )
            
            page = await context.new_page()
            try:
                # await page.goto(url, wait_until="load", timeout=30000)
                await page.goto(url, wait_until="networkidle", timeout=60000)
                stored_selector = None
                cookie_selector = await find_and_accept_cookies(page, client, stored_selector)
                if cookie_selector:
                    stored_selector = cookie_selector
                content = await page.content()
                await browser.close()
                return content
            except Exception as e:
                print(f"[WEB SCRAPER ERROR] Error during page navigation: {str(e)}")
                await browser.close()
                # Fallback to requests
                print("[WEB SCRAPER] Falling back to requests library (due to page navigation error)")
                try:
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/91.0.4472.124 Safari/537.36"
                        )
                    }
                    response = requests.get(url, headers=headers, timeout=30)
                    return response.text
                except Exception as req_err:
                    print(f"[WEB SCRAPER ERROR] Error with requests fallback: {str(req_err)}")
                    return None
    except Exception as e:
        print(f"[WEB SCRAPER ERROR] Error initializing async Playwright: {str(e)}")
        # Fallback to requests if Playwright completely fails
        print("[WEB SCRAPER] Falling back to requests library (Playwright init error)")
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=30)
            return response.text
        except Exception as req_err:
            print(f"[WEB SCRAPER ERROR] Error with requests fallback: {str(req_err)}")
            return None

async def find_and_accept_cookies(page, client: AsyncOpenAI, stored_selector=None) -> Optional[str]:
    """
    Cookie detection: tries a known selector first; if that fails,
    tries to guess which button might be a cookie consent accept.
    """
    print("Attempting to detect and accept cookies on the page")
    if stored_selector:
        try:
            print(f"Trying previously stored cookie selector: {stored_selector}")
            await page.wait_for_selector(stored_selector, timeout=3000)
            btn = await page.query_selector(stored_selector)
            if btn:
                await btn.click()
                await page.wait_for_timeout(2000)
                print("Successfully accepted cookies using stored selector")
                return stored_selector
        except:
            print("Stored cookie selector failed, falling back to general detection")
            pass

    print("Starting general cookie button detection")
    buttons = await page.query_selector_all("button")
    print(f"Found {len(buttons)} buttons on page")
    
    candidates_info = []
    
    cookie_keywords = ["accept", "allow", "agree", "consent", "accepteer", "accepteren", "toestaan", "akkoord"]
    
    for i, b in enumerate(buttons):
        try:
            txt = await page.evaluate("(element) => element.innerText || element.textContent || ''", b)
            txt = txt.strip().lower()
            
            outer_html = await page.evaluate("(element) => element.outerHTML.substring(0, 200)", b)
            
            box = await b.bounding_box()
            if box is not None:
                if any(keyword in txt for keyword in cookie_keywords):
                    print(f"Found likely cookie button: {txt}")
                    await b.click()
                    await page.wait_for_timeout(1500)
                    print(f"Successfully clicked cookie button: {txt}")
                    return f"button:nth-of-type({i+1})"
                
                candidates_info.append({
                    "index": i,
                    "text": txt,
                    "outer_html": outer_html
                })
        except Exception as e:
            print(f"Error processing button {i}: {str(e)}")
            continue
    
    print(f"Found {len(candidates_info)} visible buttons")
    
    if not candidates_info:
        print("No viable cookie buttons found")
        return None

    joined = "\n".join(
        f"Index {c['index']}: text='{c['text']}' outer_html='{c['outer_html']}'"
        for c in candidates_info
    )
    
    input_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes HTML buttons to identify cookie consent buttons."
        },
        {
            "role": "user",
            "content": f"""
We have a list of buttons found on the page. One might be a cookie/consent accept button.
Which index is the cookie accept button, if any?
Return only the integer index or 'none'.

Buttons:
{joined}
"""
        }
    ]

    try:
        print("Asking LLM to identify cookie button")
        response = await client.responses.create(
            model="gpt-4o-mini",
            input=input_messages
        )
        
        llm_response = response.output_text.strip().lower()
        print(f"LLM response: {llm_response}")
        
        if re.match(r'^\d+$', llm_response):
            idx_val = int(llm_response)
            if 0 <= idx_val < len(candidates_info):
                print(f"LLM identified button at index {idx_val} as the cookie button")
                button_index = candidates_info[idx_val]["index"]
                all_buttons = await page.query_selector_all("button")
                btn = all_buttons[button_index]
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(2000)
                    print(f"Successfully clicked cookie button at index {button_index}")
                    return f"button:nth-of-type({button_index+1})"
    except Exception as e:
        print(f"Error during LLM cookie detection: {str(e)}")
        pass

    print("Cookie detection completed but no cookie button was found or clicked")
    return None

async def extract_data_from_webpage(url: str, fields: List[str], client: AsyncOpenAI) -> Dict[str, Any]:
    """
    Extract specific data fields from a webpage (async).
    """
    print(f"[WEB SCRAPER] Starting data extraction for URL: {url}")
    print(f"[WEB SCRAPER] Fields to extract: {fields}")

    result = {}
    try:
        print(f"[WEB SCRAPER] Attempting direct extraction with async Playwright for fields: {fields}")

        # Get the HTML content (async)
        html_content = await scrape_webpage(url, client)

        if not html_content:
            print("[WEB SCRAPER ERROR] Failed to retrieve HTML content")
            return {"error": f"Failed to scrape content from {url}"}

        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract each requested field
        for field in fields:
            field_key = field.lower().replace(" ", "_")
            field_value = "Not found"

            # Example: product price extraction
            if field_key in ("product_price", "price"):
                # Check for common price selectors
                price_selectors = [
                    ".promo-price", ".price", "[data-test='price']", ".product-price",
                    ".current-price", "[itemprop='price']", ".price-container", ".offer-price",
                    ".product__price", "#priceblock_ourprice", ".price-characteristic"
                ]
                for selector in price_selectors:
                    price_elem = soup.select_one(selector)
                    if price_elem:
                        price_text = price_elem.get_text().strip()
                        # Clean up the price text
                        price_text = re.sub(r'[^0-9.,]', '', price_text)
                        if price_text:
                            field_value = price_text
                            break

            # Store the extracted value
            result[field_key] = field_value

        return result

    except Exception as e:
        error_message = f"Error extracting data from {url}: {str(e)}"
        print(f"[WEB SCRAPER ERROR] {error_message}")
        return {"error": error_message}

async def handle_web_scraper_call(url: str, fields: List[str], client: AsyncOpenAI) -> Dict[str, Any]:
    """
    Handle the web_scraper function call (async).
    """
    print("\n[WEB SCRAPER] Handling web_scraper call")
    print(f"[WEB SCRAPER] URL: {url}")
    print(f"[WEB SCRAPER] Fields: {fields}")
    print(f"[WEB SCRAPER] Calling extract_data_from_webpage...\n")

    # Because we're now fully async, we await the extraction:
    extraction_result = await extract_data_from_webpage(url, fields, client)

    print(f"[WEB SCRAPER] Extraction result: {json.dumps(extraction_result, indent=2)}")
    return extraction_result

def setup_playwright():
    """
    Install Playwright browsers if not already installed.
    You can keep this synchronous, as it only runs a subprocess to install browsers.
    """
    try:
        print("Setting up Playwright browsers...")
        playwright_install_result = subprocess.run(
            ["playwright", "install", "chromium"], 
            capture_output=True,
            text=True
        )
        if playwright_install_result.returncode == 0:
            print("Playwright browsers installed successfully")
        else:
            print(f"Error installing Playwright browsers: {playwright_install_result.stderr}")
    except Exception as e:
        print(f"Error setting up Playwright: {str(e)}")

def get_tool_definition() -> Dict[str, Any]:
    """
    Get the tool definition for web_scraper.
    """
    print("[WEB SCRAPER] Creating tool definition for web_scraper")

    tool_definition = {
        "type": "function",
        "name": "web_scraper",
        "description": "Scrape a webpage and extract specific data fields",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to scrape"
                },
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of fields to extract from the webpage"
                }
            },
            "required": ["url", "fields"]
        }
    }

    print(f"[WEB SCRAPER] Tool definition created: {json.dumps(tool_definition, indent=2)}")
    return tool_definition
