import pandas as pd
import os
import json
import tempfile
import asyncio
from typing import Dict, Any, List, Optional
from web_scraper import get_tool_definition as web_scraper_tool_definition, handle_web_scraper_call
from utils import create_schema_from_user_requests, is_url
import subprocess
from dotenv import load_dotenv
from loguru import logger
import sys
import nest_asyncio
from openai import AsyncOpenAI

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st

# Load environment variables
load_dotenv(override=True)

# Apply nest_asyncio to allow nested event loops (needed for Streamlit with async)
nest_asyncio.apply()

# Set up logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Initialize Playwright - simpler approach without sync_playwright
def setup_playwright():
    """Install Playwright browsers using subprocess directly"""
    try:
        print("Setting up Playwright browsers...")
        # Check if playwright is installed, install browsers if needed
        playwright_install_result = subprocess.run(
            ["playwright", "install", "chromium"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if playwright_install_result.returncode == 0:
            print("Playwright browsers installed successfully")
        else:
            print(f"Error installing Playwright browsers: {playwright_install_result.stderr}")
    except Exception as e:
        print(f"Error setting up Playwright: {e}")
        print("The application will continue but web scraping may not work correctly.")

# Run Playwright setup
try:
    setup_playwright()
except Exception as e:
    print(f"Playwright setup error: {e}")
    print("Web scraping functionality may be limited.")

print("==== Application Starting ====")

# Set page title and configuration
st.set_page_config(page_title="Multi-Company Data Retriever", layout="wide")

# Initialize session state
if "combined_results" not in st.session_state:
    st.session_state.combined_results = []
    
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}

def get_tools_for_input(user_input: str) -> List[Dict[str, Any]]:
    """
    Return just one of the two tools (web_search or web_scraper) 
    depending on whether 'user_input' is recognized as a URL.
    """
    # If the user_input is recognized as a URL, we only want web_scraper.
    # Otherwise, we only want web_search.

    # 1) Build your separate tool definitions:
    web_search_tool = {
        "type": "web_search",
        "search_context_size": "low",
        "user_location": {
            "type": "approximate",
            "country": "NL"
        }
    }

    # The scraper definition is the same as before:
    web_scraper_tool = web_scraper_tool_definition()

    # 2) Decide which tool to include
    if is_url(user_input, require_http_scheme=False):
        # If the text is recognized as a URL or domain,
        # provide only the web_scraper tool.
        return [web_scraper_tool]
    else:
        # If not recognized as a URL, provide only the web_search tool.
        return [web_search_tool]

async def get_openai_response(company_name, requested_data):
    """
    Get company information from OpenAI based on requested data fields using async API
    """
    print(f"\n[DEBUG] Starting get_openai_response for company: {company_name}")
    print(f"[DEBUG] Requested data fields: {requested_data}")
    
    temp_file_path = None
    try:
        # Create temp file for user schema
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
            temp_file_path = temp_file.name
            
            # Create a simple JSON structure with the requested data fields
            temp_data = {"items": requested_data}
            json.dump(temp_data, temp_file)
            temp_file.flush()  # Ensure data is written to disk
            
        print(f"[DEBUG] Created temp file at: {temp_file_path}")
        
        # Import API key
        api_key = os.environ.get("OPENAI_API_KEY")
        # print(f"[DEBUG] API key available: {api_key is not None}")
        # print(f"[DEBUG] API key: {api_key}")
        
        try:
            # Create a new client instance with explicit parameters only
            print("[DEBUG] Attempting to create AsyncOpenAI client")
            client = AsyncOpenAI(api_key=api_key)
            print("[DEBUG] AsyncOpenAI client initialized successfully")
        except Exception as e:
            print(f"[DEBUG] Error during AsyncOpenAI initialization: {e}")
            print(f"[DEBUG] Error type: {type(e)}")
            raise e
        
        # Create schema from user requests
        user_schema = create_schema_from_user_requests(
            temp_file_path,
            "company_info"
        )
        print(f"[DEBUG] Created user schema: {json.dumps(user_schema, indent=2)}")
        
        # Get the web scraper tool definition
        # web_scraper_tool = web_scraper_tool_definition()
        # print(f"[DEBUG] Web scraper tool definition: {json.dumps(web_scraper_tool, indent=2)}")
        
        # # Add web_scraper to tools list
        # tools = [
        #     {
        #         "type": "web_search",
        #         "search_context_size": "low",
        #         "user_location": {
        #             "type": "approximate",
        #             "country": "NL"
        #         }
        #     },
        #     web_scraper_tool
        # ]
        # print(f"[DEBUG] Complete tools list: {json.dumps(tools, indent=2)}")

        tools = get_tools_for_input(company_name)

        
        # Make the API request asynchronously
        print("[DEBUG] Making initial API request to OpenAI...")
        
        # Format the input according to the Responses API format
        input_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that has the following tools available: "
                    "web_scraper: ONLY to be used when the provided item is a URL"
                    "web_search: ONLY to be used when the provided item is NOT a URL"
                )
            },
            {
                "role": "user",
                "content": (
                    f"The user has provided ITEM: '{company_name}'. \n\nGet the following information: "
                    f"{', '.join(requested_data)}.\n\n REMEMBER: If the ITEM is a URL, use 'web_scraper', "
                    f"otherwise use 'web_search'"
                )
            }
        ]
        
        response = await client.responses.create(
            model="gpt-4o-mini",
            input=input_messages,
            tools=tools,
            text=user_schema
        )
        print(f"[DEBUG] Initial API response received. Status: {response.status}")
        
        # 1) Gather all function calls from the first response
        if not isinstance(response.output, list):
            print("[DEBUG] response.output is not a list, check structure:", response.output)
            output_items = [response.output]

        else:
            output_items = response.output

        print(f"[DEBUG] output_items length: {len(output_items)}")
        print(f"[DEBUG] output_items: {output_items}")

        function_calls = [
            item for item in output_items 
            if getattr(item, "type", None) in ["function_call", "web_search"]
        ]
        print(f"[DEBUG] Found {len(function_calls)} function calls in response output.")

        
        # 2) Execute each function call, build the function_call_output messages
        function_call_outputs = []
        for i, fc_item in enumerate(function_calls):
            fc_id = fc_item.id
            fc_call_id = fc_item.call_id
            fc_name = fc_item.name
            fc_args_str = fc_item.arguments
            
            print(f"[DEBUG] Function call ID: {fc_id}")
            print(f"[DEBUG] Function call ID: {fc_call_id}")
            print(f"[DEBUG] Function name: {fc_name}")
            print(f"[DEBUG] Arguments: {fc_args_str}")
            
            if fc_name == "web_scraper" and fc_args_str:
                try:
                    args_dict = json.loads(fc_args_str)
                    url = args_dict.get("url")
                    fields = args_dict.get("fields", [])
                    print(f"[DEBUG] Parsed URL: {url}")
                    print(f"[DEBUG] Parsed fields: {fields}")
                    
                    # Execute web_scraper
                    scraper_result = await handle_web_scraper_call(url, fields, client)
                    print("[DEBUG] Web scraper result:", scraper_result)
                    
                    # Build the function_call_output item
                    function_call_outputs.append({
                        "type": "function_call_output",
                        "call_id": fc_call_id,
                        "output": json.dumps(scraper_result)
                    })
                
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] JSON parse error: {e}")
        
        # 3) If we have function_call_outputs, make a follow-up request
        if function_call_outputs:
            print(f"[DEBUG] Sending follow-up API call with tool results: {json.dumps(function_call_outputs, indent=2)}")
            
            # Rebuild the input for the follow-up
            followup_input = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that has the following tools available: "
                        "web_search and web_scraper. When a user provides a URL as input, "
                        "you will use web_scraper, otherwise you will use web_search, that's a hard requirement"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"The user has provided '{company_name}', get the following information: "
                        f"{', '.join(requested_data)}.\n\n REMEMBER: If it's a URL, use 'web_scraper', "
                        "otherwise use 'web_search'"
                    )
                }
            ]
            
            # 3a) Add the original function_call messages so the model knows about them
            followup_input.extend(function_calls)
            
            # 3b) Add our function_call_output messages referencing each call
            followup_input.extend(function_call_outputs)
            
            try:
                follow_up_response = await client.responses.create(
                    model="gpt-4o-mini",
                    input=followup_input,
                    tools=tools,
                    text=user_schema
                )
                print(f"[DEBUG] Follow-up API response received. Status: {follow_up_response.status}")
                
                # Overwrite the original response with the follow-up
                response = follow_up_response
            except Exception as api_error:
                print(f"[ERROR] Error in follow-up API call: {str(api_error)}")
                # If the follow-up call fails, produce a fallback
                formatted_result = {"company_name": company_name}
                for fc_out in function_call_outputs:
                    try:
                        data = json.loads(fc_out["output"])
                        for fld in requested_data:
                            field_key = fld.lower().replace(" ", "_")
                            formatted_result[field_key] = data.get(field_key, "Not found")
                    except Exception:
                        pass
                return formatted_result
        
        # 4) Now handle the final response
        if response.status == "completed":
            print("[DEBUG] Response status is 'completed'")
            try:
                final_data = {}
                # If there's a textual output we can parse, do it here
                if hasattr(response, "output_text") and response.output_text:
                    print(f"[DEBUG] Output text content: {response.output_text}")
                    try:
                        final_data = json.loads(response.output_text)
                    except json.JSONDecodeError:
                        # Possibly parse multiple JSON blocks if needed
                        pass
                
                # As a fallback, we might also look for the last assistant message, etc.
                if not final_data and isinstance(response.output, list):
                    # Attempt to parse the last assistant message
                    message_objects = [
                        item for item in response.output 
                        if getattr(item, 'type', '') == 'message' 
                           and getattr(item, 'role', '') == 'assistant'
                    ]
                    if message_objects:
                        last_message = message_objects[-1]
                        content_list = getattr(last_message, 'content', [])
                        if content_list and isinstance(content_list, list):
                            for piece in content_list:
                                text = getattr(piece, 'text', '')
                                if text:
                                    try:
                                        final_data = json.loads(text)
                                        break
                                    except json.JSONDecodeError:
                                        pass
                
                # Default to empty dict if we still have nothing
                if not final_data:
                    final_data = {}
                
                # Add company_name to ensure we always have it
                final_data["company_name"] = company_name
                return final_data
            
            except Exception as parse_error:
                print(f"[ERROR] Exception parsing response output: {parse_error}")
                return {
                    "error": f"Error parsing final response: {str(parse_error)}",
                    "company_name": company_name
                }
        else:
            print(f"[ERROR] Response not completed. Status: {response.status}")
            return {
                "error": f"API response not completed. Status: {response.status}",
                "company_name": company_name
            }
            
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Exception in get_openai_response: {error_msg}")
        return {"error": error_msg, "company_name": company_name}
    finally:
        # Clean up the temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"[DEBUG] Removed temp file: {temp_file_path}")
        print(f"[DEBUG] Completed get_openai_response for company: {company_name}")

def run_async(func, *args, **kwargs):
    """Run an async function from a synchronous context"""
    print(f"[DEBUG] Running async function: {func.__name__} with args: {args}")
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(func(*args, **kwargs))
    print(f"[DEBUG] Async function {func.__name__} completed")
    return result

# --------------------------------
# Main application
# --------------------------------

st.title("Multi-Purpose Data finder/scraper")

st.write("Enter a list of entities and the data fields you want to retrieve for each. These can be entitites like names of companies, people, etc. Or URLs.")

col1, col2= st.columns(2)

with col1:
    companies_input = st.text_area(
        "Enter entities (one per line)",
        placeholder="Microsoft\nApple\nGoogle\nAmazon",
        height=150,
        help="Enter each entity on a new line"
    )

    custom_fields = st.text_area(
        "Enter data fields (one per line)",
        placeholder="phone number\nemail address\nwebsite\naddress\nasset class",
        height=150,
        help="Enter each field on a new line"
    )

# Parse user inputs
requested_fields = []
if custom_fields:
    requested_fields = [line.strip() for line in custom_fields.split('\n') if line.strip()]

company_names = []
if companies_input:
    company_names = [line.strip() for line in companies_input.split('\n') if line.strip()]

# # Show the inputs
# if requested_fields and company_names:
#     col_info1, col_info2 = st.columns(2)
#     with col_info1:
#         st.write(f"**Entitites to analyze ({len(company_names)}):**")
#         for c in company_names:
#             st.write(f"- {c}")
#     with col_info2:
#         st.write(f"**Fields that will be requested ({len(requested_fields)}):**")
#         for f in requested_fields:
#             st.write(f"- {f}")

if st.button("Retrieve Data for All Entitites", type="primary", disabled=not (company_names and requested_fields)):
    st.session_state.combined_results = []
    st.session_state.processing_status = {
        "total": len(company_names),
        "completed": 0,
        "current_company": "",
        "errors": []
    }

    progress_bar = st.progress(0)
    status_container = st.empty()
    results_container = st.empty()

    for i, company in enumerate(company_names):
        st.session_state.processing_status["current_company"] = company
        st.session_state.processing_status["completed"] = i

        status_text = f"Processing {company} ({i+1}/{len(company_names)})..."
        status_container.info(status_text)

        # Call our async function
        response_data = run_async(get_openai_response, company, requested_fields)

        # Check for errors
        if "error" in response_data:
            error_msg = f"Error for {company}: {response_data['error']}"
            st.session_state.processing_status["errors"].append(error_msg)

        # Collect results
        st.session_state.combined_results.append(response_data)

        # Update progress
        progress_bar.progress((i+1) / len(company_names))

        # Show partial
        if st.session_state.combined_results:
            partial_df = pd.DataFrame(st.session_state.combined_results)
            if "company_name" in partial_df.columns:
                cols = partial_df.columns.tolist()
                cols.remove("company_name")
                partial_df = partial_df[["company_name"] + cols]
            results_container.dataframe(partial_df, use_container_width=True)

    if st.session_state.combined_results:
        st.success(f"✅ Completed! Retrieved data for {len(st.session_state.combined_results)} Entities.")
        if st.session_state.processing_status["errors"]:
            with st.expander(f"🚫 Errors ({len(st.session_state.processing_status['errors'])})"):
                for err in st.session_state.processing_status["errors"]:
                    st.error(err)

        # Final results
        results_df = pd.DataFrame(st.session_state.combined_results)
        if "company_name" in results_df.columns:
            cols = results_df.columns.tolist()
            cols.remove("company_name")
            results_df = results_df[["company_name"] + cols]

        st.subheader("Results")
        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="company_data.csv",
            mime="text/csv",
        )
    else:
        st.error("No results were returned. Please try again.")

# with st.expander("ℹ️ How to use this app"):
#     st.markdown("""
#     **Instructions:**
#     1. Enter the names of Entitites you want to analyze, one per line.
#     2. Enter the data fields you want to retrieve for each entity, one per line.
#     3. Click "Retrieve Data for All Entities" to start the process.
#     4. View the results in the table and download as CSV if needed.

#     **Notes:**
#     - Processing time depends on the number of entities and fields requested.
#     - The app uses OpenAI's GPT models to retrieve entity information.
#     - Information accuracy depends on data available to the AI model.
#     """)

st.markdown("---")
st.markdown("Built with Streamlit and OpenAI Responses API")
