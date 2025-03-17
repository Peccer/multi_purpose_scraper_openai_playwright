import json
from typing import Dict, Any, List
from loguru import logger
import re
from urllib.parse import urlparse

def is_url(text: str, require_http_scheme: bool = False) -> bool:
    """
    Determine whether 'text' is a URL or domain-like string.
    
    Supports:
      - http://example.com
      - https://example.com
      - example.com (raw domain, no scheme)
      - subdomain.example.co.uk
      - optional path/query/fragment parts (e.g. example.com/path?x=1#anchor)
    
    If 'require_http_scheme' is True, then the text must explicitly be 
    an http:// or https:// URL to return True.
    Otherwise, we'll allow plain domains as well.
    
    Returns True if the text is considered a valid URL (including domain-only 
    if require_http_scheme=False). Otherwise returns False.
    """
    # Trim whitespace
    text = text.strip()

    # If no content, it's not a URL
    if not text:
        return False

    # If we require an explicit 'http' or 'https' in the string itself:
    if require_http_scheme:
        # Quick check: must start with 'http://' or 'https://'
        if not re.match(r'^(http|https)://', text.lower()):
            return False
    
    # We'll parse the string with urlparse, but if there's no scheme, 
    # prepend 'http://' so domain-only strings can still be parsed properly.
    # This is only done if the user doesn't require the scheme explicitly in the text.
    needs_scheme = not re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*://', text)
    text_for_parse = text if (require_http_scheme or not needs_scheme) else f"http://{text}"
    
    parsed = urlparse(text_for_parse)
    
    # Scheme check if require_http_scheme is True
    # e.g. user only wants to allow `http` or `https`, not ftp:// or mailto:
    if require_http_scheme and parsed.scheme not in ("http", "https"):
        return False
    
    # If there's no netloc, it's not a valid domain-based URL 
    # (urlparse might put the whole string in 'path' if no scheme was present)
    if not parsed.netloc:
        return False
    
    # Optional: we can also check if netloc is purely a domain or IP address,
    # i.e. it must contain a dot or be a valid IP. Let’s do a minimal domain check:
    # (some TLDs are just 2 letters, so it's quite tricky to handle them all with a single pattern,
    # but we at least want a `.` in there, unless you want to consider "localhost" or IP addresses).
    #
    # Below is a simple approach – requiring at least one dot.
    # For thoroughness, you might do a more advanced domain or IP check.
    if "." not in parsed.netloc:
        return False

    # If we made it here, we consider 'text' a valid URL or domain-like string
    return True



def create_schema_from_user_requests(json_file_path: str, schema_name: str = "dynamic_schema") -> Dict[str, Any]:
    """
    Transform user data requests from a JSON file into a proper schema for OpenAI API.
    
    Args:
        json_file_path: Path to the JSON file containing user requests
        schema_name: Name for the schema
        
    Returns:
        A dictionary containing the properly formatted schema for OpenAI Responses API
    """
    try:
        # Read the user requests from the JSON file
        with open(json_file_path, 'r') as file:
            file_content = file.read().strip()
            if not file_content:
                logger.error(f"JSON file is empty: {json_file_path}")
                raise ValueError("JSON file is empty")
            
            # Log the file content for debugging
            logger.debug(f"JSON file content: {file_content}")
            
            try:
                user_requests = json.loads(file_content)
            except json.JSONDecodeError as json_err:
                logger.error(f"Invalid JSON format: {json_err} - Content: {file_content}")
                raise ValueError(f"Invalid JSON format: {json_err}")
        
        # Check if the JSON has an "items" list
        if "items" not in user_requests or not isinstance(user_requests["items"], list):
            logger.error(f"JSON file must contain an 'items' list: {user_requests}")
            raise ValueError("JSON file must contain an 'items' list of requested fields")
        
        # Create properties dictionary for each requested field
        properties = {}
        required_fields = []
        
        logger.debug(f"Processing items: {user_requests['items']}")
        
        for item in user_requests["items"]:
            # Convert item to snake_case for property name
            field_name = item.lower().replace(" ", "_")
            required_fields.append(field_name)
            
            # Create property definition with appropriate description
            properties[field_name] = {
                "type": "string",
                "description": f"The {item.lower()} of the entity"
            }
        
        # Create the final schema
        custom_schema = {
            "type": "object",
            "properties": properties,
            "required": required_fields,
            "additionalProperties": False
        }
        
        # Return the complete schema structure for OpenAI API
        return {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": custom_schema
            }
        }
    except Exception as e:
        logger.error(f"Error creating schema from user requests: {e}")
        # Return a basic fallback schema
        return {
            "format": {
                "type": "json_schema",
                "name": "fallback_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error message"
                        }
                    },
                    "required": ["error"],
                    "additionalProperties": False
                }
            }
        }
