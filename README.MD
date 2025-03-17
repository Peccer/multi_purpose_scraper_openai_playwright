# Multi-Purpose Data Finder & Scraper

This repository contains a Streamlit-based application that uses OpenAI's Responses API to collect specific data fields for multiple entities. An entity can be anything from a **company name** or a **URL** or **whatever you want**. 

Depending on whether the entity is recognized as a URL, the application either:
- **Uses OpenAI’s built-in `web_search` tool** to look up general information (when the user input is **not** recognized as a URL or domain).  
- **Uses a `web_scraper` tool** (backed by Playwright or `requests`) to scrape direct information from the webpage (when the user input **is** recognized as a URL or domain).

> **In simpler terms**  
> - **If you type "Microsoft"** (no `.com` or `http://`), it will do a **web search**.  
> - **If you type "example.com"** or any **URL**, it will attempt to directly **scrape** that webpage.

---

## Table of Contents
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [How It Works (High-Level Explanation)](#how-it-works-high-level-explanation)
  - [1. Recognizing URLs vs. Plain Text](#1-recognizing-urls-vs-plain-text)
  - [2. Retrieving Data with OpenAI Tools](#2-retrieving-data-with-openai-tools)
- [Usage (Step-by-Step)](#usage-step-by-step)
- [Notes on Playwright](#notes-on-playwright)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features
1. **Multi-Entity Input**: Paste or type in multiple entities (one per line).
2. **Flexible Data Fields**: Specify the data fields you want (e.g., phone number, email address, asset class, etc.).
3. **Automatic Tool Selection**:  
   - **Non-URL input** triggers the **`web_search`** tool.  
   - **URL input** (e.g., `https://example.com`) triggers the **`web_scraper`** tool to scrape the site directly.
4. **Async OpenAI Requests**: Asynchronous calls to OpenAI for data retrieval and text processing.
5. **Progress Updates**: Real-time status and partial results shown during the batch retrieval process.
6. **Download Results**: Export the final results as a CSV file.

---

## Project Structure

```
├── multi_company_app.py             # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── utils.py                         # URL detection logic & schema creation
├── web_scraper.py                   # Logic for scraping URLs (Playwright & requests)
├── .env (not committed)             # Environment variables (e.g., OPENAI_API_KEY)
└── README.md                        # You're reading it now
```

### Brief Overview of Key Files

- **`multi_company_app.py`**  
  - Contains the Streamlit app’s UI and orchestrates how inputs are processed.  
  - Calls the appropriate OpenAI tool (web_search or web_scraper) after deciding if the input text is a URL.  

- **`utils.py`**  
  - Houses utility functions, such as `is_url`, which recognizes if a string is a URL, and `create_schema_from_user_requests` to build a JSON schema for the OpenAI response.  

- **`web_scraper.py`**  
  - Implements the `web_scraper` tool.  
  - Uses Playwright (asynchronously) to scrape a given URL. If Playwright fails, it falls back to the `requests` library.  
  - Extracts or returns basic HTML content, which is then processed by OpenAI to find specific fields.

---

## Installation & Setup

1. **Clone the repository** (or download the code):
   ```bash
   git clone https://github.com/your-username/openai-responses.git
   cd openai-responses
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This includes:
   - `openai`, `streamlit`, `pandas`, `python-dotenv`, `playwright`, `nest-asyncio`, and others.

3. **Set up Playwright browsers** (one-time setup):
   ```bash
   playwright install chromium
   ```
   > On Windows, the code in `multi_company_app.py` attempts to install automatically. But if you run into issues, use the command above.

4. **Create a `.env` file** in the project root, and add your OpenAI API Key:
   ```plaintext
   OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX
   ```
   (Ensure **never to commit** your API key to version control.)

---

## Running the Application

Launch the Streamlit app with:
```bash
streamlit run multi_company_app.py
```

Then open your browser to the URL shown in your terminal (typically `http://localhost:8501`).

---

## How It Works (High-Level Explanation)

### 1. Recognizing URLs vs. Plain Text
Inside `utils.py`, the function `is_url` checks each line you input. It looks for patterns like:
- `http://` or `https://`
- Domain-like strings with dots (e.g., `example.com`, `subdomain.example.co.uk`)

If `is_url` returns **True**, the app treats that line as a **web URL**. If it returns **False**, the app treats it as a general **search term** (like "Microsoft" or "John Doe").

### 2. Retrieving Data with OpenAI Tools
1. **If `is_url` is True** (URL recognized):
   - The app calls the **`web_scraper`** tool.  
   - Behind the scenes, it uses Playwright to load the page, possibly accept cookies, and then retrieve the HTML.  
   - After that, the HTML is **parsed** or relayed to OpenAI to look for your requested data fields.

2. **If `is_url` is False** (search term):
   - The app calls OpenAI’s built-in **`web_search`** tool to gather relevant data from across the web.  
   - OpenAI then returns the best guesses for each requested data field (e.g., phone number, address, etc.).

---

## Usage (Step-by-Step)

1. **Enter Entities**  
   In the left text area, type or paste a list of entities, each on its own line. Examples:  
   ```
   Apple
   Amazon
   https://example.com
   google.com
   ```
   Notice that `google.com` or `https://example.com` will be treated as URLs, while "Apple" or "Amazon" will be handled by web_search.

2. **Enter Data Fields**  
   In the right text area, type the data points you want to retrieve (one per line). For example:
   ```
   phone number
   email address
   website
   address
   asset class
   ```

3. **Press "Retrieve Data for All Entities"**  
   The app will:
   - For each entity, decide if it’s a URL.
   - If it’s a URL, use `web_scraper`.
   - If it’s NOT a URL, use `web_search`.

4. **Monitor Progress**  
   A progress bar and status text will show which entity is currently being processed.

5. **View Results**  
   A table will appear with your requested fields. You can then **download** the results in CSV format.

---

## Notes on Playwright

- **Playwright** is used under the hood to render modern web pages (including JavaScript) so that data can be accurately scraped.  
- If you experience issues with scraping or with cookie popups, it may be due to incomplete Playwright installation.  
- The code attempts to detect and "accept" cookie consent dialogs automatically. If you notice this does not work for certain sites, you may need to adapt that logic.

---

## Contributing
Pull requests and bug reports are welcome! If you find a bug or have a feature request, please open an issue.

---

## License
This project is open-source. Feel free to adapt or integrate it into your workflow under the terms specified in your chosen license.  