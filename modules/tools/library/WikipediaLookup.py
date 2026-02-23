import httpx

query = args.get('query')
mode = args.get('mode', 'summary')

if not query:
    result = "Error: No query provided."
else:
    try:
        base_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }
        
        resp = httpx.get(base_url, params=search_params, timeout=10.0)
        search_data = resp.json()
        
        search_results = search_data.get("query", {}).get("search", [])
        
        if not search_results:
            result = f"No Wikipedia articles found for '{query}'."
        else:
            title = search_results[0]["title"]
            
            content_params = {
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": 1,
                "format": "json",
                "redirects": 1
            }
            
            if mode == 'summary':
                content_params["exintro"] = 1
            
            resp = httpx.get(base_url, params=content_params, timeout=10.0)
            data = resp.json()
            
            pages = data["query"]["pages"]
            page_id = next(iter(pages))
            
            if page_id == "-1":
                 result = f"Error: Page '{title}' not found details."
            else:
                extract = pages[page_id].get("extract", "")
                if not extract:
                    result = f"No content found for '{title}'."
                else:
                    result = f"Title: {title}\n\n{extract}"

    except Exception as e:
        result = f"Error fetching Wikipedia data: {str(e)}"
