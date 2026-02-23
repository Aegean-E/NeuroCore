import httpx

query = args.get('query')
max_results = args.get('max_results', 5)

if not query:
    result = "Error: No query provided."
else:
    try:
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(max_results, 20),
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        response = httpx.get(base_url, params=params, timeout=15.0)
        response.raise_for_status()
        
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.text)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            result = f"No papers found for query: '{query}'"
        else:
            papers = []
            for i, entry in enumerate(entries, 1):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')[:300]
                authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)][:3]
                published = entry.find('atom:published', ns).text[:10]
                pdf_link = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_link = link.get('href')
                        break
                
                papers.append(f"**{i}. {title}**")
                papers.append(f"   Authors: {', '.join(authors)}")
                papers.append(f"   Published: {published}")
                papers.append(f"   Summary: {summary}...")
                if pdf_link:
                    papers.append(f"   PDF: {pdf_link}")
                papers.append("")
            
            result = "ArXiv Search Results:\n\n" + "\n".join(papers)
            
    except httpx.HTTPError as e:
        result = f"HTTP error occurred: {str(e)}"
    except Exception as e:
        result = f"Error searching ArXiv: {str(e)}"
