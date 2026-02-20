import httpx
import re

url = args.get('url')

if not url:
    result = "Error: No URL provided."
else:
    try:
        # Add a user agent to avoid being blocked by some sites
        headers = {"User-Agent": "NeuroCore-AI/1.0"}
        resp = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        
        # Simple HTML tag stripping
        text = resp.text
        # Remove scripts and styles
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long to prevent context overflow
        max_len = 2000
        if len(text) > max_len:
            text = text[:max_len] + "... (content truncated)"
            
        result = text
    except Exception as e:
        result = f"Error fetching URL: {str(e)}"