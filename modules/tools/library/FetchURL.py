import httpx
import re
from urllib.parse import urlparse
import socket
import ipaddress

url = args.get('url')

if not url:
    result = "Error: No URL provided."
else:
    try:
        # SSRF Protection: Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            result = "Error: Only http/https URLs are allowed."
        else:
            # SSRF Protection: Check for blocked IP ranges
            hostname = parsed.hostname
            if hostname:
                try:
                    ip = socket.gethostbyname(hostname)
                    blocked_patterns = [
                        r'^127\.',  # Loopback
                        r'^10\.',   # Private Class A
                        r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',  # Private Class B
                        r'^192\.168\.',  # Private Class C
                        r'^169\.254\.',  # Link-local (AWS metadata)
                        r'^0\.',    # Current network
                        r'^::1$',   # IPv6 loopback
                        r'^fc00:',  # IPv6 private
                        r'^fe80:',  # IPv6 link-local
                    ]
                    for pattern in blocked_patterns:
                        if re.match(pattern, ip):
                            result = f"Error: Request to private/internal IP addresses is not allowed."
                            raise ValueError("Blocked IP range")
                except socket.gaierror:
                    result = "Error: Could not resolve the provided URL."
                    raise
            
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
        # Check if we already set an error message
        if result and result.startswith("Error:"):
            pass  # Keep the existing error message
        else:
            result = f"Error fetching URL: {str(e)}"
