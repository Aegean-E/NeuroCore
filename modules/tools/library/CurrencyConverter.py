import httpx

amount = args.get('amount')
from_curr = args.get('from_currency', '').upper()
to_curr = args.get('to_currency', '').upper()

if not amount or not from_curr or not to_curr:
    result = "Error: Missing arguments. Requires 'amount', 'from_currency', and 'to_currency'."
elif from_curr == to_curr:
    result = f"{amount} {from_curr} is {amount} {to_curr}"
else:
    try:
        # Using frankfurter.app (free, no key required)
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_curr}&to={to_curr}"
        
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        converted_value = data.get('rates', {}).get(to_curr)
        
        if converted_value is not None:
            result = f"{amount} {from_curr} = {converted_value} {to_curr} (Date: {data.get('date')})"
        else:
            result = f"Error: Could not find rate for {to_curr}."
            
    except Exception as e:
        result = f"Currency conversion failed: {str(e)}"