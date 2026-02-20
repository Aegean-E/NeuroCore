from datetime import datetime
from zoneinfo import ZoneInfo

time_string = args.get('time_string')
source_tz = args.get('source_tz')
target_tz = args.get('target_tz')

try:
    # Parse the time string (handles ISO format)
    dt = datetime.fromisoformat(time_string)
    
    # Load timezones
    src_zone = ZoneInfo(source_tz)
    tgt_zone = ZoneInfo(target_tz)
    
    # Set source timezone if not present in string
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=src_zone)
        
    # Convert to target timezone
    converted_dt = dt.astimezone(tgt_zone)
    
    result = f"{time_string} ({source_tz}) is {converted_dt.strftime('%Y-%m-%d %H:%M:%S %Z%z')} ({target_tz})."

except Exception as e:
    result = f"Error converting time: {str(e)}. Please ensure time is in ISO format (YYYY-MM-DD HH:MM:SS) and timezones are valid IANA names (e.g., 'America/New_York')."