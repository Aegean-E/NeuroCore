import json
import os
import threading
import tempfile

SETTINGS_FILE = "settings.json"

DEFAULT_SETTINGS = {
    "llm_api_url": "http://localhost:1234/v1",
    "llm_api_key": "",
    "embedding_api_url": "",
    "default_model": "local-model",
    "embedding_model": "",
    "active_ai_flows": [],
    "temperature": 0.7,
    "max_tokens": 2048,
    "debug_mode": False,
    "ui_wide_mode": False,
    "ui_show_footer": True,
    "request_timeout": 60.0,
    "max_node_loops": 100
}

class SettingsManager:
    def __init__(self, file_path=SETTINGS_FILE):
        self.file_path = file_path
        # IMPORTANT: lock must be initialized before load_settings() 
        # as load_settings() uses self.lock internally
        self.lock = threading.RLock()  # Use RLock for reentrant locking
        self.settings = self.load_settings()

    def load_settings(self):
        # Lock not strictly necessary in __init__ if singleton, but good for safety
        with self.lock:
            if not os.path.exists(self.file_path):
                # If the file doesn't exist, create it with default settings
                # and return them, avoiding the circular call to save_settings.
                settings_to_load = DEFAULT_SETTINGS.copy()
                with open(self.file_path, "w") as f:
                    json.dump(settings_to_load, f, indent=4)
                return settings_to_load
            
            try:
                with open(self.file_path, "r") as f:
                    loaded_settings = json.load(f)
                    merged_settings = DEFAULT_SETTINGS.copy()
                    merged_settings.update(loaded_settings)
                    return merged_settings
            except (json.JSONDecodeError, IOError):
                return DEFAULT_SETTINGS.copy()

    def save_settings(self, new_settings):
        # Validate critical fields before saving
        validated_settings = self._validate_settings(new_settings)
        
        with self.lock:
            self.settings.update(validated_settings)
            # Use atomic write-to-temp-then-rename pattern
            dir_path = os.path.dirname(self.file_path) or "."
            with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, suffix=".tmp") as tmp:
                json.dump(self.settings, tmp, indent=4)
                tmp_path = tmp.name
            os.replace(tmp_path, self.file_path)  # Atomic on POSIX, works on Windows too
    
    def _validate_settings(self, new_settings: dict) -> dict:
        """Validate settings before saving. Returns validated settings or raises ValueError."""
        validated = {}
        
        # temperature: float between 0 and 2
        if "temperature" in new_settings:
            temp = new_settings["temperature"]
            if not isinstance(temp, (int, float)):
                raise ValueError("temperature must be a number")
            if temp < 0 or temp > 2:
                raise ValueError("temperature must be between 0 and 2")
            validated["temperature"] = float(temp)
        
        # max_tokens: positive integer
        if "max_tokens" in new_settings:
            tokens = new_settings["max_tokens"]
            if not isinstance(tokens, (int, float)):
                raise ValueError("max_tokens must be an integer")
            if int(tokens) <= 0:
                raise ValueError("max_tokens must be a positive integer")
            validated["max_tokens"] = int(tokens)
        
        # request_timeout: positive float
        if "request_timeout" in new_settings:
            timeout = new_settings["request_timeout"]
            if not isinstance(timeout, (int, float)):
                raise ValueError("request_timeout must be a number")
            if float(timeout) <= 0:
                raise ValueError("request_timeout must be positive")
            validated["request_timeout"] = float(timeout)
        
        # max_node_loops: positive integer
        if "max_node_loops" in new_settings:
            loops = new_settings["max_node_loops"]
            if not isinstance(loops, (int, float)):
                raise ValueError("max_node_loops must be an integer")
            if int(loops) <= 0:
                raise ValueError("max_node_loops must be a positive integer")
            validated["max_node_loops"] = int(loops)
        
        # debug_mode, ui_wide_mode, ui_show_footer: booleans
        for bool_field in ["debug_mode", "ui_wide_mode", "ui_show_footer"]:
            if bool_field in new_settings:
                validated[bool_field] = bool(new_settings[bool_field])
        
        # String fields that should remain strings
        for str_field in ["llm_api_url", "llm_api_key", "embedding_api_url", "default_model", "embedding_model"]:
            if str_field in new_settings:
                validated[str_field] = str(new_settings[str_field])
        
        # active_ai_flows: list of strings
        if "active_ai_flows" in new_settings:
            if not isinstance(new_settings["active_ai_flows"], list):
                raise ValueError("active_ai_flows must be a list")
            for item in new_settings["active_ai_flows"]:
                if not isinstance(item, str):
                    raise ValueError(f"active_ai_flows items must be strings, got {type(item).__name__}")
            validated["active_ai_flows"] = new_settings["active_ai_flows"]
        
        return validated

    def get(self, key, default=None):
        with self.lock:
            return self.settings.get(key, default)

# Global instance
settings = SettingsManager()
