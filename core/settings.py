import json
import os
import threading

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
    "debug_mode": True,
    "ui_wide_mode": False,
    "ui_show_footer": True,
    "request_timeout": 60.0,
    "max_node_loops": 1000
}

class SettingsManager:
    def __init__(self, file_path=SETTINGS_FILE):
        self.file_path = file_path
        self.lock = threading.Lock()
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
        with self.lock:
            self.settings.update(new_settings)
            with open(self.file_path, "w") as f:
                json.dump(self.settings, f, indent=4)

    def get(self, key, default=None):
        with self.lock:
            return self.settings.get(key, default)

# Global instance
settings = SettingsManager()
