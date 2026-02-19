import json
import os

SETTINGS_FILE = "settings.json"

DEFAULT_SETTINGS = {
    "llm_api_url": "http://localhost:1234/v1",
    "default_model": "local-model",
    "temperature": 0.7,
    "max_tokens": 2048,
    "active_ai_flow": None
}

class SettingsManager:
    def __init__(self, file_path=SETTINGS_FILE):
        self.file_path = file_path
        self.settings = self.load_settings()

    def load_settings(self):
        if not os.path.exists(self.file_path):
            # If the file doesn't exist, create it with default settings
            # and return them, avoiding the circular call to save_settings.
            settings_to_load = DEFAULT_SETTINGS.copy()
            with open(self.file_path, "w") as f:
                json.dump(settings_to_load, f, indent=4)
            return settings_to_load
        
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return DEFAULT_SETTINGS.copy()

    def save_settings(self, new_settings):
        self.settings.update(new_settings)
        with open(self.file_path, "w") as f:
            json.dump(self.settings, f, indent=4)

    def get(self, key, default=None):
        return self.settings.get(key, default)

# Global instance
settings = SettingsManager()
