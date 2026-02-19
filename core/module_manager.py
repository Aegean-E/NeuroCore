import os
import json
import importlib
from fastapi import FastAPI

MODULES_DIR = "modules"

class ModuleManager:
    def __init__(self, modules_dir=MODULES_DIR):
        self.modules_dir = modules_dir
        self.modules = self._discover_modules()

    def _discover_modules(self):
        """Finds all potential modules in the modules directory."""
        modules = {}
        if not os.path.exists(self.modules_dir):
            os.makedirs(self.modules_dir)
            return {}

        for name in os.listdir(self.modules_dir):
            module_path = os.path.join(self.modules_dir, name)
            if os.path.isdir(module_path):
                meta_path = os.path.join(module_path, "module.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        try:
                            meta = json.load(f)
                            meta['id'] = name
                            modules[name] = meta
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode module.json for {name}")
        return modules

    def load_enabled_modules(self, app: FastAPI):
        """Loads routers for all modules marked as enabled."""
        loaded_module_meta = []
        for name, meta in self.modules.items():
            if meta.get("enabled", False):
                try:
                    module = importlib.import_module(f"modules.{name}")
                    if hasattr(module, "router"):
                        app.include_router(module.router, prefix=f"/{name}", tags=[name])
                        loaded_module_meta.append(meta)
                        print(f"Loaded module: {name}")
                except Exception as e:
                    print(f"Failed to load module {name}: {e}")
        return loaded_module_meta

    def get_all_modules(self):
        """Returns a list of all discovered modules and their metadata."""
        return list(self.modules.values())

    def _toggle_module(self, module_id: str, enabled: bool):
        """Enables or disables a module by updating its metadata file."""
        if module_id not in self.modules:
            return None

        self.modules[module_id]['enabled'] = enabled
        meta_path = os.path.join(self.modules_dir, module_id, "module.json")
        with open(meta_path, "w") as f:
            json.dump(self.modules[module_id], f, indent=4)
        return self.modules[module_id]

    def enable_module(self, module_id: str):
        return self._toggle_module(module_id, True)

    def disable_module(self, module_id: str):
        return self._toggle_module(module_id, False)