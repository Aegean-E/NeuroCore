import os
import json
import importlib
from fastapi import FastAPI
from starlette.routing import Mount

from core.flow_runner import FlowRunner
MODULES_DIR = "modules"

class ModuleManager:
    def __init__(self, app: FastAPI, modules_dir=MODULES_DIR):
        self.modules_dir = modules_dir
        self.app = app
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

    def load_enabled_modules(self):
        """Loads routers for all modules marked as enabled."""
        for name, meta in self.modules.items():
            if meta.get("enabled", False):
                self._load_module_router(name)

    def _load_module_router(self, module_id: str):
        """Imports a module and adds its router to the app."""
        try:
            # Prevent loading a router that's already mounted
            if any(isinstance(r, Mount) and r.path == f"/{module_id}" for r in self.app.router.routes):
                return True

            module = importlib.import_module(f"modules.{module_id}")
            importlib.reload(module)  # Reload to pick up any changes
            if hasattr(module, "router"):
                self.app.include_router(module.router, prefix=f"/{module_id}", tags=[module_id])
                print(f"Hot-loaded module router: {module_id}")
                self.modules[module_id]['load_error'] = None
                return True
        except Exception as e:
            print(f"Failed to load module router for {module_id}: {e}")
            self.modules[module_id]['load_error'] = str(e)
        return False

    def _unload_module_router(self, module_id: str):
        """Finds and removes a module's router from the app."""
        initial_route_count = len(self.app.router.routes)
        
        # Filter out routes that have the module_id as a tag. This is how
        # include_router makes them identifiable.
        self.app.router.routes = [
            route for route in self.app.router.routes
            if not (hasattr(route, "tags") and module_id in route.tags)
        ]
        
        if len(self.app.router.routes) < initial_route_count:
            print(f"Hot-unloaded module router: {module_id}")
            return True
        return False

    def get_all_modules(self):
        """Returns a list of all discovered modules and their metadata, sorted by order key."""
        # Sort by the 'order' key in the metadata, defaulting to a high number if not present.
        return sorted(list(self.modules.values()), key=lambda m: m.get('order', 999))

    def _toggle_module(self, module_id: str, enabled: bool):
        """Enables or disables a module, updating the running app and the metadata file."""
        if module_id not in self.modules:
            return None

        if enabled:
            self._load_module_router(module_id)
        else:
            self._unload_module_router(module_id)

        self.modules[module_id]['enabled'] = enabled
        meta_path = os.path.join(self.modules_dir, module_id, "module.json")
        with open(meta_path, "w") as f:
            json.dump(self.modules[module_id], f, indent=4)
            
        # Clear the FlowRunner cache to ensure no stale executor classes are used
        FlowRunner.clear_cache()

        return self.modules[module_id]

    def enable_module(self, module_id: str):
        return self._toggle_module(module_id, True)

    def disable_module(self, module_id: str):
        return self._toggle_module(module_id, False)