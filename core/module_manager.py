import os
import json
import importlib
import sys
import threading
from fastapi import FastAPI
from starlette.routing import Mount
import logging

from core.flow_runner import FlowRunner
logger = logging.getLogger(__name__)

MODULES_DIR = "modules"

class ModuleManager:
    def __init__(self, app: FastAPI, modules_dir=MODULES_DIR):
        self.modules_dir = modules_dir
        self.app = app
        self.lock = threading.Lock()
        self.modules = self._discover_modules()
        # Runtime-only tracking for load errors (not persisted)
        self._load_errors = {}

    def _discover_modules(self):
        """Finds all potential modules in the modules directory."""
        modules = {}
        if not os.path.exists(self.modules_dir):
            os.makedirs(self.modules_dir)
            return {}

        for name in os.listdir(self.modules_dir):
            module_path = os.path.join(self.modules_dir, name)
            if os.path.isdir(module_path):
                # Skip if DISABLED file exists
                if os.path.exists(os.path.join(module_path, "DISABLED")):
                    continue
                meta_path = os.path.join(module_path, "module.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        try:
                            meta = json.load(f)
                            meta['id'] = name
                            # Don't persist load_error from file - it's runtime-only
                            meta.pop('load_error', None)
                            modules[name] = meta
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode module.json for {name}")
        return modules

    def load_enabled_modules(self):
        """Loads routers for all modules marked as enabled."""
        for name, meta in self.modules.items():
            if meta.get("enabled", False):
                self._load_module_router(name)

    def _load_module_router(self, module_id: str):
        """Imports a module and adds its router to the app."""
        # Use lock to prevent concurrent loading of the same module
        with self.lock:
            # Double-check after acquiring lock to prevent double registration
            if any(isinstance(r, Mount) and r.path == f"/{module_id}" for r in self.app.router.routes):
                return True

            # FIX: Properly reload all submodules by removing them from sys.modules first
            # This ensures hot-reload actually picks up changes in node.py, router.py, service.py, etc.
            modules_to_remove = [
                key for key in list(sys.modules.keys())
                if key.startswith(f"modules.{module_id}")
            ]
            for key in modules_to_remove:
                del sys.modules[key]

            try:
                module = importlib.import_module(f"modules.{module_id}")
                if hasattr(module, "router"):
                    self.app.include_router(module.router, prefix=f"/{module_id}", tags=[module_id])
                    logger.info(f"Hot-loaded module router: {module_id}")
                    # Clear runtime load error
                    self._load_errors[module_id] = None
                    self.modules[module_id]['load_error'] = None
                    return True
            except Exception as e:
                logger.error(f"Failed to load module router for {module_id}: {e}")
                # Store load error in runtime-only tracking, not in module metadata
                self._load_errors[module_id] = str(e)
                self.modules[module_id]['load_error'] = str(e)
            return False

    def _unload_module_router(self, module_id: str):
        """Finds and removes a module's router from the app."""
        initial_route_count = len(self.app.router.routes)
        
        # FIX: Handle both regular routes (have tags) and Mount objects (don't have tags)
        # Mount objects are identified by path pattern
        module_prefix = f"/{module_id}"
        
        def should_keep_route(route):
            # Check regular routes by tags
            if hasattr(route, "tags") and module_id in route.tags:
                return False
            # Check Mount objects by path - mounted sub-apps have path prefix
            if isinstance(route, Mount) and route.path.startswith(module_prefix):
                return False
            return True
        
        self.app.router.routes = [route for route in self.app.router.routes if should_keep_route(route)]
        
        if len(self.app.router.routes) < initial_route_count:
            logger.info(f"Hot-unloaded module router: {module_id}")
            return True
        return False

    def get_all_modules(self):
        """Returns a list of all discovered modules and their metadata, sorted by order key."""
        # Sort by the 'order' key in the metadata, defaulting to a high number if not present.
        with self.lock:
            return sorted(list(self.modules.values()), key=lambda m: m.get('order', 999))

    def _toggle_module(self, module_id: str, enabled: bool):
        """Enables or disables a module, updating the running app and the metadata file."""
        with self.lock:
            if module_id not in self.modules:
                return None

            if enabled:
                self._load_module_router(module_id)
            else:
                self._unload_module_router(module_id)

            self.modules[module_id]['enabled'] = enabled
            
            # FIX: When saving, don't include load_error (runtime-only)
            meta_to_save = {k: v for k, v in self.modules[module_id].items() if k != 'load_error'}
            
            meta_path = os.path.join(self.modules_dir, module_id, "module.json")
            with open(meta_path, "w") as f:
                json.dump(meta_to_save, f, indent=4)
                
            # Clear the FlowRunner cache to ensure no stale executor classes are used
            # (FlowRunner handles its own internal state, so calling this static method is safe)
            FlowRunner.clear_cache()

        return self.modules[module_id]

    def reorder_modules(self, module_ids: list):
        """Updates the 'order' of only modules whose order actually changed."""
        import time
        start_time = time.time()
        
        with self.lock:
            # Create a map for quick lookups
            id_to_meta = {m['id']: m for m in self.modules.values()}
            
            # FIX: Build diff first - only write to modules whose order actually changed
            modules_to_write = []
            
            for index, module_id in enumerate(module_ids):
                if module_id in id_to_meta:
                    current_order = id_to_meta[module_id].get('order')
                    # Only track if order is different
                    if current_order != index:
                        id_to_meta[module_id]['order'] = index
                        modules_to_write.append(module_id)
            
            # Write only the modules whose order changed
            for module_id in modules_to_write:
                meta_path = os.path.join(self.modules_dir, module_id, "module.json")
                if os.path.exists(meta_path):
                    # Don't include load_error in saved metadata
                    meta_to_save = {k: v for k, v in id_to_meta[module_id].items() if k != 'load_error'}
                    with open(meta_path, "w") as f:
                        json.dump(meta_to_save, f, indent=4)
        
        # Log warning if operation took too long (potential lock contention)
        elapsed = time.time() - start_time
        if elapsed > 1.0:  # Warn if taking more than 1 second
            logger.warning(f"reorder_modules took {elapsed:.2f}s - consider optimizing to reduce lock hold time")

    def enable_module(self, module_id: str):
        return self._toggle_module(module_id, True)

    def disable_module(self, module_id: str):
        return self._toggle_module(module_id, False)

    def update_module_config(self, module_id: str, new_config: dict):
        """Updates the configuration for a specific module."""
        with self.lock:
            if module_id not in self.modules:
                return None
            
            self.modules[module_id]['config'] = new_config
            
            # FIX: Don't persist load_error - it's runtime-only
            # Only save config, enabled, order - not load_error
            meta_to_save = {
                k: v for k, v in self.modules[module_id].items() 
                if k not in ('load_error',)
            }
            
            meta_path = os.path.join(self.modules_dir, module_id, "module.json")
            with open(meta_path, "w") as f:
                json.dump(meta_to_save, f, indent=4)
            return self.modules[module_id]
