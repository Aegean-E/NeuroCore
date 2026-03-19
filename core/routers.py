import json
import ast
import sys
import platform
import asyncio
import logging
import time
from datetime import datetime
from fastapi import APIRouter, Request, Form, Depends, HTTPException, Response, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

from core.settings import SettingsManager, settings
from core.dependencies import get_settings_manager, get_module_manager, get_llm_bridge, require_debug_mode
from core.module_manager import ModuleManager
from core.flow_manager import flow_manager
from core.llm import LLMBridge
from core.debug import debug_logger

logger = logging.getLogger(__name__)


router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

def format_reasoning_content(content):
    """Extracts the actual content from a raw LLM response dictionary string."""
    if isinstance(content, str) and content.strip().startswith("{") and "'choices':" in content:
        try:
            # Attempt to parse stringified dict
            data = ast.literal_eval(content)
            if isinstance(data, dict):
                # Check for OpenAI format
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    if "content" in message:
                        return message["content"]
        except (ValueError, SyntaxError, KeyError, IndexError):
            # ValueError/SyntaxError: ast.literal_eval failed to parse
            # KeyError/IndexError: Missing expected keys in parsed structure
            pass
    return content


templates.env.filters["format_reasoning"] = format_reasoning_content

def get_hardware_id():
    import uuid
    import hashlib
    import subprocess
    import os

    id_file = os.path.join("data", "hardware_id.txt")

    # Return persisted ID so hardware changes never alter it
    if os.path.exists(id_file):
        try:
            with open(id_file, "r") as f:
                persisted = f.read().strip()
            if persisted and len(persisted) >= 16:
                return persisted
        except Exception:
            pass

    def get_cmd(cmd):
        try: return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
        except Exception: return ""

    try:
        cpu_id = get_cmd("wmic cpu get ProcessorId")
        cpu_name = get_cmd("wmic cpu get Name")
        cpu_mhz = get_cmd("wmic cpu get MaxClockSpeed")
        board_sn = get_cmd("wmic baseboard get SerialNumber")
        board_mfr = get_cmd("wmic baseboard get Manufacturer")
        board_product = get_cmd("wmic baseboard get Product")
        ram = get_cmd("wmic ComputerSystem get TotalPhysicalMemory")
        disk_sn = get_cmd("wmic diskdrive where index=0 get SerialNumber")
        os_name = f"{platform.system()} {platform.release()} {platform.version()}"

        combined = f"{cpu_id}|{cpu_name}|{cpu_mhz}|{board_sn}|{board_mfr}|{board_product}|{ram}|{disk_sn}|{os_name}"
        useful = combined.replace("|", "").replace(" ", "").strip()
        if len(useful) < 10:
            raise ValueError("Specs too short")

        hardware_id = hashlib.sha256(combined.encode()).hexdigest()[:24].upper()
    except Exception:
        try:
            node = uuid.getnode()
            hardware_id = hashlib.sha256(str(node).encode()).hexdigest()[:24].upper()
        except Exception:
            hardware_id = "UNKNOWN"

    # Persist so future calls never recompute
    try:
        os.makedirs("data", exist_ok=True)
        with open(id_file, "w") as f:
            f.write(hardware_id)
    except Exception:
        pass

    return hardware_id

# Centralized definition of config keys that should be hidden from the generic JSON editor
HIDDEN_CONFIG_KEYS = {
    'memory': ['save_default_confidence', 'save_confidence_threshold', 'recall_limit', 'recall_min_score', 'consolidation_threshold', 'auto_consolidation_hours', 'arbiter_model', 'arbiter_prompt', 'similarity_threshold', 'belief_ttl_days', 'recall_access_weight'],
    'llm_module': ['temperature', 'max_tokens'],
    'chat': ['auto_rename_turns', 'auto_compact_tokens', 'compact_keep_last'],
    'agent_loop': ['max_iterations', 'max_tokens', 'temperature', 'max_llm_retries', 'retry_delay', 'timeout', 'tool_error_strategy', 'include_plan_in_context', 'include_memory_context', 'include_knowledge_context', 'include_reasoning_context', 'compact_threshold', 'compact_keep_last'],
    'telegram': ['bot_token', 'chat_id'],
    'messaging_bridge': [
        'telegram_bot_token', 'telegram_chat_id',
        'discord_bot_token', 'discord_channel_id',
        'signal_api_url', 'signal_phone_number',
        'whatsapp_api_url', 'whatsapp_api_key', 'whatsapp_instance', 'whatsapp_phone_number',
    ],
    'email_bridge': [
        'imap_host', 'imap_port', 'imap_use_ssl', 'imap_username', 'imap_password',
        'imap_folder', 'imap_filter_sender', 'poll_interval', 'mark_as_read',
        'smtp_host', 'smtp_port', 'smtp_use_tls', 'smtp_username', 'smtp_password',
        'smtp_from_address', 'reply_to_address',
    ],
}

# --- System & Navigation ---

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "index.html", {
        "modules": module_manager.get_all_modules(),
        "active_module": "dashboard",
        "settings": settings_man.settings,
        "hide_module_list": True,
        "hide_status_panel": True,
        "full_width_content": True
    })

# --- Dashboard ---

@router.get("/dashboard/gui", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})

@router.get("/dashboard/stats", response_class=HTMLResponse)
async def get_dashboard_stats(request: Request):
    import json
    from pathlib import Path
    
    base_dir = Path(__file__).resolve().parent.parent
    
    # Memory count
    memory_count = 0
    memory_path = base_dir / "data" / "memory.sqlite3"
    if memory_path.exists():
        try:
            import sqlite3
            with sqlite3.connect(str(memory_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0")
                memory_count = cursor.fetchone()[0]
        except (sqlite3.Error, OSError) as e:
            # sqlite3.Error: Database connection or query failed
            # OSError: File system issues (permissions, disk full, etc.)
            logger.warning(f"[Dashboard Stats] Could not read memory count: {e}")

    
    # Chat sessions count
    sessions_count = 0
    sessions_path = base_dir / "chat_sessions.json"
    if sessions_path.exists():
        try:
            with open(sessions_path, 'r') as f:
                sessions = json.load(f)
            sessions_count = len(sessions)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # JSONDecodeError: Corrupted JSON file
            # OSError: File read permissions or I/O issues
            # KeyError: Unexpected structure in sessions data
            logger.warning(f"[Dashboard Stats] Could not read sessions count: {e}")

    
    # Knowledge base docs count
    docs_count = 0
    kb_path = base_dir / "data" / "knowledge_base.sqlite3"
    if kb_path.exists():
        try:
            import sqlite3
            with sqlite3.connect(str(kb_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                docs_count = cursor.fetchone()[0]
        except (sqlite3.Error, OSError) as e:
            # sqlite3.Error: Database connection or query failed
            # OSError: File system issues
            logger.warning(f"[Dashboard Stats] Warning: Could not read knowledge base count: {e}")

    
    # Tools count
    tools_count = 0
    tools_path = base_dir / "modules" / "tools" / "library"
    logger.debug(f"Checking tools_path: {tools_path}, exists: {tools_path.exists()}")
    if tools_path.exists():
        try:
            files = list(tools_path.iterdir())
            py_files = [f for f in files if f.suffix == '.py' and not f.name.startswith('_') and f.name != '__init__.py']
            tools_count = len(py_files)
            logger.info(f"Found {tools_count} tools: {[f.name for f in py_files]}")
        except (OSError, PermissionError) as e:
            # OSError: Directory iteration failed
            # PermissionError: Insufficient permissions to read directory
            logger.warning(f"Error counting tools: {e}")
    else:
        logger.warning(f"Tools path does not exist: {tools_path}")


    
    html = f"""
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="group bg-slate-800/50 rounded-xl border border-slate-700/50 p-5 hover:border-blue-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/5">
            <div class="flex items-center justify-between mb-3">
                <div class="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                    <svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                </div>
                <span class="text-xs text-slate-500">Memory</span>
            </div>
            <div class="text-3xl font-bold text-white">{memory_count}</div>
            <div class="text-xs text-slate-500 mt-1">Stored Memories</div>
        </div>
        <div class="group bg-slate-800/50 rounded-xl border border-slate-700/50 p-5 hover:border-emerald-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-500/5">
            <div class="flex items-center justify-between mb-3">
                <div class="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                    <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
                </div>
                <span class="text-xs text-slate-500">Chat</span>
            </div>
            <div class="text-3xl font-bold text-white">{sessions_count}</div>
            <div class="text-xs text-slate-500 mt-1">Active Sessions</div>
        </div>
        <div class="group bg-slate-800/50 rounded-xl border border-slate-700/50 p-5 hover:border-purple-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/5">
            <div class="flex items-center justify-between mb-3">
                <div class="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                    <svg class="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path></svg>
                </div>
                <span class="text-xs text-slate-500">Knowledge</span>
            </div>
            <div class="text-3xl font-bold text-white">{docs_count}</div>
            <div class="text-xs text-slate-500 mt-1">Documents Indexed</div>
        </div>
        <div class="group bg-slate-800/50 rounded-xl border border-slate-700/50 p-5 hover:border-orange-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-orange-500/5">
            <div class="flex items-center justify-between mb-3">
                <div class="w-10 h-10 rounded-lg bg-orange-500/20 flex items-center justify-center">
                    <svg class="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                </div>
                <span class="text-xs text-slate-500">Tools</span>
            </div>
            <div class="text-3xl font-bold text-white">{tools_count}</div>
            <div class="text-xs text-slate-500 mt-1">Available Tools</div>
        </div>
    </div>
    """
    return html

@router.get("/dashboard/recent-sessions", response_class=HTMLResponse)
async def get_recent_sessions(request: Request):
    import json
    import os
    import html
    from datetime import datetime
    
    sessions_path = "chat_sessions.json"
    if not os.path.exists(sessions_path):
        return '<p class="text-slate-500 text-sm italic">No sessions yet</p>'
    
    try:
        with open(sessions_path, 'r') as f:
            sessions = json.load(f)
        
        sorted_sessions = sorted(sessions.values(), key=lambda x: x.get('updated_at', ''), reverse=True)
        recent = sorted_sessions[:5]
        
        if not recent:
            return '<p class="text-slate-500 text-sm italic">No sessions yet</p>'
        
        html_content = ""
        for s in recent:
            updated = s.get('updated_at', '')
            try:
                dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                time_ago = dt.strftime('%b %d, %H:%M')
            except (ValueError, TypeError, AttributeError):
                # ValueError: Invalid date format in updated string
                # TypeError: updated is not a string
                # AttributeError: replace/strftime called on wrong type
                time_ago = updated[:10] if len(updated) >= 10 else 'Unknown'
            
            # Escape user-supplied session name to prevent XSS
            session_name = html.escape(s.get('name', 'Untitled'))
            
            html_content += f"""
            <button onclick="htmx.ajax('GET', '/chat/gui?session_id={s['id']}', '#module-content')" class="w-full flex items-center justify-between p-3 rounded-lg hover:bg-slate-700/50 border border-transparent hover:border-slate-600/50 transition-all group text-left">
                <div class="flex items-center gap-3 min-w-0">
                    <div class="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0">
                        <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
                    </div>
                    <span class="text-sm text-slate-300 truncate">{session_name}</span>
                </div>
                <span class="text-xs text-slate-500 flex-shrink-0">{time_ago}</span>
            </button>
            """
        return html_content
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
        # JSONDecodeError: Corrupted sessions file
        # OSError: File system issues
        # KeyError/ValueError: Data structure issues
        logger.warning(f"[Recent Sessions] Warning: Could not load recent sessions: {e}")
        return f'<p class="text-slate-500 text-sm italic">Error loading sessions</p>'


@router.get("/navbar", response_class=HTMLResponse)
async def get_navbar(request: Request, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "navbar.html", {"modules": module_manager.get_all_modules(), "settings": settings_man.settings})

@router.get("/llm-status", response_class=HTMLResponse)
async def get_llm_status(request: Request, llm: LLMBridge = Depends(get_llm_bridge)):
    api_status_check = await llm.get_models()
    api_online = "error" not in api_status_check
    return f"""
    <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full {"bg-emerald-500 animate-pulse" if api_online else "bg-red-500"}"></div>
        <span class="text-xs {"text-slate-400" if api_online else "text-red-400"}">LLM Status: {"Online" if api_online else "Offline"}</span>
    </div>
    """

# --- Module Management ---

@router.get("/modules/list", response_class=HTMLResponse)
async def list_modules(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "module_list.html", {"modules": module_manager.get_all_modules()})

@router.get("/modules/{module_id}/details", response_class=HTMLResponse)
async def get_module_details(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    module = module_manager.modules.get(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    config_display = module.get('config', {}).copy()
    keys_to_hide = HIDDEN_CONFIG_KEYS.get(module_id, [])
    for key in keys_to_hide:
        config_display.pop(key, None)
        
    formatted_config = json.dumps(config_display, indent=4)
    
    # Check if there are any keys left to display
    has_visible_config = len(config_display) > 0
    
    return templates.TemplateResponse(request, "module_details.html", {
        "module": module, 
        "formatted_config": formatted_config,
        "has_visible_config": has_visible_config,
        "settings": settings_man.settings
    })

@router.get("/modules/{module_id}/default-prompt")
async def get_default_prompt(module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    """Returns the default prompt for reflection or planner modules as plain text."""
    from fastapi import Response
    
    module = module_manager.modules.get(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    config = module.get('config', {})
    
    if module_id == 'reflection':
        default_prompt = config.get('default_reflection_prompt', '')
    elif module_id == 'planner':
        default_prompt = config.get('default_planner_prompt', '')
    else:
        raise HTTPException(status_code=400, detail="Module does not support default prompts")
    
    # Return as plain text to preserve actual newlines (not escaped \n)
    return Response(content=default_prompt, media_type="text/plain")


@router.post("/modules/{module_id}/config")
async def save_module_config(request: Request, module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    try:
        form_data = await request.form()
        
        # Handle system_prompt module specially
        if module_id == "system_prompt":
            new_config = {}
            if "system_prompt" in form_data:
                new_config["system_prompt"] = form_data["system_prompt"]
            if "enabled_tools" in form_data:
                try:
                    new_config["enabled_tools"] = json.loads(form_data["enabled_tools"])
                except json.JSONDecodeError:
                    new_config["enabled_tools"] = []
        # Handle reflection module with custom prompt
        elif module_id == "reflection":
            new_config = {}
            current_module = module_manager.modules.get(module_id)
            current_config = current_module.get('config', {}) if current_module else {}
            
            # Copy existing config to preserve defaults
            new_config.update(current_config)
            
            # Update reflection_prompt (custom prompt)
            if "reflection_prompt" in form_data:
                new_config['reflection_prompt'] = form_data["reflection_prompt"]
            
            # Update inject_improvement setting
            new_config['inject_improvement'] = form_data.get("inject_improvement") == "true"
            
        # Handle planner module with custom prompt
        elif module_id == "planner":
            new_config = {}
            current_module = module_manager.modules.get(module_id)
            current_config = current_module.get('config', {}) if current_module else {}
            
            # Copy existing config to preserve defaults
            new_config.update(current_config)
            
            # Update planner_prompt (custom prompt)
            if "planner_prompt" in form_data:
                new_config['planner_prompt'] = form_data["planner_prompt"]
            
            # Update max_steps
            if "max_steps" in form_data:
                try:
                    new_config['max_steps'] = int(form_data["max_steps"])
                except ValueError:
                    pass
            
            # Update enabled setting
            new_config['enabled'] = form_data.get("enabled") == "true"
            
            # Update reasoning book integration settings
            new_config['log_to_reasoning_book'] = form_data.get("log_to_reasoning_book") == "true"
            new_config['include_reasoning_context'] = form_data.get("include_reasoning_context") == "true"
        
        # Handle agent_loop module with individual form fields
        elif module_id == "agent_loop":
            new_config = {}
            current_module = module_manager.modules.get(module_id)
            current_config = current_module.get('config', {}) if current_module else {}
            
            # Copy existing config to preserve defaults
            new_config.update(current_config)
            
            # Update loop control settings
            if "max_iterations" in form_data:
                try:
                    new_config['max_iterations'] = int(form_data["max_iterations"])
                except ValueError:
                    pass
            
            if "max_tokens" in form_data:
                try:
                    new_config['max_tokens'] = int(form_data["max_tokens"])
                except ValueError:
                    pass
            
            if "temperature" in form_data:
                try:
                    new_config['temperature'] = float(form_data["temperature"])
                except ValueError:
                    pass
            
            # Update retry & timeout settings
            if "max_llm_retries" in form_data:
                try:
                    new_config['max_llm_retries'] = int(form_data["max_llm_retries"])
                except ValueError:
                    pass
            
            if "retry_delay" in form_data:
                try:
                    new_config['retry_delay'] = float(form_data["retry_delay"])
                except ValueError:
                    pass
            
            if "timeout" in form_data:
                try:
                    new_config['timeout'] = int(form_data["timeout"])
                except ValueError:
                    pass
            
            # Update context inclusion settings (checkboxes)
            new_config['include_plan_in_context'] = form_data.get("include_plan_in_context") == "true"
            new_config['include_memory_context'] = form_data.get("include_memory_context") == "true"
            new_config['include_knowledge_context'] = form_data.get("include_knowledge_context") == "true"
            new_config['include_reasoning_context'] = form_data.get("include_reasoning_context") == "true"
            
            # Update tool error strategy
            if "tool_error_strategy" in form_data:
                new_config['tool_error_strategy'] = form_data["tool_error_strategy"]

            # Context compaction settings
            if "compact_threshold" in form_data:
                try:
                    new_config['compact_threshold'] = int(form_data["compact_threshold"])
                except ValueError:
                    pass

            if "compact_keep_last" in form_data:
                try:
                    new_config['compact_keep_last'] = int(form_data["compact_keep_last"])
                except ValueError:
                    pass

        elif module_id == "email_bridge":
            current_module = module_manager.modules.get(module_id)
            current_config = current_module.get('config', {}) if current_module else {}
            new_config = current_config.copy()
            section = form_data.get('_section', '')

            for str_key in ('imap_host', 'imap_username', 'imap_password', 'imap_folder',
                            'imap_filter_sender', 'smtp_host', 'smtp_username', 'smtp_password',
                            'smtp_from_address', 'reply_to_address'):
                if str_key in form_data:
                    new_config[str_key] = form_data[str_key]
            for int_key in ('imap_port', 'smtp_port', 'poll_interval'):
                if int_key in form_data:
                    try: new_config[int_key] = int(form_data[int_key])
                    except ValueError: pass
            # Booleans: only update for the section that was submitted
            if section in ('imap', ''):
                new_config['imap_use_ssl'] = form_data.get('imap_use_ssl') == 'true'
                new_config['mark_as_read'] = form_data.get('mark_as_read') == 'true'
            if section in ('smtp', ''):
                new_config['smtp_use_tls'] = form_data.get('smtp_use_tls') == 'true'

        else:

            # Standard JSON config
            if "config_json" not in form_data:
                return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Missing config_json field"}})})
            
            new_config = json.loads(form_data["config_json"])
        
        # Preserve hidden keys by merging from existing config
        keys_to_preserve = HIDDEN_CONFIG_KEYS.get(module_id, [])
        current_module = module_manager.modules.get(module_id)
        
        if current_module:
            current_config = current_module.get('config', {})
            for key in keys_to_preserve:
                if key in current_config:
                    new_config[key] = current_config[key]
                    
        module_manager.update_module_config(module_id, new_config)
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Configuration saved"}})})
    except json.JSONDecodeError:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid JSON format"}})})
    except (KeyError, TypeError, ValueError) as e:
        # KeyError: Missing expected form fields
        # TypeError: Invalid type operations during config processing
        # ValueError: Invalid values in form data
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Configuration error: {e}"}})})



@router.post("/modules/reorder")
async def reorder_modules(request: Request, order: str = Form(...), module_manager: ModuleManager = Depends(get_module_manager)):
    module_ids = order.split(',')
    try:
        module_manager.reorder_modules(module_ids)
        # Trigger a refresh of the navbar to reflect the new order
        return Response(status_code=200, headers={"HX-Trigger": "modulesChanged"})
    except (ValueError, KeyError, TypeError) as e:
        # ValueError: Invalid module ID format
        # KeyError: Module not found during reordering
        # TypeError: Invalid operations on module data
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Failed to reorder: {e}"}})})


@router.post("/modules/{module_id}/{action}")
async def toggle_module(request: Request, module_id: str, action: str, module_manager: ModuleManager = Depends(get_module_manager)):
    if action == "enable":
        module = module_manager.enable_module(module_id)
    elif action == "disable":
        module = module_manager.disable_module(module_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    formatted_config = json.dumps(module.get('config', {}), indent=4)
    return templates.TemplateResponse(
        request, "module_details.html", {"module": module, "formatted_config": formatted_config}, headers={"HX-Trigger": json.dumps({"modulesChanged": None, "showMessage": {"level": "success", "message": f"Module {action}d"}})}
    )


# --- AI Flow ---

@router.get("/ai-flow", response_class=HTMLResponse)
async def ai_flow_page(request: Request, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    active_flow_ids = settings_man.get("active_ai_flows", [])
    return templates.TemplateResponse(request, "ai_flow.html", {
        "modules": module_manager.get_all_modules(),
        "flows": flow_manager.list_flows(),
        "active_flow_ids": active_flow_ids,
        "settings": settings_man.settings
    })

@router.get("/ai-flow/{flow_id}", response_class=JSONResponse)
async def get_flow_data(flow_id: str):
    flow = flow_manager.get_flow(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    return flow

@router.get("/ai-flow/{flow_id}/validate", response_class=JSONResponse)
async def validate_flow(flow_id: str, request: Request):
    """Validates a flow for potential issues before execution."""
    flow = flow_manager.get_flow(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    from core.flow_runner import FlowRunner
    module_manager = request.app.state.module_manager
    
    try:
        runner = FlowRunner(flow_id, flow_override=flow)
        validation_result = runner.validate(module_manager)
        return validation_result
    except (ValueError, KeyError, TypeError) as e:
        # ValueError: Flow not found or invalid flow structure
        # KeyError: Missing required flow data
        # TypeError: Invalid data types in flow structure
        return {"valid": False, "issues": [], "warnings": [], "error": f"Validation error: {e}"}


@router.post("/ai-flow/save")
async def save_ai_flow(request: Request, name: str = Form(...), nodes: str = Form(...), connections: str = Form(...), bridges: str = Form("[]"), flow_id: str = Form(None)):
    if not flow_id:
        flow_id = None
    
    try:
        flow_manager.save_flow(name=name, nodes=json.loads(nodes), connections=json.loads(connections), bridges=json.loads(bridges), flow_id=flow_id)
    except json.JSONDecodeError:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid JSON in flow data"}})})
    
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_ids": settings.get("active_ai_flows", [])
    }, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Flow saved successfully"}})})

@router.post("/ai-flow/{flow_id}/rename", response_class=HTMLResponse)
async def rename_flow(request: Request, flow_id: str, name: str = Form(...), settings_man: SettingsManager = Depends(get_settings_manager)):
    flow_manager.rename_flow(flow_id, name)
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_ids": settings_man.get("active_ai_flows", [])
    })

@router.post("/ai-flow/{flow_id}/set-active", response_class=HTMLResponse)
async def set_active_flow(request: Request, flow_id: str, action: str = Form("toggle"), settings_man: SettingsManager = Depends(get_settings_manager)):
    # Create a copy to avoid mutating the internal settings list
    active_flows = list(settings_man.get("active_ai_flows", []))
    
    if action == "activate":
        if flow_id not in active_flows:
            active_flows.append(flow_id)
    elif action == "deactivate":
        if flow_id in active_flows:
            active_flows.remove(flow_id)
    else:
        if flow_id in active_flows:
            active_flows.remove(flow_id)
        else:
            active_flows.append(flow_id)
    
    settings_man.save_settings({"active_ai_flows": active_flows})
    
    # Auto-start the flow if it has a Repeater node
    flow = flow_manager.get_flow(flow_id)
    if flow:
        background_node_types = ["repeater_node"]
        nodes = flow.get("nodes", [])
        connections = flow.get("connections", [])
        
        # Find repeater nodes that have incoming connections
        start_nodes = []
        for n in nodes:
            if n.get("nodeTypeId") in background_node_types:
                # Check if this node has any incoming connections
                has_incoming = any(c.get("to") == n.get("id") for c in connections)
                if has_incoming:
                    start_nodes.append(n)
        
        if start_nodes:
            from core.flow_runner import FlowRunner
            from fastapi import BackgroundTasks
            
            node = start_nodes[0]
            logger.info(f"[System] Auto-starting flow '{flow.get('name')}' from {node['nodeTypeId']} '{node['id']}'.")
            
            async def run_flow():
                runner = FlowRunner(flow_id)
                await runner.run({"_repeat_count": 1}, start_node_id=node['id'])
            
            if hasattr(request.app.state, 'background_tasks'):
                task = asyncio.create_task(run_flow())
                # Add callback to remove task from set when done to prevent memory leak
                task.add_done_callback(request.app.state.background_tasks.discard)
                request.app.state.background_tasks.add(task)
    
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_ids": active_flows
    })

@router.post("/ai-flow/stop-active", response_class=HTMLResponse)
async def stop_active_flow(request: Request, settings_man: SettingsManager = Depends(get_settings_manager)):
    """Stops all active flows by clearing the active flows list and canceling tasks."""
    if hasattr(request.app.state, 'background_tasks'):
        stopped_count = 0
        tasks_to_cancel = []
        for task in list(request.app.state.background_tasks):
            if not task.done():
                task.cancel()
                tasks_to_cancel.append(task)
                stopped_count += 1
        
        # Wait for tasks to actually be cancelled
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        logger.info(f"[System] Cancelled {stopped_count} background tasks")
    
    settings_man.save_settings({"active_ai_flows": []})
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_ids": []
    }, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "info", "message": "All active flows stopped"}})})


@router.post("/ai-flow/{flow_id}/delete", response_class=HTMLResponse)
async def delete_flow(request: Request, flow_id: str, settings_man: SettingsManager = Depends(get_settings_manager)):
    # Create a copy to avoid mutating the internal settings list
    active_flows = list(settings_man.get("active_ai_flows", []))
    if flow_id in active_flows:
        active_flows.remove(flow_id)
        settings_man.save_settings({"active_ai_flows": active_flows})
    flow_manager.delete_flow(flow_id)
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_ids": active_flows
    })

@router.get("/ai-flow/{flow_id}/versions", response_class=JSONResponse)
async def get_flow_versions(flow_id: str):
    """Returns the version history for a flow (metadata only, newest first)."""
    if not flow_manager.get_flow(flow_id):
        raise HTTPException(status_code=404, detail="Flow not found")
    return flow_manager.get_versions(flow_id)


@router.get("/ai-flow/{flow_id}/versions/partial", response_class=HTMLResponse)
async def get_flow_versions_partial(request: Request, flow_id: str):
    """Returns the version history panel HTML for HTMX swap."""
    flow = flow_manager.get_flow(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    versions = flow_manager.get_versions(flow_id)
    return templates.TemplateResponse(request, "ai_flow_versions.html", {
        "flow_id": flow_id,
        "flow_name": flow.get("name", ""),
        "versions": versions,
    })


@router.post("/ai-flow/{flow_id}/rollback/{version}", response_class=HTMLResponse)
async def rollback_flow_version(request: Request, flow_id: str, version: int, settings_man: SettingsManager = Depends(get_settings_manager)):
    """Restores a flow to the specified version snapshot."""
    restored = flow_manager.rollback_version(flow_id, version)
    if restored is None:
        return Response(status_code=404, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Version not found"}})})
    return templates.TemplateResponse(request, "ai_flow_list.html", {
        "flows": flow_manager.list_flows(),
        "active_flow_ids": settings_man.get("active_ai_flows", []),
    }, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": f"Flow restored to version {version}"}})})


@router.post("/ai-flow/make-default")
async def make_active_flow_default(request: Request):
    """Overwrites the default flow with the currently active flow."""
    if flow_manager.make_active_flow_default():
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Active flow saved as Default"}})})
    return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "No active flow found"}})})

@router.post("/ai-flow/{flow_id}/run-node/{node_id}")
async def run_flow_node(flow_id: str, node_id: str, request: Request, background_tasks: BackgroundTasks):
    """Manually triggers a specific node in a flow."""
    
    flow_override = None
    try:
        if "application/json" in request.headers.get("content-type", ""):
            flow_override = await request.json()
    except (json.JSONDecodeError, KeyError, TypeError):
        # JSONDecodeError: Invalid JSON in request body
        # KeyError/TypeError: Unexpected JSON structure
        pass


    async def _run():
        try:
            runner = FlowRunner(flow_id, flow_override=flow_override)
            # Inject some default data so nodes that expect input don't fail immediately
            payload = {
                "trigger": True,
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "manual": True
            }
            await runner.run(payload, start_node_id=node_id)
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            # RuntimeError: Flow execution failed at runtime
            # ValueError: Invalid flow ID or node ID
            # TypeError/KeyError: Data structure issues during execution
            logger.warning(f"[Manual Trigger] Flow execution failed: {e}")


    background_tasks.add_task(_run)
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Node triggered"}})})

# --- Settings ---


# --- Marketplace ---

@router.get("/marketplace", response_class=HTMLResponse)
async def get_marketplace(request: Request, settings_man: SettingsManager = Depends(get_settings_manager), module_manager: ModuleManager = Depends(get_module_manager)):
    import os
    # 1. Local inventory
    modules = module_manager.get_all_modules()
    tools = {}
    tools_path = os.path.join("modules", "tools", "tools.json")
    if os.path.exists(tools_path):
         try:
             with open(tools_path, "r", encoding="utf-8") as f:
                 tools = json.load(f)
         except Exception: pass

    skills = {}
    skills_path = os.path.join("modules", "skills", "data", "skills_metadata.json")
    if os.path.exists(skills_path):
         try:
             with open(skills_path, "r", encoding="utf-8") as f:
                 skills = json.load(f)
         except Exception: pass

    # 2. Community Catalog
    catalog = []
    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    if os.path.exists(catalog_path):
         try:
             with open(catalog_path, "r", encoding="utf-8") as f:
                 catalog = json.load(f)
         except Exception: pass

    return templates.TemplateResponse(request, "marketplace.html", {
        "request": request,
        "modules": modules,
        "tools": tools,
        "skills": skills,
        "catalog": catalog,
        "active_module": "marketplace",
        "settings": settings_man.settings,
        "hardware_id": get_hardware_id()
    })

@router.post("/marketplace/upload")
async def upload_marketplace_item(
    name: str = Form(...),
    description: str = Form(""),
    type: str = Form(...), # 'module', 'tool', 'skill'
    file: UploadFile = File(...),
    image: UploadFile = File(None)
):
    import uuid
    import shutil
    import os
    
    upload_dir = os.path.join("data", "marketplace", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    item_id = str(uuid.uuid4())
    
    # Save Main File
    ext = os.path.splitext(file.filename)[1]
    save_filename = f"{item_id}{ext}"
    save_path = os.path.join(upload_dir, save_filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # Save Image Banner Optional
    image_filename = None
    if image and image.filename:
        img_ext = os.path.splitext(image.filename)[1]
        image_filename = f"{item_id}_banner{img_ext}"
        img_save_path = os.path.join(upload_dir, image_filename)
        with open(img_save_path, "wb") as f:
             shutil.copyfileobj(image.file, f)
        
    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    catalog = []
    if os.path.exists(catalog_path):
         try:
             with open(catalog_path, "r", encoding="utf-8") as f:
                 catalog = json.load(f)
         except Exception: pass
             
    entry = {
        "id": item_id,
        "name": name,
        "description": description,
        "type": type.capitalize(),
        "filename": file.filename,
        "save_filename": save_filename,
        "image_filename": image_filename,
        "uploader_id": get_hardware_id(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    catalog.append(entry)
    
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4)
        
    return RedirectResponse(url="/marketplace", status_code=303)

@router.get("/marketplace/image/{item_id}")
async def get_marketplace_image(item_id: str):
    import os
    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    if not os.path.exists(catalog_path):
         raise HTTPException(status_code=404, detail="Catalog not found")
        
    try:
         with open(catalog_path, "r", encoding="utf-8") as f:
             catalog = json.load(f)
    except Exception:
         raise HTTPException(status_code=500, detail="Failed to read catalog")
        
    item = next((i for i in catalog if i["id"] == item_id), None)
    if not item or not item.get("image_filename"):
         raise HTTPException(status_code=404, detail="Image not configured")
        
    file_path = os.path.join("data", "marketplace", "uploads", item["image_filename"])
    if not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Image missing on disk")
        
    return FileResponse(file_path)

@router.get("/marketplace/download/{item_id}")
async def download_marketplace_item(item_id: str):
    import os
    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    if not os.path.exists(catalog_path):
         raise HTTPException(status_code=404, detail="Catalog not found")
        
    try:
         with open(catalog_path, "r", encoding="utf-8") as f:
             catalog = json.load(f)
    except Exception:
         raise HTTPException(status_code=500, detail="Failed to read catalog")
        
    item = next((i for i in catalog if i["id"] == item_id), None)
    if not item:
         raise HTTPException(status_code=404, detail="Item not found")
        
    file_path = os.path.join("data", "marketplace", "uploads", item["save_filename"])
    if not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="File missing on disk")
        
    return FileResponse(file_path, filename=item["filename"])

@router.get("/marketplace/preview/{item_id}")
async def preview_marketplace_item(item_id: str):
    import os, json
    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    if not os.path.exists(catalog_path): raise HTTPException(status_code=404, detail="Catalog not found")
    try:
         with open(catalog_path, "r", encoding="utf-8") as f: catalog = json.load(f)
    except Exception: raise HTTPException(status_code=500, detail="Failed to read catalog")
    item = next((i for i in catalog if i["id"] == item_id), None)
    if not item: raise HTTPException(status_code=404, detail="Item not found")
    file_path = os.path.join("data", "marketplace", "uploads", item["save_filename"])
    if not os.path.exists(file_path): raise HTTPException(status_code=404, detail="File missing on disk")
    content = ""
    try:
         ext = os.path.splitext(item["save_filename"])[1].lower()
         if ext in ['.md', '.txt', '.json', '.py']:
             with open(file_path, "r", encoding="utf-8") as f: content = f.read()
         else: content = f"Binary file ({ext}). Preview not available."
    except Exception as e: content = f"Could not read preview: {str(e)}"
    return JSONResponse(content={"status": "success", "content": content})

@router.post("/marketplace/delete/{item_id}")
async def delete_marketplace_item(item_id: str):
    import os
    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    if not os.path.exists(catalog_path):
         raise HTTPException(status_code=404, detail="Catalog not found")
        
    try:
         with open(catalog_path, "r", encoding="utf-8") as f:
             catalog = json.load(f)
    except Exception:
         raise HTTPException(status_code=500, detail="Failed to read catalog")
        
    item = next((i for i in catalog if i["id"] == item_id), None)
    if not item:
         raise HTTPException(status_code=404, detail="Item not found")
        
    current_id = get_hardware_id()
    if item.get("uploader_id") != current_id and item.get("uploader_id") != 'local_user':
         raise HTTPException(status_code=403, detail="You can only delete items that you uploaded")
        
    # Remove files
    upload_dir = os.path.join("data", "marketplace", "uploads")
    if item.get("save_filename"):
         fpath = os.path.join(upload_dir, item["save_filename"])
         if os.path.exists(fpath): os.remove(fpath)
            
    if item.get("image_filename"):
         fpath = os.path.join(upload_dir, item["image_filename"])
         if os.path.exists(fpath): os.remove(fpath)
            
    # Update Catalog
    catalog = [i for i in catalog if i["id"] != item_id]
    with open(catalog_path, "w", encoding="utf-8") as f:
         json.dump(catalog, f, indent=4)
         
    from fastapi import Response
    return Response(status_code=200)

@router.post("/settings/skills/delete/{skill_id}")
async def delete_skill(skill_id: str):
    import os
    skills_path = os.path.join("modules", "skills", "data", "skills_metadata.json")
    if not os.path.exists(skills_path):
         raise HTTPException(status_code=404, detail="Skills Metadata not found")
        
    try:
         with open(skills_path, "r", encoding="utf-8") as f:
             skills = json.load(f)
    except Exception:
         raise HTTPException(status_code=500, detail="Failed to read skills metadata")
        
    if skill_id not in skills:
         raise HTTPException(status_code=404, detail="Skill not found")
        
    # Delete from JSON
    del skills[skill_id]
    with open(skills_path, "w", encoding="utf-8") as f:
         json.dump(skills, f, indent=4)
         
    # Delete MD file
    content_path = os.path.join("modules", "skills", "data", f"{skill_id}.md")
    if os.path.exists(content_path):
         os.remove(content_path)
         
    return JSONResponse(content={"status": "success", "message": "Skill deleted"})

@router.post("/settings/skills/save/{skill_id}")
async def save_skill(skill_id: str, instructions: str = Form(...)):
    import os
    content_path = os.path.join("modules", "skills", "data", f"{skill_id}.md")
    try:
         with open(content_path, "w", encoding="utf-8") as f:
             f.write(instructions)
    except Exception:
         raise HTTPException(status_code=500, detail="Failed to save skill content")
    return JSONResponse(content={"status": "success", "message": "Skill content saved"})

@router.post("/settings/skills/create")
async def create_skill(
    name: str = Form(...),
    description: str = Form(...),
    category: str = Form("general"),
    instructions: str = Form("")
):
    import os, re
    from datetime import datetime
    skill_id = re.sub(r'[^a-zA-Z0-9\s]', '', name).lower().replace(' ', '_')
    if not skill_id: raise HTTPException(status_code=400, detail="Invalid skill name")
    skills_path = os.path.join("modules", "skills", "data", "skills_metadata.json")
    try:
         if os.path.exists(skills_path):
              with open(skills_path, "r", encoding="utf-8") as f: skills = json.load(f)
         else: skills = {}
    except Exception: skills = {}
    if skill_id in skills: raise HTTPException(status_code=400, detail="Skill already exists")
    skills[skill_id] = {
         "name": name, "description": description, "category": category,
         "tags": [], "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat()
    }
    try:
         with open(skills_path, "w", encoding="utf-8") as f: json.dump(skills, f, indent=4)
         content_path = os.path.join("modules", "skills", "data", f"{skill_id}.md")
         with open(content_path, "w", encoding="utf-8") as f: f.write(instructions)
    except Exception: raise HTTPException(status_code=500, detail="Failed to create skill assets")
    return JSONResponse(content={"status": "success", "message": "Skill created", "id": skill_id})

@router.post("/settings/skills/upload_to_marketplace/{skill_id}")
async def upload_skill_to_marketplace(skill_id: str):
    import os, json, uuid, shutil
    from datetime import datetime
    skills_path = os.path.join("modules", "skills", "data", "skills_metadata.json")
    content_path = os.path.join("modules", "skills", "data", f"{skill_id}.md")
    if not os.path.exists(skills_path) or not os.path.exists(content_path): raise HTTPException(status_code=404, detail="Skill assets not found")
    import hashlib
    try:
         with open(skills_path, "r", encoding="utf-8") as f: skills = json.load(f)
    except Exception: raise HTTPException(status_code=500, detail="Failed to read skills metadata")
    skill = skills.get(skill_id)
    if not skill: raise HTTPException(status_code=404, detail="Skill not found")
    
    with open(content_path, "r", encoding="utf-8") as f: content_str = f.read()
    content_hash = hashlib.md5(content_str.encode("utf-8")).hexdigest()

    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    upload_dir = os.path.join("data", "marketplace", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    try:
         if os.path.exists(catalog_path):
              with open(catalog_path, "r", encoding="utf-8") as f: catalog = json.load(f)
         else: catalog = []
    except Exception: catalog = []
    
    # Deduplicate: Check if already exists in catalog
    existing_item = next((i for i in catalog if i.get("type") == "skill" and i.get("filename") == f"{skill_id}.md"), None)
    
    item_id = existing_item["id"] if existing_item else str(uuid.uuid4())
    shutil.copy(content_path, os.path.join(upload_dir, f"{item_id}.md"))
    
    if existing_item:
         # Update existing
         existing_item["name"] = skill.get("name", skill_id)
         existing_item["description"] = skill.get("description", "")
         existing_item["content_hash"] = content_hash
         existing_item["uploaded_at"] = datetime.now().isoformat()
    else:
         # Append new
         catalog.append({
              "id": item_id, "name": skill.get("name", skill_id), "description": skill.get("description", ""),
              "type": "skill", "filename": f"{skill_id}.md", "save_filename": f"{item_id}.md", 
              "uploaded_at": datetime.now().isoformat(), "uploader_id": get_hardware_id(),
              "content_hash": content_hash
         })
         
    with open(catalog_path, "w", encoding="utf-8") as f: json.dump(catalog, f, indent=4)
    return JSONResponse(content={"status": "success", "message": "Uploaded to Marketplace", "id": item_id})

@router.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request, settings_man: SettingsManager = Depends(get_settings_manager), module_manager: ModuleManager = Depends(get_module_manager)):
    import os
    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or "Unknown"
    }
    
    skills = {}
    skills_path = os.path.join("modules", "skills", "data", "skills_metadata.json")
    import hashlib
    if os.path.exists(skills_path):
         try:
             with open(skills_path, "r", encoding="utf-8") as f:
                 skills = json.load(f)
             for skill_id, skill_data in skills.items():
                 content_path = os.path.join("modules", "skills", "data", f"{skill_id}.md")
                 if os.path.exists(content_path):
                      with open(content_path, "r", encoding="utf-8") as f_content:
                          instructions = f_content.read()
                          skill_data["instructions"] = instructions
                          skill_data["content_hash"] = hashlib.md5(instructions.encode("utf-8")).hexdigest()
         except Exception: pass
         
    catalog = []
    catalog_path = os.path.join("data", "marketplace", "catalog.json")
    if os.path.exists(catalog_path):
         try:
              with open(catalog_path, "r", encoding="utf-8") as f: catalog = json.load(f)
         except Exception: pass

    return templates.TemplateResponse(request, "settings.html", {
        "settings": settings_man.settings, "modules": module_manager.get_all_modules(),
        "system_time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "system_info": system_info,
        "hardware_id": get_hardware_id(),
        "skills": skills,
        "marketplace_catalog": catalog
    })

@router.get("/system-time", response_class=HTMLResponse)
async def get_system_time(request: Request):
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

@router.get("/settings/modules-nav", response_class=HTMLResponse)
async def get_settings_modules_nav(request: Request, module_manager: ModuleManager = Depends(get_module_manager)):
    return templates.TemplateResponse(request, "settings_modules_nav.html", {"modules": module_manager.get_all_modules()})

@router.get("/footer", response_class=HTMLResponse)
async def get_footer(request: Request, settings_man: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "footer.html", {"settings": settings_man.settings})

@router.get("/settings/export/config")
async def export_config(settings_man: SettingsManager = Depends(get_settings_manager)):
    """Downloads the current settings.json file."""
    return JSONResponse(
        content=settings_man.settings,
        headers={"Content-Disposition": 'attachment; filename="neurocore_settings.json"'}
    )

@router.post("/settings/import/config")
async def import_config(request: Request, file: UploadFile = File(...), settings_man: SettingsManager = Depends(get_settings_manager)):
    """Imports settings from a JSON file."""
    try:
        content = await file.read()
        new_settings = json.loads(content)
        if not isinstance(new_settings, dict):
             return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid format: Root must be a dictionary"}})})
        
        settings_man.save_settings(new_settings)
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"settingsChanged": None, "showMessage": {"level": "success", "message": "Configuration imported successfully"}})})
    except json.JSONDecodeError:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid JSON file"}})})
    except (OSError, PermissionError, TypeError, KeyError, ValueError) as e:
        # OSError/PermissionError: File system issues during import
        # TypeError/KeyError: Invalid settings structure
        # ValueError: Invalid settings values (from _validate_settings)
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Import failed: {e}"}})})


@router.get("/settings/export/flows")
async def export_flows():
    """Downloads the current ai_flows.json file."""
    # Use thread-safe method to get flows dict with lock held
    data = flow_manager.get_all_flows_dict()
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": 'attachment; filename="ai_flows_backup.json"'}
    )

@router.post("/settings/import/flows")
async def import_flows(request: Request, file: UploadFile = File(...)):
    """Imports flows from a JSON file."""
    try:
        content = await file.read()
        flows_data = json.loads(content)
        if not isinstance(flows_data, dict):
             return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid format: Root must be a dictionary"}})})
        
        flow_manager.import_flows(flows_data)
        return Response(status_code=200, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "success", "message": "Flows imported successfully"}})})
    except json.JSONDecodeError:
        return Response(status_code=400, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": "Invalid JSON file"}})})
    except (OSError, PermissionError, TypeError, KeyError, ValueError) as e:
        # OSError/PermissionError: File system issues during import
        # TypeError/KeyError: Invalid flow data structure
        # ValueError: Invalid flow ID or configuration values
        return Response(status_code=500, headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Import failed: {e}"}})})


@router.post("/settings/reset")
async def reset_settings(request: Request, settings_man: SettingsManager = Depends(get_settings_manager)):
    """Resets settings to defaults."""
    from core.settings import DEFAULT_SETTINGS
    # Preserve the file path but overwrite content
    settings_man.save_settings(DEFAULT_SETTINGS)
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"settingsChanged": None, "showMessage": {"level": "success", "message": "Settings reset to defaults"}})})

@router.post("/settings/save")
async def save_settings_route(request: Request, settings_man: SettingsManager = Depends(get_settings_manager)):
    form_data = await request.form()
    updates = {}
    
    # Handle text fields (update only if present)
    text_fields = ["llm_api_url", "llm_api_key", "embedding_api_url", "default_model", "embedding_model"]
    for field in text_fields:
        if field in form_data:
            updates[field] = form_data[field]
    
    # Handle numeric fields
    for field in ["temperature", "max_tokens"]:
        if field in form_data:
            try:
                updates[field] = float(form_data[field])
            except (ValueError, TypeError):
                pass
            
    if "request_timeout" in form_data:
        try:
            updates["request_timeout"] = float(form_data["request_timeout"])
        except (ValueError, TypeError):
            pass
            
    if "max_node_loops" in form_data:
        try:
            updates["max_node_loops"] = int(form_data["max_node_loops"])
        except (ValueError, TypeError):
            pass
            
    # Handle debug_mode checkbox (only if the form intended to submit it)
    if "save_debug_mode" in form_data:
        updates["debug_mode"] = form_data.get("debug_mode") == "on"
    
    if "save_ui_wide_mode" in form_data:
        updates["ui_wide_mode"] = form_data.get("ui_wide_mode") == "on"
        
    if "save_ui_show_footer" in form_data:
        updates["ui_show_footer"] = form_data.get("ui_show_footer") == "on"

    # Catch validation errors and return 400 with helpful message
    try:
        settings_man.save_settings(updates)
    except ValueError as e:
        return Response(
            status_code=400, 
            headers={"HX-Trigger": json.dumps({"showMessage": {"level": "error", "message": f"Invalid settings: {str(e)}"}})}
        )
    
    return Response(status_code=200, headers={"HX-Trigger": json.dumps({"settingsChanged": None, "showMessage": {"level": "success", "message": "Settings saved successfully"}})})

# --- Debug ---

@router.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request, settings_man: SettingsManager = Depends(get_settings_manager), module_manager: ModuleManager = Depends(get_module_manager)):
    if not settings_man.get("debug_mode"):
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "debug.html", {"settings": settings_man.settings, "modules": module_manager.get_all_modules()})

@router.get("/debug/logs", response_class=HTMLResponse)
async def get_debug_logs(request: Request, _: bool = Depends(require_debug_mode)):
    return templates.TemplateResponse(request, "debug_logs.html", {"logs": debug_logger.get_logs()})

@router.get("/debug/events", response_class=JSONResponse)
async def get_debug_events(request: Request, since: float = 0, _: bool = Depends(require_debug_mode)):
    return debug_logger.get_recent_logs(since)

@router.get("/debug/agent-summary", response_class=JSONResponse)
async def get_agent_summary(request: Request, since: float = None, limit: int = 5, _: bool = Depends(require_debug_mode)):
    """Get a summary of the last agent session trace events.
    
    Query params:
        since: Optional time window in seconds (e.g., 300 = last 5 minutes, 900 = last 15 minutes)
        limit: Number of recent events to include (default: 5)
    """
    from core.session_manager import session_manager
    
    # Convert relative time (seconds) to absolute timestamp
    since_ts = None
    if since is not None:
        since_ts = time.time() - since
    
    summary = session_manager.get_trace_summary(limit=limit, since=since_ts)
    return JSONResponse(content=summary)

@router.get("/debug/summary", response_class=HTMLResponse)
async def get_debug_summary(request: Request, since: float = None, limit: int = 5, _: bool = Depends(require_debug_mode)):
    """Get the agent summary panel for the debug page.

    Query params:
        since: Optional time window in seconds (e.g., 300 = last 5 minutes, 900 = last 15 minutes)
        limit: Number of recent events to include (default: 5)
    """
    from core.session_manager import session_manager
    from core.observability import get_token_stats

    # Convert relative time (seconds) to absolute timestamp
    since_ts = None
    if since is not None:
        since_ts = time.time() - since

    summary = session_manager.get_trace_summary(limit=limit, since=since_ts)
    return templates.TemplateResponse(
        request, "debug_summary.html",
        {"summary": summary, "stats": get_token_stats()},
    )

@router.get("/debug/token-stats", response_class=JSONResponse)
async def get_token_stats_json(request: Request, _: bool = Depends(require_debug_mode)):
    """Return LLM token usage counters (total + per-model) as JSON."""
    from core.observability import get_token_stats
    return JSONResponse(content=get_token_stats())

@router.get("/debug/token-stats/partial", response_class=HTMLResponse)
async def get_token_stats_partial(request: Request, _: bool = Depends(require_debug_mode)):
    """Return the token stats panel as an HTML partial for HTMX polling."""
    from core.observability import get_token_stats
    return templates.TemplateResponse(
        request, "debug_token_stats.html", {"stats": get_token_stats()}
    )

@router.post("/debug/clear")
async def clear_debug_logs(request: Request, _: bool = Depends(require_debug_mode)):
    debug_logger.clear()
    return templates.TemplateResponse(request, "debug_logs.html", {"logs": []})

# --- Goals ---

@router.get("/goals", response_class=HTMLResponse)
async def goals_page(request: Request, module_manager: ModuleManager = Depends(get_module_manager), settings_man: SettingsManager = Depends(get_settings_manager)):
    return templates.TemplateResponse(request, "goals.html", {
        "modules": module_manager.get_all_modules(),
        "settings": settings_man.settings
    })


@router.get("/modules/{module_id}/default-config")
async def get_default_config(module_id: str, module_manager: ModuleManager = Depends(get_module_manager)):
    """Returns the default configuration for modules that support loading defaults (e.g., agent_loop)."""
    from pathlib import Path
    
    module = module_manager.modules.get(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    # Only support agent_loop for now
    if module_id != 'agent_loop':
        raise HTTPException(status_code=400, detail="Module does not support loading default config")
    
    # Read the module.json file to get default config
    module_dir = Path(__file__).resolve().parent.parent / "modules" / module_id
    module_json_path = module_dir / "module.json"
    
    if not module_json_path.exists():
        raise HTTPException(status_code=404, detail="Module configuration file not found")
    
    try:
        with open(module_json_path, 'r') as f:
            module_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        raise HTTPException(status_code=500, detail="Failed to read module configuration")
    
    # Extract default values from config schema
    config_schema = module_data.get('config', {})
    default_config = {}
    
    for key, value in config_schema.items():
        if isinstance(value, dict) and 'value' in value:
            default_config[key] = value['value']
        else:
            # For flat config values (like in chat module)
            default_config[key] = value
    
    return JSONResponse(content=default_config)
