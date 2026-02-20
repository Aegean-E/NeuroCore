import json
import os
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.dependencies import get_module_manager

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

def _load_tools():
    """Load available tools from the tools module."""
    tools_file = os.path.join(os.path.dirname(__file__), "..", "tools", "tools.json")
    if os.path.exists(tools_file):
        try:
            with open(tools_file, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

@router.get("/available-tools", response_class=HTMLResponse)
async def get_available_tools(request: Request, module_manager = Depends(get_module_manager)):
    """Get available tools with checkboxes."""
    all_tools = _load_tools()
    system_prompt_module = module_manager.modules.get("system_prompt", {})
    enabled_tools = system_prompt_module.get("config", {}).get("enabled_tools", [])
    
    tools_html = ""
    for tool_name, tool_data in all_tools.items():
        is_checked = tool_name in enabled_tools
        checked_attr = "checked" if is_checked else ""
        
        if "definition" in tool_data and "function" in tool_data["definition"]:
            func_def = tool_data["definition"]["function"]
            desc = func_def.get("description", "No description available")
            
            tools_html += f"""
            <div class="flex items-start space-x-3 p-3 rounded-lg hover:bg-slate-800/50 transition-colors">
                <input type="checkbox" id="tool_{tool_name}" value="{tool_name}" 
                       class="mt-1 w-4 h-4 rounded border-slate-600 bg-slate-950 text-blue-600 focus:ring-blue-500 cursor-pointer tool-checkbox"
                       {checked_attr}
                       onchange="updateEnabledTools()">
                <label for="tool_{tool_name}" class="cursor-pointer flex-1">
                    <div class="text-sm font-medium text-slate-300">{tool_name}</div>
                    <div class="text-xs text-slate-500">{desc}</div>
                </label>
            </div>
            """
    
    if not tools_html:
        tools_html = '<p class="text-sm text-slate-500">No tools available</p>'
    
    full_html = f"""
    <div class="space-y-2" id="tools-container">
        {tools_html}
    </div>
    <script>
        function updateEnabledTools() {{
            const checked = Array.from(document.querySelectorAll('#tools-container .tool-checkbox:checked'))
                .map(c => c.value);
            const hiddenField = document.getElementById('enabled_tools');
            if (hiddenField) {{
                hiddenField.value = JSON.stringify(checked);
            }}
        }}
    </script>
    """
    
    return full_html
