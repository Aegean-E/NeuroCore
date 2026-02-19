from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from .backend import memory_store

router = APIRouter()

@router.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    stats = memory_store.get_memory_stats()
    
    html = """
    <div class="grid grid-cols-2 gap-4 mb-6">
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Total Active</div>
            <div class="text-2xl font-bold text-slate-100">{}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Archived</div>
            <div class="text-2xl font-bold text-slate-100">{}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">User Origin</div>
            <div class="text-2xl font-bold text-blue-400">{}</div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700/50">
            <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Assistant Origin</div>
            <div class="text-2xl font-bold text-emerald-400">{}</div>
        </div>
    </div>
    """.format(stats.get('total', 0), stats.get('archived', 0), stats.get('user', 0), stats.get('assistant', 0))
    
    if stats.get('types'):
        html += """
        <h4 class="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">Memory Types</h4>
        <div class="space-y-2">
        """
        for type_name, count in stats['types'].items():
            total = stats.get('total', 1)
            percent = (count / total) * 100 if total > 0 else 0
            html += f"""
            <div class="flex items-center justify-between text-sm">
                <span class="text-slate-300 w-24">{type_name}</span>
                <div class="flex-grow mx-3 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div class="h-full bg-blue-600 rounded-full" style="width: {percent}%"></div>
                </div>
                <span class="text-slate-400 font-mono">{count}</span>
            </div>
            """
        html += "</div>"
    elif stats.get('total', 0) == 0:
        html = "<p class='text-slate-500 italic'>No memories stored yet.</p>"
        
    return html