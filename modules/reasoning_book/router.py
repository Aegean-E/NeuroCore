from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from core.dependencies import get_module_manager, get_settings_manager
from .service import service

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

@router.get("/", response_class=HTMLResponse)
async def reasoning_page(request: Request, module_manager=Depends(get_module_manager), settings_man=Depends(get_settings_manager)):
    """Serves the full page shell with the reasoning book active."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "modules": module_manager.get_all_modules(),
        "active_module": "reasoning_book",
        "settings": settings_man.settings,
        "full_width_content": True
    })

@router.get("/gui", response_class=HTMLResponse)
async def reasoning_gui(request: Request):
    """Serves the inner content of the reasoning book."""
    return templates.TemplateResponse("reasoning_book.html", {
        "request": request, 
        "thoughts": service.get_thoughts()
    })

@router.get("/thoughts", response_class=HTMLResponse)
async def get_thoughts(request: Request):
    """Returns just the list of thoughts for polling."""
    return templates.TemplateResponse("reasoning_thoughts.html", {
        "request": request, 
        "thoughts": service.get_thoughts()
    })

@router.post("/clear")
async def clear_thoughts(request: Request):
    service.thoughts.clear()
    return templates.TemplateResponse("reasoning_thoughts.html", {
        "request": request, 
        "thoughts": []
    })