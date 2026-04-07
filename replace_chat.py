import re
import os

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Router base para chat.py
chat_router_code = """from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
import traceback
import json

from auth import get_verified_user_id, verify_api_quota
from rate_limiter import RateLimiter
from db import (
    get_user_chat_sessions, get_guest_chat_sessions, get_session_owner, delete_user_agent_sessions,
    delete_single_agent_session, update_session_title, get_session_messages, get_or_create_session,
    save_message, save_message_feedback, log_api_usage
)
from memory_manager import build_memory_context, summarize_and_prune
from agent import generate_chat_title_background, chat_with_agent, chat_with_agent_stream
from services import merge_form_data_with_profile

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)

"""

# Encontrar y extraer todas las rutas que comiencen por @app.xxx("/api/chat")
# El patrón busca "@app...("/api/chat...") y luego hasta que encuentra la siguiente @app
pattern = re.compile(r'(@app\.(get|post|delete|put|patch)\("/api/chat(.*?)"\).*?)(?=@app\.)', re.DOTALL)
matches = pattern.findall(content + "\n@app.") # Añadido @app. al final para matchear el último si lo es

# Solo extraemos los últimos 8 que listamos arriba
# Sustituiremos @app por @router, y quitaremos /api/chat del path porque el router ya tiene ese prefix.

endpoints = []
for full_match, method, path in matches:
    endpoint_code = full_match
    
    # Reemplazar @app por @router
    endpoint_code = endpoint_code.replace(f'@app.{method}("/api/chat{path}")', f'@router.{method}("{path}")')
    
    # En caso de que el path sea vacío (ej: @app.post("/api/chat")), se volvería @router.post("") que a veces falla en axios,
    # pero FastAPI lo maneja. Aunque mejor lo pasamos a "/"
    endpoint_code = endpoint_code.replace(f'@router.{method}("")', f'@router.{method}("/")')
    
    endpoints.append(endpoint_code)

# Escribir chat.py
with open('routers/chat.py', 'w', encoding='utf-8') as f:
    f.write(chat_router_code + "\n".join(endpoints))

print(f"✅ Se han extraído {len(endpoints)} endpoints a routers/chat.py")

# Ahora eliminamos esos bloques de app.py
for full_match, _, _ in matches:
    content = content.replace(full_match, "")

# Registrar el router en app.py
router_import = "from routers.chat import router as chat_router\napp.include_router(chat_router)\n"
app_decl_match = re.search(r'app\.include_router\(plans_router\)\n', content)
if app_decl_match:
    content = content[:app_decl_match.end()] + router_import + content[app_decl_match.end():]
    print("Registrado chat_router")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

