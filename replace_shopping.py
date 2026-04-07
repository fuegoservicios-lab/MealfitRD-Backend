import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

shopping_router_code = """from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks
from typing import Optional
import logging

from auth import get_verified_user_id
from rate_limiter import _shopping_write_limiter, _shopping_autogen_limiter
from db import (
    get_custom_shopping_items as _get_items, update_custom_shopping_item,
    update_custom_shopping_item_status, delete_custom_shopping_item, clear_all_shopping_items,
    add_custom_shopping_items, uncheck_all_shopping_items, purge_old_shopping_items,
    deduplicate_shopping_items, log_api_usage
)
from agent import generate_auto_shopping_list
from services import regenerate_shopping_list_safe, _preserve_shopping_checkmarks

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/shopping",
    tags=["shopping"],
)

"""

pattern = re.compile(r'(@app\.(get|post|delete|put|patch)\("/api/shopping(.*?)"\).*?)(?=@app\.)', re.DOTALL)
matches = pattern.findall(content)

endpoints = []
for full_match, method, path in matches:
    endpoint_code = full_match
    endpoint_code = endpoint_code.replace(f'@app.{method}("/api/shopping{path}")', f'@router.{method}("{path}")')
    endpoint_code = endpoint_code.replace(f'@router.{method}("")', f'@router.{method}("/")')
    endpoints.append(endpoint_code)

with open('routers/shopping.py', 'w', encoding='utf-8') as f:
    f.write(shopping_router_code + "\n".join(endpoints))

print(f"✅ Se han extraído {len(endpoints)} endpoints a routers/shopping.py")

for full_match, _, _ in matches:
    content = content.replace(full_match, "")

router_import = "from routers.shopping import router as shopping_router\napp.include_router(shopping_router)\n"
app_decl_match = re.search(r'app\.include_router\(chat_router\)\n', content)
if app_decl_match:
    content = content[:app_decl_match.end()] + router_import + content[app_decl_match.end():]
    print("Registrado shopping_router")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

