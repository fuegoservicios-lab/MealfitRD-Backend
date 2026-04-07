import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

diary_router_code = """from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File, Form
from typing import Optional
import logging

from auth import get_verified_user_id
from db import log_consumed_meal, get_consumed_meals_today, save_visual_entry
from vision_agent import process_image_with_vision, get_multimodal_embedding

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/diary",
    tags=["diary"],
)

"""

pattern = re.compile(r'(@app\.(get|post|delete|put|patch)\("/api/diary(.*?)"\).*?)(?=@app\.|if __name__ ==)', re.DOTALL)
matches = pattern.findall(content)

endpoints = []
for full_match, method, path in matches:
    endpoint_code = full_match
    endpoint_code = endpoint_code.replace(f'@app.{method}("/api/diary{path}")', f'@router.{method}("{path}")')
    endpoint_code = endpoint_code.replace(f'@router.{method}("")', f'@router.{method}("/")')
    endpoints.append(endpoint_code)

with open('routers/diary.py', 'w', encoding='utf-8') as f:
    f.write(diary_router_code + "\n".join(endpoints))

print(f"✅ Se han extraído {len(endpoints)} endpoints a routers/diary.py")

for full_match, _, _ in matches:
    content = content.replace(full_match, "")

router_import = "from routers.diary import router as diary_router\napp.include_router(diary_router)\n"
app_decl_match = re.search(r'app\.include_router\(shopping_router\)\n', content)
if app_decl_match:
    content = content[:app_decl_match.end()] + router_import + content[app_decl_match.end():]
    print("Registrado diary_router")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

