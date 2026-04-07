import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Borrar api_analyze
# @app.post("/api/analyze") ... -> todo su def
pattern_analyze = re.compile(r'@app\.post\("/api/analyze"\).*?def api_analyze.*?return result\n.*?except.*?raise HTTPException\(status_code=500, detail=str\(e\)\)\n\n', re.DOTALL)
match1 = pattern_analyze.search(content)
if match1:
    content = content[:match1.start()] + content[match1.end():]
    print("Borrado api_analyze")

# Borrar api_expand_recipe
pattern_expand = re.compile(r'@app\.post\("/api/recipe/expand"\).*?def api_expand_recipe.*?return \{"success": True, "expanded_recipe": expanded_steps\}\n.*?except.*?raise HTTPException\(status_code=500, detail=str\(e\)\)\n\n', re.DOTALL)
match2 = pattern_expand.search(content)
if match2:
    content = content[:match2.start()] + content[match2.end():]
    print("Borrado api_expand_recipe")

# Borrar swap_meal
pattern_swap = re.compile(r'@app\.post\("/api/swap-meal"\).*?def api_swap_meal.*?return result\n.*?except.*?raise HTTPException\(status_code=500, detail=str\(e\)\)\n\n', re.DOTALL)
match3 = pattern_swap.search(content)
if match3:
    content = content[:match3.start()] + content[match3.end():]
    print("Borrado swap_meal")
    
# Borrar like
pattern_like = re.compile(r'@app\.post\("/api/like"\).*?def api_like.*?return \{"success": True.*?\}\n.*?except.*?return \{"error".*?\}\n\n', re.DOTALL)
match4 = pattern_like.search(content)
if match4:
    content = content[:match4.start()] + content[match4.end():]
    print("Borrado like")

# Registrar el router justo donde registramos billing_router
router_import = "from routers.plans import router as plans_router\napp.include_router(plans_router)\n"
app_decl_match = re.search(r'app\.include_router\(billing_router\)\napp\.include_router\(webhooks_router\)\n', content)
if app_decl_match:
    content = content[:app_decl_match.end()] + router_import + content[app_decl_match.end():]
    print("Registrado plans_router")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
