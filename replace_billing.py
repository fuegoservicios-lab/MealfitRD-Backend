import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Borrar la carga de módulos import httpx y los endpoints de billing y paypal webhooks
# Todo eso empieza en "import httpx" seguido de "@app.post("/api/subscription/verify")"
# Y termina en el end de "api_webhook_paypal"
pattern = re.compile(r'import httpx.*?def api_webhook_paypal.*?return \{"success": False\}\n\n', re.DOTALL)
match = pattern.search(content)

if match:
    # Eliminamos el bloque entero
    content = content[:match.start()] + content[match.end():]
    
    # Añadiremos el router registration en la seccion de CORS y webhooks
    router_import = "\nfrom routers.billing import router as billing_router, webhooks_router\napp.include_router(billing_router)\napp.include_router(webhooks_router)\n\n"
    
    # Lo inyectaremos después de app = FastAPI(lifespan=lifespan)
    app_decl_match = re.search(r'app = FastAPI\(lifespan=lifespan\)\napp\.mount\("/uploads", StaticFiles\(directory="uploads"\), name="uploads"\)\n', content)
    if app_decl_match:
        content = content[:app_decl_match.end()] + router_import + content[app_decl_match.end():]
        print("Import de Routers añadido exitosamente.")
    else:
        print("No se encontró la declaración de la app para insertar imports.")
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Bloque eliminado y reemplazado exitosamente.")
else:
    print("No se encontro el bloque de api_webhook_paypal.")
