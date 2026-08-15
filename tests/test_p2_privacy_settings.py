"""[P2-PRIVACY-SETTINGS · 2026-07-04] Test ancla de la sección Privacidad
(Configuración) + export self-service de datos (GET /api/account/export).

Contratos protegidos:
  1. Endpoint existe en app.py con throttle estricto (RateLimiter 3/5min,
     espejo de P1-ACCOUNT-DELETE-1) y guard anti-guest/anon (401).
  2. No-IDOR: el user se deriva SOLO del JWT (cero `user_id` del cliente
     en el cuerpo del endpoint — simétrico I2 / P0-AGENT-1).
  3. Quota-exempt: sin verify_api_quota / log_api_usage (lección
     P1-NEVERA-QUOTA-EXEMPT: exportar tus datos es un derecho, no consumo IA).
  4. Payload higiénico: strip de columnas `embedding` (vectores pgvector
     de 1536 floats — internos y pesados) + caps por tabla.
  5. Frontend: sección 'privacy' registrada en Settings.jsx (nav + render)
     y el botón de export llama al endpoint.

Parser-based (regex sobre source) — no levanta el stack (pytest local →
Neon cuelga).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND.parent
_APP = _BACKEND / "app.py"
_SETTINGS = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    assert path.exists(), f"No existe {path} — ¿se renombró sin actualizar el test?"
    return path.read_text(encoding="utf-8")


def _export_endpoint_body(app_src: str) -> str:
    """Recorta el cuerpo del endpoint export (desde su decorator hasta el
    siguiente decorator @app.*) para asserts function-scoped."""
    m = re.search(
        r'@app\.get\("/api/account/export"\)(.*?)\n@app\.',
        app_src,
        re.DOTALL,
    )
    assert m, "No se encontró el endpoint GET /api/account/export en app.py."
    return m.group(1)


def test_export_endpoint_exists_throttled_and_guarded():
    src = _read(_APP)
    assert re.search(r"_ACCOUNT_EXPORT_LIMITER\s*=\s*RateLimiter\(max_calls=3,\s*period_seconds=300\)", src), (
        "El export debe llevar throttle estricto 3/5min (espejo de "
        "_ACCOUNT_DELETE_LIMITER — la query carga plan_data completo)."
    )
    body = _export_endpoint_body(src)
    assert "Depends(_ACCOUNT_EXPORT_LIMITER)" in body
    assert 'verified_user_id == "guest"' in body and "401" in body, (
        "El endpoint debe rechazar guests/anon con 401."
    )


def test_export_no_idor_and_quota_exempt():
    body = _export_endpoint_body(_read(_APP))
    # No-IDOR: el endpoint NO acepta body/user_id del cliente.
    assert "Body(" not in body, (
        "P2-PRIVACY-SETTINGS: el export NO debe aceptar body del cliente — el "
        "user se deriva SOLO del JWT (simétrico I2 / P0-AGENT-1)."
    )
    # Quota-exempt (uso real, no menciones en prosa del docstring).
    for symbol in ("verify_api_quota", "log_api_usage"):
        assert not re.search(rf"\b{symbol}\s*\(", body) and f"Depends({symbol})" not in body, (
            f"P2-PRIVACY-SETTINGS: el export NO debe invocar {symbol} "
            "(exportar tus datos es un derecho, no consumo de IA)."
        )


def test_export_payload_hygiene():
    src = _read(_APP)
    assert re.search(r'_ACCOUNT_EXPORT_STRIPPED_KEYS\s*=\s*\(\s*"embedding"', src), (
        "El export debe stripear columnas `embedding` (vectores internos)."
    )
    # Tablas núcleo del export presentes en la lista canónica.
    tables_m = re.search(r"_ACCOUNT_EXPORT_TABLES\s*=\s*\((.*?)\n\)", src, re.DOTALL)
    assert tables_m, "Falta la tupla _ACCOUNT_EXPORT_TABLES."
    tables_src = tables_m.group(1)
    for core in ("user_profiles", "meal_plans", "user_inventory", "user_facts"):
        assert core in tables_src, f"Tabla núcleo {core} ausente del export."
    # Caps numéricos: toda entry lleva (tabla, col, cap int).
    assert re.search(r'\("meal_plans",\s*"user_id",\s*\d+\)', tables_src), (
        "meal_plans debe llevar cap de filas (plan_data pesa MBs)."
    )


def test_marker_bumped():
    """[reparado P1-OAUTH-CHALLENGE-COOKIE · 2026-08-10] Antes exigía la subcadena
    literal «2026-07» en el marker, así que caducó al llegar agosto y llevaba en
    rojo desde entonces. Un mes es una subcadena, no una comparación: lo que
    quería decir era «no anterior a este P-fix»."""
    src = _read(_APP)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "No se encontró _LAST_KNOWN_PFIX."
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha, f"el marker no lleva fecha legible: {m.group(1)!r}"
    assert fecha.group(1) >= "2026-07-04", f"Marker sospechosamente viejo: {m.group(1)!r}"


def test_frontend_privacy_section_wired():
    src = _read(_SETTINGS)
    assert "'privacy'" in src, "Falta 'privacy' en SECTION_IDS / sectionsConfig."
    # [P1-I18N-DASHBOARD · 2026-08-15] `label: 'Privacidad'` pasó a
    # `label: t('Privacidad')`. La PROPIEDAD vigilada —«la fila de Privacidad
    # existe en el nav de Settings, rotulada»— no cambió: en es-DO `t()` no
    # tiene catálogo y devuelve el mismo español, así que la pantalla se lee
    # igual que ayer. Cambió la GRAFÍA del valor, no el cableado.
    #
    # Se acepta el literal Y la forma envuelta, y de paso se APRIETA: antes
    # bastaba con que la subcadena `label: 'Privacidad'` apareciera en
    # CUALQUIER punto del fichero (un título suelto la habría satisfecho con la
    # fila del nav borrada). Ahora el rótulo tiene que estar dentro del MISMO
    # objeto que declara `id: 'privacy'`.
    fila = re.search(r"\{[^{}]*id:\s*'privacy'[^{}]*\}", src)
    assert fila, (
        "Falta la entry con `id: 'privacy'` en sectionsConfig (nav de Settings)."
    )
    assert re.search(
        r"label:\s*(?:'Privacidad'|t\(\s*'Privacidad'\s*\))", fila.group(0)
    ), (
        "La entry `id: 'privacy'` de sectionsConfig perdió su rótulo "
        f"'Privacidad' (ni literal ni via t()): {fila.group(0)[:160]!r}"
    )
    assert "activeSection === 'privacy'" in src, (
        "Falta el render condicional de la sección Privacidad."
    )
    assert "/api/account/export" in src, (
        "El botón Exportar datos debe llamar a GET /api/account/export."
    )
    # Enlaces de políticas del apex.
    for path in ("/data-protection", "/ai-policy"):
        assert path in src, f"Falta el enlace de política {path} en la sección."


def test_frontend_analytics_opt_out_real():
    """El toggle "Ayuda a mejorar MealfitRD" debe ser un opt-out REAL:
    trackEvent gatea en isAnalyticsOptedOut (no un switch decorativo)."""
    analytics = _read(_REPO_ROOT / "frontend" / "src" / "utils" / "analytics.js")
    assert "ANALYTICS_OPT_OUT_KEY" in analytics
    assert re.search(r"if\s*\(\s*isAnalyticsOptedOut\(\)\s*\)\s*return", analytics), (
        "trackEvent debe cortar en isAnalyticsOptedOut() — sin el gate, el "
        "toggle de Privacidad sería un ajuste falso."
    )
    settings = _read(_SETTINGS)
    # [P1-LANDING-OBS-PAPER · 2026-08-14] Esta aserción exigía ver
    # `ANALYTICS_OPT_OUT_KEY` DENTRO de Settings.jsx, o sea que la pantalla
    # escribiera la clave a pelo. Estaba atada al CÓMO, y el cómo era el bug:
    # `localStorage` es por ORIGEN, así que el opt-out escrito aquí (app.*) era
    # invisible para el landing (apex) y el usuario seguía siendo rastreado
    # después de apagarlo. Ahora se escribe por `persistAnalyticsOptOut`, que
    # deja coherentes los dos soportes (localStorage + cookie de dominio).
    #
    # Reescrita al QUÉ —«el toggle persiste el opt-out de verdad»— y además
    # endurecida: prohíbe explícitamente volver a la escritura cruda, que es la
    # regresión concreta a evitar.
    assert "handleToggleAnalytics" in settings, (
        "Falta el toggle de analytics en la sección Privacidad."
    )
    assert "persistAnalyticsOptOut" in settings, (
        "El toggle debe persistir por `persistAnalyticsOptOut` (SSOT). Escribir "
        "sólo `localStorage` deja el opt-out invisible para el apex, que es otro "
        "origen — el usuario lo apaga y el landing lo sigue rastreando."
    )
    assert not re.search(r"safeLocalStorageSet\(\s*ANALYTICS_OPT_OUT_KEY", settings), (
        "Volvió la escritura cruda de la clave del opt-out en Settings.jsx."
    )
