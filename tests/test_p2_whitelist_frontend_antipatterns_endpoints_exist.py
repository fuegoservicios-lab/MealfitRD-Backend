"""[P2-WHITELIST-AUDIT · 2026-05-12] Test parser-based que ancla la tabla
"Anti-patrones de frontend prohibidos" del CLAUDE.md a los endpoints
backend que sirven de reemplazo.

Anchor: P2-WHITELIST-AUDIT-FRONTEND-ENDPOINTS.

CLAUDE.md sección "Anti-patrones de frontend prohibidos" enumera operaciones
prohibidas (escrituras directas desde el cliente a `meal_plans`) con su
endpoint backend reemplazo. Si alguien renombra/borra un endpoint sin
actualizar la docs, el frontend que migró a la versión backend deja de
funcionar y nadie sabe dónde mirar.

Este test escanea `backend/routers/plans.py` (y eventualmente otros routers)
buscando los decoradores `@router.<method>("...path...")` correspondientes
a cada antipattern documentado.

Cada entry de la tupla es `(method, path_suffix, p_fix_anchor)`:
  - `method`: get/post/patch/delete.
  - `path_suffix`: el path completo SIN prefijo del APIRouter (e.g.
    `/{plan_id}/swap-meal/persist`, no `/api/plans/...`).
  - `p_fix_anchor`: P-fix que introdujo el endpoint (audit trail).
"""
import re
from pathlib import Path


_ROUTERS_DIR = Path(__file__).resolve().parents[1] / "routers"


# Tabla canónica extraída de CLAUDE.md "Operaciones prohibidas y sus reemplazos".
# Si CLAUDE.md añade/elimina un antipattern, sincronizá acá.
_REQUIRED_ENDPOINTS = (
    # (method, path_suffix, p_fix_anchor)
    ("post",   r"/\{plan_id\}/swap-meal/persist",      "P0-NEW-A"),
    ("post",   r"/\{plan_id\}/grocery-start-date",     "P0-NEW-B"),
    ("patch",  r"/\{plan_id\}/name",                   "P1-HIST-5"),
    ("post",   r"/\{plan_id\}/restore-local",          "P1-OPEN-1"),
    ("delete", r"/\{plan_id\}",                        "P0-HIST-1"),
    ("post",   r"/recipe/expand",                      "P1-HIST-RECIPE-1"),
    ("post",   r"/restore",                            "P0-HIST-1"),
)


def _load_router_sources() -> str:
    """Concatena todos los .py de `backend/routers/` para escanear de una vez."""
    assert _ROUTERS_DIR.exists(), (
        f"Directorio backend/routers/ no encontrado en {_ROUTERS_DIR}"
    )
    pieces = []
    for f in sorted(_ROUTERS_DIR.glob("*.py")):
        if f.name == "__init__.py":
            continue
        pieces.append(f"### FILE: {f.name}\n")
        pieces.append(f.read_text(encoding="utf-8", errors="ignore"))
        pieces.append("\n")
    return "\n".join(pieces)


def test_frontend_antipattern_replacement_endpoints_exist():
    """Cada antipattern documentado en CLAUDE.md DEBE tener su endpoint
    backend reemplazo registrado en algún router. Si alguno falta, el
    frontend que se apoyó en el contrato se rompe sin warning.
    """
    src = _load_router_sources()
    missing = []
    for method, path_pattern, anchor in _REQUIRED_ENDPOINTS:
        # Decorator literal: `@router.<method>("<path>"...)`. Regex permisivo
        # con quotes simples o dobles + posibles whitespace + posible trailing arg.
        regex = (
            rf'@router\.{re.escape(method)}\(\s*["\']'
            + path_pattern
            + r'["\']'
        )
        if not re.search(regex, src):
            missing.append(f"{method.upper()} {path_pattern} (anchor {anchor})")

    assert not missing, (
        f"Endpoints reemplazo de antipatterns frontend AUSENTES en routers/: "
        f"{missing}. CLAUDE.md sección 'Anti-patrones de frontend prohibidos' "
        f"los lista como contrato pero ningún @router.<method>() coincide. "
        f"Si renombraste un endpoint, actualizá CLAUDE.md + el .jsx que lo "
        f"invoca + este test. Si lo borraste deliberadamente, removelo de "
        f"la tabla docs primero."
    )


def test_p2_whitelist_audit_frontend_anchor_present():
    """El marker P2-WHITELIST-AUDIT-FRONTEND-ENDPOINTS debe vivir en este
    archivo para que el cross-link slug del test legacy P2-HIST-AUDIT-14 lo
    encuentre y el bump del marker sea verificable."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-WHITELIST-AUDIT" in src
    assert "P2-WHITELIST-AUDIT-FRONTEND-ENDPOINTS" in src


def test_routers_dir_has_expected_files():
    """Sanity: el directorio backend/routers/ debe contener al menos plans.py.
    Si la estructura cambia drásticamente (e.g., consolidación a un solo file),
    este test guía al refactor a actualizar también las rutas.
    """
    files = {f.name for f in _ROUTERS_DIR.glob("*.py")}
    assert "plans.py" in files, (
        f"backend/routers/plans.py no existe. Files presentes: {sorted(files)}. "
        f"Si reorganizaste los routers, actualizá _ROUTERS_DIR + la convención."
    )
