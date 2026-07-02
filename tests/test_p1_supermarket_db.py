"""[P1-SUPERMARKET-DB · 2026-07-02] Supermercado RD artificial — test ancla.

Nació como anchor MÍNIMO durante la reparación del race de OneDrive (el app.py
con include_router entró al commit P1-AUDIT-V3-BATCH mientras
`routers/supermarket.py` seguía untracked → origin/main no-importable).
Extendido el mismo día con el contrato completo del feature:

1. Import + prefix del router (el modo de fallo original).
2. Migración SSOT en AMBOS dirs (P3-MIGRATIONS-SSOT), idéntica e idempotente
   (IF NOT EXISTS + DO $$ sanity + unique index de variante).
3. TODA mutación (POST/PATCH/DELETE) gateada por `_verify_admin_token` +
   `_check_admin_rate_limit`; el GET público NO usa `verify_api_quota`
   (página de marketing, cero costo LLM) pero SÍ RateLimiter per-IP.
4. Frontend: ruta /supermercado, link en Footer, página que consume SOLO
   endpoints backend (simétrica a I6 — cero DB directa desde el cliente) y
   manda el token admin como Bearer.
5. Seed dry-run default + gate --commit + dataset completo (+200 filas),
   idempotente (ON CONFLICT DO NOTHING — no pisa ediciones de la admin UI).

Tooltip-anchor: P1-SUPERMARKET-DB-START
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g  # asegura sys.path del backend

_BACKEND = Path(g.__file__).resolve().parent
_REPO_ROOT = _BACKEND.parent

_MIGRATION_NAME = "p1_supermarket_db_2026_07_02.sql"
_ROUTER = _BACKEND / "routers" / "supermarket.py"
_SEED = _BACKEND / "scripts" / "seed_supermarket_2026_07_02.py"

_FRONTEND = _REPO_ROOT / "frontend" / "src"


# ---------------------------------------------------------------------------
# 1. Import + wiring (modo de fallo original: main no-importable)
# ---------------------------------------------------------------------------
def test_router_imports_and_prefix():
    from routers.supermarket import router
    assert router.prefix == "/api/supermarket"


def test_marker_wired_in_app_and_router():
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    r_src = _ROUTER.read_text(encoding="utf-8")
    assert "P1-SUPERMARKET-DB" in app_src
    assert "from routers.supermarket import router" in app_src
    assert "app.include_router(supermarket_router)" in app_src
    assert "router = APIRouter" in r_src


# ---------------------------------------------------------------------------
# 2. Migración SSOT dual-dir, idempotente
# ---------------------------------------------------------------------------
def test_migration_ssot_both_dirs_and_identical():
    backend_copy = _BACKEND / "migrations" / _MIGRATION_NAME
    root_copy = _REPO_ROOT / "migrations" / _MIGRATION_NAME
    assert backend_copy.exists(), (
        "P1-SUPERMARKET-DB violation: falta la migración en backend/migrations/ "
        "(P3-MIGRATIONS-SSOT)."
    )
    assert root_copy.exists(), (
        "P1-SUPERMARKET-DB violation: falta la migración en migrations/ del "
        "workspace root (P3-MIGRATIONS-SSOT)."
    )
    assert backend_copy.read_text(encoding="utf-8") == root_copy.read_text(encoding="utf-8"), (
        "P1-SUPERMARKET-DB violation: las dos copias de la migración divergieron — "
        "sincronizarlas (deben ser idénticas)."
    )


def test_migration_idempotent_with_sanity():
    source = (_BACKEND / "migrations" / _MIGRATION_NAME).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.supermarket_products" in source, (
        "P1-SUPERMARKET-DB violation: CREATE TABLE debe usar IF NOT EXISTS "
        "(P3-MIGRATION-IDEMPOTENCE-DOC)."
    )
    assert "uq_supermarket_products_variant" in source, (
        "P1-SUPERMARKET-DB violation: falta el unique index de variante "
        "(alimento+marca+presentación). Sin él, el seed pierde idempotencia y el "
        "POST no puede detectar duplicados con 409."
    )
    assert re.search(r"DO \$\$.*RAISE EXCEPTION", source, re.DOTALL), (
        "P1-SUPERMARKET-DB violation: falta el DO $$ sanity check con RAISE EXCEPTION."
    )


# ---------------------------------------------------------------------------
# 3. Gate admin en mutaciones; GET público sin paywall
# ---------------------------------------------------------------------------
def _handler_bodies(source: str) -> dict:
    """{(verbo, offset) → cuerpo} por handler @router.<verb>; cada cuerpo corta
    en el siguiente decorador (o EOF)."""
    matches = list(re.finditer(r"@router\.(get|post|patch|delete)\(", source))
    bodies = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        bodies[(m.group(1), m.start())] = source[m.start():end]
    return bodies


def test_all_mutations_require_admin_token():
    source = _ROUTER.read_text(encoding="utf-8")
    bodies = _handler_bodies(source)
    mutating = {(verb, off): body for (verb, off), body in bodies.items()
                if verb in ("post", "patch", "delete")}
    assert len(mutating) >= 3, (
        "P1-SUPERMARKET-DB violation: se esperaban al menos POST+PATCH+DELETE."
    )
    for (verb, _), body in mutating.items():
        assert "_verify_admin_token(" in body, (
            f"P1-SUPERMARKET-DB violation: el handler {verb.upper()} NO llama "
            "_verify_admin_token — abriría escritura pública a supermarket_products."
        )
        assert "_check_admin_rate_limit(" in body, (
            f"P1-SUPERMARKET-DB violation: el handler {verb.upper()} debe aplicar "
            "_check_admin_rate_limit tras el gate (P2-ADMIN-RATE-LIMIT)."
        )


def test_public_get_rate_limited_without_quota():
    source = _ROUTER.read_text(encoding="utf-8")
    assert any(verb == "get" for (verb, _) in _handler_bodies(source)), (
        "P1-SUPERMARKET-DB violation: falta el GET público del listado."
    )
    # Uso real (import o call) — la mención en docstring/comentarios es legítima.
    assert not re.search(r"import\s+.*verify_api_quota|verify_api_quota\s*[(,)]", source), (
        "P1-SUPERMARKET-DB violation: el paywall verify_api_quota NO aplica a la "
        "página pública de marketing (misma razón que la historial-quota-exemption)."
    )
    assert "RateLimiter(" in source, (
        "P1-SUPERMARKET-DB violation: el GET público debe llevar RateLimiter per-IP."
    )


# ---------------------------------------------------------------------------
# 4. Frontend: ruta + footer + página backend-only
# ---------------------------------------------------------------------------
def test_frontend_route_and_footer_link():
    app_jsx = (_FRONTEND / "App.jsx").read_text(encoding="utf-8")
    assert 'path="/supermercado"' in app_jsx, (
        "P1-SUPERMARKET-DB violation: App.jsx perdió la ruta /supermercado."
    )
    footer = (_FRONTEND / "components" / "layout" / "Footer.jsx").read_text(encoding="utf-8")
    assert "/supermercado" in footer, (
        "P1-SUPERMARKET-DB violation: Footer.jsx perdió la entrada 'Supermercado RD'."
    )


def test_frontend_page_backend_endpoints_only():
    page_path = _FRONTEND / "pages" / "SupermarketPage.jsx"
    assert page_path.exists(), "P1-SUPERMARKET-DB violation: falta SupermarketPage.jsx."
    page = page_path.read_text(encoding="utf-8")
    assert "/api/supermarket/products" in page, (
        "P1-SUPERMARKET-DB violation: la página debe consumir /api/supermarket/products."
    )
    for forbidden in ("supabase.from(", "postgrest"):
        assert forbidden not in page, (
            f"P1-SUPERMARKET-DB violation: acceso DB directo prohibido ({forbidden}) — "
            "toda mutación va por el backend (simétrica a I6)."
        )
    assert "Authorization" in page and "Bearer" in page, (
        "P1-SUPERMARKET-DB violation: el modo edición debe mandar el token admin "
        "como Bearer (lo consume _verify_admin_token)."
    )


# ---------------------------------------------------------------------------
# 5. Seed: gates + dataset completo
# ---------------------------------------------------------------------------
def test_seed_script_gated_idempotent_and_complete():
    assert _SEED.exists(), (
        "P1-SUPERMARKET-DB violation: falta scripts/seed_supermarket_2026_07_02.py."
    )
    seed = _SEED.read_text(encoding="utf-8")
    assert 'COMMIT = "--commit" in sys.argv' in seed, (
        "P1-SUPERMARKET-DB violation: el seed debe ser dry-run por default con gate "
        "--commit (patrón add_foods_batch)."
    )
    assert "ON CONFLICT" in seed and "DO NOTHING" in seed, (
        "P1-SUPERMARKET-DB violation: el seed debe ser idempotente "
        "(ON CONFLICT DO NOTHING) para no pisar ediciones de la admin UI."
    )
    n_rows = len(re.findall(r'^\s*\("', seed, re.MULTILINE))
    assert n_rows >= 200, (
        f"P1-SUPERMARKET-DB violation: el seed tiene {n_rows} filas; el dataset "
        "verificado del owner tiene +200 presentaciones. ¿Se truncó la transcripción?"
    )
