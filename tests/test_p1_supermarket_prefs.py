"""[P1-SUPERMARKET-PREFS · 2026-07-02] Marca preferida por usuario (fase 2 de la
conexión lista de compras ↔ Supermercado RD).

Parser-based (sin DB): ancla el contrato de la tabla `user_brand_preferences`
(migración en AMBOS dirs SSOT), los endpoints GET/PUT /api/supermarket/preferences
(auth user-scoped + filtro I2) y el consumo desde SupermarketBrands.jsx
(persistencia server para autenticados + fallback localStorage para guests).
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
ROUTER = BACKEND / "routers" / "supermarket.py"
COMPONENT = ROOT / "frontend" / "src" / "components" / "dashboard" / "SupermarketBrands.jsx"
MIGRATION_NAME = "p1_supermarket_prefs_2026_07_02.sql"

SRC = ROUTER.read_text(encoding="utf-8")


def test_migration_in_both_ssot_dirs():
    """P3-MIGRATIONS-SSOT: la migración vive en migrations/ Y backend/migrations/
    con contenido idéntico."""
    root_mig = ROOT / "migrations" / MIGRATION_NAME
    backend_mig = BACKEND / "migrations" / MIGRATION_NAME
    assert root_mig.exists(), f"falta {root_mig}"
    assert backend_mig.exists(), f"falta {backend_mig}"
    assert root_mig.read_text(encoding="utf-8") == backend_mig.read_text(encoding="utf-8"), (
        "drift entre migrations/ y backend/migrations/ — deben ser idénticos (P3-MIGRATIONS-SSOT)."
    )
    sql = root_mig.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.user_brand_preferences" in sql
    assert "REFERENCES public.user_profiles(id) ON DELETE CASCADE" in sql
    assert "REFERENCES public.supermarket_products(id) ON DELETE CASCADE" in sql
    assert "idx_user_brand_preferences_product" in sql, "falta el índice que cubre la FK (P2-PERF-1)"
    assert re.search(r"DO \$\$.*RAISE EXCEPTION", sql, re.DOTALL), "falta el DO $$ sanity"


def test_endpoints_exist_with_user_auth():
    assert '@router.get("/preferences")' in SRC, "falta GET /api/supermarket/preferences"
    assert '@router.put("/preferences")' in SRC, "falta PUT /api/supermarket/preferences"
    for route in ('@router.get("/preferences")', '@router.put("/preferences")'):
        start = SRC.index(route)
        body = SRC[start:start + 2500]
        assert "Depends(get_verified_user_id)" in body, (
            f"{route} sin get_verified_user_id — abriría lectura/escritura anónima de preferencias."
        )
        assert "_PREFS_LIMITER" in body, f"{route} sin RateLimiter propio"
        assert "verify_api_quota" not in body, (
            "preferencias = cero costo LLM → RateLimiter, NO paywall (historial-quota-exemption)."
        )


def test_queries_filter_by_user_id():
    """Invariante I2: toda query sobre user_brand_preferences ancla user_id."""
    for m in re.finditer(r"user_brand_preferences", SRC):
        window = SRC[max(0, m.start() - 400):m.start() + 400]
        assert re.search(r"user_id\s*=\s*%s|\(user_id, food_key", window), (
            "query sobre user_brand_preferences sin filtro/inserción anclada a user_id (I2)."
        )


def test_put_validates_active_product():
    start = SRC.index('@router.put("/preferences")')
    body = SRC[start:start + 3000]
    assert re.search(r"supermarket_products\s+WHERE\s+id\s*=\s*%s::uuid\s+AND\s+active", body), (
        "PUT debe validar que el product_id exista Y esté active — un producto "
        "oculto por la admin UI no puede quedar como preferencia."
    )


def test_frontend_consumes_prefs_with_guest_fallback():
    comp = COMPONENT.read_text(encoding="utf-8")
    assert "/api/supermarket/preferences" in comp, "el componente no usa /preferences"
    assert "fetchWithAuth" in comp, "la persistencia server debe ir autenticada (fetchWithAuth)"
    assert "localStorage" in comp.replace("safeLocalStorage", "localStorage"), (
        "falta el fallback localStorage para invitados/errores de auth."
    )
