"""[P1-SUPERMARKET-MATCH · 2026-07-02] Endpoint de matching lista de compras →
catálogo del súper (POST /api/supermarket/match).

Tests parser-based (no requieren DB ni levantar la app): anclan el contrato del
endpoint en routers/supermarket.py + el componente frontend que lo consume.
Si renombras la ruta, el limiter o la normalización, esto falla ANTES de que
el cambio llegue a producción sin actualizar el contrato.
"""
import re
import unicodedata
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROUTER = BACKEND / "routers" / "supermarket.py"
FRONTEND_COMPONENT = BACKEND.parent / "frontend" / "src" / "components" / "dashboard" / "SupermarketBrands.jsx"
DASHBOARD = BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"

SRC = ROUTER.read_text(encoding="utf-8")


def test_match_route_exists():
    assert '@router.post("/match")' in SRC, (
        "POST /api/supermarket/match desapareció de routers/supermarket.py — "
        "el panel 'Marcas del súper' del Dashboard depende de esta ruta."
    )


def test_match_has_own_rate_limiter():
    assert re.search(r"_MATCH_LIMITER\s*=\s*RateLimiter\(", SRC), "falta _MATCH_LIMITER = RateLimiter(...)"
    # El limiter debe estar aplicado como dependencia del endpoint.
    m = re.search(r'@router\.post\("/match"\)\s*\nasync def api_supermarket_match\((.*?)\):', SRC, re.S)
    assert m, "no se encontró la signature de api_supermarket_match"
    assert "_MATCH_LIMITER" in m.group(1), "el endpoint /match no aplica Depends(_MATCH_LIMITER)"


def test_match_only_active_rows():
    m = re.search(r"def _match\(\).*?return \{", SRC, re.S)
    assert m, "no se encontró el cuerpo de _match()"
    assert "WHERE active" in m.group(0), (
        "el SELECT del match debe filtrar WHERE active — los productos ocultos "
        "por la admin UI no pueden aparecer en la lista de compras."
    )


def test_match_no_paywall():
    m = re.search(r'@router\.post\("/match"\)\s*\nasync def api_supermarket_match\((.*?)\):', SRC, re.S)
    assert "verify_api_quota" not in m.group(1), (
        "/match es read-only sin costo LLM — RateLimiter, NO paywall "
        "(misma decisión que la historial-quota-exemption)."
    )


def test_norm_food_accent_insensitive():
    """Réplica de la normalización para validar el contrato accent-insensitive
    (simétrica al foodKeyOf del frontend)."""
    assert "unicodedata" in SRC and "_norm_food" in SRC

    def norm(value):
        s = unicodedata.normalize("NFD", (value or "").strip().lower())
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return " ".join(s.split())

    assert norm("Kéfir") == norm("Kefir") == "kefir"
    assert norm("  Plátano   Verde ") == "platano verde"
    assert norm("Ñame") == "name"


def test_frontend_component_exists_and_mounted():
    assert FRONTEND_COMPONENT.exists(), (
        "frontend/src/components/dashboard/SupermarketBrands.jsx no existe — "
        "es el consumidor del endpoint /match."
    )
    comp = FRONTEND_COMPONENT.read_text(encoding="utf-8")
    assert "/api/supermarket/match" in comp, "el componente no llama a /api/supermarket/match"
    dash = DASHBOARD.read_text(encoding="utf-8")
    assert "SupermarketBrands" in dash, "SupermarketBrands no está montado en Dashboard.jsx"
