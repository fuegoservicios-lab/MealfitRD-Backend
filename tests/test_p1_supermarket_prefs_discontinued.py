"""[P1-SUPERMARKET-PREFS-DISCONTINUED · 2026-07-29] Test ancla (fix round, finding 2
— lens lists). `GET /api/supermarket/preferences` devolvía la fila de una preferencia
aunque el producto elegido hubiese sido dado de baja (`sp.active = false`) por la
admin UI, y ninguna capa avisaba al usuario ni limpiaba la fila huérfana de
`user_brand_preferences`. Parser-based (mismo patrón que test_p1_supermarket_prefs.py,
sin DB real): ancla que el endpoint (a) separa activas/inactivas, (b) excluye las
inactivas de `preferences`, (c) las reporta en `discontinued`, (d) intenta borrarlas
best-effort filtrado por `user_id` (I2).
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
ROUTER = BACKEND / "routers" / "supermarket.py"
COMPONENT = ROOT / "frontend" / "src" / "components" / "dashboard" / "SupermarketBrands.jsx"

SRC = ROUTER.read_text(encoding="utf-8")


def _get_prefs_body():
    start = SRC.index('@router.get("/preferences")')
    end = SRC.index('@router.put("/preferences")')
    return SRC[start:end]


def test_marker_anchored():
    assert "P1-SUPERMARKET-PREFS-DISCONTINUED" in SRC


def test_separates_active_and_stale_rows():
    body = _get_prefs_body()
    assert re.search(r"active_rows\s*=\s*\[.*if r\.get\(\"active\"\)\]", body), (
        "el GET debe separar filas activas explícitamente"
    )
    assert re.search(r"stale_rows\s*=\s*\[.*if not r\.get\(\"active\"\)\]", body), (
        "el GET debe separar filas con producto discontinuado (active=false)"
    )


def test_preferences_only_contains_active_rows():
    body = _get_prefs_body()
    assert '"preferences": {r["food_key"]: r for r in active_rows}' in body, (
        "`preferences` debe construirse SOLO de las filas activas — un pin muerto "
        "no puede seguir apareciendo como la marca elegida."
    )


def test_discontinued_key_present_in_response():
    body = _get_prefs_body()
    assert '"discontinued"' in body, "falta el campo `discontinued` en la respuesta"


def test_stale_delete_filters_by_user_id():
    """Invariante I2: el DELETE de filas obsoletas ancla user_id."""
    body = _get_prefs_body()
    idx = body.index("DELETE FROM public.user_brand_preferences")
    window = body[max(0, idx - 200):idx + 300]
    assert re.search(r"user_id\s*=\s*%s", window), (
        "el DELETE de preferencias obsoletas no filtra por user_id (I2)"
    )
    assert "food_key = ANY(%s)" in window


def test_delete_is_best_effort_does_not_break_response():
    """Un fallo en el DELETE de limpieza NO debe convertirse en 500 — el usuario
    sigue viendo su respuesta correcta (preferences sin la fila muerta)."""
    body = _get_prefs_body()
    del_idx = body.index("DELETE FROM public.user_brand_preferences")
    try_idx = body.rindex("try:", 0, del_idx)
    except_window = body[del_idx:del_idx + 600]
    assert try_idx < del_idx, "el DELETE debe estar envuelto en try/except propio"
    assert "except Exception" in except_window, "el DELETE debe ser best-effort (fail-open)"


def test_frontend_consumes_discontinued_field():
    comp = COMPONENT.read_text(encoding="utf-8")
    assert "discontinued" in comp, (
        "SupermarketBrands.jsx debe leer `discontinued` de la respuesta del GET y "
        "avisar al usuario (antes: el pin muerto desaparecía en silencio)."
    )
