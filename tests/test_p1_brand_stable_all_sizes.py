"""[P1-BRAND-STABLE-ALL-SIZES + P2-SHOPLIST-AUTO-REFRESH · 2026-07-06]

Feedback del owner (2 pedidos):
1. Los alimentos DURADEROS (estables de despensa) deben enseñar TODAS las marcas
   y tamaños del catálogo para seleccionar ("hay personas que compran 50 lb de
   arroz si quieren, o 2 lb") — para "Arroz blanco" el picker enseñaba solo 2
   marcas (filtro ±15% al tamaño de la lista). Ahora estables = catálogo completo
   con los del tamaño de tu lista PRIMERO y luego precio asc; el filtro por
   tamaño queda solo para frescos/perecederos.
2. La lista de compras solo se actualizaba con el truco manual 30→15→30 días.
   Ahora el Dashboard dispara un recalc SILENCIOSO al cargar (una vez por plan,
   preserve_restock, fail-open) — cambios server-side (marcas default, precios
   vivos, fixes de costeo) aparecen sin tocar nada.
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
BRANDS_JSX = (BACKEND.parent / "frontend" / "src" / "components" / "dashboard"
              / "SupermarketBrands.jsx").read_text(encoding="utf-8")
DASH_JSX = (BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


# ─────────────── 1. estables: catálogo completo, tu tamaño primero ───────────────

def test_stable_items_show_full_catalog():
    assert "P1-BRAND-STABLE-ALL-SIZES" in BRANDS_JSX
    assert "MAX_STABLE_SHOWN" in BRANDS_JSX and "stableSortedVariants" in BRANDS_JSX
    assert "is_perishable === false" in BRANDS_JSX, (
        "duraderos = flag SSOT is_perishable del backend (no heurística de nombre)"
    )
    m = re.search(r"const MAX_STABLE_SHOWN = (\d+)", BRANDS_JSX)
    assert m and int(m.group(1)) >= 40, (
        "el cap de estables debe cubrir catálogos grandes (arroz = 44 variantes)"
    )


def test_stable_sort_your_size_first_then_price():
    i = BRANDS_JSX.index("const stableSortedVariants")
    body = BRANDS_JSX[i:i + 700]
    assert "matchesSize" in body and "price_rd ?? Infinity" in body, (
        "orden de estables: tamaño de tu lista primero, luego más económica"
    )


def test_stable_branch_bypasses_size_filter():
    i = BRANDS_JSX.index("const isStable = stableByKey")
    win = BRANDS_JSX[i:i + 900]
    assert "if (isStable)" in win and "stableSortedVariants(g.variants" in win
    assert "sizeFilteredVariants" in win, (
        "los NO-estables (frescos) conservan el filtro ±15% (P1-BRAND-SIZE-FILTER)"
    )


# ─────────────── 2. auto-refresh silencioso de la lista ───────────────

def test_dashboard_auto_refreshes_shopping_list():
    assert "P2-SHOPLIST-AUTO-REFRESH" in DASH_JSX
    i = DASH_JSX.index("P2-SHOPLIST-AUTO-REFRESH")
    # Ventana = hasta el cierre del useEffect (su dependency array), no un
    # número fijo de chars (el código siguiente contiene toasts legítimos).
    end = DASH_JSX.index("}, [isGuest, userProfile?.id, planData?.id", i)
    win = DASH_JSX[i:end]
    assert "_shopAutoRefreshRef" in win, "guard una-vez-por-plan (no loop de recalcs)"
    assert "recalculate-shopping-list" in win, "endpoint canónico (cero costo LLM)"
    assert "preserve_restock: true" in win, "el restock del usuario jamás se clobberea"
    assert "isGuest" in win and "isPlanExpired" in win, "gates: guests/expirados fuera"
    # Llamadas reales (toast./toast(), no la palabra en comentarios).
    assert "toast." not in win and "toast(" not in win, (
        "SILENCIOSO — sin toast (fail-open si falla)"
    )


def test_auto_refresh_guarded_once_per_plan():
    i = DASH_JSX.index("_shopAutoRefreshRef.current === planData.id")
    assert i > 0, "sin el guard por plan.id el efecto re-dispararía en cada setPlanData"
