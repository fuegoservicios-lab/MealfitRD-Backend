"""[P2-BRANDS-CANONICAL-SOURCE + P2-RECALC-STALE-FLAGS-ORDER · 2026-07-06]

Owner: "¿por qué el menú del supermercado desaparece?" (post-restock total).
Forensic (plan ff673061): weekly=48, monthly=0, main=0.

Dos bugs encadenados:
1. Las listas biweekly/monthly son HÍBRIDAS y filtran lo ya comprado en el ciclo
   (restocked_items) — restock total ⇒ activa=0 ⇒ el panel de marcas (que leía la
   lista activa) desaparecía. El GESTOR de marcas debe leer la CANÓNICA semanal
   (necesidades completas), comprado o no.
2. Orden del self-heal: con la Nevera VACÍA los flags de restock heredados son
   stale (P3-RESTOCK-STALE-RECALC-HEAL los limpia en el callback de persist) —
   pero los híbridos ya se habían filtrado con esos flags muertos ⇒ recalc
   persistía flags limpios + monthly/biweekly=0. El filtro ahora se anula ANTES
   cuando `_inv_count_at_recalc == 0`.
3. Bonus: el gate del auto-refresh saltaba listas vacías ⇒ bloqueaba su propio
   self-heal. Ahora gatea por `days` (la fuente real del recalc).
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND = os.path.join(os.path.dirname(_BACKEND), "frontend")


def _read(*parts) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


_PL = _read(_BACKEND, "routers", "plans.py")
_DASH = _read(_FRONTEND, "src", "pages", "Dashboard.jsx")


# ───────────── backend: flags stale se anulan ANTES de los híbridos ─────────────

def test_stale_flags_neutralized_before_hybrid():
    i = _PL.index("P2-RECALC-STALE-FLAGS-ORDER")
    win = _PL[i:i + 1400]
    assert "_inv_count_at_recalc == 0" in win, "invariante: nevera vacía ⇒ flags stale"
    assert "_restocked_items = None" in win and "_restocked_at = None" in win
    # y ocurre ANTES de construir los híbridos:
    j = _PL.index("_build_hybrid(scaled_7, scaled_15")
    assert i < j, "la anulación debe preceder a _build_hybrid (ahí estaba el bug de orden)"


# ───────────── frontend: panel de marcas lee la canónica ─────────────

def test_brands_panel_reads_weekly_canonical():
    assert "P2-BRANDS-CANONICAL-SOURCE" in _DASH
    i = _DASH.index("const brandsPanelList")
    win = _DASH[i:i + 700]
    assert "aggregated_shopping_list_weekly" in win, (
        "la canónica semanal es las necesidades COMPLETAS del plan — no se vacía al comprar"
    )
    assert "shoppingList={brandsPanelList}" in _DASH, "el panel consume la canónica"
    assert "brandsPanelList.length > 0" in _DASH, "el gate del panel también"


def test_auto_refresh_gates_on_days_not_list():
    i = _DASH.index("P2-SHOPLIST-AUTO-REFRESH")
    win = _DASH[i:_DASH.index("}, [isGuest, userProfile?.id, planData?.id", i)]
    assert "planData?.days" in win and "planData.days.length === 0" in win, (
        "gate por days — el gate por lista vacía bloqueaba el self-heal post-restock"
    )
    assert "aggregated_shopping_list.length === 0) return" not in win
