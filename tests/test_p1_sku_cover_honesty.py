"""[P1-SKU-COVER-HONESTY · 2026-08-02] El selector de envases permitía under-buy hasta ~33-50% en
silencio.

La regla floor `under_buy_g < over_buy_g` (⟺ frac<0.5) vivía en 3 sitios — `apply_smart_market_units`
standard path, `_find_best_sku`, `_select_market_package` (donde min-costo además FAVORECÍA el
floor) — y permitía comprar 1 envase cuando se necesitaban 1.48 (68% de cobertura). `pkg_cover_ratio`
se calcula y persiste pero nadie lo consumía para avisar (solo sobre-cobertura ≥2×,
P1-OVERCOVER-LABEL) y `_item_cycle_repurchases` clampaba `min(plano,...)` ⇒ el costo tampoco
declaraba la recompra real. Medido en prod: 18/22 planes con ≥1 ítem cover<0.9 sin nota (arroz
0.69-0.81, aceite 0.70-0.93, camarones 0.76-0.79).

Fix: el floor ya NO se retiene si el under-buy absoluto excede `SKU_FLOOR_MAX_UNDER_PCT` (10% del
total, knob `MEALFIT_SKU_FLOOR_MAX_UNDER_PCT`) — la rama `frac <= ANTI_WASTE_THRESHOLD` (colchón de
coma flotante, 2%) se conserva intacta, es política anti-desperdicio deliberada. Cuando el envase
elegido cubre <90% del ciclo (`PKG_COVER_NOTE_MIN`, knob `MEALFIT_PKG_COVER_NOTE_MIN`) y el ítem no
tiene ya un aviso de cap (`capped_by`, P1-CAPPED-STAPLE-HONESTY manda si ya existe), se sufija
"alcanza ~N de 7 días — recompra" (mismo formato que P1-CAPPED-STAPLE-HONESTY, unidad de
`trip_days=7` porque `pkg_cover_ratio` se mide en idas de 7 días, no en el ciclo completo).

⚠️ **Hallazgo aritmético (reportado, no forzado un test artificial)**: el caso propuesto originalmente
para probar la nota (arroz 647,5 g / funda 453,6 g, cover 0.70) YA NO sirve tras el fix: con el nuevo
bound, `under_buy=193.9 > 647.5×0.10=64.75` ⇒ el algoritmo ahora compra 2 fundas y la cobertura SUBE
a 1.40 (ver `test_arroz_audit_pasa_a_ceil` abajo). Y construir un caso que SÍ elija floor vía la rama
`frac <= ANTI_WASTE_THRESHOLD` (2%) es matemáticamente imposible que resulte en cover<0.9: con
floor_units=1 y frac<=0.02, `cover = floor_units/(floor_units+frac) >= 1/1.02 = 0.9804` — nunca baja
de 0.98, mucho menos de 0.9. Y vía la rama NUEVA (el bound), por construcción
`cover = 1 - under_buy/g_total >= 1 - SKU_FLOOR_MAX_UNDER_PCT = 0.90` (con el default 0.10) — nunca
ESTRICTAMENTE <0.9. Bajo los knobs default, la nota es inalcanzable para ítems recién resueltos por
estos 3 sitios (solo dispara para ítems `capped_by`, que la excluyen por diseño #3, o si un operador
sube `MEALFIT_SKU_FLOOR_MAX_UNDER_PCT` por encima del default). El test de la nota de abajo
(`test_cover_bajo_lleva_nota_alcanza`) por tanto exercita el mecanismo vía `monkeypatch` del knob al
tope del clamp (0.5) — una configuración real y soportada, no un dict manufacturado — reusando los
NÚMEROS REALES del audit (arroz 647.5g/453.6g) para mostrar que, si se afloja el bound, la nota SÍ
avisa correctamente.
"""
import pytest

import shopping_calculator as sc

CARTON = {"market_container": "cartón", "container_weight_g": 946.0}
FUNDA_ARROZ = {"market_container": "funda", "container_weight_g": 453.6}


def _por_gramos(name, grams, master_item):
    """Adapta la firma real `apply_smart_market_units(name, weight_in_lbs, unit_str, raw_qty,
    master_item)` a una necesidad expresada en gramos (el brief original asumía kwargs
    `weight_in_grams=`/`master_item=` que no existen en la firma real)."""
    lbs = grams / 453.592
    return sc.apply_smart_market_units(name, lbs, "g", grams, master_item=master_item)


# ───────────── 1. el bound del floor ─────────────

def test_floor_no_permite_underbuy_mayor_al_knob():
    # necesidad 1400 g vs cartón 946 g → frac 0.48, under_buy=454 > 1400*0.10=140: hoy (pre-fix)
    # compraba 1 (cover 0.676); con el bound debe comprar 2.
    item = _por_gramos("Leche", 1400.0, CARTON)
    assert item["market_qty"] >= 2
    assert item.get("pkg_cover_ratio", 1.0) >= 0.90


def test_underbuy_marginal_sigue_floor():
    # frac 0.0571 (caso del SKU-OVERSHOOT-FIX original): under_buy=54 <= 1400*0.10... aquí
    # 1000*0.10=100, under_buy=54<=100 → floor OK, sin nota (cover=0.946 >= 0.9).
    item = _por_gramos("Leche", 1000.0, CARTON)
    assert item["market_qty"] == 1
    assert "alcanza" not in item.get("display_qty", "")
    assert item["pkg_cover_ratio"] >= 0.9


def test_arroz_audit_pasa_a_ceil():
    """El caso real medido en prod (arroz 647.5g / funda 453.6g, cover 0.70 pre-fix) ya NO elige
    floor tras el fix: under_buy=193.9 > 647.5*0.10=64.75 → compra 2 fundas, cover sube a ~1.40."""
    item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ)
    assert item["market_qty"] == 2
    assert item["pkg_cover_ratio"] == pytest.approx(907.2 / 647.5, abs=0.01)


# ───────────── 2. la nota (vía knob al tope del clamp — ver docstring del módulo) ─────────────

def test_cover_bajo_lleva_nota_alcanza(monkeypatch):
    """Con `SKU_FLOOR_MAX_UNDER_PCT` en su tope de clamp (0.5), el under_buy real del arroz
    (193.9g de 647.5g, 30%) SIGUE cabiendo bajo el bound (647.5*0.5=323.75) → floor se retiene,
    cover=453.6/647.5≈0.70 < 0.9 y sin `capped_by` → debe llevar la nota."""
    monkeypatch.setattr(sc, "SKU_FLOOR_MAX_UNDER_PCT", 0.5)
    item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ)
    assert item["market_qty"] == 1
    assert item["pkg_cover_ratio"] < 0.9
    assert item.get("capped_by") is None
    assert "alcanza ~5 de 7 días — recompra" in item["display_qty"], item["display_qty"]
    assert "alcanza ~5 de 7 días — recompra" in item["display_string"], item["display_string"]


def test_nota_no_duplica_si_ya_esta_capado(monkeypatch):
    """Si el ítem ya tiene `capped_by` (P1-CAPPED-STAPLE-HONESTY), la nota de cover bajo NO debe
    añadir un segundo sufijo — la nota existente manda (decisión #3)."""
    monkeypatch.setattr(sc, "SKU_FLOOR_MAX_UNDER_PCT", 0.5)
    sc.reset_caps_applied_last_run()
    try:
        # Cap MARGINAL (99%, no dispara el sufijo viejo `_frac < 0.9`) — el punto es solo que
        # `capped_by` quede seteado en el resultado (`_cap_hit` requiere post<pre estrictamente).
        sc._record_cap_applied("Arroz blanco", 100.0, 99.0, "TEST-CAP")
        item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ)
        assert item["pkg_cover_ratio"] < 0.9
        # sin fracción real de cap (post==pre, no dispara el sufijo de capped-staple) pero SÍ
        # queda `capped_by` seteado — eso basta para suprimir la nota nueva.
        assert item.get("capped_by") == "TEST-CAP"
        assert item["display_qty"].count("alcanza") == 0
    finally:
        sc.reset_caps_applied_last_run()


def test_knobs_registrados_con_default_y_clamp():
    assert sc.SKU_FLOOR_MAX_UNDER_PCT == pytest.approx(0.10)
    assert sc.PKG_COVER_NOTE_MIN == pytest.approx(0.9)
    reg = sc.get_knobs_registry_snapshot() if hasattr(sc, "get_knobs_registry_snapshot") else None
    if reg is not None:
        assert "MEALFIT_SKU_FLOOR_MAX_UNDER_PCT" in reg
        assert "MEALFIT_PKG_COVER_NOTE_MIN" in reg


# ───────────── 3. `_item_cycle_repurchases` deja de clampar a plano cuando ratio<1 ─────────────

def test_repurchases_no_clampa_a_plano_cuando_ratio_bajo():
    """Un envase que NO alcanza ni una ida (`pkg_cover_ratio=0.70`) implica MÁS recompras que el
    plano ×semanas, no menos. El clamp `min(plano, ...)` ocultaba esto — decisión #5."""
    item = {"pkg_cover_ratio": 0.70}
    plano = max(1.0, 30 / 7)
    resultado = sc._item_cycle_repurchases(item, cycle_days=30, trip_days=7)
    assert resultado > plano


def test_repurchases_sigue_acotado_a_plano_cuando_ratio_alto():
    """Cuando el envase SOBRA cobertura (ratio>=1, el caso que P1-CYCLE-REPURCHASE-HONEST ya
    cubría), el comportamiento no cambia: nunca más que el plano."""
    item = {"pkg_cover_ratio": 4.0}
    plano = max(1.0, 30 / 7)
    resultado = sc._item_cycle_repurchases(item, cycle_days=30, trip_days=7)
    assert resultado <= plano


def test_repurchases_sin_senal_no_adivina():
    item = {}
    plano = max(1.0, 30 / 7)
    assert sc._item_cycle_repurchases(item, cycle_days=30, trip_days=7) == plano
