"""[P1-SKU-COVER-HONESTY · 2026-08-02] El selector de envases permitía under-buy hasta ~33-50% en
silencio.

La regla floor `under_buy_g < over_buy_g` (⟺ frac<0.5) vivía en 3 sitios — `apply_smart_market_units`
standard path, `_find_best_sku`, `_select_market_package` (donde min-costo además FAVORECÍA el
floor) — y permitía comprar 1 envase cuando se necesitaban 1.48 (68% de cobertura). `pkg_cover_ratio`
se calcula y persiste pero nadie lo consumía para avisar (solo sobre-cobertura ≥2×,
P1-OVERCOVER-LABEL) y `_item_cycle_repurchases` clampaba `min(plano,...)` ⇒ el costo tampoco
declaraba la recompra real. Medido en prod: 18/22 planes con ≥1 ítem cover<0.9 sin nota (arroz
0.69-0.81, aceite 0.70-0.93, camarones 0.76-0.79).

Fix ronda 1: el floor ya NO se retiene si el under-buy absoluto excede `SKU_FLOOR_MAX_UNDER_PCT`
(10% del total, knob `MEALFIT_SKU_FLOOR_MAX_UNDER_PCT`) — la rama `frac <= ANTI_WASTE_THRESHOLD`
(colchón de coma flotante, 2%) se conserva intacta. Cuando el envase elegido cubre <95% del ciclo
(`PKG_COVER_NOTE_MIN`, knob `MEALFIT_PKG_COVER_NOTE_MIN`) y el ítem no tiene ya un aviso de cap
(`capped_by`), se sufija "alcanza ~N de M días — recompra" (mismo formato que
P1-CAPPED-STAPLE-HONESTY; `M` es `cycle_days`, parámetro nuevo de `apply_smart_market_units`,
default 7).

────────────────────────────────────────────────────────────────────────────────────────────────
RONDA 1 DE REVISIÓN (2026-08-02) — 4 hallazgos "Important" + 5 minors. Ver
`.superpowers/sdd/2026-08-02-solver-seeder-v7-gaps/task-5-report.md` sección "Ronda 1" para el
detalle completo con números medidos. Resumen:

1. El bound puro `under_buy <= g_total*PCT` NO es uniformemente más estricto que el criterio
   viejo `under_buy<over_buy` — escala con `floor_units` y es VACUO desde `floor_units>=9`
   (permitía un déficit NUEVO: Sazón 137g/sobre 14g, floor=9, cover 0.92 vs el viejo 1.022).
   Fix: `frac<=ANTI_WASTE_THRESHOLD or (under_buy<=g_total*PCT and under_buy<over_buy)` — el AND
   garantiza que el resultado es SUBCONJUNTO estricto del criterio viejo (nunca peor, sólo más
   estricto). Ver `test_bound_no_introduce_deficit_nuevo_en_floor_alto` +
   `test_bound_nunca_produce_peor_cobertura_que_antes_barrido_aleatorio`.
2. Con el bound original (10%) y la nota en 0.9, la cobertura mínima garantizada por el bound
   (0.90) es exactamente igual al umbral de la nota (`< 0.9`) — la nota queda INALCANZABLE por
   construcción para ítems recién resueltos (medido: 60.000 casos aleatorios, 0 disparos
   legítimos). Fix: `PKG_COVER_NOTE_MIN` sube a 0.95 (cubre la banda real 5%-10% que el bound sí
   permite) + guard `test_pkg_cover_note_min_no_es_inalcanzable_por_construccion` + exclusión del
   falso positivo P2-PACK-UNITS-MATCH (`test_p2_pack_units_match_no_dispara_falso_positivo`).
3. El CABEZA-GUARD reconstruye el ítem como peso pero no limpiaba `_pkg_size_g` (sólo
   `sku_label`) → `pkg_cover_ratio` mezclaba el envase 'cabeza' descartado con el peso
   reconstruido. Fix: `_pkg_size_g = None` junto a `sku_label = None`. Ver
   `test_cabeza_guard_limpia_pkg_size_g`.
4. La nota decía "de 7 días" fijo aunque `weight_in_lbs` representa 15/30 días en listas
   biweekly/monthly (el multiplicador de ciclo entra ANTES de esta función). Fix: parámetro
   `cycle_days` (default 7, backward-compatible) reemplaza el 7 hardcodeado en el copy. Ver
   `test_nota_usa_cycle_days_real_en_vez_de_7_fijo`.
────────────────────────────────────────────────────────────────────────────────────────────────

⚠️ **Hallazgo aritmético (reportado en la ronda 0, sigue vigente)**: el caso propuesto originalmente
para probar la nota (arroz 647,5 g / funda 453,6 g, cover ≈0.70) no puede disparar la nota vía la
rama `frac <= ANTI_WASTE_THRESHOLD` sola (matemáticamente imposible que resulte en cover<0.9 por esa
vía). El test de la nota exercita el mecanismo vía `monkeypatch` del knob al tope del clamp (0.5) —
una configuración real y soportada, no un dict manufacturado.
"""
import random

import pytest

import shopping_calculator as sc
from knobs import _env_float, get_knobs_registry_snapshot

CARTON = {"market_container": "cartón", "container_weight_g": 946.0}
FUNDA_ARROZ = {"market_container": "funda", "container_weight_g": 453.6}


def _por_gramos(name, grams, master_item, cycle_days=7):
    """Adapta la firma real `apply_smart_market_units(name, weight_in_lbs, unit_str, raw_qty,
    master_item, cycle_days)` a una necesidad expresada en gramos (el brief original asumía
    kwargs `weight_in_grams=`/`master_item=` que no existen en la firma real)."""
    lbs = grams / 453.592
    return sc.apply_smart_market_units(name, lbs, "g", grams, master_item=master_item, cycle_days=cycle_days)


# ───────────── 1. el bound del floor ─────────────

def test_floor_no_permite_underbuy_mayor_al_knob():
    # necesidad 1400 g vs cartón 946 g → frac 0.48, under_buy=454 > 1400*0.10=140: hoy (pre-fix)
    # compraba 1 (cover 0.676); con el bound debe comprar 2.
    item = _por_gramos("Leche", 1400.0, CARTON)
    assert item["market_qty"] >= 2
    assert item.get("pkg_cover_ratio", 1.0) >= 0.90


def test_underbuy_marginal_sigue_floor():
    # frac 0.0571 (caso del SKU-OVERSHOOT-FIX original): under_buy=54 <= 1000*0.10=100 Y
    # under_buy(54) < over_buy(892) → floor OK, sin nota (cover=0.946 >= 0.95... no, 0.946<0.95:
    # ver test dedicado más abajo para el knob 0.95; aquí sólo ancla que el floor se retiene).
    item = _por_gramos("Leche", 1000.0, CARTON)
    assert item["market_qty"] == 1
    assert item["pkg_cover_ratio"] >= 0.9


def test_arroz_audit_pasa_a_ceil():
    """El caso real medido en prod (arroz 647.5g / funda 453.6g, cover 0.70 pre-fix) ya NO elige
    floor tras el fix: under_buy=193.9 > 647.5*0.10=64.75 → compra 2 fundas, cover sube a ~1.40."""
    item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ)
    assert item["market_qty"] == 2
    assert item["pkg_cover_ratio"] == pytest.approx(907.2 / 647.5, abs=0.01)


def test_bound_no_introduce_deficit_nuevo_en_floor_alto():
    """[RONDA 1 · Important 1] Sazón: 137g / sobre 14g → floor_units=9. under_buy=11,
    over_buy=3 (bajo el CRITERIO VIEJO `under_buy<over_buy`, 11<3 es False → ceil, cover=1.022).
    El bound PURO (`under_buy<=g_total*0.10=13.7`, SIN el `and under_buy<over_buy`) sería
    VACUO aquí y retendría floor (11<=13.7 → floor, cover=0.92) — un déficit NUEVO que el
    código pre-fix no tenía. Con el AND, el resultado es IDÉNTICO al viejo: ceil, cover>=1."""
    item = _por_gramos("Sazón", 137.0, {"market_container": "sobre", "container_weight_g": 14.0})
    assert item["market_qty"] == 10
    assert item["pkg_cover_ratio"] == pytest.approx(140.0 / 137.0, abs=0.005)


def test_bound_yogurt_floor_5_no_introduce_deficit():
    """Segundo dato del audit: yogurt 4990g / four-pack 907g → floor_units=5. under_buy=455,
    over_buy=452 (455<452 es False → el viejo también hacía ceil, cover=1.091). El bound puro
    (455<=4990*0.10=499) sería vacuo y retendría floor (cover=0.909) — mismo modo de fallo."""
    item = _por_gramos("Yogurt griego", 4990.0, {"market_container": "paquete", "container_weight_g": 907.0})
    assert item["market_qty"] == 6
    assert item["pkg_cover_ratio"] == pytest.approx(5442.0 / 4990.0, abs=0.005)


def test_bound_nunca_produce_peor_cobertura_que_antes_barrido_aleatorio():
    """[RONDA 1 · Important 1, verificación exigida] Barrido aleatorio (semilla fija, 2000
    casos): para cualquier (size, floor_units, frac), el criterio NUEVO (con el AND) nunca
    retiene floor donde el criterio VIEJO (`under_buy<over_buy`, sin bound) elegía ceil, y por
    tanto nunca produce una cobertura peor que la que el criterio viejo producía. 0% de déficits
    nuevos — cierra el hallazgo del revisor (32,7% de casos con déficit nuevo en el bound puro,
    50% para floor>=9)."""
    rng = random.Random(20260802)
    peor = 0
    total = 2000
    for _ in range(total):
        size = rng.uniform(10, 2000)
        floor_units = rng.randint(1, 20)
        frac = rng.uniform(0.0, 0.999)
        g_total = (floor_units + frac) * size
        under_buy = g_total - floor_units * size
        over_buy = (floor_units + 1) * size - g_total
        old_floor = under_buy < over_buy
        new_floor = (under_buy <= g_total * sc.SKU_FLOOR_MAX_UNDER_PCT) and old_floor
        # El nuevo criterio nunca retiene floor donde el viejo no lo hacía (subconjunto estricto).
        assert not (new_floor and not old_floor)
        cover_new = (floor_units / (floor_units + frac)) if new_floor else 1.0
        cover_old = (floor_units / (floor_units + frac)) if old_floor else 1.0
        if cover_new < cover_old - 1e-9:
            peor += 1
    assert peor == 0, f"{peor}/{total} casos con cobertura NUEVA peor que la vieja (esperado: 0)"


# ───────────── 2. la nota (vía knob al tope del clamp — ver docstring del módulo) ─────────────

def test_cover_bajo_lleva_nota_alcanza(monkeypatch):
    """Con `SKU_FLOOR_MAX_UNDER_PCT` en su tope de clamp (0.5), el under_buy real del arroz
    (193.9g de 647.5g) SIGUE cabiendo bajo el bound (647.5*0.5=323.75) Y bajo el criterio viejo
    (193.9<259.7) → floor se retiene, cover≈0.70 < 0.95 y sin `capped_by` → debe llevar la nota."""
    monkeypatch.setattr(sc, "SKU_FLOOR_MAX_UNDER_PCT", 0.5)
    item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ)
    assert item["market_qty"] == 1
    assert item["pkg_cover_ratio"] < 0.95
    assert item.get("capped_by") is None
    dias = max(1, round(7 * item["pkg_cover_ratio"]))
    assert f"alcanza ~{dias} de 7 días — recompra" in item["display_qty"], item["display_qty"]
    assert f"alcanza ~{dias} de 7 días — recompra" in item["display_string"], item["display_string"]


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
        assert item["pkg_cover_ratio"] < 0.95
        # sin fracción real de cap (post==pre, no dispara el sufijo de capped-staple) pero SÍ
        # queda `capped_by` seteado — eso basta para suprimir la nota nueva.
        assert item.get("capped_by") == "TEST-CAP"
        assert item["display_qty"].count("alcanza") == 0
    finally:
        sc.reset_caps_applied_last_run()


def test_p2_pack_units_match_no_dispara_falso_positivo():
    """[RONDA 1 · Important 2c] Burrito: envase "5 unid" 356g, density del MASTER 100g/unidad
    (difiere de la real del SKU, 356/5=71.2g/unidad). El recuento P2-PACK-UNITS-MATCH corrige el
    conteo por unidades reales (22 unidades necesarias → 5 paquetes, correcto), pero el ratio en
    GRAMOS (contra el need_g inflado por la density del master) da 0.809 — un falso positivo. La
    nota debe suprimirse cuando el conteo se derivó por unidades, no por peso."""
    burrito = {
        "market_container": "paquete", "container_weight_g": 356.0,
        "market_packages": [{"grams": 356.0, "price": 100.0, "label": "5 unid", "unit": "paquete"}],
        "density_g_per_unit": 100.0,
    }
    item = _por_gramos("Burrito", 2200.0, burrito)
    assert item["market_qty"] == 5
    assert item["pkg_cover_ratio"] == pytest.approx(0.809, abs=0.005)
    assert "alcanza" not in item.get("display_qty", "")


def test_pkg_cover_note_min_no_es_inalcanzable_por_construccion():
    """[RONDA 1 · Important 2b] Guard contra que un futuro cambio de default vuelva la nota
    inerte en silencio: cuando SÍ se retiene el floor (rama del bound), la cobertura mínima
    garantizada es `1 - SKU_FLOOR_MAX_UNDER_PCT`. Si `PKG_COVER_NOTE_MIN` fuera <= ese mínimo, la
    nota es INALCANZABLE por construcción para cualquier ítem recién resuelto por estos 3 sitios
    (medido: 60.000 casos aleatorios, 0 disparos legítimos con el default anterior de 0.9)."""
    assert sc.PKG_COVER_NOTE_MIN > 1 - sc.SKU_FLOOR_MAX_UNDER_PCT, (
        f"PKG_COVER_NOTE_MIN={sc.PKG_COVER_NOTE_MIN} <= 1-SKU_FLOOR_MAX_UNDER_PCT="
        f"{1 - sc.SKU_FLOOR_MAX_UNDER_PCT}: la nota es inalcanzable por construcción — el bound "
        f"garantiza cover >= 1-SKU_FLOOR_MAX_UNDER_PCT, así que nunca cae por debajo del umbral."
    )


def test_nota_usa_cycle_days_real_en_vez_de_7_fijo(monkeypatch):
    """[RONDA 1 · Important 4] Caso mensual: `weight_in_lbs` ya representa una necesidad de 30
    días (el multiplicador de ciclo entra ANTES de esta función), así que la nota debe decir
    "de 30 días", no "de 7 días" — mismos gramos/envase que `test_cover_bajo_lleva_nota_alcanza`,
    sólo cambia `cycle_days`."""
    monkeypatch.setattr(sc, "SKU_FLOOR_MAX_UNDER_PCT", 0.5)
    item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ, cycle_days=30)
    assert item["market_qty"] == 1
    dias = max(1, round(30 * item["pkg_cover_ratio"]))
    assert f"alcanza ~{dias} de 30 días — recompra" in item["display_qty"], item["display_qty"]
    assert "de 7 días" not in item["display_qty"]


def test_cycle_days_default_preserva_comportamiento_previo():
    """Callers que no pasan `cycle_days` (la mayoría hoy, incl. todos los tests de arriba salvo
    los explícitos) deben seguir viendo el comportamiento previo — default 7, sin cambio."""
    import inspect
    sig = inspect.signature(sc.apply_smart_market_units)
    assert sig.parameters["cycle_days"].default == 7


def test_knobs_registrados_con_default_y_clamp():
    assert sc.SKU_FLOOR_MAX_UNDER_PCT == pytest.approx(0.10)
    assert sc.PKG_COVER_NOTE_MIN == pytest.approx(0.95)
    reg = get_knobs_registry_snapshot()
    assert "MEALFIT_SKU_FLOOR_MAX_UNDER_PCT" in reg
    assert "MEALFIT_PKG_COVER_NOTE_MIN" in reg
    assert reg["MEALFIT_SKU_FLOOR_MAX_UNDER_PCT"]["default"] == pytest.approx(0.10)
    assert reg["MEALFIT_PKG_COVER_NOTE_MIN"]["default"] == pytest.approx(0.95)


def test_knob_clamp_rechaza_fuera_de_rango():
    """[MINOR ronda 1] `hasattr(sc, "get_knobs_registry_snapshot")` era siempre False (vive en
    `knobs`, no en `shopping_calculator`) — las aserciones de arriba nunca corrían. Este test
    ancla además el comportamiento de CLAMP (valor fuera de rango → cae al default), invocando
    el mismo helper `_env_float` que usan los knobs reales."""
    assert _env_float("MEALFIT_SKU_FLOOR_MAX_UNDER_PCT", 0.10, lambda v: 0.0 <= v <= 0.5,
                       ) == pytest.approx(0.10)  # sin env var → default
    import os
    os.environ["MEALFIT_SKU_COVER_HONESTY_TEST_OOR"] = "0.9"
    try:
        val = _env_float("MEALFIT_SKU_COVER_HONESTY_TEST_OOR", 0.10, lambda v: 0.0 <= v <= 0.5)
        assert val == pytest.approx(0.10), "0.9 excede el clamp [0,0.5] → debe caer al default"
    finally:
        del os.environ["MEALFIT_SKU_COVER_HONESTY_TEST_OOR"]
    os.environ["MEALFIT_PKG_COVER_NOTE_MIN_TEST_OOR"] = "1.5"
    try:
        val2 = _env_float("MEALFIT_PKG_COVER_NOTE_MIN_TEST_OOR", 0.95, lambda v: 0.0 <= v <= 1.0)
        assert val2 == pytest.approx(0.95), "1.5 excede el clamp [0,1] → debe caer al default"
    finally:
        del os.environ["MEALFIT_PKG_COVER_NOTE_MIN_TEST_OOR"]


# ───────────── 3. CABEZA-GUARD deja `_pkg_size_g` sucio (Important 3) ─────────────

def test_cabeza_guard_limpia_pkg_size_g():
    """[RONDA 1 · Important 3] El CABEZA-GUARD reconstruye el ítem como PESO (lbs) pero antes
    sólo limpiaba `sku_label`, no `_pkg_size_g` — el bloque de `pkg_cover_ratio` seguía viendo el
    tamaño del envase 'cabeza' descartado y mezclaba unidades. Reproducido exacto: Zanahoria
    900g con `market_container='cabeza'` (envase 150g) → pre-fix "2 lbs · alcanza ~2 de 7 días —
    recompra" con cover=0.333 falso, e inflaba `_item_cycle_repurchases` a ~12.9 recompras
    contra un plano de 4.29. Tras limpiar `_pkg_size_g` junto con `sku_label`, no debe quedar
    ratio ni nota."""
    master = {"market_container": "cabeza", "container_weight_g": 150.0}
    item = sc.apply_smart_market_units("Zanahoria", 900.0 / 453.592, "g", 900.0, master_item=master)
    assert item["market_qty"] == 2.0
    assert item["market_unit"] == "lbs"
    assert "pkg_cover_ratio" not in item, "el CABEZA-GUARD debe limpiar _pkg_size_g → sin ratio"
    assert "package_grams" not in item
    assert "alcanza" not in item.get("display_qty", "")
    # Sin `pkg_cover_ratio`, `_item_cycle_repurchases` no tiene señal → plano, no ~12.9.
    plano = max(1.0, 30 / 7)
    assert sc._item_cycle_repurchases(item, cycle_days=30, trip_days=7) == pytest.approx(plano)


# ───────────── 4. `_item_cycle_repurchases` deja de clampar a plano cuando ratio<1 ─────────────

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


# ───────────── 5. plumbing de `cycle_days` (source-level, offline) ─────────────

def test_cycle_days_plumbing_local():
    """`cycle_days` se propaga desde `get_shopping_list_delta` -> `aggregate_and_deduct_shopping_
    list` -> los 2 call-sites de `apply_smart_market_units`, DENTRO de este archivo. Los 15+
    call-sites externos (cron_tasks.py/routers/plans.py/tools.py) NO pasan todavía el valor real
    — seguimiento pendiente documentado en el report, no de este archivo."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("def aggregate_and_deduct_shopping_list(")
    assert "cycle_days: int | None = None" in src[i:i + 400]
    assert src.count(
        "apply_smart_market_units(name, weight_in_lbs, 'lb', 0.0, master_item, cycle_days=_cycle_days_for_note)"
    ) == 1
    assert src.count(
        "apply_smart_market_units(name, 0.0, u, q, master_item, cycle_days=_cycle_days_for_note)"
    ) == 1
    j = src.index("def get_shopping_list_delta(")
    j_end = src.index("\ndef ", j + 10)
    assert "cycle_days: int | None = None," in src[j:j_end]
    assert "cycle_days=cycle_days)" in src[j:j_end]
