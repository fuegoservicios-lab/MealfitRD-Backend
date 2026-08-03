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
RONDA 2 DE REVISIÓN (2026-08-02) — 2 hallazgos. Ver el report sección "Ronda 2" para la tabla
exhaustiva de callsites y el texto literal de los 3 casos de verificación. Resumen:

1. **NOT ADDRESSED en la ronda 1**: el plumbing de `cycle_days` no llegaba a producción — los
   callsites reales (`cron_tasks.py`, `routers/plans.py`, `tools.py`, y también
   `graph_orchestrator.py`, que el reporte de la ronda 1 no había enumerado) nunca pasaban
   `cycle_days=`. Fix: nuevo SSOT `cycle_days_for_duration(duration)` (hermano de
   `cycle_qty_multiplier`, misma tabla `_CYCLE_DAYS_BY_DURATION`) + threading mecánico en los
   ~26 callsites de periodo (biweekly/monthly) de los 4 archivos. Ver
   `test_callsites_de_periodo_llevan_cycle_days`.
2. **Rotura nueva de la ronda 1**: con `PKG_COVER_NOTE_MIN=0.95` y `round()`, Leche 1000g/cartón
   946g (cover 0,946) imprimía "alcanza ~7 de 7 días — recompra" — cobertura completa Y pide
   recomprar, contradictorio. Fix: `math.floor` en vez de `round` (da "~6 de 7", cierto) +
   supresión de la nota si el floor iguala/supera `cycle_days` (defensivo, inalcanzable hoy por
   construcción dado el gate `cover<PKG_COVER_NOTE_MIN<=1`, ver
   `test_nota_suprimida_si_floor_iguala_o_supera_cycle_days`). Restaurada la aserción perdida en
   `test_underbuy_marginal_sigue_floor` + nuevo `test_cobertura_completa_no_lleva_nota`.
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
    """[RONDA 2 · ítem 2] frac 0.0571 (caso del SKU-OVERSHOOT-FIX original): under_buy=54 <=
    1000*0.10=100 Y under_buy(54) < over_buy(892) → floor se retiene (market_qty=1). Con el
    `PKG_COVER_NOTE_MIN=0.95` de la ronda 1, cover=0,946 SÍ dispara la nota (0,946<0,95) — la
    ronda 1 dejaba este test sin aserción sobre el texto (ambigüedad "sin nota" incorrecta,
    encontrada por el revisor: con `round()` daba el contradictorio "alcanza ~7 de 7 días"). Con
    `floor()` (fix de esta ronda) da correctamente "~6 de 7 días" — falta 1 día, cierto y
    accionable. Ver `test_cobertura_completa_no_lleva_nota` para el caso SIN nota."""
    item = _por_gramos("Leche", 1000.0, CARTON)
    assert item["market_qty"] == 1
    assert item["pkg_cover_ratio"] == pytest.approx(0.946, abs=0.001)
    assert "alcanza ~6 de 7 días — recompra" in item["display_qty"], item["display_qty"]
    assert "alcanza ~7 de 7 días" not in item["display_qty"], (
        "contradicción: 'alcanza ~7 de 7' con round() decía cobertura completa Y pedía recomprar")


def test_cobertura_completa_no_lleva_nota():
    """[RONDA 2 · ítem 2] Cuando la cobertura YA es completa (cover>=1, ej. Leche 900g contra un
    cartón de 946g que sobra), la nota NO debe aparecer — no hay nada que avisar. Ancla la mitad
    positiva de la garantía: "cobertura completa ⇒ SIN nota" (la otra mitad, cover parcial ⇒ CON
    nota honesta, la ancla `test_underbuy_marginal_sigue_floor`)."""
    item = _por_gramos("Leche", 900.0, CARTON)
    assert item["pkg_cover_ratio"] >= 1.0
    assert "alcanza" not in item.get("display_qty", "")
    assert "alcanza" not in item.get("display_string", "")


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
    (193.9<259.7) → floor se retiene, cover≈0.70 < 0.95 y sin `capped_by` → debe llevar la nota.

    [RONDA 2 · ítem 2] Días con `math.floor` (no `round`) — con round, 7*0,701=4,907 redondeaba
    a 5; con floor da 4. La diferencia importa: floor nunca puede leer "completo" cuando no lo
    está (ver `test_underbuy_marginal_sigue_floor` para el caso que expuso la contradicción)."""
    import math
    monkeypatch.setattr(sc, "SKU_FLOOR_MAX_UNDER_PCT", 0.5)
    item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ)
    assert item["market_qty"] == 1
    assert item["pkg_cover_ratio"] < 0.95
    assert item.get("capped_by") is None
    dias = max(1, math.floor(7 * item["pkg_cover_ratio"]))
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
    sólo cambia `cycle_days`. [RONDA 2] días con `math.floor`, no `round` (ver ítem 2)."""
    import math
    monkeypatch.setattr(sc, "SKU_FLOOR_MAX_UNDER_PCT", 0.5)
    item = _por_gramos("Arroz blanco", 647.5, FUNDA_ARROZ, cycle_days=30)
    assert item["market_qty"] == 1
    dias = max(1, math.floor(30 * item["pkg_cover_ratio"]))
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
    list` -> los 2 call-sites de `apply_smart_market_units`, DENTRO de este archivo. [RONDA 2 ·
    ítem 1] Los ~26 call-sites externos AHORA SÍ pasan el valor real — ver
    `test_callsites_de_periodo_llevan_cycle_days` para la verificación cross-file.

    [P1-VEG-BACKFILL-HONESTY · 2026-08-02] Los 2 call-sites ganaron un kwarg más
    (`text_demand_g=...`) el mismo día — el substring exacto se actualiza para reflejar la firma
    real en vez de quedar ciego a un rename de `cycle_days=_cycle_days_for_note)` a
    `cycle_days=_cycle_days_for_note, text_demand_g=...)`."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("def aggregate_and_deduct_shopping_list(")
    assert "cycle_days: int | None = None" in src[i:i + 400]
    assert src.count(
        "apply_smart_market_units(name, weight_in_lbs, 'lb', 0.0, master_item, cycle_days=_cycle_days_for_note, text_demand_g=(text_demand_g_map or {}).get(name))"
    ) == 1
    assert src.count(
        "apply_smart_market_units(name, 0.0, u, q, master_item, cycle_days=_cycle_days_for_note, text_demand_g=(text_demand_g_map or {}).get(name))"
    ) == 1
    j = src.index("def get_shopping_list_delta(")
    j_end = src.index("\ndef ", j + 10)
    assert "cycle_days: int | None = None," in src[j:j_end]
    # [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · review final] El mapa de texto ya no viaja crudo: pasa
    # por el gate de homogeneidad `_tdg_para_agg` (vacío cuando la lista es un DELTA con deducción
    # de Nevera, porque la demanda de recetas es BRUTA). El plumbing de `cycle_days`, que es lo que
    # este test ancla, no cambia.
    # [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03 · ronda 1] El substring se extiende con
    # `apply_protein_yield=_apply_protein_yield)` (nuevo kwarg al final de la MISMA llamada) — el
    # ancla se actualiza a la línea real en vez de recortar antes del kwarg nuevo, que dejaría
    # pasar en verde un futuro editor que borre el plumbing de `cycle_days` sin tocar el final.
    assert (
        "cycle_days=cycle_days, text_demand_g_map=_tdg_para_agg, "
        "apply_protein_yield=_apply_protein_yield)"
    ) in src[j:j_end]
    # [RONDA 2 · ítem 1] SSOT hermano de `cycle_qty_multiplier`, misma tabla
    # `_CYCLE_DAYS_BY_DURATION` — evita que un callsite escriba un literal `15`/`30` suelto que
    # pueda driftear de la tabla que ya usa `cycle_qty_multiplier`.
    assert "def cycle_days_for_duration(duration: str) -> int:" in src


# [RONDA 2 · ítem 1] Los 5 archivos reales que invocan `get_shopping_list_delta` con contexto de
# duración (directo o via alias `_gsld`/`_gsld_il`/`_adb`). `agent.py` está en la lista por
# completitud — sus 2 callsites NO tienen contexto de duración (sin `cycle_qty_multiplier`), así
# que el conteo esperado ahí es 0==0 (ver tabla del report para la enumeración exhaustiva de las
# ~43 invocaciones totales, no sólo las 26 de periodo).
_EXTERNAL_CALLSITE_FILES = (
    "cron_tasks.py", "routers/plans.py", "tools.py", "graph_orchestrator.py", "agent.py",
)


def test_callsites_de_periodo_llevan_cycle_days():
    """[RONDA 2 · ítem 1] Parser-based, cross-file: cada callsite que pasa
    `cycle_qty_multiplier("biweekly")` o `("monthly")` a `get_shopping_list_delta` (directo o via
    alias `_gsld`/`_gsld_il`/`_adb`) DEBE llevar el `cycle_days=cycle_days_for_duration(...)`
    gemelo con la MISMA duración — si no, la nota de un ciclo biweekly/monthly real vuelve a
    decir "de 7 días" en silencio (el bug que esta ronda cierra; confirmado por el re-revisor
    ejecutando el código, no sólo leyéndolo).

    Conteo 1:1 por archivo y por duración en vez de un parser posicional sobre llamadas
    multi-línea (más frágil, más difícil de mantener correcto) — detecta tanto callsites nuevos
    sin `cycle_days` como un exceso accidental."""
    from pathlib import Path
    base = Path(sc.__file__).resolve().parent
    total_checked = 0
    for fname in _EXTERNAL_CALLSITE_FILES:
        src = (base / fname).read_text(encoding="utf-8")
        for duration in ("biweekly", "monthly"):
            n_mult = src.count(f'cycle_qty_multiplier("{duration}")')
            n_days = src.count(f'cycle_days_for_duration("{duration}")')
            assert n_mult == n_days, (
                f"{fname}: {n_mult} callsites de cycle_qty_multiplier('{duration}') vs "
                f"{n_days} de cycle_days_for_duration('{duration}') -- deben ser 1:1 "
                f"(cada llamada de periodo real necesita su cycle_days gemelo)"
            )
            total_checked += n_mult
    # Sanity: el test no debe pasar vacuamente (0==0 en todos lados) si el refactor renombra las
    # funciones — al menos las 26 parejas reales (13 trios × 2 duraciones) deben aparecer.
    assert total_checked >= 26, (
        f"sólo se encontraron {total_checked} callsites de periodo -- ¿un rename rompió el "
        f"parser sin que nadie lo notara?"
    )


def test_nota_suprimida_si_floor_iguala_o_supera_cycle_days(monkeypatch):
    """[RONDA 2 · ítem 2] Rama defensiva: si `math.floor(cycle_days*cover) >= cycle_days`, la
    cobertura es efectivamente completa -- no se emite "~N de N" (que sería tan contradictorio
    como el "~7 de 7" que motivó el fix a floor). Bajo los knobs por defecto esto es
    inalcanzable por construcción (el gate externo ya exige `cover < PKG_COVER_NOTE_MIN <= 1`,
    así que `floor(cycle_days*cover)` nunca alcanza `cycle_days`) -- se fuerza monkeypencheando
    `PKG_COVER_NOTE_MIN` a un valor fuera de su clamp normal para ejercitar la rama directamente,
    igual que `test_pkg_cover_note_min_no_es_inalcanzable_por_construccion` prueba la garantía
    matemática en el otro sentido."""
    monkeypatch.setattr(sc, "PKG_COVER_NOTE_MIN", 2.0)  # fuera del clamp real [0,1] -- a propósito
    item = _por_gramos("Leche", 900.0, CARTON)  # cover ~1.05, "completo"
    assert item["pkg_cover_ratio"] >= 1.0
    assert "alcanza" not in item.get("display_qty", ""), (
        "con PKG_COVER_NOTE_MIN inflado el gate externo se cumple (1.05<2.0) pero floor(7*1.05)="
        "7>=cycle_days=7 debe suprimir la nota igual -- nunca '~7 de 7'"
    )
