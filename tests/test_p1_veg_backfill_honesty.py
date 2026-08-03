"""[P1-VEG-BACKFILL-HONESTY · 2026-08-02, ronda de revisión 2026-08-03] Planes VIVOS cuya receta
(texto) contradice su propia lista de compras, sin ninguna nota — residuo de datos YA PERSISTIDOS
que Task 7 (P1-RECONCILE-CDA-DENSITY) no toca (esa arregla el lado GENERADOR para planes NUEVOS).

## Ronda de revisión 2026-08-03 (2 IMPORTANT + 1 MINOR)

1. **El emparejamiento se rompía con plurales, en silencio.** `text_demand_g_map` quedaba keyed
   por el nombre CRUDO de `expected_sum_from_recipes` ('Tomates', 'Cebollas') mientras el lado
   comprado resuelve por nombre CANÓNICO ('Tomate', 'Cebolla') — la cadena de canonicalización
   que antes vivía SOLO inline en `aggregate_and_deduct_shopping_list`. Medido en vivo: "300 g de
   tomates" → 'Tomates' (parser) vs 'Tomate' (comprado); "200 g de cebollas" → 'Cebolla'. Fix: la
   cadena se extrajo a `canonicalize_shopping_food_name()` + `_build_shopping_master_map()`
   (SSOT), reusada por AMBOS lados — `get_shopping_list_delta` ahora canonicaliza cada key de
   `expected_sum_from_recipes` antes de construir el mapa (sumando cuando varias keys crudas
   colapsan al mismo canónico). ⚠️ El tercer ejemplo medido por el revisor ("100 g de espinacas"
   → 'Espinaca' singular) NO se sostiene corriendo la cadena COMPLETA: la cola de 13 regex de
   consolidación (que siempre corre después) revierte 'Espinaca' de vuelta a 'Espinacas' —
   confirmado byte-a-byte idéntico contra el HEAD previo a esta ronda. Para espinaca crudo YA
   coincidía con canónico (net no-op, nada que arreglar); `test_canonicaliza_rucula_plural_a_
   singular_sin_reversion` ancla el caso positivo limpio equivalente (misma familia
   `canonicalize_verduras_hoja`, sin la reversión de cola).
2. **El bloque nuevo ignoraba el kill switch `CAPPED_STAPLE_HONESTY`.** Reproducido: con el knob
   en `False`, el lookup real ni corría (vive dentro de su propio `if CAPPED_STAPLE_HONESTY:`) pero
   el backstop sintético SÍ, y la nota seguía saliendo — un operador que apaga el knob en un
   incidente no lograba callarla. Fix: la rama sintética ahora exige `CAPPED_STAPLE_HONESTY` igual
   que el cap real (simétrico: sin el knob, ninguno de los dos deja `capped_by`).
3. **[MINOR] La segunda pasada del agregador (ventaneo de perecederos, OFF por default) no recibía
   `text_demand_g_map`.** El mapa ya está calculado sobre el plan completo — ahora se pasa también
   ahí para que quede coherente si ese camino se enciende en el futuro.

---
Contexto original de la tarea:

Evidencia medida en producción (SELECT, sin escritura):
  - plan `5f4bb17e`: receta dice "600 g de espárragos" en la cena; la lista SEMANAL compra
    583.33 g — una sola cena agota el 103% de la compra de la semana. `capped_by=null`, sin aviso:
    ningún cap conocido (`_CAPS_APPLIED_LAST_RUN`) explica el déficit porque espárragos no está en
    `_VEG_PER_WEEK_PER_PERSON` (P5-VEG-CAP) ni en ningún otro cap por categoría — el número
    simplemente llegó corto y nadie lo dijo.
  - plan `8d3f246a`: "470 g de tayota" vs lista 891 g (ratio 0.635 lado guard).
  - plan `cf3a81fb`: vainitas 933 g (=400×7/3, inflación de solver entregada completa) + calabacín
    1372 g/persona.

## El mecanismo (dos entregables, decisión ya tomada — ver task-8-brief.md)

(a) Mecanismo PERMANENTE en el agregador: cuando la cantidad final que `apply_smart_market_units`
    resuelve para un ítem (`base_qty`, en gramos) queda por debajo de `MEALFIT_QTY_SHORTFALL_
    NOTE_MIN` (default 0.9) veces la demanda que las RECETAS piden (`text_demand_g`, mismo parse
    que usa el guard — threaded desde `get_shopping_list_delta` vía `expected_sum_from_recipes` +
    `_normalize_food_units_to_base`), se estampa un `capped_by="qty_reconcile_v7"` SINTÉTICO. Ese
    `capped_by` alimenta el MISMO `_cap_hit` que P1-CAPPED-STAPLE-HONESTY (2026-07-26) ya consume
    para componer la nota "alcanza ~N de 30 días — recompra" — no se reimplementa el copy.

(b) Script one-shot `scripts/backfill_veg_lines_v7.py` (dry-run por defecto, NO ejecutado en esta
    sesión — DB de prod, ver header del script) para los planes YA persistidos con el defecto.

## Composición con P1-SKU-COVER-HONESTY (Task 5, mismo día)

Task 5 ya excluye de SU sufijo los ítems que traen `capped_by` (para no duplicar "alcanza...alcanza").
Estampar el `capped_by` sintético ANTES de que corra el bloque de Task 5 (mismo orden: el lookup de
`_cap_hit` corre completo, INCLUYENDO nuestro fallback, antes de que el bloque de
`pkg_cover_ratio` decida si sufija) cierra el círculo: la nota que se muestra es SIEMPRE la de
P1-CAPPED-STAPLE-HONESTY (el "dueño" histórico del copy), nunca duplicada.

## Riesgos verificados explícitamente (ver pedido de la tarea)

  1. Un ítem que YA tiene `capped_by` REAL (post<pre estricto) no debe ser pisado por el sintético
     — `_cap_hit is None` guardia la rama nueva (`test_no_pisa_un_cap_real_ya_registrado`).
  2. Nunca dos sufijos "alcanza" en el mismo `display_qty` (`test_no_duplica_sufijo_con_sku_cover_honesty`).
  3. Un ítem cuya compra SÍ cubre el texto (>=90%) no recibe nada (`test_compra_suficiente_no_recibe_nota`).

## Review final 2026-08-03 (1 CRITICAL + 1 IMPORTANT) — secciones 10 y 11

4. **[CRITICAL] El sello sintético CEGABA al coherence guard justo en los ítems que detecta.**
   `capped_pre` no es un campo de telemetría: el guard lo usa para SUSTITUIR la cantidad realmente
   comprada al construir el lado ACTUAL (P1-COHERENCE-CAPPED-PRE). Para un cap REAL eso es legítimo
   —`capped_pre` es lo que el AGREGADOR calculó por su cuenta antes del tope, un número
   independiente del lado esperado, así que un agregador equivocado sigue divergiendo—. Para el
   sello sintético, `pre_value` ES la demanda de las recetas: la misma
   `expected_sum_from_recipes(..., multiplier=effective_multiplier)` con la que el guard construye
   el lado ESPERADO. Resultado: el guard comparaba el esperado contra sí mismo. Reproducido
   ejecutando — drift de magnitud 2× (recetas 2000 g, lista 1000 g): sin el sello,
   `[{'delta_pct': 0.5}]`; con el sello, `[]`. Con el guard en modo `block` por default, ese ítem
   dejaba de escalar, de forzar retry y de aparecer en `_shopping_coherence_block_history` — o sea
   que el P-fix que venía a hacer VISIBLE el déficit lo volvía invisible para la única capa que
   puede corregirlo. Fix: el déficit sintético viaja por claves PROPIAS (`shortfall_text_g` /
   `shortfall_bought_g`) y `_extract_aggregated_food_dict` excluye explícitamente el sello de la
   sustitución (defensa doble: el sello queda PERSISTIDO en las listas, así que una lista
   construida por la versión anterior seguiría cegando al guard al releerla).
5. **[IMPORTANT] El backstop comparaba demanda BRUTA contra compra NETA.** `_text_demand_g_map`
   sale de las recetas sin restar nada, pero cuando hay `items_to_deduct` (Nevera + consumidos) el
   lado comprado es un DELTA. Reproducido: receta 2100 g + nevera 500 g + compra 1600 g → nota de
   recompra sobre algo que el usuario YA tiene en casa. Y `is_new_plan` es False POR DEFAULT, o sea
   que las superficies afectadas son la tool del coach, `mark_shopping_list_purchased`, los dos
   callsites de `agent.py` y `get_pantry_completion_list` (donde el falso positivo sería del 100%).
   Fix: el mapa se pasa sólo cuando NO hubo deducción (gate por `items_to_deduct`, no por
   `is_new_plan`: un delta con la nevera vacía también es homogéneo y ahí el mecanismo es correcto).
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc
from knobs import _env_float, get_knobs_registry_snapshot

CARTON = {"market_container": "cartón", "container_weight_g": 946.0}


def _por_gramos(name, grams, master_item=None, text_demand_g=None, cycle_days=7):
    """Adapta la firma real `apply_smart_market_units(name, weight_in_lbs, unit_str, raw_qty,
    master_item, cycle_days, text_demand_g)` a una necesidad expresada en gramos — mismo patrón
    que `test_p1_sku_cover_honesty.py::_por_gramos`."""
    lbs = grams / 453.592
    return sc.apply_smart_market_units(
        name, lbs, "g", grams, master_item=master_item, cycle_days=cycle_days,
        text_demand_g=text_demand_g,
    )


# ───────────── 1. el mecanismo dispara cuando NINGÚN cap real explica el déficit ─────────────

def test_dispara_cuando_compra_queda_bajo_90pct_del_texto():
    """Espejo del caso medido en prod (espárragos 600 g × 7 días = 4200 g de demanda; la lista
    entrega solo 3000 g, 71% — muy por debajo del 90%). Sin ningún cap registrado
    (`_CAPS_APPLIED_LAST_RUN` vacío), el mecanismo nuevo debe ser el ÚNICO que explique el
    faltante."""
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 3000.0, text_demand_g=4200.0)
    assert item.get("capped_by") == "qty_reconcile_v7"
    assert "alcanza" in item.get("display_qty", ""), item.get("display_qty")
    assert "alcanza" in item.get("display_string", "")


def test_compra_suficiente_no_recibe_nota():
    """[Riesgo 3] Si lo comprado YA cubre el 90% (o más) de lo que el texto exige, no hay nada
    que avisar — silencio es la respuesta correcta, no un bug."""
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 4000.0, text_demand_g=4200.0)  # 95.2%
    assert item.get("capped_by") is None
    assert "alcanza" not in item.get("display_qty", "")


def test_sin_text_demand_g_es_no_op():
    """Callers que no pasan `text_demand_g` (la mayoría hoy — el plumbing es nuevo) deben ver
    comportamiento previo byte-idéntico: sin el dato de referencia, no hay base para comparar."""
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 100.0, text_demand_g=None)
    assert item.get("capped_by") is None
    assert "alcanza" not in item.get("display_qty", "")


def test_umbral_respeta_el_knob(monkeypatch):
    """Con el knob relajado a 0.5, un déficit del 71% (0.71 >= 0.5) ya NO dispara."""
    monkeypatch.setattr(sc, "QTY_SHORTFALL_NOTE_MIN", 0.5)
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 3000.0, text_demand_g=4200.0)
    assert item.get("capped_by") is None


# ───────────── 2. [Riesgo 1] no pisa un cap real ya registrado ─────────────

def test_no_pisa_un_cap_real_ya_registrado():
    """Si `_CAPS_APPLIED_LAST_RUN` ya explica el déficit con post<pre estricto (cap real de
    almacenaje, ej. P5-VEG-CAP), el `capped_by` debe seguir siendo la razón REAL — el fallback
    sintético sólo debe activarse cuando NINGÚN cap real lo explicó (`_cap_hit is None`)."""
    sc.reset_caps_applied_last_run()
    try:
        sc._record_cap_applied("Cebolla", 2000.0, 600.0, "P5-VEG-CAP")
        item = _por_gramos("Cebolla", 600.0, text_demand_g=4200.0)  # dispararía el sintético solo
        assert item.get("capped_by") == "P5-VEG-CAP", (
            f"el cap sintético piso un cap REAL: capped_by={item.get('capped_by')!r}")
    finally:
        sc.reset_caps_applied_last_run()


# ───────────── 3. [Riesgo 2] composición con P1-SKU-COVER-HONESTY (Task 5) ─────────────

def test_no_duplica_sufijo_con_sku_cover_honesty():
    """Un ítem con envase (`market_packages`/`container_weight_g`) que ADEMÁS calificaría para el
    aviso de Task 5 (`pkg_cover_ratio < PKG_COVER_NOTE_MIN`) no debe terminar con DOS sufijos
    "alcanza" — Task 5 ya se abstiene cuando `capped_by` viene seteado (decisión #3); nuestro
    synthetic cap debe ser justamente lo que dispara esa abstención."""
    sc.reset_caps_applied_last_run()
    # Envase que deliberadamente sub-cubre (cover bajo) PARA que Task 5 evaluaría sufijar si no
    # fuera por nuestro capped_by sintético.
    master = {"market_container": "cartón", "container_weight_g": 200.0}
    item = _por_gramos("Espárragos", 3000.0, master_item=master, text_demand_g=4200.0)
    assert item.get("capped_by") == "qty_reconcile_v7"
    assert item.get("display_qty", "").count("alcanza") == 1, item.get("display_qty")
    assert item.get("display_string", "").count("alcanza") == 1, item.get("display_string")


# ───────────── 4. knob ─────────────

def test_knob_registrado_con_default_09():
    assert sc.QTY_SHORTFALL_NOTE_MIN == pytest.approx(0.9)
    reg = get_knobs_registry_snapshot()
    assert "MEALFIT_QTY_SHORTFALL_NOTE_MIN" in reg
    assert reg["MEALFIT_QTY_SHORTFALL_NOTE_MIN"]["default"] == pytest.approx(0.9)


def test_knob_clamp_rechaza_fuera_de_rango():
    import os
    os.environ["MEALFIT_QTY_SHORTFALL_NOTE_MIN_TEST_OOR"] = "1.5"
    try:
        val = _env_float("MEALFIT_QTY_SHORTFALL_NOTE_MIN_TEST_OOR", 0.9, lambda v: 0.0 < v <= 1.0)
        assert val == pytest.approx(0.9), "1.5 excede el clamp (0,1] -> debe caer al default"
    finally:
        del os.environ["MEALFIT_QTY_SHORTFALL_NOTE_MIN_TEST_OOR"]


# ───────────── 5. plumbing (source-level, offline) ─────────────

def test_plumbing_text_demand_g_parametro_en_apply_smart_market_units():
    import inspect
    sig = inspect.signature(sc.apply_smart_market_units)
    assert "text_demand_g" in sig.parameters
    assert sig.parameters["text_demand_g"].default is None


def test_plumbing_text_demand_g_threading_source():
    """`text_demand_g` se propaga desde `get_shopping_list_delta` -> `aggregate_and_deduct_
    shopping_list` -> los 2 call-sites de `apply_smart_market_units`. Parser-based (espejo de
    `test_cycle_days_plumbing_local` en test_p1_sku_cover_honesty.py) — anclado a texto para que
    un rename futuro falle el test antes que producción."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")

    i = src.index("def aggregate_and_deduct_shopping_list(")
    sig_end = src.index(")", i)
    assert "text_demand_g_map" in src[i:sig_end + 1]

    assert "apply_smart_market_units(name, weight_in_lbs, 'lb', 0.0, master_item, cycle_days=_cycle_days_for_note, text_demand_g=" in src
    assert "apply_smart_market_units(name, 0.0, u, q, master_item, cycle_days=_cycle_days_for_note, text_demand_g=" in src

    j = src.index("def get_shopping_list_delta(")
    j_end = src.index("\ndef ", j + 10)
    assert "expected_sum_from_recipes(" in src[j:j_end]
    assert "text_demand_g_map=" in src[j:j_end]


# ───────────── 6. end-to-end dinámico (get_shopping_list_delta), wiring real ─────────────

def test_e2e_shopping_list_delta_dispara_nota_cuando_texto_infla(monkeypatch):
    """Prueba el WIRING real (no sólo el mecanismo unitario): un plan de 1 día con "600 g de
    espárragos" en una cena, proyectado ×7 (`get_shopping_list_delta` con 1 día materializado
    proyecta a semana). Sin catálogo (worktree sin .env → `get_master_ingredients()` devuelve
    []), la aritmética real del aggregator reproduce fielmente el texto (sin caps ni pantry) — así
    que forzamos la divergencia monkeypencheando `expected_sum_from_recipes` para simular el caso
    real de producción donde el texto exige más de lo que la lista, por la razón que sea, terminó
    comprando. Esto ejercita: (a) `get_shopping_list_delta` arma `text_demand_g_map` desde
    `expected_sum_from_recipes` + `_normalize_food_units_to_base`, (b) lo threadea a través de
    `aggregate_and_deduct_shopping_list`, (c) `apply_smart_market_units` lo consume y estampa la
    nota."""
    plan = {"days": [{"meals": [{
        "ingredients": ["600 g de espárragos"],
        "ingredients_raw": ["600 g de espárragos"],
    }]}]}

    def _fake_expected(plan_data, *, apply_yield=False, multiplier=1.0):
        return {"Espárragos": {"g": 999999.0 * multiplier}}

    monkeypatch.setattr(sc, "expected_sum_from_recipes", _fake_expected)
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("capped_by") == "qty_reconcile_v7", esp
    assert "alcanza" in esp.get("display_qty", ""), esp.get("display_qty")


def test_e2e_shopping_list_delta_sin_divergencia_no_lleva_nota(monkeypatch):
    """Contrapartida honesta del test anterior: si `expected_sum_from_recipes` (mismo parse del
    guard) NO diverge de lo que el aggregator resolvió, no debe aparecer ninguna nota — el
    mecanismo nuevo no es ruidoso por default."""
    plan = {"days": [{"meals": [{
        "ingredients": ["600 g de espárragos"],
        "ingredients_raw": ["600 g de espárragos"],
    }]}]}
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("capped_by") is None, esp
    assert "alcanza" not in esp.get("display_qty", "")


# ───────────── 7. [IMPORTANT ronda 1] emparejamiento por nombre CANÓNICO, no crudo ─────────────
#
# `canonicalize_shopping_food_name`/`_build_shopping_master_map` son el SSOT extraído de
# `aggregate_and_deduct_shopping_list`. `master_map={}` (sin catálogo, worktree offline) es el
# caso realista de este repo en test — las reglas que disparan (`canonicalize_tomate`,
# `canonicalize_cebolla`, `canonicalize_verduras_hoja`) NO dependen del master_map para estos
# alimentos.
#
# ⚠️ Nota de verificación (no sólo leer el código, EJECUTARLO): el revisor midió "100 g de
# espinacas" → 'Espinaca' (singular), pero eso es lo que devuelve `canonicalize_verduras_hoja`
# EN AISLAMIENTO — corriendo la cadena COMPLETA (incluida la cola de 13 regex de consolidación,
# que corre SIEMPRE después), la regla `^espinacas?$` (con el `s` opcional) vuelve a matchear
# sobre el 'Espinaca' recién resuelto y lo revierte a 'Espinacas'. Confirmado byte-a-byte
# idéntico contra el HEAD previo a esta ronda (`git stash` + mismo input a
# `aggregate_and_deduct_shopping_list` antes/después de la extracción) — no es una regresión de
# la extracción, es el comportamiento REAL, preexistente, que el SSOT ahora también expone. Por
# esto el test de espinaca de abajo ancla 'Espinacas' (el valor real), no 'Espinaca'.

def test_canonicaliza_tomates_plural_a_tomate():
    assert sc.canonicalize_shopping_food_name("Tomates", {}) == "Tomate"


def test_canonicaliza_cebollas_plural_a_cebolla():
    assert sc.canonicalize_shopping_food_name("Cebollas", {}) == "Cebolla"


def test_canonicaliza_espinacas_es_net_no_op_por_la_cola_de_regex():
    """Ver nota de verificación arriba: la cola de consolidación revierte 'Espinaca' (singular,
    de `canonicalize_verduras_hoja`) a 'Espinacas' (plural) — el crudo YA coincide con el
    canónico para este alimento específico, así que no hay bug de emparejamiento que corregir
    aquí (a diferencia de tomate/cebolla, donde crudo≠canónico y SÍ hacía falta el fix)."""
    assert sc.canonicalize_shopping_food_name("Espinacas", {}) == "Espinacas"
    assert sc.canonicalize_shopping_food_name("Espinaca", {}) == "Espinacas"


def test_canonicaliza_rucula_plural_a_singular_sin_reversion():
    """Contraparte de espinaca que SÍ demuestra la colapsación plural→singular vía
    `canonicalize_verduras_hoja` sin que la cola de regex la revierta (rúcula no está entre las
    13 reglas de consolidación de cola) — ancla el caso positivo limpio."""
    assert sc.canonicalize_shopping_food_name("Rúculas", {}) == "Rúcula"
    assert sc.canonicalize_shopping_food_name("Rúcula", {}) == "Rúcula"


def test_los_4_alimentos_de_los_planes_reales_no_colapsan_entre_si():
    """Espárragos/tayota/vainitas/calabacín (los 4 alimentos de los planes reales 5f4bb17e/
    8d3f246a/cf3a81fb) no tienen regla de consolidación familiar — deben salir IDÉNTICOS (mismo
    nombre, singular tal cual lo escribió el LLM), confirmando que el SSOT no inventa colisiones
    donde no las había.

    [review final audit-v7-p1 · 2026-08-03 · T7] El cuerpo descartaba el resultado de la forma
    plural (`_ = sc.canonicalize...`) justo en la línea donde el comentario declaraba el
    invariante — la mitad del test que le da nombre («no colapsan ENTRE SÍ») no se comprobaba.
    Ahora se afirma: cada forma resuelve a su propia familia (la singular o la plural del MISMO
    alimento) y jamás al canónico de otro de los 4. Si una regla nueva de la cola de 13 regex
    colapsara «Calabacines» a «Calabaza», o dos de estos alimentos al mismo canónico, sus
    demandas se fusionarían en `text_demand_g_map` y el backstop mediría la suma de dos
    alimentos distintos."""
    familias = (
        ("Espárragos", "Espárragos"),  # ya es invariable en plural en es
        ("Tayota", "Tayotas"),
        ("Vainitas", "Vainitas"),  # invariable
        ("Calabacín", "Calabacines"),
    )
    resueltos: dict[str, str] = {}
    for singular, plural in familias:
        assert sc.canonicalize_shopping_food_name(singular, {}) == singular
        # La forma plural puede o no colapsar a la singular (no hay regla familiar para estos 4);
        # lo que se exige es que se quede DENTRO de su familia y no caiga en la de otro.
        canon_plural = sc.canonicalize_shopping_food_name(plural, {})
        assert canon_plural in (singular, plural), (
            f"'{plural}' resolvió a '{canon_plural}', fuera de su familia — una regla de la cola "
            f"de consolidación se lo llevó a otro alimento")
        resueltos[singular] = singular
        resueltos.setdefault(plural, canon_plural)
    # Ningún par de familias puede compartir canónico.
    canonicos = {sing: sc.canonicalize_shopping_food_name(sing, {}) for sing, _ in familias}
    assert len(set(canonicos.values())) == len(familias), (
        f"dos de los 4 alimentos colapsan al mismo canónico: {canonicos}")


def test_text_demand_g_map_matchea_nombre_canonico_no_crudo(monkeypatch):
    """[IMPORTANT ronda 1 · reproduce el bug y confirma el fix] Antes de esta ronda,
    `text_demand_g_map` quedaba keyed por el nombre CRUDO que devuelve `expected_sum_from_recipes`
    ('Rúculas', plural — así parsea el LLM la mitad de las veces) mientras el ítem comprado
    resuelve por CANÓNICO ('Rúcula', singular, vía `canonicalize_verduras_hoja`) — el `.get(name)`
    en `apply_smart_market_units` fallaba SIEMPRE para esa receta. El mock simula exactamente esa
    key cruda en plural; la receta real usa singular para que el lado comprado resuelva 'Rúcula'
    de forma determinista (sin depender de cómo pluraliza `_parse_quantity`).

    Usa rúcula (no tomate/cebolla, los ejemplos originales del revisor): esos dos también tienen
    un cap REAL por categoría (`_VEG_PER_WEEK_PER_PERSON`, P5-VEG-CAP) que dispara con las
    cantidades de este test y confundiría la aserción (`capped_by` sería 'P5-VEG-CAP', no
    'qty_reconcile_v7' — comportamiento CORRECTO por el riesgo #1, pero no es lo que esta sección
    puntual quiere aislar). Rúcula no tiene cap por categoría."""
    plan = {"days": [{"meals": [{
        "ingredients": ["300 g de rúcula"],
        "ingredients_raw": ["300 g de rúcula"],
    }]}]}

    def _fake_expected_plural(plan_data, *, apply_yield=False, multiplier=1.0):
        # Simula lo que el parser real produce para una receta que SÍ usa plural ("300 g de
        # rúculas"): key cruda 'Rúculas', pre-canonicalización.
        return {"Rúculas": {"g": 999999.0 * multiplier}}

    monkeypatch.setattr(sc, "expected_sum_from_recipes", _fake_expected_plural)
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    ruc = next(i for i in items if "c" in str(i.get("name", "")).lower()
               and "cula" in str(i.get("name", "")).lower())
    assert ruc.get("name") == "Rúcula", (
        f"el lado comprado no resolvió al canónico esperado: {ruc.get('name')!r}")
    assert ruc.get("capped_by") == "qty_reconcile_v7", (
        "la key cruda en plural del mock no emparejó con el canónico 'Rúcula' del lado "
        f"comprado: {ruc}")


# ───────────── 8. [IMPORTANT ronda 1] respeta el kill switch CAPPED_STAPLE_HONESTY ─────────────

def test_respeta_kill_switch_capped_staple_honesty(monkeypatch):
    """El backstop alimenta el MISMO `_cap_hit` que P1-CAPPED-STAPLE-HONESTY y debe respetar su
    kill switch ("Flip a False si el copy molesta; el número NO cambia con el knob, sólo se deja
    de decir"). Reproducido pre-fix: con `CAPPED_STAPLE_HONESTY=False` la nota seguía saliendo
    porque el backstop sintético no miraba el knob — un operador que lo apaga en un incidente no
    conseguía callarla."""
    monkeypatch.setattr(sc, "CAPPED_STAPLE_HONESTY", False)
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 3000.0, text_demand_g=4200.0)
    assert item.get("capped_by") is None, (
        f"con CAPPED_STAPLE_HONESTY=False no debe quedar capped_by: {item.get('capped_by')!r}")
    assert "alcanza" not in item.get("display_qty", ""), item.get("display_qty")
    assert "alcanza" not in item.get("display_string", ""), item.get("display_string")


def test_kill_switch_no_rompe_un_cap_real_existente(monkeypatch):
    """Control: el kill switch apaga TODO el mecanismo (real + sintético) simétricamente — no es
    una regresión nueva, es el mismo comportamiento que ya tenía P1-CAPPED-STAPLE-HONESTY para
    caps reales, sólo que ahora el sintético lo respeta también."""
    monkeypatch.setattr(sc, "CAPPED_STAPLE_HONESTY", False)
    sc.reset_caps_applied_last_run()
    try:
        sc._record_cap_applied("Cebolla", 2000.0, 600.0, "P5-VEG-CAP")
        item = _por_gramos("Cebolla", 600.0, text_demand_g=4200.0)
        assert item.get("capped_by") is None
        assert "alcanza" not in item.get("display_qty", "")
    finally:
        sc.reset_caps_applied_last_run()


# ───────────── 9. [MINOR ronda 1] SSOT de canonicalización compartido (plumbing) ─────────────

def test_plumbing_canonicalize_shopping_food_name_es_ssot_compartido():
    """Parser-based: `aggregate_and_deduct_shopping_list` debe llamar al MISMO
    `canonicalize_shopping_food_name` que usa `get_shopping_list_delta` — si un futuro refactor
    reintroduce una segunda copia de la cadena inline, este test debe fallar antes que produzca
    drift silencioso entre los dos lados (el bug exacto de la sección 7 de arriba)."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    assert "def canonicalize_shopping_food_name(name: str, master_map: dict) -> str:" in src
    assert "def _build_shopping_master_map() -> dict:" in src
    i = src.index("def aggregate_and_deduct_shopping_list(")
    j = src.index("\ndef ", i + 10)
    assert "canonicalize_shopping_food_name(name, master_map)" in src[i:j]
    assert "_build_shopping_master_map()" in src[i:j]


def test_plumbing_ventaneo_recibe_text_demand_g_map():
    """[MINOR ronda 1] La segunda pasada (ventaneo de perecederos, OFF por default) también debe
    recibir el mapa — ya calculado sobre el plan completo, no hay razón para no pasarlo.

    [review final · 2026-08-03] Recibe el mapa YA GATEADO (`_tdg_para_agg`), no el crudo: si la
    lista es un delta con deducción de Nevera, la segunda pasada tampoco puede comparar bruto
    contra neto. Las dos pasadas deben ver exactamente la misma decisión."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("_res_window = aggregate_and_deduct_shopping_list(")
    j = src.index(")", src.index("cycle_days=cycle_days,", i))
    assert "text_demand_g_map=_tdg_para_agg" in src[i:j + 1]


# ───────── 10. [CRITICAL review final] el sello sintético NO puede cegar al coherence guard ─────
#
# El guard es la ÚNICA superficie que puede forzar retry (columna «Bloquea retry? = Sí» de
# `coherence_surfaces_table.md`) y corre en modo `block` por default. Estos tests son de EFECTO:
# llaman al guard real y miran la lista de divergencias que devuelve. Un test que sólo comprobara
# que el campo se llama distinto no habría cazado nada — el defecto original era precisamente que
# el nombre del campo era correcto para el otro productor.

@pytest.fixture
def _sin_master_db(monkeypatch):
    """Stub de `get_master_ingredients` a `[]` — mismo patrón que
    `test_p1_shopping_recipe_coherence.py::no_master_db`. Sin esto el guard intenta cargar el
    master_map y golpea el pool de DB (que en el worktree no existe): ensucia logs y añade latencia.
    Con `[]` corre el fallback sólo-reglas-inline, suficiente para el contrato del guard."""
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [])
    yield


def _plan_con_drift(item_extra: dict | None = None) -> dict:
    """Plan de 1 día cuya receta pide 2000 g de pollo y cuya lista sólo compra 1000 g — drift de
    magnitud del 50%, muy por encima de la tolerancia default del guard (0,10)."""
    item = {
        "name": "Pollo",
        "base_qty": 1000.0,
        "base_unit": "g",
        "market_qty_numeric": 2.2,
        "market_unit": "lb",
    }
    item.update(item_extra or {})
    return {
        "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ["2000 g pollo"]}]}],
        "aggregated_shopping_list": [item],
    }


def test_el_sello_sintetico_NO_apaga_la_divergencia_de_magnitud(_sin_master_db, monkeypatch):
    """EL test del hallazgo CRITICAL. Un ítem con el sello sintético —incluso uno PERSISTIDO por
    una versión anterior, que sí traía `capped_pre`— debe seguir produciendo divergencia."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    plan = _plan_con_drift({"capped_by": "qty_reconcile_v7",
                            "capped_pre": 2000.0, "capped_post": 1000.0})
    divs = sc.run_shopping_coherence_guard(plan, multiplier=1.0)
    magnitud = [d for d in divs if d.get("magnitude")]
    assert magnitud, (
        "el sello sintético cegó al guard: sin divergencia no hay retry, no hay degradación y "
        f"no hay fila en _shopping_coherence_block_history. divs={divs}")
    assert magnitud[0]["expected_qty"] == pytest.approx(2000.0)
    assert magnitud[0]["actual_qty"] == pytest.approx(1000.0), (
        "el guard sigue sustituyendo la cantidad comprada por `capped_pre`")
    assert magnitud[0]["delta_pct"] == pytest.approx(0.5)
    # ...y en modo block la divergencia debe llegar al flag que consume `review_plan_node`.
    assert plan.get("_shopping_coherence_block"), plan.keys()


def test_un_cap_REAL_sigue_silenciando_la_divergencia(_sin_master_db, monkeypatch):
    """Contrafactual obligatorio: la exclusión debe ser QUIRÚRGICA. Si silenciara también los caps
    reales, habría revertido P1-COHERENCE-CAPPED-PRE (los topes de perecederos son una decisión de
    producto, se le comunican al usuario y NO son incoherencias)."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    plan = _plan_con_drift({"capped_by": "P5-VEG-CAP",
                            "capped_pre": 2000.0, "capped_post": 1000.0})
    divs = sc.run_shopping_coherence_guard(plan, multiplier=1.0)
    assert [d for d in divs if d.get("magnitude")] == [], divs


def test_el_item_que_produce_el_agregador_real_tampoco_ciega_al_guard(_sin_master_db, monkeypatch):
    """E2E del productor al consumidor: se construye el ítem con la función REAL (no a mano) y se
    lo pasa al guard REAL. Cierra el hueco de que el fix viva sólo en el lado del guard."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    sc.reset_caps_applied_last_run()
    item = sc.apply_smart_market_units(
        "Pollo", 1000.0 / 453.592, "lb", 0.0, master_item=None, cycle_days=7,
        text_demand_g=2000.0)
    assert item.get("capped_by") == "qty_reconcile_v7", item
    plan = {
        "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ["2000 g pollo"]}]}],
        "aggregated_shopping_list": [item],
    }
    divs = [d for d in sc.run_shopping_coherence_guard(plan, multiplier=1.0) if d.get("magnitude")]
    assert divs and divs[0]["actual_qty"] == pytest.approx(1000.0, abs=1.0), divs


def test_el_deficit_sintetico_viaja_por_claves_propias():
    """El canal `capped_pre`/`capped_post` es propiedad exclusiva de los caps REALES de
    `_CAPS_APPLIED_LAST_RUN`. El sintético usa `shortfall_*`, que ningún consumidor del guard lee."""
    sc.reset_caps_applied_last_run()
    item = _por_gramos("Espárragos", 3000.0, text_demand_g=4200.0)
    assert item.get("capped_by") == "qty_reconcile_v7"
    assert item.get("capped_pre") is None, "el sintético NO debe escribir el canal del guard"
    assert item.get("capped_post") is None
    assert item.get("shortfall_text_g") == pytest.approx(4200.0)
    assert item.get("shortfall_bought_g") == pytest.approx(3000.0, abs=1.0)


def test_un_cap_real_SI_escribe_capped_pre():
    """Simétrico del anterior: el arreglo no debe haberle quitado el canal a los caps reales (eso
    reintroduciría el falso positivo que P1-COHERENCE-CAPPED-PRE cerró)."""
    sc.reset_caps_applied_last_run()
    try:
        sc._record_cap_applied("Cebolla", 2000.0, 600.0, "P5-VEG-CAP")
        item = _por_gramos("Cebolla", 600.0, text_demand_g=2000.0)
        assert item.get("capped_by") == "P5-VEG-CAP"
        assert item.get("capped_pre") == pytest.approx(2000.0)
        assert item.get("capped_post") == pytest.approx(600.0)
        assert item.get("shortfall_text_g") is None
    finally:
        sc.reset_caps_applied_last_run()


def test_la_razon_sintetica_es_una_constante_compartida():
    """Productor y consumidor (la exclusión del guard) deben leer el MISMO nombre. Dos literales
    iguales en dos puntas del archivo son exactamente cómo se pierde una exclusión en el próximo
    rename."""
    from pathlib import Path
    assert sc.QTY_RECONCILE_SYNTHETIC_REASON == "qty_reconcile_v7"
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("def _extract_aggregated_food_dict(")
    j = src.index("\ndef ", i + 10)
    assert "QTY_RECONCILE_SYNTHETIC_REASON" in src[i:j], (
        "la exclusión del sello sintético desapareció del constructor del lado ACTUAL del guard")


# ───────── 11. [IMPORTANT review final] el backstop sólo compara magnitudes homogéneas ─────────
#
# ⚠️ El plan de estos tests tiene 7 días MATERIALIZADOS a propósito. Con un plan de 1 día,
# `get_shopping_list_delta` proyecta a la semana (`base_duration_scale = 7/num_days = 7`) y la
# deducción de la Nevera queda diluida ×7 — mi primera versión de este test usaba 1 día y PASABA
# también con el código pre-fix, o sea que no probaba nada. Con 7 días la escala es 1 y los números
# son literalmente los del caso reportado: receta 2100 g, nevera 500 g, compra 1600 g (76%).

def _plan_7d_esparragos(gramos_por_dia: float = 300.0) -> dict:
    return {"days": [{"meals": [{
        "ingredients": [f"{gramos_por_dia:g} g de espárragos"],
        "ingredients_raw": [f"{gramos_por_dia:g} g de espárragos"],
    }]} for _ in range(7)]}


def _expected_fijo(total_g: float):
    def _fake(plan_data, *, apply_yield=False, multiplier=1.0):
        return {"Espárragos": {"g": total_g * multiplier}}
    return _fake


def test_con_nevera_el_backstop_se_calla(monkeypatch):
    """EL test del hallazgo IMPORTANT, reproducido con la función real: la receta pide 2100 g, el
    usuario tiene 500 g en la Nevera y la lista (delta) compra 1600 g. Sin el gate, el backstop
    comparaba 1600 (NETO) contra 2100 (BRUTO) = 76% < 90% y estampaba una nota de recompra sobre
    algo que el usuario ya tiene en casa."""
    monkeypatch.setattr(sc, "expected_sum_from_recipes", _expected_fijo(2100.0))
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(
        None, _plan_7d_esparragos(), False, False, True, 1.0,
        inventory_override=[{"ingredient_name": "Espárragos", "quantity": 500, "unit": "g"}],
        consumed_override=[],
    )
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("base_qty") == pytest.approx(1600.0, abs=1.0), (
        f"el fixture ya no reproduce el caso (compra neta esperada 1600 g): {esp}")
    assert esp.get("capped_by") is None, (
        f"nota de recompra sobre lo que el usuario YA tiene en la Nevera: {esp}")
    assert "alcanza" not in esp.get("display_qty", ""), esp.get("display_qty")


def test_sin_nevera_el_delta_conserva_el_backstop(monkeypatch):
    """Contrafactual: el gate es por «¿hubo deducción?», NO por `is_new_plan`. Un delta con la
    nevera vacía compara neto contra neto (no se restó nada), así que el mecanismo debe seguir
    vivo — gatear por `is_new_plan` lo habría matado en este caso legítimo."""
    plan = {"days": [{"meals": [{
        "ingredients": ["600 g de espárragos"],
        "ingredients_raw": ["600 g de espárragos"],
    }]}]}

    def _fake_expected(plan_data, *, apply_yield=False, multiplier=1.0):
        return {"Espárragos": {"g": 999999.0 * multiplier}}

    monkeypatch.setattr(sc, "expected_sum_from_recipes", _fake_expected)
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(
        None, plan, False, False, True, 1.0,
        inventory_override=[], consumed_override=[],
    )
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("capped_by") == "qty_reconcile_v7", esp


def test_con_consumidos_el_backstop_tambien_se_calla(monkeypatch):
    """`items_to_deduct` = nevera + CONSUMIDOS. La segunda mitad deduce igual, así que el gate
    tiene que mirar la lista combinada y no sólo el inventario físico."""
    monkeypatch.setattr(sc, "expected_sum_from_recipes", _expected_fijo(2100.0))
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(
        None, _plan_7d_esparragos(), False, False, True, 1.0,
        inventory_override=[],
        consumed_override=["500 g de espárragos"],
    )
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("base_qty") == pytest.approx(1600.0, abs=1.0), esp
    assert esp.get("capped_by") is None, esp


def test_la_lista_canonica_conserva_el_backstop(monkeypatch):
    """La lista canónica (`is_new_plan=True`) fuerza `items_to_deduct` vacío por construcción — es
    el camino donde se MIDIÓ el caso original (espárragos del plan 5f4bb17e) y donde el mecanismo
    debe seguir intacto."""
    plan = {"days": [{"meals": [{
        "ingredients": ["600 g de espárragos"],
        "ingredients_raw": ["600 g de espárragos"],
    }]}]}

    def _fake_expected(plan_data, *, apply_yield=False, multiplier=1.0):
        return {"Espárragos": {"g": 999999.0 * multiplier}}

    monkeypatch.setattr(sc, "expected_sum_from_recipes", _fake_expected)
    sc.reset_caps_applied_last_run()
    items = sc.get_shopping_list_delta(
        None, plan, True, False, True, 1.0,
        inventory_override=[{"ingredient_name": "Espárragos", "quantity": 500, "unit": "g"}],
    )
    esp = next(i for i in items if "esp" in str(i.get("name", "")).lower())
    assert esp.get("capped_by") == "qty_reconcile_v7", esp
