"""[P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] El guard eximía el sub-suministro REAL
etiquetándolo `pantry_overdeduct` — una hipótesis imposible en las listas que evalúa.

## El agujero

`_classify_divergence_hypothesis` asignaba `pantry_overdeduct` a CUALQUIER
`0 < actual < expected × 0.5` **por umbral puro**, sin preguntar si hubo deducción de
nevera. Y `_has_severe_divergence` exime esa hipótesis de la escalada warn→block
(P1-COHERENCE-SEVERE-NO-NOISE, 2026-07-07: el artefacto conocido del delta contra
inventario no debe quemar retries).

Pero las superficies del guard comparan listas **CANÓNICAS** (`is_new_plan=True` fuerza
`physical_inventory=[]` y `consumed_ingredients=[]`, P3-CANONICAL-AGG-WEEKLY): ahí no hay
lado inventario que pueda deducir de más, así que la hipótesis es **imposible por
construcción** y el sub-suministro genuino heredaba la exención PARA SIEMPRE. Todo el
rango `0 < ratio < 0.5` era invisible a la escalada.

Caso vivo del audit: **espárragos 583,33 g comprados contra 1.400 g que las recetas
exigen — el 41,7%**. Lista canónica, cero deducción, y el guard lo archivaba como «la
nevera dedujo de más».

## El fix

El aggregator SABE si dedujo, así que estampa el sello `pantry_deduction_applied` en cada
ítem (gemelo exacto de `trip_window_days` y de `protein_yield_applied` de Task 14) y el
guard lo LEE de la lista que tiene delante en vez de adivinar. Sin deducción, el mismo
rango se clasifica `magnitude_undersupply` y SÍ cuenta como severa.

La exención original queda intacta donde era correcta: con deducción REAL, el ratio bajo
sigue siendo `pantry_overdeduct` y sigue sin escalar.

## Composición con el backstop de Task 8

Un ítem con déficit de texto lleva `shortfall_text_g`/`shortfall_bought_g` y — a propósito
— NO escribe `capped_pre`, así que `_extract_aggregated_food_dict` presenta al guard la
cantidad REALMENTE comprada (corta). Ese es exactamente el caso que esta hipótesis debe
marcar severo: en GENERACIÓN el plan todavía puede mejorar, y el retry es la respuesta
correcta. La nota del backstop es para listas que YA se entregaron.

tooltip-anchor: P2-GUARD-UNDERSUPPLY-CANONICAL
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import shopping_calculator as sc
from knobs import get_knobs_registry_snapshot

_SRC = (Path(sc.__file__).resolve().parent / "shopping_calculator.py").read_text(encoding="utf-8")

# El caso vivo del audit: 583,33 g comprados de 1.400 g exigidos = 41,7%.
_EXP_G = 1400.0
_ACT_G = 583.33
_DELTA = abs(_ACT_G - _EXP_G) / _EXP_G  # 0.583 — por encima del umbral severo 0.50


@pytest.fixture(autouse=True)
def no_master_db():
    """OFFLINE: el `.env` apunta a PRODUCCIÓN y el worktree no tiene `.env`.
    Mismo patrón que test_p2_protein_yield_canonical.py / test_p1_shopping_recipe_coherence.py."""
    with patch.object(sc, "get_master_ingredients", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _clean_knob(monkeypatch):
    monkeypatch.delenv("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)


def _div(hypothesis, *, delta=_DELTA, expected=_EXP_G, actual=_ACT_G, magnitude=True):
    return {
        "food": "Espárragos", "unit": "g", "hypothesis": hypothesis,
        "magnitude": magnitude, "delta_pct": delta,
        "expected_qty": expected, "actual_qty": actual,
    }


# ---------------------------------------------------------------------------
# 0. Marker
# ---------------------------------------------------------------------------
def test_marker_present():
    assert "P2-GUARD-UNDERSUPPLY-CANONICAL" in _SRC


# ---------------------------------------------------------------------------
# 1. El clasificador — los dos casos ancla
# ---------------------------------------------------------------------------
class TestClasificador:
    def test_undersupply_canonico_es_undersupply(self):
        """Sin deducción de nevera, 41,7% de lo exigido es sub-suministro, no overdeduct."""
        h = sc._classify_divergence_hypothesis(
            _EXP_G, _ACT_G, {"g": _EXP_G}, {"g": _ACT_G},
            food="Espárragos", pantry_deduction_applied=False,
        )
        assert h == "magnitude_undersupply"

    def test_con_nevera_conserva_pantry_overdeduct(self):
        """El MISMO ratio CON deducción real sigue siendo el artefacto conocido."""
        h = sc._classify_divergence_hypothesis(
            _EXP_G, _ACT_G, {"g": _EXP_G}, {"g": _ACT_G},
            food="Espárragos", pantry_deduction_applied=True,
        )
        assert h == "pantry_overdeduct"

    def test_default_del_parametro_es_conservador(self):
        """Caller no migrado (sin el kwarg) → comportamiento previo byte-idéntico."""
        h = sc._classify_divergence_hypothesis(
            _EXP_G, _ACT_G, {"g": _EXP_G}, {"g": _ACT_G}, food="Espárragos",
        )
        assert h == "pantry_overdeduct"

    def test_precedencia_yield_intacta(self):
        """Ratio 0,35 cae en la banda de yield de legumbre y ahí SIGUE — la
        hipótesis nueva es el paso 4, no puede robarle al 3."""
        h = sc._classify_divergence_hypothesis(
            1000.0, 350.0, {"g": 1000.0}, {"g": 350.0},
            food="Lentejas", pantry_deduction_applied=False,
        )
        assert h == "yield_uncovered"

    def test_precedencia_cap_swallowed_intacta(self):
        """Ausencia total sigue siendo `cap_swallowed_modifier` (paso 1)."""
        h = sc._classify_divergence_hypothesis(
            _EXP_G, 0.0, {"g": _EXP_G}, {}, food="Espárragos", pantry_deduction_applied=False,
        )
        assert h == "cap_swallowed_modifier"

    def test_sobreoferta_no_toca_la_hipotesis_nueva(self):
        """`act > exp` jamás es sub-suministro (sigue `unknown`, sobre-oferta de envase)."""
        h = sc._classify_divergence_hypothesis(
            100.0, 900.0, {"g": 100.0}, {"g": 900.0},
            food="Arroz", pantry_deduction_applied=False,
        )
        assert h == "unknown"


# ---------------------------------------------------------------------------
# 2. Severidad
# ---------------------------------------------------------------------------
class TestSeveridad:
    def test_default_del_knob_es_telemetry_first(self, monkeypatch):
        """[ronda 1] El default arranca APAGADO: la clasificación llega el día uno, la
        ESCALADA espera a medir volumen. Motivo medido: en T2 un `block_set` re-corre un
        cómputo determinista (mismos días, mismo inventario, mismo catálogo) → los 3 intentos
        fallan igual → re-encolado → dead letter. Es el modo de fallo de
        P1-COHERENCE-SEVERE-NO-NOISE, y la población que llega a T2 con sub-oferta ≥0.5 es
        justo la que la surface #1 ya no pudo arreglar regenerando."""
        assert sc._get_guard_undersupply_severe_knob() is False
        assert sc._has_severe_divergence([_div("magnitude_undersupply")]) is False

    def test_undersupply_es_severo_con_knob_on(self, monkeypatch):
        monkeypatch.setenv("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", "true")
        assert sc._has_severe_divergence([_div("magnitude_undersupply")]) is True

    def test_pantry_overdeduct_sigue_exento(self):
        """Regresión P1-COHERENCE-SEVERE-NO-NOISE: la exención original NO se reintroduce."""
        assert sc._has_severe_divergence([_div("pantry_overdeduct")]) is False

    def test_unknown_sobreoferta_sigue_exento(self):
        """Regresión P1-COHERENCE-SEVERE-NO-NOISE: 67 `unknown` de envase no bloquean."""
        divs = [_div("unknown", delta=8.0, expected=100.0, actual=900.0) for _ in range(67)]
        assert sc._has_severe_divergence(divs) is False

    def test_recipe_unquantified_inf_sigue_exento(self):
        """Regresión P1-COHERENCE-INF-NOT-SEVERE: sal/pimienta `inf` no bloquean."""
        divs = [_div("recipe_unquantified", delta=float("inf"), expected=0.0, actual=5.0)]
        assert sc._has_severe_divergence(divs) is False

    @pytest.mark.parametrize("knob", ["false", "true"])
    def test_el_knob_no_toca_la_etiqueta(self, monkeypatch, knob):
        """El knob gobierna la CONSECUENCIA (escalada), nunca la clasificación: con él
        apagado la observabilidad sigue completa — es la medición que justificaría
        encenderlo."""
        monkeypatch.setenv("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", knob)
        h = sc._classify_divergence_hypothesis(
            _EXP_G, _ACT_G, {"g": _EXP_G}, {"g": _ACT_G},
            food="Espárragos", pantry_deduction_applied=False,
        )
        assert h == "magnitude_undersupply"

    def test_knob_registrado(self):
        sc._get_guard_undersupply_severe_knob()
        assert "MEALFIT_GUARD_UNDERSUPPLY_SEVERE" in get_knobs_registry_snapshot()


# ---------------------------------------------------------------------------
# 3. Threading por `compare_expected_vs_aggregated`
# ---------------------------------------------------------------------------
class TestCompareThreading:
    def test_flag_false_propaga(self):
        divs = sc.compare_expected_vs_aggregated(
            {"Espárragos": {"g": _EXP_G}}, {"Espárragos": {"g": _ACT_G}},
            tolerance=0.10, pantry_deduction_applied=False,
        )
        assert len(divs) == 1
        assert divs[0]["hypothesis"] == "magnitude_undersupply"

    def test_flag_true_propaga(self):
        divs = sc.compare_expected_vs_aggregated(
            {"Espárragos": {"g": _EXP_G}}, {"Espárragos": {"g": _ACT_G}},
            tolerance=0.10, pantry_deduction_applied=True,
        )
        assert divs[0]["hypothesis"] == "pantry_overdeduct"

    def test_default_conservador(self):
        divs = sc.compare_expected_vs_aggregated(
            {"Espárragos": {"g": _EXP_G}}, {"Espárragos": {"g": _ACT_G}}, tolerance=0.10,
        )
        assert divs[0]["hypothesis"] == "pantry_overdeduct"


# ---------------------------------------------------------------------------
# 4. El sello: tri-estado (True / False / ausente)
# ---------------------------------------------------------------------------
class TestSello:
    def test_sello_true(self):
        assert sc._pantry_deduction_seal([{"name": "X", "pantry_deduction_applied": True}]) is True

    def test_sello_false(self):
        assert sc._pantry_deduction_seal([{"name": "X", "pantry_deduction_applied": False}]) is False

    def test_sello_ausente_es_none(self):
        """Lista vieja (persistida antes de este P-fix): no sabemos → `None`, y el
        caller cae al default conservador."""
        assert sc._pantry_deduction_seal([{"name": "X"}]) is None
        assert sc._pantry_deduction_seal([]) is None

    def test_un_solo_true_gana(self):
        lst = [{"name": "A", "pantry_deduction_applied": False},
               {"name": "B", "pantry_deduction_applied": True}]
        assert sc._pantry_deduction_seal(lst) is True

    def test_aggregator_estampa_false_sin_deduccion(self):
        res = sc.aggregate_and_deduct_shopping_list(["200 g de pollo"], [], structured=True)
        assert res, "el aggregator devolvió lista vacía"
        assert all(i.get("pantry_deduction_applied") is False for i in res)
        assert sc._pantry_deduction_seal(res) is False

    def test_aggregator_estampa_true_con_deduccion_efectiva(self):
        res = sc.aggregate_and_deduct_shopping_list(
            ["500 g de pollo"], ["100 g de pollo"], structured=True)
        assert res
        assert sc._pantry_deduction_seal(res) is True

    def test_deduccion_de_cero_no_cuenta(self):
        """`items_to_deduct` con cantidad 0 («Sal al gusto» en la Nevera) no dedujo NADA:
        la lista sigue siendo canónica y el sub-suministro sigue siendo real."""
        res = sc.aggregate_and_deduct_shopping_list(
            ["500 g de pollo"], ["0 g de pollo"], structured=True)
        assert res
        assert sc._pantry_deduction_seal(res) is False

    def test_lista_canonica_del_delta_lleva_sello_false(self):
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                     "ingredients_raw": ["200 g de pollo"]}]}]}
        items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
        assert items
        assert sc._pantry_deduction_seal(items) is False

    def test_delta_con_nevera_fisica_lleva_sello_true(self):
        """[ronda 1] El otro camino real hacia `items_to_deduct`: `physical_inventory`
        (la Nevera del usuario), que el delta formatea a "<qty> <unit> de <nombre>". El
        sello tiene que verlo igual que a los consumidos — si solo cubriera consumidos,
        una lista deducida por Nevera se declararía canónica y su sub-suministro (que SÍ
        puede venir de la deducción) escalaría como si no hubiera nevera."""
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                     "ingredients_raw": ["500 g de pollo"]}]}]}
        items = sc.get_shopping_list_delta(
            None, plan, False, False, True, 1.0,
            inventory_override=[{"ingredient_name": "Pollo", "quantity": 100, "unit": "g"}],
            consumed_override=[],
        )
        assert items
        assert sc._pantry_deduction_seal(items) is True

    def test_salida_categorizada_conserva_el_sello(self):
        """`categorize=True` devuelve `{categoría: [items]}` en vez de una lista plana —
        es la forma que consumen varias surfaces. `results` y `categorized_results[*]`
        comparten los mismos dicts, así que el sello debe estar en ambas."""
        res = sc.aggregate_and_deduct_shopping_list(
            ["200 g de pollo"], [], categorize=True, structured=True)
        assert isinstance(res, dict) and res
        flat = [it for items in res.values() for it in items]
        assert flat, "la salida categorizada vino vacía"
        assert all(i.get("pantry_deduction_applied") is False for i in flat)
        assert sc._pantry_deduction_seal(flat) is False


# ---------------------------------------------------------------------------
# 5. E2E del guard — el sello decide la hipótesis
# ---------------------------------------------------------------------------
def _plan(agg_items, ingredients_raw=("1400 g de espárragos",)):
    return {
        "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": list(ingredients_raw)}]}],
        "aggregated_shopping_list": agg_items,
    }


def _asparagus_item(**extra):
    item = {"name": "Espárragos", "market_qty_numeric": _ACT_G, "market_unit": "g",
            "base_qty": _ACT_G, "base_unit": "g"}
    item.update(extra)
    return item


class TestGuardE2E:
    def _run(self, plan, mode="warn"):
        sc.reset_caps_applied_last_run()
        return sc.run_shopping_coherence_guard(plan, mode_override=mode)

    def test_lista_canonica_sellada_marca_undersupply(self, monkeypatch):
        plan = _plan([_asparagus_item(pantry_deduction_applied=False)])
        divs = [d for d in self._run(plan) if d.get("magnitude")]
        assert len(divs) == 1, f"esperaba 1 divergencia de magnitud, salieron {divs}"
        assert divs[0]["hypothesis"] == "magnitude_undersupply"
        # Severa solo con el knob encendido (default telemetry-first, ver TestSeveridad).
        assert sc._has_severe_divergence(divs) is False
        monkeypatch.setenv("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", "true")
        assert sc._has_severe_divergence(divs) is True

    def test_lista_deducida_sellada_conserva_exencion(self):
        plan = _plan([_asparagus_item(pantry_deduction_applied=True)])
        divs = [d for d in self._run(plan) if d.get("magnitude")]
        assert len(divs) == 1
        assert divs[0]["hypothesis"] == "pantry_overdeduct"
        assert sc._has_severe_divergence(divs) is False

    def test_lista_sin_sello_cae_al_default_conservador(self):
        """Plan persistido ANTES de este P-fix: sin sello, comportamiento previo."""
        plan = _plan([_asparagus_item()])
        divs = [d for d in self._run(plan) if d.get("magnitude")]
        assert len(divs) == 1
        assert divs[0]["hypothesis"] == "pantry_overdeduct"

    def test_block_marca_critico(self):
        plan = _plan([_asparagus_item(pantry_deduction_applied=False)])
        self._run(plan, mode="block")
        block = plan.get("_shopping_coherence_block")
        assert block, "el sub-suministro del 41,7% debe entrar al subset crítico"
        # El guard canonicaliza a singular ("Espárrago") antes de comparar.
        assert any(str(d["food"]).startswith("Espárrago") for d in block)

    def test_composicion_con_backstop_task_8(self, monkeypatch):
        """Ítem con el sello sintético `qty_reconcile_v7`: escribe `shortfall_*` y NO
        `capped_pre`, así que el guard ve la cantidad comprada REAL (corta) y la marca
        severa. En generación (assemble) el retry es la respuesta correcta — la nota del
        backstop es para listas ya entregadas."""
        monkeypatch.setenv("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", "true")
        plan = _plan([_asparagus_item(
            pantry_deduction_applied=False,
            capped_by=sc.QTY_RECONCILE_SYNTHETIC_REASON,
            shortfall_text_g=_EXP_G, shortfall_bought_g=_ACT_G,
        )])
        divs = [d for d in self._run(plan) if d.get("magnitude")]
        assert len(divs) == 1, f"el sello sintético no debe ocultar la compra corta: {divs}"
        assert divs[0]["actual_qty"] == pytest.approx(_ACT_G, rel=0.01), (
            "si el guard leyera `capped_pre` aquí, compararía el esperado contra sí mismo"
        )
        assert divs[0]["hypothesis"] == "magnitude_undersupply"
        assert sc._has_severe_divergence(divs) is True

    def test_helper_con_block_severe_only_setea_block_set(self, monkeypatch):
        """[ronda 1] El camino REAL de la surface #3: `run_shopping_coherence_guard_and_
        append_history(..., block_severe_only=True)` escala warn→block cuando hay severas.
        Con el knob ON, el sub-suministro canónico lo dispara; con el default (OFF) no.
        Es el bit del que cuelga el retry loop de `_chunk_worker` T2, así que merece test
        propio y no solo la unidad `_has_severe_divergence`."""
        sc.reset_caps_applied_last_run()
        plan_on = _plan([_asparagus_item(pantry_deduction_applied=False)])
        monkeypatch.setenv("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", "true")
        divs, block_set = sc.run_shopping_coherence_guard_and_append_history(
            plan_on, mode_override="warn", block_severe_only=True)
        assert any(d.get("hypothesis") == "magnitude_undersupply" for d in divs)
        assert block_set is True
        assert plan_on.get("_shopping_coherence_block")

        sc.reset_caps_applied_last_run()
        plan_off = _plan([_asparagus_item(pantry_deduction_applied=False)])
        monkeypatch.setenv("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", "false")
        divs_off, block_set_off = sc.run_shopping_coherence_guard_and_append_history(
            plan_off, mode_override="warn", block_severe_only=True)
        assert block_set_off is False
        assert "_shopping_coherence_block" not in plan_off
        # …pero la telemetría del history SÍ llega: es la medición con la que se decidirá
        # encender el knob. Apagar la escalada nunca debe apagar la observabilidad.
        assert any(d.get("hypothesis") == "magnitude_undersupply" for d in divs_off)
        hist = plan_off.get("_shopping_coherence_block_history") or []
        assert hist and hist[-1]["hypotheses"].get("magnitude_undersupply") == 1


# ---------------------------------------------------------------------------
# 6. El banner del usuario no pierde el caso accionable
# ---------------------------------------------------------------------------
def test_banner_sigue_surfaceando_el_sub_suministro():
    """Pre-fix estas divergencias salían como `pantry_overdeduct`, que ES accionable
    («te puedes quedar corto»). Renombrarlas sin actualizar el set las habría borrado
    del banner en silencio — la misma clase de regresión que P1-COHERENCE-BANNER-NOISE
    existe para evitar."""
    out = sc.summarize_divergences_for_ui([_div("magnitude_undersupply")])
    assert len(out) == 1
    assert out[0]["hypothesis"] == "magnitude_undersupply"
