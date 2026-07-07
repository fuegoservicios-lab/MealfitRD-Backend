"""[P1-shop-coh-1 · 2026-05-07] Tests E2E del guard recetas↔lista.

Cierra el bucle de coherencia documentado en project_audit_p0_p1_close_2026_05_07.md:
hasta este punto el aggregator funcionaba (P0-11 clamp, P1-2 yield, P2-PDF-1
perecederos, etc) pero ningún test contrastaba `Σ(ingredientes_recetas)` vs
`Σ(items_lista_compras)`. Si el LLM produjera un plan donde un food de las
recetas no llegara a la lista (cap_swallowed) o uno extra apareciera (fantasma),
nadie lo detectaba hasta que el usuario notaba comprar de menos/más.

Cobertura por capa:

  1. `expected_sum_from_recipes(plan)` — extracción simétrica al aggregator.
  2. `compare_expected_vs_aggregated(exp, agg)` — heurísticos de hipótesis.
  3. Knobs `MEALFIT_SHOPPING_COHERENCE_GUARD` / `_TOLERANCE_PCT` — parsing seguro.
  4. `run_shopping_coherence_guard(plan_result)` — orquestación end-to-end con
     filtro de staples/urgentes y modos off/warn/block.
"""
import pytest

import shopping_calculator
from shopping_calculator import (
    expected_sum_from_recipes,
    compare_expected_vs_aggregated,
    run_shopping_coherence_guard,
    _get_coherence_guard_mode,
    _get_coherence_tolerance_pct,
)


# ---------------------------------------------------------------------------
# Helper: silenciar el fetch a master_ingredients que requiere DB
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    """Stub `get_master_ingredients` a `[]`.

    Sin esto, los tests del guard tiran un ERROR repetido al pool de DB
    cuando intenta cargar el master_map. El helper ya maneja la excepción
    pero ensucia los logs y añade latencia. Stub a [] activa el fallback
    sólo-reglas-inline (huevo/ñame/miel/ajo) que es suficiente para verificar
    el contrato del guard sin DB.

    [P3-F · 2026-05-08] `autouse=True` aplica el stub a TODOS los tests del
    archivo, no solo los que lo declaran como parámetro. Antes,
    `TestExpectedSumMultiplier` no pedía el fixture explícitamente y tiraba
    `KeyError: 'Pollo'` cuando otros tests del run habían inicializado el
    pool de DB y `_parse_quantity` devolvía "Pechuga de pollo" via
    master_map. Ese drift era ambiental (pasa en CI/pytest, no en CLI) —
    autouse cierra la inconsistencia. Mismo patrón que
    `test_p3_a_coherence_multiplier_e2e.py`.
    """
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


# ---------------------------------------------------------------------------
# 1. expected_sum_from_recipes
# ---------------------------------------------------------------------------
class TestExpectedSumFromRecipes:
    def test_golden_simple_plan(self):
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["200 g pollo", "150 g arroz"]},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert result == {"Pollo": {"g": 200.0}, "Arroz": {"g": 150.0}}

    def test_legume_yield_applied_to_seco(self):
        """`100 g lentejas cocidas` → 35g secas (factor 0.35× legumbre)."""
        plan = {"days": [{"meals": [
            {"meal": "cena", "ingredients_raw": ["100 g lentejas cocidas"]},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert result["Lentejas"]["g"] == pytest.approx(35.0, abs=0.01)

    def test_protein_no_yield_by_default(self):
        """Default `apply_yield=False` → pollo cocido NO multiplica por 1.35."""
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["1 lb pollo cocido"]},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert result["Pollo"]["lb"] == pytest.approx(1.0)

    def test_protein_yield_when_enabled(self):
        """`apply_yield=True` → 1 lb pollo cocido → 1.35 lb crudo."""
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["1 lb pollo cocido"]},
        ]}]}
        result = expected_sum_from_recipes(plan, apply_yield=True)
        assert result["Pollo"]["lb"] == pytest.approx(1.35, abs=0.01)

    def test_skip_suplemento_meals(self):
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["1 lb pollo"]},
            {"meal": "suplemento de proteina", "ingredients_raw": ["1 unidad de scoop"]},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert "Pollo" in result
        assert "Scoop" not in result

    def test_dict_ingredient_format(self):
        plan = {"days": [{"meals": [
            {"meal": "cena", "ingredients_raw": [
                {"quantity": 100, "unit": "g", "name": "avena"},
                {"quantity": 1, "unit": "mazo", "name": "cilantro"},
            ]},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert result["Avena"]["g"] == 100.0
        assert result["Cilantro"]["mazo"] == 1.0

    def test_recipe_dict_fallback(self):
        """Si `ingredients_raw` y `ingredients` están vacíos, lee de `recipe.ingredients`."""
        plan = {"days": [{"meals": [
            {"meal": "cena", "recipe": {"ingredients": ["50 g queso"]}},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert result["Queso"]["g"] == 50.0

    def test_ola_olas_coerced_to_cebolla(self):
        """Espejo del aggregator (línea 2248): `ola`/`olas` → `Cebolla`."""
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["100 g ola"]},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert "Cebolla" in result
        assert result["Cebolla"]["g"] == 100.0

    def test_aggregates_across_days_and_meals(self):
        plan = {"days": [
            {"meals": [{"meal": "desayuno", "ingredients_raw": ["1 cdta de aceite"]}]},
            {"meals": [
                {"meal": "desayuno", "ingredients_raw": ["2 cdtas de aceite"]},
                {"meal": "almuerzo", "ingredients_raw": ["1 cda aceite"]},
            ]},
        ]}
        result = expected_sum_from_recipes(plan)
        assert result["Aceite"]["cdta"] == 3.0
        assert result["Aceite"]["cda"] == 1.0

    @pytest.mark.parametrize("plan_input", [None, {}, {"days": []}, {"days": [{"meals": []}]}, "not-a-dict"])
    def test_empty_or_invalid_plan_returns_empty(self, plan_input):
        assert expected_sum_from_recipes(plan_input) == {}


# ---------------------------------------------------------------------------
# 2. compare_expected_vs_aggregated — heurísticos de hipótesis
# ---------------------------------------------------------------------------
class TestCompareExpectedVsAggregated:
    def test_golden_returns_empty(self):
        exp = {"Pollo": {"lb": 5.0}, "Arroz": {"g": 1000}}
        agg = {"Pollo": {"lb": 5.0}, "Arroz": {"g": 1010}}  # 1% < default 5%
        assert compare_expected_vs_aggregated(exp, agg) == []

    def test_cap_swallowed_modifier(self):
        """Food en expected, totalmente ausente en aggregated."""
        exp = {"Cilantro": {"mazo": 2.0}}
        divs = compare_expected_vs_aggregated(exp, {})
        assert len(divs) == 1
        assert divs[0]["food"] == "Cilantro"
        assert divs[0]["hypothesis"] == "cap_swallowed_modifier"

    def test_unit_mismatch(self, monkeypatch):
        """Food en ambos pero la unit de expected falta en aggregated.

        [P2-UNIT-CONV-1 · 2026-05-11] El converter por default está True
        post-flip. Con converter ON, `cda` (volumen) se normaliza a `ml`,
        cambiando el `unit` reportado en la divergencia y rompiendo
        `next(d for d in divs if d["unit"] == "cda")`. Forzamos converter
        OFF para preservar la semántica original del test (validar que el
        guard reporta `unit_mismatch` para una mezcla cda↔unidad sin
        normalización). Si el caso normalizado debe testarse, añadir
        sub-test separado con converter ON.
        """
        monkeypatch.setenv("MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED", "false")
        exp = {"Canela": {"cda": 3.0}}
        agg = {"Canela": {"unidad": 1.0}}  # cap exact-match en otra unit
        divs = compare_expected_vs_aggregated(exp, agg)
        cda_div = next(d for d in divs if d["unit"] == "cda")
        assert cda_div["hypothesis"] == "unit_mismatch"

    def test_yield_uncovered_protein(self):
        """Ratio 1.35× → yield proteína cocida no convertida."""
        exp = {"Pollo": {"lb": 1.0}}
        agg = {"Pollo": {"lb": 1.35}}
        divs = compare_expected_vs_aggregated(exp, agg)
        assert divs[0]["hypothesis"] == "yield_uncovered"

    def test_yield_uncovered_legume(self):
        """Ratio 0.35× → yield legumbre cocida no convertida."""
        exp = {"Lentejas": {"g": 100.0}}
        agg = {"Lentejas": {"g": 35.0}}
        divs = compare_expected_vs_aggregated(exp, agg)
        assert divs[0]["hypothesis"] == "yield_uncovered"

    def test_pantry_overdeduct(self):
        """actual < expected/2 sin matchear yield → pantry_overdeduct."""
        exp = {"Aceite": {"cda": 10.0}}
        agg = {"Aceite": {"cda": 2.0}}  # ratio 0.2, NO en yield ranges
        divs = compare_expected_vs_aggregated(exp, agg)
        assert divs[0]["hypothesis"] == "pantry_overdeduct"

    def test_ghost_inf_delta(self):
        """expected==0 & actual>0 → delta_pct=inf."""
        divs = compare_expected_vs_aggregated({}, {"Azucar": {"g": 200}})
        assert divs[0]["delta_pct"] == float("inf")
        assert divs[0]["food"] == "Azucar"

    def test_within_tolerance_no_report(self):
        exp = {"Arroz": {"g": 1000}}
        agg = {"Arroz": {"g": 1030}}  # 3% < 5% default
        assert compare_expected_vs_aggregated(exp, agg) == []

    def test_strict_tolerance_reports(self):
        exp = {"Arroz": {"g": 1000}}
        agg = {"Arroz": {"g": 1030}}
        divs = compare_expected_vs_aggregated(exp, agg, tolerance=0.01)
        assert len(divs) == 1
        assert divs[0]["delta_pct"] == pytest.approx(0.03)

    def test_sort_by_severity_inf_first(self):
        exp = {"Aceite": {"cda": 10.0}, "Pollo": {"lb": 1.0}, "Cilantro": {"mazo": 2.0}}
        agg = {"Aceite": {"cda": 9.5}, "Pollo": {"lb": 1.35}, "AzucarFantasma": {"g": 100}}
        divs = compare_expected_vs_aggregated(exp, agg)
        assert divs[0]["food"] == "AzucarFantasma"  # inf primero
        assert divs[0]["delta_pct"] == float("inf")
        # Aceite 9.5/10 = exactamente 5% delta_pct, NO supera tolerance strict.
        foods = [d["food"] for d in divs]
        assert "Aceite" not in foods

    @pytest.mark.parametrize("exp,agg", [
        (None, None),
        ({}, {}),
        ({"X": "no-dict"}, {"X": {"g": 5}}),
    ])
    def test_defensive_inputs_no_crash(self, exp, agg):
        result = compare_expected_vs_aggregated(exp, agg)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. Knobs MEALFIT_SHOPPING_COHERENCE_*
# ---------------------------------------------------------------------------
class TestCoherenceKnobs:
    @pytest.mark.parametrize("raw,expected", [
        # [P1-NEW-1 · 2026-05-10] Default + invalid fallback bumpeados a "block"
        # (eran "warn"). Razón: producción debe rechazar listas incoherentes,
        # no solo loguearlas. Rollback: env=warn.
        (None, "block"),
        ("off", "off"),
        ("warn", "warn"),
        ("block", "block"),
        ("OFF", "off"),
        ("  Block  ", "block"),
        ("", "block"),
        ("garbage", "block"),
    ])
    def test_guard_mode(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        else:
            monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raw)
        assert _get_coherence_guard_mode() == expected

    @pytest.mark.parametrize("raw,expected", [
        (None, 0.10),
        ("0.05", 0.05),
        ("0.20", 0.20),
        ("0.999", 0.999),
        ("", 0.10),
        ("garbage", 0.10),
        ("-0.5", 0.10),  # fuera de rango
        ("1.5", 0.10),
        ("0", 0.10),     # boundary excluido
        ("1", 0.10),
    ])
    def test_tolerance_pct(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
        else:
            monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raw)
        assert _get_coherence_tolerance_pct() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. run_shopping_coherence_guard — orquestación E2E
# ---------------------------------------------------------------------------
class TestRunShoppingCoherenceGuard:
    def _make_plan(self, ingredients_raw, agg_list):
        return {
            "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ingredients_raw}]}],
            "aggregated_shopping_list": agg_list,
        }

    def test_golden_no_divergences(self, no_master_db, monkeypatch):
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["200 g pollo", "150 g arroz"],
            [
                {"name": "Pollo", "market_qty_numeric": 200, "market_unit": "g"},
                {"name": "Arroz", "market_qty_numeric": 150, "market_unit": "g"},
            ],
        )
        assert run_shopping_coherence_guard(plan) == []
        assert "_shopping_coherence_block" not in plan

    def test_cap_swallowed_detected(self, no_master_db, monkeypatch):
        """[Vinculado a project_caps_asymmetry_known_issue.md] Cilantro de la
        receta totalmente ausente de la lista (caso 'cabezas' en algunas
        verduras o cap exact-match que engulle el modificador)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["200 g pollo", "1 mazo cilantro"],
            [{"name": "Pollo", "market_qty_numeric": 200, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 1
        assert divs[0]["food"] == "Cilantro"
        assert divs[0]["side"] == "expected_only"
        assert divs[0]["hypothesis"] == "cap_swallowed_modifier"

    def test_fantasma_detected(self, no_master_db, monkeypatch):
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["100 g avena"],
            [
                {"name": "Avena", "market_qty_numeric": 100, "market_unit": "g"},
                {"name": "Azucar", "market_qty_numeric": 200, "market_unit": "g"},
            ],
        )
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 1
        assert divs[0]["food"] == "Azucar"
        assert divs[0]["side"] == "aggregated_only"

    def test_staple_filtered(self, no_master_db, monkeypatch):
        """Items con `is_staple=True` se filtran del lado aggregated."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["100 g avena"],
            [
                {"name": "Avena", "market_qty_numeric": 100, "market_unit": "g"},
                {"name": "Sal", "market_qty_numeric": 1, "market_unit": "paquete", "is_staple": True},
            ],
        )
        assert run_shopping_coherence_guard(plan) == []

    def test_urgent_category_filtered(self, no_master_db, monkeypatch):
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["100 g avena"],
            [
                {"name": "Avena", "market_qty_numeric": 100, "market_unit": "g"},
                {"name": "Vitamina", "market_qty_numeric": 1, "market_unit": "ud",
                 "category": "🚨 Compra Urgente"},
            ],
        )
        assert run_shopping_coherence_guard(plan) == []

    def test_mode_off_short_circuits(self, no_master_db, monkeypatch):
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "off")
        plan = self._make_plan(
            ["1 mazo cilantro"],  # claramente missing
            [],
        )
        assert run_shopping_coherence_guard(plan) == []
        assert "_shopping_coherence_block" not in plan

    def test_mode_block_with_missing_sets_flag(self, no_master_db, monkeypatch):
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = self._make_plan(
            ["200 g pollo", "1 mazo cilantro"],
            [{"name": "Pollo", "market_qty_numeric": 200, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 1
        block = plan.get("_shopping_coherence_block")
        assert block is not None
        assert len(block) == 1
        assert block[0]["food"] == "Cilantro"

    def test_mode_block_only_fantasma_no_flag(self, no_master_db, monkeypatch):
        """Fantasmas no bloquean: pueden ser staples/legítimos."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = self._make_plan(
            ["100 g avena"],
            [
                {"name": "Avena", "market_qty_numeric": 100, "market_unit": "g"},
                {"name": "Azucar", "market_qty_numeric": 200, "market_unit": "g"},
            ],
        )
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 1
        assert "_shopping_coherence_block" not in plan

    # -----------------------------------------------------------------------
    # [P3-2 · 2026-05-08] Edge-case plan vacío + items en lista.
    #
    # Si el LLM produce accidentalmente un plan sin días/meals (regresión del
    # planner, parser fail, fallback degraded) pero la lista de compras ya
    # está poblada, el guard ve "todo es fantasma". Contrato: en mode=block
    # NO debe marcar `_shopping_coherence_block` (los fantasmas pueden ser
    # staples no marcados o catálogo legítimo); el cron de alerta 04:00 UTC
    # tampoco debe degradar el plan basándose en este caso degenerado.
    # -----------------------------------------------------------------------
    def test_empty_plan_block_mode_no_flag_on_fantasmas(self, no_master_db, monkeypatch):
        """Plan con `days=[]` + items en lista en modo block → divergencias
        `aggregated_only` reportadas, pero SIN `_shopping_coherence_block`.

        El guard no puede distinguir si la lista está poblada por error o si
        contiene staples que el filtro is_staple no marcó. Conservador: warn
        por log pero no bloquea.
        """
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = {
            "days": [],
            "aggregated_shopping_list": [
                {"name": "Pollo", "market_qty_numeric": 500, "market_unit": "g"},
                {"name": "Arroz", "market_qty_numeric": 1000, "market_unit": "g"},
            ],
        }
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 2
        assert all(d["side"] == "aggregated_only" for d in divs)
        assert "_shopping_coherence_block" not in plan

    def test_empty_plan_warn_mode_logs_but_no_flag(self, no_master_db, monkeypatch):
        """Mismo edge-case en modo warn: reporta divergencias, sin flag (warn
        nunca setea el flag por contrato — sanity check del bucle warn↔block)."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")
        plan = {
            "days": [],
            "aggregated_shopping_list": [
                {"name": "Pollo", "market_qty_numeric": 500, "market_unit": "g"},
            ],
        }
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 1
        assert divs[0]["side"] == "aggregated_only"
        assert "_shopping_coherence_block" not in plan

    def test_empty_plan_meals_empty_no_flag(self, no_master_db, monkeypatch):
        """Variante: `days=[{"meals": []}]` (días sin comidas) — mismo
        contrato que `days=[]`."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = {
            "days": [{"meals": []}, {"meals": []}],
            "aggregated_shopping_list": [
                {"name": "Pollo", "market_qty_numeric": 500, "market_unit": "g"},
            ],
        }
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 1
        assert divs[0]["side"] == "aggregated_only"
        assert "_shopping_coherence_block" not in plan

    def test_empty_plan_meals_no_ingredients_no_flag(self, no_master_db, monkeypatch):
        """Variante: meals sin `ingredients_raw` — mismo contrato."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = {
            "days": [{"meals": [{"meal": "almuerzo"}]}],
            "aggregated_shopping_list": [
                {"name": "Pollo", "market_qty_numeric": 500, "market_unit": "g"},
            ],
        }
        divs = run_shopping_coherence_guard(plan)
        assert len(divs) == 1
        assert divs[0]["side"] == "aggregated_only"
        assert "_shopping_coherence_block" not in plan

    def test_both_empty_zero_divergences_no_flag(self, no_master_db, monkeypatch):
        """Plan vacío + lista vacía: 0 divergencias, 0 flag."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = {"days": [], "aggregated_shopping_list": []}
        divs = run_shopping_coherence_guard(plan)
        assert divs == []
        assert "_shopping_coherence_block" not in plan

    def test_empty_plan_skips_magnitude_layer(self, no_master_db, monkeypatch):
        """Plan vacío + items en lista NO debe ejercitar la capa B
        (magnitudes). Garantiza que `compare_expected_vs_aggregated` no se
        invoque sobre `expected_raw={}` (early-out por contrato), evitando
        ruido en logs y posibles falsos positivos magnitud=∞.

        Validación indirecta: si la capa B se ejercitara, las divergencias
        traerían `magnitude=True`. Aquí todas deben ser `magnitude=False`.
        """
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = {
            "days": [],
            "aggregated_shopping_list": [
                {"name": "Pollo", "market_qty_numeric": 500, "market_unit": "g"},
                {"name": "Arroz", "market_qty_numeric": 1000, "market_unit": "g"},
            ],
        }
        divs = run_shopping_coherence_guard(plan)
        assert all(d.get("magnitude") is False for d in divs), (
            "Plan vacío debe activar early-out de la capa B; ningún div magnitude=True"
        )

    def test_empty_plan_with_block_action_safe_for_alert_cron(
        self, no_master_db, monkeypatch
    ):
        """[P3-2 · escenario cron 04:00 UTC] El cron `_shopping_coherence_alert_job`
        re-evalúa planes ya persistidos en modo `warn` (`mode_override='warn'`)
        para no mutar `_shopping_coherence_block` retroactivamente. Este test
        confirma que un plan vacío no provoca bloqueo accidental ni siquiera
        cuando el env apunta a `block`.
        """
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = {
            "days": [],
            "aggregated_shopping_list": [
                {"name": "Pollo", "market_qty_numeric": 500, "market_unit": "g"},
            ],
        }
        divs = run_shopping_coherence_guard(plan, mode_override="warn")
        assert len(divs) == 1
        assert "_shopping_coherence_block" not in plan, (
            "El cron de alertas (override=warn) no debe escalar plan vacío a block"
        )

    def test_exception_does_not_propagate(self, monkeypatch):
        """Si master_ingredients revienta + ingredients_raw inválido, el guard
        captura excepción interna y retorna [] sin tirar el assembly."""
        def boom():
            raise RuntimeError("simulated")
        monkeypatch.setattr(shopping_calculator, "get_master_ingredients", boom)
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = {"days": "not-a-list", "aggregated_shopping_list": []}  # invalida
        # No debe levantar.
        result = run_shopping_coherence_guard(plan)
        assert isinstance(result, list)

    def test_mode_override_off(self, no_master_db, monkeypatch):
        """`mode_override='off'` ignora env y retorna [] sin importar contenido."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")  # env=block
        plan = self._make_plan(["1 mazo cilantro"], [])  # claramente missing
        assert run_shopping_coherence_guard(plan, mode_override="off") == []

    def test_mode_override_warn_prevents_block_mutation(self, no_master_db, monkeypatch):
        """Caso del cron (Paso 7): aunque env=block, override=warn previene
        que el guard mute `_shopping_coherence_block` retroactivamente sobre
        un plan ya persistido."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = self._make_plan(
            ["200 g pollo", "1 mazo cilantro"],
            [{"name": "Pollo", "market_qty_numeric": 200, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan, mode_override="warn")
        assert len(divs) == 1
        assert "_shopping_coherence_block" not in plan

    def test_mode_override_invalid_falls_back_to_warn(self, no_master_db, monkeypatch):
        """Override con valor inválido → cae a warn (no off, no block)."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "off")  # env=off
        plan = self._make_plan(
            ["200 g pollo", "1 mazo cilantro"],
            [{"name": "Pollo", "market_qty_numeric": 200, "market_unit": "g"}],
        )
        # Override "garbage" → warn → reporta pero NO bloquea.
        divs = run_shopping_coherence_guard(plan, mode_override="garbage")
        assert len(divs) == 1
        assert "_shopping_coherence_block" not in plan


# ---------------------------------------------------------------------------
# 5. P1-C v2 — magnitudes con multiplier
# ---------------------------------------------------------------------------
class TestExpectedSumMultiplier:
    """`expected_sum_from_recipes(plan, multiplier=X)` escala las cantidades."""

    def test_default_multiplier_no_op(self):
        plan = {"days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ["200 g pollo"]}]}]}
        assert expected_sum_from_recipes(plan)["Pollo"]["g"] == 200.0

    def test_explicit_multiplier_scales(self):
        """multiplier=2.4 (familia 2A+1N×0.6) escala 200g → 480g."""
        plan = {"days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ["200 g pollo"]}]}]}
        result = expected_sum_from_recipes(plan, multiplier=2.4)
        assert result["Pollo"]["g"] == pytest.approx(480.0)

    @pytest.mark.parametrize("bad_mult", [None, 0, -1.5, float("nan"), float("inf"), "garbage", "1.0a"])
    def test_invalid_multiplier_falls_back_to_1(self, bad_mult):
        """NaN/inf/<=0/string inválido → multiplier=1.0 (no hace daño al cálculo)."""
        plan = {"days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ["200 g pollo"]}]}]}
        result = expected_sum_from_recipes(plan, multiplier=bad_mult)
        assert result["Pollo"]["g"] == pytest.approx(200.0)

    def test_multiplier_preserves_yield(self):
        """Multiplier × yield_legumbre: 100g cocidas × 0.35 × 2.0 = 70g secas."""
        plan = {"days": [{"meals": [{"meal": "cena", "ingredients_raw": ["100 g lentejas cocidas"]}]}]}
        result = expected_sum_from_recipes(plan, multiplier=2.0)
        assert result["Lentejas"]["g"] == pytest.approx(70.0, abs=0.01)


class TestMagnitudeCoherence:
    """[P1-C 2026-05-07] Capa magnitudes del guard."""

    def _make_plan(self, ingredients_raw, agg_list, multiplier_persisted=None):
        plan = {
            "days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": ingredients_raw}]}],
            "aggregated_shopping_list": agg_list,
        }
        if multiplier_persisted is not None:
            plan["calc_household_multiplier"] = multiplier_persisted
        return plan

    def test_qty_match_no_divergence(self, no_master_db, monkeypatch):
        """Multiplier=1, qty match exacto → 0 divergencias."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
        plan = self._make_plan(
            ["1000 g arroz"],
            [{"name": "Arroz", "market_qty_numeric": 1000, "market_unit": "g"}],
        )
        assert run_shopping_coherence_guard(plan, multiplier=1.0) == []

    def test_qty_partial_detected_as_magnitude(self, no_master_db, monkeypatch):
        """Receta pide 1000g, lista 200g (ratio 0.2 → pantry_overdeduct)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
        plan = self._make_plan(
            ["1000 g arroz"],
            [{"name": "Arroz", "market_qty_numeric": 200, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        # Presence/absence NO reporta (Arroz está en ambos).
        # Magnitude SÍ reporta (delta_pct = 0.8 > 0.10 default).
        mag_divs = [d for d in divs if d.get("magnitude")]
        assert len(mag_divs) == 1
        assert mag_divs[0]["food"] == "Arroz"
        assert mag_divs[0]["expected_qty"] == pytest.approx(1000.0)
        assert mag_divs[0]["actual_qty"] == pytest.approx(200.0)
        # `pantry_overdeduct` (act < exp*0.5 estricto, fuera de rangos yield).
        assert mag_divs[0]["hypothesis"] == "pantry_overdeduct"

    def test_yield_uncovered_protein_magnitude(self, no_master_db, monkeypatch):
        """Receta 1 lb pollo, lista 1.35 lb (yield 1.35× no aplicado)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
        plan = self._make_plan(
            ["1 lb pollo"],
            [{"name": "Pollo", "market_qty_numeric": 1.35, "market_unit": "lb"}],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        assert len(mag) == 1
        assert mag[0]["hypothesis"] == "yield_uncovered"

    def test_pantry_overdeduct_magnitude(self, no_master_db, monkeypatch):
        """Receta 10 cda aceite, lista 1.5 cda (sub-yield, no rango ratio)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
        plan = self._make_plan(
            ["10 cda aceite"],
            [{"name": "Aceite", "market_qty_numeric": 1.5, "market_unit": "cda"}],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        assert len(mag) == 1
        assert mag[0]["hypothesis"] == "pantry_overdeduct"

    def test_multiplier_arg_scales_expected(self, no_master_db, monkeypatch):
        """Receta 200g pollo × multiplier 2.4 = 480g esperados; lista 480g → match."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["200 g pollo"],
            [{"name": "Pollo", "market_qty_numeric": 480, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=2.4)
        mag = [d for d in divs if d.get("magnitude")]
        assert mag == []

    def test_multiplier_from_plan_data(self, no_master_db, monkeypatch):
        """Sin arg `multiplier`, lee `plan_result["calc_household_multiplier"]`."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["200 g pollo"],
            [{"name": "Pollo", "market_qty_numeric": 480, "market_unit": "g"}],
            multiplier_persisted=2.4,
        )
        divs = run_shopping_coherence_guard(plan)  # multiplier=None
        mag = [d for d in divs if d.get("magnitude")]
        assert mag == []

    def test_arg_overrides_plan_data_multiplier(self, no_master_db, monkeypatch):
        """`multiplier=1.0` arg gana sobre el cacheado en plan (caso test/debug)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["200 g pollo"],
            [{"name": "Pollo", "market_qty_numeric": 480, "market_unit": "g"}],
            multiplier_persisted=2.4,  # cacheado dice 2.4
        )
        # Pero pasamos 1.0 → expected=200, actual=480 → divergencia magnitude.
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        assert len(mag) == 1

    def test_pavo_magnitude_now_compared_v3(self, no_master_db, monkeypatch):
        """[P3-4 · 2026-05-07] v3: pavo ya NO está en lista blanca; magnitudes
        se comparan después de aplicar `canonicalize_pavo` simétricamente.

        Antes (v2): cualquier divergencia de pavo se filtraba para evitar
        falsos positivos por divergencia de canónico (Pavo vs Pechuga vs
        Jamón). v3 cierra el bucle con un mirror simétrico del aggregator.

        Caso: receta pide 200g pavo (genérico → 'Pavo'), lista trae 800g Pavo
        (4× lo de la receta) — ratio fuera de cualquier yield range, debe
        reportarse como divergencia."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["200 g pavo"],
            [{"name": "Pavo", "market_qty_numeric": 800, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        # Debe haber al menos una divergencia magnitude para Pavo.
        pavo_mag = [d for d in mag if d["food"] == "Pavo"]
        assert len(pavo_mag) >= 1, (
            f"v3 esperaba reportar magnitude divergence para Pavo; mag={mag!r}"
        )

    def test_no_double_report_presence_and_magnitude(self, no_master_db, monkeypatch):
        """Cilantro missing total: presence reporta 1 vez. Magnitude NO duplica
        (porque act_qty=0 + food en missing_in_agg)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["200 g pollo", "1 mazo cilantro"],
            [{"name": "Pollo", "market_qty_numeric": 200, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        cilantro_divs = [d for d in divs if d["food"] == "Cilantro"]
        # Solo 1 reporte (presence), no duplicado por magnitude.
        assert len(cilantro_divs) == 1
        assert cilantro_divs[0]["magnitude"] is False

    def test_block_with_magnitude_critical_sets_flag(self, no_master_db, monkeypatch):
        """Mode block + qty mitad (magnitude crítica) → setea
        `_shopping_coherence_block`."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
        plan = self._make_plan(
            ["1000 g arroz"],
            [{"name": "Arroz", "market_qty_numeric": 500, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        block = plan.get("_shopping_coherence_block")
        assert block is not None
        # Crítico: la divergencia magnitude con expected_qty>0 cuenta.
        assert any(d.get("magnitude") for d in block)

    def test_block_only_fantasma_no_flag(self, no_master_db, monkeypatch):
        """Fantasmas (delta=inf, expected=0) NO suben a block — pueden ser
        staples no marcados."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = self._make_plan(
            ["100 g avena"],
            [
                {"name": "Avena", "market_qty_numeric": 100, "market_unit": "g"},
                {"name": "Azucar", "market_qty_numeric": 200, "market_unit": "g"},
            ],
        )
        run_shopping_coherence_guard(plan, multiplier=1.0)
        assert "_shopping_coherence_block" not in plan

    def test_tolerance_pct_strict_blocks_small_delta(self, no_master_db, monkeypatch):
        """Knob TOLERANCE_PCT=0.01 → 3% delta supera tolerance, magnitude reporta.
        [P1-COHERENCE-OVERSUPPLY-STAPLES · 2026-07-07] Vehículo cambiado de Arroz a Pollo:
        la sobre-oferta de STAPLES (arroz por bolsa) ahora se filtra como ruido de envase;
        el mecanismo del knob de tolerancia se prueba con una PROTEÍNA por peso (no filtrada)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", "0.01")
        plan = self._make_plan(
            ["1000 g pollo"],
            [{"name": "Pollo", "market_qty_numeric": 1030, "market_unit": "g"}],  # 3% delta
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        assert len(mag) == 1

    def test_tolerance_pct_loose_no_report(self, no_master_db, monkeypatch):
        """Knob TOLERANCE_PCT=0.20 → 5% delta NO supera tolerance."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", "0.20")
        plan = self._make_plan(
            ["1000 g arroz"],
            [{"name": "Arroz", "market_qty_numeric": 1050, "market_unit": "g"}],  # 5% delta
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        assert mag == []

    def test_override_warn_prevents_mutation_with_magnitude(self, no_master_db, monkeypatch):
        """Cron path: env=block + override=warn + magnitude crítica → NO muta."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = self._make_plan(
            ["1000 g arroz"],
            [{"name": "Arroz", "market_qty_numeric": 500, "market_unit": "g"}],
        )
        divs = run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        assert len(mag) >= 1
        assert "_shopping_coherence_block" not in plan

    def test_staple_filtered_in_magnitude(self, no_master_db, monkeypatch):
        """is_staple=True filtra antes de magnitudes (mismo filtro que presence)."""
        monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
        plan = self._make_plan(
            ["1000 g arroz"],
            [
                {"name": "Arroz", "market_qty_numeric": 1000, "market_unit": "g"},
                # Staple con qty espuria — debe ser ignorado.
                {"name": "Sal", "market_qty_numeric": 99999, "market_unit": "g", "is_staple": True},
            ],
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        assert divs == []
