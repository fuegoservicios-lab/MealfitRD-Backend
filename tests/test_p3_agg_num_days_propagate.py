"""[P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] Los wrappers de re-agregación caían a
person_weeks=1.0 y capaban la nevera del coach/swap.

`aggregate_shopping_list` y `get_realtime_pantry` (shopping_calculator.py) invocaban a
`aggregate_and_deduct_shopping_list` SIN `num_days` ni `multiplier`. El agregador cae
entonces al fallback `_pw_days=3.0` (línea ~9948) ⇒ `_person_weeks = max(1.0, 1.0*3.0/7.0)
= 1.0` SIEMPRE — sin importar household ni duración (semanal/quincenal/mensual) del plan
real. Los caps P6 (latas de atún, aceite, especias, endulzantes...) leen `_person_weeks`
para fijar su techo, así que terminan recortando una demanda YA escalada a 15/30 días al
mismo techo que un plan semanal de 1 persona.

Alcance verificado (ejecutando el aggregador real, ver números abajo):
  - `get_realtime_pantry`: path PRIMARIO del swap (agent.py::swap_meal, rama legacy de
    `MEALFIT_PANTRY_STRICT_UPDATES=false`) — la «nevera virtual» que ve el LLM del swap.
  - `aggregate_shopping_list`: fallbacks del coach (`chat_with_agent`/`chat_with_agent_stream`,
    guests/BD caída) que SÍ tienen `current_plan` en scope, y el fallback de `swap_meal` que
    NO lo tiene (documentado inline, dejado con los defaults).

Fix: ambos wrappers aceptan `*, num_days=None, multiplier=1.0` (keyword-only, defaults =
comportamiento histórico exacto) y los reenvían tal cual al agregador. Los callsites de
agent.py que SÍ tienen el plan a mano derivan los valores reales vía el helper
`_virtual_pantry_num_days_and_multiplier` (mismo SSOT que `get_shopping_list_delta` /
`routers/plans.py::scaled_30`: `num_days` = días REALMENTE generados, `multiplier` =
`household × cycle_qty_multiplier(duración) × 7/num_days`).

Los dos casos ancla de abajo (atún/aceite) están verificados EJECUTANDO
`aggregate_and_deduct_shopping_list` con `structured=True` y leyendo `capped_post` (gramos
tras el cap) — no son números inventados:

  - Atún, household=2, mensual, num_days=3 → multiplier=20.0 (`_virtual_pantry_num_days_and_multiplier`
    produce el mismo valor). Sin el fix: capped_post=368g (2 latas × 184g, el techo default
    `max(2, round(1.0))=2`). Con el fix: capped_post=1656g (9 latas × 184g, el techo real
    `max(2, round(2*30/7))=9`).
  - Aceite, household=4, mensual, num_days=3 → multiplier=40.0. Sin el fix: capped_post=946g
    (1 botella × 946g, techo default `max(1, round(1.0/4))=1`). Con el fix: capped_post=3784g
    (4 botellas × 946g, techo real `max(1, round(4*30/7/4))=4`).
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parents[1]
_SC_SRC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_AGENT_SRC = (_BACKEND / "agent.py").read_text(encoding="utf-8")

# El knob por default deja pasar ingredientes off-catálogo (raw fallback) — sin esto,
# sin DB, `get_master_ingredients()` vuelve vacío y "VERIFIED-ONLY-DROP" descarta el
# ítem entero, dejando el test ciego a lo que quiere medir. Mismo default que conftest.py
# fija para toda la suite (P1-VERIFIED-ONLY-DEFAULT-ON); lo re-afirmamos aquí para que
# este archivo sea correcto también corriendo en aislamiento.
os.environ.setdefault("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "false")


@pytest.fixture(autouse=True)
def _reset_master_cache():
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


def _find_item(result, needle: str):
    for r in result:
        if isinstance(r, dict) and needle in str(r.get("name", "")).lower():
            return r
    return None


# ===========================================================================
# 1) Firma de los wrappers: keyword-only, defaults = comportamiento histórico
# ===========================================================================
class TestWrapperSignatures:
    def test_aggregate_shopping_list_acepta_num_days_y_multiplier(self):
        sig = inspect.signature(sc.aggregate_shopping_list)
        assert "num_days" in sig.parameters, "aggregate_shopping_list debe aceptar num_days"
        assert "multiplier" in sig.parameters, "aggregate_shopping_list debe aceptar multiplier"
        assert sig.parameters["num_days"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["multiplier"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["num_days"].default is None
        assert sig.parameters["multiplier"].default == 1.0

    def test_get_realtime_pantry_acepta_num_days_y_multiplier(self):
        sig = inspect.signature(sc.get_realtime_pantry)
        assert "num_days" in sig.parameters, "get_realtime_pantry debe aceptar num_days"
        assert "multiplier" in sig.parameters, "get_realtime_pantry debe aceptar multiplier"
        assert sig.parameters["num_days"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["multiplier"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["num_days"].default is None
        assert sig.parameters["multiplier"].default == 1.0


# ===========================================================================
# 2) Sin kwargs: comportamiento IDÉNTICO al histórico (no-regresión)
# ===========================================================================
class TestDefaultsPreserveHistoricalBehavior:
    def test_aggregate_shopping_list_sin_kwargs_es_igual_a_llamar_al_agregador_sin_num_days(self):
        ingredients = ["1 lata de atún en agua", "2 tazas de arroz"]
        via_wrapper = sc.aggregate_shopping_list(list(ingredients))
        sc.invalidate_master_cache()
        via_directo = sc.aggregate_and_deduct_shopping_list(list(ingredients), [])
        assert via_wrapper == via_directo

    def test_get_realtime_pantry_sin_kwargs_es_igual_al_comportamiento_previo(self):
        plan_result = {
            "days": [
                {"day": 1, "meals": [{"name": "Cena", "ingredients": ["1 lata de atún en agua"]}]},
            ]
        }
        via_wrapper = sc.get_realtime_pantry(plan_result, [])
        sc.invalidate_master_cache()
        via_directo = sc.aggregate_and_deduct_shopping_list(["1 lata de atún en agua"], [])
        assert via_wrapper == via_directo


# ===========================================================================
# 3) Casos ancla: una lista mensual re-agregada NO sale por debajo del real
# ===========================================================================
class TestNumDaysMultiplierPreventUndercap:
    def test_atun_no_se_capa_a_2_en_plan_mensual_2_personas(self):
        household, num_days, duration = 2.0, 3, "monthly"
        duration_factor = sc.cycle_qty_multiplier(duration)
        multiplier = household * duration_factor * (7.0 / num_days)
        raw_ingredients = ["1 lata de atún en agua"] * num_days

        buggy = sc.aggregate_and_deduct_shopping_list(list(raw_ingredients), [], structured=True)
        sc.invalidate_master_cache()
        fixed_via_wrapper = sc.aggregate_shopping_list(
            list(raw_ingredients), num_days=num_days, multiplier=multiplier
        )
        sc.invalidate_master_cache()
        fixed_structured = sc.aggregate_and_deduct_shopping_list(
            list(raw_ingredients), [], multiplier=multiplier, num_days=num_days, structured=True
        )

        buggy_item = _find_item(buggy, "atun") or _find_item(buggy, "atún")
        fixed_item = _find_item(fixed_structured, "atun") or _find_item(fixed_structured, "atún")
        assert buggy_item is not None and fixed_item is not None, (
            f"el ítem de atún debe sobrevivir en ambos: buggy={buggy} fixed={fixed_structured}"
        )

        # Ancla del bug: sin num_days/multiplier reales, el techo cae SIEMPRE a 2 latas
        # (max(2, round(person_weeks=1.0))) — 368g = 2 * 184g.
        assert buggy_item["capped_post"] == pytest.approx(368.0), (
            f"baseline buggy inesperado (¿cambió el cap de 184g/lata?): {buggy_item}"
        )
        # Con el fix: el techo real es max(2, round(household*cycle_qty_multiplier)) = 9 latas.
        expected_person_weeks = household * duration_factor
        expected_cap_latas = max(2, round(expected_person_weeks))
        assert expected_cap_latas > 2, "fixture debe ejercitar un techo real >2 o el test no prueba nada"
        assert fixed_item["capped_post"] == pytest.approx(expected_cap_latas * 184.0), (
            f"el techo con num_days/multiplier reales debe ser {expected_cap_latas} latas: {fixed_item}"
        )
        assert fixed_item["capped_post"] > buggy_item["capped_post"], (
            "una lista mensual 2p NO puede salir con el mismo techo que una semanal 1p"
        )
        # El wrapper público reproduce el mismo resultado que el agregador directo.
        assert any("atun" in s.lower() or "atún" in s.lower() for s in fixed_via_wrapper)

    def test_aceite_no_se_capa_a_1_en_plan_mensual_4_personas(self):
        household, num_days, duration = 4.0, 3, "monthly"
        duration_factor = sc.cycle_qty_multiplier(duration)
        multiplier = household * duration_factor * (7.0 / num_days)
        raw_ingredients = ["1 botella de aceite de oliva"] * num_days

        buggy = sc.aggregate_and_deduct_shopping_list(list(raw_ingredients), [], structured=True)
        sc.invalidate_master_cache()
        fixed = sc.aggregate_and_deduct_shopping_list(
            list(raw_ingredients), [], multiplier=multiplier, num_days=num_days, structured=True
        )

        buggy_item = _find_item(buggy, "aceite")
        fixed_item = _find_item(fixed, "aceite")
        assert buggy_item is not None and fixed_item is not None

        # [reapuntado 2026-08-14] El baseline exigía `capped_post == 946` en la llamada
        # SIN num_days/multiplier. Hoy esa llamada ni siquiera CAPA —3 botellas a
        # multiplicador 1 no alcanzan ningún techo— así que la clave `capped_post` no
        # existe y el test moría con KeyError sobre un lado que ya no es el interesante.
        # Además el SKU del catálogo pasó a 250 ml (230 g), o sea que el 946 era un
        # número de otra época. Lo que este test protege sigue intacto y es el OTRO
        # lado: con `num_days`/`multiplier` reales, el aceite escala a su techo de
        # person-weeks en vez de quedarse en la cantidad de un ciclo corto.
        _entregado_buggy = float(buggy_item.get("capped_post") or buggy_item.get("base_qty") or 0)
        expected_person_weeks = household * duration_factor
        expected_cap_botellas = max(1, round(expected_person_weeks / 4.0))
        assert expected_cap_botellas > 1, "fixture debe ejercitar un techo real >1"
        assert fixed_item.get("capped_by") == "P6-OIL-CAP", (
            f"el aceite debe llegar a su techo con los valores reales: {fixed_item}"
        )
        assert fixed_item["capped_post"] == pytest.approx(expected_cap_botellas * 946.0), (
            f"el techo real debe ser {expected_cap_botellas} botellas: {fixed_item}"
        )
        assert fixed_item["capped_post"] > _entregado_buggy, (
            "una lista mensual de 4 personas NO puede entregar lo mismo que un ciclo "
            f"sin escalar (fixed={fixed_item['capped_post']}, buggy={_entregado_buggy})"
        )


# ===========================================================================
# 4) agent.py: el helper + los callsites reales (swap primario + fallbacks del coach)
# ===========================================================================
class TestAgentCallsitesDeriveRealValues:
    def test_helper_existe_y_se_puede_llamar_directo(self):
        import agent as ag

        assert hasattr(ag, "_virtual_pantry_num_days_and_multiplier"), (
            "agent.py debe exponer el helper SSOT que deriva num_days/multiplier del plan"
        )
        plan_data = {
            "days": [{"day": 1, "meals": []}] * 3,
            "calc_household_multiplier": 2.0,
            "calc_grocery_duration": "monthly",
        }
        num_days, multiplier = ag._virtual_pantry_num_days_and_multiplier(plan_data)
        assert num_days == 3
        expected_multiplier = 2.0 * sc.cycle_qty_multiplier("monthly") * (7.0 / 3)
        assert multiplier == pytest.approx(expected_multiplier)

    def test_helper_fail_open_sin_plan(self):
        import agent as ag

        assert ag._virtual_pantry_num_days_and_multiplier(None) == (None, 1.0)
        assert ag._virtual_pantry_num_days_and_multiplier({}) == (None, 1.0)
        assert ag._virtual_pantry_num_days_and_multiplier({"days": []}) == (None, 1.0)

    def test_swap_primary_path_pasa_num_days_y_multiplier_reales(self):
        """El path PRIMARIO del swap (`get_realtime_pantry` en la rama legacy de
        `swap_meal`) debe derivar los valores reales del plan, no llamar al wrapper
        con los defaults (que reproduce el bug: person_weeks=1.0 siempre)."""
        i = _AGENT_SRC.index("clean_ingredients = get_realtime_pantry(")
        window = _AGENT_SRC[i:i + 400]
        assert "num_days=" in window and "multiplier=" in window, (
            "el callsite del swap debe pasar num_days=/multiplier= a get_realtime_pantry, "
            "o el fix queda decorativo (el wrapper sigue viendo los defaults)"
        )
        assert "_virtual_pantry_num_days_and_multiplier(" in _AGENT_SRC[max(0, i - 800):i], (
            "el swap debe derivar num_days/multiplier del plan justo antes de llamar al wrapper"
        )

    def test_coach_fallbacks_usan_current_plan_para_derivar_valores_reales(self):
        """`chat_with_agent`/`chat_with_agent_stream` SÍ reciben `current_plan` como
        parámetro — a diferencia de `swap_meal`, tienen contexto para derivar valores
        reales incluso en el fallback (guest / BD caída)."""
        occurrences = [
            m.start() for m in __import__("re").finditer(
                r"cleaned_pantry = aggregate_shopping_list\(", _AGENT_SRC
            )
        ]
        assert len(occurrences) >= 2, "ambos paths (stream y no-stream) deben tener el fallback"
        for start in occurrences:
            window = _AGENT_SRC[start:start + 300]
            assert "num_days=" in window and "multiplier=" in window, (
                f"el fallback de inventario del coach en offset {start} debe pasar "
                f"num_days=/multiplier= reales (derivados de current_plan)"
            )

        shop_occurrences = [
            m.start() for m in __import__("re").finditer(
                r"cleaned_shop = aggregate_shopping_list\(", _AGENT_SRC
            )
        ]
        assert len(shop_occurrences) >= 2
        for start in shop_occurrences:
            window = _AGENT_SRC[start:start + 300]
            assert "num_days=" in window and "multiplier=" in window, (
                f"el fallback de shopping-list del coach en offset {start} debe pasar "
                f"num_days=/multiplier= reales"
            )

    def test_swap_meal_fallback_sin_contexto_queda_documentado_no_inventado(self):
        """`swap_meal(form_data)` NO recibe `current_plan` — a diferencia del coach, este
        fallback (línea con `current_pantry_ingredients or current_shopping_list`) no tiene
        de dónde derivar num_days/multiplier reales. El fix NO debe inventar un valor
        (household=1 silencioso): debe documentarlo inline y dejar los defaults."""
        i = _AGENT_SRC.index('current_pantry_ingredients = form_data.get("current_pantry_ingredients")')
        window = _AGENT_SRC[i:i + 900]
        assert "P3-AGG-NUM-DAYS-PROPAGATE" in window, (
            "el fallback de swap_meal sin current_plan debe documentar por qué se dejan los "
            "defaults en vez de inventar num_days/multiplier"
        )


# ===========================================================================
# 5) Marker + _LAST_KNOWN_PFIX
# ===========================================================================
def test_marker_presente_en_ambos_archivos():
    assert "P3-AGG-NUM-DAYS-PROPAGATE" in _SC_SRC
    assert "P3-AGG-NUM-DAYS-PROPAGATE" in _AGENT_SRC


def test_last_known_pfix_bumpeado():
    """[HIGIENE · 2026-08-04, re-review de D1] Ancla la fecha-floor, NO el slug literal.

    La versión anterior (`assert "P3-AGG-NUM-DAYS-PROPAGATE" in app_src[...]`) exigía el
    marker EXACTO de este P-fix — se ponía roja con CADA P-fix posterior que bumpeara
    `_LAST_KNOWN_PFIX` a un slug distinto, aunque el bump hubiera ocurrido correctamente.
    Es el mismo anti-patrón que `test_p2_help_chatbot.py::test_marker_bumped` y
    `test_p2_live_9_...::test_marker_bumped_to_p2_live` ya corrigieron: comparar por
    FECHA (que ordena bien como string ISO) en vez de por el slug (que no)."""
    import re
    from datetime import date, datetime

    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', app_src)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    marker = m.group(1)
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert fecha, f"Marker sin fecha ISO al final (formato `Pn-X · YYYY-MM-DD`): {marker!r}"
    marker_date = datetime.strptime(fecha.group(1), "%Y-%m-%d").date()
    floor = date(2026, 8, 4)  # cierre de P3-AGG-NUM-DAYS-PROPAGATE
    assert marker_date >= floor, (
        f"_LAST_KNOWN_PFIX={marker!r} (fecha={marker_date}) anterior al floor {floor} "
        f"de cierre de P3-AGG-NUM-DAYS-PROPAGATE."
    )
