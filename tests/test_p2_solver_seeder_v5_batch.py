"""[P2-SOLVER-SEEDER-V5 · 2026-07-30] Los 14 P2 del audit solver+seeder v5.

Un archivo por oleada de audit (misma convención que `test_p2_solver_seeder_v4_batch.py`
y `test_p2_audit_v7_batch.py`): cada sección ancla UN gap, con su marker propio en el código.

  7  P2-SOLVER-ACHIEVED-HONEST     el solver acreditaba factores que el rescale NO aplicó
  8  P2-SOLVER-TOLERANCE-KNOB      knob definido y registrado que ningún camino leía
  9  P2-SWAP-SOLVER-FLAGS          copy-back inerte (hasattr sobre pydantic) + 4 claves ausentes
  10 P2-FREQ-TRACKING-CHUNK        el chunk worker no trackeaba frecuencias de ingredientes
  11 P2-PANTRY-FLOOR-ALL-CATS      el floor de nevera solo cubría proteínas
  12 P2-LIGHT-PROTEIN-POOL         el ancla liviana se sorteaba del pool equivocado
  13 P2-VEGGIE-PAIRS-ROTATE        par de vegetales por offset, sin garantía de distinción
  14 P2-VEGGIE-POOL-EMPTY-GUARD    pool vacío ⇒ ZeroDivisionError que bloquea la generación
  15 P2-BARIATRIC-WORSTDAY         banner de micros permanente para un perfil que no se puede reparar
  16 P2-CLOSER-NOTE-RE-UNIVERSE    el regex de notas no conocía 5 de los 8 wordings del closer
  17 P2-COOKED-KCAL-CACHE          hermano no migrado del caché sticky de catálogo
  18 P2-DAYGEN-SLOT-TARGETS-WIRED  feature inalcanzable: ningún caller pasaba daily_targets
  19 P2-SWAP-SLOT-KEY-MATCH        'Merienda Nocturna' resolvía a merienda_am (cuota ×1.5)
  20 P2-VEGGIE-CHANNEL-DAYGEN      la rotación de vegetales no tenía canal al day-generator

Los tests parser-based anclan por ORDEN RELATIVO (nunca ventana de bytes fija) y resuelven
los alias de import — las dos formas de anclaje que caducan solas en este repo.
"""
from __future__ import annotations

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, *rel.split("/")), encoding="utf-8") as f:
        return f.read()


_PS = _read("portion_solver.py")
_GO = _read("graph_orchestrator.py")
_AH = _read("ai_helpers.py")
_AG = _read("agent.py")
_CT = _read("cron_tasks.py")
_PL = _read("routers/plans.py")
_DG = _read("prompts/day_generator.py")
_SC = _read("schemas.py")


class _StubDB:
    """Resuelve macros por gramos/unidad sin red ni catálogo real."""

    def macros_from_ingredient_string(self, s: str):
        low = str(s).lower()
        if "miel" in low:      # 'Cdta de miel' → resoluble, PERO sin cantidad líder
            return {"kcal": 64.0, "protein": 0.0, "carbs": 17.0, "fats": 0.0}
        if "arroz" in low:
            import re
            m = re.match(r"\s*([\d.]+)\s*g\b", str(s))
            g = float(m.group(1)) if m else 100.0
            return {"kcal": 1.30 * g, "protein": .027 * g, "carbs": .28 * g, "fats": .003 * g}
        return {}


# ═════════════ 7 · P2-SOLVER-ACHIEVED-HONEST ═════════════

def test_7_achieved_no_acredita_factores_no_aplicados():
    """`rescale_ingredient_string` devuelve el string INTACTO cuando no hay cantidad líder
    ('Cdta de miel', 'Pechuga… (150g)' — formas observadas en prod). El solver sumaba
    igualmente `macros × f` a `achieved` y apendaba `f` a `factors_applied`:

      · la tarjeta prometía macros que el plato no entrega,
      · `converged=True` certificaba un target no alcanzado,
      · y el factor fantasma viajaba al rescale de `ingredients_raw` ⇒ la lista de compras
        compraba 1.8× de un plato cuyo display nunca cambió.
    """
    import portion_solver as ps
    res = ps.solve_meal_macros(["Cdta de miel", "100 g de arroz"],
                               {"protein": 60, "carbs": 90, "fats": 10}, db=_StubDB())
    i_miel = res["ingredients"].index("Cdta de miel")
    assert res["factors_applied"][i_miel] == 1.0, (
        "el string de la miel no cambió: su factor efectivo es 1.0, no el que pidió el LSQ")
    # achieved debe contar la miel a factor 1.0 (17 g de carbo), no escalada
    assert res["achieved"]["carbs"] == pytest.approx(
        17.0 + 0.28 * float(res["ingredients"][res["ingredients"].index(
            [s for s in res["ingredients"] if "arroz" in s][0])].split()[0]), abs=1.5)


def test_7_linea_escalable_sigue_escalando():
    """Regresión inversa: el fix NO puede apagar el escalado de las líneas que SÍ se re-escriben."""
    import portion_solver as ps
    res = ps.solve_meal_macros(["100 g de arroz"], {"protein": 20, "carbs": 60, "fats": 2},
                               db=_StubDB())
    assert res["ingredients"][0] != "100 g de arroz", "la línea escalable debe re-escribirse"
    assert res["factors_applied"][0] != 1.0


def test_7_ambas_rutas_del_solver_cubiertas():
    """`solve_portion_macros` (path dict) tiene el mismo defecto: si el `round(float(base_q)*f)`
    falla por cantidad no numérica, la cantidad NO se actualiza y `achieved` sí se acredita."""
    i = _PS.index("def solve_portion_macros(")
    body = _PS[i:_PS.index("\ndef solve_meal_macros(", i)]
    # Específico a propósito: la palabra "applied" ya existe en el `report` greedy de esa misma
    # función, así que un `"applied" in body` pasaría en vacío por la línea equivocada.
    assert "_f_eff" in body and "achieved[m] += e[\"macros\"][m] * _f_eff" in body, (
        "el path dict debe acreditar solo el factor EFECTIVAMENTE aplicado a la cantidad")


# ═════════════ 8 · P2-SOLVER-TOLERANCE-KNOB ═════════════

# Target = 5× las macros de la línea: el clamp max_scale (3.5) impide alcanzarlo, así que los
# ratios aterrizan en ~0.70 — MEDIDO, no supuesto. Ahí la tolerancia decide de verdad:
# 0.10 ⇒ no converge, 0.40 ⇒ converge. Con targets extremos (ratio ~0.01) el veredicto es el
# mismo para casi cualquier tolerancia y el test no distinguiría knob-vivo de knob-inerte.
_ARROZ_100G = {"protein": 2.7, "carbs": 28.0, "fats": 0.3}
_TARGET_X5 = {m: v * 5.0 for m, v in _ARROZ_100G.items()}


def test_8_knob_de_tolerancia_esta_cableado(monkeypatch):
    """El knob se definía y se auto-registraba en `_KNOBS_REGISTRY` (visible en
    /health/version) pero NINGÚN camino lo leía: las firmas conservaban `tolerance_pct=0.10`
    literal. Un A/B sobre ese knob comparaba la misma cosa consigo misma."""
    import portion_solver as ps
    monkeypatch.setattr(ps, "SOLVER_TOLERANCE_PCT", 0.40)
    laxo = ps.solve_meal_macros(["100 g de arroz"], _TARGET_X5, db=_StubDB())
    monkeypatch.setattr(ps, "SOLVER_TOLERANCE_PCT", 0.10)
    estricto = ps.solve_meal_macros(["100 g de arroz"], _TARGET_X5, db=_StubDB())
    assert laxo["converged"] is True and estricto["converged"] is False, (
        "MEALFIT_SOLVER_TOLERANCE_PCT no mueve `converged` — el knob sigue inerte")


def test_8_parametro_explicito_gana_al_knob(monkeypatch):
    """Patrón idéntico al de min_scale/max_scale: `None` → knob; valor → override del caller."""
    import portion_solver as ps
    monkeypatch.setattr(ps, "SOLVER_TOLERANCE_PCT", 0.10)
    res = ps.solve_meal_macros(["100 g de arroz"], _TARGET_X5, db=_StubDB(), tolerance_pct=0.40)
    assert res["converged"] is True, "el parámetro explícito debe ganar sobre el knob"


def test_8_firmas_usan_none_no_literal():
    for fn in ("def solve_portion_macros(", "def solve_meal_macros("):
        i = _PS.index(fn)
        head = _PS[i:_PS.index(") -> dict:", i)]
        assert "tolerance_pct: float = None" in head, f"{fn} conserva el literal 0.10"


# ═════════════ 9 · P2-SWAP-SOLVER-FLAGS ═════════════

def _copyback_block() -> str:
    """Sección COMPLETA del solver determinista del swap: desde la llamada al solver hasta el
    log de cierre. Recortar solo desde el marker del copy-back dejaba fuera la rama de
    abstención (que va ANTES, en el camino de fallo) — el test habría pasado en vacío."""
    i = _AG.index("_rs_ok = _rs_solver(")
    return _AG[i:_AG.index("P0-SWAP-DETERMINISTIC-RESCALE] porciones", i)]


def test_9_copyback_no_usa_hasattr_sobre_pydantic():
    """El copy-back era ESTRUCTURALMENTE inerte para su propósito declarado.

    En el path normal `res` es un `MealModel` de pydantic (structured output), y ese schema
    NO declara ningún `_solver_*` ni `ingredients_raw` → `hasattr(res, "_solver_not_converged")`
    es SIEMPRE False → las 6 claves de telemetría se descartaban en silencio. Las 9 declaradas
    sí copiaban, así que el bloque "funcionaba" visualmente: el fix v4 era un no-op en lo suyo.
    """
    blk = _copyback_block()
    assert "elif hasattr(res, _rk)" not in blk, (
        "el copy-back sigue dependiendo de hasattr sobre un modelo pydantic sin esos campos")


def test_9_copyback_incluye_las_claves_v4():
    """`_solver_failed_macros`, `_solver_infeasible` y `_solver_residuals` nacieron el MISMO día
    que este copy-back y quedaron fuera de la lista."""
    blk = _copyback_block()
    for k in ("_solver_failed_macros", "_solver_infeasible", "_solver_residuals"):
        assert f'"{k}"' in blk, f"{k} no viaja en el copy-back"


def test_9_abstencion_por_cobertura_tambien_viaja():
    """`_solver_abstained_coverage` se escribe en la rama donde el solver retorna False, y el
    copy-back vivía DENTRO de `if _rs_solver(...)` → era imposible de conservar por construcción.
    Sin ella, una comida swapeada con abstención por cobertura no deja rastro jamás."""
    blk = _copyback_block()
    assert "_solver_abstained_coverage" in blk


def test_9_los_flags_sobreviven_en_un_meal_pydantic(monkeypatch):
    """Funcional: el modo de fallo real necesita un MealModel, no un dict."""
    from schemas import MealModel
    m = MealModel(meal="Almuerzo", name="X", desc="d", prep_time="10 min", cals=500,
                  protein=30, carbs=40, fats=10, ingredients=["100 g de arroz"], recipe=["Mise en place: x"])
    assert not hasattr(m, "_solver_not_converged"), (
        "sanity: si MealModel empieza a declarar los flags, este gap cambia de forma")
    d = m.model_dump()
    assert "_solver_not_converged" not in d and "ingredients_raw" not in d


# ═════════════ 19 · P2-SWAP-SLOT-KEY-MATCH ═════════════

def test_19_merienda_nocturna_no_resuelve_a_merienda_am():
    """`k.split("_")[0] in _mt8` hace que `merienda_am` matchee 'merienda nocturna' por ser la
    primera merienda del dict → cuota 0.12 en vez de 0.08 (×1.5) en un plan bariátrico de 6
    comidas. El comentario del fix v4 usa EXACTAMENTE este caso como motivación, pero el diff
    solo corrigió el CONTEO de comidas, no el pareo de la clave."""
    import agent
    from nutrition_calculator import allocate_macros_per_slot
    slots = allocate_macros_per_slot({"kcal": 1200, "protein": 90, "carbs": 120, "fats": 40}, 6)
    assert "merienda_noche" in slots, "sanity: el split de 6 comidas trae merienda_noche"
    assert agent._resolve_swap_slot_key("Merienda Nocturna", slots) == "merienda_noche"


def test_19_resuelve_las_tres_meriendas():
    import agent
    from nutrition_calculator import allocate_macros_per_slot
    slots = allocate_macros_per_slot({"kcal": 1200, "protein": 90, "carbs": 120, "fats": 40}, 6)
    assert agent._resolve_swap_slot_key("Merienda AM", slots) == "merienda_am"
    assert agent._resolve_swap_slot_key("Merienda PM", slots) == "merienda_pm"
    assert agent._resolve_swap_slot_key("Merienda Nocturna", slots) == "merienda_noche"


def test_19_slots_principales_intactos():
    """Regresión inversa: desayuno/almuerzo/cena resuelven igual que siempre."""
    import agent
    from nutrition_calculator import allocate_macros_per_slot
    slots = allocate_macros_per_slot({"kcal": 2000, "protein": 120, "carbs": 200, "fats": 60}, 4)
    for mt in ("Desayuno", "Almuerzo", "Cena"):
        k = agent._resolve_swap_slot_key(mt, slots)
        assert k and k.startswith(mt.lower()[:5]), (mt, k)


def test_19_desconocido_devuelve_none():
    import agent
    from nutrition_calculator import allocate_macros_per_slot
    slots = allocate_macros_per_slot({"kcal": 2000, "protein": 120, "carbs": 200, "fats": 60}, 4)
    assert agent._resolve_swap_slot_key("Pre-entreno", slots) is None


# ═════════════ 11-14 · seeder (ai_helpers) ═════════════

def _seeder_prompt(**fd):
    import ai_helpers as ah
    return ah.get_deterministic_variety_prompt("", dict(fd), user_id=None)


def _rotation_block() -> str:
    i = _AH.index("ROTATION MODE")
    return _AH[i:_AH.index("FORCED INGREDIENT INJECTION", i)]


# ── 11 · P2-PANTRY-FLOOR-ALL-CATS ──

def test_11_floor_de_nevera_cubre_las_cuatro_categorias():
    """`P2-PANTRY-ROTATION-FLOOR` (v4) puso un PISO a las proteínas de la nevera (si aporta
    menos del mínimo, se completa con el sorteo) pero dejó las otras 3 categorías con el
    reemplazo incondicional `if extracted_c: unique_carbs = extracted_c`.

    Con UN solo carbo reconocible tras `/inventory/consume`, el padding lo cicla y los 3 días
    salen con la misma base — exactamente la monotonía (39% de días) que el fix v4 describe
    como el bug que venía a cerrar. Con vegetales es peor: 1 extraído ⇒ los 6 slots idénticos."""
    blk = _rotation_block()
    for cat, ext in (("unique_carbs", "extracted_c"), ("unique_veggies", "extracted_v"),
                     ("unique_fruits", "extracted_f")):
        assert f"{cat} = {ext}\n" not in blk and f"{cat} = {ext} " not in blk, (
            f"{cat} sigue con reemplazo incondicional (sin piso)")
        assert f"{cat} = _floor_pool({ext}" in blk, (
            f"{cat} debe pasar por el mismo piso que las proteínas")


def test_11_un_solo_carbo_no_monopoliza_los_tres_dias(caplog):
    import logging
    caplog.set_level(logging.INFO)
    out = _seeder_prompt(current_pantry_ingredients=["Arroz Blanco", "Pechuga de pollo",
                                                     "Carne de res molida"])
    # las 3 opciones del prompt no pueden compartir la MISMA base de carbohidrato
    bases = [ln for ln in out.splitlines() if "DEBE incluir obligatoriamente" in ln]
    assert len(bases) == 3, bases
    assert len(set(bases)) == 3, "las 3 opciones del día son idénticas (pool de 1 ciclado)"


# ── 12 · P2-LIGHT-PROTEIN-POOL ──

def test_12_light_pool_puede_contener_lacteos():
    """Los tokens 'queso'/'yogur'/'ricotta' se filtraban contra `filtered_veggies`, donde NO
    existe ninguno (viven en DOMINICAN_PROTEINS) → 3 de los 8 tokens no podían matchear jamás
    y el ancla acababa siendo 100% frutos secos: reforzando el 'maní en 4 de 12' que el knob
    venía a curar."""
    import ai_helpers as ah
    from constants import DOMINICAN_PROTEINS, DOMINICAN_VEGGIES_FATS
    assert not [v for v in DOMINICAN_VEGGIES_FATS
                if any(t in v.lower() for t in ("queso", "yogur", "ricotta"))], (
        "sanity: los lácteos NO están en el catálogo de vegetales/grasas")
    pool = ah._build_light_protein_pool(DOMINICAN_VEGGIES_FATS, DOMINICAN_PROTEINS, bariatric=False)
    assert any(any(t in p.lower() for t in ("queso", "yogur", "ricotta")) for p in pool), (
        "el pool del ancla liviana sigue sin poder ofrecer un lácteo")


def test_12_bariatrica_excluye_frutos_secos_enteros():
    """El slice posicional `[:4]` nunca consultaba `veggie_weights`, así que el penalty ×0.1
    por riesgo obstructivo del pouch no aplicaba: el ancla priorizaba 'Maní'/'Nueces' ENTERAS
    para un perfil post-quirúrgico. Las formas molida/mantequilla/fileteada sí son seguras."""
    import ai_helpers as ah
    from constants import DOMINICAN_PROTEINS, DOMINICAN_VEGGIES_FATS
    pool = ah._build_light_protein_pool(DOMINICAN_VEGGIES_FATS, DOMINICAN_PROTEINS, bariatric=True)
    malos = [p for p in pool
             if any(t in p.lower() for t in ("nuez", "nueces", "almendras", "mani", "maní"))
             and not any(t in p.lower() for t in ("mantequilla", "molid", "filetead"))]
    assert not malos, f"formas enteras de frutos secos en un pool bariátrico: {malos}"


# ── 13 · P2-VEGGIE-PAIRS-ROTATE ──

def test_13_par_de_vegetales_siempre_distinto():
    """`veggie_i = chosen[i]` / `veggie_ib = chosen[i+3]` es offset POSICIONAL puro. Con 3
    vegetales únicos el padding produce [a,b,c,a,b,c] y, tras el shuffle, ~20% de los días
    reciben el MISMO vegetal dos veces (40% con 2 únicos, 100% con 1) — contradiciendo la
    instrucción "2 vegetales distintos" del propio prompt. Frutas y carbos ya migraron a
    `_rotate_pairs` con dedupe; los vegetales se quedaron fuera del contrato."""
    import ai_helpers as ah
    for pool in (["A", "B", "C"], ["A", "B"], ["A", "B", "C", "D", "E", "F"]):
        slots = ah._rotate_pairs(pool)
        if slots:
            for a, b in slots:
                assert a != b, f"par idéntico con pool {pool}: {(a, b)}"


def test_13_el_seeder_usa_rotate_pairs_para_vegetales():
    assert "_rotate_pairs(chosen_veggies" in _AH, (
        "el par de vegetales debe salir del helper con dedupe, como fruta y carbo")
    for off in ("chosen_veggies[3]", "chosen_veggies[4]", "chosen_veggies[5]"):
        assert off not in _AH, (
            f"queda el offset posicional {off}: es el que producía 'Zanahoria y Zanahoria'")


def test_13_pares_distintos_en_el_prompt_real():
    """Funcional sobre el prompt entregado: ninguna opción del día puede asignar el mismo
    vegetal dos veces. Con pools grandes el bug es raro; se fuerza el caso estrecho."""
    import re
    out = _seeder_prompt(current_pantry_ingredients=["Zanahoria", "Brócoli", "Tayota",
                                                     "Pechuga de pollo", "Arroz Blanco"])
    for ln in out.splitlines():
        m = re.search(r"acompañante vegetal/grasa:\s*([^.]+)\.", ln)
        if m and " y " in m.group(1):
            a, b = [x.strip() for x in m.group(1).split(" y ", 1)]
            assert a != b, f"par de vegetales idéntico en el prompt: {m.group(1)!r}"


# ── 14 · P2-VEGGIE-POOL-EMPTY-GUARD ──

def test_14_pool_de_vegetales_vacio_no_revienta(monkeypatch):
    """El guard-clause del seeder solo cubría proteínas y carbos. Con `available_veggies=[]`
    el padding hacía `_base_veggies[len(...) % len([])]` → ZeroDivisionError en
    `_build_shared_context`, que ningún caller envuelve: el nodo falla, el retry repite el
    mismo crash y la generación queda BLOQUEADA para ese usuario hasta que edite sus dislikes."""
    import constants
    import ai_helpers as ah
    monkeypatch.setattr(ah, "_get_fast_filtered_catalogs",
                        lambda *a, **k: (list(constants.DOMINICAN_PROTEINS),
                                         list(constants.DOMINICAN_CARBS), [], []),
                        raising=False)
    monkeypatch.setattr(constants, "_get_fast_filtered_catalogs",
                        lambda *a, **k: (list(constants.DOMINICAN_PROTEINS),
                                         list(constants.DOMINICAN_CARBS), [], []))
    out = ah.get_deterministic_variety_prompt("", {"allergies": [], "dislikes": []}, user_id=None)
    assert isinstance(out, str)   # no lanza: eso es todo lo que se le pide


# ═════════════ 15 · P2-BARIATRIC-WORSTDAY ═════════════

def _panel_degraded_body() -> str:
    i = _GO.index("def _maybe_mark_panel_degraded(")
    return _GO[i:_GO.index("\ndef ", i + 10)]


def test_15_bariatrica_no_recibe_banner_micro_worst_day():
    """DETECTAR SIN PODER REPARAR.

    El micro-closer hace **skip TOTAL** en perfiles bariátricos (`MICRO_CLOSER_BARIATRIC_SKIP`,
    ON por default) porque el protocolo ASMBS cubre esos micros con SUPLEMENTACIÓN de por vida.
    Pero el sub-check 2b marca `_quality_degraded reason=micro_worst_day` cuando ≥2 micros
    accionables quedan bajo el piso en algún día — y a 1000-1200 kcal pouch-limitadas eso es
    ESTRUCTURAL (fibra 25 g, hierro 18 mg, calcio).

    Resultado: banner degradado permanente por construcción, que además se re-marca en cada
    swap/chat vía `apply_update_condition_ceilings`… y el único actor que podría limpiarlo está
    deshabilitado a propósito para ese mismo perfil. `_MICRO_WORSTDAY_EXCLUDE` excluye por MICRO
    (vit_d/omega3/vit_e), nunca por PERFIL."""
    body = _panel_degraded_body()
    i2b = body.index('reason = "micro_worst_day"')
    ctx = body[:i2b]
    assert "bariatric" in ctx.lower(), (
        "el sub-check 2b no consulta el perfil bariátrico antes de marcar el banner")


def test_15_no_bariatrico_conserva_el_banner():
    """Regresión inversa: la exención es por PERFIL, no un apagado del banner."""
    body = _panel_degraded_body()
    assert 'reason = "micro_worst_day"' in body, "el banner debe seguir existiendo"


# ═════════════ 16 · P2-CLOSER-NOTE-RE-UNIVERSE ═════════════

def test_16_el_regex_de_notas_cubre_todos_los_wordings():
    """`_CLOSER_NOTE_FOOD_RE` exigía pre ∈ {Escurre e incorpora, Incorpora, Cocina} y post ∈
    {a la preparación, a la plancha, (ya viene cocido) a la preparación}. De los 8 wordings que
    `_closer_protein_step_text` puede emitir, matcheaban 3.

    El que importa es el de food-safety: en un plato de olla con hint de precocido y línea de
    pescado FRESCO, el paso dice 'Escurre e incorpora … (ya viene cocido) al guiso' — el regex
    falla por el post 'al guiso', el `continue` salta el paso, y la rama P1-CLOSER-FRESH-COCIDO
    (que reescribe la nota cuando la línea es fresca) NUNCA corre: el usuario recibe la
    instrucción de incorporar pescado crudo sin cocerlo.

    El test DERIVA los wordings del SSOT invocando la función con todas las combinaciones — así
    un wording nuevo rompe este test antes de nacer huérfano (alinear al universo canónico, no
    mantener una lista paralela que crece por incidente)."""
    import itertools
    import re
    import graph_orchestrator as go
    emitidos = set()
    for no_cook, blended, stewy in itertools.product((False, True), repeat=3):
        for nm in ("Filete de pescado blanco", "Lentejas", "Queso fresco", "Atún en agua"):
            try:
                t = go._closer_protein_step_text(nm, no_cook, blended=blended, stewy=stewy)
            except TypeError:
                t = go._closer_protein_step_text(nm, no_cook, blended=blended)
            if isinstance(t, str) and t.strip():
                emitidos.add(t.strip())
    assert len(emitidos) >= 5, f"sanity: se esperaban varios wordings, got {len(emitidos)}"
    huerfanos = [t for t in emitidos if not go._CLOSER_NOTE_FOOD_RE.match(t.lstrip("💪 ").strip())]
    assert not huerfanos, (
        "wordings del closer que el regex de notas NO reconoce (el name-align y el rebuild "
        f"food-safety son letra muerta para ellos):\n  - " + "\n  - ".join(sorted(huerfanos)))


# ═════════════ 17 · P2-COOKED-KCAL-CACHE ═════════════

def test_17_cache_cocido_no_sella_el_fallo(monkeypatch):
    """Hermano NO migrado de `P1-CATALOG-INDEX-NO-STICKY`.

    `get_master_ingredients()` devuelve `[]` SIN excepción cuando el pool no está abierto o hay
    un blip de Neon → el índice se construye vacío, el `except` no dispara (no hubo raise), y el
    guard `if _COOKED_CATALOG_KCAL_CACHE is not None` sirve ese `{}` para toda la vida del
    worker. Consecuencia: el rewrite cocido→seco de granos queda INERTE y la lista compra ~2.76×
    de arroz, indistinguible de un motor apagado hasta el restart."""
    import graph_orchestrator as go
    import shopping_calculator as sc
    monkeypatch.setattr(go, "_COOKED_CATALOG_KCAL_CACHE", None, raising=False)
    # El gate sirve el vacío durante `MEALFIT_CATALOG_INDEX_NEG_TTL_S` (30 s) para no escanear el
    # catálogo por línea mientras está caído de verdad. Lo que se prueba aquí es que el vacío se
    # REINTENTA, no que se reintente al instante → TTL 0 en el test.
    monkeypatch.setattr(go, "_CATALOG_INDEX_NEG_TTL_S", 0.0, raising=False)
    llamadas = {"n": 0}

    def _vacio(*a, **k):
        llamadas["n"] += 1
        return []

    monkeypatch.setattr(sc, "get_master_ingredients", _vacio)
    assert go._catalog_kcal_by_name() == {}
    n1 = llamadas["n"]
    # ahora el catálogo responde: la SEGUNDA llamada debe reconstruir, no servir el {} sellado
    monkeypatch.setattr(sc, "get_master_ingredients",
                        lambda *a, **k: [{"name": "Arroz blanco", "kcal_per_100g": 358.6}])
    idx = go._catalog_kcal_by_name()
    assert idx, (
        f"el caché sirvió el fallo sellado (get_master_ingredients llamado {n1} vez/veces y "
        f"nunca reintentado): el rewrite cocido→seco queda muerto hasta el restart")


def test_17_usa_el_patron_ssot_de_los_hermanos():
    i = _GO.index("def _catalog_kcal_by_name(")
    body = _GO[i:_GO.index("\ndef ", i + 10)]
    assert "_catalog_index_should_rebuild" in body, "debe usar el gate con negative-TTL"
    assert "_catalog_index_note_failure" in body, "debe registrar el fallo, no sellarlo"


# ═════════════ 10 · P2-FREQ-TRACKING-CHUNK ═════════════

def test_10_el_chunk_worker_trackea_frecuencias():
    """`P2-FREQ-TRACKING-CHUNKED` (v4) cubrió los 2 sitios del REQUEST inicial; el worker quedó
    fuera y `increment_ingredient_frequencies` no tenía UN SOLO callsite en cron_tasks.py.

    Con la ventana rolling, tras unas semanas la mayoría de los días que el usuario come vienen
    de chunks de refill/catchup — justo los que nunca se contaban. El sorteo 1/(freq+1) y la
    fatiga por recencia (que pesa los últimos 3 días, casi siempre de refill) re-ofrecían con
    peso MÁXIMO el salmón que el usuario llevaba dos semanas recibiendo."""
    assert "_track_ingredient_frequencies" in _CT, (
        "el chunk worker sigue sin trackear frecuencias de ingredientes")
    i = _CT.index("P2-FREQ-TRACKING-CHUNK ·")
    blk = _CT[i:_CT.index("except Exception", i)]
    assert "TRACK_FREQ_ON_CHUNKED" in blk, "debe respetar el knob de rollback existente"
    assert "sorted_new_days" in blk, (
        "debe pasar SOLO los días nuevos del chunk: el request inicial ya contó los suyos y "
        "double-contarlos sesgaría el sorteo al revés")


# ═════════════ 18 · P2-DAYGEN-SLOT-TARGETS-WIRED ═════════════

def test_18_el_caller_de_produccion_pasa_daily_targets():
    """El gate era `knob AND daily_targets` y NINGUNO de los callers pasaba el argumento →
    encender `MEALFIT_DAYGEN_SLOT_TARGETS_IN_PROMPT` era un NO-OP silencioso: el canario que el
    docstring exige habría dado un prompt byte-idéntico al de control y se habría concluido que
    la cuota por slot 'no ayuda', enterrando el fix de las 13/16 infactibilidades de grasa."""
    i = _GO.index("assignment_context = build_day_assignment_context(")
    linea = _GO[i:_GO.index(")\n", i)]
    assert "daily_targets=" in linea, f"el caller de generate_single_day no pasa targets: {linea!r}"


def test_18_el_bloque_aparece_con_el_knob_encendido(monkeypatch):
    """Funcional: con knob ON y targets reales, el bloque de cuotas entra en el prompt."""
    monkeypatch.setenv("MEALFIT_DAYGEN_SLOT_TARGETS_IN_PROMPT", "true")
    from prompts.day_generator import build_day_assignment_context
    sk = {"protein_pool": ["Pollo"], "carb_pool": ["Arroz"], "fruit_pool": ["Guineo"],
          "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"]}
    con = build_day_assignment_context(sk, 1, daily_targets={"kcal": 2000, "protein": 120,
                                                             "carbs": 200, "fats": 60})
    sin = build_day_assignment_context(sk, 1)
    assert con != sin, "el knob ON con targets debe cambiar el prompt"


# ═════════════ 20 · P2-VEGGIE-CHANNEL-DAYGEN ═════════════

def test_20_el_esqueleto_puede_transportar_vegetales():
    from schemas import DaySkeletonModel
    assert "veggie_pool" in DaySkeletonModel.model_fields, (
        "DaySkeletonModel no declara el campo: la decisión del seeder no tiene canal tipado")
    d = DaySkeletonModel(day=1, assigned_technique="Guisado", protein_pool=["Pollo"],
                         carb_pool=["Arroz"], fruit_pool=["Guineo"],
                         meal_types=["Desayuno"], brief_concept="x")
    assert d.veggie_pool == [], "default vacío: los planners que no lo emitan siguen validando"


def test_20_el_daygen_emite_los_vegetales_asignados():
    from prompts.day_generator import build_day_assignment_context
    sk = {"protein_pool": ["Pollo"], "carb_pool": ["Arroz"], "fruit_pool": ["Guineo"],
          "veggie_pool": ["Brócoli", "Aguacate"], "meal_types": ["Desayuno", "Almuerzo"]}
    con = build_day_assignment_context(sk, 1)
    assert "Brócoli" in con and "Aguacate" in con, (
        "el day-gen no recibe la rotación de vegetales: genera ciego y cae en su default")


def test_20_un_solo_vegetal_no_crea_restriccion_insatisfacible():
    from prompts.day_generator import build_day_assignment_context
    sk = {"protein_pool": ["Pollo"], "carb_pool": ["Arroz"], "fruit_pool": ["Guineo"],
          "veggie_pool": ["Brócoli"], "meal_types": ["Desayuno"]}
    assert "Vegetales/Grasas Asignados" not in build_day_assignment_context(sk, 1)


def test_20_el_seeder_publica_su_reparto_como_dato():
    """El planner LLM no puede garantizar que copie el dato, así que el reparto viaja por un
    canal ESTRUCTURAL (`out_assignment`) y no por la prosa del prompt.

    La primera versión de este fix parseaba el prompt con un regex y devolvía SIEMPRE [] — los
    dos vegetales del día están en frases distintas ("…acompañante vegetal/grasa: Berro." y "En
    las DEMÁS comidas del día …, usa: Bok choy."). Habría dejado el campo vacío para siempre con
    los tests estructurales en verde: exactamente el motor-inerte que este audit persigue."""
    import ai_helpers as ah
    asig: dict = {}
    ah.get_deterministic_variety_prompt("", {"current_pantry_ingredients": ["Pechuga de pollo",
                                                                           "Arroz Blanco"]},
                                        user_id=None, out_assignment=asig)
    pares = asig.get("veggie_pairs") or []
    assert len(pares) >= 3, f"esperados ≥3 pares (uno por opción), got {pares}"
    for a, b in pares:
        assert a and b and a.lower() != b.lower(), (a, b)


def test_20_el_caller_sin_out_assignment_no_cambia():
    """El parámetro es opcional: los callers existentes no se enteran."""
    import ai_helpers as ah
    p = ah.get_deterministic_variety_prompt("", {"allergies": []}, user_id=None)
    assert isinstance(p, str) and p


def test_20_extractor_es_fail_safe():
    import graph_orchestrator as go
    assert go._extract_seeder_veggie_pairs({}) == []
    assert go._extract_seeder_veggie_pairs(None) == []
    assert go._extract_seeder_veggie_pairs({"seeder_assignment": {"veggie_pairs": [("A", "A")]}}) == []
    assert go._extract_seeder_veggie_pairs(
        {"seeder_assignment": {"veggie_pairs": [("Brócoli", "Aguacate")]}}) == [("Brócoli", "Aguacate")]
