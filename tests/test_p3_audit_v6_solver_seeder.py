"""[P3-AUDIT-V6-SOLVER-SEEDER · 2026-07-31] Los P3 del audit v6 que no tenían cobertura propia.

Cada bloque cita su marker de producción. Los que ya tenían test hermano (F2 en
`test_p2_mdda_portion_solver.py`, F7 en `test_p3_solver_seeder_v4_batch.py`, F19 en
`test_p2_solver_seeder_v4_batch.py`, C2 en `test_p3_solver_seeder_v5_batch.py`, F25/F30/F16/F26 en
`test_p3_audit_v6_locales.py`) se quedan allí, junto a lo que ya protegían.
"""
import random
import re

import pytest


# ═════════════ F3 · P3-SOLVER-LSQ-ERROR-LOUD ═════════════

def test_el_fallo_del_lsq_deja_de_ser_mudo(monkeypatch, caplog):
    """`method="greedy"` colapsa TRES causas: knob OFF, sin filas de target, y LSQ crasheado.

    Un pico de crashes era indistinguible de un rollback intencional del operador.
    """
    import portion_solver as ps

    def _boom(*_a, **_k):
        raise ValueError("matriz singular de prueba")

    monkeypatch.setattr(ps, "_box_lsq", _boom)
    monkeypatch.setattr(ps, "SOLVER_LSQ", True)
    ps._LSQ_ERR_SEEN.clear()

    entries = [{"s": "100 g de pollo", "macros": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 3.6},
                "group": "protein", "movable": True}]
    with caplog.at_level("WARNING"):
        factors, method, _hi, _lo = ps._compute_scale_factors(
            entries, {"kcal": 300, "protein": 60, "carbs": 0, "fats": 0}, 0.5, 3.5, 5.0)

    assert method == "greedy", "el valor de `method` NO cambia: el caller lo compara exacto"
    assert "ValueError" in caplog.text, f"el TIPO de la excepción no llegó al log: {caplog.text[:200]}"
    assert ps._LAST_LSQ_ERROR and "ValueError" in ps._LAST_LSQ_ERROR


def test_el_fallo_del_lsq_no_inunda_el_journal(monkeypatch, caplog):
    """Corre ~28 veces por plan: un crash sistemático no puede emitir 28 WARNING idénticos."""
    import portion_solver as ps

    monkeypatch.setattr(ps, "_box_lsq", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x")))
    monkeypatch.setattr(ps, "SOLVER_LSQ", True)
    ps._LSQ_ERR_SEEN.clear()
    entries = [{"s": "100 g de pollo", "macros": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 3.6},
                "group": "protein", "movable": True}]
    with caplog.at_level("WARNING"):
        for _ in range(5):
            ps._compute_scale_factors(entries, {"protein": 60, "kcal": 0, "carbs": 0, "fats": 0},
                                      0.5, 3.5, 5.0)
    assert caplog.text.count("P3-SOLVER-LSQ-ERROR-LOUD") == 1, "debe deduplicar por tipo"


# ═════════════ F10 · P3-FEASIBILITY-FROZEN-LINE ═════════════

def test_una_linea_congelada_no_cuenta_a_techo_completo():
    """Una línea sin cantidad líder no la mueve el solver: contarla a 3.5× miente sobre la cota."""
    import portion_solver as ps

    congelada = [{"s": "Cdta de mantequilla de maní",
                  "macros": {"kcal": 45, "protein": 2, "carbs": 1.5, "fats": 4.0},
                  "group": "fats", "movable": False}]
    rep = ps._feasibility_report(congelada, {"fats": 8.0}, 0.5, [3.5])
    assert rep and rep.get("fats") == "high", (
        f"con la única fuente de grasa congelada a 4 g, un target de 8 g NO es alcanzable: {rep}"
    )

    movible = [dict(congelada[0], movable=True)]
    assert (ps._feasibility_report(movible, {"fats": 8.0}, 0.5, [3.5]) or {}).get("fats") is None, (
        "si la línea SÍ se puede escalar, 4 g × 3.5 = 14 g cubre el target"
    )


def test_las_entries_de_strings_marcan_si_son_movibles():
    """El marcador lo pone el productor, no lo adivina el consumidor."""
    import portion_solver as ps
    from nutrition_db import rescale_ingredient_string

    assert rescale_ingredient_string("Cdta de mantequilla de maní", 2.0) == "Cdta de mantequilla de maní"
    assert rescale_ingredient_string("100 g de pollo", 2.0) != "100 g de pollo"
    src = __import__("inspect").getsource(ps.solve_meal_macros)
    assert '"movable"' in src, "las entries del path de strings deben declarar `movable`"


# ═════════════ F4 · P3-SOLVER-DICT-PATH-PARITY ═════════════

def test_los_dos_caminos_del_solver_dan_el_mismo_veredicto():
    """`solve_portion_macros` no tiene tráfico de producción; mientras siga duplicada, debe ir a la par.

    El modo de fallo real no es que se rompa, sino que alguien la cablee meses después "porque es el
    mismo algoritmo" y herede un bug ya cerrado en su hermana.
    """
    from portion_solver import solve_portion_macros, solve_meal_macros

    tgt = {"kcal": 500, "protein": 45, "carbs": 40, "fats": 12}
    a = solve_portion_macros([{"name": "pollo", "quantity": 100, "unit": "g"}], tgt)
    b = solve_meal_macros(["100 g de pollo"], tgt)
    assert a["report"]["protein"]["factor"] == pytest.approx(b["report"]["protein"]["factor"]), (
        f"los dos caminos discrepan en el factor de proteína: {a['report']} vs {b['report']}"
    )
    assert a["converged"] == b["converged"], "discrepan en el veredicto de convergencia"


# ═════════════ F6 · P3-REFINE-PARALLEL-UNGATED ═════════════

def test_el_rollback_no_reactiva_el_indice_ciego():
    """tooltip-anchor de producción: P3-REFINE-PARALLEL-UNGATED"""
    import inspect
    import portion_solver as ps

    src = inspect.getsource(ps.refine_day_portions_integer)
    i = src.index("_parallel = len(raw) == len(_disp_orig)")
    # Ventana por LÍNEAS, no por bytes: el comentario que explica el fix ocupa ~10 líneas y una
    # ventana de bytes fija dejaba el `if _RPBF:` fuera — la clase de anclaje que caduca sola.
    ventana = "\n".join(src[i:].splitlines()[:24])
    codigo = "\n".join(l for l in ventana.splitlines() if not l.lstrip().startswith("#"))
    assert "if _parallel and REFINE_RAW_BY_FOOD:" not in codigo, (
        "la verificación de paralelismo vuelve a colgar del knob de ROLLBACK: apagarlo desactivaría "
        "el detector y con largos iguales reaparecería el índice ciego"
    )
    assert "_RPBF" in codigo, "el verificador debe seguir gateado por RAW_PAIR_BY_FOOD"


# ═════════════ F8 · P3-EXEMPT-NEGATED-TOKEN ═════════════

@pytest.mark.parametrize("linea,exenta,por_que", [
    ("120 g de yogurt griego natural sin azucar", False, "el descriptor NIEGA el azúcar"),
    ("1 cda de azucar morena", True, "azúcar real"),
    ("1 cdta de sal marina", True, "sal real"),
    ("aceite de oliva y sal - sin sal anadida", True, "sigue exenta por 'aceite'"),
    ("pechuga de pollo a la plancha", False, "sin token de exención"),
])
def test_la_exencion_mira_los_negadores(linea, exenta, por_que):
    from portion_solver import _exempt_matcher
    fn = _exempt_matcher(("azucar", "sal", "aceite", "miel"))
    assert fn(linea) is exenta, f"{linea!r}: {por_que}"


# ═════════════ F21 · P3-FRUIT-PAD-KEY-SYMMETRY ═════════════

def test_el_pad_de_frutas_usa_la_misma_clave_en_ambos_chequeos():
    """'Fresa' y 'Fresas' no pueden ser las "dos frutas distintas" del día."""
    import inspect
    import ai_helpers as ah

    src = inspect.getsource(ah.get_deterministic_variety_prompt)
    # [P2-SEEDER-DAYS-COUNT · 2026-08-03] El objetivo del pad pasó de `4` fijo a `_dc + 1`
    # (una fruta más que días del chunk). Solo cambia el ancla para localizar el bloque; lo que
    # este test protege —que ambos chequeos usen `_fruit_key`— se comprueba igual.
    i = src.index("while len(unique_fruits) < _dc + 1:")
    blk = src[i: i + 900]
    codigo = "\n".join(l for l in blk.splitlines() if not l.lstrip().startswith("#"))
    assert "{f.lower() for f in unique_fruits}" not in codigo, (
        "el chequeo de DUPLICADAS vuelve a comparar en crudo mientras el de PERMITIDAS normaliza"
    )
    assert "_fruit_key(f) for f in unique_fruits" in codigo


# ═════════════ F33 · P3-VEGGIE-SLOTS-CONSUME-SIX ═════════════

def test_los_seis_vegetales_sorteados_se_usan():
    """Se sorteaban 6 y la rotación consumía 4: 2 picks inertes y el log decía 6.

    [P2-SEEDER-DAYS-COUNT · 2026-08-03] El literal `(_vg6[4], _vg6[5])` que este test anclaba se
    generalizó a `2*_dc` vegetales (2 por día del chunk, que ya no son siempre 3 días). La
    comprobación pasa de textual a FUNCIONAL sobre el reparto publicado en `out_assignment`:
    verifica lo mismo —cero picks inertes— y además para cualquier tamaño de chunk, que es
    estrictamente más fuerte que el ancla anterior."""
    import inspect
    import ai_helpers as ah

    src = inspect.getsource(ah.get_deterministic_variety_prompt)
    assert "_vg6" in src, "falta el reparto en pares disjuntos sobre los vegetales sorteados"
    assert "_veg_slots = [(_vg6[2 * _i], _vg6[2 * _i + 1]) for _i in range(_dc)]" in src, (
        "los pares disjuntos ya no salen de los 2×días sorteados"
    )
    for dias in (3, 4, 6):
        asignacion: dict = {}
        ah.get_deterministic_variety_prompt("", {"mainGoal": "gain_muscle"}, None,
                                            out_assignment=asignacion, days_count=dias)
        pares = asignacion.get("veggie_pairs") or []
        assert len(pares) == dias, f"{dias} días → {len(pares)} par(es) de vegetales"
        usados = [v for par in pares for v in par]
        assert len(set(usados)) == 2 * dias, (
            f"con {dias} días se sortean {2 * dias} vegetales y solo se asignan "
            f"{len(set(usados))}: quedan picks inertes"
        )


# ═════════════ C1 · P3-LIGHT-ANCHOR-NOT-BLOCKED ═════════════

def test_el_ancla_liviana_no_nombra_lo_prohibido():
    """El mismo prompt no puede ASIGNAR un alimento y PROHIBIRLO tres párrafos antes."""
    import inspect
    import ai_helpers as ah

    src = inspect.getsource(ah.get_deterministic_variety_prompt)
    i = src.index("_light_pool = [x for x in _build_light_protein_pool(")
    blk = src[i - 400: i + 400]
    assert "_over_light" in blk, "el pool del ancla debe excluir los sobreusados que el prompt bloquea"


# ═════════════ F24 · P3-SLOT-TARGETS-BY-NAME ═════════════

def test_la_cuota_de_slot_sigue_al_nombre_no_al_indice(monkeypatch):
    """Con meal_types en orden no-canónico, la Merienda recibía la cuota del Almuerzo."""
    import prompts.day_generator as dg

    monkeypatch.setattr(dg, "DAYGEN_SLOT_TARGETS_IN_PROMPT", True, raising=False)
    tg = {"kcal": 2000, "protein": 120, "carbs": 200, "fats": 60}
    txt = dg.build_slot_targets_block(tg, ["Desayuno", "Merienda", "Almuerzo", "Cena"])
    if not txt:
        pytest.skip("el bloque está tras un knob apagado en este entorno")
    fila = [l for l in txt.splitlines() if "Merienda" in l and "kcal" in l]
    assert fila, f"no se emitió fila de Merienda: {txt}"
    kcal = int(re.search(r"≈\s*(\d+)\s*kcal", fila[0]).group(1))
    assert kcal < 500, (
        f"la Merienda recibió {kcal} kcal: es la cuota del Almuerzo, pareada por índice"
    )


def test_las_dos_meriendas_resuelven_en_orden(monkeypatch):
    import prompts.day_generator as dg

    monkeypatch.setattr(dg, "DAYGEN_SLOT_TARGETS_IN_PROMPT", True, raising=False)
    txt = dg.build_slot_targets_block(
        {"kcal": 2000, "protein": 120, "carbs": 200, "fats": 60},
        ["Desayuno", "Merienda AM", "Almuerzo", "Merienda PM", "Cena"])
    if not txt:
        pytest.skip("knob apagado en este entorno")
    assert "Merienda AM" in txt and "Merienda PM" in txt


# ═════════════ F20 · P3-CYCLE-PERSIST-AFTER-POOLS ═════════════

def test_el_ciclo_se_persiste_despues_de_los_pools_finales():
    """tooltip-anchor de producción: P3-CYCLE-PERSIST-AFTER-POOLS"""
    import inspect
    import ai_helpers as ah

    src = inspect.getsource(ah.get_deterministic_variety_prompt)
    i_dec = src.index("_cycle_persist_pending = {")
    i_dedup = src.index("unique_proteins = _dedup_list(unique_proteins)")
    i_write = src.index("_hp[\"grocery_cycle\"] = _new_cycle")
    i_shuffle = src.index("random.shuffle(unique_proteins)")
    assert i_dec < i_dedup < i_write < i_shuffle, (
        "la escritura del ciclo debe ir DESPUÉS del dedupe (pools definitivos) y ANTES del "
        "shuffle/padding (que introduce duplicados cíclicos que no son 'base')"
    )
