"""[P2-SOLVER-PIN-FROZEN · 2026-08-03] El LSQ gastaba presupuesto de ajuste en lineas que el
propio solver ya sabia INAMOVIBLES.

`solve_meal_macros` marca cada entry con `movable` (P3-FEASIBILITY-FROZEN-LINE: una linea sin
cantidad lider — «Pechuga a la plancha (150g)», «Cdta de mantequilla de mani» — no la puede
reescribir `rescale_ingredient_string`), pero `_compute_scale_factors` NO lo consumia: el
optimizador repartia el target contando con que esa linea escala hasta 5.0x, y el descarte
post-hoc `_f_eff = 1.0` (P2-SOLVER-ACHIEVED-HONEST) tiraba el factor fantasma DESPUES de que ya
habia diluido a las moviles.

Caso medido con el fixture de abajo (target proteina 80 g):
  antes → factor crudo de la congelada 1.4276 (fantasma) · movil 0.503 · achieved P = 60.1 g
  ahora → congelada clavada a 1.0             · movil 1.231 · achieved P = 79.7 g

⚠️ EL PIN VIVE SOLO EN LA RAMA LSQ. El defecto espejo del greedy se implemento, se MIDIO sobre
312 comidas con target de 4 filas y se RETIRO: LSQ mejora 189 / empeora 19, pero greedy mejora 74
/ empeora 116 (peor caso 0.236 → 0.424). Ver `test_el_greedy_no_lleva_el_pin_ni_con_el_knob_on`.

100% OFFLINE: `StubDB` sustituye a `IngredientNutritionDB` (el .env apunta a PRODUCCION).
"""
import inspect
import re

import pytest

import portion_solver as ps


PECHUGA = "Pechuga a la plancha (150g)"   # congelada: sin cantidad lider, solo gram-hint
MANI = "Cdta de mantequilla de mani"      # congelada: unit-led sin numero
POLLO = "100 g de pollo"                  # movible
ARROZ = "150 g de arroz"                  # movible
AVENA = "1 taza de avena (50g)"           # movible

_MAC = {
    PECHUGA: {"kcal": 247.5, "protein": 46.5, "carbs": 0.0, "fats": 5.4},
    MANI: {"kcal": 45.0, "protein": 2.0, "carbs": 1.5, "fats": 4.0},
    POLLO: {"kcal": 155.0, "protein": 27.0, "carbs": 0.0, "fats": 5.0},
    ARROZ: {"kcal": 195.0, "protein": 4.0, "carbs": 42.0, "fats": 0.5},
    AVENA: {"kcal": 190.0, "protein": 6.5, "carbs": 33.0, "fats": 3.5},
}


class StubDB:
    """Catalogo offline: cero DB, cero red."""

    def macros_from_ingredient_string(self, s):
        return dict(_MAC[s]) if s in _MAC else None

    def macros_for_line(self, qty, unit, name):
        for k, v in _MAC.items():
            if name and name.lower() in k.lower():
                return dict(v)
        return None


TGT_BRIEF = {"protein": 80.0, "carbs": 0.0, "fats": 10.0}


def _solve(lines, tgt=None, **kw):
    return ps.solve_meal_macros(list(lines), dict(tgt or TGT_BRIEF), db=StubDB(), **kw)


def _entry(s, group, movable):
    return {"s": s, "macros": dict(_MAC[s]), "group": group, "movable": movable}


# ═════════════ 1 · el LSQ deja de repartir target a lo que no se mueve ═════════════

def test_la_congelada_recibe_factor_exactamente_1_no_un_fantasma():
    """El factor de una linea `movable=False` sale del optimizador ya clavado en 1.0.

    Antes salia 1.4276 y el descarte post-hoc lo borraba — pero el reparto ya estaba hecho.
    """
    entries = [_entry(PECHUGA, "protein", False), _entry(POLLO, "protein", True)]
    factors, method, _sh, _sl, _err = ps._compute_scale_factors(entries, TGT_BRIEF, 0.3, 3.5, 5.0)
    assert method == "lsq"
    assert factors[0] == 1.0, (
        f"la linea congelada sigue recibiendo un factor fantasma: {factors[0]!r}")


def test_la_movil_absorbe_el_residual_que_la_congelada_no_puede_entregar():
    res = _solve([PECHUGA, POLLO])
    f_frozen, f_movil = res["factors_applied"]
    assert f_frozen == 1.0
    # linea base medida (sin pin): 0.503 — la movil salia DILUIDA hacia abajo.
    assert f_movil > 1.15, f"la movil no absorbio el residual: {f_movil!r}"
    # proteina entregada: 60.1 g antes, ~79.7 g ahora sobre un target de 80.
    assert res["achieved"]["protein"] > 75.0, (
        f"la proteina sigue sub-entregada: {res['achieved']}")
    assert abs(res["achieved"]["protein"] / 80.0 - 1.0) < 0.05


def test_el_greedy_no_lleva_el_pin_ni_con_el_knob_on(monkeypatch):
    """ASIMETRIA DELIBERADA. El defecto espejo del greedy (`current` cuenta la congelada) se
    implemento, se MIDIO y se retiro: barrido de 312 comidas con target de 4 filas →
    LSQ mejora 189 / empeora 19, pero GREEDY mejora 74 / empeora 116 (peor caso 0.236 → 0.424).

    Causa: el greedy modela solo el macro dominante de su grupo, asi que al cubrir el residual
    arrastra la kcal/grasa embebida sin nada que lo frene. Ademas solo corre como fallback.

    Este test ancla el comportamiento PRE-FIX del greedy: si alguien "completa la migracion"
    por simetria con el LSQ, falla aqui y tiene que re-medir el barrido antes de mergear.
    """
    monkeypatch.setattr(ps, "SOLVER_LSQ", False)
    con_pin = _solve([PECHUGA, POLLO])
    assert con_pin["method"] == "greedy"
    # Comportamiento historico: factor 1.0884 repartido TAMBIEN a la congelada ⇒ 75.9 g.
    # (Con pin habria dado 1.2407 y 80.0 g — mejor en ESTA comida, peor en el agregado.)
    assert con_pin["factors_applied"][1] == pytest.approx(1.0884, abs=1e-3), (
        f"el greedy dejo de comportarse como pre-fix: {con_pin['factors_applied']}")
    assert con_pin["achieved"]["protein"] == pytest.approx(75.9, abs=0.2)

    monkeypatch.setattr(ps, "SOLVER_PIN_FROZEN", False)
    sin_pin = _solve([PECHUGA, POLLO])
    assert con_pin == sin_pin, "el knob NO debe mover ni un valor en la rama greedy"


# ═════════════ 2 · el knob apaga el cambio por completo ═════════════

def test_el_knob_off_restaura_el_comportamiento_previo_byte_a_byte(monkeypatch):
    """El rollback devuelve el dict de retorno COMPLETO, no solo el factor que miramos."""
    con_pin = _solve([PECHUGA, POLLO])
    monkeypatch.setattr(ps, "SOLVER_PIN_FROZEN", False)
    sin_pin = _solve([PECHUGA, POLLO])
    # linea base medida sobre el arbol sin el fix (golden de 23 casos):
    assert sin_pin["factors_applied"] == [1.0, pytest.approx(0.503, abs=1e-3)], (
        f"el rollback no reprodujo la linea base medida: {sin_pin['factors_applied']}")
    assert sin_pin["achieved"] == {"kcal": 325.5, "protein": 60.1, "carbs": 0.0, "fats": 7.9}
    assert sin_pin["ingredients"] == [PECHUGA, "50.3 g de pollo"]
    assert sin_pin["saturated_hi"] == 0 and sin_pin["saturated_lo"] == 0
    assert sin_pin["converged"] is False
    # ...y el pin SI cambia el dict: si no, el test de arriba seria vacuo.
    assert con_pin != sin_pin


def test_el_knob_existe_y_se_auto_registra_en_el_registry():
    """Convencion del repo: knobs `MEALFIT_*` via helper, nunca `os.environ` crudo."""
    assert isinstance(ps.SOLVER_PIN_FROZEN, bool)
    try:
        from knobs import get_knobs_registry_snapshot
    except Exception:  # pragma: no cover - knobs siempre disponible
        pytest.skip("knobs no importable")
    assert "MEALFIT_SOLVER_PIN_FROZEN" in get_knobs_registry_snapshot(), (
        "el knob no se auto-registro: invisible en /health/version")


# ═════════════ 3 · sin lineas congeladas nada cambia ═════════════

@pytest.mark.parametrize("lines,tgt", [
    ([POLLO, ARROZ], {"kcal": 600, "protein": 45, "carbs": 55, "fats": 12}),
    ([AVENA, ARROZ], {"kcal": 500, "protein": 15, "carbs": 70, "fats": 14}),
    ([POLLO], {"protein": 40, "carbs": 0, "fats": 6}),
])
def test_una_comida_sin_congeladas_sale_identica_con_el_knob_on_y_off(lines, tgt, monkeypatch):
    on = _solve(lines, tgt)
    monkeypatch.setattr(ps, "SOLVER_PIN_FROZEN", False)
    off = _solve(lines, tgt)
    assert on == off, "el pin toco una comida que no tiene ni una linea congelada"


# ═════════════ 4 · la telemetria de saturacion no se contamina ═════════════

def test_la_congelada_no_se_cuenta_como_clamp_saturado():
    """`lo=hi=1.0` haria que toda congelada contara como `saturated_hi` — la serie que motivo
    S-P2-a ("~74% de meals saturando el clamp") quedaria inflada por lineas que ni se movieron."""
    entries = [_entry(PECHUGA, "protein", False), _entry(POLLO, "protein", True)]
    _f, _m, sat_hi, sat_lo, _err = ps._compute_scale_factors(entries, TGT_BRIEF, 0.3, 3.5, 5.0)
    assert sat_hi == 0 and sat_lo == 0, (
        f"una linea PINNED no satura ningun clamp: hi={sat_hi} lo={sat_lo}")


# ═════════════ 5 · paridad del path dict (default movable=True) ═════════════

def test_el_path_dict_no_declara_movable_y_conserva_su_comportamiento(monkeypatch):
    """`solve_portion_macros` no computa `movable`: el default `True` debe dejarlo intacto."""
    ings = [{"name": "pollo", "quantity": 100, "unit": "g"},
            {"name": "arroz", "quantity": 150, "unit": "g"}]
    tgt = {"kcal": 600, "protein": 45, "carbs": 55, "fats": 12}
    on = ps.solve_portion_macros(list(ings), dict(tgt), db=StubDB())
    monkeypatch.setattr(ps, "SOLVER_PIN_FROZEN", False)
    off = ps.solve_portion_macros(list(ings), dict(tgt), db=StubDB())
    assert on == off, "el pin altero el path dict, que no declara `movable`"


def test_entries_sin_la_clave_movable_se_asumen_movibles():
    sin_clave = [{"s": POLLO, "macros": dict(_MAC[POLLO]), "group": "protein"}]
    con_clave = [dict(sin_clave[0], movable=True)]
    assert (ps._compute_scale_factors(sin_clave, TGT_BRIEF, 0.3, 3.5, 5.0)
            == ps._compute_scale_factors(con_clave, TGT_BRIEF, 0.3, 3.5, 5.0))


# ═════════════ 6 · una sola fuente de verdad sobre "frozen" ═════════════

def test_la_factibilidad_y_el_solver_dejan_de_contradecirse(monkeypatch):
    """P3-FEASIBILITY-FROZEN-LINE ya clavaba la congelada a 1.0 en las COTAS del reporte, mientras
    el solver la escalaba hasta 3.5x: DOS modelos del mismo plato que no se hablaban.

    Plato: «Cdta de mantequilla de mani» (4.0 g de grasa, CONGELADA) + «150 g de arroz» (0.5 g,
    movible). Target REALISTA de 4 filas — la forma que arma produccion. La cota frozen-aware de
    grasa es 4.0x1.0 + 0.5x3.5 = 5.75 g < 8 ⇒ veredicto 'high' (falta un PORTADOR, no escalado).

    ⚠️ El fixture NO usa un target de una sola fila. Con `{protein:0, carbs:0, fats:8}` la movil se
    va a x3.5 y "alcanza" justo la cota publicada — un numero bonito que es ARTEFACTO de haber
    quitado las filas que lo frenarian. Con las 4 filas el mismo plato da x1.18 y el deficit de
    grasa PERSISTE, que es la verdad: no hay portador.

    Lo que el pin cambia aqui no es llegar al target — es dejar de RETENER la unica linea que se
    puede mover porque el optimizador creia que el mani escalaria: 0.891 → 1.182.
    """
    lines = [MANI, ARROZ]
    tgt = {"kcal": 300, "protein": 12, "carbs": 40, "fats": 8}
    con_pin = _solve(lines, tgt)
    monkeypatch.setattr(ps, "SOLVER_PIN_FROZEN", False)
    sin_pin = _solve(lines, tgt)

    # La senal de factibilidad ya era frozen-aware y NO se toca: mismo veredicto en ambos.
    assert (con_pin["infeasible"] or {}).get("fats") == "high", (
        f"el fixture dejo de ejercitar el caso infactible: {con_pin['infeasible']}")
    assert (sin_pin["infeasible"] or {}).get("fats") == "high"
    # La congelada nunca se mueve (eso ya lo garantizaba `_f_eff`); lo que cambia es la movil.
    assert con_pin["factors_applied"][0] == sin_pin["factors_applied"][0] == 1.0
    assert sin_pin["factors_applied"][1] == pytest.approx(0.8913, abs=1e-3)
    assert con_pin["factors_applied"][1] == pytest.approx(1.1815, abs=1e-3)
    assert con_pin["factors_applied"][1] > sin_pin["factors_applied"][1], (
        "el solver sigue reteniendo la movil por contar con una linea que no se mueve")
    # El deficit de grasa que QUEDA es el que el reporte explica, no escalado sin usar.
    assert con_pin["achieved"]["fats"] < tgt["fats"]
    # Honestidad: aqui la unica movil es una bomba de carbos, asi que soltarla sobre-entrega
    # carbos (38.9 → 51.1 g). El veredicto agregado NO sale de esta comida sino del barrido de
    # 312 (LSQ: mejora 189 / empeora 19). Este test ancla el mecanismo, no la victoria.
    assert con_pin["achieved"]["carbs"] > sin_pin["achieved"]["carbs"]


def test_el_retorno_declara_cuantas_lineas_estan_congeladas():
    """`frozen_lines` es la senal honesta para "una congelada que el target QUERRIA mover es
    informacion": ninguna clave del retorno lo decia. Aditiva — los consumidores leen por `.get()`.
    """
    assert _solve([PECHUGA, POLLO])["frozen_lines"] == 1
    assert _solve([PECHUGA, MANI], {"kcal": 300, "protein": 20, "carbs": 5, "fats": 10})[
        "frozen_lines"] == 2
    assert _solve([POLLO, ARROZ], {"kcal": 600, "protein": 45, "carbs": 55, "fats": 12})[
        "frozen_lines"] == 0
    # El path dict no declara `movable`: expone la clave igual (contrato comun) y siempre en 0.
    dicts = ps.solve_portion_macros([{"name": "pollo", "quantity": 100, "unit": "g"}],
                                    {"kcal": 300, "protein": 40, "carbs": 0, "fats": 6},
                                    db=StubDB())
    assert dicts["frozen_lines"] == 0


def test_el_marker_vive_en_el_codigo_que_lo_implementa():
    src = inspect.getsource(ps._compute_scale_factors)
    assert "P2-SOLVER-PIN-FROZEN" in src, "marker ausente en `_compute_scale_factors`"
    assert "SOLVER_PIN_FROZEN" in src, "el knob no se lee donde se decide el pin"


# ═════════════ 7 · el conteo llega al meal (no es telemetria muerta) ═════════════
#
# El repo tiene norma explicita contra el codigo inerte: "el sistema mide X y nadie consume X"
# es clase de bug documentada. `frozen_lines` sale del solver; estos tests anclan que el WIRING
# lo estampa en el meal (→ `plan_data` → consultable por SQL) y que el copy-back del swap no lo
# pierde, que es exactamente como se perdieron sus tres hermanos antes (P2-SOLVER-CONVERGENCE-
# METRIC y la nota de `_solver_failed_macros` en agent.py).

class _WiringDB:
    """DB para `_apply_macro_solver_to_meal`: pechuga/pollo/arroz resueltos, el resto bulk."""

    _M = {
        "pechuga": {"protein": 46.5, "carbs": 0.0, "fats": 5.4, "kcal": 247.5},
        "pollo": {"protein": 27.0, "carbs": 0.0, "fats": 5.0, "kcal": 155.0},
        "arroz": {"protein": 4.0, "carbs": 42.0, "fats": 0.5, "kcal": 195.0},
    }

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        for tok, mac in self._M.items():
            if tok in low:
                return dict(mac)
        return None

    def macros_for_line(self, qty, unit, name):
        return self.macros_from_ingredient_string(name)

    def grams_from_ingredient_string(self, s):
        return 100.0


def _meal(lines):
    return {"name": "Comida de prueba", "ingredients": list(lines),
            "ingredients_raw": list(lines), "protein": 30, "carbs": 20, "fats": 10, "cals": 400}


def test_el_wiring_estampa_el_conteo_de_congeladas_en_el_meal():
    """(a) meal CON linea congelada ⇒ `_solver_frozen_lines` con el conteo."""
    import graph_orchestrator as g
    meal = _meal([PECHUGA, POLLO])
    assert g._apply_macro_solver_to_meal(
        meal, {"kcal": 600.0, "protein": 80.0, "carbs": 10.0, "fats": 15.0}, _WiringDB()) is True
    assert meal.get("_solver_frozen_lines") == 1, (
        f"el conteo no llego al meal (seguiria siendo telemetria muerta): "
        f"{[k for k in meal if k.startswith('_solver')]}")


def test_el_wiring_no_estampa_la_clave_cuando_no_hay_congeladas():
    """(b) meal SIN congeladas ⇒ clave AUSENTE — mismo patron que los hermanos, que solo
    estampan cuando el valor es significativo (`if _sat:`, `if _infe:`)."""
    import graph_orchestrator as g
    meal = _meal(["100 g de pollo", "150 g de arroz"])
    g._apply_macro_solver_to_meal(
        meal, {"kcal": 600.0, "protein": 45.0, "carbs": 55.0, "fats": 12.0}, _WiringDB())
    assert "_solver_frozen_lines" not in meal, (
        "una comida sin congeladas no debe cargar la clave (ruido en plan_data)")


def test_el_copy_back_del_swap_no_pierde_el_conteo():
    """(c) La lista de `agent.py` es un copy-BACK (preserva telemetria a traves del `model_dump()`
    del structured output), NO un strip: las `_solver_*` no se ocultan a la LLM, viven en
    `plan_data`. Su modo de fallo documentado es que las claves nacidas DESPUES quedan fuera —
    le paso a `_solver_failed_macros`/`_solver_infeasible`/`_solver_residuals`. Esta no.
    """
    import os
    _agent = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent.py")
    with open(_agent, encoding="utf-8") as fh:
        src = fh.read()
    i = src.index('for _rk in ("ingredients", "ingredients_raw", "recipe"')
    bloque = src[i:src.index("):", i)]
    assert '"_solver_frozen_lines"' in bloque, (
        "clave nueva fuera del copy-back: toda comida SWAPEADA perderia el conteo")
    # y los hermanos siguen ahi (no la anadimos pisando a nadie)
    for hermano in ("_solver_not_converged", "_solver_infeasible", "_solver_residuals"):
        assert f'"{hermano}"' in bloque


def test_el_conteo_viaja_en_los_logs_que_lo_vuelven_accionable():
    """No emite log propio (hay congeladas en casi toda comida, ~28 solver-runs por plan: seria
    ruido). Se ADJUNTA a los dos logs que ya existen y que lo hacen accionable."""
    import os
    _go = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "graph_orchestrator.py")
    with open(_go, encoding="utf-8") as fh:
        src = fh.read()
    # Ventana = la FUNCION entera (hasta el siguiente `def` a nivel 0), no un numero de bytes
    # a ojo: el bloque de telemetria vive a ~7.3k del `def` y una ventana corta lo dejaria
    # fuera — la clase de anclaje que caduca sola (ya mordio en test_p3_audit_v6_solver_seeder).
    i = src.index("def _apply_macro_solver_to_meal")
    _m = re.search(r"\ndef ", src[i + 10:])
    win = src[i:i + 10 + _m.start()] if _m else src[i:]
    assert 'meal["_solver_frozen_lines"] = _frozen_n' in win
    assert 'if _frozen_n:' in win, "debe estamparse solo cuando es significativo (>0)"
    assert win.count("líneas congeladas=") == 2, (
        "el conteo debe viajar en los DOS logs accionables (no-convergencia + infactibilidad)")
    assert "logger.info(f\"📐 [P2-SOLVER-PIN-FROZEN]" not in win, (
        "no debe emitir log propio: hay congeladas en casi toda comida (~28 runs/plan)")
