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

100% OFFLINE: `StubDB` sustituye a `IngredientNutritionDB` (el .env apunta a PRODUCCION).
"""
import inspect

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
    factors, method, _sh, _sl = ps._compute_scale_factors(entries, TGT_BRIEF, 0.3, 3.5, 5.0)
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


def test_el_greedy_excluye_la_contribucion_congelada_del_denominador(monkeypatch):
    """Espejo del defecto en el fallback: `current` contaba la congelada y diluia el factor."""
    monkeypatch.setattr(ps, "SOLVER_LSQ", False)
    res = _solve([PECHUGA, POLLO])
    assert res["method"] == "greedy"
    # antes: 75.9 g (factor 1.0884 repartido tambien a la congelada). Ahora: 80.0 g.
    assert abs(res["achieved"]["protein"] - 80.0) < 1.0, (
        f"el greedy sigue repartiendo mal el residual: {res['achieved']}")


# ═════════════ 2 · el knob apaga el cambio por completo ═════════════

def test_el_knob_off_restaura_el_comportamiento_previo_byte_a_byte(monkeypatch):
    monkeypatch.setattr(ps, "SOLVER_PIN_FROZEN", False)
    res = _solve([PECHUGA, POLLO])
    assert res["factors_applied"][1] == pytest.approx(0.503, abs=1e-3), (
        f"el rollback no reprodujo la linea base medida: {res['factors_applied']}")
    assert res["achieved"]["protein"] == pytest.approx(60.1, abs=0.2)


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
    _f, _m, sat_hi, sat_lo = ps._compute_scale_factors(entries, TGT_BRIEF, 0.3, 3.5, 5.0)
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
    el solver la escalaba hasta 3.5x: DOS modelos contradictorios del mismo plato.

    Plato: «Cdta de mantequilla de mani» (4.0 g de grasa, CONGELADA) + «150 g de arroz» (0.5 g,
    movible) contra un target de 8 g de grasa. La cota frozen-aware del reporte es
    4.0x1.0 + 0.5x3.5 = 5.75 g ⇒ veredicto 'high' (inalcanzable, falta un PORTADOR).

    Antes: el reporte decia "5.75 es el maximo" y el solver entregaba 4.6 g — se quedaba CORTO de su
    propio techo porque creia que el mani cubriria el hueco. Ahora entrega 5.8 g: el solver exprime
    lo unico que puede mover y el numero coincide con la cota que el reporte publica.
    """
    lines, tgt = [MANI, ARROZ], {"protein": 0.0, "carbs": 0.0, "fats": 8.0}
    res = _solve(lines, tgt)
    assert (res["infeasible"] or {}).get("fats") == "high", (
        f"el fixture dejo de ejercitar el caso infactible: {res['infeasible']}")
    assert res["achieved"]["fats"] == pytest.approx(5.75, abs=0.1), (
        f"el solver no llega a la cota que el propio reporte publica: {res['achieved']}")
    assert res["factors_applied"][1] == pytest.approx(3.5), "la unica movible debe ir al techo"

    monkeypatch.setattr(ps, "SOLVER_PIN_FROZEN", False)
    viejo = _solve(lines, tgt)
    assert viejo["achieved"]["fats"] == pytest.approx(4.6, abs=0.1), (
        "linea base de la contradiccion: reporte 'high' con cota 5.75 vs 4.6 g entregados")


def test_el_marker_vive_en_el_codigo_que_lo_implementa():
    src = inspect.getsource(ps._compute_scale_factors)
    assert "P2-SOLVER-PIN-FROZEN" in src, "marker ausente en `_compute_scale_factors`"
    assert "SOLVER_PIN_FROZEN" in src, "el knob no se lee donde se decide el pin"
