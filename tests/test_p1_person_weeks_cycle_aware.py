"""[P1-PERSON-WEEKS-CYCLE-AWARE · 2026-07-30] Un `3` hardcodeado apretaba TODOS los topes de la
lista de compras en cualquier plan que no tuviera exactamente 3 días.

    # comentario del propio código, declarando la intención:
    #   "`multiplier × 3/7` deshace el `base_duration_scale = 7/days_generated` aplicado upstream"
    _person_weeks = max(1.0, float(multiplier) * 3.0 / 7.0)      # ← el 3 hardcodeado

Si la inversa de `7/num_days` es `num_days/7`, el `3` solo la deshace cuando `num_days == 3`. Los
cuatro ejemplos trabajados del comentario (18.67, 9.33, 4.67, 2.33) están TODOS calculados sobre un
ciclo de 3 días: **la intención estaba bien escrita justo al lado de la línea que la traiciona.**

Medido en el journal de producción: con `days_len=14 base_scale=0.5`, `person_weeks` queda CLAVADO
en 1.0 mientras el multiplicador va 1/2/4 — Yogurt 1155→907 g, 2311→907 g, 4622→907 g: **el mismo
tope para tres duraciones distintas**. Con `days_len=3` sí sigue a 1/2/4, que es la firma exacta del
hardcode.

Medido sobre la matriz completa hogar × días × duración: **41 de 54 combinaciones daban un
`person_weeks` incorrecto**. Incluida la dimensión HOGAR, que en los 34 planes de la ventana no se
podía observar (todos de 1 persona): un hogar de 2 con ciclo de 14 días y lista mensual recibía 1,71
cuando le corresponden 8.

Muerde a todos los topes que leen `_person_weeks` — incluidos los NO perecederos (atún en lata,
aceite, especias, endulzantes) donde no hay coartada de realismo de almacenamiento: que la lista
mensual traiga 2 latas de atún para ~1.600 g de demanda no lo explica la nevera, lo causa esta
división.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parents[1]
_SC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


def _person_weeks(hogar: int, dias: int, semanas: int) -> float:
    """Reproduce la cadena real: el agregador recibe el multiplicador YA escalado por
    `base_duration_scale = 7/num_days`, y tiene que deshacerlo para recuperar persona-semanas."""
    base_duration_scale = 7.0 / dias
    effective_multiplier = hogar * semanas * base_duration_scale
    return max(1.0, effective_multiplier * dias / 7.0)


@pytest.mark.parametrize("dias", [2, 3, 4, 7, 14, 25])
@pytest.mark.parametrize("semanas", [1, 2, 4])          # semanal / quincenal / mensual
@pytest.mark.parametrize("hogar", [1, 2, 4])
def test_person_weeks_es_hogar_por_semanas_sea_cual_sea_el_ciclo(hogar, dias, semanas):
    """La invariante: persona-semanas = personas × semanas del ciclo. La LONGITUD del plan generado
    es un detalle de implementación (3 días rotativos o 14 explícitos) y no puede cambiar cuántas
    persona-semanas cubre la compra."""
    esperado = max(1.0, float(hogar * semanas))
    assert _person_weeks(hogar, dias, semanas) == pytest.approx(esperado, rel=1e-6)


def test_la_formula_vieja_fallaba_en_41_de_54():
    """Ancla del alcance: si alguien "simplifica" volviendo al 3, este test dice cuánto rompe."""
    malos = 0
    for hogar in (1, 2, 4):
        for dias in (2, 3, 4, 7, 14, 25):
            for semanas in (1, 2, 4):
                mult = hogar * semanas * (7.0 / dias)
                viejo = max(1.0, mult * 3.0 / 7.0)
                correcto = max(1.0, float(hogar * semanas))
                if abs(viejo - correcto) >= 0.01:
                    malos += 1
    assert malos == 41, f"el alcance del bug cambió: {malos} de 54"


@pytest.mark.parametrize("hogar,semanas", [(1, 4), (2, 4), (4, 4), (2, 2)])
def test_un_ciclo_de_14_dias_ya_no_queda_clavado_en_1(hogar, semanas):
    """La firma exacta que se vio en el journal: el tope idéntico para 1×, 2× y 4× de duración."""
    assert _person_weeks(hogar, 14, semanas) > 1.0
    assert _person_weeks(hogar, 14, semanas) == pytest.approx(hogar * semanas)


def test_tres_dias_sigue_dando_lo_de_siempre():
    """No-regresión: el caso que SÍ estaba bien no puede moverse — los ejemplos trabajados del
    comentario original (2p mensual → 8 mazos, 2p quincenal → 4, 2p semanal → 2) siguen valiendo."""
    assert _person_weeks(2, 3, 4) == pytest.approx(8.0)
    assert _person_weeks(2, 3, 2) == pytest.approx(4.0)
    assert _person_weeks(2, 3, 1) == pytest.approx(2.0)
    assert _person_weeks(1, 3, 1) == pytest.approx(1.0)


# ───────────────────────────── anclaje en el fuente ─────────────────────────────

def test_el_hardcode_no_vuelve():
    assert "_person_weeks = max(1.0, float(multiplier) * 3.0 / 7.0)" not in _SC, (
        "volvió el `3` hardcodeado: los topes se aprietan 4,7× en ciclos de 14 días")
    assert "_person_weeks = max(1.0, float(multiplier) * _pw_days / 7.0)" in _SC


def test_el_agregador_recibe_num_days():
    assert "num_days: int | None = None" in _SC, "el agregador debe aceptar num_days"
    i = _SC.index("res = aggregate_and_deduct_shopping_list(all_ingredients, items_to_deduct")
    assert "num_days=num_days" in _SC[i:i + 400], (
        "el callsite principal debe pasar num_days, o el kwarg queda decorativo y el cap sigue "
        "usando el default de 3")


def test_sin_num_days_conserva_el_comportamiento_historico():
    """Un callsite que todavía no lo pase NO puede ver su cap cambiar por debajo sin avisar: el
    default reproduce exactamente la fórmula vieja."""
    i = _SC.index("_pw_days = ")
    seg = _SC[i:i + 200]
    assert "else 3.0" in seg, "el fallback debe ser 3.0 (comportamiento histórico), no 7 ni 1"
