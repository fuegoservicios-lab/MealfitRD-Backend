"""[P1-CLOSER-BAKED-ASIDE · 2026-07-27] El closer mandaba mezclar yogur frío en una masa horneada.

## Lo que veía el owner (bollitos de maíz, plan 08114452)

    El Toque de Fuego: … Hornea por 15 minutos a 180°C … Incorpora yogurt a la
    preparación y mézclalo antes de servir.

Mezclar lácteo frío dentro de bollitos YA horneados es un disparate culinario. El lácteo blando
va AL LADO: "Sirve yogurt al lado para acompañar."

## El recorte que importa

Solo aplica a los lácteos blandos (hint soft-dairy) en platos horneados. El **atún** en unas papas
rellenas al horno SÍ se incorpora al relleno — ese wording no se toca.

tooltip-anchor: P1-CLOSER-BAKED-ASIDE
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g
from constants import strip_accents as _sa

T = g._closer_protein_step_text


# ───────────── 1. el caso del owner ─────────────

def test_horneado_mas_yogurt_va_al_lado():
    out = T("yogurt griego", False, baked=True)
    assert out == "Sirve yogurt griego al lado para acompañar."
    assert "mézclalo" not in out and "a la preparación" not in out


@pytest.mark.parametrize("lacteo", ["yogurt natural", "queso ricotta", "queso crema",
                                    "queso cottage"])
def test_cubre_los_lacteos_blandos(lacteo):
    assert "al lado" in T(lacteo, False, baked=True)


# ───────────── 2. lo que NO cambia ─────────────

def test_frio_sigue_incorporando():
    """En un plato frío (wrap, bowl) mezclar el lácteo es correcto y se conserva."""
    out = T("yogurt griego", False, baked=False)
    assert "Incorpora yogurt griego" in out and "mézclalo" in out


def test_atun_en_horneado_sigue_incorporandose():
    """El atún de unas papas rellenas al horno va AL RELLENO, no al lado."""
    out = T("atún en agua", False, precooked=True, baked=True)
    assert "Escurre e incorpora" in out or "Incorpora" in out
    assert "al lado" not in out


def test_carne_en_horneado_no_se_toca():
    out = T("pechuga de pollo", False, baked=True)
    assert "al lado para acompañar" not in out


# ───────────── 3. el detector ─────────────

def test_detector_de_horneado():
    horneado = {"name": "Bollitos de Maíz",
                "recipe": ["Mise en place: mezcla.", "El Toque de Fuego: Hornea 15 min a 180°C."]}
    frio = {"name": "Ceviche de Mero", "recipe": ["Montaje: sirve frío."]}
    assert g._meal_is_baked(horneado, _sa) is True
    assert g._meal_is_baked(frio, _sa) is False


def test_detector_fail_open():
    assert g._meal_is_baked({}, _sa) is False
    assert g._meal_is_baked({"recipe": None}, _sa) is False


def test_los_callsites_pasan_baked():
    """Ancla 'código presente, efecto ausente': la rama existe pero nadie pasa el flag.

    [reapuntado 2026-07-28 · P1-HOT-DAIRY-ASIDE] Los callsites ahora pasan
    `baked=(_meal_is_baked(...) or _meal_is_hot_cooked(...))` — el aside cubre también
    platos COCINADOS con calor (revoltillo), no solo horneados. La invariante sigue
    siendo la misma: ambos callsites deben alimentar el flag, y ahora además ambos
    deben consultar el detector de calor.
    """
    import pathlib
    src = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8")
    _wired = src.count("baked=_meal_is_baked(meal") + src.count("baked=(_meal_is_baked(meal")
    assert _wired >= 2, (
        "los dos callsites del closer deben pasar baked= o la rama jamás corre"
    )
    assert src.count("or _meal_is_hot_cooked(meal") >= 2, (
        "ambos callsites deben incluir el detector de calor (P1-HOT-DAIRY-ASIDE)"
    )
