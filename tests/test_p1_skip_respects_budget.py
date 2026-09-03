"""[P1-SKIP-RESPECTS-BUDGET · 2026-08-09] «Saltar a la última pregunta» dejaba
atrás el paso de presupuesto a medias.

REPORTE DEL OWNER (con captura): eligió «Personalizar», escribió RD$13 contra un
piso de RD$13.000 —el paso mostraba su propio aviso y su «Siguiente»
deshabilitado— y aun así el salto lo mandó al paso 14.

CAUSA: el salto solo consultaba `findFirstIncompleteField`, que comprueba
PRESENCIA de campos. Con «Personalizar» elegido, `budget` vale `'custom'`: está
presente, así que el paso 11 pasaba como completo aunque el monto fuera
inválido.

El paso SÍ sabía que estaba incompleto — su `validateExtra` lo dice, y por eso
su botón estaba deshabilitado. Lo que fallaba es que esa regla solo corría AL
PASAR POR EL PASO, y saltar es precisamente no pasar.

  *Una validación que vive en el paso no protege a quien no pasa por el paso.*

El submit ya tenía SU PROPIA copia de la regla (P1-FORM-AUDIT-BATCH, que cerró
la misma clase en la otra puerta). Con la del salto habrían sido TRES
implementaciones — por eso este P-fix las unifica en `isCustomBudgetValid`.

Tooltip-anchor: P1-SKIP-RESPECTS-BUDGET
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FLOW = (_REPO_ROOT / "frontend" / "src" / "components" / "assessment"
         / "InteractiveAssessmentFlow.jsx")


def _src() -> str:
    return _FLOW.read_text(encoding="utf-8")


def test_the_rule_lives_in_one_place():
    """Tres copias de la misma regla es como empiezan los drifts: se arregla una
    puerta y las otras dos siguen dejando pasar."""
    src = _src()
    m = re.search(r"const isCustomBudgetValid = ", src)
    assert m, (
        "P1-SKIP-RESPECTS-BUDGET: falta `isCustomBudgetValid`. La regla del piso "
        "del presupuesto personalizado debe tener UN solo sitio."
    )
    # Nadie debe re-implementarla comparando a mano contra el piso.
    inline = re.findall(r"Number\(\s*(?:fd|formData)\.budgetAmount\s*\)\s*>=", src)
    assert len(inline) <= 1, (
        f"P1-SKIP-RESPECTS-BUDGET: la comparación contra el piso aparece {len(inline)} "
        "veces. Debe estar solo dentro de `isCustomBudgetValid` — las copias a mano "
        "son las que dejaron el salto sin protección."
    )


def test_all_three_gates_use_it():
    """Las tres puertas que pueden dejar atrás el paso 11: el botón del propio
    paso (`validateExtra`), el salto y el submit."""
    src = _src()
    usos = len(re.findall(r"isCustomBudgetValid", src))
    assert usos >= 4, (  # 1 definición + 3 consumidores
        f"P1-SKIP-RESPECTS-BUDGET: `isCustomBudgetValid` aparece {usos} vez/veces; se "
        "esperan al menos 4 (definición + validateExtra + salto + submit). Si una "
        "puerta deja de usarla, vuelve a poder saltarse el paso de presupuesto."
    )
    assert re.search(r"validateExtra:\s*isCustomBudgetValid", src), (
        "P1-SKIP-RESPECTS-BUDGET: el `validateExtra` del paso de presupuesto dejó de "
        "usar el SSOT."
    )


def test_the_skip_handler_checks_the_budget():
    """La puerta del reporte. `findFirstIncompleteField` mira PRESENCIA y `budget`
    está presente con 'custom' — sin este chequeo extra el salto se va."""
    src = _src()
    m = re.search(r"const handleSkipToLastStep = \(\) => \{(.*?)\n    \};", src, re.DOTALL)
    assert m, "P1-SKIP-RESPECTS-BUDGET: no encuentro `handleSkipToLastStep`"
    body = m.group(1)
    assert "isCustomBudgetValid" in body, (
        "P1-SKIP-RESPECTS-BUDGET: el handler del salto no valida el presupuesto "
        "personalizado. `findFirstIncompleteField` no basta: comprueba presencia, y "
        "'custom' está presente aunque el monto sea inválido."
    )
    i_missing = body.index("findFirstIncompleteField")
    i_budget = body.index("isCustomBudgetValid")
    i_last = body.index("steps.length - 1")
    assert i_missing < i_budget < i_last, (
        "P1-SKIP-RESPECTS-BUDGET: el chequeo del presupuesto debe ir DESPUÉS de los "
        "campos faltantes y ANTES del salto al último paso; si va después del salto, "
        "no protege nada."
    )
