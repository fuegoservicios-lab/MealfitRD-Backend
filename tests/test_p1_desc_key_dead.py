"""[P1-DESC-KEY-DEAD · 2026-07-24] El autofix de `desc` leía una clave que no existe.

Auditoría del plan vivo `732588f8`: **7 de 12 `desc` nombran comida que el plato no tiene**
(piña→aguacate, almendras→maní, mero→pollo, salmón→pescado blanco, nueces→maní). El sistema SABE
que sustituyó — graba `_budget_substitutions` / `_protein_autofix_applied` / `_slot_autofix_applied`
— y reescribe el nombre y los pasos. El bloque que debía reescribir la descripción existía desde
P2-AUTOFIX-DESC (2026-07-06) y estaba **muerto**:

    _desc_af = meal.get("description")     ← la clave del meal es `desc`

Verificado contra la fila persistida: `sorted(meal.keys())` = [..., 'desc', 'difficulty', ...],
sin 'description' en ningún meal. Y su test ancla (`test_p2_blanch_ingredient_truth.py`) afirmaba
la MISMA clave equivocada, así que el verde no probaba nada — el modo de fallo más caro de un test
parser-based: anclar el bug en vez del contrato.
"""
from pathlib import Path

import graph_orchestrator as go

SRC = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _autofix_desc_block() -> str:
    i = SRC.index("[P2-AUTOFIX-DESC")
    return SRC[i:i + 1800]


def test_lee_la_clave_que_el_motor_escribe():
    blk = _autofix_desc_block()
    assert 'for _desc_key in ("desc", "description")' in blk, (
        "la clave real del meal es `desc`; leer solo `description` deja el bloque muerto"
    )


def test_escribe_en_la_misma_clave_que_leyo():
    blk = _autofix_desc_block()
    assert "meal[_desc_key] =" in blk, (
        "escribir en una clave fija reintroduce el bug por el otro lado (leer `desc`, "
        "escribir `description` → la desc del usuario sigue mintiendo)"
    )


def test_meals_reales_usan_desc_no_description():
    """Ancla del hecho que hizo muerto al bloque. Si algún día el motor renombra la clave,
    este test cae antes que la descripción vuelva a mentir en silencio."""
    import re
    # El motor construye/normaliza meals con la clave `desc` en múltiples surfaces.
    assert len(re.findall(r'"desc"', SRC)) > 5
    # `description` no puede volver a aparecer como ÚNICA lectura del bloque de autofix.
    assert 'meal.get("description")' not in _autofix_desc_block()


def test_el_test_ancla_hermano_ya_no_afirma_la_clave_equivocada():
    """El test que 'protegía' este bloque afirmaba `meal.get("description")` — el mismo error
    que el código. Un test parser-based que copia el bug del código no es una defensa."""
    sib = (Path(__file__).resolve().parent / "test_p2_blanch_ingredient_truth.py").read_text(encoding="utf-8")
    # Se afirma sobre la LÍNEA del assert, no sobre el archivo: el comentario que explica la
    # corrección menciona la clave vieja y un `not in sib` lo leería como si fuera código
    # (misma trampa que ya me comió una vez en PlanHydrateOnComplete.test.js).
    linea = next(l for l in sib.splitlines() if "P2-AUTOFIX-DESC" in l and l.strip().startswith("assert"))
    assert "description" not in linea, linea
    assert '"desc"' in linea, linea
