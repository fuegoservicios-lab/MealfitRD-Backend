"""[P3-BUDGET-DOCSTRING-PRICES · 2026-08-04] El docstring de `build_budget_context`
(`prompts/plan_generator.py`) afirmaba «la app no tiene base de precios por ingrediente» —
FALSO desde 2026-07-02: `master_ingredients.price_per_lb` (cheapen-pass) y
`supermarket_products` (P1-SUPERMARKET-DB) existen y alimentan costeo real (`price_lb =
m.get("price_per_lb", 0)` en este MISMO archivo).

El docstring es load-bearing en el punto EXACTO de la decisión de diseño: por qué el bloque de
presupuesto inyectado al prompt es una señal cualitativa. La razón real no es "no hay datos" sino
"los cálculos exactos viven en las palancas deterministas POST-generación" (cheapen-pass,
driver-aware, reconciliación costo-real) — pedirle al LLM que calcule precios sería redundante Y
menos preciso que esas palancas.

100% OFFLINE: solo lee el source como texto, cero import de dependencias pesadas.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_PLAN_GEN_SRC = (_BACKEND / "prompts" / "plan_generator.py").read_text(encoding="utf-8")
# Docstrings envuelven a ~70-80 cols: la frase puede partirse en un salto de línea
# ("...no tiene\nbase de precios..."). `\s+` tolera el wrap sin dejar de detectar la afirmación.
_STALE_CLAIM_RE = re.compile(r"no\s+tiene\s+base\s+de\s+precios", re.IGNORECASE)


def test_la_afirmacion_falsa_ya_no_existe_en_el_archivo():
    """«no tiene base de precios» (la frase exacta que el docstring afirmaba) debe estar
    ausente de TODO el archivo — no solo del docstring que la originó. `\\s+` tolera que el
    wrap del docstring parta la frase en un salto de línea (como en el original)."""
    m = _STALE_CLAIM_RE.search(_PLAN_GEN_SRC)
    assert m is None, (
        "el docstring de build_budget_context sigue afirmando que la app carece de datos de "
        f"precio por ingrediente — falso desde P1-SUPERMARKET-DB (2026-07-02): {m.group(0)!r}")


def test_el_docstring_referencia_las_palancas_deterministas():
    """Ancla el reemplazo: el docstring debe explicar POR QUÉ es cualitativo (diseño, no
    carencia de datos) y nombrar las 3 palancas deterministas post-gen que hacen el cálculo
    exacto — el mismo contrato que exige el brief de la tarea."""
    import sys
    sys.path.insert(0, str(_BACKEND))
    try:
        import prompts.plan_generator as pg
    finally:
        if str(_BACKEND) in sys.path:
            sys.path.remove(str(_BACKEND))
    doc = inspect.getdoc(pg.build_budget_context) or ""
    doc_norm = re.sub(r"\s+", " ", doc.lower())  # el wrap del docstring puede partir frases
    assert "P3-BUDGET-DOCSTRING-PRICES" in doc, "falta el marker del fix en el docstring"
    assert "por diseño" in doc_norm, (
        "el docstring debe explicar que la señal es cualitativa POR DISEÑO, no por falta de datos")
    for lever in ("cheapen", "driver-aware", "reconcil"):
        assert lever in doc_norm, (
            f"el docstring no referencia la palanca determinista '{lever}': {doc!r}")


def test_el_marker_del_supermercado_queda_anclado():
    """La razón por la que la afirmación era falsa (P1-SUPERMARKET-DB) debe quedar citada, para
    que un futuro lector entienda DE DÓNDE viene el dato de precio real."""
    assert "P1-SUPERMARKET-DB" in _PLAN_GEN_SRC
