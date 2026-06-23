"""[P1-PAN-PERECEDERO · 2026-05-16] Fix de categorización de pan integral
en la lista de compras.

Bug observado en lista de compras del plan aeb25e1c (2026-05-16):
  "Pan integral 1 paquete (1.3 lbs)" aparecía en sección "DESPENSA —
   ESTABLES +7 DÍAS" junto a aceite/arroz/sal. Pero pan integral fresco
   dura 5-7 días en cocina (~10d refrigerado). El usuario podría pensar
   "tengo 14+ días para usarlo" y se le mohecería.

Causa raíz:
  En `shopping_calculator._classify_perishability`, la rama de
  `_STAPLE_CATEGORIES` matchea `category="Despensa"` con prioridad #1
  y retorna "staple" SIN evaluar el nombre real. Pan integral tiene
  `category="Despensa"` + `shelf_life_days=14` (default genérico) en
  master_ingredients. La categoría "Despensa" gana antes de chequear el
  nombre o el shelf real.

Fix:
  Nueva constante `_DESPENSA_PERISHABLE_EXCEPTIONS` con los nombres
  canónicos de panes frescos (pan integral, pan de agua, pan blanco,
  pan dulce). En la rama `_STAPLE_CATEGORIES`, ANTES del `return "staple"`,
  chequear si `name_norm` matchea alguna excepción → si sí, retornar
  "perishable" en su lugar. Quirúrgico: no toca DB, no afecta items que
  YA estaban bien clasificados.

  Adicionalmente, `_STAPLE_NAME_HINTS` removed 'pan integral' (defense-
  in-depth — items legacy sin category quedarían staples por substring).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_PY = _BACKEND_ROOT / "shopping_calculator.py"


# ---------------------------------------------------------------------------
# 1. Parser-based: constants + branch
# ---------------------------------------------------------------------------
def test_despensa_perishable_exceptions_defined():
    """`_DESPENSA_PERISHABLE_EXCEPTIONS` debe estar definido como frozenset
    con los nombres canónicos de panes frescos."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    assert "_DESPENSA_PERISHABLE_EXCEPTIONS = frozenset({" in src, (
        "Constante `_DESPENSA_PERISHABLE_EXCEPTIONS` no encontrada — fix no "
        "aplicado."
    )


def test_pan_variants_in_exceptions():
    """Las 4 variantes de pan fresco DEBEN estar en la excepción:
    pan integral (caso original), pan de agua, pan blanco, pan dulce.
    Sin esto, otras variantes de pan fresco caen en el mismo bug."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    expected_breads = ["pan integral", "pan de agua", "pan blanco", "pan dulce"]
    for bread in expected_breads:
        assert f"'{bread}'" in src, (
            f"Pan variant {bread!r} ausente de _DESPENSA_PERISHABLE_EXCEPTIONS. "
            f"Sin ella, ese pan terminaría en sección 'estables +7 días'."
        )


def test_classify_branch_uses_exceptions():
    """`_classify_perishability` debe chequear `_DESPENSA_PERISHABLE_EXCEPTIONS`
    DENTRO de la rama `if cat in _STAPLE_CATEGORIES:` ANTES del
    `return "staple"`. Esa es la única posición correcta — antes de la
    rama, el cat aún no se evaluó; después, el return ya disparó."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    # Buscar el bloque de la rama staple
    branch = re.search(
        r"if cat in _STAPLE_CATEGORIES:.*?(?=if cat in _PERISHABLE_CATEGORIES:)",
        src,
        re.DOTALL,
    )
    assert branch, "Rama `_STAPLE_CATEGORIES` no encontrada en `_classify_perishability`."
    body = branch.group(0)
    assert "_DESPENSA_PERISHABLE_EXCEPTIONS" in body, (
        "La rama staple no consulta `_DESPENSA_PERISHABLE_EXCEPTIONS`. "
        "El fix está mal posicionado — el return staple disparará antes."
    )
    # Y debe retornar perishable en la excepción
    assert 'return "perishable"' in body, (
        "La rama staple no tiene `return \"perishable\"` para la excepción. "
        "El fix podría estar incompleto."
    )


def test_pan_integral_removed_from_staple_hints():
    """`'pan integral'` debe estar REMOVIDO de `_STAPLE_NAME_HINTS`. Sin esto,
    items legacy (sin category válida) podrían caer en staple por el name
    hint, contradiciendo el fix de `_DESPENSA_PERISHABLE_EXCEPTIONS`.

    Stripeamos comentarios (`# ...`) antes de chequear — el comentario que
    documenta la remoción puede mencionar 'pan integral' explicativamente
    sin contar como ocurrencia real."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    # Aislar el _STAPLE_NAME_HINTS. Non-greedy `\(.*?\)` se corta en el
    # primer `)` que aparezca dentro de un comentario (e.g. `'Atún en lata')`).
    # Usamos `^\)` (paren de cierre al inicio de línea) — convención del
    # codebase para terminar tuples multi-linea.
    hints_block = re.search(
        r"_STAPLE_NAME_HINTS = \(.*?^\)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert hints_block, "No se pudo aislar el bloque completo de _STAPLE_NAME_HINTS"
    body = hints_block.group(0)
    # Strip comments to focus on actual code lines
    body_no_comments = re.sub(r"#[^\n]*", "", body)
    assert "'pan integral'" not in body_no_comments, (
        "'pan integral' sigue en _STAPLE_NAME_HINTS como tuple item activo. "
        "Defense-in-depth roto: items legacy sin category válida lo "
        "clasificarían como staple por substring, contradicting "
        "`_DESPENSA_PERISHABLE_EXCEPTIONS`."
    )
    # Casabe y galletas SÍ deben seguir (son staples reales) — pueden estar
    # en código activo (no en comentarios).
    assert "'casabe'" in body_no_comments, (
        "Casabe se removió accidentalmente — sí es staple (cracker deshidratado, dura meses)."
    )
    assert "'galletas'" in body_no_comments, (
        "Galletas se removió accidentalmente — sí son staples (selladas, secas)."
    )


# ---------------------------------------------------------------------------
# 2. Funcional: pan integral perecedero + sin regresión
# ---------------------------------------------------------------------------
def _load_classifier():
    os.environ.setdefault("GEMINI_API_KEY", "dummy")
    os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "dummy")
    os.environ.setdefault("CRON_SECRET", "dummy")
    sys.path.insert(0, str(_BACKEND_ROOT))
    from shopping_calculator import _classify_perishability
    return _classify_perishability


def test_pan_integral_classified_perishable():
    """Caso del bug original: pan integral con category=Despensa +
    shelf_life=14 (default master_ingredients) DEBE ser perishable."""
    classify = _load_classifier()
    result = classify(
        "Pan integral",
        {"category": "Despensa", "shelf_life_days": 14},
    )
    assert result == "perishable", (
        f"BUG REGRESIÓN: pan integral clasificado como {result!r} "
        f"(esperado 'perishable'). El usuario lo verá en sección 'estables "
        f"+7 días' y se le mohecía."
    )


def test_pan_de_agua_classified_perishable():
    """Pan de agua es el más perecedero de todos (1-2d). DEBE ser perishable."""
    classify = _load_classifier()
    result = classify(
        "Pan de agua",
        {"category": "Despensa", "shelf_life_days": 14},
    )
    assert result == "perishable", (
        f"Pan de agua clasificado como {result!r}. Es el pan más rápido en "
        f"endurecerse (1-2 días); claro perishable."
    )


def test_casabe_still_staple_no_regression():
    """NO regresión: casabe es cracker totalmente deshidratado, dura meses.
    DEBE seguir siendo staple."""
    classify = _load_classifier()
    result = classify(
        "Casabe",
        {"category": "Despensa", "shelf_life_days": 14},
    )
    assert result == "staple", (
        f"REGRESIÓN: casabe clasificado como {result!r} (esperado 'staple'). "
        f"El fix de pan no debe contaminar casabe (yuca deshidratada — dura meses)."
    )


def test_galletas_still_staple_no_regression():
    """NO regresión: galletas selladas son staples reales."""
    classify = _load_classifier()
    for name in ("Galletas de soda",):  # [P3-GALLETA-ARROZ-REMOVE] galletas de arroz fuera del catálogo
        result = classify(
            name,
            {"category": "Despensa", "shelf_life_days": 30},
        )
        assert result == "staple", (
            f"REGRESIÓN: {name!r} clasificado como {result!r} (esperado 'staple')."
        )


def test_aceite_still_staple_no_regression():
    """NO regresión: aceite de oliva category=Despensa shelf=365d → staple."""
    classify = _load_classifier()
    result = classify(
        "Aceite de oliva",
        {"category": "Despensa", "shelf_life_days": 365},
    )
    assert result == "staple"


def test_arroz_still_staple_no_regression():
    """NO regresión: arroz, granos, etc. siguen siendo staples."""
    classify = _load_classifier()
    for name in ("Arroz blanco", "Lentejas", "Avena"):
        result = classify(
            name,
            {"category": "Despensa", "shelf_life_days": 365},
        )
        assert result == "staple", (
            f"REGRESIÓN: {name!r} clasificado como {result!r}."
        )


def test_pan_tostado_still_staple():
    """`Pan tostado` (palitroques, biscotti, croutons) NO contiene
    'pan integral'/'pan de agua'/'pan blanco'/'pan dulce' como substring.
    DEBE seguir siendo staple (es seco, sellado, dura semanas)."""
    classify = _load_classifier()
    result = classify(
        "Pan tostado",
        {"category": "Despensa", "shelf_life_days": 30},
    )
    assert result == "staple", (
        f"Pan tostado clasificado como {result!r}. La excepción no debe "
        f"contaminar variantes secas — solo panes frescos blandos."
    )


def test_yogurt_y_queso_fresco_still_perishable():
    """NO regresión: lácteos perecederos (categoría no-staple) siguen perishable."""
    classify = _load_classifier()
    for name in ("Yogurt griego", "Queso blanco fresco"):
        result = classify(
            name,
            {"category": "Lácteos", "shelf_life_days": 14},
        )
        assert result == "perishable", (
            f"REGRESIÓN: {name!r} clasificado como {result!r}."
        )
