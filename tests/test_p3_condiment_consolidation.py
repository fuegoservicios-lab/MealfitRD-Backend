"""[P3-CONDIMENT-CONSOLIDATION · 2026-06-22] El day-gen debe converger a UN solo vinagre/aceite por plan.

Bug observado en vivo (plan 66f4a214, 2026-06-22): un plan de 7 días usó 2 vinagres distintos (de manzana
día 1 + balsámico día 3) para ~1 cucharada en total → la lista obligaba a comprar 2 botellas de 473ml.
Como el day-gen es PARALELO (un worker por día, independientes), la consolidación cross-día se logra
haciendo que cada worker converja al MISMO vinagre base (vinagre blanco) por defecto.

Test parser-based del prompt (no requiere LLM).
"""
from __future__ import annotations

from pathlib import Path

_PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "day_generator.py").read_text(encoding="utf-8")


def test_condiment_consolidation_directive_present():
    assert "P3-CONDIMENT-CONSOLIDATION" in _PROMPT, "falta el marker de la directiva de condimentos"
    # La directiva debe nombrar el vinagre base y desincentivar variantes para cantidades mínimas.
    low = _PROMPT.lower()
    assert "vinagre blanco" in low, "la directiva debe converger a 'vinagre blanco' como base"
    assert ("balsámico" in low) or ("balsamico" in low) or ("de manzana" in low), (
        "la directiva debe mencionar las variantes a evitar (balsámico/de manzana) salvo necesidad real"
    )
