"""[P1-PLAN-LOTE-980 · 2026-11-11] El marcador vuelve a la serie del plan.

El 2026-11-03 `_LAST_KNOWN_PFIX` salio de la serie `P1-PLAN-LOTE-N` hacia
nombres sueltos (P1-PHOTO-DRINK-IDENTITY, P1-COMPARATIVE-YESTERDAY-TEMPLATE)
y ~220 tests `test_marker` de lotes anteriores (que exigen LOTE >= el suyo)
cayeron a la vez. El marcador DEBE seguir la serie: los nombres propios de
cada cambio van en sus comentarios y en su test, no en el marcador.
"""
import re
from pathlib import Path


def test_marker_sigue_la_serie_del_plan():
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', src)
    assert m, "el marcador salio de la serie P1-PLAN-LOTE-N"
    assert int(m.group(1)) >= 980
