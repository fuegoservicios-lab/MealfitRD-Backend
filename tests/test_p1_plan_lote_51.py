"""[P1-PLAN-LOTE-51 · 2026-09-15] = P1-DIARY-CLAIM-PERFECTIVE (el «hola» del dueño que recibió
«Aclarado: no registré…»). El marcador de deploy sigue la numeración de lotes; el test del
comportamiento vive en `test_p1_diary_claim_perfective.py`."""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_marcador_del_lote():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 51 and m.group(2) >= "2026-09-15"
    assert "[P1-PLAN-LOTE-51 · 2026-09-15] = [P1-DIARY-CLAIM-PERFECTIVE]" in app
    assert (_BACKEND / "tests" / "test_p1_diary_claim_perfective.py").exists()
