"""[P1-PLAN-LOTE-53 · 2026-09-15] Batería de escritura del coach y de los escáneres (encargo nocturno).

El marcador de deploy sigue la numeración de lotes. Los tests del comportamiento viven en
`test_p1_coach_battery_fixes.py` (coach) y `test_p1_scanner_audit.py` (escáneres); el método, la
rúbrica y el antes/después en `docs/coach_bateria_2026_09_15.md`.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_marcador_del_lote():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 53 and m.group(2) >= "2026-09-15"
    assert "[P1-PLAN-LOTE-53 · 2026-09-15]" in app
    for t in ("test_p1_coach_battery_fixes.py", "test_p1_scanner_audit.py"):
        assert (_BACKEND / "tests" / t).exists(), t


def test_la_bateria_y_su_rubrica_viven_en_el_repo():
    assert (_BACKEND / "scripts" / "coach_battery" / "battery.json").exists()
    assert (_BACKEND / "scripts" / "coach_battery" / "run_battery.py").exists()
    doc = (_BACKEND / "docs" / "coach_bateria_2026_09_15.md").read_text(encoding="utf-8")
    assert "Rúbrica fija" in doc and "FD1" in doc
