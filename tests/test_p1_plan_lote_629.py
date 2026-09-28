# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-629 · 2026-09-28] El toast de coherencia no viaja en el JS de arranque.

El presupuesto del arranque es 148 kB gz y estaba en 154,2 (bisección de mealfitrd-ia-9b: goteo, no un culpable). Tras
el 681 (liveUpdate y keyboardProbe diferidos) y el 702 (tabla de nombres por país fuera) quedaban ~2 kB.
`renderCoherenceWarnings.js` (~210 líneas de código) sólo llegaba al arranque por AssessmentContext, que lo usa en dos
momentos tras una respuesta del servidor (recalcular y persistir un swap); el resto de consumidores son páginas perezosas.
Ahora AssessmentContext lo pide con `import()` en esos dos momentos. El contrato de P2-AUDIT-NEW-1 se mantiene: el
contexto sigue invocando `emitCoherenceToast(toast, rd._coherence_warnings)`.

tooltip-anchor: P1-PLAN-LOTE-629
"""
import re
from pathlib import Path

import pytest

_CTX = Path(__file__).resolve().parents[2] / "frontend" / "src" / "context" / "AssessmentContext.jsx"


@pytest.fixture(scope="module")
def src():
    if not _CTX.exists():
        pytest.skip("frontend ausente (repo hermano)")
    return _CTX.read_text(encoding="utf-8")


def test_sin_import_estatico(src):
    assert not re.search(r"^import\s+\{[^}]*\}\s+from\s+['\"][^'\"]*renderCoherenceWarnings['\"]", src, re.M)


def test_los_dos_momentos_lo_piden_perezoso(src):
    llamadas = re.findall(
        r"import\('\.\./utils/renderCoherenceWarnings'\)\s*\.then\(\(\{ emitCoherenceToast \}\) => "
        r"emitCoherenceToast\(toast, rd\._coherence_warnings\)\)\.catch\(\(\) => \{\}\)", src)
    assert len(llamadas) == 2
