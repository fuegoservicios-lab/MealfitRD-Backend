# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-113 · 2026-09-19] El OTA se deja ver: revisa al volver a la app y avisa.

El dueño dijo tres veces «sigue igual» (teclado, metas, `/sonda`) y no era el código nuevo: era que NO LO TENÍA. La
revisión de paquetes corría al arrancar en frío y, al volver a la app, solo si había pasado UNA HORA desde la última.
Quien volvía a los 50 minutos de un despliegue no revisaba; cerraba del todo y abría —ahí sí descargaba, pero ese
arranque ya corría el paquete viejo—, probaba y veía lo de antes. La prueba forense: su `/sonda` llegó a
`agent_messages` como mensaje normal (el paquete que corría no traía el interceptor).
Ahora la calma entre revisiones es de un minuto, se avisa UNA vez cuando hay paquete preparado («cierra la app del
todo y vuelve a abrirla») y se confirma la primera vez que el paquete nuevo corre. Sin recarga en caliente (lote 108).
Contrato fino: `frontend/src/__tests__/lote113.test.js`."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_revisa_al_volver_con_un_minuto_de_calma():
    lu = _front("src/native/liveUpdate.js")
    assert "const REVISAR_CADA_MS = 60 * 1000;" in lu
    assert "60 * 60 * 1000" not in lu, "con una hora, volver a la app tras un despliegue no revisaba nada"
    assert "document.visibilityState === 'visible' && Date.now() - _ultimaRevision > REVISAR_CADA_MS" in lu


def test_avisa_cuando_hay_paquete_y_cuando_ya_corre_sin_recargar():
    lu = _front("src/native/liveUpdate.js")
    i = lu.index("await LiveUpdate.setNextBundle({ bundleId });")
    assert "avisarPreparado(bundleId);" in lu[i:i + 120]
    assert "if (safeLocalStorageGet(CLAVE_AVISADO, null) === bundleId) return;" in lu, "una vez por paquete"
    assert "avisarSiSeActualizo(LiveUpdate);" in lu
    codigo = "\n".join(l for l in lu.splitlines() if not l.lstrip().startswith(("//", "*", "/*")))
    assert ".reload(" not in codigo, "el aviso no trae botón de recarga: arranque en frío (lote 108, decisión 1)"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 113
