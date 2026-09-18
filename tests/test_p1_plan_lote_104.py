# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-104 · 2026-09-18] Bajo «Tus macros y micros de hoy» el subtítulo dice «N comidas registradas», sin repetir «hoy»
(el dueño, con captura: «la palabra hoy no quiero que se repita dos veces aquí»)."""
from __future__ import annotations

import json
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


def test_el_subtitulo_no_repite_hoy():
    tp = _front("src/components/dashboard/TrackingProgress.jsx")
    assert "'{n} comidas registradas'," in tp and "'{n} comida registrada'," in tp
    assert "'{n} comidas registradas hoy'" not in tp
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        d = json.loads(_front(f"src/i18n/locales/{loc}.json"))
        assert "{n} comidas registradas hoy" not in d and d.get("{n} comidas registradas", {}).get("other")


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 104
