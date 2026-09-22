"""[P1-PLAN-LOTE-167 · 2026-09-22] Francés: una sola forma de tratamiento, y la tipografía francesa también en el backend.

El catálogo tenía 752 valores con «vous» y 68 con «tu» (Configuración decía «Vos rappels…» y, dos líneas abajo, «…ton
heure habituelle»), y el backend tuteaba entero: avisos de comida y de agua, el título de los avisos del coach, su
mensaje de «no pude procesarlo» y la frase de la estrategia. Se pasa todo a «vous» —la mayoría, y el registro esperable
en una app de salud— y la directiva de idioma del coach lo pide explícitamente (antes no decía nada, y un modelo que
lee el prompt en «tu» tutea). Los espacios finos antes de «? ! ;» y el de antes de «:» también faltaban aquí.

Tooltip-anchor: P1-PLAN-LOTE-167
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_TU = re.compile(r"(?i)(?<![\w'’-])(tu|toi|ton|ta|tes|te|t['’])(?![\w-])")


def _frases_fr(rel: str) -> list[str]:
    """Los literales `"fr-FR": "…"` del fuente, evaluados como Python (los `\\u202f` pasan a ser el carácter)."""
    import ast
    src = (_BACKEND / rel).read_text(encoding="utf-8")
    return [ast.literal_eval('"' + m + '"') for m in re.findall(r'"fr-FR":\s*"((?:[^"\\]|\\.)*)"', src)]


def test_los_avisos_de_comida_y_agua_vouvoient():
    frases = _frases_fr("meal_reminders.py") + _frases_fr("hydration_reminders.py")
    textos = [f for f in frases if len(f) > 20]
    assert len(textos) >= 9, textos
    for f in textos:
        assert not _TU.search(f), f"sigue tuteando: {f}"
    assert sum("vous" in f.lower() or "votre" in f.lower() or "vos " in f.lower() for f in textos) >= 8


def test_tipografia_francesa_en_los_avisos():
    for f in _frases_fr("meal_reminders.py") + _frases_fr("hydration_reminders.py"):
        assert " ?" not in f and " !" not in f and " :" not in f and " ;" not in f, f"espacio normal ante signo doble: {f!r}"


def test_el_coach_en_frances_vouvoie():
    src = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    assert "Vouvoie TOUJOURS l'utilisateur" in src
    assert "Un mot de votre nutritionniste IA" in src and "Un mot de ton nutritionniste IA" not in src
    assert "Pouvez-vous la reformuler" in src and "Peux-tu la reformuler" not in src
    assert "Voici votre stratégie nutritionnelle" in src and "Voici ta stratégie" not in src


def test_el_catalogo_frances_ya_no_tutea():
    import json
    p = _BACKEND.parent / "frontend" / "src" / "i18n" / "locales" / "fr-FR.json"
    if not p.exists():
        import pytest
        pytest.skip("frontend ausente")
    cat = json.loads(p.read_text(encoding="utf-8"))
    tutean = [k for k, v in cat.items() if isinstance(v, str) and _TU.search(v)]
    assert tutean == [], tutean[:5]
    rectas = [k for k, v in cat.items() if isinstance(v, str) and '"' in v and "<" not in v]
    assert rectas == [], rectas


def test_el_marcador_esta_al_dia():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', src, re.M)
    assert m and int(m.group(1)) >= 167
