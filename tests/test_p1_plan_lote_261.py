# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-261 · 2026-09-25] G20 para el revisor: potasio/fósforo en renal no es un crítico agudo.

Batería rd260 (renal + gota): «El plan acumula fuentes importantes de potasio en el contexto de enfermedad renal…» era
CRÍTICO agudo — un reintento y, si el último intento repetía, el plan de emergencia.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import critico_no_agudo as cna  # noqa: E402

KP = ("El plan acumula fuentes importantes de potasio en el contexto de enfermedad renal: yautía (incluida una porción "
      "preparada en airfryer), yuca y aguacate.")
FOSFORO = "También incluye fuentes de fósforo como maní, lácteos, avena y bulgur; con enfermedad renal requieren ajuste."


def test_kp_renal_es_no_agudo():
    assert cna._critical_is_non_acute([KP]) is True
    assert cna._critical_is_non_acute([KP, FOSFORO]) is True


@pytest.mark.parametrize("texto", [
    KP + " Riesgo de hiperkalemia.",
    KP + " Puede provocar arritmia.",
    "Paciente en diálisis: el potasio del plan es excesivo.",
    "TECHO RENAL DE PROTEÍNA VIOLADO (rechazo de seguridad clínica ERC): el plan supera el cap de proteína KDIGO.",
])
def test_lo_agudo_sigue_agudo(texto):
    assert cna._critical_is_non_acute([texto]) is False


def test_con_otro_critico_agudo_al_lado_sigue_agudo():
    assert cna._critical_is_non_acute([KP, "El almuerzo contiene maní, alérgeno declarado."]) is False


def test_ancla():
    assert "P1-PLAN-LOTE-261-KP-RENAL" in (_BACKEND / "critico_no_agudo.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 261
