"""[P1-PROTEIN-REPEAT-DETERMINISTIC · 2026-07-07] El critique LLM juzgaba "misma
proteína el mismo día" por su cuenta → falsos positivos → retry loop (quema DeepSeek).

Forense (planes 5f80f797/21af5c1b/4339544f entregados): `variety_report.
same_day_protein_repeats == 0` en TODOS — el detector DETERMINISTA (word-boundary +
alias map) los pasa correctamente y al mirar los nombres NO hay repetición. Pero el
reviewer LLM flaggeaba falsos positivos ("Bollitos de Yuca Rellenos de Queso Blanco no
tiene res, pero el sistema detectó res"; 'res' en "Rellenos"/"fresas"; legumbres que en
RD se repiten a diario a propósito) → 3 retries → entrega como advisory.

Fix: el gate DETERMINISTA es la autoridad para proteína-repetida (corre por separado y
rechaza si de verdad hay repetición). El critique LLM ya NO juzga "misma proteína el
mismo día" por su cuenta — solo el carbohidrato repetido. Elimina los falsos positivos
que causaban los retries; la detección real de proteína animal la mantiene el gate
determinista + su autofix.
tooltip-anchor: P1-PROTEIN-REPEAT-DETERMINISTIC
"""
from __future__ import annotations

import graph_orchestrator as g

_INSTR = g._CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION


def test_marker_present():
    assert "P1-PROTEIN-REPEAT-DETERMINISTIC" in _INSTR


def test_llm_defers_protein_repeat_to_deterministic():
    """El critique instruye NO juzgar proteína-repetida por su cuenta."""
    low = _INSTR.lower()
    assert "no juzgues por tu cuenta" in low
    assert "misma prote" in low  # "misma PROTEÍNA el mismo día"
    assert "determinista" in low
    assert "no bajes ning" in low, "debe prohibir bajar score por proteína-repetida"


def test_slot_clause_no_longer_judges_protein():
    """La cláusula de slot ya NO pide al LLM juzgar 'la PROTEÍNA PRINCIPAL' repetida —
    solo el CARBOHIDRATO (el protein-repeat lo maneja el determinista)."""
    # El texto viejo era: "CENA NO debe repetir la PROTEÍNA PRINCIPAL ni el CARBOHIDRATO"
    assert "repetir la PROTEÍNA PRINCIPAL ni el CARBOHIDRATO" not in _INSTR, (
        "la cláusula de slot sigue pidiendo al LLM juzgar la proteína repetida (falsos positivos)"
    )
    # El carbohidrato repetido SÍ sigue siendo juicio del LLM
    assert "CARBOHIDRATO PRINCIPAL del almuerzo" in _INSTR


def test_legume_and_substring_confusion_documented():
    """La nota explica los dos modos de falso positivo (substring + legumbres RD)."""
    low = _INSTR.lower()
    assert "rellenos" in low or "fresas" in low, "documenta el falso positivo por substring"
    assert "habichuela" in low or "legumbre" in low, "documenta la exención de legumbres RD"
