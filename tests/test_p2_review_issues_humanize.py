"""[P2-REVIEW-ISSUES-HUMANIZE · 2026-07-10] Las observaciones entregadas al usuario
(`_review_issues` → toast "Plan generado con observaciones") mostraban jerga interna:
"COHERENCIA RECETAS LISTA: 3 divergencia(s) críticas (foods: Res, Cerdo, Queso fresco).
action=reject_minor." (screenshot del owner 2026-07-10). El string técnico es correcto para
logs/directivas de retry (el LLM necesita precisión), pero el usuario necesita copy es-DO claro.

Fix: `_humanize_review_issue` mapea los patrones conocidos a copy user-friendly en el boundary
de entrega (:35353); lo crudo se preserva en `_review_issues_raw` (forense). El frontend no cambia
(no se puede deployar sin shippear WIP ajeno).

tooltip-anchor: P2-REVIEW-ISSUES-HUMANIZE
"""
from pathlib import Path

import graph_orchestrator as go

_GO_SRC = Path(go.__file__).read_text(encoding="utf-8")


def test_coherence_jargon_humanized():
    raw = ("COHERENCIA RECETAS LISTA: 3 divergencia(s) críticas (foods: Res, Cerdo, "
           "Queso fresco). action=reject_minor.")
    out = go._humanize_review_issue(raw)
    assert "action=" not in out and "reject_minor" not in out
    assert "divergencia" not in out.lower()
    assert "lista de compras" in out.lower()          # dice QUÉ revisar en lenguaje humano


def test_repetition_humanized():
    raw = "REPETICIÓN DETECTADA: Los siguientes platos principales ya aparecieron en planes recientes..."
    out = go._humanize_review_issue(raw)
    assert "REPETICIÓN DETECTADA" not in out
    assert "recientes" in out.lower() or "parecen" in out.lower()


def test_same_day_protein_humanized():
    raw = "MISMA PROTEÍNA REPETIDA EL MISMO DÍA (rechazo de variedad): la misma proteína principal..."
    out = go._humanize_review_issue(raw)
    assert "rechazo de variedad" not in out
    assert "proteína" in out.lower()


def test_unknown_issue_keeps_text_but_strips_action_suffix():
    raw = "ALGO NUEVO INESPERADO: detalle técnico X. action=reject_high."
    out = go._humanize_review_issue(raw)
    assert "action=" not in out
    assert "ALGO NUEVO INESPERADO" in out             # no inventa: conserva lo desconocido


def test_humanize_failsafe():
    assert go._humanize_review_issue(None) == ""
    assert go._humanize_review_issue(123) == "123"


def test_wired_at_delivery_boundary_with_raw_preserved():
    # parser: en el path _review_failed_but_delivered se entrega humanizado + se preserva el raw
    assert "_review_issues_raw" in _GO_SRC
    i = _GO_SRC.index('"_review_failed_but_delivered"] = True')
    window = _GO_SRC[i:i + 1200]
    assert "_humanize_review_issue" in window
