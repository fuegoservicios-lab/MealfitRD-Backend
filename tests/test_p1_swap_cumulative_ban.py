"""[P1-SWAP-CUMULATIVE-BAN · 2026-07-12] El swap prohíbe las proteínas del día UPFRONT y las
prohibiciones del retry se ACUMULAN entre intentos.

Caso vivo (corr=382cf533, 02:23Z — swap 'menos tiempo' del owner): 3 intentos quemados:
huevo → pollo → LA MISMA tortilla de huevo otra vez. Dos causas:
1. La directiva de retry se construía `prompt_base + clash ACTUAL` — REEMPLAZA, no acumula:
   el ban de huevo del intento 1 se PERDÍA al banear pollo en el 2 → el 3 volvió al huevo.
   (Mismo bug que la directiva acumulada del plan-gen cerró: RESTRICCIONES ACUMULADAS.)
2. El prompt base solo llevaba una "preferencia" con NOMBRES de platos — nunca la lista de
   labels prohibidos. El generador (tunelizado en 'rápido' → tortilla) la ignoraba.

Contrato:
1. UPFRONT: si hay labels usados hoy (derivados de same_day_other_meal_blobs con el MISMO
   SSOT del gate) y el swap NO es pantry-strict → línea dura "PROHIBIDO" con los labels.
2. CUMULATIVO: `_banned_sd_labels` acumula (`|=`) los clashes de TODOS los intentos y la
   directiva de retry enumera el set COMPLETO.
3. pantry-strict conserva el wording suave (no pelear con la despensa — el gate ya tiene su
   fallback anti-slot-imposible).

tooltip-anchor: P1-SWAP-CUMULATIVE-BAN
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")


def test_upfront_hard_ban_from_gate_ssot():
    i = _AGENT.find("PROTEÍNAS PROHIBIDAS HOY")
    assert i != -1, (
        "el prompt base debe enumerar los labels ya usados hoy como PROHIBICIÓN dura "
        "(la 'preferencia' con nombres de platos quemó 3 intentos: corr=382cf533)"
    )
    win = _AGENT[max(0, i - 2500):i + 800]
    assert "_protein_gate_labels_in_text" in win, (
        "los labels prohibidos se derivan con el MISMO SSOT del gate (asimetría "
        "prompt↔gate = intentos quemados, lección P1-CRITIQUE-SLOT-PARITY)"
    )
    assert "strict_pantry" in win, (
        "pantry-strict conserva el wording suave (no pelear con la despensa)"
    )


def test_retry_directive_accumulates():
    assert "_banned_sd_labels" in _AGENT, "acumulador de labels baneados desapareció"
    i = _AGENT.find("_banned_sd_labels |= ")
    assert i != -1, "los clashes deben ACUMULARSE (|=), no reemplazarse"
    # buscar el header DESPUÉS del acumulador: varios guardrails comparten el mismo
    # header "ATENCIÓN AL INTENTO FALLIDO ANTERIOR" y el primero no es el del same-day.
    j = _AGENT.find("ATENCIÓN AL INTENTO FALLIDO ANTERIOR", i)
    assert j != -1
    win = _AGENT[j - 900:j + 900]
    assert "_banned_sd_labels" in win, (
        "la directiva de retry debe enumerar el set COMPLETO acumulado — construirla solo "
        "con el clash actual pierde los bans previos (huevo→pollo→huevo, corr=382cf533)"
    )


def test_marker_anchored():
    assert _AGENT.count("P1-SWAP-CUMULATIVE-BAN") >= 2
