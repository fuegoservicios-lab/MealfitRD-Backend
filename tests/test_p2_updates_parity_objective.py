"""[audit objetivo · P2-6 / P2-12 / P2-2] Paridad de updates con la generación.

- P2-6 (P2-VERIFIED-ONLY-UPDATE): swap y chat-modify inyectan el bloque "USA EXCLUSIVAMENTE" del catálogo
  verificado en el path no pantry-strict (cuando el usuario va de compras), gated por el mismo helper que S1.
- P2-12 (P2-PANTRY-MICRO-SOFT): en pantry-strict, una preferencia SUAVE de densidad de micros (en vez de skip total).
- P2-2 (P2-REGEN-DAY-DEFICIT-HONESTY): regenerate-day distingue déficit-por-Nevera de déficit-alcanzable.
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


# ───────────────────────── P2-6: verified-only en updates ─────────────────────────
def test_verified_block_injected_in_swap_non_pantry():
    assert "P2-VERIFIED-ONLY-UPDATE" in _AGENT
    assert "_get_verified_catalog_instruction" in _AGENT
    # gated al path no-pantry-strict (`if not clean_ingredients:`)
    assert _AGENT.count("_get_verified_catalog_instruction") >= 1


def test_verified_block_injected_in_chat_modify():
    assert "P2-VERIFIED-ONLY-UPDATE" in _TOOLS
    assert "_get_verified_catalog_instruction" in _TOOLS


# ───────────────────────── P2-12: micro-steer suave en pantry-strict ─────────────────────────
def test_pantry_micro_soft_swap():
    assert "P2-PANTRY-MICRO-SOFT" in _AGENT
    assert "DENSIDAD DE MICROS (preferencia suave" in _AGENT


def test_pantry_micro_soft_chat_modify():
    assert "P2-PANTRY-MICRO-SOFT" in _TOOLS
    assert "DENSIDAD DE MICROS (preferencia suave" in _TOOLS


# ───────────────────────── P2-2: regen-day deficit honesty ─────────────────────────
def test_regen_day_pantry_limited_flag_set_on_reverts():
    assert "P2-REGEN-DAY-DEFICIT-HONESTY" in _PLANS
    assert "_pantry_limited = False" in _PLANS
    # se setea True en AMBOS reverts (rebalance + FASE A)
    assert _PLANS.count("_pantry_limited = True") >= 2, "el flag debe setearse en los 2 reverts pantry"


def test_regen_day_warning_branches_by_cause():
    # mensaje distinto según pantry-limited vs alcanzable
    assert "Tu Nevera puede no tener suficiente para tu objetivo" in _PLANS
    assert "tu Nevera sí alcanza" in _PLANS
    # flag expuesto en la respuesta
    assert '"day_deficit_pantry_limited"' in _PLANS
