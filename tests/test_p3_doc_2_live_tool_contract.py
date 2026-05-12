"""[P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT en las 4 tools del agente
que aceptan `user_id` y tienen body real (no DUMMY).

Cierre del audit 2026-05-11 (P3 polish):
    P0-AGENT-1 cerró el IDOR vía LLM-supplied user_id mediante force-override
    en `agent.py:execute_tools`. P3-DOC-2 añade un contrato inline en cada
    tool live para que un revisor futuro entienda que:

      - El path normal (chat-agent) ya está protegido.
      - Llamadores DIRECTOS (tests, scripts, futuros endpoints HTTP que
        importen la tool sin pasar por execute_tools) DEBEN seguir las 3
        reglas: no confiar en LLM user_id, filtrar `WHERE user_id = %s`
        en SQL, y no introducir lookups globales sin filtro.

P3-NEW-7 ya cubría `modify_single_meal` (tool DUMMY interceptada). P3-DOC-2
extiende el patrón a las 4 tools live restantes con contrato adaptado:

    1. `update_form_field`        — muta `user_profiles.health_profile` +
                                    `user_facts` (delete_user_facts_by_metadata)
    2. `log_consumed_meal`        — muta `consumed_meals` + `user_inventory`
                                    (deduct_consumed_meal_from_inventory)
    3. `modify_pantry_inventory`  — muta `user_inventory` (add/deduct)
    4. `mark_shopping_list_purchased` — muta `user_inventory` (restock)

Test parser-based: scan `tools.py` para verificar que cada tool live tenga
el bloque P3-DOC-2 LIVE-TOOL CONTRACT + tooltip-anchor compartido. Si
alguien modifica una tool y borra el contrato, este test falla con
referencia al patrón canónico.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_FP = _REPO_ROOT / "backend" / "tools.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _TOOLS_FP.read_text(encoding="utf-8")


def _extract_function_body(src: str, signature_prefix: str) -> str:
    """Body desde `def <name>(...)` hasta el siguiente `@tool`/`@`/`def`
    top-level. Reusa el mismo patrón que `test_p3_new_7_*`."""
    start = src.find(signature_prefix)
    assert start > 0, f"`{signature_prefix}` no encontrado en tools.py"
    rest = src[start + len(signature_prefix):]
    nx = re.search(r"\n(@tool|def\s+\w+|@\w+)", rest)
    end = start + len(signature_prefix) + (nx.start() if nx else len(rest))
    return src[start:end]


# Tabla canónica de tools live cubiertas por P3-DOC-2. Si extendes el set
# (e.g., añades una tool nueva al agente que toma user_id), añadir aquí
# Y al docstring de la tool nueva.
_LIVE_TOOLS_WITH_CONTRACT = [
    "def update_form_field(",
    "def log_consumed_meal(",
    "def modify_pantry_inventory(",
    "def mark_shopping_list_purchased(",
]


@pytest.mark.parametrize("tool_signature", _LIVE_TOOLS_WITH_CONTRACT)
def test_each_live_tool_has_contract_marker(src: str, tool_signature: str):
    """Cada tool live tiene el bloque `[P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT`."""
    body = _extract_function_body(src, tool_signature)
    assert "[P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT" in body, (
        f"P3-DOC-2 regresión: la tool `{tool_signature}` ya NO contiene el "
        f"bloque `[P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT`. Sin él, un "
        f"refactor futuro puede introducir lookups sin filtro `user_id` y "
        f"reabrir IDOR. Restaurar el bloque antes de mergear (ver patrón "
        f"en otras tools live de tools.py)."
    )


@pytest.mark.parametrize("tool_signature", _LIVE_TOOLS_WITH_CONTRACT)
def test_each_live_tool_has_tooltip_anchor(src: str, tool_signature: str):
    """Cada tool live menciona el tooltip-anchor compartido."""
    body = _extract_function_body(src, tool_signature)
    assert "P3-DOC-2-LIVE-TOOL-CONTRACT" in body, (
        f"P3-DOC-2 regresión: la tool `{tool_signature}` no tiene el "
        f"tooltip-anchor `P3-DOC-2-LIVE-TOOL-CONTRACT`. Sin él, un grep "
        f"cross-codebase desde docs/audits no encuentra esta función como "
        f"ejemplo del patrón."
    )


@pytest.mark.parametrize("tool_signature", _LIVE_TOOLS_WITH_CONTRACT)
def test_each_live_tool_references_p0_agent_1(src: str, tool_signature: str):
    """El contrato debe explicar la conexión con P0-AGENT-1 (force-override
    upstream que protege el path normal)."""
    body = _extract_function_body(src, tool_signature)
    assert "P0-AGENT-1" in body, (
        f"P3-DOC-2 regresión: la tool `{tool_signature}` no menciona "
        f"`P0-AGENT-1`. Sin esa referencia, un revisor futuro no entiende "
        f"que el path normal (chat-agent → execute_tools) está protegido "
        f"y puede creer (incorrectamente) que la tool es vulnerable hoy."
    )


def test_grep_finds_exactly_4_anchors(src: str):
    """Cross-link: el anchor `P3-DOC-2-LIVE-TOOL-CONTRACT` aparece
    EXACTAMENTE 4 veces (1 por tool live), Y el anchor canónico
    `_LIVE_TOOLS_WITH_CONTRACT` arriba está sincronizado.

    Si añades una tool live nueva al agente que toma `user_id`, debes:
      (1) añadir el bloque LIVE-TOOL CONTRACT a su docstring/body.
      (2) añadirla a `_LIVE_TOOLS_WITH_CONTRACT` arriba.
    Este test falla si haces (1) sin (2) o viceversa.
    """
    # Solo contamos el anchor canónico, NO el "P3-DOC-2 · 2026-05-11"
    # (que aparece como header del bloque + en los docstrings/comments
    # que mencionan el marker). El anchor único es para grep.
    n_anchors = src.count("Tooltip-anchor: P3-DOC-2-LIVE-TOOL-CONTRACT")
    expected = len(_LIVE_TOOLS_WITH_CONTRACT)
    assert n_anchors == expected, (
        f"P3-DOC-2 regresión: el anchor `Tooltip-anchor: "
        f"P3-DOC-2-LIVE-TOOL-CONTRACT` aparece {n_anchors} veces en "
        f"tools.py, esperaba {expected} (una por tool live registrada en "
        f"`_LIVE_TOOLS_WITH_CONTRACT`). Si añadiste una tool nueva, "
        f"actualiza la lista en este test. Si removiste una, también. "
        f"Si el anchor desapareció de una tool existente, restaurarlo."
    )


def test_anchors_use_consistent_block_format(src: str):
    """Los 4 bloques LIVE-TOOL CONTRACT comparten un formato canónico —
    `─` (line separator) + numbered list `1.`/`2.`/`3.`. Esto facilita
    leer las 4 lado a lado."""
    for tool_signature in _LIVE_TOOLS_WITH_CONTRACT:
        body = _extract_function_body(src, tool_signature)
        contract_idx = body.find("[P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT")
        assert contract_idx >= 0, "(asegurado por test anterior)"
        # ~25 líneas tras el header debe haber `1.`, `2.`, `3.` numbered list
        contract_block = body[contract_idx: contract_idx + 2500]
        for n in ("1.", "2.", "3."):
            assert n in contract_block, (
                f"P3-DOC-2 regresión: bloque CONTRATO de `{tool_signature}` "
                f"perdió el item `{n}` de la numbered list. Las 3 reglas "
                f"(no confiar en LLM user_id, filtro SQL, no lookups "
                f"globales) son canónicas; si falta una, un revisor futuro "
                f"puede romper el contrato sin darse cuenta."
            )
