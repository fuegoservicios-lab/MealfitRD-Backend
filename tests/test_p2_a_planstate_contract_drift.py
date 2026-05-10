"""[P2-A · 2026-05-08] Drift detection: contrato P2-CANDIDATE-A vs código real.

Bug original (audit 2026-05-08):
  El contrato `[P2-CANDIDATE-A]` en `graph_orchestrator.py:2463-2496` listaba
  `_shopping_coherence_action_taken` como key viva en `plan_result`, pero el
  grep global mostraba 0 asignaciones. La key existía SOLO en el comentario
  → drift documentación↔código. Un futuro contributor leyendo el contrato y
  haciendo `plan_result.get("_shopping_coherence_action_taken")` recibe `None`
  silencioso y debugea por minutos.

Fix aplicado:
  Opción A — eliminar la línea drift del contrato. La info ya vive en
  `_shopping_coherence_block_history[-1].action_taken` (más rico: incluye
  timestamp, attempt, hypotheses count). El contrato ahora documenta esa
  estructura como fuente única.

Cobertura del test:
  - Cada key con backticks `_*` mencionada en el contrato debe aparecer
    asignada (`= ...`) o creada (`jsonb_set`/`COALESCE`) al menos UNA vez
    en algún archivo del backend fuera de:
      - el propio bloque del contrato
      - tests/
      - markdown/docs
  - Si un dev añade una nueva key al contrato sin asignarla → este test
    falla y guía al fix (asignarla o removerla del contrato).

Mismo patrón que cross-language drift detection (test_p0_form_6,
test_p1_form_14, test_p3_5_bio_ranges_parity).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_ORCHESTRATOR_PATH = _BACKEND_ROOT / "graph_orchestrator.py"

# Anclas del bloque del contrato — si se renombran, este test falla
# explícitamente con mensaje guía.
_CONTRACT_START_MARKER = "[P2-CANDIDATE-A"
_CONTRACT_END_MARKER = "============================================================"

# Keys que aparecen en el contrato como ejemplos genéricos (no son keys de
# `plan_result`, son variables Python o atributos de dict). Excluidas del scan
# para evitar falsos positivos.
_CONTRACT_PROSE_TOKENS = {
    "_token_buffers",  # mencionado como ejemplo en la prosa del contrato
}


def _extract_contract_block_lines() -> list[tuple[int, str]]:
    """Extrae las líneas del bloque P2-CANDIDATE-A. Devuelve (lineno, texto)."""
    content = _ORCHESTRATOR_PATH.read_text(encoding="utf-8").splitlines()
    in_block = False
    block: list[tuple[int, str]] = []
    end_seen_after_start = 0
    for lineno, line in enumerate(content, start=1):
        if _CONTRACT_START_MARKER in line:
            in_block = True
            block.append((lineno, line))
            continue
        if in_block:
            block.append((lineno, line))
            if _CONTRACT_END_MARKER in line:
                end_seen_after_start += 1
                # El bloque tiene 2 líneas con el marker (apertura y cierre).
                # La de apertura YA pasó (línea con [P2-CANDIDATE-A] tiene === antes).
                if end_seen_after_start >= 1 and "P2-CANDIDATE-A" not in line:
                    break
    return block


def _extract_claimed_keys(block: list[tuple[int, str]]) -> set[str]:
    """Parses el bloque y extrae los identificadores `_*` con backticks SOLO
    de líneas que son bullet items del contrato (formato `#   - \\`...\\``).

    La prosa del contrato puede mencionar funciones (`_recompute_*`,
    `_token_buffers`, etc.) en backticks como referencia — NO son keys de
    `plan_result`. Restringir la extracción a bullets evita falsos positivos
    por function names citados en explicaciones.
    """
    keys: set[str] = set()
    backtick_pattern = re.compile(r"`(_[a-zA-Z0-9_]+)`")
    bullet_pattern = re.compile(r"^\s*#\s+-\s+`")
    for _, line in block:
        if not bullet_pattern.search(line):
            continue
        for m in backtick_pattern.finditer(line):
            tok = m.group(1)
            if tok in _CONTRACT_PROSE_TOKENS:
                continue
            keys.add(tok)
    return keys


def _scan_backend_for_assignment(key: str) -> list[str]:
    """Busca asignaciones del key fuera del propio contrato. Devuelve hits.

    Patterns aceptados como evidencia de "key real":
      - `<dict>["<key>"] = ...`            (asignación dict)
      - `<dict>['<key>'] = ...`            (idem comillas simples)
      - `<dict>.pop("<key>", ...)`         (consumidor — implica que existe)
      - `<dict>.setdefault("<key>", ...)`  (idem)
      - `jsonb_set(..., '{<key>}', ...)`   (persistencia DB path-text)
      - `ARRAY['<key>', ...]`              (persistencia DB path-array)

    Si solo aparece en docstrings/comentarios → 0 hits en patterns de
    asignación → drift.
    """
    quoted_d = re.escape('"' + key + '"')
    quoted_s = re.escape("'" + key + "'")
    # Asignación, pop, setdefault, jsonb path text/array
    assign_re = re.compile(
        rf"\[\s*(?:{quoted_d}|{quoted_s})\s*\]\s*="
        rf"|\.pop\(\s*(?:{quoted_d}|{quoted_s})"
        rf"|\.setdefault\(\s*(?:{quoted_d}|{quoted_s})"
        rf"|jsonb_set\([^)]*'\{{[^}}]*{re.escape(key)}[^}}]*\}}'"
        rf"|ARRAY\s*\[\s*(?:{quoted_d}|{quoted_s})"
    )
    hits: list[str] = []
    for path in _BACKEND_ROOT.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name.startswith("_deprecated_"):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # Skip the contract block itself in graph_orchestrator.py
        if path == _ORCHESTRATOR_PATH:
            block_linenos = {ln for ln, _ in _extract_contract_block_lines()}
        else:
            block_linenos = set()
        for lineno, line in enumerate(content.splitlines(), start=1):
            if lineno in block_linenos:
                continue
            if assign_re.search(line):
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}")
                break  # Un hit por archivo es suficiente
    return hits


def test_contract_block_is_findable():
    """Sanity: el bloque del contrato existe y es parseable.

    Si este test falla, alguien renombró el marker `[P2-CANDIDATE-A` o el
    closing `=...=`. Restaurar los markers o actualizar este test.
    """
    block = _extract_contract_block_lines()
    assert block, (
        f"Bloque del contrato P2-CANDIDATE-A no encontrado en "
        f"{_ORCHESTRATOR_PATH.relative_to(_REPO_ROOT)}. "
        f"Marker esperado: {_CONTRACT_START_MARKER!r}"
    )
    # El bloque debe tener al menos 10 líneas (no es un stub vacío)
    assert len(block) >= 10, (
        f"Bloque del contrato sospechosamente corto ({len(block)} líneas). "
        "Verificar que no se truncó."
    )


def test_every_claimed_key_has_real_assignment():
    """Cada key listada en el contrato debe tener al menos 1 asignación real.

    Esta es la red de seguridad contra drift documentación↔código que motivó
    P2-A. Si falla:
      - O removes la key del contrato (si ya no se usa)
      - O añades la asignación real (si la querías mantener)
    """
    block = _extract_contract_block_lines()
    claimed = _extract_claimed_keys(block)
    assert claimed, (
        "Ninguna key extraída del contrato. ¿Cambió el formato de los "
        "bullets `_<key>` con backticks?"
    )

    drifted: dict[str, list[str]] = {}
    for key in sorted(claimed):
        hits = _scan_backend_for_assignment(key)
        if not hits:
            drifted[key] = []

    assert not drifted, (
        f"Drift detectado en contrato P2-CANDIDATE-A: las siguientes keys "
        f"están listadas pero NO tienen asignación en código activo:\n  - "
        + "\n  - ".join(sorted(drifted.keys()))
        + "\n\nFix: o remover del contrato (estilo P2-A 2026-05-08), o "
        "añadir la asignación real en el nodo apropiado."
    )


def test_action_taken_lives_in_history_not_top_level():
    """[P2-A regression] La key `_shopping_coherence_action_taken` NO debe
    re-aparecer como mirror top-level. La fuente única es
    `_shopping_coherence_block_history[-1].action_taken`.

    Si este test falla, alguien recreó el drift que P2-A 2026-05-08 cerró.
    Soluciones:
      - Si necesitas acceso top-level: actualiza el contrato + este test +
        documenta por qué (ej. consultas SQL frecuentes que prefieren
        plan_data->>'_shopping_coherence_action_taken').
      - Si fue por error: remover la asignación, leer de history[-1].
    """
    forbidden = "_shopping_coherence_action_taken"
    hits = _scan_backend_for_assignment(forbidden)
    assert not hits, (
        f"Re-introducción de la key `{forbidden}` detectada en: {hits}. "
        "Esta key fue removida en P2-A 2026-05-08 porque su contenido ya "
        "vive en `_shopping_coherence_block_history[-1].action_taken`. "
        "Si necesitas top-level mirror, justifica + actualiza P2-A memory."
    )


def test_history_action_taken_is_documented_in_contract():
    """[P2-A regression] El contrato debe seguir documentando que
    `history[-1].action_taken` es la fuente del estado del consumer.

    Sin esta nota, futuros contributors no saben dónde leer el resultado y
    podrían recrear `_shopping_coherence_action_taken` como mirror.
    """
    block = _extract_contract_block_lines()
    text = "\n".join(line for _, line in block)
    # Acepta cualquier forma de mencionar history + action_taken cerca
    assert "history[-1].action_taken" in text or "history[-1]" in text and "action_taken" in text, (
        "El contrato P2-CANDIDATE-A ya no menciona `history[-1].action_taken` "
        "como fuente del estado del consumer. Restaurar la nota — sin ella, "
        "futuros contributors podrían recrear el drift cerrado en P2-A."
    )
