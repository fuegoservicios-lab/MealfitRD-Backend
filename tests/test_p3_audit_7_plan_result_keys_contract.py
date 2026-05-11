"""[P3-AUDIT-7 . 2026-05-10] Lock-the-contract del bloque CONTRATO
claves que NO viven en PlanState en graph_orchestrator.py:2366+.

Bug original (audit 2026-05-10):
  El bloque enumera las keys de plan_result que persisten en el payload
  opaco (no como campos top-level del state, donde LangGraph strict-schema
  las filtraría). CLAUDE.md menciona el bloque pero no había test que
  enforce que las keys listadas siguen siendo:
    (a) producidas en código (escritas en algún call site), Y
    (b) consumidas en código (leídas en algún consumer).

  Un refactor que renombre una key, mueva su productor/consumer, o la
  declare en PlanState sin tocar el bloque CONTRATO causaría el bug
  equivalente a P1-G original: el flag se persistiría pero no se leería,
  o se leería pero su productor habría cambiado.

Diseño:
  1. Parsea el bloque CONTRATO (líneas ~2366-2404) extrayendo las keys
     enumeradas con bullet `- <backtick>_<key><backtick>`.
  2. Verifica que cada key aparezca como productor o consumer en al menos
     uno de los archivos en `_USAGE_FILES` (graph_orchestrator, cron_tasks,
     routers/plans, db_plans).
  3. Si una key no tiene referencias fuera del bloque, falla con copy
     explicativo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_ORCHESTRATOR = _REPO_ROOT / "backend" / "graph_orchestrator.py"

# Archivos donde las keys del CONTRATO pueden ser productores o consumers.
# `graph_orchestrator.py` es el SSOT del bloque (escribe en el dict
# plan_result y lo lee en review_plan_node). `cron_tasks.py` es co-
# productor: el chunk worker actualiza `_merged_chunk_ids`,
# `_user_forced_simplified_weeks`, etc. directamente sobre meal_plans.
# Si añades un nuevo archivo que toca estas keys, agrégalo aquí.
_USAGE_FILES = (
    _REPO_ROOT / "backend" / "graph_orchestrator.py",
    _REPO_ROOT / "backend" / "cron_tasks.py",
    _REPO_ROOT / "backend" / "routers" / "plans.py",
    _REPO_ROOT / "backend" / "db_plans.py",
)

_CONTRACT_HEADER = "[P2-CANDIDATE-A · 2026-05-08] CONTRATO"

# Patrón de captura de keys: dentro del bloque CONTRATO, bullet con backticks.
# Acepta tanto bullets one-per-line como múltiples keys en la misma línea
# (e.g., `- \`_critique_unresolved\`, \`_merged_chunk_ids\`, \`_user_forced_simplified_weeks\``).
_KEY_LINE_RE = re.compile(r"`(_[a-zA-Z_][a-zA-Z0-9_]*)`")


def _read_orchestrator() -> str:
    return _GRAPH_ORCHESTRATOR.read_text(encoding="utf-8")


def _extract_contract_block(src: str) -> str:
    """Aísla el bloque CONTRATO defensivo. Empieza en el header
    `[P2-CANDIDATE-A · ...] CONTRATO` y termina en la línea de cierre
    `# ============` siguiente."""
    start = src.find(_CONTRACT_HEADER)
    assert start >= 0, (
        f"No se encontró el header '{_CONTRACT_HEADER}' en "
        f"graph_orchestrator.py. ¿Lo renombraste? Actualiza este test."
    )
    # Busca la línea con `# =====` que cierra el bloque (≥10 iguales seguidos).
    end_pattern = re.compile(r"#\s*=={10,}", re.MULTILINE)
    m = end_pattern.search(src, pos=start + len(_CONTRACT_HEADER))
    assert m, (
        "No se encontró el cierre del bloque CONTRATO (# ======). El bloque "
        "vive en un comment-frame; si lo refactorizaste, actualiza el patrón."
    )
    return src[start:m.end()]


def _extract_contract_keys(contract_block: str) -> set[str]:
    """Extrae las keys enumeradas en el bloque CONTRATO. Filtra falsos
    positivos: solo keys que empiezan con _ (convención del contrato)
    y que aparecen como bullets '- <backtick><key><backtick>'."""
    keys: set[str] = set()
    # Encuentra líneas de bullet (espacios + `-` + backticks).
    bullet_lines = re.findall(
        r"^\s*#\s*-\s*[`].*",
        contract_block,
        re.MULTILINE,
    )
    for line in bullet_lines:
        for match in _KEY_LINE_RE.finditer(line):
            keys.add(match.group(1))
    # Sanity floor: el bloque debe enumerar al menos 4 keys (P2-CANDIDATE-A
    # original tenía 6+; si bajan, alguien rompió el parse o quitó keys).
    return keys


def test_contract_block_present():
    """El bloque CONTRATO debe existir — su ausencia tumba el invariante
    documentado de que las `plan_result` keys NO migran al state sin
    pasar por revisión."""
    src = _read_orchestrator()
    assert _CONTRACT_HEADER in src, (
        "El bloque CONTRATO defensivo `[P2-CANDIDATE-A · ...] CONTRATO: "
        "claves que NO viven en PlanState` fue removido de "
        "graph_orchestrator.py. Restaurarlo o documentar en CLAUDE.md "
        "que las plan_result keys ya no requieren contract enforcement."
    )


def test_contract_enumerates_min_keys():
    """El bloque CONTRATO enumera al menos 6 keys (floor histórico).

    Si bajan, alguien quitó keys sin migrar al state — riesgo de bug
    equivalente al original P1-G (key olvidada entre productor y consumer).
    """
    src = _read_orchestrator()
    block = _extract_contract_block(src)
    keys = _extract_contract_keys(block)
    assert len(keys) >= 6, (
        f"El bloque CONTRATO enumera solo {len(keys)} keys: {sorted(keys)}. "
        f"Floor histórico = 6 (P2-CANDIDATE-A original). Si removiste una "
        f"key intencionalmente, documenta la migración a state o la "
        f"eliminación del flag en CLAUDE.md."
    )


def test_every_contract_key_has_producer_and_consumer():
    """Cada key del bloque CONTRATO debe tener ≥1 productor y ≥1 consumer
    en `graph_orchestrator.py`.

    Productor: `plan_result["<key>"] = ...`, `plan["<key>"] = ...`, o
               `result["<key>"] = ...`.
    Consumer: `plan_result.get("<key>")`, `plan_result["<key>"]` (lectura),
              o variantes con `plan`/`result`/`plan_data`.

    Heurística textual: las dos formas son `["<key>"]` y `.get("<key>")`
    en cualquier carrier (`plan_result`, `plan`, `result`, `plan_data`).
    Si una key aparece ≥2 veces en el archivo (excluyendo el bloque
    CONTRATO mismo), asumimos productor + consumer. Si aparece solo en
    el bloque, es ghost.
    """
    src = _read_orchestrator()
    block = _extract_contract_block(src)
    keys = _extract_contract_keys(block)

    # Concatenar todos los archivos donde el contrato puede ser
    # productor/consumer. Removemos el bloque CONTRATO del orchestrator
    # para no contar sus propias menciones.
    src_outside_parts: list[str] = []
    for fp in _USAGE_FILES:
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        if fp == _GRAPH_ORCHESTRATOR:
            text = text.replace(block, "")
        src_outside_parts.append(text)
    src_outside = "\n".join(src_outside_parts)

    orphan_keys: list[str] = []
    underused_keys: list[tuple[str, int]] = []

    for key in sorted(keys):
        # Patrón: la key aparece dentro de subscript `["<key>"]` o
        # `.get("<key>")` o `'<key>'` en general como string literal.
        usage_pattern = re.compile(
            rf"""["']{re.escape(key)}["']""",
        )
        hits = usage_pattern.findall(src_outside)
        if len(hits) == 0:
            orphan_keys.append(key)
        elif len(hits) < 2:
            # Una sola referencia: probablemente solo productor o solo
            # consumer. Sospechoso pero no fatal.
            underused_keys.append((key, len(hits)))

    if orphan_keys:
        pytest.fail(
            "Las siguientes keys del bloque CONTRATO NO tienen referencias "
            "fuera del bloque mismo (ghost en el contrato):\n  - "
            + "\n  - ".join(orphan_keys)
            + "\n\nProcedimiento:\n"
            "  1. Si el campo se eliminó del pipeline, removerlo del bloque CONTRATO.\n"
            "  2. Si se renombró, actualizar tanto el bloque como los call sites.\n"
            "  3. Si se migró a PlanState, removerlo del bloque y declararlo "
            "como campo TypedDict en `class PlanState`."
        )

    if underused_keys:
        pytest.fail(
            "Las siguientes keys del bloque CONTRATO aparecen UNA sola vez "
            "fuera del bloque (probable productor sin consumer, o viceversa):\n  - "
            + "\n  - ".join(f"{k} ({n} ref)" for k, n in underused_keys)
            + "\n\nLas keys del CONTRATO requieren productor + consumer "
            "(ambas direcciones). Si una key solo se escribe (no se lee), su "
            "valor se pierde silenciosamente; si solo se lee, queda None siempre."
        )


def test_known_contract_keys_present():
    """Sanity: las keys originales documentadas en P2-CANDIDATE-A deben
    seguir en el bloque. Si quitas una, hazlo deliberadamente y elimina
    su entry de esta whitelist."""
    src = _read_orchestrator()
    block = _extract_contract_block(src)
    keys = _extract_contract_keys(block)

    expected_anchors = {
        "_shopping_coherence_block",
        "_shopping_coherence_block_history",
        "_pantry_supplement_required",
    }
    missing = expected_anchors - keys
    assert not missing, (
        f"Las siguientes keys ancla del CONTRATO P2-CANDIDATE-A fueron "
        f"removidas del bloque: {sorted(missing)}. Estas son las keys "
        f"que originalmente motivaron el bloque (gap P1-G del flujo de "
        f"coherencia). Si su migración fue intencional, actualiza el "
        f"`expected_anchors` set + documenta la migración en CLAUDE.md."
    )
