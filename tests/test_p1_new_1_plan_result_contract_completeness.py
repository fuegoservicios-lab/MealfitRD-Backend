"""[P1-NEW-1 · 2026-05-11] Inverso del test P3-AUDIT-7.

Bug original (audit 2026-05-11):
    P3-AUDIT-7 enforza que cada key del CONTRATO tenga
    productor+consumer. PERO no enforza la dirección INVERSA:
    si una key `_xxx` viaja en `plan_result["_xxx"]` o
    `result["_xxx"]` en código de producción, ¿está documentada
    en el bloque CONTRATO?

    Sin este check, un dev puede añadir una nueva key al
    `plan_result` sin enterarse que existe el contrato, y la key
    quedará indocumentada — un refactor futuro que migre el dict
    a state-level la perdería sin que ningún test falle.

    Audit 2026-05-11 detectó 16 keys huérfanas no documentadas:
    `_best_attempt_*` (5), `_review_*` (4), `_schema_*` (2),
    `_skeleton*` (2), `_profile_embedding`, `_selected_techniques`,
    `_recipe_coherence_errors`. P1-NEW-1 extendió el CONTRATO para
    incluirlas y añadió este test para prevenir reincidencia.

Estrategia:
    1. Parsear el bloque CONTRATO y extraer las keys con backticks.
    2. Greppear `plan_result["..."]`/`plan_result.get("...")`/
       `result["..."]`/`plan["..."]` etc. en graph_orchestrator.py.
    3. Para cada key encontrada que empiece con `_`, verificar que
       esté en la lista del CONTRATO. Si no, fail.

Excepciones permitidas (whitelist):
    - Keys que aparecen UNA sola vez (probable write-only que no
      requiere consumer; el test P3-AUDIT-7 ya las marca como
      underused y NO las exige en el CONTRATO).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_ORCHESTRATOR = _REPO_ROOT / "backend" / "graph_orchestrator.py"

_CONTRACT_HEADER = "[P2-CANDIDATE-A · 2026-05-08] CONTRATO"
_KEY_LINE_RE = re.compile(r"`(_[a-zA-Z_][a-zA-Z0-9_]*)`")


def _read_src() -> str:
    return _GRAPH_ORCHESTRATOR.read_text(encoding="utf-8")


def _extract_contract_block(src: str) -> str:
    start = src.find(_CONTRACT_HEADER)
    assert start >= 0, "CONTRATO header no encontrado."
    end_pattern = re.compile(r"#\s*=={10,}", re.MULTILINE)
    m = end_pattern.search(src, pos=start + len(_CONTRACT_HEADER))
    assert m, "Cierre del bloque CONTRATO no encontrado."
    return src[start:m.end()]


def _extract_documented_keys(block: str) -> set[str]:
    """Extrae keys con backticks `_xxx` enumeradas en bullets `#   - ...`."""
    keys: set[str] = set()
    bullet_lines = re.findall(r"^\s*#\s*-\s*`.*", block, re.MULTILINE)
    for line in bullet_lines:
        for match in _KEY_LINE_RE.finditer(line):
            keys.add(match.group(1))
    return keys


def _extract_used_keys(src_outside_block: str) -> dict[str, int]:
    """Extrae keys `_xxx` accedidas vía subscript/get en carriers
    `plan_result`, `plan`, `result`. Devuelve {key: count}."""
    keys: dict[str, int] = {}
    # subscript: <carrier>["<key>"] o <carrier>['<key>']
    sub_pattern = re.compile(
        r"\b(?:plan_result|plan|result)\[\s*[\"\'](_[a-zA-Z_][a-zA-Z0-9_]*)[\"\']\s*\]"
    )
    # .get accesor: <carrier>.get("<key>") o .get('<key>')
    get_pattern = re.compile(
        r"\b(?:plan_result|plan|result)\.get\(\s*[\"\'](_[a-zA-Z_][a-zA-Z0-9_]*)[\"\']"
    )
    for m in sub_pattern.finditer(src_outside_block):
        keys[m.group(1)] = keys.get(m.group(1), 0) + 1
    for m in get_pattern.finditer(src_outside_block):
        keys[m.group(1)] = keys.get(m.group(1), 0) + 1
    return keys


# Whitelist: keys que aparecen pero NO requieren estar en el CONTRATO
# (typicamente keys legacy de state migrado o helpers temporales).
# Vacía por default; añadir solo con justificación explícita.
_WHITELIST: set[str] = set()


def test_every_plan_result_key_documented_in_contract():
    """Cada key `_xxx` accedida en código de producción vía
    `plan_result`/`plan`/`result` con ≥2 references debe estar en
    el bloque CONTRATO. Las que aparecen una sola vez se permiten
    (write-only / read-only sin contraparte — P3-AUDIT-7 las marca
    como underused pero no las exige en el CONTRATO).
    """
    src = _read_src()
    block = _extract_contract_block(src)
    documented = _extract_documented_keys(block)

    # Buscar fuera del bloque CONTRATO.
    src_outside = src.replace(block, "")
    used = _extract_used_keys(src_outside)

    # Keys con ≥2 references son las que tienen productor+consumer
    # detectados — esas son las que el CONTRATO debe enumerar.
    missing = []
    for key, count in sorted(used.items()):
        if key in _WHITELIST:
            continue
        if count < 2:
            continue  # tolera underused
        if key not in documented:
            missing.append((key, count))

    if missing:
        pytest.fail(
            "Las siguientes keys `_xxx` viajan en `plan_result`/`plan`/`result` "
            "con productor+consumer detectados pero NO están en el bloque "
            "CONTRATO de graph_orchestrator.py:\n  - "
            + "\n  - ".join(f"{k} ({n} ref)" for k, n in missing)
            + "\n\nProcedimiento:\n"
            "  1. Si la key es un campo state-level legítimo, declárala en "
            "     `class PlanState` (TypedDict) y migra los call sites.\n"
            "  2. Si vive solo en `plan_result`, añádela al bloque CONTRATO "
            "     en graph_orchestrator.py:~2380 — el contrato documenta "
            "     que no se debe migrar a state sin pasar por revisión.\n"
            "  3. Si es legacy y se va a remover, deja un comment + add a "
            "     `_WHITELIST` en este test."
        )


def test_contract_minimum_keys_post_p1_new_1():
    """Post P1-NEW-1, el CONTRATO enumera ≥20 keys (6 originales +
    14+ añadidas en P1-NEW-1). Si baja el conteo, alguien removió
    keys sin pasar por aquí."""
    src = _read_src()
    block = _extract_contract_block(src)
    documented = _extract_documented_keys(block)
    assert len(documented) >= 20, (
        f"P1-NEW-1 regresión: bloque CONTRATO ahora enumera "
        f"{len(documented)} keys (esperado ≥20 post P1-NEW-1). "
        f"Si removiste keys intencionalmente y aplicaste migración a "
        f"state, baja el floor con un comentario explicativo."
    )


def test_known_p1_new_1_keys_documented():
    """Sanity: las keys que P1-NEW-1 añadió específicamente al
    CONTRATO deben seguir documentadas. Si las quitas, hazlo
    deliberadamente."""
    src = _read_src()
    block = _extract_contract_block(src)
    documented = _extract_documented_keys(block)

    p1_new_1_anchors = {
        "_best_attempt_plan",
        "_review_disclaimer",
        "_review_severity",
        "_schema_errors",
        "_skeleton",
        "_profile_embedding",
        "_selected_techniques",
    }
    missing = p1_new_1_anchors - documented
    assert not missing, (
        f"P1-NEW-1 regresión: las siguientes keys ancla del P1-NEW-1 "
        f"fueron removidas del bloque CONTRATO: {sorted(missing)}. "
        f"Si la remoción es intencional, actualiza este whitelist."
    )
