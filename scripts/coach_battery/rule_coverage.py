"""[P1-PLAN-LOTE-1004 · 2026-11-14] (G-15) Matriz automática de reglas del prompt vs tests y batería.

Extrae todos los identificadores de regla emitidos en tiempo de ejecución por
``prompts/chat_agent.py`` (``[P*-*]``, ``[BB-*]``, ``[AI-*]``), los cruza contra:
1. Tests offline en ``backend/tests/test_*.py`` (mención literal o nombre normalizado).
2. Casos de la batería en ``backend/scripts/coach_battery/battery.json`` (``RULE_TO_BATTERY_CASES``
   + referencias explícitas en notas).

También expone ``comparar_reglas_prompt(antes, despues)`` para asegurar 0 pérdida
silenciosa de reglas al comprimir o editar ``prompts/chat_agent.py`` (usado por G-03/G-04).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
BATTERY_JSON_PATH = Path(__file__).resolve().parent / "battery.json"

_BRACKET_RE = re.compile(r"\[([^\[\]\n]{3,95})\]")
_RULE_ID_RE = re.compile(r"\b((?:P\d+-[A-Za-z0-9-]+|BB-\d+[a-z]?|AI-[A-Za-z0-9-]+))\b")

# Mapeo explícito regla emitida -> casos en battery.json que ejercitan esa regla en vivo.
RULE_TO_BATTERY_CASES: dict[str, list[str]] = {
    "BB-025": ["J1", "J5", "M1"],
    "P1-DIARY-COMPLEMENTS-SECTION": ["J1", "J5", "M1"],
    "BB-027": ["M1", "J5"],
    "P1-PROMOTED-COMPLEMENT-GUARD": ["M1", "J5"],
    "BB-028": ["P1", "P2"],
    "P1-PROPORTIONAL-COOKING-FAT-SCALE": ["P1", "P2"],
    "BB-029": ["R1", "R2", "P7"],
    "P1-MICRO-PROTEIN-RESCUE-CRON": ["R1", "R2"],
    "BB-030": ["P1", "M1", "J4"],
    "P1-BEVERAGE-DOUBTS-NEUTRAL-HIERARCHY": ["P1", "M1", "J4"],
    "BB-031": ["B2", "M1", "B3"],
    "P1-CORRECT-MEAL-TARGET-DISAMBIGUATION": ["B2", "M1", "B3"],
    "BB-032": ["B1", "A1"],
    "P1-BRUNCH-SATIETY-CLAIM-SUPPRESSION": ["B1", "A1"],
    "BB-033": ["B8", "J5"],
    "P1-NIGHT-SNACK-SEMANTIC-LABEL": ["B8", "J5"],
    "BB-034": ["B6", "J5"],
    "P1-COMPARATIVE-YESTERDAY-TEMPLATE": ["B6", "J5"],
    "BB-047": ["A1", "F7"],
    "P0-NO-TECHNICAL-ERROR-LEAK": ["A1", "F7", "B2"],
    "BB-051": ["J5", "R1"],
    "P1-CATCHUP-NOCTURNO-CALIDO": ["J5", "R1"],
    "BB-055": ["B7", "H1"],
    "P0-NO-HYDRATION-TECHNICAL-LEAK": ["B7", "H1"],
    "BB-058": ["J6", "R1", "R8"],
    "P0-ISOLATED-FOOD-INTENT-DISAMBIGUATION": ["J6", "R1", "R8"],
    "BB-059": ["R5", "R6"],
    "P1-PROPOSAL-INGREDIENT-FIDELITY": ["R5", "R6"],
    "BB-066": ["A1", "J5"],
    "P0-NO-REPEATED-NIGHT-FAREWELLS": ["A1", "J5"],
    "BB-067": ["C1", "C6"],
    "P1-DIRECT-SLOT-NAME-QUERY": ["C1", "C6"],
    "BB-071": ["B1", "M1", "P2"],
    "P1-PLAN-LOTE-1059": ["W1", "W2", "W3", "W4"],   # regla R5: progreso solo del bloque 🏆
    "P1-CLEAN-MACRO-PREFIX": ["B1", "M1"],
    "P1-COMPOSITE-PLATE-VEHICLE-CONSOLIDATION": ["M1", "B1", "P2"],
    "P0-CHAT-ALLERGY-SSOT": ["F1", "F2", "F7"],
    "P1-CHAT-DIARY-CORRECT": ["B2", "M1"],
    "P1-CHAT-DIARY-WHERE": ["B1", "B6", "J5"],
    "P1-CLONE-HABITUAL-DECONTAMINATE": ["B1", "B3"],
    "P1-CONSUMED-BACKDATE": ["B6", "B5"],
    "P1-DIARY-DELTA-CORRECTION": ["B2", "M1"],
    "P1-PHOTO-DRINK-IDENTITY": ["M1", "R3", "P1"],
    "P1-PLAN-LOTE-105": ["D2", "D4"],
    "P1-PLAN-LOTE-168": ["J5", "J6"],
    "P1-PLAN-LOTE-226": ["F7", "A1"],
    "P1-PLAN-LOTE-53": ["B1", "C1", "F2"],
    "P1-PLAN-LOTE-76": ["B1", "B2", "J5"],
    "P1-PLAN-LOTE-765": ["P6", "P7", "P8"],
    "P1-PLAN-LOTE-767": ["P6", "P9", "P10"],
    "P1-PLAN-LOTE-77": ["B1", "F1", "P2"],
    "P1-PLAN-LOTE-79": ["B1", "R5"],
    "P1-PLAN-LOTE-80": ["B1", "C1"],
    "P1-PLAN-LOTE-950": ["F7", "A1"],
    "P1-PLAN-LOTE-987": ["B6", "J5"],
}


def collect_emitted_prompt_sources() -> dict[str, str]:
    """Devuelve todos los strings de reglas que ``prompts/chat_agent.py`` puede emitir."""
    from prompts import chat_agent as ca

    return {
        "CHAT_STREAM_INLINE_PROMPT": ca.CHAT_STREAM_INLINE_PROMPT,
        "CHAT_AGENT_INLINE_PROMPT": ca.CHAT_AGENT_INLINE_PROMPT,
        "CHAT_VOICE_MODE_PROMPT": ca.CHAT_VOICE_MODE_PROMPT,
        "build_tools_instructions_estaticas": ca.build_tools_instructions_estaticas(),
        "build_tools_instructions_stream_estaticas": ca.build_tools_instructions_stream_estaticas(),
        "build_tools_instructions": ca.build_tools_instructions(user_id="__probe__"),
        "build_tools_instructions_stream": ca.build_tools_instructions_stream(user_id="__probe__"),
    }


def extract_rule_ids_from_text(text: str) -> list[str]:
    """Extrae identificadores únicos de reglas dentro de corchetes ``[...]`` en un texto."""
    ids: set[str] = set()
    for m in _BRACKET_RE.finditer(text or ""):
        inside = m.group(1)
        for im in _RULE_ID_RE.finditer(inside):
            ids.add(im.group(1))
    return sorted(ids)


def extract_emitted_rules() -> dict[str, list[str]]:
    """Mapa regla_id -> lista de constantes/funciones de ``prompts/chat_agent.py`` donde aparece."""
    by_rule: dict[str, set[str]] = {}
    for src_name, text in collect_emitted_prompt_sources().items():
        for rule_id in extract_rule_ids_from_text(text):
            by_rule.setdefault(rule_id, set()).add(src_name)
    return {rid: sorted(srcs) for rid, srcs in sorted(by_rule.items())}


def comparar_reglas_prompt(antes_text: str, despues_text: str) -> dict[str, Any]:
    """Compara dos versiones del prompt asegurando cero reglas perdidas."""
    antes = set(extract_rule_ids_from_text(antes_text))
    despues = set(extract_rule_ids_from_text(despues_text))
    return {
        "antes_count": len(antes),
        "despues_count": len(despues),
        "perdidas": sorted(antes - despues),
        "nuevas": sorted(despues - antes),
    }


def _find_matching_tests(rule_id: str, test_texts: dict[str, str]) -> list[str]:
    slug = rule_id.lower().replace("-", "_")
    # Soporta P1-PLAN-LOTE-765 -> test_p1_plan_lote_765.py o mención literal en el test.
    short_lote = None
    m = re.match(r"P\d+-PLAN-LOTE-(\d+)$", rule_id)
    if m:
        short_lote = f"lote_{m.group(1)}"
    matches: list[str] = []
    for fname, content in test_texts.items():
        fname_lower = fname.lower()
        if (
            rule_id in content
            or slug in fname_lower
            or (short_lote and short_lote in fname_lower)
        ):
            matches.append(fname)
    return sorted(matches)


def build_coverage_matrix(
    tests_dir: Path | None = None,
    battery_path: Path | None = None,
) -> dict[str, Any]:
    """Construye la matriz regla -> {sources, tests, battery_cases, covered}."""
    tdir = tests_dir or (BACKEND_ROOT / "tests")
    bpath = battery_path or BATTERY_JSON_PATH

    test_texts = {
        p.name: p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(tdir.glob("test_*.py"))
    }
    battery_data = json.loads(bpath.read_text(encoding="utf-8"))
    valid_case_ids = {c["id"] for c in battery_data.get("casos") or []}

    emitted = extract_emitted_rules()
    rows: list[dict[str, Any]] = []
    uncovered: list[str] = []

    for rule_id, sources in emitted.items():
        tests = _find_matching_tests(rule_id, test_texts)
        mapped_cases = [cid for cid in RULE_TO_BATTERY_CASES.get(rule_id, []) if cid in valid_case_ids]
        # También revisa notas de battery.json por número de lote [168], [765], etc.
        m = re.match(r"P\d+-PLAN-LOTE-(\d+)$", rule_id)
        if m:
            tok = f"[{m.group(1)}]"
            for c in battery_data.get("casos") or []:
                notas = str((c.get("expect") or {}).get("notas") or "")
                if tok in notas and c["id"] not in mapped_cases:
                    mapped_cases.append(c["id"])
        covered = bool(tests or mapped_cases)
        if not covered:
            uncovered.append(rule_id)
        rows.append(
            {
                "rule_id": rule_id,
                "sources": sources,
                "tests": tests,
                "battery_cases": sorted(set(mapped_cases)),
                "covered": covered,
            }
        )

    return {
        "total_rules": len(rows),
        "covered_rules": len(rows) - len(uncovered),
        "uncovered_rules": uncovered,
        "rules": rows,
    }


def main() -> int:
    import sys

    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    ap = argparse.ArgumentParser(description="Matriz de cobertura de reglas del coach (G-15)")
    ap.add_argument("--json", action="store_true", help="Imprime matriz completa en JSON")
    ap.add_argument("--summary", action="store_true", help="Imprime resumen compacto")
    ap.add_argument("--fail-uncovered", action="store_true", help="Devuelve exit 1 si alguna regla no tiene cobertura")
    args = ap.parse_args()

    matrix = build_coverage_matrix()
    if args.json:
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
    else:
        print(
            f"Reglas emitidas: {matrix['covered_rules']}/{matrix['total_rules']} cubiertas; "
            f"sin_cobertura={matrix['uncovered_rules']}"
        )
        if not args.summary:
            for row in matrix["rules"]:
                print(
                    f"- {row['rule_id']}: tests={len(row['tests'])} "
                    f"battery={row['battery_cases']}"
                )
    if args.fail_uncovered and matrix["uncovered_rules"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
