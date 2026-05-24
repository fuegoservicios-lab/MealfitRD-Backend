#!/usr/bin/env python3
"""[P2-PROD-AUDIT-1 · 2026-05-23] Helper diagnóstico que diffs la tabla
canónica de `alert_keys` (en `docs/system_alerts_resolution_table.md`)
contra las emisiones reales en código.

Gap original (audit production-readiness 2026-05-23, B-P2-8):
    El test `test_p2_audit_4_alert_keys_documented.py` enforza la sync
    bidireccional doc ↔ código, pero cuando FALLA solo dice "missing N
    keys". El operador entonces debe:
      (a) Re-leer la tabla en el doc.
      (b) Grep `alert_key = ` en los 6 source files.
      (c) Cross-reference manual para descubrir qué row añadir/quitar.

    Costoso bajo presión de PR.

Fix:
    Este script reusa la lógica del test parser-based y produce un diff
    legible con:
      - alert_keys EMITIDOS en código pero AUSENTES de la tabla.
      - alert_keys EN TABLA pero SIN producer en código (orphan rows).
      - Snippet del callsite por cada emit para localizar el archivo.

Uso:
    ./scripts/check_alert_keys_sync.py
    ./scripts/check_alert_keys_sync.py --json    # output machine-readable

Salida humana:
    📋 ALERT KEYS — DIFF DOC ↔ CÓDIGO
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ❌ MISSING en docs/system_alerts_resolution_table.md (3):
       - scheduler_missed_<job_id>
         emit: cron_tasks.py:1234
       - chunk_lag_excessive
         emit: cron_tasks.py:5678
       - dead_lettered_chunk:<plan_id>:<week>
         emit: cron_tasks.py:9012

    ⚠️  ORPHAN rows en docs (sin productor):
       - feature_flag_X_disabled

    ✓  Sync OK: 28/31 keys.

SOP cuando hay missing:
    1. Por cada missing, añadir row en `docs/system_alerts_resolution_table.md`
       con productor / resolver / modelo (4 modelos canónicos: Auto explicit,
       Auto implicit, Handler-driven, Manual).
    2. Si es Manual, añadir SOP en `runbook_system_alerts_sops_*.md`.
    3. Re-ejecutar el script — debería decir "Sync OK".

Tooltip-anchor: P2-PROD-AUDIT-1-ALERT-KEYS-DIFF | audit 2026-05-23.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TABLE_PATH = _BACKEND_ROOT / "docs" / "system_alerts_resolution_table.md"

# Source files que emiten `alert_key = "..."`. Match contra
# `_p2_audit_4_alert_keys_documented.py` para mantener sync.
_EMITTER_FILES = [
    _BACKEND_ROOT / "cron_tasks.py",
    _BACKEND_ROOT / "db_inventory.py",
    _BACKEND_ROOT / "memory_manager.py",
    _BACKEND_ROOT / "app.py",
    _BACKEND_ROOT / "graph_orchestrator.py",
    _BACKEND_ROOT / "routers" / "billing.py",
]

_ALERT_KEY_ASSIGN = re.compile(
    r"""alert_key\s*=\s*(['"]|f['"])([^'"]+)['"]""",
    re.MULTILINE,
)


def _normalize_pattern(raw: str) -> str:
    """Normaliza f-string placeholders `{user_id}` → `<user_id>` para
    matchear el formato de la tabla canónica.
    """
    return re.sub(r"\{([^}]+)\}", r"<\1>", raw)


def _extract_from_code() -> dict[str, list[str]]:
    """Returns {alert_key_pattern: [callsite1, callsite2, ...]}."""
    out: dict[str, list[str]] = {}
    for f in _EMITTER_FILES:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.split("\n"), 1):
            m = _ALERT_KEY_ASSIGN.search(line)
            if m:
                raw = m.group(2)
                pattern = _normalize_pattern(raw)
                out.setdefault(pattern, []).append(f"{f.name}:{line_no}")
    return out


def _extract_from_doc() -> set[str]:
    """Returns set of alert_key patterns documentados en la tabla.
    Heurística: cada row tiene primer campo backtick-wrapped."""
    if not _TABLE_PATH.exists():
        sys.exit(f"❌ Doc {_TABLE_PATH} no existe — gap B-P2-8 reabierto")
    text = _TABLE_PATH.read_text(encoding="utf-8")
    keys = set()
    # Match rows tipo: | `alert_key_pattern` | ... | ... | ... |
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "alert_key" in line.lower() and "pattern" in line.lower():
            # Header row.
            continue
        if line.startswith("|---") or line.startswith("|:--"):
            continue
        # Extract first backtick-wrapped value en la primera celda.
        m = re.search(r"\|\s*`([^`]+)`", line)
        if m:
            keys.add(m.group(1).strip())
    return keys


def _diff(code: dict[str, list[str]], doc: set[str]):
    code_set = set(code.keys())
    missing = code_set - doc
    orphans = doc - code_set
    in_sync = code_set & doc
    return {
        "missing": sorted(missing),
        "orphans": sorted(orphans),
        "in_sync_count": len(in_sync),
        "code_count": len(code_set),
        "doc_count": len(doc),
        "emit_sites": code,
    }


def _print_human(report: dict) -> None:
    print()
    print("📋 ALERT KEYS — DIFF DOC ↔ CÓDIGO")
    print("━" * 60)
    print(f"  code emits : {report['code_count']:3d} unique patterns")
    print(f"  doc rows   : {report['doc_count']:3d} unique patterns")
    print(f"  in sync    : {report['in_sync_count']:3d}")
    print()
    if report["missing"]:
        print(f"❌ MISSING en docs/system_alerts_resolution_table.md ({len(report['missing'])}):")
        for k in report["missing"]:
            print(f"   - `{k}`")
            for site in report["emit_sites"].get(k, []):
                print(f"       emit: {site}")
        print()
    if report["orphans"]:
        print(f"⚠️  ORPHAN rows en docs (sin productor en código) ({len(report['orphans'])}):")
        for k in report["orphans"]:
            print(f"   - `{k}`")
        print()
    if not report["missing"] and not report["orphans"]:
        print("✅ Sync perfecto. Test `test_p2_audit_4_alert_keys_documented` debería pasar.")
        print()
    else:
        print("SOP:")
        print("  1. Por cada MISSING, añadir row a docs/system_alerts_resolution_table.md")
        print("     con columnas: alert_key | Productor | Resolver | Modelo.")
        print("  2. Por cada ORPHAN, validar:")
        print("     (a) Si el productor fue eliminado, quitar la row.")
        print("     (b) Si el productor está en un file NO escaneado (ver _EMITTER_FILES),")
        print("         añadir el file al script + al test P2-AUDIT-4.")
        print("  3. Re-ejecutar este script — debería decir Sync perfecto.")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff alert_keys código vs doc")
    parser.add_argument("--json", action="store_true", help="Output JSON machine-readable")
    args = parser.parse_args()

    code_keys = _extract_from_code()
    doc_keys = _extract_from_doc()
    report = _diff(code_keys, doc_keys)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)

    # Exit code != 0 si hay drift (útil para CI).
    return 1 if (report["missing"] or report["orphans"]) else 0


if __name__ == "__main__":
    sys.exit(main())
