"""[P1-PROD-AUDIT-1 · 2026-05-23] Monolith size cap — alerta si los
archivos gigantes crecen >threshold sin extracción concurrente.

Gap original (audit production-readiness 2026-05-23, B-P1-1):
    Archivos monolíticos:
      - graph_orchestrator.py ~14.5K líneas
      - cron_tasks.py ~27K líneas
      - shopping_calculator.py ~7.7K líneas
      - constants.py ~2.4K líneas
      - routers/plans.py ~9.7K líneas

    Refactor real (extraer a módulos) = weeks de trabajo + alto riesgo.
    Audit doc gaps-audit-2026-05.md (A2-A3) ya recomienda extracción
    incremental por dominio (`cb.py` extraction first). No es scope de
    un PR de cierre P1.

Fix (este test):
    Guardrail size cap — captura el snapshot actual + alerta si CUALQUIER
    archivo crece >10% sin documentar el bump. Esto:
      (a) Hace visible el crecimiento (operador ve PR que bump el cap →
          señal de que extracción se difirió).
      (b) NO bloquea el crecimiento legítimo — el cap es bumpeable con
          razón explicada en commit.
      (c) Documenta el snapshot baseline para tracking longitudinal.

Cuando un archivo crece y el test falla:
    - Si fue por nuevo dominio que NO encaja con el resto: extender el
      archivo Y bumpear el cap aquí. Documentar en commit por qué no se
      extrajo.
    - Si fue por extension lateral (más helpers del mismo dominio):
      considerar extracción a módulo nuevo (e.g. helpers de retry a
      `retry_helpers.py`). Bumpear cap es señal de deuda.

Snapshot caps (2026-05-23, +10% margen):
    Cada cap = current_lines * 1.10, redondeado.
    Cuando se reorganiza una zona, bumpear hacia abajo el cap (recompute
    desde el nuevo current).

Tooltip-anchor: P1-PROD-AUDIT-1-MONOLITH-CAP | audit 2026-05-23.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


# Snapshot 2026-05-23 + 10% margen. Cuando un archivo crece arriba del
# cap, FALLAR loud — operador decide si:
#   (a) Bumpear el cap (visible en PR, señal de deuda).
#   (b) Extraer subzonas a módulos nuevos (reduce el current).
#   (c) Considerar la línea como deuda explícita y mantener cap.
#
# NO bumpear sin razón documentada en commit msg.
_SIZE_CAPS_LINES = {
    "graph_orchestrator.py": 16000,    # current ~14.5K
    "cron_tasks.py":         30000,    # current ~27K
    "shopping_calculator.py": 8500,    # current ~7.7K
    "constants.py":           2700,    # current ~2.4K
    "routers/plans.py":      11000,    # current ~9.7K
    "agent.py":               2700,    # current ~2.4K
    "app.py":                 2300,    # current ~2.1K
    "ai_helpers.py":          1500,    # current ~1.3K
    "memory_manager.py":      1100,    # current ~1K
    "proactive_agent.py":      950,    # current ~865
    "tools.py":               2300,    # current ~2.1K
}


def _count_lines(path: Path) -> int:
    """Count líneas físicas del archivo (incluye blank + comment).
    Métrica más simple que LOC efectivo + estable cross-formatter.
    """
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


@pytest.mark.parametrize("relpath,cap", list(_SIZE_CAPS_LINES.items()))
def test_file_under_cap(relpath: str, cap: int):
    """Cada archivo monolithic conocido debe estar bajo su cap.

    Si falla:
      1. Mirar el diff. ¿Qué subdominio se añadió?
      2. ¿Es un dominio nuevo que NO encaja? Considerar extracción al
         módulo más cercano O crear `<domain>_helpers.py`.
      3. Si la extensión es legítima y la extracción no es viable en este
         PR: bumpear el cap aquí con razón en el commit msg
         (`refactor: bump cap de X a Y, deuda P1-MONOLITH-Z`).

    No bumpear silenciosamente — cada bump es deuda tracked.
    """
    file_path = _BACKEND_ROOT / relpath
    if not file_path.exists():
        pytest.skip(f"{relpath} no existe — si fue extraído, eliminar entry del dict.")
    lines = _count_lines(file_path)
    assert lines <= cap, (
        f"\n[P1-PROD-AUDIT-1-MONOLITH-CAP] {relpath}: {lines} líneas > cap {cap}.\n\n"
        f"Opciones:\n"
        f"  (a) Extraer subzonas: el archivo ya es grande — bumpearlo más es deuda.\n"
        f"      Ver gaps-audit-2026-05.md A2/A3 para roadmap de extracción.\n"
        f"  (b) Bumpear el cap aquí (`_SIZE_CAPS_LINES['{relpath}'] = {int(lines * 1.10)}`) +\n"
        f"      documentar en commit msg POR QUÉ no se extrajo.\n"
        f"  (c) Si el archivo fue extraído (renombrado), eliminar el entry."
    )


def test_no_new_files_above_cap():
    """Sanity: archivos en raíz del backend NO deben crecer arriba del
    threshold máximo conocido. Si un archivo nuevo aparece con >3K líneas
    (típico cap para archivos no-monolithic), eso es señal temprana de
    nuevo monolito.
    """
    new_threshold = 3000  # líneas
    excluded_dirs = {"tests", "scratch", "supabase", "prompts", "docs", ".git",
                     "__pycache__", ".github", "fixtures"}
    excluded_known_monoliths = set(_SIZE_CAPS_LINES.keys())

    suspicious = []
    for py_file in _BACKEND_ROOT.glob("*.py"):
        if py_file.name in excluded_known_monoliths:
            continue
        lines = _count_lines(py_file)
        if lines > new_threshold:
            suspicious.append((py_file.name, lines))

    # Routers + db modules — check separately.
    for sub_path in ["routers"]:
        sub_dir = _BACKEND_ROOT / sub_path
        if not sub_dir.exists():
            continue
        for py_file in sub_dir.glob("*.py"):
            relpath = f"{sub_path}/{py_file.name}"
            if relpath in excluded_known_monoliths:
                continue
            lines = _count_lines(py_file)
            if lines > new_threshold:
                suspicious.append((relpath, lines))

    if suspicious:
        msg = (
            f"\n[P1-PROD-AUDIT-1-MONOLITH-CAP] Archivos nuevos arriba del "
            f"threshold {new_threshold}L:\n"
            + "\n".join(f"  - {n}: {l} líneas" for n, l in suspicious)
            + "\n\nSi son archivos genuinamente grandes (extracted from monolith), "
            "añadir a `_SIZE_CAPS_LINES` con cap apropiado.\nSi son nuevo monolito "
            "creciendo, refactor temprano (extracción a varios módulos por dominio)."
        )
        pytest.fail(msg)
