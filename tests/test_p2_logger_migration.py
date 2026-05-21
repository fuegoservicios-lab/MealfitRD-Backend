"""[P2-LOGGER-MIGRATION · 2026-05-12] Anchor + regression guard.

Production hot-path backend files NO deben usar `print(...)` para logging
operacional. `print()` escapa del LogRecord pipeline:
  - No respeta `LOG_LEVEL` env var (filtrar WARNING+ en producción no
    funciona — todos los prints salen siempre).
  - Output mezclado con stdout/stderr del scheduler sin formato del
    logging framework (timestamp + level + module name).
  - Difícil de filtrar/parsear post-hoc por agregadores (Sentry breadcrumbs,
    log aggregators tipo Loki).

Defensas que el test enforza:
  1. Anchor `P2-LOGGER-MIGRATION` presente en `graph_orchestrator.py`
     (hot path principal, 250 conversiones).
  2. Blanket scan: ningún archivo en `PRODUCTION_PATHS` debe contener
     `print(...)` excepto por whitelist explícita.
  3. Whitelist `KNOWN_PRINT_EXEMPT_PATHS` para paths no-producción
     (scripts CLI, scratch, tests, refactors one-shot).
  4. Marker por proximidad `# [P2-LOGGER-EXEMPT: <razón>]` permite
     excepciones inline en archivos productivos cuando hay razón
     concreta documentada (ej: print explícito a stdout para CLI subcommand).

Test parser-based — no levanta el server, solo escanea source con AST.
"""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"


# Production hot-path files cubiertos por la migración 2026-05-12.
# Cualquier archivo de producción nuevo debería añadirse acá explícitamente
# (en lugar de auto-scan de `backend/*.py` que captaría también one-shots).
PRODUCTION_PATHS = [
    "backend/graph_orchestrator.py",
    "backend/fact_extractor.py",
    "backend/memory_manager.py",
    "backend/vision_agent.py",
    "backend/nutrition_calculator.py",
    "backend/db_facts.py",
    "backend/app.py",
]

# Paths NO-PRODUCTION donde print() es legítimo (CLI tools, tests, scripts
# de migración one-shot, scratch). Whitelist explícita > exclusion por glob
# para que un futuro archivo nuevo no escape por casualidad.
#
# [P3-DEBUG-TIME-CLEANUP · 2026-05-20] Vaciado tras mover `refactor.py`,
# `refactor_plans.py`, `modify_cron.py`, `recalc_now.py` a `backend/scratch/
# legacy_root_helpers/` + delete de `test.py` (UUID PII hardcoded). El audit
# `docs/gaps-audit-2026-05.md` A1 los flageó como deuda obvia. La whitelist
# queda vacía pero conserva el set + comentario para que un futuro archivo
# one-shot tenga punto de aterrizaje obvio. Si reaparecen en el árbol,
# preferir marker `# [P2-LOGGER-EXEMPT: ...]` inline en lugar de re-añadir
# entries acá.
KNOWN_PRINT_EXEMPT_PATHS: set[str] = set()


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def test_anchor_present_in_graph_orchestrator():
    src = _read("backend/graph_orchestrator.py")
    assert "P2-LOGGER-MIGRATION" in src, (
        "Falta anchor `P2-LOGGER-MIGRATION` en backend/graph_orchestrator.py. "
        "Sin anchor un futuro reader que vea logger.warning(...) en lugar "
        "de print(...) no sabrá la convención que cierra (production-grade "
        "logging via framework, NO stdout direct)."
    )


def _find_print_calls_outside_whitelist(rel: str) -> list[tuple[int, str]]:
    """Devuelve lista de `(lineno, source_line)` para cada `print(...)` que
    NO tenga marker `[P2-LOGGER-EXEMPT: ...]` en las 3 líneas previas.
    """
    src = _read(rel)
    tree = ast.parse(src)
    lines = src.splitlines()
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "print":
            continue
        ln = node.lineno
        # Look back 3 lines for exempt marker
        window_start = max(1, ln - 3)
        window = "\n".join(lines[window_start - 1 : ln])
        if "[P2-LOGGER-EXEMPT:" in window:
            continue
        violations.append((ln, lines[ln - 1] if ln <= len(lines) else ""))

    return violations


def test_no_print_in_production_paths():
    """Blanket: ninguno de los archivos de PRODUCTION_PATHS debe contener
    `print(...)` sin whitelist por proximidad."""
    all_violations: list[str] = []
    for path in PRODUCTION_PATHS:
        vs = _find_print_calls_outside_whitelist(path)
        for ln, line in vs:
            all_violations.append(f"{path}:{ln}  {line.strip()[:120]}")

    assert not all_violations, (
        f"Encontrados {len(all_violations)} `print(...)` en archivos productivos "
        f"sin marker `# [P2-LOGGER-EXEMPT: <razón>]`. "
        f"Convertir a `logger.<level>(...)` o añadir marker con razón concreta. "
        f"Primeras 5: " + " | ".join(all_violations[:5])
    )


def test_whitelist_marker_respected():
    """Sanity: el marker `[P2-LOGGER-EXEMPT: ...]` en las 3 líneas previas
    debe permitir el `print()`. Verificamos con un módulo en memoria."""
    import tempfile, os
    src = (
        "# foo\n"
        "# [P2-LOGGER-EXEMPT: CLI stdout for subcommand --print-version]\n"
        "print('mealfitrd 1.0.0')\n"
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    try:
        tmp.write(src)
        tmp.close()
        rel = os.path.relpath(tmp.name, _REPO_ROOT)
        # Inline check — call the same logic as _find_print_calls_outside_whitelist
        tree = ast.parse(src)
        lines = src.splitlines()
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                ln = node.lineno
                window = "\n".join(lines[max(1, ln - 3) - 1 : ln])
                if "[P2-LOGGER-EXEMPT:" not in window:
                    violations.append(ln)
        assert not violations, (
            "El marker `[P2-LOGGER-EXEMPT: ...]` no fue respetado. "
            "Sin esto, no hay manera de documentar excepciones legítimas."
        )
    finally:
        os.unlink(tmp.name)


def test_each_production_file_has_logger_defined():
    """Después de la migración, cada archivo de PRODUCTION_PATHS debe tener
    `logger = logging.getLogger(__name__)` (o equivalente). Sin esto, los
    `logger.<level>(...)` calls fallarían con NameError al evaluar."""
    import re
    pat = re.compile(r"\blogger\s*=\s*logging\.getLogger\b")
    missing: list[str] = []
    for path in PRODUCTION_PATHS:
        src = _read(path)
        if not pat.search(src):
            missing.append(path)
    assert not missing, (
        f"Archivos productivos sin `logger = logging.getLogger(__name__)`: "
        f"{missing}. La migración debió insertarlo automáticamente."
    )


def test_each_production_file_imports_logging():
    """`import logging` debe estar presente (top-level) en cada archivo."""
    missing: list[str] = []
    for path in PRODUCTION_PATHS:
        src = _read(path)
        if "import logging" not in src:
            missing.append(path)
    assert not missing, (
        f"Archivos productivos sin `import logging`: {missing}"
    )


def test_exempt_paths_intact():
    """Sanity: los archivos en KNOWN_PRINT_EXEMPT_PATHS deben SEGUIR
    existiendo en disco. Si alguien renombra/elimina uno, el whitelist
    queda obsoleto silenciosamente — este test fuerza actualizar el set."""
    missing: list[str] = []
    for rel in KNOWN_PRINT_EXEMPT_PATHS:
        if not (_REPO_ROOT / rel).is_file():
            missing.append(rel)
    assert not missing, (
        f"Paths en KNOWN_PRINT_EXEMPT_PATHS no existen en disco: {missing}. "
        f"Actualizar el set en el test si fueron renombrados/eliminados."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14: slug `p2_logger_migration` debe
    matchear este archivo `tests/test_<slug>*.py`."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-LOGGER-MIGRATION" in src
