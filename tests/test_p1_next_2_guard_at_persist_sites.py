"""[P1-NEXT-2 · 2026-05-11] Lock-the-contract: cada surface que escribe
`aggregated_shopping_list*` DEBE invocar
`run_shopping_coherence_guard_and_append_history(...)` (o el guard +
history append inline equivalente) en las ~80 líneas siguientes/previas.

Cierra el gap detectado en el audit 2026-05-11:
    El guard solo disparaba en `assemble_plan_node` (graph_orchestrator,
    LangGraph full-pipeline, planes ≤7d). Estos 3 surfaces construían
    `aggregated_shopping_list*` SIN guard:
      - `_chunk_worker` T2 (cron_tasks.py:~22678) — multi-week plans.
      - `/recalculate-shopping-list` (routers/plans.py:~4006) — Pantry/
        Dashboard mutations.
      - `tools.modify_single_meal` (tools.py:~514) — agent tool.
    Resultado: planes multi-week + recalcs + agent swaps podían shipear
    divergencias recetas↔lista sin retry ni `_shopping_coherence_block_history`
    populated — solo capturados post-hoc por cron diario 04:00 UTC
    sin mutación.

Fix P1-NEXT-2:
    Helper SSOT `run_shopping_coherence_guard_and_append_history` en
    `shopping_calculator.py` que invoca guard + appendea entry al
    history con cap. Los 3 callsites lo invocan en mode='warn' tras
    construir la lista.

Drift detection:
    - Nuevo surface que setee `aggregated_shopping_list` sin llamar al
      helper (o sin inline guard equivalente) → falla.
    - Refactor que mueva el helper call fuera del scope del write →
      falla.
    - El helper renombrado → falla con copy explicativo.

Whitelist:
    Marker inline `# [P1-NEXT-2 WHITELIST: <razón>]` en las 30 líneas
    previas si un futuro callsite necesita escribir aggregated SIN
    guard (e.g., reset a `[]` en fallback path donde no hay días para
    validar). Hoy no hay whitelist necesaria — los 3 callsites
    legítimos llaman al helper.

Tooltip-anchor: P1-NEXT-2-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"
_ROUTERS = _BACKEND / "routers" / "plans.py"
_TOOLS = _BACKEND / "tools.py"
_CALC = _BACKEND / "shopping_calculator.py"
_ORCH = _BACKEND / "graph_orchestrator.py"

# Surface scan targets: (file, label). Cada UPDATE/asignación
# `plan_data['aggregated_shopping_list'] = X` que NO sea inicialización
# vacía debe estar en la misma ventana que un call al helper.
_TARGET_FILES = [
    (_CRON, "cron_tasks.py"),
    (_ROUTERS, "routers/plans.py"),
    (_TOOLS, "tools.py"),
]

# Ventana de lookback/lookahead alrededor del write para buscar el guard.
_WINDOW_LINES = 80

# Helper SSOT canónico.
_HELPER_CALL_RE = re.compile(
    r"run_shopping_coherence_guard_and_append_history\s*\(",
)
# Alias por `as _coh_xxx` import.
_HELPER_ALIAS_IMPORT_RE = re.compile(
    r"from\s+shopping_calculator\s+import\s+"
    r"run_shopping_coherence_guard_and_append_history\s+as\s+(\w+)"
)
# Inline guard call (assemble_plan_node escribe pero NO usa el helper —
# tiene el bloque inline P3-NEW-C original). Aceptamos ambas formas.
_INLINE_GUARD_RE = re.compile(
    r"run_shopping_coherence_guard\s*\(",
)

# Pattern del write: solo asignaciones a la key 'aggregated_shopping_list'
# (sin sufijos _weekly/_biweekly/_monthly que siempre se asignan juntas;
# el 'aggregated_shopping_list' pelado es el delimitador del bloque).
# Detecta tanto dict-style (`plan_data["aggregated_shopping_list"] = ...`)
# como atribución plana ('aggregated_shopping_list': X) en dicts literales.
# [P2-COHERENCE-TABLE · 2026-06-18] (audit fresco P2) Además de la forma Python dict-style, capturamos la
# forma SQL `'{aggregated_shopping_list}'` de `jsonb_set` — antes el regex era ciego a ella y 3 surfaces
# (persist/clear pantry-supplement + recovery GAP-F) escapaban al contrato. El `\}` tras la key excluye los
# sufijos `{..._weekly/_biweekly/_monthly}` automáticamente (no terminan en `aggregated_shopping_list}`).
_WRITE_RE = re.compile(
    r"""['"]aggregated_shopping_list['"]\s*\]?\s*=\s*\w"""       # Python dict-style assignment
    r"""|['"]\{aggregated_shopping_list\}['"]""",                # jsonb_set SQL path (P2-COHERENCE-TABLE)
)

# Whitelist marker inline.
_WHITELIST_RE = re.compile(
    r"#\s*\[P1-NEXT-2\s+WHITELIST\s*:\s*(?P<reason>.+?)\]",
)

# Excluir matches que sean inicialización a `[]` o `None` (fallback paths
# legítimos). El write material que necesita guard es el que asigna una
# lista construida.
_INIT_EMPTY_RE = re.compile(
    r"""['"]aggregated_shopping_list['"]\s*\]?\s*=\s*(\[\s*\]|None|False|"")""",
)


def _read_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} no encontrado.")
    return path.read_text(encoding="utf-8").splitlines()


def _find_write_sites(lines: list[str]) -> list[tuple[int, str]]:
    """Devuelve [(line_no, snippet)] de asignaciones a
    aggregated_shopping_list que NO sean init vacío."""
    sites: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if not _WRITE_RE.search(line):
            continue
        # Solo nos importa la key exacta — descartar sufijos del aggregator.
        if "aggregated_shopping_list_weekly" in line:
            continue
        if "aggregated_shopping_list_biweekly" in line:
            continue
        if "aggregated_shopping_list_monthly" in line:
            continue
        # Descartar inits vacíos (fallback paths).
        if _INIT_EMPTY_RE.search(line):
            continue
        # Descartar el dict literal de _get_extreme_fallback_plan (planes
        # de contingencia matemáticos — no hay recetas que validar). Detectado
        # por la sintaxis `"aggregated_shopping_list": []` dentro de un dict
        # literal (no asignación con `=`).
        if re.search(r"""['"]aggregated_shopping_list['"]\s*:\s*\[""", line):
            continue
        sites.append((idx, stripped))
    return sites


def _has_helper_or_inline_guard_in_window(
    lines: list[str], target_line: int
) -> tuple[bool, str | None]:
    """Busca en `[target-WINDOW, target+WINDOW]` un call al helper
    canónico o al guard inline."""
    start = max(0, target_line - 1 - _WINDOW_LINES)
    end = min(len(lines), target_line + _WINDOW_LINES)
    window = "\n".join(lines[start:end])

    # Whitelist explícita?
    wl = _WHITELIST_RE.search(window)
    if wl:
        return True, f"whitelist: {wl.group('reason').strip()}"

    # Helper SSOT directo?
    if _HELPER_CALL_RE.search(window):
        return True, "helper_call"

    # Helper con alias?
    for m in _HELPER_ALIAS_IMPORT_RE.finditer(window):
        alias = m.group(1)
        alias_re = re.compile(rf"{re.escape(alias)}\s*\(")
        if alias_re.search(window):
            return True, f"helper_alias:{alias}"

    # Inline guard (legacy path como assemble_plan_node)?
    if _INLINE_GUARD_RE.search(window):
        return True, "inline_guard"

    return False, None


# ---------------------------------------------------------------------------
# 1. Contrato principal: cada write a aggregated_shopping_list tiene guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("file_path,label", _TARGET_FILES, ids=[t[1] for t in _TARGET_FILES])
def test_every_write_has_guard_in_window(file_path: Path, label: str):
    """Cada `aggregated_shopping_list = <lista construida>` en
    `cron_tasks.py`, `routers/plans.py`, `tools.py` DEBE tener una
    invocación al helper SSOT (o guard inline equivalente) en una
    ventana de ±80 líneas.
    """
    lines = _read_file(file_path)
    sites = _find_write_sites(lines)

    if not sites:
        pytest.skip(
            f"{label}: 0 writes a aggregated_shopping_list que necesiten "
            "guard. Test no aplica para este archivo."
        )

    offenders: list[str] = []
    for line_no, snippet in sites:
        ok, mechanism = _has_helper_or_inline_guard_in_window(lines, line_no)
        if ok:
            continue
        offenders.append(f"  {label}:{line_no} → {snippet[:140]}")

    assert not offenders, (
        f"P1-NEXT-2 violation: uno o más writes a `aggregated_shopping_list` "
        f"en {label} NO tienen invocación al helper "
        "`run_shopping_coherence_guard_and_append_history(...)` (ni guard "
        "inline equivalente, ni whitelist explícita) en ±80 líneas.\n\n"
        "Sin ese guard, el write puede shipear divergencias recetas↔lista "
        "sin retry ni telemetría. Cron diario 04:00 UTC las detecta tarde "
        "(usuario ya vio la lista divergente).\n\n"
        "Offenders:\n"
        + "\n".join(offenders)
        + "\n\nFix: añadir cerca del write:\n"
        "    from shopping_calculator import run_shopping_coherence_guard_and_append_history\n"
        "    run_shopping_coherence_guard_and_append_history(\n"
        "        plan_data,\n"
        "        multiplier=plan_data.get('calc_household_multiplier') or 1.0,\n"
        "        mode_override='warn',\n"
        "        action_taken='warn_only_<surface>',\n"
        "        plan_id_hint=plan_id,\n"
        "    )\n"
        "\nO añadir whitelist marker:\n"
        "    # [P1-NEXT-2 WHITELIST: razón explícita]"
    )


# ---------------------------------------------------------------------------
# 2. El helper SSOT existe en shopping_calculator.py
# ---------------------------------------------------------------------------
def test_helper_is_defined_in_shopping_calculator():
    """Defensa contra refactor que borre/renombre el helper sin
    actualizar callers."""
    if not _CALC.exists():
        pytest.fail(f"{_CALC} no encontrado.")
    src = _CALC.read_text(encoding="utf-8")
    assert re.search(
        r"^def\s+run_shopping_coherence_guard_and_append_history\s*\(",
        src,
        re.MULTILINE,
    ), (
        "P1-NEXT-2 violation: el helper "
        "`run_shopping_coherence_guard_and_append_history` no está definido "
        "en shopping_calculator.py. Sin él, los 3 callsites (T2, recalc, "
        "agent tool) pierden el SSOT. Restaurar la definición o migrar a "
        "guard inline en cada sitio (peor, drift-prone)."
    )


# ---------------------------------------------------------------------------
# 3. El helper acepta los kwargs esperados (signature contract)
# ---------------------------------------------------------------------------
def test_helper_signature_contract():
    """La signature pública del helper debe incluir los kwargs que los
    3 callsites pasan: multiplier, mode_override, attempt, action_taken,
    plan_id_hint. Si alguno desaparece, los callsites lo verán como
    TypeError en runtime."""
    src = _CALC.read_text(encoding="utf-8")
    # Extraer la firma del helper.
    sig_match = re.search(
        r"def\s+run_shopping_coherence_guard_and_append_history\s*\("
        r"(?P<args>[^)]*)\)",
        src,
        re.DOTALL,
    )
    assert sig_match, "Helper no encontrado para extracción de signature."
    args_blob = sig_match.group("args")
    expected_kwargs = {
        "multiplier", "mode_override", "attempt", "action_taken", "plan_id_hint",
    }
    for kw in expected_kwargs:
        assert kw in args_blob, (
            f"P1-NEXT-2 violation: el helper "
            f"`run_shopping_coherence_guard_and_append_history` perdió el "
            f"kwarg `{kw}` en su signature. Los callsites lo pasan; "
            f"removerlo causa TypeError en runtime. Args actuales: "
            f"{args_blob[:200]}"
        )


# ---------------------------------------------------------------------------
# 4. Cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p1_next_2"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p1_next_2`) para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee cuando "
        "el marker se bumpee a `P1-NEXT-2 · 2026-05-11`."
    )
