"""[P2-NEW-C · 2026-05-11] Test ancla: las invariantes I1..I7 del
lifecycle de `plan_id` DEBEN estar documentadas en `CLAUDE.md` con su
mención al P-fix correspondiente.

Motivación:
    CLAUDE.md es contexto auto-cargado en cada sesión nueva (lo cita el
    sistema en el preámbulo). Las invariantes del lifecycle del plan_id
    (I1..I7) capturan los contratos arquitectónicos críticos:

      - I1: plan_id nace del INSERT backend.
      - I2: toda mutación filtra `AND user_id = %s`.
      - I3: lectura cross-page valida ownership client-side.
      - I4: caches se invalidan post-mutación.
      - I5: alert plan_quality_degraded en degradación.
      - I6: mutaciones a plan_data desde frontend prohibidas.
      - I7: full-overwrite plan_data requiere advisory lock.

    Si una invariante se elimina por accidente del doc (refactor de
    CLAUDE.md, copy-paste error), los tests que enforzan cada invariante
    siguen pasando pero el CONTEXTO documental se pierde — futuros
    refactors pueden violarla sin entender por qué los tests fallan.

    Este test ancla la documentación: si I1..I7 desaparecen del doc, CI
    falla y el operador re-añade el bloque antes de mergear.

Tests:
    1. Cada invariante I1..I7 aparece en la tabla del lifecycle.
    2. Cada invariante referencia al menos un P-fix (Pn-X · YYYY-MM-DD
       o el nombre de un endpoint conocido).
    3. Las invariantes nuevas P0-NEW-A/B (I6 wiring) y P1-NEW-B/C
       (I7 wiring) están explícitamente mencionadas en su row.
    4. Cross-link slug del marker.

Tooltip-anchor: P2-NEW-C-START | gap P2 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_CLAUDE_MD = Path(__file__).resolve().parent.parent.parent / "CLAUDE.md"


@pytest.fixture(scope="module")
def claude_md_text() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lifecycle_table(claude_md_text: str) -> str:
    """Extrae el bloque de la tabla 'Invariantes del lifecycle' (markdown
    table inicial después del header `### Invariantes del lifecycle`).
    Stop ante el siguiente `###`/`---`/section header.
    """
    marker = "### Invariantes del lifecycle"
    idx = claude_md_text.find(marker)
    assert idx >= 0, (
        "P2-NEW-C regresión: el header `### Invariantes del lifecycle` "
        "no existe en CLAUDE.md. Si el doc se reorganizó, actualizar el "
        "marker en este test."
    )
    # Tomar desde el marker hasta el próximo `###` o `---` que abra la
    # próxima sección. Cap defensivo: 5000 chars.
    rest = claude_md_text[idx : idx + 5000]
    # Cortar en el próximo `### ` (header del mismo nivel) o `---` (sep).
    cut = re.search(r"\n###\s|\n---\n", rest[len(marker):])
    if cut:
        return rest[: len(marker) + cut.start()]
    return rest


# ---------------------------------------------------------------------------
# 1. Cada invariante I1..I7 está presente
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("inv_id", ["I1", "I2", "I3", "I4", "I5", "I6", "I7"])
def test_invariant_present_in_table(lifecycle_table: str, inv_id: str):
    """Cada `| I<n> |` aparece como una fila de la tabla."""
    pattern = re.compile(rf"\|\s*{inv_id}\s*\|")
    assert pattern.search(lifecycle_table), (
        f"P2-NEW-C regresión: la invariante `{inv_id}` no aparece como "
        f"fila de la tabla en `### Invariantes del lifecycle`. Si fue "
        f"eliminada por un refactor de CLAUDE.md, restaurar la fila con "
        f"su descripción + defensa (P-fix asociado). El test que enforza "
        f"`{inv_id}` puede seguir pasando, pero el contexto documental "
        f"se pierde — futuros refactors lo violarán sin entender por qué."
    )


# ---------------------------------------------------------------------------
# 2. Cada invariante referencia al menos un P-fix o callsite documentado
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "inv_id,required_token",
    [
        # I1: nace del INSERT en services.py.
        ("I1", "services.py"),
        # I2: helper update_meal_plan_data y endpoints conocidos.
        ("I2", "P1-NEW-3"),
        # I3: restorePlan client-side.
        ("I3", "P1-NEW-4"),
        # I4: invalidación de caches.
        ("I4", "P2-NEW-1"),
        # I5: alert plan_quality_degraded.
        ("I5", "plan_quality_degraded"),
        # I6: P0-NEW-A swap-meal/persist.
        ("I6", "P0-NEW-A"),
        # I7: P1-NEW-B advisory lock test.
        ("I7", "P1-NEW-B"),
    ],
)
def test_invariant_references_canonical_token(lifecycle_table: str, inv_id: str, required_token: str):
    """Cada invariante DEBE mencionar al menos su token canónico (P-fix
    o callsite documentado en la memoria del repo). Sin estas referencias,
    la fila se degrada a "documentación sin trazabilidad" — el operador
    no puede llegar al código que la enforce.
    """
    # Extraer SOLO la fila de la invariante (entre `| I<n> |` y la siguiente
    # línea que arranque con `|`, o el final del bloque).
    row_pattern = re.compile(
        rf"\|\s*{inv_id}\s*\|.*?(?=\n\|\s*I\d|\n###|\Z)",
        re.DOTALL,
    )
    row_match = row_pattern.search(lifecycle_table)
    assert row_match, (
        f"P2-NEW-C sanity: fila de `{inv_id}` no extraíble del bloque."
    )
    row = row_match.group(0)
    assert required_token in row, (
        f"P2-NEW-C regresión: la fila de `{inv_id}` en CLAUDE.md NO "
        f"menciona el token canónico `{required_token}`. Este token "
        f"ancla la invariante al P-fix o callsite que la enforce. Sin "
        f"él, el doc pierde trazabilidad. Restaurar la mención en la "
        f"columna 'Defensa' de la tabla.\n\nRow actual:\n{row.strip()[:400]}"
    )


# ---------------------------------------------------------------------------
# 3. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p2_new_c"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p2_new_c`)."
    )
