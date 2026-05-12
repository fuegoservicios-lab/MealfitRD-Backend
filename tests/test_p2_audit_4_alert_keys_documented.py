"""[P2-AUDIT-4 · 2026-05-10] Drift detection: cada `alert_key` emitido en
producción debe estar documentado en la tabla "Política de `system_alerts`
resolution" de `CLAUDE.md`, y viceversa (sin ghost entries).

Bug original (audit 2026-05-10):
  La tabla documentaba 12 alert_keys (algunos con naming drift) mientras
  el código emitía ~26 distintos. Operador SRE consultaba CLAUDE.md como
  source-of-truth y desconocía >50% de las alerts emitidas → trataba alerts
  válidas como bugs o aplicaba el resolver incorrecto.

Diseño:
  1. Parsea la tabla de CLAUDE.md extrayendo el primer campo (backtick-
     wrapped) de cada row → set de patterns documentados.
  2. Escanea `cron_tasks.py`, `db_inventory.py`, `memory_manager.py`,
     `app.py` para todas las asignaciones `alert_key = "..."` y
     `alert_key = f"..."`. Normaliza f-string placeholders (`{...}`) a
     `<>` para matchear el formato de la tabla.
  3. Verifica que cada pattern emitido tenga al menos una row matching.
  4. Verifica que cada row tenga al menos un productor en código.

Mantenimiento:
  - Cuando añadas un nuevo `alert_key` en código, añade row en CLAUDE.md
    ANTES de mergear — el test falla con copy explicativo y SOP.
  - Cuando elimines un productor, elimina la row correspondiente.
  - Si necesitas un alert temporalmente sin row (e.g., feature flag),
    añade el pattern a `_EMITTED_BUT_DRAFT` con comment de la fecha de
    cierre esperada — pero esto debe ser excepcional.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_BACKEND = _REPO_ROOT / "backend"

# Archivos donde viven los emisores de `alert_key`. Si se añade un nuevo
# emisor en otro archivo, agregarlo aquí.
#
# [P3-FINAL-1 · 2026-05-11] Se añade `graph_orchestrator.py` al detectar
# drift: P1-NEW-3 (2026-05-11) puso `_emit_plan_quality_degraded_alert`
# allí pero la lista no se actualizó, marcando la row `plan_quality_degraded`
# como ghost entry falsamente.
_EMITTER_FILES = (
    _BACKEND / "cron_tasks.py",
    _BACKEND / "db_inventory.py",
    _BACKEND / "memory_manager.py",
    _BACKEND / "app.py",
    _BACKEND / "graph_orchestrator.py",
    # [P1-BILLING-FAIL-LOUD · 2026-05-12] Añadido tras introducir
    # `_persist_billing_alert` en billing.py que emite
    # `billing_old_sub_cancel_failed:<>:<>` y `billing_cancel_failed:<>:<>`.
    _BACKEND / "routers" / "billing.py",
)

# Whitelist de patterns emitidos pero sin row (excepción documentada).
# Vacío por default — añadir solo con justificación + fecha de cierre.
_EMITTED_BUT_DRAFT: frozenset[str] = frozenset()

# F-string literales con MÚLTIPLES placeholders dinámicos donde el primero
# resuelve a un conjunto cerrado de strings literales. Para esos casos,
# expandimos el pattern emitido a las concretizaciones que realmente llegan
# a la DB para matchear las rows correspondientes de CLAUDE.md.
#
# Mantenimiento: añade aquí solo cuando una variable de tipo "kind/tipo"
# dentro del f-string toma valores de un set cerrado y CADA valor merece
# su propia row documentada (e.g., distinto resolver por tipo).
_DYNAMIC_EXPANSIONS: dict[str, tuple[str, ...]] = {
    # `app.py:221`: `f"scheduler_{event_type}_{job_id}"` con
    # `event_type ∈ {"missed", "error"}` (definido en las dos ramas previas
    # del listener). Cada tipo tiene su propia row porque la severidad y la
    # interpretación operacional difieren (missed → grace timeout; error →
    # exception en el job).
    "scheduler_<>_<>": ("scheduler_missed_<>", "scheduler_error_<>"),
}


def _normalize_pattern(raw: str) -> str:
    """Normaliza una `alert_key` para comparación.

    - Cualquier f-string placeholder `{...}` → `<>`.
    - Cualquier `<...>` se preserva (matchea el formato de la tabla).
    - Strip backticks y whitespace.
    """
    s = raw.strip().strip("`").strip()
    # Reemplaza `{anything}` por `<>` para matchear el formato de la tabla.
    s = re.sub(r"\{[^}]+\}", "<>", s)
    # Reemplaza `<anything>` por `<>` para normalizar variantes.
    s = re.sub(r"<[^>]+>", "<>", s)
    return s


def _parse_documented_patterns() -> set[str]:
    """Extrae los patterns de la primera columna de la tabla en CLAUDE.md.

    Busca el bloque entre el header "Política de `system_alerts` resolution"
    y el siguiente `## ` header. Cada row de la tabla matchea el regex
    `^| <backtick>pattern<backtick> |` (formato Markdown table).
    """  # noqa: D205  (escape de backtick evitado; ver issue Python 3.12 SyntaxWarning)
    src = _CLAUDE_MD.read_text(encoding="utf-8")

    # Aísla la sección. Empieza en el header y termina en la próxima `##`.
    section_start = src.find("## Política de `system_alerts` resolution")
    assert section_start >= 0, (
        "No se encontró el header '## Política de `system_alerts` resolution' "
        "en CLAUDE.md. ¿Lo renombraste? Actualiza este test."
    )
    section_end = src.find("\n## ", section_start + 1)
    if section_end < 0:
        section_end = len(src)
    section = src[section_start:section_end]

    # Cada row de la tabla empieza con `| \`alert_key_pattern\``.
    # Excluimos el header `| `alert_key\` (pattern) |` por su sufijo `(pattern)`.
    row_re = re.compile(r"^\|\s*`([^`]+)`\s*\|", re.MULTILINE)
    patterns = set()
    for match in row_re.finditer(section):
        raw = match.group(1)
        # Filtrar el header de la tabla.
        if raw.endswith("(pattern)") or raw == "alert_key":
            continue
        # Filtrar el subheader entre rows que contiene tipos canónicos.
        normalized = _normalize_pattern(raw)
        if normalized:
            patterns.add(normalized)
    return patterns


def _parse_emitted_patterns() -> dict[str, list[tuple[Path, int, str]]]:
    """Escanea los archivos emisores y devuelve `{normalized_pattern:
    [(file, line, raw_literal), ...]}`.

    Captura tres patrones de emisión:
      1. `alert_key = "literal"` / `alert_key = f"literal_with_{placeholders}"`
         — variable assignment, captura por línea.
      2. `"alert_key": "literal"` / `"alert_key": f"literal"`
         — dict payload (e.g., `supabase.table("system_alerts").upsert({...})`),
         captura por línea.
      3. `execute_sql_write(SQL_STR, ("literal", ...))` donde SQL_STR contiene
         `INSERT INTO system_alerts`. El `alert_key` es el primer elemento del
         tuple de params. Multi-line: usa DOTALL regex con anchor en la SQL.
    """
    out: dict[str, list[tuple[Path, int, str]]] = {}
    # Patrón 1+2: matches por línea.
    assign_re = re.compile(
        r"""alert_key\s*=\s*f?["']([^"']+)["']""",
    )
    dict_re = re.compile(
        r"""["']alert_key["']\s*:\s*f?["']([^"']+)["']""",
    )
    # Patrón 3: `execute_sql_write(SQL_STR, (literal, ...))` donde SQL_STR
    # contiene `INSERT INTO system_alerts`. El anclaje en `execute_sql_write(`
    # más el `[^"]*?` entre las triple-quotes evita el matching cross-
    # function que ocurría con un `.*?` global (un INSERT en función A
    # se emparejaba con un tuple de UPDATE en función B mucho más abajo).
    exec_re = re.compile(
        r'execute_sql_write\s*\(\s*"""[^"]*?INSERT\s+INTO\s+system_alerts[^"]*?"""\s*,\s*\(\s*[\n\s]*f?["\']([^"\'\n]+)["\']',
        re.DOTALL,
    )

    for fp in _EMITTER_FILES:
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        # Patrón 1+2: line-by-line con filtro de comentarios.
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for m in assign_re.finditer(line):
                norm = _normalize_pattern(m.group(1))
                out.setdefault(norm, []).append((fp.relative_to(_REPO_ROOT), line_no, m.group(1)))
            for m in dict_re.finditer(line):
                norm = _normalize_pattern(m.group(1))
                out.setdefault(norm, []).append((fp.relative_to(_REPO_ROOT), line_no, m.group(1)))
        # Patrón 3: multi-line, escanea text completo. Calcula línea aproximada
        # buscando el offset del match en el texto.
        for m in exec_re.finditer(text):
            raw = m.group(1)
            norm = _normalize_pattern(raw)
            # Línea del literal capturado = offset hasta el comienzo del literal.
            line_no = text.count("\n", 0, m.start(1)) + 1
            out.setdefault(norm, []).append((fp.relative_to(_REPO_ROOT), line_no, raw))
    return out


def test_documented_table_is_non_empty():
    """Sanity: la tabla de CLAUDE.md tiene al menos 10 rows (umbral
    histórico — si baja drásticamente, alguien la rompió)."""
    documented = _parse_documented_patterns()
    assert len(documented) >= 10, (
        f"La tabla 'Política de `system_alerts` resolution' en CLAUDE.md "
        f"tiene solo {len(documented)} rows (umbral histórico ≥10). "
        f"¿Borraste accidentalmente el bloque?"
    )


def test_emitter_files_exist():
    """Sanity: los archivos donde escaneamos `alert_key = ...` existen."""
    missing = [fp for fp in _EMITTER_FILES if not fp.exists()]
    assert not missing, (
        f"Archivos emisores no encontrados: {missing}. ¿Refactor? "
        f"Actualiza `_EMITTER_FILES` en este test."
    )


def test_every_emitted_alert_key_is_documented():
    """**Drift principal**: cada `alert_key` emitido por el código debe
    tener una row en la tabla de CLAUDE.md (con su pattern normalizado).

    Si este test falla, NO hagas `pytest -k` para ignorarlo. Añade la row
    al CLAUDE.md con productor + resolver + modelo siguiendo el procedimiento
    documentado en 'Cómo añadir un nuevo `alert_key`'.
    """
    documented = _parse_documented_patterns()
    emitted = _parse_emitted_patterns()

    # Expandir dynamic-name patterns a sus concretizaciones documentadas.
    expanded_emitted: dict[str, list[tuple[Path, int, str]]] = {}
    for pattern, sites in emitted.items():
        if pattern in _DYNAMIC_EXPANSIONS:
            for concrete in _DYNAMIC_EXPANSIONS[pattern]:
                expanded_emitted.setdefault(concrete, []).extend(sites)
        else:
            expanded_emitted.setdefault(pattern, []).extend(sites)

    undocumented = {
        pattern: sites
        for pattern, sites in expanded_emitted.items()
        if pattern not in documented and pattern not in _EMITTED_BUT_DRAFT
    }
    if undocumented:
        msg_lines = [
            "Los siguientes `alert_key` se emiten en código pero NO tienen row",
            "en la tabla 'Política de `system_alerts` resolution' de CLAUDE.md:",
            "",
        ]
        for pattern, sites in sorted(undocumented.items()):
            first = sites[0]
            msg_lines.append(
                f"  - `{pattern}` (raw='{first[2]}') @ {first[0]}:{first[1]}"
            )
        msg_lines.extend([
            "",
            "Procedimiento:",
            "  1. Identifica el productor exacto (archivo:línea).",
            "  2. Decide el resolver y modelo (Auto explicit/implicit, Handler-driven, Manual).",
            "  3. Añade row a CLAUDE.md con copy mínima: productor, resolver, modelo.",
            "  4. Re-ejecuta este test.",
        ])
        pytest.fail("\n".join(msg_lines))


def test_every_documented_pattern_has_emitter():
    """**Ghost detection**: cada row de la tabla debe tener al menos un
    productor en código. Si un emisor se borra sin actualizar la tabla,
    el operador asume que la alert puede aparecer pero ya no puede.
    """
    documented = _parse_documented_patterns()
    emitted = _parse_emitted_patterns()

    # Aplicar las mismas expansiones que en el test principal.
    expanded_keys: set[str] = set()
    for pattern in emitted:
        if pattern in _DYNAMIC_EXPANSIONS:
            expanded_keys.update(_DYNAMIC_EXPANSIONS[pattern])
        else:
            expanded_keys.add(pattern)

    ghost_rows = sorted(documented - expanded_keys)
    if ghost_rows:
        pytest.fail(
            "Las siguientes rows están documentadas en CLAUDE.md pero NO "
            "tienen productor en código (ghost entries):\n  - "
            + "\n  - ".join(ghost_rows)
            + "\n\nProcedimiento:\n"
            "  1. Verifica si el productor fue removido en un refactor reciente.\n"
            "  2. Si fue intencional, elimina la row de la tabla.\n"
            "  3. Si fue accidental, restaura el productor."
        )
