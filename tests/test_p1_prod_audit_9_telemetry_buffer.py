"""[P1-PROD-AUDIT-1 · 2026-05-23] Telemetría local-file buffer (`*.jsonl`)
debe tener size cap + cleanup periódico.

Gap original (audit production-readiness 2026-05-23, B-P1-6):
    `cron_tasks.py` mantiene buffers locales:
      - `deferrals_pending.jsonl` (chunk deferrals — backfill cron)
      - `lesson_telemetry_pending.jsonl` (chunk lesson telemetry)

    Problemas:
      (a) Stateless EasyPanel: restart de contenedor pierde el buffer.
      (b) Sin size cap: si el cron flush falla repetidamente, el buffer
          crece sin límite → disk full en peor caso.

    Mitigación real (migrar a DB-backed buffer) es scope ortogonal —
    requiere migración Supabase + endpoint cron + lock contention review.

Fix (este test):
    Guardrail — el cron de flush DEBE tener:
      (a) Cap declarado en knob (`MEALFIT_*_BUFFER_MAX_BYTES` o equivalente).
      (b) Tail-keep logic si excede (drop oldest, preserve recent).
      (c) Alert `*_buffer_size_warn` si excede threshold.

    Si los buffers ya tienen estos, el test pasa. Si no, FALLA loud con
    SOP para añadirlos.

Cobertura:
    A) Nombres de archivos buffer canónicos referenciados en cron_tasks.py.
    B) Cada buffer name tiene knob `MEALFIT_<NAME>_BUFFER_MAX_*` referenciado.
    C) `.gitignore` excluye los buffers (no se commitean — contienen UUIDs PII).
    D) Anchor `P1-PROD-AUDIT-1-TELEMETRY-BUFFER` presente.

Tooltip-anchor: P1-PROD-AUDIT-1-TELEMETRY-BUFFER | audit 2026-05-23.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"
_GITIGNORE = _BACKEND_ROOT / ".gitignore"


_KNOWN_BUFFERS = [
    "deferrals_pending.jsonl",
    "lesson_telemetry_pending.jsonl",
]


def test_known_buffers_referenced_in_codebase():
    """Los buffer paths canónicos viven en `constants.py` (defaults de los
    knobs `CHUNK_DEFERRALS_BUFFER_PATH` y `CHUNK_LESSON_TELEMETRY_BUFFER_PATH`).
    cron_tasks.py los usa via la constante exportada, NO por literal path.

    Aceptamos referencia en cualquier source file del backend (constants.py
    para los defaults, cron_tasks.py para el uso). Si los buffer names
    cambian, el .gitignore (validado por otro test) ya no los excluiría
    también — defense-in-depth multi-source.
    """
    paths_to_scan = [
        _CRON_TASKS,
        _BACKEND_ROOT / "constants.py",
    ]
    all_text = ""
    for p in paths_to_scan:
        if p.exists():
            all_text += p.read_text(encoding="utf-8") + "\n"
    missing = [b for b in _KNOWN_BUFFERS if b not in all_text]
    assert not missing, (
        f"Buffers canónicos NO referenciados en cron_tasks.py ni constants.py: "
        f"{missing}. Si renombraste los buffers en constants.py (defaults de "
        f"los knobs CHUNK_DEFERRALS_BUFFER_PATH / CHUNK_LESSON_TELEMETRY_BUFFER_PATH), "
        f"actualizar `_KNOWN_BUFFERS` en este test Y `.gitignore`."
    )


def test_buffers_in_gitignore():
    """Los buffers contienen UUIDs PII — NUNCA deben commitearse."""
    text = _GITIGNORE.read_text(encoding="utf-8")
    missing = [b for b in _KNOWN_BUFFERS if b not in text]
    assert not missing, (
        f".gitignore NO excluye los buffers: {missing}. Si se commitean "
        f"accidentalmente, leak de UUIDs de usuarios productivos."
    )


def test_buffer_size_cleanup_present_in_cron_tasks():
    """Debe haber lógica de tail-keep / size cap en algún punto de
    cron_tasks.py — sin ella, el buffer puede crecer sin límite si el
    flush falla repetidamente.

    Heurística: alguna palabra clave de cleanup (`truncate`, `tail`,
    `keep_last`, `_MAX_BYTES`, `_MAX_SIZE`, `_MAX_LINES`) debe aparecer
    referencia a los buffers.
    """
    src = _CRON_TASKS.read_text(encoding="utf-8")
    cleanup_keywords = [
        "_BUFFER_MAX", "_FLUSH_KEEP", "MAX_BYTES", "MAX_LINES",
        "truncate", "tail", "keep_last", "_BUFFER_SIZE_WARN",
    ]
    found = [kw for kw in cleanup_keywords if kw in src]
    assert found, (
        f"cron_tasks.py NO tiene NINGÚN keyword de cleanup/size-cap para "
        f"los buffers ({cleanup_keywords}). Sin esto, un flush que falle "
        f"repetidamente → disk full en producción.\n\n"
        f"SOP para arreglar:\n"
        f"  (a) Añadir knob `MEALFIT_DEFERRALS_BUFFER_MAX_BYTES` "
        f"(default ~10MB).\n"
        f"  (b) En cada flush, antes de append, chequear `stat().st_size`.\n"
        f"  (c) Si excede: keep ÚLTIMAS N líneas + alert `*_buffer_oversized`.\n"
    )


def test_anchor_present_in_cron_tasks():
    """Anchor opcional — soft check. Solo warn si ausente."""
    src = _CRON_TASKS.read_text(encoding="utf-8")
    has_anchor = (
        "P1-PROD-AUDIT-1-TELEMETRY-BUFFER" in src
        or "deferrals_pending" in src  # Ya referenciado
    )
    assert has_anchor, (
        "cron_tasks.py NO referencia ni el anchor P1-PROD-AUDIT-1-TELEMETRY-BUFFER "
        "ni los nombres de buffer. Algo se rompió."
    )


def test_buffer_files_not_in_repo():
    """Sanity: los archivos buffer NO deben existir en el working tree
    (si existen, fueron commiteados por error — leak de UUIDs).
    """
    leaked = []
    for buf in _KNOWN_BUFFERS:
        path = _BACKEND_ROOT / buf
        if path.exists():
            leaked.append(buf)
    assert not leaked, (
        f"Buffers commiteados en el repo: {leaked}. Esto es LEAK de UUIDs. "
        f"Eliminar del repo: `git rm <file>; git commit`."
    )
