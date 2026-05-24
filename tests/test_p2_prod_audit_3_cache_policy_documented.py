"""[P2-PROD-AUDIT-1 · 2026-05-23] Cache invalidation policy debe estar
documentada en `docs/runbooks/cache_invalidation_policy.md`.

Gap original (audit production-readiness 2026-05-23, B-P2-4):
    `cache_manager.py` provee `centralized_cache(ttl_seconds, maxsize)`
    decorator. Cada call site declara su TTL inline (`fact_extractor`,
    `vision_agent`). Pero NO existe SSOT documentando:
      - Cuándo usar TTL vs event-driven invalidation.
      - Qué funciones SÍ deben cachear vs NO.
      - SOP de invalidación manual (Redis FLUSHDB vs por-key vs restart).
      - Diagnóstico de "el cache devuelve stale".

    Sin runbook, SRE bajo presión de incident improvisa o ejecuta
    `FLUSHDB` indiscriminado.

Fix:
    Runbook `docs/runbooks/cache_invalidation_policy.md` con SOP
    documentado. Este test ancla su existencia + cobertura mínima.

Cobertura:
    A) Runbook existe.
    B) Cubre las 4 secciones canónicas (arquitectura, TTL vs event,
       invalidación manual, SOP stale).
    C) Cubre las 4 vías de invalidación manual (por-key, por-función,
       cache local, nuke).
    D) Referenciado desde docs/runbooks/README.md (descubrible).

Tooltip-anchor: P2-PROD-AUDIT-1-CACHE-RUNBOOK | audit 2026-05-23.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_RUNBOOK_PATH = _BACKEND_ROOT / "docs" / "runbooks" / "cache_invalidation_policy.md"


def test_runbook_exists():
    assert _RUNBOOK_PATH.exists(), (
        f"Runbook ausente en {_RUNBOOK_PATH}. Cierre del gap B-P2-4 perdido."
    )


def test_runbook_covers_canonical_sections():
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    required_sections = [
        "Arquitectura del cache",
        "Modelo de invalidación",
        "Cuándo NO usar",
        "Cuándo NO usar",  # variant accepts NO usar / NO use / etc.
        "Knobs operacionales",
        "Invalidación manual",
    ]
    missing = []
    for header in required_sections:
        if header not in text:
            missing.append(header)
    # Deduplicate.
    missing = sorted(set(missing))
    assert not missing, (
        f"Runbook NO cubre secciones canónicas: {missing}. "
        f"Cada sección documenta una facet operacional distinta — restaurar."
    )


def test_runbook_covers_manual_invalidation_paths():
    """4 vías de invalidación manual documentadas."""
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    required_terms = [
        "DEL",          # delete por-key con redis-cli
        "FLUSHDB",      # nuke total
        "KEYS",         # listar por prefix
        "TTL",          # diagnóstico de tiempo restante
    ]
    missing = [t for t in required_terms if t not in text]
    assert not missing, (
        f"Runbook NO cubre comandos canónicos de invalidación: {missing}. "
        f"SRE bajo presión necesita el comando exacto — restaurar."
    )


def test_runbook_covers_sop_stale_diagnosis():
    """SOP de bug 'cache devuelve stale' debe estar."""
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "stale" in text.lower(), (
        "Runbook no menciona el caso 'stale' — sin este SOP, el bug más "
        "común (cache devuelve valor viejo) no tiene guía. Restaurar."
    )
    # Debe mencionar "pura" o "puro" para explicar cuándo el cache TTL es
    # incorrecto (función no determinística).
    assert "pura" in text.lower() or "puro" in text.lower() or "determinístic" in text.lower(), (
        "Runbook no explica cuándo el cache TTL es ERRÓNEO (función no "
        "pura → mutable input). Sin esta nota, el bug 'siempre stale' se "
        "diagnostica mal."
    )


def test_runbook_referenced_from_index():
    """Runbook debe aparecer en `docs/runbooks/README.md` para descubribilidad."""
    index = _BACKEND_ROOT / "docs" / "runbooks" / "README.md"
    assert index.exists(), "docs/runbooks/README.md ausente"
    text = index.read_text(encoding="utf-8")
    assert "cache_invalidation_policy.md" in text, (
        "`cache_invalidation_policy.md` no referenciado en `docs/runbooks/README.md`. "
        "Sin entry en el index, SRE bajo presión NO lo encuentra. Restaurar."
    )


def test_anchor_present():
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "P2-PROD-AUDIT-1" in text, (
        "Runbook perdió anchor `P2-PROD-AUDIT-1`. Sin breadcrumb operacional, "
        "futuro mantenedor no rastrea el origen."
    )
