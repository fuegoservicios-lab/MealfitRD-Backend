"""[P0-HIST-LEARN-3 · 2026-05-09] Drift detection cross-language entre
el DB enum `plan_chunk_queue.status` y el map JS
`frontend/src/utils/chunkStatus.js`.

Bug original (audit Historial 2026-05-09 · gap P0):
    El check constraint del DB (P1-AUDIT-HIST-3) permite 7 valores
    canónicos para `plan_chunk_queue.status`:
      pending, processing, stale, failed, pending_user_action,
      completed, cancelled.

    El map JS solo declaraba 6 — `cancelled` ausente. Cuando un user
    abre el modal del Historial de un plan que tuvo restore (los
    chunks vivos del source quedan con `status='cancelled'` y
    `dead_letter_reason='restore_source_archived'` — routers/plans.py:
    4133+, 4175+), el chip del tab "Métricas" mostraba el snake_case
    crudo `cancelled` sin label es-DO ni severity color, asimétrico
    con los otros 6 estados que ya estaban humanizados desde
    P0-HIST-FIX-5.

Fix:
    Añadir `cancelled: 'Cancelado'` (label) + `cancelled: 'neutral'`
    (severity) al map JS. Severity neutral porque el chunk fue
    invalidado por decisión administrativa, NO por failure del
    pipeline — palette no debe escalar a warn/bad.

Drift detection (este test):
    Parsea el JS source y exige que las KEYS de ambos maps
    (LABELS + SEVERITY) sean exactamente `_CANONICAL_STATES`.
    Si:
      - El DB enum gana un estado nuevo (e.g. 'paused') sin actualizar
        el map JS → este test falla loud (key faltante).
      - Alguien retira un estado del map JS → falla loud (extra en DB
        no presente en JS).
      - Alguien añade un estado al map JS que NO está en el DB enum
        → falla loud (drift inverso).
    Mismo patrón que P0-FORM-6 / P1-FORM-14 / chunkKinds parity.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CHUNK_STATUS_JS = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "utils" / "chunkStatus.js"
)

# SSOT canónico — duplica intencionalmente el set de
# `test_p1_audit_hist_3_status_check_constraint.py:_CANONICAL_STATES`.
# Mantener ambos en sync es lo que protege el contrato cross-test:
# si un humano cambia uno solo, el otro test del par sigue verde y
# nadie se entera del drift hasta que un row con status nuevo aparece
# en producción. Este comentario + el cross-link en chunkStatus.js
# documentan el patrón.
_CANONICAL_STATES = frozenset({
    "pending",
    "processing",
    "stale",
    "failed",
    "pending_user_action",
    "completed",
    "cancelled",
})


def _js_source() -> str:
    return _CHUNK_STATUS_JS.read_text(encoding="utf-8")


def _extract_keys(js_src: str, const_name: str) -> set[str]:
    """Extrae las keys (top-level) del object literal asignado a `const_name`.
    Tolerante a comentarios `//` entre keys (los hay en CHUNK_STATUS_SEVERITY).
    Acepta keys con o sin quotes (la convención del proyecto es sin quotes
    para identifiers válidos)."""
    # Match: `const NAME = { ... };` capturando solo el body del object.
    # Greedy hasta el `};` final del const — los object literals del file
    # son top-level, no anidados.
    pat = re.compile(
        r"const\s+" + re.escape(const_name) + r"\s*=\s*\{([\s\S]*?)\};",
        re.MULTILINE,
    )
    m = pat.search(js_src)
    assert m, f"No se encontró const {const_name} en chunkStatus.js"
    body = m.group(1)
    # Strip line comments para no confundir el extractor de keys (un
    # comentario que mencione `pending:` rompería el parser ingenuo).
    body_no_comments = re.sub(r"//[^\n]*", "", body)
    # Cada entry: `key: 'value',` con key opcionalmente en quotes.
    keys = set(re.findall(
        r"^\s*['\"]?([a-z_][a-z_0-9]*)['\"]?\s*:",
        body_no_comments,
        re.MULTILINE,
    ))
    return keys


# ---------------------------------------------------------------------------
# 1. Anchor + presencia del fix
# ---------------------------------------------------------------------------
def test_chunk_status_js_exists():
    assert _CHUNK_STATUS_JS.exists(), (
        f"No se encontró chunkStatus.js en {_CHUNK_STATUS_JS}. "
        "El test cross-language requiere el archivo SSOT del frontend."
    )


def test_marker_present():
    assert "P0-HIST-LEARN-3" in _js_source(), (
        "El marker P0-HIST-LEARN-3 debe estar presente en chunkStatus.js "
        "para trazabilidad cross-archivo (memoria, app.py:_LAST_KNOWN_PFIX)."
    )


def test_cancelled_label_present():
    """El fix mínimo: `cancelled: 'Cancelado'` en CHUNK_STATUS_LABELS."""
    src = _js_source()
    assert re.search(
        r"cancelled\s*:\s*['\"]Cancelado['\"]",
        src,
    ), (
        "CHUNK_STATUS_LABELS debe declarar `cancelled: 'Cancelado'`. "
        "Sin esto, el chip del tab Métricas muestra el snake_case crudo "
        "para chunks de planes restaurados."
    )


def test_cancelled_severity_present_and_neutral():
    """Severity = 'neutral' (NO warn/bad) — el chunk fue invalidado por
    restore/cleanup, NO por failure del pipeline."""
    src = _js_source()
    # Match dentro de CHUNK_STATUS_SEVERITY específicamente.
    severity_keys = _extract_keys(src, "CHUNK_STATUS_SEVERITY")
    assert "cancelled" in severity_keys, (
        "CHUNK_STATUS_SEVERITY debe incluir `cancelled` — sin esto, "
        "getChunkStatusSeverity('cancelled') cae al fallback 'neutral' "
        "implícito pero el contrato explícito en el map evita que un "
        "futuro re-classify del bucket cambie el comportamiento sin tests."
    )
    # El valor exacto debe ser 'neutral' — verificamos vía regex por
    # tolerancia al spacing/comentarios entre key y value.
    assert re.search(
        r"cancelled\s*:\s*['\"]neutral['\"]",
        src,
    ), "Severity de `cancelled` debe ser 'neutral' (NO warn/bad)."


# ---------------------------------------------------------------------------
# 2. Drift detection: keys de ambos maps == _CANONICAL_STATES exacto
# ---------------------------------------------------------------------------
def test_labels_map_keys_match_canonical_states():
    """Si el DB enum gana/pierde un estado, este test falla loud."""
    keys = _extract_keys(_js_source(), "CHUNK_STATUS_LABELS")
    assert keys == _CANONICAL_STATES, (
        f"Drift cross-language entre DB enum y CHUNK_STATUS_LABELS:\n"
        f"  Solo en JS map: {keys - _CANONICAL_STATES}\n"
        f"  Solo en DB enum: {_CANONICAL_STATES - keys}\n"
        f"Sync el map (frontend/src/utils/chunkStatus.js) con el "
        f"CHECK constraint de plan_chunk_queue.status (migración "
        f"p1_audit_hist_3_*.sql)."
    )


def test_severity_map_keys_match_canonical_states():
    """Mismo invariante para el map de severity."""
    keys = _extract_keys(_js_source(), "CHUNK_STATUS_SEVERITY")
    assert keys == _CANONICAL_STATES, (
        f"Drift cross-language entre DB enum y CHUNK_STATUS_SEVERITY:\n"
        f"  Solo en JS map: {keys - _CANONICAL_STATES}\n"
        f"  Solo en DB enum: {_CANONICAL_STATES - keys}\n"
    )


def test_labels_and_severity_keys_are_identical():
    """Ambos maps deben cubrir el MISMO set — un estado con label pero
    sin severity (o viceversa) es bug de mantenimiento."""
    labels = _extract_keys(_js_source(), "CHUNK_STATUS_LABELS")
    severity = _extract_keys(_js_source(), "CHUNK_STATUS_SEVERITY")
    assert labels == severity, (
        f"CHUNK_STATUS_LABELS y CHUNK_STATUS_SEVERITY tienen keys "
        f"distintas:\n  Solo en LABELS: {labels - severity}\n"
        f"  Solo en SEVERITY: {severity - labels}\n"
        "Los dos maps DEBEN cubrir el mismo set de estados."
    )


# ---------------------------------------------------------------------------
# 3. Cross-link al SSOT del backend (test sibling)
# ---------------------------------------------------------------------------
def test_cross_link_to_canonical_states_test():
    """El JS source debe documentar el cross-link al test SSOT — sin esto,
    un futuro mantenedor que añada un estado al DB enum no sabría que
    debe sincronizar también el JS."""
    src = _js_source()
    assert (
        "test_p0_hist_learn_3" in src
        or "_CANONICAL_STATES" in src
        or "p1_audit_hist_3" in src.lower()
    ), (
        "chunkStatus.js debe documentar el SSOT del DB enum (e.g. cita "
        "a la migración P1-AUDIT-HIST-3 o al test sibling). Sin esto, "
        "un futuro mantenedor no sabría que cambios en el JS map "
        "requieren bump del CHECK constraint del DB."
    )
