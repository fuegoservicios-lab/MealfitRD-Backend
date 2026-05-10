"""[P2-HIST-NEW-4 · 2026-05-09] Tests de paridad cross-language entre
los `chunk_kind` que usa el backend al encolar chunks y el catálogo
del frontend `utils/chunkKinds.js`.

Bug original (audit profundo Historial 2026-05-09):
    El badge del tab Métricas mostraba `chunk_kind` snake_case crudo
    ("rolling_refill"). Asimetría con _TIER_LABELS que humaniza.

    Fix 100% client-side (helper `getChunkKindLabel` con map es-DO).
    Este test cierra el cross-link del marker (P2-HIST-AUDIT-14
    requiere `tests/test_p2_hist_new_4*.py`) Y detecta drift cuando
    alguien introduce un chunk_kind nuevo en backend sin actualizar
    el catálogo frontend.

Cobertura backend:
    1. Anchor del marker en History.jsx Y en el helper chunkKinds.js.
    2. Helper frontend cubre los 3 kinds canónicos (initial_plan,
       rolling_refill, catchup).
    3. Drift detection: cualquier `chunk_kind="..."` literal en
       backend código de prod (NO tests) debe estar en el catálogo.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
_HELPER_PATH = _REPO_ROOT / "frontend" / "src" / "utils" / "chunkKinds.js"
_HISTORY_PATH = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


# Archivos del código de prod (no tests) donde aparecen literales
# `chunk_kind="..."` o `chunk_kind='...'`. Tests fixtures usan
# valores arbitrarios que no necesariamente reflejan kinds del
# enum canónico — los excluimos para evitar falsos positivos.
_PROD_FILES = [
    _BACKEND_DIR / "cron_tasks.py",
    _BACKEND_DIR / "routers" / "plans.py",
]


def _extract_helper_codes() -> set[str]:
    """Extrae las keys del map en chunkKinds.js (lo parseamos via
    regex porque importar JS desde Python no es trivial)."""
    text = _HELPER_PATH.read_text(encoding="utf-8")
    # Buscar el bloque `const CHUNK_KIND_LABELS = { ... }`.
    m = re.search(
        r"const\s+CHUNK_KIND_LABELS\s*=\s*\{([\s\S]+?)\}\s*;",
        text,
    )
    assert m is not None, "No pude extraer CHUNK_KIND_LABELS del helper."
    body = m.group(1)
    # Cada line tiene `<key>: '<label>',`. Extraemos las keys.
    keys = set()
    for line_match in re.finditer(r"^\s*([a-z][a-z0-9_]*)\s*:", body, re.MULTILINE):
        keys.add(line_match.group(1))
    return keys


def _extract_backend_kinds() -> set[str]:
    """Extrae los literales `chunk_kind="..."` o `chunk_kind='...'`
    del código de prod del backend."""
    kinds = set()
    for path in _PROD_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(
            r"""chunk_kind\s*=\s*['"]([a-z_]+)['"]""",
            text,
        ):
            kinds.add(m.group(1))
    return kinds


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    text = _HISTORY_PATH.read_text(encoding="utf-8")
    assert "[P2-HIST-NEW-4" in text, (
        "Marker `P2-HIST-NEW-4` debe aparecer en History.jsx donde "
        "se aplica el helper getChunkKindLabel al badge."
    )


def test_marker_present_in_helper():
    text = _HELPER_PATH.read_text(encoding="utf-8")
    assert "[P2-HIST-NEW-4" in text, (
        "Marker `P2-HIST-NEW-4` debe aparecer en chunkKinds.js — "
        "documenta el contrato cross-language."
    )


# ---------------------------------------------------------------------------
# 2. Helper cubre kinds canónicos
# ---------------------------------------------------------------------------
def test_helper_covers_canonical_kinds():
    """Los 3 kinds canónicos del enum del backend deben estar en el
    helper. Sin esto, el badge cae al fallback (code crudo) en lugar
    de mostrar la etiqueta humana."""
    helper_keys = _extract_helper_codes()
    canonical = {"initial_plan", "rolling_refill", "catchup"}
    missing = canonical - helper_keys
    assert not missing, (
        f"Helper chunkKinds.js NO cubre kinds canónicos: {missing}. "
        f"Helper actualmente contiene: {sorted(helper_keys)}."
    )


# ---------------------------------------------------------------------------
# 3. Drift detection: paridad backend ⟷ frontend
# ---------------------------------------------------------------------------
def test_no_orphan_backend_kinds():
    """Cualquier `chunk_kind="..."` literal en código de prod DEBE
    estar en el catálogo del helper. Si alguien agrega un kind nuevo
    en backend sin actualizar el frontend, el badge se cae al code
    crudo (snake_case visible al user) — este test loud-faila para
    detectarlo en CI."""
    helper_keys = _extract_helper_codes()
    backend_kinds = _extract_backend_kinds()
    orphans = backend_kinds - helper_keys
    assert not orphans, (
        f"Backend usa chunk_kind(s) que el frontend NO mapea: {orphans}.\n"
        f"Helper frontend (chunkKinds.js) tiene: {sorted(helper_keys)}.\n"
        f"Backend prod files: {sorted(backend_kinds)}.\n"
        f"Agrega entradas a `CHUNK_KIND_LABELS` en chunkKinds.js o "
        f"justifica con un comentario por qué este kind NO debe "
        f"aparecer en UI (e.g. interno, deprecated)."
    )


# [P0-HIST-FIX-5 · 2026-05-09] Set de kinds que aparecen en DB pero
# NO como literal `chunk_kind="..."` en código de prod (legacy o
# producidos por paths dinámicos). Permitimos tenerlos en el
# helper para no mostrar snake_case al user, pero documentamos el
# mismatch aquí para que un dev futuro lo entienda en lugar de
# limpiar el helper creyendo que es un orphan.
_HELPER_ONLY_LEGACY_KINDS = {
    # `first_chunk`: alias semántico de `initial_plan` que aparece
    # en plan_chunk_queue.chunk_kind para chunks creados antes del
    # rename a `initial_plan`. Helper lo mapea a "Inicial" (mismo
    # label que `initial_plan`) para que el badge sea coherente.
    "first_chunk",
}


def test_no_orphan_helper_kinds():
    """Defensivo: si el helper tiene un kind que NUNCA se usa en
    backend prod, probablemente fue agregado por error o el backend
    lo eliminó. Loud-faila para que un dev lo limpie o restaure el
    backend.

    [P0-HIST-FIX-5 · 2026-05-09] Excepción: kinds que aparecen en DB
    pero no como literal en código (legacy / dynamic paths) están en
    `_HELPER_ONLY_LEGACY_KINDS` y se excluyen del orphan check."""
    helper_keys = _extract_helper_codes()
    backend_kinds = _extract_backend_kinds()
    orphans = helper_keys - backend_kinds - _HELPER_ONLY_LEGACY_KINDS
    assert not orphans, (
        f"Helper frontend tiene chunk_kind(s) que NO aparecen en "
        f"backend prod: {orphans}.\n"
        f"Backend prod kinds: {sorted(backend_kinds)}.\n"
        f"Helper kinds: {sorted(helper_keys)}.\n"
        f"Si el kind es legacy o ya no se usa, eliminarlo del "
        f"catálogo evita confusión."
    )


# ---------------------------------------------------------------------------
# 4. Sanity: el regex parser no rompe ante código del repo
# ---------------------------------------------------------------------------
def test_extract_returns_non_empty():
    """Sanity check: si el regex parser falló (e.g. file movido o
    refactor que cambió el shape del const), ambas extracciones
    devolverían vacío y los tests pasarían vacuamente."""
    helper_keys = _extract_helper_codes()
    backend_kinds = _extract_backend_kinds()
    assert len(helper_keys) >= 3, (
        f"Helper extraction returned {len(helper_keys)} keys — "
        f"probable bug del regex parser. Got: {helper_keys}."
    )
    assert len(backend_kinds) >= 2, (
        f"Backend kinds extraction returned {len(backend_kinds)} — "
        f"probable bug del regex parser. Got: {backend_kinds}."
    )
