"""[P2-EMBED-TELEMETRY · 2026-05-24] Regression guards del par natural de
P1-EMBEDDING-CACHE-BOUNDED (mismo día).

Scope:
  El endpoint `/health/version` ([backend/app.py](backend/app.py)) gana 3
  gauges nuevos para observabilidad de los embedding caches bounded:
    - `embedding_cache_size`         — len(_embedding_cache) actual
    - `pantry_embeddings_cache_size` — len(_pantry_embeddings_cache) actual
    - `embedding_cache_maxsize`      — knob `MEALFIT_EMBEDDING_CACHE_MAXSIZE`

Sin estas keys un operador no puede detectar saturación del cache (size
topado contra maxsize → eviction LRU continua → cache hit rate degradado)
antes de que afecte performance. Cierra el follow-up P2 documentado en
[`memory/project_p1_prod_final_3_2026_05_24.md`].

Estrategia: parser-based — NO importamos `app.py` (init Sentry + crons).
Anclamos por regex sobre el source del endpoint + el comment tooltip
`P2-EMBED-TELEMETRY` para que un refactor cosmético no borre el feature.

P-fix umbrella anchor: `P2-EMBED-TELEMETRY` — slug `p2_embed_telemetry`
matchea este archivo y satisface el cross-link enforcer P2-HIST-AUDIT-14.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_CONSTANTS_PY = _BACKEND_ROOT / "constants.py"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sección 1 — Anchor presence
# ---------------------------------------------------------------------------
def test_app_py_has_p2_embed_telemetry_anchor():
    """app.py debe contener el tooltip-anchor `P2-EMBED-TELEMETRY` para que
    un refactor cosmético no borre el feature sin que este test grite."""
    text = _read(_APP_PY)
    assert "P2-EMBED-TELEMETRY" in text, (
        "app.py no contiene anchor `P2-EMBED-TELEMETRY`. Un refactor pudo "
        "haber borrado las gauges del cache."
    )


# ---------------------------------------------------------------------------
# Sección 2 — 3 keys nuevas presentes en el payload retornado
# ---------------------------------------------------------------------------
def test_health_version_payload_includes_embedding_cache_size():
    """El dict de respuesta de `health_version()` debe incluir la key
    `"embedding_cache_size"`."""
    text = _read(_APP_PY)
    assert re.search(
        r'["\']embedding_cache_size["\']\s*:\s*embedding_cache_size',
        text,
    ), (
        "`health_version()` no expone `embedding_cache_size` en el payload."
    )


def test_health_version_payload_includes_pantry_embeddings_cache_size():
    """El dict de respuesta debe incluir la key `"pantry_embeddings_cache_size"`."""
    text = _read(_APP_PY)
    assert re.search(
        r'["\']pantry_embeddings_cache_size["\']\s*:\s*pantry_embeddings_cache_size',
        text,
    ), (
        "`health_version()` no expone `pantry_embeddings_cache_size` en el "
        "payload."
    )


def test_health_version_payload_includes_embedding_cache_maxsize():
    """El dict de respuesta debe incluir la key `"embedding_cache_maxsize"`
    (gauge del knob `MEALFIT_EMBEDDING_CACHE_MAXSIZE`)."""
    text = _read(_APP_PY)
    assert re.search(
        r'["\']embedding_cache_maxsize["\']\s*:\s*embedding_cache_maxsize',
        text,
    ), (
        "`health_version()` no expone `embedding_cache_maxsize` en el payload. "
        "Sin esta gauge, el operador ve `embedding_cache_size` pero no sabe "
        "contra qué cap comparar."
    )


# ---------------------------------------------------------------------------
# Sección 3 — Best-effort import + default -1 (no romper endpoint)
# ---------------------------------------------------------------------------
def test_telemetry_block_imports_from_constants():
    """El bloque telemetry debe importar `_embedding_cache`,
    `_pantry_embeddings_cache` y `_EMBEDDING_CACHE_MAXSIZE` desde `constants`.
    Si los nombres cambian en constants.py este test falla y guía al fix."""
    text = _read(_APP_PY)
    # Pattern flexible para alias `as _ec`/`as _pec`/`as _ec_max`.
    pattern = re.compile(
        r"from\s+constants\s+import\s*\("
        r"[\s\S]{0,200}_embedding_cache[\s\S]{0,200}"
        r"_pantry_embeddings_cache[\s\S]{0,200}"
        r"_EMBEDDING_CACHE_MAXSIZE",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "El bloque P2-EMBED-TELEMETRY no importa los 3 símbolos esperados "
        "desde constants.py (`_embedding_cache`, `_pantry_embeddings_cache`, "
        "`_EMBEDDING_CACHE_MAXSIZE`)."
    )


def test_telemetry_block_is_best_effort():
    """El bloque telemetry debe estar en try/except — un fallo del import
    no debe romper el endpoint completo (`/health/version` debe responder
    200 incluso si constants.py cambia un símbolo). El default -1 sirve
    como sentinel `unknown`."""
    text = _read(_APP_PY)
    # Buscar el block: anchor + try + except.
    block_pattern = re.compile(
        r"P2-EMBED-TELEMETRY[\s\S]{0,2500}"
        r"embedding_cache_size:\s*int\s*=\s*-1[\s\S]{0,800}"
        r"try:[\s\S]{0,800}except\s+Exception:[\s\S]{0,200}pass",
        re.MULTILINE,
    )
    assert block_pattern.search(text), (
        "El bloque P2-EMBED-TELEMETRY no respeta el patrón "
        "`size: int = -1; try: ... except Exception: pass`. Sin esto, "
        "una excepción del import deja el endpoint 500 y los blackbox "
        "monitors externos van a alertar falsamente."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Symbols exported by constants.py (paridad con import del app.py)
# ---------------------------------------------------------------------------
def test_constants_exports_embedding_cache_symbols():
    """`constants.py` debe exponer los 3 símbolos que `app.py` importa.
    Defensa simétrica al test del bloque app.py: si renombras en
    constants.py sin actualizar app.py, este test te avisa."""
    text = _read(_CONSTANTS_PY)
    for symbol in (
        "_embedding_cache",
        "_pantry_embeddings_cache",
        "_EMBEDDING_CACHE_MAXSIZE",
    ):
        # Asignación o declaración a nivel módulo.
        pattern = re.compile(
            rf"^{re.escape(symbol)}\s*=",
            re.MULTILINE,
        )
        assert pattern.search(text), (
            f"`constants.py` no declara `{symbol}` a nivel módulo. "
            "El bloque P2-EMBED-TELEMETRY del `/health/version` fallará "
            "al import y reportará gauges = -1."
        )


# ---------------------------------------------------------------------------
# Sección 5 — Marker bump
# ---------------------------------------------------------------------------
def test_marker_bumped_to_p2_embed_telemetry():
    """[Relajado por P1-PROD-FINAL-4 · 2026-05-24] Sibling date-floor:
    el marker debe tener fecha >= 2026-05-24 (día del cierre P2-EMBED-TELEMETRY).
    Exact-match original removido tras supersede — mismo patrón que
    P1-PROD-FINAL-1 (relajado por P2-PROD-FINAL-2), P2-PROD-FINAL-2 (relajado
    por P1-PROD-FINAL-3), P1-PROD-FINAL-3 (relajado por P2-EMBED-TELEMETRY)."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor 2026-05-24 "
        f"(día del cierre P2-EMBED-TELEMETRY)."
    )


def test_marker_date_meets_p2_embed_telemetry_floor():
    """Date-floor sibling para futuros supersedes."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor {floor}."
    )
