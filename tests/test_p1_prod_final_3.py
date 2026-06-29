"""[P1-PROD-FINAL-3 · 2026-05-24] Regression guards del bundle P1 que
cierra los 2 P1 residuales detectados tras P2-PROD-FINAL-2 (audit 2026-05-24).

Bundle scope (2 P1):
  1. **P1-EMBEDDING-CACHE-BOUNDED** — `backend/constants.py` reemplaza los
     dicts globales `_embedding_cache = {}` y `_pantry_embeddings_cache = {}`
     por instancias de `_BoundedEmbeddingCache` (LRU con maxsize). Pre-fix
     crecían monotónicamente (1-5 MB/mes en RAM del worker) → OOM eventual
     en el VPS Oracle sin alerta. Knob `MEALFIT_EMBEDDING_CACHE_MAXSIZE`.
  2. **P1-FRONTEND-LOCALSTORAGE-HOT-PATHS** — 6 callsites raw
     `localStorage.setItem(...)` en hot paths migrados a `safeLocalStorageSet`:
       - Settings.jsx:~589 (preferencia notificaciones)
       - Dashboard.jsx (2x — finally del enable-push + dismiss handler)
       - Pantry.jsx (post-recalc mealfit_plan)
       - Plan.jsx (guest session id — golden path crítico)
       - IOSInstallPrompt.jsx (dismiss flag — irónico el archivo iOS-only)
     Pre-fix lanzaban uncaught SecurityError/QuotaExceededError en iOS
     Safari Private Mode → handler abortaba a mitad de side-effects.

Estrategia de los tests:
  - Parser-based (regex sobre source). NO ejecutamos JSX ni cargamos el
    módulo `constants` (que dispara init de embeddings + Supabase client).
  - Tooltip-anchors `P1-PROD-FINAL-3 · 2026-05-24` o
    `P1-EMBEDDING-CACHE-BOUNDED` en source detectan refactors que borren
    los fixes.

P-fix umbrella anchor: `P1-PROD-FINAL-3` — slug `p1_prod_final_3` matchea
este archivo y satisface el cross-link enforcer `test_p2_hist_audit_14`.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"
_APP_PY = _BACKEND_ROOT / "app.py"
_CONSTANTS_PY = _BACKEND_ROOT / "constants.py"
_SETTINGS_JSX = _FRONTEND_SRC / "pages" / "Settings.jsx"
_DASHBOARD_JSX = _FRONTEND_SRC / "pages" / "Dashboard.jsx"
_PANTRY_JSX = _FRONTEND_SRC / "pages" / "Pantry.jsx"
_PLAN_JSX = _FRONTEND_SRC / "pages" / "Plan.jsx"
_IOS_PROMPT_JSX = _FRONTEND_SRC / "components" / "IOSInstallPrompt.jsx"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sección 1 — P1-EMBEDDING-CACHE-BOUNDED
# ---------------------------------------------------------------------------
def test_constants_no_raw_dict_for_embedding_caches():
    """`constants.py` NO debe declarar `_embedding_cache = {}` o
    `_pantry_embeddings_cache = {}` raw. Sin bound, RAM del worker crece
    monotónicamente con cada string único de ingrediente."""
    text = _read(_CONSTANTS_PY)
    raw_pattern = re.compile(
        r"^(?:_embedding_cache|_pantry_embeddings_cache)\s*=\s*\{\s*\}\s*$",
        re.MULTILINE,
    )
    matches = raw_pattern.findall(text)
    assert not matches, (
        f"constants.py declara cache raw `= {{}}` ({len(matches)} match): "
        "regresión del fix P1-EMBEDDING-CACHE-BOUNDED. Usar "
        "`_BoundedEmbeddingCache(_EMBEDDING_CACHE_MAXSIZE)`."
    )


def test_constants_has_bounded_embedding_cache_class():
    """`_BoundedEmbeddingCache` debe existir como clase con dict-like API
    (__contains__/__getitem__/__setitem__) + LRU eviction."""
    text = _read(_CONSTANTS_PY)
    assert "class _BoundedEmbeddingCache" in text, (
        "constants.py no define `_BoundedEmbeddingCache`."
    )
    # API mínima cubierta.
    for method in ("__contains__", "__getitem__", "__setitem__", "__len__"):
        assert method in text, (
            f"`_BoundedEmbeddingCache` no implementa `{method}` — "
            "callsites legacy de `dict.in/[]/=` fallarán."
        )
    # Eviction LRU debe usar OrderedDict + popitem(last=False).
    assert "OrderedDict" in text, (
        "`_BoundedEmbeddingCache` no importa OrderedDict — sin LRU no "
        "hay garantía de qué entry se evicta primero."
    )
    assert "popitem(last=False)" in text, (
        "`_BoundedEmbeddingCache` no llama `popitem(last=False)` — "
        "sin eviction el bound no se respeta."
    )


def test_constants_caches_are_instances_of_bounded_class():
    """Las dos variables `_embedding_cache` y `_pantry_embeddings_cache`
    deben ser instancias de `_BoundedEmbeddingCache`, no dict raw."""
    text = _read(_CONSTANTS_PY)
    for name in ("_embedding_cache", "_pantry_embeddings_cache"):
        pattern = re.compile(
            rf"^{re.escape(name)}\s*=\s*_BoundedEmbeddingCache\(",
            re.MULTILINE,
        )
        assert pattern.search(text), (
            f"`{name}` en constants.py no se asigna como instancia de "
            "`_BoundedEmbeddingCache(...)`."
        )


def test_constants_knob_for_cache_maxsize_registered():
    """El maxsize debe leerse via `_env_int` (auto-registro en
    `_KNOBS_REGISTRY` → visible en `/health/version`). Hardcoded value
    sería deuda silenciosa."""
    text = _read(_CONSTANTS_PY)
    assert "MEALFIT_EMBEDDING_CACHE_MAXSIZE" in text, (
        "Knob `MEALFIT_EMBEDDING_CACHE_MAXSIZE` no declarado. Operador "
        "no puede subir/bajar el cap sin redeploy."
    )
    # Debe pasar por _env_int (cualquier alias importado del módulo knobs).
    assert re.search(
        r"_env_int[^\(]*\(\s*[\"']MEALFIT_EMBEDDING_CACHE_MAXSIZE[\"']",
        text,
    ), (
        "Knob `MEALFIT_EMBEDDING_CACHE_MAXSIZE` no se lee via `_env_int`. "
        "Sin auto-registro no aparece en el snapshot de `/health/version`."
    )


def test_constants_has_anchor_p1_embedding_cache_bounded():
    """Tooltip-anchor debe existir para detectar refactors cosméticos."""
    text = _read(_CONSTANTS_PY)
    assert "P1-EMBEDDING-CACHE-BOUNDED" in text, (
        "constants.py no contiene anchor `P1-EMBEDDING-CACHE-BOUNDED`. "
        "Un refactor podría borrar el bound sin que este test grite."
    )


# ---------------------------------------------------------------------------
# Sección 2 — P1-FRONTEND-LOCALSTORAGE-HOT-PATHS
# ---------------------------------------------------------------------------
_HOT_PATH_FILES_AND_KEYS = (
    (_SETTINGS_JSX, "mealfit_notifications"),
    (_DASHBOARD_JSX, "mealfit_push_onboarding_seen"),
    (_PANTRY_JSX, "mealfit_plan"),
    (_PLAN_JSX, "mealfit_guest_session_id"),
    (_IOS_PROMPT_JSX, "dismissed_ios_prompt"),
)


def test_hot_path_callsites_use_safe_localstorage_set():
    """Cada uno de los 5 archivos de hot path debe usar `safeLocalStorageSet`
    para su key específica, NO `localStorage.setItem(...)` raw."""
    failures = []
    for path, key in _HOT_PATH_FILES_AND_KEYS:
        text = _read(path)
        # safeLocalStorageSet('<key>', ...) debe estar presente.
        safe_pattern = re.compile(
            rf"safeLocalStorageSet\(\s*['\"]{re.escape(key)}['\"]"
        )
        if not safe_pattern.search(text):
            failures.append(f"{path.name}: falta safeLocalStorageSet('{key}', ...)")
        # raw localStorage.setItem('<key>', ...) NO debe quedar.
        raw_pattern = re.compile(
            rf"localStorage\.setItem\(\s*['\"]{re.escape(key)}['\"]"
        )
        if raw_pattern.search(text):
            failures.append(
                f"{path.name}: raw `localStorage.setItem('{key}', ...)` aún presente — "
                "migrar a safeLocalStorageSet."
            )
    assert not failures, "Fallos hot-path:\n" + "\n".join(failures)


def test_hot_path_files_import_safe_localstorage_set():
    """Cada archivo migrado debe importar `safeLocalStorageSet` desde el
    helper SSOT `utils/safeLocalStorage`."""
    failures = []
    for path, _key in _HOT_PATH_FILES_AND_KEYS:
        text = _read(path)
        # Combinar dos chequeos: hay alguna import line que mencione
        # safeLocalStorage SSOT y safeLocalStorageSet aparece en el archivo.
        if "safeLocalStorageSet" not in text:
            failures.append(f"{path.name}: nombre `safeLocalStorageSet` ausente.")
            continue
        if not re.search(
            r"from\s+['\"][./]+utils/safeLocalStorage['\"]",
            text,
        ):
            failures.append(
                f"{path.name}: ningún import desde `utils/safeLocalStorage`."
            )
    assert not failures, "Fallos de import:\n" + "\n".join(failures)


def test_plan_jsx_uses_safe_localstorage_for_guest_session():
    """`Plan.jsx` es golden path crítico: si la lectura/escritura del
    `mealfit_guest_session_id` falla, el SSE nunca arranca para guests.
    Anchor explícito P1-PROD-FINAL-3."""
    text = _read(_PLAN_JSX)
    assert "P1-PROD-FINAL-3" in text, (
        "Plan.jsx no tiene el anchor `P1-PROD-FINAL-3` — el comentario "
        "que documenta por qué este migration es golden-path crítico "
        "fue borrado."
    )
    # Tanto getter como setter migrados (Plan.jsx también leía raw).
    assert "safeLocalStorageGet('mealfit_guest_session_id'" in text, (
        "Plan.jsx no usa safeLocalStorageGet para 'mealfit_guest_session_id'."
    )
    assert "safeLocalStorageSet('mealfit_guest_session_id'" in text, (
        "Plan.jsx no usa safeLocalStorageSet para 'mealfit_guest_session_id'."
    )


# ---------------------------------------------------------------------------
# Sección 3 — Marker bump
# ---------------------------------------------------------------------------
def test_last_known_pfix_marker_meets_p1_prod_final_3_floor():
    """`_LAST_KNOWN_PFIX` en app.py debe tener fecha >= 2026-05-24 (cierre
    del bundle P1-PROD-FINAL-3).

    [Relajado del exact-match `P1-PROD-FINAL-3 · 2026-05-24` original al
    superseder por P2-EMBED-TELEMETRY mismo día. Mismo patrón documentado
    en P1-PROD-FINAL-1 / P2-PROD-FINAL-2. Las otras 10 assertions del
    archivo enforzan que los fixes del bundle siguen vivos.]"""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor {floor} "
        "(P1-PROD-FINAL-3). Si revertiste el marker debes también revertir "
        "las migraciones del bundle (constants.py cache + 5 frontend hot paths)."
    )


def test_marker_date_meets_p1_prod_final_3_floor():
    """Date-floor sibling para futuros supersedes: el marker debe tener
    fecha >= 2026-05-24 (cierre del bundle)."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor {floor} "
        "(P1-PROD-FINAL-3). Si revertiste el marker debes también revertir "
        "las migraciones del bundle (constants.py cache + 5 frontend hot paths)."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Sanity del cap CLAUDE.md
# ---------------------------------------------------------------------------
def test_claude_md_still_under_cap_after_bundle():
    """El bundle P1-PROD-FINAL-3 NO toca CLAUDE.md (todo el contenido nuevo
    vive en `backend/tests/`, source code y memoria). Sanity del cap para
    detectar que un edit accidental no haya engordado CLAUDE.md."""
    claude_md = _REPO_ROOT / "CLAUDE.md"
    cap_test = _BACKEND_ROOT / "tests" / "test_p3_claudemd_cap.py"
    cap_text = cap_test.read_text(encoding="utf-8")
    cap_match = re.search(r"_DEFAULT_CAP\s*=\s*(\d+)", cap_text)
    assert cap_match, "No pude parsear _DEFAULT_CAP del test cap."
    cap = int(cap_match.group(1))
    size = claude_md.stat().st_size
    assert size <= cap, (
        f"CLAUDE.md = {size} > cap {cap}. El bundle P1-PROD-FINAL-3 no "
        "debería haber tocado CLAUDE.md — ¿edit accidental?"
    )
