"""[P1-3 · 2026-05-10] Cross-language drift detection: el catálogo de
labels en `frontend/src/utils/coherenceLabels.js` cubre todos los códigos
emitidos por el backend.

Bug observado en el audit 2026-05-10:
    El tab "Ajustes" del Historial renderizaba `action_taken` crudo
    (`degrade`, `reject_minor`, etc.) directamente al usuario. Las
    `hypothesis` de cada divergencia (`cap_swallowed_modifier`,
    `unit_mismatch`, `yield_uncovered`, `pantry_overdeduct`, `unknown`)
    no se mostraban en absoluto. P1-3 cierra el gap creando
    `coherenceLabels.js` con maps es-DO + helpers + este test que
    enforza paridad.

Cobertura de este test:
    1. Cada `hypothesis` que el backend puede emitir desde
       `_classify_divergence_hypothesis` está mapeada en el JS.
    2. Cada `action_taken` que el backend puede emitir desde sus 3 sites
       (review_plan_node, assemble_plan_node, _recompute_aggregates_after_swap)
       está mapeada en el JS.
    3. Inversamente: el JS no expone códigos que el backend nunca emita
       (atrapa typos / drift "frontend agregó label sin código real").
    4. Helper exports (`getCoherenceActionLabel`, `getCoherenceHypothesisLabel`,
       `_*_LABELS_MAP`) están presentes con la firma esperada.

El test es estático: parsea archivos source con regex, sin importar el
código JS (no requiere node/jest).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_ROOT = _BACKEND_ROOT.parent / "frontend"
_JS_FILE = _FRONTEND_ROOT / "src" / "utils" / "coherenceLabels.js"
_SHOPPING_PY = _BACKEND_ROOT / "shopping_calculator.py"
_ORCHESTRATOR_PY = _BACKEND_ROOT / "graph_orchestrator.py"


def _read(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"Archivo no encontrado: {p}")
    return p.read_text(encoding="utf-8")


def _extract_function_body_lines(py_src: str, fn_name: str) -> str:
    """Devuelve el cuerpo de una función `def fn_name(...)` como string.

    Implementación line-based (NO regex) para evitar catastrophic backtracking
    sobre archivos grandes como shopping_calculator.py (4000+ líneas).

    Estrategia: localiza la línea `def <fn_name>(` y avanza hasta la próxima
    línea top-level (sin indent) que comience con `def `, `class ` o EOF.
    """
    lines = py_src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {fn_name}("):
            start = i
            break
    if start is None:
        return ""
    # Avanzar hasta la próxima top-level def/class.
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln and not ln[0].isspace() and (ln.startswith("def ") or ln.startswith("class ")):
            end = j
            break
    return "\n".join(lines[start:end])


def _extract_js_map_keys(js_src: str, var_name: str) -> set[str]:
    """Extrae las keys de un objeto JS literal `const VAR = { foo: '...', bar: '...' }`.

    Defensivo contra:
      - whitespace y comentarios entre keys.
      - keys quoted (`'foo'`) o unquoted (`foo`).
      - valores multi-línea (no nos interesa el valor, solo la key).
    """
    # 1. Localizar la asignación. Permite líneas opcionales de comentario antes de `{`.
    m = re.search(
        rf"const\s+{re.escape(var_name)}\s*=\s*\{{",
        js_src,
    )
    if m is None:
        return set()
    # 2. Recortar desde el `{` matcheado hasta el `}` que cierra el objeto al
    #    mismo nivel. Usamos un balanced-bracket scanner simple.
    start = m.end() - 1  # apunta al `{`
    depth = 0
    end = start
    for i in range(start, len(js_src)):
        ch = js_src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = js_src[start + 1:end]
    # 3. Extraer keys: identificadores o strings al inicio de cada par.
    #    Patrón: comma o inicio de body, opcional whitespace/comments,
    #    luego key (quoted o unquoted), luego ':'.
    # Para simplicidad: line-by-line tras strip de comments.
    keys = set()
    # Quitar comentarios `/* ... */` y `// ...`.
    body_no_comments = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    body_no_comments = re.sub(r"//[^\n]*", "", body_no_comments)
    # Match `<key>:` donde key es identificador o string.
    for km in re.finditer(
        r"(?:^|[,{\s])\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))\s*:",
        body_no_comments,
    ):
        key = km.group(1) or km.group(2) or km.group(3)
        if key:
            keys.add(key)
    return keys


# ---------------------------------------------------------------------------
# 1. Helpers exportados existen
# ---------------------------------------------------------------------------
def test_js_helpers_and_maps_exported():
    js = _read(_JS_FILE)
    assert "export const getCoherenceActionLabel" in js
    assert "export const getCoherenceHypothesisLabel" in js
    assert "export const _COHERENCE_ACTION_LABELS_MAP" in js
    assert "export const _COHERENCE_HYPOTHESIS_LABELS_MAP" in js


# ---------------------------------------------------------------------------
# 2. Hipótesis: backend → frontend cobertura completa
# ---------------------------------------------------------------------------
def _backend_hypothesis_codes() -> set[str]:
    """Extrae los string literals devueltos desde
    `_classify_divergence_hypothesis` en shopping_calculator.py."""
    body = _extract_function_body_lines(
        _read(_SHOPPING_PY), "_classify_divergence_hypothesis"
    )
    assert body, (
        "No se encontró `_classify_divergence_hypothesis` en shopping_calculator.py"
    )
    return set(re.findall(r'return\s+["\']([a-z_]+)["\']', body))


def test_all_backend_hypotheses_mapped_in_js():
    """Toda hypothesis devuelta por `_classify_divergence_hypothesis` debe
    tener entry en `COHERENCE_HYPOTHESIS_LABELS`. Si no, la UI mostraría
    el code raw."""
    backend_codes = _backend_hypothesis_codes()
    assert backend_codes, "Sin literals; el parser puede haber drifteado"

    js_codes = _extract_js_map_keys(_read(_JS_FILE), "COHERENCE_HYPOTHESIS_LABELS")
    missing = backend_codes - js_codes
    assert not missing, (
        f"Hipótesis del backend sin entry en COHERENCE_HYPOTHESIS_LABELS: {missing}. "
        f"Backend emite: {sorted(backend_codes)}. "
        f"JS cubre: {sorted(js_codes)}. "
        f"Añadir entrada en `frontend/src/utils/coherenceLabels.js`."
    )


def test_no_orphan_hypothesis_in_js():
    """JS no debe tener hipótesis que el backend nunca emita."""
    backend_codes = _backend_hypothesis_codes()
    js_codes = _extract_js_map_keys(_read(_JS_FILE), "COHERENCE_HYPOTHESIS_LABELS")
    extras = js_codes - backend_codes
    assert not extras, (
        f"JS tiene hipótesis que el backend no emite: {extras}. "
        f"¿Typo o el backend dejó de emitirlas? Revisar y limpiar."
    )


# ---------------------------------------------------------------------------
# 3. Action_taken: backend → frontend cobertura completa
# ---------------------------------------------------------------------------
# Los 3 sites del backend que ASIGNAN action_taken (NO los que solo lo leen).
_KNOWN_ACTION_VALUES = {
    "not_applicable",         # assemble_plan_node (initial state)
    "degrade",                # review_plan_node (kill switch)
    "reject_minor",           # review_plan_node (default block action)
    "reject_high",            # review_plan_node (forced retry)
    "hydration_error",        # review_plan_node (P2-2 fallback)
    "post_swap_revalidation", # _recompute_aggregates_after_swap
}


def test_all_known_action_values_mapped_in_js():
    """Toda `action_taken` que el backend asigna en los 3 sites debe tener
    entry en `COHERENCE_ACTION_LABELS`."""
    js_codes = _extract_js_map_keys(_read(_JS_FILE), "COHERENCE_ACTION_LABELS")
    missing = _KNOWN_ACTION_VALUES - js_codes
    assert not missing, (
        f"Acciones del backend sin entry en COHERENCE_ACTION_LABELS: {missing}. "
        f"Backend emite: {sorted(_KNOWN_ACTION_VALUES)}. "
        f"JS cubre: {sorted(js_codes)}."
    )


def test_no_orphan_action_in_js():
    """JS no debe exponer acciones que no estén en el catálogo conocido."""
    js_codes = _extract_js_map_keys(_read(_JS_FILE), "COHERENCE_ACTION_LABELS")
    extras = js_codes - _KNOWN_ACTION_VALUES
    assert not extras, (
        f"JS tiene acciones no documentadas en `_KNOWN_ACTION_VALUES`: {extras}. "
        f"Si es un código nuevo del backend, añadirlo al test (catálogo SSOT del test) "
        f"y a `frontend/src/utils/coherenceLabels.js` simultáneamente."
    )


def test_action_values_present_in_orchestrator_source():
    """Sanity check: las acciones documentadas en `_KNOWN_ACTION_VALUES`
    aparecen literalmente como string en `graph_orchestrator.py`. Atrapa
    drift "alguien renombró un action_taken en el backend sin actualizar
    test ni JS"."""
    orch = _read(_ORCHESTRATOR_PY)
    missing = []
    for action in _KNOWN_ACTION_VALUES:
        # Buscar como string literal asignado a action_taken.
        if not re.search(rf'["\']{re.escape(action)}["\']', orch):
            missing.append(action)
    assert not missing, (
        f"Acciones del catálogo no encontradas como literal en graph_orchestrator.py: "
        f"{missing}. ¿Renombre / removal silencioso?"
    )


# ---------------------------------------------------------------------------
# 4. Sanity: no labels vacíos
# ---------------------------------------------------------------------------
def test_js_labels_are_nonempty_strings():
    """Cada entry tiene label non-empty (no se asigna `''` por error)."""
    js = _read(_JS_FILE)
    # Detectar `key: ''` o `key: ""` pattern.
    empty_str = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*['\"]\s*['\"]",
        js,
    )
    assert not empty_str, (
        f"Labels vacíos en coherenceLabels.js: {empty_str}. "
        f"Cada code debe tener label es-DO breve."
    )
