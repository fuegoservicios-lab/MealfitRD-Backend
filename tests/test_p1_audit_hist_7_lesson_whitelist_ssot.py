"""[P1-AUDIT-HIST-7 · 2026-05-09] Tests del SSOT
``constants.LESSON_COUNT_EVENT_WHITELIST`` y de la consolidación
del catálogo de events de lecciones.

Bug original (audit Historial 2026-05-09):
    `_LESSON_COUNT_EVENT_WHITELIST` vivía en `routers/plans.py` como
    tupla literal in-place. Cualquier consumidor adicional (admin
    tool, dashboard, monitoring SRE, futuras rutas) tenía que
    duplicar la lista o leer el atributo privado vía `getattr` (lo
    que hacía el test P1-HIST-AUDIT-5). El test cubría drift entre
    la tupla y los events emitidos por `cron_tasks.py`, pero NO
    detectaba si algún consumidor copiara la tupla literal en otro
    sitio.

Fix:
    Mover la tupla a `constants.LESSON_COUNT_EVENT_WHITELIST`
    (SSOT). `routers/plans.py` la importa con alias privado
    `_LESSON_COUNT_EVENT_WHITELIST` para retrocompat con tests
    existentes (P1-HIST-AUDIT-5, P2-HIST-AUDIT-2). Cualquier
    consumidor nuevo importa la pública desde constants.

Cobertura:
    1. Anchor del marker en `constants.py`.
    2. `LESSON_COUNT_EVENT_WHITELIST` está exportada en `constants`
       y es la tupla canónica (4 events).
    3. `routers/plans.py` importa de `constants` (no redefine
       inline).
    4. Alias privado `_LESSON_COUNT_EVENT_WHITELIST` en `plans`
       apunta al MISMO objeto (`is` identity, no solo `==`).
    5. Drift detection: cualquier otro archivo del backend que
       contenga la tupla literal `("lesson_synthesized_low_confidence",
       "synth_propagated_to_prompt", ...)` falla el test — debe
       importar de constants en su lugar.
    6. Tests existentes (P1-HIST-AUDIT-5, P2-HIST-AUDIT-2) siguen
       leyendo el atributo via `routers.plans` sin romper.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CONSTANTS_PATH = _BACKEND_ROOT / "constants.py"
_PLANS_PATH = _BACKEND_ROOT / "routers" / "plans.py"

_EXPECTED_EVENTS = frozenset({
    "lesson_synthesized_low_confidence",
    "synth_propagated_to_prompt",
    "recent_lessons_partial_synthesis",
    "indefinite_pause_unblocked",
})


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_in_constants_py():
    """`constants.py` debe citar el marker P1-AUDIT-HIST-7 en el
    bloque de definición — sin esto, un grep + git blame no llega
    al fix."""
    text = _CONSTANTS_PATH.read_text(encoding="utf-8")
    assert "P1-AUDIT-HIST-7" in text, (
        "constants.py debe citar `P1-AUDIT-HIST-7` en el comentario "
        "load-bearing del bloque LESSON_COUNT_EVENT_WHITELIST."
    )


def test_marker_in_plans_py_at_import_site():
    """`routers/plans.py` debe citar el marker donde reemplaza la
    tupla in-place por el import. Sin esto, un futuro refactor que
    'limpie' el comentario podría también deshacer el import."""
    text = _PLANS_PATH.read_text(encoding="utf-8")
    assert "P1-AUDIT-HIST-7" in text


# ---------------------------------------------------------------------------
# 2. Exportación pública desde constants
# ---------------------------------------------------------------------------
def test_constants_exports_lesson_count_event_whitelist():
    """`constants.LESSON_COUNT_EVENT_WHITELIST` (público, sin
    underscore) debe existir y contener exactamente los 4 events
    semánticos."""
    import constants
    whitelist = getattr(constants, "LESSON_COUNT_EVENT_WHITELIST", None)
    assert whitelist is not None, (
        "constants.py debe exportar `LESSON_COUNT_EVENT_WHITELIST`."
    )
    assert isinstance(whitelist, tuple), (
        f"`LESSON_COUNT_EVENT_WHITELIST` debe ser tuple (immutable). "
        f"Got: {type(whitelist)}."
    )
    assert set(whitelist) == _EXPECTED_EVENTS, (
        f"Whitelist diverge del catálogo esperado:\n"
        f"  esperado: {sorted(_EXPECTED_EVENTS)}\n"
        f"  actual:   {sorted(whitelist)}"
    )


# ---------------------------------------------------------------------------
# 3. routers/plans.py importa de constants (no redefine in-place)
# ---------------------------------------------------------------------------
def test_plans_imports_whitelist_from_constants():
    """`routers/plans.py` debe usar `from constants import
    LESSON_COUNT_EVENT_WHITELIST as _LESSON_COUNT_EVENT_WHITELIST`
    (o equivalente). Sin esto, redefinir la tupla in-place vuelve
    a abrir el riesgo de drift."""
    text = _PLANS_PATH.read_text(encoding="utf-8")
    # Buscar el import explícito desde constants. Aceptamos varios
    # patrones razonables.
    pattern = re.compile(
        r"from\s+constants\s+import\s+[^#\n]*LESSON_COUNT_EVENT_WHITELIST",
        re.IGNORECASE,
    )
    assert pattern.search(text), (
        "routers/plans.py debe importar `LESSON_COUNT_EVENT_WHITELIST` "
        "desde `constants` (SSOT). Sin esto, la consolidación P1-AUDIT-"
        "HIST-7 está rota."
    )


def test_plans_does_not_redefine_whitelist_inline():
    """Defensa contra regresión: el archivo NO debe contener una
    asignación de tupla con los 4 events literales — eso indicaría
    una redefinición que rompe el SSOT."""
    text = _PLANS_PATH.read_text(encoding="utf-8")
    # Buscamos la tupla literal completa con los 4 strings exactos
    # en cualquier orden. Si aparece, alguien redefinió.
    # Patrón: `= (` seguido de los 4 events en alguna combinación.
    # Más simple: contar cuántas asignaciones hay con los 4 strings
    # juntos.
    redefinition_pattern = re.compile(
        r"=\s*\(\s*"
        r"['\"]lesson_synthesized_low_confidence['\"]\s*,\s*"
        r"['\"]synth_propagated_to_prompt['\"]\s*,\s*"
        r"['\"]recent_lessons_partial_synthesis['\"]\s*,\s*"
        r"['\"]indefinite_pause_unblocked['\"]",
        re.IGNORECASE | re.DOTALL,
    )
    matches = redefinition_pattern.findall(text)
    assert not matches, (
        f"routers/plans.py contiene una redefinición inline de la "
        f"tupla whitelist ({len(matches)} matches). El SSOT vive en "
        f"constants.py — importar desde allá."
    )


# ---------------------------------------------------------------------------
# 4. Identity check: alias y SSOT son el MISMO objeto
# ---------------------------------------------------------------------------
def test_plans_alias_is_same_object_as_constants():
    """`plans._LESSON_COUNT_EVENT_WHITELIST` debe ser el MISMO objeto
    que `constants.LESSON_COUNT_EVENT_WHITELIST` (Python tuple
    interning + import as alias). Si alguien copia el valor a una
    nueva tupla, `is` falla aunque `==` pase — y el riesgo de
    divergencia futura vuelve.
    """
    import constants
    from routers import plans as plans_module
    assert plans_module._LESSON_COUNT_EVENT_WHITELIST is constants.LESSON_COUNT_EVENT_WHITELIST, (
        "El alias debe ser el MISMO objeto que el SSOT (no una copia "
        "que pasa `==`). Patrón correcto: "
        "`from constants import LESSON_COUNT_EVENT_WHITELIST as _LESSON_COUNT_EVENT_WHITELIST`."
    )


# ---------------------------------------------------------------------------
# 5. Drift detection cross-archivo: nadie más redefine
# ---------------------------------------------------------------------------
def test_no_other_file_redefines_whitelist_literal():
    """Recorrer el árbol del backend buscando redefiniciones
    inline de la tupla. Solo `constants.py` (SSOT) debe contenerla.
    `routers/plans.py` la importa. El test cubre la regla "una
    sola fuente": si un dev añade la tupla en otro módulo, falla.
    """
    redefinition_pattern = re.compile(
        r"=\s*\(\s*"
        r"['\"]lesson_synthesized_low_confidence['\"]\s*,\s*"
        r"['\"]synth_propagated_to_prompt['\"]\s*,\s*"
        r"['\"]recent_lessons_partial_synthesis['\"]\s*,\s*"
        r"['\"]indefinite_pause_unblocked['\"]",
        re.IGNORECASE | re.DOTALL,
    )

    offenders = []
    for py_path in _BACKEND_ROOT.rglob("*.py"):
        # Skip venv/dist y el SSOT mismo (constants.py); skip tests
        # (este test contiene los strings esperados como referencia).
        rel = py_path.relative_to(_BACKEND_ROOT).as_posix()
        if (
            rel == "constants.py"
            or rel.startswith("venv/")
            or rel.startswith(".venv/")
            or rel.startswith("tests/")
        ):
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if redefinition_pattern.search(text):
            offenders.append(rel)

    assert not offenders, (
        f"Estos archivos REDEFINEN la tupla whitelist inline en lugar "
        f"de importarla desde constants:\n  {offenders}\n"
        f"Reemplazar por: `from constants import LESSON_COUNT_EVENT_WHITELIST`."
    )


# ---------------------------------------------------------------------------
# 6. Backward-compat: tests legacy siguen funcionando
# ---------------------------------------------------------------------------
def test_legacy_attribute_access_via_getattr_still_works():
    """Tests pre-P1-AUDIT-HIST-7 (P1-HIST-AUDIT-5 línea 86) leen
    el atributo via `getattr(plans_module, '_LESSON_COUNT_EVENT_WHITELIST')`.
    Tras la migración a import-as-alias, el atributo del módulo
    sigue presente y resoluble.
    """
    from routers import plans as plans_module
    whitelist = getattr(plans_module, "_LESSON_COUNT_EVENT_WHITELIST", None)
    assert whitelist is not None, (
        "El alias privado `_LESSON_COUNT_EVENT_WHITELIST` debe "
        "seguir resoluble como atributo del módulo plans para "
        "retrocompat con tests existentes."
    )
    assert set(whitelist) == _EXPECTED_EVENTS
