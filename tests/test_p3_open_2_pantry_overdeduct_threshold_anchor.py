"""[P3-OPEN-2 · 2026-05-11] Anchor del threshold `pantry_overdeduct` y
criterio de trigger para subirlo de 0.5 a 0.75 (deferred sin code-change).

Contexto:
    `_classify_divergence_hypothesis` (shopping_calculator.py) clasifica una
    divergencia receta↔lista como `pantry_overdeduct` cuando
    `actual < expected * threshold`. El threshold actual es 0.5 — un caso
    real auditado (audit 2026-05-10) mostró que ratios 0.50-0.75 (e.g.,
    receta 3kg pollo, nevera promete 2kg → ratio 0.67) caen al bucket
    `unknown` en lugar de `pantry_overdeduct`, dificultando el triage
    operacional.

    P3-NEW-5 (2026-05-10) documentó el trigger inline pero NO cambió el
    threshold (se mantuvo conservador). Este test ancla:

      1. El **valor actual** del knob (`0.5`) en el código de prod.
      2. El **knob name** canónico (`MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD`).
      3. El **criterio de trigger** documentado: `>25% de los unknown del
         bucket en pipeline_metrics correlacionan con sobrededucción real`.

    Si SRE decide bumpear el threshold (porque telemetría cumple el
    criterio), debe:
      - Cambiar el default literal en `shopping_calculator.py` (de 0.5
        a 0.75 o lo que sea).
      - Bumpear `_EXPECTED_DEFAULT` abajo para que CI tenga el nuevo
        valor anclado.
      - Actualizar la sección "Trigger para actuar" del comentario
        inline en `shopping_calculator.py`.

    Drift detection: si alguien cambia el default sin bumpear este test,
    falla con copy explícito apuntando al criterio.

Tooltip-anchor: P3-OPEN-2-START | gap P3 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SHOPPING = _REPO_ROOT / "backend" / "shopping_calculator.py"

# Default actual del threshold. Si cambia, bumpear acá Y actualizar el
# comentario inline de "Trigger para actuar" en shopping_calculator.py.
_EXPECTED_DEFAULT = 0.5

# Knob name canónico. NO renombrar sin migración explícita — operadores
# con scripts que lo seteen via env var asumen este nombre exacto.
_EXPECTED_KNOB_NAME = "MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD"


def _read_shopping_src() -> str:
    return _SHOPPING.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Knob name y default anclados
# ---------------------------------------------------------------------------
def test_overdeduct_threshold_default_unchanged() -> None:
    """El default del threshold debe permanecer en `_EXPECTED_DEFAULT`
    hasta que SRE ejecute el bump deliberadamente (con justificación
    en pipeline_metrics).
    """
    src = _read_shopping_src()
    # Match laxo: el knob puede invocarse vía `_knob_env_float(...)` o
    # similar. Buscamos la cadena `MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD`
    # seguida de cualquier whitespace/sep y el literal float default.
    pat = re.compile(
        r"['\"]" + re.escape(_EXPECTED_KNOB_NAME) + r"['\"]"
        r"\s*,\s*"
        r"(?P<default>[0-9]+\.[0-9]+|[0-9]+)",
        re.MULTILINE,
    )
    m = pat.search(src)
    assert m, (
        f"P3-OPEN-2: knob `{_EXPECTED_KNOB_NAME}` no encontrado con default "
        f"literal en `shopping_calculator.py`. Si el knob fue renombrado o "
        f"refactorizado, actualizar este test PRIMERO (y verificar que el "
        f"comentario inline de trigger sigue vigente)."
    )
    default = float(m.group("default"))
    assert abs(default - _EXPECTED_DEFAULT) < 1e-9, (
        f"P3-OPEN-2: el default de `{_EXPECTED_KNOB_NAME}` cambió de "
        f"{_EXPECTED_DEFAULT} a {default}. Esto requiere:\n"
        f"  1. Evidencia en `pipeline_metrics` de que el criterio se cumple "
        f"(>25% de unknowns son overdeducts reales — ver comentario inline en "
        f"shopping_calculator.py sección 'Trigger para actuar').\n"
        f"  2. Bumpear `_EXPECTED_DEFAULT` en este test al nuevo valor.\n"
        f"  3. Actualizar la memoria P3-OPEN-2 con la fecha + evidencia + "
        f"nuevo valor.\n"
        f"Si el cambio fue accidental, revertirlo. El test bloquea para "
        f"forzar la decisión explícita."
    )


# ---------------------------------------------------------------------------
# 2. Knob registrado en _KNOBS_REGISTRY (auto-poblado por _knob_env_float)
# ---------------------------------------------------------------------------
def test_overdeduct_knob_uses_env_helper() -> None:
    """El knob debe leerse vía un helper que lo registre en
    `_KNOBS_REGISTRY` (uno de `_knob_env_float`, `_env_float`, etc.),
    no como un hardcoded literal. Sin esto, operadores no pueden
    overridearlo en runtime y SRE no puede ver el valor activo en
    `/admin/knobs`.
    """
    src = _read_shopping_src()
    pat = re.compile(
        r"_(?:knob_)?env_float\s*\([\s\S]{0,200}?"
        + re.escape(_EXPECTED_KNOB_NAME),
        re.MULTILINE,
    )
    assert pat.search(src), (
        f"P3-OPEN-2: knob `{_EXPECTED_KNOB_NAME}` debe leerse vía "
        f"`_knob_env_float` o `_env_float` para auto-registrarse en "
        f"`_KNOBS_REGISTRY`. Hardcoded literal o lectura directa de "
        f"`os.environ` saltan el sistema de observabilidad de knobs."
    )


# ---------------------------------------------------------------------------
# 3. Criterio de trigger documentado inline (anclado, no se puede borrar
#    silenciosamente)
# ---------------------------------------------------------------------------
def test_trigger_criterion_documented_inline() -> None:
    """El comentario inline en `shopping_calculator.py` debe mencionar
    explícitamente el criterio: `>25%` correlación + `pipeline_metrics`.

    Si alguien borra esa documentación sin actualizar este test, fail.
    Sin el criterio escrito, un SRE futuro no sabe cuándo es seguro bumpear.
    """
    src = _read_shopping_src()
    # Match laxo: cualquier mención de "25%" + "pipeline_metrics" dentro
    # del comentario de la función `_classify_divergence_hypothesis`.
    has_25_pct = ">25%" in src or ">= 25%" in src or "25 %" in src or "25%" in src
    has_metrics = "pipeline_metrics" in src
    has_trigger_section = "Trigger" in src or "trigger" in src

    assert has_25_pct and has_metrics and has_trigger_section, (
        "P3-OPEN-2: el comentario inline del threshold `pantry_overdeduct` "
        "debe documentar el criterio canónico de trigger: `>25%` de unknowns "
        "correlacionan con overdeducts reales según `pipeline_metrics`. "
        "Detectado: "
        f"25%={has_25_pct}, pipeline_metrics={has_metrics}, "
        f"trigger-section={has_trigger_section}."
    )


# ---------------------------------------------------------------------------
# 4. Slug del marker en filename
# ---------------------------------------------------------------------------
def test_marker_anchor_present() -> None:
    """Filename contiene `p3_open_2` para cross-link audit."""
    assert "p3_open_2" in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug `p3_open_2`."
    )
