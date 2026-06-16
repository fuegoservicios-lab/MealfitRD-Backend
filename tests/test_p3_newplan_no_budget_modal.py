"""[P3-NEWPLAN-NO-BUDGET-MODAL · 2026-05-23] Decisión de producto:
la opción "Opciones económicas" / "Opciones más económicas" del modal
"¿Por qué quieres actualizar?" (regenerate full plan) era ortogonal al
contrato del producto.

El comportamiento DEFAULT del regenerate ya respeta la nevera + lista
de compras (el frontend pasa ``current_pantry_ingredients`` a
``/api/plans/generate``). El hint "💰 ECONÓMICAS" del prompt era una
preferencia de COSTO, no de PANTRY — sugería falsamente al usuario que
los demás reasons NO usaban su nevera.

Decisión owner 2026-05-23:
> "elimina lo de 'opciones más económicas' de raíz ya que no tiene
> sentido porque solo se usaran alimentos que esten en la nevera y si
> la nevera esta vacia debe usar los que estan en la lista de compras
> pdf"

Mirror del precedente swap-meal [[p3_swap_pantry_default_2026_05_22]]
donde la opción "Usar solo lo que tengo" se removió por la misma razón.

## Fix

1. **Frontend** ``Dashboard.jsx`` — opción ``{ id: 'budget', ... }``
   removida de AMBOS arrays del modal new-plan (path ``isPlanExpired``
   y path ciclo activo) + hover info band del 'budget' eliminado.
2. **Backend** ``ai_helpers.py`` — branch ``elif update_reason ==
   'budget':`` eliminado del prompt builder. Comentario explicativo
   preservado como ancla narrativa. ``pantry_first`` preservado para
   back-compat con callers legacy.

## Back-compat

Callers legacy que emitan ``update_reason='budget'`` siguen aceptados
(no raise) — simplemente caen al default sin context_extras extra,
mismo comportamiento que cualquier reason no-matched.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_newplan_no_budget_modal`` ↔ filename
``test_p3_newplan_no_budget_modal.py``.
"""
import pathlib
import re

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"

DASHBOARD_JSX = (FRONTEND_ROOT / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
AI_HELPERS_PY = (BACKEND_ROOT / "ai_helpers.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Frontend: opción 'budget' del new-plan modal removida
# ---------------------------------------------------------------------------

def test_newplan_modal_no_longer_exposes_economic_label():
    """El modal new-plan NO debe exponer las labels JSX
    ``"Opciones económicas"`` ni ``"Opciones más económicas"`` (ambos
    paths del modal según ``isPlanExpired``)."""
    # JSX literal — match precise para evitar false-positives con
    # menciones narrativas del removal en comments
    assert not re.search(
        r"label:\s*['\"]Opciones económicas['\"]",
        DASHBOARD_JSX,
    ), (
        "Property `label: 'Opciones económicas'` (path isPlanExpired) "
        "aún presente en Dashboard.jsx — la opción debe estar removida."
    )
    assert not re.search(
        r"label:\s*['\"]Opciones más económicas['\"]",
        DASHBOARD_JSX,
    ), (
        "Property `label: 'Opciones más económicas'` (path ciclo activo) "
        "aún presente en Dashboard.jsx — la opción debe estar removida."
    )


def test_newplan_modal_no_budget_id_with_economic_desc():
    """No debe quedar un objeto JSX `{ id: 'budget', ...desc:'...bajo costo'... }`
    en el array del modal new-plan. Match scope-aware sobre la property
    JSX `desc:` para evitar false-positives con comments narrativos del
    removal que mencionen las frases como contexto histórico."""
    # Property JSX literal — `desc: 'Priorizar ingredientes de bajo costo'`
    assert not re.search(
        r"desc:\s*['\"]Priorizar ingredientes de bajo costo['\"]",
        DASHBOARD_JSX,
    ), (
        "JSX property `desc: 'Priorizar ingredientes de bajo costo'` aún "
        "presente en el array del modal (path isPlanExpired)."
    )
    # Property JSX literal — `desc: 'Ingredientes de bajo costo'`
    assert not re.search(
        r"desc:\s*['\"]Ingredientes de bajo costo['\"]",
        DASHBOARD_JSX,
    ), (
        "JSX property `desc: 'Ingredientes de bajo costo'` aún presente "
        "en el array del modal (path ciclo activo)."
    )


def test_newplan_modal_hover_info_band_no_budget_branch():
    """El hover info band NO debe tener un branch
    ``hoveredOption === 'budget'`` — sin botón, sin hover."""
    assert "hoveredOption === 'budget'" not in DASHBOARD_JSX, (
        "Hover info band aún tiene branch `hoveredOption === 'budget'`. "
        "Sin la opción visible no debe haber path de hover."
    )
    # Sanity: el copy específico del banner económico debe estar gone
    assert "Económico:" not in DASHBOARD_JSX, (
        "Copy `<strong>Económico:</strong>` aún presente en el hover."
    )


def test_newplan_modal_preserves_other_options_intact():
    """[FUNCIONAL] Las demás opciones del new-plan modal NO deben verse
    afectadas por el removal."""
    expected_options_present = [
        "id: 'variety'",
        "id: 'time'",
        "id: 'cravings'",
        "id: 'weekend'",  # weekendOption
        "id: 'dislike'",
    ]
    for opt in expected_options_present:
        assert opt in DASHBOARD_JSX, (
            f"Opción esperada `{opt}` desapareció — scope del removal "
            f"debe ser SOLO 'budget' del modal new-plan."
        )


def test_newplan_marker_anchor_preserved():
    """El marker ancla del removal debe permanecer en Dashboard.jsx
    para que un futuro reader sepa POR QUÉ se removió."""
    assert "P3-NEWPLAN-NO-BUDGET-MODAL" in DASHBOARD_JSX, (
        "Marker ancla `P3-NEWPLAN-NO-BUDGET-MODAL` ausente — un futuro "
        "maintainer podría re-añadir la opción sin entender por qué."
    )


# ---------------------------------------------------------------------------
# Section B — Backend: branch `update_reason == 'budget'` removido
# ---------------------------------------------------------------------------

def test_backend_no_budget_branch_in_prompt_builder():
    """``ai_helpers.py`` NO debe tener ``elif update_reason == 'budget':``
    como branch activo. El hint "💰 ECONÓMICAS" del prompt se elimina
    porque era ortogonal al contrato real (pantry-based)."""
    assert "elif update_reason == 'budget':" not in AI_HELPERS_PY, (
        "Branch `elif update_reason == 'budget':` aún activo en "
        "ai_helpers.py. Debe estar removido — el hint económico "
        "confundía la semántica del flow."
    )
    # El copy específico del hint tampoco debe quedar como string literal
    assert "💰 [INTENCIÓN DEL USUARIO]" not in AI_HELPERS_PY, (
        "Copy del hint 💰 ECONÓMICAS aún presente como string literal."
    )


def test_backend_marker_anchor_preserved():
    """El marker ancla `P3-NEWPLAN-NO-BUDGET-MODAL` debe permanecer en
    ai_helpers.py como ancla narrativa del removal."""
    assert "P3-NEWPLAN-NO-BUDGET-MODAL" in AI_HELPERS_PY, (
        "Marker ancla ausente en ai_helpers.py."
    )


def test_backend_pantry_first_branch_preserved_for_back_compat():
    """``elif update_reason == 'pantry_first':`` debe seguir presente —
    es back-compat para callers legacy que aún emiten ese reason."""
    assert "elif update_reason == 'pantry_first':" in AI_HELPERS_PY, (
        "Branch `pantry_first` debe seguir presente para back-compat. "
        "Si lo eliminas, callers legacy fallan silenciosamente."
    )


def test_backend_other_reasons_branches_intact():
    """[PARSER] Los branches de los demás reasons (variety/dislike/time/
    similar/cravings/weekend) NO deben verse afectados por el removal.
    NOTA: 'variety' es el PRIMER branch del chain → usa `if` no `elif`."""
    # Variety es el primer branch (if, no elif)
    assert "if update_reason == 'variety':" in AI_HELPERS_PY, (
        "Branch `if update_reason == 'variety':` (primer branch del chain) "
        "desapareció — el removal debe ser scope SOLO 'budget'."
    )
    # Los demás son elif
    expected_elif_branches = [
        "elif update_reason == 'dislike':",
        "elif update_reason == 'time':",
        "elif update_reason == 'similar':",
        "elif update_reason == 'cravings':",
        "elif update_reason == 'weekend':",
    ]
    for branch in expected_elif_branches:
        assert branch in AI_HELPERS_PY, (
            f"Branch esperado `{branch}` desapareció — scope del removal "
            f"debe ser SOLO 'budget'."
        )


# ---------------------------------------------------------------------------
# Section C — Back-compat: 'budget' como input sigue aceptado sin error
# ---------------------------------------------------------------------------

def test_budget_input_no_longer_matched_but_not_rejected():
    """[FUNCIONAL] Si un caller legacy emite ``update_reason='budget'``,
    el flow NO debe rechazar — simplemente cae al default (sin
    context_extras extra, mismo path que cualquier reason no-matched)."""
    # Verificamos que NO hay validación que rechace 'budget' como input
    # inválido. El backend debe aceptar el string sin error.
    # (Esta sanity es estructural — no hay un endpoint que valide
    # estrictamente la lista de reasons aceptados.)
    assert "update_reason" in AI_HELPERS_PY, (
        "`update_reason` debe seguir siendo procesado por ai_helpers.py."
    )
    # No debe haber un raise/error para reasons no-matched
    assert "raise ValueError" not in AI_HELPERS_PY[
        AI_HELPERS_PY.find("update_reason == 'variety'"):
        AI_HELPERS_PY.find("update_reason == 'weekend':") + 500
    ], (
        "El elif chain de update_reason NO debe levantar ValueError para "
        "reasons no-matched (sin esto, 'budget' legacy crashearía)."
    )


# ---------------------------------------------------------------------------
# Section D — Marker bumped
# ---------------------------------------------------------------------------

def test_marker_bumped():
    """``_LAST_KNOWN_PFIX`` debe estar bumpeado a un marker real y fresco.

    Originalmente pineaba el literal exacto
    ``P3-NEWPLAN-NO-BUDGET-MODAL · 2026-05-23`` (el valor al cerrar ESTE
    P-fix). Pero ``_LAST_KNOWN_PFIX`` es un marker rolling: CADA P-fix
    posterior lo bumpea (ver convención en CLAUDE.md + SSOT
    ``test_p3_1_last_known_pfix_freshness``), así que pinear el literal lo
    rompe en el primer cierre subsiguiente. La intención real del test era
    "el marker SÍ se bumpeó para este feature, no quedó en un placeholder
    pre-feature". Reflejamos el prod actual verificando que el marker:
      1. existe y sigue el formato canónico ``Pn-X · YYYY-MM-DD``, y
      2. tiene fecha >= la fecha de este P-fix (2026-05-23) — confirma que
         hubo bump (este feature o uno posterior), nunca regresión.
    """
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"(?P<val>[^"]+)"', APP_PY
    )
    assert m is not None, "Marker `_LAST_KNOWN_PFIX` no encontrado en app.py."
    marker = m.group("val")
    fmt = re.match(
        r"^P\d+(?:-[A-Z0-9]+)+\s+·\s+(?P<date>\d{4}-\d{2}-\d{2})$", marker
    )
    assert fmt is not None, (
        f"`_LAST_KNOWN_PFIX={marker!r}` no sigue el formato canónico "
        f"`Pn-X · YYYY-MM-DD` (ver test_p3_1_last_known_pfix_freshness)."
    )
    from datetime import date as _date
    marker_date = _date.fromisoformat(fmt.group("date"))
    assert marker_date >= _date(2026, 5, 23), (
        f"`_LAST_KNOWN_PFIX={marker!r}` tiene fecha < 2026-05-23 (fecha de "
        f"este P-fix) → el marker regresó a un valor pre-feature."
    )
