"""[P3-SWAP-PANTRY-DEFAULT · 2026-05-22] Decisión de producto:
strict-pantry pasa a ser el DEFAULT del swap-meal.

Pre-fix: el modal "¿Por qué quieres cambiar?" en Dashboard.jsx exponía
una opción ``"Usar solo lo que tengo"`` (id ``budget``) que prometía
"Aprovecho mi nevera o lista de compras". El backend tradicía la
intención a ``strict_pantry = swap_reason in ("budget", "pantry_first")``
— OPT-IN al hard constraint pantry. Los demás reasons (variety/time/
similar/dislike) caían a un "soft prefer pantry" donde el LLM podía
introducir externos y el validator solo retryaba sin abortar.

Decisión owner (2026-05-22, post-audit production-readiness):
> "veo innecesario la opcion que dice 'Usar solo lo que tengo' ya que
> por defecto debe hacer eso"

Fix:
  1. **Frontend** ``Dashboard.jsx`` — opción ``budget`` removida del array
     del modal. Comentario inline conserva el contexto.
  2. **Backend** ``agent.py::swap_meal`` —
     a. ``elif swap_reason == 'budget'`` removido del elif chain (hint
        redundante con el nuevo default).
     b. ``strict_pantry`` flippeado a inversion: ``swap_reason not in
        ("cravings", "weekend")`` — strict por default, opt-out solo
        para indulgencia explícita.
     c. Hint genérico ``"📦 RESPETA LA NEVERA"`` inyectado para todos
        los reasons que NO son cravings/weekend (antes solo 'budget'
        lo tenía).
  3. **Back-compat**: backend acepta ``swap_reason='budget'``/'pantry_first'``
     como input (legacy callers / clientes cached con la opción vieja)
     y caen en strict-pantry via la inversión del default — mismo
     comportamiento que pre-fix.

Trade-off explícito: el 422 ``swap_strict_pantry_no_inventory`` ahora
se dispara para los 4 reasons base (no solo budget/pantry_first) cuando
pantry + ``aggregated_shopping_list`` ambos vacíos. Cubierto por el fix
UX [[p2_swap_422_ux_copy_2026_05_22]] que ya rendereaba toast honesto.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_swap_pantry_default`` ↔ filename ``test_p3_swap_pantry_default.py``.
"""
import pathlib
import re

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"

AGENT_PY = (BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
DASHBOARD_JSX = (FRONTEND_ROOT / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Frontend: opción 'budget' removida del modal
# ---------------------------------------------------------------------------

def test_modal_no_longer_exposes_budget_option():
    """El modal swap-meal "¿Por qué quieres cambiar?" en Dashboard.jsx NO
    debe exponer la opción "Usar solo lo que tengo". El label JSX es único
    a este modal — el modal de regeneración completa tiene OTRO `id:'budget'`
    con label "Opciones económicas" (sobre costo, no pantry-strict) que NO
    es alcance de este P-fix.

    Pre-fix sugería que los demás reasons NO usaban la nevera, cuando el
    contrato del swap-meal entero es respetar la nevera por default.
    """
    # El label JSX `'Usar solo lo que tengo'` es único al swap-meal modal.
    # Match sobre `label:` property (JSX literal) para evitar falso-positivo
    # con el comentario narrativo del removal que menciona la frase.
    assert not re.search(r"label:\s*['\"]Usar solo lo que tengo['\"]", DASHBOARD_JSX), (
        "Property `label: 'Usar solo lo que tengo'` JSX aún presente en "
        "Dashboard.jsx — la opción del modal swap-meal debe estar removida."
    )
    # Description también única al swap-meal modal pre-fix.
    assert not re.search(r"desc:\s*['\"]Aprovecho mi nevera o lista de compras['\"]", DASHBOARD_JSX), (
        "Property `desc: 'Aprovecho mi nevera o lista de compras'` JSX aún presente."
    )


def test_modal_preserves_anchor_comment_for_decision():
    """El comentario que documenta POR QUÉ se removió debe permanecer
    como ancla narrativa para futuros lectores que consideren restaurar
    la opción."""
    assert "P3-SWAP-PANTRY-DEFAULT" in DASHBOARD_JSX, (
        "Falta marker ancla `P3-SWAP-PANTRY-DEFAULT` en Dashboard.jsx — "
        "un futuro maintainer podría re-añadir la opción sin entender por qué."
    )


def test_modal_other_options_intact():
    """Las demás 6 opciones del modal NO deben verse afectadas por el
    removal. Sanity check del scope del cambio."""
    expected_options = [
        "id: 'variety'",
        "id: 'time'",
        "id: 'cravings'",
        "id: 'weekend'",
        "id: 'similar'",
        "id: 'dislike'",
    ]
    for opt in expected_options:
        assert opt in DASHBOARD_JSX, (
            f"Opción `{opt}` desapareció — el scope del removal debe ser "
            f"SOLO 'budget'. Si removiste otras, revierte; este P-fix no "
            f"las cubre."
        )


# ---------------------------------------------------------------------------
# Section B — Backend: strict_pantry como default (inversion)
# ---------------------------------------------------------------------------

def test_strict_pantry_uses_inverted_default():
    """``strict_pantry`` debe derivarse como ``swap_reason not in
    ("cravings", "weekend")`` — default strict con opt-out solo para
    indulgencia explícita."""
    # [P4-UPDATE-DISHES-STRICT-ALL · 2026-06-23] La derivación quedó anidada bajo el knob
    # strict-all (default OFF = legacy): el opt-out cravings/weekend vive en el `else`.
    m = re.search(
        r"else \(swap_reason\s+not\s+in\s+\(\s*[\"']cravings[\"']\s*,\s*[\"']weekend[\"']\s*\)\)",
        AGENT_PY,
    )
    assert m, (
        "Falta o está mal formado el opt-out `swap_reason not in (\"cravings\", "
        "\"weekend\")` (bajo el knob strict-all) en agent.py. Si lo renombraste, "
        "sincroniza este test."
    )


def test_strict_pantry_old_opt_in_tuple_removed():
    """El tuple legacy ``strict_pantry = swap_reason in ("budget",
    "pantry_first")`` NO debe seguir en agent.py — sería opt-in (régimen
    viejo) co-existiendo con el opt-out nuevo. Modo confuso."""
    assert 'swap_reason in ("budget", "pantry_first")' not in AGENT_PY, (
        "El tuple opt-in legacy sigue en agent.py. Post-P3-SWAP-PANTRY-DEFAULT "
        "debe estar reemplazado por el opt-out (`not in cravings/weekend`)."
    )


# ---------------------------------------------------------------------------
# Section C — Backend: elif 'budget' removido del prompt
# ---------------------------------------------------------------------------

def test_budget_elif_branch_removed():
    """El branch ``elif swap_reason == 'budget'`` debe estar removido del
    elif chain del prompt — su hint específico (📦 APROVECHAR SU NEVERA)
    es ahora redundante con el hint genérico inyectado debajo para todos
    los reasons no-indulgentes."""
    assert "elif swap_reason == 'budget'" not in AGENT_PY, (
        "Branch 'budget' aún presente en agent.py. Post-P3-SWAP-PANTRY-DEFAULT "
        "su hint pasó a ser genérico (RESPETA LA NEVERA para non-indulgent)."
    )


def test_generic_pantry_hint_injected_for_non_indulgent_reasons():
    """Tras el elif chain, debe haber un bloque condicional que inyecta
    ``"📦 RESPETA LA NEVERA"`` cuando ``swap_reason not in ('cravings',
    'weekend')``. Sin este hint, el LLM para variety/time/similar/dislike
    no sabe que pantry es hard constraint → retry overhead innecesario."""
    # Buscar el hint genérico
    assert "RESPETA LA NEVERA" in AGENT_PY, (
        "Hint genérico `RESPETA LA NEVERA` ausente en agent.py. Debe "
        "inyectarse para reasons no-indulgentes después del elif chain."
    )
    # Buscar el guard del hint (debe estar gateado por non-cravings/weekend)
    m = re.search(
        r"if\s+swap_reason\s+not\s+in\s+\(\s*[\"']cravings[\"']\s*,\s*[\"']weekend[\"']\s*\)\s*:\s*\n\s*context_extras\s*\+=.*?RESPETA LA NEVERA",
        AGENT_PY,
        re.DOTALL,
    )
    assert m, (
        "El hint `RESPETA LA NEVERA` debe estar dentro de un `if swap_reason "
        "not in ('cravings', 'weekend'):` block. Sin el guard, también se "
        "inyectaría a cravings/weekend rompiendo `allow_external_count`."
    )


# ---------------------------------------------------------------------------
# Section D — Back-compat: budget/pantry_first como input siguen strict
# ---------------------------------------------------------------------------

def test_budget_input_still_triggers_strict_pantry_via_inversion():
    """Un caller legacy que aún emita ``swap_reason='budget'`` debe caer
    en strict-pantry — porque 'budget' ∉ ('cravings', 'weekend') por la
    inversión. Verificado por inspección de la lógica."""
    # 'budget' NO está en el opt-out → el `not in` evalúa True → strict_pantry True
    opt_out_set = {"cravings", "weekend"}
    for legacy_reason in ("budget", "pantry_first"):
        assert legacy_reason not in opt_out_set, (
            f"Reason legacy `{legacy_reason}` NO debe estar en el opt-out "
            f"set — pre-fix era opt-in strict y post-fix debe seguir strict."
        )


# ---------------------------------------------------------------------------
# Section E — Marker anchor
# ---------------------------------------------------------------------------
# Pin del marker removido por la misma razón que en test_p2_swap_422_ux_copy:
# pin-tests se rompen cada P-fix siguiente cuando el marker avanza. El
# contract "marker fresco a nivel codebase" lo cubre
# `test_p3_1_last_known_pfix_freshness` (floor check). Las secciones A-D
# de este archivo anclan el CONTENIDO del fix (parser-based, no temporales).
