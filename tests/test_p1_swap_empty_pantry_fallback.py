"""[P1-SWAP-EMPTY-PANTRY-FALLBACK · 2026-05-22] Test del fallback a
`aggregated_shopping_list` cuando la pantry virtual está vacía.

Pre-fix (GAP-1 audit production-readiness 2026-05-22):
    `agent.py::swap_meal` solo intentaba 2 fuentes para `clean_ingredients`:

      1. `get_realtime_pantry(plan_data, consumed_ingredients)` — calcula
         pantry virtual descontando los consumidos del aggregate del plan.
         Si TODO se consumió, retorna [].
      2. `form_data["current_pantry_ingredients"]` / `current_shopping_list` —
         lo que el frontend envía. Si el frontend NO incluye este campo
         (clientes legacy / contexto restaurado del historial), también [].

    Cuando ambas eran [] Y `swap_reason` NO era strict (variety, taste,
    allergy, etc.), el fallback caía al hardcoded `["Pollo", "Arroz",
    "Aguacate"]` ignorando la `aggregated_shopping_list` del plan — el PDF
    que el user TIENE en su mano como su "nevera futura comprometida".

    Requisito del owner audit 2026-05-22:
    > "si la nevera está vacía debe tomar en cuenta la lista de compras
    > pdf para crear los platos personalizados"

Cierre P1-SWAP-EMPTY-PANTRY-FALLBACK:
    En `agent.py::swap_meal`, ANTES del log warning "FREE_GENERATION",
    si `clean_ingredients == []` Y el user no es guest, se hace fetch
    explícito del último plan del user y se lee
    `plan_data['aggregated_shopping_list']` como fallback. Knob
    `MEALFIT_SWAP_EMPTY_PANTRY_FALLBACK_TO_SHOPPING_LIST=false`
    desactiva (vuelve al comportamiento legacy hardcoded).

    Espejo del patrón ya implementado en
    `tools.py::execute_modify_single_meal:570-576` (que mezcla pantry +
    aggregated SIEMPRE — el chat-agent ya cumplía el requisito).

Tooltip-anchor: P1-SWAP-EMPTY-PANTRY-FALLBACK | audit 2026-05-22
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PY = _REPO_ROOT / "backend" / "agent.py"
_TOOLS_PY = _REPO_ROOT / "backend" / "tools.py"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def swap_meal_body(agent_src: str) -> str:
    """Extrae el cuerpo de `def swap_meal(form_data: dict):` hasta el
    siguiente `def ` de top-level."""
    idx = agent_src.find("def swap_meal(form_data: dict):")
    assert idx > 0, "def swap_meal no encontrado en agent.py"
    # Limitamos a ~12kb para captar la función entera (es larga)
    next_def = re.search(r"\ndef\s+[a-zA-Z_]", agent_src[idx + 100:])
    end = (idx + 100 + next_def.start()) if next_def else len(agent_src)
    return agent_src[idx:end]


# ---------------------------------------------------------------------------
# Section A — Fallback wiring en agent.py::swap_meal
# ---------------------------------------------------------------------------
class TestAgentSwapMealFallback:

    def test_marker_present(self, swap_meal_body: str):
        """El comment block que documenta el fix DEBE estar presente
        (tooltip-anchor para que un renombre falle el test antes que
        producción)."""
        assert "P1-SWAP-EMPTY-PANTRY-FALLBACK" in swap_meal_body, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK regresión: el comment block "
            "que documenta el fix fue removido del cuerpo de swap_meal. "
            "Restaurar el comment block que sirve de tooltip-anchor."
        )

    def test_reads_aggregated_shopping_list_from_plan_data(
        self, swap_meal_body: str
    ):
        """El fallback DEBE leer `aggregated_shopping_list` del plan_data
        del user (no del form_data — eso es el fallback anterior)."""
        # Verificamos que dentro del fallback se invoca
        # `get_latest_meal_plan_with_id` Y se lee
        # `aggregated_shopping_list` del plan_data.
        assert "get_latest_meal_plan_with_id" in swap_meal_body, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK regresión: el fallback no "
            "carga el plan del user via `get_latest_meal_plan_with_id`. "
            "Sin esto, no hay forma de leer `aggregated_shopping_list`."
        )
        assert "aggregated_shopping_list" in swap_meal_body, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK regresión: `aggregated_"
            "shopping_list` no se referencia. El fallback fue removido "
            "o degradado al hardcoded legacy."
        )

    def test_knob_consulted_for_kill_switch(self, swap_meal_body: str):
        """El knob `MEALFIT_SWAP_EMPTY_PANTRY_FALLBACK_TO_SHOPPING_LIST`
        DEBE consultarse antes del fallback, para que se pueda desactivar
        sin redeploy si introduce regresiones."""
        assert "MEALFIT_SWAP_EMPTY_PANTRY_FALLBACK_TO_SHOPPING_LIST" in swap_meal_body, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK regresión: el knob de "
            "kill-switch no se consulta. El fix está hardcoded sin rollback "
            "rápido — viola la convención del repo "
            "(`Cambios de comportamiento que pueden necesitar revertirse "
            "sin redeploy van como knob`)."
        )

    def test_fallback_runs_only_after_existing_paths_failed(
        self, swap_meal_body: str
    ):
        """El bloque del nuevo fallback DEBE aparecer DESPUÉS de los dos
        existing paths (`get_realtime_pantry` + `current_pantry_ingredients`).
        Si va antes, sobrescribe la pantry real del user con la lista del
        plan — bug funcional."""
        idx_realtime = swap_meal_body.find("get_realtime_pantry")
        idx_frontend = swap_meal_body.find("current_pantry_ingredients")
        idx_fallback = swap_meal_body.find(
            "MEALFIT_SWAP_EMPTY_PANTRY_FALLBACK_TO_SHOPPING_LIST"
        )
        assert idx_realtime > 0
        assert idx_frontend > 0
        assert idx_fallback > 0
        assert idx_fallback > idx_realtime, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK regresión: el fallback corre "
            "ANTES de `get_realtime_pantry`. Eso ignora la pantry real "
            "(la deducción consumed) y propondría platos con ingredientes "
            "ya gastados."
        )
        assert idx_fallback > idx_frontend, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK regresión: el fallback corre "
            "ANTES del fallback al `current_pantry_ingredients` del "
            "frontend. Eso ignora el state del cliente (potencialmente "
            "más actualizado que el plan persistido en BD)."
        )

    def test_fallback_only_for_authenticated_users(
        self, swap_meal_body: str
    ):
        """Guests (`user_id == 'guest'`) NO deben golpear BD — no tienen
        plan persistido y la query sería desperdicio. El fallback debe
        gatearse por `user_id != 'guest'`."""
        # El bloque del fallback debe contener una condición que excluya
        # guests. Buscamos `user_id != "guest"` o `user_id and user_id !=`
        # dentro del fallback (después del marker).
        idx = swap_meal_body.find("P1-SWAP-EMPTY-PANTRY-FALLBACK")
        assert idx > 0
        block = swap_meal_body[idx:idx + 2500]
        has_guest_gate = (
            'user_id != "guest"' in block
            or "user_id != 'guest'" in block
            or "user_id and user_id" in block
        )
        assert has_guest_gate, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK regresión: el fallback no "
            "excluye explícitamente a guests. Eso resulta en query "
            "innecesaria a BD para guest sessions."
        )


# ---------------------------------------------------------------------------
# Section B — Sanity en tools.py (el surface del chat-agent ya cumplía)
# ---------------------------------------------------------------------------
class TestToolsAlreadyCompliant:
    """`tools.py::execute_modify_single_meal` ya combinaba pantry física
    + aggregated_shopping_list. Solo verificamos que el patrón sigue ahí
    (no regresionó por refactor ortogonal)."""

    def test_tools_reads_aggregated_in_modify(self, tools_src: str):
        idx_fn = tools_src.find("def execute_modify_single_meal(")
        assert idx_fn > 0
        # Captura el bloque inicial de extracción de clean_ingredients
        body = tools_src[idx_fn:idx_fn + 4000]
        assert "aggregated_shopping_list" in body, (
            "P1-SWAP-EMPTY-PANTRY-FALLBACK (tools.py sanity) regresión: "
            "`execute_modify_single_meal` ya no lee "
            "`aggregated_shopping_list` del plan_data — el chat-agent "
            "perdió la cobertura del fallback que el bundle preserva."
        )


# ---------------------------------------------------------------------------
# Section C — Marker cross-link
# ---------------------------------------------------------------------------
def test_marker_anchor_filename():
    expected_slug = "p1_swap_empty_pantry_fallback"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "Filename debe contener `p1_swap_empty_pantry_fallback`."
    )
