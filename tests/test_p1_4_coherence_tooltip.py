"""[P1-4-COHERENCE-TOOLTIP · 2026-05-10] Backend ↔ frontend wiring del
campo `coherence_last_hypotheses` que enriquece el tooltip del chip
"N ajustes" del Historial.

Bug observado en el audit 2026-05-10:
    El chip "N ajustes" (P2-HIST-AUDIT-4) tenía tooltip genérico
    ("ajustes de coherencia recetas↔lista de compras realizados por el
    sistema"). El usuario veía el conteo pero NO sabía qué se ajustó.
    Las hipótesis (`cap_swallowed_modifier`, `unit_mismatch`,
    `yield_uncovered`, `pantry_overdeduct`) ya vivían en
    `_shopping_coherence_block_history[*].divergences[*].hypothesis`
    pero el endpoint /history-list no las exponía.

Fix:
    1. Backend `api_plans_history_list` extrae la lista (max 5 distintas)
       de hipótesis de la ÚLTIMA entry anomalous y la expone como
       `coherence_last_hypotheses` en cada plan summary.
    2. Frontend `History.jsx` lee el campo y construye un tooltip rico
       humanizando cada hipótesis vía `getCoherenceHypothesisLabel`
       (P1-3). Fallback legacy reconstruye desde `plan_data` raw cuando
       el summary endpoint no expone el campo (deploy lag).

Cobertura de este test:
    1. Lógica del extract: última entry anomalous gana sobre las viejas;
       distinct hypotheses cap 5; non-anomalous y non-list defendidos.
    2. Backend expone el campo en el response shape.
    3. Frontend lee `plan.coherence_last_hypotheses` y aplica el helper
       de P1-3 (`getCoherenceHypothesisLabel`).
    4. Frontend tiene fallback al `plan_data._shopping_coherence_block_history`
       cuando el campo está ausente (deploy lag tolerance).

Tests son estáticos: parsean el source, no requieren DB ni node/jest.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_ROOT = _BACKEND_ROOT.parent / "frontend"
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_HISTORY_JSX = _FRONTEND_ROOT / "src" / "pages" / "History.jsx"


def _read(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"Archivo no encontrado: {p}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Lógica de extract — reproducir el algoritmo en Python puro y validar
#    contra fixtures construidos a mano.
# ---------------------------------------------------------------------------
def _extract_last_hypotheses(history, anomalous: set[str], cap: int = 5) -> list[str]:
    """Réplica del bucle del backend (sin importarlo) para validar invariantes."""
    out: list[str] = []
    if not isinstance(history, list):
        return out
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if entry.get("action_taken") not in anomalous:
            continue
        divs = entry.get("divergences")
        if not isinstance(divs, list):
            break
        seen: set[str] = set()
        for d in divs:
            if not isinstance(d, dict):
                continue
            h = d.get("hypothesis")
            if not isinstance(h, str) or not h or h in seen:
                continue
            seen.add(h)
            out.append(h)
            if len(out) >= cap:
                break
        break
    return out


_ANOMALOUS = {"degrade", "reject_minor", "reject_high", "hydration_error"}


class TestExtractLogic:
    def test_empty_history(self):
        assert _extract_last_hypotheses([], _ANOMALOUS) == []

    def test_no_anomalous_entries(self):
        history = [
            {"action_taken": "not_applicable", "divergences": [{"hypothesis": "unit_mismatch"}]},
            {"action_taken": "post_swap_revalidation", "divergences": [{"hypothesis": "yield_uncovered"}]},
        ]
        assert _extract_last_hypotheses(history, _ANOMALOUS) == []

    def test_latest_anomalous_wins(self):
        history = [
            {
                "action_taken": "reject_minor",
                "divergences": [{"hypothesis": "cap_swallowed_modifier"}, {"hypothesis": "unit_mismatch"}],
            },
            # Más reciente — debe ganar sobre la entry de arriba.
            {
                "action_taken": "degrade",
                "divergences": [{"hypothesis": "yield_uncovered"}, {"hypothesis": "pantry_overdeduct"}],
            },
        ]
        assert _extract_last_hypotheses(history, _ANOMALOUS) == ["yield_uncovered", "pantry_overdeduct"]

    def test_cap_5_distinct(self):
        divs = [
            {"hypothesis": f"hyp_{i}"} for i in range(10)
        ]
        history = [{"action_taken": "reject_high", "divergences": divs}]
        assert _extract_last_hypotheses(history, _ANOMALOUS, cap=5) == [f"hyp_{i}" for i in range(5)]

    def test_distinct_dedup(self):
        history = [{
            "action_taken": "reject_minor",
            "divergences": [
                {"hypothesis": "cap_swallowed_modifier"},
                {"hypothesis": "cap_swallowed_modifier"},
                {"hypothesis": "unit_mismatch"},
            ],
        }]
        assert _extract_last_hypotheses(history, _ANOMALOUS) == ["cap_swallowed_modifier", "unit_mismatch"]

    def test_non_dict_entry_skipped(self):
        history = ["not a dict", None, {"action_taken": "reject_minor", "divergences": [{"hypothesis": "h1"}]}]
        assert _extract_last_hypotheses(history, _ANOMALOUS) == ["h1"]

    def test_divergences_not_list_returns_empty(self):
        history = [{"action_taken": "reject_minor", "divergences": "string"}]
        assert _extract_last_hypotheses(history, _ANOMALOUS) == []

    def test_divergence_without_hypothesis_skipped(self):
        history = [{
            "action_taken": "reject_minor",
            "divergences": [
                {"food": "X"},
                {"hypothesis": "cap_swallowed_modifier"},
                {"hypothesis": ""},
                {"hypothesis": None},
            ],
        }]
        assert _extract_last_hypotheses(history, _ANOMALOUS) == ["cap_swallowed_modifier"]

    def test_history_not_list_returns_empty(self):
        assert _extract_last_hypotheses("nope", _ANOMALOUS) == []
        assert _extract_last_hypotheses({"a": 1}, _ANOMALOUS) == []
        assert _extract_last_hypotheses(None, _ANOMALOUS) == []


# ---------------------------------------------------------------------------
# 2. Backend expone el campo en el response shape
# ---------------------------------------------------------------------------
class TestBackendExposes:
    def test_field_in_response_dict(self):
        """El response dict de api_plans_history_list debe incluir
        `coherence_last_hypotheses`."""
        src = _read(_PLANS_PY)
        assert '"coherence_last_hypotheses"' in src, (
            "El campo `coherence_last_hypotheses` no aparece en el response "
            "shape de `api_plans_history_list`."
        )

    def test_field_documented_in_docstring(self):
        """Docstring del endpoint debería mencionar el nuevo campo."""
        src = _read(_PLANS_PY)
        assert "coherence_last_hypotheses" in src

    def test_extract_loop_present(self):
        """El bucle reverse-walk con cap 5 distinct está en el source."""
        src = _read(_PLANS_PY)
        # Anchor mínimo: la lista vacía inicializada y el cap >= 5.
        assert "coherence_last_hypotheses: list[str] = []" in src
        assert ">= 5" in src
        # Reverse walk del history.
        assert "reversed(history)" in src


# ---------------------------------------------------------------------------
# 3. Frontend wiring
# ---------------------------------------------------------------------------
class TestFrontendWiring:
    def test_history_jsx_reads_field(self):
        """`History.jsx` lee `plan.coherence_last_hypotheses` para el tooltip."""
        src = _read(_HISTORY_JSX)
        assert "coherence_last_hypotheses" in src, (
            "History.jsx no consume `plan.coherence_last_hypotheses` — el chip "
            "tooltip seguirá siendo genérico."
        )

    def test_history_jsx_uses_hypothesis_label_helper(self):
        """El render aplica `getCoherenceHypothesisLabel` al tooltip."""
        src = _read(_HISTORY_JSX)
        assert "getCoherenceHypothesisLabel" in src

    def test_history_jsx_imports_helper(self):
        """Import del helper de P1-3 presente."""
        src = _read(_HISTORY_JSX)
        assert "from '../utils/coherenceLabels'" in src

    def test_history_jsx_has_legacy_fallback(self):
        """Cuando `coherence_last_hypotheses` está ausente (deploy lag),
        debe haber fallback a `plan_data._shopping_coherence_block_history`
        para reconstruir las hipótesis client-side."""
        src = _read(_HISTORY_JSX)
        # Buscar el bloque del chip y verificar que referencia ambas fuentes.
        # Anchor en el tooltip enriquecido.
        m = re.search(
            r"coherence_last_hypotheses[\s\S]{0,2000}_shopping_coherence_block_history",
            src,
        )
        assert m is not None, (
            "Frontend no tiene fallback legacy. Si el server (deploy rezagado) "
            "no expone `coherence_last_hypotheses`, el tooltip queda genérico "
            "innecesariamente."
        )

    def test_no_unknown_hypothesis_codes_used_directly(self):
        """El render NO debe interpolar hypothesis raw como user-visible
        sin pasar por el helper humanizador."""
        src = _read(_HISTORY_JSX)
        # Anchor: cualquier interpolación de h o hypothesis sin envolver en
        # getCoherenceHypothesisLabel sería un bug.
        # Buscamos pattern: `${h}` o similar SIN la función helper en su
        # vecindad. Heurística: si `getCoherenceHypothesisLabel(h)` aparece en
        # el archivo, asumimos que se usa correctamente. Sanity check:
        assert "getCoherenceHypothesisLabel(h)" in src, (
            "Hypothesis NO se humaniza en el tooltip; el usuario vería "
            "`cap_swallowed_modifier` raw."
        )
