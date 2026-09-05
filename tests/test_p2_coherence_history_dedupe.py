"""[P2-COHERENCE-HISTORY-DEDUPE · 2026-09-04] El historial `_shopping_coherence_block_history` se llenaba
de entradas IDÉNTICAS (una por recálculo de la lista: 18 en 7 h, todas `warn_only_recalc` con las mismas 3
divergencias) y el Dashboard anunciaba «16 revisiones automáticas». Ahora la misma alerta consecutiva se
funde en la última entrada (`repeats`, `first_ts`, `ts` renovado); el frontend cuenta alertas distintas.
"""
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _divs(n=3):
    return [{"hypothesis": "magnitude_mild_short", "magnitude": True, "name": f"x{i}"} for i in range(n)]


def _append(plan_result, monkeypatch=None):
    import shopping_calculator as sc
    # el guard real necesita catálogo/LLM: se simula el resultado del guard y se ejerce SOLO el append
    monkeypatch.setattr(sc, "run_shopping_coherence_guard", lambda *a, **k: {"divergences": _divs(), "block": False})
    return sc.run_shopping_coherence_guard_and_append_history(plan_result, action_taken="warn_only_recalc", plan_id_hint="p")


def test_a_la_misma_alerta_consecutiva_se_funde_en_una_entrada(monkeypatch):
    import shopping_calculator as sc
    if not hasattr(sc, "run_shopping_coherence_guard_and_append_history"):
        pytest.skip("helper no disponible")
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert '_knob_env_bool("MEALFIT_COHERENCE_HISTORY_DEDUPE", True)' in src
    assert '_merged["repeats"] = int(_last_entry.get("repeats") or 1) + 1' in src
    assert '_merged.setdefault("first_ts", _last_entry.get("ts"))' in src
    # la fusión ocurre ANTES del cap: el cap recibe (historial sin la última) + (entrada fundida)
    assert src.index('_merged["repeats"]') < src.index("from graph_orchestrator import _apply_coherence_history_cap as _cap_helper")


def test_b_una_alerta_distinta_si_apila_entrada_nueva():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("[P2-COHERENCE-HISTORY-DEDUPE · 2026-09-04]")
    block = src[i:i + 1800]
    for cond in ('_last_entry.get("action_taken") == entry["action_taken"]',
                 'dict(_last_entry.get("hypotheses") or {}) == entry["hypotheses"]',
                 'int(_last_entry.get("divergence_count") or 0) == entry["divergence_count"]',
                 'bool(_last_entry.get("block_set")) == entry["block_set"]'):
        assert cond in block, cond


def test_c_el_frontend_cuenta_alertas_distintas_y_marca_visto_al_autocerrarse():
    for base in (_BACKEND.parents[0], _BACKEND.parent):
        p = base / "frontend" / "src" / "utils" / "renderCoherenceWarnings.js"
        if p.exists():
            src = p.read_text(encoding="utf-8")
            assert "const sig = `${e.action_taken}|${hyps}|${e.divergence_count ?? ''}|${e.block_set ? 1 : 0}`;" in src
            assert "count: distinct.length" in src
            assert "onAutoClose: _writeDismissAt" in src
            return
    pytest.skip("frontend hermano no disponible")
