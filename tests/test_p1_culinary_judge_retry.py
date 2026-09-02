"""[P1-CULINARY-JUDGE-RETRY · 2026-09-02] El juez culinario reintenta como el resto de nodos.

Medido en prod (02-sep): 6 errores de proveedor en el día, 4 cayeron en el juez (fail-open,
plan sin juicio) porque era la ÚNICA llamada con `max_retries=0`; los otros nodos se
recuperaron con los 15 reintentos del cliente. Knob `MEALFIT_CULINARY_JUDGE_MAX_RETRIES`
(default 1, clamp [0, 3]); el `asyncio.wait_for` exterior escala con los intentos, porque
con el timeout fijo de antes habría matado el reintento antes de que sirviera.

Tooltip-anchor: P1-CULINARY-JUDGE-RETRY | max_retries=CULINARY_JUDGE_MAX_RETRIES
"""
import importlib
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "graph_orchestrator.py").read_text(encoding="utf-8")


def _judge_block() -> str:
    i = SRC.index("tooltip-anchor: P1-CULINARY-JUDGE\"\"\"")
    j = SRC.index("except Exception as _cj_e:", i)
    return SRC[i:j]


def test_both_judge_clients_use_the_knob_not_zero():
    blk = _judge_block()
    assert "max_retries=0" not in blk, "el juez volvió a max_retries=0 (fail-open ante cualquier hipo)"
    assert blk.count("max_retries=CULINARY_JUDGE_MAX_RETRIES") == 2, "las DOS construcciones (thinking / no-thinking)"


def test_outer_wait_for_scales_with_retries():
    blk = _judge_block()
    assert re.search(r"asyncio\.wait_for\(_judge\.ainvoke\(_msg\), timeout=CULINARY_JUDGE_TIMEOUT_S \* \(1 \+ CULINARY_JUDGE_MAX_RETRIES\) \+ 5\)", blk),         "el timeout exterior debe caber (1 + reintentos) intentos"


def test_knob_default_and_clamp(monkeypatch):
    go = importlib.import_module("graph_orchestrator")
    assert go.CULINARY_JUDGE_MAX_RETRIES == 1
    monkeypatch.setenv("MEALFIT_CULINARY_JUDGE_MAX_RETRIES", "9")
    assert max(0, min(3, go._env_int("MEALFIT_CULINARY_JUDGE_MAX_RETRIES", 1))) == 3
    monkeypatch.setenv("MEALFIT_CULINARY_JUDGE_MAX_RETRIES", "-2")
    assert max(0, min(3, go._env_int("MEALFIT_CULINARY_JUDGE_MAX_RETRIES", 1))) == 0


def test_knob_registered():
    go = importlib.import_module("graph_orchestrator")
    assert "MEALFIT_CULINARY_JUDGE_MAX_RETRIES" in go.get_knobs_registry_snapshot()
