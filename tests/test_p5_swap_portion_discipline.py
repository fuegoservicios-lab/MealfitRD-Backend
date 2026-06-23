"""[P5-SWAP-PORTION-DISCIPLINE · 2026-06-23] El bloque REGLA DE RECICLAJE de `swap_meal`
debe instruir PORCIONES MODERADAS de un solo plato — sin esto el LLM proponía cantidades
grandes y el pantry guard rechazaba por `over_limit` → reintentos (swap lento). Ancla
parser-based: si alguien quita la disciplina de porción, este test falla antes que prod."""
from pathlib import Path

_AGENT = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")


def test_anchor_present():
    assert "P5-SWAP-PORTION-DISCIPLINE" in _AGENT


def test_portion_instruction_in_recycle_rule():
    assert "REGLA DE RECICLAJE" in _AGENT, "el bloque de despensa debe existir"
    assert "porciones MODERADAS" in _AGENT or "porciones moderadas" in _AGENT.lower(), \
        "falta la instrucción de porciones moderadas (causa de los over_limit/retries)"
    assert "inventario es LIMITADO" in _AGENT or "inventario es limitado" in _AGENT.lower(), \
        "el prompt debe recordar que el inventario es limitado (no usar de más)"
