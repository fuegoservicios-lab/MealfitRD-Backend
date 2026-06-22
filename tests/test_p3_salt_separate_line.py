"""[P3-SALT-SEPARATE-LINE · 2026-06-22] Sal y pimienta deben ir en renglones SEPARADOS.

Bug observado en vivo (plan e819b76b, 2026-06-22): las recetas usaban "Sal y pimienta al gusto" como UN
solo renglón de `ingredients` en 7 comidas, pero la lista de compras no tenía SAL — el resolver del
shopping mapea cada renglón a UN alimento y "sal y pimienta" → solo "Pimienta negra", perdiendo la sal.

Fix: (1) prompt del day-gen pide emitir Sal y Pimienta separadas; (2) backstop determinista en el day-gen
que separa cualquier renglón que mencione sal Y pimienta en dos ingredientes.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PROMPT = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")


def test_prompt_requires_separate_condiment_lines():
    assert "P3-SALT-SEPARATE-LINE" in _PROMPT
    low = _PROMPT.lower()
    assert "sal y pimienta" in low, "el prompt debe nombrar el anti-patrón 'sal y pimienta' a evitar"
    assert "sal al gusto" in low and "pimienta negra al gusto" in low


def test_daygen_has_deterministic_split_backstop():
    assert "P3-SALT-SEPARATE-LINE" in _GRAPH, "falta el backstop determinista en el day-gen"
    # El split aparece DENTRO de generate_days_parallel_node (post-proceso por comida).
    start = _GRAPH.find("async def generate_days_parallel_node(")
    body = _GRAPH[start: start + 60000]
    assert 'append("Sal al gusto")' in body
    assert 'append("Pimienta negra al gusto")' in body
    assert r"\bsal\b" in body and r"\bpimienta\b" in body


def test_word_boundary_safety():
    """`\\bsal\\b` debe separar 'sal y pimienta' pero NO disparar en 'salsa'/'ensalada' (falsos positivos)."""
    sal_re = re.compile(r"\bsal\b")
    pim_re = re.compile(r"\bpimienta\b")
    # Caso real → split
    assert sal_re.search("sal y pimienta negra al gusto") and pim_re.search("sal y pimienta negra al gusto")
    # Falsos positivos que NO deben tratarse como combo sal+pimienta:
    assert not sal_re.search("salsa de soya")           # 'salsa' no es 'sal'
    assert not sal_re.search("ensalada de lechuga")     # 'ensalada' no es 'sal'
    # 'salsa y pimienta' tiene pimienta pero NO 'sal' standalone → no se fuerza el split de sal
    assert not sal_re.search("salsa picante y pimienta")
