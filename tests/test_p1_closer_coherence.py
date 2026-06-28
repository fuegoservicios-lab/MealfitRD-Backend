"""[P1-CLOSER-COHERENCE · 2026-06-27] El cerrador de proteína inyectaba un 2º queso INCOHERENTE (mozzarella) a un
batido que ya tenía ricotta, "como fuente principal de proteína", con cantidades absurdas (4.73g) y etiqueta
duplicada "(4.73g)". Detectado en vivo (captura del usuario: "Batido Refrescante de Lechosa con Queso Ricotta").

6 fixes (workflow de diagnóstico + verificación adversaria):
 A. _NO_COOK_SAFE_PROTEIN_HINT: solo quesos BATIBLES (excluye mozzarella/cheddar/parmesano/gouda/de freír).
 B. Congruencia del closer por token ESPECÍFICO (ricotta/mozzarella), no el genérico "queso" → escala el MISMO queso.
 C. Sin hint duplicado "(Ng)" en los 3 sites del closer + quantize FINAL a gramos humanos (múltiplo de 5).
 D. _GEN_INCONGRUENT_IN_BATIDO dropea quesos firmes de un batido (backstop si el LLM los genera).
 E. Wording "reforzar la proteína" (no "fuente principal") en comida ligera/batido.
tooltip-anchor: P1-CLOSER-COHERENCE
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---- FIX A ----
def test_a_no_cook_excludes_firm_cheeses():
    hints = g._NO_COOK_SAFE_PROTEIN_HINT
    assert not any(h in "queso mozzarella" for h in hints), "mozzarella NO debe ser no-cook-safe"
    for firm in ("queso parmesano", "queso cheddar", "queso gouda", "queso de freir", "queso de hoja"):
        assert not any(h in firm for h in hints), f"{firm} NO debe ser no-cook-safe"


def test_a_no_cook_keeps_blendable():
    hints = g._NO_COOK_SAFE_PROTEIN_HINT
    for soft in ("queso ricotta", "queso cottage", "queso crema", "yogurt griego"):
        assert any(h in soft for h in hints), f"{soft} SÍ debe seguir no-cook-safe"


# ---- FIX B ----
def test_b_congruence_specific_token_not_generic():
    """Con un batido que tiene ricotta: 'mozzarella' NO debe declarar congruencia (token 'queso' es genérico);
    'ricotta' SÍ (escala el mismo queso)."""
    meal_text = "batido de lechosa con queso ricotta 1 cda de queso ricotta canela"
    moz_toks = [t for t in "queso mozzarella".split() if len(t) >= 4 and t not in g._CLOSER_GENERIC_PROTEIN_WORDS]
    ric_toks = [t for t in "queso ricotta".split() if len(t) >= 4 and t not in g._CLOSER_GENERIC_PROTEIN_WORDS]
    assert moz_toks == ["mozzarella"] and ric_toks == ["ricotta"]
    assert not any(t in meal_text for t in moz_toks), "mozzarella no debe ser congruente con un plato de ricotta"
    assert any(t in meal_text for t in ric_toks), "ricotta sí debe ser congruente"


def test_b_generic_words_blacklisted():
    for w in ("queso", "carne", "pescado", "yogur", "yogurt"):
        assert w in g._CLOSER_GENERIC_PROTEIN_WORDS


def test_b_knob_and_logic_present():
    assert "CLOSER_CONGRUENCE_FULLNAME" in _SRC
    assert "_CLOSER_GENERIC_PROTEIN_WORDS" in _SRC


# ---- FIX C ----
def test_c_no_duplicate_gram_hint_in_closer_sites():
    """Ninguno de los 3 sites del closer debe seguir construyendo el hint duplicado '(Ng)'."""
    assert 'g de {name_disp} ({grams}g)"' not in _SRC
    assert 'g de {nm}{cook} ({grams}g)"' not in _SRC
    assert "({grams_food}g)\"" not in _SRC


def test_c_final_quantize_after_protein_floor():
    body = _SRC[_SRC.index("async def assemble_plan_node"):]
    body = body[:body.index("\nasync def ", 5)] if "\nasync def " in body[5:] else body
    i_floor = body.index("_repair_protein_floor_post_caps(days")
    i_quant = body.index("_apply_portion_quantization({\"days\": days}")
    assert i_floor < i_quant, "el quantize final debe correr DESPUÉS del re-cierre de proteína (última mutación)"
    assert "ASSEMBLE_FINAL_QUANTIZE" in _SRC


# ---- FIX D ----
def test_d_firm_cheeses_droppable_from_batido():
    inc = g._GEN_INCONGRUENT_IN_BATIDO
    for firm in ("mozzarella", "parmesano", "cheddar", "gouda", "queso de freir", "queso de hoja"):
        assert firm in inc, f"{firm} debe ser dropeable de un batido"


def test_d_blendable_cheeses_not_dropped():
    inc = g._GEN_INCONGRUENT_IN_BATIDO
    for soft in ("ricotta", "cottage", "queso crema"):
        assert soft not in inc, f"{soft} NO debe dropearse de un batido (es batible)"


# ---- FIX E ----
def test_e_light_meal_wording_reforzar():
    assert "para reforzar la proteína de esta comida" in _SRC
    # ya no se afirma "fuente principal" de forma incondicional en el closer ligero
    assert "como fuente principal de \n" not in _SRC
