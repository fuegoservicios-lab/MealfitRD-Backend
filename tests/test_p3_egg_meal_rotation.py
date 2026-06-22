"""[P3-EGG-MEAL-ROTATION · 2026-06-22] Refuerzo del prompt: el huevo como proteína en MÁXIMO 1
comida/día → rota proteínas desde el PRIMER intento.

Caso visto en vivo (2026-06-22, plan de invitado gain_muscle): el LLM puso huevo en 6 de 12 comidas
→ el gate de variedad (P3-VARIETY-HARD-GATE, `egg_meals > max(3, round(total*0.25))`) lo rechazó →
un retry caro (~90-210s). Causa raíz: el prompt SOLO capeaba la CANTIDAD de huevo/día (≤3 enteros),
pero recomendaba huevo en 2 comidas/día (desayuno + otra) = 6 en el plan → sobre el cap del gate
(~4 de 12). El gate cuenta CADA comida con huevo (entero O claras).

Fix (prompt-only, day_generator.py): se añade un cap explícito de Nº DE COMIDAS con huevo (≤1/día,
idealmente desayuno) + rotación obligatoria a otras proteínas en las demás comidas + la razón (el
gate). El efecto (menos rechazos de variedad desde el intento 1) se valida observando generaciones
en vivo; este test ancla que la directiva existe en el prompt.
"""
from prompts import day_generator


def _src():
    return open(day_generator.__file__, encoding="utf-8").read()


def test_marker_presente():
    assert "P3-EGG-MEAL-ROTATION" in _src()


def test_cap_de_numero_de_comidas_con_huevo():
    src = _src()
    # La directiva debe limitar el NÚMERO de comidas con huevo (no solo la cantidad/día).
    assert "Nº DE COMIDAS" in src or "número de comidas" in src.lower()
    assert "MÁXIMO **1 comida de ESTE DÍA**" in src or "MÁXIMO 1 comida" in src


def test_menciona_que_claras_tambien_cuentan():
    # CLAVE: el gate cuenta huevo en CUALQUIER forma (entero o claras) → el prompt debe avisarlo,
    # si no el LLM usaría claras en muchas comidas y trip-earía igual.
    src = _src()
    assert "entero o claras" in src.lower() or "entero O claras" in src


def test_directiva_de_rotacion_de_proteinas():
    src = _src()
    # Debe mandar ROTAR a otras proteínas en las comidas sin huevo.
    assert "ROTA" in src and ("pollo" in src.lower() and "pescado" in src.lower())


def test_se_conserva_el_cap_de_cantidad():
    # El cap de CANTIDAD pre-existente (≤3 enteros, ≤6 claras) NO debe perderse.
    src = _src()
    assert "3 unidades enteras" in src
    assert "6 claras" in src
