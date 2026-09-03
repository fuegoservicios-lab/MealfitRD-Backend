# [P1-SWAP-EMPTY-PANTRY-WAIVER · 2026-08-09] El swap heredaba el modo estricto global
# (D2: strict todos los motivos) pero NO la exención de nevera-vacía que la GENERACIÓN
# tiene a propósito: un guest (o Nevera real sin ítems) recibía un universo imposible,
# el LLM agotaba retries y moría en SWAP_STRICT_PANTRY_NO_INVENTORY. Primera medición
# de la telemetría resucitada (2026-08-09): 4/4 fallos guest con ese ÚNICO error_code
# (36% de los swaps). Mejora REAL de producto: guests y usuarios nuevos pueden cambiar
# platos; el estricto sigue intacto para quien SÍ tiene Nevera.
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_AGENT = open(os.path.join(os.path.dirname(__file__), "..", "agent.py"), encoding="utf-8").read()


def test_waiver_existe_antes_del_uso_del_universo():
    # [P0-SWAP-WAIVER-UNBOUND · 2026-08-09] La versión original de este test validaba
    # PROXIMIDAD a «if clean_ingredients:» — había VARIOS y el waiver quedó anclado al
    # equivocado, ~400 líneas ANTES de que `strict_pantry` naciera → UnboundLocalError →
    # 500 en el 100% de los swaps durante ~1 día. El ORDEN correcto (nace strict_pantry →
    # waiver → raise honesto) lo verifica por AST test_p0_swap_waiver_unbound.py; aquí
    # queda la existencia + kill switch + que desactiva de verdad.
    i = _AGENT.find("P1-SWAP-EMPTY-PANTRY-WAIVER")
    assert i > 0, "el waiver de nevera vacía desapareció del swap"
    blk = _AGENT[i: i + 1800]
    assert "strict_pantry = False" in blk, "el waiver debe DESACTIVAR strict, no solo loguear"
    assert "MEALFIT_SWAP_EMPTY_PANTRY_WAIVER" in blk, "kill switch obligatorio"


def test_condicion_exacta_strict_y_vacio():
    m = re.search(r"if strict_pantry and not clean_ingredients and _swap_empty_waiver_on:", _AGENT)
    assert m, ("la condición debe ser strict Y universo vacío Y knob ON — "
               "cualquier otra forma desactivaría strict con nevera poblada (error simétrico)")


def test_raise_honesto_sigue_vivo():
    # Para strict con Nevera NO vacía cuyo LLM agote intentos, el soft-fail honesto se queda:
    # quitar el raise habría sido el error simétrico (fallback deshonesto con nevera real).
    assert "SWAP_STRICT_PANTRY_NO_INVENTORY: el usuario eligió una razón" in _AGENT
