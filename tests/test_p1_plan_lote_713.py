# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-713 · 2026-09-28] Las frases que `/blocked-reasons` manda al usuario no hablan como el código.

`reason_to_text` (routers/plans.py) explicaba un bloque cancelado o detenido con «Chunk cancelado por restore», «tras
una interrupción del worker. El cron lo retomará…» y «El sistema marcó este chunk…». El frontend las pinta TAL CUAL en
español (P1-PLAN-LOTE-225 solo las traduce a los otros idiomas), así que el usuario dominicano leía la jerga del
servidor. Ahora dicen «bloque», «el servidor», «el sistema». Si una frase vuelve a nombrar el mecanismo, este test lo
dice; que el frontend las siga conociendo lo vigila `test_p1_plan_lote_225`.

tooltip-anchor: P1-PLAN-LOTE-713
"""
import ast
import pathlib
import re

_PLANS = pathlib.Path(__file__).resolve().parent.parent / "routers" / "plans.py"
_JERGA = re.compile(r"\b(chunks?|worker|cron|restore|pipeline|fallback|queue)\b", re.IGNORECASE)


def _frases_de_reason_to_text():
    """Los textos de `reason_to_text` y de `_UNKNOWN_REASON_TEMPLATE` (asignaciones dentro del endpoint)."""
    arbol = ast.parse(_PLANS.read_text(encoding="utf-8"))
    frases = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("reason_to_text", "_UNKNOWN_REASON_TEMPLATE") for t in nodo.targets
        ):
            frases += [
                n.value for n in ast.walk(nodo.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and " " in n.value
                and not n.value.startswith("/")
            ]
    assert frases, "no se encontró reason_to_text en routers/plans.py"
    return frases


def test_las_frases_de_blocked_reasons_no_usan_jerga():
    frases = _frases_de_reason_to_text()
    assert frases, "reason_to_text ya no tiene frases: el test no vigilaría nada"
    con_jerga = [f for f in frases if _JERGA.search(f)]
    assert not con_jerga, f"frases al usuario con jerga del servidor: {con_jerga}"


def test_la_frase_de_bloque_interrumpido_es_la_nueva():
    frases = _frases_de_reason_to_text()
    assert "Un bloque del plan se interrumpió en el servidor y quedó pendiente. El sistema lo retomará solo." in frases
