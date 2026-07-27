"""[P1-CLOSER-INTO-MONTAJE · 2026-07-27] El closer 💪 se quedó fuera de la fusión de ayer.

## El caso, visto por el owner en su propio plan

Merienda "Lechosa Fresca con Queso Ricotta y Yogurt":

    [0] Mise en place: Corta lechosa en cubos pequeños.
    [1] 💪 Incorpora yogurt a la preparación y mézclalo antes de servir.
    [2] Montaje: Coloca lechosa en un bowl. Añade el queso ricotta por encima
        y espolvorea canela. Sirve frío.

El paso [1] se lee como lo que es —un parche automático— y el Montaje ni menciona el yogurt.

## Por qué pasaba

`P1-COMPLEMENT-INTO-MONTAJE` (2026-07-26) ya cura esto: en platos fríos funde el ingrediente
dentro del Montaje en vez de dejar un paso suelto. Pero cubrió **un solo camino**, el del
complemento. El closer de proteína inserta su paso por otro sitio y siguió soltándolo.

Medido sobre el plan vivo: verbo de cocción detectado NINGUNO, paso Montaje presente — o sea que
cumplía las dos condiciones de la fusión. Y en 3 h de logs del VPS la fusión se registró **1 vez**,
justamente porque cubría un camino de dos.

Es el patrón de predicados/caminos duplicados que ya cerró P1-PANTRY-GATE-SSOT. Por eso el fix
REUTILIZA `_merge_complement_into_montaje` en vez de escribir una segunda fusión: dos
implementaciones del mismo predicado divergen.

## El gate importante

No basta con "el plato es frío". El paso del closer no siempre significa "añádelo y ya":

    Incorpora X a la preparación y mézclalo   -> fundir es equivalente        ✔
    Agrega X a la licuadora y licúa           -> fundir PIERDE la licuadora   �’
    Cocina X a la plancha o hervido           -> fundir SALTA la cocción      ✗
    Escurre e incorpora X (ya viene cocido)   -> fundir pierde el escurrido   ✗

Se filtra por WORDING (la rama fría «Incorpora … mézclal…»), que es la única equivalente. Misma
asimetría que documenta el helper: un falso positivo cuesta conservar un paso feo pero correcto;
un falso negativo manda al plato un ingrediente sin cocinar.

tooltip-anchor: P1-CLOSER-INTO-MONTAJE
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g


_MONTAJE_FRIO = ("Montaje: Coloca lechosa en un bowl. Añade el queso ricotta por encima "
                 "y espolvorea canela. Sirve frío.")


def _funde(pasos: list, nm: str, *, blended: bool = False, no_cook: bool = True) -> tuple:
    """Reproduce la decisión del closer: (texto_del_paso, fundio?, pasos_resultantes)."""
    paso = f"💪 {g._closer_protein_step_text(nm, no_cook, blended=blended)}"
    rama_fria = ("Incorpora " in paso) and ("mézclal" in paso)
    rec = list(pasos)
    fundio = bool(g.COMPLEMENT_INTO_MONTAJE and rama_fria
                  and g._merge_complement_into_montaje(rec, [nm]))
    return paso, fundio, rec


# ───────────── 1. el caso del owner ─────────────

def test_merienda_fria_funde_en_el_montaje():
    pasos = ["Mise en place: Corta lechosa en cubos pequeños.", _MONTAJE_FRIO]
    _paso, fundio, rec = _funde(pasos, "yogurt")
    assert fundio, "el caso frío del plan vivo DEBE fundirse; era el que salía con paso suelto"
    assert "yogurt" in rec[-1].lower(), f"el Montaje debe nombrar el yogurt: {rec[-1]}"
    assert len(rec) == len(pasos), "fundir no debe añadir pasos"


def test_el_paso_suelto_desaparece():
    """La razón de ser del fix: que el usuario no lea el parche."""
    pasos = ["Mise en place: Corta lechosa en cubos pequeños.", _MONTAJE_FRIO]
    _paso, _f, rec = _funde(pasos, "yogurt")
    assert not any("💪" in s for s in rec), f"quedó el paso suelto: {rec}"


# ───────────── 2. lo que NO debe fundirse (la parte que protege) ─────────────

def test_plato_cocinado_conserva_el_paso():
    """Fundir en el emplatado un plato que se cocina podría saltarse la cocción."""
    pasos = ["Mise en place: Pica la cebolla.",
             "El Toque de Fuego: Sofríe la cebolla y cocina 5 minutos.",
             "Montaje: Sirve caliente."]
    _paso, fundio, _rec = _funde(pasos, "queso cottage")
    assert not fundio, "un plato con cocción NO puede fundir el ingrediente en el emplatado"


def test_licuado_conserva_el_paso():
    """«Agrega a la licuadora y licúa» no es equivalente a «Termina con X»."""
    pasos = ["Mise en place: Pela el guineo.", "Montaje: Sirve el batido en un vaso."]
    paso, fundio, _rec = _funde(pasos, "yogurt", blended=True)
    assert "licuadora" in paso.lower(), f"el wording del licuado cambió: {paso}"
    assert not fundio, "fundir un licuado perdería la instrucción de licuar"


def test_sin_montaje_no_funde():
    pasos = ["Mise en place: Corta la fruta.", "Sirve en un bowl."]
    _paso, fundio, _rec = _funde(pasos, "yogurt")
    assert not fundio, "sin paso Montaje no hay dónde fundir"


def test_si_el_montaje_ya_lo_menciona_no_duplica():
    pasos = ["Mise en place: Corta lechosa.",
             "Montaje: Coloca lechosa y el yogurt en un bowl."]
    _paso, fundio, _rec = _funde(pasos, "yogurt")
    assert not fundio, "el Montaje ya nombra el yogurt: fundir lo duplicaría"


# ───────────── 3. ancla: UNA sola fusión, no dos ─────────────

def test_reutiliza_el_helper_existente():
    """Si alguien escribe una segunda fusión en el closer, los dos predicados divergen — es el
    fallo que P1-PANTRY-GATE-SSOT costó cerrar."""
    import inspect
    src = inspect.getsource(g)
    i = src.index("P1-CLOSER-INTO-MONTAJE")
    bloque = src[i:i + 2600]
    assert "_merge_complement_into_montaje(rec, [nm])" in bloque, (
        "el closer debe REUTILIZAR el helper de fusión, no implementar el suyo"
    )


def test_el_gate_es_por_wording_no_solo_por_plato_frio():
    """Ancla del razonamiento: filtrar solo por 'plato frío' fundiría también el licuado."""
    import inspect
    src = inspect.getsource(g)
    i = src.index("P1-CLOSER-INTO-MONTAJE")
    bloque = src[i:i + 2600]
    assert "_rama_fria" in bloque and "mézclal" in bloque, (
        "el gate por wording desapareció: volvería a fundirse el licuado / lo que se cocina"
    )


def test_el_knob_de_rollback_sigue_gobernando():
    """Mismo knob que la fusión original: un solo interruptor para las dos rutas."""
    import inspect
    src = inspect.getsource(g)
    i = src.index("P1-CLOSER-INTO-MONTAJE")
    assert "COMPLEMENT_INTO_MONTAJE" in src[i:i + 2600]
