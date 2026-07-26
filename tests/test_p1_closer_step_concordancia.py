"""[P1-CLOSER-STEP-CONCORDANCIA · 2026-07-26] La concordancia la manda el NÚCLEO, no la última
palabra del sintagma.

El paso que el cerrador de proteína añade a la receta decidía número así:

    _plural = _nm_sa.endswith("es") or _nm_sa.endswith("as")   # ← última palabra

y el comentario que lo acompañaba decía *"'res' NO es plural"*. Cierto — pero el nombre que
llega del catálogo no es "res", es **"Carne de res"**, que sí termina en "es". Resultado
entregado al usuario en el plan vivo `0afa0ed5`:

    "Cocina Carne de res a la plancha o hervid**os** y sírve**los** como proteína del plato."
    "Añade Carne de res al guiso… Incorpóra**los** con cuidado para no deshacer el resto."

"Carne" es femenino singular. Y el mismo fallo estaba en la rama de lácteo blando
("mézclalos") y en la de legumbres, donde "incorpóralos" estaba directamente fijo pese a que
habichuelas y lentejas son femeninas.

`_closer_step_agreement` toma el núcleo (lo que va antes del primer " de "; si no hay
complemento, la primera palabra) y devuelve `(plural, femenino)`. El género de los sustantivos
de comida no se deduce de la terminación —"carne" acaba en -e y es femenino, "atún" en -n y es
masculino— así que hay dos conjuntos explícitos para los frecuentes y la regla -a/-as para el
resto, con fallback a masculino singular = comportamiento previo.

tooltip-anchor: P1-CLOSER-STEP-CONCORDANCIA
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g
from constants import strip_accents


def _acuerdo(nombre):
    return g._closer_step_agreement(strip_accents(nombre.lower()))


def _texto(nombre, **kw):
    return g._closer_protein_step_text(nombre, no_cook=False, **kw)


# ───────────── 1. el caso vivo ─────────────

def test_carne_de_res_es_femenino_singular():
    assert _acuerdo("Carne de res") == (False, True)


def test_el_paso_entregado_al_usuario_ya_concuerda():
    t = _texto("Carne de res")
    assert "hervida" in t and "sírvela" in t
    assert "hervidos" not in t and "sírvelos" not in t


def test_la_rama_de_guiso_tambien():
    t = _texto("Carne de res", stewy=True)
    assert "Incorpórala" in t, t
    assert "Incorpóralos" not in t


# ───────────── 2. lo que ya funcionaba debe seguir ─────────────

@pytest.mark.parametrize("nombre,plural,fem", [
    ("Camarones", True, False),
    ("Huevos", True, False),
    ("Filete de res", False, False),
    ("Pechuga de pollo", False, True),
    ("Atún en agua", False, False),
    ("Pollo", False, False),
    ("Habichuelas negras", True, True),
])
def test_nucleos_conocidos(nombre, plural, fem):
    assert _acuerdo(nombre) == (plural, fem), nombre


def test_los_plurales_reales_no_se_degradan():
    """P1-RECIPE-QUALITY-100 cerró "Cocina camarones… o hervido y sírvelo". No debe volver."""
    t = _texto("Camarones")
    assert "hervidos" in t and "sírvelos" in t


# ───────────── 3. las otras dos ramas con el mismo fallo ─────────────

def test_lacteo_blando_concuerda():
    t = g._closer_protein_step_text("Carne de res", no_cook=True)
    assert "mézclala" in t, t
    assert "mézclalos" not in t


def test_legumbre_femenina_concuerda():
    t = _texto("Habichuelas negras")
    assert "incorpóralas" in t, t
    assert "incorpóralos" not in t


# ───────────── 4. bordes ─────────────

@pytest.mark.parametrize("valor", ["", None, "   "])
def test_vacios_caen_a_masculino_singular(valor):
    assert g._closer_step_agreement(valor) == (False, False)


def test_nombre_desconocido_usa_la_regla_general():
    """-a → femenino; cualquier otra terminación → masculino (comportamiento previo)."""
    assert _acuerdo("Tilapia") == (False, True)
    assert _acuerdo("Bacalao") == (False, False)


def test_el_nucleo_se_toma_antes_del_complemento():
    """"Pechuga de pollo": el núcleo es 'pechuga' (f), no 'pollo' (m). Si esto se invierte,
    vuelve la clase entera del bug."""
    assert _acuerdo("Pechuga de pollo")[1] is True
    assert _acuerdo("Filete de res")[1] is False


def test_ninguna_rama_decide_por_la_ultima_palabra():
    """Ancla de la CLASE: si reaparece el `endswith("es")` sobre el nombre completo, vuelve
    "Carne de res → hervidos"."""
    import inspect
    cuerpo = inspect.getsource(g._closer_protein_step_text)
    assert '_nm_sa.endswith("es")' not in cuerpo, \
        "la concordancia debe salir de _closer_step_agreement, no del final de la cadena"
    assert cuerpo.count("_closer_step_agreement(") >= 3, \
        "las tres ramas (lácteo, legumbre, plancha/guiso) deben consultarla"
