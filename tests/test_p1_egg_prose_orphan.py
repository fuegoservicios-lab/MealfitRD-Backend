"""[P1-EGG-PROSE-ORPHAN · 2026-07-27] El wrap cocinaba el yogur como si fueran huevos.

## Lo que veía el owner

    Vierte yogurt griego entero BATIDAS y cocina 2-3 minutos, revolviendo suavemente,
    hasta que CUAJEN.
    …coloca encima yogurt griego entero REVUELTAS con kale.

Esos femeninos plurales apuntan a unos huevos que ya no están en el plato.

## La causa

`RECIPE-COHERENCE-AUTOFIX` sustituye la mención huérfana con `_orphan_pat`, que matchea **solo el
sustantivo** (`\\b(?:huevo)(?:es|s)?\\b`). Cambia el alimento y deja intactos el participio y el
verbo que concordaban con él.

Medido sobre 164 comidas de 14 planes vivos: **2 (1.2%)**, y una de las dos es de las que el guard
de honestidad de nombre marcó como degradadas — la misma familia que las 14 degradaciones por
'queso' ausente de los logs.

## Por qué se BORRA el participio y no se concuerda

"yogurt batido" seguiría siendo falso: nadie bate el yogur para que cuaje. El verbo sí admite un
neutro ("hasta que tome consistencia").

⚠️ NO es lo mismo que `_EGG_ADJ_CONCORD_RX` (P1-RECIPE-AUDIT-6): aquel arregla la concordancia
cuando el huevo **sigue presente** ("el huevo fritos" → "el huevo frito"). Aquí el huevo ya no
existe y la palabra sobra entera.

tooltip-anchor: P1-EGG-PROSE-ORPHAN
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g
from constants import strip_accents as _sa


R = g._reparar_concordancia_huevo


# ───────────── 1. los dos pasos reales del owner ─────────────

def test_el_wrap_del_owner():
    out = R("Vierte yogurt griego entero batidas y cocina por 2-3 minutos, "
            "revolviendo suavemente, hasta que cuajen.", ["huevo"], _sa)
    assert "batidas" not in out
    assert "cuajen" not in out
    assert "tome consistencia" in out
    assert "yogurt griego entero" in out, "el alimento real no se puede perder"


def test_el_montaje_del_owner():
    out = R("Coloca encima yogurt griego entero revueltas con kale, enrolla firmemente.",
            ["huevo"], _sa)
    assert "revueltas" not in out
    assert "yogurt griego entero" in out and "kale" in out


@pytest.mark.parametrize("palabra", ["batidas", "batidos", "revueltas", "revueltos",
                                     "cuajadas", "pochados"])
def test_participios_de_huevo_se_retiran(palabra):
    out = R(f"Vierte el yogurt {palabra} en la sartén.", ["huevo"], _sa)
    assert palabra not in out


def test_tambien_con_claras_como_huerfano():
    out = R("Mezcla el yogurt batidas hasta que cuaje.", ["claras de huevo"], _sa)
    assert "batidas" not in out and "cuaje" not in out


# ───────────── 2. el plato que SÍ lleva huevo no se toca ─────────────

@pytest.mark.parametrize("paso", [
    "Vierte los huevos batidos y cocina hasta que cuajen.",
    "Sirve los huevos revueltos con cilantro.",
    "Bate el huevo con una pizca de sal y pimienta.",
])
def test_si_el_huerfano_no_es_huevo_no_se_toca(paso):
    """Ancla de la CLASE: en un plato con huevo, 'batidos' y 'hasta que cuajen' son CORRECTOS.
    Aplicar la reparación sin el gate destrozaría prosa buena."""
    assert R(paso, ["camaron"], _sa) == paso


@pytest.mark.parametrize("claves", [None, [], ["pollo"], ["queso blanco"]])
def test_sin_huevo_entre_los_huerfanos_es_identidad(claves):
    paso = "Vierte los huevos batidos y cocina hasta que cuajen."
    assert R(paso, claves, _sa) == paso


# ───────────── 3. fail-safe e idempotencia ─────────────

def test_idempotente():
    una = R("Vierte yogurt batidas hasta que cuajen.", ["huevo"], _sa)
    assert R(una, ["huevo"], _sa) == una


@pytest.mark.parametrize("basura", [None, 123, "", "   "])
def test_no_revienta(basura):
    R(basura, ["huevo"], _sa)


def test_excepcion_devuelve_el_paso_intacto():
    class Explota:
        def __iter__(self):
            raise RuntimeError("boom")
    assert R("Vierte el yogurt batidas.", Explota(), _sa) == "Vierte el yogurt batidas."


# ───────────── 4. está conectado al autofix ─────────────

def test_el_autofix_invoca_la_reparacion():
    """Ancla 'código presente, efecto ausente': el helper podría existir y no llamarse nunca.
    Debe correr DESPUÉS de la sustitución del sustantivo — antes no habría nada que reparar."""
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    i = src.index("_reparar_concordancia_huevo(_s, _orphan_keys")
    j = src.rindex("_s = _p.sub(_repl, _s)", 0, i)
    assert j < i, "la reparación debe ir DESPUÉS de sustituir el sustantivo huérfano"
