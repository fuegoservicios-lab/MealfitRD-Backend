# -*- coding: utf-8 -*-
"""[P1-LIBRARY-V3-SIN-SUSTITUTO · 2026-09-08] Un sustituto ofrecido es un sustituto usado.

## Qué pasó

`P0-DONENESS-TEMPERATURA` (07-sep, mismo día) cerró un fallo real: la §21 del prompt del día
decía «hasta que el interior no tenga partes rosadas. Aquí no es textura, es SEGURIDAD» — nombra
la seguridad y da acto seguido el único criterio que no sirve para ella.

Al arreglarlo añadí, por mi cuenta, una señal de respaldo *«para quien no tenga termómetro»*:
jugos claros, carne firme, sin zonas rosadas. Y el juicio a ciegas del dueño **bajó**: de
**3 «la serviría» / 16 «dudoso» / 1 «no»** a **1/19/0**. Nueve de las veinte notas decían lo
mismo: *«eliminar los jugos claros, la firmeza y la ausencia de zonas rosadas como sustitutos del
termómetro; mantener la comprobación de 74 °C»*.

Tenía razón, y **el orden no salvaba nada**: yo había puesto la temperatura primero y el respaldo
después, creyendo que eso lo hacía subordinado. No lo hace — quien no tenga termómetro lee la
frase que le permite cocinar sin él. Es la misma forma que el palillo del 07-sep: una cláusula
bienintencionada que se convierte en la puerta de salida.

## Por qué este fichero existe aparte

Son tres contratos distintos sobre la misma regla, y separarlos es lo que hace que un renombre
falle el test correcto:

  · `test_p1_daygen_doneness_inside` — que HAYA señal de interior.
  · `test_p0_doneness_temperatura`   — que para la carne esa señal sea la TEMPERATURA.
  · éste                             — que no haya NINGUNA alternativa a medirla, ni en el prompt
                                       del día ni en las 140 recetas ya escritas.

El tercero hace falta porque la biblioteca es DATO ya generado: arreglar el prompt no reescribe
lo que ya está en `data/registry/`. Un prompt limpio con una biblioteca sucia sirve la frase
insegura igual.
"""
import json
import re
import unicodedata
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_LIBRERIA = _BACKEND / "data" / "registry" / "recipe_library_do_v1.json"

# Las formas concretas que el dueño pidió eliminar, más la que las introduce.
_SUSTITUTOS = (
    (r"jugos?\s+(?:salen\s+)?claros", "jugos claros"),
    (r"sin\s+(?:zonas?|partes?)\s+rosad", "sin zonas rosadas"),
    (r"no\s+(?:tenga|queda|quede|hay)\s+(?:partes?|zonas?)\s+rosad", "sin partes rosadas"),
    (r"se\s+separa\s+del\s+hueso\s+sin\s+resistencia", "se separa del hueso"),
    (r"si\s+no\s+tienes?\s+term[oó]metro", "alternativa explícita"),
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


@pytest.fixture(scope="module")
def biblioteca() -> dict:
    if not _LIBRERIA.exists():
        pytest.skip("la biblioteca no está en el árbol")
    return json.loads(_LIBRERIA.read_text(encoding="utf-8"))


def test_ninguna_receta_ofrece_un_sustituto_del_termometro(biblioteca):
    """El contrato duro. En la v2 eran **25 de 140** las que lo ofrecían."""
    culpables = []
    for tid, r in (biblioteca.get("por_id") or {}).items():
        txt = _norm(" ".join(r.get("pasos") or []))
        for rx, etiqueta in _SUSTITUTOS:
            if re.search(rx, txt):
                culpables.append(f"{tid} [{etiqueta}]")
                break
    assert not culpables, (
        "vuelven los sustitutos del termómetro en la biblioteca publicada: "
        f"{culpables[:8]}{'…' if len(culpables) > 8 else ''}")


def test_la_carne_sigue_llevando_su_temperatura(biblioteca):
    """El guard del péndulo: quitar el sustituto no puede haberse llevado por delante la
    temperatura. Si un plato con carne pierde los 74/71/63, esto deja de ser un endurecimiento
    y pasa a ser una regresión de seguridad."""
    con_temp = sum(
        1 for r in (biblioteca.get("por_id") or {}).values()
        if re.search(r"\b(74|71|63)\s*.?\s*C\b", " ".join(r.get("pasos") or [])))
    assert con_temp >= 35, (
        f"sólo {con_temp} recetas citan una temperatura interna; medido el 08-sep: 40 platos con "
        f"carne, 40 con temperatura. Una caída aquí es que se perdió la regla, no que hay menos "
        f"carne")


def test_la_procedencia_dice_lo_que_NO_esta_resuelto(biblioteca):
    """Un bloque de procedencia que sólo lista logros es publicidad. La v2 decía «140/140
    limpias» mientras servía un criterio inseguro; lo que evita repetirlo es dejar escrito, junto
    al dato, qué sigue sin medirse."""
    p = biblioteca.get("procedencia") or {}
    assert p.get("sin_resolver"), "falta el apartado de lo que la biblioteca NO arregla"
    sr = _norm(p["sin_resolver"])
    assert "crudo" in sr and "cocido" in sr, (
        "los gramos crudo-vs-cocido son el hallazgo más serio de la ronda del dueño (cambian el "
        "cálculo nutricional) y no están arreglados: tiene que constar")
    assert "juzgada por un humano todavia" in _norm(p.get("veredicto_humano", "")), (
        "la v3 no ha pasado juicio humano; decir lo contrario es exactamente el error que este "
        "P-fix corrige")


def test_el_prompt_del_dia_tampoco_ofrece_sustituto():
    """La otra mitad: la biblioteca cubre los platos ya escritos, el prompt cubre todo lo que se
    genere de aquí en adelante."""
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as P

    i = P.find("21. CÓMO SE SABE QUE ESTÁ LISTO POR DENTRO")
    if i < 0:
        pytest.skip("§21 no está en el prompt (knob apagado)")
    bloque = P[i:]
    # OJO: el prompt SÍ nombra las tres señales, y debe hacerlo — las nombra para PROHIBIRLAS.
    # Por eso aquí NO se puede reusar `_SUSTITUTOS`: aplicado al prompt daría falso positivo
    # justo sobre la frase que arregla el bug. Se ancla la prohibición, no la ausencia.
    assert "NO ofrezcas ningún sustituto" in bloque
    assert "ofrecerlo es que se use" in bloque, (
        "falta el porqué de la prohibición; una prohibición sin razón se ignora en cuanto estorba")


def test_la_migracion_de_avena_existe_y_es_idempotente():
    """[P1-LIBRARY-V3-SIN-SUSTITUTO] El segundo hallazgo de la ronda, y no es de prosa: `Avena`
    no admitía 'tostar' aunque una plantilla del registry se llama «avena tostada». Lo destapó la
    biblioteca al pasar por el escáner de producción, no una auditoría del catálogo."""
    sql = _BACKEND / "migrations" / "p1_avena_tostada_prep_method_2026_09_08.sql"
    assert sql.exists(), "falta la migración; el falso positivo V1 sigue vivo en producción"
    t = sql.read_text(encoding="utf-8")
    assert "NOT ('tostar' = ANY(prep_methods))" in t, (
        "sin el guard, re-ejecutar la migración añade 'tostar' otra vez: no es idempotente")
    assert "RAISE EXCEPTION" in t, "falta el sanity check del patrón de la casa"
    assert "Leche de avena" in t, (
        "la migración debe decir por qué `Leche de avena` se queda fuera; si no, alguien la añade "
        "por simetría y «tuesta la leche de avena» deja de dispararse")
