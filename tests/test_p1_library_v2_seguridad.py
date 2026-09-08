# -*- coding: utf-8 -*-
"""[P1-LIBRARY-V2-SEGURIDAD · 2026-09-08] La biblioteca, regenerada con el criterio correcto.

Las 140 recetas de la v1 se escribieron con la regla que decía «hasta que no quede rosado» para la
carne. El juicio a ciegas del dueño sobre almuerzos y cenas —**3 «la serviría» / 16 «dudoso» / 1
«no»**, contra 17/3/0 en desayunos— lo destapó: **13 de los 16 «dudoso» pedían temperatura interna**.

Los desayunos habían pasado con 17/20 sólo porque casi no llevan carne. Y el escáner de coherencia
culinaria daba **140/140 limpias** con la regla defectuosa dentro: mide que no se pida lo que no hay
y que lo crudo se cocine, **no si un criterio de cocción es seguro**.

## Lo que se regeneró y lo que se midió

Cuatro líneas del prompt, cada una atada a una queja concreta:

| queja del dueño | v1 | v2 |
|---|---|---|
| «sustituir el color por 74 °C» (13 notas) | 0/40 con temperatura | **40/40** |
| «precisar técnica sin aceite» (~6 notas) | 67 | 79 dicen antiadherente o agua |
| «no prometer textura fija del queso» | 9 con «no se funde» | **0** |
| «precisar precalentado» | 17/25 | **27/27** |

Los 5 que además mencionan «rosado» lo hacen como el respaldo documentado, DESPUÉS de la
temperatura: «comprueba 74 °C con termómetro; **si no tienes**, los jugos deben salir claros».

## Dos correcciones de mi propio instrumento, en la misma medición

El detector de «platos con carne» daba 45 y el número real es 40: `fresas` contiene «res» y
**re·pollo** contiene «pollo». Es la misma clase de bug por subcadena que este repo ya documentó con
«que·**so f·res·co**», cometida por mí mientras medía si había arreglado un fallo de seguridad.

*Un detector por subcadena encuentra palabras dentro de otras palabras, y el número que produce
parece igual de sólido que uno correcto.*
"""
import json
import re
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parent.parent / "data" / "registry"
_LIB = _DIR / "recipe_library_do_v1.json"
_REG = _DIR / "dish_registry_do_v1.json"

# Palabra completa a propósito: por subcadena, `repollo` trae «pollo» y `fresas` trae «res».
# Y se mira el INGREDIENTE, no el texto del paso: ahí «la carne se separa en hojas» (pescado) y
# «chuleta de soya» entraban como carne de matadero. Las dos correcciones salieron de mirar los
# casos que el test acusaba, no de razonar mejor.
_CARNE = re.compile(r"\b(pollo|pavo|res|cerdo|chivo|carne|pechuga|molida|costilla|chuleta)\b", re.I)
_NO_ES_CARNE = re.compile(r"\bsoya\b|\bsoja\b|\bvegetal\b", re.I)
_TEMP = re.compile(r"7[41]\s*°?\s*C|63\s*°?\s*C")


@pytest.fixture(scope="module")
def lib():
    if not _LIB.exists():
        pytest.skip("la biblioteca no está en el árbol")
    return json.loads(_LIB.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ingredientes_por_id():
    reg = json.loads(_REG.read_text(encoding="utf-8"))
    return {t["template_id"]: [str((c or {}).get("name") or "") for c in (t.get("constituents") or [])]
            for t in reg.get("templates") or []}


def _con_carne(lib, ings_por_id):
    """Platos cuya LISTA lleva carne de matadero. El pescado va aparte: su criterio es que la carne
    se separe en hojas, y ése sí es el correcto."""
    out = []
    for t, r in lib["por_id"].items():
        for ing in ings_por_id.get(t) or []:
            if _CARNE.search(ing) and not _NO_ES_CARNE.search(ing):
                out.append((t, r))
                break
    return out


def test_toda_receta_con_carne_lleva_TEMPERATURA(lib, ingredientes_por_id):
    """El contrato de seguridad, sobre el dato y no sobre el prompt.

    El prompt ya lo ancla `test_p0_doneness_temperatura`; esto comprueba que las recetas que se van
    a SERVIR lo cumplen. Un prompt correcto con un dato viejo sirve el dato viejo.
    """
    sin = [t for t, r in _con_carne(lib, ingredientes_por_id)
           if not _TEMP.search(" ".join(r["pasos"]))]
    assert not sin, (
        f"{len(sin)} receta(s) con carne sin temperatura interna: {sin[:4]}. "
        "Regenerar con `scratchpad/genera_biblioteca_rd.py` (prompt v6).")


def test_el_color_solo_aparece_como_respaldo(lib, ingredientes_por_id):
    """Mencionar «rosado» está bien SI la temperatura va antes: es la salida para quien no tiene
    termómetro. Está mal si es el único criterio, que es lo que el dueño rechazó 13 veces."""
    malas = []
    for t, r in _con_carne(lib, ingredientes_por_id):
        txt = " ".join(r["pasos"])
        if re.search(r"rosad", txt, re.I) and not _TEMP.search(txt):
            malas.append(t)
    assert not malas, f"el color volvió a ser criterio único en: {malas[:4]}"


def test_no_promete_como_se_comporta_el_queso(lib):
    """La regla anterior sobre-generalizó en la dirección contraria: pasó de «no digas que se funde»
    a afirmar que NUNCA se funde. El dueño pidió no prometer NINGÚN comportamiento."""
    malas = [t for t, r in lib["por_id"].items()
             if re.search(r"no se (funde|derrite)", " ".join(r.get("pasos") or []), re.I)]
    assert not malas, f"vuelven las promesas sobre el queso: {malas[:4]}"


def test_la_procedencia_cuenta_el_fallo_de_la_v1(lib):
    """Dentro de un mes, «140 recetas revisadas» sin más suena a garantía. El fichero debe llevar
    encima que su v1 tenía un fallo de seguridad y que el escáner no lo vio."""
    p = lib.get("procedencia") or {}
    assert "SEGURIDAD" in (p.get("veredicto_humano") or ""), (
        "la procedencia perdió el motivo de la regeneración")
    assert "NO mide seguridad" in (p.get("escaner") or ""), (
        "el fichero debe decir qué NO mide su propio escáner: dio 140/140 con la regla del color")
