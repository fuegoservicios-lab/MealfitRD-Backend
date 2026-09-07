# -*- coding: utf-8 -*-
"""[P1-REGISTRY-TITLE-TRUTH · 2026-09-07] El título de una plantilla no puede contradecir su plato.

`Panqueques de harina de ARROZ con fresas` traía como constituyente `Harina de TRIGO`. Lo encontró
el dueño juzgando la biblioteca a ciegas, no un linter — y ningún prompt podía arreglarlo: el
modelo usó correctamente lo que la plantilla le dio. **Un defecto en el registro es peor que uno en
un plan**: el plan es de un usuario, la plantilla se sirve a todos los que caigan en ella.

## Por qué la regla es de CONTRADICCIÓN y no de ausencia

Mi primer detector preguntaba «¿el título nombra un alimento que falta en `constituents`?» y salió
inservible: 13 hallazgos, casi todos falsos por la relación genérico↔específico —«Yogur» contra
`Yogurt griego sin azúcar`, «Espaguetis» contra `Pasta integral`, «Cerdo» contra `Costilla de
cerdo`— y **el único caso real no aparecía**, porque «harina de arroz» resuelve a `Harina de trigo`
por el alias «harina».

La pregunta correcta es de contradicción: mismo núcleo, calificativo incompatible. Así el
genérico↔específico deja de disparar, porque «Yogur» no lleva calificativo que contradecir.

## Dos trampas que este test lleva dentro

1. **Los pares tienen que SOLAPAR.** En «panqueques de harina de arroz» un `findall` normal consume
   «panqueques de harina» y jamás llega a «harina de arroz», que es justo el par buscado. Con
   lookahead el detector encontró 1 caso; sin él encontraba 0 — y ese 0 parecía una buena noticia.
2. **La lista de exclusión puede tapar el hallazgo.** Al añadirle palabras para callar dos falsos
   positivos, el contador bajó a 0 por la razón equivocada. Por eso el test fija el caso conocido
   (`test_el_caso_que_lo_origino_seria_detectado`): un detector que no puede volver a encontrar lo
   que ya encontró no está midiendo nada.
"""
import json
import re
import unicodedata
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_REGISTRY = _BACKEND / "data" / "registry"

# Tras «de», palabras que describen origen, forma o una nota entre paréntesis — no la identidad
# del alimento. Mantener esta lista CORTA: cada entrada es un hallazgo que dejamos de ver.
_NO_IDENTIDAD = {
    "casa", "la", "el", "los", "las", "hoja", "hojas", "corral", "campo", "temporada",
    "monte", "olla", "sarten", "horno", "cuchara", "mano", "siempre", "diario",
    "despensa", "nevera", "cerdo", "res", "pollo",
}

# Lookahead: los pares DEBEN solapar (ver trampa 1 del docstring).
_PAR = re.compile(r"(?=\b([a-zñ]{4,})\s+de\s+([a-zñ]{3,})\b)")


def _sa(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def contradicciones(nombre: str, constituyentes: list[str]) -> list[str]:
    """Calificativos que el título promete y los constituyentes desmienten.

    Tres condiciones a la vez: (1) el título trae «<núcleo> de <calificativo>»; (2) algún
    constituyente EMPIEZA por ese núcleo —si no, el núcleo no es un alimento de este plato—;
    (3) el calificativo no aparece en ningún nombre de constituyente.
    """
    cons = [_sa(c) for c in constituyentes]
    blob = " ".join(cons)
    fuera = []
    for nucleo, calif in _PAR.findall(_sa(nombre)):
        if calif in _NO_IDENTIDAD:
            continue
        if not any(c.split()[:1] == [nucleo] for c in cons):
            continue
        if re.search(r"\b" + re.escape(calif), blob):
            continue
        fuera.append(f"{nucleo} de {calif}")
    return fuera


def _snapshots():
    return sorted(_REGISTRY.glob("dish_registry_*_v1.json"))


def test_hay_snapshots_que_auditar():
    """Sin este guard, borrar los snapshots dejaría el test en verde sin mirar nada."""
    assert len(_snapshots()) == 6, [p.name for p in _snapshots()]


@pytest.mark.parametrize("snap", _snapshots(), ids=lambda p: p.stem.split("_")[2])
def test_ningun_titulo_contradice_sus_constituyentes(snap):
    reg = json.loads(snap.read_text(encoding="utf-8"))
    plantillas = reg.get("templates") or []
    assert plantillas, f"{snap.name} sin plantillas"
    malas = []
    for t in plantillas:
        nom = ((t.get("editorial") or {}).get("display_name") or {}).get("es") or t.get("name") or ""
        cons = [str(c.get("name") or c.get("canonical") or "") for c in (t.get("constituents") or [])]
        for promesa in contradicciones(nom, cons):
            malas.append(f"[{t.get('template_id')}] {nom!r} promete «{promesa}» y trae {cons}")
    assert not malas, "títulos que contradicen su plato:\n  " + "\n  ".join(malas)


def test_el_caso_que_lo_origino_seria_detectado():
    """El detector tiene que poder volver a encontrar lo que ya encontró una vez.

    Es el ancla contra las dos trampas: si alguien quita el lookahead o mete «arroz» en la lista de
    exclusión, el barrido de arriba dará 0 y parecerá una buena noticia.
    """
    assert contradicciones("Panqueques de harina de arroz con fresas",
                           ["Harina de trigo", "Huevo", "Leche", "Fresas"]) == ["harina de arroz"]


@pytest.mark.parametrize("nombre,cons", [
    ("Yogur con frutas picadas y avena tostada", ["Yogurt griego sin azúcar", "Avena", "Mango"]),
    ("Espaguetis con atún y salsa criolla", ["Pasta integral", "Atún en agua", "Salsa de tomate"]),
    ("Pescado con coco estilo Samaná", ["Filete de pescado blanco", "Leche de coco", "Arroz blanco"]),
    ("Rollitos de lechuga con res mechada", ["Lechuga romana", "Carne de res", "Tomate"]),
    ("Chicharrón de cerdo con tostones", ["Chicharrón", "Plátano verde", "Pique"]),
])
def test_el_generico_contra_el_especifico_no_dispara(nombre, cons):
    """Los cinco falsos positivos que hundieron la primera versión del detector.

    «Yogur» y `Yogurt griego sin azúcar` son el mismo alimento nombrado con más o menos precisión;
    un detector que los acusa es ruido, y el ruido es lo que hace que se apague un detector.
    """
    assert contradicciones(nombre, cons) == []
