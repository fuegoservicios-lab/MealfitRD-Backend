# -*- coding: utf-8 -*-
"""[P1-LIBRARY-V4-PESO-Y-TIEMPO · 2026-09-08] Las cinco causas que quedaban, todas suyas.

El juicio a ciegas de la v3 dio **8 «la serviría» / 12 «dudoso» / 0 «no»** en almuerzo y cena,
contra 1/19/0 (v2) y 3/16/1 (v1). Lo importante está en lo que las notas NO dicen: ninguna de las
20 pide el sustituto del termómetro de vuelta ni discute los 74 °C. `P0-DONENESS` quedó cerrado, y
**ninguna defensa automática podría haberlo dicho** — el escáner daba 140/140 con la regla insegura.

Clasificadas sus 12 «dudoso», quedaban cinco causas. Una era defecto de DATO (`arroz amarillo`,
cerrada en `P1-REGISTRY-COLOR-PROMETE`). Las otras cuatro son de prosa, más una quinta que era una
regla MÍA disparando mal.

## A · El estado del peso — y por qué el problema era más pequeño de lo que yo mismo dije

Seis de sus veinte notas: «los 70 g de arroz se pesan en crudo», «si los 150 g de chivo incluyen
hueso», «si los 180 g de gandules son secos, frescos o cocidos: **esto cambia el cálculo
nutricional**», «los 120 g de edamame son granos sin vaina», «los 90 g de atún son peso escurrido»,
«los 100 g de huevo son sin cáscara».

Lo llamé su hallazgo más serio, y al MEDIRLO resultó más barato: el catálogo cuenta **crudo/seco y
porción comestible, 10 de 10 sondas** (arroz 359 kcal contra 360 crudo / 130 cocido; avena 382;
lentejas 362), y por `fdc_id` de USDA — chivo sin hueso `174375`, edamame desvainado `168411`, atún
escurrido `334194`, pollo crudo `2646170`. `nutrition_db`, `dish_registry` y la lista de compras
leen todos el mismo `kcal_per_100g`. **El cálculo no está mal**; lo que falta es que la receta diga
en qué estado pesar. Quien pese 70 g de arroz ya hecho come un tercio: el plan acierta y la
ejecución falla, que es otro problema y se cierra con una frase.

*Estuve a punto de reportar el atún como el outlier del sistema con una cifra de USDA que recordaba
mal. Comprobar el `fdc_id` en vez de fiarme de mi memoria tumbó el hallazgo.*

## C · La causa que decidí NO anclar con un test

El «reparto incompleto» (los 30 g de queso de los que un paso usa la mitad) sale en **5 de sus 20**
notas. Escribí un detector léxico y encuentra **2 en 140**: el mío busca «la mitad de X» y el suyo
es semántico. **No mide lo que él mide**, así que va como regla del prompt y no como test — un
escáner que da verde sobre un defecto vivo es peor que no tenerlo, y hoy pasó dos veces.

## Lo que este fichero ancla

Los números medidos, para que una regeneración futura que los empeore falle aquí en vez de llegar
al plato. Son suelos, no objetivos: por eso van holgados respecto a lo medido el 08-sep.
"""
import json
import re
import unicodedata
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_LIBRERIA = _BACKEND / "data" / "registry" / "recipe_library_do_v1.json"

# Los alimentos cuyo peso cambia al cocinarse o vienen con algo que se descarta.
_PIDE_ESTADO = ("arroz", "habichuela", "gandul", "lenteja", "garbanzo", "avena", "quinoa",
                "semola", "pasta", "espagueti", "bulgur", "frijol", "chivo", "atun", "edamame",
                "huevo")
_DICE_ESTADO = re.compile(
    r"\b(en\s+crudo|peso\s+(?:en\s+)?(?:crudo|seco)|ya\s+escurrid|sin\s+c[aá]scara|"
    r"se\s+pesan?\s+(?:crudo|seco|en\s+crudo|sin)|sin\s+hueso|desgranad|sin\s+vaina|"
    r"crudos?\b|secos?\b)", re.I)
_GRASAS = ("aceite", "mantequilla", "manteca", "margarina")
_AGUA_EN_VEZ = re.compile(r"agua\s+en\s+vez\s+de\s+aceite|chorrito\s+de\s+agua\s+en\s+vez", re.I)


def _sa(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


@pytest.fixture(scope="module")
def lib():
    if not _LIBRERIA.exists():
        pytest.skip("la biblioteca no está en el árbol")
    return json.loads(_LIBRERIA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def con_ingredientes(lib):
    """Empareja cada receta con los constituyentes de SU plantilla. La biblioteca guarda sólo
    `pasos` a propósito (no duplica lo que ya es del registry), así que la pregunta «¿qué lleva
    este plato?» sólo se puede contestar cruzando los dos ficheros."""
    reg = _BACKEND / "data" / "registry" / "dish_registry_do_v1.json"
    if not reg.exists():
        pytest.skip("falta el snapshot del registry")
    tpl = {t["template_id"]: t
           for t in (json.loads(reg.read_text(encoding="utf-8")).get("templates") or [])}
    fuera = []
    for tid, r in (lib.get("por_id") or {}).items():
        t = tpl.get(tid)
        if not t:
            continue
        ings = _sa(" ".join(str(c.get("name") or "") for c in (t.get("constituents") or [])))
        fuera.append((t.get("name"), ings, _sa(" ".join(r.get("pasos") or []))))
    assert len(fuera) >= 130, "casi ninguna receta cruzó con su plantilla: el emparejamiento falla"
    return fuera


def test_el_estado_del_peso_se_dice_en_la_mayoria(con_ingredientes):
    """Medido el 08-sep: 67 de 83 (80 %), desde 17 de 83 (20 %) en la v3. El suelo es 60 % —
    holgado, porque esto es una regla de prosa y el modelo no la cumple al 100 %."""
    pide = [(n, p) for n, i, p in con_ingredientes
            if set(re.findall(r"[a-z0-9]+", i)) & set(_PIDE_ESTADO)]
    dice = [n for n, p in pide if _DICE_ESTADO.search(p)]
    assert len(pide) >= 60, f"sólo {len(pide)} platos piden estado: el emparejamiento se rompió"
    pct = 100 * len(dice) // len(pide)
    assert pct >= 60, (
        f"sólo {len(dice)}/{len(pide)} ({pct} %) dicen en qué estado se pesa; el 08-sep eran 80 %. "
        f"Sin esa frase, quien pese arroz ya cocido come un tercio de lo que el plan calculó")


def test_ninguna_receta_ofrece_agua_teniendo_grasa_en_la_lista(con_ingredientes):
    """Regla MÍA que disparaba mal: «sofríe con un chorrito de agua en vez de aceite» en un plato
    que lleva mantequilla. Lo cazó el dueño, no el escáner. Medido: 3 en la v3, 0 en la v4."""
    malas = [n for n, i, p in con_ingredientes
             if any(g in i for g in _GRASAS) and _AGUA_EN_VEZ.search(p)]
    assert not malas, (
        "vuelven a ofrecer agua como sustituto teniendo grasa comprada en la lista: "
        f"{malas[:6]}")


def test_los_tiempos_van_con_su_senal_no_solos(lib):
    """El tiempo ACOMPAÑA a la señal, nunca la sustituye. Se ancla que la inmensa mayoría de los
    pasos con minutos los presenten como aproximados («unos 12-15 minutos»), que es la forma que
    deja sitio a la señal real."""
    con_tiempo = orientativos = 0
    for r in (lib.get("por_id") or {}).values():
        for p in r.get("pasos") or []:
            if re.search(r"\b\d+\s*(?:-|a|–)?\s*\d*\s*(?:min|hora)", p, re.I):
                con_tiempo += 1
                if re.search(r"\b(orientativ|aproximad|unos\s+\d|alrededor|m[aá]s o menos)",
                             _sa(p)):
                    orientativos += 1
    assert con_tiempo >= 200, f"sólo {con_tiempo} pasos con tiempo: la regla se perdió"
    pct = 100 * orientativos // con_tiempo
    assert pct >= 85, f"sólo {pct} % de los pasos con tiempo lo dan como aproximado (medido: 98 %)"


def test_la_procedencia_no_se_atribuye_un_juicio_que_no_tuvo(lib):
    """La v2 decía «140/140 limpias» mientras servía un criterio inseguro. Lo que evita repetirlo
    es dejar escrito, junto al dato, qué NO se ha comprobado."""
    p = lib.get("procedencia") or {}
    assert "no ha sido juzgada por un humano" in _sa(p.get("veredicto_humano", "")), (
        "la v4 no ha pasado juicio humano; decir lo contrario es el error que ya cometimos")
    assert "MUESTRA" in p.get("veredicto_humano", "")
    sr = _sa(p.get("sin_resolver", ""))
    assert "orden de operaciones" in sr, (
        "el orden de cocina real lo nombró el dueño tres veces y la v4 no lo arregla: tiene que "
        "constar, o el siguiente lo dará por cerrado")
    assert "chenchen" in sr and "maiz partido" in sr, (
        "el chenchén necesita maíz partido y `Maíz partido` no existe en el catálogo: es una "
        "decisión de producto pendiente, no una omisión")
    assert p.get("medido_v3_a_v4"), "faltan las cifras de la ronda"
