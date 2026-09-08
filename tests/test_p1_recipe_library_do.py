# -*- coding: utf-8 -*-
"""[P1-RECIPE-LIBRARY-DO · 2026-09-08] La biblioteca de recetas de RD, escrita una sola vez.

Hoy el motor le pide a un LLM que invente la receta de cada plato en el momento: cuesta dinero cada
vez, se espera, y **el mismo plato sale escrito distinto cada vez**. Ahí nacieron varios de los
defectos que este repo lleva persiguiendo.

La propuesta es escribirlas OFFLINE, revisarlas, y que el runtime SELECCIONE. Su economía se
sostiene en una asimetría: offline, una receta mala cuesta un borrador tirado, así que el portero
puede ser brutal.

## Lo que decidió que era viable

Tres rondas de juicio a ciegas del dueño sobre 20 desayunos: **8 → 11 → 17** «la serviría» de 20, 0
rechazadas. El umbral (≥70 % ⇒ «llenar la biblioteca es un cron») estaba escrito ANTES de ver el
resultado. Y las tres rondas corrigieron **reglas mías**, nunca el modelo:

  · una prohibía toda cifra, incluidas las de cocción — pero la temperatura no escala con la porción;
  · dos se contradecían: una prohibía cifras y otra exigía repartir un ingrediente «explícitamente»,
    y un reparto no se puede hacer explícito sin una proporción;
  · el palillo se convirtió en ritual (7 de 20, incluidas 4 que el dueño aprobó);
  · el agua estaba prohibida junto a la sal y el aceite, así que el modelo inventaba apaños.

## Lo que este test ancla, y lo que NO

Ancla la forma: cobertura, sin cantidades de ingrediente, ids que existen, procedencia escrita.

**NO ancla que las recetas sean buenas.** El escáner dice que son coherentes —que no piden lo que no
hay, que cocinan lo crudo, que reparten lo que se comparte—; eso no es un paladar. El veredicto
humano cubre una MUESTRA de 20, no las 140, y el fichero lo dice en su propia procedencia para que
nadie lo olvide dentro de un mes.
"""
import json
import re
from pathlib import Path

import pytest

_B = Path(__file__).resolve().parent.parent
_LIB = _B / "data" / "registry" / "recipe_library_do_v1.json"
_REG = _B / "data" / "registry" / "dish_registry_do_v1.json"

# Las mismas cifras que el generador prohíbe: cantidades de INGREDIENTE, no de cocción.
# «180 °C» y «15 minutos» son parámetros y deben poder ir.
_CIFRA = re.compile(
    r"\d+\s*(?:g|gr|gramos|ml|kg|tazas?|cdas?|cdtas?|cucharad|unidad|huevos?|claras?)|[½¼¾⅓⅔⅛]",
    re.I)


@pytest.fixture(scope="module")
def lib():
    if not _LIB.exists():
        pytest.skip("la biblioteca no está en el árbol")
    return json.loads(_LIB.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reg():
    return json.loads(_REG.read_text(encoding="utf-8"))


def test_cubre_todas_las_plantillas_usables(lib, reg):
    usables = {t["template_id"] for t in reg["templates"]
               if t.get("constituents") and not t.get("excluded")}
    faltan = sorted(usables - set(lib["por_id"]))
    assert not faltan, (
        f"{len(faltan)} plantillas usables del registry sin receta: {faltan[:5]}. "
        "Regenerar con `scratchpad/genera_biblioteca_rd.py` (reanudable).")


def test_ninguna_receta_apunta_a_una_plantilla_que_no_existe(lib, reg):
    """Una receta huérfana es peor que ninguna: se sirve un texto de un plato retirado."""
    ids = {t["template_id"] for t in reg["templates"]}
    sobran = sorted(set(lib["por_id"]) - ids)
    assert not sobran, f"recetas de plantillas inexistentes: {sobran[:5]}"


def test_los_pasos_no_llevan_cantidades_de_ingrediente(lib):
    """Es la condición que hace viable la biblioteca: sin cantidades, la misma receta sirve para
    cualquier porción y se escribe UNA vez. Con cifras habría que escribir una por porción."""
    malas = []
    for tid, r in lib["por_id"].items():
        for p in r["pasos"]:
            if _CIFRA.search(p):
                malas.append((tid, p[:70]))
    assert not malas, (
        f"{len(malas)} paso(s) con cantidad de ingrediente — la biblioteca deja de servir para "
        f"cualquier porción: {malas[:3]}")


def test_toda_receta_tiene_entre_3_y_5_pasos_con_texto(lib):
    for tid, r in lib["por_id"].items():
        n = len(r["pasos"])
        assert 3 <= n <= 5, f"{tid}: {n} pasos"
        assert all(isinstance(p, str) and len(p.strip()) > 20 for p in r["pasos"]), tid


def test_no_duplica_lo_que_ya_es_del_registry(lib):
    """Sólo `template_id` y `pasos`. El nombre y los ingredientes viven en el registry.

    Duplicarlos crea dos fuentes que divergen — la clase de fallo que este repo ya pagó con
    `P1-REGISTRY-TITLE-TRUTH` (el título prometía una harina que el plato no llevaba).
    """
    for tid, r in lib["por_id"].items():
        assert set(r) <= {"franja", "pasos"}, f"{tid} guarda campos del registry: {sorted(r)}"


def test_la_procedencia_dice_que_NO_estan_revisadas_una_a_una(lib):
    """Dentro de un mes nadie va a recordar si un humano miró estas 140. El fichero debe decirlo."""
    p = lib.get("procedencia") or {}
    for k in ("escrito_por", "veredicto_humano", "escaner", "regenerar", "sin_cantidades"):
        assert p.get(k), f"falta la procedencia `{k}`"
    assert "MUESTRA" in p["veredicto_humano"], (
        "la procedencia debe dejar claro que el juicio humano cubre una muestra, no las 140")


def test_sigue_INERTE_y_el_dia_que_no_lo_este_sera_a_sabiendas(lib):
    """Ningún módulo de producción lee todavía la biblioteca.

    Cuando alguien enchufe la selección al runtime, este test fallará — y esa es la idea: el cambio
    que hace que el mismo plato salga SIEMPRE igual es el que de verdad cambia el sistema, y merece
    ser deliberado, no un efecto colateral. Al enchufarlo, actualizar este test con el call site.
    """
    assert lib.get("estado", "").startswith("INERTE")
    lectores = []
    for py in _B.rglob("*.py"):
        if any(x in py.parts for x in ("tests", "scripts", "__pycache__", ".venv")):
            continue
        try:
            if "recipe_library_do" in py.read_text(encoding="utf-8", errors="ignore"):
                lectores.append(py.name)
        except Exception:
            continue
    assert not lectores, (
        f"la biblioteca dejó de ser inerte: la leen {lectores}. Si es deliberado, actualiza este "
        "test y el campo `estado` del fichero, y añade la cobertura de la ruta de selección.")
