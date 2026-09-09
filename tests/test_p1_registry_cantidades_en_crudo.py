# -*- coding: utf-8 -*-
"""[P1-REGISTRY-CANTIDADES-EN-CRUDO · 2026-09-09] El registry nombra en «plato» lo que mide en «crudo».

## De dónde sale esto

Revisión HUMANA del dueño sobre los 35 platos de despensa del 09-sep: 35 de 35 marcados «cambiar»,
**cero rechazados por motivos culturales**. Las 35 notas hablaban de exactitud, y dos temas se
repetían por encima de todos:

  · 35/35 — «45 días sin nevera» leído como la vida del plato YA COCINADO.
  · 32/35 — los gramos son del ingrediente CRUDO, y `serving_g` los presenta como una ración.

Medido después: `serving_g == suma de gramos crudos` en **179 de 179** plantillas. No eran 35 platos,
era el registry entero.

## Por qué esto NO se arregla renombrando

Los dos números son CORRECTOS para lo que el motor pregunta:

  · `days_fresh_min` responde «¿puedo COCINAR esto el día 25 del ciclo?», que es una pregunta sobre
    los ingredientes; `pantry_durability.template_fits` siempre lo dijo así.
  · las macros se calculan sobre gramos crudos contra un catálogo en crudo, que es la forma correcta.

Y las claves entran en `snapshot_hash`: renombrarlas expira las 6 firmas curatoriales y cambia el
`registry_hash` de los planes vivos. Pagar eso por un campo que todos sus consumidores usan bien, y
que hoy no llega a ningún usuario, es desproporcionado.

Lo que sí puede pudrirse es el SIGNIFICADO. `derive_logistics` decía «cuántos días aguanta el plato»
y `template_fits` decía «cuyos constituyentes aguantan»: dos comentarios contradictorios sobre un
mismo campo, y el próximo lector se cree el que le convenga. Estos tests anclan cuál de los dos es
verdad, en el CÓDIGO y en los DATOS.

## El daño que estos tests existen para impedir

Que alguien enseñe `days_fresh_min` a un usuario como «este plato dura 35 días». Es comida: una
tortilla de papa dura tres. Por eso `test_ninguna_cantidad_en_crudo_llega_al_usuario` escanea el
frontend — el día que aparezca ahí, esto falla antes que el despliegue.
"""
import json
import pathlib
import re

import pytest

_BACKEND = pathlib.Path(__file__).resolve().parent.parent
_REG = _BACKEND / "data" / "registry"
_FRONT = _BACKEND.parent / "frontend" / "src"

#: Campos cuyo nombre habla del plato servido y cuyo contenido habla del ingrediente crudo.
CAMPOS_EN_CRUDO = ("serving_g", "days_fresh_min", "days_with_freezer_min")


def _snapshots():
    return sorted(_REG.glob("dish_registry_*_v1.json"))


@pytest.fixture(scope="module")
def registros():
    fuera = {}
    for p in _snapshots():
        try:
            fuera[p.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return fuera


# --------------------------------------------------------------------------- los DATOS


def test_serving_g_es_la_suma_de_los_gramos_CRUDOS(registros):
    """El invariante que el nombre no dice. Si alguien empieza a escribir ahí el peso tras
    cocción, esta prueba cae — y esa es la conversación que hay que tener, no un cambio callado."""
    if not registros:
        pytest.skip("no hay snapshots en el árbol")
    fallos = []
    total = 0
    for nombre, snap in registros.items():
        for t in snap.get("templates") or []:
            crudos = sum(float(c.get("grams") or 0) for c in (t.get("constituents") or []))
            declarado = float(t.get("serving_g") or 0)
            total += 1
            if abs(crudos - declarado) > 0.51:
                fallos.append(f"{nombre}:{t.get('name')} crudo={crudos:.1f} serving_g={declarado:.1f}")
    assert total, "ningún template inspeccionado: el instrumento no midió nada"
    assert not fallos, (
        "`serving_g` dejó de ser la suma de gramos crudos en:\n  " + "\n  ".join(fallos[:10]))


def test_la_durabilidad_es_el_MINIMO_de_sus_ingredientes(registros):
    """`days_fresh_min` es el eslabón más débil, no un promedio ni el del ingrediente principal.

    Un plato no puede declararse más duradero que su ingrediente más perecedero; si lo hiciera, el
    ciclo de una sola compra lo programaría el día 25 y ese día no habría con qué cocinarlo.
    """
    import pantry_durability as pd

    if not registros:
        pytest.skip("no hay snapshots en el árbol")
    fallos = []
    for nombre, snap in registros.items():
        for t in snap.get("templates") or []:
            cons = t.get("constituents") or []
            if not cons:
                continue
            declarado = (t.get("logistics") or {}).get("days_fresh_min")
            if declarado is None:
                continue
            esperado = pd.durability_of(cons).get("days_fresh_min")
            if esperado is not None and int(declarado) > int(esperado):
                fallos.append(f"{nombre}:{t.get('name')} declara {declarado} d, su peor "
                              f"ingrediente aguanta {esperado} d")
    assert not fallos, (
        "plantillas que se declaran MÁS duraderas que su peor ingrediente:\n  "
        + "\n  ".join(fallos[:10]))


# --------------------------------------------------------------------------- el CÓDIGO


def test_los_dos_comentarios_dicen_LO_MISMO():
    """La contradicción que la revisión humana encontró, cerrada donde nace.

    `derive_logistics` (productor) y `template_fits` (consumidor) hablan del mismo número. Hasta el
    09-sep uno decía «el plato» y el otro «sus constituyentes». Un campo con dos significados
    documentados no tiene ninguno.
    """
    import inspect

    import dish_registry as dr
    import pantry_durability as pd

    productor = inspect.getsource(dr.derive_logistics)
    consumidor = inspect.getsource(pd.template_fits)

    # `[\s#]+` y no `\s+`: la frase cruza un salto de línea dentro de un comentario, así que entre
    # las dos palabras hay un `#`. Que este test cayera por eso la primera vez es la prueba de que
    # sí mira el código y no una constante suya.
    assert re.search(r"INGREDIENTES[\s#]+CRUDOS", productor), (
        "`derive_logistics` dejó de declarar que sus días son de los INGREDIENTES crudos")
    assert "constituyentes" in consumidor.lower(), (
        "`template_fits` dejó de declarar que mide los constituyentes")
    assert not re.search(r"cu[áa]ntos d[íi]as aguanta el plato sin congelador", productor), (
        "volvió la redacción que dice que el número es la vida del PLATO cocinado")


def test_serving_g_se_documenta_como_crudo_donde_se_calcula():
    import inspect

    import dish_registry as dr

    src = inspect.getsource(dr.compile_template)
    assert re.search(r"gramos\s+CRUDOS", src), (
        "`compile_template` dejó de declarar que `serving_g` es la suma en crudo")


# --------------------------------------------------------------------------- el USUARIO


def test_ninguna_cantidad_en_crudo_llega_al_usuario():
    """El guard de verdad. Es COMIDA: enseñar «este plato dura 35 días» sobre una tortilla de papa
    no es un bug de producto, es una afirmación falsa sobre algo que alguien se va a comer.

    Hoy la cifra no sale del backend (medido el 09-sep: cero referencias en `frontend/src`). El día
    que alguien la pinte en una pantalla, esta prueba falla ANTES del despliegue y obliga a
    etiquetarla como vida de los INGREDIENTES o a calcular el rendimiento real tras cocción.
    """
    if not _FRONT.exists():
        pytest.skip("el frontend no está en este árbol")
    hallazgos = []
    for p in list(_FRONT.rglob("*.jsx")) + list(_FRONT.rglob("*.js")):
        try:
            texto = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for campo in CAMPOS_EN_CRUDO:
            if campo in texto:
                hallazgos.append(f"{p.relative_to(_FRONT)} usa {campo}")
    assert not hallazgos, (
        "una cantidad medida en CRUDO llegó a la interfaz:\n  " + "\n  ".join(hallazgos)
        + "\n\nSi es deliberado, etiquétala como vida de los INGREDIENTES (no del plato) o "
          "calcula el peso tras cocción, y luego afloja este test citando esa decisión.")


# --------------------------------------------------------------------------- la REVISIÓN


def test_la_revision_humana_del_09_sep_consta_sin_ascenderse():
    """La firma sigue diciendo la verdad: hubo revisión humana, y NO fue una aprobación.

    El dueño marcó los 35 platos como «cambiar» y ninguno como aprobado. Registrarlo como
    `kind: human` sugeriría un visto bueno que no existe; borrarlo perdería una revisión real.
    Consta como lo que fue: una pasada de PRECISIÓN, con sus hallazgos y su conteo.
    """
    ruta = _REG / "cultural_curation_review_v1.json"
    if not ruta.exists():
        pytest.skip("no hay fichero de firma curatorial")
    firma = json.loads(ruta.read_text(encoding="utf-8"))
    perfil = (firma.get("profiles") or {}).get("dominican_criolla") or {}
    decisiones = " ".join(str(d) for d in (perfil.get("decisions") or []))

    assert "P1-REGISTRY-CANTIDADES-EN-CRUDO" in decisiones, (
        "la revisión humana del 09-sep no consta en la firma curatorial")
    assert "35" in decisiones, "el conteo de la revisión no consta"
    assert firma.get("kind") == "automated", (
        "la firma se ascendió a revisión aprobada: el dueño marcó los 35 platos como «cambiar» "
        "y aprobó CERO. Ascenderla es el defecto que P1-REVIEW-KIND-HONEST cerró.")
