# -*- coding: utf-8 -*-
"""[P1-MINUTOS-DE-LA-RECETA · 2026-09-10] El tiempo del plato estaba escrito y la ficha lo contradecía.

## De dónde sale

Tercera ronda de juicio humano sobre los 19 platos de proteína: **19 de 19 «cambiar»**, y al
clasificar las notas, **13 de las 19 pedían corregir el tiempo**. «Yogurt griego con guineo» —pelar
un guineo y ponerlo encima del yogurt— anunciaba **30 minutos**.

## El mecanismo

`derive_logistics` sacaba `prep_minutes_est` de una tabla de 6 filas por TÉCNICA con **30 de
defecto**. Medido: **21 de 198** plantillas tenían una técnica que no casaba ninguna fila —16 de
ellas `crudo`— y se llevaban el 30 en silencio. *Un defecto que se aplica sin decirlo no es un
defecto: es un dato inventado con la misma cara que uno medido.*

Mientras tanto, la receta escrita **ya declara sus tiempos paso a paso**. Medido contra los 13
números que pidió el dueño: la suma de los pasos acierta **13 de 13** (±5 min).

## Las dos trampas que la medición destapó

1. **Sumar pasos que corren a la vez.** La primera versión infló «Víveres guisados con garbanzos» a
   **125 min**, porque sumaba los 25 de pelar los víveres encima de los 90 del hervor — y el paso
   dice literalmente «*Mientras* los garbanzos cuecen». Un paso con marca de solape sólo añade lo
   que se le salga por arriba del anterior.
2. **La sonda que valida no puede tener su propia copia de la función.** La primera revalidación dio
   13/13 con el parser VIEJO copiado dentro de la sonda; sólo al apuntarla a
   `dish_registry.minutos_de_los_pasos` estaba midiendo lo que se despliega.
"""
import json
import pathlib

import pytest

import dish_registry as dr

DATOS = pathlib.Path(dr.REGISTRY_DIR)


@pytest.fixture(scope="module")
def registro_do():
    return json.loads((DATOS / "dish_registry_do_v1.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recetas_do():
    return dr.recipe_steps_index("do")


# ── el lector de tiempos ─────────────────────────────────────────────────────
def test_sin_receta_devuelve_None_no_cero():
    """`None` = «no hay receta, usa la tabla». Un 0 haría pasar por instantáneo lo desconocido."""
    assert dr.minutos_de_los_pasos(None) is None
    assert dr.minutos_de_los_pasos([]) is None


def test_una_receta_sin_tiempos_vale_el_piso_no_la_tabla():
    """Montar un yogurt no es un dato ausente: es un plato de ensamblaje."""
    assert dr.minutos_de_los_pasos(["Pela el guineo y córtalo en ruedas.",
                                    "Sirve el yogurt y coloca el guineo encima."]) == dr.PISO_MINUTOS


@pytest.mark.parametrize("paso,esperado", [
    (["Cocina 12 minutos."], 15),                       # 12 + 3 de margen -> redondeo a 5
    (["Cocina 10-15 minutos."], 20),                    # del rango manda el TOPE
    (["Licúa 45-60 segundos."], 5),                     # segundos: 1 min + margen -> piso
    (["Sofríe 3-4 minutos", "Hornea 12-15 minutos"], 25),      # 4 + 15 + 3 = 22 -> 25
])
def test_lee_el_tiempo_que_el_paso_declara(paso, esperado):
    assert dr.minutos_de_los_pasos(paso) == esperado


def test_un_paso_que_dice_mientras_corre_DENTRO_del_anterior():
    """La trampa que infló «Víveres guisados» a 125 min: sumaba lo que la receta dice que solapa."""
    en_serie = ["Hierve los garbanzos 60-90 minutos.", "Pela la yuca 20-25 minutos."]
    solapado = ["Hierve los garbanzos 60-90 minutos.",
                "Mientras los garbanzos cuecen, pela la yuca 20-25 minutos."]
    assert dr.minutos_de_los_pasos(en_serie) == 120
    assert dr.minutos_de_los_pasos(solapado) == 95, "volvió a sumar dos cosas que pasan a la vez"


def test_lo_que_se_sale_por_arriba_del_solape_SI_cuenta():
    """«Mientras» no es «gratis»: un paso más largo que aquel dentro del que corre sigue añadiendo."""
    assert dr.minutos_de_los_pasos(["Sofríe 5 minutos.",
                                    "Mientras tanto, hornea 40 minutos."]) == 45


# ── el cableado en la compilación ────────────────────────────────────────────
def test_la_receta_gana_a_la_tabla(registro_do, recetas_do):
    plantilla = {"template_id": "tpl_x", "name": "X", "technique": "guisado"}   # tabla diría 50
    lg = dr.derive_logistics(plantilla, [], {}, ["Cocina 10 minutos."])
    assert lg["prep_minutes_est"] == 15 and lg["prep_minutes_source"] == "receta"
    lg2 = dr.derive_logistics(plantilla, [], {}, None)
    assert lg2["prep_minutes_est"] == 50 and lg2["prep_minutes_source"] == "tecnica"


def test_ninguna_tecnica_del_registro_cae_al_defecto(registro_do):
    """El 30 de relleno tenía 21 clientes silenciosos. Si vuelve a tenerlos, que se vea aquí."""
    caidas = [t["name"] for t in registro_do["templates"]
              if t["logistics"].get("prep_minutes_source") == "defecto"]
    assert not caidas, f"{len(caidas)} plantillas se llevan el 30 por defecto: {caidas[:6]}"


def test_el_snapshot_dice_de_donde_sale_cada_numero(registro_do):
    for t in registro_do["templates"]:
        assert t["logistics"].get("prep_minutes_source") in ("receta", "tecnica", "defecto"), t["name"]


def test_casi_todo_el_registro_do_tiene_su_tiempo_escrito(registro_do):
    de_receta = sum(1 for t in registro_do["templates"]
                    if t["logistics"]["prep_minutes_source"] == "receta")
    assert de_receta >= 190, f"sólo {de_receta} de {len(registro_do['templates'])} leen su receta"


def test_el_snapshot_coincide_con_lo_que_dicen_las_recetas(registro_do, recetas_do):
    """El snapshot en disco no puede haber quedado atrás respecto al recetario."""
    malos = []
    for t in registro_do["templates"]:
        pasos = recetas_do.get(t["template_id"])
        if not pasos:
            continue
        esperado = dr.minutos_de_los_pasos(pasos)
        if t["logistics"]["prep_minutes_est"] != esperado:
            malos.append((t["name"], t["logistics"]["prep_minutes_est"], esperado))
    assert not malos, f"{len(malos)} plantillas con el tiempo rancio: {malos[:5]}"


# ── el caso del dueño, textual ───────────────────────────────────────────────
@pytest.mark.parametrize("nombre,tope", [
    ("Yogurt griego con guineo", 10),                     # pidió 5
    ("Queso cottage con lechosa", 10),                    # pidió 5
    ("Sardinas en lata con casabe", 10),                  # pidió 5
    ("Yogurt griego con avena tostada y maní", 15),       # pidió 10
    ("Sardinas guisadas con casabe y tomate", 25),        # pidió 15-20
])
def test_los_platos_que_el_dueno_cronometro(registro_do, nombre, tope):
    t = next((x for x in registro_do["templates"] if x["name"] == nombre), None)
    assert t is not None, f"desapareció del registro: {nombre}"
    assert t["logistics"]["prep_minutes_est"] <= tope, (
        f"«{nombre}» sigue anunciando {t['logistics']['prep_minutes_est']} min")


def test_el_recetario_entra_en_el_hash_de_la_fuente():
    """Si cambiar un tiempo de la receta no mueve el snapshot, el número puede quedar rancio."""
    import inspect
    src = inspect.getsource(dr.compile_library)
    assert "recipe_minutes" in src, "la receta decide un campo y no entra en `source_material`"
