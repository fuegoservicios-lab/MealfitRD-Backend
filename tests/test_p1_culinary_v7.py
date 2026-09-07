# -*- coding: utf-8 -*-
"""[P1-CULINARY-V7 · 2026-09-07] Los tres detectores que salieron del golden set humano.

De las 25 comidas que NINGUNA capa marcó, el dueño encontró defecto en 19 (recall ponderado
12,3 % determinista y 15,2 % juez). Sus notas —escritas en 79 de 80 casos— agrupan esos defectos
ciegos, y tres de las clases son mecanizables:

    V7a  la lista compra N piezas y los pasos usan MENOS      (espejo de V6, que solo ve el exceso)
    V7b  el mismo alimento dos veces con unidades incompatibles
    V7c  legumbre declarada SECA que ningún paso remoja ni hierve

Y una cuarta cosa que no era un detector nuevo: **V3 estaba roto**.

## Los casos sintéticos, no el golden set

Este fichero prueba el CONTRATO con comidas fabricadas. El golden set no sirve para eso: dejó de
ser independiente en cuanto se leyeron sus notas para diseñar estos detectores. Medirse contra él
sería medir cuánto se aprendieron las respuestas — el sobreajuste que este proyecto ya tiene
documentado con el juez al 89 %.

Cifra de DESARROLLO, dicha como tal: sobre esas 80, V7 + el arreglo de V3 cazan **7 de los 19
defectos ciegos con 0 falsos positivos** sobre las 9 que el humano marcó correctas. No es la
precisión de V7 — para eso hace falta una muestra nueva que el dueño no haya visto.

## Las tres lecciones de diseño, cada una con su guard

1. **`cocid*` NO es una acción de cocción.** La primera versión de V7c lo incluía y no disparó ni
   una vez teniendo cinco casos: el paso dice «escurre las lentejas y las habichuelas negras
   COCIDAS» y eso no prueba que se cocieran — es exactamente la contradicción que se busca.
   *Usar el síntoma como coartada es cómo un detector se ciega a sí mismo.*

2. **La acción y el alimento, en la misma CLÁUSULA.** «calienta la plancha; cocina la berenjena;
   añade las lentejas» cuece la berenjena. Comprobando por PASO, ese `cocina` daba por cocidas
   unas lentejas que nadie coció.

3. **El número gramatical solo es evidencia con enteros ≥ 2.** «1½ guineos» y un paso que dice
   «el guineo» no se contradicen: media pieza no tiene plural. Ése era el único falso positivo.

Y el arreglo de V3: un prefijo de UNA palabra que nombra una FORMA (harina, pasta, crema, masa…)
lo produce la prosa a partir de cualquier ingrediente. «muele la avena hasta obtener una HARINA
fina» daba por mencionada la «Harina de trigo», que estaba huérfana — 40 g comprados y jamás
usados. 18ª colisión por subcadena documentada en el proyecto, esta vez dentro de un guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402

_CATALOGO = [
    {"name": "Tortilla integral"}, {"name": "Huevo"}, {"name": "Aguacate"},
    {"name": "Habichuelas rojas"}, {"name": "Berenjena"}, {"name": "Lentejas"},
    {"name": "Harina de trigo"}, {"name": "Avena"}, {"name": "Ají cubanela"},
    {"name": "Arroz blanco"}, {"name": "Guineo"}, {"name": "Pan integral"},
]


@pytest.fixture(scope="module")
def idx():
    return cc.build_culinary_index(_CATALOGO)


def _checks(meal, idx):
    v = (cc._v3_huerfanos({}, meal, idx) + cc._v7a_lista_compra_de_mas({}, meal, idx)
         + cc._v7b_duplicado_incompatible({}, meal, idx) + cc._v7c_seco_sin_coccion({}, meal, idx))
    return {x["check"] for x in v}, v


# ─────────────────────────────────────────────────────────── V7a · lo que se compra y sobra

def test_v7a_dos_tortillas_y_los_pasos_hablan_de_una(idx):
    """El caso del dueño: «declara dos tortillas, pero el procedimiento solo utiliza una»."""
    meal = {"ingredients": ["2 tortillas integrales", "2 huevos"],
            "recipe": ["Mise en place: bate los huevos", "Montaje: rellena la tortilla y enrolla"]}
    checks, _ = _checks(meal, idx)
    assert "V7a" in checks


def test_v7a_calla_si_los_pasos_la_nombran_en_plural(idx):
    """Una sola mención en plural basta: el reparto entre pasos es legítimo."""
    meal = {"ingredients": ["2 tortillas integrales"],
            "recipe": ["Calienta las tortillas y rellénalas"]}
    checks, _ = _checks(meal, idx)
    assert "V7a" not in checks


def test_v7a_no_confunde_media_pieza_con_un_plural(idx):
    """«1½ guineos» y «el guineo» no se contradicen — media pieza no tiene plural.

    Era el ÚNICO falso positivo del detector sobre el golden set.
    """
    meal = {"ingredients": ["1½ guineos medianos"], "recipe": ["Corta el guineo en rodajas"]}
    checks, _ = _checks(meal, idx)
    assert "V7a" not in checks


def test_v7a_el_singular_COLECTIVO_no_es_una_contradiccion(idx):
    """«2 tomates» + «Ralla el tomate» NO es un defecto: tras rallar, el singular es colectivo.

    Este falso positivo exacto ya se había identificado el **2026-07-31** —está escrito en el
    docstring de `test_trampa_fp_plural_singular` y tiene un fixture commiteado para impedirlo— y
    V7a lo reintrodujo por no haber buscado antes lo que el repo ya sabía. Lo cazó ese guard, no
    yo.

    La señal es el VERBO, no una lista de alimentos: «rellena la tortilla» sí significa una
    tortilla, y una lista de «masificables» habría que mantenerla a mano para siempre.
    """
    meal = {"ingredients": ["2 tomates"],
            "recipe": ["Ralla el tomate y resérvalo", "Incorpora el arroz y el tomate rallado"]}
    checks, _ = _checks(meal, idx)
    assert "V7a" not in checks


def test_v7a_el_masificador_se_mira_por_CLAUSULA(idx):
    """«ralla el queso; coloca las tortillas» ralla el queso, no las tortillas."""
    meal = {"ingredients": ["2 tortillas integrales", "2 huevos"],
            "recipe": ["Ralla el queso; rellena la tortilla con el huevo"]}
    checks, _ = _checks(meal, idx)
    assert "V7a" in checks


def test_v7a_no_cuenta_una_cucharada_como_pieza(idx):
    """«2 cucharadas de avena» es una MEDIDA: contarla como pieza haría de cada especia un FP."""
    meal = {"ingredients": ["2 cucharadas de avena"], "recipe": ["Añade la avena"]}
    checks, _ = _checks(meal, idx)
    assert "V7a" not in checks


def test_v7a_no_pisa_a_v3(idx):
    """Si NINGÚN paso lo menciona es huérfano (V3), no sobrante (V7a)."""
    meal = {"ingredients": ["2 tortillas integrales", "2 huevos"],
            "recipe": ["Bate los huevos y sirve"]}
    checks, _ = _checks(meal, idx)
    assert "V3" in checks and "V7a" not in checks


# ─────────────────────────────────────────────────────── V7b · el duplicado incompatible

def test_v7b_el_mismo_alimento_en_piezas_y_en_gramos(idx):
    """«½ ají» y «50 g de ají cubanela»: no se sabe cuánto comprar. V4 no lo ve — compara g con g."""
    meal = {"ingredients": ["½ ají cubanela", "50 g de ají cubanela"],
            "recipe": ["Sofríe el ají cubanela"]}
    checks, _ = _checks(meal, idx)
    assert "V7b" in checks


def test_v7b_calla_con_dos_lineas_de_la_misma_familia(idx):
    """Dos gramajes del mismo alimento son reparto legítimo entre preparaciones, no ambigüedad."""
    meal = {"ingredients": ["100 g de huevo", "50 g de huevo"], "recipe": ["Bate el huevo"]}
    checks, _ = _checks(meal, idx)
    assert "V7b" not in checks


# ────────────────────────────────────────────────────── V7c · lo seco que nadie cuece

def test_v7c_habichuelas_secas_sin_remojo_ni_coccion(idx):
    meal = {"ingredients": ["50 g de habichuelas rojas secas"],
            "recipe": ["Monta las habichuelas rojas sobre la tostada"]}
    checks, v = _checks(meal, idx)
    assert "V7c" in checks


def test_v7c_el_estado_cocido_NO_cuenta_como_haberlo_cocido(idx):
    """La lección 1. «escurre las lentejas cocidas» sin cocerlas es la CONTRADICCIÓN, no la coartada.

    Con `cocid*` dentro de las acciones, V7c no disparaba ni una vez teniendo cinco casos.
    """
    meal = {"ingredients": ["25 g de lentejas secas"],
            "recipe": ["Escurre las lentejas cocidas y sírvelas"]}
    checks, v = _checks(meal, idx)
    assert "V7c" in checks
    assert any("COCIDO sin cocerlo" in x["detail"] for x in v), "debe señalar la contradicción"
    assert any(x["severity"] == "major" for x in v if x["check"] == "V7c")


def test_v7c_la_coccion_debe_estar_en_la_misma_clausula(idx):
    """La lección 2: «cocina la berenjena; añade las lentejas» NO cuece las lentejas."""
    meal = {"ingredients": ["25 g de lentejas secas", "1 berenjena"],
            "recipe": ["Calienta la plancha; cocina la berenjena; añade las lentejas"]}
    checks, _ = _checks(meal, idx)
    assert "V7c" in checks


def test_v7c_calla_cuando_si_se_cuece(idx):
    meal = {"ingredients": ["50 g de habichuelas rojas secas"],
            "recipe": ["Remoja las habichuelas rojas 8 horas y hiérvelas 45 minutos"]}
    checks, _ = _checks(meal, idx)
    assert "V7c" not in checks


def test_v7c_solo_mira_lo_que_cambia_al_cocerse(idx):
    """Una hoja «seca» no es una legumbre cruda: el detector no puede volverse un cazador de adjetivos."""
    meal = {"ingredients": ["1 cdta de orégano seco"], "recipe": ["Espolvorea el orégano"]}
    checks, _ = _checks(meal, idx)
    assert "V7c" not in checks


# ──────────────────────────────────────────────────────── V3 · el prefijo que la cegaba

def test_v3_la_harina_de_trigo_huerfana_bajo_una_harina_de_avena(idx):
    """El caso 028ad9ed64: «muele la avena hasta obtener una harina fina» daba por mencionada la
    harina de trigo. 40 g comprados y jamás usados."""
    meal = {"ingredients": ["40 g de harina de trigo", "¼ taza de avena molida"],
            "recipe": ["Muele la avena hasta obtener una harina fina y forma las arepitas"]}
    checks, v = _checks(meal, idx)
    assert "V3" in checks
    assert any(x["food"] == "Harina de trigo" for x in v if x["check"] == "V3")


def test_v3_sigue_aceptando_el_prefijo_de_una_IDENTIDAD(idx):
    """«Arroz blanco» → «el arroz» tiene que seguir funcionando: es una identidad, no una forma."""
    assert cc._mencionado_por_prefijo("Arroz blanco", "sirve el arroz caliente", ["Arroz blanco"])


def test_las_formas_genericas_son_pocas_y_explicitas():
    """Una lista corta y nombrada: si crece sin medir, V3 empieza a producir falsos positivos."""
    assert "harina" in cc._V3_FORMA_GENERICA and "pasta" in cc._V3_FORMA_GENERICA
    assert "arroz" not in cc._V3_FORMA_GENERICA and "huevo" not in cc._V3_FORMA_GENERICA
    assert len(cc._V3_FORMA_GENERICA) <= 20


# ──────────────────────────────────────────────────────────────── el encadenado

def test_las_tres_capas_entran_en_el_scan():
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    for fn in ("_v7a_lista_compra_de_mas", "_v7b_duplicado_incompatible", "_v7c_seco_sin_coccion"):
        assert f"out.extend({fn}(day, meal, index))" in src, f"{fn} no está encadenada en el scan"


def test_las_tres_son_fail_open():
    """Ninguna puede tumbar una generación: heredan la regla de V5 y V6."""
    for fn in (cc._v7a_lista_compra_de_mas, cc._v7b_duplicado_incompatible, cc._v7c_seco_sin_coccion):
        assert fn({}, {"ingredients": None, "recipe": 42}, None) == []


def test_la_cuarta_clase_queda_anotada_y_no_implementada():
    """«la mitad del ajo queda sin usar» exige seguir cantidades ENTRE pasos. No entra en V7, y que
    esté escrito evita que se dé por cubierta."""
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    assert "usado A MEDIAS" in src and "Queda anotada, no olvidada" in src
