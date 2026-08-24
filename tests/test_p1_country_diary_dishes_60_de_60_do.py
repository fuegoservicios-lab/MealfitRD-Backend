"""[P1-COUNTRY-DIARY-DISHES-60-DE-60-DO · 2026-08-23] G25: los 60 platos del diario eran 60
platos dominicanos, y el modo seguimiento se vende como producto independiente.

MEDIDO antes de tocar: `load_dishes()` → 60 claves, las 60 dominicanas. Búsqueda por término:
paella 0 · gazpacho 0 · cocido 0 · fabada 0 · tacos 0 · pozole 0 · arepa 0 · ajiaco 0 · bandeja 0.
Un usuario beta abre «Registrar comida», busca lo que acaba de comer y no encuentra NADA de su
cocina: tiene que componerla ingrediente a ingrediente —el trabajo que el componedor existe para
evitarle— o registrarla mal, que es peor porque contamina sus macros del día.

DE DÓNDE SALEN LOS PLATOS. No hay recetas inventadas: las plantillas por país
(`dish_templates_{es,mx,co,pr}.json`) ya traían `constituents` curados con gramos porque el
generador de PLANES las usa. `scripts/build_country_dishes.py` las lee y resuelve cada
constituyente contra `master_ingredients` — el mismo camino que los 60 dominicanos. Los macros
son la suma de filas del catálogo entre el peso final, no una estimación.

SE AÑADE, NUNCA SE QUITA. `P2-DIARY-CATALOG-COUNTRY` decidió no filtrar por país y esa decisión
es correcta: un dominicano en Madrid sigue comiendo mangú.

LO QUE NO SE PUBLICA. 18 platos mexicanos quedaron fuera porque les faltaba UN ingrediente que
el catálogo no tiene: «Tortilla de maíz», o sea su base. Sus macros no serían aproximados: serían
bajos por la cantidad exacta que falta, y el diario los suma al total del día. Una comida
registrada de menos es peor que una no registrada, porque parece registrada. Eso deja un hueco
REAL del catálogo anotado aquí para que se pueda cerrar.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_DATA = _BACKEND / "data"

_FICHEROS_PAIS = ("spanish_dishes.json", "mexican_dishes.json",
                  "colombian_dishes.json", "puertorican_dishes.json")


def _cargar(nombre: str) -> dict:
    return (json.loads(io.open(_DATA / nombre, encoding="utf-8").read()) or {}).get("dishes") or {}


@pytest.fixture(scope="module")
def platos():
    import food_search
    food_search._dishes_cache = None  # el cache es de proceso; esta suite mide el fichero
    return food_search.load_dishes()


# ── el defecto, en los términos que lo destaparon ─────────────────────────────

@pytest.mark.parametrize("termino", ["paella", "gazpacho", "fabada", "arepa", "ajiaco", "mofongo"])
def test_el_buscador_encuentra_platos_de_las_cocinas_beta(platos, termino):
    hits = [k for k, v in platos.items()
            if termino in k or termino in str(v.get("label", "")).lower()]
    assert hits, f"«{termino}» sigue devolviendo cero: el beta no puede registrar lo que come"


def test_los_60_dominicanos_siguen_todos(platos):
    """Se cierra AÑADIENDO. Si esta cuenta baja, alguien filtró por país — que es justo la
    decisión que P2-DIARY-CATALOG-COUNTRY tomó al revés y por buenas razones."""
    dominicanos = _cargar("dominican_dishes.json")
    assert len(dominicanos) >= 60
    faltan = [k for k in dominicanos if k not in platos]
    assert not faltan, f"desaparecieron platos dominicanos del buscador: {faltan[:5]}"


def test_el_catalogo_crecio_de_verdad(platos):
    dominicanos = _cargar("dominican_dishes.json")
    assert len(platos) > len(dominicanos) + 100, (
        f"solo {len(platos)} platos: los catálogos por país no se están uniendo"
    )


# ── y lo que NO puede colarse ────────────────────────────────────────────────

@pytest.mark.parametrize("fichero", _FICHEROS_PAIS)
def test_ningun_plato_publicado_tiene_ingredientes_sin_resolver(fichero):
    """EL contrato de calidad: un plato al que le falta un ingrediente da macros BAJOS por la
    cantidad exacta que falta, y el diario los suma al día del usuario."""
    malos = {k: v["resolution_coverage"] for k, v in _cargar(fichero).items()
             if v.get("resolution_coverage", 0) < 1.0}
    assert not malos, f"{fichero} publica platos incompletos: {list(malos)[:5]}"


@pytest.mark.parametrize("fichero", _FICHEROS_PAIS)
def test_los_macros_son_positivos_y_plausibles(fichero):
    """Un plato con 0 kcal o con 900 kcal/100g es un error de cómputo, no un plato."""
    for slug, v in _cargar(fichero).items():
        p = v["per_100g"]
        assert 20 <= p["kcal"] <= 600, f"{slug}: {p['kcal']} kcal/100g está fuera de rango"
        assert p["protein"] >= 0 and p["carbs"] >= 0 and p["fats"] >= 0, slug
        assert v["finished_g"] > 0, slug


@pytest.mark.parametrize("fichero", _FICHEROS_PAIS)
def test_la_forma_es_la_misma_que_la_dominicana(fichero):
    """El buscador y el registro leen las mismas claves para todos: una forma distinta rompería
    el componedor sólo para los platos nuevos, y en silencio."""
    dominicano = next(iter(_cargar("dominican_dishes.json").values()))
    obligatorias = {"label", "method", "finished_g", "per_100g", "constituents",
                    "resolution_coverage"}
    assert obligatorias <= set(dominicano), "cambió la forma del catálogo dominicano"
    for slug, v in _cargar(fichero).items():
        faltan = obligatorias - set(v)
        assert not faltan, f"{fichero}:{slug} no trae {faltan}"


def test_el_hueco_del_catalogo_queda_declarado():
    """Los 18 platos mexicanos omitidos no son un capricho: falta «Tortilla de maíz» en
    `master_ingredients`. Mientras el hueco exista, el porqué tiene que estar escrito donde se
    toma la decisión — si alguien añade la fila y regenera, este test se lo recuerda."""
    src = io.open(_BACKEND / "scripts" / "build_country_dishes.py", encoding="utf-8").read()
    assert "Tortilla de maíz" in src
    assert "OMITIDO" in src, "el script dejó de nombrar los platos que no publica"


def test_load_dishes_no_se_cae_si_falta_un_fichero(monkeypatch, tmp_path):
    """Degradar a los que haya es correcto; reventar el registro por alimento no lo es."""
    import food_search
    food_search._dishes_cache = None
    real = Path.read_text

    def _falla_para_espanoles(self, *a, **kw):
        if self.name == "spanish_dishes.json":
            raise FileNotFoundError(self.name)
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _falla_para_espanoles)
    d = food_search.load_dishes()
    assert len(d) > 60, "sin un fichero de país el buscador debe seguir sirviendo el resto"
    food_search._dishes_cache = None
