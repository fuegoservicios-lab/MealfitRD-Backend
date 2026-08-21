"""[P1-SHOPLIST-SANITY-CAP · 2026-08-21] La lista mandaba comprar 15 sobres de pimienta y 10
frascos de orégano.

Medido HOY sobre los planes vivos (`aggregated_shopping_list_weekly`):

    Pimienta negra      15 sobres   (14,2 g cada uno)
    Canela en polvo     14 sobres
    Orégano             10 frascos  ·  RD$810
    Maíz dulce          9 latas     ·  RD$495     ← NO es un condimento
    Alcachofa           8 unidades  ·  RD$2.760   ← NO es despensa

NO ES UN DEFECTO DEL SISTEMA DE PAÍSES: aparece igual en planes dominicanos. Lo cierra esta ola
porque la auditoría lo encontró de paso y porque el usuario beta lo sufre igual — pero sin el
consuelo del precio para notar el absurdo, ya que en beta los importes van suprimidos.

Y no es sólo legibilidad: **está preciado**. Los 10 frascos de orégano valen RD$810 y entran en
`shopping_cost_summary`, así que contaminan el banner de presupuesto y la reconciliación
costo-real-vs-presupuesto. Un plan puede parecer «excedido» por un especiero.

POR QUÉ AQUÍ CAPAR SÍ ES HONESTO, Y EN P1-COUNTRY-KEEP-RESPECT-QTY NO LO ERA. Allí el 150 g fijo
IGNORABA una demanda real (653 g de almejas) y el usuario compraba de menos. Aquí la demanda
ESTIMADA es la que está mal: un frasco de orégano de 90 g dura meses, y «1 orégano» repetido 30
días no son 30 frascos. Se acota la presentación porque el consumo real de un condimento no
escala con el número de recetas que lo mencionan.

EL PREDICADO SALE DEL DATO, NO DE UNA LISTA A MANO. No existe categoría «especias» en
`master_ingredients` — todo cae en 'Despensa' junto al arroz y el maíz en lata. Lo que sí separa
es el ENVASE: las especias vienen en 14-100 g y la comida de despensa en 425-907 g. Así que el
tope se aplica a Despensa con `container_weight_g` pequeño, y se adapta solo cuando el catálogo
crezca. Una lista de nombres habría que mantenerla, y su fallo sería silencioso.

Cubre:
  A. El tope acota los condimentos y escala con la duración del ciclo.
  B. NO toca comida de verdad: ni el maíz en lata, ni el arroz, ni las verduras.
  C. El costo estimado se recalcula sobre la cantidad acotada (si no, el banner seguiría sucio).
  D. Knob de rollback.
  E. Parser-based.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SC_PATH = _BACKEND_ROOT / "shopping_calculator.py"


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


# ── A. El tope ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dias,tope", [(7, 1), (15, 2), (30, 3)])
def test_el_tope_escala_con_la_duracion_del_ciclo(sc, dias, tope):
    """Una pizca son ~0,3 g y un sobre 14 g: tres comidas al día durante un mes son ~27 g, o sea
    dos sobres. El tope da tres — generoso frente al consumo real y a años luz de los quince que
    la lista pedía."""
    assert sc._condiment_package_cap(dias) == tope


@pytest.mark.parametrize("dias", [1, 0, None, "basura", 999])
def test_el_tope_nunca_es_absurdo_ni_revienta(sc, dias):
    """Corre en el camino caliente del agregador: una excepción aquí rompe la lista entera. Y el
    tope nunca puede ser 0 — eso borraría el condimento de la lista, que es el defecto contrario
    (lista incompleta sin aviso, el miedo explícito del dueño)."""
    got = sc._condiment_package_cap(dias)
    assert isinstance(got, int) and 1 <= got <= 4


@pytest.mark.parametrize("nombre,envase_g", [
    ("Pimienta negra", 14.2), ("Canela en polvo", 14.2), ("Orégano", 90.0), ("Tomillo", 14.0),
])
def test_un_condimento_se_reconoce_por_su_envase(sc, nombre, envase_g):
    """Los cuatro que la lista viva desbordó."""
    assert sc._is_condiment_presentation("Despensa", envase_g) is True


# ── B. No toca comida de verdad ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre,categoria,envase_g", [
    ("Maíz dulce en granos", "Despensa", 425.0),   # 9 latas puede ser legítimo para un mes
    ("Arroz blanco", "Despensa", 907.0),
    ("Alcachofa", "Vegetales", 128.0),             # 8 alcachofas = 8 comidas con alcachofa
    ("Leche entera", "Lácteos", 946.0),
    ("Pollo", "Proteínas", 0.0),
])
def test_la_comida_real_no_se_capa(sc, nombre, categoria, envase_g):
    """El error opuesto —y peor— sería capar comida: el usuario compraría de menos y se quedaría
    sin cenar. Por eso el predicado es estrecho por dos lados: categoría Despensa Y envase
    pequeño. La alcachofa ni siquiera es Despensa; el maíz lo es pero viene en lata de 425 g."""
    assert sc._is_condiment_presentation(categoria, envase_g) is False


def test_un_envase_desconocido_no_se_capa(sc):
    """Sin `container_weight_g` no sabemos qué es. Fail-open: no capar. Capar a ciegas podría
    dejar corto un alimento de verdad, y el coste de no capar es un ítem feo — asimetría clara."""
    assert sc._is_condiment_presentation("Despensa", None) is False
    assert sc._is_condiment_presentation("Despensa", 0) is False


# ── C. El costo sigue a la cantidad ─────────────────────────────────────────────────────────────

def test_el_costo_se_recalcula_sobre_la_cantidad_acotada(sc):
    """Si se capa la cantidad y no el costo, el banner de presupuesto sigue contando RD$810 de
    orégano — o sea que el defecto que más duele (un plan que parece excedido por un especiero)
    sobreviviría al arreglo.

    Se mide el EFECTO sobre un objeto real, no la presencia del literal en el fuente: la primera
    versión buscaba «estimated_cost» en una ventana de 3000 chars tras el marker y sólo alcanzaba
    la prosa del comentario — otra vez el número mágico en vez del cuerpo de la función."""
    market_obj = {
        "name": "Orégano", "market_qty_numeric": 10.0, "market_qty": "10",
        "market_unit": "frasco", "display_qty": "10 frascos",
        "estimated_cost_rd": 810.0,
    }
    recorto = sc._apply_condiment_sanity_cap(
        market_obj, {"container_weight_g": 90.0}, "DESPENSA", 30)
    assert recorto is True
    assert market_obj["market_qty_numeric"] == 3.0
    assert market_obj["estimated_cost_rd"] == pytest.approx(243.0, abs=1.0), (
        "el costo no siguió a la cantidad: el banner de presupuesto seguiría contando 10 frascos"
    )
    assert "3" in market_obj["display_qty"]


def test_lo_que_ya_esta_bajo_el_tope_no_se_toca(sc):
    """Control: un condimento con 2 frascos en un ciclo mensual está dentro del tope y sale
    intacto — el helper no puede convertirse en un redondeo universal."""
    obj = {"name": "Comino", "market_qty_numeric": 2.0, "market_unit": "pote",
           "display_qty": "2 potes", "estimated_cost_rd": 120.0}
    assert sc._apply_condiment_sanity_cap(obj, {"container_weight_g": 28.0}, "DESPENSA", 30) is False
    assert obj["market_qty_numeric"] == 2.0 and obj["estimated_cost_rd"] == 120.0


def test_el_maiz_en_lata_no_se_capa_aunque_desborde(sc):
    """El caso que más importa del lado negativo: 9 latas de maíz en un ciclo mensual son
    plausibles, y capar comida deja al usuario sin cenar. El envase (425 g) lo excluye."""
    obj = {"name": "Maíz dulce en granos", "market_qty_numeric": 9.0, "market_unit": "lata",
           "display_qty": "9 latas", "estimated_cost_rd": 495.0}
    assert sc._apply_condiment_sanity_cap(obj, {"container_weight_g": 425.0}, "DESPENSA", 30) is False
    assert obj["market_qty_numeric"] == 9.0


# ── D. Knob ─────────────────────────────────────────────────────────────────────────────────────

def test_el_knob_permite_revertir(sc, monkeypatch):
    """Camino caliente del agregador ⇒ knob propio, según la convención del repo."""
    monkeypatch.setenv("MEALFIT_SHOPLIST_SANITY_CAP", "false")
    assert sc._shoplist_sanity_cap_enabled() is False
    monkeypatch.delenv("MEALFIT_SHOPLIST_SANITY_CAP", raising=False)
    assert sc._shoplist_sanity_cap_enabled() is True


# ── E. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_predicado_no_es_una_lista_de_nombres():
    """Una lista de especias a mano habría que mantenerla cada vez que el catálogo crece, y su
    fallo sería SILENCIOSO (una especia nueva desbordando la lista sin que nadie se entere). El
    predicado sale del dato: categoría + peso del envase."""
    src = _SC_PATH.read_text(encoding="utf-8", errors="replace")
    i = src.find("def _is_condiment_presentation")
    assert i > 0
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    for especia in ("oregano", "orégano", "pimienta", "canela", "comino"):
        assert especia not in cuerpo.lower(), (
            f"el predicado nombra «{especia}»: es una lista a mano disfrazada"
        )
