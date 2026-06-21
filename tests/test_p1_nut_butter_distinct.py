"""[P1-NUT-BUTTER-DISTINCT · 2026-06-21] La mantequilla/crema/pasta de un fruto
seco es un PRODUCTO DISTINTO del fruto seco crudo y NO debe consolidarse a la
nuez base.

Bug en vivo (2026-06-21): una receta decía "mantequilla de maní" pero la lista de
compras mostraba "Maní" (maní crudo, frasco RD$185) en vez de "Mantequilla de maní"
(frasco RD$117). El usuario compraría el producto equivocado y la receta contradecía
la lista.

Causa raíz: `canonicalize_frutos_secos` matcheaba `\bman[íi]\b` DENTRO de
"mantequilla de maní" → retornaba "Maní". El maní es fruto seco, así que la
mantequilla de maní se colapsaba al maní crudo.

Fix: un early-return None para las formas mantequilla/crema/pasta de cualquier
fruto seco (productos distintos) ANTES de las reglas de consolidación. Devolver
None deja que el `master_map` resuelva el producto por su propio nombre
("Mantequilla de maní", que tiene "crema de maní" como alias).

La función es el SSOT compartido por:
  - el aggregator de la lista de compras (`aggregate_and_deduct_shopping_list`)
  - el coherence guard (`_canonicalize_for_coherence`)
…así que el mismo fix mantiene ambos lados simétricos (recetas ↔ lista).
"""
import re

import shopping_calculator
from shopping_calculator import canonicalize_frutos_secos


# ---------------------------------------------------------------------------
# 1. Regresión directa: las formas mantequilla/crema/pasta NO colapsan al crudo
# ---------------------------------------------------------------------------
def test_mantequilla_de_mani_no_colapsa_a_mani():
    # EL bug reportado en vivo: la mantequilla de maní NO es maní crudo.
    assert canonicalize_frutos_secos("Mantequilla de maní") is None
    assert canonicalize_frutos_secos("mantequilla de mani") is None


def test_crema_y_pasta_y_butter_de_frutos_secos_no_colapsan():
    formas_distintas = [
        "crema de maní",
        "crema de mani",
        "pasta de maní",
        "mantequilla de almendra",
        "mantequilla de almendras",
        "crema de avellana",
        "peanut butter",
        "almond butter",
    ]
    for nombre in formas_distintas:
        assert canonicalize_frutos_secos(nombre) is None, (
            f"{nombre!r} es un producto distinto del fruto seco crudo; debe ser "
            f"None para que master_map lo resuelva por su propio nombre, NO colapsar."
        )


# ---------------------------------------------------------------------------
# 2. No-regresión: los frutos secos CRUDOS siguen consolidando a su canónico
# ---------------------------------------------------------------------------
def test_frutos_secos_crudos_siguen_consolidando():
    casos = {
        "maní": "Maní",
        "mani tostado": "Maní",
        "100g de maní tostado sin sal": "Maní",
        "cacahuate": "Maní",
        "cacahuetes": "Maní",
        "almendras tostadas": "Almendras",
        "almendra laminada": "Almendras",
        "nueces": "Nueces",
        "nuez de castilla": "Nueces",
        "avellanas": "Avellanas",
        "pistachos": "Pistachos",
        "pecanas": "Pecanas",
    }
    for nombre, esperado in casos.items():
        assert canonicalize_frutos_secos(nombre) == esperado, (
            f"{nombre!r} crudo debe seguir consolidando a {esperado!r}."
        )


def test_la_palabra_mani_aislada_sigue_siendo_mani():
    # Sanity: el early-return solo aplica a "<butter> de <X>", no a "X" solo.
    assert canonicalize_frutos_secos("maní") == "Maní"
    assert canonicalize_frutos_secos("mani") == "Maní"


# ---------------------------------------------------------------------------
# 3. Anchor del tooltip en el source de prod (un renombre falla el test antes
#    de romper producción)
# ---------------------------------------------------------------------------
def test_tooltip_anchor_presente_en_source():
    src = re.sub(
        r"\s+",
        " ",
        open(shopping_calculator.__file__, encoding="utf-8").read(),
    )
    assert "P1-NUT-BUTTER-DISTINCT" in src, (
        "El marker P1-NUT-BUTTER-DISTINCT debe vivir en shopping_calculator.py "
        "como ancla del early-return; si se renombra, este test lo detecta."
    )
