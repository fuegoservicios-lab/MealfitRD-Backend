"""[P1-CULINARY-HASTA-DORAR · 2026-08-19] «hasta dorar» describe el punto de cocción,
no ordena saltear.

`VERB_TO_METHOD` mapeaba `dora(?!d[oa]s?\\b)\\w*` → `saltear`. El lookahead excluye
"dorado/dorada/dorados/doradas" (participios, que describen un resultado) pero **no el
infinitivo**, así que «Hornea las papas hasta dorar» disparaba V1 contra todo alimento
del paso sin `saltear` en `prep_methods`.

MEDIDO, no supuesto (33 planes reales de prod, 2026-08-19): **12 de 63 violaciones V1
eran esto** — 19% de ruido. Y los acusados eran quienes menos tocan el fuego: Aceite de
oliva, Miel, Vainilla, Mango, Linaza, Plátano maduro. Pasa porque V1 acusa a cualquier
alimento nombrado en un paso largo multi-cláusula, y estos pasos («El Toque de Fuego:
…») encadenan media receta.

Importa más de lo que parece: con el guard en `warn` esto es ruido de telemetría, pero
`P1-CULINARY-CONTRACT-BLOCK` (F2) quiere escalarlo a `block`. Un 19% de falsos positivos
convierte esa escalada en rechazos de planes buenos.

Este test ancla las dos mitades: que el punto de cocción NO dispare, y que el imperativo
SÍ lo siga haciendo — «dora» es el sellado que abre cada guiso dominicano y romperlo
sería la regresión que la Task-5 del P-fix original ya pagó una vez.

tooltip-anchor: P1-CULINARY-HASTA-DORAR
"""
from __future__ import annotations

import re

import pytest

import culinary_coherence as cc


def _metodos(texto: str) -> set:
    """Métodos que los patrones de producción extraen de un paso."""
    return {met for rx, met in cc._VERB_RES if rx.search(cc._norm(texto))}


# ───────────────── el punto de cocción NO es una instrucción ─────────────────

@pytest.mark.parametrize("paso", [
    "Hornea las papas 18 minutos hasta dorar.",
    "Cocina las arepitas hasta dorarlas por ambos lados.",
    "Hierve la papa y luego llévala al horno hasta dorarla.",
    "Deja el pan en el horno hasta dorarlos ligeramente.",
])
def test_hasta_dorar_no_produce_saltear(paso):
    assert "saltear" not in _metodos(paso), (
        f"«{paso}» describe el punto de cocción; no ordena saltear")


# ───────────────── el imperativo SÍ sigue siendo saltear ─────────────────────

@pytest.mark.parametrize("paso", [
    "Dora la cebolla en la sartén.",
    "Dóralo por ambos lados.",
    "Dora la pechuga de pollo y reserva.",
    "Sofríe el ajo y el ají.",
    "Saltea el pimiento 3 minutos.",
])
def test_el_imperativo_sigue_disparando(paso):
    assert "saltear" in _metodos(paso), (
        f"«{paso}» SÍ es una instrucción de saltear; romper esto revive la regresión "
        "de la Task-5 (el sellado que abre cada guiso dominicano)")


def test_los_participios_siguen_exentos():
    """Lo que el patrón ya excluía antes de este fix no debe cambiar."""
    for paso in ("Sirve con la cebolla dorada.", "Usa los plátanos dorados.",
                 "Agrega el sofrito ya preparado."):
        assert "saltear" not in _metodos(paso), paso


# ─────────────────────── el mecanismo, no solo el efecto ────────────────────

def test_el_lookbehind_vive_en_el_patron_de_saltear():
    """Guard estructural: si alguien reescribe la alternancia y pierde el lookbehind,
    los tests de arriba se caen — pero este dice POR QUÉ, que es lo que ahorra la
    segunda investigación."""
    clave = next((k for k, v in cc.VERB_TO_METHOD.items() if v == "saltear"), None)
    assert clave, "no existe ninguna clave que resuelva a 'saltear'"
    assert "(?<!hasta )dora" in clave, (
        "el patrón de 'saltear' perdió el lookbehind `(?<!hasta )`: «hasta dorar» "
        "volverá a acusar de salteado a aceites, miel y frutas")


def test_sigue_siendo_una_sola_clave_para_saltear():
    """El P-fix original fusionó saltear/sofreír/dorar en UNA clave a propósito: dos
    claves que resuelven al mismo método producían la violación V1 DUPLICADA."""
    claves = [k for k, v in cc.VERB_TO_METHOD.items() if v == "saltear"]
    assert len(claves) == 1, (
        f"{len(claves)} claves resuelven a 'saltear'; duplicarán cada violación")


def test_el_lookbehind_es_de_ancho_fijo():
    """`re` solo admite lookbehind de ancho fijo. Si alguien lo generaliza a algo
    variable (p.ej. `(?<!hasta\\s+)`), el módulo revienta al importar — y como el scan
    es fail-open, el fallo se vería como 'ninguna violación', no como un error."""
    clave = next(k for k, v in cc.VERB_TO_METHOD.items() if v == "saltear")
    re.compile(clave)  # basta con que compile


# ──────────────────────────── el efecto de punta a punta ────────────────────

def test_el_escaneo_completo_no_acusa_al_aceite_por_un_hasta_dorar():
    """Reproducción del caso real medido: un paso largo que menciona aceite y termina
    en «hasta dorar». Antes acusaba al aceite de estar salteado."""
    catalogo = [
        {"name": "Aceite de oliva", "prep_methods": ["ninguno"], "ready_to_eat": True},
        {"name": "Papa", "prep_methods": ["hervir", "hornear", "freir"], "ready_to_eat": False},
    ]
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Papas al horno",
        "ingredients": ["200 g Papa", "10 g Aceite de oliva"],
        "recipe": ["Mezcla la papa con el aceite de oliva y hornea 20 minutos hasta dorar."],
    }]}]}
    viols = cc.culinary_contract_scan(plan, catalogo)
    acusados = [v["food"] for v in viols]
    assert "Aceite de oliva" not in acusados, (
        f"el aceite sigue acusado por un «hasta dorar»: {viols}")
