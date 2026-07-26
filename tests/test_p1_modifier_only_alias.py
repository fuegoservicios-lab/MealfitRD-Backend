"""[P1-MODIFIER-ONLY-ALIAS · 2026-07-26] La lista de compras traía PLÁTANO para un desayuno de MANGO.

## El caso

Plan vivo `01d63a5b`, desayuno del día 1: `½ mango maduro`. La lista de compras no tenía mango
— tenía **Plátano, 3 Uds.** El usuario compra plátanos para un desayuno de mango.

    _parse_quantity('½ mango maduro')  ->  (0.5, 'unidad', 'Plátano maduro')

## Por qué

El catálogo tiene `Plátano maduro` con el alias **`'maduro'`**, a secas. El resolvedor busca
cada alias como palabra completa DENTRO del texto (tiers 2 y 4), recorriéndolos por longitud
DESCENDENTE. Así que `'maduro'` (6 letras) se evalúa antes que `'mango'` (5) y gana.

Afecta a todo alimento **masculino cuyo nombre sea más corto que el modificador**. Confirmado
en dos: mango y kiwi. Las femeninas se salvan por casualidad — "pera madura" no casa con
`'maduro'` — lo cual no es una defensa, es suerte ortográfica.

## Lo que lo hacía invisible

El aggregator de la lista y el coherence guard usan **el mismo parser**. Los dos lados
convertían mango en plátano, coincidían, y la guarda no reportaba nada: *coherentemente
equivocados*. Por eso `presence_count: 0` en un plan al que le faltaba una fruta.

Es la séptima instancia en esta sesión de la misma familia —matchear por un token compartido
en vez de por el núcleo— después de `'sal'`⊂`'salsa'`, `'pollo'`⊂`'repollo'`, `'ajo'`⊂`'abajo'`,
`'batido'` adjetivo, `'agua'`⊂`'aguacate'` y la concordancia por última palabra.

tooltip-anchor: P1-MODIFIER-ONLY-ALIAS
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


def _food(s: str) -> str:
    return sc._parse_quantity(s)[2]


# ───────────── 1. los casos vivos ─────────────

def test_el_mango_del_desayuno():
    assert _food("½ mango maduro") == "Mango"


def test_el_mango_con_gramos_entre_parentesis():
    assert _food("1 mango maduro (200g)") == "Mango"


def test_el_kiwi():
    assert _food("1 kiwi maduro") == "Kiwi"


# ───────────── 2. el plátano NO se pierde ─────────────

@pytest.mark.parametrize("linea", [
    "1 platano maduro",
    "2 platanos maduros",
    "1 plátano maduro mediano",
])
def test_el_platano_maduro_sigue_resolviendo(linea):
    assert _food(linea) == "Plátano maduro", linea


def test_el_modificador_SOLO_sigue_resolviendo():
    """Los tiers de match EXACTO conservan los alias-modificador: si alguien escribe "maduro"
    a secas, resolver a plátano maduro es defendible. Lo que se prohíbe es que secuestre un
    texto que ya nombra OTRO alimento."""
    assert _food("maduro") == "Plátano maduro"


# ───────────── 3. no se rompió el resto del catálogo ─────────────

@pytest.mark.parametrize("linea,esperado", [
    ("1 guineo maduro", "Guineo"),
    ("1 aguacate maduro", "Aguacate"),
    ("1 lechosa madura", "Lechosa"),
    ("200g de pollo", "Pechuga de pollo"),
    ("1 taza de arroz blanco", "Arroz blanco"),
    ("queso cottage bajo en grasa", "Queso cottage"),
])
def test_resoluciones_conocidas_intactas(linea, esperado):
    assert _food(linea) == esperado, linea


def test_los_alias_de_dos_palabras_no_se_tocan():
    """"coco maduro" es un alias legítimo de Coco: DOS palabras, no un modificador suelto.
    El filtro solo descarta los que son únicamente un modificador."""
    assert "coco maduro" not in sc._MODIFIER_ONLY_ALIASES


# ───────────── 4. contrato del filtro ─────────────

def test_la_lista_de_modificadores_no_contiene_alimentos():
    """Si alguien mete un alimento aquí, deja de poder resolverse dentro de un texto."""
    for sospechoso in ("mango", "platano", "pollo", "arroz", "queso", "huevo", "coco"):
        assert sospechoso not in sc._MODIFIER_ONLY_ALIASES, sospechoso


def test_esta_sin_acentos():
    """El caller compara contra `strip_accents`; una entrada acentuada nunca casaría."""
    from constants import strip_accents
    for w in sc._MODIFIER_ONLY_ALIASES:
        assert w == strip_accents(w), w
        assert w == w.lower(), w


def test_los_tiers_de_contains_usan_la_lista_filtrada():
    """Ancla de la CLASE: si un tier vuelve a recorrer `all_aliases` buscando DENTRO del texto,
    el modificador vuelve a secuestrar el match."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("_aliases_for_contains = [")
    bloque = src[i:src.index("# ── INTENTO 5", i)]
    assert bloque.count("for alias_stripped, master_name in _aliases_for_contains:") == 2, \
        "los dos tiers de búsqueda-dentro-del-texto deben usar la lista filtrada"
    assert "for alias_stripped, master_name in all_aliases:" in bloque, \
        "los tiers de match EXACTO deben seguir viendo la lista completa"
