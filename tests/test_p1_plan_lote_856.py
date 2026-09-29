# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-856 · 2026-09-29] Alias que faltan en el catálogo de los países beta.

LO QUE SE VIO (validación beta G24, ES D1 Almuerzo): la lista dice «100 g de filete de merluza» y el paso «mide 170 g
de merluza»; la comida sale con `_recipe_contract_final=None` porque el índice del contrato no empareja «merluza» con
NINGUNA fila — y los medidores de números (V4) ni lo miden. La evidencia suponía una fila «Filete de merluza» sin su
alias corto: NO EXISTE (354 filas leídas el 29-sep). La merluza es un pescado blanco magro sin fila propia, igual que el
chillo, que ya es alias de «Filete de pescado blanco» (P2-WHITE-FISH-ALIAS-SPLIT: un alias resuelve a UNA fila; mero y
tilapia salieron de la genérica porque tienen fila, la merluza no). Además la lista de compras la perdía entera.

LO QUE SE MIDIÓ (sin IA; catálogo = SELECT de prod; corpus = 834 planes guardados, 10 480 comidas, 432 beta):
  · auditoría de las 140 filas beta (núcleo sin «Filete de», «en lata», «fresco»…, y su cabeza): 74 núcleos, 3 ya
    resuelven, 64 ambiguos (token o alias de otra fila, varias filas beta, forma genérica, o el SSOT de tracking los
    colapsa a otro término), 7 sin ambigüedad estructural, de los que 5 lo son de SENTIDO (chocolate, jarabe, especias,
    sémola, bolitas) y 2 entran (panceta, ron);
  · pasada por el corpus (líneas y menciones numéricas de los pasos que el índice no resuelve): merluza (ES, 3 comidas),
    almendras laminadas (ES/CO/DO, 3), chile piquín en polvo (MX, 6);
  · replay antes/después: 0 claves del índice perdidas o con otro dueño; sólo cambian textos donde casa un alias nuevo;
    de 7 580 líneas únicas re-parseadas, 5 cambian (las esperadas, ninguna DO); el contrato reescribe 1 comida (la de
    la evidencia: «mide 100 g de merluza») y el scan capa 1 gana 1 V4 (el mismo 170 frente a 100, ahora medido).

QUÉ ANCLA ESTE TEST (parser + la foto del catálogo, sin DB — el dueño aprueba y aplica la migración):
  A. idempotencia y forma; B. los pares exactos y la paridad UPDATE↔sanity; C. ningún núcleo ambiguo entra;
  D. aplicada a la foto: cada alias pasa de no resolver a resolver a SU fila, sin robar ninguna clave; E. el caso de la
  evidencia (contrato y V4) rojo→verde; F. la Nevera; G. el índice del contrato ve los alias nuevos sin reiniciar
  (knob `MEALFIT_RECIPE_CONTRACT_INDEX_BY_CONTENT`). tooltip-anchor: P1-PLAN-LOTE-856
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_MIG_NAME = "p1_plan_lote_856_alias_beta_2026_09_29.sql"
_MIG = _BACKEND / "migrations" / _MIG_NAME
_FIX = _BACKEND / "tests" / "fixtures" / "alias_catalogo_2026_09_29.json"

ESPERADOS = {
    ("Filete de pescado blanco", "merluza"),
    ("Filete de pescado blanco", "filete de merluza"),
    ("Almendras fileteadas", "almendras laminadas"),
    ("Chile en polvo", "chile piquín en polvo"),
    ("Chile en polvo", "chile piquin en polvo"),
    ("Panceta ibérica", "panceta"),
    ("Ron de cocina", "ron"),
}

# Núcleos que la auditoría descartó por AMBIGUOS (lista para el dueño): ninguno puede entrar como alias.
AMBIGUOS = {
    "chile", "chiles", "frijol", "frijoles", "queso", "chorizo", "jamon", "crema", "salsa", "tortilla", "tortillas", "pan",
    "harina", "masa", "mezcla", "aceite", "almendra", "almendras", "nueces", "nuez", "azucar", "huevos", "papas", "hoja",
    "carne", "suero", "panecillos", "galletas", "lomo", "longaniza", "salchicha", "sazonador", "chuleta", "aderezo",
    "judias", "arandanos", "membrillo", "ensalada", "chili", "flor", "aceitunas", "platano", "maiz", "soya", "pollo",
    "chocolate", "jarabe", "especias", "semola", "bolitas", "pulpa", "arroz rojo", "salsa verde",
}

# ES D1 Almuerzo de la validación G24 (tal cual se guardó, con `_recipe_contract_final=None`).
MERLUZA_ES = {
    "name": "Merluza a la plancha con pimentón, batata asada, brócoli y percebes",
    "ingredients": [
        "100 g de filete de merluza", "1 batata mediana", "250 g de brócoli", "¾ cucharadita de pimentón dulce",
        "1 diente de ajo", "2 cucharadas de aceite de oliva virgen extra", "½ limón", "155 g de percebes cocidos",
    ],
    "recipe": [
        "Mise en place: corta la batata en rodajas finas, separa el brócoli en ramilletes y pica el ajo; mide 170 g de "
        "merluza, ¾ cucharadita de pimentón, 2 cucharadas de aceite y el zumo de medio limón.",
        "El Toque de Fuego: asa la batata con la mitad del aceite en horno precalentado a 220 °C durante 20-25 minutos, "
        "hasta que esté tierna al pincharla. Cuece el brócoli al vapor 6-8 minutos. Cocina la merluza en sartén con el "
        "aceite restante, el ajo y el pimentón, 3-4 minutos por lado, hasta que esté opaca y se desmenuce fácilmente. "
        "Cocina percebes a la plancha o hervidos y sírvelos como proteína del plato.",
        "Montaje: sirve la merluza con el zumo de limón, las rodajas de batata asada y el brócoli.",
    ],
}

_UPDATE = re.compile(
    r"UPDATE public\.master_ingredients\s+SET aliases = array_append\(COALESCE\(aliases, ARRAY\[\]::text\[\]\), "
    r"'((?:[^']|'')*)'\)\s+WHERE name = '((?:[^']|'')*)'\s+AND NOT \('((?:[^']|'')*)' = ANY\(COALESCE\(aliases, "
    r"ARRAY\[\]::text\[\]\)\)\);")
_PAR = re.compile(r"\(\s*'((?:[^']|'')*)'\s*,\s*'((?:[^']|'')*)'\s*\)")


def _n(s: str) -> str:
    from constants import strip_accents
    return " ".join(strip_accents(str(s).lower()).split())


@pytest.fixture(scope="module")
def sql():
    assert _MIG.exists(), f"falta la migración {_MIG_NAME}"
    return _MIG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pares(sql):
    out = []
    for alias, fila, guarda in _UPDATE.findall(sql):
        assert alias == guarda, f"la guarda de idempotencia no es la del alias: {alias!r} vs {guarda!r}"
        out.append((fila.replace("''", "'"), alias.replace("''", "'")))
    return out


@pytest.fixture(scope="module")
def foto():
    return json.loads(_FIX.read_text(encoding="utf-8"))["filas"]


def _aplicar(foto, pares):
    cat = copy.deepcopy(foto)
    por_nombre = {r["name"]: r for r in cat}
    for fila, alias in pares:
        al = por_nombre[fila].get("aliases") or []
        if alias not in al:
            por_nombre[fila]["aliases"] = al + [alias]
    return cat


# ── A. forma e idempotencia ───────────────────────────────────────────────────────────────

def test_a_marker_y_forma_idempotente(sql):
    assert "[P1-PLAN-LOTE-856 · 2026-09-29]" in sql
    codigo = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    # sólo añade alias: ni DDL, ni borrados, ni altas, ni columnas de precio o nutrientes
    for prohibido in ("CREATE ", "ALTER ", "DROP ", "DELETE ", "INSERT ", "TRUNCATE", "price_", "kcal_", "_per_100g",
                      "array_remove", "SET name"):
        assert prohibido not in codigo, prohibido
    n_updates = codigo.count("UPDATE public.master_ingredients")
    assert n_updates == len(_UPDATE.findall(sql)) == 7, "cada UPDATE lleva la guarda «AND NOT (alias = ANY(...))»"
    assert codigo.count("DO $$") == 3 and codigo.count("END $$;") == 3
    assert codigo.count("RAISE EXCEPTION") >= 4


def test_a_sin_surrogates_ni_fuera_del_bmp(sql):
    assert not any(0xD800 <= ord(c) <= 0xDFFF or ord(c) > 0xFFFF for c in sql)


# ── B. pares exactos y paridad con las sanity ─────────────────────────────────────────────

def test_b_pares_exactos(pares):
    assert len(pares) == len(set(pares)), "alias repetido"
    assert set(pares) == ESPERADOS


def test_b_las_sanity_verifican_los_mismos_pares(sql, pares):
    bloques = re.findall(r"FROM \(VALUES\n(.*?)\n\s*\) AS v\(fila, alias\)", sql, re.S)
    assert len(bloques) == 2, "la sanity previa y la posterior listan los pares"
    for b in bloques:
        assert {(f.replace("''", "'"), a.replace("''", "'")) for f, a in _PAR.findall(b)} == set(pares)
    filas = re.search(r"FROM \(VALUES (.*?)\) AS f\(fila\)", sql, re.S)
    assert filas, "la sanity previa exige que existan las filas destino"
    assert set(re.findall(r"'((?:[^']|'')*)'", filas.group(1))) == {f for f, _ in pares}


# ── C. ningún núcleo ambiguo ──────────────────────────────────────────────────────────────

def test_c_ningun_nucleo_ambiguo_entra(pares):
    assert not ({_n(a) for _, a in pares} & AMBIGUOS)


def test_c_la_foto_confirma_la_ambiguedad_de_los_descartes(foto):
    """Los descartes no son opinión: cada uno de estos es token o alias de ≥2 filas del catálogo vivo."""
    from culinary_coherence import _index_entry
    for nucleo in ("chile", "frijoles", "queso", "chorizo", "jamon", "crema", "salsa", "tortilla", "almendra", "nueces"):
        rx = _index_entry(nucleo, nucleo, {})["rx"]
        duenos = {r["name"] for r in foto for k in [r["name"]] + list(r["aliases"] or []) if rx.search(_n(k))}
        assert len(duenos) >= 2, (nucleo, duenos)


# ── D. aplicada a la foto ─────────────────────────────────────────────────────────────────

def test_d_la_foto_no_tiene_fila_filete_de_merluza(foto):
    nombres = {_n(r["name"]) for r in foto}
    assert "filete de merluza" not in nombres and "merluza" not in nombres
    assert not any("merluza" in _n(a) for r in foto for a in r["aliases"] or [])
    assert "chillo" in (next(r for r in foto if r["name"] == "Filete de pescado blanco")["aliases"])


def test_d_cada_alias_pasa_a_resolver_a_su_fila_y_a_ninguna_otra(foto, pares):
    from culinary_coherence import build_culinary_index, find_catalog_foods
    antes, despues = build_culinary_index(foto), build_culinary_index(_aplicar(foto, pares))
    for fila, alias in pares:
        assert find_catalog_foods(alias, antes) == [], (alias, find_catalog_foods(alias, antes))
        assert find_catalog_foods(alias, despues) == [fila], (alias, find_catalog_foods(alias, despues))
    # un alias sólo AÑADE claves: ninguna existente se pierde ni cambia de dueño
    assert set(antes) <= set(despues)
    assert all(antes[k]["name"] == despues[k]["name"] for k in antes)
    # y cada alias nuevo lo reclama una sola fila (nombre o alias), sin acentos ni mayúsculas
    cat = _aplicar(foto, pares)
    for fila, alias in pares:
        duenos = {r["name"] for r in cat for k in [r["name"]] + list(r["aliases"] or []) if _n(k) == _n(alias)}
        assert duenos == {fila}, (alias, duenos)


def test_d_los_nucleos_ambiguos_siguen_sin_resolver(foto, pares):
    from culinary_coherence import build_culinary_index, find_catalog_foods
    despues = build_culinary_index(_aplicar(foto, pares))
    for nucleo in ("chile", "frijoles", "queso", "almendras", "nueces", "platano", "chocolate", "jarabe", "tortilla"):
        assert find_catalog_foods(nucleo, despues) == [], nucleo


def test_d_las_filas_destino(foto, pares):
    por = {r["name"]: r for r in foto}
    beta = {f for f, _ in pares if por[f]["pais_beta"]}
    assert beta == {"Chile en polvo", "Panceta ibérica", "Ron de cocina"}
    # las dos filas DO que usan los planes beta, declaradas: la genérica del pescado blanco y las almendras
    assert {f for f, _ in pares} - beta == {"Filete de pescado blanco", "Almendras fileteadas"}


# ── E. el caso de la evidencia: contrato y medidor ───────────────────────────────────────

def test_e_contrato_merluza_rojo_a_verde(foto, pares):
    import recipe_contract
    from culinary_coherence import build_culinary_index
    sin, con = copy.deepcopy(MERLUZA_ES), copy.deepcopy(MERLUZA_ES)
    r0 = recipe_contract.reconcile_meal(sin, build_culinary_index(foto))
    r1 = recipe_contract.reconcile_meal(con, build_culinary_index(_aplicar(foto, pares)))
    assert r0["reescritas"] == 0 and "mide 170 g de merluza" in sin["recipe"][0]
    assert r1["reescritas"] == 1 and "mide 100 g de merluza" in con["recipe"][0]
    assert con["ingredients"] == MERLUZA_ES["ingredients"], "la lista manda: no se toca"


def test_e_el_medidor_v4_lo_ve(foto, pares):
    from culinary_coherence import culinary_contract_scan
    plan = {"days": [{"day": 1, "meals": [copy.deepcopy(MERLUZA_ES)]}]}
    v0 = [v for v in culinary_contract_scan(plan, foto) if v["check"] == "V4"]
    v1 = [v for v in culinary_contract_scan(plan, _aplicar(foto, pares)) if v["check"] == "V4"]
    assert v0 == []
    assert [(v["food"], v["detail"]) for v in v1] == [
        ("Filete de pescado blanco", "ingrediente declara 100 g, pasos declaran 170 g")]


# ── F. la Nevera ──────────────────────────────────────────────────────────────────────────

def test_f_nevera_sinonimos_si_calificativos_no(foto, pares, monkeypatch):
    import constants
    import shopping_calculator as sc
    cat = _aplicar(foto, pares)
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: cat)
    constants._reset_pantry_alias_index_cache()
    try:
        assert constants.pantry_names_match("merluza", "Filete de pescado blanco")
        assert constants.pantry_names_match("almendras laminadas", "Almendras fileteadas")
        # «panceta» ⊂ «Panceta ibérica»: un calificativo quitado no es sinónimo en la Nevera (P2-PANTRY-REGIONAL-SYNONYMS)
        assert not constants.pantry_names_match("panceta", "Panceta ibérica")
    finally:
        constants._reset_pantry_alias_index_cache()


# ── G. el índice del contrato ve los alias nuevos sin reiniciar ───────────────────────────

def _cat_min(con_merluza: bool):
    fila = {"name": "Filete de pescado blanco", "category": "Proteínas", "aliases": ["chillo"],
            "prep_methods": ["plancha"], "ready_to_eat": False}
    if con_merluza:
        fila["aliases"] = ["chillo", "merluza"]
    return [fila, {"name": "Ajo", "category": "Vegetales", "aliases": [], "prep_methods": ["saltear"], "ready_to_eat": False}]


def test_g_index_for_catalog_ve_un_alias_nuevo_con_el_mismo_numero_de_filas(monkeypatch):
    import recipe_contract
    from culinary_coherence import find_catalog_foods
    monkeypatch.delenv("MEALFIT_RECIPE_CONTRACT_INDEX_BY_CONTENT", raising=False)
    monkeypatch.setattr(recipe_contract, "_INDEX_CACHE", {"index": None, "n": -1})
    assert find_catalog_foods("merluza", recipe_contract.index_for_catalog(_cat_min(False))) == []
    assert find_catalog_foods("merluza", recipe_contract.index_for_catalog(_cat_min(True))) == ["Filete de pescado blanco"]


def test_g_knob_apagado_conserva_la_cache_por_tamano(monkeypatch):
    import recipe_contract
    from culinary_coherence import find_catalog_foods
    monkeypatch.setenv("MEALFIT_RECIPE_CONTRACT_INDEX_BY_CONTENT", "false")
    monkeypatch.setattr(recipe_contract, "_INDEX_CACHE", {"index": None, "n": -1})
    assert find_catalog_foods("merluza", recipe_contract.index_for_catalog(_cat_min(False))) == []
    assert find_catalog_foods("merluza", recipe_contract.index_for_catalog(_cat_min(True))) == []


def test_g_knob_documentado():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_RECIPE_CONTRACT_INDEX_BY_CONTENT` | `True` | [P1-PLAN-LOTE-856]" in doc
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-856-INDEX-BY-CONTENT" in src
