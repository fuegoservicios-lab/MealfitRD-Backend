"""[P1-PROTEIN-CLOSER-COUNTRY · 2026-08-21] El closer del piso de proteína elegía de un pool
dominicano hardcodeado, con el pool del país a un kwarg de distancia.

`_safe_high_density_proteins` es el builder de candidatos del closer: la lista de proteínas de
alta densidad, allergen-safe y diet-safe, entre las que el motor elige cuando un plato no llega al
piso de proteína del slot. Iteraba sobre `DOMINICAN_PROTEINS` sin ningún parámetro de país,
mientras `COUNTRY_POOLS[cc]['proteins']` —23 nombres para ES, 20 para MX, curados en Fase 2— ya
existía y sólo lo leía el camino degradado.

Consecuencia: a un español al que le falta proteína en la cena se le SIEMBRA un ingrediente real,
en gramos, elegido de una lista donde no están el jamón serrano, las gambas, los boquerones ni el
cordero. Y es el closer, no el LLM: esto ocurre DESPUÉS de la generación, así que ningún prompt lo
evita.

LA UNIÓN, NO LA SUSTITUCIÓN. `DOMINICAN_PROTEINS` son 49 nombres y la mayoría son universales
—Pollo, Cerdo, Res, Pescado, Atún, Huevos, lentejas, garbanzos, quesos—: reemplazarlo dejaría a un
español sin pollo, que es peor que el problema que arregla. El pool del país se antepone (gana los
empates de magrez) y el dominicano se conserva **menos los nombres con gentilicio**, que hoy es
uno solo: 'Salami Dominicano'. El filtro es una comprobación sobre el NOMBRE, no una tabla nueva
de exclusiones — la clase de tabla que P1-DIET-CANON-SSOT prohibió, y que además yo habría escrito
mal: la primera versión de este test exigía excluir también 'Longaniza', y el código correcto lo
refutó recordándome que la longaniza es un embutido español real.

Cubre:
  A. Byte-identidad dominicana y con el knob apagado.
  B. El país beta recibe sus proteínas propias.
  C. La unión: las universales sobreviven; las de gentilicio dominicano, no.
  D. Los filtros clínicos (alergia y dieta) siguen corriendo sobre el pool ampliado.
  E. El closer que consume los candidatos también sabe de país.
  F. Parser-based.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


class _FakeInfo:
    def __init__(self, name):
        self.name = name
        self.protein = 25.0
        self.kcal = 120.0


class _FakeDB:
    """Todo nombre resuelve con la misma densidad: así el test mide QUÉ nombres entran al pool,
    no cómo se ordenan por magrez — que es otra propiedad y tiene sus propios tests."""
    def lookup(self, name):
        return _FakeInfo(name)


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _pool(go, country=None, allergies=None, diet=None):
    kwargs = {"diet": diet}
    if country is not None:
        kwargs["country"] = country
    out = go._safe_high_density_proteins(allergies, _FakeDB(), min_protein=18.0, **kwargs)
    return [n for (_ln, n, _i) in out]


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_do_es_identico_a_no_declarar_pais(go, knob_on):
    assert _pool(go, "DO") == _pool(go)


def test_el_pais_beta_cae_a_dominicano_con_el_knob_apagado(go, monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert _pool(go, "ES") == _pool(go, "DO")


# ── B. El país beta recibe sus proteínas ────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc,propia", [
    ("ES", "Jamón serrano"), ("ES", "Gambas"), ("ES", "Boquerones"),
    ("MX", "Cecina"), ("MX", "Chorizo mexicano"),
    ("CO", "Sobrebarriga"), ("PR", "Jamonilla"), ("US", "Pavo ahumado"),
])
def test_el_pais_beta_recibe_sus_proteinas(go, knob_on, cc, propia):
    """RED pre-fix: ninguna. Se saltan las que su propio pool no declare — el test comprueba lo
    que el pool de Fase 2 dice tener, no lo que yo suponga que debería tener."""
    from constants import COUNTRY_POOLS
    if propia not in (COUNTRY_POOLS.get(cc, {}).get("proteins") or []):
        pytest.skip(f"'{propia}' no está en el pool {cc} de Fase 2 — este test no lo inventa")
    assert propia in _pool(go, cc)


def test_el_pool_beta_es_distinto_del_dominicano(go, knob_on):
    assert _pool(go, "ES") != _pool(go, "DO")


# ── C. La unión ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("universal", ["Pollo", "Huevos", "Lentejas", "Garbanzos", "Atún"])
def test_las_proteinas_universales_sobreviven_en_beta(go, knob_on, universal):
    """El error opuesto sería sustituir el pool en vez de ampliarlo: un español sin pollo, sin
    huevos y sin lentejas está peor que antes. La mayoría de `DOMINICAN_PROTEINS` es universal."""
    assert universal in _pool(go, "ES")


@pytest.mark.parametrize("gentilicio", ["Salami Dominicano"])
def test_las_proteinas_con_gentilicio_dominicano_no_viajan_a_beta(go, knob_on, gentilicio):
    """El único nombre del pool dominicano que un español no puede comprar TAL CUAL. El filtro
    mira el NOMBRE (lleva gentilicio), no una tabla de exclusiones que mantener.

    La primera versión de este test también exigía excluir 'Longaniza', y el código correcto lo
    refutó: la longaniza es un embutido ESPAÑOL real, no un dominicanismo. El filtro por gentilicio
    acierta justo por no ser una lista a mano — una lista la habría escrito yo con ese error
    dentro."""
    from constants import DOMINICAN_PROTEINS
    if gentilicio not in DOMINICAN_PROTEINS:
        pytest.skip(f"'{gentilicio}' ya no está en DOMINICAN_PROTEINS")
    assert gentilicio in _pool(go, "DO"), "el pool dominicano debe conservarlos"
    assert gentilicio not in _pool(go, "ES")


# ── D. Los filtros clínicos siguen corriendo sobre el pool ampliado ─────────────────────────────

def test_la_alergia_filtra_tambien_las_proteinas_del_pais(go, knob_on):
    """Lo más importante después de que funcione: las proteínas nuevas entran POR el mismo camino,
    así que el filtro de alérgenos las ve. Un español alérgico a mariscos no puede recibir gambas
    sembradas por el closer."""
    p = _pool(go, "ES", allergies=["mariscos"])
    assert "Gambas" not in p and "Almejas" not in p
    assert "Camarones" not in p, "el filtro dejó de correr sobre las dominicanas"
    assert "Pollo" in p


def test_la_dieta_filtra_tambien_las_proteinas_del_pais(go, knob_on):
    """Espejo para el eje de dieta: un vegetariano español no puede recibir jamón serrano."""
    p = _pool(go, "ES", diet="vegetariana")
    assert "Jamón serrano" not in p and "Chorizo español" not in p
    assert "Pollo" not in p, "el filtro de dieta dejó de correr sobre las dominicanas"


# ── E. El closer que los consume ────────────────────────────────────────────────────────────────

def test_el_closer_acepta_pais_y_lo_usa_en_su_rebuild_interno(go):
    """`_close_protein_gap_for_meal` reconstruye candidatos por dentro para el caso de lácteo
    dulce. Sin país ahí, ese sub-caso volvería al pool dominicano aunque el caller ya lo hubiera
    resuelto — el hueco por el que se cuelan los arreglos a medias."""
    import inspect
    assert "country" in inspect.signature(go._close_protein_gap_for_meal).parameters


# ── F. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_fuente_declara_el_marker_y_reusa_los_ssot():
    src = _GO_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P1-PROTEIN-CLOSER-COUNTRY" in src
    i = src.find("def _country_protein_pool")
    assert i > 0, "el helper del pool por país desapareció"
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    assert "COUNTRY_POOLS" in cuerpo, "no reusa el pool por país que Fase 2 ya construyó"
    assert "country_for_form_data" in cuerpo, "no deriva el país por la única puerta"
    # Y el builder tiene que CONSUMIRLO, no volver a iterar el pool dominicano a mano.
    j = src.find("def _safe_high_density_proteins")
    _fin2 = src.find("\ndef ", j + 1)
    builder = src[j:_fin2 if _fin2 > 0 else len(src)]
    assert "_country_protein_pool(country)" in builder
    assert "for name in DOMINICAN_PROTEINS:" not in builder


def test_la_puerta_unica_es_resoluble_en_el_ambito_del_modulo(go):
    """El fallo que casi se me va vivo: los call sites del closer llaman a `country_for_form_data`
    por su nombre desnudo, pero en `graph_orchestrator` ese símbolo existía SÓLO como import local
    dentro de `_build_shared_context`. En runtime habrían lanzado `NameError` — y la suite no lo
    vio porque no ejercita esa rama del piso de proteína.

    Un test de comportamiento no puede cubrir esto sin montar el pipeline entero; una comprobación
    de ÁMBITO sí, y cuesta una línea. Mismo espíritu que los guards de «feature inerte» del repo:
    lo que importa no es que la función exista, es que el sitio que la llama pueda verla."""
    assert hasattr(go, "country_for_form_data"), (
        "`country_for_form_data` volvió a ser sólo un import local: los call sites que la usan "
        "por nombre desnudo lanzarán NameError en runtime"
    )


def test_los_tres_call_sites_externos_pasan_el_pais():
    """El builder con parámetro y los callers sin pasarlo sería la forma clásica de feature
    inerte que este repo ya pagó dos veces."""
    src = _GO_PATH.read_text(encoding="utf-8", errors="replace")
    llamadas = [i for i in range(len(src))
                if src.startswith("_safe_high_density_proteins(", i)]
    assert len(llamadas) >= 4, "cambiaron los call sites: revisa el barrido"
    sin_pais = [i for i in llamadas if "country" not in src[i:i + 400]]
    assert not sin_pais, (
        f"{len(sin_pais)} call site(s) de _safe_high_density_proteins siguen sin pasar país"
    )
