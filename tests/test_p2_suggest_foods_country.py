"""[P2-SUGGEST-FOODS-COUNTRY · 2026-08-21] «¿Qué como para más hierro?» — y le salía huitlacoche.

`suggest_foods_for_nutrient` es la tool que el coach usa cuando el usuario pregunta cómo subir un
micronutriente. Rankea sobre `get_master_ingredients()` **entero**: las 347 filas, con las de los
seis países mezcladas. Filtra por alergias, rechazos y dieta — nunca por país.

Resultado para un español: de las seis sugerencias, unas son dominicanas (yautía, casabe), otras
mexicanas (huitlacoche, nopal), otras colombianas (chontaduro) y otras estadounidenses (pretzels).
El consejo es correcto en lo nutricional y **incomprable** en lo práctico, que para una tool cuyo
único trabajo es decir «cómete esto» equivale a no funcionar.

MISMO PREDICADO QUE EL CATÁLOGO DEL GENERADOR, a propósito. La regla ya existe y ya está decidida:
sobrevive lo que tiene precio RD (que incluye los básicos universales — arroz, pollo, huevo) más lo
que reclama el keep sin precio DE SU PAÍS. Escribir aquí un segundo criterio sería el espejo que
driftea, que es la forma precisa del defecto que costó la costura (a) del guard de coherencia y las
tres tablas de dieta de `P1-DIET-CANON-SSOT`.

Hereda también la limitación del generador, y se dice: a un español se le siguen pudiendo sugerir
alimentos dominicanos con precio (casabe, yautía) porque «tiene precio RD» es la única señal de
comprabilidad que existe hoy. Acotar eso es curación de datos, no plomería — el mismo pendiente que
`P1-BETA-FRAGMENT-DEPTH`.

Cubre:
  A. Las sugerencias respetan el país del usuario.
  B. Byte-identidad dominicana.
  C. No se rompe lo que ya funcionaba (alergias, dieta, techo/piso).
  D. Fail-open: sin país o sin catálogo, la conducta de siempre.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def tools():
    import tools as _t
    return _t


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


@pytest.fixture
def perfil(monkeypatch, tools):
    """Perfil sintético: el país se inyecta por caso."""
    def _hacer(country=None, **hp_extra):
        hp = {"allergies": [], "dislikes": [], "dietType": "balanced"}
        if country is not None:
            hp["country"] = country
        hp.update(hp_extra)
        monkeypatch.setattr(tools, "get_user_profile", lambda uid: {"health_profile": hp})
        return hp
    return _hacer


@pytest.fixture(scope="module")
def hay_catalogo(tools):
    from shopping_calculator import get_master_ingredients
    if not (get_master_ingredients() or []):
        pytest.skip("catálogo no disponible (sin DB)")
    return True


def _sugeridos(tools, nutriente, top_n=12):
    """Los nombres que la tool devuelve, extraídos de su propio formato («- Nombre: 12g …»)."""
    r = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": nutriente, "top_n": top_n}))
    return [ln[2:].split(":", 1)[0].strip() for ln in r.splitlines() if ln.startswith("- ")]


# ── A. Las sugerencias respetan el país ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
@pytest.mark.parametrize("nutriente", ["hierro", "fibra", "calcio", "potasio"])
def test_todo_lo_sugerido_es_comprable_en_el_pais_del_usuario(tools, hay_catalogo, knob_on, perfil,
                                                              cc, nutriente):
    """Invariante ESTRUCTURAL, y esa forma es deliberada: la primera versión de este test llevaba
    una lista de alimentos «claramente ajenos» (huitlacoche, nopal, chontaduro, pretzels) y pasaba
    en verde **sin que el código filtrara nada** — esos alimentos simplemente no rankean en el
    top-N de esos nutrientes. Elegí las sondas por plausibilidad en vez de por medición, que es
    justo el error que este repo tiene registrado varias veces: un guard que no puede fallar es una
    coartada.

    Lo que de verdad salía para un español: Chile de árbol, Chile mulato, Chile guajillo, Chile
    ancho, Chile pasilla, Achiote, Flor de Jamaica, Queso de papa. Ninguno estaba en mi lista.

    Así que ahora no se enumera nada: se comprueba CADA sugerencia contra el mismo predicado de
    comprabilidad que usa el catálogo del generador. No puede quedarse vacuo."""
    import shopping_calculator as sc
    perfil(country=cc)
    filas = {r["name"]: r for r in (sc.get_master_ingredients() or [])}

    def _comprable(nombre):
        r = filas.get(nombre)
        if r is None:
            return True  # no resuelve a fila: no es este gap
        if (r.get("price_per_lb") or 0) > 0 or (r.get("price_per_unit") or 0) > 0:
            return True
        return sc.is_country_catalog_unpriced_item(nombre, country=cc)

    ajenos = [n for n in _sugeridos(tools, nutriente) if not _comprable(n)]
    assert not ajenos, f"{cc}/{nutriente}: se sugieren alimentos de otro país: {ajenos}"


def test_el_caso_medido_que_abrio_el_gap(tools, hay_catalogo, knob_on, perfil):
    """El RED concreto, anclado con los nombres que la medición devolvió — no con los que yo
    imaginé. Si vuelven, este test lo dice por su nombre."""
    from constants import strip_accents
    perfil(country="ES")
    vistos = {strip_accents(n.lower()) for n in _sugeridos(tools, "potasio")}
    for mexicano in ("chile mulato", "chile guajillo", "chile ancho", "chile pasilla", "achiote"):
        assert strip_accents(mexicano) not in vistos, (
            f"a un español se le vuelve a sugerir {mexicano!r} para el potasio"
        )


def test_cada_pais_sigue_viendo_lo_suyo(tools, hay_catalogo, knob_on, perfil):
    """El error opuesto —y peor— sería estrechar hasta dejar la tool sin nada que decir."""
    for cc in ("ES", "MX", "CO", "PR", "US", "DO"):
        perfil(country=cc)
        assert _sugeridos(tools, "fibra", 6), f"{cc}: la tool se quedó sin sugerencias"


# ── B. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_el_usuario_dominicano_ve_lo_mismo_que_antes(tools, hay_catalogo, knob_on, perfil):
    """RD conserva el catálogo completo con precio, que es el que siempre tuvo."""
    perfil(country="DO")
    con_pais = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "hierro", "top_n": 6}))
    perfil(country=None)
    sin_pais = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "hierro", "top_n": 6}))
    assert con_pais == sin_pais


def test_con_el_knob_apagado_no_se_filtra(tools, hay_catalogo, monkeypatch, perfil):
    """Contrato de rollback: apagado ⇒ conducta pre-sistema-de-países."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    perfil(country="ES")
    es = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "hierro", "top_n": 12}))
    perfil(country="DO")
    do = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "hierro", "top_n": 12}))
    assert es == do, "con el knob apagado el país no debe cambiar nada"


# ── C. Lo que ya funcionaba ─────────────────────────────────────────────────────────────────────

def test_las_alergias_siguen_filtrando(tools, hay_catalogo, knob_on, perfil):
    """P0-CHAT-ALLERGY-SSOT: el filtro clínico es lo único que esta tool NO puede perder."""
    perfil(country="ES", allergies=["Lácteos"])
    from constants import strip_accents
    r = strip_accents(str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "calcio", "top_n": 12})).lower())
    for lacteo in ("leche", "queso", "yogurt"):
        assert lacteo not in r, f"se sugiere {lacteo!r} a alguien con alergia a lácteos"


def test_el_techo_sigue_devolviendo_los_mas_bajos(tools, hay_catalogo, knob_on, perfil):
    """Sodio es un TECHO: la tool debe dar los más BAJOS, no los más ricos."""
    perfil(country="ES")
    r = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "sodio", "top_n": 6}))
    assert "más bajos en" in r


# ── D. Fail-open ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("basura", [None, "", "ZZ", "basura"])
def test_un_pais_ilegible_no_deja_al_usuario_sin_consejo(tools, hay_catalogo, knob_on, perfil,
                                                         basura):
    perfil(country=basura)
    r = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "hierro", "top_n": 6}))
    assert "por cada 100g" in r


def test_si_el_perfil_revienta_la_tool_sigue_respondiendo(tools, hay_catalogo, knob_on, monkeypatch):
    def _boom(uid):
        raise RuntimeError("DB caída")
    monkeypatch.setattr(tools, "get_user_profile", _boom)
    r = str(tools.suggest_foods_for_nutrient.invoke(
        {"user_id": "u1", "nutrient": "hierro", "top_n": 6}))
    assert "por cada 100g" in r
