"""[P1-JUDGE-SEVERITY-COUNTRY · 2026-08-21] El juez culinario ya sabía QUIÉN era y qué platos
citar, pero seguía graduando con ojos dominicanos.

F1-T3 re-ancló la identidad del juez («Eres un juez culinario dominicano experto» → «experto en la
cocina de España») y F2-T5 sustituyó el catálogo de ejemplos por el del país. Las dos son
correctas y este P-fix no las toca. Lo que quedó es la PROSA que decide cuándo algo está mal —
medido en el render ES:

    «severidad ('minor' si es cosmético/discutible, 'high' si un DOMINICANO lo vería como un
     error claro)»
    «...guisados SON cenas DOMINICANAS legítimas y frecuentes en la vida real»
    «...la creatividad DOMINICANA legítima (fusiones, adaptaciones...)»
    «...dentro de un patrón culinario real (DOMINICANO o internacional)»
    y los ejemplos sueltos «casabe» y «sancocho» en las descripciones de slot

POR QUÉ IMPORTA MÁS QUE UN NOMBRE MAL PUESTO. Esa frase no describe: **calibra**. El juez decide
si una divergencia es `minor` o `high` según lo que un dominicano pensaría, y `high` es lo que
escala a retry — un retry que se PAGA en tokens. Una paella o unos boquerones en vinagre pueden
parecerle raros a ojos dominicanos y perfectamente normales a ojos españoles: el resultado es que
el plan correcto se rechaza, se regenera con dinero y vuelve a rechazarse.

Y hay una segunda cara: los `issues` del juez se le muestran al usuario **verbatim**. Un español
lee que su cena «no es una cena dominicana legítima».

SIN TABLA DE GENTILICIOS. Decir «un español», «un mexicano», «un colombiano» exigiría una tabla
de demónimos — la 2ª tabla que P1-DIET-CANON-SSOT prohíbe. Se reformula con el nombre del país,
que ya está en `COUNTRY_PROFILES[cc]['name_es']`: «si alguien de España lo vería como un error
claro». Menos elegante y sin tabla que mantener.

Cubre:
  A. Byte-identidad dominicana.
  B. El criterio de severidad se calibra con el país del usuario.
  C. La prosa deja de llamar «dominicano» a lo legítimo.
  D. Los ejemplos de comida usan el SSOT de neutralización, no una tabla nueva.
  E. Lo que F1/F2 ya arreglaron sigue arreglado (control de no-regresión).
  F. La rúbrica no se vació.
"""
from __future__ import annotations

import re

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


@pytest.fixture(autouse=True)
def cache_limpia(go):
    go._CULINARY_JUDGE_RUBRIC_CACHE.clear()
    yield
    go._CULINARY_JUDGE_RUBRIC_CACHE.clear()


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_la_rubrica_dominicana_no_cambia(go, knob_on):
    do = go._culinary_judge_rubric_for_country("DO")
    assert "un dominicano lo vería" in do
    assert "cenas dominicanas legítimas" in do


def test_la_byte_identidad_dominicana_no_depende_de_la_cache(go, knob_on):
    """El fixture de este fichero limpia `_CULINARY_JUDGE_RUBRIC_CACHE` entre casos, y así destapó
    una fragilidad real: la identidad dominicana descansaba ENTERAMENTE en que el dict naciera
    pre-sembrado con `{"DO": _CULINARY_JUDGE_RUBRIC}`. Vaciar la caché —un test, un reload, un
    refactor— hacía que 'DO' cayera por la rama beta y recibiera la rúbrica re-anclada, EN
    SILENCIO. Una garantía que descansa en un estado mutable no es una garantía.

    Ahora hay un return temprano y este test lo ancla contra el objeto exacto."""
    go._CULINARY_JUDGE_RUBRIC_CACHE.clear()
    assert go._culinary_judge_rubric_for_country("DO") is go._CULINARY_JUDGE_RUBRIC
    assert go._culinary_judge_rubric_for_country(None) is go._CULINARY_JUDGE_RUBRIC
    assert go._culinary_judge_rubric_for_country("basura") is go._CULINARY_JUDGE_RUBRIC


# ── B. La severidad se calibra con el país ──────────────────────────────────────────────────────

@pytest.mark.parametrize("cc,pais", [("ES", "España"), ("MX", "México"), ("CO", "Colombia")])
def test_la_severidad_se_calibra_con_el_pais_del_usuario(go, knob_on, cc, pais):
    """La frase no describe: CALIBRA. `high` es lo que escala a retry, y el retry se paga en
    tokens — un plan español correcto rechazado por ojos dominicanos se regenera con dinero y
    vuelve a rechazarse."""
    r = go._culinary_judge_rubric_for_country(cc)
    assert "un dominicano lo vería" not in r, f"{cc}: la severidad sigue calibrada en dominicano"
    assert pais in r


# ── C. La prosa ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_ningun_gentilicio_dominicano_sobrevive(go, knob_on, cc):
    """Los `issues` del juez se le enseñan al usuario VERBATIM: un español no puede leer que su
    cena «no es una cena dominicana legítima»."""
    r = go._culinary_judge_rubric_for_country(cc)
    assert not re.search(r"dominican", r, re.I), (
        f"{cc}: la rúbrica sigue diciendo «dominicano» en su prosa de calibración"
    )


# ── D. Los ejemplos de comida ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("token", ["casabe", "sancocho"])
def test_los_ejemplos_de_comida_pasan_por_el_ssot(go, knob_on, token):
    """Reusa `constants.neutralize_do_lexicon`, que ya sirve al planner, al prompt de variedad y a
    los ejemplos clínicos. Una cuarta tabla a mano es lo que P1-DIET-CANON-SSOT costó una vez."""
    assert token not in go._culinary_judge_rubric_for_country("ES").lower()


# ── E. No-regresión de lo que F1/F2 ya arreglaron ───────────────────────────────────────────────

def test_la_identidad_del_juez_sigue_re_anclada(go, knob_on):
    """F1-T3. Control de no-regresión: este P-fix toca la prosa de calibración, no la identidad."""
    assert "cocina de España" in go._culinary_judge_rubric_for_country("ES")


def test_el_catalogo_de_ejemplos_sigue_siendo_el_del_pais(go, knob_on):
    """F2-T5. Otro control: el bloque de platos ya se sustituye por el del país."""
    r = go._culinary_judge_rubric_for_country("ES")
    assert "PLATOS DE ESPAÑA" in r.upper() or "PLATOS DOMINICANOS" not in r.upper()


# ── F. La rúbrica no se vació ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX"])
def test_la_rubrica_beta_conserva_su_contenido(go, knob_on, cc):
    """Vaciar la calibración sería peor que calibrarla mal: el juez dejaría de distinguir `minor`
    de `high` y todo escalaría a retry, o nada lo haría."""
    do, beta = go._culinary_judge_rubric_for_country("DO"), go._culinary_judge_rubric_for_country(cc)
    assert len(beta) > len(do) * 0.85
    assert "minor" in beta and "high" in beta
