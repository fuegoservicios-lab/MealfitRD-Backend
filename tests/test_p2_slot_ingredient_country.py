"""[P2-SLOT-INGREDIENT-COUNTRY · 2026-08-21] Al español se le rechazaba el desayuno «por no ser
dominicano».

`slot_ingredient_violations` caza la regla más dura del dueño —arroz en el desayuno— mirando los
INGREDIENTES, porque el detector por nombre es name-only a propósito y «Bowl energético criollo» con
150 g de arroz se le escapaba.

Fase 1 ya hizo país-aware la parte que decide: el flag `hard` se overridea a `False` fuera de
República Dominicana, así que en beta el issue no fuerza retry. **Lo que quedó dominicano es el
TEXTO**, y el texto es lo que viaja:

    «COMIDA FUERA DE HORARIO (rechazo de coherencia cultural **es-DO**): … arroz/**locrio** no
     corresponde al **desayuno dominicano** aunque el nombre no lo diga.»

Ese string entra en los `issues` del reviewer, que se le muestran al usuario **verbatim** — el
mismo camino por el que `P1-JUDGE-SEVERITY-COUNTRY` tuvo que re-anclar la prosa del juez. Un
español lee que su desayuno se rechaza por no parecerse a uno dominicano, y de paso se le cita
«locrio», un plato que no existe en su cocina.

LA REGLA EN SÍ NO SE TOCA, y conviene decir por qué: arroz para desayunar tampoco es español ni
mexicano, así que el detector sigue midiendo lo mismo en los seis países. Lo que cambia es cómo se
CUENTA — y el flag `hard`, que ya estaba resuelto, sigue igual.

El segundo consumidor (`slot_coherence_backstop_for_meal`) ya estaba cubierto: sólo añade su línea
`… (coherencia de horario es-DO)` cuando el país es DO, así que en beta no emite nada. Este P-fix
no lo toca; se ancla para que siga así.

Cubre:
  A. Byte-identidad dominicana del texto.
  B. El texto beta no invoca lo dominicano ni cita platos criollos.
  C. La regla sigue midiendo (no se vacía el detector).
  D. Lo que ya estaba bien: el flag `hard` y el segundo consumidor.
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


def _plan_con_arroz_en_desayuno():
    return [{
        "day": 1,
        "meals": [{
            "meal": "Desayuno",
            "name": "Bowl energético",
            "ingredients": ["150 g de Arroz blanco", "2 Huevos"],
            "preparation_steps": [],
        }],
    }]


def _issues(go, country):
    return go._detect_slot_appropriateness(_plan_con_arroz_en_desayuno(),
                                           {"country": country}) or []


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_el_texto_dominicano_no_cambia(go, knob_on):
    issues = _issues(go, "DO")
    assert issues, "el detector dejó de cazar arroz en el desayuno para RD"
    t = " ".join(str(i.get("text") or "") for i in issues)
    assert "es-DO" in t and "desayuno dominicano" in t


# ── B. El texto beta ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_el_texto_beta_no_invoca_lo_dominicano(go, knob_on, cc):
    """Los `issues` del reviewer se le enseñan al usuario VERBATIM — el mismo camino por el que
    P1-JUDGE-SEVERITY-COUNTRY tuvo que re-anclar la prosa del juez."""
    for i in _issues(go, cc):
        t = str(i.get("text") or "")
        assert not re.search(r"dominican|es-DO", t, re.I), (
            f"{cc}: el rechazo del desayuno sigue apelando a lo dominicano: {t[:160]}"
        )


@pytest.mark.parametrize("cc", ["ES", "MX", "CO"])
def test_el_texto_beta_no_cita_platos_criollos(go, knob_on, cc):
    """«locrio» es un plato dominicano. Citárselo a un español no le dice nada — y sugiere que el
    sistema cree que cocina otra cosa."""
    for i in _issues(go, cc):
        t = str(i.get("text") or "").lower()
        for criollo in ("locrio", "moro", "mangu", "mangú"):
            assert criollo not in t, f"{cc}: el texto cita «{criollo}»"


# ── C. La regla sigue midiendo ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["DO", "ES", "MX", "CO", "PR", "US"])
def test_el_detector_sigue_cazando_el_arroz_en_los_seis(go, knob_on, cc):
    """Arroz para desayunar tampoco es español ni mexicano: la regla mide lo mismo en los seis
    países. Vaciarla en beta sería cambiar un texto mal escrito por una regla ausente."""
    issues = _issues(go, cc)
    assert issues, f"{cc}: el detector dejó de cazar arroz en los ingredientes del desayuno"
    assert any("arroz" in str(i.get("text") or "").lower() for i in issues)


def test_el_texto_beta_sigue_diciendo_QUE_HACER(go, knob_on):
    """Un rechazo sin salida es peor que uno mal redactado. La sugerencia de arreglo (`_fix`) tiene
    que sobrevivir a la neutralización."""
    for i in _issues(go, "ES"):
        t = str(i.get("text") or "")
        assert "Cámbialo" in t or "cambia" in t.lower(), f"el texto beta perdió la salida: {t[:160]}"


# ── D. Lo que ya estaba bien ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc,esperado", [("DO", True), ("ES", False), ("MX", False)])
def test_el_flag_hard_sigue_siendo_pais_aware(go, knob_on, cc, esperado):
    """Fase 1 ya lo resolvió: en beta el issue no fuerza retry. Se ancla para que la
    neutralización del texto no lo rompa de rebote."""
    issues = _issues(go, cc)
    assert issues
    assert any(bool(i.get("hard")) is esperado for i in issues)


def test_el_segundo_consumidor_sigue_mudo_en_beta(go, knob_on):
    """`slot_coherence_backstop_for_meal` ya sólo emite su línea `(coherencia de horario es-DO)`
    cuando el país es DO. Este P-fix no lo toca; se ancla para que siga así."""
    # La firma es `(meal, meal_type, country)` — el slot va SUELTO, no dentro del dict, y el país
    # es el tercer argumento, no un form_data. Mi primera versión le pasaba `{"country": ...}` como
    # meal_type y el backstop devolvía vacío para los dos, así que el test habría «pasado» en beta
    # por la razón equivocada: no porque callara, sino porque nunca llegó a mirar.
    meal = {"name": "Bowl energético", "ingredients": ["150 g de Arroz blanco"],
            "preparation_steps": []}
    do = go.slot_coherence_backstop_for_meal(meal, "Desayuno", "DO") or []
    es = go.slot_coherence_backstop_for_meal(meal, "Desayuno", "ES") or []
    assert any("es-DO" in str(x) for x in do), "el backstop dejó de emitir para RD"
    assert not any("es-DO" in str(x) for x in es), "el backstop emite prosa es-DO en beta"
