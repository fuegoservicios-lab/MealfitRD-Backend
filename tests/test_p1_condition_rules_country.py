"""[P1-CONDITION-RULES-COUNTRY · 2026-08-21] La capa clínica no contenía la palabra `country`.

`grep -c country condition_rules.py` daba **0**. Ese módulo hace dos cosas para el usuario más
frágil del sistema —el que declara una condición médica— y las dos eran país-ciegas:

  1. SUSTITUCIÓN QUIRÚRGICA PROACTIVA (P0-ALLERGEN-SUBS). A un celíaco le reescribe «pan
     integral», «pan de agua», «tostada» y «galletas de soda» a **«Casabe»** — un cracker de
     yuca dominicano que no se vende en España. Su desayuno queda literalmente incomprable, y es
     precisamente el usuario para el que la sustitución es load-bearing: sin ella, el plan cae al
     path crítico→fallback.

  2. EL EJEMPLO QUE DIRIGE EL DÍA. La regla `bariatric` inyecta un «📋 EJEMPLO DE UN DÍA
     BARIÁTRICO CORRECTO … GENERA ASÍ» escrito entero en Casabe, Auyama y Yogurt griego. Medido:
     `build_condition_prompt({'country':'ES', …}) == build_condition_prompt({'country':'DO', …})`
     → **True**, 6118 chars byte-idénticos.

ENCUADRE HONESTO: AQUÍ NO HAY RIESGO CLÍNICO. La sustitución sigue siendo gluten-free y segura —
el casabe no tiene gluten. Lo que falla es la **disponibilidad y la reconocibilidad local**. Se
clasifica P1 por el bloque bariátrico, que no sugiere sino que DIRIGE el día entero con la orden
«GENERA ASÍ».

LA DECISIÓN DE DISEÑO: HUECO HONESTO ANTES QUE ALIMENTO INSERVIBLE. Para un país beta no existe
un «Casabe local» que el catálogo garantice. La tentación es mapear casabe→«tortitas de arroz»,
pero eso sería inventar disponibilidad que nadie midió. Se aplica el criterio que el propio módulo
ya tiene escrito para lácteos/huevo/maní (la «DECISIÓN HONESTA» de su comentario): cuando no hay
target que resuelva, **no se sustituye** y el residual se declara — el plan cae al path
crítico→fallback, que es la conducta correcta y ya existe, en vez de entregar un desayuno que el
usuario no puede comprar.

Excepción: los targets que YA son panhispánicos (Arroz blanco, Harina de maíz precocida, Pechuga
de pollo, Quinoa) se conservan para beta. El corte es por ALIMENTO, no por regla.

Cubre:
  A. Byte-identidad dominicana y con el knob apagado.
  B. Los targets criollos no viajan a un país beta.
  C. Los targets panhispánicos SÍ sobreviven (no se tira la regla entera).
  D. El bloque bariátrico deja de dictar un día dominicano.
  E. Las firmas no cambian: el país sale de `form_data`, que ya llegaba.
  F. Parser-based.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CR_PATH = _BACKEND_ROOT / "condition_rules.py"


@pytest.fixture(scope="module")
def cr():
    import condition_rules as _cr
    return _cr


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _targets(cr, country, alergia="gluten"):
    subs = cr.collect_allergen_substitutions({"country": country, "allergies": [alergia]})
    return [s.get("replacement") for s in subs]


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

def test_el_dominicano_conserva_sus_sustituciones(cr, knob_on):
    """Control primero: en RD el casabe es la respuesta correcta y no se mueve."""
    assert "Casabe" in _targets(cr, "DO")


def test_con_el_knob_apagado_el_beta_cae_a_dominicano(cr, monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert _targets(cr, "ES") == _targets(cr, "DO")


def test_el_prompt_dominicano_no_cambia(cr, knob_on):
    fd = {"country": "DO", "medicalConditions": ["Cirugía bariátrica"]}
    assert "Casabe" in cr.build_condition_prompt(fd)


# ── B. Los targets criollos no viajan ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_el_celiaco_beta_no_recibe_casabe(cr, knob_on, cc):
    """RED pre-fix: los 5 recibían «Casabe». Para un español es un alimento que no puede comprar,
    y la sustitución existe justamente para que su desayuno SÍ sea comprable."""
    assert "Casabe" not in _targets(cr, cc), f"{cc}: sigue recibiendo casabe"


def test_la_embarazada_beta_no_recibe_tayota(cr, knob_on):
    """Mismo mecanismo en la escalera uterotónica: cundeamor → «Tayota»."""
    src = _CR_PATH.read_text(encoding="utf-8", errors="replace")
    assert "Tayota" in src, "la escalera de embarazo cambió: revisa este guard"
    fd = {"country": "ES", "medicalConditions": ["Embarazo"]}
    subs = cr.collect_allergen_substitutions(fd) or []
    assert "Tayota" not in [s.get("replacement") for s in subs]


# ── C. Los panhispánicos sobreviven ─────────────────────────────────────────────────────────────

def test_los_targets_panhispanicos_siguen_sustituyendo_en_beta(cr, knob_on):
    """El error opuesto sería tirar la regla entera: «pasta de trigo → Arroz blanco» y «harina de
    trigo → Harina de maíz precocida» son correctos en los 6 países, y quitarlos dejaría al
    celíaco español SIN ninguna sustitución — peor que el problema que se arregla."""
    t = _targets(cr, "ES")
    assert t, "el celíaco español se quedó sin ninguna sustitución"
    assert "Arroz blanco" in t


def test_el_hueco_es_honesto_no_inventado(cr, knob_on):
    """No se mapea casabe→«tortitas de arroz»: eso sería inventar disponibilidad que nadie midió.
    Se omite la sustitución y el plan cae al path crítico→fallback, que ya existe y es correcto —
    el mismo criterio que el módulo ya aplica a lácteos/huevo/maní."""
    t = _targets(cr, "ES")
    for inventado in ("Tortitas de arroz", "Pan sin gluten", "Galletas de arroz"):
        assert inventado not in t, f"se inventó un target beta ('{inventado}') sin medir su catálogo"


# ── D. El bloque bariátrico ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("criollo", ["Casabe", "Auyama"])
def test_el_dia_bariatrico_beta_no_se_dicta_en_criollo(cr, knob_on, criollo):
    """No SUGIERE: DIRIGE, y termina con «GENERA ASÍ». Es el bloque de mayor autoridad del prompt
    clínico."""
    fd = {"country": "ES", "medicalConditions": ["Cirugía bariátrica"]}
    out = cr.build_condition_prompt(fd)
    assert criollo not in out, f"el día bariátrico español sigue dictando «{criollo}»"


def test_el_bloque_bariatrico_beta_sigue_siendo_util(cr, knob_on):
    """Vaciarlo sería el otro error: la FORMA que enseña (proteína primero, porciones pequeñas en
    gramos, sin azúcar, sin bebida junto al sólido) es clínica y universal — lo que sobra son los
    nombres de los alimentos."""
    fd = {"country": "ES", "medicalConditions": ["Cirugía bariátrica"]}
    out = cr.build_condition_prompt(fd)
    assert "GENERA ASÍ" in out
    assert "proteína" in out.lower()
    assert len(out) > 3000, "el bloque clínico bariátrico se quedó sin contenido"


# ── E/F. Firmas y anclas ────────────────────────────────────────────────────────────────────────

def test_las_firmas_no_cambian(cr):
    """Las dos puertas YA reciben `form_data`, así que el país sale de dentro: cero cambios de
    firma y cero cambios en los call sites (`prompts/plan_generator.py`, `agent.py`, `tools.py`)."""
    import inspect
    assert list(inspect.signature(cr.build_condition_prompt).parameters) == ["form_data"]
    assert list(inspect.signature(cr.collect_allergen_substitutions).parameters)[0] == "form_data"


def test_el_fuente_declara_el_marker_y_la_puerta_unica():
    src = _CR_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P1-CONDITION-RULES-COUNTRY" in src
    assert "country_for_form_data" in src, (
        "la capa clínica no deriva el país por la única puerta del motor"
    )
