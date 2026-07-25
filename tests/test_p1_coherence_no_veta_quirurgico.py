"""[P1-COHERENCE-NO-VETA-QUIRURGICO · 2026-07-25] 150 s de regeneración para cuadrar un queso.

Forense de la corrida `corr=1156a13c` (plan 8fa68e8b, 8 min 20 s, 3 intentos). El intento 2 se
rechazó por DOS razones:

    ❌ FRUTA REPETIDA EL MISMO DÍA                      ← day-attributable, whitelisted
    ❌ COHERENCIA RECETAS LISTA: 1 divergencia crítica  ← no whitelisted
       (foods: Queso fresco). action=reject_minor

`_surgical_reject_targets` exige que **todas** las razones sean reparables por día; una sola que
no lo sea devuelve `None` y el plan entero vuelve al planificador. Resultado: una regeneración
completa (~150 s) para arreglar una cantidad de queso.

## Por qué la coherencia no necesita vetar

El path quirúrgico **re-ensambla** (medido en esta misma corrida: 25 s frente a 77 s de la pasada
completa), y ese re-ensamblado reconstruye la lista de compras y vuelve a correr el guard. La
reparación determinista que la divergencia necesita ya viene incluida. Si sobrevive al
re-ensamblado, el siguiente rechazo enruta normal y no se ha perdido nada.

Se mantiene conservador en las dos direcciones que importan, y ambas están ancladas abajo:
  · cualquier razón que no sea whitelisted **ni** de coherencia → retry completo (sin cambios);
  · **sólo** coherencia, sin ninguna razón atribuible a un día → retry completo, porque el
    quirúrgico no tendría ningún día que reparar.
"""
import pytest

import graph_orchestrator as go


_FRUTA = ("FRUTA REPETIDA EL MISMO DÍA (rechazo de variedad): una misma fruta dulce aparece "
          "en 2+ comidas del mismo día.")
_COH = "COHERENCIA RECETAS LISTA: 1 divergencia(s) críticas (foods: Queso fresco). action=reject_minor."
_BANDA = "BANDA DE MACROS fuera de rango en 3 días (proteína 0.62)."


@pytest.fixture(autouse=True)
def _detector_dice_que_el_dia_2_es_el_culpable(monkeypatch):
    """Aísla la REGLA DE ENRUTADO del detector de variedad.

    `_surgical_reject_targets` sólo devuelve días si los detectores deterministas nombran un
    subconjunto ESTRICTO de culpables. Lo que se prueba aquí es qué razones habilitan o vetan la
    ruta, no si el detector sabe ver fruta repetida (eso lo cubre la suite de variedad). Sin este
    aislamiento el test mediría dos cosas y fallaría por la que no vigila — que es justo lo que
    pasó al escribirlo con un plan sintético sin fruta repetida.
    """
    monkeypatch.setattr(go, "build_variety_report",
                        lambda *_a, **_k: {"issues": ["Día 2: fruta repetida el mismo día"]})
    monkeypatch.setattr(go, "_detect_slot_appropriateness", lambda *_a, **_k: [])
    monkeypatch.setattr(go, "_detect_slot_incoherence", lambda *_a, **_k: [])


def _state(reasons, n_days=3):
    days = [{"day": i + 1, "meals": [{"name": f"Comida {i}", "ingredients": ["100 g de pollo"]}]}
            for i in range(n_days)]
    return {"rejection_reasons": list(reasons), "plan_result": {"days": days}}


# ───────────── 1. el caso medido ─────────────

def test_coherencia_junto_a_fruta_NO_manda_al_planificador():
    """El caso exacto del intento 2 de corr=1156a13c."""
    assert go._surgical_reject_targets(_state([_FRUTA, _COH])) is not None


def test_solo_fruta_sigue_yendo_al_quirurgico():
    """Comportamiento previo intacto."""
    assert go._surgical_reject_targets(_state([_FRUTA])) is not None


# ───────────── 2. lo que NO puede aflojarse ─────────────

def test_solo_coherencia_va_al_retry_COMPLETO():
    """Sin ninguna razón atribuible a un día no hay nada que reparar quirúrgicamente: habilitar la
    ruta aquí produciría una 'reparación' de cero días y quemaría el intento igual."""
    assert go._surgical_reject_targets(_state([_COH])) is None


def test_una_razon_ajena_sigue_vetando():
    """La banda de macros es cross-día: el retry completo es la herramienta correcta."""
    assert go._surgical_reject_targets(_state([_FRUTA, _BANDA])) is None
    assert go._surgical_reject_targets(_state([_FRUTA, _COH, _BANDA])) is None


def test_prosa_del_reviewer_sigue_vetando():
    assert go._surgical_reject_targets(
        _state([_FRUTA, "El plan no incluye NINGUNA preparación transformada"])) is None


def test_plan_de_un_solo_dia_no_entra():
    assert go._surgical_reject_targets(_state([_FRUTA, _COH], n_days=1)) is None


def test_sin_razones_no_entra():
    assert go._surgical_reject_targets(_state([])) is None


# ───────────── 3. knob y forma ─────────────

def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(go, "COHERENCE_NO_VETA_QUIRURGICO", False)
    assert go._surgical_reject_targets(_state([_FRUTA, _COH])) is None, \
        "con el knob OFF la coherencia debe volver a vetar"


def test_la_lista_de_re_juzgadas_es_estrecha():
    """Sólo entran razones que el RE-ENSAMBLADO vuelve a evaluar por sí solo. Meter aquí algo que
    el re-ensamblado no re-evalúa haría que el quirúrgico lo ignore en silencio."""
    assert go._SURGICAL_REJECT_REJUDGED_PREFIXES == ("coherencia recetas lista",)


def test_no_se_solapa_con_la_whitelist():
    for p in go._SURGICAL_REJECT_REJUDGED_PREFIXES:
        assert p not in go._SURGICAL_REJECT_SAFE_PREFIXES
