"""[P1-PLAN-TITLE-I18N · 2026-08-20] El titulo del Historial seguia en espanol.

Reportado con captura: la app en `en-US`, los platos y las macros ya en ingles, y el
titulo del plan en espanol arriba del todo.

LA CAUSA, medida en produccion antes de teorizar. `plan_data->>'name'` es NULL en los
8 planes vivos: el nombre del plan vive en la COLUMNA `meal_plans.name`, no dentro del
jsonb. Fase 1c lo leia del jsonb:

    _plan_name_raw = plan_data.get("name")      # -> siempre None

asi que `plan_name_pending` salia None, al LLM nunca se le pedia `plan_name`, y el
bloque que escribe `pd["_display"][locale]["name"]` no llegaba a ejecutarse nunca. La
funcionalidad estaba INERTE mientras los meals SI se enriquecian — de ahi lo raro del
sintoma: media pantalla traducida y el titulo no.

`plan_data["name"]` solo existe si el plan fue RENOMBRADO alguna vez (el
`PATCH /api/plans/{id}/name` escribe la columna Y el jsonb, por P1-HIST-5). O sea que
el titulo solo se habria traducido en planes renombrados.

POR QUE LOS TESTS NO LO VIERON. Los de fase 1c construian el plan con
`plan_data={"name": ...}` a mano. `_make_plan` —el helper por defecto— NO pone `name`,
o sea que la forma real de produccion estaba disponible y los casos del titulo eligieron
la otra. Un fixture que no se parece a produccion prueba el codigo contra un mundo que
no existe, y aqui el mundo inventado era justo el unico donde la funcion servia.

EL ARREGLO separa dos valores que fase 1c habia colapsado en uno:
  - el TEXTO a traducir: el jsonb si lo trae (plan renombrado), si no la columna.
  - el SNAPSHOT del guard TOCTOU: SIEMPRE el valor del jsonb, aunque sea None. El
    mutator compara `pd.get("name")` contra el, y un rename concurrente CREA ese campo
    (None != "Nuevo" -> mismatch detectado). Pasar ahi el texto de la columna romperia
    el guard al reves: None nunca igualaria al titulo y no se escribiria jamas — que es
    exactamente el bug, desplazado un paso.

tooltip-anchor: P1-PLAN-TITLE-I18N
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plan_display_i18n as pdi  # noqa: E402


# ------------------------------------------------------------------ dobles

class _FakeResponse:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata or {"input_tokens": 10, "output_tokens": 10}


class _FakeLLM:
    NEXT_RESPONSE = None
    captured_prompts: list = []

    def __init__(self, **kwargs):
        pass

    def invoke(self, prompt):
        _FakeLLM.captured_prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        return _FakeLLM.NEXT_RESPONSE


_MEAL = {
    "name": "Habichuelas guisadas",
    "description": "Guiso dominicano de habichuelas rojas.",
    "recipe": ["Sofreir el sazon en aceite caliente."],
    "ingredients": ["30 g de Habichuelas rojas"],
}

_RESPUESTA = (
    '{"meals":[{"i":0,'
    '"name":"Stewed red beans",'
    '"description":"Traditional Dominican stew with red beans.",'
    '"recipe":["Saute the seasoning in hot oil."],'
    '"ingredients":["30 g red beans (Habichuelas rojas)"]}],'
    '"plan_name":"Strong Seasoning, Balanced Life"}'
)


@pytest.fixture
def motor(monkeypatch):
    """Estado compartido + dobles. `state["plan_data"]` simula la fila jsonb y
    `state["columna_name"]` la COLUMNA `meal_plans.name` (la distincion es el P-fix)."""
    state = {"plan_data": None, "columna_name": None, "columna_consultada": 0}

    _FakeLLM.NEXT_RESPONSE = None
    _FakeLLM.captured_prompts = []

    monkeypatch.setattr(pdi, "_try_claim_enrich_lock_cross_worker",
                        lambda plan_id, locale, day_indices: True)
    monkeypatch.setattr(pdi, "_release_enrich_lock_cross_worker",
                        lambda plan_id, locale, day_indices: None)
    monkeypatch.setattr(pdi, "_circuit_breaker_can_proceed", lambda model_name: True)
    monkeypatch.setattr(pdi, "build_chat_llm", lambda model, **kwargs: _FakeLLM(**kwargs))
    monkeypatch.setattr(pdi, "_fetch_plan_data", lambda plan_id, user_id: state["plan_data"])

    def _fake_columna(plan_id, user_id):
        state["columna_consultada"] += 1
        return state["columna_name"]

    monkeypatch.setattr(pdi, "_fetch_plan_name_column", _fake_columna)

    def _fake_atomic(plan_id, mutator, user_id=None, **kwargs):
        result = mutator(state["plan_data"])
        if isinstance(result, dict):
            state["plan_data"] = result
        return state["plan_data"]

    monkeypatch.setattr(pdi, "update_plan_data_atomic", _fake_atomic)
    monkeypatch.setattr(pdi, "log_llm_usage_event", lambda **kwargs: None)
    return state


def _titulo_traducido(state, locale="en-US"):
    disp = state["plan_data"].get("_display")
    if not isinstance(disp, dict):
        return None
    entrada = disp.get(locale)
    return entrada.get("name") if isinstance(entrada, dict) else None


# ------------------------------------------------------------------ la regresion

def test_plan_con_la_forma_REAL_de_produccion_traduce_su_titulo(motor):
    """EL TEST QUE FALTABA. Sin `name` en el jsonb —los 8 planes vivos— y con la
    columna puesta: antes ni se pedia `plan_name` al LLM ni se escribia nada."""
    motor["plan_data"] = {"days": [{"meals": [dict(_MEAL)]}]}   # sin "name": produccion
    motor["columna_name"] = "Sazon Fuerte, Vida en Equilibrio"
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(_RESPUESTA)

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert _titulo_traducido(motor) == "Strong Seasoning, Balanced Life", (
        "el titulo no se tradujo: la funcionalidad vuelve a estar inerte para los "
        "planes que nunca se renombraron, que son TODOS los de produccion")


def test_el_titulo_llega_al_prompt_del_LLM(motor):
    """Aguas arriba del persist: si no se pide, no hay nada que escribir."""
    motor["plan_data"] = {"days": [{"meals": [dict(_MEAL)]}]}
    motor["columna_name"] = "Sazon Fuerte, Vida en Equilibrio"
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(_RESPUESTA)

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert _FakeLLM.captured_prompts, "no hubo llamada al LLM"
    assert "Sazon Fuerte, Vida en Equilibrio" in _FakeLLM.captured_prompts[0]


def test_un_plan_renombrado_usa_el_jsonb_y_no_consulta_la_columna(motor):
    """El camino que fase 1c si cubria sigue igual, y sin query de mas."""
    motor["plan_data"] = {"name": "Nombre Puesto a Mano", "days": [{"meals": [dict(_MEAL)]}]}
    motor["columna_name"] = "NO-DEBERIA-USARSE"
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(_RESPUESTA)

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert motor["columna_consultada"] == 0, "consulto la columna teniendo el jsonb"
    assert "Nombre Puesto a Mano" in _FakeLLM.captured_prompts[0]
    assert _titulo_traducido(motor) == "Strong Seasoning, Balanced Life"


# ------------------------------------------------------------------ el guard TOCTOU

def test_un_rename_concurrente_descarta_la_traduccion(motor):
    """La otra mitad del arreglo. El snapshot es el valor del JSONB (aqui None); si un
    rename corre durante la llamada LLM, CREA `pd["name"]` y el mutator ve la
    diferencia. Sin esta parte se pegaria la traduccion del titulo viejo sobre el
    nuevo — y con el snapshot mal elegido no se escribiria NUNCA."""
    motor["plan_data"] = {"days": [{"meals": [dict(_MEAL)]}]}
    motor["columna_name"] = "Sazon Fuerte, Vida en Equilibrio"

    class _LLMQueRenombra(_FakeLLM):
        def invoke(self, prompt):
            # Simula el rename concurrente: ocurre MIENTRAS el LLM responde.
            motor["plan_data"]["name"] = "Nombre Nuevo del Usuario"
            return super().invoke(prompt)

    import plan_display_i18n as _pdi
    _pdi.build_chat_llm = lambda model, **kwargs: _LLMQueRenombra(**kwargs)
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(_RESPUESTA)

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert _titulo_traducido(motor) is None, (
        "se pego la traduccion del titulo viejo encima de un plan ya renombrado")


# ------------------------------------------------------------------ bordes

def test_sin_titulo_en_ninguna_parte_no_rompe_ni_escribe(motor):
    """Fail-open: el enriquecimiento de meals sigue su curso."""
    motor["plan_data"] = {"days": [{"meals": [dict(_MEAL)]}]}
    motor["columna_name"] = None
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(_RESPUESTA)

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert _titulo_traducido(motor) is None
    assert motor["plan_data"]["days"][0]["meals"][0].get("_display"), (
        "el titulo ausente no puede impedir que los platos se traduzcan")


def test_un_titulo_ya_traducido_no_se_vuelve_a_pedir(motor):
    """Evita pagar la traduccion del mismo titulo en cada enriquecimiento."""
    motor["plan_data"] = {
        "days": [{"meals": [dict(_MEAL)]}],
        "_display": {"en-US": {"name": "Already There"}},
    }
    motor["columna_name"] = "Sazon Fuerte, Vida en Equilibrio"
    _FakeLLM.NEXT_RESPONSE = _FakeResponse(_RESPUESTA)

    pdi.enrich_plan_display("plan-1", "user-1", "en-US")

    assert _titulo_traducido(motor) == "Already There"
    assert "Sazon Fuerte" not in _FakeLLM.captured_prompts[0]


def test_la_columna_se_lee_filtrando_por_user_id():
    """I2: toda lectura de `meal_plans` filtra por `user_id`. El helper nuevo no es
    una excepcion — es un SELECT nuevo sobre una tabla user-scoped.

    SIN el fixture `motor` a proposito: instala un doble de este mismo helper, y con
    el puesto `inspect.getsource` devuelve el codigo del DOBLE. El test pasaba a
    inspeccionar el fake en vez de produccion — un verificador que mira lo que el no
    es no verifica nada.
    """
    import inspect
    fuente = inspect.getsource(pdi._fetch_plan_name_column)
    assert "meal_plans" in fuente, "no es el helper real"
    assert "AND user_id = %s" in fuente, "el SELECT del titulo no filtra por user_id"
