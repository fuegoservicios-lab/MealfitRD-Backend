"""[P1-DISPLAY-LOTE-POR-COMIDAS · 2026-08-21] El lote se medía en DÍAS y el coste lo
fijan las COMIDAS.

`MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS` vale 4 por defecto. Un plan normal trae 4-5
comidas al día, así que el lote ORDINARIO son 16-20 comidas — y esa es la unidad que
paga el tope de salida de 8.000 tokens.

LO QUE SE MIDIÓ (271 comidas de los 6 planes vivos con días, 2026-08-21). Reconstruida
la forma exacta del JSON de respuesta —`{day_idx, meal_idx, name, desc, ingredients[],
recipe[]}`— sobre el texto fuente, a 3 chars/token y +20 % de inflación por traducir al
francés o al portugués:

    16 comidas  ~6.642 tok de media · ~10.544 en el peor caso   → SE PASA
    20 comidas  ~8.302 tok de media · ~12.963 en el peor caso   → SE PASA DE MEDIA

O sea: el lote por defecto se pasa del tope en el peor caso, y un plan con 5 comidas
diarias (existe: 76a6836d, 55 comidas en 11 días) se pasa **de media**. En el peor caso
medido caben 11 comidas antes de tocar el tope.

QUÉ PASABA AL PASARSE, que es la parte que duele: la salida se trunca, el JSON no
parsea, `last_skip_reason = "json_parse_error"` y `continue`. Sin retry, sin split y
—hasta `P2-DISPLAY-SIN-TELEMETRIA-RESULTADO`— sin rastro. El usuario se queda ese tramo
del plan en español y el gasto ya se pagó: `_emit_usage_telemetry` corre justo tras el
invoke, así que se cobró la llamada y se tiró el resultado entero.

EL ARREGLO, en tres piezas y en este orden:

  1. Trocear por TAMAÑO PROYECTADO de la salida, no por días. El texto fuente está en
     la mano cuando se decide el lote — estimarlo no cuesta una llamada extra.
  2. Split-and-retry: si un lote no parsea y trae más de una comida, se parte en dos y
     se reintenta cada mitad. Un lote grande envenenado deja de perderse entero.
  3. Techo de invocaciones, porque el split es recursivo y un modelo que devuelve
     basura para TODO haría crecer el trabajo sin fin.

LO QUE NO SE TOCA: `MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS` sigue existiendo como tope
duro por lote. Quitarlo dejaría el troceo entero a merced de una estimación, y una
estimación que se equivoca por lo alto no tiene suelo.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
import json
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MARKER = "P1-DISPLAY-LOTE-POR-COMIDAS"


@pytest.fixture()
def mod():
    import plan_display_i18n

    return importlib.reload(plan_display_i18n)


def _target(day_idx: int, meal_idx: int, chars: int = 1100) -> dict:
    """Una comida del tamaño medido en producción (~1.100 chars de salida)."""
    relleno = "Pollo guisado con vegetales criollos y arroz blanco. " * 4
    n_lineas = max(1, chars // 220)
    return {
        "day_idx": day_idx,
        "meal_idx": meal_idx,
        "name": "Pollo guisado criollo",
        "description": relleno[:120],
        "ingredients": [f"{i + 1}. Pechuga de pollo 180 g" for i in range(6)],
        "recipe": [relleno[:200] for _ in range(n_lineas)],
    }


def _lote_de(n: int, comidas_por_dia: int = 4) -> list[dict]:
    return [_target(i // comidas_por_dia, i % comidas_por_dia) for i in range(n)]


# ============================================================
# 1 · El troceo se dimensiona por la salida, no por los días
# ============================================================


def test_existe_el_particionador_por_tamano(mod) -> None:
    assert hasattr(mod, "_particionar_targets"), (
        f"No existe `_particionar_targets`. El troceo sigue siendo "
        f"`requested_day_indices[i:i+batch_days]`, que mide días. [{_MARKER}]"
    )


def test_un_lote_de_20_comidas_se_parte(mod) -> None:
    """El caso medido: 20 comidas proyectan ~8.302 tokens DE MEDIA contra un tope de
    8.000. Si sale un solo lote, se está reproduciendo el bug."""
    lotes = mod._particionar_targets(_lote_de(20), max_output_tokens=8000)
    assert len(lotes) >= 2, (
        f"20 comidas salieron en {len(lotes)} lote(s). Proyectan ~8.302 tokens de "
        f"salida de media: un solo lote se trunca y se descarta entero. [{_MARKER}]"
    )
    assert sum(len(x) for x in lotes) == 20, "el particionador perdió comidas por el camino"


def test_ningun_lote_proyecta_por_encima_del_tope(mod) -> None:
    for n in (12, 16, 20, 28, 40):
        for lote in mod._particionar_targets(_lote_de(n), max_output_tokens=8000):
            tok = mod._tokens_estimados(lote)
            assert tok < 8000, (
                f"un lote de {len(lote)} comidas proyecta {tok:.0f} tokens con un tope "
                f"de 8.000 (partiendo de {n} comidas). [{_MARKER}]"
            )


def test_una_sola_comida_gigante_no_se_pierde(mod) -> None:
    """El particionador no puede partir una comida por la mitad. Si una sola ya se pasa,
    tiene que salir igualmente en su propio lote —que el LLM la trunque es otro
    problema— y NUNCA desaparecer del reparto."""
    gigante = _target(0, 0, chars=40000)
    lotes = mod._particionar_targets([gigante, _target(0, 1)], max_output_tokens=8000)
    total = [t for lote in lotes for t in lote]
    assert len(total) == 2, (
        f"se perdió una comida: entraron 2 y salieron {len(total)}. Un target que no "
        f"cabe tiene que salir solo, no evaporarse. [{_MARKER}]"
    )


def test_el_tope_en_dias_sigue_acotando_por_arriba(mod) -> None:
    """`BATCH_DAYS` no se elimina: una estimación que se equivoca por lo alto necesita
    un suelo duro. Con comidas diminutas el límite lo pone el tope, no el tamaño."""
    diminutas = [
        {"day_idx": i, "meal_idx": 0, "name": "Té", "description": "",
         "ingredients": [], "recipe": []}
        for i in range(200)
    ]
    lotes = mod._particionar_targets(diminutas, max_output_tokens=8000)
    assert len(lotes) > 1, (
        f"200 comidas diminutas salieron en un solo lote de {len(lotes[0])}. Sin tope "
        f"duro, todo el troceo depende de la estimación. [{_MARKER}]"
    )


def test_el_estimador_crece_con_el_contenido(mod) -> None:
    """MUTACIÓN DE CONTROL. Un estimador que devuelva una constante hace pasar todo lo
    de arriba y no mide nada."""
    poco = mod._tokens_estimados([_target(0, 0, chars=200)])
    mucho = mod._tokens_estimados([_target(0, 0, chars=8000)])
    assert mucho > poco * 3, (
        f"el estimador dio {poco:.0f} para una comida pequeña y {mucho:.0f} para una "
        f"40 veces mayor: no está mirando el contenido. [{_MARKER}]"
    )


def test_el_estimador_cuenta_la_inflacion_al_traducir(mod) -> None:
    """El francés y el portugués se alargan sobre el español; estimar 1:1 sobre el
    fuente deja el margen justo del lado equivocado."""
    t = _target(0, 0)
    import json as _json

    crudo = len(_json.dumps(t, ensure_ascii=False))
    assert mod._tokens_estimados([t]) > crudo / 4.0, (
        f"la estimación es demasiado optimista: no parece contar ni la inflación de la "
        f"traducción ni un ratio conservador de chars/token. [{_MARKER}]"
    )


# ============================================================
# 2 · Split-and-retry en vez de descarte
# ============================================================


def _plan_de(n_dias: int, comidas_por_dia: int = 5) -> dict:
    return {
        "days": [
            {
                "meals": [
                    {
                        "name": f"Comida {d}-{m}",
                        "desc": "Descripción de la comida en español.",
                        "ingredients": ["Pechuga de pollo 180 g", "Arroz blanco 100 g"],
                        "recipe": ["Mise en place: pica la cebolla.", "Cocina 20 min."],
                    }
                    for m in range(comidas_por_dia)
                ]
            }
            for d in range(n_dias)
        ]
    }


def _instalar_motor(mod, monkeypatch, plan_data: dict) -> dict:
    """Los mismos dobles que usa `test_p1_plan_display_i18n.py`, en corto."""
    estado = {"plan_data": plan_data, "invokes": [], "telemetria": 0}

    monkeypatch.setattr(mod, "_try_claim_enrich_lock_cross_worker",
                        lambda plan_id, locale, day_indices: True)
    monkeypatch.setattr(mod, "_release_enrich_lock_cross_worker",
                        lambda plan_id, locale, day_indices: None)
    monkeypatch.setattr(mod, "_circuit_breaker_can_proceed", lambda model_name: True)
    monkeypatch.setattr(mod, "_fetch_plan_data", lambda plan_id, user_id: estado["plan_data"])
    monkeypatch.setattr(mod, "log_llm_usage_event",
                        lambda **kw: estado.__setitem__("telemetria", estado["telemetria"] + 1))

    def _persist(plan_id, mutator, user_id=None, **kw):
        r = mutator(estado["plan_data"])
        if isinstance(r, dict):
            estado["plan_data"] = r
        return estado["plan_data"]

    monkeypatch.setattr(mod, "update_plan_data_atomic", _persist)
    return estado


def _llm_que_trunca(mod, monkeypatch, estado: dict, umbral: int):
    """Un LLM que devuelve basura cuando el lote trae mas de `umbral` comidas —el modo
    de fallo real: la salida se corta a mitad de JSON— y responde bien cuando cabe."""
    def _build(model, **kw):
        llm = MagicMock()

        def _invoke(mensajes):
            # El prompt lista `0. NAME: …` por comida — NO emite `"i":` (eso solo
            # sale una vez, en el ejemplo de formato de la directiva). Contar por la
            # forma que uno IMAGINA en vez de por la que el codigo escribe daba
            # `invokes=[1]` con diez comidas dentro.
            prompt = mensajes[0].content
            n = prompt.count('. NAME: ')
            estado["invokes"].append(n)
            r = MagicMock()
            if n > umbral:
                r.content = '{"meals":[{"i":0,"name":"Trunca'
            else:
                r.content = json.dumps({
                    "meals": [
                        {"i": k, "name": f"Dish {k}", "description": "Desc.",
                         "recipe": ["Mise en place: chop.", "Cook 20 min."],
                         "ingredients": ["Chicken breast 180 g", "White rice 100 g"]}
                        for k in range(n)
                    ]
                })
            return r

        llm.invoke = _invoke
        return llm

    monkeypatch.setattr(mod, "build_chat_llm", _build)


def test_e2e_un_lote_truncado_se_recupera_partiendolo(mod, monkeypatch) -> None:
    """EL CORAZON DEL P-FIX, de punta a punta.

    Antes: el lote de 10 comidas se truncaba, `json_parse_error`, `continue`, y las 10
    se quedaban en espanol con la llamada ya cobrada. Ahora se parte hasta que cabe.
    """
    estado = _instalar_motor(mod, monkeypatch, _plan_de(2, comidas_por_dia=5))
    _llm_que_trunca(mod, monkeypatch, estado, umbral=5)

    r = mod.enrich_plan_display("plan-1", "user-1", "en-US")

    assert r["enriched_meals"] == 10, (
        f"se recuperaron {r['enriched_meals']} de 10 comidas. Con el split, un lote "
        f"truncado no puede perderse entero. [{_MARKER}] · invokes={estado['invokes']}"
    )
    assert len(estado["invokes"]) >= 3, (
        f"solo hubo {len(estado['invokes'])} invocacion(es): el lote grande no se "
        f"llego a partir. [{_MARKER}]"
    )
    assert max(estado["invokes"]) > 5, "sanity: el primer lote tenia que ser el grande"


def test_e2e_sin_split_el_tramo_se_perderia(mod, monkeypatch) -> None:
    """MUTACION DE CONTROL del e2e: con el LLM fallando SIEMPRE, no se escribe nada y
    —esto es lo nuevo— el ciclo lo dice en vez de callarse."""
    estado = _instalar_motor(mod, monkeypatch, _plan_de(2, comidas_por_dia=5))
    _llm_que_trunca(mod, monkeypatch, estado, umbral=0)   # nada le vale

    r = mod.enrich_plan_display("plan-1", "user-1", "en-US")

    assert r["enriched_meals"] == 0
    assert r["skipped"] in ("json_parse_error", "invocation_budget_exhausted"), (
        f"motivo inesperado: {r['skipped']!r}. [{_MARKER}]"
    )
    assert len(estado["invokes"]) <= mod._max_invocaciones_por_ciclo(1), (
        f"{len(estado['invokes'])} invocaciones con un modelo que falla siempre: el "
        f"techo no esta acotando el split recursivo. [{_MARKER}]"
    )


def test_dividir_un_lote_de_una_comida_no_se_puede(mod) -> None:
    """El caso base de la recursión. Sin él, partir un lote de 1 devuelve `([], [x])` o
    `([x], [])` y el bucle no termina nunca."""
    izq, der = mod._dividir_lote([_target(0, 0)])
    assert izq == [] and der == [], (
        f"`_dividir_lote` sobre una sola comida devolvió {izq!r} / {der!r}. Tiene que "
        f"declararse indivisible, o el reintento entra en bucle. [{_MARKER}]"
    )


def test_hay_techo_de_invocaciones(mod) -> None:
    """El split es recursivo: un modelo que devuelve basura para TODO haría crecer el
    trabajo hasta agotar el plan comida a comida."""
    assert hasattr(mod, "_max_invocaciones_por_ciclo"), (
        f"No hay techo de invocaciones. Con split-and-retry, un modelo que falla "
        f"siempre convierte 4 llamadas en 40. [{_MARKER}]"
    )
    for n_lotes, esperado_min in ((1, 1), (5, 5)):
        techo = mod._max_invocaciones_por_ciclo(n_lotes)
        assert techo >= esperado_min, f"techo {techo} < lotes iniciales {n_lotes}"
        assert techo <= n_lotes * 10, f"techo {techo} demasiado laxo para {n_lotes} lotes"


# ============================================================
# 3 · La pérdida definitiva deja rastro
# ============================================================


def test_una_comida_perdida_del_todo_se_reporta(mod) -> None:
    """Tras el split, un lote de UNA comida que sigue sin parsear es pérdida definitiva.
    Ese es el evento que nadie veía."""
    import inspect

    fuente = inspect.getsource(mod.enrich_plan_display)
    assert "targets_perdidos" in fuente, (
        f"el resumen del ciclo no cuenta las comidas perdidas definitivamente tras el "
        f"split. Sin ese número, «se escribieron 12 de 20» es indistinguible de «había "
        f"12». [{_MARKER}]"
    )


def test_el_knob_de_dias_sigue_registrado(mod) -> None:
    """No se sustituye un knob por otro a escondidas: `BATCH_DAYS` sigue vivo como tope
    duro y tiene que seguir siendo configurable."""
    assert mod._plan_display_i18n_batch_days() == 4
    with patch.dict(os.environ, {"MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS": "7"}):
        assert mod._plan_display_i18n_batch_days() == 7
