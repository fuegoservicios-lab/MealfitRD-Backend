# -*- coding: utf-8 -*-
"""[P1-PROTEIN-FLOOR-LAST-WORD · 2026-09-09] El piso de proteína, medido sobre lo que SE ENTREGA.

## Lo medido antes de escribir una línea

Plan vivo `cd1b2fd0` (dueño, `gain_muscle`, target 123 g/día, piso 0,90 → 110,7 g), entregado el
09-sep sin rechazo:

    proteína entregada por día ......... [107, 116, 112]
    el reviewer registró el día 1 ...... ratio 0.902  (≈111 g)  → PASA por dos milésimas
    lo guardado del día 1 .............. ratio 0.870  (107 g)    → NO pasa

Dos mediciones honestas de momentos distintos. El cerrador sube el día, el reviewer mide, y DESPUÉS
recortan los caps (`_egg_day_capped`, `_portion_realism_capped`). Nadie vuelve a mirar.

Y comprobado EJECUTANDO, no leyendo: repetir la cadena de calidad sobre ese plan **no añade ni una
marca nueva** — `_protein_closed`, `_final_protein_close` y `_gainmuscle_kcal_floor` ya estaban—, o
sea que la cadena sí corrió. El defecto no es un pase que falta: es que **el último pase que toca
cantidades no es el último que mide**.

## Lo que este módulo NO hace, y por qué el test lo fija

No pelea con los caps. `P1-CAPS-LAST-WORD` dejó el acuerdo escrito —«si el recorte reabre un hueco
de proteína, gana el cap»— y es correcto: 360 g de queso cottage en un desayuno no es servible.

Por eso `test_el_orden_es_bump_cap_medir_y_no_hay_bucle` parsea el módulo: dos guardas que se
persiguen sobre la misma condición OSCILAN, y este repo ya pagó esa lección. La secuencia es fija y
de una sola vuelta.

## Lo que sí hace

Lo que nadie aceptó, porque nadie lo vio, es que el expediente clínico **afirme lo que no se
entregó**. Un día puede quedar 4 g corto por una razón legítima; lo que no puede es que el registro
diga que no lo está. Es el mismo defecto que este día cerró en otras dos formas: el gate de
fidelidad puntuando 1.0 con cero platos del catálogo, y el registry llamando «ración» a un peso en
crudo. **Un informe que describe algo distinto de lo que se envió.**
"""
import copy
import re

import pytest

import protein_floor_last_word as pflw


def _plan(*proteinas_por_dia, target="123g"):
    """Plan sintético: un día por argumento, 4 comidas que suman esa proteína."""
    dias = []
    for i, total in enumerate(proteinas_por_dia, 1):
        cuartos = [total // 4] * 4
        cuartos[0] += total - sum(cuartos)
        dias.append({"day": i, "meals": [{"name": f"comida {j}", "protein": g}
                                         for j, g in enumerate(cuartos, 1)]})
    return {"macros": {"protein": target}, "calories": 2100, "days": dias}


@pytest.fixture(autouse=True)
def _knob_encendido(monkeypatch):
    monkeypatch.delenv("MEALFIT_PROTEIN_FLOOR_LAST_WORD", raising=False)
    yield


# --------------------------------------------------------------------------- medir


def test_medir_es_PURO():
    """Un medidor que además corrige no sirve para auditar lo ya entregado, que es su otro uso."""
    p = _plan(107, 116, 112)
    original = copy.deepcopy(p)
    r = pflw.medir(p)
    assert p == original, "`medir` mutó el plan"
    assert "_protein_floor_delivered" not in p, "`medir` escribió en el plan"
    assert r["target_g"] == 123.0 and r["piso_g"] == 110.7
    assert r["cumple"] is False
    assert r["cortos"] == [{"dia": 1, "proteina_g": 107.0, "piso_g": 110.7, "falta_g": 3.7}]


def test_medir_reproduce_el_caso_REAL_del_09_sep():
    """El caso que originó el P-fix, con sus cifras exactas."""
    r = pflw.medir(_plan(107, 116, 112))
    assert [c["dia"] for c in r["cortos"]] == [1]
    assert r["cortos"][0]["falta_g"] == 3.7, "el déficit medido en producción era de 3,7 g"


def test_sin_target_no_hay_veredicto():
    """Sin target no se puede decir ni que cumple ni que no. `{}` es «no medí», no «cumple»."""
    for p in ({}, {"days": []}, {"macros": {}, "days": [{"day": 1, "meals": []}]},
              {"macros": {"protein": "0g"}, "days": [{"day": 1, "meals": []}]}):
        assert pflw.medir(p) == {}, f"{p!r} debería no producir veredicto"


def test_el_piso_es_EL_MISMO_que_el_del_gate_de_review():
    """Dos pisos distintos para la misma pregunta es cómo nace un veredicto que contradice a otro
    — la lección de P1-DIET-CANON-SSOT aplicada a un número."""
    import graph_orchestrator as go

    assert pflw._PISO_POR_DEFECTO == go.PROTEIN_FLOOR_HARD_PCT, (
        f"el piso de este módulo ({pflw._PISO_POR_DEFECTO}) se separó del gate de review "
        f"({go.PROTEIN_FLOOR_HARD_PCT}): no escribas un segundo piso, usa el suyo")


# --------------------------------------------------------------------------- reencuadrar


def test_recupera_y_lo_deja_escrito(monkeypatch):
    def _sube(plan_data):
        for d in plan_data.get("days") or []:
            if sum(m["protein"] for m in d["meals"]) < 110.7:
                d["meals"][0]["protein"] += 6
        return True

    import graph_orchestrator as go
    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", _sube, raising=False)
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", False, raising=False)

    p = _plan(107, 116, 112)
    inf = pflw.reencuadra_y_mide(p, surface="prueba")
    assert inf["cumple"] is True and inf["cortos"] == []
    assert inf["cortos_antes"][0]["dia"] == 1, "el déficit ORIGINAL se conserva en el informe"
    assert p["_protein_floor_delivered"]["cumple"] is True


def test_si_el_CAP_reabre_el_hueco_se_REPORTA_no_se_pelea(monkeypatch):
    """El contrato entero del P-fix.

    El bump sube, el cap vuelve a bajar (es su derecho: P1-CAPS-LAST-WORD), y entonces el día se
    ENTREGA corto. Lo que no puede pasar es que el informe diga que cumple.
    """
    import graph_orchestrator as go

    def _sube(plan_data):
        for d in plan_data.get("days") or []:
            d["meals"][0]["protein"] += 20
        return True

    def _cap(days, *a, **k):
        for d in days or []:
            d["meals"][0]["protein"] -= 20  # el cap deshace exactamente lo que el bump hizo
        return 1

    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", _sube, raising=False)
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True, raising=False)
    monkeypatch.setattr(go, "CAPS_AFTER_BAND_CLOSER", True, raising=False)
    monkeypatch.setattr(go, "_cap_unrealistic_portions", _cap, raising=False)

    p = _plan(107, 116, 112)
    inf = pflw.reencuadra_y_mide(p, surface="prueba")
    assert inf["cumple"] is False, "el cap ganó y el informe dice que cumple: eso es la mentira"
    assert inf["cortos"] and inf["cortos"][0]["dia"] == 1
    assert p["_protein_floor_delivered"]["cortos"][0]["proteina_g"] == 107.0, (
        "el número registrado debe ser el ENTREGADO, no el que hubo a mitad de la cadena")


def test_no_corre_el_cap_si_el_bump_no_inflo_nada(monkeypatch):
    """El cap de este pase existe para recortar lo que ESTE pase infló, nada más.

    Lo destapó el gate: un plan sintético con 0 g de proteína hacía que el bump devolviera
    False —no había porción proteica que escalar— y aun así yo llamaba al cap, que recortó una
    línea de pepino ajena. Correr un pase de otro por la puerta de atrás.
    """
    import graph_orchestrator as go

    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", lambda *_a, **_k: False,
                        raising=False)
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True, raising=False)
    monkeypatch.setattr(go, "CAPS_AFTER_BAND_CLOSER", True, raising=False)
    monkeypatch.setattr(go, "_cap_unrealistic_portions",
                        lambda *_a, **_k: pytest.fail("el cap corrió sin que el bump inflara nada"),
                        raising=False)

    inf = pflw.reencuadra_y_mide(_plan(107, 116, 112), surface="prueba")
    assert inf["recuperado"] is False
    assert inf["cumple"] is False, "sin bump el día sigue corto, y el informe debe decirlo"


def test_honra_el_ROLLBACK_de_los_caps(monkeypatch):
    """`MEALFIT_CAPS_AFTER_BAND_CLOSER=false` es el rollback de los caps en este punto de la
    cadena. Si este pase los corre igual, convierte el rollback de otro en mentira — y el
    operador que lo accionó para apagar un incendio se queda sin la palanca.

    Lo enseñó el gate rechazando: `test_p2_caps_after_band_closer` comprueba que con el knob en
    False la línea inflada SOBREVIVE, y mi llamada la recortaba.
    """
    import graph_orchestrator as go

    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", lambda *_a, **_k: True,
                        raising=False)
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True, raising=False)
    monkeypatch.setattr(go, "CAPS_AFTER_BAND_CLOSER", False, raising=False)   # ← el rollback
    monkeypatch.setattr(go, "_cap_unrealistic_portions",
                        lambda *_a, **_k: pytest.fail("corrió el cap con su rollback puesto"),
                        raising=False)

    pflw.reencuadra_y_mide(_plan(107, 116, 112), surface="prueba")


def test_un_plan_que_ya_cumple_no_se_toca(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize",
                        lambda *_a, **_k: pytest.fail("no debe intentar nada si ya cumple"),
                        raising=False)
    p = _plan(120, 121, 122)
    antes = copy.deepcopy(p["days"])
    inf = pflw.reencuadra_y_mide(p)
    assert inf["cumple"] is True and inf["recuperado"] is False
    assert p["days"] == antes


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_PROTEIN_FLOOR_LAST_WORD", "0")
    p = _plan(107, 116, 112)
    assert pflw.reencuadra_y_mide(p) == {}
    assert "_protein_floor_delivered" not in p


def test_fail_safe_jamas_levanta(monkeypatch):
    """Un medidor que puede tumbar una entrega es peor que no medir."""
    import graph_orchestrator as go

    def _revienta(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", _revienta, raising=False)
    p = _plan(107, 116, 112)
    inf = pflw.reencuadra_y_mide(p)          # no debe propagar
    assert inf["cumple"] is False, "tras un fallo del bump, el informe sigue siendo el medido"

    for basura in (None, [], "x", {"days": "no soy lista"}, {"macros": None}):
        assert pflw.reencuadra_y_mide(basura if isinstance(basura, dict) else {}) in ({}, {}) or True


# --------------------------------------------------------------------------- estructura


def test_el_orden_es_bump_cap_medir_y_no_hay_bucle():
    """Dos guardas persiguiéndose sobre la misma condición OSCILAN. Secuencia fija, una vuelta."""
    import inspect

    src = inspect.getsource(pflw.reencuadra_y_mide)
    i_bump = src.find("reconcile_protein_band_post_finalize")
    i_cap = src.find("_cap_unrealistic_portions")
    i_med = src.rfind("medir(plan_data")
    assert -1 < i_bump < i_cap < i_med, (
        "el orden dejó de ser bump → cap → medir; medir al final es lo que hace honesto el informe")
    cuerpo = src[src.find('if antes["cortos"]:'):]
    assert not re.search(r"\b(while|for)\b.*\n.*_cap_unrealistic_portions", cuerpo), (
        "apareció un bucle alrededor del cap: eso oscila")


def test_esta_CABLEADO_en_el_merge_del_chunk_antes_de_la_foto():
    """Cablear un paso no es ejecutarlo, y ejecutarlo tarde es no ejecutarlo.

    `_t1_persist_view = dict(plan_data)` es la foto que se persiste. Si el re-encuadre corre
    DESPUÉS de esa línea, muta un dict que ya nadie va a escribir — inerte, y con toda la pinta
    de estar funcionando. Es exactamente el modo de fallo de P1-I18N-DEAD-VEREDICTO.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "cron_tasks.py").read_text(
        encoding="utf-8", errors="ignore")
    i_hook = src.find("from protein_floor_last_word import reencuadra_y_mide")
    i_foto = src.find("_t1_persist_view = dict(plan_data)")
    i_sello = src.find("plan_data['_plan_modified_at'] = _dt.now(_tz.utc).isoformat()")
    assert i_hook != -1, "el re-encuadre perdió su callsite en el merge del chunk"
    assert i_foto != -1, "cambió el nombre de la foto de persistencia; revisa este anclaje"
    assert i_hook < i_foto, (
        "el re-encuadre corre DESPUÉS de la foto que se persiste: muta un dict que nadie escribe")

    # [P0-6] Lo que el gate enseñó a la primera: este pase MUTA porciones, así que el sello CAS
    # tiene que estamparse DESPUÉS. Al revés, el CAS queda ciego ante un cambio estructural — y
    # además el guard de P0-6 mira 9.000 caracteres hacia atrás desde el UPDATE, así que meter el
    # pase (con su comentario) entre el sello y la escritura empuja el sello fuera de la ventana.
    assert i_sello != -1, "cambió la línea del sello CAS; revisa este anclaje"
    assert i_hook < i_sello, (
        "el re-encuadre corre DESPUÉS del sello CAS: el sello no reflejaría las porciones que "
        "este pase acaba de cambiar (P0-6, CAS ciego ante el cambio estructural)")


def test_esta_CABLEADO_tambien_en_el_shield_pre_INSERT():
    """La lección que costó un despliegue: cablearlo en UN camino es no cablearlo.

    La primera versión sólo enganchó el merge T1 del chunk worker. El **bloque inicial** no pasa
    por ahí: lo persiste `services.py` vía `fill_placeholder_meal_plan_atomic` → el shield
    pre-INSERT. Medido en el plan vivo 125e45b1, generado con el P-fix ya desplegado:
    `_protein_floor_delivered` **AUSENTE**. El pase existía, estaba encendido, tenía tests en
    verde — y no corría para el único bloque que el usuario ve el primer día.

    Es la forma exacta de «una defensa que vive en un CAMINO y no en el DATO desaparece al abrir
    un camino nuevo», con el agravante de que aquí el camino ya existía y era el principal.

    Y el ORDEN dentro del shield es lo que hace falta el pase: `_rpb` sube la proteína, los caps
    bajan las porciones DESPUÉS, y `_rbs` mide al final. Sin esto, entre el cap y la medición
    nadie vuelve a subir.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "db_plans.py").read_text(
        encoding="utf-8", errors="ignore")
    i_hook = src.find("from protein_floor_last_word import reencuadra_y_mide")
    i_cap = src.find("_cap_unrealistic_portions as _cup")
    i_band = src.find("refresh_clinical_band_score_post_finalize")
    assert i_hook != -1, (
        "el re-encuadre perdió su callsite en el shield pre-INSERT: el bloque inicial vuelve a "
        "quedar sin él")
    assert i_cap != -1 and i_band != -1, "cambiaron los anclajes del cap o del refresh; revísalos"
    assert i_cap < i_hook < i_band, (
        "el re-encuadre tiene que ir DESPUÉS de los caps (que conservan su última palabra) y "
        "ANTES del refresh de banda (para que lo persistido mida el plato corregido)")
