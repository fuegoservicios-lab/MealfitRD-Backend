# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-846 · 2026-09-29] Edad mínima de 18 (auditoría App Store, fila 16.2 y §A.9).

Los Términos (§2) y la Privacidad (§11) dicen «solo mayores de 18»; el formulario aceptaba de 12 a 100 y mandaba los
datos de salud de un menor a la IA. Aquí se ancla:
  1. el SSOT (`edad_minima.py`): qué es «menor», el 422 `underage` y su cuerpo;
  2. la paridad del mínimo: router, coach y formulario dicen 18;
  3. cada puerta por la que la edad llega a la generación lleva el rechazo, y va ANTES que el resto de validaciones;
  4. conducta: un menor recibe 422 `underage` sin que nada más corra, un adulto sigue su camino, el perfil falla
     abierto, `PATCH /api/profile` no guarda la edad de un menor y el coach no la guarda ni pide otra;
  5. P1-MINOR-SAFETY-GATE sigue encendido como capa extra.
Sin base de datos: el `.env` de desarrollo apunta a producción y ningún test la toca.
"""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest
from fastapi import BackgroundTasks, HTTPException, Response

import edad_minima as em

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT_FV = _BACKEND.parent / "frontend" / "src" / "config" / "formValidation.js"
UID = "11111111-2222-4333-8444-555555555555"


# ═════════════════════════════════════════════ 1. el SSOT
@pytest.mark.parametrize("valor", [17, "17", 1, "1", "15", "17,9", "17.99", 12, 5.5, " 16 "])
def test_menor_es_una_edad_legible_de_1_a_17(valor):
    assert em.es_menor_de_edad(valor) is True


@pytest.mark.parametrize("valor", [18, "18", "18.0", 30, "100", 0, "0", -3, "0.5", None, "", "abc", "15 años",
                                   True, False, float("nan"), float("inf"), [], {}])
def test_no_es_menor_lo_adulto_ni_lo_ilegible(valor):
    # Lo ilegible (0, negativo, texto) no es «menor»: eso lo rechaza el rango (`invalid_biometric_range`).
    assert em.es_menor_de_edad(valor) is False


def test_el_422_lleva_el_codigo_el_campo_y_el_minimo():
    with pytest.raises(HTTPException) as e:
        em.rechazar_si_menor("30", "16", origen="test")
    assert e.value.status_code == 422
    d = e.value.detail
    assert d["code"] == d["error_code"] == "underage"
    assert d["field"] == "age" and d["min_age"] == 18
    assert d["message"] == "Bioboros es solo para mayores de 18 años."


def test_sin_menores_no_hace_nada():
    em.rechazar_si_menor("30", None, "", "abc", origen="test")


def test_el_perfil_se_mira_despues_de_la_peticion_y_falla_abierto(monkeypatch):
    import db
    lecturas = []

    def _lee(sql, params=None, fetch_one=False, **_):
        lecturas.append((" ".join(sql.split()), params))
        return {"age": "16"}

    monkeypatch.setattr(db, "execute_sql_query", _lee)
    with pytest.raises(HTTPException) as e:
        em.rechazar_si_menor_en_perfil(UID, "30", origen="test")
    assert e.value.detail["code"] == "underage"
    assert lecturas == [("SELECT health_profile->>'age' AS age FROM user_profiles WHERE id = %s", (UID,))]

    # La edad de la petición ya basta: no se lee la base.
    lecturas.clear()
    with pytest.raises(HTTPException):
        em.rechazar_si_menor_en_perfil(UID, "15", origen="test")
    assert lecturas == []

    # Adulto en el perfil: pasa. Invitado / sin cuenta: no se lee nada.
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: {"age": 44})
    em.rechazar_si_menor_en_perfil(UID, origen="test")
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: pytest.fail("un invitado no tiene perfil"))
    em.rechazar_si_menor_en_perfil(None, origen="test")
    em.rechazar_si_menor_en_perfil("guest", origen="test")

    # Base caída: fail-open (la puerta principal es la edad de la petición; un perfil ya no guarda menores).
    def _boom(*a, **k):
        raise RuntimeError("sin base")

    monkeypatch.setattr(db, "execute_sql_query", _boom)
    em.rechazar_si_menor_en_perfil(UID, origen="test")


# ═════════════════════════════════════════════ 2. paridad del mínimo
def test_el_minimo_es_18_en_el_router_el_coach_y_el_formulario():
    from routers import plans as rp
    import tools

    assert em.EDAD_MINIMA == 18
    assert rp._BIO_RANGES["age"] == (18, 100)
    assert tools._CHAT_BIO_RANGES["age"] == (18, 100)
    fv = _FRONT_FV.read_text(encoding="utf-8")
    m = re.search(r"\n\s*age:\s*\{\s*min:\s*(\d+)\s*,\s*max:\s*(\d+)", fv)
    assert m and (int(m.group(1)), int(m.group(2))) == (18, 100), "BIO_RANGES.age del formulario no es 18-100"
    assert "export const esMenorDeEdad" in fv, "el espejo del formulario (`esMenorDeEdad`) desapareció"


# ═════════════════════════════════════════════ 3. cada puerta lleva el rechazo, y primero
def _fuente_de(rel: str, nombre: str) -> str:
    src = (_BACKEND / rel).read_text(encoding="utf-8")
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre:
            return ast.get_source_segment(src, n)
    raise AssertionError(f"{rel} no define {nombre}")


_PUERTAS = [
    # (fichero, función, llamada que debe llevar)
    ("routers/plans.py", "api_analyze", "rechazar_si_menor(data.get(\"age\")"),
    ("routers/plans.py", "api_analyze_stream", "rechazar_si_menor(data.get(\"age\")"),
    ("generation_inputs.py", "validate_generation_request", "rechazar_si_menor(data.get(\"age\")"),
    ("routers/plans.py", "api_swap_meal", "rechazar_si_menor_en_perfil("),
    ("routers/plans.py", "api_regenerate_day", "rechazar_si_menor_en_perfil(verified_user_id, data.get(\"age\")"),
    ("routers/plans.py", "api_fix_sodium_day", "rechazar_si_menor_en_perfil(verified_user_id"),
    ("routers/plans.py", "api_retry_chunk", "rechazar_si_menor_en_perfil(verified_user_id"),
    ("routers/plans.py", "api_regenerate_dead_lettered_simplified", "rechazar_si_menor_en_perfil(verified_user_id"),
    ("routers/plans.py", "api_regen_degraded_chunks", "rechazar_si_menor_en_perfil(verified_user_id"),
    ("routers/user_data.py", "api_patch_profile", "validar_edad_del_perfil(hp_patch.get(\"age\")"),   # ronda 1
]


@pytest.mark.parametrize("fichero, funcion, llamada", _PUERTAS)
def test_cada_puerta_de_la_generacion_rechaza_al_menor(fichero, funcion, llamada):
    assert llamada in _fuente_de(fichero, funcion), f"{fichero}::{funcion} sin el 422 underage"


@pytest.mark.parametrize("fichero, funcion", [
    ("routers/plans.py", "api_analyze"),
    ("routers/plans.py", "api_analyze_stream"),
    ("generation_inputs.py", "validate_generation_request"),
])
def test_en_el_formulario_el_rechazo_va_antes_que_cualquier_otra_validacion(fichero, funcion):
    src = _fuente_de(fichero, funcion)
    i = src.index("rechazar_si_menor(")
    for despues in ("_close_medical_freetext_scope(", "_hydrate_country_from_profile_for_submit(",
                    "_validate_form_data_min(", "_validate_form_data_ranges("):
        assert i < src.index(despues), f"{funcion}: el 422 underage debe ir antes de {despues}"


def test_el_perfil_rechaza_antes_de_escribir():
    src = _fuente_de("routers/user_data.py", "api_patch_profile")
    assert src.index("validar_edad_del_perfil(") < src.index("update_user_health_profile_atomic(")


# ═════════════════════════════════════════════ 4. conducta
_MENOR = {"user_id": "guest", "session_id": None, "age": "15"}


def test_analyze_menor_422_underage_antes_que_nada():
    from routers import plans as rp
    with pytest.raises(HTTPException) as e:
        rp.api_analyze(BackgroundTasks(), Response(), data=dict(_MENOR), verified_user_id=None, _rl=None, _ia=None)
    assert e.value.status_code == 422 and e.value.detail["code"] == "underage"


def test_analyze_adulto_sigue_a_las_demas_validaciones():
    # Adulto con el formulario a medias: el rechazo por edad NO salta y responde la validación de siempre.
    from routers import plans as rp
    with pytest.raises(HTTPException) as e:
        rp.api_analyze(BackgroundTasks(), Response(), data={"user_id": "guest", "age": "30"},
                       verified_user_id=None, _rl=None, _ia=None)
    assert e.value.status_code == 422 and e.value.detail["code"] == "missing_required_fields"


def test_analyze_stream_menor_422_antes_de_abrir_el_stream():
    from routers import plans as rp
    with pytest.raises(HTTPException) as e:
        asyncio.run(rp.api_analyze_stream(request=None, background_tasks=BackgroundTasks(), data=dict(_MENOR),
                                          verified_user_id=None, _rl=None, _ia=None))
    assert e.value.status_code == 422 and e.value.detail["code"] == "underage"


def test_la_cola_de_generacion_rechaza_al_menor():
    from generation_inputs import validate_generation_request
    with pytest.raises(HTTPException) as e:
        validate_generation_request({"user_id": UID, "age": "17"}, UID)
    assert e.value.detail["code"] == "underage"


def test_swap_de_un_invitado_menor_422(monkeypatch):
    from routers import plans as rp
    with pytest.raises(HTTPException) as e:
        rp.api_swap_meal(BackgroundTasks(), data={"user_id": "guest", "age": "14"}, verified_user_id=None,
                         _rl=None, _ia=None)
    assert e.value.detail["code"] == "underage"


def test_arreglar_sodio_con_un_perfil_menor_422(monkeypatch):
    from routers import plans as rp
    monkeypatch.setattr(em, "edad_del_perfil", lambda uid: "16")
    with pytest.raises(HTTPException) as e:
        rp.api_fix_sodium_day("plan-x", data={}, verified_user_id=UID, _rl=None, _ia=None)
    assert e.value.detail["code"] == "underage"


def test_patch_perfil_no_guarda_la_edad_de_un_menor(monkeypatch):
    import db
    from routers import user_data as ud
    monkeypatch.setattr(db, "update_user_health_profile_atomic",
                        lambda *a, **k: pytest.fail("no debe escribir la edad de un menor"))
    body = ud.ProfilePatchBody(health_profile={"age": 15, "weight": 50})
    with pytest.raises(HTTPException) as e:
        asyncio.run(ud.api_patch_profile(body=body, verified_user_id=UID, _rl=None))
    assert e.value.status_code == 422 and e.value.detail["code"] == "underage"


@pytest.mark.parametrize("valor", ["15", "17 años", "12"])
def test_el_coach_no_guarda_la_edad_de_un_menor_ni_pide_otra(valor):
    import tools
    ok, mensaje, _ = tools._valor_canonico_del_formulario("age", valor)
    assert ok is False
    assert "mayores de 18" in mensaje and "no le pidas otra edad" in mensaje
    assert "Pregúntale al usuario" not in mensaje


def test_el_coach_si_guarda_la_de_un_adulto():
    import tools
    assert tools._valor_canonico_del_formulario("age", "30 años") == (True, "30", None)
    ok, mensaje, _ = tools._valor_canonico_del_formulario("age", "300")
    assert ok is False and "18-100" in mensaje


# ═════════════════════════════════════════════ 5. la capa extra sigue ahí
def test_el_gate_de_menores_sigue_encendido_por_defecto():
    import nutrition_calculator as nc
    assert nc.MINOR_SAFETY_GATE_ENABLED is True
    res = nc.get_nutrition_targets({"weight": 60, "weightUnit": "kg", "height": 165, "age": 15,
                                    "gender": "male", "activityLevel": "sedentary", "mainGoal": "lose_fat"})
    assert (res.get("minor_safety") or {}).get("applied") is True


# ═════════════════════════════════════════════ ronda 1
@pytest.mark.parametrize("valor", ["15 años", "1e1", "1_5", "15abc", "inf", "nan", "", "  "])
def test_r1_solo_un_numero_limpio_es_una_edad(valor):
    # `float("1_5")` era 15 y `float("1e1")` 10: el formulario (`EDAD_LIMPIA`) no los lee y el backend tampoco.
    assert em.edad_declarada(valor) is None and em.es_menor_de_edad(valor) is False


def test_r1_la_expresion_limpia_es_la_misma_en_python_y_en_js():
    py = (_BACKEND / "edad_minima.py").read_text(encoding="utf-8")
    js = _FRONT_FV.read_text(encoding="utf-8")
    m_py = re.search(r'_EDAD_LIMPIA = re\.compile\(r"([^"]+)"\)', py)
    m_js = re.search(r"const EDAD_LIMPIA = /(.+)/;", js)
    assert m_py and m_js and m_py.group(1) == m_js.group(1), (m_py and m_py.group(1), m_js and m_js.group(1))
    assert "Math.trunc" in js[js.index("export const edadDeclarada"):][:400]


def test_r1_el_mensaje_sale_de_la_constante():
    src = (_BACKEND / "edad_minima.py").read_text(encoding="utf-8")
    assert 'MENSAJE_MENOR = f"Bioboros es solo para mayores de {EDAD_MINIMA} años."' in src
    assert em.MENSAJE_MENOR == f"Bioboros es solo para mayores de {em.EDAD_MINIMA} años."
    from routers import plans as rp
    assert em.EDAD_MAXIMA == rp._BIO_RANGES["age"][1]


@pytest.mark.parametrize("valor", ["abc", 0, "0", -3, "250", "15 años"])
def test_r1_patch_perfil_rechaza_la_edad_ilegible_con_el_rango_de_siempre(monkeypatch, valor):
    import db
    from routers import user_data as ud
    monkeypatch.setattr(db, "update_user_health_profile_atomic", lambda *a, **k: pytest.fail("no debe escribir"))
    body = ud.ProfilePatchBody(health_profile={"age": valor})
    with pytest.raises(HTTPException) as e:
        asyncio.run(ud.api_patch_profile(body=body, verified_user_id=UID, _rl=None))
    assert e.value.status_code == 422 and e.value.detail["code"] == "invalid_biometric_range"
    assert e.value.detail["errors"][0]["field"] == "age"
    assert e.value.detail["errors"][0]["accepted_range"] == [18, 100]


def test_r1_patch_perfil_deja_guardar_un_adulto(monkeypatch):
    import db
    from routers import user_data as ud
    escrito = {}

    def _atomico(uid, fusionar):
        hp = {}
        fusionar(hp)
        escrito.update(hp)
        return hp

    monkeypatch.setattr(db, "update_user_health_profile_atomic", _atomico)
    body = ud.ProfilePatchBody(health_profile={"age": "30"})
    assert asyncio.run(ud.api_patch_profile(body=body, verified_user_id=UID, _rl=None)) == {"success": True}
    assert escrito == {"age": "30"}


# ─── el chat del coach
@pytest.mark.parametrize("funcion", ["api_chat_stream", "api_chat"])
def test_r1_el_chat_rechaza_al_menor_antes_de_guardar_y_tras_fundir_el_perfil(funcion):
    src = _fuente_de("routers/chat.py", funcion)
    pre = src.index('rechazar_si_menor((form_data or {}).get("age") if isinstance(form_data, dict) else None')
    primera_escritura = re.search(r"\n\s+save_message(?:_with_attachments)?\(", src).start()
    assert pre < primera_escritura and pre < src.index("merge_form_data_with_profile(")
    fundido = src.index("merge_form_data_with_profile(")
    post = src.index('rechazar_si_menor((form_data or {}).get("age"), origen="chat (perfil fundido)")')
    assert fundido < post


def test_r1_el_merge_del_chat_no_escribe_la_edad_de_un_menor(monkeypatch):
    import services
    escrito = {}

    def _atomico(uid, mutador):
        hp = {}
        if mutador(hp) is not False:
            escrito.update(hp)

    monkeypatch.setattr(services, "get_user_profile", lambda uid: {"health_profile": {}})
    monkeypatch.setattr(services, "update_user_health_profile_atomic", _atomico)
    services.merge_form_data_with_profile(UID, {"age": "15", "weight": "60"})
    assert escrito == {"weight": "60"}

    creado = {}
    monkeypatch.setattr(services, "get_user_profile", lambda uid: None)
    monkeypatch.setattr(services, "upsert_user_profile", lambda uid, hp: creado.update(hp))
    services.merge_form_data_with_profile(UID, {"age": "16", "gender": "female"})
    assert creado == {"gender": "female"}

    escrito.clear()
    monkeypatch.setattr(services, "get_user_profile", lambda uid: {"health_profile": {}})
    services.merge_form_data_with_profile(UID, {"age": "40"})
    assert escrito == {"age": "40"}


def test_r1_el_chat_responde_422_si_el_formulario_trae_un_menor(monkeypatch):
    # Conducta real de /api/chat: el 422 sale antes de guardar el mensaje ni fundir el perfil.
    from routers import chat as ch
    monkeypatch.setattr(ch, "save_message", lambda *a, **k: pytest.fail("no debe guardar el mensaje"))
    monkeypatch.setattr(ch, "merge_form_data_with_profile", lambda *a, **k: pytest.fail("no debe fundir"))
    monkeypatch.setattr(ch, "_resolve_chat_local_time", lambda d, t, u: (d, t))
    import db_chat
    monkeypatch.setattr(db_chat, "get_session_owner", lambda sid: None)
    with pytest.raises(HTTPException) as e:
        ch.api_chat(BackgroundTasks(), data={"session_id": "s1", "prompt": "hola", "form_data": {"age": "14"}},
                    verified_user_id=UID, _ia=None)
    assert e.value.status_code == 422 and e.value.detail["code"] == "underage"


# ─── la herramienta del coach que genera un plan
def test_r1_el_coach_no_genera_un_plan_a_un_menor(monkeypatch):
    import tools
    monkeypatch.setattr(tools, "run_plan_pipeline", lambda *a, **k: pytest.fail("no debe generar"))
    out = tools.execute_generate_new_plan(UID, {"age": "15", "weight": "60"})
    assert out.startswith("ERROR:") and em.MENSAJE_MENOR in out

    # sin edad en el formulario, la del perfil consolidado
    monkeypatch.setattr(tools, "get_user_profile", lambda uid: {"health_profile": {"age": 13, "weight": 50}})
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: False)
    out = tools.execute_generate_new_plan(UID, {})
    assert out.startswith("ERROR:") and em.MENSAJE_MENOR in out
