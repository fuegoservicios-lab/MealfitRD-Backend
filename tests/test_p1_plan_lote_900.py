"""[P1-PLAN-LOTE-900 · 2026-09-29] El coach cambia los ajustes de la app y abre sus pantallas.

El dueño, en modo voz (29-sep 17:39 UTC): «Activa la hidratación». El coach no tenía herramienta para el interruptor y
ANOTÓ UN VASO DE AGUA («Marqué un vaso de agua, así que ya llevas 1 de 9»). Pidió: «que si le pido cualquier cosa como
esa que lo haga, quiero que tenga 100 % acceso a todo de la app».
"""
from __future__ import annotations


import pytest

import ajustes_de_la_app as aj


def _cambio(resultado: str):
    texto, cambio = aj.extraer_marcador(resultado)
    assert aj.MARCADOR not in texto, "el modelo no debe ver el JSON"
    return texto, cambio


# ── 1. El marcador ─────────────────────────────────────────────────────────────────────────────────────────────

def test_el_marcador_se_extrae_y_desaparece():
    r = aj._con_marcador("Hecho.", {"tema": "dark"})
    texto, cambio = aj.extraer_marcador(r)
    assert (texto, cambio) == ("Hecho.", {"tema": "dark"})
    assert aj.extraer_marcador("sin marcador") == ("sin marcador", None)
    assert aj.extraer_marcador(None) == (None, None)


# ── 2. Los ajustes del servidor, con las funciones de Configuración ───────────────────────────────────────────

@pytest.fixture
def db_falsa(monkeypatch):
    import db_profiles
    import hydration_reminders
    visto = {}
    monkeypatch.setattr(db_profiles, "update_water_tracker_enabled", lambda uid, v: visto.setdefault("agua", (uid, v)) and True)
    monkeypatch.setattr(db_profiles, "update_long_term_memory_enabled", lambda uid, v: visto.setdefault("memoria", (uid, v)) and True)
    monkeypatch.setattr(hydration_reminders, "al_encender", lambda uid: visto.setdefault("al_encender", uid))
    return visto


def test_activa_la_hidratacion_enciende_la_tarjeta_y_no_anota_un_vaso(db_falsa):
    texto, cambio = _cambio(aj.cambiar_ajuste("u1", "hidratación", "true"))
    assert db_falsa["agua"] == ("u1", True)
    assert db_falsa["al_encender"] == "u1", "encenderla pone a cero los avisos ignorados (lote 135)"
    assert cambio == {"hidratacion": True}
    assert "no anotaste ningún vaso" in texto


def test_ocultar_la_hidratacion(db_falsa):
    _t, cambio = _cambio(aj.cambiar_ajuste("u1", "agua", "apagar"))
    assert db_falsa["agua"] == ("u1", False) and cambio == {"hidratacion": False}
    assert "al_encender" not in db_falsa


def test_memoria(db_falsa):
    _t, cambio = _cambio(aj.cambiar_ajuste("u1", "memoria", "false"))
    assert db_falsa["memoria"] == ("u1", False) and cambio == {"memoria": False}


def test_nevera_dice_lo_que_quedo(monkeypatch):
    import nevera_opcional
    monkeypatch.setattr(nevera_opcional, "interruptor_disponible", lambda: True)
    monkeypatch.setattr(nevera_opcional, "fijar_nevera", lambda uid, v: True)
    monkeypatch.setattr(nevera_opcional, "estado_nevera", lambda uid: {"activa": True})
    _t, cambio = _cambio(aj.cambiar_ajuste("u1", "nevera", "true"))
    assert cambio == {"nevera": True}
    # pidió apagarla y el modo no lo permite: se dice, y la pantalla recibe lo que QUEDÓ
    import cron_tasks
    monkeypatch.setattr(cron_tasks, "try_unfreeze_plan_for_user", lambda uid: None)
    texto, cambio = _cambio(aj.cambiar_ajuste("u1", "nevera", "false"))
    assert cambio == {"nevera": True} and "no cambió" in texto


def test_generador_apagar_y_encender(monkeypatch):
    import plan_mode
    monkeypatch.setattr(plan_mode, "pause_plan_generation", lambda uid: {"plan_mode": "tracking"})
    monkeypatch.setattr(aj, "_tiene_plan", lambda uid: True)
    _t, cambio = _cambio(aj.cambiar_ajuste("u1", "generador de planes", "false"))
    assert cambio == {"generador_de_planes": "tracking", "tenia_plan": True}

    monkeypatch.setattr(plan_mode, "get_plan_mode", lambda uid: {"plan_mode": "tracking"})
    monkeypatch.setattr(plan_mode, "resume_plan_generation", lambda uid: {"plan_mode": "plan", "plan_expired": False})
    texto, cambio = _cambio(aj.cambiar_ajuste("u1", "planes", "reanudar"))
    assert cambio["generador_de_planes"] == "plan" and "reanudados" in texto


def test_encender_los_planes_sin_plan_abre_el_formulario_y_no_reanuda(monkeypatch):
    import plan_mode
    monkeypatch.setattr(plan_mode, "get_plan_mode", lambda uid: {"plan_mode": "tracking"})
    monkeypatch.setattr(aj, "_tiene_plan", lambda uid: False)
    monkeypatch.setattr(plan_mode, "resume_plan_generation", lambda uid: pytest.fail("sin plan no hay nada que reanudar"))
    _t, cambio = _cambio(aj.cambiar_ajuste("u1", "generador_de_planes", "true"))
    assert cambio == {"pantalla": "formulario"}


def test_interruptor_operativo_apagado_no_miente(monkeypatch):
    import plan_mode
    monkeypatch.setattr(plan_mode, "pause_plan_generation", lambda uid: {"plan_mode": "plan", "skipped": "switch_off"})
    texto, cambio = _cambio(aj.cambiar_ajuste("u1", "generador_de_planes", "false"))
    assert cambio is None and "no se puede" in texto


def test_recordatorios_por_la_fusion_atomica_del_perfil(monkeypatch):
    import db
    hp = {"avisos_comida": False, "allergies": ["Maní"]}
    monkeypatch.setattr(db, "update_user_health_profile_atomic", lambda uid, f: hp.update(f(hp)) or hp)
    _t, cambio = _cambio(aj.cambiar_ajuste("u1", "recordatorios_de_comida", "true"))
    assert cambio == {"avisos_comida": True}
    assert hp == {"avisos_comida": True, "allergies": ["Maní"]}, "fusiona la clave, no pisa el perfil"


def test_un_fallo_de_la_base_no_se_confirma(monkeypatch):
    import db_profiles
    monkeypatch.setattr(db_profiles, "update_water_tracker_enabled", lambda uid, v: False)
    texto, cambio = _cambio(aj.cambiar_ajuste("u1", "hidratacion", "true"))
    assert cambio is None and texto.startswith("ERROR") and "No digas que se hizo" in texto


def test_invitado_y_valores_raros():
    assert "cuenta" in aj.cambiar_ajuste("guest", "hidratacion", "true")
    assert aj.MARCADOR not in aj.cambiar_ajuste("u1", "hidratacion", "quizás")
    assert "No reconozco" in aj.cambiar_ajuste("u1", "borrar cuenta", "true")


# ── 3. Tema, idioma y pantallas: los aplica el teléfono ───────────────────────────────────────────────────────

@pytest.mark.parametrize("valor,tema", [("oscuro", "dark"), ("modo claro", "light"), ("claro", "light"), ("sistema", "system")])
def test_tema(valor, tema):
    _t, cambio = _cambio(aj.cambiar_ajuste("guest", "tema", valor))
    assert (cambio or {}).get("tema") == tema


def test_idioma():
    _t, cambio = _cambio(aj.cambiar_ajuste("u1", "idioma", "inglés"))
    assert cambio == {"idioma": "en-US"}
    assert aj.MARCADOR not in aj.cambiar_ajuste("u1", "idioma", "klingon")


def test_pantallas():
    assert _cambio(aj.abrir_pantalla("Nevera"))[1] == {"pantalla": "nevera"}
    assert _cambio(aj.abrir_pantalla("progreso"))[1] == {"pantalla": "progreso"}
    assert _cambio(aj.abrir_pantalla("configuración", "alergias"))[1] == {"pantalla": "configuracion", "seccion": "health"}
    assert _cambio(aj.abrir_pantalla("ajustes", "capacidades"))[1] == {"pantalla": "configuracion", "seccion": "preferences"}
    assert aj.MARCADOR not in aj.abrir_pantalla("la luna")


# ── 4. El cableado: tools, prompt, estado del turno y evento done ─────────────────────────────────────────────

def test_las_tools_estan_en_agent_tools_y_delegan_en_el_modulo():
    import inspect
    import tools
    nombres = [t.name for t in tools.agent_tools]
    assert "cambiar_ajuste_de_la_app" in nombres and "abrir_pantalla_de_la_app" in nombres
    assert "ajustes_de_la_app.cambiar_ajuste(user_id, ajuste, valor)" in inspect.getsource(tools.cambiar_ajuste_de_la_app.func)
    assert "no anota un vaso" in tools.cambiar_ajuste_de_la_app.description


def test_el_prompt_lo_dice_en_los_dos_builders():
    from prompts import chat_agent
    for b in (chat_agent.build_tools_instructions("u1"), chat_agent.build_tools_instructions_stream("u1")):
        assert "cambiar_ajuste_de_la_app" in b and "abrir_pantalla_de_la_app" in b
        assert "es ENCENDER LA TARJETA, no anotar un vaso" in b


def test_execute_tools_quita_el_marcador_y_el_done_lo_entrega():
    import inspect
    import agent
    assert "ajustes_de_app" in agent.ChatState.__annotations__, "fuera del schema LangGraph descarta la clave en silencio"
    src = inspect.getsource(agent.execute_tools)
    assert "tool_result, _ajuste = extraer_marcador(tool_result)" in src
    assert '"ajustes_de_app": ajustes_de_app or None,' in src
    full = inspect.getsource(agent)
    assert full.count('"ajustes_de_app": None,') >= 2, "los DOS inputs lo reinician por turno"
    assert "'ajustes_de_app': ajustes_de_app})}" in full


def test_la_tabla_de_tools_documenta_las_dos():
    import os
    doc = open(os.path.join(os.path.dirname(aj.__file__), "docs", "agent_tools_user_id_table.md"), encoding="utf-8").read()
    assert "`cambiar_ajuste_de_la_app`" in doc and "`abrir_pantalla_de_la_app`" in doc
