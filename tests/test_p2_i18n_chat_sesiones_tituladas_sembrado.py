"""[P2-I18N-CHAT-SESIONES-TITULADAS-POR-MENSAJE-SEMBRADO · 2026-08-23] Tras generar el
plan, ``routers/plans.py`` siembra en la sesión de chat un mensaje firmado como del
USUARIO («Generar plan para mi objetivo: lose_fat») y la respuesta del coach, los dos en
español fijo y con el CÓDIGO del objetivo pegado. Medido: 90 de 106 sesiones se titulan con
ese mensaje — es lo primero que el usuario ve de su historial, y en francés salía en
español con un identificador en inglés.

Cierre: ``prompts.chat_agent.plan_seed_messages(locale, goal_code)`` (SSOT por idioma, el
objetivo se GLOSA al escribir la prosa, el código sigue siendo el identificador del motor)
y el call site resuelve el locale del perfil, best-effort, guest ⇒ español.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = _BACKEND / "routers" / "plans.py"
_MARKER = "P2-I18N-CHAT-SESIONES-TITULADAS-POR-MENSAJE-SEMBRADO"


@pytest.fixture(scope="module")
def seed():
    from prompts.chat_agent import plan_seed_messages
    return plan_seed_messages


@pytest.mark.parametrize("locale,goal_esperado,coach_fragmento", [
    ("en-US", "lose fat", "Here is your"),
    ("pt-BR", "perder gordura", "Aqui está"),
    ("fr-FR", "perdre du gras", "Voici"),
    ("it-IT", "perdere grasso", "Ecco"),
])
def test_el_par_sembrado_sale_en_el_idioma_del_usuario(seed, locale, goal_esperado, coach_fragmento):
    usuario, coach = seed(locale, "lose_fat")
    assert goal_esperado in usuario, f"{locale}: el objetivo no se glosó: {usuario!r}"
    assert "lose_fat" not in usuario, f"{locale}: el CÓDIGO del objetivo se coló en la prosa"
    assert "Generar plan" not in usuario, f"{locale}: cayó al español"
    assert coach_fragmento in coach, f"{locale}: la respuesta del coach no se tradujo: {coach!r}"


@pytest.mark.parametrize("goal", ["lose_fat", "gain_muscle", "maintenance", "performance"])
def test_los_cuatro_objetivos_del_formulario_tienen_glosa_en_los_cinco_idiomas(seed, goal):
    # «maintenance» y «performance» son su propia glosa en inglés (y «performance» en
    # francés), así que «el código no aparece» sería un test falso: se mide que la prosa
    # termina con la etiqueta de la TABLA de ese idioma, y que la tabla la tiene.
    from prompts.chat_agent import _PLAN_SEED_GOAL_LABELS
    for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        assert goal in _PLAN_SEED_GOAL_LABELS[loc], f"{loc} no glosa {goal}"
        usuario, _ = seed(loc, goal)
        assert usuario.endswith(_PLAN_SEED_GOAL_LABELS[loc][goal]), f"{loc}/{goal}: {usuario!r}"


@pytest.mark.parametrize("locale", ["es-DO", None, "", "xx-YY", 7])
def test_lo_desconocido_cae_al_espanol_sin_lanzar(seed, locale):
    usuario, coach = seed(locale, "gain_muscle")
    assert usuario == "Generar plan para mi objetivo: ganar músculo"
    assert coach.startswith("¡Aquí tienes tu estrategia")


def test_un_codigo_de_objetivo_desconocido_se_escribe_tal_cual(seed):
    """Mejor «lose_weight_fast» que inventar una etiqueta."""
    usuario, _ = seed("fr-FR", "lose_weight_fast")
    assert usuario.endswith("lose_weight_fast")
    usuario, _ = seed("fr-FR", None)
    assert usuario.endswith("Desconocido")


def test_el_call_site_ya_no_pasa_literales_al_save_message_sembrado():
    """AST de `routers/plans.py`: en el bloque que siembra el chat, los `save_message(…, "user",
    X)` y `(…, "model", X)` reciben NOMBRES (los que devuelve `plan_seed_messages`), no
    literales. Un literal aquí es español fijo firmado como del usuario, otra vez."""
    tree = ast.parse(_PLANS.read_text(encoding="utf-8"))
    sembrados = []
    usa_ssot = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "prompts.chat_agent":
            if any(a.name == "plan_seed_messages" for a in node.names):
                usa_ssot = True
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        nombre = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if nombre != "save_message" or len(node.args) < 3:
            continue
        rol = node.args[1]
        if isinstance(rol, ast.Constant) and rol.value in ("user", "model"):
            sembrados.append((rol.value, node.args[2], node.lineno))
    assert usa_ssot, f"routers/plans.py no importa plan_seed_messages [{_MARKER}]"
    assert sembrados, "no se encontró el save_message sembrado en routers/plans.py"
    for rol, contenido, linea in sembrados:
        assert not isinstance(contenido, (ast.Constant, ast.JoinedStr)), (
            f"routers/plans.py:{linea}: save_message(role={rol!r}) vuelve a recibir un literal "
            f"— español fijo firmado como del usuario [{_MARKER}]")


def test_el_literal_espanol_vive_solo_en_el_ssot():
    """El texto español sigue existiendo (es el suelo), pero en UN sitio."""
    from prompts import chat_agent
    assert "Generar plan para mi objetivo: {goal}" in chat_agent._PLAN_SEED_USER["es-DO"]
    assert "Generar plan para mi objetivo" not in _PLANS.read_text(encoding="utf-8").replace(
        "# ", ""), "el literal sembrado volvió a routers/plans.py"
