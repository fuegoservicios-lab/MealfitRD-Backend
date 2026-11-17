"""[P1-PLAN-LOTE-1064 · 2026-11-17] El botón se llama «Cambiar plato» (frase normal, decisión de diseño del dueño).

Los avisos del plan, el prompt del coach y los errores del servidor lo nombraban «Cambiar Plato»; la glosa del
frontend (`ReviewIssuesClaro`) empareja el texto del backend palabra por palabra, así que los dos lados cambian a la vez.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MODULOS = ("agent.py", "graph_orchestrator.py", "prompts/chat_agent.py", "routers/plans.py", "constants.py", "cron_tasks.py")


def test_ningun_texto_del_servidor_dice_cambiar_plato_con_mayuscula():
    malos = [m for m in MODULOS if "Cambiar Plato" in (RAIZ / m).read_text(encoding="utf-8")]
    assert malos == []


def test_los_avisos_siguen_nombrando_el_boton():
    texto = (RAIZ / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "usa «Cambiar plato» si quieres más variedad" in texto
