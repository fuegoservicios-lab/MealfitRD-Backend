"""[P1-PLAN-LOTE-1086 · 2026-11-18] La respuesta del coach salía dos veces (pre-herramienta y post-herramienta casi
iguales, con « oqo?» entre medias) y así se guardaba."""
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from turno_casi_repetido import parecido, quitar_casi_repetidos, quitar_repeticion_entera

ANTES = ("Anoté tu **desayuno**: pernil de cerdo al horno con **3 lonjas de queso** y **4 panecillos Hawaiian rolls** "
         "(incluido el que ya habías comido), aprox. **850 kcal** y **56 g de proteína**. Cuando almuerces, cuéntame "
         "qué comiste y lo anoto.  oqo?")
DESPUES = ("Anoté tu **desayuno**: pernil de cerdo al horno con **3 lonjas de queso** y **4 panecillos Hawaiian rolls**, "
           "incluido el que ya habías comido: aprox. **850 kcal** y **56 g de proteína**. Cuando almuerces, cuéntame qué "
           "comiste y lo anoto.")


def test_el_caso_del_18_nov_sale_una_vez():
    assert parecido(ANTES, DESPUES) >= 0.6
    assert quitar_casi_repetidos([ANTES, DESPUES]) == [DESPUES]


def test_la_narracion_corta_y_lo_distinto_se_quedan():
    narr = "Lo anoto y te digo cómo vas."
    assert quitar_casi_repetidos([narr, DESPUES]) == [narr, DESPUES]
    otra = ("Te faltan unos 60 g de proteína para tu meta de hoy; en la cena, pollo o pescado con ensalada te lo "
            "resuelve sin pasarte de calorías.")
    assert quitar_casi_repetidos([DESPUES, otra]) == [DESPUES, otra]
    assert quitar_casi_repetidos([DESPUES]) == [DESPUES]
    assert quitar_casi_repetidos([]) == []
    # un bloque mucho más corto que el siguiente no es «el mismo» aunque comparta palabras
    assert quitar_casi_repetidos(["Anoté tu desayuno: pernil con queso y panecillos, 850 kcal y 56 g.", DESPUES + " " + otra])[0].startswith("Anoté")


def test_el_ultimo_nunca_se_quita():
    assert quitar_casi_repetidos([DESPUES, ANTES]) == [ANTES]


def test_la_misma_respuesta_repetida_dentro_de_un_texto_sale_una_vez():
    # lo guardado el 18-nov: UNA respuesta del modelo con las dos copias separadas por un salto de línea
    assert quitar_repeticion_entera(ANTES + "\n" + DESPUES) == DESPUES
    # una respuesta normal de dos párrafos distintos no se toca
    normal = DESPUES + "\n" + ("Te faltan unos 60 g de proteína para tu meta de hoy; en la cena, pollo o pescado con "
                               "ensalada te lo resuelve sin pasarte de calorías.")
    assert quitar_repeticion_entera(normal) == normal
    assert quitar_repeticion_entera("Hola") == "Hola" and quitar_repeticion_entera("") == ""


def test_engancha_en_la_reconstruccion_del_turno():
    import agent
    msgs = [
        HumanMessage(content="La carne es pernil…"),
        AIMessage(content=ANTES, tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "c1"}]),
        ToolMessage(content="ok", tool_call_id="c1"),
        AIMessage(content=DESPUES),
    ]
    out = agent._build_final_content_from_messages(msgs)
    assert out.count("Anoté tu") == 1 and "oqo" not in out
    una = [HumanMessage(content="La carne es pernil…"), AIMessage(content=ANTES + "\n" + DESPUES)]
    assert agent._build_final_content_from_messages(una) == DESPUES
    src = (Path(__file__).resolve().parents[1] / "agent.py").read_text(encoding="utf-8")
    assert '__import__("turno_casi_repetido").quitar_casi_repetidos(parts)' in src
    assert '__import__("turno_casi_repetido").quitar_repeticion_entera(' in src
