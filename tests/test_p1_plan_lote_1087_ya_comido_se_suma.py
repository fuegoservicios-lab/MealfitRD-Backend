"""[P1-PLAN-LOTE-1087 · 2026-11-18] «con 4 panecillos… y ya me había comido uno» + foto con 4: lo ya comido NO se ve en
la foto y se suma (5), salvo que diga que está incluido."""
from prompts.chat_agent import _DUDAS_INSTRUCCION


def test_la_regla_va_en_las_instrucciones_de_la_foto():
    assert "LO QUE YA SE COMIÓ ANTES DE LA FOTO SE SUMA [P1-PLAN-LOTE-1087]" in _DUDAS_INSTRUCCION
    assert "4 panecillos en la foto + «ya me había comido uno» = 5 panecillos" in _DUDAS_INSTRUCCION
    assert "«eran 4 y ya me comí uno» = 4 en total" in _DUDAS_INSTRUCCION


def test_la_instruccion_de_dudas_llega_a_los_turnos_con_foto():
    import prompts.chat_agent as pa
    src = open(pa.__file__, encoding="utf-8").read()
    # la constante se usa (no es texto muerto)
    assert src.count("_DUDAS_INSTRUCCION") >= 2
