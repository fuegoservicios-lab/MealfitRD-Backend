"""[P2-COACH-COUNTRY · 2026-08-21] El coach le hablaba de mangú a un español.

Recibe siempre la misma `<biblioteca_culinaria_local>`: seis platos dominicanos con sus tiempos de
digestión (Mofongo 5-6 h, Yaroa 6+ h, La Bandera 4,5 h…). Y el prompt no los deja como contexto
opcional — le da una **ORDEN**: «TIENES LA ORDEN de citar explícitamente sus horas estimadas de
digestión» al ver uno de esos platos. Un español leyendo «toma 5 horas digerir ese Mofongo» recibe
una reprimenda por algo que no comió, y el mecanismo del producto que regaña CON FUNDAMENTO queda
inerte para 5 de los 6 países.

MITAD DEL GAP ESTABA MAL DIAGNOSTICADA, y lo descubrí escribiendo este fichero. La auditoría citaba
la persona: «CULTURA GASTRONÓMICA DOMINICANA». Medido: esa línea vive SÓLO en
`CHAT_SYSTEM_PROMPT_BASE` y `CHAT_STREAM_SYSTEM_PROMPT_BASE`, que `agent.py` **importa y no usa**.
Los tres prompts que sí llegan al modelo tienen **cero** marcas dominicanas. Mi primera versión de
este test falló por eso — estaba leyendo la constante equivocada, igual que el audit.

Como el helper de persona que había escrito se quedaba sin call site, lo borré: añadir código
muerto para cerrar un gap sería la ironía exacta de esta ola, donde el defecto de fondo se ha
repetido tres veces como «la función existe, es correcta y nadie la llama».

QUÉ NO SE HACE: no se inventan tiempos de digestión para la paella, el pozole ni la bandeja paisa.
Un tiempo de digestión es una AFIRMACIÓN CLÍNICA y fabricarla de memoria es lo que costó la
auditoría de procedencia del catálogo. En beta el coach conserva el PRINCIPIO y pierde el catálogo
ajeno: menos capacidad de la ideal, y honesta. Curarlo con fuente es trabajo de contenido, hermano
de P1-BETA-FRAGMENT-DEPTH.

IDIOMA ≠ PAÍS: `locale` mueve la PROSA del coach (F2-T3) y `country` mueve la COCINA. Un dominicano
puede leer la app en inglés y seguir comiendo mangú.

Cubre:
  A. Byte-identidad dominicana.
  B. La persona: lo que el audit señalaba era prosa MUERTA (se ancla para que siga muerta).
  C. La biblioteca dominicana no viaja a beta.
  D. El principio sobrevive (no se vacía el mecanismo).
  E. Los call sites pasan el país.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


# ── A. Byte-identidad dominicana ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["DO", None, "", "basura"])
def test_la_biblioteca_dominicana_no_cambia(knob_on, cc):
    """Lo desconocido se comporta como RD, igual que `canonicalize_country`."""
    from constants import CULINARY_KNOWLEDGE_BASE, culinary_knowledge_base_for_country
    assert culinary_knowledge_base_for_country(cc) == CULINARY_KNOWLEDGE_BASE


def test_con_el_knob_apagado_todo_el_mundo_recibe_la_dominicana(monkeypatch):
    """Contrato de rollback del sistema de países."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    from constants import CULINARY_KNOWLEDGE_BASE, culinary_knowledge_base_for_country
    assert culinary_knowledge_base_for_country("ES") == CULINARY_KNOWLEDGE_BASE


# ── B. La persona: el audit señalaba PROSA MUERTA ──────────────────────────────────────────────

def test_los_prompts_VIVOS_del_coach_no_llevan_marca_dominicana():
    """La auditoría citaba «CULTURA GASTRONÓMICA DOMINICANA» en la persona del coach. Medido: esa
    línea vive SÓLO en `CHAT_SYSTEM_PROMPT_BASE` y `CHAT_STREAM_SYSTEM_PROMPT_BASE`, que `agent.py`
    **importa y no usa**. Los tres prompts que sí llegan al modelo —AGENT_INLINE, STREAM_INLINE y
    VOICE— tienen CERO marcas dominicanas.

    O sea que esa mitad del gap estaba ya cerrada y nadie lo sabía; lo que seguía vivo era la
    BIBLIOTECA (abajo). Se ancla la propiedad para que un futuro añadido a la persona falle aquí
    en vez de llegar a un español, y para que nadie vuelva a diagnosticar el defecto leyendo la
    constante equivocada — como me pasó a mí al escribir la primera versión de este fichero."""
    import re
    from prompts.chat_agent import (CHAT_AGENT_INLINE_PROMPT, CHAT_STREAM_INLINE_PROMPT,
                                    CHAT_VOICE_MODE_PROMPT)
    for nombre, prompt in (("CHAT_AGENT_INLINE_PROMPT", CHAT_AGENT_INLINE_PROMPT),
                           ("CHAT_STREAM_INLINE_PROMPT", CHAT_STREAM_INLINE_PROMPT),
                           ("CHAT_VOICE_MODE_PROMPT", CHAT_VOICE_MODE_PROMPT)):
        hits = re.findall(r"[Dd]ominican\w*|[Mm]ofongo|[Mm]ang[uú]|[Yy]aroa", prompt)
        assert not hits, f"{nombre} volvió a nombrar lo dominicano en la persona: {sorted(set(hits))}"


def test_las_dos_constantes_muertas_siguen_sin_call_site():
    """Corolario del anterior, y la razón por la que este P-fix NO las tocó: si algún día alguien
    las cablea, hereda la persona dominicana y este test lo obliga a mirarlo."""
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent.joinpath("agent.py").read_text(
        encoding="utf-8", errors="replace")
    for muerta in ("CHAT_SYSTEM_PROMPT_BASE", "CHAT_STREAM_SYSTEM_PROMPT_BASE"):
        assert src.count(muerta) <= 1, (
            f"{muerta} pasó a tener call site: lleva «CULTURA GASTRONÓMICA DOMINICANA» dentro"
        )


# ── C. La biblioteca ajena no viaja ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
@pytest.mark.parametrize("plato", ["Mofongo", "Mangú", "Yaroa", "Sancocho", "Pica Pollo"])
def test_los_platos_dominicanos_no_se_le_citan_a_un_usuario_beta(knob_on, cc, plato):
    """El prompt no ofrece estos platos como contexto: ORDENA citarlos. Un español leyendo «toma 5
    horas digerir ese Mofongo» recibe una reprimenda por algo que no comió."""
    from constants import culinary_knowledge_base_for_country, strip_accents
    r = strip_accents(culinary_knowledge_base_for_country(cc).lower())
    assert strip_accents(plato.lower()) not in r


# ── D. El principio sobrevive ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cc", ["ES", "MX", "CO"])
def test_el_mecanismo_no_se_vacia(knob_on, cc):
    """Vaciar la biblioteca sería peor que dejarla mal: el coach perdería la capacidad de razonar
    sobre digestión, que es la parte del producto que regaña con fundamento. Se conserva el
    PRINCIPIO sin inventar cifras para platos que nadie ha medido."""
    from constants import culinary_knowledge_base_for_country
    r = culinary_knowledge_base_for_country(cc)
    assert r.strip(), f"{cc}: biblioteca vacía"
    assert "<biblioteca_culinaria_local>" in r, "el prompt referencia esa etiqueta por su nombre"
    assert "digest" in r.lower(), "se perdió el concepto de digestión, que es el mecanismo entero"


@pytest.mark.parametrize("cc", ["ES", "MX", "CO"])
def test_no_se_inventan_tiempos_de_digestion_para_platos_locales(knob_on, cc):
    """La otra mitad del criterio: un tiempo de digestión es una afirmación clínica. Fabricar
    «Paella: 4 horas» de memoria es la clase que costó la auditoría de procedencia del catálogo."""
    import re
    from constants import culinary_knowledge_base_for_country
    r = culinary_knowledge_base_for_country(cc)
    assert not re.search(r"\b(paella|pozole|bandeja paisa|paella|paelha)\b", r, re.I)


# ── E. Los call sites ───────────────────────────────────────────────────────────────────────────

def test_el_agente_pide_la_biblioteca_por_pais():
    """El modo de fallo repetido de esta ola: la función existe, es correcta y nadie la llama."""
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent.joinpath("agent.py").read_text(
        encoding="utf-8", errors="replace")
    directas = src.count("{CULINARY_KNOWLEDGE_BASE}")
    porpais = src.count("culinary_knowledge_base_for_country")
    assert porpais >= 4, f"solo {porpais} call sites por país (esperados 4)"
    assert directas == 0, (
        f"quedan {directas} inyecciones directas de la biblioteca dominicana sin pasar por el país"
    )
