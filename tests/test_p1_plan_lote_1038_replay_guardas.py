"""[P1-PLAN-LOTE-1038 · A-15 · 2026-11-16] Cada guarda por patrones del coach tiene su tasa medida sobre mensajes reales.

El replay (`scripts/coach_battery/replay_guardas.py`) corre contra la base; aquí se fija su VEREDICTO (recuentos en
`replay_tasas.json`, sin texto de usuarios) y las clases de fallo que encontró, con frases sintéticas equivalentes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_DIR = _BACKEND / "scripts" / "coach_battery"

_spec = importlib.util.spec_from_file_location("replay_guardas", _DIR / "replay_guardas.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

TASAS = json.loads((_DIR / "replay_tasas.json").read_text(encoding="utf-8"))


def test_toda_guarda_del_replay_tiene_su_tasa_medida():
    """Añadir una guarda al replay sin medirla (o medirla sin anotarla) rompe aquí."""
    assert set(replay.guardas()) == set(TASAS["guardas"])
    assert TASAS["mensajes"] >= 400 and TASAS["fecha"] >= "2026-11-16"
    for nombre, g in TASAS["guardas"].items():
        assert g["aciertos"] + g["falsos_positivos"] == g["disparos"], f"{nombre}: quedan disparos sin etiquetar"
        assert g["nota"].strip()


def test_ninguna_guarda_queda_con_falsos_positivos_ni_escapes_conocidos():
    for nombre, g in TASAS["guardas"].items():
        assert g["falsos_positivos"] == 0, f"{nombre}: el replay encontró falsos positivos y no se arreglaron"
        assert g["escapes_conocidos"] == 0, f"{nombre}: hay un escape conocido sin arreglar"


def test_el_corpus_no_se_versiona():
    seguidos = {p.name for p in _DIR.iterdir() if p.is_file()}
    assert "replay_tasas.json" in seguidos
    crudo = (_DIR / "replay_tasas.json").read_text(encoding="utf-8")
    assert '"mensajes_texto"' not in crudo and len(crudo) < 6000, "el veredicto son recuentos, no mensajes"


# ───────────────────────────────────────────────────────────── las clases de fallo que encontró el replay
@pytest.mark.parametrize("texto", [
    "1 taza · A la plancha o al horno",
    "Plátano maduro · Al horno / a la plancha",
    "me comi un plantano grande maduro y dos huevos",
    "le eché una planta de cilantro",
])
def test_plancha_y_platano_no_son_un_turno_dificil(texto):
    g = replay.guardas()
    assert g["intencion_dificil"](texto) is False


@pytest.mark.parametrize("texto", [
    "quiero cambiar mi plan porque soy diabético",
    "¿mis planes sirven con hipertensión?",
    "¿puedo comer mariscos si soy alérgico?",
    "necesito sustituir el arroz, tengo enfermedad renal",
])
def test_lo_que_si_es_intencion_dificil_sigue_siendolo(texto):
    assert replay.guardas()["intencion_dificil"](texto) is True


@pytest.mark.parametrize("texto", [
    "Ay no, por favor, yo no me comí eso, bórralo",
    "oye, creo que te equivocaste con las calorías",
    "mira que lo anotaste mal, eran dos",
    "Te equivocaste",
    "no, eso no fue lo que comí",
])
def test_la_queja_se_ve_aunque_no_abra_el_mensaje(texto):
    assert replay.guardas()["queja"](texto) is True


@pytest.mark.parametrize("texto", [
    "Elimina las yemas ya que me comí los huevos sin yemas",      # corrige SU registro: no es una queja del coach
    "Quita el pan, al final no me lo comí",
    "agrega el desayuno que borré sin querer",
    "Oh vale, deja la que está, me confundí",
    "me comí 2 huevos y un plátano",
])
def test_corregir_el_propio_registro_no_es_una_queja(texto):
    assert replay.guardas()["queja"](texto) is False


def test_la_queja_solo_se_busca_al_empezar_el_mensaje():
    largo = "Hoy desayuné mangú con huevos y salami, almorcé arroz con pollo y ensalada. " * 3
    assert len(largo) > 200
    assert replay.guardas()["queja"](largo + "te equivocaste ayer") is False


# ───────────────────────────────────────────────────────────── el script
def test_anonimiza_antes_de_enseñar_un_mensaje():
    t = replay.anonimizar("soy ana@correo.com,  mi id 61a13831-1111-2222-3333-444455556666 y mi cel +1 (809) 555-1234\nme comí 2 huevos")
    assert t == "soy <correo>, mi id <id> y mi cel <tel> me comí 2 huevos"
    assert len(replay.anonimizar("x" * 500)) == 180
    assert replay.anonimizar("me comí 250 g de pollo y 2 huevos") == "me comí 250 g de pollo y 2 huevos", "las cantidades no son teléfonos"


def test_medir_cuenta_disparos_y_no_se_cae_si_una_guarda_falla():
    def rota(t):
        raise ValueError("x")
    r = replay.medir(["a", "bb", "ccc"], {"larga": lambda t: len(t) >= 2, "rota": rota})
    assert r == {"disparos": {"larga": [1, 2], "rota": []}, "errores": {"larga": 0, "rota": 3}}


def test_el_script_solo_lee():
    fuente = (_DIR / "replay_guardas.py").read_text(encoding="utf-8").upper()
    for verbo in ("INSERT ", "UPDATE ", "DELETE ", "OPENAI", "GEMINI"):
        assert verbo not in fuente, f"el replay no escribe ni llama a modelos ({verbo.strip()})"
