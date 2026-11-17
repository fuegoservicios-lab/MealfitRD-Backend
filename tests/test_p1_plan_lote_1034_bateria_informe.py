"""[P1-PLAN-LOTE-1034 · 2026-11-17] Lo que faltaba del lote 1034 del plan: la regla R6 en el manual, los casos I1/I2
de la batería y que el arnés compile (un stub roto no lo veía ningún test: el script no lo importa nadie)."""
import json
import py_compile
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
BAT = BACKEND / "scripts" / "coach_battery"


def test_los_scripts_de_la_bateria_compilan():
    for f in sorted(BAT.glob("*.py")):
        py_compile.compile(str(f), doraise=True)


def test_r6_en_el_manual():
    src = (BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    assert "R6. INFORME PARA SU MÉDICO [P1-PLAN-LOTE-1034]" in src
    assert "generar_informe_para_medico" in src and "ni diagnostiques" in src


def test_casos_i1_i2_y_su_preset():
    casos = {c["id"]: c for c in json.loads((BAT / "battery.json").read_text(encoding="utf-8"))["casos"]}
    assert casos["I1"]["expect"]["tools"] == ["generar_informe_para_medico"]
    assert "informe" in json.dumps(casos["I2"]["expect"]["assert_spec"], ensure_ascii=False)
    from scripts.coach_battery import evaluador
    assert evaluador.INFORME_PRESET_IDS == ("I1", "I2")


def test_enfermedad_abre_la_herramienta_del_informe():
    src = (BACKEND / "agent.py").read_text(encoding="utf-8")
    m = re.search(r'needed\.add\("generar_informe_para_medico"\)', src)
    assert m
    linea = src[:m.start()].rsplit("re.search(", 1)[1]
    assert "enfermedad" in linea


def test_el_stub_de_progreso_cuadra_con_el_bloque_fijo():
    src = (BAT / "run_battery.py").read_text(encoding="utf-8")
    stub = src[src.index("def _stub_consultar_progreso"):src.index("_STUBS = {")]
    assert "Mejor semana: S9 (3/7, 1★)" in stub and "S10 (2026-11-15): 2/7 en meta" in stub
