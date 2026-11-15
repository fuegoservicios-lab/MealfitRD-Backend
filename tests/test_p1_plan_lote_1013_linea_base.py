"""[P1-PLAN-LOTE-1013 · A-02] Línea base de calidad del coach.

La corrida real del 14-nov (19 casos cortos, `gpt-6-luna`): 14/19, 74 llamadas, US$ 0,0457, caché 66 %, p50 4,3 s.
Fallan M1, D2, F7, J5, J6. F7 y N1 eran un falso fallo del arnés (`value` vs `new_value`); el resto son del coach (ver la memoria
del lote). Aquí se fija lo que se arregló: el arnés y el corte de la fuga de razonamiento.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_BACKEND = Path(__file__).resolve().parent.parent
_BAT = _BACKEND / "scripts" / "coach_battery"

M1_REAL = (
    "Anoté tu **cena**: puré de plátano maduro con pollo desmenuzado, aprox. **520 kcal** y **32 g de proteína**. "
    "Hoy llevas **520 kcal** y **32 g de proteína** registradas. [UI_ACTION: REFRESH_INVENTORY] \n<lemma> \n\n---\n\n"
    "We must reconcile diary context said earlier 1210 but tool says total 520 only. Tool explicitly total real today "
    "already 520, despite context discrepancy; use result. Need remove.\nAnoté tu **cena**: puré de plátano maduro con "
    "pollo desmenuzado, aprox. **520 kcal** y **32 g de proteína**. [UI_ACTION: REFRESH_INVENTORY]"
)


def test_se_corta_la_fuga_real_del_caso_m1():
    from fuga_de_razonamiento import cortar_razonamiento_filtrado as cortar

    out = cortar(M1_REAL)
    assert out.startswith("Anoté tu **cena**") and "lemma" not in out and "We must" not in out and "reconcile" not in out
    assert out.count("Anoté tu") == 1


def test_el_corte_por_raya_sin_etiqueta():
    from fuga_de_razonamiento import cortar_razonamiento_filtrado as cortar

    t = "Listo, anoté tu desayuno.\n\n---\n\nLet's double check the totals before answering."
    assert cortar(t) == "Listo, anoté tu desayuno."


def test_no_toca_respuestas_legitimas():
    from fuga_de_razonamiento import cortar_razonamiento_filtrado as cortar

    legitimos = [
        "Hoy llevas 1.200 kcal.\n\n---\n\nTe faltan 40 g de proteína; una pechuga lo cubre.",   # raya + español
        "Pide un **Chicken Wrap**.\n\n---\n\nOkay no, mejor una ensalada.",                      # «Okay» sin coma inglesa tras la raya
        "El plato se llama «We Must Eat Well» en la carta.",                                      # texto inglés suelto
        "Usa *lemma* de limón.",                                                                  # la palabra sola no es una etiqueta
        "",
        None,
        "Respuesta normal sin nada raro.",
    ]
    for t in legitimos:
        assert cortar(t) == t, t


def test_si_lo_anterior_al_corte_esta_vacio_se_devuelve_intacto():
    from fuga_de_razonamiento import cortar_razonamiento_filtrado as cortar

    t = "<lemma>\nTexto que es toda la respuesta"
    assert cortar(t) == t


def test_el_sanitizador_del_wire_aplica_el_corte():
    import agent

    out = agent._sanitize_chat_output_for_wire(M1_REAL)
    assert "lemma" not in out and "We must" not in out and "[UI_ACTION: REFRESH_INVENTORY]" in out


# ── arnés ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _casos():
    return {c["id"]: c for c in json.loads((_BAT / "battery.json").read_text(encoding="utf-8"))["casos"]}


def test_los_casos_nuevos_estan_y_el_preset_corto_sigue_en_19():
    import importlib

    ev = importlib.import_module("evaluador") if str(_BAT) in sys.path else None
    if ev is None:
        sys.path.insert(0, str(_BAT))
        ev = importlib.import_module("evaluador")
    c = _casos()
    for i in ("N1", "N2", "N3", "N4"):
        assert i in c
    assert len(ev.SHORT_PRESET_IDS) == 19


def test_f7_y_n1_buscan_en_new_value_que_es_el_argumento_real():
    c = _casos()
    for i in ("F7", "N1"):
        checks = c[i]["expect"]["assert_spec"]["args_checks"]
        campos = [f for chk in checks for t in chk.get("text_all", []) for f in t["fields"]]
        assert campos == ["new_value"], (i, campos)
    # el argumento de verdad de la herramienta
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "def update_form_field(user_id: str, field: str, new_value: str)" in src


def test_el_evaluador_da_por_buenos_f7_y_n1_con_lo_que_hizo_el_coach():
    sys.path.insert(0, str(_BAT))
    import evaluador as ev

    c = _casos()
    t_f7 = [{"tools": [{"tool": "update_form_field", "args": {"field": "allergies", "new_value": "Leche"}, "result": "ok"}],
             "final": "Tendré en cuenta tu alergia a la leche.", "errors": [], "contaminated": []}]
    r = ev.evaluar_caso(c["F7"], t_f7, t_f7[0]["final"])
    assert r["passed"], r["failures"]
    t_n1 = [{"tools": [{"tool": "update_form_field", "args": {"field": "allergies", "new_value": "Intolerancia a la lactosa"}, "result": "ok"}],
             "final": "Guardé tu intolerancia a la lactosa.", "errors": [], "contaminated": []}]
    r = ev.evaluar_caso(c["N1"], t_n1, t_n1[0]["final"])
    assert r["passed"], r["failures"]


def test_el_arnes_no_anota_en_el_ledger_una_corrida_sin_llamadas():
    src = (_BAT / "run_battery.py").read_text(encoding="utf-8")
    assert 'COST["calls"] <= 0' in src and "NO se anota en el ledger" in src
    assert "default=str" in src   # `results.json` con FoodLabel
    assert "OPENAI_API_KEY falta o es un marcador" in src


def test_el_stub_de_ajustes_usa_las_funciones_puras_y_no_toca_la_base():
    src = (_BAT / "run_battery.py").read_text(encoding="utf-8")
    assert '"cambiar_ajuste_de_la_app": _stub_cambiar_ajuste' in src
    i = src.index("def _stub_cambiar_ajuste")
    cuerpo = src[i:src.index("def _stub_update_form_field")]
    assert "execute_sql" not in cuerpo and "update_user" not in cuerpo


def test_existe_la_linea_base_en_el_ledger_y_el_evaluador_la_reconoce():
    sys.path.insert(0, str(_BAT))
    import evaluador as ev

    entradas = ev.load_ledger(_BAT / "ledger.jsonl")
    base = ev.latest_baseline_entry(entradas, preset="short", exclude_label="otra")
    assert base is not None and base["label"] == "base_20261114"
    assert base["model"] == "gpt-6-luna" and base["total"] == 19 and base["passed"] == 15
    assert base["calls"] > 0 and base["cost_usd"] > 0          # nunca una fila de una corrida que no llamó al modelo
    assert set(base["failed_ids"]) == {"M1", "D2", "J5", "J6"}
