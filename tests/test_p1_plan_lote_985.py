"""[P1-PLAN-LOTE-985 · 2026-11-11] Con restricciones, el aviso nombra opciones ya pasadas por el backstop.

Caso vivo (cuenta de prueba con hipertensión, 5:30 p. m.): «elige algo que ya sepas que te cae bien y que sea bajo en
sal» — el lote 161 le prohibía nombrar comida y el aviso no decía nada.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import opciones_del_aviso as oa  # noqa: E402


def _veta(palabra):
    return lambda nombre, _perfil: [f"contiene {palabra}"] if palabra in nombre.lower() else []


def test_el_backstop_filtra_y_el_resto_pasa():
    ops = oa.opciones_aptas({"medicalConditions": ["Hipertensión"]}, "Merienda", violaciones=_veta("huevo"))
    assert ops and len(ops) <= oa.MAX_OPCIONES
    assert not any("huevo" in o.lower() for o in ops)


def test_lo_que_no_le_gusta_sale():
    ops = oa.opciones_aptas({}, "Merienda", no_le_gusta=["Guineos"], violaciones=lambda n, p: [])
    assert not any("guineo" in o.lower() for o in ops)


def test_alergia_al_mani_quita_frutos_secos():
    ops = oa.opciones_aptas({"allergies": ["Maní"]}, "Merienda", violaciones=lambda n, p: [])
    assert not any("nueces" in o.lower() for o in ops)
    ops = oa.opciones_aptas({"allergies": "Maní"}, "Merienda", violaciones=lambda n, p: [])   # string, no lista
    assert not any("nueces" in o.lower() for o in ops)


def test_backstop_que_falla_veta():
    def _rompe(n, p):
        raise RuntimeError("x")
    assert oa.opciones_aptas({}, "Cena", violaciones=_rompe) == []
    assert oa.bloque({}, "Cena", violaciones=_rompe) == ""


def test_franja_desconocida_y_knob(monkeypatch):
    assert oa.opciones_aptas({}, "Brunch", violaciones=lambda n, p: []) == []
    monkeypatch.setenv("MEALFIT_NUDGE_SAFE_OPTIONS", "0")
    assert oa.bloque({}, "Cena", violaciones=lambda n, p: []) == ""


def test_bloque_dice_que():
    b = oa.bloque({}, "Merienda", violaciones=lambda n, p: [])
    assert "ÚNICAS comidas que puedes nombrar" in b and "di qué" in b


def test_cableado_en_el_cron():
    src = (_BACKEND / "proactive_agent.py").read_text(encoding="utf-8")
    assert 'if _ia_ok and _ctx_aviso["restringido"] else "")' in src
    assert "prompt += _opciones_aviso   # [P1-PLAN-LOTE-985]" in src
    assert '"no_le_gusta": no_le_gusta,' in src
    i_bloque = src.index('prompt += _ctx_aviso["bloque"]')
    assert i_bloque < src.index("prompt += _opciones_aviso")   # las opciones van DESPUÉS de las restricciones


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 985
