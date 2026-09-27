# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-556 · 2026-09-27] «Cambiar plato»: los tres cerradores de proteína ven dieta y rechazos.

Auditoría del formulario: en `agent.swap_meal` el cerrador del reintento y el del tope de porción elegían sin `diet=` y
con sólo `allergies` (el del tope corre después del último chequeo: «No me gusta: atún» → «atún en agua» guardado), y
la inspiración leía `dietType`/`diet` cuando el router manda `diet_type`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")


def _swap():
    m = re.search(r"\ndef swap_meal\(", _AGENT)
    nxt = re.search(r"\ndef ", _AGENT[m.start() + 1:])
    return _AGENT[m.start():m.start() + 1 + nxt.start()]


def test_los_pools_de_los_tres_cerradores_llevan_rechazos_y_dieta():
    b = _swap()
    assert "_restr556 = __import__(\"constants\").alergias_y_rechazos(form_data)" in b
    assert "_pc_pool(_pc_allergies, _tu_db_holder[0], country=_swap_country, diet=diet_type)" in b
    assert "_safe_high_density_proteins(_restr556, _cl_db, country=_swap_country, diet=diet_type)" in b
    assert "_safe_pc(_restr556, _cap_db, min_protein=18.0," in b
    assert "_safe_pc(allergies" not in b and "_safe_high_density_proteins(allergies" not in b


def test_los_cierres_tambien():
    b = _swap()
    assert b.count("allergies=_restr556") >= 2
    assert b.count("diet=diet_type") >= 6


def test_la_inspiracion_lee_la_clave_del_router():
    assert 'form_data.get("dietType") or form_data.get("diet") or form_data.get("diet_type")' in _swap()


def test_rechazos_en_el_pool():
    from constants import alergias_y_rechazos
    assert alergias_y_rechazos({"allergies": ["Maní"], "dislikes": ["Atún"]}) == ["Maní", "Atún"]
