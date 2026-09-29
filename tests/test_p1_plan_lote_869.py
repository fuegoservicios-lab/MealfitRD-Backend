# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-869 · 2026-09-29] Una proteína que sólo nombra la NOTA de seguridad no está servida: recibe su paso.

Batería rdv636 (estudiante económico, cena D3) pasada por el escudo actual: «Canoas crujientes de plátano verde… con
edamame y pechuga de pollo» — «60 g de pechuga de pollo» en la lista y NINGÚN paso la cocina ni la sirve; sólo la nota
«⚠️ Seguridad alimentaria: el pollo/cerdo debe cocinarse por completo…». `ensure_protein_step_parity` (P0-2) se daba por
cumplido porque su blob incluía las notas. Corpus: 19 de 8.550 comidas (pescado, pollo, claras de los «Huevos y Avena»).
"""
from __future__ import annotations

import graph_orchestrator as go


class _Db:
    def macros_from_ingredient_string(self, line):
        s = str(line).lower()
        if "pechuga de pollo" in s:
            return {"name": "Pechuga de pollo", "protein": 18.6}
        if "platano" in s or "plátano" in s:
            return {"name": "Plátano verde", "protein": 1.0}
        return None


def _plan():
    return {"days": [{"day": 3, "meals": [{
        "meal": "Cena",
        "name": "Canoas crujientes de plátano verde rellenas de queso fresco con edamame y pechuga de pollo",
        "ingredients": ["½ plátano verde mediano", "60 g de pechuga de pollo"],
        "recipe": ["Mise en place: pela ½ plátano verde y córtalo a lo largo en dos mitades.",
                   "El Toque de Fuego: pincela las mitades de plátano con aceite y cocínalas en la airfryer 15-18 min.",
                   "Montaje: sirve las canoas de plátano verde recién hechas.",
                   "⚠️ Seguridad alimentaria: el pollo/cerdo debe cocinarse por completo (interior sin partes rosadas, "
                   "~74°C)."]}]}]}


def test_el_huevo_de_la_nota_lo_repone_el_863_no_la_paridad(monkeypatch):
    """«Huevos y Avena» (plan degradado): 3 huevos y 60 g de claras sólo en la nota. Un paso de paridad para las claras
    («Cocina clara de huevo…») haría que el 863 ya no pusiera el que cuaja huevos Y claras."""
    monkeypatch.setattr(go, "_ingredient_is_protein_dominant", lambda line, db: "huevo" in str(line).lower())

    class _DbHuevo:
        def macros_from_ingredient_string(self, line):
            return {"name": "Clara de huevo" if "clara" in str(line).lower() else "Huevo", "protein": 10.0}
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Huevos y Avena", "ingredients": ["3 huevos", "60 g de clara de huevo", "30 g de avena"],
        "recipe": ["Mise en place: pesa cada ingrediente.", "El Toque de Fuego: calienta la avena cocida 2-3 minutos.",
                   "Montaje: sirve en bowl.", go._FOOD_SAFETY_NOTE_NOCOOK]}]}]}
    assert go.ensure_protein_step_parity(plan, db=_DbHuevo()) == 0


def test_la_pechuga_que_solo_nombra_la_nota_recibe_su_paso(monkeypatch):
    monkeypatch.setattr(go, "_ingredient_is_protein_dominant", lambda line, db: "pechuga" in str(line).lower())
    plan = _plan()
    assert go.ensure_protein_step_parity(plan, db=_Db()) == 1
    rec = plan["days"][0]["meals"][0]["recipe"]
    assert any("pechuga de pollo" in p.lower() and not p.startswith("⚠") for p in rec), rec
    assert go.ensure_protein_step_parity(plan, db=_Db()) == 0, "idempotente"
