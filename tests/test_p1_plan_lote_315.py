# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-315 · 2026-09-25] Lo que un paso ya mete en la preparación no se sirve «al lado».

Plan real del perfil del dueño (25-sep), «Batido frío de guanábana con leche, almendras y queso cottage»: «💪 Agrega
queso cottage a la licuadora y licúa hasta integrar» y, en el Montaje, «… Acompaña con queso cottage». Dos destinos para
los mismos 40 g. 6 de 354 comidas de las baterías, casi todas batidos."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go
import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_batido_no_sirve_al_lado_lo_que_ya_licuo():
    pasos = ["Mise en place: retira las semillas de 215 g de guanábana y mide 100 g de leche.",
             "💪 Agrega queso cottage a la licuadora y licúa hasta integrar.",
             "Montaje: licúa la guanábana, la leche y el hielo hasta obtener un batido homogéneo; sirve frío."]
    antes = list(pasos)
    assert go._mention_cooked_complement_in_montaje(pasos, ["queso cottage"]) is False
    assert pasos == antes, pasos


def test_lo_preparado_aparte_se_sigue_mencionando():
    pasos = ["El Toque de Fuego: cocina el huevo en una sartén 3 min.", "💪 Calienta el queso cottage aparte 1 min.",
             "Montaje: sirve el huevo."]
    assert go._mention_cooked_complement_in_montaje(pasos, ["queso cottage"]) is True
    assert pasos[-1].endswith("Acompaña con queso cottage."), pasos[-1]


def test_la_regla_sola():
    assert pc.ya_va_dentro("agrega el queso cottage a la licuadora y licua", "queso cottage") is True
    assert pc.ya_va_dentro("licua la guanabana, la leche y el queso cottage", "queso cottage") is True
    assert pc.ya_va_dentro("incorpora 40 g de queso a la masa", "queso") is True
    assert pc.ya_va_dentro("sirve el queso cottage en un plato aparte", "queso cottage") is False
    assert pc.ya_va_dentro(None, None) is False


def test_ancla():
    assert "ya_va_dentro(_otros, _tok):  # [P1-PLAN-LOTE-315]" in (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
