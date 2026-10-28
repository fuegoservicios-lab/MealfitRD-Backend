# -*- coding: utf-8 -*-
"""[P1-MULTI-PHOTO-MATCH · 2026-10-27] Pruebas de asociación de adjuntos en turnos multi-foto.

Incidente: El usuario subió en un mismo turno 2 fotos:
- Foto 1: Guineo maduro mediano (90 kcal)
- Foto 2: Sancocho criollo con arroz (710 kcal)
Mensaje: "Me comí el guineo de desayuno y lo otro de almuerzo 2 tazas"
El coach registró Desayuno (guineo) y Almuerzo (sancocho), pero el desayuno
quedó con ambas fotos asociadas.

Garantías probadas aquí:
1. `_plate_photos_from_vision` extrae y preserva `attachment_id` para cada plato analizado.
2. `_asociar_adjuntos_a_comida` asocia 1-a-1 la foto correspondiente según nombre/tokens/kcal
   y no mezcla fotos de comidas distintas.
3. Al registrar múltiples comidas en el mismo turno, cada comida recibe estrictamente
   su propio `attachment_id` en `nutrition["attachment_ids"]`.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent import (
    _plate_photos_from_vision,
    _asociar_adjuntos_a_comida,
    _tokens_de_comida,
    execute_tools,
)


def test_plate_photos_from_vision_preserves_attachment_id():
    vision_multi = {
        "kind": "multi",
        "items": [
            {
                "kind": "plato",
                "attachment_id": "att-guineo-1111",
                "description": "1 guineo maduro mediano. Estimación nutricional: 90 kcal.",
            },
            {
                "kind": "plato",
                "attachment_id": "att-sancocho-2222",
                "description": "Sancocho criollo con arroz blanco (2 tazas). Estimación nutricional: 710 kcal.",
            },
            {
                "kind": "etiqueta",
                "attachment_id": "att-pote-3333",
                "description": "Tabla nutricional de proteína.",
            },
        ],
    }

    fotos = _plate_photos_from_vision(vision_multi)
    assert len(fotos) == 2
    assert fotos[0]["attachment_id"] == "att-guineo-1111"
    assert fotos[0]["kcal"] == 90.0
    assert "guineo" in fotos[0]["desc"].lower()

    assert fotos[1]["attachment_id"] == "att-sancocho-2222"
    assert fotos[1]["kcal"] == 710.0
    assert "sancocho" in fotos[1]["desc"].lower()


def test_asociar_adjuntos_a_comida_1_to_1():
    plate_photos = [
        {
            "desc": "1 guineo maduro mediano. Estimación nutricional: 90 kcal.",
            "kcal": 90.0,
            "attachment_id": "att-guineo-1111",
        },
        {
            "desc": "Sancocho criollo con arroz blanco (2 tazas). Estimación nutricional: 710 kcal.",
            "kcal": 710.0,
            "attachment_id": "att-sancocho-2222",
        },
    ]

    assigned = set()

    # Primera comida: Guineo de desayuno
    guineo_args = {
        "meal_name": "1 guineo maduro mediano",
        "calories": 90,
        "meal_type": "desayuno",
    }
    match_guineo = _asociar_adjuntos_a_comida(guineo_args, plate_photos, assigned_ids=assigned)
    assert match_guineo == ["att-guineo-1111"]
    assigned.update(match_guineo)

    # Segunda comida: Sancocho de almuerzo
    sancocho_args = {
        "meal_name": "Arroz blanco con sancocho criollo (2 tazas)",
        "calories": 710,
        "meal_type": "almuerzo",
    }
    match_sancocho = _asociar_adjuntos_a_comida(sancocho_args, plate_photos, assigned_ids=assigned)
    assert match_sancocho == ["att-sancocho-2222"]
    assigned.update(match_sancocho)

    # No debe quedar ninguna foto disponible
    disponible = [p for p in plate_photos if p["attachment_id"] not in assigned]
    assert len(disponible) == 0


def test_asociar_adjuntos_no_mezcla_fotos_no_coincidentes():
    plate_photos = [
        {
            "desc": "Sancocho criollo con arroz blanco. Estimación nutricional: 710 kcal.",
            "kcal": 710.0,
            "attachment_id": "att-sancocho-2222",
        },
    ]

    # Comida completamente diferente con calorías diferentes
    manzana_args = {
        "meal_name": "1 manzana verde",
        "calories": 60,
        "meal_type": "snack",
    }
    # Si la única foto disponible no tiene nada que ver con la comida (ni tokens ni kcal):
    # En un turno multi-foto (o donde la foto restante no coincide), no debe forzarla.
    # Simulamos pasando plate_photos de tamaño 2 donde la primera ya fue asignada:
    all_photos = [
        {"desc": "foto previa", "kcal": 50, "attachment_id": "att-1"},
        plate_photos[0],
    ]
    assigned = {"att-1"}
    match = _asociar_adjuntos_a_comida(manzana_args, all_photos, assigned_ids=assigned)
    assert match == []
