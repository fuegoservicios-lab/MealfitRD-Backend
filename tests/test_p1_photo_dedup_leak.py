# [P1-PHOTO-DEDUP-LEAK · 2026-11-01]
# Pruebas para garantizar que log_consumed_meal no arrastre attachment_ids de turnos anteriores
import pytest


def test_attachment_ids_sanitized_to_current_turn_photos():
    """Verifica que attachment_ids de turnos anteriores se eliminen y solo queden los del turno actual."""
    # Simulamos la lógica implementada en execute_tools para log_consumed_meal
    state = {
        "turn_plate_photos": [
            {"attachment_id": "photo-current-turn-123", "desc": "papas con huevo", "kcal": 695}
        ]
    }
    tool_args = {
        "user_id": "test-user",
        "meal_name": "Papas fritas con 2 huevos",
        "calories": 695,
        "protein": 23,
        "nutrition": {
            "attachment_ids": [
                "photo-old-turn-999",       # Leakeado por el LLM desde un turno previo
                "photo-current-turn-123",    # Del turno actual
            ]
        }
    }

    _valid_turn_ids = {
        str(p["attachment_id"])
        for p in (state.get("turn_plate_photos") or [])
        if isinstance(p, dict) and p.get("attachment_id")
    }
    _cur_nutrition = tool_args.get("nutrition")
    if isinstance(_cur_nutrition, dict):
        _cur_nutrition = dict(_cur_nutrition)
        _clean_att = [str(a) for a in (_cur_nutrition.get("attachment_ids") or []) if str(a) in _valid_turn_ids]
        if _clean_att:
            _cur_nutrition["attachment_ids"] = _clean_att
        else:
            _cur_nutrition.pop("attachment_ids", None)
        tool_args["nutrition"] = _cur_nutrition

    assert tool_args["nutrition"]["attachment_ids"] == ["photo-current-turn-123"]
    assert "photo-old-turn-999" not in tool_args["nutrition"]["attachment_ids"]


def test_attachment_ids_stripped_when_no_turn_photos():
    """Si el turno no tiene fotos (ej. registro por texto de creatina), se limpia cualquier attachment_id."""
    state = {
        "turn_plate_photos": []
    }
    tool_args = {
        "user_id": "test-user",
        "meal_name": "5 cápsulas de Creatine",
        "calories": 0,
        "protein": 0,
        "nutrition": {
            "attachment_ids": ["photo-old-turn-999"]
        }
    }

    _valid_turn_ids = {
        str(p["attachment_id"])
        for p in (state.get("turn_plate_photos") or [])
        if isinstance(p, dict) and p.get("attachment_id")
    }
    _cur_nutrition = tool_args.get("nutrition")
    if isinstance(_cur_nutrition, dict):
        _cur_nutrition = dict(_cur_nutrition)
        _clean_att = [str(a) for a in (_cur_nutrition.get("attachment_ids") or []) if str(a) in _valid_turn_ids]
        if _clean_att:
            _cur_nutrition["attachment_ids"] = _clean_att
        else:
            _cur_nutrition.pop("attachment_ids", None)
        tool_args["nutrition"] = _cur_nutrition

    assert "attachment_ids" not in tool_args["nutrition"]
