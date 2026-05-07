import pytest
from cron_tasks import _filter_days_by_fresh_pantry

def test_smart_shuffle_strict_pantry_validation():
    """
    Test de aceptación (P0-5): Verifica que el umbral estricto (0.9)
    y la validación descartan días con ingredientes faltantes.
    """
    # Pool: 1 día con 4 bases.
    # Pantry: Solo 3 bases presentes. (ratio 3/4 = 0.75)
    # Como el min_match_ratio subió a 0.9, el pool filtrado debería estar vacío.
    
    mock_pool = [
        {
            "meals": [
                {"name": "Comida 1", "ingredients": ["pollo", "arroz", "cebolla", "ajo"]}
            ],
            "_technique": "grill"
        }
    ]
    
    mock_pantry = ["pollo", "arroz", "cebolla"]
    
    filtered_pool = _filter_days_by_fresh_pantry(mock_pool, mock_pantry)
    
    assert len(filtered_pool) == 0, "El día no debe entrar al plan final por ratio menor a 0.9"

def test_smart_shuffle_accepts_when_ratio_is_met():
    """
    Test para validar que si el ratio >= 0.9, el día sí es aceptado en el pool inicial.
    """
    # Ratio: 9/10 = 0.9. Debería aceptarlo.
    mock_pool = [
        {
            "meals": [
                {"name": "Comida 1", "ingredients": ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"]}
            ]
        }
    ]
    mock_pantry = ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"]
    
    filtered_pool = _filter_days_by_fresh_pantry(mock_pool, mock_pantry)
    
    assert len(filtered_pool) == 1, "Debería aceptar un día con >= 90% de match"
