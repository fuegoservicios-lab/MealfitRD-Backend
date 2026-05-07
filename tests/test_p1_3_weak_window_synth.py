"""[P1-3] Tests para `_synthesize_user_history_lifetime_summary`.

Cuando el chunk N+1 tiene ventana rolling vacía y no hay `_lifetime_lessons_summary`
heredado (típico de chunk 2 de un primer plan 7d), el sistema debe sintetizar
contexto histórico desde:
  - `meal_rejections` (rechazos permanentes del onboarding).
  - `meal_plans` recientes (últimos 6 meses, ingredientes y nombres).

Sin esta síntesis el LLM solo recibe `_last_chunk_learning` (1 chunk), perdiendo
señales valiosas como "el usuario rechazó pollo frito en onboarding".
"""
from unittest.mock import patch


def test_returns_none_for_guest_user():
    from cron_tasks import _synthesize_user_history_lifetime_summary
    assert _synthesize_user_history_lifetime_summary("guest") is None
    assert _synthesize_user_history_lifetime_summary(None) is None


def test_returns_none_when_user_has_no_history():
    """Usuario verdaderamente nuevo: sin rechazos, sin planes pasados."""
    from cron_tasks import _synthesize_user_history_lifetime_summary

    with patch("db_chat.get_active_rejections", return_value=[]), \
         patch("db_plans.get_recent_plans", return_value=[]):
        result = _synthesize_user_history_lifetime_summary("user-new")

    assert result is None


def test_synthesizes_from_rejections_only():
    """Usuario con rechazos del onboarding pero sin planes previos."""
    from cron_tasks import _synthesize_user_history_lifetime_summary

    rejections = [
        {"meal_name": "Pollo Frito", "meal_type": "almuerzo"},
        {"meal_name": "Hígado Encebollado", "meal_type": "cena"},
        {"meal_name": "Sopa de Pescado", "meal_type": "almuerzo"},
    ]
    with patch("db_chat.get_active_rejections", return_value=rejections), \
         patch("db_plans.get_recent_plans", return_value=[]):
        result = _synthesize_user_history_lifetime_summary("user-1")

    assert result is not None
    assert result["synthesized_from_user_history"] is True
    assert "Pollo Frito" in result["top_rejection_hits"]
    assert "Hígado Encebollado" in result["top_rejection_hits"]
    assert "Sopa de Pescado" in result["top_rejection_hits"]
    assert result["top_repeated_bases"] == []
    assert result["top_repeated_meal_names"] == []


def test_synthesizes_from_recent_plans_only():
    """Usuario sin rechazos pero con planes recientes."""
    from cron_tasks import _synthesize_user_history_lifetime_summary

    recent_plans = [
        {
            "days": [
                {"meals": [
                    {"name": "Arroz con Pollo", "ingredients": ["pollo", "arroz"]},
                    {"name": "Bistec a la Plancha", "ingredients": ["res", "cebolla"]},
                ]},
            ]
        },
    ]
    with patch("db_chat.get_active_rejections", return_value=[]), \
         patch("db_plans.get_recent_plans", return_value=recent_plans):
        result = _synthesize_user_history_lifetime_summary("user-2")

    assert result is not None
    assert "Arroz con Pollo" in result["top_repeated_meal_names"]
    assert "Bistec a la Plancha" in result["top_repeated_meal_names"]
    # Bases canónicas normalizadas — al menos pollo y res deben estar.
    assert "pollo" in result["top_repeated_bases"]
    assert "res" in result["top_repeated_bases"]


def test_synthesizes_combined_sources():
    """Mix completo: rechazos + planes recientes."""
    from cron_tasks import _synthesize_user_history_lifetime_summary

    rejections = [{"meal_name": "Mondongo"}]
    recent_plans = [
        {"days": [{"meals": [{"name": "Pollo al Curry", "ingredients": ["pollo"]}]}]}
    ]
    with patch("db_chat.get_active_rejections", return_value=rejections), \
         patch("db_plans.get_recent_plans", return_value=recent_plans):
        result = _synthesize_user_history_lifetime_summary("user-3")

    assert result is not None
    assert "Mondongo" in result["top_rejection_hits"]
    assert "Pollo al Curry" in result["top_repeated_meal_names"]
    assert result["synthesized_from_user_history"] is True
    assert result["_lifetime_window_days"] == 180


def test_caps_field_sizes():
    """Listas largas se truncan a tamaños razonables para no inflar el prompt."""
    from cron_tasks import _synthesize_user_history_lifetime_summary

    rejections = [{"meal_name": f"Plato Rechazado {i}"} for i in range(50)]
    plans = [{"days": [
        {"meals": [
            {"name": f"Comida {i}", "ingredients": [f"ingrediente_{j}" for j in range(10)]}
            for i in range(40)
        ]},
    ]}]

    with patch("db_chat.get_active_rejections", return_value=rejections), \
         patch("db_plans.get_recent_plans", return_value=plans):
        result = _synthesize_user_history_lifetime_summary("user-4")

    assert result is not None
    assert len(result["top_rejection_hits"]) <= 20
    assert len(result["top_repeated_bases"]) <= 20
    assert len(result["top_repeated_meal_names"]) <= 15


def test_handles_db_errors_gracefully():
    """Si una de las queries falla, la otra fuente sigue produciendo señal."""
    from cron_tasks import _synthesize_user_history_lifetime_summary

    rejections = [{"meal_name": "Pollo Frito"}]
    with patch("db_chat.get_active_rejections", return_value=rejections), \
         patch("db_plans.get_recent_plans",
               side_effect=RuntimeError("DB temporarily unavailable")):
        result = _synthesize_user_history_lifetime_summary("user-5")

    assert result is not None
    assert "Pollo Frito" in result["top_rejection_hits"]
    assert result["top_repeated_bases"] == []


def test_skips_malformed_rejection_entries():
    """Entries que no son dict o no tienen meal_name se ignoran sin crash."""
    from cron_tasks import _synthesize_user_history_lifetime_summary

    rejections = [
        None,
        "not a dict",
        {"meal_name": "Plato Real"},
        {"other_field": "no name"},
    ]
    with patch("db_chat.get_active_rejections", return_value=rejections), \
         patch("db_plans.get_recent_plans", return_value=[]):
        result = _synthesize_user_history_lifetime_summary("user-6")

    assert result is not None
    assert result["top_rejection_hits"] == ["Plato Real"]
