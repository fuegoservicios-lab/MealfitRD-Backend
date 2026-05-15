"""[P1-DIARY-PROMPT-INJECTION · 2026-05-15] Anchor parser-based: el endpoint
`/api/diary/upload` NO debe concatenar texto instructivo (poison_pill) al
`description` que luego se persiste a `visual_diary` y llega al chat agent.

Contexto del bug original:
    El endpoint pre-fix detectaba "calorías altas + hora crítica" y, cuando
    la regla determinista disparaba, concatenaba al `description` un string
    de instrucciones para el LLM downstream ("INSTRUCCIÓN CLÍNICA PRIORITARIA
    DE SISTEMA: ... TIENES LA ORDEN DIRECTA Y OBLIGATORIA de cambiar tu
    tono..."). El string se persistía a `visual_diary.description` y
    quedaba embebido en futuros contextos del chat agent.

    Vector de explotación: `calories` proviene de `vision_agent` → LLM
    derivado de la imagen del usuario → controlable. `tz_offset_mins` viene
    del body del cliente → controlable. Un atacante podía construir un
    payload para disparar la regla y dejar texto instructivo persistido.

Fix:
    Se eliminó la concatenación. La signal chrono-nutrición ahora se canaliza
    via `pipeline_metrics(node='chrono_nutrition_red_alert', ...)` (structured,
    no-text). El response del endpoint expone `red_alert: bool` + metadata
    para que el frontend decida UX sin que la LLM reciba texto poisoned.

Tooltip-anchor: P1-DIARY-PROMPT-INJECTION-START
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_DIARY_PY = _BACKEND / "routers" / "diary.py"


@pytest.fixture(scope="module")
def diary_src() -> str:
    return _DIARY_PY.read_text(encoding="utf-8")


def test_no_poison_pill_concatenation(diary_src: str):
    """Regresión: el string instructivo "INSTRUCCIÓN CLÍNICA PRIORITARIA"
    NO debe aparecer en diary.py, y NO debe haber `description += <something>`
    en el flujo de upload.
    """
    assert "INSTRUCCIÓN CLÍNICA PRIORITARIA" not in diary_src, (
        "P1-DIARY-PROMPT-INJECTION regresión: el poison_pill literal vuelve a "
        "aparecer en diary.py. Vector prompt-injection reabierto."
    )
    assert "description += poison_pill" not in diary_src, (
        "P1-DIARY-PROMPT-INJECTION regresión: la concatenación volvió a "
        "introducirse — texto instructivo controlable por vision_agent se "
        "estaría persistiendo a visual_diary.description."
    )


def test_chrono_red_alert_emits_pipeline_metric(diary_src: str):
    """El branch que detectaba chrono-violation debe emitir un tick estructurado
    a `pipeline_metrics` con node `chrono_nutrition_red_alert` (signal canal).
    """
    assert "chrono_nutrition_red_alert" in diary_src, (
        "P1-DIARY-PROMPT-INJECTION: el reemplazo estructural (node="
        "'chrono_nutrition_red_alert' en pipeline_metrics) no aparece."
    )
    # Verificar que el INSERT a pipeline_metrics está en el flujo del upload.
    # Ventana 1200 chars cubre INSERT INTO + VALUES tupla + json.dumps con
    # node literal "chrono_nutrition_red_alert".
    assert re.search(
        r"INSERT INTO pipeline_metrics[\s\S]{0,1200}chrono_nutrition_red_alert",
        diary_src,
    ), (
        "P1-DIARY-PROMPT-INJECTION: el INSERT a pipeline_metrics con node="
        "'chrono_nutrition_red_alert' no aparece dentro de la ventana esperada."
    )


def test_response_exposes_structured_red_alert_flag(diary_src: str):
    """El response del endpoint debe exponer `red_alert: bool` (estructurado)
    para que el frontend reaccione sin recibir poison_pill text.
    """
    assert re.search(r'"red_alert":\s*chrono_red_alert', diary_src), (
        "P1-DIARY-PROMPT-INJECTION: el response no expone `red_alert` como "
        "flag estructurado — la signal chrono no llega al cliente sin texto."
    )


def test_marker_tooltip_present(diary_src: str):
    """Tooltip-anchor para que renames disparen este test (convención repo)."""
    assert "P1-DIARY-PROMPT-INJECTION" in diary_src, (
        "Marker tooltip ausente — si renombras el bloque, actualiza este test."
    )
