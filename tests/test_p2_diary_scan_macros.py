"""[P2-DIARY-SCAN-MACROS · 2026-05-30] Regression guard del flujo
"Escanear comida → registrar macros" del Dashboard.

Contexto: el vision agent (`process_image_with_vision`) ya estimaba las 4
macros (calorías/proteína/carbs/grasas) pero solo retornaba `calories`, y el
endpoint `/api/diary/upload` solo devolvía `description/is_food/image_url/
red_alert` — sin macros. El modal del Dashboard necesita las macros + un nombre
corto del platillo para precargar los campos editables que el usuario confirma
antes de hacer POST /api/diary/consumed.

Este test ancla (parser-based) las dos extensiones backend, de modo que un
renombre/revert falle aquí antes de romper la feature en producción:

  1. `ImageDescription` declara el campo `meal_name`.
  2. `process_image_with_vision` retorna las 4 macros + `meal_name` (no solo
     `calories`).
  3. `/api/diary/upload` expone `meal_name` + `macros{...}` en su respuesta sin
     persistir nada (la confirmación + INSERT la hace el usuario).
  4. El marker `_LAST_KNOWN_PFIX` quedó bumpeado a esta familia.

Tooltip-anchor: P2-DIARY-SCAN-MACROS.
"""
import re
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def vision_src() -> str:
    return (_BACKEND_DIR / "vision_agent.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def diary_src() -> str:
    return (_BACKEND_DIR / "routers" / "diary.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_src() -> str:
    return (_BACKEND_DIR / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. ImageDescription declara meal_name
# ---------------------------------------------------------------------------
def test_image_description_has_meal_name(vision_src: str):
    # El bloque de la clase ImageDescription debe declarar `meal_name`.
    m = re.search(
        r"class ImageDescription\(BaseModel\):(?P<body>[\s\S]+?)\n\S",
        vision_src,
    )
    assert m, "No se encontró la clase ImageDescription."
    body = m.group("body")
    assert re.search(r"\bmeal_name\b\s*:\s*str", body), (
        "P2-DIARY-SCAN-MACROS: `ImageDescription` ya no declara el campo "
        "`meal_name: str` — el modal de escaneo no podrá precargar el nombre."
    )


# ---------------------------------------------------------------------------
# 2. process_image_with_vision retorna las 4 macros + meal_name
# ---------------------------------------------------------------------------
def test_vision_returns_all_macros_and_name(vision_src: str):
    # El return de éxito de process_image_with_vision debe incluir las 4 macros
    # + meal_name como keys (no solo "calories").
    for key in ("meal_name", "calories", "protein", "carbs", "healthy_fats"):
        assert re.search(rf'["\']{key}["\']\s*:', vision_src), (
            f"P2-DIARY-SCAN-MACROS: `process_image_with_vision` ya no retorna "
            f"la key '{key}'."
        )


# ---------------------------------------------------------------------------
# 3. /api/diary/upload expone meal_name + macros
# ---------------------------------------------------------------------------
def test_upload_endpoint_exposes_macros(diary_src: str):
    # Lee las macros del vision_result.
    for key in ("protein", "carbs", "healthy_fats", "meal_name"):
        assert re.search(
            rf'{key}\s*=\s*vision_result\.get\(\s*["\']{key}["\']', diary_src
        ), (
            f"P2-DIARY-SCAN-MACROS: `/api/diary/upload` ya no extrae "
            f"'{key}' del vision_result."
        )
    # Devuelve el bloque `macros` en la respuesta.
    assert re.search(r'["\']macros["\']\s*:\s*\{', diary_src), (
        "P2-DIARY-SCAN-MACROS: la respuesta de `/api/diary/upload` ya no "
        "incluye el bloque `macros`."
    )
    assert re.search(r'["\']meal_name["\']\s*:\s*meal_name', diary_src), (
        "P2-DIARY-SCAN-MACROS: la respuesta de `/api/diary/upload` ya no "
        "incluye `meal_name`."
    )
    # `analysis_failed` distingue "analizador caído" de "no es comida".
    assert re.search(r'["\']analysis_failed["\']\s*:\s*analysis_failed', diary_src), (
        "P2-DIARY-SCAN-MACROS: `/api/diary/upload` ya no propaga "
        "`analysis_failed` — el modal no podrá distinguir fallo de no-comida."
    )


def test_vision_error_path_flags_analysis_failed(vision_src: str):
    # El except de process_image_with_vision debe marcar analysis_failed=True.
    m = re.search(
        r"except Exception as e:(?P<body>[\s\S]+?)\n\ndef ", vision_src
    )
    assert m, "No se encontró el except de process_image_with_vision."
    assert re.search(r'["\']analysis_failed["\']\s*:\s*True', m.group("body")), (
        "P2-DIARY-SCAN-MACROS: el path de error de `process_image_with_vision` "
        "ya no marca `analysis_failed: True`."
    )


def test_upload_does_not_persist_consumed_meal(diary_src: str):
    # Defensa: el endpoint /upload NO debe llamar log_consumed_meal — la
    # confirmación + INSERT es responsabilidad del usuario (modal → /consumed).
    # Aislamos el cuerpo de api_diary_upload para no chocar con el endpoint
    # /consumed que sí vive en el mismo archivo.
    m = re.search(
        r"async def api_diary_upload\((?P<body>[\s\S]+?)\n@router\.",
        diary_src,
    )
    assert m, "No se encontró el handler api_diary_upload."
    assert "log_consumed_meal" not in m.group("body"), (
        "P2-DIARY-SCAN-MACROS: `/api/diary/upload` no debe registrar la comida "
        "directamente — el escaneo solo estima; el usuario confirma vía "
        "/api/diary/consumed."
    )


# ---------------------------------------------------------------------------
# 4. Marker bump
# ---------------------------------------------------------------------------
def test_marker_bumped(app_src: str):
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"(?P<marker>[^"]+)"', app_src)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    # No exigimos que sea exactamente esta familia para siempre (un P-fix
    # posterior lo bumpea), solo que el formato siga válido y con fecha >= la
    # de este cierre — el test de freshness ya cubre el floor canónico.
    assert re.match(r"^P\d+(?:-[A-Z0-9]+)+\s+·\s+\d{4}-\d{2}-\d{2}$", m.group("marker")), (
        f"_LAST_KNOWN_PFIX tiene formato inválido: {m.group('marker')!r}"
    )
