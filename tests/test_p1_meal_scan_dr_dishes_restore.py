"""[P1-MEAL-SCAN-DR-DISHES-RESTORE · 2026-07-28] Restaura el prompt/schema
afinados (P1-MEAL-SCAN-DR-DISHES · 2026-07-12) al dispatch CLOUD real.

Regresion: la migracion del meal-scan al provider cloud (P0-LLM-PROVIDER-MIGRATION
-> P1-VISION-LUNA -> P1-VISION-NO-LOCAL) dejo `_dispatch_openai_compatible_vision`
con un prompt GENERICO inline + el schema `ImageDescription` (cap "max ~6
palabras" en `meal_name` — EXACTAMENTE el bug que `_MEAL_VISION_PROMPT`
(P1-MEAL-SCAN-DR-DISHES) ya habia cerrado para el provider local). El trio
`_MEAL_VISION_PROMPT` / `_MEAL_VISION_SCHEMA` / `_coerce_meal_scan` quedo con
CERO callers de produccion — verificado con grep antes de tocar nada (solo
aparecian en su propia definicion y en tests que assertan sobre el prompt
DIRECTO, nunca sobre el dispatch real).

`test_p1_meal_scan_dr_dishes.py` (el test pre-existente del fix original)
importa `_MEAL_VISION_PROMPT` DIRECTO del modulo y asserta sobre su
contenido — eso sigue siendo verdad diga lo que diga `_dispatch_openai_
compatible_vision`. Esas assertions pasaban VACIAMENTE mientras la regresion
vivia: el prompt afinado EXISTIA en el archivo pero nunca viajaba al LLM real.
Este archivo cierra ese gap con tests que muerden el CALL-SITE real (parser
acotado al codigo vivo, mas un test funcional que monkeypatchea
`_invoke_structured_vision` y captura los argumentos exactos que
`_dispatch_openai_compatible_vision` le pasa).

Fix: el dispatch ahora usa `_MEAL_VISION_PROMPT` + `_MealVisionResult` (bridge
Pydantic de `_MEAL_VISION_SCHEMA` — `with_structured_output` en este modulo
SIEMPRE recibe una clase Pydantic, nunca un dict JSON-schema crudo; ver
comentario junto a `_MealVisionResult` en vision_agent.py) + `_coerce_meal_scan`
para normalizar la respuesta al contrato que ya consumen `routers/diary.py`
y `ScanMealModal.jsx` (ambos YA leian `photo_kind`/`items` con fallback —
codigo que quedaba muerto sin este fix).

Verificado en vivo contra la API real de gpt-5.6-luna (VPS, script scratch,
imagen sintetica Pillow) — ver reporte de la sesion para el output literal.

Tooltip-anchor: P1-MEAL-SCAN-DR-DISHES-RESTORE
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _src(rel: str = "vision_agent.py") -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parser acotado al CALL-SITE real de _dispatch_openai_compatible_vision.
#
# La ventana arranca en la llamada real a `_invoke_structured_vision` y
# termina en el `except Exception as e:` que la envuelve — deliberadamente
# EXCLUYE el docstring de la funcion (que narra a proposito la historia del
# bug y menciona "ImageDescription"/"maximo 6 palabras" como HISTORIA, no
# como codigo vivo — mismo patron tolerado que test_p1_vision_no_local.py).
# Sin esta acotacion, la narrativa produciria falsos positivos O negativos.
# ---------------------------------------------------------------------------

_CALL_SITE_START = "response = await _invoke_structured_vision("
_CALL_SITE_END = "except Exception as e:"


def _dispatch_call_window(src: str) -> str:
    start = src.find(_CALL_SITE_START)
    end = src.find(_CALL_SITE_END, start if start != -1 else 0)
    assert start != -1 and end != -1, (
        "no se encontro el call-site de _dispatch_openai_compatible_vision "
        "(marcadores _CALL_SITE_START/_CALL_SITE_END desactualizados tras "
        "un refactor — re-anclar)."
    )
    return src[start:end]


def _dispatch_uses_tuned_prompt(window: str) -> bool:
    """True si la ventana referencia el prompt/schema/normalizador AFINADOS
    (P1-MEAL-SCAN-DR-DISHES) y NO el trio generico que causo la regresion."""
    return (
        "_MEAL_VISION_PROMPT" in window
        and "_MealVisionResult" in window
        and "_coerce_meal_scan(" in window
        and "ImageDescription" not in window
        and "Describe detalladamente" not in window
        and "máx ~6 palabras" not in window
    )


def test_dispatch_window_uses_tuned_prompt_schema_and_coercion():
    """El call-site REAL de vision_agent.py hoy debe pasar este detector."""
    window = _dispatch_call_window(_src())
    assert _dispatch_uses_tuned_prompt(window), (
        "_dispatch_openai_compatible_vision ya no referencia "
        "_MEAL_VISION_PROMPT / _MealVisionResult / _coerce_meal_scan, o "
        "volvio a traer el prompt generico / ImageDescription — la "
        "regresion P1-MEAL-SCAN-DR-DISHES-RESTORE reaparecio.\n"
        f"Ventana analizada:\n{window}"
    )


# --- Bite proof: el detector de arriba SI habria atrapado la regresion real.
#
# `_OLD_BROKEN_WINDOW` es una reconstruccion literal del call-site que corria
# en produccion ANTES de este P-fix (leido directamente del archivo antes de
# editarlo). Si `_dispatch_uses_tuned_prompt` no lo rechaza, el test de
# arriba seria un guard vacio — exactamente el modo de fallo que este
# archivo existe para cerrar.

_OLD_BROKEN_WINDOW = (
    "response = await _invoke_structured_vision(\n"
    "            image_bytes,\n"
    "            \"Describe detalladamente todos los alimentos, ingredientes"
    " o platillos que ves en esta imagen. Si es una nevera, lista el"
    " contenido visible. Si no hay comida, indícalo. Da también un nombre"
    " corto del platillo en español dominicano (`meal_name`, máx ~6"
    " palabras). También proporciona una estimación de las calorías,"
    " gramos de proteína, gramos de carbohidratos y gramos de grasas"
    " saludables (solo el número) totales en la imagen (usa 0 si no es"
    " comida).\",\n"
    "            ImageDescription,\n"
    "        )\n\n"
    "        description = response.description if response and"
    " hasattr(response, 'description') else \"Imagen sin descripción"
    " clara.\"\n"
    "        is_food = response.is_food if response and hasattr(response,"
    " 'is_food') else False\n"
    "        calories = 0\n"
    "        protein = 0\n"
    "        carbs = 0\n"
    "        healthy_fats = 0\n"
    "        meal_name = \"\"\n\n"
    "        if is_food:\n"
    "            calories = response.calories if hasattr(response,"
    " 'calories') else 0\n"
    "            meal_name = (response.meal_name if hasattr(response,"
    " 'meal_name') else \"\") or \"\"\n"
    "        return {\n"
    "            \"description\": description,\n"
    "            \"is_food\": is_food,\n"
    "            \"meal_name\": meal_name,\n"
    "            \"calories\": calories,\n"
    "            \"protein\": protein,\n"
    "            \"carbs\": carbs,\n"
    "            \"healthy_fats\": healthy_fats,\n"
    "        }\n\n    "
)


def test_bite_detector_rejects_the_real_pre_fix_code():
    """MUERDE de verdad: el call-site EXACTO que corria en produccion antes
    de este P-fix (prompt generico inline + `ImageDescription`, cero
    referencia a `_MEAL_VISION_PROMPT`/`_MealVisionResult`/`_coerce_meal_scan`)
    debe ser RECHAZADO por el detector."""
    assert not _dispatch_uses_tuned_prompt(_OLD_BROKEN_WINDOW), (
        "el detector NO habria atrapado el codigo pre-fix real — el test "
        "de wiring de arriba seria un guard vacio ante un revert real."
    )


def test_bite_detector_accepts_a_synthetic_fixed_window():
    """Direccion opuesta: una ventana sintetica que SI hace lo correcto pasa
    (confirma que el detector no esta simplemente rechazando todo)."""
    fixed = (
        "response = await _invoke_structured_vision(\n"
        "    image_bytes,\n"
        "    _MEAL_VISION_PROMPT,\n"
        "    _MealVisionResult,\n"
        ")\n"
        "data = response.model_dump() if response else {}\n"
        "return _coerce_meal_scan(data)\n\n    "
    )
    assert _dispatch_uses_tuned_prompt(fixed)


# ---------------------------------------------------------------------------
# 2. Funcional: monkeypatch de _invoke_structured_vision (un nivel mas
#    profundo que _dispatch_openai_compatible_vision) para exercitar el
#    CUERPO real de la funcion — incluye la normalizacion via
#    _coerce_meal_scan y la construccion del dict de retorno.
# ---------------------------------------------------------------------------


def test_dispatch_calls_invoke_with_tuned_prompt_and_schema(monkeypatch):
    import vision_agent as va

    captured = {}

    async def _fake_invoke(image_bytes, prompt, schema):
        captured["image_bytes"] = image_bytes
        captured["prompt"] = prompt
        captured["schema"] = schema
        return va._MealVisionResult(
            photo_kind="plato",
            is_food=True,
            meal_name="Mangu con huevo frito, salami y queso frito criollo",
            description="Mangu, huevo frito, salami, queso frito, cebolla roja encurtida",
            calories=750,
            protein=35,
            carbs=80,
            healthy_fats=20,
            items=[],
        )

    monkeypatch.setattr(va, "_invoke_structured_vision", _fake_invoke)

    out = asyncio.run(va._dispatch_openai_compatible_vision(b"fake-jpeg-bytes"))

    # El prompt/schema pasados deben ser LOS AFINADOS, no un literal genérico
    # ni ImageDescription.
    assert captured["prompt"] is va._MEAL_VISION_PROMPT, (
        "_dispatch_openai_compatible_vision NO uso _MEAL_VISION_PROMPT — "
        f"uso: {captured['prompt']!r}"
    )
    assert captured["schema"] is va._MealVisionResult, (
        "_dispatch_openai_compatible_vision NO uso el bridge Pydantic "
        f"_MealVisionResult — uso: {captured['schema']!r}"
    )

    # Las 7 keys legacy que routers/diary.py, agent.py y el frontend ya leen
    # DEBEN seguir presentes con los valores esperados.
    for key in ("description", "is_food", "meal_name", "calories",
                "protein", "carbs", "healthy_fats"):
        assert key in out, f"falta key legacy {key!r} en el resultado del dispatch"

    assert out["is_food"] is True
    assert out["calories"] == 750
    assert out["protein"] == 35
    assert out["carbs"] == 80
    assert out["healthy_fats"] == 20

    # Additive keys (diary.py y ScanMealModal.jsx ya los leen con fallback —
    # no rompen ningun consumidor existente).
    assert out["photo_kind"] == "plato"
    assert out["items"] == []

    # El cap de 6 palabras desaparecio del path vivo: un meal_name de 8+
    # palabras sobrevive INTACTO (antes: ImageDescription lo pedia "max ~6
    # palabras" en el prompt genérico).
    assert out["meal_name"] == "Mangu con huevo frito, salami y queso frito criollo"
    assert len(out["meal_name"].split()) > 6, (
        "el meal_name deberia superar las 6 palabras — si esto falla, "
        "revisar que _coerce_meal_scan no este truncando por palabras."
    )

    # _coerce_meal_scan aplico su normalizacion (paridad del sufijo de
    # estimacion — requisito explicito: NO perder el sufijo que consumen
    # otros surfaces).
    assert "Estimaci" in out["description"] and "750" in out["description"], (
        "el sufijo de estimacion de macros desaparecio de la description — "
        "_coerce_meal_scan no se esta aplicando."
    )


def test_dispatch_items_mode_reaches_live_path(monkeypatch):
    """Modo 'items' (compra/alimentos sueltos) — antes de este fix, el
    dispatch generico SIEMPRE devolvia is_food/description de un solo modo
    ('plato' implicito); el modo 'items' de _coerce_meal_scan (P1-CHAT-VISION-GEMMA)
    quedaba inalcanzable desde el path cloud aunque diary.py y ScanMealModal.jsx
    ya lo manejaran."""
    import vision_agent as va

    async def _fake_invoke(image_bytes, prompt, schema):
        return va._MealVisionResult(
            photo_kind="items",
            is_food=True,
            items=[va._MealVisionItem(name="pechuga de pollo", quantity=2, unit="lb")],
        )

    monkeypatch.setattr(va, "_invoke_structured_vision", _fake_invoke)

    out = asyncio.run(va._dispatch_openai_compatible_vision(b"fake-jpeg-bytes"))

    assert out["photo_kind"] == "items"
    assert out["items"] == [{"name": "pechuga de pollo", "quantity": 2.0, "unit": "lb"}]
    # Modo items: macros en 0, meal_name vacio (contrato de _coerce_meal_scan).
    assert out["calories"] == 0 and out["meal_name"] == ""


def test_dispatch_error_path_unchanged(monkeypatch):
    """El path de error (analysis_failed=True) NO debe cambiar de forma —
    requisito explicito del fix."""
    import vision_agent as va

    async def _boom(image_bytes, prompt, schema):
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(va, "_invoke_structured_vision", _boom)

    out = asyncio.run(va._dispatch_openai_compatible_vision(b"fake-jpeg-bytes"))

    assert out == {
        "description": "Error analizando imagen.",
        "is_food": False,
        "analysis_failed": True,
        "meal_name": "",
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
    }


# ---------------------------------------------------------------------------
# 3. ImageDescription sigue definida (test_p2_diary_scan_macros.py depende
#    de que la clase exista con el campo meal_name) pero YA NO tiene
#    callers en el call-site vivo — la ventana de arriba ya lo prueba;
#    esto ancla puntualmente que NO se borro (borrarla romperia ese otro
#    test, fuera del alcance de este P-fix).
# ---------------------------------------------------------------------------


def test_image_description_kept_for_backcompat_but_unused_live():
    import vision_agent as va

    assert hasattr(va, "ImageDescription"), (
        "ImageDescription no debe borrarse — test_p2_diary_scan_macros.py "
        "depende de que la clase siga definida con el campo meal_name."
    )
    window = _dispatch_call_window(_src())
    assert "ImageDescription" not in window, (
        "ImageDescription sigue viva en el call-site del dispatch — la "
        "regresion no se cerro."
    )


# ---------------------------------------------------------------------------
# 4. Marker supersession-proof (mismo patron que test_p1_vision_luna.py /
#    test_p1_vision_no_local.py) — own-name-or-later-date. Un `startswith`
#    seria una mina: rechazaria cualquier P-fix futuro con OTRO nombre sin
#    importar la fecha.
# ---------------------------------------------------------------------------

_OUR_MARKER_SLUG = "P1-MEAL-SCAN-DR-DISHES-RESTORE"
_OUR_MARKER_DATE = "2026-07-28"


def _marker_supersedes_or_matches(marker: str) -> bool:
    if _OUR_MARKER_SLUG in marker:
        return True
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    if not fecha:
        return False
    return fecha.group(1) >= _OUR_MARKER_DATE


def test_marker_bumped_to_dr_dishes_restore_or_later():
    src = _src("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX en app.py"
    marker = m.group(1)
    assert _marker_supersedes_or_matches(marker), (
        f"`_LAST_KNOWN_PFIX={marker!r}` no es P1-MEAL-SCAN-DR-DISHES-RESTORE "
        f"ni tiene fecha >= {_OUR_MARKER_DATE} -- stale respecto a este P-fix."
    )


@pytest.mark.parametrize("marker_viejo", [
    "P1-VISION-NO-LOCAL · 2026-07-27",
    "P0-ALGUN-FIX-VIEJO · 2026-01-01",
])
def test_bite_marker_mas_viejo_es_rechazado(marker_viejo):
    """MUERDE de verdad: un marker de OTRO P-fix con fecha ANTERIOR debe
    fallar la validacion."""
    assert not _marker_supersedes_or_matches(marker_viejo)


@pytest.mark.parametrize("marker_nuevo", [
    "P1-VISION-NO-LOCAL · 2026-07-28",
    "P2-ALGUN-FIX-FUTURO · 2026-08-01",
])
def test_bite_marker_de_otro_pfix_pero_mas_nuevo_es_aceptado(marker_nuevo):
    """MUERDE de verdad (direccion opuesta): un marker de OTRO P-fix pero
    con fecha >= la nuestra DEBE pasar — esto es lo que un `startswith`
    NUNCA aceptaria."""
    assert _marker_supersedes_or_matches(marker_nuevo)


def test_marker_slug_matches_this_test_file():
    """Cross-link con test_p2_hist_audit_14_marker_test_link.py: el slug
    derivado del marker debe matchear el nombre de ESTE archivo."""
    slug = _OUR_MARKER_SLUG.replace("-", "_").lower()
    assert slug == "p1_meal_scan_dr_dishes_restore"
    this_file = os.path.basename(__file__)
    assert this_file.startswith(f"test_{slug}"), (
        f"slug {slug!r} no matchea el nombre de este archivo {this_file!r}"
    )
