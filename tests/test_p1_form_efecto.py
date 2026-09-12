# -*- coding: utf-8 -*-
"""[P1-FORM-EFECTO · 2026-09-09] Cada paso del asistente tiene que CAMBIAR algo.

Un campo que se le pide al usuario y que nadie lee es una mentira barata: le cobras un
paso del asistente, le pides un dato personal, y su plan sale idéntico. Esta suite es la
red que impide que aparezca el primero.

## Lo que se midió al escribirla (09-sep, sobre el perfil real del plan c4098931)

**41 de 41 campos** del formulario tienen sitio de lectura real en producción. Ninguno
huérfano, ninguno atrapado en la capa de transporte. El reparto por dónde muerden:

* 21 cambian el prompt o la política compilada (dieta, alergias, presupuesto, ciclo de
  compra, congelador, condiciones médicas, suplementos, motivación, sueño, estrés…).
* 10 mueven los objetivos nutricionales (edad, sexo, peso, talla, actividad, meta,
  ritmo, grasa corporal, peso objetivo, unidad).
* el resto vive en superficies propias: `scheduleType` en las ventanas de comida del
  coach y el diario, `habitCaffeine`/`habitSmoking` en `condition_rules`, `totalDays`
  en la planificación de bloques, `struggles` en el orquestador.

**`scheduleType` no cambia los platos**, solo el coach y las ventanas horarias. Es
defendible, pero conviene saberlo: quien elige «turno de noche» en el asistente del PLAN
no ve un plan distinto.

## Dos trampas que se cobraron esta auditoría, ambas del instrumento

1. **Los suplementos viajan como CLAVE, no como etiqueta.** El formulario manda
   `creatine`; probarlo con `"Creatina"` hace que el validador lo descarte y el prompt
   diga «no seleccionó suplementos específicos» — que es exactamente el síntoma de un
   bug real. No lo era: era la sonda.
2. **El entorno local no es producción.** Sin `MEALFIT_COUNTRY_SYSTEM`,
   `country_for_form_data` colapsa CUALQUIER país a `DO` y la cultura a
   `dominican_criolla`. Medido así, `country` y `cultureProfiles` parecen inertes.
   Con el knob puesto —como está en producción— ES resuelve a `market_country: ES` y a
   `spain_mediterranea`. *Una auditoría que corre con otra configuración audita otra
   cosa.*
"""
import json
import os
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"

# Los 41 campos que el asistente recoge. Excluye lo que escribe el MOTOR en el perfil
# (previous_meals, grocery_cycle, reflection_history…): eso no es una respuesta del usuario.
CAMPOS_DEL_FORMULARIO = [
    "activityLevel", "age", "allergies", "appMode", "batchCooking", "bodyFat", "budget",
    "budgetAmount", "budgetCurrency", "cookingTime", "country", "cultureProfiles",
    "dietType", "dislikes", "freezerMode", "freshTopup", "gender", "goalPace",
    "groceryDuration", "habitAlcohol", "habitCaffeine", "habitSmoking", "habitWater",
    "height", "householdSize", "includeSupplements", "mainGoal", "mealOrganization",
    "medicalConditions", "medications", "motivation", "otherAllergies", "otherConditions",
    "otherDislikes", "otherMedications", "otherStruggles", "planSource", "scheduleType",
    "selectedSupplements", "sleepHours", "stapleFoods", "stressLevel", "struggles",
    "targetWeight", "totalDays", "waistCm", "weight", "weightUnit",
]

_EXCLUIDOS = ("tests", "__pycache__", "venv-test", "data", "migrations", "docs",
              ".git", "scripts", "node_modules")
# `routers/` es TRANSPORTE: valida y guarda. Que un campo solo aparezca ahí significa
# que se le pide al usuario, se comprueba… y nadie lo usa para nada.
_SOLO_TRANSPORTE = ("routers", "landing_benchmarks.py")


def _fuentes():
    out = {}
    for raiz, dirs, ficheros in os.walk(_BACKEND):
        dirs[:] = [d for d in dirs if d not in _EXCLUIDOS]
        for f in ficheros:
            if f.endswith(".py"):
                r = Path(raiz) / f
                rel = str(r.relative_to(_BACKEND)).replace("\\", "/")
                try:
                    out[rel] = r.read_text(encoding="utf-8")
                except Exception:                                      # noqa: BLE001
                    pass
    return out


_FUENTES = _fuentes()


def _campos_del_merge_de_texto_libre():
    """Los `other*` no se leen con `get(...)`: entran por la tabla de merge P1-2, que los
    funde en su array canónico (`otherStruggles` → `struggles`).

    Es una segunda vía de consumo REAL y con sitio fijo, así que se reconoce por su
    nombre en vez de aflojar el patrón de lectura a una simple mención — que convertiría
    el test en un buscador de cadenas y le quitaría los dientes.
    """
    src = _FUENTES.get("graph_orchestrator.py", "")
    bloque = re.search(r"_OTHER_TEXT_FIELD_MAP\s*=\s*\((.*?)\n\)", src, re.DOTALL)
    if not bloque:
        return set()
    return set(re.findall(r"""["'](\w+)["']""", bloque.group(1)))


_MERGEADOS = _campos_del_merge_de_texto_libre()


@pytest.mark.parametrize("campo", CAMPOS_DEL_FORMULARIO)
def test_cada_campo_del_asistente_lo_LEE_alguien(campo):
    """Ningún paso del asistente puede quedarse en la capa de transporte.

    Se busca la LECTURA (`get("campo")` / `["campo"]`), no la mención: una cadena suelta
    en un comentario o en una lista de validación no es consumo. La única excepción son
    los campos de texto libre, que se consumen fundiéndose en su array canónico.
    """
    pat = re.compile(r"""(?:get\(\s*["']""" + re.escape(campo) + r"""["']|"""
                     r"""\[\s*["']""" + re.escape(campo) + r"""["']\s*\])""")
    lectores = [ruta for ruta, src in _FUENTES.items()
                if pat.search(src) and not ruta.startswith(_SOLO_TRANSPORTE)]
    assert lectores or campo in _MERGEADOS, (
        f"`{campo}` se le pide al usuario y NADIE lo lee fuera de la capa de "
        f"transporte, ni entra por el merge de texto libre: es un paso del asistente "
        f"que no cambia su plan")


def test_la_tabla_de_merge_sigue_existiendo():
    """Si alguien borra `_OTHER_TEXT_FIELD_MAP`, los cuatro campos de texto libre vuelven
    a quedar huérfanos — y el test de arriba dejaría de acusarlo porque su excepción se
    vaciaría en silencio. Una excepción que se auto-cancela es peor que no tenerla."""
    assert {"otherAllergies", "otherConditions", "otherDislikes", "otherStruggles"} <= _MERGEADOS, (
        f"la tabla de merge de texto libre perdió campos: {sorted(_MERGEADOS)}")


# ------------------------------------------------------------------ suplementos
def _claves_frontend():
    src = (_FRONT / "components" / "assessment" / "questions" / "QSupplements.jsx").read_text(
        encoding="utf-8")
    bloque = re.search(r"const SUPPLEMENT_META\s*=\s*\{(.*?)\n\};", src, re.DOTALL)
    assert bloque, "SUPPLEMENT_META desapareció de QSupplements.jsx"
    return set(re.findall(r"^\s*(\w+)\s*:\s*\{", bloque.group(1), re.MULTILINE))


def test_las_claves_de_suplementos_coinciden_en_las_DOS_puntas():
    """El formulario manda CLAVES (`creatine`), no etiquetas.

    Si una punta añade un suplemento y la otra no, el validador lo descarta en silencio y
    el prompt pasa a decir «no seleccionó suplementos específicos» — con lo que el modelo
    recomienda LIBREMENTE lo que el usuario no pidió. El fallo es invisible: el plan sale,
    con suplementos, y nadie nota que no son los suyos.
    """
    from constants import SUPPLEMENT_NAMES

    front, back = _claves_frontend(), set(SUPPLEMENT_NAMES)
    assert front == back, (
        f"drift de suplementos · solo en el formulario: {sorted(front - back)} · "
        f"solo en el backend: {sorted(back - front)}")


def test_seleccionar_un_suplemento_CAMBIA_lo_que_se_le_pide_al_modelo():
    """Funcional: con el interruptor puesto, elegir Creatina tiene que producir una lista
    EXACTA, no el texto de «recomienda tú»."""
    from prompts.plan_generator import build_supplements_context

    base = {"includeSupplements": True, "mainGoal": "gain_muscle", "dietType": "balanced",
            "age": 21, "gender": "male"}
    vacio = build_supplements_context({**base, "selectedSupplements": []}) or ""
    elegido = build_supplements_context({**base, "selectedSupplements": ["creatine"]}) or ""

    assert vacio and elegido, "el contexto de suplementos vino vacío con el gate ENCENDIDO"
    assert elegido != vacio, (
        "elegir un suplemento no cambia el prompt: el usuario marca Creatina y el modelo "
        "recibe el mismo texto que si no hubiera marcado nada")
    assert "Creatina" in elegido, f"el suplemento elegido no llega al prompt: {elegido[:200]!r}"


def test_apagar_el_interruptor_PROHIBE_los_suplementos():
    """El caso contrario importa igual: sin opt-in, ni un gramo de suplemento."""
    from prompts.plan_generator import build_supplements_context

    fd = {"includeSupplements": False, "selectedSupplements": ["creatine"],
          "mainGoal": "gain_muscle"}
    txt = build_supplements_context(fd) or ""
    assert "Creatina" not in txt, (
        "con `includeSupplements=False` el prompt sigue nombrando un suplemento")


# --------------------------------------------- el país no puede colapsar en silencio
def test_el_pais_del_formulario_sobrevive_hasta_la_politica(monkeypatch):
    """Con el sistema de países ENCENDIDO —como está en producción— el país elegido debe
    llegar al mercado y a la cultura.

    Se fuerza el knob a propósito: sin él, `country_for_form_data` colapsa cualquier país
    a `DO` y esta comprobación mediría el default, no el contrato. Es la trampa que se
    cobró la auditoría del 09-sep.

    [P1-PLAN-LOTE-13 · 2026-09-12] Con `monkeypatch.setenv`, no con `os.environ[...] =`: la
    versión anterior dejaba el knob maestro ENCENDIDO para el resto del worker y además hacía
    `importlib.reload(constants)`. Eso volvía «flaky» a TRES tests de otros ficheros —la ruta USD
    del piso de presupuesto, la tortilla de maíz del guard de preparaciones y la identidad
    `go._RENAL_CONDITION_TERMS is constants.RENAL_CONDITION_TERMS`— que pasaban aislados y caían
    cuando este fichero corría antes en el mismo proceso. El knob se lee POR LLAMADA, así que
    el reload nunca hizo falta.
    """
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    from constants import country_for_form_data

    for cc in ("DO", "ES", "MX", "US", "CO", "PR"):
        assert country_for_form_data({"country": cc}) == cc, (
            f"el país {cc} del formulario se pierde antes de llegar al mercado")
