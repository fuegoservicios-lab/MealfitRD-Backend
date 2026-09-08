# -*- coding: utf-8 -*-
"""[P1-DETERMINISTIC-DAY-CANARY · 2026-09-08] Encenderlo para UNO antes que para todos.

El día determinista cambia **qué come** el usuario. Sin canario por usuario, encender el knob
cambia la dieta de todos a la vez y la única marcha atrás es otro despliegue — el `.env` de
producción vive en el VPS y el deploy no lo sube (`--exclude="backend/.env"`), así que el ciclo
«enciendo, miro, apago» es de minutos, no de segundos.

El repo ya tenía el patrón (`MEALFIT_PLAN_POLICY_ENFORCE_USERS`, «dueño → test → flip») y no
haberlo copiado era exactamente lo que separaba «funciona en mi medición» de «se puede poner
delante de usuarios».

**La identidad se lee de `form_data["user_id"]`, y eso está MEDIDO, no supuesto.** Casi lo doy por
inerte con una sonda rota: conté cuántos planes vivos llevan `user_id` dentro de
`plan_data->'form_data'` y salió 0 de 97 — pero `plan_data` no persiste `form_data` en ningún
plan, así que el 0 era vacío. La medida buena está en `pipeline_metrics`, que escribe ese mismo
campo en runtime: `generate_day_1` tiene 151 filas con `user_id` de 158 (las 7 sin él son
invitados). El cron ya lo estampa explícitamente para los bloques 2+.
"""
import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _src() -> str:
    return (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")


def test_el_canario_existe_y_es_el_que_decide():
    """`build_day_for_skeleton` debe consultar el CANARIO, no el knob global a secas.

    Si vuelve a llamar a `deterministic_day_enabled()` directamente, el estado del medio —el
    único que hace esto desplegable— deja de existir sin que nada falle.
    """
    src = _src()
    assert "def deterministic_day_for_user" in src
    cuerpo = src.split("def build_day_for_skeleton(")[1][:900]
    assert "deterministic_day_for_user(" in cuerpo, (
        "el punto de entrada no consulta el canario: encenderlo vuelve a ser todo-o-nada")
    assert "if not deterministic_day_enabled():" not in cuerpo, (
        "vuelve a decidir con el knob global a secas — el canario queda decorativo")


def test_la_identidad_sale_de_donde_el_pipeline_la_lleva():
    """`form_data["user_id"]` es la clave que el pipeline transporta de verdad.

    Medido en `pipeline_metrics`: 151 de 158 filas de `generate_day_1` traen `user_id`, y ese
    valor sale de `form_data.get("user_id")`. El cron lo estampa explícito para los bloques 2+
    (`cron_tasks.py`, los dos snapshots de refill y renovación). Cambiar esta clave por una
    inventada deja el canario sin poder identificar a nadie — inerte con toda la apariencia de
    estar enchufado.
    """
    cuerpo = _src().split("def build_day_for_skeleton(")[1][:900]
    assert '"user_id"' in cuerpo, "no lee `form_data['user_id']`: el canario no puede reconocer a nadie"


def test_los_tres_estados_del_canario():
    """Global ON ⇒ todos · global OFF + en la lista ⇒ sólo él · nada ⇒ nadie."""
    import os
    import deterministic_day as dd

    previos = (os.environ.pop("MEALFIT_DETERMINISTIC_DAY", None),
               os.environ.pop("MEALFIT_DETERMINISTIC_DAY_USERS", None))
    try:
        assert dd.deterministic_day_for_user("u1") is False, "sin nada encendido no entra nadie"
        assert dd.deterministic_day_for_user(None) is False

        os.environ["MEALFIT_DETERMINISTIC_DAY_USERS"] = "u1, U2 "
        assert dd.deterministic_day_for_user("u1") is True
        assert dd.deterministic_day_for_user("u2") is True, "la comparación normaliza caso y espacios"
        assert dd.deterministic_day_for_user("u3") is False, "un tercero NO debe entrar por la lista"
        assert dd.deterministic_day_for_user(None) is False, (
            "sin identidad no se puede afirmar que el usuario está en la lista: el invitado se "
            "queda fuera, que es el lado seguro")

        os.environ.pop("MEALFIT_DETERMINISTIC_DAY_USERS")
        os.environ["MEALFIT_DETERMINISTIC_DAY"] = "1"
        assert dd.deterministic_day_for_user("cualquiera") is True, "el knob global sigue mandando"
        assert dd.deterministic_day_for_user(None) is True, (
            "con el knob global encendido no hace falta identidad — si no, los invitados quedarían "
            "fuera de un flip que se declaró global")
    finally:
        for k, v in zip(("MEALFIT_DETERMINISTIC_DAY", "MEALFIT_DETERMINISTIC_DAY_USERS"), previos):
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_una_lista_vacia_no_abre_la_puerta():
    """`MEALFIT_DETERMINISTIC_DAY_USERS=""` debe ser «nadie», nunca «todos».

    Es el modo de fallo clásico de un split por env var: una cadena vacía se parte en `[""]`, el
    `in` casa contra un `user_id` vacío y el canario se vuelve global sin que nadie lo decida.
    """
    import os
    import deterministic_day as dd

    previos = (os.environ.pop("MEALFIT_DETERMINISTIC_DAY", None),
               os.environ.pop("MEALFIT_DETERMINISTIC_DAY_USERS", None))
    try:
        for vacia in ("", "   ", ",", " , ,"):
            os.environ["MEALFIT_DETERMINISTIC_DAY_USERS"] = vacia
            assert dd.deterministic_day_for_user("quien-sea") is False, (
                f"con la lista {vacia!r} entró alguien: una lista vacía debe cerrar, no abrir")
            assert dd.deterministic_day_for_user("") is False
    finally:
        for k, v in zip(("MEALFIT_DETERMINISTIC_DAY", "MEALFIT_DETERMINISTIC_DAY_USERS"), previos):
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_hacen_falta_DOS_knobs_y_esto_lo_deja_por_escrito():
    """`MEALFIT_DETERMINISTIC_DAY` SOLO no hace nada. Hace falta también la biblioteca.

    Lo descubrí perdiendo el tiempo: encendí el knob del día, medí, y salían 0 días construidos.
    La causa es que `construir_comida` devuelve `None` cuando no hay receta congelada —a
    propósito, porque sin ella no hay determinismo del TEXTO— y `recipe_for_dish_name` empieza con
    `if not library_select_enabled(): return None`, que es el knob `MEALFIT_RECIPE_LIBRARY_SELECT`.

    La dependencia es correcta; lo que estaba mal es que no estuviera escrita en ninguna parte. Un
    operador que encienda un knob y no vea cambio alguno concluye que la feature no sirve.
    """
    src = _src()
    assert "recipe_for_dish_name" in src
    assert "return None" in src.split("if not meal.get(\"recipe\")")[1][:200], (
        "sin receta congelada la comida debe devolver None — el LLM la hace")
    doc = (_BACKEND / "docs" / "deterministic_day.md")
    assert doc.exists(), "el SOP de encendido tiene que existir: dos knobs no se adivinan"
    t = doc.read_text(encoding="utf-8")
    assert "MEALFIT_DETERMINISTIC_DAY" in t and "MEALFIT_RECIPE_LIBRARY_SELECT" in t, (
        "el doc no nombra los DOS knobs: es justo el dato que me costó una medición en falso")
    assert "MEALFIT_DETERMINISTIC_DAY_USERS" in t, "ni el canario, que es la vía de encendido segura"


def test_el_modulo_sigue_compilando():
    ast.parse(_src())
