"""[P2-DIARY-SLOTS · 2026-07-31] El día por sus FRANJAS, y sin horas inventadas.

Rediseño del cajón `DiaryHistory` a los 20 minutos de nacer. No fue por gusto:
la v1 dibujaba una LÍNEA HORARIA y la hora no existe.

`tools.log_consumed_meal` sella `consumed_at = now() - N días` al retrodatar, así
que una cena registrada a las 10:51 de la mañana salía como "10:51 · CENA".
Medido en la fila real del incidente:

    consumed = 2026-07-30 14:51:04.580099+00
    created  = 2026-07-31 14:51:04.597502+00

El mismo minuto, 24 h de diferencia: `consumed_at` es literalmente
`created_at - 1 día`. La hora que la v1 mostraba era el instante del REGISTRO.
*La señal de la que colgaba todo el diseño descansaba sobre un dato inventado.*

Lo que SÍ es real es la FRANJA — la nombró el usuario. Y al dibujar el día por
sus franjas, el enorme vacío bajo una sola comida deja de ser un problema de
espaciado y pasa a ser la respuesta a "¿qué me falta?": desayuno, almuerzo y
merienda están ahí, sin registro. *El hueco era información que la v1 se negaba
a mostrar.*
"""

from pathlib import Path

FRONT = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "components" / "dashboard"


def _src() -> str:
    return (FRONT / "DiaryHistory.jsx").read_text(encoding="utf-8")


def test_no_se_muestra_una_hora_RETRODATADA():
    """Se detecta comparando los DÍAS de ambas marcas.

    No hace falta epsilon: la pregunta es "¿lo anotaste otro día?", no "¿cuánto
    se parecen los relojes?".
    """
    src = _src()
    assert "horaFiable" in src, "no existe el guard de hora retrodatada"
    assert "aISO(creado) !== aISO(consumido)" in src, (
        "el guard no compara los DÍAS de consumed_at y created_at, así que no "
        "distingue una hora real de la del registro"
    )


def test_cuando_la_hora_no_es_fiable_se_dice_la_VERDAD():
    """Callar la hora no basta: el usuario merece saber por qué falta, y
    "lo anotaste el viernes" es a la vez cierto y útil."""
    src = _src()
    assert "Lo anotaste el" in src, (
        "sin la nota de retrodatado, una comida sin hora parece un dato perdido"
    )


def test_el_dia_se_dibuja_por_FRANJAS_incluidas_las_vacias():
    src = _src()
    for f in ("desayuno", "almuerzo", "merienda", "cena"):
        assert f"'{f}'" in src, f"la franja {f} no se dibuja"
    assert "Sin registro" in src, "las franjas vacías no se muestran"


def test_las_franjas_van_en_el_orden_del_DIA():
    """Con las horas fuera de juego, el orden lo tiene que dar la estructura.
    El enum del backend no sirve: es alfabético/arbitrario."""
    src = _src()
    orden = [src.index(f"'{f}'") for f in ("desayuno", "almuerzo", "merienda", "cena")]
    assert orden == sorted(orden), (
        "las franjas no están declaradas en orden cronológico del día"
    )


def test_el_total_lleva_su_objetivo_al_lado():
    """420 kcal es mucho o poco según el objetivo; suelto no dice nada y obliga
    a recordar el denominador de la card de hoy."""
    src = _src()
    assert "targetCalories" in src and "quotaOf" in src


def test_el_disparador_no_se_estira_a_todo_el_ancho():
    """`.mealsSection` es una COLUMNA FLEX.

    Un `inline-flex` dentro de ella se estira por el `align-items: stretch` por
    defecto, y el botón quedaba como un rectángulo vacío a todo el ancho con
    cuatro palabras en la esquina: pesaba mucho más de lo que vale y se leía
    como un campo de formulario. *El display del hijo no manda si el padre es
    flex.*
    """
    css = (FRONT / "DiaryHistory.module.css").read_text(encoding="utf-8")
    bloque = css.split(".trigger {")[1].split("}")[0]
    assert "align-self: flex-end" in bloque, (
        "sin `align-self`, el disparador se estira a todo el ancho de la card"
    )


def test_el_disparador_pesa_MENOS_que_la_accion_principal():
    """`.scanBtn` (Escanear comida) lleva borde: es la acción principal. Mirar
    el pasado es secundario y no debe competir — la jerarquía la marca el peso,
    no el tamaño."""
    css = (FRONT / "DiaryHistory.module.css").read_text(encoding="utf-8")
    bloque = css.split(".trigger {")[1].split("}")[0]
    assert "border: 1px solid transparent" in bloque, (
        "el disparador vuelve a llevar borde visible y compite con la acción "
        "principal de la card"
    )


def test_una_sola_gramatica_para_la_ausencia_de_dato():
    """Punteado = sin dato, en las DOS mitades del cajón: el riel de un día sin
    registro en la tira y la regla de una franja vacía. Si cada mitad inventara
    su propio código visual, el usuario tendría que aprender dos."""
    css = (FRONT / "DiaryHistory.module.css").read_text(encoding="utf-8")
    assert "dashed" in css.split(".railEmpty")[1][:220], "el riel vacío no es punteado"
    assert "dashed" in css.split(".slotEmpty .slotRule")[1][:220], (
        "la franja vacía no usa el mismo punteado que el riel"
    )
