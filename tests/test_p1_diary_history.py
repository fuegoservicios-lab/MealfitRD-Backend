"""[P1-DIARY-HISTORY · 2026-07-31] Ver el diario de días pasados.

El coach registra hacia atrás (`days_ago`): "cené dos panes" dicho por la mañana
va al diario de AYER. Pero la única superficie que mostraba el diario era la card
"Progreso en Tiempo Real", que es SOLO hoy.

Caso medido: el owner registró correctamente su cena de anoche, miró el panel en
cero y reportó "no se registró" — la fila estaba en `consumed_meals`, fechada el
día anterior a las 18:51 RD. *Un registro correcto que el usuario no puede ver es
indistinguible de uno que falló.*

`GET /consumed/{user_id}` ya aceptaba `date` arbitraria, así que el día concreto
no necesitaba backend nuevo. Lo que faltaba era el RESUMEN del rango: sin él la
tira de días solo puede pintar fechas y el usuario tiene que ir tocando una a una
para descubrir cuáles tienen registro. La alternativa (14 llamadas al endpoint de
un día) multiplica por 14 el round-trip para la misma información.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DIARY_PY = BACKEND / "routers" / "diary.py"
FRONT = BACKEND.parent / "frontend" / "src" / "components" / "dashboard"


def _fuente_endpoint() -> str:
    from routers.diary import api_get_consumed_range
    return inspect.getsource(api_get_consumed_range)


# ------------------------------------------------------- I2: el filtro user_id

def test_la_sql_filtra_por_user_id():
    """[I2] Toda lectura user-scoped filtra en SQL, no solo en el guard.

    El guard de ownership de arriba ya rechaza el IDOR, pero la invariante del
    repo exige las dos capas: si mañana alguien refactoriza el guard, el SQL
    sigue acotando. Un endpoint nuevo es exactamente donde se olvida.
    """
    src = _fuente_endpoint()
    assert re.search(r"WHERE\s+user_id\s*=\s*%s", src), (
        "P1-DIARY-HISTORY: la consulta de rango no filtra `WHERE user_id = %s`"
    )


def test_rechaza_leer_el_diario_de_otro_usuario():
    src = _fuente_endpoint()
    assert "verified_user_id != user_id" in src and "403" in src, (
        "P1-DIARY-HISTORY: falta el guard de ownership — cualquiera podría leer "
        "el resumen del diario ajeno pasando otro user_id en la URL"
    )


# --------------------------------------------------- entradas del cliente

def test_days_viene_clampado():
    """`days` lo elige el cliente. Sin tope, `days=99999` convierte un endpoint
    de UI en un escaneo de tabla."""
    src = _fuente_endpoint()
    assert re.search(r"min\(int\(days\),\s*90\)", src), (
        "P1-DIARY-HISTORY: `days` sin clamp superior"
    )


def test_el_offset_horario_esta_acotado_a_zonas_reales():
    src = _fuente_endpoint()
    assert "840" in src, (
        "P1-DIARY-HISTORY: `tzOffset` sin clamp. Viene del cliente y entra en "
        "`make_interval`; el rango real de zonas es ±14 h (840 min)."
    )


def test_agrupa_en_la_zona_del_usuario_no_en_UTC():
    """Agrupar en UTC partiría las cenas dominicanas en dos días: en RD (UTC-4)
    todo lo comido después de las 20:00 caería en el día siguiente — el mismo
    desfase que ya causó un bug de diario en esta base."""
    src = _fuente_endpoint()
    assert "make_interval(mins =>" in src, (
        "P1-DIARY-HISTORY: la agrupación por día no descuenta el offset del "
        "usuario, así que corta el día en UTC"
    )


def test_no_hay_interpolacion_de_strings_en_la_sql():
    """Los tres valores del cliente (offset, user_id, days) van como parámetros.

    Se comprueba sobre el AST y no con un grep de comillas: el objetivo es que
    la sentencia sea una constante literal, no que no aparezca ningún `%`.
    """
    arbol = ast.parse(DIARY_PY.read_text(encoding="utf-8"))
    objetivo = next(
        (n for n in ast.walk(arbol)
         if isinstance(n, ast.FunctionDef) and n.name == "api_get_consumed_range"),
        None,
    )
    assert objetivo is not None, "no encuentro `api_get_consumed_range`"
    for nodo in ast.walk(objetivo):
        if isinstance(nodo, ast.JoinedStr):  # f-string
            texto = ast.unparse(nodo)
            assert "consumed_meals" not in texto, (
                f"SQL construida con f-string: {texto[:90]}"
            )


# ------------------------------------------------------- la UI que lo consume

def test_el_drawer_existe_y_lo_abre_la_card_de_hoy():
    """El componente solo sirve si algo lo abre — y tiene que abrirse también
    con el día VACÍO, que es cuando el usuario necesita mirar atrás."""
    tp = (FRONT / "TrackingProgress.jsx").read_text(encoding="utf-8")
    assert "DiaryHistoryTrigger" in tp and "<DiaryHistory" in tp, (
        "la card de hoy no monta el drawer de días pasados"
    )
    # El disparador va DESPUÉS del ternario vacío/no-vacío: si quedara dentro
    # de la rama con comidas, desaparecería justo en el caso reportado.
    i_ternario = tp.index("Aún no registras comidas hoy")
    i_trigger = tp.index("<DiaryHistoryTrigger")
    assert i_trigger > i_ternario, (
        "el disparador quedó dentro de la rama con comidas: con el día en cero "
        "—el caso que originó esto— no habría forma de abrir el historial"
    )


def test_las_fechas_se_construyen_en_hora_LOCAL():
    """`toISOString()` da UTC: en RD convertiría 'hoy 21:00' en 'mañana'. Es el
    mismo desfase que causó que una cena se registrara en el día equivocado."""
    src = (FRONT / "DiaryHistory.jsx").read_text(encoding="utf-8")
    # Anclado al CÓDIGO, no al vocabulario: la propia explicación de por qué NO
    # se usa esa API contiene su nombre, y la primera versión de este test se
    # cayó con su propio comentario. El `not in` va contra la línea que DECIDE.
    codigo = "\n".join(
        l for l in src.splitlines()
        if not l.lstrip().startswith(("//", "*", "/*"))
    )
    assert ".toISOString(" not in codigo, (
        "P1-DIARY-HISTORY: se cuela una llamada a toISOString() — las fechas del "
        "selector deben construirse en hora local"
    )
    assert "getFullYear()" in src and "getMonth()" in src


def test_la_linea_del_dia_ordena_por_hora_de_CONSUMO():
    """El endpoint devuelve por `created_at` (orden de REGISTRO). Ordenar por eso
    rompe justo el caso que motivó la pantalla: una cena registrada al día
    siguiente aparecería fuera de sitio en su propia línea de tiempo."""
    src = (FRONT / "DiaryHistory.jsx").read_text(encoding="utf-8")
    assert "lista.sort(" in src and "consumed_at" in src, (
        "la línea del día no reordena por hora de consumo"
    )


def test_un_dia_sin_registro_se_distingue_de_uno_a_cero():
    css = (FRONT / "DiaryHistory.module.css").read_text(encoding="utf-8")
    assert ".railEmpty" in css and "dashed" in css, (
        "sin un estilo propio para el día sin registro, 'no comí nada anotado' y "
        "'no registré nada' se ven idénticos — y la ausencia de dato ES un dato"
    )
