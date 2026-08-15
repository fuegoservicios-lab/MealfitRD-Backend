"""[P2-PUSH-RESPECTS-PAUSE · 2026-08-15] Las notificaciones push del canal de
planes: ni avisan de un plan que el usuario apagó, ni llevan a una ruta muerta.

Dos defectos del MISMO canal, el que la auditoría del modo contador señaló como
su mayor hueco: «se revisaron 3 de 28 call sites; ahí puede haber más».

H17 · «TU PLAN PARECE ATRASADO» DESPUÉS DE PAUSAR
    `_detect_chronic_deferrals` agrupa `chunk_deferrals` de las últimas
    `CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS` y empuja un push. Las filas de un
    usuario que acaba de pausar siguen dentro de esa ventana, así que durante
    horas recibe en el teléfono que su plan «parece atrasado» y que «verifique su
    zona horaria» — de un plan que él apagó, y cuyos reintentos son el RESULTADO
    de la pausa, no una avería.

    La consulta ya existía bien resuelta en los cuatro crons que sí filtran (el
    nudge de zero-log, el freeze sweep, `_process_pending_shopping_lists`, el
    BG-refill). Se le añade a este el MISMO `NOT EXISTS … plan_mode='tracking'`:
    filtrar en SQL y no en Python es lo que evita construir en memoria una lista
    de destinatarios que luego hay que acordarse de podar.

H20 · UN DEEPLINK A UNA RUTA QUE NO EXISTE
    Un push apuntaba a `/shopping-list`. Las rutas reales del producto son
    `/dashboard/shopping` (y `/dashboard/pantry`): `/shopping-list` no está en
    `App.jsx`. Tocar la notificación abría la app en un 404 — el peor momento
    posible, porque el push ya había prometido «revisa tu lista de compras».

    Es el único de los 18 deeplinks del fichero que no apuntaba a `/dashboard`:
    un valor huérfano se detecta comparándolo con sus hermanos, no leyéndolo solo.

Tooltip-anchor: P2-PUSH-RESPECTS-PAUSE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"
_APP_JSX = (Path(__file__).resolve().parent.parent.parent
            / "frontend" / "src" / "App.jsx")


def _fuente() -> str:
    return _CRON.read_text(encoding="utf-8")


def _cuerpo(nombre: str) -> str:
    src = _fuente()
    i = src.find(f"\ndef {nombre}(")
    assert i >= 0, f"[P2-PUSH-RESPECTS-PAUSE] No existe {nombre}"
    j = src.find("\ndef ", i + 1)
    return re.sub(r"^\s*#.*$", "", src[i:j if j > 0 else len(src)], flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# H17 · el aviso de «plan atrasado» no alcanza a quien pausó
# ---------------------------------------------------------------------------

def test_los_deferrals_cronicos_excluyen_a_quien_esta_en_pausa():
    cuerpo = _cuerpo("_detect_chronic_deferrals")
    assert "plan_mode" in cuerpo, (
        "[P2-PUSH-RESPECTS-PAUSE] `_detect_chronic_deferrals` no mira `plan_mode`.\n"
        "Sus filas de `chunk_deferrals` siguen dentro de la ventana horas después de "
        "pausar, así que el usuario del contador recibe en el teléfono que su plan "
        "«parece atrasado» — de un plan que él apagó, y por unos reintentos que son "
        "el RESULTADO de la pausa."
    )


def test_el_filtro_va_en_SQL_y_no_en_python():
    """Filtrar en la consulta evita construir destinatarios que luego hay que podar."""
    cuerpo = _cuerpo("_detect_chronic_deferrals")
    i_sql = cuerpo.find("FROM chunk_deferrals")
    i_bucle = cuerpo.find("for user_id, row in")
    i_filtro = cuerpo.find("plan_mode")
    assert i_sql > 0 and i_bucle > 0, "No se encontró la consulta o el bucle de envío"
    assert i_filtro < i_bucle, (
        "[P2-PUSH-RESPECTS-PAUSE] El filtro de modo aparece DESPUÉS del bucle que "
        "despacha los push. Debe estar en la consulta: es el patrón que ya usan los "
        "cuatro crons que filtran bien."
    )


def test_el_patron_es_el_mismo_que_el_de_los_crons_que_ya_filtran():
    cuerpo = _cuerpo("_detect_chronic_deferrals")
    assert re.search(r"NOT EXISTS", cuerpo), (
        "[P2-PUSH-RESPECTS-PAUSE] Se esperaba el mismo `NOT EXISTS (… "
        "plan_mode='tracking')` que usan el nudge de zero-log y el freeze sweep. "
        "Un segundo dialecto para la misma pregunta es una divergencia futura."
    )


# ---------------------------------------------------------------------------
# H20 · ningún deeplink apunta a una ruta inexistente
# ---------------------------------------------------------------------------

def _rutas_reales() -> set[str]:
    if not _APP_JSX.exists():
        pytest.skip("App.jsx no disponible")
    return set(re.findall(r'path="(/[^"]*)"', _APP_JSX.read_text(encoding="utf-8")))


def test_todo_deeplink_de_push_apunta_a_una_ruta_que_existe():
    rutas = _rutas_reales()
    # Sin comentarios: un ejemplo en prosa no es un deeplink.
    codigo = re.sub(r"^\s*#.*$", "", _fuente(), flags=re.MULTILINE)
    enlaces = set(re.findall(r'url="(/[^"]*)"', codigo))
    assert enlaces, "[P2-PUSH-RESPECTS-PAUSE] No se encontró ningún deeplink; ¿cambió la forma?"

    huerfanos = sorted(
        u for u in enlaces
        if u.split("?")[0] not in rutas and not any(
            u.split("?")[0].startswith(r.rstrip("*")) for r in rutas if r.endswith("*")
        )
    )
    assert not huerfanos, (
        f"[P2-PUSH-RESPECTS-PAUSE] Deeplinks de push a rutas que NO existen en "
        f"App.jsx: {huerfanos}\n"
        "Tocar esa notificación abre la app en un 404, y justo después de haberle "
        "prometido algo al usuario. Las rutas reales de compras son "
        "`/dashboard/shopping` y `/dashboard/pantry`."
    )
