"""[P1-GAMMA-NOTIF-REVIVE · 2026-07-28] La notificación de diversidad nació muerta.

## Lo que decía el log del VPS (23:50, tras el deploy del 27)

    [P0-gamma] Error enviando notif de diversidad para plan d48bbe7a… week 2:
    ImportError: cannot import name 'create_notification' from 'services'

`create_notification` **no existió nunca** en `services.py` (0 definiciones en el repo). Cada vez
que el gate de diversidad de nevera intentaba avisar al usuario, el ImportError caía en el except
y quedaba como una línea de ERROR — la notificación jamás se envió, desde el día en que se
escribió. Solo se vio 1 vez en 7 días de journal porque el camino (refill semana 2 con nevera
degradando la variedad) es raro; la suite lo pisaba y también lo tragaba.

## El arreglo

El mecanismo real del repo es `_dispatch_push_notification` (bg_executor acotado, fail-safe) — el
mismo que usan el zero-log CTA y los nudges de nevera. URL de la Nevera: `/dashboard/pantry`.

tooltip-anchor: P1-GAMMA-NOTIF-REVIVE
"""
from __future__ import annotations

import pathlib
import re

import cron_tasks

_SRC = pathlib.Path(cron_tasks.__file__).with_suffix(".py").read_text(encoding="utf-8")


def test_no_queda_ningun_import_del_fantasma():
    """`create_notification` no existe en services.py: cualquier import es un ImportError seguro."""
    assert "from services import create_notification" not in _SRC
    assert not re.search(r"\bcreate_notification\s*\(", _SRC), (
        "algún callsite sigue invocando la función fantasma"
    )


def test_services_confirma_que_la_funcion_no_existe():
    """Si algún día alguien la CREA en services, este test avisa para reevaluar el canal
    (hoy el push es el único mecanismo de notificación del repo)."""
    _sv = pathlib.Path(cron_tasks.__file__).parent / "services.py"
    assert "def create_notification" not in _sv.read_text(encoding="utf-8")


def test_la_notif_de_diversidad_usa_el_dispatch_real():
    i = _SRC.find("notif de diversidad")
    assert i != -1
    # el bloque del gate (hacia atrás hasta el try) debe despachar por el mecanismo vivo
    bloque = _SRC[max(0, i - 2200):i]
    assert "_dispatch_push_notification(" in bloque, (
        "el gate de diversidad debe usar _dispatch_push_notification — el import fantasma "
        "dejaba la notificación muerta desde su nacimiento"
    )
    assert '"/dashboard/pantry"' in bloque, "la notificación de nevera debe llevar a la Nevera"


def test_el_dispatch_existe_y_es_fail_safe():
    import inspect
    src = inspect.getsource(cron_tasks._dispatch_push_notification)
    assert "submit_bg_task" in src and "except Exception" in src
