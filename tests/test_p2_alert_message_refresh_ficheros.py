# -*- coding: utf-8 -*-
"""[P2-ALERT-MESSAGE-REFRESH-FICHEROS · 2026-09-08] La lista de emisores se DERIVA, no se hereda.

`P2-ALERT-MESSAGE-REFRESH` arregló 32 upserts que refrescaban `metadata` y no `message`. Se quedó
corto en **4**, y la causa no fue el detector: fue la lista de ficheros.

«Los 9 ficheros productores» los tomé de `test_p2_audit_4_alert_keys_documented`, que es el SSOT de
las **claves documentadas** — no de los ficheros que EMITEN. Escanear el backend da **13**:

    agent.py  bg_executor.py  plan_jobs.py  rate_limiter.py  services.py

De esos cinco, **cuatro tenían el bug intacto**; sólo `agent.py` estaba bien.

*Una lista heredada hereda también el límite de la pregunta que la creó.*

Este fichero ancla la corrección estructural: que la lista salga de un `rglob` y no de la memoria de
nadie. Es el mismo cierre que `P1-RAW-INDEX-INVENTORY-FICHEROS` el mismo día — **los dos guards que
escribí hoy nacieron con el mismo hueco**, vigilar sólo donde ya sabía mirar, que es exactamente la
forma del defecto que ambos existen para impedir.
"""
import inspect
from pathlib import Path

import test_p2_alert_message_refresh as hermano

_BACKEND = Path(__file__).resolve().parents[1]


def _emisores_reales() -> set:
    fuera = set()
    for path in sorted(_BACKEND.rglob("*.py")):
        rel = path.relative_to(_BACKEND).as_posix()
        if rel.startswith(("tests/", "scripts/", "migrations/")) or "site-packages" in rel:
            continue
        if "INSERT INTO system_alerts" in path.read_text(encoding="utf-8", errors="replace"):
            fuera.add(rel)
    return fuera


def test_la_lista_del_guard_cubre_TODOS_los_emisores():
    """Si esto falla, un fichero empezó a emitir alertas y el guard del mensaje no lo mira.

    No se arregla añadiéndolo a mano a ninguna lista: la lista se deriva. Falla sólo si alguien
    vuelve a fijarla.
    """
    faltan = _emisores_reales() - set(hermano._PRODUCTORES)
    assert not faltan, f"emisores fuera del guard del mensaje: {sorted(faltan)}"


def test_la_lista_NO_esta_escrita_a_mano():
    """El arreglo estructural: `_PRODUCTORES` sale de recorrer el árbol, no de un literal.

    Con la lista fija, los 4 emisores nuevos habrían seguido con el mensaje congelado hasta que
    alguien se acordara — y acordarse es justo lo que falló.
    """
    src = inspect.getsource(hermano)
    assert "def _productores(" in src
    assert "rglob" in src
    assert '"cron_tasks.py", "db_inventory.py"' not in src, "volvió la lista literal"


def test_los_cuatro_que_se_escaparon_estan_arreglados():
    """Los que la lista heredada dejó fuera. Nombrados uno a uno: si mañana alguno vuelve a perder
    el refresco, el mensaje dice cuál sin tener que abrir el diff."""
    for fichero in ("bg_executor.py", "plan_jobs.py", "rate_limiter.py", "services.py"):
        faltan = hermano._upserts_sin_refresco(_BACKEND / fichero)
        assert not faltan, f"{fichero}: volvió a perder `message = EXCLUDED.message` en {faltan}"
