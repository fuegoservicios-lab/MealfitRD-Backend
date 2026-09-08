# -*- coding: utf-8 -*-
"""[P2-ALERT-MESSAGE-REFRESH · 2026-09-08] El mensaje de la alerta se congelaba en la PRIMERA vez.

Las alertas se upsertean por `alert_key`. La mayoría de los emisores hacían:

    ON CONFLICT (alert_key) DO UPDATE
    SET triggered_at = NOW(), metadata = EXCLUDED.metadata, resolved_at = NULL

sin `message = EXCLUDED.message`. Resultado: en cada re-emisión la **metadata** se refresca y el
**mensaje que lee el operador se queda congelado en la primera ocurrencia, para siempre**.

## Cómo apareció, y lo que costó

Investigando `plan_display_i18n_degraded:fr-FR` (1 usuario real con la app en francés y el plan en
español), la alerta se contradecía a sí misma:

    message  → «Motivo: no_valid_meals. El plan 6b97a4e7…»   ← plan BORRADO
    metadata → {"reason": "no_meals", "plan_id": "3957a669…"} ← plan VIVO

Leí el mensaje, perseguí `6b97a4e7`, salió «PLAN BORRADO» y estuve a un paso de archivarlo como
deuda muerta. El plan realmente afectado seguía vivo. *Un diagnóstico que envejece sin avisar cuesta
más que no tenerlo: dirige la investigación al sitio equivocado con toda la confianza.*

## Medido antes de arreglar

58 upserts de alerta en los 9 ficheros que creí productores. **36 no refrescaban el mensaje** — pero
eso sólo hace daño si el mensaje lleva datos volátiles, así que se separó: **14 tienen mensaje
estático** (inocuo) y **19 interpolan `plan_id`, `chunk_id`, contadores o ventanas** — ésos mienten.

Se arreglaron los 32 alcanzables (uniformemente, también los estáticos: en ellos es un no-op y
evita tener que decidir caso por caso otra vez). Sin crecer una sola línea: la cláusula se apila en
la que ya existía, porque `graph_orchestrator.py` está a 2 líneas de su techo.

## Y los 4 que se me escaparon, por heredar una lista

«Los 9 ficheros productores» los tomé de `test_p2_audit_4_alert_keys_documented`, que es el SSOT de
las CLAVES documentadas — **no de los ficheros que emiten**. Escanear el backend da **13**, y los
cuatro que faltaban (`bg_executor`, `plan_jobs`, `rate_limiter`, `services`) tenían el bug intacto.

*Una lista heredada hereda también el límite de la pregunta que la creó.* La lista se deriva ahora
con un `rglob`, así que no hay que acordarse de actualizarla — el mismo hueco que
`P1-RAW-INDEX-INVENTORY-FICHEROS` cerró en el ratchet hermano, encontrado el mismo día por la misma
pregunta: *¿qué es lo que este guard NO vigila?*

Este test es lo que impide que el 37º nazca sin ella.
"""
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

# [P2-ALERT-MESSAGE-REFRESH-FICHEROS · 2026-09-08] La lista se DERIVA, no se hereda.
#
# La primera version tomaba los 9 ficheros de `test_p2_audit_4_alert_keys_documented` — que es el
# SSOT de las CLAVES documentadas, no de los ficheros que EMITEN. Escanear el backend da 13, y los
# 4 que faltaban (`bg_executor`, `plan_jobs`, `rate_limiter`, `services`) tenian el bug intacto:
# refrescaban `metadata` y no `message`. Mi arreglo de 32 sitios se quedo corto por confiar en una
# lista que respondia a OTRA pregunta.
#
# *Una lista heredada hereda tambien el limite de la pregunta que la creo.* Derivarla cuesta un
# `rglob` y no hay que acordarse de nada.
def _productores() -> list:
    fuera = []
    for path in sorted(_BACKEND.rglob("*.py")):
        rel = path.relative_to(_BACKEND).as_posix()
        if rel.startswith(("tests/", "scripts/", "migrations/")) or "site-packages" in rel:
            continue
        if "INSERT INTO system_alerts" in path.read_text(encoding="utf-8", errors="replace"):
            fuera.append(rel)
    return fuera


_PRODUCTORES = _productores()
_ANCLA = "ON CONFLICT (alert_key) DO UPDATE"
# Insensible al espaciado: `app.py` escribe `message=EXCLUDED.message` sin espacios y una comparación
# literal lo marcaba como incumplidor. Un detector que no tolera el estilo del repo acusa de un
# defecto que no existe, y esa acusación cuesta la confianza del test entero.
_REFRESCA_RX = re.compile(r"message\s*=\s*EXCLUDED\.message")


def _upserts_sin_refresco(path: Path):
    """Bloques `ON CONFLICT (alert_key) DO UPDATE` cuya cláusula SET no toca `message`."""
    if not path.exists():
        return []
    lineas = path.read_text(encoding="utf-8").split("\n")
    fuera = []
    for i, l in enumerate(lineas):
        if _ANCLA not in l:
            continue
        # La cláusula SET vive en las pocas líneas siguientes, antes de cerrar el string.
        ventana = "\n".join(lineas[i + 1:i + 9])
        cabeza = ventana.split('"""')[0].split('",')[0]
        if "SET" not in cabeza and "SET" not in l:
            continue
        if not _REFRESCA_RX.search(cabeza):
            fuera.append(i + 1)
    return fuera


@pytest.mark.parametrize("fichero", _PRODUCTORES)
def test_todo_upsert_de_alerta_refresca_el_mensaje(fichero):
    """Si esto falla, alguien añadió un emisor cuyo mensaje envejecerá sin avisar.

    El arreglo es una palabra: `message = EXCLUDED.message,` en la cláusula SET. Apílala en la línea
    que ya existe en vez de añadir una — `graph_orchestrator.py` vive pegado a su techo.
    """
    faltan = _upserts_sin_refresco(_BACKEND / fichero)
    assert not faltan, (
        f"{fichero}: upserts de alerta sin `message = EXCLUDED.message` en las líneas {faltan}. "
        "En la re-emisión la metadata se refresca y el mensaje se queda congelado en la PRIMERA "
        "ocurrencia — el operador lee datos viejos con toda la confianza.")


def test_el_detector_encuentra_de_verdad_un_upsert_sin_refresco(tmp_path):
    """Un test que no puede fallar no mide nada: se le da un fichero roto y debe verlo.

    (El primer intento de arreglo masivo insertó la cláusula FUERA de la comilla en los emisores que
    usan concatenación implícita de strings, y rompió la sintaxis. Por eso el detector mira la
    cláusula SET dentro del bloque y no el fichero entero.)
    """
    roto = tmp_path / "roto.py"
    roto.write_text(
        'execute_sql_write("""\n'
        '    INSERT INTO system_alerts (alert_key) VALUES (%s)\n'
        '    ON CONFLICT (alert_key) DO UPDATE\n'
        '    SET triggered_at = NOW(),\n'
        '        metadata = EXCLUDED.metadata,\n'
        '        resolved_at = NULL\n'
        '""", (k,))\n', encoding="utf-8")
    assert _upserts_sin_refresco(roto), "el detector no ve un upsert sin refresco"


def test_las_dos_formas_de_string_estan_cubiertas():
    """Los emisores usan bloque triple Y concatenación implícita de literales. Cubrir sólo una deja
    la mitad del contrato sin defender — y fue justo la forma que rompió mi primer intento."""
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert re.search(r'"SET triggered_at = NOW\(\), message = EXCLUDED\.message', src), \
        "no queda ningún upsert en concatenación implícita: revisa si el arreglo se perdió"
    assert re.search(r"SET triggered_at = NOW\(\),\n\s+message = EXCLUDED\.message", src), \
        "no queda ningún upsert en bloque triple: revisa si el arreglo se perdió"
