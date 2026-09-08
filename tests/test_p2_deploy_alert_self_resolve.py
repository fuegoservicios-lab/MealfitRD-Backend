# -*- coding: utf-8 -*-
"""[P2-DEPLOY-ALERT-SELF-RESOLVE · 2026-09-08] La doc prometía un resolver que no existía.

`docs/system_alerts_resolution_table.md` decía, para `deploy_lag_marker_stale` y
`deploy_lag_drift_vs_expected`,
que el resolver era «cron re-eval tras bump», modelo **Auto (implicit)**.

No era verdad. Las dos ramas de `_alert_deploy_lag_marker_stale` hacían sólo:

    INSERT INTO system_alerts ... ON CONFLICT (alert_key) DO UPDATE SET resolved_at = NULL

o sea **reabrir**. No había ninguna rama que cerrara la alerta cuando la condición desaparecía. Una
vez encendidas, se quedaban encendidas para siempre.

## Medido antes de tocar nada

El 08-sep, `deploy_lag_marker_stale` llevaba abierta **desde el 02-sep** — con el marcador bumpeado
tres veces ese mismo día y `drift=false` verificado tras cada deploy. La condición había
desaparecido hacía días y el aviso seguía en rojo.

*Un aviso que no puede apagarse deja de ser un aviso*, y la doc empeoraba el asunto: invitaba a
confiar en que se apagaba solo. Es la forma que `CLAUDE.md` ya nombra para otra cosa — «una
excepción documentada que no existe es peor que ninguna».

## Lo que NO se tocó

`temporal_gate_proactive:*` acumula 18 filas sin resolver desde julio y parecía el mismo caso.
**No lo es**: su resolver (`G12-TEMPORAL-GATE-RESOLVE`) está vivo — 2 de 20 resueltas, ambas
posteriores al fix del 30-may. Las 18 restantes son planes cuyo gate nunca llegó a pasar, que es la
conducta correcta. Se comprobó antes de acusar.

## Y un fallo de mi primer intento, que este test cazó

Cerré la señal B con la clave `deploy_drift` — que es el **`alert_type`**, no el `alert_key`. La
clave real es `deploy_lag_drift_vs_expected`, así que mi resolver actualizaba cero filas: un no-op
silencioso, la misma forma que llevo todo el día persiguiendo. Lo encontró el test de paridad
doc↔código, no yo.
"""
import ast
import inspect
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _fuente():
    return (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def test_el_resolver_existe_y_cierra_de_verdad():
    import cron_tasks

    src = inspect.getsource(cron_tasks._resolver_alerta_deploy)
    assert "UPDATE system_alerts SET resolved_at = NOW()" in src
    assert "resolved_at IS NULL" in src, (
        "sin el filtro, un re-cierre pisaría la marca de cuándo se resolvió la primera vez")


@pytest.mark.parametrize("clave", ["deploy_lag_marker_stale", "deploy_lag_drift_vs_expected"])
def test_las_dos_senales_llaman_al_resolver(clave):
    """Las dos ramas tienen que cerrar, no sólo la primera: eran dos INSERT simétricos y arreglar
    una sola habría dejado la otra encendida para siempre, que es el fallo original a medias."""
    import cron_tasks

    src = inspect.getsource(cron_tasks._alert_deploy_lag_marker_stale)
    assert f'_resolver_alerta_deploy("{clave}")' in src


def test_el_cierre_NUNCA_convive_con_el_insert_que_abre():
    """El cierre no puede estar en el mismo bloque que el INSERT que enciende la alerta.

    Si conviviera, cerraría la alerta que acaba de abrir y el aviso desaparecería justo cuando la
    condición SÍ existe — peor que no cerrarlo nunca.

    (Primera versión de este test marcaba cualquier llamada dentro de un `If.body` y suspendía al
    `elif`, que es un `ast.If` anidado en el `orelse` y por tanto la rama CORRECTA. Un test que no
    distingue `elif` de `if` acusa al código de su propio error de lectura.)
    """
    import cron_tasks

    arbol = ast.parse(inspect.getsource(cron_tasks._alert_deploy_lag_marker_stale))
    conflictos = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.If):
            continue
        for cuerpo in (nodo.body, nodo.orelse):
            txt = chr(10).join(ast.unparse(x) for x in cuerpo)
            if "_resolver_alerta_deploy" in txt and "INSERT INTO system_alerts" in txt:
                conflictos.append(txt[:80])
    assert not conflictos, f"el cierre convive con el INSERT que abre: {conflictos}"


def test_la_doc_y_el_codigo_ya_dicen_lo_mismo():
    """La tabla canónica prometía «cron re-eval tras bump». Ahora es cierto — y este test es lo que
    impide que vuelva a ser una promesa: si alguien quita el resolver, la doc queda mintiendo otra
    vez y esto falla antes."""
    doc = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    # `deploy_drift` es el alert_type; la CLAVE es `deploy_lag_drift_vs_expected`. Confundirlos hizo
    # que mi primera versión del arreglo cerrara una clave inexistente — un no-op silencioso.
    for clave in ("deploy_lag_marker_stale", "deploy_lag_drift_vs_expected"):
        fila = [l for l in doc.split("\n") if l.startswith(f"| `{clave}`")]
        assert fila, f"{clave} salió de la tabla canónica"
        assert "cron re-eval" in fila[0], f"{clave}: la doc ya no promete el resolver que existe"


def test_el_insert_sigue_reabriendo():
    """El `resolved_at = NULL` del UPSERT es load-bearing y NO se tocó: si la condición vuelve tras
    haberse cerrado, la alerta tiene que reabrirse. Cerrar y no poder reabrir es el fallo simétrico.
    """
    src = _fuente()
    assert src.count("resolved_at = NULL") >= 2


@pytest.mark.parametrize("respuesta,escribe", [
    ([{"alerta_abierta": 1}], True),    # la forma REAL de la DB viva
    ({"alerta_abierta": 1}, True),      # dict suelto (algunos mocks)
    ([], False),                        # nada abierto
    (None, False),
    ({"value": "otra cosa"}, False),    # respuesta a OTRA pregunta: no autoriza a cerrar
])
def test_acepta_la_forma_REAL_de_execute_sql_query(respuesta, escribe):
    """`execute_sql_query` devuelve una LISTA de filas contra la base viva.

    Mi primera versión sólo aceptaba `dict` y por eso salía por la puerta de atrás: el resolver era
    **inerte en producción** con los tests en verde, porque el mock del test hermano sí devuelve un
    dict. Lo cazó ejecutarlo contra la base real, no el mock.

    El último caso es el simétrico: una fila que responde a OTRA consulta no debe autorizar el
    cierre — por eso se pide una columna con nombre propio y se exige ESA.
    """
    from unittest.mock import MagicMock, patch

    import cron_tasks

    w = MagicMock()
    with patch("cron_tasks.execute_sql_query", return_value=respuesta), \
         patch("cron_tasks.execute_sql_write", w):
        cron_tasks._resolver_alerta_deploy("deploy_lag_marker_stale")
    assert w.called is escribe
