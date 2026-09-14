# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-37 · 2026-09-13] La cola lenta de la CI era Sentry sin DSN.

`--durations=30` (run 34787121743) nombró la cola: los 7 tests de `test_p1_arq27_f3_candidateset.py` pasaban
630-750 s CADA UNO en el setup (`build_blueprint`), 80 de los 88 min de la pata. Aislado, el mismo setup tarda
3,5 s. La diferencia la ponía `import app`: inicializa Sentry y, aunque sin DSN no envía nada, su integración
de logging convertía cada `logging.error` en un evento completo —serializando las variables locales de cada
marco— para tirarlo al final por falta de transporte (13-21 ms por evento). Sin base de datos,
`get_master_ingredients` registra un error POR LLAMADA y `build_blueprint` hace 32.564 llamadas.

Lo que este test fija: sin DSN, `sentry_sdk.init` no instala integraciones (con DSN, idéntico a antes); la suite
nunca tiene DSN (conftest lo vacía antes de que nadie cargue el entorno: sin `override`, una variable presente
gana); y un `logging.error` después de `import app` no construye ningún evento.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _app_src() -> str:
    return (_BACKEND / "app.py").read_text(encoding="utf-8")


def _bloque_init() -> str:
    m = re.search(r"sentry_sdk\.init\(\s*(.*?)\n\)", _app_src(), re.DOTALL)
    assert m, "no se encontró el bloque sentry_sdk.init(...) en app.py"
    return m.group(1)


def test_sin_dsn_sentry_no_instala_integraciones_y_con_dsn_todo_igual():
    src, blq = _app_src(), _bloque_init()
    assert re.search(r'^_SENTRY_DSN = \(os\.environ\.get\("SENTRY_DSN"\) or ""\)\.strip\(\) or None$', src, re.M)
    assert "dsn=_SENTRY_DSN," in blq
    assert "default_integrations=_SENTRY_DSN is not None," in blq
    assert "auto_enabling_integrations=_SENTRY_DSN is not None," in blq
    assert "tooltip-anchor: P1-PLAN-LOTE-37-SENTRY-SIN-DSN" in src
    # `test_p1_sentry_pii_scrubbing_backend` ejecuta aislado lo que hay entre los helpers y el init: la variable
    # que lee `os` tiene que vivir ANTES de los helpers o ese exec se queda sin `os`.
    assert src.index("_SENTRY_DSN = ") < src.index("_SENSITIVE_KEY_SUBSTRINGS = (")


def test_la_suite_nunca_tiene_dsn():
    assert os.environ.get("SENTRY_DSN") == "", "conftest vacía SENTRY_DSN: la suite no habla con Sentry"
    conftest = (_BACKEND / "tests" / "conftest.py").read_text(encoding="utf-8")
    i = conftest.index('_os_conftest.environ["SENTRY_DSN"] = ""')
    assert i < conftest.index("import db_core"), "debe fijarse antes de cualquier import que cargue el entorno"


def test_un_error_de_log_tras_import_app_no_construye_un_evento(monkeypatch):
    import sentry_sdk
    import sentry_sdk.client as sentry_client
    from sentry_sdk.integrations.logging import LoggingIntegration

    import app  # noqa: F401  — el import que inicializa Sentry

    cliente = sentry_sdk.get_client()
    assert not cliente.dsn
    assert cliente.get_integration(LoggingIntegration) is None
    eventos: list = []
    monkeypatch.setattr(sentry_client._Client, "_prepare_event", lambda self, *a, **k: eventos.append(1))
    logging.getLogger("p1_plan_lote_37").error("un error de prueba")
    assert eventos == [], "sin DSN, un error de log no debe construir un evento de Sentry"


def test_el_techo_de_la_ci_es_tres_veces_la_pata_mas_larga():
    """[P1-PLAN-LOTE-37 (bis) · 2026-09-14] Con la cola arreglada las patas cierran en 10,5 y 8,3 min (run
    34799738022): el techo baja de 120 a 30, tres veces la pata más larga. Si vuelve a subir, que sea con una
    medición al lado, no como margen a ciegas; y `--durations` se queda, para que la próxima cola tenga nombre."""
    ci = (_BACKEND / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    techos = re.findall(r"^\s+timeout-minutes:\s*(\d+)\s*$", ci, re.M)
    assert techos, "el job backend-tests debe declarar timeout-minutes"
    assert all(int(t) <= 30 for t in techos), f"techo {techos} min: con patas de 8-11 min, 30 es margen; más esconde una regresión"
    assert "P1-PLAN-LOTE-37 (bis)" in ci
    assert "--durations=30" in ci, "sin --durations la próxima cola no tiene nombre"


def test_docs_plan_marker():
    for doc in ("knobs_reference.md", "plan_pendientes_2026_09_11.md", "advisors_aceptados.md"):
        assert "P1-PLAN-LOTE-37" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _app_src())
    assert m and int(m.group(1)) >= 37
