# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-658 · 2026-09-28] «Tu plan parece atrasado» solo le pide la zona horaria a quien no la tiene.

El aviso de aplazamientos crónicos (`reason='temporal_gate'`: el bloque anterior aún no termina) mandaba al usuario
«Detectamos N reintentos… Verifica que tu zona horaria esté correcta» — a la cuenta 0257d89d, con su zona horaria bien
puesta (240), le llegó una y otra vez entre el 24 y el 26-sep, y en español aunque su app está en francés: el cuerpo
era una f-string, que el catálogo de push no puede traducir. La causa real de esos aplazamientos era del sistema (la
deriva del ancla que corrige el lote 652), no algo que el usuario pudiera arreglar.

Ahora la alerta del operador (`chronic_deferrals:<user>`) se escribe siempre; el push SOLO cuando el perfil no tiene
zona horaria (lo único que el usuario sí puede arreglar abriendo la app), con texto fijo que está en el catálogo.

tooltip-anchor: P1-PLAN-LOTE-658
"""
import cron_tasks


def _preparar(monkeypatch, tz_del_perfil):
    pushes, alertas = [], []
    filas = [{"user_id": "u1", "meal_plan_id": "p1", "week_number": 3, "deferral_count": 42, "last_at": None,
              "tz_perfil": None if tz_del_perfil is None else str(tz_del_perfil)}]

    def consulta(q, p=None, **k):
        return filas if "FROM chunk_deferrals" in q else []

    monkeypatch.setattr(cron_tasks, "execute_sql_query", consulta)
    monkeypatch.setattr(cron_tasks, "execute_sql_write",
                        lambda q, p=None, **k: alertas.append(p[0]) if "INSERT INTO system_alerts" in q else None)
    monkeypatch.setattr(cron_tasks, "_dispatch_push_notification", lambda **k: pushes.append(k))
    monkeypatch.setattr(cron_tasks, "_resolve_cleared_chronic_deferral_alerts", lambda *a, **k: None)
    return pushes, alertas


def test_con_zona_horaria_solo_avisa_al_operador(monkeypatch):
    pushes, alertas = _preparar(monkeypatch, 240)
    cron_tasks._detect_chronic_deferrals()
    assert alertas == ["chronic_deferrals:u1"]
    assert pushes == []


def test_sin_zona_horaria_el_usuario_recibe_un_texto_traducible(monkeypatch):
    from push_i18n import push_catalog_keys
    pushes, alertas = _preparar(monkeypatch, None)
    cron_tasks._detect_chronic_deferrals()
    assert alertas == ["chronic_deferrals:u1"]
    assert len(pushes) == 1
    assert pushes[0]["title"] in push_catalog_keys() and pushes[0]["body"] in push_catalog_keys()
    assert "reintentos" not in pushes[0]["body"]


def test_la_zona_horaria_viene_en_la_misma_consulta():
    import inspect
    src = inspect.getsource(cron_tasks._detect_chronic_deferrals)
    assert "LEFT JOIN user_profiles up_tz ON up_tz.id = cd.user_id" in src
    assert "_get_user_tz_minutes_optional(" not in src
