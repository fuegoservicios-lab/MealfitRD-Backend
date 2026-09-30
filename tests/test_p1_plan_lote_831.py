"""[P1-PLAN-LOTE-831 · 2026-09-29] Panel · Cuentas: la lista con actividad EN NÚMEROS, el CSV, la ficha ampliada
(actividad, ajustes y cuenta de prueba), las marcas de prueba (una o varias), el historial y el resumen de ajustes.

Spec docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md §4.1-§4.3, §7, §8, §13 (raíz del
workspace); contratos 1-8 del plan. Todo con bases falsas: nada de este fichero toca Neon.

Lo que cubre cada bloque (y por qué fallaría contra una versión a medias):
  1. la búsqueda escapa `%`, `_` y `\\` (si no, «%» listaría todas las cuentas);
  2. el SQL de la lista: filtros y orden de lista cerrada, paginación, las definiciones de §4.1 (hilos SSOT, escaneos,
     gasto, días activos) y el total en la misma consulta;
  3. la fila cumple el contrato (`FilaCuenta`): prueba, admin, plan efectivo con cortesía y sin ella;
  4. el CSV: BOM, cabecera, sin columnas de contenido, fórmulas neutralizadas, 5.000 filas como mucho;
  5. la ficha ampliada y su actividad (embudo, plataformas, extras), que carga aunque falle un bloque;
  6. el router: interruptor apagado ⇒ 404 y ficha idéntica a la del 774; admin, cabecera, rastro antes de responder
     y 503 si no se anota; los códigos de las marcas pasan tal cual; el lote fuera de `/cuentas/…`; limitadores.
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_cuentas as ac
import admin_cuentas_lista as acl
import ajustes_cuenta
import cuentas_prueba as cp
import routers.admin as ra
from auth import get_verified_user_id
from db import USER_CHAT_THREAD_IDS_SQL
from tests.test_p1_plan_lote_830 import _BD as _BDPrueba

_BACKEND = Path(__file__).resolve().parents[1]
ADMIN = "11111111-1111-1111-1111-111111111111"
ADMIN2 = "22222222-2222-2222-2222-222222222222"
UID = "33333333-3333-3333-3333-333333333333"
OTRO = "55555555-5555-5555-5555-555555555555"
NADIE = "77777777-7777-7777-7777-777777777777"
H = {"X-Admin-Accion": "1"}
T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
MOTIVO = "tester del beta cerrado"

_CLAVES_FILA = {"user_id", "email", "nombre", "alta", "plan_pagado", "plan_efectivo", "es_admin", "prueba",
                "actividad", "modo", "idioma", "pais"}
_CLAVES_ACTIVIDAD = {"ultima", "comidas_total", "comidas_30d", "planes", "mensajes_coach", "escaneos",
                     "gasto_ia_30d_usd", "dias_activos_30d"}
_CLAVES_ACTIVIDAD_FICHA = _CLAVES_ACTIVIDAD | {"comidas_por_dia_activo", "dias_con_agua_30d", "registros_peso",
                                               "bloques_fallidos_30d", "pulgares_abajo", "plataformas",
                                               "avisos_abiertos", "embudo", "modo", "idioma", "pais"}
_CLAVES_EMBUDO = {"alta", "formulario", "primer_plan", "primera_comida", "primer_mensaje", "primer_escaneo"}


def _fila(uid=UID, **k) -> dict:
    """Una fila tal como la devuelve la consulta de la lista (las columnas de `filas` + `total`)."""
    base = {"user_id": uid, "email": "ana@correo.com", "nombre": "Ana", "alta": T0 - timedelta(days=20),
            "plan_pagado": "gratis", "plan_mode": "plan", "locale": "es-DO", "pais": "DO",
            "prueba_desde": None, "prueba_aviso_visto_at": None,
            "comidas_total": 0, "comidas_30d": 0, "comidas_ultima": None, "planes": 0,
            "mensajes_coach": 0, "mensajes_ultima": None, "escaneos": 0, "micros_30d": 0, "escaneo_ultima": None,
            "peso_ultima": None, "agua_ultima": None,
            "dias_activos_30d": 0, "ultima": None, "total": 1}
    base.update(k)
    return base


class _BDLista:
    """Las consultas de `admin_cuentas_lista`. Entiende SOLO lo que el módulo emite: un SQL nuevo revienta aquí."""

    def __init__(self):
        self.filas: list = []
        self.total_aparte = 0
        self.cortesias: list = []
        self.cortesias_rotas = False
        self.correos: list = []
        self.correos_rotos = False
        self.extras: dict = {}
        self.extras_rotos = False
        self.consultas: list = []

    def query(self, q, p=None, fetch_one=False, fetch_all=False):
        q = " ".join(q.split())
        self.consultas.append((q, p))
        if q.startswith("WITH base AS"):
            if "SELECT count(*) AS n FROM filas" in q:
                return {"n": self.total_aparte}
            limite, desde = p[-2], p[-1]
            return [dict(f) for f in self.filas[desde:desde + limite]]
        if "FROM public.account_grants" in q:
            if self.cortesias_rotas:
                raise RuntimeError("account_grants no existe")
            return list(self.cortesias)
        if q.startswith("SELECT id::text AS id, email FROM public.user_profiles WHERE id = ANY"):
            if self.correos_rotos:
                raise RuntimeError("sin DB")
            return [c for c in self.correos if c["id"] in p[0]]
        if q.startswith("SELECT (SELECT count(*) FROM public.water_intake_log"):
            if self.extras_rotos:
                raise RuntimeError("push_subscriptions no existe")
            return dict(self.extras)
        raise AssertionError(f"consulta que el fake no conoce: {q[:140]}")

    def de_la_lista(self):
        return [(q, p) for q, p in self.consultas if q.startswith("WITH base AS") and "count(*) AS n" not in q]

    def pagina(self):
        paginas = self.de_la_lista()
        assert paginas, "no hubo consulta de la lista"
        return paginas[-1]


@pytest.fixture
def bd(monkeypatch):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    monkeypatch.delenv("MEALFIT_ACCOUNT_GRANTS", raising=False)
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", ADMIN)
    b = _BDLista()
    monkeypatch.setattr(acl, "execute_sql_query", b.query)
    return b


# ═════════════════════════════════════════════ 1. la búsqueda
@pytest.mark.parametrize("entrada,patron", [
    ("ana", "%ana%"),
    ("  Ana@Correo.com ", "%Ana@Correo.com%"),
    ("50%", "%50\\%%"),
    ("a_b", "%a\\_b%"),
    ("c:\\x", "%c:\\\\x%"),
    ("%", "%\\%%"),
    ("_", "%\\_%"),
    ("\\", "%\\\\%"),
    ("\\%_", "%\\\\\\%\\_%"),
])
def test_el_patron_de_busqueda_escapa_los_comodines(entrada, patron):
    # Review Focus 3: «%», «_» o «\» literales no pueden listar todas las cuentas.
    assert acl.patron_de_busqueda(entrada) == patron


@pytest.mark.parametrize("vacio", ["", "   ", None])
def test_sin_texto_no_hay_busqueda(vacio):
    assert acl.patron_de_busqueda(vacio) is None


def test_la_busqueda_va_como_parametro_sobre_correo_y_nombre_con_escape(bd):
    acl.listar("50%_x", "actividad", "todas", 1)
    q, p = bd.pagina()
    assert "(p.email ILIKE %s ESCAPE '\\' OR p.full_name ILIKE %s ESCAPE '\\')" in q
    assert p[:2] == ("%50\\%\\_x%", "%50\\%\\_x%")
    assert "50%" not in q, "el texto del cliente nunca se interpola en el SQL"


def test_sin_busqueda_no_hay_ilike_ni_parametros_de_busqueda(bd):
    acl.listar("", "actividad", "todas", 1)
    q, p = bd.pagina()
    assert "ILIKE" not in q and p == (50, 0)


# ═════════════════════════════════════════════ 2. el SQL de la lista
_FILTROS = {
    "todas": "FROM filas WHERE TRUE ORDER BY",
    "prueba": "FROM filas WHERE prueba_desde IS NOT NULL ORDER BY",
    "sin_marcar": "FROM filas WHERE prueba_desde IS NULL ORDER BY",
    "activas_7d": "FROM filas WHERE ultima >= now() - interval '7 days' ORDER BY",
    "inactivas_14d": "FROM filas WHERE COALESCE(ultima, alta) < now() - interval '14 days' ORDER BY",
    "con_plan": "FROM filas WHERE planes > 0 ORDER BY",
    "seguimiento": "FROM filas WHERE plan_mode = 'tracking' ORDER BY",
}
_ORDENES = {
    "actividad": "ORDER BY ultima DESC NULLS LAST, alta DESC, user_id LIMIT %s OFFSET %s",
    "alta": "ORDER BY alta DESC, user_id LIMIT %s OFFSET %s",
    "comidas": "ORDER BY comidas_total DESC, ultima DESC NULLS LAST, user_id LIMIT %s OFFSET %s",
    "gasto": "ORDER BY micros_30d DESC, ultima DESC NULLS LAST, user_id LIMIT %s OFFSET %s",
}


@pytest.mark.parametrize("filtro", sorted(_FILTROS))
def test_cada_filtro_llega_al_sql(bd, filtro):
    acl.listar("", "actividad", filtro, 1)
    assert _FILTROS[filtro] in bd.pagina()[0]


@pytest.mark.parametrize("orden", sorted(_ORDENES))
def test_cada_orden_llega_al_sql(bd, orden):
    acl.listar("", orden, "todas", 1)
    assert _ORDENES[orden] in bd.pagina()[0]


def test_filtros_y_ordenes_son_los_del_contrato():
    assert set(acl.FILTROS) == set(_FILTROS) and set(acl.ORDENES) == set(_ORDENES)


@pytest.mark.parametrize("orden,filtro", [("x", "todas"), ("actividad", "x"), ("alta; DROP", "todas")])
def test_orden_o_filtro_fuera_de_la_lista_no_llegan_al_sql(bd, orden, filtro):
    with pytest.raises(ValueError):
        acl.listar("", orden, filtro, 1)
    with pytest.raises(ValueError):
        acl.exportar_csv("", orden, filtro)
    assert bd.consultas == []


def test_la_paginacion_es_de_50(bd):
    bd.filas = [_fila(f"{i:08d}-0000-0000-0000-000000000000", total=120) for i in range(120)]
    r = acl.listar("", "alta", "todas", 3)
    assert bd.pagina()[1] == (50, 100)
    assert (r["pagina"], r["por_pagina"], r["total"], len(r["cuentas"])) == (3, 50, 120, 20)


def test_el_total_sale_de_la_misma_consulta(bd):
    bd.filas = [_fila(total=7)]
    assert acl.listar("", "actividad", "todas", 1)["total"] == 7
    assert "count(*) OVER () AS total" in bd.pagina()[0]
    assert len(bd.de_la_lista()) == 1 and not any("count(*) AS n FROM filas" in q for q, _ in bd.consultas)


def test_una_pagina_mas_alla_del_final_cuenta_el_total_aparte(bd):
    bd.total_aparte = 3
    r = acl.listar("ana", "actividad", "prueba", 5)
    assert r == {"cuentas": [], "total": 3, "pagina": 5, "por_pagina": 50}
    q, p = next((q, p) for q, p in bd.consultas if "SELECT count(*) AS n FROM filas" in q)
    assert "FROM filas WHERE prueba_desde IS NOT NULL" in q and p == ("%ana%", "%ana%")


def test_la_primera_pagina_vacia_es_total_cero_sin_otra_consulta(bd):
    assert acl.listar("", "actividad", "todas", 1) == {"cuentas": [], "total": 0, "pagina": 1, "por_pagina": 50}
    assert len(bd.consultas) == 1


def test_las_definiciones_de_la_actividad_son_las_del_spec(bd):
    """§4.1: comidas = consumed_meals; planes = meal_plans; mensajes = role 'user' en los HILOS de la cuenta (el SSOT,
    que incluye la respuesta del coach con user_id NULL); escaneos = vision_scan; gasto = cost_usd_micros en 30 d;
    día activo = día UTC con comida o mensaje; última = lo más reciente que HIZO la persona (ver el test siguiente)."""
    acl.listar("", "actividad", "todas", 1)
    q = bd.pagina()[0]
    hilos = " ".join(USER_CHAT_THREAD_IDS_SQL.replace("%s", "p.id").split())
    assert q.count(f"m.role = 'user' AND (m.user_id = p.id OR m.session_id::text IN ({hilos}))") == 2, (
        "mensajes y días activos: lo que escribió la persona, en sus hilos o con su id")
    assert "FROM public.consumed_meals c WHERE c.user_id = p.id" in q
    assert "FROM public.meal_plans x WHERE x.user_id = p.id" in q
    assert "count(*) FILTER (WHERE e.node = 'vision_scan') AS escaneos" in q
    assert "sum(e.cost_usd_micros) FILTER (WHERE e.created_at >= now() - interval '30 days')" in q
    assert "(c.consumed_at AT TIME ZONE 'UTC')::date" in q and "(m.created_at AT TIME ZONE 'UTC')::date" in q
    assert "GREATEST(comidas_ultima, mensajes_ultima, escaneo_ultima, peso_ultima, agua_ultima) AS ultima" in q
    assert "FROM public.cuentas_de_prueba t WHERE t.user_id = p.id AND t.quitada_at IS NULL" in q
    for contenido in ("meal_name", "ingredients", "content", "health_profile ->> 'weight'", "attachments"):
        assert contenido not in q, f"la lista no lee contenido: {contenido}"


def test_la_ultima_actividad_es_solo_lo_que_hace_la_persona(bd):
    """Ruling del controlador (29-sep): «última actividad» = solo acciones de la PERSONA — comidas, sus mensajes al
    coach, escaneos, peso y agua. El chunk worker atribuye a la cuenta la IA que corre en segundo plano
    (`llm_usage_events.user_id`): contarla dejaba «activas» a cuentas abandonadas. Es el SQL, y el SQL alimenta a la vez
    la columna `ultima`, el orden `actividad` y los filtros `activas_7d` / `inactivas_14d`."""
    acl.listar("", "actividad", "todas", 1)
    q = bd.pagina()[0]
    # las cinco fuentes, y solo ellas, en el GREATEST
    m = re.search(r"GREATEST\(([^)]*)\) AS ultima", q)
    assert m and [x.strip() for x in m.group(1).split(",")] == [
        "comidas_ultima", "mensajes_ultima", "escaneo_ultima", "peso_ultima", "agua_ultima"]
    for fuera in ("ia_ultima", "planes_ultima"):
        assert fuera not in q, f"{fuera}: no es una acción de la persona"
    # el escaneo: SOLO `vision_scan`; el resto de `llm_usage_events` no marca actividad…
    assert "max(e.created_at) FILTER (WHERE e.node = 'vision_scan') AS escaneo_ultima" in q
    assert q.count("max(e.created_at)") == 1, "ningún `max` sobre TODOS los eventos de IA"
    # …pero el gasto sigue siendo el de TODOS los eventos (el dinero sale igual)
    assert ("COALESCE(sum(e.cost_usd_micros) FILTER (WHERE e.created_at >= now() - interval '30 days'), 0) "
            "AS micros_30d") in q, "el gasto: todos los eventos, sin filtro de `node`"
    # peso y agua: sus tablas, por cuenta, con su marca de tiempo
    assert "SELECT max(w.created_at) AS ultima FROM public.weight_log w WHERE w.user_id = p.id" in q
    assert "SELECT max(a.updated_at) AS ultima FROM public.water_intake_log a WHERE a.user_id = p.id" in q
    # y la lista, el orden y los filtros hablan de esa `ultima` (no de otra columna)
    assert "ultima DESC NULLS LAST" in _ORDENES["actividad"]
    assert "ultima >= now() - interval '7 days'" in _FILTROS["activas_7d"]
    assert "COALESCE(ultima, alta) < now() - interval '14 days'" in _FILTROS["inactivas_14d"]
    # la ficha usa la MISMA consulta, así que hereda la misma definición
    bd.consultas.clear()
    bd.filas = [_fila()]
    bd.extras = dict(_EXTRAS)
    acl.actividad_de(UID)
    assert "GREATEST(comidas_ultima, mensajes_ultima, escaneo_ultima, peso_ultima, agua_ultima) AS ultima" in bd.pagina()[0]


def test_la_ultima_actividad_sale_tal_cual_de_la_consulta(bd):
    """`ultima` no se recalcula en Python: es la columna del SQL (una cuenta sin acciones suyas queda en None)."""
    bd.filas = [_fila(ultima=T0 - timedelta(days=3), comidas_ultima=T0 - timedelta(days=3)), _fila(OTRO, ultima=None)]
    con, sin = acl.listar("", "actividad", "todas", 1)["cuentas"]
    assert con["actividad"]["ultima"] == (T0 - timedelta(days=3)).isoformat()
    assert sin["actividad"]["ultima"] is None


# ═════════════════════════════════════════════ 3. la fila (FilaCuenta)
def test_la_fila_cumple_el_contrato(bd):
    bd.filas = [_fila(comidas_total=57, comidas_30d=12, planes=2, mensajes_coach=9, escaneos=3, micros_30d=1234567,
                      dias_activos_30d=6, ultima=T0, prueba_desde=T0 - timedelta(days=1))]
    r = acl.listar("", "actividad", "todas", 1)
    f = r["cuentas"][0]
    assert set(f) == _CLAVES_FILA and set(f["actividad"]) == _CLAVES_ACTIVIDAD
    assert f["actividad"] == {"ultima": T0.isoformat(), "comidas_total": 57, "comidas_30d": 12, "planes": 2,
                              "mensajes_coach": 9, "escaneos": 3, "gasto_ia_30d_usd": 1.23, "dias_activos_30d": 6}
    assert f["prueba"] == {"estado": "aviso_pendiente", "desde": (T0 - timedelta(days=1)).isoformat()}
    assert (f["user_id"], f["email"], f["nombre"], f["alta"]) == (UID, "ana@correo.com", "Ana",
                                                                  (T0 - timedelta(days=20)).isoformat())
    assert (f["plan_pagado"], f["plan_efectivo"], f["es_admin"]) == ("gratis", "gratis", False)
    assert (f["modo"], f["idioma"], f["pais"]) == ("plan", "es-DO", "DO")
    json.dumps(r)                                          # serializable tal cual


def test_el_estado_de_la_prueba_es_el_de_exigir_prueba(bd, monkeypatch):
    bd.filas = [_fila(prueba_desde=T0, prueba_aviso_visto_at=T0 + timedelta(hours=1)), _fila(OTRO, prueba_desde=T0)]
    uno, otro = acl.listar("", "actividad", "todas", 1)["cuentas"]
    assert uno["prueba"]["estado"] == "activa" and otro["prueba"]["estado"] == "aviso_pendiente"
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", "false")
    otro = acl.listar("", "actividad", "todas", 1)["cuentas"][1]
    assert otro["prueba"]["estado"] == "activa", "sin exigir el aviso, la marca ya abre el detalle"


def test_sin_datos_los_numeros_son_cero_y_lo_opcional_null(bd):
    bd.filas = [_fila(comidas_total=None, micros_30d=None, plan_mode=None, locale=None, pais=None, plan_pagado=None)]
    f = acl.listar("", "actividad", "todas", 1)["cuentas"][0]
    assert f["actividad"]["comidas_total"] == 0 and f["actividad"]["gasto_ia_30d_usd"] == 0.0
    assert f["actividad"]["ultima"] is None and f["prueba"] is None
    assert (f["modo"], f["idioma"], f["pais"], f["plan_pagado"]) == ("plan", None, None, "gratis")


def test_seguimiento_es_el_modo_tracking(bd):
    bd.filas = [_fila(plan_mode="tracking")]
    assert acl.listar("", "actividad", "todas", 1)["cuentas"][0]["modo"] == "tracking"


def test_la_lista_ensena_a_los_admin_con_su_etiqueta(bd, monkeypatch):
    # El resumen de ajustes los excluye; la LISTA los enseña marcados (tier admin o la lista del .env del panel).
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", f"{ADMIN2.upper()}")
    bd.filas = [_fila(ADMIN, plan_pagado="admin"), _fila(ADMIN2), _fila(UID)]
    admin, del_env, persona = acl.listar("", "actividad", "todas", 1)["cuentas"]
    assert admin["es_admin"] and admin["plan_efectivo"] == "admin"
    assert del_env["es_admin"] and not persona["es_admin"]


def test_el_plan_efectivo_superpone_la_cortesia_vigente(bd):
    bd.filas = [_fila(UID), _fila(OTRO, plan_pagado="ultra")]
    bd.cortesias = [{"user_id": UID, "plan": "plus"}, {"user_id": OTRO, "plan": "basic"}]
    uno, otro = acl.listar("", "actividad", "todas", 1)["cuentas"]
    assert (uno["plan_pagado"], uno["plan_efectivo"]) == ("gratis", "plus")
    assert (otro["plan_pagado"], otro["plan_efectivo"]) == ("ultra", "ultra"), "lo pagado nunca baja"
    q, p = next((q, p) for q, p in bd.consultas if "FROM public.account_grants" in q)
    assert "DISTINCT ON (g.user_id)" in q and "g.kind = 'plan'" in q and "ORDER BY g.user_id, g.created_at DESC" in q
    assert "revoked_at IS NULL AND starts_at <= now() AND (ends_at IS NULL OR ends_at > now())" in q
    assert sorted(p[0]) == sorted([UID, OTRO]) and sorted(p[1]) == ["basic", "plus", "ultra"]


def test_sin_regalos_legibles_o_apagados_queda_lo_pagado(bd, monkeypatch):
    bd.filas = [_fila(UID)]
    bd.cortesias = [{"user_id": UID, "plan": "plus"}]
    bd.cortesias_rotas = True
    assert acl.listar("", "actividad", "todas", 1)["cuentas"][0]["plan_efectivo"] == "gratis"
    bd.cortesias_rotas = False
    monkeypatch.setenv("MEALFIT_ACCOUNT_GRANTS", "false")
    bd.consultas.clear()
    assert acl.listar("", "actividad", "todas", 1)["cuentas"][0]["plan_efectivo"] == "gratis"
    assert not any("account_grants" in q for q, _ in bd.consultas), "apagados, ni se consultan"


# ═════════════════════════════════════════════ 4. el CSV
def _ajustes_de_varias(monkeypatch, por_uid):
    vistos = []

    def _falso(uids):
        vistos.append(list(uids))
        return {u: por_uid[u] for u in uids if u in por_uid}
    monkeypatch.setattr(ajustes_cuenta, "ajustes_de_varias", _falso)
    return vistos


def test_el_csv_lleva_bom_cabecera_y_una_fila_por_cuenta(bd, monkeypatch):
    bd.filas = [_fila(UID, comidas_total=5, micros_30d=2500000, prueba_desde=T0), _fila(OTRO, nombre="Beto")]
    vistos = _ajustes_de_varias(monkeypatch, {UID: {"ajustes": [
        {"clave": "water_tracker_enabled", "estado": "apagado", "valor": False},
        {"clave": "locale", "estado": "valor", "valor": "en-US"}], "ajustes_dispositivo": {}}})
    texto, n = acl.exportar_csv("", "actividad", "todas")
    assert texto.startswith("\ufeff") and n == 2 and vistos == [[UID, OTRO]]
    filas = list(csv.reader(io.StringIO(texto[1:])))
    cabecera = [*acl.COLUMNAS_FILA, *ajustes_cuenta.columnas_csv()]
    assert filas[0] == cabecera and len(filas) == 3
    uno = dict(zip(cabecera, filas[1]))
    assert (uno["user_id"], uno["actividad.comidas_total"], uno["actividad.gasto_ia_30d_usd"]) == (UID, "5", "2.50")
    assert (uno["prueba.estado"], uno["es_admin"]) == ("aviso_pendiente", "no")
    assert (uno["ajuste.water_tracker_enabled"], uno["ajuste.locale"]) == ("apagado", "en-US")
    otro = dict(zip(cabecera, filas[2]))
    assert otro["nombre"] == "Beto" and otro["prueba.estado"] == "" and otro["ajuste.locale"] == ""
    assert acl.csv_de("", "actividad", "todas") == texto


def test_las_columnas_del_csv_son_la_fila_aplanada_sin_contenido():
    assert acl.COLUMNAS_FILA == (
        "user_id", "email", "nombre", "alta", "plan_pagado", "plan_efectivo", "es_admin", "prueba.estado",
        "prueba.desde", "actividad.ultima", "actividad.comidas_total", "actividad.comidas_30d", "actividad.planes",
        "actividad.mensajes_coach", "actividad.escaneos", "actividad.gasto_ia_30d_usd", "actividad.dias_activos_30d",
        "modo", "idioma", "pais")
    columnas = [c.lower() for c in (*acl.COLUMNAS_FILA, *ajustes_cuenta.columnas_csv())]
    for contenido in ("plato", "meal", "ingred", "texto", "content", "health", "alerg", "dieta", "condicion",
                      "medica", "foto", "attachment", "motivo"):
        assert not any(contenido in c for c in columnas), f"el CSV nunca lleva contenido: {contenido}"
    # La unidad de peso es un ajuste; el peso, la altura o la edad son el perfil de salud (spec §13.1).
    salud = {"weight", "peso", "height", "altura", "age", "edad", "gender", "sexo", "allergies", "medications"}
    assert not salud & {c.split(".")[-1] for c in columnas}


def test_el_csv_neutraliza_las_formulas_de_hoja_de_calculo(bd, monkeypatch):
    bd.filas = [_fila(UID, nombre="=HYPERLINK(\"http://x\")", email="+1@x.com"), _fila(OTRO, nombre="@SUM(A1)")]
    _ajustes_de_varias(monkeypatch, {})
    filas = list(csv.reader(io.StringIO(acl.csv_de("", "actividad", "todas")[1:])))
    cab = filas[0]
    assert filas[1][cab.index("nombre")] == "'=HYPERLINK(\"http://x\")"
    assert filas[1][cab.index("email")] == "'+1@x.com" and filas[2][cab.index("nombre")] == "'@SUM(A1)"


def test_el_csv_pide_como_mucho_5000_filas_con_los_mismos_filtros(bd, monkeypatch):
    _ajustes_de_varias(monkeypatch, {})
    acl.exportar_csv("ana_", "gasto", "con_plan")
    q, p = bd.pagina()
    assert p == ("%ana\\_%", "%ana\\_%", 5000, 0) and acl.MAX_CSV == 5000
    assert _FILTROS["con_plan"] in q and _ORDENES["gasto"] in q


def test_si_los_ajustes_fallan_no_hay_csv(bd, monkeypatch):
    bd.filas = [_fila(UID)]

    def _roto(uids):
        raise RuntimeError("ajustes_dispositivo no existe")
    monkeypatch.setattr(ajustes_cuenta, "ajustes_de_varias", _roto)
    with pytest.raises(RuntimeError):
        acl.exportar_csv("", "actividad", "todas")


def test_un_csv_vacio_es_solo_la_cabecera(bd, monkeypatch):
    monkeypatch.setattr(ajustes_cuenta, "ajustes_de_varias", lambda uids: pytest.fail("sin cuentas no se piden"))
    texto, n = acl.exportar_csv("", "actividad", "prueba")
    assert n == 0 and len(list(csv.reader(io.StringIO(texto[1:])))) == 1


# ═════════════════════════════════════════════ 5. la ficha ampliada
_EXTRAS = {"dias_con_agua_30d": 5, "registros_peso": 2, "bloques_fallidos_30d": 1, "pulgares_abajo": 3,
           "primer_mensaje": T0 - timedelta(days=3), "primer_plan": T0 - timedelta(days=10),
           "primera_comida": T0 - timedelta(days=9), "primer_escaneo": None,
           "formulario_enviado": T0 - timedelta(days=11), "avisos_abiertos": 2, "plataformas_push": ["ios"],
           "suscripciones_web": 0, "plataformas_consentimiento": ["android", None, "otra"],
           "plataformas_dispositivo": ["web", "raro"]}


def test_la_actividad_de_la_ficha(bd):
    bd.filas = [_fila(comidas_30d=12, dias_activos_30d=4, plan_mode="tracking", locale="en-US", pais="US")]
    bd.extras = dict(_EXTRAS)
    a = acl.actividad_de(UID)
    assert set(a) == _CLAVES_ACTIVIDAD_FICHA and set(a["embudo"]) == _CLAVES_EMBUDO
    assert a["comidas_por_dia_activo"] == 3.0
    assert (a["dias_con_agua_30d"], a["registros_peso"], a["bloques_fallidos_30d"], a["pulgares_abajo"],
            a["avisos_abiertos"]) == (5, 2, 1, 3, 2)
    assert a["plataformas"] == ["web", "ios", "android"], "vistas, en orden fijo y solo las tres conocidas"
    assert a["embudo"] == {
        "alta": (T0 - timedelta(days=20)).isoformat(), "formulario": (T0 - timedelta(days=11)).isoformat(),
        "primer_plan": (T0 - timedelta(days=10)).isoformat(), "primera_comida": (T0 - timedelta(days=9)).isoformat(),
        "primer_mensaje": (T0 - timedelta(days=3)).isoformat(), "primer_escaneo": None}
    assert (a["modo"], a["idioma"], a["pais"]) == ("tracking", "en-US", "US")
    json.dumps(a)


def test_la_ficha_usa_la_misma_consulta_que_la_lista_filtrada_por_su_id(bd):
    bd.filas = [_fila()]
    bd.extras = dict(_EXTRAS)
    acl.actividad_de(UID)
    q, p = bd.pagina()
    assert "WHERE p.id = %s)" in q and p == (UID, 1, 0)


def test_los_extras_de_la_ficha_toman_el_uid_en_cada_parametro_y_los_hilos_del_ssot(bd):
    bd.filas = [_fila()]
    bd.extras = dict(_EXTRAS)
    acl.actividad_de(UID)
    q, p = next((q, p) for q, p in bd.consultas if q.startswith("SELECT (SELECT count(*) FROM public.water_intake_log"))
    assert p and set(p) == {UID} and len(p) == q.count("%s")
    hilos = " ".join(USER_CHAT_THREAD_IDS_SQL.split())
    assert f"(m.user_id = %s OR m.session_id::text IN ({hilos}))" in q
    assert "m.feedback = 'down'" in q, "los 👎 van en las respuestas del coach: solo el hilo dice de quién son"
    assert "strpos(a.alert_key, %s) > 0" in q and "a.resolved_at IS NULL" in q
    assert "q.status = 'failed'" in q and "pm.metadata ->> 'event' = 'wizard_submit'" in q
    for contenido in ("m.content", "meal_name", "ingredients", "weight ", "glasses AS", "attachments"):
        assert contenido not in q, f"la actividad son números: {contenido}"


def test_si_fallan_solo_los_extras_la_actividad_no_se_pierde(bd, caplog):
    """La consulta de los extras (agua, peso, cola, alertas, tokens…) tiene su propio try: si falla, la ficha conserva
    los números de la lista y el embudo que sale de ella, y los extras salen en cero / [] / null con su aviso."""
    bd.filas = [_fila(comidas_total=7, comidas_30d=6, dias_activos_30d=3, planes=2, mensajes_coach=4, escaneos=1,
                      micros_30d=1500000, ultima=T0, plan_mode="tracking", locale="en-US", pais="US")]
    bd.extras_rotos = True
    a = acl.actividad_de(UID)
    assert set(a) == _CLAVES_ACTIVIDAD_FICHA and set(a["embudo"]) == _CLAVES_EMBUDO
    assert (a["comidas_total"], a["comidas_30d"], a["planes"], a["mensajes_coach"], a["escaneos"]) == (7, 6, 2, 4, 1)
    assert (a["gasto_ia_30d_usd"], a["dias_activos_30d"], a["ultima"]) == (1.5, 3, T0.isoformat())
    assert a["comidas_por_dia_activo"] == 2.0
    assert (a["dias_con_agua_30d"], a["registros_peso"], a["bloques_fallidos_30d"], a["pulgares_abajo"],
            a["avisos_abiertos"]) == (0, 0, 0, 0, 0)
    assert a["plataformas"] == []
    assert a["embudo"] == {"alta": (T0 - timedelta(days=20)).isoformat(), "formulario": None, "primer_plan": None,
                           "primera_comida": None, "primer_mensaje": None, "primer_escaneo": None}
    assert (a["modo"], a["idioma"], a["pais"]) == ("tracking", "en-US", "US")
    assert any("P1-PLAN-LOTE-831" in r.getMessage() and "push_subscriptions" in r.getMessage() for r in caplog.records)
    json.dumps(a)


def test_si_falla_la_consulta_de_la_fila_la_actividad_si_lanza(bd, monkeypatch):
    """Lo que no se traga es la consulta PRINCIPAL: sin la fila no hay «base» que enseñar (`ampliar_ficha` la vuelve
    null y la ficha carga igual)."""
    def _roto(q, p=None, fetch_one=False, fetch_all=False):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(acl, "execute_sql_query", _roto)
    with pytest.raises(RuntimeError):
        acl.actividad_de(UID)


def test_el_formulario_es_el_primer_envio_o_si_no_el_primer_plan(bd):
    bd.filas = [_fila()]
    bd.extras = {**_EXTRAS, "formulario_enviado": None}
    assert acl.actividad_de(UID)["embudo"]["formulario"] == (T0 - timedelta(days=10)).isoformat()
    bd.extras = {**_EXTRAS, "formulario_enviado": T0 - timedelta(days=5)}          # el plan fue antes (invitado)
    assert acl.actividad_de(UID)["embudo"]["formulario"] == (T0 - timedelta(days=10)).isoformat()
    bd.extras = {**_EXTRAS, "formulario_enviado": None, "primer_plan": None}
    assert acl.actividad_de(UID)["embudo"]["formulario"] is None


def test_la_web_se_ve_por_su_suscripcion_push(bd):
    bd.filas = [_fila()]
    bd.extras = {**_EXTRAS, "plataformas_push": None, "plataformas_consentimiento": None,
                 "plataformas_dispositivo": None, "suscripciones_web": 2}
    assert acl.actividad_de(UID)["plataformas"] == ["web"]
    bd.extras = {**bd.extras, "suscripciones_web": 0}
    assert acl.actividad_de(UID)["plataformas"] == []


def test_sin_dias_activos_las_comidas_por_dia_son_cero(bd):
    bd.filas = [_fila(comidas_30d=0, dias_activos_30d=0)]
    bd.extras = dict(_EXTRAS)
    assert acl.actividad_de(UID)["comidas_por_dia_activo"] == 0.0


def test_la_actividad_de_una_cuenta_que_no_existe(bd):
    assert acl.actividad_de(UID) is None
    bd.consultas.clear()
    assert acl.actividad_de("guest") is None and bd.consultas == [], "un id que no es uuid no consulta la base"


def _base_774() -> dict:
    return {"user_id": UID, "email": "ana@correo.com", "nombre": "Ana", "alta": "2026-09-01T00:00:00+00:00",
            "plan_pagado": "gratis", "plan_efectivo": "gratis", "es_admin": False,
            "suscripcion": {"estado": None, "fin": None, "paypal": False}, "cortesia": None,
            "creditos": {"usados": 3, "plan": 10, "regalo": 0, "tope": 10}, "regalos": []}


def test_ampliar_ficha_anade_los_tres_bloques_sin_tocar_lo_del_774(monkeypatch):
    base = _base_774()
    copia = json.loads(json.dumps(base))
    monkeypatch.setattr(acl, "actividad_de", lambda uid: {"comidas_total": 4} if uid == UID else None)
    monkeypatch.setattr(ajustes_cuenta, "ajustes_de", lambda uid: {
        "ajustes": [{"clave": "locale", "estado": "valor", "valor": "es-DO"}],
        "ajustes_dispositivo": {"web": {"tema": "dark", "at": "2026-09-29T10:00:00+00:00"}}})
    monkeypatch.setattr(acl, "prueba_de", lambda uid: {"estado": "activa"})
    f = acl.ampliar_ficha(base)
    assert base == copia, "no muta la ficha que recibe"
    assert {k: f[k] for k in base} == base
    assert f["actividad"] == {"comidas_total": 4} and f["prueba"] == {"estado": "activa"}
    assert f["ajustes"] == [{"clave": "locale", "estado": "valor", "valor": "es-DO"}]
    assert f["ajustes_dispositivo"] == {"web": {"tema": "dark", "at": "2026-09-29T10:00:00+00:00"}}


def test_ampliar_ficha_carga_aunque_falle_cada_bloque(monkeypatch, caplog):
    def _roto(uid):
        raise RuntimeError("migración sin aplicar")
    for nombre in ("actividad_de", "prueba_de"):
        monkeypatch.setattr(acl, nombre, _roto)
    monkeypatch.setattr(ajustes_cuenta, "ajustes_de", _roto)
    f = acl.ampliar_ficha(_base_774())
    assert f["actividad"] is None and f["prueba"] is None
    assert f["ajustes"] is None and f["ajustes_dispositivo"] == {}
    assert f["email"] == "ana@correo.com"
    assert sum("P1-PLAN-LOTE-831" in r.getMessage() for r in caplog.records) >= 3


def test_ampliar_ficha_de_nada_es_nada():
    assert acl.ampliar_ficha(None) is None


def test_la_ficha_ampliada_etiqueta_a_los_admin_con_la_misma_regla_que_la_lista(bd, monkeypatch):
    """La lista dice `es_admin` por el tier `admin` O por la lista del .env del panel; la ficha del 774 solo mira el
    tier. Ya ampliada, no puede decir otra cosa que la lista sobre la misma cuenta."""
    monkeypatch.setattr(acl, "actividad_de", lambda uid: None)
    monkeypatch.setattr(ajustes_cuenta, "ajustes_de", lambda uid: {"ajustes": [], "ajustes_dispositivo": {}})
    monkeypatch.setattr(acl, "prueba_de", lambda uid: None)
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", f"{ADMIN2.upper()}, {OTRO}")
    casos = [(ADMIN, "admin"), (ADMIN2, "gratis"), (OTRO, "plus"), (UID, "gratis"), (NADIE, "ultra")]
    bd.filas = [_fila(uid, plan_pagado=pagado) for uid, pagado in casos]
    en_la_lista = {c["user_id"]: c["es_admin"] for c in acl.listar("", "actividad", "todas", 1)["cuentas"]}
    assert en_la_lista == {ADMIN: True, ADMIN2: True, OTRO: True, UID: False, NADIE: False}
    for uid, pagado in casos:
        base = {**_base_774(), "user_id": uid, "plan_pagado": pagado}     # el 774: es_admin False salvo el tier
        ficha = acl.ampliar_ficha(base)
        assert ficha["es_admin"] is en_la_lista[uid], f"{uid}: la ficha contradice a la lista"
        assert base["es_admin"] is False, "no muta la ficha que recibe"
    monkeypatch.delenv("MEALFIT_ADMIN_USER_IDS")
    assert acl.ampliar_ficha({**_base_774(), "plan_pagado": "admin"})["es_admin"] is True, "el tier basta"
    assert acl.ampliar_ficha({**_base_774(), "user_id": ADMIN2})["es_admin"] is False, "sin lista, solo el tier"


def test_apagado_la_ficha_no_pasa_por_ampliar_y_sigue_siendo_la_del_774(panel, monkeypatch):
    """Solo el camino con el interruptor encendido recalcula `es_admin` (`ampliar_ficha`): apagado, ni se llama."""
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", f"{ADMIN},{UID}")
    monkeypatch.setattr(ac, "ficha", lambda uid: _base_774())
    monkeypatch.setattr(acl, "ampliar_ficha", lambda f: pytest.fail("apagado no se amplía"))
    r = panel.cliente().get(f"/api/admin/cuentas/{UID}")
    assert r.status_code == 200 and r.json()["cuenta"]["es_admin"] is False


def test_la_prueba_de_la_ficha(bd, monkeypatch):
    marca = {"id": "m2", "marcada_por": ADMIN, "marcada_at": T0, "motivo": MOTIVO, "aviso_visto_at": None}
    historial = [
        {**marca, "quitada_at": None, "quitada_por": None, "quitada_por_la_persona": False, "motivo_quitar": None},
        {"id": "m1", "marcada_por": ADMIN2, "marcada_at": T0 - timedelta(days=5), "motivo": "primera vez",
         "aviso_visto_at": T0 - timedelta(days=5), "quitada_at": T0 - timedelta(days=2), "quitada_por": UID,
         "quitada_por_la_persona": True, "motivo_quitar": None}]
    monkeypatch.setattr(cp, "marca_viva", lambda uid: dict(marca))
    monkeypatch.setattr(cp, "historial", lambda uid: [dict(h) for h in historial])
    bd.correos = [{"id": ADMIN, "email": "dueno@bioboros.com"}]
    p = acl.prueba_de(UID)
    assert p == {
        "estado": "aviso_pendiente", "desde": T0.isoformat(), "motivo": MOTIVO, "marcada_por": "dueno@bioboros.com",
        "aviso_visto_at": None,
        "historial": [
            {"desde": T0.isoformat(), "hasta": None, "motivo": MOTIVO, "motivo_quitar": None,
             "quitada_por_la_persona": False, "marcada_por": "dueno@bioboros.com"},
            {"desde": (T0 - timedelta(days=5)).isoformat(), "hasta": (T0 - timedelta(days=2)).isoformat(),
             "motivo": "primera vez", "motivo_quitar": None, "quitada_por_la_persona": True, "marcada_por": None}]}
    assert UID not in json.dumps(p) and ADMIN2 not in json.dumps(p), "ni ids del personal ni el de la persona"


def test_una_cuenta_que_nunca_fue_de_prueba_no_tiene_bloque(bd, monkeypatch):
    """`None` SOLO si jamás se marcó: ni marca viva ni historial."""
    monkeypatch.setattr(cp, "marca_viva", lambda uid: None)
    monkeypatch.setattr(cp, "historial", lambda uid: [])
    assert acl.prueba_de(UID) is None


@pytest.mark.parametrize("quien", ["admin", "persona"])
def test_quitada_la_marca_la_ficha_dice_sin_marca_y_trae_el_historial(bd, monkeypatch, quien):
    """Enmienda del controlador (29-sep): hubo marcas y ninguna vive (la quitó un admin o salió la propia persona) ⇒
    `estado: "sin_marca"`, `desde`/`motivo`/`marcada_por`/`aviso_visto_at` en null y el `historial` (cuándo y quién)."""
    quitada = {"id": "m1", "marcada_por": ADMIN, "marcada_at": T0 - timedelta(days=5), "motivo": MOTIVO,
               "aviso_visto_at": T0 - timedelta(days=4), "quitada_at": T0 - timedelta(days=2),
               "quitada_por": ADMIN2 if quien == "admin" else UID, "quitada_por_la_persona": quien == "persona",
               "motivo_quitar": "terminó la ronda" if quien == "admin" else None}
    monkeypatch.setattr(cp, "marca_viva", lambda uid: None)
    monkeypatch.setattr(cp, "historial", lambda uid: [dict(quitada)])
    bd.correos = [{"id": ADMIN, "email": "dueno@bioboros.com"}, {"id": ADMIN2, "email": "otro@bioboros.com"}]
    p = acl.prueba_de(UID)
    assert p == {
        "estado": "sin_marca", "desde": None, "motivo": None, "marcada_por": None, "aviso_visto_at": None,
        "historial": [{"desde": (T0 - timedelta(days=5)).isoformat(), "hasta": (T0 - timedelta(days=2)).isoformat(),
                       "motivo": MOTIVO, "motivo_quitar": "terminó la ronda" if quien == "admin" else None,
                       "quitada_por_la_persona": quien == "persona", "marcada_por": "dueno@bioboros.com"}]}
    assert set(p) == {"estado", "desde", "motivo", "marcada_por", "aviso_visto_at", "historial"}, "la misma forma"
    for id_ in (UID, ADMIN, ADMIN2):
        assert id_ not in json.dumps(p), "ni ids del personal ni el de la persona"


def test_la_ficha_ampliada_de_una_cuenta_que_salio_lleva_sin_marca(bd, monkeypatch):
    monkeypatch.setattr(acl, "actividad_de", lambda uid: None)
    monkeypatch.setattr(ajustes_cuenta, "ajustes_de", lambda uid: {"ajustes": [], "ajustes_dispositivo": {}})
    monkeypatch.setattr(cp, "marca_viva", lambda uid: None)
    monkeypatch.setattr(cp, "historial", lambda uid: [{
        "id": "m1", "marcada_por": ADMIN, "marcada_at": T0 - timedelta(days=5), "motivo": MOTIVO,
        "aviso_visto_at": None, "quitada_at": T0, "quitada_por": UID, "quitada_por_la_persona": True,
        "motivo_quitar": None}])
    assert acl.ampliar_ficha(_base_774())["prueba"]["estado"] == "sin_marca"


def test_la_lista_sigue_con_prueba_null_para_quien_no_tiene_marca_viva(bd, monkeypatch):
    """Solo la FICHA cuenta el pasado de la marca; la fila de la lista es la marca viva (su filtro «prueba» también) y no
    lee el historial."""
    monkeypatch.setattr(cp, "historial", lambda uid: pytest.fail("la lista no lee el historial"))
    bd.filas = [_fila(prueba_desde=None)]
    assert acl.listar("", "actividad", "todas", 1)["cuentas"][0]["prueba"] is None


def test_si_los_correos_no_se_leen_la_prueba_sale_sin_ellos(bd, monkeypatch):
    marca = {"id": "m", "marcada_por": ADMIN, "marcada_at": T0, "motivo": MOTIVO, "aviso_visto_at": T0}
    monkeypatch.setattr(cp, "marca_viva", lambda uid: dict(marca))
    monkeypatch.setattr(cp, "historial", lambda uid: [{**marca, "quitada_at": None, "quitada_por_la_persona": False,
                                                       "motivo_quitar": None}])
    bd.correos_rotos = True
    p = acl.prueba_de(UID)
    assert p["estado"] == "activa" and p["marcada_por"] is None and p["historial"][0]["marcada_por"] is None


# ═════════════════════════════════════════════ 6. el router
_LIMITADORES = ("_CUENTAS_LISTA_LIMITER", "_CUENTAS_LECTURA_LIMITER", "_CUENTAS_ESCRITURA_LIMITER")


class _Panel:
    def __init__(self):
        self.rastro: list = []
        self.orden: list = []
        self.rastro_roto = False

    def anotar(self, admin, accion, objetivo=None, detalle=None):
        self.orden.append("rastro")
        if self.rastro_roto:
            raise RuntimeError("admin_access_log no escribe")
        self.rastro.append((accion, objetivo, detalle))

    @staticmethod
    def cliente(uid=ADMIN):
        app = FastAPI()
        app.include_router(ra.router)
        app.dependency_overrides[get_verified_user_id] = lambda: uid
        return TestClient(app)


def _vaciar_limitadores():
    for nombre in _LIMITADORES:
        getattr(ra, nombre)._hits.clear()


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_PANEL", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", ADMIN)
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    p = _Panel()
    monkeypatch.setattr(ra, "registrar_acceso", p.anotar)
    _vaciar_limitadores()
    yield p
    _vaciar_limitadores()


_RUTAS_NUEVAS = [
    ("get", "/api/admin/cuentas?orden=alta", None),
    ("get", "/api/admin/cuentas.csv", None),
    ("post", f"/api/admin/cuentas/{UID}/prueba", {"motivo": MOTIVO}),
    ("post", f"/api/admin/cuentas/{UID}/prueba/quitar", {"motivo": MOTIVO}),
    ("post", "/api/admin/pruebas/lote", {"user_ids": [UID], "motivo": MOTIVO}),
    ("get", f"/api/admin/cuentas/{UID}/ajustes/historial", None),
    ("get", "/api/admin/ajustes/resumen", None),
]


def _pedir(c, metodo, ruta, cuerpo, cabeceras=H):
    return c.get(ruta) if metodo == "get" else c.post(ruta, json=cuerpo, headers=cabeceras)


def _nada_se_toca(monkeypatch):
    def _no(*a, **k):
        pytest.fail("con el interruptor apagado no se llega al módulo")
    for mod, nombres in ((acl, ("listar", "exportar_csv", "ampliar_ficha")),
                         (cp, ("marcar", "quitar", "marcar_varias")),
                         (ajustes_cuenta, ("historial", "resumen"))):
        for nombre in nombres:
            monkeypatch.setattr(mod, nombre, _no)


@pytest.mark.parametrize("metodo,ruta,cuerpo", _RUTAS_NUEVAS)
def test_con_el_interruptor_apagado_las_rutas_nuevas_son_404(panel, monkeypatch, metodo, ruta, cuerpo):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    _nada_se_toca(monkeypatch)
    r = _pedir(panel.cliente(), metodo, ruta, cuerpo)
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, "no se anuncia: el mismo 404 que un extraño"
    assert panel.rastro == []


def test_con_el_interruptor_apagado_la_ficha_es_la_del_774(panel, monkeypatch):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    _nada_se_toca(monkeypatch)
    monkeypatch.setattr(ac, "ficha", lambda uid: _base_774() if uid == UID else None)
    monkeypatch.setattr(ac, "buscar_por_correo", lambda correo: UID)
    c = panel.cliente()
    assert c.get(f"/api/admin/cuentas/{UID}").json() == {"cuenta": _base_774()}
    assert c.post("/api/admin/cuentas/buscar", json={"email": "ana@correo.com"}, headers=H).json() == {
        "cuenta": _base_774()}


@pytest.mark.parametrize("metodo,ruta,cuerpo", _RUTAS_NUEVAS)
def test_quien_no_es_admin_recibe_404(panel, monkeypatch, metodo, ruta, cuerpo):
    _nada_se_toca(monkeypatch)
    assert _pedir(panel.cliente(ADMIN2), metodo, ruta, cuerpo).status_code == 404


@pytest.mark.parametrize("metodo,ruta,cuerpo", [x for x in _RUTAS_NUEVAS if x[0] == "post"])
def test_los_post_sin_la_cabecera_de_accion_son_403(panel, monkeypatch, metodo, ruta, cuerpo):
    _nada_se_toca(monkeypatch)
    assert _pedir(panel.cliente(), metodo, ruta, cuerpo, cabeceras={}).status_code == 403


def _lista_falsa(panel, monkeypatch, respuesta=None):
    llamadas = []

    def _listar(buscar, orden, filtro, pagina):
        panel.orden.append("consulta")
        llamadas.append((buscar, orden, filtro, pagina))
        return respuesta or {"cuentas": [{"user_id": UID}, {"user_id": OTRO}], "total": 2, "pagina": pagina,
                             "por_pagina": 50}
    monkeypatch.setattr(acl, "listar", _listar)
    return llamadas


def test_la_lista_anota_su_rastro_antes_de_responder(panel, monkeypatch):
    llamadas = _lista_falsa(panel, monkeypatch)
    r = panel.cliente().get("/api/admin/cuentas?buscar=ana&orden=gasto&filtro=activas_7d&pagina=2")
    assert r.status_code == 200 and r.json()["total"] == 2 and len(r.json()["cuentas"]) == 2
    assert llamadas == [("ana", "gasto", "activas_7d", 2)]
    assert panel.orden == ["consulta", "rastro"]
    assert panel.rastro == [("listar_cuentas", None,
                             {"con_busqueda": True, "orden": "gasto", "filtro": "activas_7d", "pagina": 2, "n": 2})]
    assert "buscar" not in panel.rastro[0][2]


def test_el_rastro_de_la_lista_y_del_csv_nunca_guarda_lo_buscado(panel, monkeypatch):
    """Enmienda del controlador (29-sep): lo buscado puede ser un correo y la política promete que el registro no guarda
    correos. El rastro dice SI hubo búsqueda (`con_busqueda`), jamás cuál; una búsqueda en blanco no es una búsqueda."""
    _lista_falsa(panel, monkeypatch)
    monkeypatch.setattr(acl, "exportar_csv", lambda buscar, orden, filtro: ("﻿x\r\n", 3))
    c = panel.cliente()
    secreto = "ana.perez@correo.com"
    assert c.get("/api/admin/cuentas", params={"buscar": secreto, "filtro": "prueba"}).status_code == 200
    assert c.get("/api/admin/cuentas.csv", params={"buscar": secreto, "orden": "alta"}).status_code == 200
    assert c.get("/api/admin/cuentas", params={"buscar": "   "}).status_code == 200
    assert c.get("/api/admin/cuentas.csv").status_code == 200
    lista, csv_, lista_en_blanco, csv_sin_busqueda = [(a, d) for a, _, d in panel.rastro]
    assert lista == ("listar_cuentas", {"con_busqueda": True, "orden": "actividad", "filtro": "prueba", "pagina": 1,
                                        "n": 2})
    assert csv_ == ("exportar_cuentas", {"con_busqueda": True, "filtro": "todas", "n": 3})
    assert lista_en_blanco[1]["con_busqueda"] is False
    assert csv_sin_busqueda == ("exportar_cuentas", {"con_busqueda": False, "filtro": "todas", "n": 3})
    for _, _, detalle in panel.rastro:
        assert "buscar" not in detalle
        for texto in (secreto, "correo.com", "@"):
            assert texto not in json.dumps(detalle), f"el rastro guarda lo buscado: {detalle}"


def test_la_lista_por_defecto(panel, monkeypatch):
    llamadas = _lista_falsa(panel, monkeypatch)
    assert panel.cliente().get("/api/admin/cuentas").status_code == 200
    assert llamadas == [("", "actividad", "todas", 1)]


@pytest.mark.parametrize("consulta", ["orden=nombre", "filtro=admins", "pagina=0", "orden=", "buscar=" + "x" * 255])
def test_la_lista_rechaza_lo_que_no_esta_en_el_contrato(panel, monkeypatch, consulta):
    _nada_se_toca(monkeypatch)
    assert panel.cliente().get(f"/api/admin/cuentas?{consulta}").status_code == 422


def test_si_el_rastro_de_la_lista_falla_no_hay_datos(panel, monkeypatch):
    _lista_falsa(panel, monkeypatch)
    panel.rastro_roto = True
    r = panel.cliente().get("/api/admin/cuentas")
    assert r.status_code == 503 and "cuentas" not in r.json() and UID not in r.text


def test_si_la_lista_no_se_lee_503_sin_rastro(panel, monkeypatch):
    def _roto(*a):
        raise RuntimeError("cuentas_de_prueba no existe")
    monkeypatch.setattr(acl, "listar", _roto)
    r = panel.cliente().get("/api/admin/cuentas")
    assert r.status_code == 503 and panel.rastro == []


def test_el_csv_por_http(panel, monkeypatch):
    pedido = []

    def _exportar(buscar, orden, filtro):
        panel.orden.append("consulta")
        pedido.append((buscar, orden, filtro))
        return "\ufeffuser_id,email\r\n" + f"{UID},ana@correo.com\r\n", 1
    monkeypatch.setattr(acl, "exportar_csv", _exportar)
    r = panel.cliente().get("/api/admin/cuentas.csv?buscar=ana&orden=alta&filtro=prueba")
    assert r.status_code == 200 and pedido == [("ana", "alta", "prueba")]
    assert r.headers["content-type"] == "text/csv; charset=utf-8"
    hoy = datetime.now(timezone.utc).strftime("%Y%m%d")
    assert r.headers["content-disposition"] == f'attachment; filename="cuentas-{hoy}.csv"'
    assert r.headers.get("cache-control") == "no-store"
    assert r.content.startswith(b"\xef\xbb\xbf") and r.content.decode("utf-8-sig").startswith("user_id,email")
    assert panel.orden == ["consulta", "rastro"]
    accion, objetivo, detalle = panel.rastro[0]
    assert (accion, objetivo) == ("exportar_cuentas", None)
    assert detalle == {"con_busqueda": True, "filtro": "prueba", "n": 1}, "y nunca el texto buscado"
    assert "ana" not in json.dumps(detalle)


def test_el_csv_sin_rastro_o_sin_datos_es_503(panel, monkeypatch):
    monkeypatch.setattr(acl, "exportar_csv", lambda *a: ("\ufeffx\r\n", 0))
    panel.rastro_roto = True
    r = panel.cliente().get("/api/admin/cuentas.csv")
    assert r.status_code == 503 and not r.content.startswith(b"\xef\xbb\xbf")
    panel.rastro_roto = False

    def _roto(*a):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(acl, "exportar_csv", _roto)
    assert panel.cliente().get("/api/admin/cuentas.csv").status_code == 503


def test_el_csv_no_lo_captura_la_ficha(panel, monkeypatch):
    monkeypatch.setattr(acl, "exportar_csv", lambda *a: ("\ufeffx\r\n", 0))
    monkeypatch.setattr(ac, "ficha", lambda uid: pytest.fail("cuentas.csv no es una cuenta"))
    assert panel.cliente().get("/api/admin/cuentas.csv").status_code == 200


def test_la_ficha_ampliada_con_el_interruptor(panel, monkeypatch):
    monkeypatch.setattr(ac, "ficha", lambda uid: _base_774() if uid == UID else None)
    monkeypatch.setattr(ac, "buscar_por_correo", lambda correo: UID)

    def _ampliar(f):
        panel.orden.append("consulta")
        return {**f, "actividad": {"comidas_total": 1}}
    monkeypatch.setattr(acl, "ampliar_ficha", _ampliar)
    c = panel.cliente()
    r = c.get(f"/api/admin/cuentas/{UID}")
    assert r.status_code == 200 and r.json()["cuenta"] == {**_base_774(), "actividad": {"comidas_total": 1}}
    assert panel.orden == ["consulta", "rastro"] and panel.rastro == [("ver_cuenta", UID, {})]
    r = c.post("/api/admin/cuentas/buscar", json={"email": "ana@correo.com"}, headers=H)
    assert r.json()["cuenta"]["actividad"] == {"comidas_total": 1}
    assert c.get(f"/api/admin/cuentas/{OTRO}").status_code == 404


@pytest.fixture
def marcas(panel, monkeypatch):
    """La marca de verdad (`cuentas_prueba`) sobre la base falsa del lote 830; la ficha que se relee, de mentira."""
    b = _BDPrueba()
    monkeypatch.setattr(cp, "execute_sql_query", b.query)
    monkeypatch.setattr(cp, "execute_sql_write", b.write)
    monkeypatch.setattr(cp, "registrar_acceso", b.anotar)
    monkeypatch.setattr(ac, "ficha", lambda uid: {"user_id": uid, "email": "ana@correo.com"})
    monkeypatch.setattr(acl, "ampliar_ficha", lambda f: {**f, "prueba": cp.marca_viva(f["user_id"]) and "viva"}
                        if f else f)
    return b


def test_marcar_y_quitar_por_http(panel, marcas):
    c = panel.cliente()
    r = c.post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H)
    assert r.status_code == 200 and r.json() == {"ok": True, "cuenta": {"user_id": UID, "email": "ana@correo.com",
                                                                         "prueba": "viva"}}
    assert [a for a, *_ in marcas.rastro] == ["marcar_prueba"], "el rastro lo escribe cuentas_prueba, antes del INSERT"
    r = c.post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H)
    assert (r.status_code, r.json()) == (409, {"detail": "ya_marcada"})
    r = c.post(f"/api/admin/cuentas/{UID}/prueba/quitar", json={"motivo": "terminó la prueba"}, headers=H)
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["cuenta"]["prueba"] is None
    r = c.post(f"/api/admin/cuentas/{UID}/prueba/quitar", json={"motivo": "otra vez"}, headers=H)
    assert (r.status_code, r.json()) == (409, {"detail": "sin_marca"})


def test_si_releer_la_ficha_falla_la_marca_guardada_no_es_un_500(panel, marcas, monkeypatch):
    # Como los regalos: la marca YA está; un 500 invitaría a reintentar (y el reintento daría 409 ya_marcada).
    def _rota(uid):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(ac, "ficha", _rota)
    r = panel.cliente().post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H)
    assert (r.status_code, r.json()) == (200, {"ok": True, "cuenta": None}) and len(marcas.filas) == 1


def test_buscar_a_nadie_con_el_interruptor_no_amplia_nada(panel, monkeypatch):
    monkeypatch.setattr(ac, "buscar_por_correo", lambda correo: None)
    monkeypatch.setattr(acl, "ampliar_ficha", lambda f: pytest.fail("sin cuenta no hay nada que ampliar"))
    r = panel.cliente().post("/api/admin/cuentas/buscar", json={"email": "x@y.z"}, headers=H)
    assert r.json() == {"cuenta": None} and panel.rastro == [("buscar_cuenta", None, {"encontrada": False})]


def test_volver_a_marcar_tras_salir_ella_exige_confirmar_la_vuelta(panel, marcas):
    c = panel.cliente()
    assert c.post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H).status_code == 200
    assert cp.salir(UID) is True
    r = c.post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H)
    assert (r.status_code, r.json()) == (409, {"detail": "salio_ella"})
    r = c.post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": "me pidió volver", "confirmar_vuelta": True},
               headers=H)
    assert r.status_code == 200 and r.json()["cuenta"]["prueba"] == "viva"


@pytest.mark.parametrize("ruta,cuerpo,status,detalle", [
    (f"/api/admin/cuentas/{UID}/prueba", {"motivo": "no"}, 422, "motivo"),
    (f"/api/admin/cuentas/{UID}/prueba/quitar", {"motivo": "  "}, 422, "motivo"),
    (f"/api/admin/cuentas/{NADIE}/prueba", {"motivo": MOTIVO}, 404, "no_existe"),
    ("/api/admin/pruebas/lote", {"user_ids": [UID] * 101, "motivo": MOTIVO}, 422, "demasiadas"),
    ("/api/admin/pruebas/lote", {"user_ids": [UID], "motivo": "x"}, 422, "motivo"),
])
def test_los_codigos_de_la_marca_pasan_tal_cual(panel, marcas, ruta, cuerpo, status, detalle):
    r = panel.cliente().post(ruta, json=cuerpo, headers=H)
    assert (r.status_code, r.json()) == (status, {"detail": detalle})
    assert marcas.filas == []


def test_si_el_rastro_de_la_marca_falla_503_y_sin_marca(panel, marcas):
    marcas.rastro_roto = True
    r = panel.cliente().post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H)
    assert r.status_code == 503 and marcas.filas == []


@pytest.mark.parametrize("ruta,cuerpo", [
    (f"/api/admin/cuentas/{UID}/prueba", {"motivo": MOTIVO}),
    (f"/api/admin/cuentas/{UID}/prueba/quitar", {"motivo": MOTIVO}),
    ("/api/admin/pruebas/lote", {"user_ids": [UID], "motivo": MOTIVO}),
])
def test_un_fallo_inesperado_al_marcar_o_quitar_es_503_y_se_registra(panel, marcas, monkeypatch, caplog, ruta, cuerpo):
    """Igual que las lecturas (`_sin_datos`): lo que NO es una regla de la marca (`ErrorPrueba`) —la base que no
    responde— sale como 503 con una frase, sin el error dentro, y el error queda en el log. Antes era un 500 pelado."""
    def _rota(*a, **k):
        raise RuntimeError("sin DB: secreto-interno")
    monkeypatch.setattr(cp, "execute_sql_query", _rota)
    r = panel.cliente().post(ruta, json=cuerpo, headers=H)
    assert r.status_code == 503 and "No se pudo completar" in r.json()["detail"]
    assert "secreto-interno" not in r.text and "RuntimeError" not in r.text, "el error va al log, no a la respuesta"
    assert marcas.filas == [] and marcas.rastro == [], "nada se escribió ni se anotó"
    assert any("P1-PLAN-LOTE-831" in rec.getMessage() and "secreto-interno" in rec.getMessage()
               and rec.levelname == "ERROR" for rec in caplog.records)


def test_las_reglas_de_la_marca_siguen_saliendo_con_su_codigo(panel, marcas):
    """El `except Exception` nuevo va DESPUÉS del de `ErrorPrueba`: 409/404/422 no se vuelven 503."""
    c = panel.cliente()
    c.post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H)
    r = c.post(f"/api/admin/cuentas/{UID}/prueba", json={"motivo": MOTIVO}, headers=H)
    assert (r.status_code, r.json()) == (409, {"detail": "ya_marcada"})
    r = c.post(f"/api/admin/cuentas/{NADIE}/prueba/quitar", json={"motivo": MOTIVO}, headers=H)
    assert (r.status_code, r.json()) == (409, {"detail": "sin_marca"})
    r = c.post("/api/admin/pruebas/lote", json={"user_ids": [UID] * 101, "motivo": MOTIVO}, headers=H)
    assert (r.status_code, r.json()) == (422, {"detail": "demasiadas"})


def test_los_marcadores_del_router_llevan_su_fecha():
    """Convención del repo: los comentarios nuevos llevan `[P1-PLAN-LOTE-8xx · 2026-09-29]` (los mensajes de log, no)."""
    src = (_BACKEND / "routers" / "admin.py").read_text(encoding="utf-8")
    sin_fecha = [n for n, linea in enumerate(src.splitlines(), 1)
                 if "#" in linea and "[P1-PLAN-LOTE-831]" in linea.split("#", 1)[1]]
    assert not sin_fecha, f"comentarios con el marcador sin fecha en las líneas {sin_fecha}"


def test_marcar_varias_por_http(panel, marcas):
    ids = [UID, OTRO, NADIE, UID, "no-es-un-uuid"]
    r = panel.cliente().post("/api/admin/pruebas/lote", json={"user_ids": ids, "motivo": MOTIVO}, headers=H)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "resultados": [
        {"user_id": UID, "resultado": "marcada"}, {"user_id": OTRO, "resultado": "marcada"},
        {"user_id": NADIE, "resultado": "no_existe"}, {"user_id": UID, "resultado": "ya_marcada"},
        {"user_id": "no-es-un-uuid", "resultado": "no_existe"}]}
    assert [(a, o) for a, o, _ in marcas.rastro] == [("marcar_prueba", UID), ("marcar_prueba", OTRO)]


def test_la_ruta_del_lote_no_la_captura_la_de_una_cuenta():
    rutas = {(r.path, m) for r in ra.router.routes for m in getattr(r, "methods", ())}
    assert ("/api/admin/pruebas/lote", "POST") in rutas
    assert not any(p.startswith("/api/admin/cuentas/") and p.endswith("/lote") for p, _ in rutas)


def test_el_historial_de_ajustes(panel, monkeypatch):
    pedidos = []

    def _historial(uid, dias):
        panel.orden.append("consulta")
        pedidos.append((uid, dias))
        return [{"at": T0.isoformat(), "clave": "locale", "etiqueta": "Idioma", "antes": "es-DO", "despues": "en-US",
                 "origen": "app"}]
    monkeypatch.setattr(ajustes_cuenta, "historial", _historial)
    c = panel.cliente()
    r = c.get(f"/api/admin/cuentas/{UID}/ajustes/historial?dias=30")
    assert r.status_code == 200 and r.json()["cambios"][0]["clave"] == "locale" and pedidos == [(UID, 30)]
    assert panel.orden == ["consulta", "rastro"] and panel.rastro == [("ver_ajustes", UID, {"dias": 30, "n": 1})]
    assert c.get(f"/api/admin/cuentas/{UID}/ajustes/historial").status_code == 200 and pedidos[-1] == (UID, 90)
    for dias in (0, 366):
        assert c.get(f"/api/admin/cuentas/{UID}/ajustes/historial?dias={dias}").status_code == 422
    panel.rastro_roto = True
    r = c.get(f"/api/admin/cuentas/{UID}/ajustes/historial")
    assert r.status_code == 503 and "cambios" not in r.json()


def test_el_historial_que_no_se_lee_es_503_sin_rastro(panel, monkeypatch):
    def _roto(uid, dias):
        raise RuntimeError("ajustes_cambios no existe")
    monkeypatch.setattr(ajustes_cuenta, "historial", _roto)
    assert panel.cliente().get(f"/api/admin/cuentas/{UID}/ajustes/historial").status_code == 503
    assert panel.rastro == []


def test_el_resumen_de_ajustes_es_agregado_y_sin_rastro(panel, monkeypatch):
    pedidos = []
    resumen = {"cuentas": 3, "ajustes": [], "cambios": [], "dispositivo": {"tema": {}, "notificaciones_permiso": {},
                                                                          "plataformas": {}}}
    monkeypatch.setattr(ajustes_cuenta, "resumen", lambda dias: pedidos.append(dias) or resumen)
    c = panel.cliente()
    assert c.get("/api/admin/ajustes/resumen?dias=7").json() == resumen and pedidos == [7]
    assert c.get("/api/admin/ajustes/resumen").status_code == 200 and pedidos[-1] == 30
    assert c.get("/api/admin/ajustes/resumen?dias=91").status_code == 422
    assert panel.rastro == []

    def _roto(dias):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(ajustes_cuenta, "resumen", _roto)
    assert c.get("/api/admin/ajustes/resumen").status_code == 503


# ═════════════════════════════════════════════ 7. forma del router y limitadores
def _ruta(path, metodo):
    return next(r for r in ra.router.routes if r.path == path and metodo in r.methods)


def _dependencias(ruta):
    return [d.dependency for d in ruta.dependencies]


_NUEVAS = [("/api/admin/cuentas", "GET"), ("/api/admin/cuentas.csv", "GET"),
           ("/api/admin/cuentas/{user_id}/prueba", "POST"), ("/api/admin/cuentas/{user_id}/prueba/quitar", "POST"),
           ("/api/admin/pruebas/lote", "POST"), ("/api/admin/cuentas/{user_id}/ajustes/historial", "GET"),
           ("/api/admin/ajustes/resumen", "GET")]


@pytest.mark.parametrize("path,metodo", _NUEVAS)
def test_cada_ruta_nueva_pasa_por_el_interruptor_y_un_limitador(path, metodo):
    deps = _dependencias(_ruta(path, metodo))
    assert ra._exigir_knob_pruebas in deps
    assert any(d in deps for d in (ra._CUENTAS_LISTA_LIMITER, ra._CUENTAS_LECTURA_LIMITER,
                                   ra._CUENTAS_ESCRITURA_LIMITER))
    assert deps.index(ra._exigir_knob_pruebas) < min(deps.index(d) for d in deps if hasattr(d, "max_calls")), (
        "apagado, ni cuenta en el cupo")
    if metodo == "POST":
        assert ra._exigir_cabecera in deps and ra._CUENTAS_ESCRITURA_LIMITER in deps


def test_la_lista_y_el_csv_comparten_su_limitador_y_el_historial_usa_el_de_lectura():
    assert ra._CUENTAS_LISTA_LIMITER in _dependencias(_ruta("/api/admin/cuentas", "GET"))
    assert ra._CUENTAS_LISTA_LIMITER in _dependencias(_ruta("/api/admin/cuentas.csv", "GET"))
    assert ra._CUENTAS_LECTURA_LIMITER in _dependencias(_ruta("/api/admin/cuentas/{user_id}/ajustes/historial", "GET"))


def test_el_limitador_de_la_lista_tiene_su_par_propio():
    src = (_BACKEND / "routers" / "admin.py").read_text(encoding="utf-8")
    m = re.search(r"_CUENTAS_LISTA_LIMITER = RateLimiter\(max_calls=(\d+), period_seconds=(\d+)\)", src)
    assert m, "el limitador de la lista existe con sus números a la vista"
    assert re.fullmatch(r"_[A-Z_]+_LIMITER", "_CUENTAS_LISTA_LIMITER"), "el guard de limitadores no admite dígitos"
    pares = []
    for f in [*_BACKEND.glob("*.py"), *(_BACKEND / "routers").glob("*.py")]:
        pares += re.findall(r"RateLimiter\(\s*(?:max_calls\s*=\s*)?(\d+)\s*,\s*(?:period_seconds\s*=\s*)?(\d+)\s*\)",
                            f.read_text(encoding="utf-8"))
    assert pares.count(m.groups()) == 1, "Redis comparte la ventana por par (rl:<max>:<periodo>:<uid>)"


def test_el_orden_y_el_filtro_del_router_son_los_del_modulo():
    import typing
    firma = typing.get_type_hints(ra.api_admin_listar_cuentas)
    assert set(typing.get_args(firma["orden"])) == set(acl.ORDENES)
    assert set(typing.get_args(firma["filtro"])) == set(acl.FILTROS)


def test_ninguna_ruta_del_panel_cobra_cuota():
    src = (_BACKEND / "routers" / "admin.py").read_text(encoding="utf-8")
    assert "verify_api_quota" not in src
    assert all("/admin/" not in r.path.replace("/api/admin", "", 1) for r in ra.router.routes)
