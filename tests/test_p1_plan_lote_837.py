"""[P1-PLAN-LOTE-837 · 2026-09-29] Los ajustes de cada cuenta: registro, historial por trigger y ajustes del dispositivo
(spec 2026-09-29-admin-cuentas-actividad-pruebas-design §13.1, §13.2, §13.3 y §13.7).

Un ajuste es MODO DE USO, no contenido: el perfil de salud no es un ajuste y nunca sale de aquí. Todo con la base falsa:
nada de este fichero toca Neon. El SQL de la migración se revisa como texto (el controlador lo verifica en el despliegue
con BEGIN/ROLLBACK); lo que sí se ejecuta aquí es el Python que lo alimenta.

Lo que cubre cada bloque (y por qué cada uno fallaría contra un módulo hecho a medias):
  1. el origen en la MISMA sentencia (`sql_con_origen`, `origen_de_ajustes`);
  2. el registro: fuentes que existen, cobertura del spec, claves vigiladas = las del trigger, las `avisos_*` de la app;
  3. `ajustes_de`: estados, «Otros ajustes», tipos raros, nada de contenido, cambiado_at/origen, fuentes rotas;
  4. `historial`, `resumen`, `columnas_csv`/`fila_csv`;
  5. `guardar_dispositivo` (lista cerrada, knob) y `PUT /api/profile/ajustes-dispositivo`;
  6. la purga, su cron y la exportación;
  7. los escritores: el coach y los apagados automáticos pasan por `sql_con_origen`;
  8. la migración (forma, trigger, idempotencia, copias idénticas).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ajustes_cuenta as ac

_BACKEND = Path(__file__).resolve().parents[1]
_MIG = "p1_plan_lote_837_ajustes_cambios_2026_09_29.sql"
UID = "33333333-3333-3333-3333-333333333333"
OTRA = "44444444-4444-4444-4444-444444444444"
TERCERA = "55555555-5555-5555-5555-555555555555"
ADMIN = "11111111-1111-1111-1111-111111111111"
T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
_SET_CONFIG = "set_config('mealfit.origen_ajuste', '{}', true)"


def _plano(q: str) -> str:
    return " ".join(str(q).split())


def _vacio(v) -> bool:
    """Lo que el SQL cuenta como vacío: null, false, "", [] y {} (un 0 NO lo es: jsonb 0 ≠ jsonb false)."""
    return v is None or v is False or (isinstance(v, (str, list, dict)) and len(v) == 0)


class _BD:
    """`user_profiles` y compañía en memoria. Entiende SOLO las sentencias que el módulo emite: cualquier otra revienta,
    así que un SQL nuevo no pasa sin que alguien mire este fake. La proyección del perfil imita la del SQL (solo las
    claves de ajustes + `avisos_*`, y de los paneles solo si están rellenos y su `updatedAt`)."""

    def __init__(self):
        self.perfiles: dict = {}
        self.tablas = {"device_push_tokens": {}, "push_subscriptions": {}, "user_brand_preferences": {},
                       "user_inventory": {}, "apple_signin_tokens": {}}
        self.kv: dict = {}
        self.cambios: list = []
        self.lecturas: list = []
        self.escrituras: list = []
        self.rotas: set = set()
        self.escritura_rota = None

    def perfil(self, uid, **cols):
        fila = {c: None for c in ac._COLUMNAS_PERFIL}
        fila.update(plan_mode="plan", long_term_memory_enabled=True, water_tracker_enabled=True, locale="es-DO",
                    ai_training_consent=False, ajustes_dispositivo={}, health_profile={})
        fila.update(cols)
        self.perfiles[uid] = fila
        return fila

    def cambio(self, uid, clave, antes, despues, origen, at):
        self.cambios.append({"id": len(self.cambios) + 1, "user_id": uid, "clave": clave, "antes": antes,
                             "despues": despues, "origen": origen, "at": at})

    def _proyeccion(self, uid, claves):
        f = self.perfiles[uid]
        hp = f.get("health_profile")
        hp = hp if isinstance(hp, dict) else {}
        fila = {c: f.get(c) for c in ac._COLUMNAS_PERFIL}
        fila["id"] = uid
        fila["hp"] = {k: v for k, v in hp.items() if k in claves or k.startswith("avisos_")}

        def _panel(clave):
            v = hp.get(clave)
            if not isinstance(v, dict):
                return {"relleno": False, "updatedAt": None}
            up = v.get("updatedAt")
            return {"relleno": any(k != "updatedAt" and not _vacio(x) for k, x in v.items()),
                    "updatedAt": None if up is None else str(up)}

        sf = hp.get("staple_foods")
        fila["hp_resumen"] = {"super_personalization": _panel("super_personalization"),
                              "clinical_profile": _panel("clinical_profile"),
                              "staple_foods": len(sf) if isinstance(sf, list) else None}
        return fila

    def _ordenados(self, filas):
        return sorted(filas, key=lambda c: (c["at"], c["id"]), reverse=True)

    def query(self, q, p=None, fetch_one=False, fetch_all=False):
        q = _plano(q)
        self.lecturas.append((q, p))
        for frag in self.rotas:
            if frag in q:
                raise RuntimeError(f"sin base: {frag}")
        if "FROM public.user_profiles p" in q and "AS hp_resumen" in q:
            claves, filtro = p
            if "WHERE p.id = ANY(%s::uuid[])" in q:
                ids = [u for u in filtro if u in self.perfiles]
            elif "WHERE p.id::text <> ALL(%s::text[])" in q:
                ids = [u for u in self.perfiles if u not in filtro]
            else:
                raise AssertionError(f"filtro de perfiles desconocido: {q[-120:]}")
            return [self._proyeccion(u, claves) for u in ids]
        for tabla, filas in self.tablas.items():
            if f"FROM public.{tabla} WHERE" in q:
                assert q.endswith("user_id = ANY(%s::uuid[]) GROUP BY user_id") or tabla == "apple_signin_tokens", q
                (ids,) = p
                return [{"id": u, "v": v} for u, v in filas.items() if u in ids]
        if "FROM public.app_kv_store WHERE key = ANY(%s::text[])" in q:
            (claves,) = p
            return [{"key": k, "value": v, "updated_at": t} for k, (v, t) in self.kv.items() if k in claves]
        if q == "SELECT count(*) AS n, max(at) AS ultimo FROM public.ajustes_cambios":     # el canario de la purga
            return {"n": len(self.cambios), "ultimo": max((c["at"] for c in self.cambios), default=None)}
        if "FROM public.ajustes_cambios" in q and q.endswith("GROUP BY clave, origen"):
            dias, fuera = p
            cuenta: dict = {}
            for c in self.cambios:
                if c["at"] >= T0 - timedelta(days=dias) and c["user_id"] not in fuera:
                    cuenta[(c["clave"], c["origen"])] = cuenta.get((c["clave"], c["origen"]), 0) + 1
            return [{"clave": k, "origen": o, "n": n} for (k, o), n in cuenta.items()]
        if "FROM public.ajustes_cambios WHERE user_id = %s AND at >= now() - make_interval(days => %s)" in q:
            uid, dias, limite = p
            filas = [c for c in self.cambios if c["user_id"] == uid and c["at"] >= T0 - timedelta(days=dias)]
            return [dict(c) for c in self._ordenados(filas)[:limite]]
        if "FROM public.ajustes_cambios WHERE user_id = %s ORDER BY at DESC, id DESC LIMIT %s" in q:
            uid, limite = p
            return [dict(c) for c in self._ordenados([c for c in self.cambios if c["user_id"] == uid])[:limite]]
        raise AssertionError(f"consulta que el fake no conoce: {q[:160]}")

    def write(self, q, p=None, returning=False, **k):
        q = _plano(q)
        self.escrituras.append((q, p))
        if self.escritura_rota:
            raise self.escritura_rota
        if q.startswith("UPDATE public.user_profiles SET ajustes_dispositivo = jsonb_set("):
            ruta, datos, uid = p
            if uid not in self.perfiles:
                return []
            disp = self.perfiles[uid].get("ajustes_dispositivo")
            disp = dict(disp) if isinstance(disp, dict) else {}
            disp[ruta[0]] = json.loads(datos)
            self.perfiles[uid]["ajustes_dispositivo"] = disp
            return [{"id": uid}]
        if q.startswith("DELETE FROM public.ajustes_cambios"):
            (dias,) = p
            viejos = [c for c in self.cambios if c["at"] < T0 - timedelta(days=dias)]
            self.cambios = [c for c in self.cambios if c not in viejos]
            return [{"id": c["id"]} for c in viejos]
        raise AssertionError(f"escritura que el fake no conoce: {q[:160]}")


@pytest.fixture
def bd(monkeypatch):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    monkeypatch.delenv("MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS", raising=False)
    b = _BD()
    monkeypatch.setattr(ac, "execute_sql_query", b.query)
    monkeypatch.setattr(ac, "execute_sql_write", b.write)
    return b


def _por_clave(ajustes):
    return {a["clave"]: a for a in ajustes}


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _cuerpo(src: str, firma: str) -> str:
    """El cuerpo de una función de nivel de módulo: desde su `def` hasta el siguiente `def`/`@`/`class` en columna 0."""
    i = src.index(firma)
    m = re.search(r"\n(?:def |async def |@|class )", src[i + len(firma):])
    return src[i:i + len(firma) + (m.start() if m else len(src))]


# ═════════════════════════════════════════════ 1. el origen en la MISMA sentencia
_UPDATE_SIMPLE = "UPDATE user_profiles SET water_tracker_enabled = %s WHERE id = %s RETURNING id"


def test_sin_origen_la_sentencia_no_cambia_ni_un_caracter():
    assert ac.origen_actual() is None
    assert ac.sql_con_origen(_UPDATE_SIMPLE) is _UPDATE_SIMPLE
    assert ac.sql_con_origen(_UPDATE_SIMPLE, None) is _UPDATE_SIMPLE


@pytest.mark.parametrize("origen", ["app", "coach", "sistema"])
def test_con_origen_la_marca_va_en_el_from_de_la_misma_sentencia(origen):
    sql = ac.sql_con_origen(_UPDATE_SIMPLE, origen)
    assert sql == (f"UPDATE user_profiles SET water_tracker_enabled = %s FROM (SELECT {_SET_CONFIG.format(origen)}) "
                   f"AS _origen WHERE id = %s RETURNING id")
    assert sql.count("%s") == _UPDATE_SIMPLE.count("%s"), "el origen va como literal: los parámetros no se mueven"


def test_el_where_de_nivel_cero_es_el_del_update_no_el_de_una_subconsulta_ni_un_literal():
    sql = ("UPDATE user_profiles p\n   SET nota = 'x WHERE y', n = (SELECT count(*) FROM t WHERE t.a = p.id)\n"
           " WHERE p.id IN (SELECT q.id FROM user_profiles q WHERE q.x = %s)\n   AND p.y IS NULL\nRETURNING p.id")
    out = ac.sql_con_origen(sql, "sistema")
    marca = f"FROM (SELECT {_SET_CONFIG.format('sistema')}) AS _origen "
    assert out.index(marca) == sql.index(" WHERE p.id IN") + 1
    assert out.replace(marca, "") == sql


def test_si_el_update_ya_tiene_from_la_marca_se_suma_a_la_lista():
    sql = "UPDATE a SET x = b.x FROM b WHERE a.id = b.id AND a.id = %s"
    assert ac.sql_con_origen(sql, "coach") == (
        f"UPDATE a SET x = b.x FROM b , (SELECT {_SET_CONFIG.format('coach')}) AS _origen WHERE a.id = b.id AND a.id = %s")


@pytest.mark.parametrize("sql", ["SELECT 1 FROM user_profiles WHERE id = %s",
                                 "UPDATE user_profiles SET x = 1",
                                 "DELETE FROM user_profiles WHERE id = %s"])
def test_una_sentencia_que_no_es_update_set_where_se_rechaza(sql):
    with pytest.raises(ValueError):
        ac.sql_con_origen(sql, "coach")


@pytest.mark.parametrize("origen", ["persona", "", "coach'); DROP TABLE x; --", "COACH"])
def test_un_origen_fuera_de_la_lista_se_rechaza_nunca_llega_al_sql(origen):
    with pytest.raises(ValueError):
        ac.sql_con_origen(_UPDATE_SIMPLE, origen)
    with pytest.raises(ValueError):
        with ac.origen_de_ajustes(origen):
            pass


def test_el_bloque_de_origen_se_aplica_dentro_y_se_restaura_fuera_tambien_con_error():
    with ac.origen_de_ajustes("coach"):
        assert ac.origen_actual() == "coach"
        assert _SET_CONFIG.format("coach") in ac.sql_con_origen(_UPDATE_SIMPLE)
        assert _SET_CONFIG.format("sistema") in ac.sql_con_origen(_UPDATE_SIMPLE, "sistema"), "el explícito manda"
        with ac.origen_de_ajustes("sistema"):
            assert ac.origen_actual() == "sistema"
        assert ac.origen_actual() == "coach"
    assert ac.origen_actual() is None
    with pytest.raises(RuntimeError):
        with ac.origen_de_ajustes("coach"):
            raise RuntimeError("falla dentro")
    assert ac.origen_actual() is None
    assert ac.sql_con_origen(_UPDATE_SIMPLE) is _UPDATE_SIMPLE


def test_el_guc_y_los_origenes_son_los_del_trigger():
    sql = _mig()
    assert ac.GUC_ORIGEN == "mealfit.origen_ajuste" and ac.ORIGENES == ("app", "coach", "sistema")
    assert "COALESCE(NULLIF(current_setting('mealfit.origen_ajuste', true), ''), 'app')" in sql
    assert "IF v_origen NOT IN ('app', 'coach', 'sistema') THEN" in sql


# ═════════════════════════════════════════════ 2. el registro
_BASE_USER_PROFILES = {"id", "email", "full_name", "created_at", "health_profile", "plan_tier", "logging_preference"}
_FUENTES_DEL_SPEC = {
    ("columna", "plan_mode"), ("columna", "logging_preference"), ("columna", "long_term_memory_enabled"),
    ("columna", "water_tracker_enabled"), ("columna", "nevera_enabled"), ("columna", "locale"),
    ("columna", "analytics_consent"), ("columna", "ai_training_consent"), ("columna", "ai_consent_version"),
    ("perfil", "avisos_comida"), ("perfil", "avisos_agua"), ("perfil", "avisos_por_comida.desayuno"),
    ("perfil", "avisos_por_comida.almuerzo"), ("perfil", "avisos_por_comida.merienda"),
    ("perfil", "avisos_por_comida.cena"), ("perfil", "country"), ("perfil", "groceryDuration"), ("perfil", "budget"),
    ("perfil", "budgetCurrency"), ("perfil", "weightUnit"), ("perfil", "super_personalization"),
    ("perfil", "clinical_profile"), ("perfil", "staple_foods"),
    ("tabla", "device_push_tokens"), ("tabla", "push_subscriptions"), ("tabla", "user_brand_preferences"),
    ("tabla", "user_inventory"), ("tabla", "apple_signin_tokens"),
    ("kv", "hydration_state:"), ("kv", "avisos_locales:"), ("kv", "plan_invite:"),
    ("dispositivo", "tema"), ("dispositivo", "notificaciones_permiso"), ("dispositivo", "alertas_activadas"),
    ("dispositivo", "analitica_vetada"), ("dispositivo", "barra_plegada"), ("dispositivo", "unidad_altura"),
    ("dispositivo", "avatar_elegido"), ("dispositivo", "plataforma"), ("dispositivo", "pwa"),
    ("dispositivo", "app_build"),
}


def _migraciones() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted((_BACKEND / "migrations").glob("*.sql")))


def test_el_registro_tiene_forma_y_cubre_lo_que_pide_el_spec():
    claves = [a.clave for a in ac.REGISTRO]
    assert len(claves) == len(set(claves)), "claves repetidas"
    etiquetas = [a.etiqueta for a in ac.REGISTRO]
    assert len(etiquetas) == len(set(etiquetas)), "etiquetas repetidas: en la ficha no se distinguirían"
    assert ac.GRUPOS == ("Uso", "Avisos", "Capacidades", "Privacidad", "Plan", "Dispositivo")
    for a in ac.REGISTRO:
        assert a.grupo in ac.GRUPOS, a
        assert a.fuente[0] in ("columna", "perfil", "tabla", "kv", "dispositivo"), a
        assert a.tipo in ac.TIPOS, a
        assert a.etiqueta and a.etiqueta != a.clave, a
    assert {a.fuente for a in ac.REGISTRO} >= _FUENTES_DEL_SPEC, _FUENTES_DEL_SPEC - {a.fuente for a in ac.REGISTRO}
    assert {a.grupo for a in ac.REGISTRO if a.fuente[0] == "dispositivo"} == {"Dispositivo"}


def test_cada_columna_del_registro_existe_en_una_migracion_o_en_la_base():
    sql = _migraciones()
    columnas = ({a.fuente[1] for a in ac.REGISTRO if a.fuente[0] == "columna"} | set(ac.COLUMNAS_VIGILADAS)
                | set(ac._COLUMNAS_PERFIL))
    faltan = [c for c in sorted(columnas)
              if c not in _BASE_USER_PROFILES and not re.search(rf"ADD COLUMN (IF NOT EXISTS )?{c}\b", sql)]
    assert not faltan, f"columnas del registro que ninguna migración crea: {faltan}"


def test_cada_tabla_kv_y_clave_de_dispositivo_del_registro_existe():
    sql = _migraciones()
    import db_profiles
    for a in ac.REGISTRO:
        tipo, nombre = a.fuente
        if tipo == "tabla":
            assert re.search(rf"(CREATE|ALTER) TABLE (IF NOT EXISTS )?(public\.)?{nombre}\b", sql), nombre
        elif tipo == "kv":
            assert nombre in db_profiles._USER_SCOPED_KV_PREFIXES, f"{nombre}: el borrado de cuenta no lo limpia"
        elif tipo == "dispositivo":
            assert nombre in set(ac.CLAVES_DISPOSITIVO) | {"plataforma", "pwa", "app_build"}, nombre
    import hydration_reminders
    import plan_invite
    assert {hydration_reminders.PREFIJO_ESTADO, hydration_reminders.PREFIJO_CANAL_LOCAL, plan_invite.PREFIJO} == {
        a.fuente[1] for a in ac.REGISTRO if a.fuente[0] == "kv"}


def test_el_perfil_de_salud_no_es_un_ajuste():
    """Peso, edad, alergias, dieta, condiciones, medicamentos… jamás se proyectan: solo las claves de ajustes."""
    sensibles = {"weight", "age", "gender", "height", "allergies", "dietType", "medicalConditions", "medications",
                 "dislikes", "freeText", "super_personalization", "clinical_profile", "staple_foods"}
    assert not sensibles & set(ac._HP_VALORES)
    assert set(ac._HP_VALORES) == set(ac.CLAVES_PERFIL_VIGILADAS)


def test_las_claves_vigiladas_son_las_del_trigger_y_todas_tienen_etiqueta():
    sql = _mig()
    cuerpo = sql.split("CREATE OR REPLACE FUNCTION public.registrar_cambio_ajustes()", 1)[1].split("$body$;", 1)[0]
    arrays = re.findall(r"FOREACH v_clave IN ARRAY ARRAY\[(.*?)\]::text\[\]", cuerpo, re.DOTALL)
    assert len(arrays) == 2, arrays
    columnas, perfil = ([x.strip().strip("'") for x in a.split(",")] for a in arrays)
    assert tuple(columnas) == ac.COLUMNAS_VIGILADAS
    assert tuple(perfil) == ac.CLAVES_PERFIL_VIGILADAS
    for clave in ac.COLUMNAS_VIGILADAS + ac.CLAVES_PERFIL_VIGILADAS:
        assert ac.etiqueta_de_cambio(clave) != clave, f"{clave}: el historial lo enseñaría sin nombre"
    assert ac.etiqueta_de_cambio("clave_rara") == "clave_rara", "una clave desconocida se enseña tal cual"


def test_cada_clave_avisos_que_escribe_la_app_esta_en_el_registro():
    """§13.1: una `avisos_*` nueva sale sola en «Otros ajustes»; este test pide darle su sitio y su etiqueta."""
    src = _BACKEND.parent / "frontend" / "src"
    if not src.exists():
        pytest.skip("sin el repo del frontend al lado")
    usadas = set()
    for f in list(src.rglob("*.js")) + list(src.rglob("*.jsx")):
        if "__tests__" in f.parts or "node_modules" in f.parts:
            continue
        usadas |= set(re.findall(r"['\"](avisos_[a-z_]+)['\"]", f.read_text(encoding="utf-8", errors="replace")))
    registradas = {a.fuente[1].split(".")[0] for a in ac.REGISTRO if a.fuente[0] == "perfil"}
    assert usadas, "el escáner no encontró ninguna clave avisos_: revisa el test"
    assert usadas <= registradas, f"claves avisos_ de la app sin registrar: {usadas - registradas}"


# ═════════════════════════════════════════════ 3. ajustes_de
_PERFIL_DE_EJEMPLO = {
    "avisos_agua": False,
    "avisos_por_comida": {"cena": {"activo": False, "hora": "20:15"}, "almuerzo": {"hora": "12:30"}},
    "country": "ES", "weightUnit": "kg", "avisos_foo": [1, 2],
    "super_personalization": {"freeText": "odio el cilantro", "likes": ["mangú"], "updatedAt": "2026-09-20T10:00:00Z"},
    "clinical_profile": {"updatedAt": "2026-09-21T10:00:00Z", "labs": {}},
    "staple_foods": ["arroz", "habichuelas"],
    "allergies": ["maní"], "medicalConditions": ["diabetes tipo 2"], "medications": ["metformina"], "weight": 181,
}


def test_ajustes_de_un_perfil_de_ejemplo(bd):
    bd.perfil(UID, nevera_enabled=None, health_profile=dict(_PERFIL_DE_EJEMPLO))
    r = ac.ajustes_de(UID)
    aj = _por_clave(r["ajustes"])
    assert aj["nevera_enabled"]["estado"] == "automatico", "Nevera NULL = automática"
    assert aj["avisos_comida"]["estado"] == "encendido", "ausente = encendido por defecto"
    assert aj["avisos_agua"]["estado"] == "apagado"
    assert (aj["avisos_por_comida.cena"]["estado"], aj["avisos_por_comida.cena"]["valor"]) == ("apagado", "20:15")
    assert (aj["avisos_por_comida.almuerzo"]["estado"], aj["avisos_por_comida.almuerzo"]["valor"]) == ("encendido", "12:30")
    assert (aj["avisos_por_comida.desayuno"]["estado"], aj["avisos_por_comida.desayuno"]["valor"]) == ("encendido", None)
    assert (aj["country"]["estado"], aj["country"]["valor"]) == ("valor", "ES")
    assert aj["groceryDuration"]["estado"] == "sin_elegir"
    assert (aj["weightUnit"]["estado"], aj["weightUnit"]["valor"]) == ("valor", "kg")
    assert (aj["locale"]["estado"], aj["locale"]["valor"]) == ("valor", "es-DO")
    assert (aj["plan_mode"]["estado"], aj["plan_mode"]["valor"]) == ("encendido", "plan")
    assert aj["logging_preference"]["estado"] == "apagado", "NULL = manual (el modo automático apagado)"
    assert aj["long_term_memory_enabled"]["estado"] == "encendido"
    assert aj["analytics_consent"]["estado"] == "sin_elegir", "NULL = nunca se le preguntó"
    assert aj["ai_training_consent"]["estado"] == "apagado"
    assert aj["ai_consent"]["estado"] == "sin_elegir"
    sp = aj["super_personalization"]
    assert (sp["estado"], sp["valor"], sp["cambiado_at"]) == ("valor", "relleno", "2026-09-20T10:00:00Z")
    cp = aj["clinical_profile"]
    assert (cp["estado"], cp["cambiado_at"]) == ("sin_elegir", "2026-09-21T10:00:00Z"), "guardado vacío no es relleno"
    assert (aj["staple_foods"]["estado"], aj["staple_foods"]["valor"]) == ("valor", 2)
    otros = [a for a in r["ajustes"] if a["grupo"] == "Otros ajustes"]
    assert otros == [{"clave": "avisos_foo", "etiqueta": "avisos_foo", "grupo": "Otros ajustes", "estado": "valor",
                      "valor": [1, 2], "cambiado_at": None, "origen": None}]
    for a in r["ajustes"]:
        assert set(a) == {"clave", "etiqueta", "grupo", "estado", "valor", "cambiado_at", "origen"}, a
        assert a["estado"] in ac.ESTADOS, a
    assert r["ajustes_dispositivo"] == {}
    todo = json.dumps(r, ensure_ascii=False, default=str)
    for secreto in ("cilantro", "mangú", "maní", "diabetes", "metformina", "labs", "arroz", "habichuelas", "181"):
        assert secreto not in todo, f"«{secreto}» es contenido de salud: no puede salir en los ajustes"


def test_la_proyeccion_del_perfil_pide_solo_las_claves_de_ajustes(bd):
    bd.perfil(UID, health_profile=dict(_PERFIL_DE_EJEMPLO))
    ac.ajustes_de(UID)
    q, p = next((q, p) for q, p in bd.lecturas if "AS hp_resumen" in q)
    assert p[0] == list(ac._HP_VALORES) and p[1] == [UID]
    assert "WHERE e.key = ANY(%s::text[]) OR left(e.key, 7) = 'avisos_'" in q
    assert "p.health_profile," not in q and "p.health_profile " not in q.split("AS hp_resumen")[1], (
        "el perfil de salud entero no viaja: solo su proyección")
    for panel in ("super_personalization", "clinical_profile"):
        assert f"p.health_profile -> '{panel}' ->> 'updatedAt'" in q, panel
    assert "jsonb_array_length(p.health_profile -> 'staple_foods')" in q


def test_aunque_la_proyeccion_trajera_contenido_la_ficha_no_lo_ensena(bd, monkeypatch):
    """Segunda capa: si un día la consulta devolviera más claves, el Python solo lee las del registro y `avisos_*`."""
    bd.perfil(UID, health_profile=dict(_PERFIL_DE_EJEMPLO))
    original = bd._proyeccion

    def _fuga(uid, claves):
        fila = original(uid, claves)
        fila["hp"] = dict(bd.perfiles[uid]["health_profile"])      # el perfil entero, como si la SQL fallara
        return fila
    monkeypatch.setattr(bd, "_proyeccion", _fuga)
    todo = json.dumps(ac.ajustes_de(UID), ensure_ascii=False, default=str)
    for secreto in ("cilantro", "maní", "diabetes", "metformina", "arroz"):
        assert secreto not in todo, secreto


def test_tipos_raros_no_revientan(bd):
    """Review Focus 4: una lista donde se espera un bool, `null`, un número… la ficha no revienta."""
    bd.perfil(UID, water_tracker_enabled=None, logging_preference="raro", analytics_consent="sí",
              plan_mode="otro", locale=None, ajustes_dispositivo=["no", "es", "objeto"],
              health_profile={"avisos_comida": [1], "avisos_agua": None, "avisos_por_comida": "roto",
                              "avisos_bar": {"x": None}, "avisos_baz": None, "country": 5, "budget": {"a": 1},
                              "budgetCurrency": "x" * 500, "super_personalization": "texto",
                              "clinical_profile": ["lista"], "staple_foods": {"a": 1}})
    r = ac.ajustes_de(UID)
    aj = _por_clave(r["ajustes"])
    assert (aj["avisos_comida"]["estado"], aj["avisos_comida"]["valor"]) == ("valor", [1])
    assert aj["avisos_agua"]["estado"] == "encendido", "null = ausente = encendido"
    assert aj["water_tracker_enabled"]["estado"] == "encendido"
    assert (aj["logging_preference"]["estado"], aj["logging_preference"]["valor"]) == ("valor", "raro")
    assert (aj["plan_mode"]["estado"], aj["plan_mode"]["valor"]) == ("valor", "otro")
    assert (aj["analytics_consent"]["estado"], aj["analytics_consent"]["valor"]) == ("valor", "sí")
    assert (aj["locale"]["estado"], aj["locale"]["valor"]) == ("valor", "es-DO"), "NULL = el idioma por defecto"
    for comida in ("desayuno", "almuerzo", "merienda", "cena"):
        assert (aj[f"avisos_por_comida.{comida}"]["estado"], aj[f"avisos_por_comida.{comida}"]["valor"]) == (
            "valor", "roto")
    assert (aj["country"]["estado"], aj["country"]["valor"]) == ("valor", 5)
    assert aj["budget"]["estado"] == "valor"
    assert len(aj["budgetCurrency"]["valor"]) <= ac.MAX_TEXTO + 1, "un texto largo se recorta"
    assert aj["super_personalization"]["estado"] == "sin_elegir" and aj["clinical_profile"]["estado"] == "sin_elegir"
    assert aj["staple_foods"]["estado"] == "sin_elegir"
    otros = {a["clave"]: a for a in r["ajustes"] if a["grupo"] == "Otros ajustes"}
    assert set(otros) == {"avisos_bar", "avisos_baz"} and otros["avisos_baz"]["valor"] is None
    assert r["ajustes_dispositivo"] == {}, "un ajustes_dispositivo que no es objeto no revienta: sale vacío"
    json.dumps(r)


def test_health_profile_que_no_es_un_objeto_da_los_valores_por_defecto(bd):
    bd.perfil(UID, health_profile=["no", "soy", "un", "objeto"])
    aj = _por_clave(ac.ajustes_de(UID)["ajustes"])
    assert aj["avisos_comida"]["estado"] == "encendido" and aj["country"]["estado"] == "sin_elegir"
    assert (aj["weightUnit"]["estado"], aj["weightUnit"]["valor"]) == ("valor", "lb"), "lb es el defecto"


def test_las_tablas_y_el_kv_dan_su_estado(bd):
    bd.perfil(UID)
    bd.tablas["device_push_tokens"][UID] = ["android", "ios"]
    bd.tablas["push_subscriptions"][UID] = 2
    bd.tablas["user_brand_preferences"][UID] = 3
    bd.tablas["user_inventory"][UID] = 1
    bd.tablas["apple_signin_tokens"][UID] = True
    bd.kv["hydration_state:" + UID] = ({"auto_off_at": "2026-09-27T09:00:00+00:00", "nudges_ignorados": 4},
                                        T0 - timedelta(days=2))
    bd.kv["avisos_locales:" + UID] = ({"canal": "local"}, datetime.now(timezone.utc) - timedelta(hours=5))
    bd.kv["plan_invite:" + UID] = (json.dumps({"shown_at": "2026-09-20T10:00:00+00:00",
                                               "dismissed_at": "2026-09-21T10:00:00+00:00"}), T0)
    aj = _por_clave(ac.ajustes_de(UID)["ajustes"])
    assert (aj["push_app"]["estado"], aj["push_app"]["valor"]) == ("encendido", "android,ios")
    assert (aj["push_web"]["estado"], aj["push_web"]["valor"]) == ("encendido", 2)
    assert (aj["marcas_elegidas"]["estado"], aj["marcas_elegidas"]["valor"]) == ("valor", 3)
    assert (aj["suplementos"]["estado"], aj["suplementos"]["valor"]) == ("valor", 1)
    assert (aj["entro_con_apple"]["estado"], aj["entro_con_apple"]["valor"]) == ("valor", "sí")
    h = aj["hidratacion_apagada_sola"]
    assert (h["estado"], h["valor"], h["cambiado_at"], h["origen"]) == (
        "valor", "sí", "2026-09-27T09:00:00+00:00", "sistema")
    assert aj["avisos_locales"]["estado"] == "encendido"
    inv = aj["invitacion_al_plan"]
    assert (inv["estado"], inv["valor"], inv["cambiado_at"]) == ("valor", "descartada", "2026-09-21T10:00:00+00:00")
    # y una cuenta sin nada de eso
    bd.perfil(OTRA)
    otra = _por_clave(ac.ajustes_de(OTRA)["ajustes"])
    assert otra["push_app"]["estado"] == "apagado" and otra["push_web"]["estado"] == "apagado"
    assert otra["marcas_elegidas"]["estado"] == "sin_elegir" and otra["suplementos"]["valor"] == 0
    assert otra["entro_con_apple"]["valor"] == "no" and otra["hidratacion_apagada_sola"]["valor"] == "no"
    assert otra["avisos_locales"]["estado"] == "sin_elegir" and otra["invitacion_al_plan"]["estado"] == "sin_elegir"


def test_el_canal_local_caduca_a_las_72_horas(bd):
    bd.perfil(UID)
    bd.kv["avisos_locales:" + UID] = ({"canal": "local"}, datetime.now(timezone.utc) - timedelta(hours=73))
    assert _por_clave(ac.ajustes_de(UID)["ajustes"])["avisos_locales"]["estado"] == "apagado"


def test_el_canal_local_toma_sus_horas_de_hydration_reminders(bd, monkeypatch):
    """Una sola definición de cuánto vale un teléfono sincronizado: `hydration_reminders.HORAS_DE_ALCANCE_LOCAL`. Antes
    era una copia a mano de 72 que se habría desfasado en silencio el día que alguien cambiara la del agua."""
    import hydration_reminders as hr
    assert hr.HORAS_DE_ALCANCE_LOCAL == 72 and ac._horas_canal_local() == 72
    assert not hasattr(ac, "_HORAS_CANAL_LOCAL"), "la copia a mano ya no existe"
    assert "from hydration_reminders import HORAS_DE_ALCANCE_LOCAL" in _src("ajustes_cuenta.py")
    bd.perfil(UID)
    bd.kv["avisos_locales:" + UID] = ({"canal": "local"}, datetime.now(timezone.utc) - timedelta(hours=30))
    assert _por_clave(ac.ajustes_de(UID)["ajustes"])["avisos_locales"]["estado"] == "encendido"     # 30 h < 72 h
    monkeypatch.setattr(hr, "HORAS_DE_ALCANCE_LOCAL", 24)
    assert ac._horas_canal_local() == 24
    assert _por_clave(ac.ajustes_de(UID)["ajustes"])["avisos_locales"]["estado"] == "apagado"        # 30 h > 24 h


def test_el_idioma_pedido_al_coach_se_documenta_como_cambio_de_la_app():
    """El coach solo devuelve el marcador `{"idioma": …}`; lo GUARDA el cliente, así que el historial lo registra como
    `app` (lo mismo que un cambio de `update_form_field`). Solo `cambiar_ajuste_de_la_app` corre en el bloque `coach`."""
    doc = " ".join(ac.__doc__.split())
    assert "Tampoco el IDIOMA que la persona le pide al coach" in doc
    assert "lo GUARDA el cliente" in doc and "también queda como `app`" in doc
    assert "update_form_field" in doc, "junto al otro caso conocido"
    src = _src("ajustes_de_la_app.py")
    assert src.count('origen_de_ajustes("coach")') == 1, "un único bloque `coach`: el de los ajustes del servidor"
    assert "_con_marcador(" in _cuerpo(src, "def _cambiar_idioma"), "el idioma va en el marcador, no se escribe aquí"


def test_el_permiso_para_la_ia_en_sus_cuatro_estados(bd, monkeypatch):
    import consentimientos
    v = consentimientos.AI_CONSENT_VERSION
    casos = {
        UID: dict(ai_consent_version=v, ai_consent_at=T0, ai_cn_transfer_at=T0),
        OTRA: dict(ai_consent_version=v, ai_consent_at=T0 - timedelta(days=2), ai_cn_transfer_at=T0,
                   ai_consent_revoked_at=T0 - timedelta(days=1)),
        TERCERA: dict(ai_consent_version="ia-2020-01", ai_consent_at=T0, ai_cn_transfer_at=T0),
    }
    esperado = {UID: ("encendido", v), OTRA: ("apagado", v), TERCERA: ("valor", "ia-2020-01")}
    for uid, cols in casos.items():
        bd.perfil(uid, **cols)
        a = _por_clave(ac.ajustes_de(uid)["ajustes"])["ai_consent"]
        assert (a["estado"], a["valor"]) == esperado[uid], uid
    assert _por_clave(ac.ajustes_de(OTRA)["ajustes"])["ai_consent"]["cambiado_at"] == (T0 - timedelta(days=1)).isoformat()


def test_los_ajustes_del_dispositivo_por_plataforma_y_el_mas_reciente_manda(bd):
    bd.perfil(UID, ajustes_dispositivo={
        "web": {"tema": "dark", "pwa": True, "alertas_activadas": True, "at": "2026-09-28T10:00:00+00:00",
                "clave_vieja": "x"},
        "ios": {"tema": "light", "notificaciones_permiso": "granted", "app_build": "1.0 (106)",
                "at": "2026-09-29T08:00:00+00:00"},
        "consola": {"tema": "dark"},
    })
    r = ac.ajustes_de(UID)
    assert set(r["ajustes_dispositivo"]) == {"web", "ios"}, "solo plataformas de la lista"
    assert "clave_vieja" not in r["ajustes_dispositivo"]["web"], "solo claves de la lista"
    aj = _por_clave(r["ajustes"])
    assert (aj["tema"]["valor"], aj["tema"]["cambiado_at"]) == ("light", "2026-09-29T08:00:00+00:00")
    assert (aj["alertas_activadas"]["estado"], aj["alertas_activadas"]["cambiado_at"]) == (
        "encendido", "2026-09-28T10:00:00+00:00"), "la clave sale del informe más nuevo QUE LA TRAE"
    assert aj["notificaciones_permiso"]["valor"] == "granted" and aj["pwa"]["estado"] == "encendido"
    assert aj["app_build"]["valor"] == "1.0 (106)"
    assert (aj["plataformas"]["estado"], aj["plataformas"]["valor"]) == ("valor", "ios,web")
    assert aj["barra_plegada"]["estado"] == "sin_elegir"


def test_cambiado_at_y_origen_salen_del_historial_y_si_no_de_su_marca(bd):
    bd.perfil(UID, nevera_enabled=False, nevera_auto_off_at=T0 - timedelta(days=3), water_tracker_enabled=False,
              plan_mode="tracking", plan_mode_changed_at=T0 - timedelta(days=10),
              health_profile={"avisos_por_comida": {"cena": {"activo": False}, "desayuno": {"activo": True}}})
    bd.kv["hydration_state:" + UID] = ({"auto_off_at": (T0 - timedelta(days=2)).isoformat()}, T0 - timedelta(days=2))
    bd.cambio(UID, "avisos_por_comida", {"desayuno": {"activo": True}},
              {"desayuno": {"activo": True}, "cena": {"activo": False}}, "coach", T0 - timedelta(hours=5))
    aj = _por_clave(ac.ajustes_de(UID)["ajustes"])
    assert (aj["avisos_por_comida.cena"]["cambiado_at"], aj["avisos_por_comida.cena"]["origen"]) == (
        (T0 - timedelta(hours=5)).isoformat(), "coach")
    assert aj["avisos_por_comida.desayuno"]["cambiado_at"] is None, "ese cambio no tocó el desayuno"
    assert (aj["nevera_enabled"]["cambiado_at"], aj["nevera_enabled"]["origen"]) == (
        (T0 - timedelta(days=3)).isoformat(), "sistema"), "sin historial: la marca del apagado automático"
    assert (aj["water_tracker_enabled"]["cambiado_at"], aj["water_tracker_enabled"]["origen"]) == (
        (T0 - timedelta(days=2)).isoformat(), "sistema")
    assert (aj["plan_mode"]["cambiado_at"], aj["plan_mode"]["origen"]) == ((T0 - timedelta(days=10)).isoformat(), None)
    bd.cambio(UID, "plan_mode", "plan", "tracking", "coach", T0 - timedelta(hours=1))
    aj = _por_clave(ac.ajustes_de(UID)["ajustes"])
    assert (aj["plan_mode"]["cambiado_at"], aj["plan_mode"]["origen"]) == ((T0 - timedelta(hours=1)).isoformat(), "coach")
    q, p = next((q, p) for q, p in bd.lecturas if q.startswith("SELECT") and "ORDER BY at DESC, id DESC LIMIT %s" in q
                and "make_interval" not in q)
    assert p == (UID, ac.MAX_HISTORIAL)


def test_una_fuente_rota_quita_sus_ajustes_y_no_tumba_la_ficha(bd):
    bd.perfil(UID)
    bd.rotas |= {"FROM public.apple_signin_tokens", "FROM public.app_kv_store", "FROM public.ajustes_cambios"}
    claves = {a["clave"] for a in ac.ajustes_de(UID)["ajustes"]}
    assert {"entro_con_apple", "hidratacion_apagada_sola", "avisos_locales", "invitacion_al_plan"}.isdisjoint(claves), (
        "sin su fuente no se inventa un estado: se omite")
    assert {"nevera_enabled", "push_app", "suplementos", "tema"} <= claves


def test_si_el_perfil_no_se_puede_leer_lanza_y_si_no_existe_devuelve_vacio(bd):
    assert ac.ajustes_de("guest") == {"ajustes": [], "ajustes_dispositivo": {}} and bd.lecturas == []
    assert ac.ajustes_de(UID) == {"ajustes": [], "ajustes_dispositivo": {}}, "sin perfil"
    bd.rotas.add("FROM public.user_profiles p")
    with pytest.raises(RuntimeError):
        ac.ajustes_de(UID)


def test_ajustes_de_varias_en_una_consulta_por_fuente(bd):
    for u in (UID, OTRA, TERCERA):
        bd.perfil(u, nevera_enabled=(u == UID))
    r = ac.ajustes_de_varias([UID, OTRA, "guest", UID])
    assert set(r) == {UID, OTRA}
    assert _por_clave(r[UID]["ajustes"])["nevera_enabled"]["estado"] == "encendido"
    assert _por_clave(r[OTRA]["ajustes"])["nevera_enabled"]["estado"] == "apagado"
    perfiles = [p for q, p in bd.lecturas if "AS hp_resumen" in q]
    assert len(perfiles) == 1 and sorted(perfiles[0][1]) == sorted([UID, OTRA])
    assert not any("FROM public.ajustes_cambios" in q for q, _ in bd.lecturas), "en bloque no se lee el historial"
    assert ac.ajustes_de_varias([]) == {} and ac.ajustes_de_varias(["guest"]) == {}


# ═════════════════════════════════════════════ 4. historial, resumen y CSV
def test_historial_mas_nuevo_primero_con_etiqueta_y_limite(bd):
    bd.cambio(UID, "nevera_enabled", None, False, "sistema", T0 - timedelta(days=1))
    bd.cambio(UID, "locale", "es-DO", "en-US", "app", T0 - timedelta(hours=2))
    bd.cambio(UID, "avisos_comida", True, False, "coach", T0 - timedelta(hours=2))
    bd.cambio(UID, "plan_mode", "plan", "tracking", "app", T0 - timedelta(days=200))
    bd.cambio(OTRA, "locale", "es-DO", "fr-FR", "app", T0)
    h = ac.historial(UID)
    assert [c["clave"] for c in h] == ["avisos_comida", "locale", "nevera_enabled"], "90 días, lo más nuevo primero"
    assert h[0] == {"at": (T0 - timedelta(hours=2)).isoformat(), "clave": "avisos_comida",
                    "etiqueta": ac.etiqueta_de_cambio("avisos_comida"), "antes": True, "despues": False,
                    "origen": "coach"}
    q, p = next((q, p) for q, p in bd.lecturas if "make_interval" in q)
    assert p == (UID, 90, ac.MAX_HISTORIAL) and ac.MAX_HISTORIAL == 500
    assert len(ac.historial(UID, dias=365)) == 4
    assert [p for q, p in bd.lecturas if "make_interval" in q][-1][1] == 365
    ac.historial(UID, dias=0)
    ac.historial(UID, dias=9999)
    assert [p[1] for q, p in bd.lecturas if "make_interval" in q][-2:] == [1, 365], "acotado a 1..365"
    n = len(bd.lecturas)
    assert ac.historial("guest") == [] and len(bd.lecturas) == n, "un id que no es uuid no consulta"


def test_resumen_cuenta_por_ajuste_sin_admins_y_cambios_por_origen(bd, monkeypatch):
    import admin_metricas
    monkeypatch.setattr(admin_metricas, "_ids_fuera", lambda: [ADMIN])
    bd.perfil(UID, nevera_enabled=None, locale="es-DO",
              ajustes_dispositivo={"web": {"tema": "dark", "pwa": True, "at": "2026-09-29T10:00:00+00:00"}})
    bd.perfil(OTRA, nevera_enabled=False, locale="en-US", health_profile={"avisos_comida": False},
              ajustes_dispositivo={"ios": {"tema": "dark", "notificaciones_permiso": "denied", "at": "2026-09-29T10:00:00+00:00"},
                                   "web": {"tema": "light", "at": "2026-09-28T10:00:00+00:00"}})
    bd.perfil(TERCERA, nevera_enabled=True, health_profile={"avisos_foo": True})
    bd.perfil(ADMIN, nevera_enabled=False, locale="it-IT")
    bd.cambio(UID, "nevera_enabled", None, False, "sistema", T0 - timedelta(days=2))
    bd.cambio(OTRA, "nevera_enabled", None, False, "sistema", T0 - timedelta(days=3))
    bd.cambio(OTRA, "avisos_comida", True, False, "coach", T0 - timedelta(days=1))
    bd.cambio(UID, "locale", "en-US", "es-DO", "app", T0 - timedelta(days=60))
    bd.cambio(ADMIN, "locale", "es-DO", "it-IT", "app", T0 - timedelta(days=1))
    r = ac.resumen(dias=30)
    assert set(r) == {"cuentas", "ajustes", "cambios", "dispositivo"}
    assert r["cuentas"] == 3
    aj = {a["clave"]: a for a in r["ajustes"]}
    assert [a["clave"] for a in r["ajustes"]] == [a.clave for a in ac.REGISTRO], "todas, en el orden del registro"
    assert aj["nevera_enabled"]["conteo"] == {"encendido": 1, "apagado": 1, "automatico": 1, "sin_elegir": 0,
                                              "valores": {}}
    assert aj["locale"]["conteo"]["valores"] == {"es-DO": 2, "en-US": 1}, "la cuenta admin no cuenta"
    assert (aj["avisos_comida"]["conteo"]["encendido"], aj["avisos_comida"]["conteo"]["apagado"]) == (2, 1)
    assert aj["nevera_enabled"]["etiqueta"] and aj["nevera_enabled"]["grupo"] == "Capacidades"
    assert "avisos_foo" not in aj, "«Otros ajustes» no entra en el agregado"
    cambios = {c["clave"]: c for c in r["cambios"]}
    assert cambios["nevera_enabled"]["por_origen"] == {"app": 0, "coach": 0, "sistema": 2}
    assert cambios["avisos_comida"]["por_origen"] == {"app": 0, "coach": 1, "sistema": 0}
    assert "locale" not in cambios, "el de hace 60 días queda fuera del periodo y el del admin no cuenta"
    assert r["cambios"][0]["clave"] == "nevera_enabled" and cambios["nevera_enabled"]["etiqueta"]
    q, p = next((q, p) for q, p in bd.lecturas if "GROUP BY clave, origen" in q)
    assert p == (30, [ADMIN])
    assert r["dispositivo"]["tema"] == {"dark": 2, "light": 1}, "un informe por cuenta y plataforma"
    assert r["dispositivo"]["notificaciones_permiso"] == {"denied": 1}
    assert r["dispositivo"]["plataformas"] == {"web": 2, "ios": 1, "pwa": 1}
    ac.resumen(dias=500)
    assert [p for q, p in bd.lecturas if "GROUP BY clave, origen" in q][-1][0] == 90, "acotado a 1..90"


def test_resumen_sin_historial_legible_da_cambios_vacios(bd, monkeypatch):
    import admin_metricas
    monkeypatch.setattr(admin_metricas, "_ids_fuera", lambda: [])
    bd.perfil(UID)
    bd.rotas.add("FROM public.ajustes_cambios")
    r = ac.resumen()
    assert r["cuentas"] == 1 and r["cambios"] == []


def test_csv_una_columna_por_ajuste_del_registro(bd):
    bd.perfil(UID, nevera_enabled=None, health_profile={"country": "DO", "budget": "=HYPERLINK(\"x\")",
                                                        "avisos_foo": True})
    cols = ac.columnas_csv()
    assert cols == [f"ajuste.{a.clave}" for a in ac.REGISTRO]
    fila = ac.fila_csv(ac.ajustes_de(UID)["ajustes"])
    assert len(fila) == len(cols) and all(isinstance(x, str) for x in fila)
    valores = dict(zip(cols, fila))
    assert valores["ajuste.nevera_enabled"] == "automatico"
    assert valores["ajuste.country"] == "DO", "un enumerado sale con su valor"
    assert valores["ajuste.avisos_comida"] == "encendido"
    assert valores["ajuste.budget"].startswith("'="), "una fórmula no se ejecuta al abrir el CSV"
    assert ac.fila_csv([]) == [""] * len(cols) and ac.fila_csv(None) == [""] * len(cols)


# ═════════════════════════════════════════════ 5. ajustes del dispositivo
_INFORME = {
    "plataforma": "ios", "pwa": "yes", "app_build": "  1.0.3 (106)  ", "otra": "x",
    "ajustes": {"tema": "dark", "notificaciones_permiso": "maybe", "alertas_activadas": 1, "analitica_vetada": False,
                "barra_plegada": True, "unidad_altura": "ft", "avatar_elegido": False, "desconocida": True},
}


def test_guardar_dispositivo_con_el_knob_apagado_no_escribe(bd):
    bd.perfil(UID)
    assert ac.guardar_dispositivo(UID, dict(_INFORME)) is False
    assert bd.escrituras == []


def test_guardar_dispositivo_guarda_solo_la_lista_cerrada_por_plataforma(bd, monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    bd.perfil(UID, ajustes_dispositivo={"web": {"tema": "light", "at": "2026-09-01T00:00:00+00:00"}})
    assert ac.guardar_dispositivo(UID, dict(_INFORME)) is True
    q, p = bd.escrituras[0]
    assert q == ("UPDATE public.user_profiles SET ajustes_dispositivo = jsonb_set(CASE WHEN "
                 "jsonb_typeof(ajustes_dispositivo) = 'object' THEN ajustes_dispositivo ELSE '{}'::jsonb END, "
                 "%s::text[], %s::jsonb, true) WHERE id = %s RETURNING id")
    ruta, datos, uid = p
    datos = json.loads(datos)
    assert ruta == ["ios"] and uid == UID
    at = datos.pop("at")
    assert datetime.fromisoformat(at).tzinfo is not None
    assert datos == {"app_build": "1.0.3 (106)", "tema": "dark", "analitica_vetada": False, "barra_plegada": True,
                     "unidad_altura": "ft", "avatar_elegido": False}
    disp = bd.perfiles[UID]["ajustes_dispositivo"]
    assert set(disp) == {"web", "ios"} and disp["web"]["tema"] == "light", "cada plataforma en su clave"


@pytest.mark.parametrize("cuerpo", [
    {"plataforma": "consola", "ajustes": {"tema": "dark"}},
    {"ajustes": {"tema": "dark"}},
    {"plataforma": ["ios"]},
    "no soy un objeto",
    None,
])
def test_guardar_dispositivo_sin_plataforma_valida_no_escribe(bd, monkeypatch, cuerpo):
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    bd.perfil(UID)
    assert ac.guardar_dispositivo(UID, cuerpo) is False and bd.escrituras == []


def test_guardar_dispositivo_cuerpos_raros_y_base_caida(bd, monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    bd.perfil(UID)
    assert ac.guardar_dispositivo("guest", {"plataforma": "web"}) is False and bd.escrituras == []
    assert ac.guardar_dispositivo(UID, {"plataforma": "web", "ajustes": ["tema"], "app_build": "x" * 65}) is True
    assert json.loads(bd.escrituras[-1][1][1]).keys() == {"at"}, "lo que no se entiende se descarta, el informe queda"
    assert ac.guardar_dispositivo(UID, {"plataforma": "web", "app_build": "<script>"}) is True
    assert "app_build" not in json.loads(bd.escrituras[-1][1][1])
    assert ac.guardar_dispositivo(OTRA, {"plataforma": "web"}) is False, "sin perfil no hay fila"
    bd.escritura_rota = RuntimeError("sin DB")
    assert ac.guardar_dispositivo(UID, {"plataforma": "web"}) is False, "nunca lanza"


def _user_data():
    from routers import user_data
    return user_data


def test_el_endpoint_del_dispositivo_responde_el_contrato(monkeypatch):
    ud = _user_data()
    vistos = []
    monkeypatch.setattr(ac, "guardar_dispositivo", lambda uid, cuerpo: vistos.append((uid, cuerpo)) or True)
    r = asyncio.run(ud.api_put_ajustes_dispositivo(data={"plataforma": "web"}, verified_user_id=UID))
    assert r == {"ok": True, "guardado": True} and vistos == [(UID, {"plataforma": "web"})]
    monkeypatch.setattr(ac, "guardar_dispositivo", lambda uid, cuerpo: False)
    assert asyncio.run(ud.api_put_ajustes_dispositivo(data={}, verified_user_id=UID)) == {"ok": True, "guardado": False}
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(ud.api_put_ajustes_dispositivo(data={"plataforma": "web"}, verified_user_id=None))
    assert ei.value.status_code == 401


def test_el_endpoint_del_dispositivo_tiene_su_limitador_propio_y_no_pasa_por_la_cuota():
    src = _src("routers/user_data.py")
    assert "_AJUSTES_DISPOSITIVO_LIMITER = RateLimiter(max_calls=6, period_seconds=60)" in src
    i = src.index('@router.put("/profile/ajustes-dispositivo")')
    bloque = src[i:i + 1500].split("\n@router.", 1)[0]
    assert "Depends(_AJUSTES_DISPOSITIVO_LIMITER)" in bloque and "verify_api_quota" not in bloque
    assert re.fullmatch(r"_[A-Z_]+_LIMITER", "_AJUSTES_DISPOSITIVO_LIMITER"), "el guard de limitadores no admite dígitos"
    pares = []
    for f in [*_BACKEND.glob("*.py"), *(_BACKEND / "routers").glob("*.py")]:
        pares += re.findall(r"RateLimiter\(\s*(?:max_calls\s*=\s*)?(\d+)\s*,\s*(?:period_seconds\s*=\s*)?(\d+)\s*\)",
                            f.read_text(encoding="utf-8"))
    assert pares.count(("6", "60")) == 1, "el par (6, 60) es solo suyo: Redis comparte la ventana por par"


# ═════════════════════════════════════════════ 6. purga, cron y exportación
def test_la_purga_usa_el_plazo_del_knob_acotado_y_nunca_lanza(bd, monkeypatch):
    assert ac.dias_de_historial() == 730
    bd.cambio(UID, "locale", "es-DO", "en-US", "app", T0 - timedelta(days=800))
    bd.cambio(UID, "locale", "en-US", "es-DO", "app", T0 - timedelta(days=10))
    assert ac.purgar_cambios_antiguos() == 1 and len(bd.cambios) == 1
    q, p = bd.escrituras[-1]
    assert q == "DELETE FROM public.ajustes_cambios WHERE at < now() - make_interval(days => %s) RETURNING id"
    assert p == (730,)
    for crudo, dias in (("10", 730), ("5000", 730), ("abc", 730), ("90", 90), ("3650", 3650)):
        monkeypatch.setenv("MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS", crudo)
        assert ac.dias_de_historial() == dias, crudo
    bd.escritura_rota = RuntimeError("sin DB")
    assert ac.purgar_cambios_antiguos() == 0


def test_la_purga_deja_un_canario_diario_con_el_conteo_y_el_ultimo_cambio(bd, caplog):
    """El trigger que llena la tabla avisa de sus fallos con un WARNING de POSTGRES, que no llega a los logs de la app:
    una tabla que dejó de llenarse pasaría meses callada. La pasada diaria de la purga deja `count(*)` y `max(at)`."""
    bd.cambio(UID, "locale", "es-DO", "en-US", "app", T0 - timedelta(days=800))
    bd.cambio(UID, "locale", "en-US", "es-DO", "app", T0 - timedelta(days=10))
    bd.cambio(OTRA, "nevera_enabled", None, False, "sistema", T0 - timedelta(days=2))
    caplog.set_level(logging.INFO, logger="ajustes_cuenta")
    assert ac.purgar_cambios_antiguos() == 1
    assert "SELECT count(*) AS n, max(at) AS ultimo FROM public.ajustes_cambios" in [q for q, _ in bd.lecturas]
    canario = [r.getMessage() for r in caplog.records if "canario" in r.getMessage()]
    assert len(canario) == 1 and "P1-PLAN-LOTE-837" in canario[0]
    assert "2 cambios" in canario[0], "el conteo es el de DESPUÉS de purgar"
    assert (T0 - timedelta(days=2)).isoformat() in canario[0], "y el último es el `max(at)` de lo que queda"


def test_el_canario_de_una_tabla_vacia_dice_ninguno(bd, caplog):
    caplog.set_level(logging.INFO, logger="ajustes_cuenta")
    assert ac.purgar_cambios_antiguos() == 0
    canario = [r.getMessage() for r in caplog.records if "canario" in r.getMessage()]
    assert len(canario) == 1 and "0 cambios" in canario[0] and "ninguno" in canario[0]


def test_el_canario_corre_aunque_la_purga_falle_y_nunca_la_rompe(bd, caplog):
    caplog.set_level(logging.INFO, logger="ajustes_cuenta")
    bd.escritura_rota = RuntimeError("sin DB")
    assert ac.purgar_cambios_antiguos() == 0
    assert any("canario" in r.getMessage() for r in caplog.records), "la purga rota no apaga el canario"
    # el canario mismo roto (solo él lee `max(at)`): la purga sigue devolviendo lo que borró y no lanza
    caplog.clear()
    bd.escritura_rota = None
    bd.rotas.add("max(at)")
    bd.cambio(UID, "locale", "es-DO", "en-US", "app", T0 - timedelta(days=800))
    assert ac.purgar_cambios_antiguos() == 1 and bd.cambios == []
    assert any("canario" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)
    bd.escritura_rota = RuntimeError("sin DB")           # purga Y canario rotos: tampoco lanza
    assert ac.purgar_cambios_antiguos() == 0


def test_el_knob_del_plazo_queda_en_el_inventario(monkeypatch):
    from knobs import get_knobs_registry_snapshot
    monkeypatch.delenv("MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS", raising=False)
    ac.dias_de_historial()
    assert get_knobs_registry_snapshot()["MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS"]["default"] == 730


def test_el_cron_diario_de_la_purga_esta_registrado():
    from cron_tasks import register_plan_chunk_scheduler

    fake_scheduler = MagicMock()
    fake_scheduler.get_job.return_value = None
    register_plan_chunk_scheduler(fake_scheduler)
    llamadas = [c for c in fake_scheduler.add_job.call_args_list if c.kwargs.get("id") == "purge_ajustes_cambios"]
    assert len(llamadas) == 1
    llamada = llamadas[0]
    registrada = getattr(llamada.args[0], "__wrapped__", llamada.args[0])
    assert registrada is ac.purgar_cambios_antiguos
    assert llamada.args[1] == "interval" and llamada.kwargs.get("hours") == 24
    assert llamada.kwargs.get("max_instances") == 1 and llamada.kwargs.get("coalesce") is True
    assert "jitter" in llamada.kwargs  # pasó por `_add_job_jittered`


def test_la_exportacion_lleva_el_historial_ordenado_por_at():
    src = _src("app.py")
    tablas = re.search(r"_ACCOUNT_EXPORT_TABLES\s*=\s*\((.*?)\n\)", src, re.DOTALL).group(1)
    assert re.search(r'\("ajustes_cambios",\s*"user_id",\s*5000\)', tablas)
    import app as app_module
    assert app_module._ACCOUNT_EXPORT_ORDER["ajustes_cambios"] == "at DESC"
    sql, params = app_module._account_export_query("ajustes_cambios", "user_id", 5000, UID, "SELECT 1")
    assert "FROM (SELECT * FROM public.ajustes_cambios) AS t WHERE user_id = %s" in sql
    assert "ORDER BY at DESC LIMIT 5000" in sql and "created_at" not in sql, (
        "la tabla no tiene created_at: sin su orden la exportación saldría incompleta")
    assert params[1:] == (UID,)


# ═════════════════════════════════════════════ 7. los escritores ponen su origen
def _captura(destino, devuelve=True):
    def _w(q, p=None, returning=False, **k):
        destino.append((_plano(q), p))
        return [{"id": UID}] if returning else devuelve
    return _w


@pytest.fixture
def perfiles_db(monkeypatch):
    import db_profiles
    escrito = []
    monkeypatch.setattr(db_profiles, "connection_pool", object())
    monkeypatch.setattr(db_profiles, "execute_sql_write", _captura(escrito))
    return escrito


def test_los_interruptores_de_configuracion_no_cambian_su_sql(perfiles_db):
    """Sin bloque de origen, la sentencia es la de siempre: el trigger registra `app` (la persona)."""
    import db_profiles
    assert db_profiles.update_water_tracker_enabled(UID, True) is True
    assert db_profiles.update_long_term_memory_enabled(UID, False) is True
    assert [q for q, _ in perfiles_db] == [
        "UPDATE user_profiles SET water_tracker_enabled = %s WHERE id = %s RETURNING id",
        "UPDATE user_profiles SET long_term_memory_enabled = %s WHERE id = %s RETURNING id"]


def test_dentro_del_bloque_del_coach_los_interruptores_llevan_su_origen(perfiles_db):
    import db_profiles
    with ac.origen_de_ajustes("coach"):
        db_profiles.update_water_tracker_enabled(UID, False)
        db_profiles.update_long_term_memory_enabled(UID, False)
    assert all(_SET_CONFIG.format("coach") in q for q, _ in perfiles_db) and len(perfiles_db) == 2
    assert [p for _, p in perfiles_db] == [(False, UID), (False, UID)], "los parámetros no se mueven"


class _Cursor:
    def __init__(self, destino, hp):
        self.destino, self.hp, self.fila = destino, hp, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, q, p=None):
        self.destino.append((_plano(q), p))
        if _plano(q).startswith("SELECT health_profile FROM user_profiles"):
            self.fila = {"health_profile": dict(self.hp)}

    def fetchone(self):
        return self.fila


class _Conexion:
    def __init__(self, destino, hp):
        self.destino, self.hp = destino, hp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def transaction(self):
        return self

    def cursor(self, row_factory=None):
        return _Cursor(self.destino, self.hp)


class _Pool:
    def __init__(self, destino, hp):
        self.destino, self.hp = destino, hp

    def connection(self):
        return _Conexion(self.destino, self.hp)


def test_la_fusion_atomica_del_perfil_lleva_el_origen_del_bloque(monkeypatch):
    import db_profiles
    sentencias = []
    monkeypatch.setattr(db_profiles, "connection_pool", _Pool(sentencias, {"avisos_comida": True}))

    def _apagar(hp):
        hp["avisos_comida"] = False
    db_profiles.update_user_health_profile_atomic(UID, _apagar)
    assert sentencias[-1][0] == "UPDATE user_profiles SET health_profile = %s::jsonb WHERE id = %s"
    with ac.origen_de_ajustes("coach"):
        db_profiles.update_user_health_profile_atomic(UID, _apagar)
    ultima = sentencias[-1][0]
    assert ultima.startswith("UPDATE user_profiles SET health_profile = %s::jsonb FROM (SELECT ")
    assert _SET_CONFIG.format("coach") in ultima and ultima.endswith("AS _origen WHERE id = %s")


def test_la_tool_del_coach_cambia_cada_ajuste_dentro_del_bloque_coach(monkeypatch):
    """`cambiar_ajuste_de_la_app` (§13.2): cada escritor que alcanza corre con el origen `coach`."""
    import ajustes_de_la_app as aj
    import db
    import db_profiles
    import hydration_reminders
    import nevera_opcional
    import plan_mode
    vistos = []

    def _anota(nombre, devuelve):
        return lambda *a, **k: vistos.append((nombre, ac.origen_actual())) or devuelve
    monkeypatch.setattr(db_profiles, "update_water_tracker_enabled", _anota("agua", True))
    monkeypatch.setattr(db_profiles, "update_long_term_memory_enabled", _anota("memoria", True))
    monkeypatch.setattr(hydration_reminders, "al_encender", _anota("al_encender", None))
    monkeypatch.setattr(nevera_opcional, "interruptor_disponible", lambda: True)
    monkeypatch.setattr(nevera_opcional, "fijar_nevera", _anota("nevera", True))
    monkeypatch.setattr(nevera_opcional, "estado_nevera", lambda uid: {"activa": True})
    monkeypatch.setattr(plan_mode, "pause_plan_generation", _anota("pausa", {"plan_mode": "tracking"}))
    monkeypatch.setattr(aj, "_tiene_plan", lambda uid: True)
    monkeypatch.setattr(db, "update_user_health_profile_atomic", _anota("perfil", {"avisos_agua": False}))
    for ajuste, valor in (("hidratacion", "encender"), ("memoria", "apagar"), ("nevera", "encender"),
                          ("generador_de_planes", "apagar"), ("recordatorios_de_agua", "apagar")):
        assert "ERROR" not in aj.cambiar_ajuste(UID, ajuste, valor), ajuste
    assert vistos == [("agua", "coach"), ("al_encender", "coach"), ("memoria", "coach"), ("nevera", "coach"),
                      ("pausa", "coach"), ("perfil", "coach")]
    assert ac.origen_actual() is None, "fuera de la tool, el origen vuelve a ser el de la persona"
    cuerpo = _cuerpo(_src("ajustes_de_la_app.py"), "def cambiar_ajuste(")
    assert 'with origen_de_ajustes("coach"):' in cuerpo


def test_la_nevera_la_fija_con_el_origen_del_bloque_y_se_apaga_sola_como_sistema(monkeypatch):
    import nevera_opcional as no
    escrito = []
    monkeypatch.setattr(no, "execute_sql_write", _captura(escrito))
    no.fijar_nevera(UID, True)
    assert escrito[-1][0] == "UPDATE user_profiles SET nevera_enabled = %s, nevera_auto_off_at = NULL WHERE id = %s RETURNING id"
    with ac.origen_de_ajustes("coach"):
        no.fijar_nevera(UID, False)
    assert _SET_CONFIG.format("coach") in escrito[-1][0] and escrito[-1][1] == (False, UID)
    # los dos apagados automáticos, y el encendido automático al guardar algo (la apagó el sistema)
    monkeypatch.setattr(no, "interruptor_disponible", lambda: True)
    monkeypatch.setattr(no, "_auto_apagado_encendido", lambda: True)
    monkeypatch.setattr(no, "apagable_en_modo_plan", lambda: True)
    assert no.apagar_por_plan_vacio(UID) is True
    assert _SET_CONFIG.format("sistema") in escrito[-1][0] and escrito[-1][1] == (UID,)
    assert no.apagar_neveras_sin_uso(limite=7) == [UID]
    sql, params = escrito[-1]
    assert _SET_CONFIG.format("sistema") in sql and params == (48, 48, 7)
    assert sql.index("FROM (SELECT set_config") < sql.index("WHERE p.id IN (") < sql.index("FROM user_profiles q")
    monkeypatch.setattr(no, "_perfil_nevera", lambda uid: {"plan_mode": "tracking", "nevera_enabled": False,
                                                           "nevera_auto_off_at": "2026-09-27T00:00:00Z"})
    assert no.encender_por_uso(UID) == "encendida"
    assert _SET_CONFIG.format("sistema") in escrito[-1][0] and "nevera_enabled = NULL" in escrito[-1][0]
    with ac.origen_de_ajustes("coach"):
        no.apagar_por_plan_vacio(UID)
    assert _SET_CONFIG.format("sistema") in escrito[-1][0], "el apagado automático es del sistema también bajo el coach"


def test_el_generador_lleva_el_origen_solo_en_la_bandera_del_perfil(monkeypatch):
    import plan_mode as pm
    escrito = []
    monkeypatch.setattr(pm, "PLAN_MODE_SWITCH_ENABLED", True)
    monkeypatch.setattr(pm, "execute_sql_write", _captura(escrito))
    monkeypatch.setattr(pm, "execute_sql_query", lambda *a, **k: {"plan_mode": "tracking", "paused_days": 2})
    monkeypatch.setattr(pm, "_revive_paused_chunks", lambda uid: {"revived": 0, "plans": 0})
    with ac.origen_de_ajustes("coach"):
        pm.pause_plan_generation(UID)
        pm.resume_plan_generation(UID)
    perfil = [q for q, _ in escrito if q.startswith("UPDATE user_profiles")]
    otras = [q for q, _ in escrito if not q.startswith("UPDATE user_profiles")]
    assert len(perfil) == 2 and all(_SET_CONFIG.format("coach") in q for q in perfil)
    assert otras and not any("set_config" in q for q in otras), "las colas y los planes no son ajustes"
    escrito.clear()
    pm.pause_plan_generation(UID)
    assert "set_config" not in escrito[0][0], "desde Configuración, sin marca: la persona"


def test_la_hidratacion_se_apaga_sola_con_origen_sistema(monkeypatch):
    import db
    import hydration_reminders as hr
    import utils_push
    from routers import plans as rp
    vistos = []
    monkeypatch.setattr(hr, "_leer_estado", lambda _u: {"ignored_since": (T0 - timedelta(hours=49)).isoformat(),
                                                        "nudges": 4})
    monkeypatch.setattr(hr, "_guardar_estado", lambda _u, e: None)
    monkeypatch.setattr(hr, "_hubo_agua_desde", lambda _u, _d: False)
    monkeypatch.setattr(db, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(db, "get_water_intake_glasses_today", lambda _u, _f: 0)
    monkeypatch.setattr(db, "update_water_tracker_enabled", lambda _u, v: vistos.append((v, ac.origen_actual())) or True)
    monkeypatch.setattr(rp, "_compute_water_goal", lambda _u: {"goal": 9})
    monkeypatch.setattr(utils_push, "send_push_notification", lambda *a, **k: True)
    assert hr.revisar_usuario(UID, "es-DO", True, ahora=T0)["accion"] == "apagado"
    assert vistos == [(False, "sistema")] and ac.origen_actual() is None


def test_los_escritores_pasan_por_sql_con_origen():
    """Test por fuente: si alguien reescribe una de estas sentencias sin el helper, el origen se pierde en silencio."""
    perfiles = _src("db_profiles.py")
    for firma in ("def update_water_tracker_enabled(", "def update_long_term_memory_enabled(",
                  "def update_user_health_profile_atomic(", "def update_user_health_profile("):
        assert "sql_con_origen(" in _cuerpo(perfiles, firma), firma
    nevera = _src("nevera_opcional.py")
    assert "sql_con_origen(" in _cuerpo(nevera, "def fijar_nevera(")
    assert 'sql_con_origen(_SQL_APAGAR, "sistema")' in _cuerpo(nevera, "def apagar_neveras_sin_uso(")
    assert '"sistema")' in _cuerpo(nevera, "def apagar_por_plan_vacio(")
    assert '"sistema")' in _cuerpo(nevera, "def encender_por_uso(")
    pm = _src("plan_mode.py")
    for firma in ("def pause_plan_generation(", "def resume_plan_generation("):
        assert "sql_con_origen(" in _cuerpo(pm, firma), firma
    assert 'with origen_de_ajustes("sistema"):' in _cuerpo(_src("hydration_reminders.py"), "def revisar_usuario(")


# ═════════════════════════════════════════════ 8. la migración
def _mig() -> str:
    return (_BACKEND / "migrations" / _MIG).read_text(encoding="utf-8")


def _codigo(sql: str) -> str:
    return "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))


def test_migracion_la_tabla_con_las_columnas_exactas():
    sql = _mig()
    cuerpo = sql.split("CREATE TABLE IF NOT EXISTS public.ajustes_cambios (", 1)[1].split(");", 1)[0]
    columnas = [" ".join(l.strip().rstrip(",").split()) for l in cuerpo.strip().splitlines() if l.strip()]
    assert columnas == [
        "id BIGSERIAL PRIMARY KEY",
        "user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE",
        "clave TEXT NOT NULL",
        "antes JSONB",
        "despues JSONB",
        "origen TEXT NOT NULL DEFAULT 'app'",
        "at TIMESTAMPTZ NOT NULL DEFAULT now()",
    ]
    plano = _plano(sql)
    assert "CREATE INDEX IF NOT EXISTS ajustes_cambios_user_at_idx ON public.ajustes_cambios (user_id, at DESC);" in plano
    assert "CREATE INDEX IF NOT EXISTS ajustes_cambios_at_idx ON public.ajustes_cambios (at DESC);" in plano
    assert "REVOKE ALL ON public.ajustes_cambios FROM PUBLIC;" in plano
    assert ("ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS ajustes_dispositivo JSONB NOT NULL "
            "DEFAULT '{}'::jsonb;") in plano
    assert "CHECK (jsonb_typeof(ajustes_dispositivo) = 'object')" in plano


def test_migracion_la_funcion_invoker_con_search_path_vacio_y_nombres_calificados():
    sql = _mig()
    funcion = sql.split("CREATE OR REPLACE FUNCTION public.registrar_cambio_ajustes()", 1)[1].split("$body$;", 1)[0]
    cabecera = _plano(funcion.split("AS $body$", 1)[0])
    assert cabecera == "RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER SET search_path = ''"
    assert funcion.count("INSERT INTO public.ajustes_cambios (user_id, clave, antes, despues, origen)") == 2
    assert "INSERT INTO ajustes_cambios" not in funcion, "con search_path vacío, todo nombre va calificado"
    assert "EXCEPTION WHEN OTHERS THEN" in funcion and "RAISE WARNING" in funcion, (
        "un fallo al registrar no puede tumbar el UPDATE de la persona")
    nevera = _plano(funcion)
    assert ("CASE WHEN v_clave = 'nevera_enabled' AND NEW.nevera_enabled IS FALSE AND OLD.nevera_auto_off_at IS NULL "
            "AND NEW.nevera_auto_off_at IS NOT NULL THEN 'sistema' ELSE v_origen END") in nevera
    assert "RETURN NULL;" in funcion


def _trigger(sql: str) -> str:
    return _plano(sql.split("CREATE TRIGGER trg_ajustes_cambios", 1)[1].split(";", 1)[0])


def test_migracion_el_trigger_mira_solo_lo_vigilado():
    """Review Focus 5: un UPDATE de `fact_locked_at` (el lock del extractor) ni entra; uno que cambia una clave vigilada
    sí, y la función escribe una fila por clave."""
    sql = _mig()
    trg = _trigger(sql)
    assert sql.index("DROP TRIGGER IF EXISTS trg_ajustes_cambios ON public.user_profiles;") < sql.index(
        "CREATE TRIGGER trg_ajustes_cambios")
    columnas = ", ".join(ac.COLUMNAS_VIGILADAS)
    assert trg.startswith(f"AFTER UPDATE OF {columnas}, health_profile ON public.user_profiles FOR EACH ROW WHEN (")
    assert trg.endswith("EXECUTE FUNCTION public.registrar_cambio_ajustes()")
    when = trg.split("WHEN (", 1)[1].rsplit(") EXECUTE FUNCTION", 1)[0]
    for vetada in ("fact_locked_at", "updated_at"):
        assert vetada not in when and vetada not in trg, vetada
    for c in ac.COLUMNAS_VIGILADAS:
        assert f"OLD.{c} IS DISTINCT FROM NEW.{c}" in when, c
    for k in ac.CLAVES_PERFIL_VIGILADAS:
        assert f"(OLD.health_profile -> '{k}') IS DISTINCT FROM (NEW.health_profile -> '{k}')" in when, k
    condiciones = re.split(r"\s+OR\s+", when.strip())
    assert len(condiciones) == len(ac.COLUMNAS_VIGILADAS) + len(ac.CLAVES_PERFIL_VIGILADAS), "ni una más"


def test_migracion_es_idempotente_se_autoverifica_y_no_nombra_auth():
    sql = _mig()
    codigo = _codigo(sql)
    assert not re.search(r"CREATE\s+(UNIQUE\s+)?(TABLE|INDEX)\s+(?!IF NOT EXISTS)", codigo)
    assert not re.search(r"DROP\s+(TABLE|INDEX|COLUMN)", codigo)
    assert "CREATE OR REPLACE FUNCTION public.registrar_cambio_ajustes()" in codigo
    assert codigo.index("DROP CONSTRAINT IF EXISTS user_profiles_ajustes_dispositivo_objeto_chk") < codigo.index(
        "ADD CONSTRAINT user_profiles_ajustes_dispositivo_objeto_chk")
    assert "auth." not in codigo and "TO authenticated" not in codigo
    assert codigo.count("RAISE EXCEPTION") >= 5
    assert codigo.index("faltan columnas en user_profiles") < codigo.index("CREATE TRIGGER trg_ajustes_cambios"), (
        "las columnas vigiladas se comprueban ANTES de nombrarlas en el trigger")
    assert "[P1-PLAN-LOTE-837 · 2026-09-29]" in sql


def test_migracion_copia_identica_en_la_raiz_del_workspace():
    backend = _BACKEND / "migrations" / _MIG
    raiz = _BACKEND.parent / "migrations" / _MIG
    assert raiz.exists(), "el SSOT de migraciones vive en DOS directorios (P3-MIGRATIONS-SSOT): falta la copia raíz"
    assert raiz.read_bytes() == backend.read_bytes(), "las dos copias difieren"
