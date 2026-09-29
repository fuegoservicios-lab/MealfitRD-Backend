"""[P1-PLAN-LOTE-830 · 2026-09-29] Cuentas de prueba: la tabla y las reglas (spec 2026-09-29-admin-cuentas-actividad-
pruebas-design §3 y §13.4).

Una marca la pone el admin con un motivo, la ve la propia persona y ella puede salir cuando quiera. El contenido de la
cuenta solo se abre mientras la marca está VIVA y la persona ya vio el aviso, y se comprueba en cada petición, sin
caché. Todo con la base falsa: nada de este fichero toca Neon.

Lo que cubre cada bloque (y por qué cada uno fallaría contra un módulo hecho a medias):
  1. los dos interruptores y `exigir_prueba` (knob apagado, sin marca, aviso pendiente, sin caché);
  2. `marcar`: rastro ANTES del INSERT, doble marca, vuelta tras salir ella, carreras y motivo;
  3. `quitar`, `salir` y `aviso_visto`;
  4. `marcar_varias`;
  5. la migración (forma, idempotencia, copias idénticas) y la exportación de la cuenta.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import cuentas_prueba as cp
import regalos_cuenta as rc

_BACKEND = Path(__file__).resolve().parents[1]
_MIG = "p1_plan_lote_830_cuentas_de_prueba_2026_09_29.sql"
ADMIN = "11111111-1111-1111-1111-111111111111"
ADMIN2 = "22222222-2222-2222-2222-222222222222"
UID = "33333333-3333-3333-3333-333333333333"
OTRO = "55555555-5555-5555-5555-555555555555"
TERCERA = "88888888-8888-8888-8888-888888888888"
NADIE = "77777777-7777-7777-7777-777777777777"      # sin perfil
MOTIVO = "prueba del flujo de pagos"


class UniqueViolation(Exception):
    """Lleva el nombre de `psycopg.errors.UniqueViolation`: el módulo distingue el fallo por el NOMBRE de la clase."""


class ForeignKeyViolation(Exception):
    """Ídem con `psycopg.errors.ForeignKeyViolation`."""


class _BD:
    """`cuentas_de_prueba` en memoria. Entiende SOLO las sentencias que el módulo emite: cualquier otra revienta, así
    que un SQL nuevo no pasa sin que alguien mire este fake."""

    def __init__(self):
        self.perfiles = {UID, OTRO}
        self.filas: list = []
        self.reloj = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        self.rastro_roto = False
        self.insert_roto = None
        self.update_roto = None
        self.update_sin_efecto = False
        self.vaciar()

    def vaciar(self):
        """Los contadores a cero, sin tocar las filas: para mirar solo lo que hace la llamada que viene."""
        self.lecturas, self.escrituras, self.rastro, self.orden = [], [], [], []

    def _ahora(self):
        self.reloj += timedelta(minutes=1)
        return self.reloj

    # ── lecturas
    def query(self, q, p=None, fetch_one=False, fetch_all=False):
        q = " ".join(q.split())
        self.lecturas.append(q)
        if "FROM public.user_profiles WHERE id = %s" in q:
            return {"id": p[0]} if p[0] in self.perfiles else None
        if "FROM public.cuentas_de_prueba WHERE user_id = %s AND quitada_at IS NULL" in q:
            vivas = [f for f in self.filas if f["user_id"] == p[0] and f["quitada_at"] is None]
            return dict(vivas[0]) if vivas else None
        if "FROM public.cuentas_de_prueba WHERE user_id = %s ORDER BY marcada_at DESC" in q:
            propias = sorted((f for f in self.filas if f["user_id"] == p[0]),
                             key=lambda f: f["marcada_at"], reverse=True)
            if fetch_one:
                return dict(propias[0]) if propias else None
            return [dict(f) for f in propias]
        raise AssertionError(f"consulta que el fake no conoce: {q[:140]}")

    # ── escrituras
    def write(self, q, p=None, returning=False, **k):
        q = " ".join(q.split())
        self.orden.append("escritura")
        self.escrituras.append((q, p))
        if q.startswith("INSERT INTO public.cuentas_de_prueba"):
            if self.insert_roto:
                raise self.insert_roto
            mid, uid, admin, motivo = p
            if uid not in self.perfiles:
                raise ForeignKeyViolation("user_id")
            if any(f["user_id"] == uid and f["quitada_at"] is None for f in self.filas):
                raise UniqueViolation("cuentas_de_prueba_una_viva_idx")       # el índice único parcial
            self.filas.append({
                "id": mid, "user_id": uid, "marcada_por": admin, "marcada_at": self._ahora(), "motivo": motivo,
                "aviso_visto_at": None, "quitada_at": None, "quitada_por": None, "quitada_por_la_persona": False,
                "motivo_quitar": None})
            return [{"id": mid}] if returning else True
        if q.startswith("UPDATE public.cuentas_de_prueba SET aviso_visto_at = now()"):
            for f in self.filas:
                if f["user_id"] == p[0] and f["aviso_visto_at"] is None and f["quitada_at"] is None:
                    f["aviso_visto_at"] = self._ahora()
            return True
        if q.startswith("UPDATE public.cuentas_de_prueba SET quitada_at = now()"):
            if self.update_roto:
                raise self.update_roto
            if self.update_sin_efecto:
                return []
            if "quitada_por_la_persona = true" in q:                              # salir: la propia persona
                assert "quitada_por = user_id" in q
                vivas = [f for f in self.filas if f["user_id"] == p[0] and f["quitada_at"] is None]
                for f in vivas:
                    f.update(quitada_at=self._ahora(), quitada_por=f["user_id"], quitada_por_la_persona=True)
            else:                                                                 # quitar: un admin
                admin, motivo, mid = p
                vivas = [f for f in self.filas if f["id"] == mid and f["quitada_at"] is None]
                for f in vivas:
                    f.update(quitada_at=self._ahora(), quitada_por=admin, quitada_por_la_persona=False,
                             motivo_quitar=motivo)
            return [{"id": f["id"]} for f in vivas]
        raise AssertionError(f"escritura que el fake no conoce: {q[:140]}")

    def anotar(self, admin, accion, objetivo=None, detalle=None):
        self.orden.append("rastro")
        if self.rastro_roto:
            raise RuntimeError("admin_access_log no escribe")
        self.rastro.append((accion, objetivo, detalle))


@pytest.fixture
def bd(monkeypatch):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    b = _BD()
    monkeypatch.setattr(cp, "execute_sql_query", b.query)
    monkeypatch.setattr(cp, "execute_sql_write", b.write)
    monkeypatch.setattr(cp, "registrar_acceso", b.anotar)
    return b


def _encender(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")


def _viva(bd, uid=UID, visto=False):
    """Una marca viva (y, con `visto`, con el aviso ya enseñado) y los contadores del fake a cero."""
    cp.marcar(ADMIN, uid, MOTIVO)
    if visto:
        cp.aviso_visto(uid)
    bd.vaciar()


def _rota(*a, **k):
    raise RuntimeError("sin DB")


# ═════════════════════════════════════════════ 1. interruptores y exigir_prueba
def test_los_dos_interruptores_y_sus_defaults(monkeypatch):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    assert cp.activo() is False, "el maestro nace apagado: se enciende con el texto legal ya publicado"
    assert cp.requiere_aviso() is True, "sin correo, el contenido espera a que la persona vea el aviso en la app"
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", "false")
    assert cp.activo() is True and cp.requiere_aviso() is False


def test_los_dos_interruptores_quedan_en_el_inventario_de_knobs(monkeypatch):
    from knobs import get_knobs_registry_snapshot
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    cp.activo()
    cp.requiere_aviso()
    snap = get_knobs_registry_snapshot()
    assert snap["MEALFIT_ADMIN_TEST_ACCOUNTS"]["default"] is False
    assert snap["MEALFIT_ADMIN_TEST_REQUIRE_NOTICE"]["default"] is True


def test_error_prueba_lleva_estado_y_detalle():
    e = cp.ErrorPrueba(409, "ya_marcada")
    assert (e.status, e.detalle, str(e)) == (409, "ya_marcada", "ya_marcada")


def test_exigir_prueba_con_el_knob_apagado_es_403_y_ni_mira_la_base(bd):
    _viva(bd, visto=True)          # la marca existe y la persona vio el aviso…
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.exigir_prueba(UID)      # …pero el interruptor maestro está apagado
    assert (ei.value.status, ei.value.detalle) == (403, "no_es_prueba")
    assert bd.lecturas == []


def test_exigir_prueba_sin_marca_es_403(bd, monkeypatch):
    _encender(monkeypatch)
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.exigir_prueba(UID)
    assert (ei.value.status, ei.value.detalle) == (403, "no_es_prueba")


def test_exigir_prueba_con_marca_viva_sin_aviso_visto_es_409(bd, monkeypatch):
    _encender(monkeypatch)
    _viva(bd)
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.exigir_prueba(UID)
    assert (ei.value.status, ei.value.detalle) == (409, "aviso_pendiente")


def test_exigir_prueba_con_el_aviso_visto_devuelve_la_marca_viva(bd, monkeypatch):
    _encender(monkeypatch)
    _viva(bd, visto=True)
    m = cp.exigir_prueba(UID)
    assert m["motivo"] == MOTIVO and m["marcada_por"] == ADMIN and m["aviso_visto_at"] is not None


def test_sin_exigir_el_aviso_pasa_aunque_la_persona_no_lo_haya_visto(bd, monkeypatch):
    _encender(monkeypatch)
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", "false")
    _viva(bd)
    assert cp.exigir_prueba(UID)["aviso_visto_at"] is None


@pytest.mark.parametrize("quien", ["persona", "admin"])
def test_quitar_la_marca_corta_la_siguiente_vista_sin_cache(bd, monkeypatch, quien):
    """Review Focus 1: entre dos peticiones la marca se quita (ella o el admin) y la SEGUNDA vista falla."""
    _encender(monkeypatch)
    _viva(bd, visto=True)
    assert cp.exigir_prueba(UID)["motivo"] == MOTIVO
    if quien == "persona":
        assert cp.salir(UID) is True
    else:
        cp.quitar(ADMIN, UID, "ya no hace falta")
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.exigir_prueba(UID)
    assert (ei.value.status, ei.value.detalle) == (403, "no_es_prueba")


def test_si_la_marca_no_se_puede_leer_no_hay_vista(bd, monkeypatch):
    _encender(monkeypatch)
    monkeypatch.setattr(cp, "execute_sql_query", _rota)
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.exigir_prueba(UID)
    assert ei.value.status == 503


def test_estado_de_refleja_lo_que_exigir_prueba_haria(monkeypatch):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    con = {"aviso_visto_at": datetime.now(timezone.utc)}
    sin = {"aviso_visto_at": None}
    assert cp.estado_de(None) is None
    assert cp.estado_de(con) == "activa" and cp.estado_de(sin) == "aviso_pendiente"
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", "false")
    assert cp.estado_de(sin) == "activa", "sin exigir el aviso, el detalle está abierto: no puede decir «esperando»"


def test_para_la_persona(bd, monkeypatch):
    _viva(bd)
    assert cp.para_la_persona(UID) is None, "con el knob apagado la app no le enseña nada"
    _encender(monkeypatch)
    assert cp.para_la_persona(OTRO) is None, "sin marca"
    assert cp.para_la_persona(UID) == {"desde": rc.iso(bd.filas[0]["marcada_at"]), "aviso_visto": False}
    cp.aviso_visto(UID)
    assert cp.para_la_persona(UID)["aviso_visto"] is True
    cp.salir(UID)
    assert cp.para_la_persona(UID) is None, "quitada la marca ya no hay aviso"


def test_para_la_persona_no_rompe_el_perfil(bd, monkeypatch):
    _encender(monkeypatch)
    assert cp.para_la_persona("guest") is None and bd.lecturas == [], "un id que no es uuid no consulta"
    monkeypatch.setattr(cp, "execute_sql_query", _rota)
    assert cp.para_la_persona(UID) is None, "una lectura fallida deja el perfil sin el bloque, no lo revienta"


# ═════════════════════════════════════════════ 2. marcar
def test_marcar_anota_el_rastro_ANTES_de_escribir(bd):
    r = cp.marcar(ADMIN, UID, "  prueba   del flujo  de pagos ")
    assert bd.orden == ["rastro", "escritura"]
    accion, objetivo, detalle = bd.rastro[0]
    assert (accion, objetivo) == ("marcar_prueba", UID)
    assert detalle["motivo"] == "prueba del flujo de pagos" and detalle["marca_id"] == r["id"]
    sql, p = bd.escrituras[0]
    assert sql.startswith("INSERT INTO public.cuentas_de_prueba")
    assert p == (r["id"], UID, ADMIN, "prueba del flujo de pagos")
    fila = bd.filas[0]
    assert fila["marcada_por"] == ADMIN and fila["quitada_at"] is None and fila["aviso_visto_at"] is None


def test_marcar_dos_veces_es_409_ya_marcada_sin_rastro_ni_fila_nueva(bd):
    cp.marcar(ADMIN, UID, MOTIVO)
    bd.vaciar()
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar(ADMIN2, UID, MOTIVO)
    assert (ei.value.status, ei.value.detalle) == (409, "ya_marcada")
    assert bd.orden == [] and len(bd.filas) == 1


def test_si_el_rastro_falla_no_hay_marca_ni_insert(bd):
    bd.rastro_roto = True
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar(ADMIN, UID, MOTIVO)
    assert ei.value.status == 503
    assert bd.escrituras == [] and bd.filas == []


@pytest.mark.parametrize("error,status,detalle", [
    (UniqueViolation("idx"), 409, "ya_marcada"),       # dos admins a la vez: el índice único parcial corta la carrera
    (ForeignKeyViolation("fk"), 404, "no_existe"),     # la cuenta se borró entre la lectura y la escritura
    (RuntimeError("sin DB"), 500, None),
])
def test_si_el_insert_falla_queda_marcar_prueba_fallo(bd, error, status, detalle):
    bd.insert_roto = error
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar(ADMIN, UID, MOTIVO)
    assert ei.value.status == status
    if detalle:
        assert ei.value.detalle == detalle
    assert [a for a, _, _ in bd.rastro] == ["marcar_prueba", "marcar_prueba_fallo"]
    assert bd.orden == ["rastro", "escritura", "rastro"]
    assert bd.rastro[1][2]["error"] == type(error).__name__ and bd.filas == []


def test_el_texto_libre_del_rastro_va_solo_bajo_las_claves_que_borra_el_cierre_de_cuenta(bd):
    """El rastro (`admin_access_log`) sobrevive a la cuenta. Al cerrarla, `delete_account_data` (lote 841) le quita las
    claves `motivo` y `error`: si el motivo de una marca viajara bajo otra clave, quedaría en un log sin dueño."""
    cp.marcar(ADMIN, UID, MOTIVO)                                    # marcar_prueba
    bd.update_roto = RuntimeError("sin DB")
    with pytest.raises(cp.ErrorPrueba):
        cp.quitar(ADMIN, UID, "fin de la ronda")                     # quitar_prueba + quitar_prueba_fallo
    bd.update_roto = None
    cp.quitar(ADMIN, UID, "fin de la ronda")                         # quitar_prueba
    bd.insert_roto = RuntimeError("sin DB")
    with pytest.raises(cp.ErrorPrueba):
        cp.marcar(ADMIN, UID, "otra ronda")                          # marcar_prueba + marcar_prueba_fallo
    assert {a for a, _, _ in bd.rastro} == {"marcar_prueba", "marcar_prueba_fallo", "quitar_prueba",
                                            "quitar_prueba_fallo"}
    for accion, _, d in bd.rastro:
        texto = {k for k, v in d.items() if isinstance(v, str) and k != "marca_id"}
        assert texto <= {"motivo", "error"}, f"{accion}: texto libre bajo {texto - {'motivo', 'error'}}"
    assert "- 'motivo' - 'error'" in (_BACKEND / "db_profiles.py").read_text(encoding="utf-8"), (
        "el borrado de cuenta ya no quita `motivo` y `error` del rastro: el motivo de la marca sobreviviría a la cuenta")


@pytest.mark.parametrize("motivo", [None, "", "   ", "ab", "x" * 301, 12])
def test_el_motivo_fuera_de_3_a_300_es_422_y_no_se_escribe_nada(bd, motivo):
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar(ADMIN, UID, motivo)
    assert (ei.value.status, ei.value.detalle) == (422, "motivo")
    assert bd.orden == [] and bd.filas == []


@pytest.mark.parametrize("motivo", ["abc", "x" * 300])
def test_los_extremos_del_motivo_se_aceptan(bd, motivo):
    assert cp.marcar(ADMIN, UID, motivo)["motivo"] == motivo


@pytest.mark.parametrize("uid", [NADIE, "guest", "", None])
def test_marcar_una_cuenta_que_no_existe_es_404_sin_rastro(bd, uid):
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar(ADMIN, uid, MOTIVO)
    assert (ei.value.status, ei.value.detalle) == (404, "no_existe")
    assert bd.orden == []


def test_tras_salir_ella_volver_a_marcar_exige_confirmar_la_vuelta(bd):
    cp.marcar(ADMIN, UID, MOTIVO)
    assert cp.salir(UID) is True
    bd.vaciar()
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar(ADMIN, UID, MOTIVO)
    assert (ei.value.status, ei.value.detalle) == (409, "salio_ella")
    assert bd.orden == [] and len(bd.filas) == 1, "ni rastro ni fila nueva"
    r = cp.marcar(ADMIN, UID, "la persona me pidió volver", confirmar_vuelta=True)
    assert len(bd.filas) == 2 and cp.marca_viva(UID)["id"] == r["id"]
    assert bd.rastro[0][2]["vuelta_tras_salir"] is True, "el rastro dice que fue una vuelta confirmada"


def test_si_la_ultima_marca_la_quito_un_admin_se_vuelve_a_marcar_sin_confirmar(bd):
    cp.marcar(ADMIN, UID, MOTIVO)
    cp.quitar(ADMIN2, UID, "ya no hace falta")
    r = cp.marcar(ADMIN, UID, "otra ronda de pruebas")
    assert cp.marca_viva(UID)["id"] == r["id"] and bd.rastro[-1][2]["vuelta_tras_salir"] is False


def test_solo_cuenta_la_ultima_marca_para_saber_si_salio_ella(bd):
    cp.marcar(ADMIN, UID, MOTIVO)
    cp.salir(UID)                                                        # 1.ª: salió ella
    cp.marcar(ADMIN, UID, "vuelta pedida", confirmar_vuelta=True)
    cp.quitar(ADMIN, UID, "terminado")                                   # 2.ª: la quitó un admin
    cp.marcar(ADMIN, UID, "tercera ronda")                               # ya no hay nada que confirmar
    assert cp.marca_viva(UID) is not None and len(bd.filas) == 3


# ═════════════════════════════════════════════ 3. marca_viva, historial, quitar, salir y aviso_visto
def test_marca_viva_lee_las_cinco_columnas_del_contrato(bd):
    assert cp.marca_viva(UID) is None
    q = bd.lecturas[-1]
    for c in ("id::text AS id", "marcada_por::text AS marcada_por", "marcada_at", "motivo", "aviso_visto_at"):
        assert c in q, f"marca_viva debe traer {c}"
    assert "quitada_at IS NULL" in q
    assert cp.marca_viva("guest") is None and len(bd.lecturas) == 1, "un id que no es uuid no consulta"


def test_historial_devuelve_todas_las_filas_la_mas_nueva_primero(bd):
    assert cp.historial(UID) == []
    a = cp.marcar(ADMIN, UID, "primera ronda")
    cp.salir(UID)
    b = cp.marcar(ADMIN, UID, "segunda ronda", confirmar_vuelta=True)
    h = cp.historial(UID)
    assert [f["id"] for f in h] == [b["id"], a["id"]]
    assert h[1]["quitada_por_la_persona"] is True and h[1]["quitada_at"] is not None and h[0]["quitada_at"] is None
    q = next(x for x in reversed(bd.lecturas) if "ORDER BY marcada_at DESC" in x)
    for c in ("quitada_at", "quitada_por::text AS quitada_por", "quitada_por_la_persona", "motivo_quitar"):
        assert c in q, f"historial debe traer {c}"
    assert cp.historial("guest") == []


def test_quitar_anota_antes_de_escribir_y_no_borra_la_fila(bd):
    m = cp.marcar(ADMIN, UID, MOTIVO)
    bd.vaciar()
    r = cp.quitar(ADMIN2, UID, "  fin de   la ronda ")
    assert bd.orden == ["rastro", "escritura"]
    accion, objetivo, detalle = bd.rastro[0]
    assert (accion, objetivo) == ("quitar_prueba", UID)
    assert detalle["marca_id"] == m["id"] and detalle["motivo"] == "fin de la ronda"
    fila = bd.filas[0]
    assert fila["quitada_at"] is not None and fila["quitada_por"] == ADMIN2
    assert fila["quitada_por_la_persona"] is False and fila["motivo_quitar"] == "fin de la ronda"
    assert r["id"] == m["id"] and r["user_id"] == UID
    assert cp.marca_viva(UID) is None and len(cp.historial(UID)) == 1, "quitar no borra: el historial es la tabla"
    assert not any(q.startswith("DELETE") for q, _ in bd.escrituras)


def test_quitar_sin_marca_viva_es_409_sin_marca(bd):
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.quitar(ADMIN, UID, "no hay nada que quitar")
    assert (ei.value.status, ei.value.detalle) == (409, "sin_marca") and bd.orden == []


def test_quitar_con_el_rastro_roto_no_quita_la_marca(bd):
    _viva(bd)
    bd.rastro_roto = True
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.quitar(ADMIN, UID, "fin de la ronda")
    assert ei.value.status == 503 and bd.escrituras == []
    assert cp.marca_viva(UID) is not None


def test_quitar_valida_el_motivo(bd):
    _viva(bd)
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.quitar(ADMIN, UID, "no")
    assert (ei.value.status, ei.value.detalle) == (422, "motivo") and bd.orden == []


def test_quitar_una_carrera_perdida_es_409_y_deja_el_fallo_en_el_rastro(bd):
    """Entre la lectura y el UPDATE la persona salió (o lo quitó otro admin): este admin no quitó nada."""
    _viva(bd)
    bd.update_sin_efecto = True
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.quitar(ADMIN, UID, "fin de la ronda")
    assert (ei.value.status, ei.value.detalle) == (409, "sin_marca")
    assert [a for a, _, _ in bd.rastro] == ["quitar_prueba", "quitar_prueba_fallo"]


def test_quitar_con_el_update_roto_deja_quitar_prueba_fallo(bd):
    _viva(bd)
    bd.update_roto = RuntimeError("sin DB")
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.quitar(ADMIN, UID, "fin de la ronda")
    assert ei.value.status == 500
    assert [a for a, _, _ in bd.rastro] == ["quitar_prueba", "quitar_prueba_fallo"]


def test_salir_funciona_con_el_knob_apagado_y_deja_a_la_persona_como_quien_quito(bd):
    assert cp.activo() is False
    cp.marcar(ADMIN, UID, MOTIVO)
    bd.vaciar()
    assert cp.salir(UID) is True
    fila = bd.filas[0]
    assert fila["quitada_por"] == UID and fila["quitada_por_la_persona"] is True and fila["quitada_at"] is not None
    assert bd.rastro == [], "no es una acción del personal: no va a admin_access_log"
    assert cp.salir(UID) is False, "idempotente: sin marca viva no hay nada que quitar"
    assert cp.salir(OTRO) is False, "una cuenta que nunca fue de prueba"
    n = len(bd.escrituras)
    assert cp.salir("guest") is False and len(bd.escrituras) == n, "un id que no es uuid no escribe"


def test_el_sql_de_salir_y_del_aviso_solo_toca_marcas_vivas(bd):
    cp.marcar(ADMIN, UID, MOTIVO)
    cp.aviso_visto(UID)
    cp.salir(UID)
    ups = [q for q, _ in bd.escrituras if q.startswith("UPDATE")]
    aviso = next(q for q in ups if "SET aviso_visto_at = now()" in q)
    salir = next(q for q in ups if "quitada_por_la_persona = true" in q)
    assert "AND aviso_visto_at IS NULL AND quitada_at IS NULL" in aviso, "el aviso solo se anota la primera vez"
    assert "quitada_por = user_id" in salir and "WHERE user_id = %s AND quitada_at IS NULL" in salir


def test_aviso_visto_solo_anota_la_primera_vez(bd):
    cp.marcar(ADMIN, UID, MOTIVO)
    assert cp.marca_viva(UID)["aviso_visto_at"] is None
    cp.aviso_visto(UID)
    primera = cp.marca_viva(UID)["aviso_visto_at"]
    assert primera is not None
    cp.aviso_visto(UID)
    assert cp.marca_viva(UID)["aviso_visto_at"] == primera


def test_aviso_visto_no_toca_una_marca_quitada_ni_falla_sin_marca(bd):
    cp.marcar(ADMIN, UID, MOTIVO)
    cp.salir(UID)
    assert cp.aviso_visto(UID) is None            # la marca ya no está viva
    cp.aviso_visto(OTRO)                          # nunca tuvo
    n = len(bd.escrituras)
    cp.aviso_visto("guest")                       # ni es un uuid
    assert bd.filas[0]["aviso_visto_at"] is None and len(bd.escrituras) == n


# ═════════════════════════════════════════════ 4. marcar_varias
def test_marcar_varias_mas_de_100_es_422_y_no_toca_nada(bd):
    ids = [str(uuid.uuid4()) for _ in range(101)]
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar_varias(ADMIN, ids, MOTIVO)
    assert (ei.value.status, ei.value.detalle) == (422, "demasiadas")
    assert bd.lecturas == [] and bd.orden == []


def test_marcar_varias_acepta_exactamente_100(bd):
    ids = [str(uuid.uuid4()) for _ in range(100)]
    r = cp.marcar_varias(ADMIN, ids, MOTIVO)
    assert len(r) == cp.MAX_LOTE == 100 and {x["resultado"] for x in r} == {"no_existe"}


def test_marcar_varias_devuelve_el_resultado_de_cada_id_en_su_orden(bd):
    bd.perfiles.add(TERCERA)
    cp.marcar(ADMIN, OTRO, MOTIVO)
    cp.salir(OTRO)                                # OTRO salió ella
    cp.marcar(ADMIN, TERCERA, MOTIVO)             # TERCERA ya tiene marca viva
    bd.vaciar()
    r = cp.marcar_varias(ADMIN, [UID, TERCERA, OTRO, NADIE, "guest"], "tanda de pruebas")
    assert r == [
        {"user_id": UID, "resultado": "marcada"},
        {"user_id": TERCERA, "resultado": "ya_marcada"},
        {"user_id": OTRO, "resultado": "salio_ella"},
        {"user_id": NADIE, "resultado": "no_existe"},
        {"user_id": "guest", "resultado": "no_existe"},
    ]
    assert [(a, o) for a, o, _ in bd.rastro] == [("marcar_prueba", UID)], "rastro solo de la que se marcó de verdad"
    assert cp.marca_viva(UID) is not None and cp.marca_viva(OTRO) is None


def test_marcar_varias_deja_una_fila_de_rastro_por_cuenta_y_antes_de_cada_insert(bd):
    bd.perfiles.add(TERCERA)
    r = cp.marcar_varias(ADMIN, [UID, TERCERA], MOTIVO)
    assert [x["resultado"] for x in r] == ["marcada", "marcada"]
    assert bd.orden == ["rastro", "escritura", "rastro", "escritura"]
    assert [(a, o) for a, o, _ in bd.rastro] == [("marcar_prueba", UID), ("marcar_prueba", TERCERA)]


def test_marcar_varias_con_un_motivo_invalido_es_422_antes_de_tocar_nada(bd):
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar_varias(ADMIN, [UID], "no")
    assert (ei.value.status, ei.value.detalle) == (422, "motivo") and bd.lecturas == []


def test_marcar_varias_si_el_rastro_falla_corta_el_lote(bd):
    bd.rastro_roto = True
    with pytest.raises(cp.ErrorPrueba) as ei:
        cp.marcar_varias(ADMIN, [UID, OTRO], MOTIVO)
    assert ei.value.status == 503 and bd.escrituras == [] and bd.filas == []


def test_marcar_varias_sin_ids_no_hace_nada(bd):
    assert cp.marcar_varias(ADMIN, [], MOTIVO) == [] and bd.lecturas == []


# ═════════════════════════════════════════════ 5. la migración
def _sql() -> str:
    return (_BACKEND / "migrations" / _MIG).read_text(encoding="utf-8")


def test_migracion_la_tabla_con_las_columnas_exactas_del_spec():
    sql = _sql()
    cuerpo = sql.split("CREATE TABLE IF NOT EXISTS public.cuentas_de_prueba (", 1)[1].split(");", 1)[0]
    columnas = [" ".join(l.strip().rstrip(",").split()) for l in cuerpo.strip().splitlines() if l.strip()]
    assert columnas == [
        "id UUID PRIMARY KEY DEFAULT gen_random_uuid()",
        "user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE",
        "marcada_por UUID NOT NULL",
        "marcada_at TIMESTAMPTZ NOT NULL DEFAULT now()",
        "motivo TEXT NOT NULL",
        "aviso_visto_at TIMESTAMPTZ",
        "quitada_at TIMESTAMPTZ",
        "quitada_por UUID",
        "quitada_por_la_persona BOOLEAN NOT NULL DEFAULT false",
        "motivo_quitar TEXT",
    ]


def test_migracion_los_check_se_reemiten_y_el_indice_unico_es_parcial():
    sql = _sql()
    for c in ("cuentas_de_prueba_motivo_chk", "cuentas_de_prueba_motivo_quitar_chk", "cuentas_de_prueba_quitada_chk"):
        assert f"DROP CONSTRAINT IF EXISTS {c}" in sql and f"ADD CONSTRAINT {c}" in sql, c
        assert sql.index(f"DROP CONSTRAINT IF EXISTS {c}") < sql.index(f"ADD CONSTRAINT {c}"), (
            f"{c}: el DROP va ANTES del ADD (idempotencia)")
    plano = " ".join(sql.split())
    assert "CHECK (char_length(motivo) BETWEEN 3 AND 300)" in plano
    assert "CHECK (motivo_quitar IS NULL OR char_length(motivo_quitar) BETWEEN 3 AND 300)" in plano
    assert "CHECK (quitada_at IS NULL OR quitada_por IS NOT NULL)" in plano
    assert ("CREATE UNIQUE INDEX IF NOT EXISTS cuentas_de_prueba_una_viva_idx "
            "ON public.cuentas_de_prueba (user_id) WHERE quitada_at IS NULL;") in plano, "una sola marca viva por cuenta"
    assert "CREATE INDEX IF NOT EXISTS cuentas_de_prueba_user_idx" in plano


def test_migracion_es_idempotente_cierra_la_tabla_y_se_autoverifica():
    sql = _sql()
    codigo = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    assert "REVOKE ALL ON public.cuentas_de_prueba FROM PUBLIC;" in sql
    assert "RAISE EXCEPTION" in sql
    # cada CREATE lleva IF NOT EXISTS y ningún DROP toca tablas o índices: solo constraints
    assert not re.search(r"CREATE\s+(UNIQUE\s+)?(TABLE|INDEX)\s+(?!IF NOT EXISTS)", codigo)
    assert not re.search(r"DROP\s+(TABLE|INDEX)", codigo)
    # `auth.` no existe en Neon: una migración que lo nombre revienta al aplicarla (test_p1_consumption_ledger)
    assert "auth." not in codigo and "TO authenticated" not in codigo
    assert "[P1-PLAN-LOTE-830 · 2026-09-29]" in sql


def test_migracion_copia_identica_en_la_raiz_del_workspace():
    backend = _BACKEND / "migrations" / _MIG
    raiz = _BACKEND.parent / "migrations" / _MIG
    assert raiz.exists(), "el SSOT de migraciones vive en DOS directorios (P3-MIGRATIONS-SSOT): falta la copia raíz"
    assert raiz.read_bytes() == backend.read_bytes(), "las dos copias difieren"


# ═════════════════════════════════════════════ 6. la exportación de la cuenta
def _app_src() -> str:
    return (_BACKEND / "app.py").read_text(encoding="utf-8")


def test_la_exportacion_lleva_la_tabla_y_quita_los_ids_del_personal():
    src = _app_src()
    tablas = re.search(r"_ACCOUNT_EXPORT_TABLES\s*=\s*\((.*?)\n\)", src, re.DOTALL).group(1)
    assert re.search(r'\("cuentas_de_prueba",\s*"user_id",\s*100\)', tablas), (
        "la persona tiene derecho a ver que su cuenta fue marcada como de prueba (Privacidad §5)")
    quitadas = re.search(r"_ACCOUNT_EXPORT_STRIPPED_KEYS\s*=\s*\((.*?)\)", src).group(1)
    assert quitadas.lstrip().startswith('"embedding"'), "`embedding` sigue primero (test_p2_privacy_settings)"
    for campo in ("marcada_por", "quitada_por"):
        assert f'"{campo}"' in quitadas, f"{campo} es el id del PERSONAL que marcó o quitó: no sale en la exportación"
    for campo in ("motivo", "motivo_quitar", "aviso_visto_at", "quitada_por_la_persona"):
        assert f'"{campo}"' not in quitadas, f"{campo} es información sobre la persona: debe seguir saliendo"


def test_la_consulta_de_la_exportacion_ordena_por_marcada_at():
    """La tabla NO tiene `created_at` (el orden por defecto del export): sin su propio ORDER BY, la consulta reventaría
    en producción y CADA exportación saldría con `complete: false` y la tabla en `omitted`."""
    import app as app_module
    sql, params = app_module._account_export_query("cuentas_de_prueba", "user_id", 100, UID, "SELECT 1")
    assert "FROM (SELECT * FROM public.cuentas_de_prueba) AS t WHERE user_id = %s" in sql
    assert "ORDER BY marcada_at DESC LIMIT 100" in sql and "created_at" not in sql
    assert {"embedding", "marcada_por", "quitada_por"} <= set(params[0]) and params[1:] == (UID,)
