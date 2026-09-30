"""[P1-PLAN-LOTE-838 · 2026-09-29] Marcar las cuentas existentes desde la consola, y la doc canónica del panel de cuentas.

Spec docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md §10 y §13.5 (raíz del workspace).
Todo con bases falsas: nada de este fichero toca Neon ni abre un pool.

Lo que cubre cada bloque (y por qué fallaría contra una versión a medias):
  1. el script `scripts/marcar_cuentas_prueba.py`: sin `--aplicar` NO escribe (solo lista); con el interruptor apagado se
     niega ANTES de abrir el pool o leer nada; `--admin` tiene que ser un admin de la lista; el motivo, de 3 a 300; un
     correo pedido o excluido que no existe se rechaza (una errata en una exclusión marcaría a quien no debía); marca en
     tandas de 100 con la función de verdad (`cuentas_prueba.marcar_varias`: mismo rastro que el panel); un corte a
     mitad sale con 1 y dice qué quedó; y solo imprime ASCII;
  2. la doc `docs/admin_cuentas_actividad_pruebas.md`: nombra CADA ruta del router (una ruta nueva sin documentar cae
     aquí) con su limitador y sus números reales, cada acción del rastro, los knobs, las tablas y el orden del despliegue;
     y `docs/panel_admin.md` la enlaza.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

import admin_cuentas_lista as acl
import admin_prueba_detalle as apd
import ajustes_cuenta
import cuentas_prueba as cp
import routers.admin as ra
import routers.user_data as ud
from rate_limiter import RateLimiter
from tests.test_p1_plan_lote_830 import _BD as _BDPrueba

_BACKEND = Path(__file__).resolve().parents[1]
_SCRIPT = _BACKEND / "scripts" / "marcar_cuentas_prueba.py"
_DOC = _BACKEND / "docs" / "admin_cuentas_actividad_pruebas.md"
_PANEL_DOC = _BACKEND / "docs" / "panel_admin.md"
ADMIN = "11111111-1111-1111-1111-111111111111"
ADMIN2 = "22222222-2222-2222-2222-222222222222"
FUERA = "99999999-9999-9999-9999-999999999999"        # un uuid válido que NO está en MEALFIT_ADMIN_USER_IDS
UID = "33333333-3333-3333-3333-333333333333"
OTRO = "55555555-5555-5555-5555-555555555555"
MOTIVO = "beta cerrada"


def _cargar():
    spec = importlib.util.spec_from_file_location("marcar_cuentas_prueba", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cuerpo(src: str, firma: str) -> str:
    """El cuerpo de una función de nivel de módulo: desde su `def` hasta el siguiente `def`/`@`/`class` en columna 0."""
    i = src.index(firma)
    m = re.search(r"\n(?:def |async def |@|class |if __name__)", src[i + len(firma):])
    return src[i:i + len(firma) + (m.start() if m else len(src))]


class _Consola:
    """Lo que el script toca fuera de sí mismo, en memoria: el pool, la lectura de `user_profiles` y `marcar_varias`.
    `orden` deja ver qué se tocó y en qué orden."""

    def __init__(self, capsys):
        self.capsys = capsys
        self.cuentas: list = []
        self.orden: list = []
        self.lecturas: list = []          # el filtro de cada lectura: None = todas, o la lista de correos
        self.marcadas: list = []          # (admin, ids, motivo) de cada llamada a `marcar_varias`
        self.resultado = lambda uid: "marcada"
        self.falla_en = None              # nº de llamada a `marcar_varias` que revienta…
        self.falla_con = cp.ErrorPrueba(503, "No se pudo registrar la accion; no se hizo ningun cambio.")
        self.falla_la_lectura = None
        self.falla_el_pool = None
        self.msc = None

    def abrir_pool(self):
        self.orden.append("pool")
        if self.falla_el_pool:
            raise self.falla_el_pool

    def leer(self, correos=None):
        self.orden.append("leer")
        self.lecturas.append(None if correos is None else list(correos))
        if self.falla_la_lectura:
            raise self.falla_la_lectura
        if correos is None:
            return [dict(c) for c in self.cuentas]
        return [dict(c) for c in self.cuentas if str(c.get("email") or "").lower() in correos]   # `lower(email) = ANY(…)`

    def marcar_varias(self, admin, ids, motivo):
        self.orden.append("marcar")
        self.marcadas.append((admin, list(ids), motivo))
        if self.falla_en == len(self.marcadas):
            raise self.falla_con
        return [{"user_id": u, "resultado": self.resultado(u)} for u in ids]

    def correr(self, *args):
        """`(código de salida, lo impreso)`. `SystemExit` (argparse) sale como su código."""
        try:
            codigo = self.msc.main(list(args))
        except SystemExit as e:
            codigo = e.code
        return codigo, self.capsys.readouterr().out


@pytest.fixture
def consola(monkeypatch, capsys):
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", f"{ADMIN}, {ADMIN2.upper()}")
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    c = _Consola(capsys)
    c.msc = _cargar()
    monkeypatch.setattr(c.msc, "_cargar_entorno", lambda: None)       # el .env de esta máquina no puede cambiar el resultado
    monkeypatch.setattr(c.msc, "_abrir_pool", c.abrir_pool)
    monkeypatch.setattr(c.msc, "_leer_cuentas", c.leer)
    monkeypatch.setattr(cp, "marcar_varias", c.marcar_varias)
    return c


def _cuentas(n: int) -> list:
    return [{"id": f"{i:08d}-0000-4000-8000-000000000000", "email": f"cuenta{i}@correo.com"} for i in range(n)]


# ═════════════════════════════════════════════ 1. el script
def test_sin_aplicar_lista_lo_que_haria_y_no_escribe_nada(consola, monkeypatch):
    consola.cuentas = _cuentas(3)
    monkeypatch.setattr(cp, "execute_sql_write", lambda *a, **k: pytest.fail("escribió sin --aplicar"))
    monkeypatch.setattr(cp, "registrar_acceso", lambda *a, **k: pytest.fail("dejó rastro sin --aplicar"))
    codigo, salida = consola.correr("--todas", "--motivo", MOTIVO, "--admin", ADMIN)
    assert codigo == 0
    assert consola.marcadas == [], "sin --aplicar `marcar_varias` no se llama"
    assert consola.orden == ["pool", "leer"] and consola.lecturas == [None], "abre el pool y SOLO lee"
    assert "SIMULACION" in salida and "no se escribio nada" in salida and "--aplicar" in salida
    assert "cuentas a marcar: 3" in salida
    for c in consola.cuentas:
        assert c["email"] in salida and c["id"][:8] in salida


def test_con_aplicar_marca_en_tandas_de_100_y_cuenta_los_resultados(consola):
    consola.cuentas = _cuentas(250)
    ids = [c["id"] for c in consola.cuentas]

    def resultado(u):
        i = ids.index(u)
        return "no_existe" if i == 249 else "salio_ella" if i % 60 == 59 else "ya_marcada" if i % 10 == 0 else "marcada"
    consola.resultado = resultado
    codigo, salida = consola.correr("--todas", "--motivo", "  beta   cerrada  ", "--admin", ADMIN, "--aplicar")
    assert codigo == 0
    assert cp.MAX_LOTE == 100 and [len(t) for _, t, _ in consola.marcadas] == [100, 100, 50]
    assert [u for _, t, _ in consola.marcadas for u in t] == ids, "todas, en su orden, sin repetir"
    assert {(a, m) for a, _, m in consola.marcadas} == {(ADMIN, "beta cerrada")}, "el motivo va normalizado"
    esperado = {r: sum(1 for u in ids if resultado(u) == r) for r in ("marcada", "ya_marcada", "salio_ella", "no_existe")}
    assert all(n > 0 for n in esperado.values()) and sum(esperado.values()) == 250
    for r, n in esperado.items():
        assert f"  {r}: {n}\n" in salida, f"el conteo de {r}"
    for u in (u for u in ids if resultado(u) == "salio_ella"):
        assert f"{u[:8]}  cuenta{ids.index(u)}@correo.com" in salida, "las que salieron ellas, con su correo"
    assert "tanda 3/3: 50 cuentas" in salida


def test_con_la_funcion_de_verdad_deja_el_mismo_rastro_que_el_panel(monkeypatch, capsys):
    """Sin el doble de `marcar_varias`: el script llama a la función SSOT y el rastro sale ANTES de cada INSERT."""
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", ADMIN)
    b = _BDPrueba()
    monkeypatch.setattr(cp, "execute_sql_query", b.query)
    monkeypatch.setattr(cp, "execute_sql_write", b.write)
    monkeypatch.setattr(cp, "registrar_acceso", b.anotar)
    msc = _cargar()
    monkeypatch.setattr(msc, "_cargar_entorno", lambda: None)
    monkeypatch.setattr(msc, "_abrir_pool", lambda: None)
    monkeypatch.setattr(msc, "_leer_cuentas", lambda correos=None: [{"id": UID, "email": "ana@x.com"},
                                                                     {"id": OTRO, "email": "beto@x.com"}])
    assert msc.main(["--todas", "--motivo", MOTIVO, "--admin", ADMIN, "--aplicar"]) == 0
    assert [(a, o) for a, o, _ in b.rastro] == [("marcar_prueba", UID), ("marcar_prueba", OTRO)]
    assert b.orden == ["rastro", "escritura", "rastro", "escritura"], "una fila de rastro por cuenta, antes de su INSERT"
    assert {f["marcada_por"] for f in b.filas} == {ADMIN} and len(b.filas) == 2
    assert "  marcada: 2\n" in capsys.readouterr().out
    assert msc.main(["--todas", "--motivo", MOTIVO, "--admin", ADMIN, "--aplicar"]) == 0
    assert "  ya_marcada: 2\n" in capsys.readouterr().out, "relanzarlo no duplica marcas"
    assert len(b.filas) == 2


def test_correos_y_excluir_se_comparan_sin_mayusculas_ni_blancos_y_sin_repetir(consola):
    consola.cuentas = [{"id": UID, "email": "Ana@X.com"}, {"id": OTRO, "email": "beto@x.com"},
                       {"id": ADMIN2, "email": "cris@x.com"}]
    codigo, salida = consola.correr("--correos", "ANA@x.com, beto@x.com ,ana@x.com", "--motivo", MOTIVO,
                                    "--admin", ADMIN, "--aplicar")
    assert codigo == 0 and consola.lecturas == [["ana@x.com", "beto@x.com"]], "minúsculas, sin blancos, sin repetir"
    assert consola.marcadas == [(ADMIN, [UID, OTRO], MOTIVO)]
    consola.marcadas.clear()
    codigo, salida = consola.correr("--todas", "--excluir-correos", " CRIS@x.com ", "--motivo", MOTIVO,
                                    "--admin", ADMIN, "--aplicar")
    assert codigo == 0 and consola.marcadas == [(ADMIN, [UID, OTRO], MOTIVO)], "la excluida no se marca"
    assert "excluida  22222222  cris@x.com" in salida and "cuentas a marcar: 2 | excluidas: 1" in salida


@pytest.mark.parametrize("argumentos", [
    ("--correos", "ana@x.com,zzz@x.com"),                            # uno no existe
    ("--todas", "--excluir-correos", "erata@x.com"),                 # la errata en una exclusión marcaría a quien no debía
    ("--correos", "ana@x.com", "--excluir-correos", "beto@x.com"),   # excluir a quien no estaba pedido
])
def test_un_correo_que_no_corresponde_a_ninguna_cuenta_se_rechaza_sin_marcar_nada(consola, argumentos):
    consola.cuentas = [{"id": UID, "email": "ana@x.com"}, {"id": OTRO, "email": "beto@x.com"}]
    for extra in ((), ("--aplicar",)):
        codigo, salida = consola.correr(*argumentos, "--motivo", MOTIVO, "--admin", ADMIN, *extra)
        assert codigo == 2 and "RECHAZADO" in salida and "no se marco nada" in salida
        assert consola.marcadas == []


def test_sin_cuentas_no_hay_nada_que_marcar(consola):
    codigo, salida = consola.correr("--todas", "--motivo", MOTIVO, "--admin", ADMIN, "--aplicar")
    assert codigo == 0 and consola.marcadas == [] and "no hay cuentas que marcar" in salida


@pytest.mark.parametrize("extra", [(), ("--aplicar",)])
def test_con_el_interruptor_apagado_se_niega_antes_de_tocar_nada(consola, monkeypatch, extra):
    consola.cuentas = _cuentas(2)
    for apagado in (None, "false"):
        if apagado is None:
            monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS")
        else:
            monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", apagado)
        codigo, salida = consola.correr("--todas", "--motivo", MOTIVO, "--admin", ADMIN, *extra)
        assert codigo == 2 and "RECHAZADO" in salida and "MEALFIT_ADMIN_TEST_ACCOUNTS" in salida
        assert "apagado" in salida and salida.isascii()
        assert consola.orden == [] and consola.marcadas == [], "ni el pool, ni una lectura, ni una marca"


@pytest.mark.parametrize("argv,razon", [
    (["--todas", "--motivo", MOTIVO, "--admin", "no-es-un-uuid"], "uuid"),
    (["--todas", "--motivo", MOTIVO, "--admin", FUERA], "MEALFIT_ADMIN_USER_IDS"),   # bien formado, pero no es admin
    (["--todas", "--motivo", "ab", "--admin", ADMIN], "motivo"),
    (["--todas", "--motivo", "x" * 301, "--admin", ADMIN], "motivo"),
    (["--todas", "--motivo", "   ", "--admin", ADMIN], "motivo"),
    (["--correos", " , ", "--motivo", MOTIVO, "--admin", ADMIN], "vacio"),
])
def test_argumentos_invalidos_se_rechazan_antes_de_leer_la_base(consola, argv, razon):
    codigo, salida = consola.correr(*argv, "--aplicar")
    assert codigo == 2 and "RECHAZADO" in salida and razon in salida
    assert consola.orden == [] and consola.marcadas == []


def test_el_admin_puede_ser_cualquiera_de_la_lista_en_mayusculas_o_minusculas(consola):
    consola.cuentas = _cuentas(1)
    for admin in (ADMIN, ADMIN2, ADMIN2.upper()):
        assert consola.correr("--todas", "--motivo", MOTIVO, "--admin", admin)[0] == 0


@pytest.mark.parametrize("argumentos", [
    ("--todas", "--correos", "a@x.com", "--motivo", MOTIVO, "--admin", ADMIN),        # las dos a la vez
    ("--motivo", MOTIVO, "--admin", ADMIN),                                            # ninguna
    ("--todas", "--admin", ADMIN),                                                     # sin motivo
    ("--todas", "--motivo", MOTIVO),                                                   # sin admin
])
def test_todas_y_correos_son_excluyentes_y_lo_demas_es_obligatorio(consola, argumentos):
    codigo, _ = consola.correr(*argumentos)
    assert codigo == 2 and consola.orden == []


def test_la_base_que_no_responde_no_marca_nada(consola):
    consola.falla_la_lectura = RuntimeError("sin DB: secreto-interno")
    codigo, salida = consola.correr("--todas", "--motivo", MOTIVO, "--admin", ADMIN, "--aplicar")
    assert codigo == 1 and "FATAL" in salida and "RuntimeError" in salida and "secreto-interno" not in salida
    assert consola.marcadas == []
    consola.falla_la_lectura = None
    consola.falla_el_pool = RuntimeError("pool cerrado")
    codigo, salida = consola.correr("--todas", "--motivo", MOTIVO, "--admin", ADMIN, "--aplicar")
    assert codigo == 1 and "FATAL" in salida and consola.marcadas == []


@pytest.mark.parametrize("error,esperado", [
    (cp.ErrorPrueba(503, "No se pudo registrar la accion; no se hizo ningun cambio."), "503 No se pudo registrar"),
    (RuntimeError("sin DB"), "RuntimeError"),
])
def test_un_corte_a_mitad_sale_con_1_y_dice_que_quedo_hecho(consola, error, esperado):
    consola.cuentas = _cuentas(250)
    consola.falla_en, consola.falla_con = 2, error
    codigo, salida = consola.correr("--todas", "--motivo", MOTIVO, "--admin", ADMIN, "--aplicar")
    assert codigo == 1 and len(consola.marcadas) == 2, "se para en la tanda que falló: no sigue con la 3.ª"
    assert "CORTADO en la tanda 2 de 3" in salida and esperado in salida
    assert "ya_marcada" in salida, "dice que relanzar devuelve las hechas como ya_marcada"
    assert "  marcada: 100\n" in salida, "y lo que sí quedó hecho (la 1.ª tanda)"


def test_solo_imprime_ascii(consola):
    consola.cuentas = [{"id": UID, "email": "peña@correo.com"}, {"id": OTRO, "email": "ñandú@x.com"}]
    consola.resultado = lambda u: "salio_ella"
    for extra in ((), ("--aplicar",)):
        codigo, salida = consola.correr("--todas", "--motivo", "beta — sesión ñ", "--admin", ADMIN, *extra)
        assert codigo == 0 and salida.isascii(), salida
        assert "pe?a@correo.com" in salida
    assert consola.msc._parser().format_help().isascii(), "la ayuda tampoco lleva acentos"


def test_el_script_solo_habla_por_say_y_no_escribe_por_su_cuenta():
    src = _SCRIPT.read_text(encoding="utf-8")
    llamadas = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "print"]
    assert len(llamadas) == 1, "un solo `print`, dentro de `_say` (que lo pasa a ASCII)"
    assert "_ascii(texto)" in _cuerpo(src, "def _say")
    codigo = src.split('"""', 2)[2]            # sin el docstring del módulo
    assert not re.search(r"\b(INSERT|UPDATE|DELETE)\b", codigo), (
        "el script solo LEE `user_profiles`: escribir es de `cuentas_prueba.marcar_varias` (mismas reglas, mismo rastro)")
    assert "execute_sql_write" not in src and "marcar_varias(" in src and "cuentas_prueba.MAX_LOTE" in src


def test_el_orden_de_main_interruptor_luego_pool_luego_lectura_luego_escritura():
    cuerpo = _cuerpo(_SCRIPT.read_text(encoding="utf-8"), "def main")
    orden = ["parse_args(", "cuentas_prueba.activo()", "admin_ids()", "_abrir_pool()", "_leer_cuentas(correos)",
             "if not args.aplicar:", "cuentas_prueba.marcar_varias("]
    posiciones = [cuerpo.index(x) for x in orden]
    assert posiciones == sorted(posiciones), (
        "el interruptor va PRIMERO (antes del pool y de cualquier lectura); la escritura, la última y tras `--aplicar`")


def test_las_lecturas_del_script_son_selects_parametrizados(monkeypatch):
    import db
    vistos = []

    def _falso(q, p=None, fetch_one=False, fetch_all=False):
        vistos.append((" ".join(q.split()), p))
        return [{"id": UID, "email": "a@x.com"}]
    monkeypatch.setattr(db, "execute_sql_query", _falso)
    msc = _cargar()
    assert msc._leer_cuentas() == [{"id": UID, "email": "a@x.com"}]
    assert msc._leer_cuentas(["a@x.com", "b@x.com"])
    (q1, p1), (q2, p2) = vistos
    assert q1 == "SELECT id::text AS id, email FROM public.user_profiles ORDER BY created_at, id" and p1 is None
    assert q2 == ("SELECT id::text AS id, email FROM public.user_profiles WHERE lower(email) = ANY(%s::text[]) "
                  "ORDER BY created_at, id")
    assert p2 == (["a@x.com", "b@x.com"],), "los correos van como parámetro, nunca dentro del SQL"


def test_abrir_el_pool_como_pide_el_sop_de_sql_forense(monkeypatch):
    import db_core

    class _Pool:
        abierto = 0

        def open(self):
            self.abierto += 1
    pool = _Pool()
    monkeypatch.setattr(db_core, "connection_pool", pool)
    msc = _cargar()
    msc._abrir_pool()
    assert pool.abierto == 1
    monkeypatch.setattr(db_core, "connection_pool", None)
    with pytest.raises(RuntimeError):
        msc._abrir_pool()            # un pool sin configurar no se disimula: cada lectura saldría vacía


# ═════════════════════════════════════════════ 2. la doc
def _doc() -> str:
    return _DOC.read_text(encoding="utf-8")


def _filas_de_rutas(doc: str) -> list:
    """`[(método, ruta, limitador|None)]` de las tablas «Rutas» de la doc (la 1.ª celda empieza por `MÉTODO /ruta`)."""
    filas = []
    for linea in doc.splitlines():
        m = re.match(r"\|\s*`(GET|POST|PUT) (/[^`?]*)", linea)
        if not m:
            continue
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        lim = re.search(r"`(_[A-Z_]+_LIMITER)`", celdas[1]) if len(celdas) > 1 else None
        filas.append((m.group(1), m.group(2), lim.group(1) if lim else None))
    return filas


def _limitadores_de(ruta) -> set:
    """Los `RateLimiter` de una ruta, del decorador (`dependencies=[…]`) o de sus parámetros (`Depends(_X_LIMITER)`)."""
    return {d.call for d in ruta.dependant.dependencies if isinstance(d.call, RateLimiter)}


def _ruta(router, metodo, path):
    return next((r for r in router.routes if r.path == path and metodo in r.methods), None)


# Lo que el router ya tenía antes de este bloque (lotes 577 y 774): no es de esta doc.
_ANTERIORES = {("GET", "/api/admin/yo"), ("GET", "/api/admin/metricas"), ("POST", "/api/admin/cuentas/{user_id}/creditos"),
               ("POST", "/api/admin/cuentas/{user_id}/cortesia"), ("POST", "/api/admin/regalos/{grant_id}/revocar")}
_DETALLE = "/api/admin/cuentas/{user_id}/prueba/<sección>"
_PREFIJO_DETALLE = "/api/admin/cuentas/{user_id}/prueba/"


def test_la_doc_existe_y_el_panel_la_enlaza():
    assert _DOC.exists()
    panel = _PANEL_DOC.read_text(encoding="utf-8")
    assert "(admin_cuentas_actividad_pruebas.md)" in panel, "panel_admin.md enlaza la doc canónica"
    assert "marcar_cuentas_prueba.py" in panel and "MEALFIT_ADMIN_TEST_ACCOUNTS" in panel


def test_la_doc_nombra_cada_ruta_del_panel_con_su_limitador_real():
    doc = _doc()
    filas = [(m, "/api/admin" + p, lim) for m, p, lim in _filas_de_rutas(doc) if not p.startswith("/api/")]
    en_la_doc = {(m, p) for m, p, _ in filas}
    # cada ruta del router que NO es de antes tiene su fila (las 8 del detalle, en una sola con sus secciones)
    for r in ra.router.routes:
        for metodo in sorted(r.methods):
            if (metodo, r.path) in _ANTERIORES:
                continue
            if r.path.startswith(_PREFIJO_DETALLE) and metodo == "GET":
                seccion = r.path[len(_PREFIJO_DETALLE):]
                assert f"`{seccion}`" in doc, f"la doc no nombra la sección del detalle `{seccion}`"
                assert ("GET", _DETALLE) in en_la_doc
            else:
                assert (metodo, r.path) in en_la_doc, f"la doc no nombra {metodo} {r.path}"
    # y cada fila de la doc apunta a una ruta que existe, con el limitador que la ruta usa DE VERDAD
    for metodo, path, lim in filas:
        rutas = ([r for r in ra.router.routes if r.path.startswith(_PREFIJO_DETALLE) and metodo in r.methods]
                 if path == _DETALLE else [_ruta(ra.router, metodo, path)])
        assert rutas and all(rutas), f"la doc nombra {metodo} {path}, que no existe"
        assert lim, f"{metodo} {path}: la fila no dice su limitador"
        for r in rutas:
            assert getattr(ra, lim) in _limitadores_de(r), f"{metodo} {r.path}: no usa {lim}"


def test_la_doc_nombra_las_rutas_de_la_persona_con_su_limitador_real():
    persona = {(m, p): lim for m, p, lim in _filas_de_rutas(_doc()) if p.startswith("/api/")}
    assert set(persona) == {("GET", "/api/profile"), ("POST", "/api/profile/prueba/aviso-visto"),
                            ("POST", "/api/profile/prueba/salir"), ("PUT", "/api/profile/ajustes-dispositivo")}
    for (metodo, path), lim in persona.items():
        ruta = _ruta(ud.router, metodo, path)
        assert ruta is not None, f"{metodo} {path} no existe"
        if lim:
            assert getattr(ud, lim) in _limitadores_de(ruta), f"{metodo} {path}: no usa {lim}"
    assert persona[("GET", "/api/profile")] is None
    assert persona[("POST", "/api/profile/prueba/salir")] == persona[("POST", "/api/profile/prueba/aviso-visto")] == (
        "_PRUEBA_LIMITER")


def test_la_doc_da_cada_limitador_con_sus_numeros_reales():
    dichos: dict = {}
    for nombre, maximo, periodo in re.findall(r"`(_[A-Z_]+_LIMITER)` \((\d+)/(\d+)\)", _doc()):
        dichos.setdefault(nombre, set()).add((int(maximo), int(periodo)))      # cada mención, no solo la última
    assert set(dichos) == {"_CUENTAS_LISTA_LIMITER", "_CUENTAS_LECTURA_LIMITER", "_CUENTAS_ESCRITURA_LIMITER",
                           "_PRUEBA_DETALLE_LIMITER", "_PRUEBA_LIMITER", "_AJUSTES_DISPOSITIVO_LIMITER"}
    for nombre, pares in dichos.items():
        real = getattr(ra, nombre, None) or getattr(ud, nombre)
        assert pares == {(real.max_calls, real.period)}, f"{nombre}: la doc dice {sorted(pares)}, el código {real.max_calls}/{real.period}"


def test_la_doc_nombra_cada_accion_del_rastro():
    doc = _doc()
    admin = (_BACKEND / "routers" / "admin.py").read_text(encoding="utf-8")
    marcas = (_BACKEND / "cuentas_prueba.py").read_text(encoding="utf-8")
    acciones = (set(re.findall(r'_anotar_vista\(admin_id, "(\w+)"', admin))
                | set(re.findall(r'_anotar\(admin_id, "(\w+)"', marcas)))
    assert {"listar_cuentas", "exportar_cuentas", "ver_cuenta", "buscar_cuenta", "ver_ajustes", "ver_prueba",
            "marcar_prueba", "quitar_prueba"} <= acciones, acciones
    for accion in sorted(acciones):
        assert f"`{accion}`" in doc, f"la doc no nombra la acción del rastro `{accion}`"
    assert "`marcar_prueba_fallo`" in doc and "`quitar_prueba_fallo`" in doc


def test_la_doc_nombra_los_knobs_con_su_default_real_las_tablas_las_migraciones_el_cron_y_el_script(monkeypatch):
    doc = _doc()
    for knob in ("MEALFIT_ADMIN_TEST_ACCOUNTS", "MEALFIT_ADMIN_TEST_REQUIRE_NOTICE",
                 "MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS"):
        monkeypatch.delenv(knob, raising=False)
    for knob, defecto in (("MEALFIT_ADMIN_TEST_ACCOUNTS", cp.activo()), ("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE",
                                                                        cp.requiere_aviso()),
                          ("MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS", ajustes_cuenta.dias_de_historial())):
        assert f"`{knob}` | `{defecto}` |" in doc, f"la doc no da el default real de {knob}: {defecto}"
    for nombre in ("cuentas_de_prueba", "ajustes_cambios", "ajustes_dispositivo", "trg_ajustes_cambios",
                   "registrar_cambio_ajustes", "purge_ajustes_cambios", "scripts/marcar_cuentas_prueba.py",
                   "admin_access_log", "X-Admin-Accion", "MEALFIT_ADMIN_PANEL", "MEALFIT_ADMIN_USER_IDS"):
        assert nombre in doc, f"la doc no nombra {nombre}"
    for migracion in ("p1_plan_lote_830_cuentas_de_prueba_2026_09_29.sql", "p1_plan_lote_837_ajustes_cambios_2026_09_29.sql"):
        assert migracion in doc and (_BACKEND / "migrations" / migracion).exists(), migracion


def test_la_doc_explica_el_porque_legal_y_cuando_se_abre_el_contenido():
    doc = " ".join(_doc().split())
    for frase in ("Privacidad §5", "§2", "§9", "Protección de datos §5", "sin aviso por correo",
                  "desde que usted ve ese aviso", "SOLO de las cuentas de prueba",
                  "se comprueba en CADA petición, sin caché", "409 `aviso_pendiente`", "`motivo` y `error`",
                  "POST /api/profile/prueba/salir` funciona SIEMPRE"):
        assert frase in doc, frase


def test_la_doc_dice_como_anadir_un_ajuste_y_los_tests_que_cita_existen():
    doc = _doc()
    seccion = doc.split("## Cómo añadir un ajuste nuevo al registro", 1)[1].split("\n## ", 1)[0]
    for paso in ("REGISTRO", "COLUMNAS_VIGILADAS", "CLAVES_PERFIL_VIGILADAS", "CLAVES_DISPOSITIVO",
                 "_USER_SCOPED_KV_PREFIXES", "migración nueva", "las DOS carpetas"):
        assert paso in seccion, paso
    import db_profiles
    assert all(hasattr(ajustes_cuenta, n) for n in ("REGISTRO", "COLUMNAS_VIGILADAS", "CLAVES_PERFIL_VIGILADAS",
                                                     "CLAVES_DISPOSITIVO")) and hasattr(db_profiles, "_USER_SCOPED_KV_PREFIXES")
    todos = "\n".join(p.read_text(encoding="utf-8") for p in (_BACKEND / "tests").glob("test_*.py"))
    nombres = {n for n, es_fichero in re.findall(r"\b(test_[a-z0-9_]+)(\.py)?", seccion) if not es_fichero}
    assert len(nombres) >= 4, nombres
    for nombre in sorted(nombres):
        assert f"def {nombre}(" in todos, f"la doc cita el test `{nombre}`, que ya no existe"


def test_el_orden_de_despliegue_de_la_doc():
    despliegue = _doc().split("## Despliegue, en este orden", 1)[1].split("\n## ", 1)[0]
    pasos = ["**Migraciones**", "**Desplegar**", "**Publicar los textos legales**", "**Encender el interruptor:**",
             "**Marcar las cuentas existentes**"]
    posiciones = [despliegue.index(p) for p in pasos]
    assert posiciones == sorted(posiciones), "migraciones → desplegar → textos legales → interruptor → script"
    assert "apply_migration.py" in despliegue and "--status" in despliegue and "ROLLBACK" in despliegue


def test_los_numeros_de_la_doc_son_los_del_codigo():
    doc = " ".join(_doc().split())
    assert cp.MAX_LOTE == 100 and "Marca en tandas de 100" in doc and "(≤ 100 ids)" in doc
    assert acl.MAX_CSV == 5000 and "El CSV corta en 5.000 filas" in doc
    assert (apd.MARGEN_ANTES_H, apd.MARGEN_DESPUES_H) == (12, 14) and "12 h antes, 14 h después" in doc
    assert ajustes_cuenta.MAX_HISTORIAL == 500 and "(≤ 500 cambios)" in doc
    assert "[90, 3650]" in doc
