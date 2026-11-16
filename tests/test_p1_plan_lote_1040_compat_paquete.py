"""[P1-PLAN-LOTE-1040 · A-33 · 2026-11-16] El servidor no rompe a los paquetes de la app que siguen vivos.

Tras un deploy, el teléfono corre el paquete OTA ANTERIOR hasta su segundo arranque (14-nov: un chat corrió con el
paquete viejo justo tras el deploy). El contrato del stream fija lo que el cliente lee de cada evento:
  · cada evento que el servidor emite lleva los campos mínimos de su tipo (AST sobre los emisores);
  · el cliente no lee un campo que el contrato no conozca (lectura de `AgentPage.jsx`);
  · el servidor anota qué paquetes le hablan, para saber cuándo se puede retirar algo.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import compat_paquete as cp  # noqa: E402

_EMISORES = ("agent.py", "routers/chat.py", "coach_turno_repetido.py", "turno_desacoplado.py")


def _emisiones():
    """(fichero, línea, tipo, claves) de cada dict literal {'type': <tipo del contrato>, ...} de los emisores."""
    out = []
    for rel in _EMISORES:
        p = _BACKEND / rel
        if not p.exists():
            continue
        arbol = ast.parse(p.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Dict):
                continue
            claves = [k.value for k in nodo.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if "type" not in claves:
                continue
            v = nodo.values[[k.value if isinstance(k, ast.Constant) else None for k in nodo.keys].index("type")]
            if isinstance(v, ast.Constant) and v.value in cp.CONTRATO_STREAM:
                abiertos = any(k is None for k in nodo.keys)      # {**base, ...}: no se puede juzgar
                out.append((rel, nodo.lineno, v.value, set(claves), abiertos))
    return out


def test_el_servidor_emite_los_cuatro_tipos_que_el_cliente_lee():
    tipos = {t for _, _, t, _, _ in _emisiones()}
    assert tipos == set(cp.CONTRATO_STREAM), tipos


def test_cada_emision_lleva_los_campos_minimos_de_su_tipo():
    faltan = [f"{rel}:{ln} {t} sin {sorted(cp.CAMPOS_MINIMOS[t] - claves)}"
              for rel, ln, t, claves, abiertos in _emisiones()
              if not abiertos and not cp.CAMPOS_MINIMOS[t] <= claves]
    assert not faltan, "un paquete vivo no podría pintar estos eventos:\n" + "\n".join(faltan)


def test_cada_campo_opcional_del_contrato_lo_manda_algun_emisor():
    vistos = {}
    for _, _, t, claves, _ in _emisiones():
        vistos.setdefault(t, set()).update(claves)
    for t, opcionales in cp.CAMPOS_OPCIONALES.items():
        assert opcionales <= vistos.get(t, set()), f"{t}: el contrato promete {sorted(opcionales - vistos.get(t, set()))}"


def _campos_que_lee_el_cliente():
    fe = _BACKEND.parent / "frontend" / "src" / "pages" / "AgentPage.jsx"
    if not fe.exists():
        pytest.skip("sin el frontend al lado")
    src = fe.read_text(encoding="utf-8")
    i = src.index("if (dataObj.type === 'progress') {")
    j = src.index("} else if (dataObj.type === 'error') {", i)
    k = src.index("\n                                    }", j + 10)
    return set(re.findall(r"dataObj\.([a-zA-Z_]+)", src[i:k])) - {"type"}


def test_el_cliente_solo_lee_campos_del_contrato():
    contrato = set().union(*cp.CONTRATO_STREAM.values())
    nuevos = _campos_que_lee_el_cliente() - contrato
    assert not nuevos, (f"AgentPage lee {sorted(nuevos)} y el contrato no lo conoce: añádelo a compat_paquete "
                        "(y que el servidor lo mande) o un paquete viejo/nuevo quedará leyendo un campo que no llega")


# ───────────────────────────────────────────────────────────── paquetes vivos
class _KV:
    def __init__(self):
        self.valor = None
        self.escrituras = 0

    def query(self, sql, params=None, **k):
        return {"value": self.valor} if self.valor is not None else None

    def write(self, sql, params=None, **k):
        self.escrituras += 1
        self.valor = json.loads(params[1])


@pytest.fixture(autouse=True)
def _limpio():
    cp.reset_para_tests()
    yield
    cp.reset_para_tests()


@pytest.mark.parametrize("valor,esperado", [
    ("ota:20261116-054412", "ota:20261116-054412"),
    ("web:20261116-054510", "web:20261116-054510"),
    ("web:?", None), ("", None), (None, None), ("ota:" + "x" * 60, None),
    ("ota:1; DROP TABLE x", None), ({"a": 1}, None),
])
def test_solo_entra_un_paquete_con_forma(valor, esperado):
    assert cp.paquete_limpio(valor) == esperado


def test_registra_un_paquete_nuevo_al_momento_y_luego_como_mucho_cada_10_min():
    kv = _KV()
    assert cp.registrar_paquete("ota:20261116-054412", 1000.0, kv.query, kv.write) is True
    assert cp.registrar_paquete("ota:20261116-054412", 1100.0, kv.query, kv.write) is False
    assert cp.registrar_paquete("web:20261116-054510", 1200.0, kv.query, kv.write) is True     # nuevo: se escribe
    assert cp.registrar_paquete("ota:20261116-054412", 1000.0 + cp.ESCRIBIR_CADA_S + 300, kv.query, kv.write) is True
    assert kv.escrituras == 3 and set(kv.valor) == {"ota:20261116-054412", "web:20261116-054510"}


def test_lo_que_no_tiene_forma_no_se_escribe():
    kv = _KV()
    assert cp.registrar_paquete("web:?", 1000.0, kv.query, kv.write) is False
    assert kv.escrituras == 0


def test_los_paquetes_vivos_por_tipo_con_el_mas_viejo():
    kv = _KV()
    kv.valor = {"ota:20261101-000000": 1000.0, "ota:20261116-054412": 5000.0, "web:20261116-054510": 5000.0,
                "ota:20260901-000000": 5000.0 - 40 * 86400}
    v = cp.paquetes_vivos(dias=7, ahora=5000.0 + 3600, execute_sql_query=kv.query)
    assert v["n"] == 3 and v["por_tipo"] == {"ota": 2, "web": 1}
    assert v["mas_viejo"]["ota"] == "ota:20261101-000000" and v["mas_nuevo"]["ota"] == "ota:20261116-054412"
    assert v["protocolo_stream"] == cp.PROTOCOLO_STREAM


def test_un_fallo_de_la_base_no_rompe_nada():
    def roto(*a, **k):
        raise RuntimeError("neon")
    import time
    assert cp.registrar_paquete("ota:20261116-054412", time.time(), roto, roto) is False
    assert cp.paquetes_vivos(execute_sql_query=roto)["n"] == 1    # el de la memoria del proceso


def test_cableado_en_el_stream_y_en_health_version():
    chat = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
    assert '__import__("compat_paquete").registrar_paquete(data.get("bundle"))' in chat
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "from compat_paquete import paquetes_vivos_cacheado" in app and '"paquetes": paquetes,' in app


def test_el_cliente_manda_su_paquete_en_el_cuerpo_no_en_una_cabecera():
    fe = _BACKEND.parent / "frontend" / "src"
    if not fe.exists():
        pytest.skip("sin el frontend al lado")
    agente = (fe / "pages" / "AgentPage.jsx").read_text(encoding="utf-8")
    assert "bundle: paqueteDelCliente()," in agente
    perf = (fe / "utils" / "chatPerf.js").read_text(encoding="utf-8")
    assert "export const paqueteDelCliente" in perf
    assert "X-Bundle" not in agente and "X-Mealfit-Bundle" not in perf, "una cabecera nueva exigiría su preflight CORS"
