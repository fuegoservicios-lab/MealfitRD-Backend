# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-747 · 2026-09-28] Una Nevera que la guarda ya refutó no se vuelve a probar con 3 corridas del LLM.

Caso REAL (plan 3957a669, la única usuaria externa con relleno ligado a su Nevera). Su Nevera nació con UNA compra
(04-sep, «📦 [RESTOCK] 41/41») y nunca cambió: ni compras, ni consumos; lo único que se movió fueron las reservas.
Los bloques 3, 4, 7 y 9 del relleno siguieron el mismo camino, medido en el journal del VPS:

    intento 1 → «INEXISTENTES: aguacate, edamame, guineo…» → intento 2 → lo mismo → intento 3 → lo mismo
    → pausa `pending_user_action:pantry_violation_after_retries` (17-28 min de pipeline tirados)
    → 12 h de TTL → modo flexible → entrega con 🚨 Compra Urgente.

Los 4 bloques acabaron entregados por la vía flexible; lo gastado en esos intentos descartados fue el 49 % de todo su
gasto de LLM (0,94 de 1,92 US$). Desde el segundo, el resultado era predecible ANTES de llamar al modelo: la misma guarda
ya había refutado esa misma Nevera y la Nevera no había cambiado.

El pre-chequeo (`nevera_refutada.prechequeo`, antes del bucle de reintentos del worker):
  1. SSOT de las guardas: si `_pantry_gate_waiver_reason` da un motivo (flexible, advisory, invitado, Nevera virtual,
     autonomía del `initial_plan` —y con ella P1-FIRST-PURCHASE-PAUSE—), no hace nada. Ninguna guarda decide sola.
  2. Evidencia: la última refutación del plan (`plan_data._pantry_refutation`, escrita por la guarda de existencia al
     agotar sus reintentos) con sus líneas INEXISTENTES, la huella BRUTA de la Nevera de ANTES de los intentos
     (∩ la de después: una compra durante la corrida no entra), los alimentos DISPONIBLES en las dos lecturas y la
     huella del GENERADOR que falló (hash corto: proveedor, modelos, esfuerzo del day-gen y tier). Un éxito estricto
     de la guarda la borra. Ni `registrar` ni `olvidar` sellan `_plan_modified_at` (contabilidad interna).
  3. La Nevera no creció desde entonces (huella bruta: ni alimento nuevo ni más cantidad) — las reservas no cuentan:
     suben y bajan solas y no son una compra —, y no hay hoy disponible un alimento que entonces estaba reservado
     entero.
  4. La MISMA medida de la guarda (`validate_ingredients_against_pantry` + `compras_pequenas.tolerar`) sigue
     rechazando esas líneas contra la Nevera de hoy.
  5. El bloque sigue vivo (`_validate_chunk_pre_llm`).
  ⇒ entra directo por donde el sistema acababa: la pausa `pantry_violation_after_retries` (mismo helper con la marca
     `_pantry_pause_precheck_at`, mismo push si la pausa ocurrió); el recovery de siempre decide lo demás.
La evidencia caduca (`MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H`, 168 h), vale sólo para el mismo generador
(proveedor + modelos por tier) y `MEALFIT_PANTRY_REFUTED_PRECHECK_EPOCH` la invalida toda sin apagar el pre-chequeo.
Knob `MEALFIT_PANTRY_REFUTED_PRECHECK` (True) — apagado, conducta previa exacta (ni lee ni escribe).

tooltip-anchor: P1-PLAN-LOTE-747
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_MOD = _BACKEND / "nevera_refutada.py"

# La Nevera REAL de 3957a669 (user_inventory, kind='food'): 41 filas, todas del restock del 04-sep, sin cambios después.
INVENTARIO = [
    ("Plátano", 1, "unidad"), ("Sal", 453.6, "g"), ("Ajo", 1, "paquete (4 uds.)"), ("Orégano", 45, "g"),
    ("Pimienta negra", 14.2, "g"), ("Vainilla", 141.7, "g"), ("Mantequilla de maní", 453.6, "g"),
    ("Cilantro", 1, "mazo"), ("Polvo de hornear", 150, "g"), ("Canela en polvo", 14.2, "g"), ("Comino", 28.3, "g"),
    ("Aceite de oliva", 230, "g"), ("Laurel", 100, "g"), ("Queso de hoja", 0.25, "lb"), ("Perejil", 1, "mazo"),
    ("Leche descremada", 1031.3, "g"), ("Yogurt", 1960, "g"), ("Uva", 0.75, "lb"), ("Costilla de cerdo", 0.5, "lb"),
    ("Ají cubanela", 0.5, "lb"), ("Limón", 3, "unidad"), ("Mango", 4, "unidad"), ("Soya texturizada", 200, "g"),
    ("Pasta integral", 500, "g"), ("Lechosa", 1, "unidad"), ("Tayota", 2.75, "lb"), ("Cebollín", 1, "mazo"),
    ("Piña", 1, "unidad"), ("Avena", 650, "g"), ("Casabe", 35, "g"), ("Huevo", 1, "cartón (20 uds.)"),
    ("Maní", 800, "g"), ("Harina de trigo", 907.18, "g"), ("Nabo", 2, "unidad"), ("Calabacín", 1.25, "lb"),
    ("Tomate", 1.75, "lb"), ("Habichuelas blancas", 800, "g"), ("Cebolla", 1.5, "lb"), ("Tilapia", 5, "unidad"),
    ("Queso blanco", 0.25, "lb"), ("Ají morrón", 9, "unidad"),
]

# Líneas INEXISTENTES de los intentos de cada refutación (journal del VPS, 10/11/21-sep).
REFUTADAS_S3 = ["10 g de aguacate", "10 g de arroz blanco", "45 g de guineo"]
REFUTADAS_S4 = ["½ guineo", "10 g de semillas de linaza", "120 g de edamame cocido", "30 g de proteína whey",
                "20 g de semillas de girasol", "15 g de aguacate", "30 g de arroz blanco crudo"]
#: Generador de los tests (proveedor | modelo del usuario | modelo del day-gen | esfuerzo | tier) y su huella: el hash
#: corto de «época del knob|generador» (plan_data llega al cliente: ningún nombre de proveedor en la evidencia).
MODELOS = "zai|glm-5.3-flash|gpt-6-luna|low|gratis"
GENERADOR = __import__("hashlib").sha256(("1|" + MODELOS).encode("utf-8")).hexdigest()[:12]

REFUTADAS_S7 = ["15 g de aguacate", "95 g de edamame cocido", "2 rebanadas de pan integral familiar",
                "80 g de camarones cocidos", "½ guineo maduro (74 g)", "15 g de arroz blanco crudo",
                "15 g de berenjena", "150 g de edamame cocido"]


def _filas(inv):
    return [{"ingredient_name": n, "quantity": q, "unit": u} for n, q, u in inv]


def _nevera_neta(inv, reservado=None):
    """Las líneas que la guarda ve (formato de `get_user_inventory_net`), con parte reservada descontada."""
    from shopping_calculator import get_plural_unit
    reservado = reservado or {}
    out = []
    for n, q, u in inv:
        q = float(q) - float(reservado.get(n, 0))
        if q <= 0:
            continue
        qr = f"{q:.2f}".rstrip("0").rstrip(".")
        out.append(f"{qr} {n}" if u == "unidad" else f"{qr} {get_plural_unit(q, u)} de {n}")
    return sorted(out)


def _resultado_guarda(lineas):
    """El texto con el que `validate_ingredients_against_pantry` rechaza la existencia (lo que registra el worker)."""
    return ("ERRORES DE DESPENSA HALLADOS OBLIGANDO A CORREGIR:\n"
            f"- Ingredientes COMPLETAMENTE INEXISTENTES en inventario: {', '.join(lineas)}.\n"
            "Corrige tu respuesta bajando las porciones estrictamente numéricas al límite exacto, O eliminando/"
            "sustituyendo ingredientes.")


@pytest.fixture
def nr(monkeypatch):
    import nevera_refutada as m
    monkeypatch.delenv("MEALFIT_PANTRY_REFUTED_PRECHECK", raising=False)
    monkeypatch.delenv("MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H", raising=False)
    monkeypatch.setenv("MEALFIT_INITIAL_CHUNK_PANTRY_AUTONOMY", "true")
    escrituras, metricas = [], []
    monkeypatch.delenv("MEALFIT_PANTRY_REFUTED_PRECHECK_EPOCH", raising=False)
    monkeypatch.setattr(m, "_leer_bruto", lambda user_id: _filas(INVENTARIO))
    monkeypatch.setattr(m, "_escribir", lambda sql, params: escrituras.append((sql, params)))
    monkeypatch.setattr(m, "_metrica", lambda **kw: metricas.append(kw))
    # El generador que escribiría el bloque (proveedor + modelos por tier): fijo en los tests.
    monkeypatch.setattr(m, "_generador_actual", lambda user_id: MODELOS, raising=False)
    m._escrituras, m._metricas = escrituras, metricas
    return m


@pytest.fixture
def worker(monkeypatch):
    """Los dos efectos del camino de siempre, capturados (jamás tocan la DB ni mandan pushes)."""
    import cron_tasks
    pausas, pushes = [], []
    monkeypatch.setattr(cron_tasks, "_pause_chunk_for_pantry_refresh",
                        lambda task_id, user_id, week_number, fresh_inventory, reason="empty_pantry", notify=True,
                        precheck=False:
                        pausas.append({"task_id": task_id, "week": week_number, "reason": reason,
                                       "n": len(fresh_inventory or [])}))
    monkeypatch.setattr(cron_tasks, "_dispatch_push_notification", lambda **kw: pushes.append(kw))
    monkeypatch.setattr(cron_tasks, "_validate_chunk_pre_llm", lambda task_id, meal_plan_id, user_id: "ok")
    return pausas, pushes


def _evidencia(lineas, inv=INVENTARIO, horas=10):
    at = (datetime.now(timezone.utc) - timedelta(hours=horas)).isoformat()
    import nevera_refutada as m
    return {"v": 3, "at": at, "week": 4, "chunk_kind": "rolling_refill", "lines": list(lineas),
            "bruto": m.huella_bruta(_filas(inv)), "disponible": m.huella_disponible(_filas(inv)), "gen": GENERADOR}


def _pre(m, *, plan_data, chunk_kind="rolling_refill", form_data=None, snap=None):
    fd = {"current_pantry_ingredients": _nevera_neta(INVENTARIO), "_fresh_pantry_source": "live"}
    fd.update(form_data or {})
    return m.prechequeo(task_id="t-1", user_id="u-1", meal_plan_id="p-1", week_number=7, chunk_kind=chunk_kind,
                        snap=snap or {}, form_data=fd, plan_data=plan_data, country="DO")


# ─────────────── la medida ───────────────

def test_la_huella_bruta_solo_crece_con_una_compra(nr):
    antes = nr.huella_bruta(_filas(INVENTARIO))
    assert not nr.mas_rica(antes, antes)
    comido = [(n, q / 2, u) if n == "Yogurt" else (n, q, u) for n, q, u in INVENTARIO]
    assert not nr.mas_rica(nr.huella_bruta(_filas(comido)), antes), "consumir no es comprar"
    sin_uva = [r for r in INVENTARIO if r[0] != "Uva"]
    assert not nr.mas_rica(nr.huella_bruta(_filas(sin_uva)), antes)
    assert nr.mas_rica(nr.huella_bruta(_filas(INVENTARIO + [("Aguacate", 2, "unidad")])), antes), "alimento nuevo"
    mas = [(n, q * 2, u) if n == "Huevo" else (n, q, u) for n, q, u in INVENTARIO]
    assert nr.mas_rica(nr.huella_bruta(_filas(mas)), antes), "más cantidad de lo que ya había"
    otra_unidad = [(n, 900, "g") if n == "Tilapia" else (n, q, u) for n, q, u in INVENTARIO]
    assert nr.mas_rica(nr.huella_bruta(_filas(otra_unidad)), antes), "otra unidad: no se compara, se asume compra"


def test_las_lineas_refutadas_son_la_union_de_los_intentos(nr):
    lineas = nr.lineas_refutadas(_resultado_guarda(["45 g de guineo", "10 g de aguacate"]),
                                 _resultado_guarda(["10 g de aguacate", "30 g de proteína whey"]), True, None,
                                 "CANTIDADES: excede el límite de Huevo")
    assert lineas == ["45 g de guineo", "10 g de aguacate", "30 g de proteína whey"]


# ─────────────── la evidencia ───────────────

def test_la_evidencia_vale_mientras_la_nevera_no_cambie(nr):
    ev = _evidencia(REFUTADAS_S4)
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") == ev


def test_sin_evidencia_o_caducada_no_hay_prediccion(nr, monkeypatch):
    assert nr.evidencia_vigente({}, _nevera_neta(INVENTARIO), "u-1", "DO") is None
    assert nr.evidencia_vigente({"_pantry_refutation": {"at": "basura"}}, _nevera_neta(INVENTARIO), "u-1", "DO") is None
    vieja = _evidencia(REFUTADAS_S4, horas=24 * 15)
    assert nr.evidencia_vigente({"_pantry_refutation": vieja}, _nevera_neta(INVENTARIO), "u-1", "DO") is None
    monkeypatch.setenv("MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H", "400")
    assert nr.evidencia_vigente({"_pantry_refutation": vieja}, _nevera_neta(INVENTARIO), "u-1", "DO") == vieja
    sin_lineas = dict(_evidencia([]), lines=[])
    assert nr.evidencia_vigente({"_pantry_refutation": sin_lineas}, _nevera_neta(INVENTARIO), "u-1", "DO") is None


def test_una_compra_anula_la_evidencia(nr, monkeypatch):
    ev = _evidencia(REFUTADAS_S4)
    comprado = INVENTARIO + [("Guineo", 6, "unidad")]
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: _filas(comprado))
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(comprado), "u-1", "DO") is None


def test_sin_poder_leer_la_nevera_bruta_no_se_predice(nr, monkeypatch):
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: None)
    ev = _evidencia(REFUTADAS_S4)
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None


def test_si_la_nevera_de_hoy_ya_cubre_lo_refutado_no_se_predice(nr):
    """La misma medida que la guarda: si hoy lo refutado entra (o son compras pequeñas), se deja probar al LLM."""
    ev = _evidencia(["2 huevos", "100 g de avena", "30 g de maní"])
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None
    pequenas = _evidencia(["15 g de aguacate"])
    assert nr.evidencia_vigente({"_pantry_refutation": pequenas}, _nevera_neta(INVENTARIO), "u-1", "DO") is None


# ─────────────── el pre-chequeo en el worker ───────────────

def test_entra_directo_por_la_pausa_de_siempre(nr, worker):
    pausas, pushes = worker
    ev = _evidencia(REFUTADAS_S4)
    assert _pre(nr, plan_data={"_pantry_refutation": ev}) is True
    assert pausas == [{"task_id": "t-1", "week": 7, "reason": "pantry_violation_after_retries", "n": 41}]
    assert len(pushes) == 1 and pushes[0]["title"] == "Tu plan necesita revisión de ingredientes"
    assert pushes[0]["user_id"] == "u-1" and pushes[0]["url"] == "/dashboard"
    assert nr._metricas and nr._metricas[0]["lineas"] == len(REFUTADAS_S4)
    assert nr._escrituras == [], "el pre-chequeo NO reescribe la evidencia: su edad sigue siendo la de la refutación real"


@pytest.mark.parametrize("extra, snap, fuente", [
    ({"_pantry_flexible_mode": True}, {}, "live"),
    ({}, {"_pantry_flexible_mode": True}, "live"),
    ({"_pantry_advisory_only": True}, {}, "live"),
    ({"_nevera_virtual": True}, {}, "live"),
    ({}, {}, "guest"),
])
def test_el_waiver_ssot_manda(nr, worker, extra, snap, fuente):
    pausas, pushes = worker
    fd = dict(extra, _fresh_pantry_source=fuente)
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)}, form_data=fd, snap=snap) is False
    assert pausas == [] and pushes == []


def test_la_autonomia_del_initial_plan_y_la_primera_compra_no_se_tocan(nr, worker, monkeypatch):
    """Mismo waiver que la guarda de existencia: `initial_plan` no la pasa (P1-FIRST-PURCHASE-PAUSE vive ANTES, en el
    gate pre-pipeline, y no se adelanta). Sólo con la autonomía apagada el pre-chequeo actúa, igual que la guarda."""
    pausas, _ = worker
    ev = {"_pantry_refutation": _evidencia(REFUTADAS_S4)}
    assert _pre(nr, plan_data=ev, chunk_kind="initial_plan") is False and pausas == []
    monkeypatch.setenv("MEALFIT_INITIAL_CHUNK_PANTRY_AUTONOMY", "false")
    assert _pre(nr, plan_data=ev, chunk_kind="initial_plan") is True and len(pausas) == 1


def test_sin_nevera_no_hay_guarda_ni_prechequeo(nr, worker):
    pausas, _ = worker
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)},
                form_data={"current_pantry_ingredients": []}) is False
    assert pausas == []


def test_knob_apagado_conducta_previa_exacta(nr, worker, monkeypatch):
    pausas, _ = worker
    monkeypatch.setenv("MEALFIT_PANTRY_REFUTED_PRECHECK", "false")
    leidas = []
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: leidas.append(user_id) or _filas(INVENTARIO))
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)}) is False
    assert nr.huella_inicial("u-1") is None
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),),
                        inicio=_inicio_de(nr)) is False
    assert nr.olvidar("p-1", "u-1", {"_pantry_refutation": _evidencia(REFUTADAS_S4)}) is False
    assert pausas == [] and nr._escrituras == [] and leidas == []


def test_un_fallo_al_decidir_deja_correr_al_llm(nr, worker, monkeypatch):
    pausas, _ = worker

    def _roto(user_id):
        raise RuntimeError("DB caída")

    monkeypatch.setattr(nr, "_leer_bruto", _roto)
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)}) is False and pausas == []


# ─────────────── la escritura de la evidencia ───────────────

def test_registrar_escribe_quirurgico_y_con_dueno(nr):
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill",
                        (_resultado_guarda(REFUTADAS_S4[:3]), _resultado_guarda(REFUTADAS_S4[3:])),
                        inicio=_inicio_de(nr)) is True
    (sql, params), = nr._escrituras
    assert "jsonb_set" in sql and "'{_pantry_refutation}'" in sql
    assert "_plan_modified_at" not in sql, "contabilidad interna: no promueve el plan a activo (ronda 2, punto 1)"
    assert re.search(r"WHERE id = %s AND user_id = %s\s*$", sql.strip())
    ev = json.loads(params[0])
    assert params[1:] == ("p-1", "u-1")
    assert ev["lines"] == REFUTADAS_S4 and ev["week"] == 4 and ev["chunk_kind"] == "rolling_refill"
    assert ev["bruto"] == nr.huella_bruta(_filas(INVENTARIO))
    assert ev["disponible"] == nr.huella_disponible(_filas(INVENTARIO))
    assert ev["gen"] == GENERADOR and ev["v"] == 3


def test_sin_lineas_inexistentes_no_hay_evidencia(nr):
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", ("CANTIDADES: excede", True, None),
                        inicio=_inicio_de(nr)) is False
    assert nr._escrituras == []


def test_el_ciclo_completo_registrar_y_despues_prechequear(nr, worker):
    pausas, _ = worker
    nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),),
                 inicio=nr.huella_inicial("u-1"))
    ev = json.loads(nr._escrituras[0][1][0])
    assert _pre(nr, plan_data={"_pantry_refutation": ev}) is True and len(pausas) == 1


# ─────────────── con los datos reales de 3957a669 ───────────────

@pytest.mark.parametrize("bloque, refutadas, reservado_hoy", [
    ("S4 (11-sep, tras la refutación del S3)", REFUTADAS_S3, {"Yogurt": 250, "Avena": 80, "Huevo": 0.2}),
    ("S7 (21-sep, tras la del S4)", REFUTADAS_S4, {"Yogurt": 125, "Tilapia": 1, "Plátano": 1}),
    ("S9 (27-sep, tras la del S7)", REFUTADAS_S7, {}),
])
def test_los_tres_bloques_refutados_que_se_podian_prever(nr, worker, bloque, refutadas, reservado_hoy):
    """La Nevera bruta no cambió desde el 04-sep; lo refutado sigue fuera de ella ⇒ los tres bloques entran por la pausa
    sin sus 3 intentos. Las reservas (lo único que se movía) no cuentan como compra."""
    pausas, _ = worker
    ev = _evidencia(refutadas, horas=24 * 6)
    fd = {"current_pantry_ingredients": _nevera_neta(INVENTARIO, reservado_hoy)}
    assert _pre(nr, plan_data={"_pantry_refutation": ev}, form_data=fd) is True, bloque
    assert pausas[-1]["reason"] == "pantry_violation_after_retries"


def test_si_hubiera_comprado_lo_que_faltaba_el_llm_si_corre(nr, worker, monkeypatch):
    pausas, _ = worker
    comprado = INVENTARIO + [("Aguacate", 2, "unidad"), ("Guineo", 6, "unidad"), ("Edamame", 500, "g")]
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: _filas(comprado))
    fd = {"current_pantry_ingredients": _nevera_neta(comprado)}
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S7)}, form_data=fd) is False
    assert pausas == []


# ─────────────── el cableado en el worker (fuente) ───────────────

def _cuerpo_worker() -> str:
    tree = ast.parse(_CRON)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_chunk_worker")
    return ast.get_source_segment(_CRON, fn) or ""


def test_el_prechequeo_va_despues_de_refrescar_la_nevera_y_antes_del_llm():
    w = _cuerpo_worker()
    gancho = w.index('__import__("nevera_refutada").prechequeo(')
    assert w.index("form_data = _refresh_chunk_pantry(") < gancho < w.index("for _pantry_attempt in range(")
    linea = w[w.rfind("\n", 0, gancho):w.index("\n", gancho)]
    assert "return" in linea and "P1-PLAN-LOTE-747" in linea
    for kw in ("chunk_kind=chunk_kind", "snap=snap", "form_data=form_data", "plan_data=prior_plan_data",
               "country=_pantry_guard_country"):
        assert kw in linea, kw


def test_la_refutacion_se_registra_antes_de_pisar_la_correccion_anterior():
    w = _cuerpo_worker()
    pausa = w.index('reason="pantry_violation_after_retries",')
    gancho = w.rindex('__import__("nevera_refutada").registrar(', 0, pausa)
    pisada = w.rindex('form_data["_pantry_correction"] = str(_val_result)[:1000]', 0, pausa)
    assert gancho < pisada, "registrar debe leer la corrección del intento anterior ANTES de que se pise"
    assert w.rindex("Violación persistente tras", 0, pausa) < gancho
    linea = w[w.rfind("\n", 0, gancho):w.index("\n", gancho)]
    assert 'form_data.get("_pantry_correction")' in linea and "_val_result" in linea
    assert "inicio=_nr_inicio" in linea, "la base de la evidencia es la Nevera de ANTES de los intentos"


def test_el_modulo_consulta_la_ssot_con_los_mismos_argumentos_que_la_guarda():
    src = _MOD.read_text(encoding="utf-8")
    codigo = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    tree = ast.parse(src)
    llamadas = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", getattr(n.func, "id", "")) == "_pantry_gate_waiver_reason"]
    assert len(llamadas) == 1
    guarda = next(n for n in ast.walk(ast.parse(_CRON)) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_exist_waiver" for t in n.targets))
    assert {k.arg for k in llamadas[0].keywords} == {k.arg for k in guarda.value.keywords}
    for flag in ('"_pantry_flexible_mode"', '"_pantry_advisory_only"', "'_pantry_flexible_mode'"):
        assert flag not in codigo, f"{flag}: el pre-chequeo no lee flags de nevera por su cuenta (P1-PANTRY-GATE-SSOT)"


def test_knob_registrado_y_marker_anclado(nr):
    from knobs import _KNOBS_REGISTRY
    nr.activo()
    nr.max_edad_h()
    nr.epoca()
    assert "MEALFIT_PANTRY_REFUTED_PRECHECK" in _KNOBS_REGISTRY
    assert "MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H" in _KNOBS_REGISTRY
    assert "MEALFIT_PANTRY_REFUTED_PRECHECK_EPOCH" in _KNOBS_REGISTRY
    assert _CRON.count("P1-PLAN-LOTE-747") >= 2
    assert "tooltip-anchor: P1-PLAN-LOTE-747" in _MOD.read_text(encoding="utf-8")


# ─────────────── ronda 1 de la revisión adversaria ───────────────
# Cada test reproduce un defecto de la revisión (rojo contra 438185c3) y fija su arreglo.

def test_defecto1_una_compra_durante_la_corrida_no_entra_en_la_base(nr, worker, monkeypatch):
    """La base de la evidencia es la Nevera de ANTES de los intentos (∩ la de después), no la del final: lo que el
    usuario compra mientras el LLM corre no lo vio ninguna guarda y no puede quedar «refutado» una semana."""
    pausas, _ = worker
    antes = nr.huella_bruta(_filas(INVENTARIO))
    comprado = INVENTARIO + [("Pechuga de pollo", 2, "lb")]  # compra a mitad de la corrida (no es lo refutado)
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: _filas(comprado))
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),),
                        inicio=_inicio_de(nr)) is True
    ev = json.loads(nr._escrituras[-1][1][0])
    assert "pechuga de pollo|lb" not in ev["bruto"], "la compra de durante la corrida entró en la base refutada"
    assert ev["bruto"] == antes
    # El bloque siguiente, con esa compra en la Nevera, SÍ corre el LLM.
    fd = {"current_pantry_ingredients": _nevera_neta(comprado)}
    assert _pre(nr, plan_data={"_pantry_refutation": ev}, form_data=fd) is False and pausas == []


def test_defecto1_la_base_es_la_interseccion_y_sin_lectura_inicial_no_hay_evidencia(nr, monkeypatch):
    antes = nr.huella_bruta(_filas(INVENTARIO))
    comido = [(n, q / 2, u) if n == "Yogurt" else (n, q, u) for n, q, u in INVENTARIO if n != "Uva"]
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: _filas(comido))
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),), inicio=_inicio_de(nr))
    ev = json.loads(nr._escrituras[-1][1][0])
    assert ev["bruto"] == nr.huella_bruta(_filas(comido)), "min(antes, después) por alimento"
    n = len(nr._escrituras)
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),),
                        inicio=None) is False
    assert len(nr._escrituras) == n


def test_defecto1_la_huella_inicial_se_toma_antes_de_refrescar_la_nevera():
    w = _cuerpo_worker()
    gancho = w.index('__import__("nevera_refutada").prechequeo(')
    refresco = w.rindex("form_data = _refresh_chunk_pantry(", 0, gancho)
    inicial = w.rindex('_nr_inicio = __import__("nevera_refutada").huella_inicial(user_id)', 0, gancho)
    assert inicial < refresco, "la huella inicial debe leerse ANTES de capturar la Nevera que validará la guarda"
    assert w[inicial:refresco].count("\n") <= 1, "pegada al refresco: nada entre la lectura bruta y la neta"
    assert w.count("_nr_inicio = ") == 1


def test_defecto2_un_exito_estricto_borra_la_evidencia(nr):
    plan_data = {"_pantry_refutation": _evidencia(REFUTADAS_S4), "days": []}
    assert nr.olvidar("p-1", "u-1", plan_data) is True
    (sql, params), = nr._escrituras
    assert "plan_data - '_pantry_refutation'" in sql
    assert "_plan_modified_at" not in sql, "corre DENTRO de la corrida: sellar disparaba el CAS del propio worker"
    assert re.search(r"WHERE id = %s AND user_id = %s AND plan_data \? '_pantry_refutation'\s*$", sql.strip())
    assert params == ("p-1", "u-1")
    assert "_pantry_refutation" not in plan_data, "la copia en memoria del worker tampoco la conserva"
    assert nr.olvidar("p-1", "u-1", plan_data) is False and len(nr._escrituras) == 1, "sin evidencia no escribe"


def test_defecto2_el_worker_olvida_la_evidencia_cuando_la_guarda_aprueba_sin_waiver():
    w = _cuerpo_worker()
    rama = w.index("if _val_result is True:")
    olvido = w.index('__import__("nevera_refutada").olvidar(meal_plan_id, user_id, prior_plan_data)', rama)
    assert olvido < w.index("_qty_mode = (", rama), "se olvida en cuanto la EXISTENCIA pasa, antes de las cantidades"
    assert w.rindex("if _pantry_snapshot and _exist_waiver:", 0, rama) < rama, "sólo tras descartar el waiver"


def test_defecto3_otro_generador_invalida_la_evidencia(nr, monkeypatch):
    ev = _evidencia(REFUTADAS_S4)
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") == ev
    monkeypatch.setattr(nr, "_generador_actual", lambda user_id: "openai|gpt-6-luna|gpt-6-luna")
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None
    monkeypatch.setattr(nr, "_generador_actual", lambda user_id: None)
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None, \
        "sin poder saber qué generador correría, no se predice"


def test_defecto3_la_epoca_del_knob_invalida_toda_evidencia(nr, monkeypatch):
    ev = _evidencia(REFUTADAS_S4)
    monkeypatch.setenv("MEALFIT_PANTRY_REFUTED_PRECHECK_EPOCH", "2")
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None


def test_defecto3_evidencia_sin_generador_no_vale(nr):
    ev = dict(_evidencia(REFUTADAS_S4))
    ev.pop("gen")
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None


def test_defecto3_la_caducidad_por_defecto_es_una_semana(nr):
    assert nr.max_edad_h() == 168
    seis = _evidencia(REFUTADAS_S4, horas=24 * 6)
    ocho = _evidencia(REFUTADAS_S4, horas=24 * 8)
    assert nr.evidencia_vigente({"_pantry_refutation": seis}, _nevera_neta(INVENTARIO), "u-1", "DO") == seis
    assert nr.evidencia_vigente({"_pantry_refutation": ocho}, _nevera_neta(INVENTARIO), "u-1", "DO") is None


def test_defecto3_el_generador_real_se_resuelve_sin_lanzar(monkeypatch):
    """Sin sustituir la resolución: la huella real del generador (proveedor | modelo del usuario | modelo y esfuerzo del
    day-gen | tier) sale como texto; si algo de la resolución falla, None (y entonces ni se registra ni se predice).
    El tier sí se sustituye en los DOS módulos que lo leen, y ninguna consulta llega a la DB."""
    import nevera_refutada as m
    import llm_provider
    import graph_orchestrator
    import db
    consultas = []
    # [ronda 2 · punto 5] Ninguna consulta REAL del tier (con `.env` iría a producción).
    monkeypatch.setattr(db, "get_user_plan_tier", lambda user_id: consultas.append(user_id) or "gratis")
    monkeypatch.setattr(llm_provider, "get_user_tier", lambda user_id: "gratis")
    monkeypatch.setattr(graph_orchestrator, "get_user_tier", lambda user_id: "gratis")  # su propia copia (import por nombre)
    g = m._generador_actual("u-747-sin-db")
    assert consultas == [], "el test consultó el tier real: `graph_orchestrator` tiene su propia copia de get_user_tier"
    assert isinstance(g, str) and g.count("|") == 4 and llm_provider.llm_provider_name() in g
    assert g.endswith("|gratis")
    monkeypatch.setattr(llm_provider, "llm_provider_name", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert m._generador_actual("u-747-sin-db") is None
    assert consultas == []


@pytest.mark.parametrize("estado", ["plan_missing", "chunk_terminal", "chunk_unknown"])
def test_defecto5_sin_pausa_ni_push_si_el_bloque_ya_no_esta_vivo(nr, worker, monkeypatch, estado):
    """El pre-chequeo corre antes de `_validate_chunk_pre_llm`: un bloque cancelado o un plan borrado entre la recogida
    y el pre-chequeo no se pausa ni avisa — se devuelve al bucle, cuya 1.ª vuelta limpia (cancelar/liberar)."""
    import cron_tasks
    pausas, pushes = worker
    monkeypatch.setattr(cron_tasks, "_validate_chunk_pre_llm", lambda task_id, meal_plan_id, user_id: estado)
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)}) is False
    assert pausas == [] and pushes == [] and nr._metricas == []


def test_defecto5_sin_push_si_la_pausa_fue_desplazada(nr, worker, monkeypatch):
    import cron_tasks
    _, pushes = worker
    llamadas = []
    monkeypatch.setattr(cron_tasks, "_pause_chunk_for_pantry_refresh",
                        lambda *a, **kw: llamadas.append(kw) or False)
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)}) is True, "el bloque ya no es nuestro"
    assert len(llamadas) == 1 and pushes == [], "C1-PAUSE-CAS: desplazado ⇒ ni pausa ni push"


def test_defecto5_la_pausa_dice_si_ocurrio(monkeypatch):
    import cron_tasks
    monkeypatch.setattr(cron_tasks, "execute_sql_query", lambda *a, **kw: {"pipeline_snapshot": {}})
    monkeypatch.setattr(cron_tasks, "_dispatch_pantry_nudge", lambda user_id: True)
    monkeypatch.setattr(cron_tasks, "_cas_pause_chunk_to_pending_user_action", lambda *a: False)
    assert cron_tasks._pause_chunk_for_pantry_refresh("t", "u", 3, [], reason="x") is False
    monkeypatch.setattr(cron_tasks, "_cas_pause_chunk_to_pending_user_action", lambda *a: True)
    assert cron_tasks._pause_chunk_for_pantry_refresh("t", "u", 3, [], reason="x") is True


def test_defecto6_la_huella_ni_se_corta_ni_depende_del_orden(nr):
    filas = [{"ingredient_name": f"Alimento {i:03d}", "quantity": 1, "unit": "g"} for i in range(450)]
    h = nr.huella_bruta(filas)
    assert len(h) == 450, "cortar la huella deja fuera alimentos (y con ellos las compras de esos alimentos)"
    assert nr.huella_bruta(list(reversed(filas))) == h
    src = _MOD.read_text(encoding="utf-8")
    assert "ORDER BY" in src[src.index("def _leer_bruto"):src.index("def _escribir")]


def test_defecto7_el_push_tiene_una_sola_copia():
    import push_i18n
    import nevera_refutada as m
    assert "Tu plan necesita revisión de ingredientes" not in _CRON, "el texto vive en UN sitio (nevera_refutada)"
    assert m.PUSH_TITULO in push_i18n.push_catalog_keys() and m.PUSH_CUERPO in push_i18n.push_catalog_keys()
    w = _cuerpo_worker()
    pausa = w.index('reason="pantry_violation_after_retries",')
    assert '__import__("nevera_refutada").avisar(user_id)' in w[pausa:pausa + 400]


def test_defecto8_el_prechequeo_no_escribe_una_correccion_que_nadie_lee(nr, worker):
    fd = {"current_pantry_ingredients": _nevera_neta(INVENTARIO), "_fresh_pantry_source": "live"}
    assert nr.prechequeo(task_id="t-1", user_id="u-1", meal_plan_id="p-1", week_number=7, chunk_kind="rolling_refill",
                         snap={}, form_data=fd, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)},
                         country="DO") is True
    assert "_pantry_correction" not in fd


def test_defecto9_la_pausa_del_prechequeo_lleva_su_marca(nr, worker, monkeypatch):
    import cron_tasks
    llamadas = []
    monkeypatch.setattr(cron_tasks, "_pause_chunk_for_pantry_refresh", lambda *a, **kw: llamadas.append(kw))
    assert _pre(nr, plan_data={"_pantry_refutation": _evidencia(REFUTADAS_S4)}) is True
    assert llamadas[0].get("precheck") is True and llamadas[0].get("reason") == "pantry_violation_after_retries"


def test_defecto9_el_helper_sella_y_limpia_la_marca(monkeypatch):
    """La marca sólo describe la pausa VIVA del pre-chequeo: una pausa posterior por la vía normal la quita y
    resolver la pausa la borra (`_PANTRY_PAUSE_LIVE_KEYS`)."""
    import cron_tasks
    escritos = []
    monkeypatch.setattr(cron_tasks, "execute_sql_query",
                        lambda *a, **kw: {"pipeline_snapshot": {"_pantry_pause_precheck_at": "vieja"}})
    monkeypatch.setattr(cron_tasks, "_dispatch_pantry_nudge", lambda user_id: True)
    monkeypatch.setattr(cron_tasks, "_cas_pause_chunk_to_pending_user_action",
                        lambda task_id, snap_json, tag: escritos.append(json.loads(snap_json)) or True)
    cron_tasks._pause_chunk_for_pantry_refresh("t", "u", 3, [], reason="pantry_violation_after_retries", precheck=True)
    cron_tasks._pause_chunk_for_pantry_refresh("t", "u", 3, [], reason="pantry_violation_after_retries")
    assert escritos[0]["_pantry_pause_precheck_at"] != "vieja"
    assert "_pantry_pause_precheck_at" not in escritos[1]
    assert "_pantry_pause_precheck_at" in cron_tasks._PANTRY_PAUSE_LIVE_KEYS


# ─────────────── ronda 2 de la revisión (re-verificación) ───────────────
# Cada test reproduce un punto de la re-verificación (rojo contra 5e1f82c0) y fija su arreglo.

def _hash12(texto: str) -> str:
    import hashlib
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]


def _filas_res(inv, reservado=None):
    """Filas de `_leer_bruto` con su `reserved_quantity` (lo que la Nevera neta descuenta)."""
    reservado = reservado or {}
    return [{"ingredient_name": n, "quantity": q, "unit": u, "reserved_quantity": float(reservado.get(n, 0))}
            for n, q, u in inv]


def _inicio_de(m, inv=INVENTARIO, reservado=None):
    """Lo que `huella_inicial` devuelve para esa Nevera (bruta + lo disponible)."""
    filas = _filas_res(inv, reservado)
    return {"bruto": m.huella_bruta(filas), "disponible": m.huella_disponible(filas)}


def _sql_de(nombre_fn: str) -> str:
    """El SQL que una función del módulo pasa a `_escribir` (sin docstring ni logs)."""
    src = _MOD.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == nombre_fn)
    llamada = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_escribir")
    return "".join(c.value for c in ast.walk(llamada.args[0]) if isinstance(c, ast.Constant) and isinstance(c.value, str))


def test_r2_punto1_registrar_y_olvidar_no_sellan_plan_modified_at():
    """`_plan_modified_at` elige el plan ACTIVO (`GREATEST(created_at, _plan_modified_at)`: restore, rename, Historial,
    coach) y es el CAS del worker. `registrar` corre tras minutos de LLM sin re-comprobar que el bloque siga vivo: si el
    usuario creó o restauró otro plan, sellar el viejo lo promovía a activo (el fallo de P2-HIST-RENAME-NO-PROMOTE). Y
    `olvidar` corre DENTRO de la misma corrida: su sello disparaba el CAS del propio worker (modo degradado ⇒ merge
    abortado y bloque re-encolado). Son contabilidad interna: perder una escritura sólo hace que corra el LLM."""
    for fn in ("registrar", "olvidar"):
        sql = _sql_de(fn)
        assert "UPDATE meal_plans" in sql, fn
        assert "_plan_modified_at" not in sql, f"{fn}: la evidencia no es una edición del plan; no lo promueve a activo"
        assert "AND user_id = %s" in sql, fn


def test_r2_punto1_las_escrituras_reales_no_llevan_el_sello(nr):
    nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),), inicio=_inicio_de(nr))
    nr.olvidar("p-1", "u-1", {"_pantry_refutation": _evidencia(REFUTADAS_S4)})
    assert len(nr._escrituras) == 2
    for sql, _params in nr._escrituras:
        assert "_plan_modified_at" not in sql


def test_r2_punto2_otro_tier_es_otro_generador(monkeypatch):
    """gratis y plus resuelven el MISMO modelo con los defaults, pero el day-gen de plus corre con esfuerzo `medium` (y
    otro revisor clínico): quien se pasa a plus no arrastra una semana la evidencia del generador barato."""
    import nevera_refutada as m
    import llm_provider
    import graph_orchestrator
    monkeypatch.setattr(graph_orchestrator, "_openai_key_available", lambda: True)
    huellas = {}
    for tier in ("gratis", "plus"):
        monkeypatch.setattr(llm_provider, "get_user_tier", lambda user_id, _t=tier: _t)
        monkeypatch.setattr(graph_orchestrator, "get_user_tier", lambda user_id, _t=tier: _t)
        huellas[tier] = m.huella_generador("u-747-tier")
    assert huellas["gratis"] and huellas["plus"]
    assert huellas["gratis"] != huellas["plus"], "otro tier (otro esfuerzo del day-gen) ⇒ se vuelve a probar de verdad"


def test_r2_punto3_la_huella_del_generador_no_saca_nombres_de_proveedor(nr):
    """`plan_data` llega al cliente (y a su localStorage): la evidencia guarda un hash corto, no `deepseek|glm-…`."""
    assert nr.huella_generador("u-1") == _hash12("1|" + MODELOS)
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),),
                        inicio=_inicio_de(nr)) is True
    crudo = nr._escrituras[-1][1][0]
    ev = json.loads(crudo)
    assert re.fullmatch(r"[0-9a-f]{12}", ev["gen"])
    for nombre in ("zai", "glm", "gpt", "luna", "deepseek", "openai"):
        assert nombre not in crudo.lower(), nombre


def test_r2_punto4_un_alimento_liberado_de_reservas_vuelve_a_probar(nr, worker, monkeypatch):
    """Al refutar, el mango estaba reservado entero (el LLM no lo vio y puso guineo). Hoy se liberó y vuelve a estar
    disponible: la Nevera BRUTA no creció, lo refutado sigue fuera, pero el LLM tiene hoy una fruta que entonces no tenía
    ⇒ se prueba de verdad en vez de 12 h de pausa."""
    pausas, _ = worker
    ev = dict(_evidencia(REFUTADAS_S4))
    ev["disponible"] = _inicio_de(nr, reservado={"Mango": 4})["disponible"]
    assert "mango|unidad" not in ev["disponible"]
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: _filas_res(INVENTARIO))
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None
    assert _pre(nr, plan_data={"_pantry_refutation": ev}) is False and pausas == []


def test_r2_punto4_lo_que_hoy_esta_reservado_no_cuenta_como_novedad(nr, worker, monkeypatch):
    pausas, _ = worker
    ev = dict(_evidencia(REFUTADAS_S4))
    ev["disponible"] = _inicio_de(nr)["disponible"]
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: _filas_res(INVENTARIO, {"Mango": 4, "Yogurt": 1960}))
    fd = {"current_pantry_ingredients": _nevera_neta(INVENTARIO, {"Mango": 4, "Yogurt": 1960})}
    assert _pre(nr, plan_data={"_pantry_refutation": ev}, form_data=fd) is True and len(pausas) == 1


def test_r2_punto4_registrar_guarda_lo_disponible_en_las_dos_lecturas(nr, monkeypatch):
    """La base de lo disponible es la de ANTES ∩ la de DESPUÉS (igual que la bruta): lo que se liberó a mitad de corrida
    no lo vieron todos los intentos."""
    inicio = _inicio_de(nr, reservado={"Mango": 4})
    monkeypatch.setattr(nr, "_leer_bruto", lambda user_id: _filas_res(INVENTARIO, {"Piña": 1}))
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),), inicio=inicio) is True
    ev = json.loads(nr._escrituras[-1][1][0])
    assert "mango|unidad" not in ev["disponible"] and "pina|unidad" not in ev["disponible"]
    assert "tilapia|unidad" in ev["disponible"]
    assert ev["v"] == 3


def test_r2_punto4_evidencia_sin_lo_disponible_no_vale(nr):
    ev = dict(_evidencia(REFUTADAS_S4))
    ev.pop("disponible", None)
    assert nr.evidencia_vigente({"_pantry_refutation": ev}, _nevera_neta(INVENTARIO), "u-1", "DO") is None


def test_r2_punto4_la_lectura_trae_lo_reservado():
    src = _MOD.read_text(encoding="utf-8")
    lectura = src[src.index("def _leer_bruto"):src.index("def _escribir")]
    assert "reserved_quantity" in lectura and "kind = 'food'" in lectura


def test_r2_punto6_el_push_del_worker_solo_si_la_pausa_ocurrio():
    """C1-PAUSE-CAS: `_pause_chunk_for_pantry_refresh` devuelve False si el bloque ya no es nuestro (desplazado o
    cancelado por un plan nuevo). El worker avisaba igual: push sobre un plan que el usuario acababa de reemplazar."""
    worker_fn = next(n for n in ast.walk(ast.parse(_CRON)) if isinstance(n, ast.FunctionDef) and n.name == "_chunk_worker")
    ramas = [n for n in ast.walk(worker_fn) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
             and getattr(n.test.func, "id", "") == "_pause_chunk_for_pantry_refresh"
             and any(k.arg == "reason" and getattr(k.value, "value", None) == "pantry_violation_after_retries"
                     for k in n.test.keywords)]
    assert len(ramas) == 1, "la pausa tras agotar los reintentos debe condicionar el push a que ocurrió"
    cuerpo = "\n".join(ast.get_source_segment(_CRON, st) or "" for st in ramas[0].body)
    assert '__import__("nevera_refutada").avisar(user_id)' in cuerpo
    assert _cuerpo_worker().count('__import__("nevera_refutada").avisar(user_id)') == 1
