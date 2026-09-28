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
     agotar sus reintentos) con sus líneas INEXISTENTES y la huella BRUTA de la Nevera de ese momento.
  3. La Nevera no creció desde entonces (huella bruta: ni alimento nuevo ni más cantidad) — las reservas no cuentan:
     suben y bajan solas y no son una compra.
  4. La MISMA medida de la guarda (`validate_ingredients_against_pantry` + `compras_pequenas.tolerar`) sigue
     rechazando esas líneas contra la Nevera de hoy.
  ⇒ entra directo por donde el sistema acababa: la pausa `pantry_violation_after_retries` (mismo helper, mismo push);
     el recovery de siempre decide lo demás (TTL → flexible, tope de ciclos).
La evidencia caduca (`MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H`, 336 h): cada dos semanas se vuelve a probar de verdad.
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
    monkeypatch.setattr(m, "_leer_bruto", lambda user_id: _filas(INVENTARIO))
    monkeypatch.setattr(m, "_escribir", lambda sql, params: escrituras.append((sql, params)))
    monkeypatch.setattr(m, "_metrica", lambda **kw: metricas.append(kw))
    m._escrituras, m._metricas = escrituras, metricas
    return m


@pytest.fixture
def worker(monkeypatch):
    """Los dos efectos del camino de siempre, capturados (jamás tocan la DB ni mandan pushes)."""
    import cron_tasks
    pausas, pushes = [], []
    monkeypatch.setattr(cron_tasks, "_pause_chunk_for_pantry_refresh",
                        lambda task_id, user_id, week_number, fresh_inventory, reason="empty_pantry", notify=True:
                        pausas.append({"task_id": task_id, "week": week_number, "reason": reason,
                                       "n": len(fresh_inventory or [])}))
    monkeypatch.setattr(cron_tasks, "_dispatch_push_notification", lambda **kw: pushes.append(kw))
    return pausas, pushes


def _evidencia(lineas, inv=INVENTARIO, horas=10):
    at = (datetime.now(timezone.utc) - timedelta(hours=horas)).isoformat()
    import nevera_refutada as m
    return {"v": 1, "at": at, "week": 4, "chunk_kind": "rolling_refill", "lines": list(lineas),
            "bruto": m.huella_bruta(_filas(inv))}


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
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),)) is False
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
                        (_resultado_guarda(REFUTADAS_S4[:3]), _resultado_guarda(REFUTADAS_S4[3:]))) is True
    (sql, params), = nr._escrituras
    assert "jsonb_set" in sql and "'{_pantry_refutation}'" in sql and "'{_plan_modified_at}'" in sql
    assert re.search(r"WHERE id = %s AND user_id = %s\s*$", sql.strip())
    ev = json.loads(params[0])
    assert params[1:] == ("p-1", "u-1")
    assert ev["lines"] == REFUTADAS_S4 and ev["week"] == 4 and ev["chunk_kind"] == "rolling_refill"
    assert ev["bruto"] == nr.huella_bruta(_filas(INVENTARIO))


def test_sin_lineas_inexistentes_no_hay_evidencia(nr):
    assert nr.registrar("p-1", "u-1", 4, "rolling_refill", ("CANTIDADES: excede", True, None)) is False
    assert nr._escrituras == []


def test_el_ciclo_completo_registrar_y_despues_prechequear(nr, worker):
    pausas, _ = worker
    nr.registrar("p-1", "u-1", 4, "rolling_refill", (_resultado_guarda(REFUTADAS_S4),))
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
    assert "MEALFIT_PANTRY_REFUTED_PRECHECK" in _KNOBS_REGISTRY
    assert "MEALFIT_PANTRY_REFUTED_PRECHECK_MAX_AGE_H" in _KNOBS_REGISTRY
    assert _CRON.count("P1-PLAN-LOTE-747") >= 2
    assert "tooltip-anchor: P1-PLAN-LOTE-747" in _MOD.read_text(encoding="utf-8")
