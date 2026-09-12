# -*- coding: utf-8 -*-
"""[P0-CULINARY-BASELINE · 2026-09-06] La línea base de coherencia culinaria, ANTES de mejorarla.

Congelar la foto es el primer paso del benchmark, y no es burocracia: **una medición posterior al
efecto no mide el efecto**. Este repo ya lo pagó una vez —«recontar mañana» no era un entregable— y
por eso la línea base se toma, se escribe con fecha y se commitea antes de tocar nada.

    python scripts/culinary_baseline.py                 # la foto de hoy
    python scripts/culinary_baseline.py --json          # para diffear contra la congelada
    python scripts/culinary_baseline.py --congelar      # reescribe docs/culinary_baseline.json

## Qué mide, y por qué las dos capas por separado

- **Contrato determinista** (`culinary_contract_scan`, V1–V5): reglas que no opinan. V1 técnica
  imposible, V2 estado imposible, V3 ingrediente que ningún paso usa, V4 gramos que no cuadran,
  V5 paso que usa lo que la lista no trae.
- **Juez culinario** (LLM, `_culinary_judge_history`): incoherencia de prosa — pasos que se
  contradicen, nombres que prometen lo que el plato no tiene, combinaciones absurdas.

Se publican **separadas y con su solapamiento**, nunca sumadas en un índice único. La medición del
2026-09-06 explica por qué: de las 314 comidas que marcan entre las dos, **solo 23 coinciden**.
Fundirlas en un número escondería que ninguna sustituye a la otra.

## La foto del 2026-09-06 (96 planes · 1.186 comidas)

    contrato determinista ... 133 comidas   11,2 %   V1=80 V3=55 V4=34 V5=11 V2=4
    juez culinario .......... 227 comidas   19,1 %   paso_incoherente=96
                                                     nombre_no_corresponde=53
                                                     combo_absurdo=50
                                                     slot_inapropiado=28
                                                     tecnica_impropia=23
    coinciden ...............  23 comidas
    solo determinista ....... 110   ·   solo juez ... 204

**Cuidado al leer esto como «calidad».** El juez es un LLM sin verdad de referencia: no se conoce su
precisión ni su recall. Que señale el 19,1 % no significa que el 19,1 % de las comidas esté mal;
significa que él lo cree. Convertir esa tasa en una nota, o calibrar un umbral contra ella, es el
overfitting que este repo ya pagó en agosto. Lo que falta es el golden set humano
(`scripts/culinary_golden_sample.py`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

CONGELADA = _BACKEND / "docs" / "culinary_baseline.json"


def medir(planes: int = 120) -> dict:
    """La foto de la VENTANA VIVA (`ORDER BY created_at DESC LIMIT n` sobre el `plan_data` de hoy).

    [P1-PLAN-LOTE-18 · 2026-09-12] (C0) Sirve para mirar la flota de hoy; NO para congelar una línea base: la
    ventana viva no se puede volver a medir mañana (el shift encoge los días, la purga se lleva planes). Para eso
    está `medir_corpus` sobre un corpus FIJO (`scripts/congela_corpus_culinario.py`).
    """
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        cat = [dict(r) for r in c.execute(
            "SELECT name, aliases, category, ready_to_eat, prep_methods "
            "FROM master_ingredients").fetchall()]
        filas = c.execute(
            "SELECT id, plan_data FROM meal_plans WHERE plan_data->'days' IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s", (planes,)).fetchall()
    return _medir_filas(filas, cat)


def medir_corpus(corpus: dict, fichero: "str | None" = None) -> dict:
    """[P1-PLAN-LOTE-18 · 2026-09-12] (C0) La misma medición sobre un corpus FIJO: filas y catálogo salen del
    fichero congelado, la huella publicada es la del corpus (contenido + catálogo) y `corpus.fijo = True`.
    Medirlo dos veces da las mismas cifras por construcción; si difieren, cambió el código
    (`computation.reglas_huella`), no el corpus. tooltip-anchor: P1-PLAN-LOTE-18-CORPUS-FIJO"""
    from culinary_corpus import filas_para_medir

    r = _medir_filas(filas_para_medir(corpus), corpus.get("catalogo_filas") or [])
    r["corpus"]["huella_ventana_viva"] = r["corpus"]["huella"]
    r["corpus"]["huella"] = corpus["huella"]
    r["corpus"]["fijo"] = True
    r["corpus"]["fichero"] = fichero
    r["corpus"]["congelado_at"] = corpus.get("congelado_at")
    r["corpus"]["nota"] = ("Corpus FIJO (fichero congelado con huella de contenido + catálogo): dos mediciones con "
                           "la misma huella miden lo MISMO; una diferencia de cifras es del código, no del cron.")
    return r


def _medir_filas(filas: list, cat: list) -> dict:
    """El medidor, común a la ventana viva y al corpus fijo: `filas` son `{"id", "plan_data"}`."""
    from culinary_coherence import (culinary_contract_scan_status, judgment_covers_delivered,
                                    resolve_judge_violations, judge_evaluation_state)
    from culinary_corpus import huella_catalogo, huella_reglas

    comidas = 0
    # [P1-BASELINE-REPRODUCIBLE - 2026-09-07] La foto anterior guardaba `planes: 96` y
    # `comidas: 1186` y NADA mas. No bastaba: el corpus es `ORDER BY created_at DESC LIMIT n`
    # sobre `plan_data` VIVO, y el shift del cron ENCOGE la ventana de dias de un plan ya
    # existente. Medido: 14 h despues de congelarla, los MISMOS 96 planes daban 1.182 comidas
    # -- dos planes mutaron a las 00:00 y 00:30 y se llevaron 4 comidas por delante.
    #
    # O sea que una re-medicion no comparaba "antes vs despues del cambio": comparaba dos
    # corpus distintos, y cualquier mejora o empeoramiento incluia esa deriva sin decirlo.
    # Se graba la HUELLA del corpus para que la proxima corrida pueda afirmar si mide lo mismo.
    huella_corpus = hashlib.sha256()
    ids = []
    por_check, por_tipo = collections.Counter(), collections.Counter()
    con_det, con_juez = set(), set()
    cobertura = collections.Counter()   # P1-JUDGE-REVISION-STAMP: si / no / desconocido
    # [P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) Lo que antes se colapsaba en «lista vacía»: el estado del scan por
    # plan, el estado del juez sobre la versión ENTREGADA, y cada hallazgo atado (o no) a una comida que existe.
    # tooltip-anchor: P1-PLAN-LOTE-22-RECONCILIA
    estado_scan, estado_juez = collections.Counter(), collections.Counter()
    hall_det = collections.Counter()      # con_comida / sin_comida
    hall_juez = collections.Counter()     # vigentes/obsoletos/desconocidos/no_disponibles + con_comida/ambiguos/sin_comida
    con_juez_vigente, por_tipo_vigente = set(), collections.Counter()
    ocurrencias = set()                   # (pid, day, meal_index) de cada comida que existe
    det_occ = set()                       # (pid, day, meal_index) marcadas por el determinista

    for f in filas:
        pid = str(f["id"])
        pd = f["plan_data"] or {}
        ids.append(pid[:8])
        _dias = pd.get("days") or []
        for _pos, d in enumerate(_dias, 1):
            comidas += len(d.get("meals") or [])
            for _mi in range(len(d.get("meals") or [])):
                ocurrencias.add((pid, d.get("day") if d.get("day") is not None else _pos, _mi))
        # el nº de dias es justo lo que el shift mueve: entra en la huella
        huella_corpus.update(f"{pid}:{len(_dias)}:".encode())
        _viols, _est = culinary_contract_scan_status(pd, cat)
        estado_scan[_est["status"]] += 1
        for v in _viols:
            por_check[str(v.get("check"))] += 1
            con_det.add((pid, v.get("day"), str(v.get("meal"))))
            _occ = (pid, v.get("day"), v.get("meal_index"))
            hall_det["con_comida" if _occ in ocurrencias else "sin_comida"] += 1
            if _occ in ocurrencias:
                det_occ.add(_occ)
        estado_juez[judge_evaluation_state(pd)["estado"]] += 1
        for h in (pd.get("_culinary_judge_history") or []):
            if not isinstance(h, dict):
                continue
            # [C1] ¿esta entrada habla de lo entregado? y ¿cada queja suya tiene UNA comida a la que atarse?
            _cubre = judgment_covers_delivered(h, pd)
            hall_juez["no_disponibles" if h.get("status") == "unavailable" else
                      "vigentes" if _cubre else ("obsoletos" if _cubre is False else "desconocidos")] += 1
            _orig = [x for x in (h.get("violations") or []) if isinstance(x, dict)]
            for x, v in zip(_orig, resolve_judge_violations(pd, _orig)):
                # [C1] tres formas de estar atada: por SELLO (la comida juzgada sigue entregada, quizá en otro día tras
                # el shift), por índice/franja con la entrada vigente, o por índice/franja de una entrada obsoleta
                # (atada a una comida que existe, pero que ya no es la juzgada: cuenta como hallazgo, no como vigente).
                if v.get("resolucion") == "por_sello":
                    _occ = (pid,) + tuple(v["ocurrencia_actual"])
                    hall_juez["con_comida"] += 1
                    hall_juez["vigentes_por_sello"] += 1
                    _vig = True
                elif v.get("meal_index") is None:
                    hall_juez["ambiguos" if v.get("resolucion") == "ambigua" else "sin_comida"] += 1
                    continue
                else:
                    _occ = (pid, v.get("day"), v.get("meal_index"))
                    hall_juez["con_comida"] += 1
                    # sellada al juzgar (`x`) y NO reencontrada por sello ⇒ la comida juzgada ya no se entrega así;
                    # sin sello (entradas anteriores a C1) ⇒ manda el sello del plan de la entrada
                    _vig = False if x.get("meal_seal") else (_cubre is True)
                if _vig:
                    con_juez_vigente.add(_occ)
                    por_tipo_vigente[str(v.get("tipo"))] += 1
            # [P1-JUDGE-REVISION-STAMP · 2026-09-06] ¿Esta entrada juzgó lo que se ENTREGÓ, o una
            # versión que el pipeline reparó después? Tres estados, y el tercero manda: sin sello
            # (todo lo generado antes del P-fix) la pregunta no es decidible, y colapsarlo hacia
            # cualquier lado fabricaría una cifra. Se cuenta aparte y se publica aparte.
            cubre = judgment_covers_delivered(h, pd)
            cobertura["si" if cubre else ("no" if cubre is False else "desconocido")] += 1
            for v in (h.get("violations") or []):
                if isinstance(v, dict):
                    por_tipo[str(v.get("tipo"))] += 1
                    con_juez.add((pid, v.get("day"), str(v.get("meal"))))

    def pct(n):
        return round(100.0 * n / comidas, 1) if comidas else None

    return {
        "planes": len(filas), "comidas": comidas,
        "corpus": {
            "huella": huella_corpus.hexdigest()[:16],
            "fijo": False,
            "plan_ids": sorted(ids),
            "nota": ("`plan_data` es VIVO: el shift encoge los dias de un plan ya existente. "
                     "Si la huella cambia, las dos fotos NO son comparables aunque coincidan "
                     "los ids."),
        },
        # [P1-PLAN-LOTE-18 · 2026-09-12] (C0) Qué código produjo las cifras. Con el corpus fijo, dos mediciones con
        # la misma `corpus.huella` y distinta `reglas_huella` miden el efecto del código — y sólo entonces.
        "computation": {"reglas_huella": huella_reglas(), "catalogo_huella": huella_catalogo(cat),
                        "catalogo_filas": len(cat or [])},
        "determinista": {"comidas": len(con_det), "pct": pct(len(con_det)),
                         "por_check": dict(por_check.most_common())},
        "juez": {"comidas": len(con_juez), "pct": pct(len(con_juez)),
                 "por_tipo": dict(por_tipo.most_common())},
        # [P1-JUDGE-REVISION-STAMP] Cuantas entradas del juez se sabe que juzgaron lo
        # ENTREGADO. Va al lado de la tasa, no dentro: no la corrige, la CALIFICA.
        #
        # [P1-MEASUREMENT-INTEGRITY - 2ª iter] `decidible` se CALCULA, no se afirma a mano. La
        # version anterior la escribi directamente en el JSON... y el primer `--congelar` se la
        # llevo por delante, junto con las cifras de la medicion. Es exactamente la trampa que
        # habia diagnosticado una hora antes y solo cerre a medias: movi la `advertencia` al
        # codigo y deje su bloque hermano en el fichero. Un dato corregido a mano en algo que un
        # guion regenera es una correccion con fecha de caducidad, y esta vez la pague yo.
        "juez_sobre_lo_entregado": {
            "conteo": dict(cobertura),
            "decidible": cobertura.get("desconocido", 0) == 0,
            "motivo": (None if cobertura.get("desconocido", 0) == 0 else
                       "hay entradas de `_culinary_judge_history` sin `judged_fingerprint` "
                       "(anteriores a P1-JUDGE-REVISION-STAMP): de ellas NO se puede saber si "
                       "juzgaron lo entregado"),
            "medicion_2026_09_06": MEDICION_QUEJAS_OBSOLETAS,
        },
        "solapamiento": {"ambas": len(con_det & con_juez),
                         "solo_determinista": len(con_det - con_juez),
                         "solo_juez": len(con_juez - con_det)},
        # [P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-01) Estado explícito de evaluación y reconciliación con el
        # denominador. `juez` (arriba) sigue siendo «cuántas comidas señaló el juez en algún momento» (histórico);
        # `juez_entregado` son SOLO las quejas de entradas que juzgaron la versión entregada, atadas a una comida
        # que existe. La partición de las comidas ENTREGADAS suma exactamente `comidas` — si no, `reconcilia` es False
        # y hay un defecto en el medidor, no en los planes.
        "estado_evaluacion": {"contrato": dict(estado_scan), "juez": dict(estado_juez)},
        "hallazgos": {"determinista": dict(hall_det), "juez": dict(hall_juez)},
        "juez_entregado": {"comidas": len(con_juez_vigente), "pct": pct(len(con_juez_vigente)),
                           "por_tipo": dict(por_tipo_vigente.most_common())},
        "particion": _particion(ocurrencias, det_occ, con_juez_vigente, comidas),
        # Se guarda EXPLICITO para que nadie lo derive de las tasas y se engañe.
        # [P1-JUDGE-REVISION-STAMP · 2026-09-06] La segunda frase NO es decorativa y por eso vive
        # AQUÍ y no solo en el JSON congelado: `--congelar` reescribe el fichero entero, así que
        # una advertencia enmendada a mano en el JSON se habría borrado en silencio en el próximo
        # congelado. La advertencia tiene que nacer del mismo sitio que el dato.
        "advertencia": ADVERTENCIA,
    }


def _particion(ocurrencias: set, det_occ: set, con_juez_vigente: set, comidas: int) -> dict:
    """[P1-PLAN-LOTE-22] (C1) Cada comida ENTREGADA cae en UN cubo: ambas / solo_determinista / solo_juez_vigente /
    ninguna, por ocurrencia (pid, día, índice). La suma tiene que ser el denominador: `reconcilia` lo comprueba en vez
    de suponerlo. Las alertas históricas sin comida a la que atarse no entran aquí: van en `hallazgos`."""
    c = collections.Counter()
    for occ in ocurrencias:
        d, j = occ in det_occ, occ in con_juez_vigente
        c["ambas" if (d and j) else "solo_determinista" if d else "solo_juez_vigente" if j else "ninguna"] += 1
    total = sum(c.values())
    return {**{k: c.get(k, 0) for k in ("ambas", "solo_determinista", "solo_juez_vigente", "ninguna")},
            "total": total, "denominador": comidas, "reconcilia": total == comidas}


#: [P1-JUDGE-REVISION-STAMP · 2026-09-06] Dos razones para no leer la tasa del juez como calidad,
#: no una. La segunda se midió DESPUÉS de congelar la foto: parte de sus quejas describen un
#: estado que el pipeline REPARÓ antes de entregar.
#: [P1-JUDGE-REVISION-STAMP] La medicion puntual que respalda la segunda razon de la advertencia.
#: Vive en el codigo por lo mismo que ella: `--congelar` reescribe el JSON entero.
MEDICION_QUEJAS_OBSOLETAS = {
    "quejas_falta_en_la_lista_juzgables": 37,
    "nombraban_algo_que_hoy_SI_esta": 6,
    "queso_no_lonjeable_en_lonjas_en_planes_vivos": 0,
    "lineas_de_queso_con_lonja_revisadas": 23,
}

ADVERTENCIA = (
    "el juez es un LLM sin verdad de referencia: su tasa NO es la tasa de defectos reales. "
    "Calibrar contra ella es overfitting. Y hay una segunda razon, medida el 2026-09-06: parte de "
    "sus quejas describen un estado que el pipeline REPARO antes de entregar — de 37 quejas "
    "juzgables del tipo «X no aparece en la lista», 6 nombraban algo que SI esta en el plan "
    "entregado, y el sub-patron mas citado («lonjas/pedazos de queso» sobre queso cottage) "
    "aparece 0 veces de 23 en planes vivos. Las entradas SIN `judged_fingerprint` no permiten "
    "saber cuales juzgaron lo entregado: leelas como «cuantas comidas el juez senalo en algun "
    "momento», nunca como «cuantas se entregaron mal»."
)


def _corpus_comparable(r: dict, previa: dict | None) -> "bool | None":
    """¿La foto congelada mide el MISMO corpus que esta? `None` = no se puede saber.

    [P1-BASELINE-REPRODUCIBLE · 2026-09-07] Las fotos anteriores a este P-fix no llevan huella,
    y ahí la respuesta honesta es «no se sabe» — nunca «sí». Restar dos cifras de corpus distintos
    y presentar la diferencia como el efecto de un cambio es fabricar un resultado.
    """
    if not previa:
        return None
    a = ((r.get("corpus") or {}).get("huella"))
    b = ((previa.get("corpus") or {}).get("huella"))
    if not a or not b:
        return None
    return a == b


def render(r: dict, previa: dict | None = None) -> str:
    o = [f"planes {r['planes']} · comidas {r['comidas']}", ""]
    if (r.get("corpus") or {}).get("fijo"):
        o.insert(1, f"  corpus FIJO {r['corpus'].get('fichero')} · huella {r['corpus']['huella']} · "
                    f"reglas {(r.get('computation') or {}).get('reglas_huella')}")
    _cmp = _corpus_comparable(r, previa)
    if _cmp is False:
        o += ["  ⛔ EL CORPUS CAMBIO desde la foto congelada: los deltas de abajo NO son el efecto",
              "     de ningun cambio de codigo. `plan_data` es vivo y el shift encoge los dias de",
              "     un plan ya existente (medido: 4 comidas menos en 14 h, mismos 96 planes).", ""]
    elif _cmp is None and previa:
        o += ["  ⚠  La foto congelada no lleva huella de corpus (es anterior a",
              "     P1-BASELINE-REPRODUCIBLE): no se puede saber si mide lo mismo. Trata los",
              "     deltas como orientativos, no como el efecto de un cambio.", ""]
    for capa, etiq in (("determinista", "contrato determinista (V1-V7)"), ("juez", "juez culinario")):
        v = r[capa]
        linea = f"  {etiq:32s} {v['comidas']:5d} comidas   {v['pct']} %"
        if previa and previa.get(capa):
            d = v["comidas"] - previa[capa]["comidas"]
            linea += f"   ({d:+d} vs congelada)"
        o.append(linea)
        detalle = v.get("por_check") or v.get("por_tipo") or {}
        o.append("      " + "  ".join(f"{k}={n}" for k, n in detalle.items()))
    s = r["solapamiento"]
    o += ["", f"  coinciden {s['ambas']}   ·   solo determinista {s['solo_determinista']}   ·   "
              f"solo juez {s['solo_juez']}",
          "", "  Las dos capas apenas se solapan: ninguna sustituye a la otra, y por eso NO se",
          "  suman en un indice unico."]
    cob = (r.get("juez_sobre_lo_entregado") or {}).get("conteo") or {}
    if cob:
        o += ["", "  entradas del juez que juzgaron LO ENTREGADO: "
                  + "  ".join(f"{k}={n}" for k, n in cob.items()),
              "  («desconocido» = sin `judged_fingerprint`, anterior a P1-JUDGE-REVISION-STAMP; "
              "no cuenta a ningun lado)"]
    ev = r.get("estado_evaluacion") or {}
    if ev:
        o += ["", f"  estado de evaluacion · contrato {ev.get('contrato')} · juez (sobre lo entregado) {ev.get('juez')}"]
    je = r.get("juez_entregado") or {}
    if je:
        o.append(f"  juez SOBRE LO ENTREGADO: {je.get('comidas')} comidas   {je.get('pct')} %   "
                 + "  ".join(f"{k}={n}" for k, n in (je.get("por_tipo") or {}).items()))
    pa = r.get("particion") or {}
    if pa:
        o.append(f"  particion de las {pa.get('denominador')} comidas: ambas {pa.get('ambas')} · solo det {pa.get('solo_determinista')}"
                 f" · solo juez vigente {pa.get('solo_juez_vigente')} · ninguna {pa.get('ninguna')} · "
                 f"{'RECONCILIA' if pa.get('reconcilia') else 'NO RECONCILIA (defecto del medidor)'}")
    ha = r.get("hallazgos") or {}
    if ha:
        o.append(f"  hallazgos: determinista {ha.get('determinista')} · juez {ha.get('juez')}  "
                 "(«sin_comida»/«ambiguos» se informan aparte: no se reparten a nadie)")
    o += ["", "  " + r["advertencia"]]
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=int, default=120)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--congelar", action="store_true",
                    help="congela la linea base medida sobre --corpus en docs/culinary_baseline_<fecha>.json")
    ap.add_argument("--corpus", help="[C0] fichero de corpus FIJO (scripts/congela_corpus_culinario.py); "
                                     "sin el, se mide la ventana VIVA")
    ap.add_argument("--verificar", action="store_true",
                    help="[C0] re-mide --corpus y lo compara con su linea base congelada: exit 0 si reproduce, "
                         "3 si mismas reglas y cifras distintas, 4 si no hay linea base para esa huella")
    a = ap.parse_args()

    if a.corpus:
        from culinary_corpus import cargar
        corpus = cargar(a.corpus)
        r = medir_corpus(corpus, fichero=Path(a.corpus).as_posix())
        destino = baseline_path_for(corpus)
        previa = _leer(destino)
        if a.verificar:
            return verificar(r, previa, destino)
        if a.congelar:
            destino.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
            print(f"linea base sobre corpus FIJO congelada en {destino.relative_to(_BACKEND).as_posix()} "
                  f"(huella {r['corpus']['huella']}, reglas {r['computation']['reglas_huella']})")
            return 0
        print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r, previa))
        return 0

    if a.congelar or a.verificar:
        # [P1-PLAN-LOTE-18 · 2026-09-12] (C0) La ventana viva no se puede volver a medir mañana: la foto del 6-sep
        # dejo de ser reproducible en 14 h. Una linea base se congela SOLO sobre un corpus fijo.
        print("--congelar/--verificar exigen --corpus: una linea base sobre la ventana VIVA no es reproducible. "
              "Congela el corpus con scripts/congela_corpus_culinario.py y pasa el fichero.")
        return 2
    r = medir(a.planes)
    previa = _leer(CONGELADA)
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r, previa))
    return 0


def baseline_path_for(corpus: dict) -> Path:
    """`docs/culinary_baseline_<YYYY_MM_DD>.json`, con la fecha del congelado del corpus: una linea base por corpus.
    `docs/culinary_baseline.json` (la foto viva del 6/7-sep) se conserva tal cual: es historia, no comparable."""
    fecha = str(corpus.get("congelado_at") or "")[:10].replace("-", "_") or "sin_fecha"
    return _BACKEND / "docs" / f"culinary_baseline_{fecha}.json"


def _leer(p: Path) -> "dict | None":
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


#: Lo que tiene que reproducirse entre dos mediciones del mismo corpus fijo.
CIFRAS = ("planes", "comidas", "determinista", "juez", "solapamiento", "juez_sobre_lo_entregado",
          # [P1-PLAN-LOTE-22] (C1) también tienen que reproducirse: estados, hallazgos atados y la partición
          "estado_evaluacion", "hallazgos", "juez_entregado", "particion")


def verificar(r: dict, previa: "dict | None", destino: Path) -> int:
    """[P1-PLAN-LOTE-18] (C0) ¿La medicion de hoy sobre el corpus fijo reproduce la congelada? Tres salidas:
    0 reproduce (o el delta es del codigo, dicho); 3 mismas reglas y cifras distintas (el medidor no es
    determinista); 4 no hay linea base comparable para esta huella."""
    if not previa:
        print(f"sin linea base congelada para este corpus ({destino.name}): congela primero con --congelar")
        return 4
    if not _corpus_comparable(r, previa):
        print(f"la linea base {destino.name} es de OTRO corpus (huella "
              f"{(previa.get('corpus') or {}).get('huella')} != {r['corpus']['huella']}): no comparable")
        return 4
    iguales = all(r.get(k) == previa.get(k) for k in CIFRAS)
    reglas_hoy = (r.get("computation") or {}).get("reglas_huella")
    reglas_base = (previa.get("computation") or {}).get("reglas_huella")
    if iguales:
        print(f"REPRODUCIBLE: mismas cifras sobre el corpus {r['corpus']['huella']} (reglas {reglas_hoy})")
        return 0
    if reglas_hoy != reglas_base:
        print(f"CIFRAS DISTINTAS con reglas distintas ({reglas_base} -> {reglas_hoy}): el delta es del CODIGO, "
              f"no del corpus. Revisalo y, si es el esperado, re-congela con --congelar.")
        print(render(r, previa))
        return 0
    print(f"NO REPRODUCIBLE: mismo corpus ({r['corpus']['huella']}), mismas reglas ({reglas_hoy}) y cifras "
          f"distintas: el medidor no es determinista. Investigar antes de leer ningun delta.")
    print(render(r, previa))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
