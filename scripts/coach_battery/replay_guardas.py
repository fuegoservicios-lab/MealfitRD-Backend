# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1038 · A-15 · 2026-11-16] Replay de las guardas por patrones del coach sobre mensajes REALES (nivel N1).

Las guardas deterministas del coach (qué es una queja, qué es una pregunta y no un registro, «se me acabó X» no es un
gusto…) se escribieron mirando tres o cuatro frases; la de G-20 fallaba en 8 de 9 frases reales. Este script las pasa
por los mensajes reales de `agent_messages` y dice, por guarda, cuántos mensajes dispara. Gratis: no llama a ningún
modelo y solo LEE la base.

Uso (en el servidor, que es donde está la base):

    python scripts/coach_battery/replay_guardas.py                 # recuentos por guarda
    python scripts/coach_battery/replay_guardas.py --ver queja     # además, los mensajes que dispara esa guarda
    python scripts/coach_battery/replay_guardas.py --ver solo_pregunta --max 400

Los mensajes que imprime `--ver` van ANONIMIZADOS (correos, teléfonos e identificadores fuera) y recortados; son para
que una persona los etiquete (acierto / falso positivo). No se guardan en el repo ni en logs: el veredicto —recuentos y
fecha— es lo único que se versiona, en `replay_tasas.json`, y `tests/test_p1_plan_lote_1038_replay_guardas.py` lo fija.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

_CORREO = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TELEFONO = re.compile(r"(?<!\d)\+?\d[\d\s().-]{7,}\d(?!\d)")


def anonimizar(texto: str, largo: int = 180) -> str:
    t = _CORREO.sub("<correo>", str(texto or ""))
    t = _UUID.sub("<id>", t)
    t = _TELEFONO.sub("<tel>", t)
    return " ".join(t.split())[:largo]


def guardas() -> dict:
    """nombre -> predicado(texto) -> bool. Solo guardas que deciden por el TEXTO del usuario (funciones puras)."""
    from chat_hard_turn import _RE_HARD_INTENT
    from chat_prompt_opt import es_turno_objetivo_sin_sentimiento
    from coach_senales_calidad import es_queja
    from contenido_externo import motivo as orden_al_modelo
    from guarda_dos_comidas_un_mensaje import franja_de_la_foto
    from guarda_nevera_no_es_gusto import bloqueo_si_es_la_nevera
    from guarda_pregunta_no_es_registro import bloqueo_si_solo_pregunta
    return {
        "queja": lambda t: bool(es_queja(t)),
        "solo_pregunta": lambda t: bloqueo_si_solo_pregunta("log_consumed_meal", t) is not None,
        "nevera_no_es_gusto": lambda t: bloqueo_si_es_la_nevera("update_form_field", {"field": "dislikes"}, t) is not None,
        "franja_de_la_foto": lambda t: franja_de_la_foto(t) is not None,
        "intencion_dificil": lambda t: bool(_RE_HARD_INTENT.search(t or "")),
        "sin_sentimiento": lambda t: bool(es_turno_objetivo_sin_sentimiento(t)),
        "orden_al_modelo": lambda t: orden_al_modelo(t) is not None,
    }


def medir(textos: list, predicados: dict) -> dict:
    """{guarda: [índices de los mensajes que dispara]}. Un predicado que lanza cuenta como error, no como disparo."""
    disparos = {nombre: [] for nombre in predicados}
    errores = {nombre: 0 for nombre in predicados}
    for i, t in enumerate(textos):
        for nombre, pred in predicados.items():
            try:
                if pred(t):
                    disparos[nombre].append(i)
            except Exception:  # noqa: BLE001
                errores[nombre] += 1
    return {"disparos": disparos, "errores": errores}


def leer_mensajes(limite: int) -> list:
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"]) as c:
        cur = c.cursor()
        cur.execute(
            "SELECT content FROM public.agent_messages WHERE role = 'user' AND content IS NOT NULL AND btrim(content) <> '' "
            "ORDER BY created_at DESC LIMIT %s", (int(limite),))
        return [str(r[0]) for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2000, help="cuántos mensajes recientes repasar")
    ap.add_argument("--ver", default="", help="guarda cuyos disparos se imprimen (anonimizados) para etiquetar")
    ap.add_argument("--muestra", type=int, default=80, help="como mucho cuántos disparos imprimir con --ver")
    args = ap.parse_args()

    textos = leer_mensajes(args.max)
    predicados = guardas()
    r = medir(textos, predicados)
    print(json.dumps({"mensajes": len(textos),
                      "disparos": {k: len(v) for k, v in r["disparos"].items()},
                      "errores": {k: v for k, v in r["errores"].items() if v}}, ensure_ascii=False))
    if args.ver:
        if args.ver not in r["disparos"]:
            print(f"guarda desconocida: {args.ver} (hay: {', '.join(predicados)})", file=sys.stderr)
            return 2
        idx = r["disparos"][args.ver]
        paso = max(1, len(idx) // max(1, args.muestra))
        for i in idx[::paso][:args.muestra]:
            print(f"- {anonimizar(textos[i])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
