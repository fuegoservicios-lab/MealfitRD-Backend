"""[P1-PLAN-DISPLAY-I18N · Task 5 · 2026-08-19] One-shot: puebla
`master_ingredients.name_en` (gloss en inglés para la lista de compras
BILINGÜE — fase 1b de docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md,
regla de oro: el usuario cocina en su idioma pero COMPRA en español, la lista
NUNCA sale en inglés puro).

DISPLAY-ONLY: `name_en` NUNCA entra a `normalize_name`, aliases, matchers ni
`pantry_names_match` — la identidad de una fila del catálogo sigue resolviendo
EXCLUSIVAMENTE por `name` (español canónico). Este script SOLO escribe `name_en`,
jamás toca `name`.

Flujo:
  1. Lee TODAS las filas de `master_ingredients` (columna `name`).
  2. UNA llamada LLM flash batch pidiendo el contrato JSON estricto
     `{"items":[{"name":"...","name_en":"..."}]}`.
  3. Imprime la tabla `name -> name_en` completa a stdout, SIEMPRE (auditoría
     del dueño antes de decidir `--commit`).
  4. Sin `--commit` (default = dry-run): no toca la DB.
     Con `--commit`: persiste `UPDATE master_ingredients SET name_en = %s
     WHERE name = %s` fila por fila, ya validada.

Fail-loud: si el LLM devuelve MENOS filas que las pedidas, o nombres que no
matchean ninguna fila real del catálogo (ni exacto ni accent/case-insensitive),
el script aborta con `sys.exit(1)` ANTES de escribir nada en DB — preferimos
tener que re-correrlo manualmente a dejar la columna a medio poblar sin que
nadie lo note (misma filosofía fail-loud que el resto del repo, ver I8/P2-NEXT-4).

⚠️ Gasta una llamada LLM real incluso en `--dry-run` — NO ejecutar sin
autorización explícita del dueño (convención `feedback_api_spend_caution`:
benchmarks/scripts que llaman LLM solo corren DIRIGIDOS). El controller de
esta task decide cuándo correrlo y si promueve a `--commit`.

Uso:
    NEON_DATABASE_URL(_POOLED) + ZAI_API_KEY en .env
    python scripts/fill_catalog_name_en.py                 # dry-run (default)
    python scripts/fill_catalog_name_en.py --commit         # persiste
    python scripts/fill_catalog_name_en.py --model glm-5.3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND, ".env"))
except Exception:
    pass

import psycopg
from langchain_core.messages import HumanMessage, SystemMessage

from llm_provider import build_chat_llm, GLM_FLASH
from constants import strip_accents

NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")

_JSON_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_PROMPT_TEMPLATE = (
    "You translate a Latin American grocery catalog's food names into English, for a SHOPPING "
    "LIST label read by an English-speaking user living abroad who still buys these foods by "
    "their Spanish name. This is a short GROCERY gloss, not a recipe translation.\n\n"
    "STRICT RULES:\n"
    "1. Translate ONLY the food/ingredient itself into US English (e.g. 'Pechuga de pollo' -> "
    "'Chicken breast', 'Habichuelas rojas' -> 'Red beans'). NO brand names, NO quantities/units, "
    "NO preparation instructions.\n"
    "2. Even if a name has no perfect English equivalent (a regional dish name), ALWAYS give the "
    "closest short descriptive English gloss — never leave an item untranslated or skip it.\n"
    "3. Reply with ONLY valid JSON, no markdown, no text outside the JSON. Return EXACTLY one "
    "item per input name, in the SAME order, with this exact contract:\n"
    '{{"items":[{{"name":"<original Spanish name, unchanged>","name_en":"<English gloss>"}}]}}\n\n'
    "NAMES ({n} total):\n{names_block}"
)


def _build_prompt(names: list) -> str:
    names_block = "\n".join(f"{i}. {n}" for i, n in enumerate(names))
    return _PROMPT_TEMPLATE.format(n=len(names), names_block=names_block)


def _parse_json_response(raw: str):
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = _JSON_CODE_FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def fetch_catalog_names(conn, only_missing: bool = False) -> list:
    """[Ola final · FF-10] `only_missing=True` acota el SELECT a las filas SIN gloss
    (`name_en IS NULL`). Re-ejecutar el script completo re-paga la llamada LLM sobre las
    347 filas del catálogo — con esta flag un re-run tras una fila nueva (o tras un
    fallo parcial) cuesta lo que cuestan esas pocas filas, no el catálogo entero.
    """
    sql = "SELECT name FROM master_ingredients"
    if only_missing:
        sql += " WHERE name_en IS NULL"
    sql += " ORDER BY name"
    rows = conn.execute(sql).fetchall()
    return [r[0] for r in rows if r and r[0]]


def translate_batch(names: list, model: str, timeout_s: float = 180.0) -> dict:
    """UNA llamada LLM para TODO el catálogo (contrato del brief: 'UNA llamada flash
    batch'). Retorna `{name: name_en}` SOLO tras validar que CADA nombre pedido
    resolvió a exactamente una traducción — fail-loud (RuntimeError) si algo no
    matchea o faltan filas; el caller decide si eso aborta el proceso.
    """
    prompt = _build_prompt(names)
    llm = build_chat_llm(model, temperature=0.1, timeout=timeout_s, max_output_tokens=16000)
    # [P1-I18N-GLM-USER-TURN] GLM exige un turno de usuario: solo system => 400/1214
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Proceed. Reply with ONLY the JSON.")])
    raw = getattr(response, "content", "") or ""
    parsed = _parse_json_response(raw)
    if parsed is None or not isinstance(parsed.get("items"), list):
        raise RuntimeError(
            f"[FAIL-LOUD] respuesta LLM no es JSON válido con 'items' -- raw[:300]={raw[:300]!r}"
        )

    requested_exact = {n: n for n in names}
    requested_normalized = {strip_accents(n).strip().lower(): n for n in names}

    out: dict = {}
    unmatched: list = []
    for item in parsed["items"]:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("name")
        gloss = item.get("name_en")
        if not isinstance(raw_name, str) or not isinstance(gloss, str) or not gloss.strip():
            continue
        matched = requested_exact.get(raw_name.strip())
        if matched is None:
            matched = requested_normalized.get(strip_accents(raw_name).strip().lower())
        if matched is None:
            unmatched.append(raw_name)
            continue
        out[matched] = gloss.strip()

    if unmatched:
        preview = unmatched[:10]
        raise RuntimeError(
            f"[FAIL-LOUD] {len(unmatched)} nombre(s) devueltos por el LLM no matchean ninguna "
            f"fila pedida del catálogo (ni exacto ni accent/case-insensitive): {preview}"
            + ("..." if len(unmatched) > len(preview) else "")
        )

    if len(out) < len(names):
        faltantes = [n for n in names if n not in out]
        preview = faltantes[:10]
        raise RuntimeError(
            f"[FAIL-LOUD] el LLM devolvió {len(out)}/{len(names)} traducciones -- faltan "
            f"{len(faltantes)}: {preview}" + ("..." if len(faltantes) > len(preview) else "")
        )

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", action="store_true",
        help="Persiste UPDATE en DB. Sin esta flag (default): dry-run, solo imprime.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Explícito: NO persiste (dry-run YA es el default sin --commit). "
             "Mutuamente excluyente con --commit.",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Solo las filas SIN gloss (name_en IS NULL) -- re-runs baratos: no re-paga "
             "la llamada LLM sobre el catálogo entero.",
    )
    parser.add_argument(
        "--model", default=GLM_FLASH,
        help=f"Modelo LLM (default: {GLM_FLASH}).",
    )
    args = parser.parse_args()
    # [Ola final · FF-10] `--dry-run` era un flag INERTE: `commit = bool(args.commit)`
    # lo ignoraba, así que `--commit --dry-run` ESCRIBÍA. Un flag de seguridad que no
    # protege es peor que ninguno (invita a confiar en él). Ahora la combinación es un
    # error de uso -- `parser.error` sale con código 2, sin tocar la DB.
    if args.commit and args.dry_run:
        parser.error("--commit y --dry-run son mutuamente excluyentes: elige uno.")
    commit = bool(args.commit)

    if not NEON:
        print("FATAL: falta NEON_DATABASE_URL(_POOLED) en el entorno.")
        sys.exit(1)

    # [cierre Task 6 · 2026-08-19] La conexión de LECTURA se abre y CIERRA antes de la
    # llamada LLM: sostenerla durante la traducción (~1-3 min) hacía que Neon la matara
    # por idle y el exit del context manager reventara con ProtocolViolation («server
    # conn crashed?») — medido en el primer dry-run real. En --commit habría sido fatal:
    # los UPDATEs habrían corrido sobre conexión muerta. Lectura → cerrar → LLM →
    # conexión NUEVA solo para escribir.
    with psycopg.connect(NEON) as conn:
        names = fetch_catalog_names(conn, only_missing=bool(args.only_missing))
    if not names:
        if args.only_missing:
            # [Ola final · FF-10] Con --only-missing, "cero filas" es el estado
            # DESEADO (catálogo completo), no un fallo -- salir 0 para que un
            # re-run idempotente en un pipeline no rompa.
            print("OK: 0 filas sin name_en -- el catálogo ya está completo.")
            return
        print("FATAL: master_ingredients no devolvió ninguna fila.")
        sys.exit(1)

    # [P2-LOGGER-EXEMPT: CLI subcommand a stdout] -- este script es un CLI
    # one-shot, no un módulo de producción importado por app.py; su output
    # ES el producto (tabla de auditoría para el dueño antes de --commit).
    print(f"Catálogo: {len(names)} filas. model={args.model!r} commit={commit}")

    try:
        translations = translate_batch(names, args.model)
    except Exception as e:
        print(f"FATAL: traducción falló -- {e}")
        sys.exit(1)

    print(f"\n{'name (es)':<42} name_en")
    print("-" * 90)
    for n in names:
        print(f"{n:<42} {translations.get(n, '')}")

    if commit:
        written = 0
        with psycopg.connect(NEON) as conn:
            with conn.cursor() as cur:
                for n, gloss in translations.items():
                    cur.execute(
                        "UPDATE master_ingredients SET name_en = %s WHERE name = %s",
                        (gloss, n),
                    )
                    written += cur.rowcount
            conn.commit()
        print(f"\nCOMMIT: {written} fila(s) de master_ingredients actualizadas con name_en.")
    else:
        print(
            f"\nDRY-RUN: {len(translations)} traducciones listas, NADA persistido. "
            f"Corre con --commit para escribir en master_ingredients.name_en."
        )


if __name__ == "__main__":
    main()
