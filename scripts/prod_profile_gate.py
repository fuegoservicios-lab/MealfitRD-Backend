# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-33 · 2026-09-13] F8: el paso del gate con el perfil de knobs de PRODUCCIÓN.

`tests/conftest.py` apaga a propósito cinco gates (`setdefault(..., "false")`) y la suite deja en su default de código otros
tantos que producción enciende. Resultado, medido el 2026-09-11: **la suite entera mide un producto distinto del que se
entrega**. Corrida bajo `prod_profile.perfil_completo()` (los 33 knobs del `.env` del VPS más los tres que allí corren en su
default `True`): 104 fallos de 24.270, y `MEALFIT_VERIFIED_INGREDIENTS_ONLY=true` explicaba 56 de 56 en los cinco ficheros
peores — harnesses que construyen planes con alimentos sintéticos fuera del catálogo, que el filtro de verificados descarta.

Esos fallos no se «arreglan» a ciegas. Este script es el paso que faltaba: correr la suite CON el perfil de producción, menos
una lista de exclusión donde cada fichero dice por qué no se mide así (`tests/prod_profile_excluded.txt`), menos la
cuarentena de la CI. La lista es NEGATIVA a propósito: un test nuevo entra por defecto al paso de producción, y quien quiera
sacarlo tiene que escribir el motivo.

Cómo inyecta el perfil sin tocar `conftest`: exportando las variables al entorno del subproceso. `setdefault` no pisa una
variable que ya existe, y los knobs se leen del entorno al importar — así que el perfil llega entero.

La batería (`tests/test_p1_arq27_f3_bateria.py`) corre DESPUÉS y con el entorno normal: aplica el perfil por dentro
(`perfil_aplicado()`) y además afirma que la suite diverge de producción, cosa que bajo el perfil exportado deja de ser
verdad a propósito.

Uso:
    python scripts/prod_profile_gate.py                         # el paso (exit ≠ 0 si falla el subconjunto o la batería)
    python scripts/prod_profile_gate.py --listar                # qué entorno pone y qué excluye, sin correr nada
    python scripts/prod_profile_gate.py --sin-exclusiones --junitxml <f>   # medición: la suite entera bajo el perfil
    python scripts/prod_profile_gate.py --medir <f.xml> --out <artefacto.json> [--atribuir]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import prod_profile  # noqa: E402

EXCLUIDOS = _BACKEND / "tests" / "prod_profile_excluded.txt"
CI = _BACKEND / ".github" / "workflows" / "ci.yml"
BATERIA = "tests/test_p1_arq27_f3_bateria.py"
#: El knob que explicó todos los fallos de los cinco ficheros peores en la medición del 09-11: se prueba primero.
SOSPECHOSO = "MEALFIT_VERIFIED_INGREDIENTS_ONLY"


def cuarentena() -> list[str]:
    """La cuarentena de la CI (SSOT: la variable `QUARANTINE` del workflow)."""
    m = re.search(r'QUARANTINE="([^"]*)"', CI.read_text(encoding="utf-8"))
    return m.group(1).split() if m else []


def excluidos() -> dict[str, str]:
    """`tests/<fichero>.py  # <motivo>` por línea; líneas vacías y comentarios de cabecera se ignoran."""
    out: dict[str, str] = {}
    if EXCLUIDOS.exists():
        for ln in EXCLUIDOS.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            ruta, _, motivo = s.partition("#")
            out[ruta.strip()] = motivo.strip()
    return out


def entorno_perfil(base: dict | None = None, sin: tuple = ()) -> dict:
    """El entorno del subproceso: el del proceso (o `base`) + el perfil de producción, menos los knobs de `sin`
    (quitar un knob = volver al valor de la suite: el `setdefault` de conftest o el default del código)."""
    env = dict(os.environ if base is None else base)
    for k, v in prod_profile.perfil_completo().items():
        if k in sin:
            env.pop(k, None)
        else:
            env[k] = v
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def comando_pytest(ignorar: list[str], workers: int, junit: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short", "-m", "not e2e", "-p", "no:cacheprovider"]
    if workers > 0:
        cmd += ["-n", str(workers), "--dist", "loadfile", "--max-worker-restart=4"]
    for f in ignorar:
        cmd += ["--ignore", f]
    if junit:
        cmd.append(f"--junitxml={junit}")
    return cmd


def _cabecera(ignorar: list[str]) -> None:
    div = list(prod_profile.divergencias(dict(os.environ)))
    print(f"[PERFIL_PROD] leído {prod_profile.PROFILE_READ_AT} de {prod_profile.PROFILE_SOURCE}")
    print(f"[PERFIL_PROD] {len(prod_profile.perfil_completo())} knobs; el perfil cambia {len(div)} respecto de este entorno:")
    for k, actual, prod in div:
        print(f"    {k}: {actual} → {prod}")
    exc = excluidos()
    print(f"[PERFIL_PROD] excluidos {len(exc)} + cuarentena {len(cuarentena())} → {len(ignorar)} ficheros fuera")


def paso(workers: int, sin_exclusiones: bool = False, junit: str | None = None) -> int:
    ignorar = cuarentena() + ([] if sin_exclusiones else sorted(excluidos()))
    _cabecera(ignorar)
    t0 = time.time()
    r1 = subprocess.run(comando_pytest(ignorar, workers, junit), cwd=_BACKEND, env=entorno_perfil()).returncode
    if sin_exclusiones:
        print(f"[PERFIL_PROD] medición: pytest exit={r1} en {time.time() - t0:.0f}s")
        return r1
    # La batería aplica el perfil por dentro y afirma que la suite diverge: corre con el entorno normal.
    r2 = subprocess.run([sys.executable, "-m", "pytest", BATERIA, "-q", "--tb=short", "-p", "no:cacheprovider"],
                        cwd=_BACKEND).returncode
    print(f"[PERFIL_PROD] subconjunto exit={r1} · batería exit={r2} · {time.time() - t0:.0f}s")
    return r1 or r2


# ─────────────── medición ───────────────

def leer_junit(ruta: str) -> dict:
    ficheros: dict[str, dict] = {}
    total = {"tests": 0, "fallos": 0, "pasados": 0, "saltados": 0}
    for tc in ET.parse(ruta).getroot().iter("testcase"):
        cls = tc.get("classname") or ""
        partes = cls.split(".")
        if len(partes) < 2 or partes[0] != "tests":
            continue
        f = f"tests/{partes[1]}.py"
        d = ficheros.setdefault(f, {"fallos": 0, "pasados": 0, "saltados": 0})
        total["tests"] += 1
        if tc.find("failure") is not None or tc.find("error") is not None:
            d["fallos"] += 1
            total["fallos"] += 1
        elif tc.find("skipped") is not None:
            d["saltados"] += 1
            total["saltados"] += 1
        else:
            d["pasados"] += 1
            total["pasados"] += 1
    return {"total": total, "ficheros": ficheros}


def _fallos_aislado(fichero: str, sin: tuple = ()) -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", fichero, "-q", "--tb=no", "-p", "no:cacheprovider"],
                       cwd=_BACKEND, env=entorno_perfil(sin=sin), capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    ult = (r.stdout.strip().splitlines() or [""])[-1]
    n = sum(int(x) for x in re.findall(r"(\d+) (?:failed|error)", ult))
    if r.returncode not in (0, 1) and n == 0:
        n = -1  # no se pudo correr (colección rota, etc.)
    return n


def atribuir(fichero: str) -> dict:
    """Qué knob del perfil explica los fallos de un fichero: el conjunto mínimo cuya retirada lo deja en verde."""
    knobs = list(prod_profile.perfil_completo())
    runs = 0
    f0 = _fallos_aislado(fichero)
    runs += 1
    if f0 == 0:
        return {"veredicto": "no_reproduce_aislado", "knobs": [], "aislado_fallos": 0, "runs": runs}
    if _fallos_aislado(fichero, sin=tuple(knobs)) != 0:
        return {"veredicto": "falla_sin_perfil", "knobs": [], "aislado_fallos": f0, "runs": runs + 1}
    runs += 1
    if SOSPECHOSO in knobs and _fallos_aislado(fichero, sin=(SOSPECHOSO,)) == 0:
        return {"veredicto": "knob", "knobs": [SOSPECHOSO], "aislado_fallos": f0, "runs": runs + 1}
    runs += 1

    def minimo(cands: list[str]) -> list[str]:
        nonlocal runs
        if len(cands) <= 1:
            return cands
        a, b = cands[: len(cands) // 2], cands[len(cands) // 2:]
        runs += 1
        if _fallos_aislado(fichero, sin=tuple(a)) == 0:
            return minimo(a)
        runs += 1
        if _fallos_aislado(fichero, sin=tuple(b)) == 0:
            return minimo(b)
        return cands

    k = minimo(knobs)
    return {"veredicto": "knob" if len(k) == 1 else "combinacion", "knobs": k, "aislado_fallos": f0, "runs": runs}


def medir(junit: str, out: str, con_atribucion: bool, workers: int) -> int:
    datos = leer_junit(junit)
    con_fallos = sorted(f for f, d in datos["ficheros"].items() if d["fallos"])
    if con_atribucion and con_fallos:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for f, a in zip(con_fallos, ex.map(atribuir, con_fallos)):
                datos["ficheros"][f]["atribucion"] = a
                print(f"  {f}: {a['veredicto']} {a['knobs']} ({a['runs']} corridas)")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_BACKEND, capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        sha = None
    art = {
        "schema": "2026-09-13.f8",
        "fecha": date.today().isoformat(),
        "git_sha": sha,
        "metodo": ("la suite (`-m 'not e2e'`, sin la cuarentena de la CI) bajo `prod_profile.perfil_completo()` exportado al "
                   "entorno; atribución por fichero: el conjunto mínimo de knobs cuya retirada deja el fichero en verde "
                   "corriendo SOLO ese fichero (primero el sospechoso de la medición del 09-11)"),
        "perfil": {"leido": prod_profile.PROFILE_READ_AT, "fuente": prod_profile.PROFILE_SOURCE,
                   "knobs": prod_profile.perfil_completo()},
        "corrida": datos["total"],
        "ficheros_con_fallos": len(con_fallos),
        "ficheros": {f: datos["ficheros"][f] for f in con_fallos},
    }
    Path(out).write_text(json.dumps(art, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[PERFIL_PROD] {datos['total']} · {len(con_fallos)} ficheros con fallos → {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--sin-exclusiones", action="store_true", help="medición: la suite entera bajo el perfil")
    ap.add_argument("--junitxml")
    ap.add_argument("--medir", metavar="JUNIT")
    ap.add_argument("--out")
    ap.add_argument("--atribuir", action="store_true")
    a = ap.parse_args(argv)
    if a.listar:
        _cabecera(cuarentena() + sorted(excluidos()))
        for f, m in sorted(excluidos().items()):
            print(f"  {f}  # {m}")
        return 0
    if a.medir:
        if not a.out:
            ap.error("--medir exige --out")
        return medir(a.medir, a.out, a.atribuir, a.workers)
    return paso(a.workers, a.sin_exclusiones, a.junitxml)


if __name__ == "__main__":
    sys.exit(main())
