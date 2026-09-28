# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-795 · 2026-09-28] Copias y cifras de la app que eran falsas (decisiones del dueño).

1. «Multi-condición clínica … (p. ej. DM2 + renal) … Verificado en agosto»: la matriz medida en agosto
   (`CLINICAL.n` = 20 perfiles, `docs/landing_benchmarks.md`) NO tuvo perfil renal — los perfiles 21-25
   (`renal_hta` incluido) se añadieron después y no se han corrido. El ejemplo pasa a una combinación que
   SÍ se midió (perfil 12, DM2 + HTA + colesterol).
2. El sub de la fila clínica de CAPS («DM2 · renal · HTA · alergias») nombraba renal: mismo motivo.
3. «Tu primer plan, calculado, en cinco minutos»: lo medido es bloque 1 p50 ~4 min y hasta ~10. Un
   número en el titular sería falso para la mitad de los usuarios.
4. `VERIFIED_FOODS_LABEL` decía «200+» con 354 filas en `master_ingredients` (SELECT del 28-sep).
5. El Aviso Médico (§6) sólo nombraba el 9-1-1 de RD: a un usuario de España (112) o de Colombia (123)
   le daba un número que allí no es el de emergencias. Texto nuevo de G94, el mismo que el landing,
   y la regla: cada país de `COUNTRY_PROFILES` con SU número (SSOT `emergency_number`, lote 741).

tooltip-anchor: P1-PLAN-LOTE-795
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import constants

_SRC = Path(__file__).resolve().parent.parent.parent / "frontend" / "src"


def _leer(rel: str) -> str:
    p = _SRC / rel
    if not p.is_file():
        pytest.skip(f"{rel} no está en este árbol")
    return p.read_text(encoding="utf-8")


def _aviso_medico() -> str:
    texto = _leer("pages/legal/LegalPages.jsx")
    ini = texto.find("export const MedicalDisclaimer")
    assert ini != -1, "no se encontró el componente del Aviso Médico"
    return texto[ini: texto.find("</LegalLayout>", ini)]


def _seccion_6(aviso: str) -> str:
    ini = aviso.find("<h3>6. Emergencias Médicas</h3>")
    assert ini != -1, "el Aviso Médico perdió su §6 «Emergencias Médicas»"
    return aviso[ini: aviso.find("<h3>", ini + 5)]


def _numero(n: str) -> str:
    return n.replace("-", "")   # 9-1-1 ≡ 911: la grafía dominicana y la del resto marcan lo mismo


# ── 5. Aviso Médico §6 ───────────────────────────────────────────────────────────────────────────

def test_cada_pais_con_su_numero_de_emergencias():
    s6 = _seccion_6(_aviso_medico())
    por_pais: dict[str, str] = {}
    for numero, paises in re.findall(r"<strong>([\d-]+)</strong> en ([^;.]+)", s6):
        for pais in re.split(r",\s*|\s+y\s+", paises):
            if pais.strip():
                por_pais[pais.strip()] = _numero(numero)
    for codigo, perfil in constants.COUNTRY_PROFILES.items():
        nombre = perfil["name_es"]
        assert nombre in por_pais, f"§6 del Aviso Médico no nombra {nombre} ({codigo})"
        assert por_pais[nombre] == perfil["emergency_number"], (
            f"§6 da a {nombre} el {por_pais[nombre]}; su número es el {perfil['emergency_number']}"
        )


def test_el_resto_del_mundo_sigue_cubierto():
    s6 = _seccion_6(_aviso_medico())
    assert "número de emergencias del lugar donde se encuentre" in s6
    assert "Si se encuentra en cualquier otro país, use el número de emergencias local." in s6
    assert "sala de emergencia más cercana" in s6 and "médico tratante" in s6


def test_la_fecha_del_aviso_medico_se_movio():
    texto = _leer("pages/legal/LegalPages.jsx")
    m = re.search(r'title="Aviso Médico"\s+lastUpdated="([^"]+)"', texto)
    assert m and m.group(1) == "28 de Septiembre, 2026"


# ── 1-2. La clínica: nada de «renal» en lo que se presenta como verificado ──────────────────────

def test_features_no_verifica_un_perfil_que_no_se_midio():
    features = _leer("pages/FeaturesPage.jsx")
    lineas = [l for l in features.splitlines() if "Verificado" in l]
    assert lineas, "FeaturesPage ya no tiene la línea «Verificado …» que este test vigila"
    for l in lineas:
        assert "renal" not in l.lower(), f"se afirma verificado un perfil renal que nunca se midió: {l.strip()[:160]}"
    assert any("DM2 + HTA + colesterol" in l for l in lineas), (
        "el ejemplo debe ser una combinación que la matriz de agosto SÍ midió (perfil 12)"
    )


def test_caps_clinico_no_nombra_renal():
    bench = _leer("data/benchmark.js")
    m = re.search(r"key:\s*'clinical',.*?sub:\s*'([^']+)'", bench, re.S)
    assert m, "no se encontró la fila clínica de CAPS"
    assert "renal" not in m.group(1).lower(), f"CAPS vuelve a nombrar renal: {m.group(1)}"


# ── 3. Sin número de minutos en el cierre ─────────────────────────────────────────────────────────

def test_el_cierre_no_promete_cinco_minutos():
    cierre = _leer("components/home/ClosingBand.jsx")
    assert "cinco minutos" not in cierre
    assert not re.search(r"en \d+ minutos", cierre)
    assert "Tu primer plan, calculado, en minutos." in cierre


# ── 4. El catálogo ─────────────────────────────────────────────────────────────────────────────────

def test_catalogo_300_mas_con_su_medicion():
    facts = _leer("data/systemFacts.js")
    assert "export const VERIFIED_FOODS_LABEL = '300+';" in facts
    assert "354" in facts and "2026-09-28" in facts, "el label debe citar el conteo y la fecha de la medición"
