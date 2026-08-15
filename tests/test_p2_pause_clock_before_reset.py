"""[P2-PAUSE-CLOCK-BEFORE-RESET · 2026-08-15] La ventana de reanudación se mide
ANTES de reiniciar el reloj que la mide.

EL BUG. `resume_plan_generation` hace, en este orden:

    paso 1  UPDATE user_profiles SET plan_mode='plan',
              plan_mode_changed_at = CASE WHEN plan_mode <> 'plan' THEN NOW() … END
    paso 4  SELECT NOW() - plan_mode_changed_at  AS paused_days

Reanudar siempre viene de `plan_mode='tracking'`, así que el CASE del paso 1
SIEMPRE entra y estampa `NOW()`. Cuando el paso 4 mide, la resta es contra un
valor puesto milisegundos antes: `paused_days` es 0 SIEMPRE, y por tanto
`expired = paused_days > MEALFIT_PLAN_PAUSE_MAX_RESUME_DAYS` es False SIEMPRE.

Es decir: el único aviso que el sistema tiene sobre la ventana de 30 días es
código inalcanzable. Un usuario que vuelve a los seis meses recibe «la generación
continúa donde quedó» y se le reencola un plan cuyas fechas son historia.

LA FORMA DEL ARREGLO. No se mueve el paso 1 —su orden es load-bearing: la bandera
va PRIMERO porque encolar con el gate puesto deja chunks que el pickup ignora—.
Lo que se mueve es la LECTURA: se captura `plan_mode_changed_at` antes de tocar
nada, y el paso 4 pasa a ser aritmética sobre ese snapshot.

    Un reloj que se pone a cero antes de leerlo no mide: sólo confirma que lo
    acabas de poner a cero.

Tooltip-anchor: P2-PAUSE-CLOCK-BEFORE-RESET
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_PM = Path(__file__).resolve().parent.parent / "plan_mode.py"


def _cuerpo_resume() -> str:
    src = _PM.read_text(encoding="utf-8")
    i = src.find("def resume_plan_generation(")
    assert i >= 0, "[P2-PAUSE-CLOCK-BEFORE-RESET] No existe resume_plan_generation"
    j = src.find("\ndef ", i + 1)
    cuerpo = src[i:j if j > 0 else len(src)]
    return re.sub(r"^\s*#.*$", "", cuerpo, flags=re.MULTILINE)


def test_el_reloj_se_lee_antes_de_reiniciarse():
    cuerpo = _cuerpo_resume()
    i_lectura = cuerpo.find("plan_mode_changed_at")
    i_reset = cuerpo.find("plan_mode_changed_at = CASE")
    assert i_reset > 0, (
        "[P2-PAUSE-CLOCK-BEFORE-RESET] Desapareció el reinicio del reloj en el paso 1. "
        "Si el cambio fue intencional, actualiza este guard."
    )
    assert i_lectura < i_reset, (
        "[P2-PAUSE-CLOCK-BEFORE-RESET] `plan_mode_changed_at` se REINICIA antes de "
        "leerse. Reanudar siempre viene de 'tracking', así que el CASE siempre "
        "estampa NOW() y `paused_days` sale 0 SIEMPRE: la ventana de reanudación "
        "nunca puede vencer y `plan_expired` es código inalcanzable."
    )


def test_la_medida_no_vuelve_a_consultar_la_columna_ya_pisada():
    """El paso 4 debe ser aritmética sobre el snapshot, no un segundo SELECT."""
    cuerpo = _cuerpo_resume()
    i_reset = cuerpo.find("plan_mode_changed_at = CASE")
    despues = cuerpo[i_reset:]
    assert "EXTRACT(EPOCH FROM (NOW() - plan_mode_changed_at))" not in despues, (
        "[P2-PAUSE-CLOCK-BEFORE-RESET] Después de reiniciar el reloj sigue habiendo "
        "un SELECT que mide contra esa misma columna. Da 0 por construcción."
    )


def test_paused_days_sigue_en_el_contrato_de_salida():
    """El arreglo no puede cambiar la forma que el endpoint ya consume."""
    cuerpo = _cuerpo_resume()
    for clave in ("paused_days", "plan_expired", "plan_mode", "plan_status"):
        assert clave in cuerpo, (
            f"[P2-PAUSE-CLOCK-BEFORE-RESET] Falta `{clave}` en el retorno de "
            "resume_plan_generation: el endpoint y el frontend lo leen."
        )


def test_la_bandera_sigue_yendo_primero():
    """El orden del paso 1 es load-bearing; lo que se adelanta es la LECTURA."""
    cuerpo = _cuerpo_resume()
    i_flag = cuerpo.find("SET plan_mode = 'plan'")
    i_revive = cuerpo.find("_revive_paused_chunks")
    assert i_flag > 0 and i_revive > 0, "Faltan el UPDATE de bandera o el revive"
    assert i_flag < i_revive, (
        "[P2-PAUSE-CLOCK-BEFORE-RESET] El revive de la cola quedó ANTES de levantar "
        "la bandera. Revivir con el gate puesto deja chunks `pending` que el pickup "
        "ignora — invisibles. Ese orden no se toca."
    )
