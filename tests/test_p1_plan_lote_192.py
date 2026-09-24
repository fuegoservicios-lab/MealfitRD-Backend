# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-192 · 2026-09-24] Compartir el día + la Nevera opcional en modo contador + micros inverosímiles.

Lote hecho EN PARALELO a la serie del generador (sesión aparte, worktrees `nevera-compartir`, número reservado con la
otra sesión). Tres encargos del dueño del 23-sep:

  · «Agrega una forma de compartir tus macros y micros del día para enviárselo a amigos en WhatsApp» — botón
    «Compartir» en «Tus macros y micros de hoy»: imagen en canvas + texto, WhatsApp por `wa.me`, copiar, y descarga
    solo en la web (P1-COMPARTIR-DIA; contrato fino en frontend `compartirDia/tarjetaDelDia/ShareDaySheet`).
  · «Cuando el generador de planes esté desactivado, que la Nevera también tenga una opción en Configuración para
    desactivarla» + «si en 48 horas no se usa, que se desactive sola» — UNA regla en `nevera_opcional.py`, servida en el
    perfil; el diario no descuenta, el coach no la ve, la app la oculta; encenderla a mano es definitivo
    (P1-NEVERA-OPCIONAL; contrato fino en `test_p1_nevera_opcional.py`, doc `docs/nevera_opcional.md`).
  · La captura del dueño decía potasio 9.815 mg: «8 rodajas de plátano maduro» se contaba como 8 plátanos enteros.
    Una comida cuyos renglones pesan más de 1,5× sus kcal no aporta micros (P1-DIARY-MICROS-PLAUSIBLE; el lector de
    unidades del generador queda para su serie)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_la_regla_de_la_nevera_vive_en_un_solo_sitio():
    import nevera_opcional as no
    assert no.nevera_activa_de({"plan_mode": "tracking", "nevera_enabled": False}) is False
    # [P1-PLAN-LOTE-217 · 2026-09-24] la Nevera también se apaga en modo plan (antes: «en modo plan, activa siempre»)
    assert no.nevera_activa_de({"plan_mode": "plan", "nevera_enabled": False}) is False
    assert no.nevera_activa_de({"plan_mode": "tracking", "nevera_enabled": None}) is True


def test_el_apagado_automatico_esta_registrado_y_documentado():
    ct = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert 'id="nevera_auto_off"' in ct[ct.index("def register_plan_chunk_scheduler("):]
    doc = (_BACKEND / "docs" / "nevera_opcional.md").read_text(encoding="utf-8")
    assert "MEALFIT_NEVERA_AUTO_OFF" in doc and "nevera_reloj_desde" in doc


def test_la_guarda_de_micros_inverosimiles_esta_en_el_dia():
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert 'm["micros"] = micros_plausibles(m["micros"], m.get("calories"))' in diary


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 192 and m.group(2) >= "2026-09-24"
