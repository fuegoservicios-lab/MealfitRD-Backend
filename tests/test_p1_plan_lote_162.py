"""[P1-PLAN-LOTE-162 · 2026-09-22] Auditoría de la beta en modo contador — lo que se cierra por OTA (frontend).

Anclas desde el backend (el único sitio que ve los dos árboles) de lo que el lote 162 cambió en el frontend. Las
pruebas funcionales viven en `frontend/src/__tests__/lote162.test.jsx` (con el doble fiel del plugin y la tarjeta
de agua montada de verdad con el reloj simulado); aquí se fija que el arreglo sigue en su sitio.

1. **Android 14+ y las alarmas exactas.** El plugin de avisos locales programa EXACTO por defecto y, sin el permiso
   «Alarmas y recordatorios» (que en Android 14 nace denegado), abre la pantalla del sistema en CADA `schedule()`. La
   app reprograma al abrir, al volver, tras cada comida y tras cada turno del coach: el tester acababa sacado a
   Ajustes una y otra vez. Ahora: exacta solo con el permiso dado; el permiso se pide desde Configuración.
2. **Agua.** Con la app abierta desde anoche, el primer vaso de la mañana salía con la fecha de AYER y el servidor
   sobrescribía el total de ese día.
3. **Los interruptores de avisos** mandaban el formulario entero (copia local congelada) y se pisaban entre sí.
4. Memoria del coach visible para toda cuenta; escáner que solo habla de la Nevera si está en uso; Historial vacío
   del contador con salida a los días anteriores; «Ver días anteriores» que recalcula hoy; borrado a medias que no
   se anuncia como hecho; Android sin instrucciones de iPhone; hoja de Google cerrada con salida al correo.

Tooltip-anchor: P1-PLAN-LOTE-162
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _f(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"frontend ausente: {rel}")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_alarma_exacta_solo_con_el_permiso_dado():
    av = _f("src/utils/avisosDeComida.js")
    assert "const exacta = (await _alarmaExactaConcedida(LN)) === true;" in av
    # [P1-PLAN-LOTE-166] la misma línea suma el canal propio de Android (solo si se pudo crear); lo que se vigila aquí
    # —exacta SOLO con el permiso dado— no cambia.
    assert ".map((n) => ({ ...n, isExactNotification: exacta, ...(canal ? { channelId: CANAL_ANDROID } : {}) }));" in av
    # la consulta tiene tope: un método que no contesta no puede colgar la reprogramación
    assert "_TOPE_CONSULTA_ALARMA_MS" in av and "Promise.race([" in av
    # el permiso lo pide la PERSONA, desde Configuración
    st = _f("src/pages/Settings.jsx")
    assert "const ok = await pedirAlarmaExacta();" in st
    assert "alarmaExactaPendiente().then(" in st


def test_el_agua_pone_la_fecha_al_dia_antes_de_escribir():
    wt = _f("src/components/dashboard/WaterTracker.jsx")
    assert "setCurrentDate(hoy);\n            loadIntake(hoy);" in wt
    assert "const fechaAlDia = useCallback(() => {" in wt
    assert "if (!loading && fechaAlDia()) persist(glassesRef.current + n);" in wt


def test_los_interruptores_de_avisos_guardan_solo_su_clave():
    st = _f("src/pages/Settings.jsx")
    i = st.index("const cambiarPrefDeAviso")
    cuerpo = st[i:i + 1600]
    assert "body: JSON.stringify({ health_profile: { [clave]: valor } })," in cuerpo
    assert "safeUpdateHealthProfile(" not in cuerpo
    sfs = _f("src/config/secureFormStorage.js")
    # [P1-PLAN-LOTE-216] la lista creció con `avisos_por_comida` (interruptor y hora de cada recordatorio): lo que se
    # vigila aquí —los dos interruptores no viajan en el formulario— no cambia.
    m = re.search(r"export const CLAVES_CON_CONTROL_PROPIO = Object\.freeze\(\[([^\]]*)\]\);", sfs)
    assert m, "CLAVES_CON_CONTROL_PROPIO desapareció de secureFormStorage.js"
    assert {"avisos_comida", "avisos_agua"} <= set(re.findall(r"'([^']+)'", m.group(1)))
    assert "for (const clave of CLAVES_CON_CONTROL_PROPIO) delete base[clave];" in sfs


def test_la_memoria_del_coach_no_depende_del_tier():
    st = _f("src/pages/Settings.jsx")
    assert "{!isGuest && ltmEnabled !== null && (" in st
    assert "El Cerebro IA está disponible a partir del plan" not in st
    assert not re.search(r"[{,]\s*isPremium\s*[,}]", st)


def test_historial_y_dias_anteriores_del_contador():
    h = _f("src/pages/History.jsx")
    assert "onClick={enModoContador ? _verDiasAnteriores : _encenderElPlan}" in h
    tp = _f("src/components/dashboard/TrackingProgress.jsx")
    assert "if (consumirAbrirDiasAnteriores()) setHistoryOpen(true);" in tp
    dh = _f("src/components/dashboard/DiaryHistory.jsx")
    assert "const hoyISO = useMemo(() => aISO(new Date()), [open]);" in dh


def test_escaner_borrado_google_y_android():
    m = _f("src/components/dashboard/ScanMealModal.jsx")
    assert "if (ausentes.length > 0 && (bajaron > 0 || neveraConCosas === true)) {" in m
    d = _f("src/components/account/DeleteAccountSection.jsx")
    assert "if (_resultado && _resultado.success === false) {" in d
    lg = _f("src/pages/Login.jsx")
    assert "if (vuelta.detalle) console.error('[Google Android] hoja cerrada con detalle:', vuelta.detalle);" in lg
    pl = _f("src/config/platform.js")
    assert "export function nativePlatform() {" in pl


@pytest.mark.parametrize("loc", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_los_textos_nuevos_estan_traducidos(loc):
    cat = json.loads(_f(f"src/i18n/locales/{loc}.json"))
    for k in (
        "Tus avisos pueden llegar con unos minutos de margen. Para que suenen a su hora exacta, permite «Alarmas y recordatorios».",
        "Ver mis días anteriores",
        "Aviso del coach",
        "Si no pudiste entrar con Google, entra con tu correo: te mandamos un código de 6 dígitos.",
    ):
        assert cat.get(k), f"{loc}: falta «{k}»"


def test_el_marcador_esta_al_dia():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', src, re.M)
    assert m and int(m.group(1)) >= 162
