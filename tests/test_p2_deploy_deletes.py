"""[P2-DEPLOY-DELETES · 2026-08-14] El deploy del frontend borra lo que tú borraste.

`tar -xzf` sólo AÑADE y ACTUALIZA. Nunca borra. Así que un archivo que sacas del
repo sobrevive en `/opt/mealfit` para siempre, y vite lo vuelve a copiar a cada
release como si siguiera vivo.

Cómo se descubrió: el dueño pidió quitar el isotipo del dashboard. Lo borré del
repo, lo commiteé, desplegué con el gate verde… y `https://bioboros.com/bioboros-mark.png`
seguía devolviendo `200 image/png`. **Un borrado que no llega a producción no es
un borrado: es una creencia.**

El inventario del día lo confirmó como clase, no como incidente:

  - `public/` — 13 huérfanos, 6,4 MB republicados en CADA release (y hay 5 vivas).
    Entre ellos `mealfit-mark-dark.png`, que la auditoría del landing había
    borrado esa misma mañana *precisamente* para adelgazar el bundle. Su trabajo
    llevaba horas siendo inerte en producción sin que nadie lo notara.
  - `src/` — 26 huérfanos, y estos son peores que peso muerto. `supabase.js` y
    `config/api.js` seguían ahí después de que el repo los eliminara: un `import`
    a un módulo ya inexistente compilaría en el VPS y explotaría en un checkout
    limpio. El build de producción mintiendo a favor es la peor dirección posible
    para una mentira.

El fix vacía esos dos directorios antes de extraer. Este guard defiende las tres
propiedades que lo hacen seguro — y la CUARTA, que es la que de verdad importa:
que a nadie se le ocurra "generalizarlo" al backend.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEPLOY = _REPO_ROOT / "deploy-mealfit.ps1"


def _codigo_sin_comentarios() -> str:
    """El script sin comentarios PowerShell.

    Imprescindible aquí: el comentario que explica POR QUÉ se borra `src/` cita
    literalmente el `rm -rf`, así que un guard que leyera el fichero entero se
    daría por satisfecho con la explicación y jamás miraría el comando. Es el
    modo de fallo que ya nos costó un guard este mes.
    """
    if not _DEPLOY.exists():
        pytest.fail(f"[P2-DEPLOY-DELETES] No existe {_DEPLOY}")
    texto = _DEPLOY.read_text(encoding="utf-8", errors="replace")
    texto = re.sub(r"<#.*?#>", "", texto, flags=re.DOTALL)  # bloque de ayuda
    return re.sub(r"^\s*#.*$", "", texto, flags=re.MULTILINE)


def _rutas_que_se_borran(codigo: str) -> list[str]:
    """TODAS las rutas de TODOS los `rm -rf`, no sólo la primera de cada uno.

    Esto lo escribió una mutación que sobrevivió. La primera versión buscaba
    `rm -rf <ruta-prohibida>` con la ruta pegada al comando, así que un
    `rm -rf .../frontend/src .../frontend/public /opt/mealfit/backend/tests`
    pasaba invisible: el destino peligroso era el TERCER argumento y el patrón
    no sabía cruzar los espacios. Un guard que sólo mira el primer argumento de
    un comando variádico no vigila el comando: vigila su prefijo.
    """
    rutas: list[str] = []
    for m in re.finditer(r"rm\s+(?:-[a-zA-Z]+\s+)*", codigo):
        # Los argumentos llegan hasta el siguiente separador de shell o el
        # cierre de la cadena de PowerShell que envuelve el comando ssh.
        resto = re.split(r"&&|\|\||[;|'\"\n]", codigo[m.end():], maxsplit=1)[0]
        rutas.extend(tok for tok in resto.split() if tok.startswith("/"))
    return rutas


def _comando_ssh_del_frontend() -> str:
    """La línea que construye `$feOut`: el único ssh que compila el frontend."""
    for linea in _codigo_sin_comentarios().splitlines():
        if "$feOut = ssh" in linea:
            return linea
    pytest.fail(
        "[P2-DEPLOY-DELETES] No encuentro la línea `$feOut = ssh ...` en "
        "deploy-mealfit.ps1. Si la renombraste, actualiza este guard: sin ella "
        "no puede comprobar nada y pasaría en verde sobre un script cualquiera."
    )


# ---------------------------------------------------------------------------
# 1. El frontend SÍ borra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("directorio", ["src", "public"])
def test_el_deploy_del_frontend_vacia_el_directorio_antes_de_extraer(directorio: str):
    cmd = _comando_ssh_del_frontend()
    objetivo = f"/opt/mealfit/frontend/{directorio}"

    assert objetivo in cmd, (
        f"[P2-DEPLOY-DELETES] El deploy del frontend ya no vacía `{objetivo}`.\n\n"
        "`tar -xzf` sólo añade y actualiza: sin este borrado, cada archivo que "
        "saques del repo sobrevive en el VPS para siempre y vite lo republica.\n\n"
        "Medido el 2026-08-14 antes del fix: 13 huérfanos en public/ (6,4 MB por "
        "release) y 26 en src/, incluido el isotipo que el dueño acababa de pedir "
        "quitar — seguía sirviéndose con 200 tras un deploy verde."
    )

    pos_rm = cmd.find("rm -rf")
    pos_extraer = cmd.find("tar -xzf")
    assert 0 <= pos_rm < pos_extraer, (
        "[P2-DEPLOY-DELETES] El `rm -rf` debe ir ANTES del `tar -xzf`.\n\n"
        "Después no limpia: borraría lo que el tar acaba de escribir y el build "
        "se quedaría sin fuentes. El orden es la mitad del fix."
    )


def test_el_borrado_va_detras_de_un_seguro_que_valida_el_paquete():
    cmd = _comando_ssh_del_frontend()

    pos_listar = cmd.find("tar -tzf")
    pos_rm = cmd.find("rm -rf")
    assert 0 <= pos_listar < pos_rm, (
        "[P2-DEPLOY-DELETES] Falta el `tar -tzf ... >/dev/null` DELANTE del `rm -rf`.\n\n"
        "Es el seguro: si la subida llegó truncada (ya pasó — incidente scp/tar "
        "del 2026-07-02), el listado falla y el borrado no llega a correr. Sin él, "
        "un paquete corrupto deja el VPS sin fuentes Y sin nada que extraer."
    )


# ---------------------------------------------------------------------------
# 2. El backend NO borra — y este es el test caro
# ---------------------------------------------------------------------------


def test_el_deploy_del_backend_nunca_vacia_su_directorio():
    """La asimetría es deliberada y hay secretos de producción en juego.

    `/opt/mealfit/backend/.env` tiene los valores propios del VPS (`REDIS_URL`,
    `CRON_SECRET`, claves de proveedor) y el tar lo excluye A PROPÓSITO para no
    pisarlos con los locales. Un `rm -rf` "simétrico" ahí no dejaría un archivo
    de más: dejaría el backend sin credenciales, en un deploy que parece normal.

    Por eso el guard no vigila que el fix esté puesto, sino que NO se extienda.
    """
    prohibidos = [
        ruta
        for ruta in _rutas_que_se_borran(_codigo_sin_comentarios())
        if ruta.rstrip("/") == "/opt/mealfit" or ruta.startswith("/opt/mealfit/backend")
    ]
    assert not prohibidos, (
        "[P2-DEPLOY-DELETES] El deploy borra dentro de /opt/mealfit/backend: "
        f"{prohibidos}\n\n"
        "NO se hace, y no es una omisión: ahí vive el `.env` de producción que el "
        "tar excluye a propósito. Vaciar ese directorio borra los secretos del "
        "VPS en un deploy de aspecto rutinario — el peor sitio para descubrir un "
        "fallo.\n\n"
        "Si de verdad hace falta limpiar huérfanos del backend, hazlo con una "
        "lista explícita de rutas, nunca vaciando el directorio."
    )


# ---------------------------------------------------------------------------
# 3. La nota del encabezado dice la verdad
# ---------------------------------------------------------------------------


def test_la_ayuda_del_script_documenta_la_asimetria():
    """Una nota que describe el comportamiento viejo es peor que ninguna nota.

    El encabezado decía «este script no borra del VPS; bórralo a mano». Si se
    queda así después del fix, el operador que la lea seguirá borrando a mano en
    el frontend (inofensivo) y, lo importante, no sabrá que en el BACKEND esa
    instrucción sigue siendo obligatoria.
    """
    texto = _DEPLOY.read_text(encoding="utf-8", errors="replace")
    cabecera = texto.split("#>")[0]

    assert "P2-DEPLOY-DELETES" in cabecera, (
        "[P2-DEPLOY-DELETES] La nota de «Borrados» del encabezado no cita el "
        "marker. Sin él, la nota y el comportamiento pueden divergir sin que "
        "nadie lo note al leer el script."
    )
    assert re.search(r"backend", cabecera, re.IGNORECASE), (
        "[P2-DEPLOY-DELETES] La nota de «Borrados» debe decir explícitamente que "
        "en el BACKEND sigues teniendo que borrar a mano. Es la mitad que se "
        "olvida: el frontend ya se limpia solo, así que el lector asume que todo "
        "se limpia solo."
    )
