# Panel admin · Cuentas, actividad y cuentas de prueba

[P1-PLAN-LOTE-830 … 838 · 2026-09-29] Doc canónica del bloque 830-839. Spec: `docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md`
(raíz del workspace) · plan: `docs/superpowers/plans/2026-09-29-admin-cuentas-pruebas.md`. El panel de base (quién entra,
el rastro, las métricas) está en [`panel_admin.md`](panel_admin.md).

Tests: `test_p1_plan_lote_830.py` (las marcas), `_831.py` (lista, ficha, CSV y rutas), `_832.py` (detalle de prueba),
`_835.py` (lo que ve la persona), `_837.py` (ajustes e historial), `_838.py` (este doc y el script).

## Qué es

En `/admin` → **Cuentas** el dueño ve TODAS las cuentas con su correo, su actividad **en números** y los **ajustes** que
cada persona tiene encendidos o apagados (y cuándo los cambió). En las cuentas que él marca como **de prueba** —con un
motivo, y que la propia persona ve en la app— abre además su **contenido**. Todo detrás del interruptor
`MEALFIT_ADMIN_TEST_ACCOUNTS` (apagado por defecto).

## Tablas

Las migraciones son idénticas en `migrations/` y `backend/migrations/` y se aplican con `scripts/apply_migration.py`.

| Tabla | Migración | Qué guarda |
|---|---|---|
| `public.cuentas_de_prueba` | `p1_plan_lote_830_cuentas_de_prueba_2026_09_29.sql` | Una fila por marca: `marcada_por`, `marcada_at`, `motivo` (3-300), `aviso_visto_at`, `quitada_at`, `quitada_por`, `quitada_por_la_persona`, `motivo_quitar`. Índice único parcial `WHERE quitada_at IS NULL`: **una sola marca viva por cuenta**. Quitar NO borra: la tabla es su propio historial. |
| `public.ajustes_cambios` | `p1_plan_lote_837_ajustes_cambios_2026_09_29.sql` | Un cambio de ajuste por fila: `clave`, `antes`, `despues` (jsonb), `origen` (`app` / `coach` / `sistema`), `at`. La llena un trigger. |
| `user_profiles.ajustes_dispositivo` | la misma 837 | jsonb `{"web": {…, "at"}, "ios": …, "android": …}`: lo que solo existe en el teléfono (tema, permiso de notificaciones…). Solo lo escribe `ajustes_cuenta.guardar_dispositivo`, con una lista cerrada de claves. El trigger NO lo vigila. |

Las tres cuelgan de `user_profiles` (`ON DELETE CASCADE`: se van con la cuenta) y salen en la exportación «Mis datos»;
`cuentas_de_prueba` sin `marcada_por` ni `quitada_por`, que son ids del personal.

## Interruptores

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_ADMIN_TEST_ACCOUNTS` | `False` | El maestro. Apagado: las rutas nuevas del panel responden el mismo 404 que a un extraño, la ficha es exactamente la del lote 774, el perfil no trae `cuenta_de_prueba` y `PUT /api/profile/ajustes-dispositivo` responde `guardado: false` sin escribir. **`POST /api/profile/prueba/salir` funciona SIEMPRE**: dejar de ser vista nunca depende de un interruptor. |
| `MEALFIT_ADMIN_TEST_REQUIRE_NOTICE` | `True` | El detalle de una cuenta de prueba exige `aviso_visto_at`: que la persona ya haya visto el aviso en la app. |
| `MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS` | `730` | Plazo del historial de ajustes, acotado a [90, 3650]. Lo purga el cron diario `purge_ajustes_cambios`. |

El panel sigue necesitando `MEALFIT_ADMIN_PANEL=true` y el id del dueño en `MEALFIT_ADMIN_USER_IDS`.

## Rutas

Todas del panel bajo `/api/admin`, con `require_admin` a nivel de router (404 a quien no es admin). Los POST exigen la
cabecera `X-Admin-Accion: 1`. Ninguna lleva `verify_api_quota`: cero IA. El rastro es una fila de `admin_access_log`
escrita ANTES de responder (o de escribir): si no se puede anotar, 503 sin datos.

| Ruta | Limitador | Rastro (`accion`) |
|---|---|---|
| `GET /cuentas?buscar&orden&filtro&pagina` | `_CUENTAS_LISTA_LIMITER` (45/60) | `listar_cuentas` `{con_busqueda, orden, filtro, pagina, n}`: **nunca el texto buscado** (puede ser un correo) |
| `GET /cuentas.csv` | `_CUENTAS_LISTA_LIMITER` (45/60) | `exportar_cuentas` `{con_busqueda, filtro, n}` |
| `GET /cuentas/{user_id}` (ficha) | `_CUENTAS_LECTURA_LIMITER` (30/60) | `ver_cuenta` |
| `POST /cuentas/buscar` (por correo exacto) | `_CUENTAS_LECTURA_LIMITER` (30/60) | `buscar_cuenta` |
| `POST /cuentas/{user_id}/prueba` | `_CUENTAS_ESCRITURA_LIMITER` (20/60) | `marcar_prueba` (y `marcar_prueba_fallo` si el INSERT falla) |
| `POST /cuentas/{user_id}/prueba/quitar` | `_CUENTAS_ESCRITURA_LIMITER` (20/60) | `quitar_prueba` (y `quitar_prueba_fallo`) |
| `POST /pruebas/lote` (≤ 100 ids) | `_CUENTAS_ESCRITURA_LIMITER` (20/60) | `marcar_prueba`, una fila por cuenta marcada |
| `GET /cuentas/{user_id}/ajustes/historial` (≤ 500 cambios) | `_CUENTAS_LECTURA_LIMITER` (30/60) | `ver_ajustes` `{dias, n}` |
| `GET /ajustes/resumen` | `_CUENTAS_LECTURA_LIMITER` (30/60) | ninguno: es un agregado, sin cuentas admin |
| `GET /cuentas/{user_id}/prueba/<sección>` con `formulario` · `comidas` · `planes` · `planes/{plan_id}` · `conversaciones` · `conversaciones/{session_id}` · `adjuntos/{attachment_id}` · `actividad` | `_PRUEBA_DETALLE_LIMITER` (90/60) | `ver_prueba` `{seccion, objeto}` |

`POST /pruebas/lote` vive FUERA de `/cuentas/…` a propósito: `/cuentas/{user_id}/…` lo capturaría y daría 422. Los pares
(max, periodo) de los limitadores son únicos en el repo: Redis cuenta por par (`rl:<max>:<periodo>:<uid>`).

Códigos de las marcas (los devuelve el router tal cual): 409 `ya_marcada`, 409 `salio_ella` (la última marca la quitó
ELLA: hace falta `confirmar_vuelta` y que la persona haya pedido volver), 409 `sin_marca`, 404 `no_existe`, 422
`motivo` / `demasiadas`. Un fallo inesperado de la base al LEER es un 503 «No se pudo completar la acción»; si falla la escritura DESPUÉS de
anotar el rastro, 500 con su fila `…_fallo` en el registro.

Del lado de la PERSONA (`routers/user_data.py`, sesión verificada, sin cuota):

| Ruta | Limitador | Qué hace |
|---|---|---|
| `GET /api/profile` | ninguno propio (sesión verificada) | `profile.cuenta_de_prueba` `{desde, aviso_visto}` o `null`; con el knob apagado la clave ni aparece |
| `POST /api/profile/prueba/aviso-visto` | `_PRUEBA_LIMITER` (11/60) | anota, solo la primera vez y en una marca viva, que vio el aviso |
| `POST /api/profile/prueba/salir` | `_PRUEBA_LIMITER` (11/60) | sale del modo de prueba (idempotente; sin knob también) |
| `PUT /api/profile/ajustes-dispositivo` | `_AJUSTES_DISPOSITIVO_LIMITER` (6/60) | informa los ajustes del teléfono; claves y valores fuera de la lista se descartan |

## Qué ve cada vista, y por qué

- **Lista, CSV y ficha (CUALQUIER cuenta):** identidad, plan, país / idioma / modo y la actividad **en números**
  (comidas, planes, mensajes al coach, escaneos, gasto de IA, días activos, última actividad; en la ficha además el embudo
  alta → formulario → primer plan → primera comida → primer mensaje → primer escaneo) y los **ajustes**. Nunca contenido.
  «Última actividad» es lo que HIZO la persona —comida, mensaje suyo al coach, escaneo, peso, agua—: no cuenta la IA de
  fondo del generador, que se anota a nombre de la cuenta. El perfil de salud (peso, edad, sexo, alergias, dieta,
  condiciones, medicamentos, súper personalización, perfil clínico) NO es un ajuste: de los dos paneles solo se ve si están
  rellenos y cuándo se guardaron.
- **Historial y resumen de ajustes:** cuándo cambió cada ajuste y quién lo cambió; el resumen de Métricas cuenta, sin las
  cuentas admin, cuántas lo tienen encendido, apagado, automático o sin elegir.
- **Detalle de una cuenta de PRUEBA:** el formulario, las comidas (rango de fechas), los planes con sus bloques, las
  conversaciones con sus fotos y una línea de tiempo. Solo aquí sale contenido, y solo si: (1) el interruptor está
  encendido, (2) la marca está VIVA —se comprueba en CADA petición, sin caché: si la persona sale o el admin la quita, la
  siguiente vista ya da 403— y (3) la persona ya vio el aviso (409 `aviso_pendiente` si no; sin rastro y sin datos). Un plan,
  un hilo o una foto de OTRA cuenta responde 404 aunque exista. Orden de cada vista: interruptor (404) → parámetros (422)
  → marca (403/409) → fila `ver_prueba` → lectura. Solo lectura: no se edita nada ni se entra como ella.

**El porqué legal.** La política publicada solo permitía al soporte ver los datos de la cuenta y los regalos, y decía que
no se ven conversaciones, fotos ni perfil de salud. Por eso el 29-sep se ampliaron (lote 836, en las dos copias: el apex y
`LegalPages.jsx`): Privacidad §5 (acceso del equipo: los números y ajustes de cualquier cuenta, y el contenido SOLO de las
cuentas de prueba «desde que usted ve ese aviso»), §2 (qué guardamos: ajustes, historial, marca de prueba), §9 (el rastro se
conserva 24 meses y, al borrar la cuenta, pierde el id y los motivos: por eso el texto libre del rastro va SOLO bajo las
claves `motivo` y `error`, que `db_profiles.delete_account_data` quita) y Protección de datos §5 (finalidad: probar y mejorar
el servicio, solo en cuentas de prueba avisadas). Decisión del dueño: **sin aviso por correo**; el aviso de la app es la
única notificación y llega antes de cualquier acceso al contenido (`MEALFIT_ADMIN_TEST_REQUIRE_NOTICE`). Marcar a un cliente
real le enseña el aviso y queda quién lo hizo y por qué; si sale, volver a marcarlo exige confirmarlo aparte.

## El historial de ajustes: el trigger y el origen

- **Registro SSOT:** `backend/ajustes_cuenta.py::REGISTRO` (un `Ajuste(clave, etiqueta, grupo, fuente, tipo, por_defecto)`
  por ajuste) lo leen la ficha, el resumen y el CSV. Una clave `avisos_*` de `health_profile` que no esté en el registro
  sale igual en «Otros ajustes» con su valor tal cual.
- **Trigger:** `trg_ajustes_cambios` (`AFTER UPDATE OF <columnas vigiladas>, health_profile ON public.user_profiles FOR
  EACH ROW`, función `public.registrar_cambio_ajustes()` con `SET search_path = ''`). Su `WHEN` compara SOLO las columnas y
  claves vigiladas (`ajustes_cuenta.COLUMNAS_VIGILADAS` y `CLAVES_PERFIL_VIGILADAS`): un UPDATE de `fact_locked_at` ni se
  considera. Escribe una fila por clave que cambia y cubre a TODOS los escritores sin tocarlos. Un fallo suyo es un WARNING:
  nunca tumba el UPDATE de la persona.
- **Origen (`app` / `coach` / `sistema`):** el escritor lo fija EN LA MISMA SENTENCIA con `UPDATE … FROM (SELECT
  set_config('mealfit.origen_ajuste', '<origen>', true)) AS _origen WHERE …` (`ajustes_cuenta.sql_con_origen`; el bloque
  `origen_de_ajustes("coach")` lo aplica a los escritores compartidos). Funciona porque el backend va en autocommit: la
  transacción ES la sentencia, el trigger corre al final de ella y la variable desaparece con ella (no se arrastra por el
  pool). NO sirve un `SELECT set_config(…)` suelto antes del UPDATE. Sin marca, `app`. La Nevera que se apaga pasando
  `nevera_auto_off_at` de NULL a un valor en el mismo UPDATE es `sistema` aunque nadie lo marque.

**Límites conocidos:**
- Solo `cambiar_ajuste_de_la_app` corre en el origen `coach`. Las demás herramientas del coach (`update_form_field` con el
  país o el presupuesto) y el idioma que se le pide —lo guarda el cliente— quedan como `app`, que el panel etiqueta «la persona».
- El WARNING del trigger sale en el log de Postgres, no en el de la app. Por eso la pasada diaria de la purga deja un
  **canario** en el log de la app (`canario del historial de ajustes: N cambios, el último: …`): un `max(at)` que no avanza
  con la app en uso es la señal de una tabla que dejó de llenarse.
- Los ajustes del dispositivo llegan cuando la persona abre la app después del despliegue (a lo sumo una vez al día): una
  cuenta que no la abre sale «sin elegir». `unidad_altura` no distingue una elección de un valor por defecto.
- La ventana de las comidas del detalle rebasa las fechas pedidas (12 h antes, 14 h después): los días son los del admin,
  las filas están en UTC y las personas viven entre UTC−4 y UTC+2. En los extremos puede aparecer una comida del día vecino,
  con su hora UTC entera.
- La lista calcula los números de TODAS las cuentas en cada página (subconsultas por cuenta): sobra para cientos; con miles
  habría que paginar antes de calcular. El CSV corta en 5.000 filas.

## Cómo añadir un ajuste nuevo al registro

1. Añade su `Ajuste(...)` a `REGISTRO` (`ajustes_cuenta.py`), con clave y etiqueta ÚNICAS y un `grupo` de `GRUPOS`.
   `test_p1_plan_lote_837.py::test_el_registro_tiene_forma_y_cubre_lo_que_pide_el_spec` lo comprueba.
2. Según su `fuente`: una `columna` debe existir en una migración (`ADD COLUMN`); una `tabla`, en un `CREATE|ALTER TABLE`; un
   prefijo `kv` debe estar en `db_profiles._USER_SCOPED_KV_PREFIXES` (si no, «Eliminar cuenta» no lo limpia); una clave de
   `dispositivo` debe estar en `CLAVES_DISPOSITIVO` (y la app debe informarla).
3. Si es una columna o una clave de `health_profile` cuyo cambio quieres en el historial: añádela a `COLUMNAS_VIGILADAS` /
   `CLAVES_PERFIL_VIGILADAS` y RE-EMITE el trigger en una migración nueva (idempotente, copiada a las DOS carpetas de
   migraciones): los dos `FOREACH` de la función y el `WHEN`. `test_las_claves_vigiladas_son_las_del_trigger_y_todas_tienen_etiqueta`
   compara las tuplas de Python con la migración; ajústalo a la migración vigente.
4. Si es una `avisos_*` que escribe la app, `test_cada_clave_avisos_que_escribe_la_app_esta_en_el_registro` falla hasta que
   la registres (mientras tanto sale sola en «Otros ajustes»).
5. NUNCA registres una clave del perfil de salud (`test_el_perfil_de_salud_no_es_un_ajuste`): un ajuste es modo de uso, no
   contenido. El CSV y el resumen salen del registro solos; `test_p1_plan_lote_831.py` vigila que no haya columnas de contenido.

## Despliegue, en este orden

1. **Migraciones** (con el interruptor apagado no cambia nada visible): `python backend/scripts/apply_migration.py
   migrations/p1_plan_lote_830_cuentas_de_prueba_2026_09_29.sql --apply` y lo mismo con la `…837_ajustes_cambios…`
   (la 837 pide antes las de `plan_mode`, la Nevera, `locale` y `user_consents`: su primer bloque `DO` lo dice). Comprueba
   `--status` (0 pendientes) y el trigger dentro de `BEGIN; … ROLLBACK;` sobre una cuenta: un UPDATE de
   `water_tracker_enabled` deja una fila `app`, con `set_config('mealfit.origen_ajuste', 'coach', true)` en el mismo UPDATE
   deja `coach`, y uno de `fact_locked_at` no deja ninguna.
2. **Desplegar** backend y frontend (`deploy-mealfit.ps1`). Con el interruptor apagado, el despliegue es invisible.
3. **Publicar los textos legales** (lote 836) en las dos copias: el apex y `LegalPages.jsx`.
4. **Encender el interruptor:** `MEALFIT_ADMIN_TEST_ACCOUNTS=true` en `/opt/mealfit/backend/.env` y reiniciar
   `mealfit-backend`. Comprobar: `GET /api/admin/cuentas` con la sesión del dueño → 200, con otra cuenta → 404. Marcha atrás:
   apagarlo (todo vuelve a 404; `salir` sigue funcionando).
5. **Marcar las cuentas existentes** con el script de abajo.
6. La persona ve el aviso una vez (con el detalle aún cerrado: «Esperando a que vea el aviso»); en cuanto lo ve, la marca
   pasa a `activa` y el detalle se abre.

## Marcar las cuentas existentes: `scripts/marcar_cuentas_prueba.py`

Lo mismo que `POST /pruebas/lote`, con la MISMA función (`cuentas_prueba.marcar_varias`): mismas reglas y mismo rastro
(una fila `marcar_prueba` por cuenta marcada, antes de escribir).

```
python scripts/marcar_cuentas_prueba.py --todas --motivo "beta cerrada" --admin <uuid>            # simulación (por defecto)
python scripts/marcar_cuentas_prueba.py --todas --excluir-correos a@x.com --motivo "…" --admin <uuid> --aplicar
python scripts/marcar_cuentas_prueba.py --correos ana@x.com,beto@y.com --motivo "amigos" --admin <uuid> --aplicar
```

- **Sin `--aplicar` no escribe nada:** lista las cuentas que marcaría y las excluidas.
- Lo primero que hace es `cuentas_prueba.activo()`: con el interruptor apagado se NIEGA (salida 2).
- `--admin` debe estar en `MEALFIT_ADMIN_USER_IDS`; `--motivo` de 3 a 300 caracteres; `--todas` son TODAS las filas de
  `user_profiles` (también el personal y el revisor de Apple: `--excluir-correos`, o quitar la marca luego desde el panel).
  Un correo pedido o excluido que no corresponde a ninguna cuenta se rechaza: una errata en una exclusión marcaría a quien
  no debía.
- Marca en tandas de 100 e imprime cuántas quedaron `marcada`, `ya_marcada`, `salio_ella` (no se remarcan solas: se marcan
  una a una desde el panel con la confirmación de la vuelta) y `no_existe`. Si una tanda falla a mitad, corta (salida 1):
  las ya marcadas quedan con su rastro y relanzarlo las devuelve como `ya_marcada`.
- Abre el pool de Neon como pide el SOP de SQL forense y solo imprime ASCII.

tooltip-anchor: cuentas_de_prueba, ajustes_cambios, ajustes_dispositivo, marcar_cuentas_prueba (test_p1_plan_lote_838.py)
