# El libro de migraciones (`public.schema_migrations`)

[P2-I18N-MIGRACIONES-SIN-LIBRO · 2026-08-23] Una fila por fichero de `migrations/` aplicado a
Neon. Lo escribe `scripts/apply_migration.py`; nadie más.

## Por qué existe

Hasta el 2026-08-23 el runner ejecutaba el fichero y no dejaba rastro. «¿Está aplicada?» era
una auditoría a mano contra `information_schema` (tablas, índices, columnas, constraints,
funciones), que sólo ve DDL: 44 de los 110 ficheros son de DATOS (`UPDATE`, `DO $$`) y contra
esos la auditoría no ve nada. La migración de i18n del 21-ago (`locale` nullable) se aplicó
a mano y quedó anotada en una doc — que es donde las cosas se pierden.

Medido ese día con la auditoría: **110 ficheros y UNO sin aplicar en producción**,
`p3_country_db_check_2026_08_22.sql` (índice + `CHECK` de país), sin que nada lo dijera.

## Los tres verbos

```bash
# ¿Qué falta? (solo lectura; exit 0 = al día, 4 = hay pendientes o cambiadas, 3 = no hay libro)
python backend/scripts/apply_migration.py --status

# Aplicar Y anotar (dry-run sin --apply)
python backend/scripts/apply_migration.py migrations/x.sql --apply --note "por qué"

# Anotar SIN ejecutar: aplicada a mano, o superseded (di cómo lo verificaste)
python backend/scripts/apply_migration.py migrations/x.sql --record --note "aplicada a mano el 22-ago, verificado: \d user_profiles"
```

`--status` distingue tres estados por fichero: **al día** (fila con el mismo sha256),
**PENDIENTE** (sin fila) y **aplicada con OTRO contenido** (la fila tiene otro sha256: el
fichero cambió después de aplicarse — este repo los edita para añadir sanity checks; revisa el
diff y re-aplica, que es idempotente, o `--record`).

## El backfill del 2026-08-23

Las 109 anteriores al libro se anotaron con `--record` y una `note` que dice CÓMO se verificó
cada grupo, porque «aplicada» sin evidencia es la misma mentira con una tabla:

| Grupo | Nota | Verificación |
|---|---|---|
| DDL (`CREATE`/`ALTER`) | `objetos verificados en information_schema/pg_catalog` | Auditoría por regex: cada tabla/índice/columna/constraint/función que el fichero crea existe en Neon. |
| Datos (sin DDL) | `asumida aplicada: el producto corre sobre ella` | No hay objeto que mirar. Es una suposición, y la nota lo dice. |
| `p1_audit_1_drop_dead_webhook_trigger.sql` | `migración de DROP, ausencia del objeto verificada` | El trigger y la función NO existen: eso es lo que la migración hace. |
| `p1_form_9_health_profile_jsonb_merge.sql`, `p1_new_5_user_fk_cascade_consolidate.sql` | `superseded … no aplica en Neon` | Era Supabase: una RPC sustituida por `PATCH /api/profile` y FKs a `auth.users`, esquema que en Neon no existe. |
| `p3_country_db_check_2026_08_22.sql` | **sin fila, PENDIENTE** | Es del plan de países; la aplica quien lo lleve, con `--apply`. |

Dos falsos positivos de la auditoría que NO son pendientes: `p0_3_backfill_plan_anchors.sql`
(la constraint que «falta» está citada en un comentario del fichero) y las constraints
`permite`/`*_user_id_fkey` de `p1_new_5` (regex sobre prosa y sobre un esquema que no existe).

## Lo que sigue sin resolver

- Nada IMPIDE aplicar a mano fuera del runner. El libro registra lo que pasa por él; un
  `psql` suelto sigue siendo invisible hasta el siguiente `--status`.
- `--status` no mira la base, mira el libro. Una fila anotada con `--record` vale lo que
  valga su `note`.
- Test: [`test_p2_i18n_migraciones_sin_libro.py`](../tests/test_p2_i18n_migraciones_sin_libro.py)
  (idempotencia de la migración, paridad de las dos copias, `clasificar()` pura, el orden
  ejecutar→anotar, y que sin libro se avisa en vez de reventar).
