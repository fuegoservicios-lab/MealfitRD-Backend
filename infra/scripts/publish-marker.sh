#!/bin/bash
# Publica _LAST_KNOWN_PFIX (leído de app.py) al KV app_kv_store.expected_last_known_pfix.
# Mantiene sincronizado el detector de deploy-lag (P0-1) tras cada deploy de backend,
# evitando la alerta `deploy_lag_drift_vs_expected`. Idempotente.
set -e
cd /opt/mealfit/backend
/home/ubuntu/miniforge3/envs/mealfit/bin/python -c "from dotenv import load_dotenv; load_dotenv(); import runpy; runpy.run_path('scripts/publish_pfix_marker.py', run_name='__main__')"
