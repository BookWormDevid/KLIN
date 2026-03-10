#!/bin/sh

set -euo pipefail

readonly admin_user="${POSTGRES_USER:?POSTGRES_USER is required}"
readonly app_db_name="${APP_DB_NAME:?APP_DB_NAME is required}"
readonly app_db_user="${APP_DB_USER:?APP_DB_USER is required}"
readonly app_db_password="${APP_DB_PASSWORD:?APP_DB_PASSWORD is required}"
readonly airflow_db_name="${AIRFLOW_DB_NAME:?AIRFLOW_DB_NAME is required}"
readonly airflow_db_user="${AIRFLOW_DB_USER:?AIRFLOW_DB_USER is required}"
readonly airflow_db_password="${AIRFLOW_DB_PASSWORD:?AIRFLOW_DB_PASSWORD is required}"
readonly metrics_db_name="${ML_METRICS_DB_NAME:-ml}"

psql_base=(
  psql
  -h
  postgresql
  -U
  "${admin_user}"
  -d
  postgres
  -v
  ON_ERROR_STOP=1
)

until pg_isready -h postgresql -U "${admin_user}" -d postgres >/dev/null 2>&1; do
  sleep 2
done

ensure_role() {
  local role_name="$1"
  local role_password="$2"

  if ! "${psql_base[@]}" -tAc \
    "SELECT 1 FROM pg_roles WHERE rolname = '${role_name}'" | grep -q 1; then
    "${psql_base[@]}" -c \
      "CREATE ROLE \"${role_name}\" LOGIN PASSWORD '${role_password}'"
    return
  fi

  "${psql_base[@]}" -c \
    "ALTER ROLE \"${role_name}\" WITH LOGIN PASSWORD '${role_password}'"
}

ensure_database() {
  local db_name="$1"
  local db_owner="$2"

  if ! "${psql_base[@]}" -tAc \
    "SELECT 1 FROM pg_database WHERE datname = '${db_name}'" | grep -q 1; then
    "${psql_base[@]}" -c \
      "CREATE DATABASE \"${db_name}\" OWNER \"${db_owner}\""
    return
  fi

  "${psql_base[@]}" -c \
    "ALTER DATABASE \"${db_name}\" OWNER TO \"${db_owner}\""
}

repair_database_ownership() {
  local db_name="$1"
  local db_owner="$2"

  psql -h postgresql -U "${admin_user}" -d "${db_name}" -v ON_ERROR_STOP=1 <<SQL
ALTER SCHEMA public OWNER TO "${db_owner}";
DO \$\$
DECLARE
  row record;
BEGIN
  FOR row IN
    SELECT tablename
    FROM pg_tables
    WHERE schemaname = 'public'
  LOOP
    EXECUTE format('ALTER TABLE public.%I OWNER TO %I', row.tablename, '${db_owner}');
  END LOOP;

  FOR row IN
    SELECT sequence_name
    FROM information_schema.sequences
    WHERE sequence_schema = 'public'
  LOOP
    EXECUTE format('ALTER SEQUENCE public.%I OWNER TO %I', row.sequence_name, '${db_owner}');
  END LOOP;

  FOR row IN
    SELECT table_name
    FROM information_schema.views
    WHERE table_schema = 'public'
  LOOP
    EXECUTE format('ALTER VIEW public.%I OWNER TO %I', row.table_name, '${db_owner}');
  END LOOP;

  FOR row IN
    SELECT matviewname
    FROM pg_matviews
    WHERE schemaname = 'public'
  LOOP
    EXECUTE format('ALTER MATERIALIZED VIEW public.%I OWNER TO %I', row.matviewname, '${db_owner}');
  END LOOP;
END
\$\$;
GRANT ALL ON SCHEMA public TO "${db_owner}";
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "${db_owner}";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "${db_owner}";
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO "${db_owner}";
SQL
}

ensure_role "${app_db_user}" "${app_db_password}"
ensure_database "${app_db_name}" "${app_db_user}"
repair_database_ownership "${app_db_name}" "${app_db_user}"

ensure_role "${airflow_db_user}" "${airflow_db_password}"
ensure_database "${airflow_db_name}" "${airflow_db_user}"
repair_database_ownership "${airflow_db_name}" "${airflow_db_user}"

ensure_database "${metrics_db_name}" "${admin_user}"
