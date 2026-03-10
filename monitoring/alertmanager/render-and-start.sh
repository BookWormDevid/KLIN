#!/bin/sh
set -eu

template_path="/etc/alertmanager/alertmanager.yml.tmpl"
rendered_path="/tmp/alertmanager.yml"
unresolved_path="/tmp/alertmanager.unresolved"
storage_path="${ALERTMANAGER_STORAGE_PATH:-/alertmanager}"

awk '
function esc(value) {
  gsub(/\\/, "\\\\", value)
  gsub(/&/, "\\\\&", value)
  return value
}
{
  line = $0
  for (key in ENVIRON) {
    replacement = esc(ENVIRON[key])
    gsub("\\$\\{" key "\\}", replacement, line)
  }
  print line
}' "$template_path" > "$rendered_path"

if awk 'match($0, /\$\{[A-Za-z_][A-Za-z0-9_]*\}/) { print NR ":" $0; found = 1 } END { exit found ? 0 : 1 }' "$rendered_path" > "$unresolved_path"; then
  echo "Unresolved placeholders remain in Alertmanager config:" >&2
  cat "$unresolved_path" >&2
  exit 1
fi

exec /bin/alertmanager --config.file="$rendered_path" --storage.path="$storage_path"
