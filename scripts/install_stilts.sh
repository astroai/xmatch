#!/usr/bin/env bash
# Create a `stilts` wrapper from an installed TOPCAT application bundle.
set -euo pipefail

TOPCAT_APP_NAME="TOPCAT.app"
INSTALL_DIR="${STILTS_INSTALL_DIR:-$HOME/bin}"
STILTS_SCRIPT_PATH="${INSTALL_DIR}/stilts"
IFS=: read -r -a SEARCH_DIRS <<< "${TOPCAT_SEARCH_DIRS:-/Applications}"

find_topcat_jar() {
    local search_dir topcat_path jar_path
    for search_dir in "${SEARCH_DIRS[@]}"; do
        [[ -d "$search_dir" ]] || continue
        for topcat_path in "$search_dir/$TOPCAT_APP_NAME" "$search_dir"/*/"$TOPCAT_APP_NAME"; do
            [[ -d "$topcat_path" ]] || continue
            printf 'Found TOPCAT at: %s\n' "$topcat_path" >&2
            jar_path=$(find "$topcat_path/Contents/Resources" -type f -name 'topcat-extra.jar' -print -quit 2>/dev/null || true)
            if [[ -z "$jar_path" ]]; then
                jar_path=$(find "$topcat_path/Contents/Resources" -type f -name 'topcat-lite.jar' -print -quit 2>/dev/null || true)
            fi
            if [[ -n "$jar_path" ]]; then
                printf 'Using JAR: %s\n' "$jar_path" >&2
                printf '%s\n' "$jar_path"
                return 0
            fi
            printf 'Error: TOPCAT has no topcat-extra.jar or topcat-lite.jar: %s\n' "$topcat_path" >&2
            return 1
        done
    done
    printf 'Error: Could not find %s in configured search directories: %s\n' \
        "$TOPCAT_APP_NAME" "${SEARCH_DIRS[*]}" >&2
    return 1
}

TOPCAT_JAR_PATH=$(find_topcat_jar) || exit 1
mkdir -p "$INSTALL_DIR"
printf -v JAR_PATH_QUOTED '%q' "$TOPCAT_JAR_PATH"

cat > "$STILTS_SCRIPT_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
JAR_PATH=$JAR_PATH_QUOTED
if ! command -v java >/dev/null 2>&1; then
    echo "Error: Java command not found. Install Java to run STILTS." >&2
    exit 1
fi
if [[ ! -f "\$JAR_PATH" ]]; then
    echo "Error: TOPCAT JAR file not found at \$JAR_PATH" >&2
    exit 1
fi
exec java -jar "\$JAR_PATH" -stilts "\$@"
EOF
chmod +x "$STILTS_SCRIPT_PATH"
printf 'Created executable wrapper: %s\n' "$STILTS_SCRIPT_PATH"
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    printf 'Add this directory to PATH to run `stilts`: %s\n' "$INSTALL_DIR"
fi
