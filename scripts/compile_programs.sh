#!/usr/bin/env bash
# compile_programs.sh
# Compiles every .c file in programs/ to a binary in binaries/
# Usage: bash scripts/compile_programs.sh [jobs]

set -euo pipefail

PROG_DIR="programs"
BIN_DIR="binaries"
LOG="compile_errors.log"
JOBS="${1:-$(nproc)}"

mkdir -p "$BIN_DIR"
> "$LOG"

echo "[compile] Using $JOBS parallel jobs"
echo "[compile] Source: $PROG_DIR → $BIN_DIR"

compile_one() {
    src="$1"
    base="$(basename "$src" .c)"
    out="$BIN_DIR/$base"

    if gcc -O0 -o "$out" "$src" 2>>"$LOG"; then
        echo "OK $base"
    else
        echo "FAIL $base" | tee -a "$LOG"
    fi
}
export -f compile_one
export LOG BIN_DIR

# GNU parallel or xargs fallback
if command -v parallel &>/dev/null; then
    find "$PROG_DIR" -name "*.c" | parallel -j "$JOBS" compile_one {}
else
    find "$PROG_DIR" -name "*.c" | xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {}
fi

compiled=$(find "$BIN_DIR" -maxdepth 1 -type f | wc -l)
echo ""
echo "[compile] Done — $compiled binaries in $BIN_DIR/"
errors=$(wc -l < "$LOG")
[ "$errors" -gt 0 ] && echo "[compile] $errors error lines logged → $LOG"
