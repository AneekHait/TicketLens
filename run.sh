#!/usr/bin/env bash
# TicketLens launcher for macOS and Linux. The counterpart to run.bat, and it
# follows the same shape: validate Python, build/repair .venv, install pinned
# deps, health-check the local LLM engine once per machine, then launch.
#
# Differences from run.bat that are platform facts, not choices:
#   * wheels/ ships a win_amd64 llama.cpp wheel only, so there is nothing local
#     to prefer here -- we always go to the prebuilt CPU index.
#   * repair_llm.ps1 / build_wheel.bat are PowerShell and Windows-only, so the
#     engine health check is inlined below instead of shelling out to them.
#   * llama-cpp-python 0.3.32 publishes no macOS x86_64 wheel (those stopped at
#     0.3.2), so Intel Macs must compile it. Handled explicitly further down.
#
# Usage:  ./run.sh            normal launch
#         ./run.sh --rebuild  discard .venv and reinstall from scratch
set -uo pipefail

# Resolve the real script directory even when invoked through a symlink, so the
# relative paths below (src/, requirements.txt) always resolve.
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
    DIR=$(cd -P "$(dirname "$SOURCE")" && pwd)
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
REPO=$(cd -P "$(dirname "$SOURCE")" && pwd)
cd "$REPO" || exit 1

VENV=.venv
VPY="$VENV/bin/python"
MARKER="$VENV/.deps_installed"
LLAMA_MARK="$VENV/.llama_ok"
QT_MARK="$VENV/.qt_ok"
WHEEL_INDEX_CPU="https://abetlen.github.io/llama-cpp-python/whl/cpu"
# Apple Silicon gets the Metal build. The cpu index publishes a macOS arm64
# wheel too, but it is compiled without Metal, so n_gpu_layers=-1 would be a
# silent no-op and the LLM would plod along on the CPU.
WHEEL_INDEX_METAL="https://abetlen.github.io/llama-cpp-python/whl/metal"
LLAMA_PIN="0.3.32"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    B=$'\033[1m'; RED=$'\033[31m'; YEL=$'\033[33m'; RST=$'\033[0m'
else
    B=''; RED=''; YEL=''; RST=''
fi
say()  { printf '%s[TicketLens]%s %s\n' "$B" "$RST" "$*"; }
warn() { printf '%s[TicketLens]%s %sWARNING:%s %s\n' "$B" "$RST" "$YEL" "$RST" "$*" >&2; }
die()  { printf '\n%sERROR:%s %s\n\n' "$RED" "$RST" "$*" >&2; exit 1; }

# --- version (single source of truth, read before Python exists) -------------
# Matches MAJOR.MINOR.PATCH only, which sidesteps quoting the quote characters.
APPVER=$(sed -n 's/^__version__.*=[^0-9]*\([0-9][0-9.]*\).*/\1/p' src/__init__.py 2>/dev/null | head -1)
APPVER=${APPVER:-unknown}

# --- platform ----------------------------------------------------------------
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
    Darwin) PLATFORM=macOS ;;
    Linux)  PLATFORM=Linux ;;
    *)      die "Unsupported OS '$OS'. Use run.bat on Windows; this script covers macOS and Linux." ;;
esac

cat <<BANNER

 ============================================================
   TicketLens   v$APPVER
   Local AI Ticket Analytics
   Privacy-first ticket clustering and insights - your ticket
   data stays on this machine (AI models download once, on
   first run, from Hugging Face).
   Free and open source, Apache-2.0 licensed.
   https://github.com/AneekHait/TicketLens
 ============================================================

BANNER

if [ "${1:-}" = "--rebuild" ]; then
    say "Rebuilding the virtual environment from scratch..."
    rm -rf "$VENV"
fi

# --- locate a supported interpreter (3.11-3.14) ------------------------------
# Only needed when the venv must be created; once it exists we use its own
# python and never consult PATH again.
FOUND_TOO_OLD=""
find_python() {
    local cand ver maj min
    for cand in python3.12 python3.13 python3.11 python3.14 python3 python; do
        command -v "$cand" >/dev/null 2>&1 || continue
        ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
        maj=${ver%%.*}
        min=${ver##*.}
        [ "$maj" = "3" ] || continue
        # Remember the newest too-old interpreter so the failure can name it
        # rather than claiming nothing is installed.
        if [ "$min" -lt 11 ]; then
            FOUND_TOO_OLD=$ver
            continue
        fi
        if [ "$min" -gt 14 ]; then
            warn "Python $ver is newer than the tested range (3.11-3.14)."
            warn "Some native packages may lack prebuilt wheels; continuing anyway."
        fi
        PYBIN=$cand
        PYVER=$ver
        return 0
    done
    return 1
}

# --- a .venv copied from another machine hardcodes an absolute interpreter ----
# path in pyvenv.cfg and will not run here. Same trap as on Windows: detect it
# and rebuild rather than failing later with a confusing error. Removing the
# venv also drops .deps_installed, so dependencies reinstall.
if [ -x "$VPY" ] && ! "$VPY" -c 'import sys' >/dev/null 2>&1; then
    say "The existing .venv is not usable on this machine (likely copied) - rebuilding..."
    rm -rf "$VENV"
fi

if [ ! -x "$VPY" ]; then
    if ! find_python; then
        if [ -n "$FOUND_TOO_OLD" ]; then
            die "Python $FOUND_TOO_OLD is too old.
  TicketLens needs CPython 3.11-3.14 (3.12 recommended); pandas 3.x requires 3.11+.
  Install a newer interpreter alongside it - this does not replace the system one:
    macOS : brew install python@3.12
    Debian: sudo apt install python3.12 python3.12-venv
    Fedora: sudo dnf install python3.12
  Then re-run this script."
        fi
        die "No supported Python found on PATH.
  TicketLens needs CPython 3.11-3.14 (3.12 recommended); pandas 3.x requires 3.11+.
    macOS : brew install python@3.12
    Debian: sudo apt install python3.12 python3.12-venv
    Fedora: sudo dnf install python3.12
  Then re-run this script."
    fi
    say "Using $PYBIN (Python $PYVER)"
    say "Creating virtual environment in $VENV ..."
    if ! "$PYBIN" -m venv "$VENV"; then
        die "Could not create the virtual environment.
  On Debian/Ubuntu the venv module ships separately: sudo apt install python3-venv"
    fi
fi

# --- first run: install the pinned dependency set ----------------------------
if [ ! -f "$MARKER" ]; then
    say "Installing dependencies into $VENV - this can take several minutes..."
    "$VPY" -m pip install --upgrade pip || die "Could not upgrade pip."

    # wheels/ holds a win_amd64 wheel only, so there is no local wheel to prefer
    # on this platform; always go to the prebuilt index.
    if [ "$PLATFORM" = macOS ] && [ "$ARCH" = "arm64" ]; then
        WHEEL_INDEX=$WHEEL_INDEX_METAL
        say "Apple Silicon: using the Metal llama.cpp build (GPU-accelerated labelling)."
    else
        WHEEL_INDEX=$WHEEL_INDEX_CPU
    fi
    PIP_ARGS=(-r requirements.txt --find-links wheels --extra-index-url "$WHEEL_INDEX")

    if [ "$PLATFORM" = macOS ] && [ "$ARCH" = "x86_64" ]; then
        # Checked against the index: llama-cpp-python publishes no macOS x86_64
        # wheel past 0.3.2, and requirements.txt pins $LLAMA_PIN. Forcing
        # --only-binary here would fail outright, so allow the source build and
        # say up front what it needs.
        warn "Intel Mac detected. No prebuilt llama.cpp wheel exists for macOS x86_64
             at the pinned version, so it will be compiled from source (a few
             minutes). This needs the Xcode command line tools."
        if ! xcode-select -p >/dev/null 2>&1; then
            die "Xcode command line tools are missing. Install them first:
    xcode-select --install
  Then re-run this script."
        fi
    else
        # Everywhere else a prebuilt wheel exists (manylinux/musllinux x86_64 and
        # aarch64, macOS arm64). Refuse a silent source build so a missing wheel
        # is a clear error rather than a surprise 20-minute compile.
        PIP_ARGS+=(--only-binary llama-cpp-python)
    fi

    # If a matching local wheel exists in wheels/, install it first to avoid upstream index bugs
    for local_whl in wheels/llama_cpp_python-$LLAMA_PIN-*.whl; do
        if [ -f "$local_whl" ]; then
            "$VPY" -m pip install --no-deps "$local_whl" || true
            break
        fi
    done

    if ! "$VPY" -m pip install "${PIP_ARGS[@]}"; then
        die "Dependency installation failed - see the messages above.
  Fix the issue and re-run this script to retry."
    fi
    printf 'ok' > "$MARKER"
    say "Setup complete."
    if [ "$PLATFORM" = macOS ] && [ "$ARCH" = "arm64" ]; then
        say "Tip: the embedding step can use the Apple GPU too --"
        say "     Settings -> Acceleration -> \"Apple Metal / MPS\". No install needed."
        say "     ./install_mlx.sh adds the experimental MLX engine on top."
    fi
fi

# --- one-time per-machine Qt preflight ---------------------------------------
# PySide6 wheels bundle Qt but not its system dependencies. On a headless or
# minimal Linux image the first symptom is "could not load the Qt platform
# plugin xcb" at launch, which says nothing about which package is missing.
if [ "$PLATFORM" = Linux ] && [ "$(cat "$QT_MARK" 2>/dev/null)" != "$(hostname)" ]; then
    if "$VPY" -c 'import PySide6.QtWidgets' >/dev/null 2>&1; then
        hostname > "$QT_MARK"
    else
        warn "PySide6 could not be imported. Qt needs system libraries pip does not install:
    Debian/Ubuntu: sudo apt install libgl1 libegl1 libxkbcommon-x11-0 libdbus-1-3 \\
                                    libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \\
                                    libxcb-randr0 libxcb-render-util0 libxcb-shape0
    Fedora/RHEL:   sudo dnf install mesa-libGL libxkbcommon-x11 xcb-util-cursor \\
                                    xcb-util-wm xcb-util-keysyms xcb-util-renderutil
  Install those and re-run. Attempting to launch anyway."
    fi
fi

# --- one-time per-machine AI engine health check -----------------------------
# Verify-only; this never compiles. Run in a subprocess on purpose: a wheel built
# for the wrong instruction set does not raise, it kills the interpreter with
# SIGILL, so only an exit code can detect it. The marker is stamped with the
# hostname so the check re-runs if the folder is copied to another machine.
if [ "$(cat "$LLAMA_MARK" 2>/dev/null)" != "$(hostname)" ]; then
    say "Verifying the local AI engine for this machine (one-time)..."
    if "$VPY" -c 'import llama_cpp' >/dev/null 2>&1; then
        hostname > "$LLAMA_MARK"
    else
        rc=$?
        if [ "$rc" -gt 128 ]; then
            warn "The llama.cpp engine crashed with signal $((rc - 128)) on import -- the
             prebuilt wheel does not match this CPU. Rebuild it for this machine:
               $VPY -m pip install --force-reinstall --no-binary llama-cpp-python \\
                 llama-cpp-python==$LLAMA_PIN"
        else
            warn "The local LLM is not enabled on this machine yet."
        fi
        warn "Clustering will run with keyword-based labels instead of AI labels."
    fi
fi

# --- launch -------------------------------------------------------------------
say "Launching..."
exec "$VPY" main.py
