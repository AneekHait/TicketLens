#!/usr/bin/env bash
# Install the optional MLX (Apple Silicon) acceleration stack into the existing
# .venv. The counterpart to install_openvino.bat, and like it this is opt-in:
# TicketLens runs perfectly well without it.
#
# Read this before running: you probably do not need it. Apple GPU support for
# the two slow steps is already available with no extra install --
#   embeddings : Settings -> Acceleration -> "Apple Metal / MPS" (PyTorch MPS)
#   LLM labels : run.sh installs the Metal llama.cpp wheel on arm64 Macs
# MLX is a separate engine that may be faster again for some models. It is
# experimental here, and the embedding path is numerically verified against
# PyTorch on load and refused if it disagrees.
set -uo pipefail

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
    DIR=$(cd -P "$(dirname "$SOURCE")" && pwd)
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
REPO=$(cd -P "$(dirname "$SOURCE")" && pwd)
cd "$REPO" || exit 1

VPY=".venv/bin/python"

echo
echo " ============================================================"
echo "   TicketLens - MLX acceleration (Apple Silicon, experimental)"
echo " ============================================================"
echo

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: MLX only runs on Apple Silicon (arm64 macOS)."
    echo "       This machine is $(uname -s)/$(uname -m). Nothing to do."
    exit 1
fi

if [ ! -x "$VPY" ]; then
    echo "ERROR: no virtual environment at $VPY."
    echo "       Run ./run.sh once first to create it, then re-run this script."
    exit 1
fi

echo "LICENSE NOTE: mlx and mlx-lm are MIT, but mlx-embeddings is GPL-3.0."
echo "  TicketLens (Apache-2.0) does not bundle it - this script fetches it from"
echo "  PyPI into your own .venv. If you later redistribute a bundle containing"
echo "  it, that distribution must comply with GPL-3.0. Skip this script if you"
echo "  would rather not: the Apple GPU is already used via MPS and Metal."
echo
printf "Continue? [y/N]: "
read -r reply
case "$reply" in
    [Yy]*) ;;
    *) echo "Cancelled. Nothing was installed."; exit 0 ;;
esac
echo

echo "[TicketLens] Installing MLX into .venv ..."
if ! "$VPY" -m pip install -r requirements-mlx.txt; then
    echo
    echo "ERROR: MLX installation failed - see the messages above."
    exit 1
fi

echo
echo "[TicketLens] Verifying the install..."
"$VPY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src import mlx_backend as m
print("  " + m.describe())
if not (m.mlx_lm_available() or m.mlx_embeddings_available()):
    print("  MLX imported nothing usable; TicketLens will keep using the defaults.")
    raise SystemExit(1)
PYEOF
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "[TicketLens] Done. Restart TicketLens, then:"
    echo "   - Settings -> Acceleration -> \"MLX (Apple Silicon - experimental)\""
    echo "   - Settings -> AI Model     -> any \"MLX - ...\" entry"
    echo
    echo "   The MLX embedding backend is checked against PyTorch on load and"
    echo "   silently refused if the vectors disagree, so your clusters cannot"
    echo "   change without you being told."
else
    echo "[TicketLens] MLX did not verify. The app still works on its defaults."
fi
exit $rc
