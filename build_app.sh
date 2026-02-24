#!/bin/bash
# ============================================================
# Build MMPS.app for macOS (using Homebrew)
#
# Run from the repo root:
#     chmod +x build_app.sh
#     ./build_app.sh
#
# Output:  dist/MMPS.app
# ============================================================

set -e

echo "========================================"
echo "  MMPS App Builder (macOS + Homebrew)"
echo "========================================"

# ----------------------------------------------------------
# 1. Homebrew
# ----------------------------------------------------------
echo ""
echo "[1/5] Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "  Homebrew not found — installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "  Homebrew found: $(brew --prefix)"
fi

# ----------------------------------------------------------
# 2. Python
# ----------------------------------------------------------
echo ""
echo "[2/5] Checking Python..."
if ! brew list python@3 &>/dev/null 2>&1; then
    echo "  Installing Python 3 via Homebrew..."
    brew install python@3
else
    echo "  Python already installed"
fi

PYTHON="$(brew --prefix python@3)/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(brew --prefix)/bin/python3"
fi
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(which python3)"
fi
echo "  Using: $PYTHON ($($PYTHON --version))"

# ----------------------------------------------------------
# 3. Virtual environment
# ----------------------------------------------------------
echo ""
echo "[3/5] Setting up virtual environment..."
VENV_DIR=".venv_build"
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ----------------------------------------------------------
# 4. Dependencies
# ----------------------------------------------------------
echo ""
echo "[4/5] Installing dependencies..."
pip install --quiet \
    PyQt5 \
    numpy \
    Pillow \
    tifffile \
    scikit-image \
    scipy \
    matplotlib \
    opencv-python \
    pyinstaller

# ----------------------------------------------------------
# 5. Build
# ----------------------------------------------------------
echo ""
echo "[5/5] Building MMPS.app..."
pyinstaller MMPS.spec --noconfirm

# Clean up build artifacts (keep dist/)
rm -rf build

deactivate

echo ""
echo "========================================"
echo "  BUILD COMPLETE"
echo "========================================"
echo ""
echo "  App:     dist/MMPS.app"
echo ""
echo "  To run:        open dist/MMPS.app"
echo "  To install:    cp -r dist/MMPS.app /Applications/"
echo ""
