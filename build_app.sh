#!/bin/bash
# ============================================================
# MMPS App Builder — paste this ENTIRE script into Terminal
#
# Builds MMPS.app from MMPSv2.py with a generated .icns icon.
# Handles everything: Homebrew, Python, deps, icon, packaging.
#
# Output: ~/Desktop/MMPS.app
# ============================================================

set -e

# =========================
# CONFIG — edit these paths
# =========================
APP_NAME="MMPS"
SCRIPT_PATH="$HOME/Microglia/MMPSv2.py"
BUNDLE_ID="com.mmps.microglia"
APP_VERSION="2.0"
OUTPUT_DIR="$HOME/Desktop"

# =========================
# 1. Homebrew
# =========================
echo ""
echo "========================================"
echo "  MMPS App Builder"
echo "========================================"

echo ""
echo "[1/6] Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "  OK"
fi

# =========================
# 2. Python
# =========================
echo ""
echo "[2/6] Checking Python..."
if ! brew list python@3 &>/dev/null 2>&1; then
    brew install python@3
fi

PYTHON="$(brew --prefix python@3)/bin/python3"
[[ ! -x "$PYTHON" ]] && PYTHON="$(brew --prefix)/bin/python3"
[[ ! -x "$PYTHON" ]] && PYTHON="$(which python3)"
echo "  $($PYTHON --version)"

# =========================
# 3. Virtual environment
# =========================
echo ""
echo "[3/6] Setting up build environment..."
BUILD_DIR=$(mktemp -d)
"$PYTHON" -m venv "$BUILD_DIR/venv"
source "$BUILD_DIR/venv/bin/activate"
pip install --upgrade pip --quiet

# =========================
# 4. Dependencies
# =========================
echo ""
echo "[4/6] Installing dependencies..."
pip install --quiet \
    PyQt5 numpy Pillow tifffile scikit-image \
    scipy matplotlib opencv-python pyinstaller

# =========================
# 5. Generate .icns icon
# =========================
echo ""
echo "[5/6] Generating app icon..."

ICONSET_DIR="$BUILD_DIR/${APP_NAME}.iconset"
mkdir -p "$ICONSET_DIR"

# Generate icon PNGs using Python/Pillow (same microglia design as the app)
"$BUILD_DIR/venv/bin/python3" - "$ICONSET_DIR" <<'PYEOF'
import sys, math
from PIL import Image, ImageDraw

def draw_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    scale = size / 256.0

    # Branches
    branches = [
        (0, 90), (45, 80), (90, 85), (135, 75),
        (180, 88), (225, 82), (270, 78), (315, 86),
        (22, 60), (67, 55), (112, 65), (157, 58),
        (202, 62), (247, 57), (292, 63), (337, 60),
    ]
    w = max(2, int(5 * scale))
    for angle_deg, length in branches:
        a = math.radians(angle_deg)
        x1 = cx + int(30 * scale * math.cos(a))
        y1 = cy + int(30 * scale * math.sin(a))
        x2 = cx + int(length * scale * math.cos(a))
        y2 = cy + int(length * scale * math.sin(a))
        draw.line([(x1, y1), (x2, y2)], fill=(100, 180, 255), width=w)
        for fork in (-25, 25):
            fa = a + math.radians(fork)
            fx = x2 + int(20 * scale * math.cos(fa))
            fy = y2 + int(20 * scale * math.sin(fa))
            draw.line([(x2, y2), (fx, fy)], fill=(100, 180, 255), width=w)

    # Soma
    r = int(28 * scale)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 140, 220))
    # Nucleus
    nr = int(9 * scale)
    draw.ellipse([cx - nr, cy - nr - int(2 * scale),
                  cx + nr, cy + nr - int(2 * scale)], fill=(180, 220, 255))
    return img

out = sys.argv[1]
for size in [16, 32, 64, 128, 256, 512, 1024]:
    icon = draw_icon(size)
    icon.save(f"{out}/icon_{size}x{size}.png")
    if size <= 512:
        icon2x = draw_icon(size * 2)
        icon2x.save(f"{out}/icon_{size}x{size}@2x.png")
PYEOF

# Convert to .icns using macOS iconutil
ICNS_PATH="$BUILD_DIR/${APP_NAME}.icns"
iconutil -c icns "$ICONSET_DIR" -o "$ICNS_PATH"
echo "  Icon created"

# =========================
# 6. Build .app
# =========================
echo ""
echo "[6/6] Building ${APP_NAME}.app..."

pyinstaller \
    --noconfirm \
    --windowed \
    --name "$APP_NAME" \
    --icon "$ICNS_PATH" \
    --osx-bundle-identifier "$BUNDLE_ID" \
    --hidden-import PyQt5 \
    --hidden-import PyQt5.QtWidgets \
    --hidden-import PyQt5.QtCore \
    --hidden-import PyQt5.QtGui \
    --hidden-import numpy \
    --hidden-import PIL \
    --hidden-import PIL.Image \
    --hidden-import tifffile \
    --hidden-import skimage \
    --hidden-import skimage.restoration \
    --hidden-import skimage.color \
    --hidden-import skimage.measure \
    --hidden-import scipy \
    --hidden-import scipy.ndimage \
    --hidden-import scipy.stats \
    --hidden-import matplotlib \
    --hidden-import matplotlib.path \
    --hidden-import cv2 \
    --exclude-module tkinter \
    --distpath "$OUTPUT_DIR" \
    --workpath "$BUILD_DIR/build" \
    --specpath "$BUILD_DIR" \
    "$SCRIPT_PATH"

deactivate

# Clean up temp build dir
rm -rf "$BUILD_DIR"

echo ""
echo "========================================"
echo "  BUILD COMPLETE"
echo "========================================"
echo ""
echo "  ${OUTPUT_DIR}/${APP_NAME}.app"
echo ""
echo "  Double-click to run, or:"
echo "    open ${OUTPUT_DIR}/${APP_NAME}.app"
echo ""
echo "  To install to Applications:"
echo "    mv ${OUTPUT_DIR}/${APP_NAME}.app /Applications/"
echo ""
