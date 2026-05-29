#!/bin/bash
# ============================================================================
# Build MMPSv2 as a macOS .app bundle using PyInstaller
#
# Prerequisites:
#   pip install pyinstaller PyQt5 numpy scipy scikit-image opencv-python-headless \
#               tifffile Pillow matplotlib
#
# Usage:
#   cd /path/to/Microglia
#   bash build_app.sh
#
# Output:
#   dist/MMPSv2.app    — double-click to run
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  Building MMPSv2.app"
echo "======================================"

# Check for required files
if [ ! -f "MMPSv2.12.py" ]; then
    echo "ERROR: MMPSv2.12.py not found in current directory"
    echo "  Copy it here or run this script from the directory containing it"
    exit 1
fi

# Look for icon
ICON=""
for candidate in \
    "$HOME/Desktop/MMPS.icns" \
    "$HOME/MMPS.icns" \
    "$HOME/Downloads/MMPS.icns" \
    "./MMPS.icns"; do
    if [ -f "$candidate" ]; then
        ICON="$candidate"
        echo "Found icon: $ICON"
        break
    fi
done

if [ -z "$ICON" ]; then
    echo "WARNING: MMPS.icns not found. Building without custom icon."
    echo "  Place MMPS.icns in this directory or on your Desktop to include it."
fi

# Check PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

# Check key dependencies are installed
echo "Checking dependencies..."
python3 -c "
import PyQt5, numpy, scipy, skimage, cv2, tifffile, PIL, matplotlib
print('All dependencies found')
" || {
    echo ""
    echo "Missing dependencies. Install with:"
    echo "  pip install PyQt5 numpy scipy scikit-image opencv-python-headless tifffile Pillow matplotlib"
    exit 1
}

# Copy icon to build directory if found elsewhere
if [ -n "$ICON" ] && [ ! -f "MMPS.icns" ]; then
    cp "$ICON" ./MMPS.icns
    echo "Copied icon to build directory"
fi

# Clean previous builds
rm -rf build/MMPSv2 dist/MMPSv2 dist/MMPSv2.app

echo ""
echo "Running PyInstaller..."
echo ""

# Build using the spec file
pyinstaller MMPSv2.spec --noconfirm

echo ""
echo "======================================"

if [ -d "dist/MMPSv2.app" ]; then
    APP_SIZE=$(du -sh "dist/MMPSv2.app" | cut -f1)
    echo "  BUILD SUCCESSFUL"
    echo ""
    echo "  App:  dist/MMPSv2.app  ($APP_SIZE)"
    echo ""
    echo "  To run:  open dist/MMPSv2.app"
    echo "  To install: drag dist/MMPSv2.app to /Applications"
    echo ""
    echo "  NOTE: On first launch macOS may block it."
    echo "  Fix: System Settings > Privacy & Security > Open Anyway"
    echo "  Or:  Right-click the app > Open"
    echo "======================================"
else
    echo "  BUILD FAILED"
    echo "  Check the output above for errors"
    echo "======================================"
    exit 1
fi
