#!/bin/bash
# ============================================================
# Build MMPS.app — run from the Microglia folder
# ============================================================
set -e

# Generate MMPS.icns if it doesn't exist
if [[ ! -f MMPS.icns ]]; then
    echo "Generating MMPS.icns..."
    ICONSET="MMPS.iconset"
    mkdir -p "$ICONSET"
    python3 - "$ICONSET" <<'PYEOF'
import sys, math
from PIL import Image, ImageDraw
def draw_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    s = size / 256.0
    branches = [
        (0,90),(45,80),(90,85),(135,75),(180,88),(225,82),(270,78),(315,86),
        (22,60),(67,55),(112,65),(157,58),(202,62),(247,57),(292,63),(337,60),
    ]
    w = max(2, int(5 * s))
    for ang, l in branches:
        a = math.radians(ang)
        x1, y1 = cx + int(30*s*math.cos(a)), cy + int(30*s*math.sin(a))
        x2, y2 = cx + int(l*s*math.cos(a)), cy + int(l*s*math.sin(a))
        d.line([(x1,y1),(x2,y2)], fill=(100,180,255), width=w)
        for f in (-25, 25):
            fa = a + math.radians(f)
            d.line([(x2,y2),(x2+int(20*s*math.cos(fa)),y2+int(20*s*math.sin(fa)))], fill=(100,180,255), width=w)
    r = int(28*s)
    d.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(60,140,220))
    nr = int(9*s)
    d.ellipse([cx-nr,cy-nr-int(2*s),cx+nr,cy+nr-int(2*s)], fill=(180,220,255))
    return img
out = sys.argv[1]
for sz in [16,32,64,128,256,512,1024]:
    draw_icon(sz).save(f"{out}/icon_{sz}x{sz}.png")
    if sz <= 512:
        draw_icon(sz*2).save(f"{out}/icon_{sz}x{sz}@2x.png")
PYEOF
    iconutil -c icns "$ICONSET" -o MMPS.icns
    rm -rf "$ICONSET"
    echo "MMPS.icns created"
fi

# Build
python3 -m PyInstaller --onefile --windowed --name "MMPS" --icon MMPS.icns MMPSv2.py

echo ""
echo "Done: dist/MMPS.app"
