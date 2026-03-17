# -*- mode: python ; coding: utf-8 -*-

import sys, os

# Icon file — .icns for macOS, .ico for Windows (optional — skipped if not found)
if sys.platform == 'darwin':
    ICON = 'MMPS.icns' if os.path.exists('MMPS.icns') else None
else:
    ICON = 'MMPS.ico' if os.path.exists('MMPS.ico') else None

a = Analysis(
    ['MMPSv2.12.py'],
    pathex=[],
    binaries=[],
    datas=[('3DMicroglia.py', '.')],
    hiddenimports=['PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui', 'numpy', 'PIL', 'PIL.Image', 'tifffile', 'imagecodecs', 'imagecodecs._imcd', 'skimage', 'skimage.restoration', 'skimage.color', 'skimage.measure', 'skimage.measure._moments_cy', 'skimage.measure._find_contours_cy', 'skimage.measure._pnpoly', 'skimage.measure._label', 'skimage.morphology', 'scipy', 'scipy.ndimage', 'scipy.stats', 'matplotlib', 'matplotlib.path', 'cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib.backends.backend_tkagg'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MMPS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MMPS',
)

# macOS: create a proper .app bundle so Gatekeeper doesn't rescan on every launch
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='MMPS.app',
        icon=ICON,
        bundle_identifier='com.mmps.microglia',
        info_plist={
            'CFBundleShortVersionString': '2.0.0',
            'NSHighResolutionCapable': True,
        },
    )
