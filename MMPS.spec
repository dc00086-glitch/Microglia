# -*- mode: python ; coding: utf-8 -*-


import sys, os

# Icon file — .icns for macOS, .ico for Windows
if sys.platform == 'darwin':
    ICON = 'MMPS.icns'
else:
    ICON = 'MMPS.ico' if os.path.exists('MMPS.ico') else None

a = Analysis(
    ['MMPSv2.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui', 'numpy', 'PIL', 'PIL.Image', 'tifffile', 'skimage', 'skimage.restoration', 'skimage.color', 'skimage.measure', 'scipy', 'scipy.ndimage', 'scipy.stats', 'matplotlib', 'matplotlib.path', 'cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='MMPS',
)
