# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MMPSv2 (Microglia Morphology Processing Suite)

Usage:
    pyinstaller MMPSv2.spec

Produces: dist/MMPSv2.app (macOS) or dist/MMPSv2/ (Windows/Linux)
"""

import os
import sys

block_cipher = None

# Paths — adjust if your files are elsewhere
SCRIPT = 'MMPSv2.12.py'
ICON = os.path.expanduser('~/Desktop/MMPS.icns')

# Collect extra data files to bundle alongside the executable
datas = []

# Bundle 3DMicroglia.py if it exists (optional 3D module)
if os.path.isfile('3DMicroglia.py'):
    datas.append(('3DMicroglia.py', '.'))

# Bundle the icon file so _get_app_icon() can find it at runtime
if os.path.isfile(ICON):
    datas.append((ICON, '.'))
elif os.path.isfile('MMPS.icns'):
    datas.append(('MMPS.icns', '.'))

a = Analysis(
    [SCRIPT],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PIL',
        'PIL.Image',
        'PIL.TiffImagePlugin',
        'tifffile',
        'skimage',
        'skimage.restoration',
        'skimage.color',
        'skimage.measure',
        'skimage.segmentation',
        'skimage.filters',
        'scipy',
        'scipy.ndimage',
        'scipy.stats',
        'matplotlib',
        'matplotlib.path',
        'cv2',
        'numpy',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.sip',
        'csv',
        'json',
        'glob',
        'math',
        'multiprocessing',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib.backends.backend_tkagg',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MMPSv2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,  # macOS: accept file drops
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON if os.path.isfile(ICON) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MMPSv2',
)

# macOS .app bundle
app = BUNDLE(
    coll,
    name='MMPSv2.app',
    icon=ICON if os.path.isfile(ICON) else None,
    bundle_identifier='com.mmps.microgliaanalysis',
    info_plist={
        'CFBundleName': 'MMPSv2',
        'CFBundleDisplayName': 'MMPSv2 - Microglia Analysis',
        'CFBundleShortVersionString': '2.12',
        'CFBundleVersion': '2.12.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        'LSMinimumSystemVersion': '10.14.0',
    },
)
