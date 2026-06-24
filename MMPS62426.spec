# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MMPS62426 (Mapping Microglia Parameters Software)

Usage:
    pyinstaller MMPS62426.spec

Produces: dist/MMPS62426.app (macOS)
"""

import os

block_cipher = None

# Paths — adjust if your files are elsewhere
SCRIPT = 'MMPS62426.py'
ICON = os.path.expanduser('~/Desktop/MMPS.icns')

# Collect extra data files to bundle alongside the executable
datas = []

# Bundle the icon file so it can be found at runtime
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
    name='MMPS62426',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
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
    name='MMPS62426',
)

# macOS .app bundle
app = BUNDLE(
    coll,
    name='MMPS62426.app',
    icon=ICON if os.path.isfile(ICON) else None,
    bundle_identifier='com.mmps.microgliaanalysis',
    info_plist={
        'CFBundleName': 'MMPS',
        'CFBundleDisplayName': 'MMPS - Mapping Microglia Parameters Software',
        'CFBundleShortVersionString': '6.24.26',
        'CFBundleVersion': '6.24.26',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        'LSMinimumSystemVersion': '10.14.0',
    },
)
