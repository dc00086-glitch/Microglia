# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MMPS (Microglia Morphology Processing Suite).

Usage:
    pip install pyinstaller
    pyinstaller MMPS.spec

Produces a standalone app:
    macOS  ->  dist/MMPS.app
    Windows -> dist/MMPS/MMPS.exe
"""

import sys
import os

block_cipher = None

a = Analysis(
    ['MMPSv2.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'numpy',
        'PIL',
        'PIL.Image',
        'tifffile',
        'skimage',
        'skimage.restoration',
        'skimage.color',
        'skimage.measure',
        'scipy',
        'scipy.ndimage',
        'scipy.stats',
        'matplotlib',
        'matplotlib.path',
        'cv2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- macOS: .app bundle ---
if sys.platform == 'darwin':
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
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='MMPS',
    )
    app = BUNDLE(
        coll,
        name='MMPS.app',
        icon=None,  # Replace with 'MMPS.icns' if you create one
        bundle_identifier='com.mmps.microglia',
        info_plist={
            'CFBundleName': 'MMPS',
            'CFBundleDisplayName': 'MMPS - Microglia Morphology Processing Suite',
            'CFBundleShortVersionString': '2.0',
            'NSHighResolutionCapable': True,
        },
    )

# --- Windows: folder with .exe ---
else:
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
        console=False,  # No terminal window
        icon=None,  # Replace with 'MMPS.ico' if you create one
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='MMPS',
    )
