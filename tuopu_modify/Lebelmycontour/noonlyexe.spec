# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

APP_NAME  = "LabelMy Contour V1.1"
ICON_FILE = "logo.ico"
VER_FILE  = "verinfo.txt"
MANIFEST  = "app.manifest"

datas = collect_data_files('vedo')
datas += [(ICON_FILE, ".")]   # 👈 把 logo.ico 放到 dist/APP_NAME/ 根目录

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['vedo'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    icon=[ICON_FILE],      # EXE 资源图标（保留）
    version=VER_FILE,
    manifest=MANIFEST,
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
    name=APP_NAME,
)
