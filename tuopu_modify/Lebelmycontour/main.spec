# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

APP_NAME  = "LabelMy Contour V1.1"
ICON_FILE = "logo.ico"
VER_FILE  = "verinfo.txt"     # 如无此文件，可改为 None 或注释掉 version= 行
MANIFEST  = "app.manifest"    # 如不想用，置为 None 或注释掉 manifest= 行

# 依赖与资源
datas = collect_data_files('vedo')     # vedo 的字体/资源
datas += [(ICON_FILE, ".")]            # 运行时要能找到 logo.ico（解包到 _MEIPASS 根）

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

# —— onefile：把二进制/zipfiles/datas 都直接塞进 EXE 中 ——
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,               # ✅ 先 binaries
    a.zipfiles,               # ✅ 再 zipfiles（你之前漏了）
    a.datas,                  # ✅ 最后 datas
    [],
    name=APP_NAME,
    icon=[ICON_FILE],         # EXE 外壳图标
    version=VER_FILE,         # 可选：没有就改 None
    manifest=MANIFEST,        # 可选：没有就改 None
    console=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,      # 用系统临时目录解包（默认即可）
)
