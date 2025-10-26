#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
# ---- 必须在 QApplication 之前：绑定任务栏图标到你的 EXE ----
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            u"com.senjer.LabelMyContour.V1_1"  # 自定义且稳定的ID
        )
    except Exception:
        pass

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from qfluentwidgets import setTheme, Theme, qconfig
from app_gui import MainWindow

# ---- 路径解析：支持 onefile / onedir / 源码运行 ----
def resource_path(rel: str) -> str:
    if hasattr(sys, "_MEIPASS"):                              # onefile 展开目录
        base = sys._MEIPASS
    elif getattr(sys, "frozen", False):                       # onedir，可执行文件所在目录
        base = os.path.dirname(sys.executable)
    else:                                                     # 源码运行：当前文件目录
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)

def load_app_icon() -> QIcon:
    """优先从 Qt 资源加载（若你有 resources.qrc），否则回落到物理文件。"""
    # 1) Qt 资源（如果你用过：<qresource prefix="/res"><file>logo.ico</file>）
    icon = QIcon(":/res/logo.ico")
    if not icon.isNull():
        return icon
    # 2) 物理文件（打包已带上）
    for candidate in ("logo.ico", "resources/logo.ico", "icons/logo.ico"):
        p = resource_path(candidate)
        ic = QIcon(p)
        if not ic.isNull():
            return ic
    return QIcon()  # 兜底（空）

def main():
    # 可选高DPI缩放策略（弃用 AA_UseHighDpiPixmaps，不再设置）
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication(sys.argv)

    # 主题
    setTheme(Theme.DARK)
    try:
        qconfig.load()
    except Exception:
        pass

    # 图标：应用级 + 窗口级 都设置，确保任务栏/左上角统一显示
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    w = MainWindow()
    if not app_icon.isNull():
        try:
            w.setWindowIcon(app_icon)
        except Exception:
            pass

    w.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
