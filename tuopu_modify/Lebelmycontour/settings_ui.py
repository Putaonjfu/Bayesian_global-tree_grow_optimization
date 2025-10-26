# settings_ui.py
from __future__ import annotations
from typing import Tuple

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSpinBox, QPushButton, QColorDialog, QComboBox    # ← 新增 QComboBox
)
from qfluentwidgets import CardWidget, CaptionLabel, BodyLabel


# --------- 小工具：颜色互转 & 按钮底色 ----------
def _rgbf_to_qcolor(rgb: Tuple[float, float, float]) -> QColor:
    r, g, b = rgb
    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)
    return QColor(r, g, b)

def _qcolor_to_rgbf(c: QColor) -> Tuple[float, float, float]:
    return (c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0)

def _color_btn_style(c: QColor) -> str:
    return (
        "QPushButton{"
        "border:1px solid #3a3a3a;border-radius:6px;"
        "min-width:56px;min-height:28px;"
        f"background: rgb({c.red()},{c.green()},{c.blue()});"
        "}"
    )

def _bump_font(w, step: int = 1, bold: bool = False):
    """字号放大且自动增高，避免与下方控件重叠。"""
    f: QFont = w.font()
    if f.pointSize() > 0:
        f.setPointSize(f.pointSize() + step)
    else:
        px = f.pixelSize() if f.pixelSize() > 0 else 14
        f.setPixelSize(int(px * (1.0 + 0.12 * step)))
    if bold:
        f.setBold(True)
    w.setFont(f)


# ======================= 设置页 =======================
class SettingsInterface(QWidget):
    # —— 可视化参数 ——（主窗体会连接这些信号）
    pointSizeChanged    = Signal(int)
    pointColorChanged   = Signal(tuple)   # (r,g,b) in [0,1]
    edgeWidthChanged    = Signal(float)
    edgeColorChanged    = Signal(tuple)
    selEdgeWidthChanged = Signal(float)
    selEdgeColorChanged = Signal(tuple)

    # —— 主题 —— 'dark' / 'light'
    themeChanged        = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsInterface")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # -------- 卡片 1：可视化样式 --------
        self.cardVis = CardWidget(self)
        layVis = QVBoxLayout(self.cardVis)
        layVis.setContentsMargins(12, 12, 12, 12)
        layVis.setSpacing(10)

        titleVis = CaptionLabel("可视化样式", self.cardVis)
        _bump_font(titleVis, step=2, bold=True)       # 栏目标题：加大两号、加粗
        titleVis.setContentsMargins(0, 0, 0, 6)       # 与内容留点间距
        layVis.addWidget(titleVis)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        # 点大小
        labPoint = BodyLabel("点云大小", self.cardVis); _bump_font(labPoint, 1)
        self.spinPoint = QSpinBox(self.cardVis); self.spinPoint.setRange(1, 20); _bump_font(self.spinPoint, 1)
        grid.addWidget(labPoint, 0, 0, Qt.AlignLeft)
        grid.addWidget(self.spinPoint, 0, 1)

        # 点颜色
        labPointC = BodyLabel("点云颜色", self.cardVis); _bump_font(labPointC, 1)
        self.btnPointColor = QPushButton(self.cardVis); _bump_font(self.btnPointColor, 1); self.btnPointColor.setMinimumHeight(28)
        grid.addWidget(labPointC, 0, 2, Qt.AlignLeft)
        grid.addWidget(self.btnPointColor, 0, 3)

        # 线宽
        labEdge = BodyLabel("线框粗细", self.cardVis); _bump_font(labEdge, 1)
        self.spinEdge = QSpinBox(self.cardVis); self.spinEdge.setRange(1, 20); _bump_font(self.spinEdge, 1)
        grid.addWidget(labEdge, 1, 0, Qt.AlignLeft)
        grid.addWidget(self.spinEdge, 1, 1)

        # 线颜色
        labEdgeC = BodyLabel("线框颜色", self.cardVis); _bump_font(labEdgeC, 1)
        self.btnEdgeColor = QPushButton(self.cardVis); _bump_font(self.btnEdgeColor, 1); self.btnEdgeColor.setMinimumHeight(28)
        grid.addWidget(labEdgeC, 1, 2, Qt.AlignLeft)
        grid.addWidget(self.btnEdgeColor, 1, 3)

        # 选中线宽
        labSel = BodyLabel("选中线粗细", self.cardVis); _bump_font(labSel, 1)
        self.spinSelEdge = QSpinBox(self.cardVis); self.spinSelEdge.setRange(1, 30); _bump_font(self.spinSelEdge, 1)
        grid.addWidget(labSel, 2, 0, Qt.AlignLeft)
        grid.addWidget(self.spinSelEdge, 2, 1)

        # 选中线颜色
        labSelC = BodyLabel("选中线颜色", self.cardVis); _bump_font(labSelC, 1)
        self.btnSelColor = QPushButton(self.cardVis); _bump_font(self.btnSelColor, 1); self.btnSelColor.setMinimumHeight(28)
        grid.addWidget(labSelC, 2, 2, Qt.AlignLeft)
        grid.addWidget(self.btnSelColor, 2, 3)

        layVis.addLayout(grid)
        root.addWidget(self.cardVis)

        # -------- 卡片 2：界面主题 --------
        self.cardTheme = CardWidget(self)
        layTheme = QVBoxLayout(self.cardTheme)
        layTheme.setContentsMargins(12, 12, 12, 12)
        layTheme.setSpacing(10)

        titleTheme = CaptionLabel("界面主题", self.cardTheme)
        _bump_font(titleTheme, step=2, bold=True)
        titleTheme.setContentsMargins(0, 0, 0, 6)
        layTheme.addWidget(titleTheme)

        row = QHBoxLayout()
        self.btnDark  = QPushButton("Dark",  self.cardTheme);  _bump_font(self.btnDark, 1);  self.btnDark.setMinimumHeight(30)
        self.btnLight = QPushButton("Light", self.cardTheme); _bump_font(self.btnLight, 1); self.btnLight.setMinimumHeight(30)
        row.addWidget(self.btnDark); row.addWidget(self.btnLight); row.addStretch(1)
        layTheme.addLayout(row)
        root.addWidget(self.cardTheme)

        root.addStretch(1)

        # ---------- 信号绑定（即改即生效） ----------
        self.spinPoint.valueChanged.connect(lambda v: self.pointSizeChanged.emit(int(v)))
        self.spinEdge.valueChanged.connect(lambda v: self.edgeWidthChanged.emit(float(v)))
        self.spinSelEdge.valueChanged.connect(lambda v: self.selEdgeWidthChanged.emit(float(v)))

        self.btnPointColor.clicked.connect(lambda: self._pick_color(self.btnPointColor, self.pointColorChanged))
        self.btnEdgeColor.clicked.connect(lambda: self._pick_color(self.btnEdgeColor,  self.edgeColorChanged))
        self.btnSelColor.clicked.connect(lambda: self._pick_color(self.btnSelColor,   self.selEdgeColorChanged))

        self.btnDark.clicked.connect(lambda: self.themeChanged.emit("dark"))
        self.btnLight.clicked.connect(lambda: self.themeChanged.emit("light"))

    # ------ 外部回填初值 ------
    def init_from_values(
        self,
        point_size: int,
        point_color: Tuple[float, float, float],
        edge_width: float,
        edge_color: Tuple[float, float, float],
        sel_edge_width: float,
        sel_edge_color: Tuple[float, float, float],
    ):
        self.spinPoint.setValue(int(point_size))
        self.spinEdge.setValue(int(round(edge_width)))
        self.spinSelEdge.setValue(int(round(sel_edge_width)))
        self._set_btn_color(self.btnPointColor, _rgbf_to_qcolor(point_color))
        self._set_btn_color(self.btnEdgeColor,  _rgbf_to_qcolor(edge_color))
        self._set_btn_color(self.btnSelColor,   _rgbf_to_qcolor(sel_edge_color))

    # ------ 内部：颜色拾取 ------
    def _pick_color(self, btn: QPushButton, signal):
        base = btn.palette().button().color() if btn.palette() else QColor(230, 230, 230)
        c = QColorDialog.getColor(base, self, "选择颜色")
        if not c.isValid():
            return
        self._set_btn_color(btn, c)
        signal.emit(_qcolor_to_rgbf(c))

    def _set_btn_color(self, btn: QPushButton, c: QColor):
        btn.setText(f"#{c.red():02X}{c.green():02X}{c.blue():02X}")
        btn.setStyleSheet(_color_btn_style(c))
