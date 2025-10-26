#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os
import sys
import subprocess

from PySide6.QtCore import Qt, QTimer, QEvent

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QSizePolicy, QPlainTextEdit, QFrame, QTreeWidget, QAbstractItemView, QHeaderView, QTextBrowser, QMessageBox, QTreeWidgetItem, QComboBox, QColorDialog

from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, FluentIcon,
    InfoBar, InfoBarPosition, PrimaryPushButton, PushButton,
    BodyLabel, CaptionLabel, CardWidget
)
from controller import Controller
from viewer_embed import ViewerEmbed
from legacy_bridge import LegacyBridge, AppContext
from settings_ui import SettingsInterface


def _fi(*names):
    for n in names:
        icon = getattr(FluentIcon, n, None)
        if icon is not None:
            return icon
    return FluentIcon.INFO


class StatRow(QWidget):
    def __init__(self, name: str, value: str = "--", parent: QWidget | None = None):
        super().__init__(parent)
        self.nameLabel = CaptionLabel(name, self)
        self.valueLabel = BodyLabel(value, self)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.nameLabel, 0, Qt.AlignLeft)
        lay.addStretch(1)
        lay.addWidget(self.valueLabel, 0, Qt.AlignRight)

    def setValue(self, v: str):
        self.valueLabel.setText(v)


class DatasetInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("datasetInterface")
        self.firstTitle = CaptionLabel("FuncA:删除冗余线框", self)
        self.pathLabel = BodyLabel("未选择", self)
        self.totalRow = StatRow("总文件数：", "--", self)
        self.doneRow  = StatRow("已处理：", "--", self)
        self.todoRow  = StatRow("待处理：", "--", self)

        self.loadBtn    = PrimaryPushButton("加载数据", self)
        self.refreshBtn = PushButton("刷新统计", self)
        self.refreshBtn.setEnabled(False)

        btnBar = QHBoxLayout()
        btnBar.addWidget(self.loadBtn)
        btnBar.addWidget(self.refreshBtn)
        btnBar.addStretch(1)

        card = CardWidget(self)
        cardLay = QVBoxLayout(card)
        cardLay.addWidget(CaptionLabel("当前路径：", self))
        cardLay.addWidget(self.pathLabel)
        cardLay.addSpacing(8)
        cardLay.addWidget(self.totalRow)
        cardLay.addWidget(self.doneRow)
        cardLay.addWidget(self.todoRow)
        cardLay.addSpacing(12)
        cardLay.addLayout(btnBar)

        rootLay = QVBoxLayout(self)
        rootLay.setContentsMargins(20, 20, 20, 20)
        rootLay.setSpacing(12)
        rootLay.addWidget(card)
        rootLay.addStretch(1)

        # ====== 小标题（第二块的标题）======
        self.secondTitle = CaptionLabel("FuncB:对比数据", self)
        rootLay.insertWidget(0, self.firstTitle)
        rootLay.insertWidget(rootLay.count() - 1, self.secondTitle)

        # ====== 第二块：精简版（仅“当前路径/总文件数/加载按钮”）======
        self.pathLabel2 = BodyLabel("未选择", self)
        self.totalRow2  = StatRow("总文件数：", "--", self)

        self.loadBtn2    = PrimaryPushButton("加载数据", self)
        self.refreshBtn2 = PushButton("刷新统计", self)
        self.refreshBtn2.setEnabled(False)

        btnBar2 = QHBoxLayout()
        btnBar2.addWidget(self.loadBtn2)
        btnBar2.addWidget(self.refreshBtn2)
        btnBar2.addStretch(1)

        card2 = CardWidget(self)
        cardLay2 = QVBoxLayout(card2)
        cardLay2.addWidget(CaptionLabel("当前路径：", self))
        cardLay2.addWidget(self.pathLabel2)
        cardLay2.addSpacing(8)
        cardLay2.addWidget(self.totalRow2)
        cardLay2.addSpacing(12)
        cardLay2.addLayout(btnBar2)

        rootLay.insertWidget(rootLay.count() - 1, card2)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


def app_path(*p):
    """打包后（_MEIPASS）或源码状态下，统一拿资源路径。"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base.joinpath(*p))


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()

        # === 1) 窗口基础信息 / 图标 ===
        self.setWindowTitle("Label MyContour")
        self.resize(1200, 800)
        icon_path = app_path("logo.ico")
        if os.path.exists(icon_path):
            from PySide6 import QtGui
            self.setWindowIcon(QtGui.QIcon(icon_path))
        self.setMinimumSize(960, 600)

        # === 2) 控制器 / 兼容桥 ===
        self.controller = Controller()
        self.bridge = LegacyBridge(Path(__file__).resolve().parent)

        # === 3) 数据集运行态 ===
        self._dataset_root: Path | None = None
        self._out_root: Path | None = None
        self._all_ids: list[str] = []
        self._current_id: str | None = None

        # === 4) 数据页 ===
        self.datasetInterface = DatasetInterface(self)

        # === 5) 视图页（原） ===
        self.viewerInterface = ViewerEmbed(self)
        self.viewerPage = QWidget(self)
        self.viewerPage.setObjectName("viewerPage")
        _vp_layout = QVBoxLayout(self.viewerPage)
        _vp_layout.setContentsMargins(0, 0, 0, 0)

        self.viewerTopCard = CardWidget(self.viewerPage)
        _tc = QHBoxLayout(self.viewerTopCard)
        _tc.setContentsMargins(16, 12, 16, 12)
        _tc.setSpacing(24)

        # 左列
        _left = QVBoxLayout()
        _left.setSpacing(6)
        self.vi_idRow = StatRow("当前ID：", "-", self.viewerTopCard)
        self.vi_edgeRow = StatRow("标记边数：", "0", self.viewerTopCard)
        self.vi_progRow = StatRow("进度：", "0 / 0", self.viewerTopCard)
        self.vi_selRow = StatRow("选择模式：", "关闭 (F)", self.viewerTopCard)
        self.vi_todoRow = StatRow("待处理：", "0", self.viewerTopCard)
        _left.addWidget(self.vi_idRow)
        _left.addWidget(self.vi_edgeRow)
        _left.addWidget(self.vi_progRow)
        _left.addWidget(self.vi_selRow)
        _left.addWidget(self.vi_todoRow)

        # 中列
        _midBox = QVBoxLayout()
        _midBox.setSpacing(6)
        self.vi_opsTitle = CaptionLabel("操作队列：", self.viewerTopCard)
        self.vi_opsTitle.setObjectName("sectionHeader")
        self.vi_opsPanel = QFrame(self.viewerTopCard)
        self.vi_opsPanel.setObjectName("opsPanel")
        _opsLay = QVBoxLayout(self.vi_opsPanel)
        _opsLay.setContentsMargins(8, 8, 8, 8)
        _opsLay.setSpacing(0)
        self.vi_opsView = QPlainTextEdit(self.vi_opsPanel)
        self.vi_opsView.setReadOnly(True)
        self.vi_opsView.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.vi_opsView.setPlaceholderText("操作队列将显示在此处（每次框选/点击都会追加）")
        self.vi_opsView.setMinimumHeight(96)
        _opsLay.addWidget(self.vi_opsView)
        _midBox.addWidget(self.vi_opsTitle, 0)
        _midBox.addWidget(self.vi_opsPanel, 1)
        self.vi_opsPanel.setStyleSheet("""
        #opsPanel {
            background: #1f1f1f;
            border: 1px solid #303030;
            border-radius: 8px;
        }
        #opsPanel QPlainTextEdit {
            background: transparent;
            border: none;
            color: #E6E6E6;
            selection-background-color: #264F78;
            font-family: Consolas, "Cascadia Code", "Fira Code", monospace;
            font-size: 13pt;
        }
        """)

        # 右列
        _right = QVBoxLayout()
        _right.setSpacing(8)
        self.vi_btn_select = PrimaryPushButton("选择模式 (F)", self.viewerTopCard)
        self.vi_btn_undo = PushButton("撤销 (Z)", self.viewerTopCard)
        self.vi_btn_clear = PushButton("重置本数据 (Y)", self.viewerTopCard)
        self.vi_btn_save_next = PrimaryPushButton("保存并下一个 (S)", self.viewerTopCard)
        self.vi_btn_prev = PushButton("上一个 (P)", self.viewerTopCard)
        self.vi_btn_open = PushButton("打开文件夹", self.viewerTopCard)
        for _w in [self.vi_btn_select, self.vi_btn_undo, self.vi_btn_clear,
                   self.vi_btn_save_next, self.vi_btn_prev, self.vi_btn_open]:
            _w.setEnabled(True)
            _right.addWidget(_w)
        _right.addStretch(1)

        _tc.addLayout(_left, 1)
        _tc.addLayout(_midBox, 1)
        _tc.addLayout(_right, 0)
        _vp_layout.addWidget(self.viewerTopCard)
        _vp_layout.addWidget(self.viewerInterface, 1)

        # === 7) 视图页（副本） ===
        self.viewerInterface2 = ViewerEmbed(self)
        # 同步样式
        self.viewerInterface2.set_point_size(getattr(self.viewerInterface, "_point_size_default", 4))
        pc = getattr(self.viewerInterface, "_point_color", (0.95, 0.95, 0.95))
        self.viewerInterface2.set_point_color(*pc)
        ew = getattr(self.viewerInterface, "_edge_width", 3.0)
        ec = getattr(self.viewerInterface, "_edge_color", (1.0, 1.0, 1.0))
        self.viewerInterface2.set_edge_style(width=ew, color=ec)
        sew = getattr(self.viewerInterface, "_sel_edge_width", 4.0)
        sec = getattr(self.viewerInterface, "_sel_edge_color", (1.0, 0.2, 0.2))
        self.viewerInterface2.set_selected_edge_style(width=sew, color=sec)

        self.viewerPage2 = QWidget(self)
        self.viewerPage2.setObjectName("viewerPage2")
        _vp2_layout = QVBoxLayout(self.viewerPage2)
        _vp2_layout.setContentsMargins(0, 0, 0, 0)

        self.viewerTopCard2 = CardWidget(self.viewerPage2)
        _tc2 = QHBoxLayout(self.viewerTopCard2)
        _tc2.setContentsMargins(16, 12, 16, 12)
        _tc2.setSpacing(24)

        # 左列（仅 当前ID / 进度）
        # 左列（新增：跳转到ID下拉）
        _left2 = QVBoxLayout()
        _left2.setSpacing(6)

        # 下拉：跳转到ID
        self.vi2_idCombo = QComboBox(self.viewerTopCard2)
        self.vi2_idCombo.setMinimumWidth(220)
        self.vi2_idCombo.currentIndexChanged.connect(self._viz2_on_id_combo_changed)

        _rowCombo = QHBoxLayout()
        _rowCombo.setSpacing(6)
        _rowCombo.addWidget(CaptionLabel("跳转到ID：", self.viewerTopCard2), 0)
        _rowCombo.addWidget(self.vi2_idCombo, 1)
        _left2.addLayout(_rowCombo)

        # 原有：当前ID/进度
        self.vi2_idRow = StatRow("当前ID：", "-", self.viewerTopCard2)
        self.vi2_progRow = StatRow("进度：", "0 / 0", self.viewerTopCard2)
        _left2.addWidget(self.vi2_idRow)
        _left2.addWidget(self.vi2_progRow)

        # 中列：三列表格（文件 / 颜色 / 粗细）
        _midBox2 = QVBoxLayout()
        _midBox2.setSpacing(6)
        self.vi2_filesTitle = CaptionLabel("可展示文件：", self.viewerTopCard2)
        _midBox2.addWidget(self.vi2_filesTitle, 0)

        self.vi2_tree = QTreeWidget(self.viewerTopCard2)
        self.vi2_tree.setColumnCount(3)
        self.vi2_tree.setHeaderLabels(["文件", "颜色", "粗细"])
        self.vi2_tree.setRootIsDecorated(False)
        self.vi2_tree.setAlternatingRowColors(True)
        self.vi2_tree.setUniformRowHeights(True)
        self.vi2_tree.setSelectionMode(QAbstractItemView.NoSelection)
        self.vi2_tree.header().setStretchLastSection(False)
        self.vi2_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.vi2_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.vi2_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        _midBox2.addWidget(self.vi2_tree, 1)

        # 行状态：item -> {path,color,width}
        self._viz2_row_state = {}

        # 右列（按钮占位，不绑定）
        # 右列：按钮（新版）
        _right2 = QVBoxLayout();
        _right2.setSpacing(8)

        self.v2_btn_prevA = PushButton("上一份数据 (A)", self.viewerTopCard2)
        self.v2_btn_nextD = PrimaryPushButton("下一份数据 (D)", self.viewerTopCard2)
        self.v2_btn_markS = PushButton("记录ID (S)", self.viewerTopCard2)
        self.v2_btn_openQ = PushButton("打开文件夹 (Q)", self.viewerTopCard2)

        for _w in [self.v2_btn_prevA, self.v2_btn_nextD, self.v2_btn_markS, self.v2_btn_openQ]:
            _right2.addWidget(_w)
        _right2.addStretch(1)

        # 按钮绑定（先实现“下一份数据”）
        self.v2_btn_nextD.clicked.connect(self._viz2_next_id)
        self.v2_btn_prevA.clicked.connect(self._viz2_prev_id)  # 先占位，稍后可补
        self.v2_btn_markS.clicked.connect(self._onViz2RecordClicked)  # 先占位，稍后可补
        self.v2_btn_openQ.clicked.connect(self._viz2_open_current_folder)  # 先占位，稍后可补

        _tc2.addLayout(_left2, 1)
        _tc2.addLayout(_midBox2, 1)
        _tc2.addLayout(_right2, 0)
        _tc2.setStretch(0, 3)
        _tc2.setStretch(1, 8)
        _tc2.setStretch(2, 1)

        _vp2_layout.addWidget(self.viewerTopCard2)
        _vp2_layout.addWidget(self.viewerInterface2, 1)
        self._viz2_style_template: dict[str, dict] = {}  # filename -> {"checked":bool, "color":(r,g,b), "width":float}

        # === 6) 设置页 ===
        self.settingsInterface = SettingsInterface(self)

        # 设置信号 → 两个 viewer
        self.settingsInterface.pointSizeChanged.connect(
            lambda v: (self.viewerInterface.set_point_size(v), self.viewerInterface2.set_point_size(v))
        )
        self.settingsInterface.pointColorChanged.connect(
            lambda rgb: (self.viewerInterface.set_point_color(*rgb), self.viewerInterface2.set_point_color(*rgb))
        )
        self.settingsInterface.edgeWidthChanged.connect(
            lambda v: (self.viewerInterface.set_edge_style(width=v), self.viewerInterface2.set_edge_style(width=v))
        )
        self.settingsInterface.edgeColorChanged.connect(
            lambda rgb: (self.viewerInterface.set_edge_style(color=rgb), self.viewerInterface2.set_edge_style(color=rgb))
        )
        self.settingsInterface.selEdgeWidthChanged.connect(
            lambda v: (self.viewerInterface.set_selected_edge_style(width=v),
                       self.viewerInterface2.set_selected_edge_style(width=v))
        )
        self.settingsInterface.selEdgeColorChanged.connect(
            lambda rgb: (self.viewerInterface.set_selected_edge_style(color=rgb),
                         self.viewerInterface2.set_selected_edge_style(color=rgb))
        )

        # 绑定主题切换
        self.settingsInterface.themeChanged.connect(self._onThemeChanged)

        # 用 viewer 当前值回填设置页
        self.settingsInterface.init_from_values(
            point_size=getattr(self.viewerInterface, "_point_size_default", 4),
            point_color=getattr(self.viewerInterface, "_point_color", (0.95, 0.95, 0.95)),
            edge_width=getattr(self.viewerInterface, "_edge_width", 3.0),
            edge_color=getattr(self.viewerInterface, "_edge_color", (1.0, 1.0, 1.0)),
            sel_edge_width=getattr(self.viewerInterface, "_sel_edge_width", 4.0),
            sel_edge_color=getattr(self.viewerInterface, "_sel_edge_color", (1.0, 0.2, 0.2))
        )

        # === 8) 导航 ===
        self._initNavigation()

        # === 9) 页面间信号 ===
        self.datasetInterface.loadBtn.clicked.connect(self._onLoadClicked)
        self.datasetInterface.loadBtn2.clicked.connect(self._onLoadSecondClicked)
        self.datasetInterface.refreshBtn.clicked.connect(self._refreshStats)

        self._bindSelectionUI()
        self.vi_btn_save_next.clicked.connect(self._onSaveNextClicked)
        self.vi_btn_prev.clicked.connect(self._onPrevClicked)
        self.vi_btn_undo.clicked.connect(self._onUndoClicked)
        self.vi_btn_open.clicked.connect(self._onOpenOutputClicked)

        # === 10) 快捷键 ===
        self._shortcutS = QShortcut(QKeySequence("S"), self); self._shortcutS.setContext(Qt.ApplicationShortcut); self._shortcutS.activated.connect(self._onSaveNextClicked)
        self._shortcutY = QShortcut(QKeySequence("Y"), self); self._shortcutY.setContext(Qt.ApplicationShortcut); self._shortcutY.activated.connect(self._onResetCurrentClicked)
        self._shortcutZ = QShortcut(QKeySequence("Z"), self); self._shortcutZ.setContext(Qt.ApplicationShortcut); self._shortcutZ.activated.connect(self._onUndoClicked)
        self._shortcutO = QShortcut(QKeySequence("O"), self); self._shortcutO.setContext(Qt.ApplicationShortcut); self._shortcutO.activated.connect(self._onOpenOutputClicked)

        # —— 视图（副本）页快捷键 ——（应用级，随时可用）
        self._shortcutA2 = QShortcut(QKeySequence("A"), self)
        self._shortcutA2.setContext(Qt.ApplicationShortcut)
        self._shortcutA2.activated.connect(self._viz2_prev_id)

        self._shortcutD2 = QShortcut(QKeySequence("D"), self)
        self._shortcutD2.setContext(Qt.ApplicationShortcut)
        self._shortcutD2.activated.connect(self._viz2_next_id)

        self._shortcutS2 = QShortcut(QKeySequence("S"), self)
        self._shortcutS2.setContext(Qt.ApplicationShortcut)
        self._shortcutS2.activated.connect(self._viz2_mark_current)

        self._shortcutQ2 = QShortcut(QKeySequence("Q"), self)
        self._shortcutQ2.setContext(Qt.ApplicationShortcut)
        self._shortcutQ2.activated.connect(self._viz2_open_current_folder)

        # === 11) 初始页 & 队列文本 ===
        self.switchTo(self.datasetInterface)
        self.viewerInterface.opsChanged.connect(self._onOpsChanged)
        self._onOpsChanged("（空）")
        self._apply_fixed_font_scale()

        # === 12) 几何节流 ===
        self._geom_busy = False
        self._geomTimer = QTimer(self)
        self._geomTimer.setSingleShot(True)
        self._geomTimer.timeout.connect(self._end_geometry_change)
        try:
            self.navigationInterface.installEventFilter(self)
        except Exception:
            pass

        # === 13) 无边框缩放兼容 ===
        try:
            if hasattr(self, "setResizeEnabled"): self.setResizeEnabled(True)
            if hasattr(self, "setResizeBorderWidth"): self.setResizeBorderWidth(8)
            elif hasattr(self, "setBorderWidth"): self.setBorderWidth(8)
        except Exception:
            pass

        # === 14) 鼠标追踪 ===
        self.setMouseTracking(True)
        self._closing = False

    # ---------------- Navigation ----------------
    def _initNavigation(self):
        self.addSubInterface(self.datasetInterface, _fi("FOLDER"), "数据加载", NavigationItemPosition.TOP)
        self.addSubInterface(self.viewerPage, _fi("VIEW", "ZOOM"), "删除冗余线框", NavigationItemPosition.TOP)
        self.addSubInterface(self.viewerPage2, _fi("VIEW", "ZOOM"), "数据集可视化", NavigationItemPosition.TOP)
        self.addSubInterface(self.settingsInterface, _fi("SETTING"), "设置", NavigationItemPosition.TOP)

        # 说明
        about = QWidget(self)
        about.setObjectName("aboutInterface")
        aboutLay = QVBoxLayout(about)
        aboutLay.setContentsMargins(18, 18, 18, 18)
        aboutLay.setSpacing(12)

        self._helpView = QTextBrowser(about)
        self._helpView.setOpenExternalLinks(True)
        self._helpView.setReadOnly(True)
        self._helpView.document().setDefaultStyleSheet("""
          body, h1, h2, h3, p, li, th, td, code, pre { color:#ffffff !important; }
          a { color:#8ab4ff !important; }
        """)
        self._helpView.setHtml(self._build_help_html())
        aboutLay.addWidget(self._helpView, 1)
        self._apply_help_theme(self._get_current_theme_name())
        self.addSubInterface(about, _fi("INFO"), "说明", NavigationItemPosition.BOTTOM)

    # ---------------- Utils ----------------
    def _ctx(self) -> AppContext:
        return AppContext(dataset_root=self._dataset_root, out_dir=self.controller.out_dir, window=self, controller=self.controller)

    def _toast_missing(self, name: str):
        InfoBar.warning(title="功能未接入", content=f"未找到旧工程中的 “{name}” 实现（可在 legacy_hooks.register() 显式绑定）。",
                        orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=3000, parent=self)

    def _toast_ok(self, title: str, content: str = ""):
        InfoBar.success(title=title, content=content, orient=Qt.Horizontal,
                        isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=1800, parent=self)

    def _read_all_ids(self, root: Path) -> list[str]:
        all_path = root / "all.txt"
        if not all_path.exists():
            return []
        with open(all_path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _read_done_ids(self) -> set[str]:
        if not self._out_root:
            return set()
        out_txt = self._out_root / "out.txt"
        if not out_txt.exists():
            return set()
        with open(out_txt, "r", encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}

    def _append_done(self, id_str: str):
        if not self._out_root:
            return
        self._out_root.mkdir(parents=True, exist_ok=True)
        out_txt = self._out_root / "out.txt"
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write(id_str + "\n")

    def _remove_done_last_occurrence(self, id_str: str) -> bool:
        if not self._out_root:
            return False
        out_txt = self._out_root / "out.txt"
        if not out_txt.exists():
            return False
        with open(out_txt, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == id_str:
                idx = i
                break
        if idx < 0:
            return False
        del lines[idx]
        with open(out_txt, "w", encoding="utf-8") as f:
            for ln in lines:
                if ln.strip():
                    f.write(ln.strip() + "\n")
        return True

    def _update_progress_rows(self):
        done = self._read_done_ids()
        total = len(self._all_ids)
        self.vi_progRow.setValue(f"{len(done)} / {total}")
        self.vi_todoRow.setValue(str(max(0, total - len(done))))

    # ---------------- 加载/刷新 ----------------
    def _onLoadClicked(self):
        directory = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if not directory:
            return

        p = Path(directory)
        stats = self.controller.analyze_dataset(p)
        self._dataset_root = self.controller.dataset_root
        self._out_root = self.controller.out_dir
        self._all_ids = self._read_all_ids(self._dataset_root)

        self.datasetInterface.pathLabel.setText(str(self._dataset_root))
        self.datasetInterface.totalRow.setValue(str(stats["total"]))
        self.datasetInterface.doneRow.setValue(str(stats["processed"]))
        self.datasetInterface.todoRow.setValue(str(stats["pending"]))
        self.datasetInterface.refreshBtn.setEnabled(True)

        pending_ids = self.controller.pending_sources() or []
        if not pending_ids:
            done = self._read_done_ids()
            pending_ids = [i for i in self._all_ids if i not in done]

        if not pending_ids:
            InfoBar.success(title="该数据已处理完毕", content="未发现待处理的 ID，已全部完成。",
                            orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=2500, parent=self)
            try:
                self.viewerInterface.set_selection_mode(False)
                self.viewerInterface.clear()
                self.viewerInterface.force_full_render()
            except Exception:
                pass
            self.switchTo(self.datasetInterface)
            return

        self._current_id = pending_ids[0]
        self.switchTo(self.viewerPage)
        self.viewerInterface.show_id(self._dataset_root, self._current_id)
        self.viewerInterface.force_full_render()

        self.vi_idRow.setValue(self._current_id)
        self.vi_edgeRow.setValue("0")
        self._update_progress_rows()
        self.vi_selRow.setValue("关闭 (F)")

        self.bridge.call("load_data", self._ctx())

    def _refreshStats(self):
        if not self._dataset_root:
            return
        p = self._dataset_root
        stats = self.controller.analyze_dataset(p)
        self.datasetInterface.totalRow.setValue(str(stats["total"]))
        self.datasetInterface.doneRow.setValue(str(stats["processed"]))
        self.datasetInterface.todoRow.setValue(str(stats["pending"]))
        InfoBar.info(title="已刷新统计",
                     content=f"总 {stats['total']}，已 {stats['processed']}，待 {stats['pending']}",
                     orient=Qt.Horizontal, isClosable=True,
                     position=InfoBarPosition.TOP_RIGHT, duration=2000, parent=self)

    # ---------------- 顶部动作（保留原桥接） ----------------
    def _onSaveClicked(self):
        if not self.bridge.call("save_results", self._ctx()):
            self._toast_missing("保存结果")

    def _onExportClicked(self):
        if not self.bridge.call("export_results", self._ctx()):
            self._toast_missing("导出")

    def _onRunClicked(self):
        if not self.bridge.call("run_pipeline", self._ctx()):
            self._toast_missing("运行流程")

    def _onNextClicked(self):
        if not self.bridge.call("next_item", self._ctx()):
            self._toast_missing("下一项")

    def _onUndoClicked(self):
        self.viewerInterface.undo_last_operation()
        try:
            n = len(self.viewerInterface.selected_ids())
            self.vi_edgeRow.setValue(str(n))
        except Exception:
            pass

    # ---------------- 选择模式 & 重置本数据 ----------------
    def _bindSelectionUI(self):
        self.vi_btn_select.clicked.connect(self._toggleSelectMode)
        self._shortcutF = QShortcut(QKeySequence("F"), self)
        self._shortcutF.setContext(Qt.ApplicationShortcut)
        self._shortcutF.activated.connect(self._toggleSelectMode)
        self.vi_selRow.setValue("关闭 (F)")
        self.vi_btn_clear.clicked.connect(self._onResetCurrentClicked)

    def _onResetCurrentClicked(self):
        if getattr(self, "_current_id", None) is None:
            return
        self.viewerInterface.clear_selection()
        self.vi_edgeRow.setValue("0")
        try:
            self.viewerInterface.force_full_render()
        except Exception:
            pass

    def _toggleSelectMode(self):
        self.viewerInterface.toggle_select_mode()
        on = self.viewerInterface.is_selection_mode()
        self.vi_selRow.setValue("开启 (F)" if on else "关闭 (F)")

    # ---------------- 上一个/保存并下一个 ----------------
    def _onPrevClicked(self):
        if not self._dataset_root or not self._all_ids or not self._current_id:
            InfoBar.warning(title="无法回退", content="当前没有已加载的数据。",
                            orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=2000, parent=self)
            return
        try:
            cur_idx = self._all_ids.index(self._current_id)
        except ValueError:
            cur_idx = -1
        if cur_idx <= 0:
            InfoBar.info(title="已是第一个数据", content="没有上一个可回退的 ID。",
                         orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=2000, parent=self)
            return

        prev_id = self._all_ids[cur_idx - 1]
        ret = QMessageBox.question(
            self, "回到上一个？",
            f"进入上一个数据将删去其当前结果：\n"
            f"• 从 out.txt 中移除「{prev_id}」的一条记录\n"
            f"• 删除输出 obj 文件（若存在）\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return

        try:
            obj_path = (self._out_root / "obj" / f"{prev_id}.obj") if self._out_root else None
            if obj_path and obj_path.exists():
                obj_path.unlink(missing_ok=True)
        except Exception:
            pass

        removed = False
        try:
            removed = self._remove_done_last_occurrence(prev_id)
        except Exception:
            removed = False

        self._update_progress_rows()

        self.viewerInterface.clear_selection()
        self._current_id = prev_id
        self.vi_idRow.setValue(prev_id)
        try:
            self.viewerInterface.show_id(self._dataset_root, prev_id)
            self.viewerInterface.force_full_render()
        except Exception as e:
            InfoBar.error(title="加载失败", content=f"加载 {prev_id} 失败：{e}",
                          orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=3000, parent=self)
            return

        self.vi_edgeRow.setValue("0")
        InfoBar.success(title="已回到上一个",
                        content=f"{'已清除旧结果，' if removed else ''}当前编辑：{prev_id}",
                        orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=2000, parent=self)

    def _onSaveNextClicked(self):
        if not self._dataset_root or not self._out_root or not self._current_id:
            InfoBar.warning(title="未加载数据", content="请先在“数据”页加载数据集。",
                            orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=2000, parent=self)
            return

        out_obj = self._out_root / "obj" / f"{self._current_id}.obj"
        try:
            nv, ne = self.viewerInterface.export_pruned_obj(out_obj)
        except Exception as e:
            InfoBar.error(title="保存失败", content=f"写入 OBJ 失败：{e}",
                          orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=3000, parent=self)
            return

        self._append_done(self._current_id)
        self._update_progress_rows()
        self._toast_ok("已保存", f"写入 {out_obj.name}（v={nv}, l={ne}）")

        next_id = self._pick_next_pending_id(self._current_id)
        if not next_id:
            InfoBar.success(title="完成", content="全部项已处理完毕。",
                            orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=2500, parent=self)
            try:
                self.viewerInterface.set_selection_mode(False)
                self.viewerInterface.clear()
                self.viewerInterface.force_full_render()
            except Exception:
                pass
            self.vi_idRow.setValue("-"); self.vi_todoRow.setValue("0"); self.vi_edgeRow.setValue("0"); self.vi_progRow.setValue("0 / 0"); self.vi_selRow.setValue("关闭 (F)")
            self.datasetInterface.pathLabel.setText("未选择")
            self.datasetInterface.totalRow.setValue("--"); self.datasetInterface.doneRow.setValue("--"); self.datasetInterface.todoRow.setValue("--")
            self.datasetInterface.refreshBtn.setEnabled(False)
            self.switchTo(self.datasetInterface)
            return

        self._current_id = next_id
        self.vi_idRow.setValue(next_id)
        try:
            self.viewerInterface.show_id(self._dataset_root, next_id)
            self.viewerInterface.force_full_render()
        except Exception as e:
            InfoBar.error(title="切换失败", content=f"加载 {next_id} 失败：{e}",
                          orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=3000, parent=self)
            return
        self.vi_edgeRow.setValue("0")

    # ---------------- 队列文本变化 ----------------
    def _onOpsChanged(self, text: str):
        try:
            self.vi_opsView.setPlainText(text)
            sb = self.vi_opsView.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception:
            pass
        try:
            n = len(self.viewerInterface.selected_ids())
            self.vi_edgeRow.setValue(str(n))
        except Exception:
            pass

    def _onOpenOutputClicked(self):
        if not self._out_root:
            InfoBar.warning(title="没有输出目录", content="请先加载数据并保存一次结果。",
                            orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT, duration=2000, parent=self)
            return
        obj_dir = self._out_root / "obj"
        target_file = obj_dir / f"{self._current_id}.obj" if (self._current_id) else None
        if target_file and target_file.exists():
            ok = self._reveal_in_file_manager(target_file, select=True)
            if not ok:
                self._reveal_in_file_manager(target_file.parent, select=False)
            return

        if obj_dir.exists():
            self._reveal_in_file_manager(obj_dir, select=False)
        else:
            self._reveal_in_file_manager(self._out_root, select=False)

    def _reveal_in_file_manager(self, path: Path, *, select: bool) -> bool:
        try:
            p = Path(path).resolve()
            if sys.platform.startswith("win"):
                if select and p.exists():
                    subprocess.Popen(["explorer", "/select,", str(p)])
                    return True
                folder = p if p.is_dir() else p.parent
                os.startfile(str(folder))  # type: ignore[attr-defined]
                return True
            elif sys.platform == "darwin":
                if select and p.exists():
                    subprocess.run(["open", "-R", str(p)])
                    return True
                folder = p if p.is_dir() else p.parent
                subprocess.run(["open", str(folder)])
                return True
            else:
                folder = p if p.is_dir() else p.parent
                subprocess.Popen(["xdg-open", str(folder)])
                return True
        except Exception:
            return False

    # ---------------- 主题/样式 ----------------
    def _onThemeChanged(self, name: str):
        setTheme = None; ThemeEnum = None
        try:
            from qfluentwidgets import setTheme, Theme as ThemeEnum
        except Exception:
            try:
                from qfluentwidgets import setTheme, FluentTheme as ThemeEnum
            except Exception:
                pass

        if setTheme and ThemeEnum:
            try:
                setTheme(ThemeEnum.DARK if name.lower() == "dark" else ThemeEnum.LIGHT)
            except Exception:
                pass

        if name.lower() == "dark":
            css = """
            #opsPanel { background:#1f1f1f; border:1px solid #303030; border-radius:8px; }
            #opsPanel QPlainTextEdit { background:transparent; border:none; color:#E6E6E6;
                selection-background-color:#264F78; font-family:Consolas,"Cascadia Code","Fira Code",monospace; font-size:13pt; }
            """
        else:
            css = """
            #opsPanel { background:#ffffff; border:1px solid #d0d0d0; border-radius:8px; }
            #opsPanel QPlainTextEdit { background:transparent; border:none; color:#2b2b2b;
                selection-background-color:#BBD7FB; font-family:Consolas,"Cascadia Code","Fira Code",monospace; font-size:13pt; }
            """
        try:
            self.vi_opsPanel.setStyleSheet(css)
        except Exception:
            pass
    def _pick_next_pending_id(self, after_id: str | None) -> str | None:
        done = self._read_done_ids()
        if not self._all_ids:
            return None
        start_idx = 0
        if after_id in self._all_ids:
            start_idx = self._all_ids.index(after_id) + 1
        for i in range(start_idx, len(self._all_ids)):
            if self._all_ids[i] not in done:
                return self._all_ids[i]
        for i in range(0, start_idx):
            if self._all_ids[i] not in done:
                return self._all_ids[i]
        return None

    def _apply_fixed_font_scale(self):
        qss = """
        BodyLabel,
        CaptionLabel,
        QPlainTextEdit,
        PushButton,
        PrimaryPushButton { font-size: 13pt; }
        CaptionLabel#sectionHeader { font-size: 14pt; font-weight: 600; }
        """
        try:
            self.setStyleSheet(self.styleSheet() + qss)
        except Exception:
            pass

    # ---------------- 尺寸节流 ----------------
    def resizeEvent(self, e):
        self._begin_geometry_change()
        super().resizeEvent(e)

    def eventFilter(self, obj, ev):
        if obj is getattr(self, "navigationInterface", None) and ev.type() in (QEvent.Resize, QEvent.Move):
            self._begin_geometry_change()
        return super().eventFilter(obj, ev)

    def _begin_geometry_change(self):
        if getattr(self, "_closing", False):
            return
        try:
            if not self._geom_busy:
                self._geom_busy = True
                self.viewerInterface.setUpdatesEnabled(False)
        except Exception:
            pass
        self._geomTimer.start(180)

    def _end_geometry_change(self):
        if getattr(self, "_closing", False):
            return
        try:
            self.viewerInterface.setUpdatesEnabled(True)
            self.viewerInterface.force_full_render()
        except Exception:
            pass
        self._geom_busy = False

    def _get_current_theme_name(self) -> str:
        try:
            from qfluentwidgets import isDarkTheme
            if callable(isDarkTheme) and isDarkTheme():
                return "dark"
        except Exception:
            pass
        try:
            from qfluentwidgets import qconfig
            th = getattr(qconfig, "theme", None)
            name = str(th).lower()
            if "dark" in name: return "dark"
            if "light" in name: return "light"
        except Exception:
            pass
        return "dark"

    def _apply_help_theme(self, name: str | None = None):
        if not hasattr(self, "_helpView") or self._helpView is None:
            return
        theme = (name or self._get_current_theme_name()).lower()
        bg = "#1f1f1f" if theme == "dark" else "#ffffff"
        self._helpView.setStyleSheet(f"""
            QTextBrowser {{
                background: {bg};
                border: none;
                font-size: 13pt;
                line-height: 1.55;
            }}
            QAbstractScrollArea > QWidget#qt_scrollarea_viewport {{
                background: {bg};
            }}
        """)

    # ---------------- 第二块：加载/填充/渲染 ----------------
    def _onLoadSecondClicked(self):
        """第二块“加载数据”（副本页使用）。"""
        root_dir = QFileDialog.getExistingDirectory(
            self, "选择包含 all.txt 与 data/ 的根目录",
            str(self._dataset_root or Path.home())
        )
        if not root_dir:
            return
        root = Path(root_dir)
        all_txt = root / "all.txt"
        data_dir = root / "data"

        if not all_txt.is_file() or not data_dir.is_dir():
            self._notify_error("目录结构不正确", "需要包含 all.txt 与 data/ 目录。")
            return

        try:
            ids: list[str] = []
            with all_txt.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        ids.append(s)
        except Exception as e:
            self._notify_error("读取 all.txt 失败", str(e))
            return

        if not ids:
            self._notify_error("all.txt 为空", "未解析到任何 ID。")
            return

        first_id = ids[0]
        first_id_dir = data_dir / first_id
        if not first_id_dir.is_dir():
            self._notify_error("缺少首个 ID 的目录", f"未找到目录：data/{first_id}")
            return

        try:
            file_names = [p.name for p in first_id_dir.rglob("*") if p.is_file()]
        except Exception as e:
            self._notify_error("遍历文件失败", str(e))
            return

        # 更新第二块显示
        self.datasetInterface.pathLabel2.setText(str(root))
        self._set_stat_row_value(self.datasetInterface.totalRow2, str(len(file_names)))

        # 缓存给“视图（副本）”
        self._ds2_root = root
        self._ds2_ids = ids
        self._ds2_first_id = first_id
        self._ds2_first_files = file_names

        self._ds2_current_idx = 0  # 当前是第 0 个（从 0 计）
        self._ds2_current_id = first_id  # 当前 ID

        # 填充下拉，默认选中第一个
        self._viz2_fill_id_combo(ids)
        self._viz2_select_combo_by_id(first_id)

        # 跳转并填充表格（这步是关键）
        self.switchTo(self.viewerPage2)
        self._viz2_populate_file_list(self._ds2_current_id)  # ← 传入当前 ID

        self._notify_ok("加载成功", f"共 {len(ids)} 个 ID；首个 ID='{first_id}'，文件数 {len(file_names)}。")




    def _set_stat_row_value(self, row, text: str):
        try:
            row.setValue(text)
        except Exception:
            try:
                row.valueLabel.setText(text)
            except Exception:
                pass

    def _notify_ok(self, title: str, content: str = ""):
        try:
            InfoBar.success(title, content, parent=self, position=InfoBarPosition.TOP_RIGHT, duration=2500)
        except Exception:
            QMessageBox.information(self, title, content)

    def _notify_error(self, title: str, content: str = ""):
        try:
            InfoBar.error(title, content, parent=self, position=InfoBarPosition.TOP_RIGHT, duration=3500)
        except Exception:
            QMessageBox.critical(self, title, content)

    # ———— 视图（副本）表格/交互 ————
    def _viz2_populate_file_list(self, id_str: str | None = None):
        # 1) 选出要显示的 ID
        ids = getattr(self, "_ds2_ids", []) or []
        root = getattr(self, "_ds2_root", None)
        if not ids or not root:
            return

        # 没传就用当前；都没有就退回第一项
        if id_str is None:
            id_str = getattr(self, "_ds2_current_id", None) or ids[0]

        # 同步当前索引
        try:
            self._ds2_current_idx = ids.index(id_str)
        except ValueError:
            self._ds2_current_idx = 0
            id_str = ids[0]

        # 2) 计算该 ID 下要展示的文件
        base_dir = (root / "data" / id_str).resolve()
        try:
            files = [p.name for p in base_dir.rglob("*") if p.is_file()]
        except Exception:
            files = []

        self.vi2_tree.blockSignals(True)
        self.vi2_tree.clear()
        self._viz2_row_state.clear()

        width_choices = list(range(1, 21))  # 1..20

        tmpl = getattr(self, "_viz2_style_template", None) or {}

        for fname in files:
            abs_path = str((base_dir / fname).resolve())
            item = QTreeWidgetItem(self.vi2_tree, [fname, "", ""])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

            # —— 应用模板（有就用模板，否则默认：未勾选、白色、宽度3）——
            trow = tmpl.get(fname, {})
            default_checked = bool(trow.get("checked", False))
            default_color = tuple(trow.get("color", (1.0, 1.0, 1.0)))
            default_width = float(trow.get("width", 3.0))

            item.setCheckState(0, Qt.Checked if default_checked else Qt.Unchecked)

            # 颜色按钮
            color_btn = PushButton(" ", self.vi2_tree)
            color_btn.setFixedWidth(36);
            color_btn.setFixedHeight(20)
            self._apply_btn_color(color_btn, default_color)
            self.vi2_tree.setItemWidget(item, 1, color_btn)

            # 宽度下拉（1..20）
            width_box = QComboBox(self.vi2_tree)
            for w in width_choices:
                width_box.addItem(str(w), float(w))
            try:
                width_box.setCurrentIndex(width_choices.index(int(round(default_width))))
            except Exception:
                width_box.setCurrentIndex(width_choices.index(3))
            self.vi2_tree.setItemWidget(item, 2, width_box)

            # 保存行状态（用于后续 show_layers）
            self._viz2_row_state[item] = {
                "path": abs_path,
                "color": default_color,
                "width": float(width_box.currentData()),
            }

            color_btn.clicked.connect(lambda _, it=item: self._viz2_pick_color_for(it))
            width_box.currentIndexChanged.connect(lambda _, it=item: self._viz2_width_changed_for(it))

        self.vi2_tree.blockSignals(False)

        # 只连一次也可以，这里简单处理：每次重建后重连
        self.vi2_tree.itemChanged.connect(self._viz2_item_toggled)
        self._viz2_apply_layers_from_tree(reset_camera_on_add=True)
        self._viz2_select_combo_by_id(id_str)

        # 左侧显示 ID
        try:
            self.vi2_idRow.setValue(id_str)
            self.vi2_progRow.setValue(f"{self._ds2_current_idx + 1} / {len(ids)}")
        except Exception:
            pass
        # 4) 清空旧图层，确保与新列表同步（可选）
        try:
            self.viewerInterface2.clear_layers()
        except Exception:
            pass


    @staticmethod
    def _apply_btn_color(btn, rgb):
        r, g, b = [int(round(x * 255)) for x in rgb]
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #666;")

    def _viz2_item_toggled(self, item, col):
        if col != 0 or item not in self._viz2_row_state:
            return
        checked = (item.checkState(0) == Qt.Checked)
        self._viz2_apply_layers_from_tree(reset_camera_on_add=checked)

    def _viz2_pick_color_for(self, item):
        if item not in self._viz2_row_state:
            return
        from PySide6 import QtGui
        st = self._viz2_row_state[item]
        orig = st["color"]
        qcol = QColorDialog.getColor(QtGui.QColor.fromRgbF(*orig), self, "选择颜色")
        if not qcol.isValid():
            return
        rgb = (qcol.redF(), qcol.greenF(), qcol.blueF())
        st["color"] = rgb
        btn = self.vi2_tree.itemWidget(item, 1)
        if btn:
            self._apply_btn_color(btn, rgb)
        if item.checkState(0) == Qt.Checked:
            self._viz2_apply_layers_from_tree()

    def _viz2_width_changed_for(self, item):
        if item not in self._viz2_row_state:
            return
        box = self.vi2_tree.itemWidget(item, 2)
        if box:
            st = self._viz2_row_state[item]
            st["width"] = float(box.currentData())
            if item.checkState(0) == Qt.Checked:
                self._viz2_apply_layers_from_tree()

    def _viz2_apply_layers_from_tree(self, *, reset_camera_on_add=False):
        layers = []
        for i in range(self.vi2_tree.topLevelItemCount()):
            it = self.vi2_tree.topLevelItem(i)
            if it in self._viz2_row_state and it.checkState(0) == Qt.Checked:
                st = self._viz2_row_state[it]
                layers.append((st["path"], st["color"], st["width"]))

        try:
            self.viewerInterface2.show_layers(layers, reset_camera_if_first=reset_camera_on_add)
            if reset_camera_on_add and layers:
                self.viewerInterface2.reset()
        except Exception as e:
            self._notify_error("刷新可视化失败", str(e))

    # ---------------- 说明 HTML ----------------

    def _build_help_html(self) -> str:
        return """<h2>使用说明</h2>

<h3>一、线框标注（核心功能）</h3>
<ul>
  <li><b>加载数据</b>：加载数据文件夹dataset_root，测试中为加载demo1文件夹。</li>
  <li><b>选择模式（F）</b>：切换框选模式并锁定/解锁视角；拖拽橡皮筋可批量选中/反选边。</li>
  <li><b>操作队列</b>：每次（点击/框选）会以一组 ID 入队，采用奇偶翻转统计，界面中部实时展示。</li>
  <li><b>可视化设置</b>：在“设置”页可调整点云大小与颜色、线框粗细与颜色、选中样式与主题。</li>
</ul>
<table>
  <tr><th>操作</th><th>按钮文字</th><th>快捷键</th><th>说明</th></tr>
  <tr><td>切换选择模式</td><td>选择模式 (F)</td><td><code>F</code></td><td>开启/关闭框选并锁定/解锁视角。</td></tr>
  <tr><td>撤销上一步</td><td>撤销 (Z)</td><td><code>Z</code></td><td>移除队列中最后一组操作并刷新渲染。</td></tr>
  <tr><td>重置当前数据</td><td>重置本数据 (Y)</td><td><code>Y</code></td><td>清空当前 ID 的选择与队列。</td></tr>
</table>
<ul><li>数据文件夹结构</li></ul>
<pre><code>dataset_root/
├─ all.txt               # 待处理 ID 列表（每行一个）
├─ obj/                  # 输入：线框 OBJ
│  ├─ id1.obj
│  └─ ...
├─ xyz/                  # 输入：点云 XYZ
│  ├─ id1.xyz
│  └─ ...
└─ out/
   ├─ out.txt            # 已完成的 ID（程序自动维护）
   └─ obj/               # 输出：结果模型
      ├─ id1.obj
      └─ ...
</code></pre>

<h3>二、数据管理与导出（核心功能）</h3>
<ol>
  <li><b>加载数据</b>：加载数据文件夹dataset_root（测试中为加载demo2文件夹）。</li>
  <li><b>可视化操作</b>：勾选需要可视化的文件，自动进行可视化，并且可以调整对应参数。</li>
  <li><b>上一个/下一个(A/D)</b>：加载不同的数据，请确保所有数据所含的文件一致。</li>
  <li><b>标记id</b>：会记录改数据的ID，存储到dataset_root（测试中为加载demo2文件夹），便于筛选异常数据。</li>
</ol>
<ul><li>数据文件夹结构</li></ul>
<pre><code>dataset_root/
├─ all.txt                    # 待处理 ID 列表（每行一个）
└─ data/                      # 各 ID 的输入数据
   ├─ id1/
   │  ├─ data_type1.xyz       # 点云（XYZ）
   │  ├─ data_type2.obj       # 线框（OBJ）
   │  └─ ...                  # 其它文件
   ├─ id2/
   │  ├─ data_type1.xyz
   │  ├─ data_type2.obj
   │  └─ ...
   └─ ...                     # 更多 ID
</code></pre>

<hr />
<p><i>Lebel MyContour — 版本：v1.1.0　作者：Senjer Feng</i></p>
"""


    def closeEvent(self, e):
        # Stop timers / detach compare overlays to reduce GL warnings on exit
        try:
            if hasattr(self, "_geomTimer"):
                self._geomTimer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "viewerInterface2"):
                self.viewerInterface2.end_compare_mode()
        except Exception:
            pass
        try:
            super().closeEvent(e)
        except Exception:
            pass

    def closeEvent(self, e):
        # 标记进入关闭阶段，禁止再触发任何强制渲染
        self._closing = True
        try:
            # 停止节流计时器，移除来自导航的频繁 resize 事件
            if hasattr(self, "_geomTimer"):
                self._geomTimer.stop()
            if hasattr(self, "navigationInterface"):
                try:
                    self.navigationInterface.removeEventFilter(self)
                except Exception:
                    pass
        except Exception:
            pass
        # 先让两个嵌入的 VTK 视图干净下线
        try:
            if hasattr(self, "viewerInterface") and self.viewerInterface:
                self.viewerInterface.shutdown_vtk()
        except Exception:
            pass
        try:
            if hasattr(self, "viewerInterface2") and self.viewerInterface2:
                self.viewerInterface2.shutdown_vtk()
        except Exception:
            pass
        # 交给父类继续关闭
        super().closeEvent(e)

    def _viz2_set_current_id(self, id_str: str):
        ids = getattr(self, "_ds2_ids", []) or []
        if not ids:
            return
        if id_str not in ids:
            # 不在列表里就不跳
            return
        self._ds2_current_id = id_str
        self._ds2_current_idx = ids.index(id_str)
        self._viz2_populate_file_list(id_str)

    def _viz2_capture_template_from_tree(self) -> dict[str, dict]:
        """把当前表格里的 勾选/颜色/宽度 收集成模板，key=文件名（不含路径）"""
        tmpl = {}
        for i in range(self.vi2_tree.topLevelItemCount()):
            it = self.vi2_tree.topLevelItem(i)
            if it in self._viz2_row_state:
                st = self._viz2_row_state[it]
                fname = it.text(0)
                checked = (it.checkState(0) == Qt.Checked)
                tmpl[fname] = {
                    "checked": checked,
                    "color": tuple(st["color"]),
                    "width": float(st["width"]),
                }
        return tmpl

    def _viz2_next_id(self):
        ids = getattr(self, "_ds2_ids", []) or []
        root = getattr(self, "_ds2_root", None)
        idx = int(getattr(self, "_ds2_current_idx", 0))

        if not ids or root is None:
            return
        if idx >= len(ids) - 1:
            # 最后一个：不做任何事（你说“没反应就行”）
            return

        # 1) 先把当前表格设置存成模板（文件名 -> 勾选/颜色/宽度）
        self._viz2_style_template = self._viz2_capture_template_from_tree()

        # 2) 下一份 ID
        next_idx = idx + 1
        next_id = ids[next_idx]

        # 3) 检查文件名集合是否一致；不一致则清空模板（回默认）
        base_dir = (root / "data" / next_id).resolve()
        try:
            next_files = sorted([p.name for p in base_dir.rglob("*") if p.is_file()])
        except Exception:
            next_files = []

        if self._viz2_style_template:
            if set(next_files) != set(self._viz2_style_template.keys()):
                # 结构不一致：重置模板
                self._viz2_style_template = {}

        # 4) 更新当前索引与ID，刷新列表（会应用模板）
        self._ds2_current_idx = next_idx
        self._ds2_current_id = next_id
        self._viz2_populate_file_list(next_id)
        self._viz2_select_combo_by_id(self._ds2_current_id)

    def _viz2_prev_id(self):
        # 先占位，后面实现与 _viz2_next_id 对称的逻辑即可
        ids = getattr(self, "_ds2_ids", []) or []
        root = getattr(self, "_ds2_root", None)
        idx = int(getattr(self, "_ds2_current_idx", 0))
        if not ids or root is None:
            return
        if idx <= 0:
            return
        self._viz2_style_template = self._viz2_capture_template_from_tree()
        prev_idx = idx - 1
        prev_id = ids[prev_idx]
        base_dir = (root / "data" / prev_id).resolve()
        try:
            prev_files = sorted([p.name for p in base_dir.rglob("*") if p.is_file()])
        except Exception:
            prev_files = []
        if self._viz2_style_template and set(prev_files) != set(self._viz2_style_template.keys()):
            self._viz2_style_template = {}
        self._ds2_current_idx = prev_idx
        self._ds2_current_id = prev_id
        self._viz2_populate_file_list(prev_id)
        self._viz2_select_combo_by_id(self._ds2_current_id)

    def _viz2_mark_current(self):
        # 先做个最小实现：把当前ID存内存 + 提示
        mid = getattr(self, "_ds2_current_id", None)
        if not mid:
            return
        if not hasattr(self, "_ds2_marked_ids"):
            self._ds2_marked_ids = set()
        self._ds2_marked_ids.add(mid)
        self._notify_ok("已记录", f"ID: {mid}")

    def _viz2_open_current_folder(self):
        root = getattr(self, "_ds2_root", None)
        mid = getattr(self, "_ds2_current_id", None)
        if not (root and mid):
            return
        self._reveal_in_file_manager((root / "data" / mid), select=False)

    def _onViz2RecordClicked(self):
        """把当前 ID 追加记录到 self._ds2_root/record.txt（去重）"""
        root = getattr(self, "_ds2_root", None)
        cur_id = getattr(self, "_ds2_current_id", None)

        if not root or not cur_id:
            try:
                InfoBar.warning(title="未加载数据", content="请先在“视图（副本）”加载数据。",
                                orient=Qt.Horizontal, isClosable=True,
                                position=InfoBarPosition.TOP_RIGHT, duration=2000, parent=self)
            except Exception:
                QMessageBox.warning(self, "未加载数据", "请先在“视图（副本）”加载数据。")
            return

        record_path = Path(root) / "record.txt"
        try:
            # 读取已存在的条目（去重）
            existed = set()
            if record_path.exists():
                with record_path.open("r", encoding="utf-8", errors="ignore") as f:
                    existed = {ln.strip() for ln in f if ln.strip()}

            if cur_id in existed:
                try:
                    InfoBar.info(title="已记录过", content=f"{cur_id} 已存在于 {record_path.name}",
                                 orient=Qt.Horizontal, isClosable=True,
                                 position=InfoBarPosition.TOP_RIGHT, duration=1800, parent=self)
                except Exception:
                    QMessageBox.information(self, "已记录过", f"{cur_id} 已存在于 {record_path.name}")
                return

            # 追加写入末尾
            with record_path.open("a", encoding="utf-8") as f:
                f.write(cur_id + "\n")

            try:
                InfoBar.success(title="已记录", content=f"写入 {record_path.name}: {cur_id}",
                                orient=Qt.Horizontal, isClosable=True,
                                position=InfoBarPosition.TOP_RIGHT, duration=1800, parent=self)
            except Exception:
                QMessageBox.information(self, "已记录", f"写入 {record_path.name}: {cur_id}")

        except Exception as e:
            try:
                InfoBar.error(title="记录失败", content=str(e),
                              orient=Qt.Horizontal, isClosable=True,
                              position=InfoBarPosition.TOP_RIGHT, duration=3000, parent=self)
            except Exception:
                QMessageBox.critical(self, "记录失败", str(e))

    def _viz2_fill_id_combo(self, ids: list[str]):
        """用 all.txt 的ID填充下拉（显示成 1..N 递增）。"""
        self.vi2_idCombo.blockSignals(True)
        self.vi2_idCombo.clear()
        N = len(ids)
        for i, id_str in enumerate(ids, start=1):  # 1..N
            # 任选其一的显示格式
            # disp = f"{i}-{id_str}"
            disp = f"{i}/{N}  {id_str}"
            self.vi2_idCombo.addItem(disp, userData=id_str)  # 用 itemData 存真实ID
        self.vi2_idCombo.blockSignals(False)

    def _viz2_select_combo_by_id(self, id_str: str):
        """让下拉框选中指定 ID（不触发切换）。"""
        if not hasattr(self, "vi2_idCombo"):
            return
        self.vi2_idCombo.blockSignals(True)
        for i in range(self.vi2_idCombo.count()):
            if self.vi2_idCombo.itemData(i) == id_str:
                self.vi2_idCombo.setCurrentIndex(i)
                break
        self.vi2_idCombo.blockSignals(False)

    def _viz2_on_id_combo_changed(self, idx: int):
        """下拉选择变更 → 切换到对应ID。"""
        if idx < 0 or idx >= self.vi2_idCombo.count():
            return
        id_str = self.vi2_idCombo.itemData(idx)
        if not id_str:
            return
        # 复用“模板迁移 + 切换ID”的逻辑
        self._viz2_switch_to_id(id_str)

    def _viz2_switch_to_id(self, id_str: str):
        """切到任意ID：保存当前表格模板 → 检查新ID文件集合 → 应用/清空模板 → 刷新。"""
        ids = getattr(self, "_ds2_ids", []) or []
        root = getattr(self, "_ds2_root", None)
        if not ids or root is None or id_str not in ids:
            return

        # 1) 记录当前模板（文件名 -> 勾选/颜色/宽度）
        self._viz2_style_template = self._viz2_capture_template_from_tree()

        # 2) 新ID文件集合；若与模板的 key 不一致，清空模板（避免错配）
        base_dir = (root / "data" / id_str).resolve()
        try:
            new_files = sorted([p.name for p in base_dir.rglob("*") if p.is_file()])
        except Exception:
            new_files = []
        if self._viz2_style_template and set(new_files) != set(self._viz2_style_template.keys()):
            self._viz2_style_template = {}

        # 3) 更新索引/ID并刷新列表（会自动应用模板）
        self._ds2_current_id = id_str
        try:
            self._ds2_current_idx = ids.index(id_str)
        except ValueError:
            self._ds2_current_idx = 0

        self._viz2_populate_file_list(id_str)
        # 同步左侧显示与下拉选中
        self.vi2_idRow.setValue(id_str)
        self.vi2_progRow.setValue(f"{self._ds2_current_idx + 1} / {len(ids)}")
        self._viz2_select_combo_by_id(id_str)







