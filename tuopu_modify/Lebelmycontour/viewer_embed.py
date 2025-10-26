#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import List, Set, Optional, Tuple

from PySide6.QtCore import Signal, QPoint, Qt, QEvent, QRect, QSize
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QRubberBand
from typing import Iterable

# ---- VTK / numpy / vedo ----
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.all as vtk
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOPLY import vtkPLYReader

import numpy as np

# 保留原有标志位与变量名，避免动到其它代码
_VTK_OK, _VTK_ERR = True, None
_NP_OK,  _NP_ERR  = True, None
_VEDO_OK, _VEDO_ERR = True, None



# ================== 工具：选择队列（奇偶切换） ==================
class EdgeSelectionQueue:
    def __init__(self):
        # 每个元素是一组 ids（一次操作/一次框选/一次点击）
        self.ops: List[List[int]] = []

    def push_group(self, ids: Iterable[int]):
        """把同一轮操作（一次框选/一次点击）作为一个分组入队"""
        group = [int(i) for i in ids if i is not None]
        if group:
            self.ops.append(group)

    def push_one(self, i: int):
        """单个 id 也按一组处理，便于统一撤销"""
        self.push_group([i])

    def undo(self, n: int = 1):
        for _ in range(max(0, n)):
            if self.ops:
                self.ops.pop()

    def _iter_flat(self):
        for g in self.ops:
            for i in g:
                yield i

    def selected_set(self) -> Set[int]:
        """奇偶翻转（对所有组内 id 展开后 XOR 计数）"""
        if not self.ops:
            return set()
        bit = {}
        for i in self._iter_flat():
            bit[i] = bit.get(i, 0) ^ 1
        return {i for i, v in bit.items() if v == 1}

    # 辅助：用于界面显示的摘要
    def summarize(self, per_line_limit: int = 12) -> str:
        if not self.ops:
            return "（空）"
        lines = []
        for k, g in enumerate(self.ops, 1):
            if len(g) <= per_line_limit:
                body = ", ".join(str(x) for x in g)
            else:
                head = ", ".join(str(x) for x in g[:per_line_limit])
                body = f"{head} …(+{len(g)-per_line_limit})"
            lines.append(f"{k}. toggle {len(g)} edges: [{body}]")
        return "\n".join(lines)

# ================== 主部件：QVTK + 锁相机选择 ==================
class ViewerEmbed(QFrame):
    """
    - 纯 VTK 渲染（QVTKRenderWindowInteractor + vtkRenderer）
    - show_id(root, id) / show_pair(xyz, obj)：默认同时加载点云与 OBJ 边
    - F 键或按钮切换“选择模式”：锁相机 + 橡皮筋框选边
    - 选择逻辑：用队列做奇偶切换；未选为灰，选中为红色加粗覆盖
    """
    # 可选：打开/关闭调试打印
    DEBUG = False
    opsChanged = Signal(str)  # 队列文本变化时发出

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewerInterface")
        # --- Compare mode (multi-file overlay, no selection) ---
        self._cmp_mode = False
        self._cmp_layers = {}  # path -> {type:'points'|'lines', 'actor':vtkActor, 'color':(r,g,b), 'width':float}
        self.setFrameShape(QFrame.StyledPanel)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        if not _VTK_OK:
            lay.addWidget(QLabel(f"VTK 不可用: {_VTK_ERR}", self))
            self._ready = False
            return

        # ---- 基础 VTK 场景 ----
        self.vtk = QVTKRenderWindowInteractor(self)
        lay.addWidget(self.vtk)

        # 关键：让 Qt 别给这个部件刷底色，同时声明这是不透明绘制区域
        self.vtk.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.vtk.setAutoFillBackground(False)

        # 保险：把 Qt 小部件的“窗口色”也设成黑（即便 Qt 真的去刷，也刷黑）
        try:
            pal = self.vtk.palette()
            pal.setColor(self.vtk.backgroundRole(), Qt.black)  # 或 QPalette.Window
            self.vtk.setPalette(pal)
        except Exception:
            pass

        self._ren = vtk.vtkRenderer()
        # 关键：背景+Alpha 都强制为黑且不透明
        self._ren.SetBackground(0.0, 0.0, 0.0)
        try:
            self._ren.SetBackgroundAlpha(1.0)  # VTK>=9 支持
        except Exception:
            pass

        self._rw = self.vtk.GetRenderWindow()

        # 关键：关闭 Alpha 位平面，避免“透明”背景把底层 Qt 白色露出来
        try:
            self._rw.SetAlphaBitPlanes(0)
        except Exception:
            pass

        self._rw.AddRenderer(self._ren)
        self._iren = self._rw.GetInteractor()

        # 默认交互：Trackball（常用）
        try:
            self._trackball = vtkInteractorStyleTrackballCamera()
            self._iren.SetInteractorStyle(self._trackball)
        except Exception:
            self._trackball = None

        try:
            self.vtk.Initialize()
        except Exception:
            pass

        # ---- 运行期状态 ----
        self._ready = True
        self._primed = False

        # 选择模式与交互相关
        self._sel_mode = False
        self._old_style = None          # 退出时恢复
        self._user_style = None         # vtkInteractorStyleUser
        self._obs_tokens: dict = {}     # VTK 观察者 Token
        self._dragging = False
        self._press_xy: Optional[Tuple[int, int]] = None
        self._qt_drag_active = False    # Qt/VTK 双通道互斥

        # 场景对象
        self._points_actor = None
        self._edges_actor = None        # 灰色全边
        self._sel_actor = None          # 红色选中覆盖
        self._edges_poly: Optional['vtk.vtkPolyData'] = None
        self._verts_np: Optional[np.ndarray] = None  # (N,3)
        self._edges_ij: List[tuple[int, int]] = []   # (i,j) 序列
        self._edge_count: int = 0

        # 选择队列
        self._queue = EdgeSelectionQueue()

        # 橡皮筋（Qt）
        self._rubber: Optional[QRubberBand] = None
        self._rubber_origin: Optional[QPoint] = None

        # 尺寸 & 坐标系换算
        self._last_widget_size: Optional[Tuple[int, int]] = None
        self._update_rw_size()
        self._widget_filter_installed = False

        # —— 线/点默认样式 ——（底层线框更白更粗）
        self._edge_color = (1.0, 1.0, 1.0)  # 纯白
        self._edge_width = 3.0
        self._sel_edge_color = (1.0, 0.2, 0.2)
        self._sel_edge_width = 4.0
        self._point_size_default = 4
        self._point_color = (0.95, 0.95, 0.95)

        # 抗锯齿
        try:
            self._rw.SetMultiSamples(8)
        except Exception:
            pass

    def _ops_summary(self) -> str:
        return self._queue.summarize()

    # -------------------- Qt 生命周期 --------------------
    def showEvent(self, e):
        super().showEvent(e)
        if self._ready and not self._primed:
            self._primed = True
            try:
                self._ren.ResetCamera()
                self._rw.Render()
            except Exception:
                pass

    # -------------------- 选择模式 --------------------
    def is_selection_mode(self) -> bool:
        return bool(self._sel_mode)

    def toggle_select_mode(self):
        self.set_selection_mode(not self._sel_mode)

    def set_selection_mode(self, enabled: bool):
        if not self._ready or enabled == self._sel_mode:
            return
        self._sel_mode = bool(enabled)
        try:
            if self._sel_mode:
                # 进入选择模式：锁相机（User style）
                try:
                    self._old_style = self._iren.GetInteractorStyle()
                except Exception:
                    self._old_style = None
                self._attach_user_style()
            else:
                # 退出：恢复交互样式，清理状态
                self._detach_user_style()
                if self._old_style is not None:
                    self._iren.SetInteractorStyle(self._old_style)
                elif self._trackball is not None:
                    self._iren.SetInteractorStyle(self._trackball)
                self._dragging = False
                self._press_xy = None
                self._qt_drag_active = False
                if self._rubber:
                    self._rubber.hide()
                    self._rubber_origin = None
        except Exception as e:
            print("[ViewerEmbed] set_selection_mode failed:", e)
        try:
            self._rw.Render()
        except Exception:
            pass

    # -------------------- 场景基础操作 --------------------
    def clear(self):
        if not self._ready:
            return
        actors = self._ren.GetActors()
        actors.InitTraversal()
        to_remove = []
        for _ in range(actors.GetNumberOfItems()):
            a = actors.GetNextActor()
            if a is not None:
                to_remove.append(a)
        for a in to_remove:
            self._ren.RemoveActor(a)

        self._points_actor = None
        self._edges_actor = None
        self._sel_actor = None
        self._edges_poly = None
        self._verts_np = None
        self._edges_ij = []
        self._edge_count = 0
        self._queue = EdgeSelectionQueue()
        try:
            self._rw.Render()
        except Exception:
            pass

        # ★ 新增：切换文件/清空场景时，让中间的队列视图也同步重置
        self.opsChanged.emit(self._ops_summary())


    def reset(self):
        if not self._ready:
            return
        self._ren.ResetCamera()
        self._rw.Render()

    # -------------------- 构建与显示 --------------------
    def _actor_from_vedo(self, vobj):
        for name in ("actor", "_actor", "vtkactor", "vtkActor"):
            if hasattr(vobj, name):
                return getattr(vobj, name)
        if hasattr(vobj, "GetActors"):
            col = vobj.GetActors()
            col.InitTraversal()
            return col.GetNextActor()
        return None

    def _add_xyz_actor(self, xyz_path: Path, point_size: int = 3):
        arr = np.loadtxt(str(xyz_path), dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        pts_np = np.asarray(arr[:, :3], dtype=np.float64)

        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetDataTypeToDouble()  # ★ 强制双精度
        vtk_pts.SetNumberOfPoints(len(pts_np))
        for i, (x, y, z) in enumerate(pts_np):
            vtk_pts.SetPoint(i, float(x), float(y), float(z))

        verts = vtk.vtkCellArray()
        for i in range(len(pts_np)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.SetVerts(verts)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetPointSize(self._point_size_default)
        prop.SetColor(*self._point_color)
        try:
            prop.SetRenderPointsAsSpheres(True)  # 让点渲染为小球，视觉更平滑（可选）
        except Exception:
            pass
        actor.PickableOff()
        self._points_actor = actor
        self._ren.AddActor(actor)

    def _parse_obj_vertices_edges(self, obj_path: Path) -> Optional[tuple[np.ndarray, List[tuple[int, int]]]]:
        """
        OBJ 边模型：
        - 优先使用 `l i j [k ...]`（折线 -> 相邻成段）
        - 若无 `l` 则用 `f` 拆边（环边）
        - 支持负索引、带斜杠索引（取顶点项）
        返回: (vertices[N,3], edges[(i,j)]，0-based，i<j 去重)
        """
        try:
            verts: List[List[float]] = []
            edges_l: List[tuple[int, int]] = []
            edges_f: List[tuple[int, int]] = []
            has_l = False

            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    ln = raw.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if ln.startswith("v "):
                        ps = ln.split()
                        if len(ps) >= 4:
                            try:
                                verts.append([float(ps[1]), float(ps[2]), float(ps[3])])
                            except Exception:
                                pass
                    elif ln.startswith("l "):
                        has_l = True
                        toks = ln.split()[1:]
                        idxs: List[int] = []
                        for t in toks:
                            s = t.split("/")[0]
                            if not s:
                                continue
                            try:
                                idxs.append(int(s))
                            except Exception:
                                continue
                        for a, b in zip(idxs[:-1], idxs[1:]):
                            edges_l.append((a, b))
                    elif ln.startswith("f "):
                        toks = ln.split()[1:]
                        idxs: List[int] = []
                        for t in toks:
                            s = t.split("/")[0]
                            if not s:
                                continue
                            try:
                                idxs.append(int(s))
                            except Exception:
                                continue
                        if len(idxs) >= 2:
                            cyc = idxs[1:] + idxs[:1]
                            for a, b in zip(idxs, cyc):
                                edges_f.append((a, b))

            if not verts:
                return None
            n = len(verts)

            def norm_one(i: int) -> Optional[int]:
                # OBJ: 正数 1-based；负数相对末尾
                j = (i - 1) if i > 0 else (n + i)
                return j if 0 <= j < n else None

            def norm_pair(a: int, b: int) -> Optional[tuple[int, int]]:
                ia, ib = norm_one(a), norm_one(b)
                if ia is None or ib is None or ia == ib:
                    return None
                return (ia, ib) if ia < ib else (ib, ia)

            edges = [norm_pair(a, b) for (a, b) in (edges_l if has_l else edges_f)]
            edges = [e for e in edges if e is not None]
            edges = sorted(set(edges))  # 去重
            return (np.array(verts, dtype=float), edges)
        except Exception as e:
            print("[ViewerEmbed] parse obj failed:", e)
            return None

    def _build_edges_from_vertices_edges(self, vertices: np.ndarray, edges: List[tuple[int, int]]):
        """根据顶点/边构建灰色线框 + 红色覆盖层（占位），并缓存投影数据"""
        # 缓存
        self._verts_np = np.asarray(vertices, float)
        self._edges_ij = [(int(i), int(j)) for (i, j) in edges]
        self._edge_count = len(self._edges_ij)

        # vtkPoints
        pts = vtk.vtkPoints()
        pts.SetDataTypeToDouble()
        for p in self._verts_np:
            pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

        # vtkCellArray -> Lines
        carr = vtk.vtkCellArray()
        for (i, j) in self._edges_ij:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, j)
            carr.InsertNextCell(line)

        edges_poly = vtk.vtkPolyData()
        edges_poly.SetPoints(pts)
        edges_poly.SetLines(carr)
        self._edges_poly = edges_poly

        # 底层线框（更白更粗）
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(edges_poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self._edges_poly = edges_poly
        self._edges_actor = actor
        self._style_lines(actor, self._edge_color, self._edge_width)
        self._ren.AddActor(actor)

        # 顶层红边（初始空；用 DataSetMapper 承接 ExtractSelection 的 UGrid）
        selMapper = vtk.vtkDataSetMapper()
        selMapper.SetInputData(vtk.vtkUnstructuredGrid())
        selActor = vtk.vtkActor()
        selActor.SetMapper(selMapper)
        selActor.GetProperty().SetColor(1.0, 0.2, 0.2)
        selActor.GetProperty().SetLineWidth(self._sel_edge_width * 2)
        selActor.GetProperty().SetLighting(False)
        self._sel_actor = selActor
        self._ren.AddActor(selActor)

        if self.DEBUG:
            print("[DEBUG] edges:", self._edge_count)

        try:
            self._rw.Render()
        except Exception:
            pass

    # -------------------- 选择 & 高亮 --------------------
    def _on_ids_toggled(self, ids: List[int]):
        self._queue.push_group(ids)
        self._update_selected_actor(self._queue.selected_set())
        self.opsChanged.emit(self._ops_summary())

    def get_ops(self) -> List[int]:
        """兼容旧逻辑：返回扁平化的一维操作序列"""
        # self._queue.ops 是 List[List[int]]，这里扁平化回 List[int]
        return [i for g in self._queue.ops for i in g]

    def get_ops_grouped(self) -> List[List[int]]:
        """返回分组后的操作队列（每组=一次框选/点击）"""
        return [list(g) for g in self._queue.ops]

    def undo_last_operation(self):
        self._queue.undo(1)
        self._update_selected_actor(self._queue.selected_set())
        self.opsChanged.emit(self._ops_summary())

    def apply_selection_ops(self, ops_groups: List[List[int]]):
        """用一串‘分组操作’重建当前选择"""
        self._queue = EdgeSelectionQueue()
        for g in (ops_groups or []):
            self._queue.push_group(g)
        self._update_selected_actor(self._queue.selected_set())
        self.opsChanged.emit(self._ops_summary())

    def _update_selected_actor(self, selected: Set[int]):
        """把选中的边作为红层显示；白层只显示未选中的边（避免重叠）"""
        if self._edges_poly is None:
            return

        # ---------- 1) 红色层：沿用你原来的写法 ----------
        idarr = vtk.vtkIdTypeArray()
        for i in sorted(int(x) for x in selected):
            if 0 <= i < self._edge_count:
                idarr.InsertNextValue(i)

        selNode = vtk.vtkSelectionNode()
        selNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selNode.SetSelectionList(idarr)

        selection = vtk.vtkSelection()
        selection.AddNode(selNode)

        extract = vtk.vtkExtractSelection()
        extract.SetInputData(0, self._edges_poly)
        extract.SetInputData(1, selection)
        extract.Update()

        ug_selected = extract.GetOutput()  # vtkUnstructuredGrid

        if self._sel_actor is None:
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(ug_selected)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.2, 0.2)         # 沿用你原来的红色控制名
            actor.GetProperty().SetLineWidth(self._sel_edge_width)
            actor.GetProperty().SetLighting(False)
            self._sel_actor = actor
            self._ren.AddActor(actor)
        else:
            mapper = self._sel_actor.GetMapper()
            if isinstance(mapper, vtk.vtkDataSetMapper):
                mapper.SetInputData(ug_selected)
            else:
                m = vtk.vtkDataSetMapper()
                m.SetInputData(ug_selected)
                self._sel_actor.SetMapper(m)

        # ---------- 2) 白色层：改成“未选集合” ----------
        if self._edges_actor is not None:
            base_mapper = self._edges_actor.GetMapper()

            # 无选择：白层恢复显示全部
            if not selected:
                if isinstance(base_mapper, vtk.vtkDataSetMapper):
                    base_mapper.SetInputData(self._edges_poly)   # 直接喂回 PolyData
                else:
                    m = vtk.vtkDataSetMapper()
                    m.SetInputData(self._edges_poly)
                    self._edges_actor.SetMapper(m)
            else:
                # 构造未选 id 列表
                base_ids = vtk.vtkIdTypeArray()
                # 选中是 set，查找 O(1)
                for i in range(self._edge_count):
                    if i not in selected:
                        base_ids.InsertNextValue(i)

                baseNode = vtk.vtkSelectionNode()
                baseNode.SetFieldType(vtk.vtkSelectionNode.CELL)
                baseNode.SetContentType(vtk.vtkSelectionNode.INDICES)
                baseNode.SetSelectionList(base_ids)

                baseSel = vtk.vtkSelection()
                baseSel.AddNode(baseNode)

                baseExtract = vtk.vtkExtractSelection()
                baseExtract.SetInputData(0, self._edges_poly)
                baseExtract.SetInputData(1, baseSel)
                baseExtract.Update()

                ug_unselected = baseExtract.GetOutput()

                if isinstance(base_mapper, vtk.vtkDataSetMapper):
                    base_mapper.SetInputData(ug_unselected)
                else:
                    m = vtk.vtkDataSetMapper()
                    m.SetInputData(ug_unselected)
                    self._edges_actor.SetMapper(m)

            # 保持白层样式（名字不改）
            self._style_lines(self._edges_actor, self._edge_color, self._edge_width)

        # ---------- 3) 刷新 ----------
        try:
            self._ren.ResetCameraClippingRange()
            self._rw.Render()
        except Exception:
            pass


    # 框选（屏幕空间命中：端点/中点/与矩形边相交）
    def _perform_box_select(self, x0: int, y0: int, x1: int, y1: int) -> List[int]:
        if self._verts_np is None or not self._edges_ij:
            return []

        rx0, rx1 = (x0, x1) if x0 <= x1 else (x1, x0)
        ry0, ry1 = (y0, y1) if y0 <= y1 else (y1, y0)

        def world_to_display_xy(p3):
            self._ren.SetWorldPoint(float(p3[0]), float(p3[1]), float(p3[2]), 1.0)
            self._ren.WorldToDisplay()
            dx, dy, _ = self._ren.GetDisplayPoint()
            return float(dx), float(dy)

        def inside(x, y): return (rx0 <= x <= rx1) and (ry0 <= y <= ry1)

        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        def seg_inter(a, b, c, d):
            if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]) or \
               max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
                return False
            return (cross(a, b, c) * cross(a, b, d) <= 0) and (cross(c, d, a) * cross(c, d, b) <= 0)

        rect_edges = [((rx0, ry0), (rx1, ry0)), ((rx1, ry0), (rx1, ry1)),
                      ((rx1, ry1), (rx0, ry1)), ((rx0, ry1), (rx0, ry0))]

        hit = []
        for eid, (i, j) in enumerate(self._edges_ij):
            a = world_to_display_xy(self._verts_np[i])
            b = world_to_display_xy(self._verts_np[j])

            if inside(*a) or inside(*b):
                hit.append(eid); continue
            mid = ((a[0]+b[0])*0.5, (a[1]+b[1])*0.5)
            if inside(*mid):
                hit.append(eid); continue
            for r0, r1 in rect_edges:
                if seg_inter(a, b, r0, r1):
                    hit.append(eid); break

        if self.DEBUG:
            print("[SEL] pick:", hit)
        return hit

    # -------------------- 选择模式：锁相机 & 事件通道 --------------------
    def _attach_user_style(self):
        if not _VTK_OK:
            return
        self._user_style = vtk.vtkInteractorStyleUser()
        self._iren.SetInteractorStyle(self._user_style)

        # 保证投影矩阵稳定
        try: self._iren.Initialize()
        except Exception: pass
        try: self._iren.Enable()
        except Exception: pass
        try: self._rw.Render()
        except Exception: pass

        # VTK 观察者（优先级 1.0）
        self._obs_tokens.clear()
        self._obs_tokens["LeftButtonPressEvent"]   = self._iren.AddObserver("LeftButtonPressEvent",   self._on_lpress,   1.0)
        self._obs_tokens["LeftButtonReleaseEvent"] = self._iren.AddObserver("LeftButtonReleaseEvent", self._on_lrelease, 1.0)
        self._obs_tokens["MouseMoveEvent"]         = self._iren.AddObserver("MouseMoveEvent",         self._on_lmove,    1.0)

        # 禁用其它交互
        for ev in ("RightButtonPressEvent","MiddleButtonPressEvent",
                   "MouseWheelForwardEvent","MouseWheelBackwardEvent"):
            self._obs_tokens[ev] = self._iren.AddObserver(ev, lambda o,e: None, 1.0)

        # Qt 事件兜底（一次安装即可）
        if not self._widget_filter_installed:
            self.vtk.installEventFilter(self)
            self._widget_filter_installed = True

    def _detach_user_style(self):
        if not self._user_style:
            return
        try:
            for _, tag in list(self._obs_tokens.items()):
                if tag is not None:
                    self._iren.RemoveObserver(tag)
        except Exception:
            pass
        self._obs_tokens.clear()
        self._user_style = None

    # VTK 事件（当 Qt 正在拖拽时静音，避免双触发抵消）
    def _on_lpress(self, obj, evt):
        if self._qt_drag_active:
            return
        try:
            self._press_xy = self._iren.GetEventPosition()
            self._dragging = True
            if self.DEBUG: print("[SEL] VTK press:", self._press_xy)
        except Exception:
            self._press_xy = None
            self._dragging = False

    def _on_lmove(self, obj, evt):
        pass

    def _on_lrelease(self, obj, evt):
        if self._qt_drag_active:
            return
        if not self._dragging or self._press_xy is None:
            self._dragging = False
            self._press_xy = None
            return
        try:
            x0, y0 = self._press_xy
            x1, y1 = self._iren.GetEventPosition()
            ids = self._perform_box_select(x0, y0, x1, y1)
            if ids:
                self._on_ids_toggled(ids)
        except Exception as e:
            print("[ViewerEmbed] box-select failed:", e)
        finally:
            self._dragging = False
            self._press_xy = None
            try: self._rw.Render()
            except Exception: pass

    # -------------------- Qt 事件（主通道 + 橡皮筋） --------------------
    def eventFilter(self, obj, event):
        if obj is self.vtk and self._sel_mode:
            et = event.type()
            if et == QEvent.Resize:
                self._update_rw_size()
            elif et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._on_qt_lpress(int(event.position().x()), int(event.position().y()))
            elif et == QEvent.MouseMove and (event.buttons() & Qt.LeftButton):
                self._on_qt_lmove(int(event.position().x()), int(event.position().y()))
            elif et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._on_qt_lrelease(int(event.position().x()), int(event.position().y()))
        return super().eventFilter(obj, event)

    def _on_qt_lpress(self, x_qt: int, y_qt: int):
        self._qt_drag_active = True  # 互斥
        x, y = self._qt_to_vtk_xy(x_qt, y_qt)
        self._press_xy = (x, y)
        self._dragging = True

        # 橡皮筋
        try:
            if self._rubber is None:
                self._rubber = QRubberBand(QRubberBand.Rectangle, self.vtk)
            self._rubber_origin = QPoint(int(x_qt), int(y_qt))
            self._rubber.setGeometry(QRect(self._rubber_origin, QSize()))
            self._rubber.show()
            self._rubber.raise_()
        except Exception:
            self._rubber = None
            self._rubber_origin = None

        if self.DEBUG:
            print("[SEL] QT  press:", (x, y), " from Qt:", (x_qt, y_qt), " rw_size:", self._last_widget_size)

    def _on_qt_lmove(self, x_qt: int, y_qt: int):
        if not self._sel_mode or self._rubber is None or self._rubber_origin is None:
            return
        rect = QRect(self._rubber_origin, QPoint(x_qt, y_qt)).normalized()
        self._rubber.setGeometry(rect)

    def _on_qt_lrelease(self, x_qt: int, y_qt: int):
        if not self._dragging or self._press_xy is None:
            self._qt_drag_active = False
            self._dragging = False
            self._press_xy = None
            if self._rubber: self._rubber.hide()
            self._rubber_origin = None
            return

        if self._rubber:
            self._rubber.hide()
        self._rubber_origin = None

        x1, y1 = self._qt_to_vtk_xy(x_qt, y_qt)
        x0, y0 = self._press_xy
        ids = self._perform_box_select(x0, y0, x1, y1)
        if self.DEBUG:
            print(f"[SEL] QT  release: {(x1,y1)}  rect=({x0},{y0})-({x1},{y1})  from Qt={(x_qt,y_qt)} size={self._last_widget_size}")
            print("[SEL] picked (QT):", ids)
        if ids:
            self._on_ids_toggled(ids)

        self._qt_drag_active = False
        self._dragging = False
        self._press_xy = None
        try: self._rw.Render()
        except Exception: pass

    # -------------------- 坐标与尺寸 --------------------
    def _update_rw_size(self):
        """更新渲染窗口的物理像素尺寸（VTK 事件坐标使用物理像素）"""
        try:
            w, h = self._rw.GetSize()
            if w <= 0 or h <= 0:
                dpr = float(getattr(self.vtk.windowHandle(), "devicePixelRatio", lambda: 1.0)())
                w = int(max(1, round(self.vtk.width() * dpr)))
                h = int(max(1, round(self.vtk.height() * dpr)))
        except Exception:
            dpr = float(getattr(self.vtk.windowHandle(), "devicePixelRatio", lambda: 1.0)())
            w = int(max(1, round(self.vtk.width() * dpr)))
            h = int(max(1, round(self.vtk.height() * dpr)))
        self._last_widget_size = (w, h)

    def _qt_to_vtk_xy(self, x_qt: int, y_qt: int) -> tuple[int, int]:
        """Qt 逻辑像素 -> VTK 物理像素（左上->左下 + DPR）"""
        self._update_rw_size()
        w, h = self._last_widget_size or (1, 1)
        try:
            dpr = float(self.vtk.windowHandle().devicePixelRatio())
        except Exception:
            dpr = 1.0
        x = int(round(x_qt * dpr))
        y = int(round(y_qt * dpr))
        return int(max(0, min(w - 1, x))), int(max(0, min(h - 1, h - 1 - y)))

    # -------------------- 外部接口 --------------------
    def show_pair(self, xyz_path: Optional[Path], obj_path: Optional[Path], *, point_size: int = 3):
        """同时加载点云与 OBJ；任一不存在则跳过。"""
        if not self._ready:
            return
        self.clear()
        if xyz_path and Path(xyz_path).exists() and str(xyz_path).lower().endswith(".xyz"):
            try:
                self._add_xyz_actor(Path(xyz_path), point_size=point_size)
            except Exception as e:
                print("[ViewerEmbed] 点云加载失败:", e)
        if obj_path and Path(obj_path).exists():
            try:
                parsed = self._parse_obj_vertices_edges(Path(obj_path))
                if parsed is None:
                    print("[ViewerEmbed] OBJ 无顶点，跳过:", obj_path)
                else:
                    verts, edges = parsed
                    if not edges:
                        print("[ViewerEmbed] OBJ 无边可用，跳过:", obj_path)
                    else:
                        self._build_edges_from_vertices_edges(verts, edges)
            except Exception as e:
                print("[ViewerEmbed] OBJ 加载失败:", e)
        self.reset()

    def show_id(self, root: Path, id_str: str, *, point_size: int = 3):
        """根据 id 同时展示 xyz 与 obj（默认都加载）。"""
        root = Path(root)
        xyz_path: Optional[Path] = root / "xyz" / f"{id_str}.xyz"
        obj_path: Optional[Path] = root / "obj" / f"{id_str}.obj"

        if not xyz_path.exists() and (root / "xyz").exists():
            cands = list((root / "xyz").glob(f"{id_str}.*"))
            xyz_path = cands[0] if cands else None
        if not obj_path.exists() and (root / "obj").exists():
            cands = list((root / "obj").glob(f"{id_str}.*"))
            obj_path = cands[0] if cands else None

        self.show_pair(xyz_path if xyz_path and xyz_path.exists() else None,
                       obj_path if obj_path and obj_path.exists() else None,
                       point_size=point_size)

    def show_path(self, path: Path):
        """兼容旧接口：传入 xyz 或 obj，推断 root/id 后转 show_id()。"""
        if not self._ready:
            return
        p = Path(path)
        root = p.parent.parent if (p.parent.name.lower() in ("xyz", "obj") and p.parent.parent.exists()) else p.parent
        self.show_id(root, p.stem)

    # ---- 选择队列公共 API（供外部调用） ----
    def push_edge_ids(self, ids: List[int]):
        self._on_ids_toggled(ids)

    def selected_ids(self) -> List[int]:
        return sorted(int(x) for x in self._queue.selected_set())

    def clear_selection(self):
        self._queue = EdgeSelectionQueue()
        self._update_selected_actor(set())
        self.opsChanged.emit(self._ops_summary())

    # === 新增：导出/保存相关 ===
    def get_ops(self) -> List[int]:
        """兼容旧逻辑：返回扁平化的一维操作序列"""
        return [i for g in self._queue.ops for i in g]

    def compute_pruned_graph(self, delete_eids: Optional[Set[int]] = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        计算“删边并去孤点”后的图：
        - delete_eids 为空则使用当前奇数次出现（selected）的边索引
        - 返回: 新 vertices (M,3) 与新 edges [(i,j)] (0-based)
        """
        if self._verts_np is None:
            return np.zeros((0, 3), float), []
        del_set = set(delete_eids) if delete_eids is not None else set(self.selected_ids())
        keep_edges = [e for idx, e in enumerate(self._edges_ij) if idx not in del_set]

        if not keep_edges:
            # 全删：没有边，也没有需要的点
            return np.zeros((0, 3), float), []

        used = sorted(set([i for (i, j) in keep_edges] + [j for (i, j) in keep_edges]))
        old2new = {old: ni for ni, old in enumerate(used)}
        verts_new = self._verts_np[used]
        edges_new = [(old2new[i], old2new[j]) for (i, j) in keep_edges]
        return verts_new, edges_new

    def export_pruned_obj(self, out_path: Path) -> Tuple[int, int]:
        """
        将当前“奇数次被选中的边”删除、去掉孤点，保存为 OBJ（v + l）。
        返回: (新顶点数, 新边数)
        """
        V, E = self.compute_pruned_graph()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for v in V:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for (i, j) in E:
                f.write(f"l {i+1} {j+1}\n")
        return len(V), len(E)

    def force_full_render(self):
        """强制刷新一帧；关闭/已释放时直接返回，避免 wglMakeCurrent 报错。"""
        # 任何一个为空，或已经标记不可用，就不再渲染
        if (not getattr(self, "_ready", False)) or (self._rw is None) or (self._ren is None) or (self._iren is None):
            return
        # 组件不可见也不渲染（常见于关窗阶段）
        if not self.isVisible():
            return
        try:
            self._rw.Render()
        except Exception:
            return
        try:
            self._ren.ResetCameraClippingRange()
        except Exception:
            pass
        try:
            self._iren.Render()
        except Exception:
            pass

    def _style_lines(self, actor, color, width):
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetLineWidth(float(width))
        prop.SetLighting(False)
        try:
            prop.SetRenderLinesAsTubes(True)
        except Exception:
            pass

    def set_point_size(self, s: int):
        self._point_size_default = int(max(1, s))
        if self._points_actor:
            try:
                self._points_actor.GetProperty().SetPointSize(self._point_size_default)
            except Exception:
                pass
            self.force_full_render()

    def set_point_color(self, r: float, g: float, b: float):
        self._point_color = (float(r), float(g), float(b))
        if self._points_actor:
            try:
                self._points_actor.GetProperty().SetColor(*self._point_color)
            except Exception:
                pass
            self.force_full_render()

    def set_edge_style(self, *, width: float | None = None, color: tuple | None = None):
        if width is not None:
            self._edge_width = float(width)
        if color is not None:
            self._edge_color = tuple(color)
        if self._edges_actor:
            self._style_lines(self._edges_actor, self._edge_color, self._edge_width)
            self.force_full_render()

    def set_selected_edge_style(self, *, width: float | None = None, color: tuple | None = None):
        if width is not None:
            self._sel_edge_width = float(width)
        if color is not None:
            self._sel_edge_color = tuple(color)
        if self._sel_actor:
            try:
                prop = self._sel_actor.GetProperty()
                prop.SetLineWidth(self._sel_edge_width)
                prop.SetColor(*self._sel_edge_color)
            except Exception:
                pass
            self.force_full_render()
    # -------------------- 多文件叠加（供“视图（副本）”使用） --------------------
    def _make_lines_actor(self, vertices: np.ndarray, edges: List[tuple[int, int]],
                          color=(1.0, 1.0, 1.0), width=3.0) -> vtk.vtkActor:
        """从 (V,E) 创建一个独立的线框 actor，不修改主场景缓存（供多图层使用）"""
        # vtkPoints
        pts = vtk.vtkPoints()
        pts.SetDataTypeToDouble()
        for p in np.asarray(vertices, dtype=float):
            pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

        # vtkCellArray -> Lines
        carr = vtk.vtkCellArray()
        for (i, j) in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(i))
            line.GetPointIds().SetId(1, int(j))
            carr.InsertNextCell(line)

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(carr)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self._style_lines(actor, color, width)
        actor.PickableOff()
        return actor

    def _ensure_layer_store(self):
        """延迟初始化图层存储：path -> {'actor':vtkActor,'color':(r,g,b),'width':float}"""
        if not hasattr(self, "_layer_actors"):
            self._layer_actors: dict[str, dict] = {}

    def add_obj_wire(self, obj_path: str | Path, *, color=(1.0, 1.0, 1.0), width=3.0) -> bool:
        """
        读取“仅 v/l 的 OBJ”并以独立图层叠加到当前场景。
        注意：这不会改动 self._edges_poly/self._edges_actor（不影响主选择逻辑）。
        """
        if not self._ready:
            return False
        self._ensure_layer_store()
        p = str(Path(obj_path).resolve())
        if p in self._layer_actors:
            # 已存在则只更新样式
            self.update_layer_style(p, color=color, width=width)
            return True
        parsed = self._parse_obj_vertices_edges(Path(p))
        if not parsed:
            return False
        verts, edges = parsed
        if len(edges) == 0:
            return False
        actor = self._make_lines_actor(verts, edges, color=color, width=width)
        self._ren.AddActor(actor)
        self._layer_actors[p] = {"actor": actor, "color": tuple(color), "width": float(width)}
        try:
            self._ren.ResetCameraClippingRange()
            self._rw.Render()
        except Exception:
            pass
        return True

    def remove_layer(self, obj_path: str | Path):
        """移除一个叠加图层"""
        if not self._ready:
            return
        self._ensure_layer_store()
        p = str(Path(obj_path).resolve())
        st = self._layer_actors.get(p)
        if st:
            try:
                self._ren.RemoveActor(st["actor"])
            except Exception:
                pass
            self._layer_actors.pop(p, None)
            try:
                self._rw.Render()
            except Exception:
                pass

    def clear_layers(self):
        """移除所有叠加图层（不影响主白层/红层/点云）"""
        if not self._ready:
            return
        self._ensure_layer_store()
        for p, st in list(self._layer_actors.items()):
            try:
                self._ren.RemoveActor(st["actor"])
            except Exception:
                pass
        self._layer_actors.clear()
        try:
            self._rw.Render()
        except Exception:
            pass

    def update_layer_style(self, obj_path: str | Path, *, color=None, width=None):
        """更新图层样式"""
        self._ensure_layer_store()
        p = str(Path(obj_path).resolve())
        st = self._layer_actors.get(p)
        if not st:
            return
        if color is not None:
            st["color"] = tuple(color)
        if width is not None:
            st["width"] = float(width)
        try:
            self._style_lines(st["actor"], st["color"], st["width"])
            self._rw.Render()
        except Exception:
            pass

    def show_layers(self, layers: list[tuple[str | Path, tuple[float,float,float], float]]):
        """
        批量对齐到 layers 列表：[(path, color, width), ...]
        - 新增：add_obj_wire
        - 已有：update_layer_style
        - 缺失：remove_layer
        """
        if not self._ready:
            return
        self._ensure_layer_store()
        want = {str(Path(p).resolve()): (tuple(c), float(w)) for (p, c, w) in layers}
        # 移除不需要的
        for p in list(self._layer_actors.keys()):
            if p not in want:
                self.remove_layer(p)
        # 新增/更新
        for p, (c, w) in want.items():
            if p in self._layer_actors:
                self.update_layer_style(p, color=c, width=w)
            else:
                self.add_obj_wire(p, color=c, width=w)
        try:
            self._ren.ResetCameraClippingRange()
            self._rw.Render()
        except Exception:
            pass


    # ================== Compare Mode (multi-file overlay, no selection) ==================
    def begin_compare_mode(self):
        """Enter compare mode: disable selection and prepare an isolated overlay pipeline."""
        try:
            self.set_selection_mode(False)
        except Exception:
            pass
        self._cmp_mode = True

    def end_compare_mode(self):
        """Leave compare mode and remove all overlay actors."""
        try:
            if getattr(self, "_cmp_layers", None):
                for info in list(self._cmp_layers.values()):
                    try:
                        self._ren.RemoveActor(info.get("actor"))
                    except Exception:
                        pass
                self._cmp_layers.clear()
                try:
                    self._rw.Render()
                except Exception:
                    pass
        except Exception:
            pass
        self._cmp_mode = False

    def show_layers(self, layers, *, reset_camera_if_first=True):
        """
        Align overlay to the given list of (path, color(tuple), width(float)).
        - points: width -> point size
        - lines:  width -> line width
        """
        if not self._ready:
            return
        want = {}
        for tup in (layers or []):
            if not tup:
                continue
            try:
                p, col, w = tup
                want[str(p)] = {"color": tuple(col), "width": float(w)}
            except Exception:
                continue

        had_zero_before = (len(self._cmp_layers) == 0)

        # Remove stale
        for path in list(self._cmp_layers.keys()):
            if path not in want:
                try:
                    self._ren.RemoveActor(self._cmp_layers[path]["actor"])
                except Exception:
                    pass
                self._cmp_layers.pop(path, None)

        # Add / update
        for path, st in want.items():
            if path not in self._cmp_layers:
                actor, typ = self._cmp_make_actor_for_file(Path(path))
                if actor is None:
                    continue
                if typ == "points":
                    try:
                        prop = actor.GetProperty()
                        prop.SetColor(*st["color"])
                        prop.SetPointSize(max(1, int(round(st["width"]))))
                        try:
                            prop.SetRenderPointsAsSpheres(True)
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    self._style_lines(actor, st["color"], st["width"])
                try:
                    self._ren.AddActor(actor)
                except Exception:
                    continue
                self._cmp_layers[path] = {"type": typ, "actor": actor, "color": st["color"], "width": st["width"]}
            else:
                info = self._cmp_layers[path]
                actor = info["actor"]
                info["color"] = st["color"]
                info["width"] = st["width"]
                if info["type"] == "points":
                    try:
                        prop = actor.GetProperty()
                        prop.SetColor(*st["color"])
                        prop.SetPointSize(max(1, int(round(st["width"]))))
                    except Exception:
                        pass
                else:
                    self._style_lines(actor, st["color"], st["width"])

        try:
            if had_zero_before and len(self._cmp_layers) > 0 and reset_camera_if_first:
                self._ren.ResetCamera()
            self._rw.Render()
        except Exception:
            pass

    def _cmp_make_actor_for_file(self, path: Path):
        """Return (vtkActor, 'points'|'lines') or (None, None)."""
        p = Path(path)
        if not p.exists():
            return None, None
        ext = p.suffix.lower()
        try:
            if ext in (".xyz", ".txt"):
                import numpy as _np
                arr = _np.loadtxt(str(p), dtype=float, usecols=(0,1,2))
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 3)
                return self._actor_from_points_numpy(arr), "points"
            elif ext == ".ply":
                try:
                    reader = vtkPLYReader()
                    reader.SetFileName(str(p))
                    reader.Update()
                    poly = reader.GetOutput()
                    from vtkmodules.vtkFiltersCore import vtkVertexGlyphFilter
                    vg = vtkVertexGlyphFilter()
                    vg.SetInputData(poly)
                    vg.Update()
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(vg.GetOutput())
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.PickableOff()
                    return actor, "points"
                except Exception:
                    return None, None
            elif ext == ".obj":
                parsed = self._parse_obj_vertices_edges(p)
                if not parsed:
                    return None, None
                verts, edges = parsed
                pts = vtk.vtkPoints()
                pts.SetDataTypeToDouble()
                for v in verts:
                    pts.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
                carr = vtk.vtkCellArray()
                for (i, j) in edges:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, int(i))
                    line.GetPointIds().SetId(1, int(j))
                    carr.InsertNextCell(line)
                poly = vtk.vtkPolyData()
                poly.SetPoints(pts)
                poly.SetLines(carr)
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputData(poly)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.PickableOff()
                return actor, "lines"
            else:
                return None, None
        except Exception:
            return None, None

    def _actor_from_points_numpy(self, pts_np):
        """Create a VTK actor from Nx3 numpy array as point sprites."""
        import numpy as _np
        pts_np = _np.asarray(pts_np, float)
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetDataTypeToDouble()
        vtk_pts.SetNumberOfPoints(len(pts_np))
        for i, (x, y, z) in enumerate(pts_np):
            vtk_pts.SetPoint(i, float(x), float(y), float(z))
        verts = vtk.vtkCellArray()
        for i in range(len(pts_np)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.SetVerts(verts)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        try:
            actor.GetProperty().SetRenderPointsAsSpheres(True)
        except Exception:
            pass
        actor.PickableOff()
        return actor

    def shutdown_vtk(self):
        """在 Qt 释放窗口句柄前，干净释放 VTK/交互/观察者，避免 wglMakeCurrent 报错。"""
        try:
            # 1) 退出选择模式并移除事件过滤
            try:
                self.set_selection_mode(False)
            except Exception:
                pass
            try:
                if getattr(self, "_widget_filter_installed", False):
                    self.vtk.removeEventFilter(self)
                    self._widget_filter_installed = False
            except Exception:
                pass

            # 2) 停用交互器 & 观察者
            try:
                if self._iren is not None:
                    try:
                        self._iren.RemoveAllObservers()
                    except Exception:
                        pass
                    try:
                        self._iren.Disable()
                    except Exception:
                        pass
            except Exception:
                pass

            # 3) 删掉橡皮筋、图层与演员（不再调用 Render）
            try:
                if self._rubber:
                    self._rubber.hide()
                self._rubber = None
                self._rubber_origin = None
            except Exception:
                pass
            try:
                # 移除叠加图层
                if hasattr(self, "_layer_actors"):
                    for st in list(self._layer_actors.values()):
                        try:
                            self._ren.RemoveActor(st["actor"])
                        except Exception:
                            pass
                    self._layer_actors.clear()
            except Exception:
                pass
            try:
                # 移除场景所有 actor
                if self._ren is not None:
                    actors = self._ren.GetActors()
                    actors.InitTraversal()
                    to_remove = []
                    for _ in range(actors.GetNumberOfItems()):
                        a = actors.GetNextActor()
                        if a:
                            to_remove.append(a)
                    for a in to_remove:
                        try:
                            self._ren.RemoveActor(a)
                        except Exception:
                            pass
            except Exception:
                pass

            # 4) Finalize 渲染窗口，释放 GL 资源
            try:
                if self._rw is not None:
                    self._rw.Finalize()
            except Exception:
                pass

        finally:
            # 5) 断开强引用，标记不可用，后续任何刷新都直接 return
            self._ready = False
            self._sel_mode = False
            self._ren = None
            self._rw = None
            self._iren = None


