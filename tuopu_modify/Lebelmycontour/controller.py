#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, List, Optional
import sys

# 识别为“源数据”的扩展名（用于回退枚举/兜底匹配）
POINT_EXTS: Set[str] = {".xyz", ".ply", ".pcd", ".pts", ".txt", ".npz", ".npy", ".obj"}


class Controller:
    """数据集控制器（ID 驱动版）"""

    def __init__(self) -> None:
        self.dataset_root: Optional[Path] = None
        self.out_dir: Optional[Path] = None

        # 运行态（加载后可用）
        self.ids_all: List[str] = []
        self.done_ids: Set[str] = set()
        self.todo_ids: List[str] = []
        self.idx: int = 0

    # ----------------------------------------------------------------------
    # 入口：统计 + 初始化运行态（ID 驱动：all.txt + deal_res/out.txt）
    # ----------------------------------------------------------------------
    def analyze_dataset(self, root: Path) -> Dict[str, object]:
        """
        统计数据集规模并初始化运行态（优先基于 all.txt；无则回退到文件枚举推断 ID）。
        返回给 UI：{total, processed, pending, out_dir}
        """
        root = Path(root)
        self.dataset_root = root

        out_dir = root / "deal_res"
        (out_dir / "obj").mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        # 读取全量 ID（优先 all.txt；无则从 xyz/ 或可识别后缀推断）
        self.ids_all = self._read_ids(root)

        # 读取已完成 ID（out.txt + deal_res/obj/*.obj 的 stem）
        self.done_ids = self._read_processed_ids(out_dir)

        # 计算待处理
        self.todo_ids = [i for i in self.ids_all if i not in self.done_ids]
        self.idx = 0

        return {
            "total": len(self.ids_all),
            "processed": len(self.done_ids),
            "pending": len(self.todo_ids),
            "out_dir": out_dir,
        }

    # ----------------------------------------------------------------------
    # 新约定：返回“待处理 ID 列表”（以前 pending_sources 返回 Path）
    # ----------------------------------------------------------------------
    def pending_sources(self, limit: Optional[int] = None) -> List[str]:
        """
        返回待处理 ID 列表（按 all.txt 顺序减去已完成）。
        ⚠️ 破坏性改动：以前返回的是 Path，现在返回 ID（str）。
        """
        if limit is None:
            return list(self.todo_ids)
        return list(self.todo_ids[:max(0, int(limit))])

    # （可选）当前/前后导航基于 ID
    def current_id(self) -> Optional[str]:
        if 0 <= self.idx < len(self.todo_ids):
            return self.todo_ids[self.idx]
        return None

    def next_id(self) -> Optional[str]:
        if self.idx + 1 < len(self.todo_ids):
            self.idx += 1
            return self.todo_ids[self.idx]
        return None

    def prev_id(self) -> Optional[str]:
        if self.idx - 1 >= 0:
            self.idx -= 1
            return self.todo_ids[self.idx]
        return None

    # ----------------------------------------------------------------------
    # ID → 路径映射（按惯例目录 xyz/ 与 obj/；带兜底模糊匹配）
    # ----------------------------------------------------------------------
    def id_to_paths(self, root: Optional[Path], id_str: str) -> Dict[str, Optional[Path]]:
        """
        把 id 映射到 { "xyz": Path|None, "obj": Path|None }
        - 优先匹配 xyz/<id>.xyz 与 obj/<id>.obj
        - 若精确文件不存在，则在各目录下做 <id>.* 的兜底匹配（取第一个）
        """
        if root is None:
            root = self.dataset_root
        if root is None:
            return {"xyz": None, "obj": None}

        root = Path(root)
        xyz_dir, obj_dir = root / "xyz", root / "obj"

        xyz_path = xyz_dir / f"{id_str}.xyz"
        obj_path = obj_dir / f"{id_str}.obj"

        if not xyz_path.exists() and xyz_dir.exists():
            cands = list(xyz_dir.glob(f"{id_str}.*"))
            if cands:
                xyz_path = cands[0]
        if not obj_path.exists() and obj_dir.exists():
            cands = list(obj_dir.glob(f"{id_str}.*"))
            if cands:
                obj_path = cands[0]

        return {
            "xyz": xyz_path if xyz_path.exists() else None,
            "obj": obj_path if obj_path.exists() else None,
        }

    def resolve_xyz_path(self, id_str: str) -> Optional[Path]:
        return self.id_to_paths(self.dataset_root, id_str)["xyz"]

    def resolve_obj_path(self, id_str: str) -> Optional[Path]:
        return self.id_to_paths(self.dataset_root, id_str)["obj"]

    # ----------------------------------------------------------------------
    # 回退/工具
    # ----------------------------------------------------------------------
    def list_sources(self) -> List[Path]:
        """
        仅用于回退/调试：枚举数据根目录下“可识别后缀”的文件。
        （新流程推荐通过 ID 去解析路径）
        """
        if not self.dataset_root:
            return []
        return sorted(
            p for p in self.dataset_root.rglob("*")
            if p.is_file() and p.suffix.lower() in POINT_EXTS
        )

    # 读取“全量 ID”
    def _read_ids(self, root: Path) -> List[str]:
        """
        读取 ID 的优先级：
        1) all.txt 的非空首列
        2) xyz/*.xyz 的 stem
        3) 根目录下所有可识别后缀文件的 stem（去重保序）
        """
        root = Path(root)
        ids: List[str] = []

        all_txt = root / "all.txt"
        if all_txt.exists():
            try:
                lines = all_txt.read_text("utf-8", errors="ignore").splitlines()
            except Exception as e:
                print(f"[Controller] read all.txt failed: {e}", file=sys.stderr)
                lines = []
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                ids.append(s.split()[0])
        else:
            xyz_dir = root / "xyz"
            if xyz_dir.exists():
                ids = [p.stem for p in sorted(xyz_dir.glob("*.xyz"))]
            else:
                # 最后兜底：扫描所有可识别后缀
                stems: List[str] = []
                for p in sorted(root.rglob("*")):
                    if p.is_file() and p.suffix.lower() in POINT_EXTS:
                        stems.append(p.stem)
                ids = stems

        # 去重但保持顺序
        seen: Set[str] = set()
        uniq_ids = [x for x in ids if not (x in seen or seen.add(x))]
        return uniq_ids

    # 读取“已完成 ID”
    def _read_processed_ids(self, out_dir: Path) -> Set[str]:
        """
        以 deal_res/out.txt + deal_res/obj/*.obj 的 stem 作为已完成集合
        （两者并集；行按首列取 id）
        """
        processed: Set[str] = set()

        out_txt = out_dir / "out.txt"
        if out_txt.exists():
            try:
                for ln in out_txt.read_text("utf-8", errors="ignore").splitlines():
                    t = ln.strip()
                    if not t:
                        continue
                    processed.add(t.split()[0])
            except Exception as e:
                print(f"[Controller] read out.txt failed: {e}", file=sys.stderr)

        od = out_dir / "obj"
        if od.exists():
            for p in od.glob("*.obj"):
                processed.add(p.stem)

        return processed
