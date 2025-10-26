#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LegacyBridge：尝试自动对接你们之前菜单栏里的功能。
策略：
1) 在工程目录扫描可导入模块；
2) 在模块里查找以下函数名（或同义名），找到就接上；
3) 调用时按函数签名把可用参数注入（只传它需要的）。

支持的目标函数（越多越好）：
- load_data / open_project
- save_results / save_project / save
- export_results / export_annotations / export
- run_pipeline / process / predict / segment / refine_labels
- prev_item / prev
- next_item / next
- undo
- redo
- open_settings / settings
"""

from __future__ import annotations
import sys, pkgutil, importlib, inspect, types, os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any

EXCLUDES = {
    "app_gui", "main", "controller", "viewer_embed", "legacy_bridge",
    "hotkeys", "settings_ui"
}

NAME_ALIASES = {
    "save": "save_results",
    "save_project": "save_results",
    "export": "export_results",
    "export_annotations": "export_results",
    "process": "run_pipeline",
    "predict": "run_pipeline",
    "segment": "run_pipeline",
    "refine_labels": "run_pipeline",
    "prev": "prev_item",
    "next": "next_item",
    "open_project": "load_data",
    "settings": "open_settings",
}

TARGETS = [
    "load_data",
    "save_results",
    "export_results",
    "run_pipeline",
    "prev_item",
    "next_item",
    "undo",
    "redo",
    "open_settings",
]

@dataclass
class AppContext:
    dataset_root: Optional[Path] = None
    out_dir: Optional[Path] = None
    window: Optional[object] = None
    controller: Optional[object] = None

class LegacyBridge:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        self.funcs: Dict[str, Callable[..., Any]] = {}
        self._discover()

    def _discover(self):
        """扫描工程根目录下的包/模块并收集目标函数"""
        for m in pkgutil.iter_modules([str(self.project_root)]):
            name = m.name
            if name in EXCLUDES or name.startswith("_"):
                continue
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue

            # 直接在模块作用域寻找函数
            for attr_name, obj in vars(mod).items():
                if not callable(obj):
                    continue
                key = self._normalize(attr_name)
                if key in TARGETS and key not in self.funcs:
                    self.funcs[key] = obj

            # 同义名映射
            for alias, target in NAME_ALIASES.items():
                if alias in vars(mod) and callable(getattr(mod, alias)) and target not in self.funcs:
                    self.funcs[target] = getattr(mod, alias)

        # 兜底：如果某些还没找到，允许在“legacy_hooks.py”里显式注册
        try:
            hooks = importlib.import_module("legacy_hooks")
            if hasattr(hooks, "register"):
                mapping = hooks.register()
                if isinstance(mapping, dict):
                    for k, v in mapping.items():
                        key = self._normalize(k)
                        if callable(v) and key in TARGETS and key not in self.funcs:
                            self.funcs[key] = v
        except Exception:
            pass

    def _normalize(self, name: str) -> str:
        return NAME_ALIASES.get(name, name)

    # -------- 对外动作（若未接入，返回 False） --------
    def call(self, name: str, ctx: AppContext) -> bool:
        key = self._normalize(name)
        fn = self.funcs.get(key)
        if not fn:
            return False
        try:
            kwargs = self._select_kwargs(fn, ctx)
            fn(**kwargs)
            return True
        except Exception as e:
            print(f"[LegacyBridge] 调用 {key} 失败: {e}")
            return False

    def _select_kwargs(self, fn: Callable, ctx: AppContext) -> dict:
        """根据函数签名挑选可传入的参数"""
        sig = inspect.signature(fn)
        support = {
            "dataset_root": ctx.dataset_root,
            "out_dir": ctx.out_dir,
            "window": ctx.window,
            "controller": ctx.controller,
        }
        use = {}
        for p in sig.parameters.values():
            if p.name in support and support[p.name] is not None:
                use[p.name] = support[p.name]
        return use
