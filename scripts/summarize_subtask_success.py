#!/usr/bin/env python3
"""统计 metrics JSON 中 subtask_success 字段的成功率。

用法:
  python summarize_subtask_success.py /path/to/turning_on_radio_ep60_st3.json
  python summarize_subtask_success.py /path/to/metrics_dir          # 递归扫描目录下所有 .json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collect_json_paths(target: Path) -> list[Path]:
    target = target.expanduser().resolve()
    if target.is_file():
        if target.suffix.lower() != ".json":
            raise ValueError(f"不是 .json 文件: {target}")
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.json"))
    raise ValueError(f"路径不存在: {target}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "path",
        type=Path,
        help="单个 metrics .json 文件，或包含多个 .json 的目录（目录会递归扫描）",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出统计结果",
    )
    args = p.parse_args()

    try:
        files = collect_json_paths(args.path)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    if not files:
        print(f"未找到任何 .json: {args.path.expanduser().resolve()}")
        return 0

    n_true = 0
    n_false = 0
    errors: list[str] = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{path}: 无法解析 JSON — {e}")
            continue

        if "subtask_success" not in data:
            errors.append(f"{path}: 缺少字段 subtask_success")
            continue

        v = data["subtask_success"]
        if not isinstance(v, bool):
            errors.append(f"{path}: subtask_success 不是 bool: {type(v).__name__} = {v!r}")
            continue

        if v:
            n_true += 1
        else:
            n_false += 1

    total = n_true + n_false
    rate_pct = (100.0 * n_true / total) if total else 0.0

    if errors:
        print("警告 (已跳过):", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)

    root = args.path.expanduser().resolve()
    if args.json:
        out = {
            "path": str(root),
            "is_directory": root.is_dir(),
            "files_scanned": len(files),
            "files_valid": total,
            "subtask_success_true": n_true,
            "subtask_success_false": n_false,
            "total": total,
            "success_rate_pct": round(rate_pct, 4),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    kind = "目录 (递归)" if root.is_dir() else "单文件"
    print(f"路径: {root}  |  {kind}")
    print(f"扫描 .json 数量: {len(files)}  |  有效记录 (含 subtask_success): {total}")
    if total:
        print(
            f"subtask_success: True={n_true}, False={n_false}, "
            f"成功率 {rate_pct:.2f}% ({n_true}/{total})"
        )
    else:
        print("无有效 subtask_success 记录，无法计算成功率。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
