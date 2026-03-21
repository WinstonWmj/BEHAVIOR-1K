#!/usr/bin/env python3
"""按 manipulating_object_id 汇总 metrics JSON 中 success 的 true/false 数量。

用法:
  python summarize_metrics_success_by_object.py /path/to/metrics
  python summarize_metrics_success_by_object.py /path/to/logs --recursive
  python summarize_metrics_success_by_object.py . --group combined   # 整份 JSON 只算一类（多 id 用排序后逗号拼接）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def iter_json_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("*.json"))
    return sorted(root.glob("*.json"))


def normalize_object_ids(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x) != ""]
    return [str(raw)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "folder",
        type=Path,
        help="包含 metrics *.json 的目录",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归扫描子目录中的 .json",
    )
    p.add_argument(
        "--group",
        choices=("per_id", "combined"),
        default="per_id",
        help="per_id: 列表中每个 id 各计一次; combined: 整条记录只归到「排序拼接」的一类 (默认 per_id)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 而非表格",
    )
    args = p.parse_args()
    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"错误: 不是目录: {root}", file=sys.stderr)
        return 1

    files = iter_json_files(root, args.recursive)
    if not files:
        print(f"未找到 .json: {root}" + (" (递归)" if args.recursive else ""))
        return 0

    # object_id -> {"true": n, "false": n}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"true": 0, "false": 0})
    errors: list[str] = []
    # 每个 JSON 文件只计一次（与按 object 分行统计互补）
    files_true = 0
    files_false = 0
    files_parsed = 0

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{path}: {e}")
            continue

        if "success" not in data:
            errors.append(f"{path}: 缺少字段 success")
            continue
        success = data["success"]
        if not isinstance(success, bool):
            errors.append(f"{path}: success 不是 bool: {type(success).__name__}")
            continue

        files_parsed += 1
        if success:
            files_true += 1
        else:
            files_false += 1

        key_bucket = "true" if success else "false"
        ids = normalize_object_ids(data.get("manipulating_object_id"))

        if args.group == "combined":
            label = ",".join(sorted(ids)) if ids else "<empty>"
            counts[label][key_bucket] += 1
        else:
            if not ids:
                counts["<empty>"][key_bucket] += 1
            else:
                for oid in ids:
                    counts[oid][key_bucket] += 1

    if errors:
        print("警告 (已跳过):", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)

    rows = []
    for oid in sorted(counts.keys()):
        c = counts[oid]
        t, f = c["true"], c["false"]
        tot = t + f
        rate = (100.0 * t / tot) if tot else 0.0
        rows.append(
            {
                "manipulating_object_id": oid,
                "success_true": t,
                "success_false": f,
                "total": tot,
                "success_rate_pct": round(rate, 2),
            }
        )

    # 按 object 汇总行的合计（per_id 且一条记录多个 id 时，会大于文件数）
    sum_true = sum(r["success_true"] for r in rows)
    sum_false = sum(r["success_false"] for r in rows)
    sum_tot = sum_true + sum_false
    sum_rate = (100.0 * sum_true / sum_tot) if sum_tot else 0.0

    file_tot = files_true + files_false
    file_rate = (100.0 * files_true / file_tot) if file_tot else 0.0

    if args.json:
        out = {
            "folder": str(root),
            "recursive": args.recursive,
            "group_mode": args.group,
            "files_scanned": len(files),
            "files_parsed_ok": files_parsed,
            "by_object": rows,
            "sum_by_object_rows": {
                "success_true": sum_true,
                "success_false": sum_false,
                "total": sum_tot,
                "success_rate_pct": round(sum_rate, 2),
            },
            "per_file": {
                "success_true": files_true,
                "success_false": files_false,
                "total": file_tot,
                "success_rate_pct": round(file_rate, 2),
            },
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"目录: {root}  |  扫描 {len(files)} 个 .json (有效 {files_parsed})  |  group={args.group}\n")
    print(
        "按文件 (每个 JSON 计一次):  "
        f"success=true {files_true}, false {files_false}, 合计 {file_tot}, 成功率 {file_rate:.2f}%\n"
    )
    w_oid = max(len("manipulating_object_id"), max((len(r["manipulating_object_id"]) for r in rows), default=0))
    head = f"{'manipulating_object_id':<{w_oid}}  {'true':>6}  {'false':>6}  {'total':>6}  {'rate%':>8}"
    print("按 manipulating_object_id:")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['manipulating_object_id']:<{w_oid}}  {r['success_true']:>6}  "
            f"{r['success_false']:>6}  {r['total']:>6}  {r['success_rate_pct']:>7.2f}%"
        )
    print("-" * len(head))
    print(f"{'SUM(各行)':<{w_oid}}  {sum_true:>6}  {sum_false:>6}  {sum_tot:>6}  {sum_rate:>7.2f}%")
    if args.group == "per_id" and sum_tot != file_tot:
        print(
            "\n说明: SUM(各行) 与「按文件」合计不同，通常是因为同一 JSON 里 manipulating_object_id 有多个 id，"
            "per_id 模式下每个 id 各记一次。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
