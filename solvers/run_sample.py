#!/usr/bin/env python3
"""
run_sample.py - Chọn N config ngẫu nhiên từ test_config_v2.txt rồi chạy run_test.py.

Mục đích: tránh phải chạy toàn bộ 85 config mỗi lần test thuật toán.

Ví dụ:
    python3 run_sample.py --method GreedyBFS
    python3 run_sample.py --method GreedyBFS --n 10
    python3 run_sample.py --method all --n 6 --sample-seed 123
    python3 run_sample.py --method GreedyBFS --tiers S A B   # chỉ lấy từ tầng S, A, B
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_blocks(filepath: str) -> list[tuple[str, str]]:
    """
    Đọc file config và trả về list (name, raw_text_block).
    raw_text_block bao gồm từ dòng [CONFIG] đến [END] (kèm newline cuối).
    """
    blocks: list[tuple[str, str]] = []
    current_lines: list[str] = []
    current_name: str = ""
    in_block = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.split("#", 1)[0].strip()
            if stripped == "[CONFIG]":
                in_block = True
                current_lines = [line]
                current_name = ""
            elif in_block and stripped.startswith("name") and "=" in stripped:
                current_name = stripped.split("=", 1)[1].strip()
                current_lines.append(line)
            elif stripped == "[END]" and in_block:
                current_lines.append(line)
                blocks.append((current_name, "".join(current_lines)))
                current_lines = []
                in_block = False
            elif in_block:
                current_lines.append(line)

    return blocks


def tier_of(name: str) -> str:
    """Trả về tên tầng dựa vào prefix tên config (S/A/B/C/D/E/F)."""
    if not name:
        return "?"
    return name[0].upper()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy run_test.py với N config ngẫu nhiên từ test_config_v2.txt"
    )
    parser.add_argument(
        "--config", default=os.path.join(SCRIPT_DIR, "test_config_v2.txt"),
        help="File config nguồn (mặc định: test_config_v2.txt)",
    )
    parser.add_argument(
        "--out", default=os.path.join(SCRIPT_DIR, "results_v2"),
        help="Thư mục kết quả (mặc định: results_v2/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed cho run_test.py (không ảnh hưởng đến sampling)",
    )
    parser.add_argument(
        "--method", default="GreedyBFS",
        help="Phương pháp chạy: GreedyBFS / VRPOrToolsSolver / ACOSolver / MAPDCBSSolver / all",
    )
    parser.add_argument(
        "--n", type=int, default=6,
        help="Số config ngẫu nhiên cần chọn (mặc định: 6)",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=None,
        help="Seed cho random sampling (mặc định: random mỗi lần)",
    )
    parser.add_argument(
        "--tiers", nargs="*", default=None,
        help="Giới hạn lấy config từ các tầng cụ thể, VD: --tiers S A B. "
             "Mặc định: lấy từ tất cả tầng.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Chỉ in danh sách config được chọn, không chạy.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[ERROR] Không tìm thấy file config: {args.config}")

    # Đọc tất cả blocks
    all_blocks = extract_blocks(args.config)
    if not all_blocks:
        sys.exit(f"[ERROR] Không đọc được config nào từ {args.config}")

    # Lọc theo tầng nếu có
    if args.tiers:
        tiers_upper = {t.upper() for t in args.tiers}
        pool = [(name, block) for name, block in all_blocks if tier_of(name) in tiers_upper]
        if not pool:
            available = sorted({tier_of(n) for n, _ in all_blocks})
            sys.exit(
                f"[ERROR] Không có config nào thuộc tầng {args.tiers}. "
                f"Tầng có sẵn: {available}"
            )
    else:
        pool = all_blocks

    # Sample
    n = min(args.n, len(pool))
    rng = random.Random(args.sample_seed)
    sampled = rng.sample(pool, n)

    tier_summary = {}
    for name, _ in sampled:
        t = tier_of(name)
        tier_summary[t] = tier_summary.get(t, 0) + 1
    tier_str = "  ".join(f"{t}×{cnt}" for t, cnt in sorted(tier_summary.items()))

    print(f"Pool: {len(pool)} configs  →  chọn {n}: {', '.join(name for name, _ in sampled)}")
    print(f"Phân bố tầng: {tier_str}")

    if args.list:
        return

    # Ghi temp config file
    content = "".join(block for _, block in sampled)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="mapd_sample_", text=True)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)

        run_test = os.path.join(SCRIPT_DIR, "run_test.py")
        cmd = [
            sys.executable, run_test,
            "--config", tmp_path,
            "--out", args.out,
            "--seed", str(args.seed),
            "--method", args.method,
        ]
        print(f"Chạy: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=False)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
