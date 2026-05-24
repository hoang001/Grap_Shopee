#!/usr/bin/env python3
"""
generate_phase2.py — Sinh config MÔ PHỎNG Phase 2, RANDOM tham số ẩn theo seed.

Mục tiêu: mô phỏng GẦN ĐÚNG config Phase 2 mà KHÔNG hardcode bộ tham số ẩn. Mỗi
"draw" (mỗi seed) cho một bộ surge/hotspot/map khác nhau, bốc trong range neo theo
chính logic env tự sinh ở Phase 1 (env.py:_resolve_generation_params) rồi nới rộng
để stress test. Nhờ vậy đánh giá đo độ ROBUST trên một PHÂN PHỐI kịch bản Phase 2,
không phải một bộ số may rủi.

KHUNG CỐ ĐỊNH (không random): 8 mức quy mô P1..P8 trải tới biên Phase 2
(N≤100, C≤25, G≤1500, T≤2400) — để các draw cùng quy mô, so sánh được.

RANDOM mỗi draw (tham số ẩn / chưa biết trước):
  • map: chọn topology + density + seed ngẫu nhiên (map Phase 2 ta chưa biết).
  • surge_amplitude A  ~ phần lớn U(2,6), thi thoảng cực đoan U(6,10).
  • số cửa sổ surge      ~ {1,2,2,3}; mỗi cửa sổ start/duration random.
  • số hotspot + vị trí  ~ random trên ô trống, số lượng theo C.
  • λ₀: KHÔNG ghi → env tự lấy G/T (đúng baseline; tổng đơn vẫn = G).

Solver KHÔNG đọc các tham số này (chỉ dùng heatmap quan sát) nên random hóa chỉ
đổi MÔI TRƯỜNG, không rò rỉ vào thuật toán.

Dùng:
    python3 generate_phase2.py --out test_config_phase2.txt --seed 0   # 1 draw để xem
    # Monte Carlo nhiều draw: dùng eval_phase2_mc.py (import build_strings).
"""
from __future__ import annotations

import argparse
import random
from typing import List, Tuple

from generate_configs import (
    _fmt_config,
    _free_cells,
    _k_maxes,
    _w_maxes,
    asymmetric_map,
    bottleneck_map,
    divided_h_map,
    divided_v_map,
    grid_obstacles_map,
    maze_map,
    rooms_4_map,
    sparse_map,
    validate_configs,
)

# Khung quy mô cố định: (name, N, C, G, T). Trải từ vừa tới trần Phase 2.
SCALE_SLOTS: List[Tuple[str, int, int, int, int]] = [
    ("P1", 20, 5, 120, 720),
    ("P2", 30, 7, 220, 960),
    ("P3", 40, 9, 350, 1200),
    ("P4", 50, 12, 500, 1440),
    ("P5", 60, 14, 650, 1600),
    ("P6", 75, 18, 900, 1900),
    ("P7", 90, 22, 1200, 2200),
    ("P8", 100, 25, 1500, 2400),
]


def _random_map(rng: random.Random, N: int):
    """Chọn topology + tham số ngẫu nhiên; mọi builder đều đảm bảo liên thông."""
    ms = rng.randint(0, 10**9)
    pick = rng.choice([
        "sparse", "sparse", "sparse",   # phổ biến nhất
        "maze", "asymmetric",
        "divided_h", "divided_v", "rooms", "grid", "bottleneck",
    ])
    if pick == "sparse":
        return sparse_map(N, rng.uniform(0.08, 0.18), ms)
    if pick == "maze":
        return maze_map(N, rng.uniform(0.15, 0.22), ms)
    if pick == "asymmetric":
        return asymmetric_map(N, ms)
    if pick == "divided_h":
        return divided_h_map(N, rng.randint(2, 4))
    if pick == "divided_v":
        return divided_v_map(N, rng.randint(2, 4))
    if pick == "rooms":
        return rooms_4_map(N)
    if pick == "grid":
        return grid_obstacles_map(N)
    return bottleneck_map(N)


def _random_surge(rng: random.Random, T: int) -> Tuple[float, List[Tuple[int, int]]]:
    """Bốc surge_amplitude + các cửa sổ. Range neo env.py rồi nới rộng."""
    # 75% biên độ vừa, 25% cực đoan.
    amp = rng.uniform(2.0, 6.0) if rng.random() < 0.75 else rng.uniform(6.0, 10.0)
    n_win = rng.choice([1, 2, 2, 3])
    lo = int(0.10 * T)
    windows: List[Tuple[int, int]] = []
    for _ in range(n_win):
        dur = rng.randint(max(20, T // 8), max(21, T // 4))
        hi = max(lo + 1, int(0.85 * T) - dur)
        s = rng.randint(lo, hi)
        windows.append((s, min(T - 1, s + dur)))
    return round(amp, 1), sorted(windows)


def _random_hotspots(rng: random.Random, grid, C: int) -> List[Tuple[int, int]]:
    """Số hotspot theo C (neo env: ~C/2, chặn 4), vị trí random trên ô trống."""
    free = _free_cells(grid)
    n = rng.randint(1, min(4, max(1, C // 3)))
    return rng.sample(free, min(n, len(free)))


def build_strings(seed: int) -> List[str]:
    """Sinh 8 config (chuỗi) cho MỘT draw, tham số ẩn random theo seed."""
    rng = random.Random(seed)
    cfgs: List[str] = []
    for name, N, C, G, T in SCALE_SLOTS:
        grid = _random_map(rng, N)
        amp, windows = _random_surge(rng, T)
        hotspots = _random_hotspots(rng, grid, C)
        cfgs.append(_fmt_config(
            name, N=N, C=C, G=G, T=T,
            k_maxes=_k_maxes(C), w_maxes=_w_maxes(C), grid=grid,
            surge_amplitude=amp, surge_windows=windows, hotspots=hotspots,
        ))
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_config_phase2.txt")
    ap.add_argument("--seed", type=int, default=0, help="Seed draw tham số ẩn")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    cfgs = build_strings(args.seed)
    header = (
        f"# test_config_phase2.txt — MÔ PHỎNG Phase 2, draw seed={args.seed}.\n"
        "# Tham số ẩn (surge/hotspot/map) RANDOM theo seed, KHÔNG hardcode.\n"
        "# Chạy Monte Carlo nhiều draw bằng eval_phase2_mc.py để đo robust.\n"
        "[SEED]\nbase_seed = 42\n\n"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(cfgs))
    print(f"Đã ghi {len(cfgs)} configs (draw seed={args.seed}) ra {args.out}")
    if not args.no_validate:
        validate_configs(cfgs)


if __name__ == "__main__":
    main()
