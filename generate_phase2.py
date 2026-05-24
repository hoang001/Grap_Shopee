#!/usr/bin/env python3
"""
generate_phase2.py — Mô phỏng bộ config Phase 2 để stress-test solver.

Phase 2 (theo problem.md + env.py) công bố ĐẦY ĐỦ 4 knob sinh đơn ẩn ở Phase 1:
    lambda0, surge_amplitude, surge_windows, hotspots
và nới biên: N <= 100, C <= 25, G <= 1500, T <= 2400, tối đa 8 config.

Bộ này KHÔNG nhằm tối ưu điểm — nó cố tình phủ rộng để đo độ THÍCH NGHI:
  • N ramp 20 → 100 (kiểm tra scale của BFS-from-static + planner).
  • Nhiều topology (open / sparse / maze / rooms / divided / bottleneck / grid).
  • Surge từ nhẹ tới cực đoan (A = 3 → 12), 1–3 cửa sổ, 1–3 hotspot trải vị trí.
  • Cửa sổ surge đặt sớm / giữa / muộn để thử phản ứng ở các pha khác nhau.

LƯU Ý cơ chế env (env.py:_resolve_generation_params): chỉ khi config có ĐỒNG
THỜI surge_windows VÀ hotspots thì env mới dùng chúng; nếu thiếu một trong hai,
env tự sinh ngẫu nhiên. Generator này luôn phát cả hai để mô phỏng đúng Phase 2.
lambda0 mặc định = G/T (đúng như env tự suy ra) nên không cần ghi đè.

    python3 generate_phase2.py --out test_config_phase2.txt
    python3 eval_dynamic.py --config test_config_phase2.txt --nmax 100 \
            --timeout 300 --seeds 42 7 123
"""
from __future__ import annotations

import argparse

from generate_configs import (
    _find_free,
    _fmt_config,
    _k_maxes,
    _w_maxes,
    asymmetric_map,
    bottleneck_map,
    divided_h_map,
    grid_obstacles_map,
    maze_map,
    open_map,
    rooms_4_map,
    sparse_map,
    validate_configs,
)


def _hotspots(grid, coords):
    """Ánh xạ danh sách (r,c) mong muốn về ô trống gần nhất, loại trùng."""
    out = []
    for r, c in coords:
        cell = _find_free(grid, r, c)
        if cell not in out:
            out.append(cell)
    return out


def build() -> list:
    """8 config Phase 2 mô phỏng, N tăng dần tới 100."""
    cfgs = []

    # P1 — N=20, surge nhẹ giữa episode, 1 hotspot trung tâm. Baseline scale.
    g = sparse_map(20, 0.10, 901)
    cfgs.append(_fmt_config(
        "P1", N=20, C=5, G=120, T=720,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=g,
        surge_amplitude=3.0, surge_windows=[(200, 320)],
        hotspots=_hotspots(g, [(10, 10)]),
    ))

    # P2 — N=30, 2 cửa sổ surge (sớm + muộn), 2 hotspot đối góc → cầu nhảy vùng.
    g = sparse_map(30, 0.10, 902)
    cfgs.append(_fmt_config(
        "P2", N=30, C=7, G=220, T=960,
        k_maxes=_k_maxes(7), w_maxes=_w_maxes(7), grid=g,
        surge_amplitude=4.0, surge_windows=[(120, 260), (640, 800)],
        hotspots=_hotspots(g, [(6, 6), (24, 24)]),
    ))

    # P3 — N=40, map chia đôi (cổng hẹp) + surge mạnh dồn về 1 nửa → nút cổ chai.
    g = divided_h_map(40, n_gaps=3)
    cfgs.append(_fmt_config(
        "P3", N=40, C=9, G=350, T=1200,
        k_maxes=_k_maxes(9), w_maxes=_w_maxes(9), grid=g,
        surge_amplitude=6.0, surge_windows=[(300, 520)],
        hotspots=_hotspots(g, [(8, 20), (8, 8)]),
    ))

    # P4 — N=50, maze (nhiều vật cản) → stress heuristic A* + dist-cache.
    g = maze_map(50, 0.20, 904)
    cfgs.append(_fmt_config(
        "P4", N=50, C=12, G=500, T=1440,
        k_maxes=_k_maxes(12), w_maxes=_w_maxes(12), grid=g,
        surge_amplitude=5.0, surge_windows=[(400, 640)],
        hotspots=_hotspots(g, [(12, 12), (38, 38), (12, 38)]),
    ))

    # P5 — N=60, 4 phòng nối qua tâm + surge cực đoan ở tâm (điểm tranh chấp nhất).
    g = rooms_4_map(60)
    cfgs.append(_fmt_config(
        "P5", N=60, C=14, G=650, T=1600,
        k_maxes=_k_maxes(14), w_maxes=_w_maxes(14), grid=g,
        surge_amplitude=8.0, surge_windows=[(500, 760)],
        hotspots=_hotspots(g, [(30, 30)]),
    ))

    # P6 — N=75, bottleneck 1-ô + 2 surge → kiểm tra chống deadlock hành lang.
    g = bottleneck_map(75)
    cfgs.append(_fmt_config(
        "P6", N=75, C=18, G=900, T=1900,
        k_maxes=_k_maxes(18), w_maxes=_w_maxes(18), grid=g,
        surge_amplitude=6.0, surge_windows=[(350, 560), (1100, 1350)],
        hotspots=_hotspots(g, [(20, 37), (55, 37)]),
    ))

    # P7 — N=90, grid vật cản đều + 3 hotspot trải, surge dài → cầu phân tán động.
    g = grid_obstacles_map(90)
    cfgs.append(_fmt_config(
        "P7", N=90, C=22, G=1200, T=2200,
        k_maxes=_k_maxes(22), w_maxes=_w_maxes(22), grid=g,
        surge_amplitude=5.0, surge_windows=[(600, 1000), (1500, 1800)],
        hotspots=_hotspots(g, [(15, 15), (45, 75), (75, 30)]),
    ))

    # P8 — N=100, C=25, G=1500, T=2400: TRẦN Phase 2. Map bất đối xứng + surge
    # cực đoan muộn → kiểm tra scale tối đa và phản ứng surge cuối episode.
    g = asymmetric_map(100, 905)
    cfgs.append(_fmt_config(
        "P8", N=100, C=25, G=1500, T=2400,
        k_maxes=_k_maxes(25), w_maxes=_w_maxes(25), grid=g,
        surge_amplitude=10.0, surge_windows=[(900, 1300), (1900, 2300)],
        hotspots=_hotspots(g, [(20, 20), (80, 80), (20, 80), (80, 20)]),
    ))

    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_config_phase2.txt")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    cfgs = build()
    header = (
        "# test_config_phase2.txt — MÔ PHỎNG Phase 2 (KHÔNG phải config chính thức).\n"
        "# N<=100, C<=25, G<=1500, T<=2400, 8 configs. surge/hotspot công bố đầy đủ.\n"
        "# Dùng để đo adaptivity của solver, không để tinh chỉnh tham số.\n"
        "[SEED]\nbase_seed = 42\n\n"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(cfgs))
    print(f"Đã ghi {len(cfgs)} configs ra {args.out}")
    if not args.no_validate:
        validate_configs(cfgs)


if __name__ == "__main__":
    main()
