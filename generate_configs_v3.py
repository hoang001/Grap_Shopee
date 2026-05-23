#!/usr/bin/env python3
"""
generate_configs_v3.py — Bộ test động gọn (N<=30) để đo adaptivity.

Tái sử dụng map-generator của generate_configs.py. Bộ này nhỏ hơn v2 để chạy
nhanh qua nhiều seed, nhưng vẫn phủ: nhiều loại map, nhiều mức surge, nhiều
vị trí/thời điểm hotspot, và dải N rộng tới 30.

    python3 generate_configs_v3.py --out test_config_v3.txt
"""
import argparse

from generate_configs import (
    _find_free,
    bottleneck_map,
    divided_h_map,
    grid_obstacles_map,
    maze_map,
    open_map,
    rooms_4_map,
    ring_map,
    sparse_map,
    _fmt_config,
    validate_configs,
)


def build() -> list:
    cfgs = []

    # --- nhỏ, sanity ---
    cfgs.append(_fmt_config("V1", N=7, C=2, G=12, T=120,
                            k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=open_map(7)))
    cfgs.append(_fmt_config("V2", N=10, C=2, G=25, T=240,
                            k_maxes=[3, 3], w_maxes=[20.0, 20.0],
                            grid=sparse_map(10, 0.12, 210)))

    # --- N=12: surge/hotspot variety (spatial + intensity) ---
    m12 = sparse_map(12, 0.10, 212)
    hs_tl = _find_free(m12, 2, 2)
    hs_br = _find_free(m12, 9, 9)
    hs_ct = _find_free(m12, 5, 5)
    cfgs.append(_fmt_config("V3", N=12, C=3, G=40, T=360,
                            k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=m12,
                            surge_amplitude=3.0, surge_windows=[(60, 130)], hotspots=[hs_tl]))
    cfgs.append(_fmt_config("V4", N=12, C=3, G=40, T=360,
                            k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=m12,
                            surge_amplitude=5.0, surge_windows=[(180, 260)], hotspots=[hs_br]))
    cfgs.append(_fmt_config("V5", N=12, C=3, G=40, T=360,
                            k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=m12,
                            surge_amplitude=8.0, surge_windows=[(80, 140), (240, 300)],
                            hotspots=[hs_ct, hs_tl]))
    cfgs.append(_fmt_config("V6", N=12, C=3, G=40, T=360,
                            k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0],
                            grid=divided_h_map(12, 2),
                            surge_amplitude=3.0, surge_windows=[(90, 170)],
                            hotspots=[_find_free(divided_h_map(12, 2), 3, 6)]))

    # --- N=14-15: nhiều map type, surge timing khác nhau ---
    cfgs.append(_fmt_config("V7", N=14, C=4, G=55, T=480,
                            k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
                            grid=open_map(14),
                            surge_amplitude=4.0, surge_windows=[(120, 200)],
                            hotspots=[_find_free(open_map(14), 7, 7)]))
    m15 = sparse_map(15, 0.12, 215)
    cfgs.append(_fmt_config("V8", N=15, C=4, G=60, T=480,
                            k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0], grid=m15,
                            surge_amplitude=3.0, surge_windows=[(336, 432)],  # late surge
                            hotspots=[_find_free(m15, 12, 12)]))
    cfgs.append(_fmt_config("V9", N=15, C=4, G=60, T=480,
                            k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
                            grid=maze_map(15, 0.20, 216)))
    cfgs.append(_fmt_config("V10", N=15, C=4, G=60, T=480,
                            k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
                            grid=rooms_4_map(15)))

    # --- N=16-20: large-ish, mixed structures ---
    cfgs.append(_fmt_config("V11", N=16, C=4, G=65, T=500,
                            k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
                            grid=sparse_map(16, 0.10, 240)))
    cfgs.append(_fmt_config("V12", N=18, C=5, G=80, T=600,
                            k_maxes=[3, 3, 2, 2, 3], w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0],
                            grid=sparse_map(18, 0.10, 243),
                            surge_amplitude=4.0, surge_windows=[(150, 260)],
                            hotspots=[_find_free(sparse_map(18, 0.10, 243), 9, 9)]))
    cfgs.append(_fmt_config("V13", N=18, C=5, G=80, T=600,
                            k_maxes=[3, 3, 2, 2, 3], w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0],
                            grid=bottleneck_map(18)))
    cfgs.append(_fmt_config("V14", N=20, C=5, G=100, T=720,
                            k_maxes=[3, 3, 2, 2, 3], w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0],
                            grid=sparse_map(20, 0.10, 247)))
    cfgs.append(_fmt_config("V15", N=20, C=5, G=100, T=720,
                            k_maxes=[3, 3, 2, 2, 3], w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0],
                            grid=ring_map(20)))
    cfgs.append(_fmt_config("V16", N=20, C=5, G=100, T=720,
                            k_maxes=[3, 3, 2, 2, 3], w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0],
                            grid=grid_obstacles_map(20),
                            surge_amplitude=5.0, surge_windows=[(200, 320), (500, 600)],
                            hotspots=[_find_free(grid_obstacles_map(20), 4, 4),
                                      _find_free(grid_obstacles_map(20), 15, 15)]))

    # --- N=22-30: scalability ---
    cfgs.append(_fmt_config("V17", N=22, C=5, G=110, T=770,
                            k_maxes=[3, 3, 2, 2, 3], w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0],
                            grid=sparse_map(22, 0.10, 260)))
    cfgs.append(_fmt_config("V18", N=25, C=6, G=140, T=875,
                            k_maxes=[3, 3, 2, 2, 3, 3], w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0, 20.0],
                            grid=sparse_map(25, 0.10, 261),
                            surge_amplitude=4.0, surge_windows=[(300, 500)],
                            hotspots=[_find_free(sparse_map(25, 0.10, 261), 12, 12)]))
    cfgs.append(_fmt_config("V19", N=28, C=7, G=165, T=980,
                            k_maxes=[3, 3, 2, 2, 3, 3, 2],
                            w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0, 20.0, 30.0],
                            grid=divided_h_map(28, 3)))
    cfgs.append(_fmt_config("V20", N=30, C=7, G=185, T=1050,
                            k_maxes=[3, 3, 2, 2, 3, 3, 2],
                            w_maxes=[20.0, 20.0, 30.0, 30.0, 20.0, 20.0, 30.0],
                            grid=sparse_map(30, 0.10, 262)))

    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_config_v3.txt")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    cfgs = build()
    header = (
        "# test_config_v3.txt — bộ động gọn (N<=30, 20 configs)\n"
        "# Dùng với eval_dynamic.py qua nhiều base_seed để đo adaptivity.\n"
        "[SEED]\nbase_seed = 42\n\n"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(cfgs))
    print(f"Đã ghi {len(cfgs)} configs ra {args.out}")
    if not args.no_validate:
        validate_configs(cfgs)


if __name__ == "__main__":
    main()
