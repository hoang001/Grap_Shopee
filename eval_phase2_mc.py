#!/usr/bin/env python3
"""
eval_phase2_mc.py — Monte Carlo đánh giá solver trên config Phase 2 MÔ PHỎNG.

Ý tưởng: KHÔNG chấm trên một bộ tham số ẩn cố định (dễ overfit/may rủi). Thay vào
đó bốc K "draw" tham số ẩn khác nhau (surge/hotspot/map random theo seed, xem
generate_phase2.py); với mỗi draw lại chạy M order-stream seed khác nhau. Tổng cộng
K×M lần/slot quy mô (P1..P8). Báo cáo trung bình ± lệch chuẩn của net_reward,
%giao, %đúng hạn theo từng slot → đo độ ROBUST trên cả phân phối kịch bản.

  • draw seed  → quyết định surge_amplitude, cửa sổ, hotspot, map  (tham số ẩn).
  • stream seed → quyết định realization luồng đơn trên cùng kịch bản đó.

Solver KHÔNG đọc tham số ẩn nên random hóa chỉ đổi môi trường.

Ví dụ:
    python3 eval_phase2_mc.py --draws 3 --streams 1 --method MAPDCBSSolver --timeout 360
    python3 eval_phase2_mc.py --draws 5 --streams 2 --nmax 60     # bỏ qua N>60 cho nhanh
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER_DIR = os.path.join(SCRIPT_DIR, "solvers")
for p in (SCRIPT_DIR, SOLVER_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from env import DeliveryEnv, load_config  # noqa: E402
from generate_phase2 import build_strings  # noqa: E402


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def _load_solver(method: str):
    sources = {
        "GreedyBFS": "greedy_bfs.py",
        "VRPOrToolsSolver": "vrp_ortools.py",
        "ACOSolver": "aco_solver.py",
        "MAPDCBSSolver": "mapd_cbs_solver.py",
    }
    path = os.path.join(SOLVER_DIR, sources[method])
    spec = importlib.util.spec_from_file_location(method, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, method)


def _configs_for_draw(draw_seed: int) -> List[dict]:
    """Sinh chuỗi config cho draw, ghi file tạm rồi parse về list cfg dict."""
    cfgs_str = build_strings(draw_seed)
    text = "[SEED]\nbase_seed = 42\n\n" + "\n".join(cfgs_str)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        tmp = fh.name
    try:
        return load_config(tmp)
    finally:
        os.unlink(tmp)


def _run_one(solver_cls, cfg: dict, env_seed: int, timeout: int) -> dict:
    env = DeliveryEnv(cfg, seed=env_seed)
    solver = solver_cls(env)
    if timeout > 0:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
    t0 = time.time()
    try:
        res = solver.run()
    except _Timeout:
        return {"status": "TIMEOUT", "delivery_rate": 0.0, "on_time_rate": 0.0,
                "net_reward": 0.0, "wall": time.time() - t0}
    finally:
        if timeout > 0:
            signal.alarm(0)
    res["wall"] = time.time() - t0
    return res


def _fmt_stat(vals: List[float]) -> str:
    if not vals:
        return "    n/a"
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:7.1f}±{sd:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=3, help="Số bộ tham số ẩn khác nhau")
    ap.add_argument("--streams", type=int, default=1, help="Số order-stream / draw")
    ap.add_argument("--method", default="MAPDCBSSolver")
    ap.add_argument("--nmax", type=int, default=100, help="Bỏ qua slot có N > nmax")
    ap.add_argument("--timeout", type=int, default=360, help="Giới hạn giây mỗi lần chạy; 0=tắt")
    ap.add_argument("--draw-base", type=int, default=1000, help="Seed gốc cho các draw")
    ap.add_argument("--stream-base", type=int, default=7, help="Seed gốc cho order-stream")
    args = ap.parse_args()

    solver_cls = _load_solver(args.method)
    print(f"Method={args.method}  draws={args.draws}  streams={args.streams}  "
          f"nmax={args.nmax}  timeout={args.timeout}s")
    print("=" * 96)
    hdr = (f"{'Slot':<6}{'draw':>5}{'strm':>5}{'N':>4}{'C':>3}{'G':>5}"
           f"{'%Giao':>8}{'%Đúng':>8}{'net':>10}{'t(s)':>8}")
    print(hdr)
    print("-" * 96)

    agg: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"dr": [], "ot": [], "net": [], "wall": []})
    meta: Dict[str, tuple] = {}
    t_start = time.time()

    for d in range(args.draws):
        cfgs = _configs_for_draw(args.draw_base + d)
        for cfg in cfgs:
            if cfg["N"] > args.nmax:
                continue
            meta[cfg["name"]] = (cfg["N"], cfg["C"], cfg["G"])
            for m in range(args.streams):
                env_seed = args.stream_base + d * 101 + m
                res = _run_one(solver_cls, cfg, env_seed, args.timeout)
                dr = res.get("delivery_rate", 0.0)
                ot = res.get("on_time_rate", 0.0)
                net = res.get("net_reward", 0.0)
                wall = res.get("wall", 0.0)
                agg[cfg["name"]]["dr"].append(dr)
                agg[cfg["name"]]["ot"].append(ot)
                agg[cfg["name"]]["net"].append(net)
                agg[cfg["name"]]["wall"].append(wall)
                N, C, G = meta[cfg["name"]]
                tag = " TMO" if res.get("status") == "TIMEOUT" else ""
                print(f"{cfg['name']:<6}{d:>5}{m:>5}{N:>4}{C:>3}{G:>5}"
                      f"{dr:>7.1f}%{ot:>7.1f}%{net:>10.1f}{wall:>8.2f}{tag}")

    print("=" * 96)
    print("TỔNG HỢP theo slot (trung bình ± lệch chuẩn qua mọi draw×stream):")
    print(f"{'Slot':<6}{'N':>4}{'C':>3}{'G':>5}   {'%Giao':>13}{'%Đúng':>15}{'net':>16}")
    print("-" * 96)
    for name, _, _, _, _ in __import__("generate_phase2").SCALE_SLOTS:
        if name not in agg:
            continue
        N, C, G = meta[name]
        a = agg[name]
        print(f"{name:<6}{N:>4}{C:>3}{G:>5}   {_fmt_stat(a['dr'])}  "
              f"{_fmt_stat(a['ot'])}  {_fmt_stat(a['net'])}")
    all_net = [v for a in agg.values() for v in a["net"]]
    print("-" * 96)
    print(f"Tổng net trung bình/slot cộng lại: {sum(statistics.mean(a['net']) for a in agg.values()):.1f}")
    print(f"Tổng thời gian: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
