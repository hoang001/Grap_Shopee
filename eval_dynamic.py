#!/usr/bin/env python3
"""
eval_dynamic.py — Đánh giá khả năng thích nghi của solver trên môi trường động.

Khác run_test.py ở chỗ: chạy CÙNG bộ config với NHIỀU base_seed khác nhau,
mỗi seed cho một realization surge/hotspot/order-stream khác nhau. Mục tiêu là
đo độ ROBUST, không phải net_reward trên một episode cố định.

Tiêu chí đạt: mỗi (config, seed) "pass" nếu delivery_rate >= THR và
on_time_rate >= THR. Báo cáo tỉ lệ pass trên toàn bộ.

Ví dụ:
    python3 eval_dynamic.py --config test_config_v3.txt --seeds 42 7 123
    python3 eval_dynamic.py --config test_config_v3.txt --nmax 20 --timeout 60
"""
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import time
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER_DIR = os.path.join(SCRIPT_DIR, "solvers")
for p in (SCRIPT_DIR, SOLVER_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from env import DeliveryEnv, load_config  # noqa: E402

THR = 70.0


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def _config_seed(name: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{base_seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_solver(method: str):
    import importlib.util

    sources = {
        "GreedyBFS": "greedy_bfs.py",
        "VRPOrToolsSolver": "vrp_ortools.py",
        "ACOSolver": "aco_solver.py",
        "MAPDCBSSolver": "mapd_cbs_solver.py",
    }
    fname = sources[method]
    path = os.path.join(SOLVER_DIR, fname)
    spec = importlib.util.spec_from_file_location(method, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, method)


def run_one(solver_cls, cfg: dict, base_seed: int, timeout: int) -> dict:
    seed = _config_seed(cfg["name"], base_seed)
    env = DeliveryEnv(cfg, seed=seed)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(SCRIPT_DIR, "test_config_v3.txt"))
    ap.add_argument("--method", default="MAPDCBSSolver")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 123])
    ap.add_argument("--nmax", type=int, default=30, help="Bỏ qua config có N > nmax")
    ap.add_argument("--timeout", type=int, default=120, help="Giới hạn giây mỗi (config,seed); 0 = tắt")
    ap.add_argument("--tiers", nargs="*", default=None)
    args = ap.parse_args()

    configs = load_config(args.config)
    if args.tiers:
        keep = {t.upper() for t in args.tiers}
        configs = [c for c in configs if c["name"][0].upper() in keep]
    configs = [c for c in configs if c["N"] <= args.nmax]

    solver_cls = load_solver(args.method)

    print(f"Method={args.method}  seeds={args.seeds}  nmax={args.nmax}  "
          f"configs={len(configs)}  timeout={args.timeout}s")
    print("=" * 92)
    header = f"{'Config':<8}{'N':>4}{'C':>3}{'G':>5}  {'seed':>5}{'%Giao':>8}{'%Đúng':>8}{'net':>10}{'t(s)':>8}  PASS"
    print(header)
    print("-" * 92)

    rows: List[Tuple] = []
    n_pass = 0
    n_total = 0
    t_start = time.time()
    for cfg in configs:
        for sd in args.seeds:
            res = run_one(solver_cls, cfg, sd, args.timeout)
            dr = res.get("delivery_rate", 0.0)
            ot = res.get("on_time_rate", 0.0)
            net = res.get("net_reward", 0.0)
            wall = res.get("wall", 0.0)
            passed = (dr >= THR and ot >= THR and res.get("status") != "TIMEOUT")
            n_pass += int(passed)
            n_total += 1
            rows.append((cfg["name"], sd, dr, ot, net, wall, passed))
            tag = "OK " if passed else ("TMO" if res.get("status") == "TIMEOUT" else "no ")
            print(f"{cfg['name']:<8}{cfg['N']:>4}{cfg['C']:>3}{cfg['G']:>5}  {sd:>5}"
                  f"{dr:>7.1f}%{ot:>7.1f}%{net:>10.1f}{wall:>8.2f}  {tag}")

    print("=" * 92)
    # Tổng hợp theo config: config "đạt" nếu pass trên TẤT CẢ seed.
    by_cfg = {}
    for name, sd, dr, ot, net, wall, passed in rows:
        by_cfg.setdefault(name, []).append(passed)
    cfg_pass = sum(1 for v in by_cfg.values() if all(v))
    print(f"Pass (config,seed): {n_pass}/{n_total} = {100.0*n_pass/max(n_total,1):.1f}%")
    print(f"Config pass mọi seed: {cfg_pass}/{len(by_cfg)} = {100.0*cfg_pass/max(len(by_cfg),1):.1f}%")
    print(f"Tổng thời gian: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
