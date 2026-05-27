"""
generate_random_tests.py — Tạo bộ test ngẫu nhiên để kiểm tra độ tổng quát hóa.

Mục tiêu: kiểm tra solver trên các topology chưa thấy bao giờ,
  không phải chỉ tối ưu theo benchmark cố định.

Topology được hỗ trợ:
  open        — grid gần như trống, vài vật cản rải rác
  corridor    — hai vùng kết nối bởi một hành lang hẹp (horizontal hoặc vertical)
  maze        — vật cản dày đặc ngẫu nhiên (~20-35%)
  asymmetric  — vùng tự do tập trung lệch về một phía

Cách dùng:
  python generate_random_tests.py [--n-configs N] [--seed SEED] [--out FILE]

Ví dụ:
  python generate_random_tests.py --n-configs 8 --seed 99 --out test_random.txt
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import List, Tuple, Optional

Grid = List[List[int]]


def make_border(N: int) -> Grid:
    grid = [[0] * N for _ in range(N)]
    for i in range(N):
        grid[0][i] = grid[N - 1][i] = grid[i][0] = grid[i][N - 1] = 1
    return grid


def is_connected(grid: Grid, N: int) -> bool:
    """Kiểm tra toàn bộ ô tự do có liên thông không."""
    start = next(
        ((r, c) for r in range(N) for c in range(N) if grid[r][c] == 0),
        None
    )
    if start is None:
        return False
    visited, q = {start}, deque([start])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    total_free = sum(1 for r in range(N) for c in range(N) if grid[r][c] == 0)
    return len(visited) == total_free


def add_random_walls(grid: Grid, N: int, density: float, rng: random.Random) -> Grid:
    """Thêm tường ngẫu nhiên với mật độ cho trước, duy trì liên thông."""
    candidates = [(r, c) for r in range(1, N - 1) for c in range(1, N - 1)]
    rng.shuffle(candidates)
    target = int(density * len(candidates))
    added = 0
    for r, c in candidates:
        if added >= target:
            break
        if grid[r][c] == 1:
            continue
        grid[r][c] = 1
        if not is_connected(grid, N):
            grid[r][c] = 0
        else:
            added += 1
    return grid


def make_open(N: int, rng: random.Random) -> Grid:
    """Grid mở: chỉ viền + vài vật cản thưa (~8%)."""
    grid = make_border(N)
    add_random_walls(grid, N, density=0.08, rng=rng)
    return grid


def make_corridor(N: int, rng: random.Random) -> Grid:
    """Hai vùng lớn kết nối bởi một hành lang hẹp (1-3 ô).

    Hành lang có thể ngang (horizontal) hoặc dọc (vertical).
    """
    grid = make_border(N)
    horizontal = rng.choice([True, False])
    gap_width = rng.choice([1, 2, 2])  # 1-cell hoặc 2-cell (2 xác suất cao hơn)

    if horizontal:
        # Tường ngang ở hàng giữa, gap ở vị trí ngẫu nhiên
        mid = N // 2
        gap_start = rng.randint(1, N - 1 - gap_width)
        for c in range(1, N - 1):
            if gap_start <= c < gap_start + gap_width:
                grid[mid][c] = 0
            else:
                grid[mid][c] = 1
    else:
        # Tường dọc ở cột giữa, gap ở hàng ngẫu nhiên
        mid = N // 2
        gap_start = rng.randint(1, N - 1 - gap_width)
        for r in range(1, N - 1):
            if gap_start <= r < gap_start + gap_width:
                grid[r][mid] = 0
            else:
                grid[r][mid] = 1

    # Thêm một ít vật cản vào mỗi vùng
    add_random_walls(grid, N, density=0.06, rng=rng)
    return grid


def make_maze(N: int, rng: random.Random) -> Grid:
    """Grid mê lộ: vật cản dày đặc ngẫu nhiên (15-35%)."""
    density = rng.uniform(0.15, 0.35)
    grid = make_border(N)
    add_random_walls(grid, N, density=density, rng=rng)
    return grid


def make_asymmetric(N: int, rng: random.Random) -> Grid:
    """Vùng tự do tập trung lệch: một phần grid có mật độ vật cản cao hơn nhiều.

    Mô phỏng bản đồ thực tế: kho hàng một bên, khu giao hàng bên kia.
    """
    grid = make_border(N)
    split = rng.randint(N // 3, 2 * N // 3)
    dense_left = rng.choice([True, False])

    # Vùng dày đặc (~30%) và vùng thưa (~5%)
    for r in range(1, N - 1):
        for c in range(1, N - 1):
            is_dense_side = (c < split) if dense_left else (c >= split)
            density_here = 0.30 if is_dense_side else 0.05
            if rng.random() < density_here:
                grid[r][c] = 1

    if not is_connected(grid, N):
        # Xóa tường cho đến khi liên thông
        walls = [(r, c) for r in range(1, N - 1) for c in range(1, N - 1) if grid[r][c] == 1]
        rng.shuffle(walls)
        for r, c in walls:
            if is_connected(grid, N):
                break
            grid[r][c] = 0
    return grid


TOPOLOGY_MAKERS = {
    "open": make_open,
    "corridor": make_corridor,
    "maze": make_maze,
    "asymmetric": make_asymmetric,
}


def grid_to_str(grid: Grid, N: int) -> str:
    return "\n".join(" ".join(str(grid[r][c]) for c in range(N)) for r in range(N))


def make_config(name: str, N: int, topology: str, load: str, rng: random.Random) -> str:
    """Tạo một [CONFIG] block hoàn chỉnh."""
    # Số shipper theo N
    if N <= 15:
        C = rng.randint(3, 5)
    elif N <= 20:
        C = rng.randint(4, 6)
    elif N <= 30:
        C = rng.randint(5, 8)
    else:
        C = rng.randint(7, 12)

    # Tỉ lệ đơn hàng theo mức độ tải
    load_factors = {"light": 0.8, "medium": 1.2, "heavy": 1.8, "extreme": 2.5}
    factor = load_factors[load]
    G = int(C * N * factor / 5)
    G = max(G, C * 3)  # ít nhất 3 đơn/shipper

    # T: bước thời gian — thời gian trung bình để shipper đi được ~30% diagonal
    avg_dist = int(N * 0.5)
    base_T = G * avg_dist // C
    T_factor = rng.uniform(0.7, 1.3)
    T = max(int(base_T * T_factor), G * 3)
    T = (T // 10) * 10  # làm tròn về bội của 10

    # K_max và W_max cho từng shipper
    K_choices = [2, 2, 3, 3]
    W_choices = [20.0, 20.0, 30.0]
    K_max = " ".join(str(rng.choice(K_choices)) for _ in range(C))
    W_max = " ".join(str(rng.choice(W_choices)) for _ in range(C))

    maker = TOPOLOGY_MAKERS[topology]
    grid = maker(N, rng)

    # Kiểm tra liên thông sau cùng
    assert is_connected(grid, N), f"Grid {name} không liên thông!"

    map_str = grid_to_str(grid, N)

    return (
        f"[CONFIG]\n"
        f"name    = {name}\n"
        f"N       = {N}\n"
        f"C       = {C}\n"
        f"G       = {G}\n"
        f"T       = {T}\n"
        f"K_max   = {K_max}\n"
        f"W_max   = {W_max}\n"
        f"[MAP]\n"
        f"{map_str}\n"
        f"[END]\n"
    )


def generate(n_configs: int, seed: int, out_path: str) -> None:
    rng = random.Random(seed)

    # Đảm bảo coverage: ít nhất 1 config mỗi topology
    topologies = list(TOPOLOGY_MAKERS.keys())
    loads = ["light", "medium", "heavy", "extreme"]
    sizes_small = [10, 12, 15, 18, 20]
    sizes_large = [22, 25, 28, 30, 35, 40]

    configs: List[str] = []
    config_count = 0

    # Phần 1: một config cho mỗi topology × một mức tải ngẫu nhiên
    for topo in topologies:
        load = rng.choice(["medium", "heavy"])
        # Xen kẽ grid nhỏ và lớn
        size_pool = sizes_small if config_count % 2 == 0 else sizes_large
        N = rng.choice(size_pool)
        name = f"R{config_count + 1}_{topo[:4].upper()}_{load[:3].upper()}"
        try:
            cfg = make_config(name, N, topo, load, rng)
            configs.append(cfg)
            config_count += 1
        except AssertionError:
            pass  # bỏ qua nếu không liên thông (hiếm)

    # Phần 2: các config ngẫu nhiên còn lại
    while config_count < n_configs:
        topo = rng.choice(topologies)
        load = rng.choice(loads)
        is_large = rng.random() < 0.5
        N = rng.choice(sizes_large if is_large else sizes_small)
        name = f"R{config_count + 1}_{topo[:4].upper()}_{load[:3].upper()}"
        try:
            cfg = make_config(name, N, topo, load, rng)
            configs.append(cfg)
            config_count += 1
        except AssertionError:
            pass

    lines = [
        "# =============================================================",
        "# test_random.txt — Bộ test ngẫu nhiên (auto-generated)",
        f"# Seed: {seed}  |  Configs: {len(configs)}",
        "# Topology: open / corridor / maze / asymmetric",
        "# Load: light / medium / heavy / extreme",
        "# =============================================================",
        "",
        f"[SEED]",
        f"base_seed = {seed}",
        "",
    ]
    lines.extend("\n".join(c.splitlines()) + "\n" for c in configs)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved {len(configs)} configs to {out_path}")

    # In summary
    print("\nGenerated configs:")
    for cfg in configs:
        name_line = next(l for l in cfg.splitlines() if l.startswith("name"))
        n_line = next(l for l in cfg.splitlines() if l.strip().startswith("N "))
        c_line = next(l for l in cfg.splitlines() if l.strip().startswith("C "))
        g_line = next(l for l in cfg.splitlines() if l.strip().startswith("G "))
        t_line = next(l for l in cfg.splitlines() if l.strip().startswith("T "))
        N = int(n_line.split("=")[1])
        C = int(c_line.split("=")[1])
        G = int(g_line.split("=")[1])
        T = int(t_line.split("=")[1])
        name = name_line.split("=")[1].strip()
        print(f"  {name:<30} N={N:<3} C={C} G={G:<4} T={T}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo bộ test ngẫu nhiên cho MAPD solver")
    parser.add_argument("--n-configs", type=int, default=8, help="Số config cần tạo (default 8)")
    parser.add_argument("--seed", type=int, default=77, help="Random seed (default 77)")
    parser.add_argument("--out", default="test_random.txt", help="File output (default test_random.txt)")
    args = parser.parse_args()

    generate(args.n_configs, args.seed, args.out)
