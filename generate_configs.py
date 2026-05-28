#!/usr/bin/env python3
"""
generate_configs.py - Sinh test_config_v2.txt với 106 configs đa dạng.

Cách chạy:
    python3 generate_configs.py --out test_config_v2.txt

Tổng quan tầng:
  S  ( 5): N=5-7,    smoke / sanity
  A  (15): N=8-10,   biến thể C / map type / G-T ratio
  B  (20): N=11-15,  surge tường minh, map phức tạp, shipper hỗn hợp
  C  (15): N=16-20,  large scale
  D  (10): edge cases
  E  (10): N=22-50,  medium-large
  F  (10): N=55-100, large to very large
  X  (10): cùng map, hotspot ở 5 vị trí khác nhau — đo spatial adaptability
  Y  ( 6): cùng map/hotspot/timing, amplitude 0→12 — đo surge intensity response
  Z  ( 5): cùng map/hotspot/amplitude, timing thay đổi — đo temporal adaptability
"""

import argparse
import random
from collections import deque
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Map generators
# ---------------------------------------------------------------------------

def _make_border(N: int) -> List[List[int]]:
    """Lưới rỗng chỉ có viền tường."""
    grid = [[0] * N for _ in range(N)]
    for i in range(N):
        grid[0][i] = grid[N - 1][i] = 1
        grid[i][0] = grid[i][N - 1] = 1
    return grid


def _is_connected(grid: List[List[int]]) -> bool:
    """BFS kiểm tra tất cả ô trống có liên thông không."""
    N = len(grid)
    free = [(r, c) for r in range(N) for c in range(N) if grid[r][c] == 0]
    if not free:
        return False
    visited = {free[0]}
    q = deque([free[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0:
                visited.add((nr, nc))
                q.append((nr, nc))
    return len(visited) == len(free)


def _free_cells(grid: List[List[int]]) -> List[Tuple[int, int]]:
    N = len(grid)
    return [(r, c) for r in range(N) for c in range(N) if grid[r][c] == 0]


def open_map(N: int) -> List[List[int]]:
    """Map trống, chỉ viền tường."""
    return _make_border(N)


def sparse_map(N: int, density: float = 0.10, seed: int = 0) -> List[List[int]]:
    """
    Map với vật cản ngẫu nhiên, đảm bảo liên thông.

    N < 30: thêm từng vật cản, kiểm tra liên thông mỗi lần (an toàn nhất).
    N >= 30: đặt tất cả cùng lúc rồi kiểm tra một lần (nhanh hơn cho map lớn).
    """
    if N >= 30:
        return _sparse_map_fast(N, density, seed)
    return _sparse_map_careful(N, density, seed)


def _sparse_map_careful(N: int, density: float, seed: int) -> List[List[int]]:
    """Thêm từng vật cản + BFS sau mỗi lần; dùng cho N nhỏ."""
    rng = random.Random(seed)
    interior = [(r, c) for r in range(1, N - 1) for c in range(1, N - 1)]
    n_obs = int(len(interior) * density)
    grid = _make_border(N)
    rng.shuffle(interior)
    placed = 0
    for r, c in interior:
        if placed >= n_obs:
            break
        grid[r][c] = 1
        if _is_connected(grid):
            placed += 1
        else:
            grid[r][c] = 0
    return grid


def _sparse_map_fast(N: int, density: float, seed: int) -> List[List[int]]:
    """Đặt tất cả vật cản cùng lúc, giảm mật độ nếu mất liên thông; dùng cho N lớn."""
    rng = random.Random(seed)
    interior = [(r, c) for r in range(1, N - 1) for c in range(1, N - 1)]
    for attempt_density in [density, density * 0.75, density * 0.50, density * 0.25]:
        if attempt_density < 0.01:
            break
        grid = _make_border(N)
        n_obs = int(len(interior) * attempt_density)
        for r, c in rng.sample(interior, n_obs):
            grid[r][c] = 1
        if _is_connected(grid):
            return grid
    return _make_border(N)


def divided_h_map(N: int, n_gaps: int = 2) -> List[List[int]]:
    """Tường ngang chia đôi map, có n_gaps lỗ hổng."""
    grid = _make_border(N)
    mid = N // 2
    for c in range(1, N - 1):
        grid[mid][c] = 1
    step = max(1, (N - 2) // (n_gaps + 1))
    for i in range(1, n_gaps + 1):
        pos = 1 + i * step
        if 1 <= pos <= N - 2:
            grid[mid][pos] = 0
    if not _is_connected(grid):
        grid[mid][N // 2] = 0
    return grid


def divided_v_map(N: int, n_gaps: int = 2) -> List[List[int]]:
    """Tường dọc chia đôi map, có n_gaps lỗ hổng."""
    grid = _make_border(N)
    mid = N // 2
    for r in range(1, N - 1):
        grid[r][mid] = 1
    step = max(1, (N - 2) // (n_gaps + 1))
    for i in range(1, n_gaps + 1):
        pos = 1 + i * step
        if 1 <= pos <= N - 2:
            grid[pos][mid] = 0
    if not _is_connected(grid):
        grid[N // 2][mid] = 0
    return grid


def bottleneck_map(N: int) -> List[List[int]]:
    """Hai khu vực nối bởi 1 ô duy nhất."""
    grid = _make_border(N)
    mid = N // 2
    for c in range(1, N - 1):
        grid[mid][c] = 1
    grid[mid][mid] = 0
    if not _is_connected(grid):
        grid[mid][mid - 1] = 0
    return grid


def rooms_4_map(N: int) -> List[List[int]]:
    """4 khu vực góc nối qua trung tâm (dạng +)."""
    grid = _make_border(N)
    mid_r, mid_c = N // 2, N // 2
    for c in range(1, N - 1):
        grid[mid_r][c] = 1
    for r in range(1, N - 1):
        grid[r][mid_c] = 1
    for gap in range(1, min(3, mid_c)):
        if 1 <= mid_c - gap <= N - 2:
            grid[mid_r][mid_c - gap] = 0
            break
    for gap in range(1, min(3, N - 1 - mid_c)):
        if 1 <= mid_c + gap <= N - 2:
            grid[mid_r][mid_c + gap] = 0
            break
    for gap in range(1, min(3, mid_r)):
        if 1 <= mid_r - gap <= N - 2:
            grid[mid_r - gap][mid_c] = 0
            break
    for gap in range(1, min(3, N - 1 - mid_r)):
        if 1 <= mid_r + gap <= N - 2:
            grid[mid_r + gap][mid_c] = 0
            break
    grid[mid_r][mid_c] = 0
    if not _is_connected(grid):
        return open_map(N)
    return grid


def corridor_h_map(N: int) -> List[List[int]]:
    """2 khu vực trái/phải nối bởi hành lang hẹp ở giữa."""
    grid = _make_border(N)
    div_c = N // 3
    div_c2 = 2 * N // 3
    mid_r = N // 2
    for r in range(1, N - 1):
        if r != mid_r:
            if 1 <= div_c <= N - 2:
                grid[r][div_c] = 1
            if 1 <= div_c2 <= N - 2:
                grid[r][div_c2] = 1
    if not _is_connected(grid):
        return divided_v_map(N, n_gaps=2)
    return grid


def ring_map(N: int) -> List[List[int]]:
    """Map mở với vật cản hình chữ nhật rỗng ở giữa."""
    grid = _make_border(N)
    inn, out = N // 4, 3 * N // 4
    for r in range(inn, out + 1):
        for c in range(inn, out + 1):
            if (r == inn or r == out or c == inn or c == out) and 1 <= r <= N - 2 and 1 <= c <= N - 2:
                grid[r][c] = 1
    if not _is_connected(grid):
        grid[inn][N // 2] = 0
    return grid


def grid_obstacles_map(N: int) -> List[List[int]]:
    """Các block vật cản 2×2 theo dạng lưới đều."""
    grid = _make_border(N)
    step = 4
    for r in range(2, N - 2, step):
        for c in range(2, N - 2, step):
            for dr in range(2):
                for dc in range(2):
                    nr, nc = r + dr, c + dc
                    if 1 <= nr <= N - 2 and 1 <= nc <= N - 2:
                        grid[nr][nc] = 1
    if not _is_connected(grid):
        return sparse_map(N, 0.12)
    return grid


def maze_map(N: int, density: float = 0.22, seed: int = 0) -> List[List[int]]:
    return sparse_map(N, density, seed)


def asymmetric_map(N: int, seed: int = 0) -> List[List[int]]:
    """Vật cản tập trung ở nửa bản đồ."""
    grid = _make_border(N)
    rng = random.Random(seed)
    dense_cells = [(r, c) for r in range(1, N // 2) for c in range(1, N // 2)]
    sparse_cells = [(r, c) for r in range(N // 2, N - 1) for c in range(N // 2, N - 1)]
    rng.shuffle(dense_cells)
    rng.shuffle(sparse_cells)
    placed = 0
    for r, c in dense_cells:
        if placed >= int(len(dense_cells) * 0.20):
            break
        grid[r][c] = 1
        if _is_connected(grid):
            placed += 1
        else:
            grid[r][c] = 0
    placed = 0
    for r, c in sparse_cells:
        if placed >= int(len(sparse_cells) * 0.05):
            break
        grid[r][c] = 1
        if _is_connected(grid):
            placed += 1
        else:
            grid[r][c] = 0
    return grid


# ---------------------------------------------------------------------------
# Shipper parameter helpers
# ---------------------------------------------------------------------------

def _k_maxes(C: int) -> List[int]:
    """Phân bổ K_max theo pattern [3,3,2,2,3,...] cho C shipper."""
    pattern = [3, 3, 2, 2, 3]
    return [pattern[i % len(pattern)] for i in range(C)]


def _w_maxes(C: int) -> List[float]:
    """Phân bổ W_max theo pattern [20,20,30,30,20,...] cho C shipper."""
    pattern = [20.0, 20.0, 30.0, 30.0, 20.0]
    return [pattern[i % len(pattern)] for i in range(C)]


# ---------------------------------------------------------------------------
# Config formatter
# ---------------------------------------------------------------------------

def _fmt_map(grid: List[List[int]]) -> str:
    return "\n".join(" ".join(map(str, row)) for row in grid)


def _fmt_config(
    name: str,
    N: int,
    C: int,
    G: int,
    T: int,
    k_maxes: List[int],
    w_maxes: List[float],
    grid: List[List[int]],
    surge_amplitude: Optional[float] = None,
    surge_windows: Optional[List[Tuple[int, int]]] = None,
    hotspots: Optional[List[Tuple[int, int]]] = None,
) -> str:
    lines = [
        "[CONFIG]",
        f"name    = {name}",
        f"N       = {N}",
        f"C       = {C}",
        f"G       = {G}",
        f"T       = {T}",
        f"K_max   = {' '.join(map(str, k_maxes))}",
        f"W_max   = {' '.join(f'{w:.1f}' for w in w_maxes)}",
    ]
    if surge_amplitude is not None:
        lines.append(f"surge_amplitude = {surge_amplitude:.1f}")
    if surge_windows:
        sw = " ".join(f"{s} {e}" for s, e in surge_windows)
        lines.append(f"surge_windows = {sw}")
    if hotspots:
        hs = " ".join(f"{r} {c}" for r, c in hotspots)
        lines.append(f"hotspots = {hs}")
    lines.append("[MAP]")
    lines.append(_fmt_map(grid))
    lines.append("[END]")
    lines.append("")
    return "\n".join(lines)


def _find_free(grid: List[List[int]], r: int, c: int) -> Tuple[int, int]:
    """Tìm ô trống gần nhất với (r, c) trên grid."""
    free = _free_cells(grid)
    return min(free, key=lambda x: abs(x[0] - r) + abs(x[1] - c))


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

def build_all_configs() -> List[str]:
    configs: List[str] = []

    # -----------------------------------------------------------------------
    # Tier S — Smoke (5 configs, N=5-7)
    # -----------------------------------------------------------------------

    configs.append(_fmt_config(
        "S1", N=5, C=1, G=5, T=120,
        k_maxes=[3], w_maxes=[20.0], grid=open_map(5),
    ))
    configs.append(_fmt_config(
        "S2", N=6, C=2, G=8, T=120,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=open_map(6),
    ))
    configs.append(_fmt_config(
        "S3", N=7, C=1, G=8, T=120,
        k_maxes=[3], w_maxes=[20.0], grid=sparse_map(7, 0.08, 1),
    ))
    configs.append(_fmt_config(
        "S4", N=7, C=2, G=12, T=120,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=sparse_map(7, 0.10, 2),
    ))
    configs.append(_fmt_config(
        "S5", N=7, C=3, G=10, T=120,
        k_maxes=[2, 2, 2], w_maxes=[20.0, 20.0, 20.0], grid=divided_h_map(7, 2),
    ))

    # -----------------------------------------------------------------------
    # Tier A — Small (15 configs, N=8-10)
    # -----------------------------------------------------------------------

    base_map_10 = sparse_map(10, 0.10, 10)

    # A1-A5: biến thể C
    configs.append(_fmt_config(
        "A1", N=10, C=1, G=15, T=240,
        k_maxes=[3], w_maxes=[20.0], grid=base_map_10,
    ))
    configs.append(_fmt_config(
        "A2", N=10, C=2, G=20, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=base_map_10,
    ))
    configs.append(_fmt_config(
        "A3", N=10, C=3, G=28, T=240,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=base_map_10,
    ))
    configs.append(_fmt_config(
        "A4", N=10, C=4, G=30, T=240,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0], grid=base_map_10,
    ))
    configs.append(_fmt_config(
        "A5", N=10, C=2, G=50, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=base_map_10,
    ))

    # A6-A10: biến thể map type
    configs.append(_fmt_config(
        "A6", N=10, C=2, G=25, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=open_map(10),
    ))
    configs.append(_fmt_config(
        "A7", N=10, C=2, G=25, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=sparse_map(10, 0.15, 11),
    ))
    configs.append(_fmt_config(
        "A8", N=10, C=2, G=25, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=divided_h_map(10, 2),
    ))
    configs.append(_fmt_config(
        "A9", N=10, C=2, G=25, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=corridor_h_map(10),
    ))
    configs.append(_fmt_config(
        "A10", N=10, C=2, G=25, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=rooms_4_map(10),
    ))

    # A11-A15: biến thể G/T ratio
    map_a = sparse_map(10, 0.10, 12)
    configs.append(_fmt_config(
        "A11", N=10, C=2, G=10, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=map_a,
    ))
    configs.append(_fmt_config(
        "A12", N=10, C=2, G=25, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=map_a,
    ))
    configs.append(_fmt_config(
        "A13", N=10, C=2, G=42, T=240,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=map_a,
    ))
    configs.append(_fmt_config(
        "A14", N=10, C=2, G=25, T=120,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=map_a,
    ))
    configs.append(_fmt_config(
        "A15", N=10, C=2, G=25, T=480,
        k_maxes=[3, 3], w_maxes=[20.0, 20.0], grid=map_a,
    ))

    # -----------------------------------------------------------------------
    # Tier B — Medium (20 configs, N=11-15)
    # -----------------------------------------------------------------------

    # B1-B4: surge tường minh
    map_b1 = sparse_map(12, 0.10, 20)
    hs_b1 = [_find_free(map_b1, 3, 3), _find_free(map_b1, 8, 8)]
    configs.append(_fmt_config(
        "B1", N=12, C=3, G=40, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=map_b1,
        surge_amplitude=3.0, surge_windows=[(55, 115), (215, 275)], hotspots=hs_b1,
    ))

    map_b2 = divided_h_map(12, 2)
    hs_b2 = [_find_free(map_b2, 3, 5)]
    configs.append(_fmt_config(
        "B2", N=12, C=3, G=40, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=map_b2,
        surge_amplitude=2.5, surge_windows=[(60, 130)], hotspots=hs_b2,
    ))

    map_b3 = sparse_map(14, 0.12, 21)
    hs_b3 = [_find_free(map_b3, 3, 3), _find_free(map_b3, 10, 10)]
    configs.append(_fmt_config(
        "B3", N=14, C=4, G=55, T=480,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0], grid=map_b3,
        surge_amplitude=3.0, surge_windows=[(70, 150), (270, 340)], hotspots=hs_b3,
    ))

    map_b4 = open_map(14)
    hs_b4 = [_find_free(map_b4, 7, 7)]
    configs.append(_fmt_config(
        "B4", N=14, C=4, G=55, T=480,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0], grid=map_b4,
        surge_amplitude=4.0, surge_windows=[(100, 180)], hotspots=hs_b4,
    ))

    # B5-B12: biến thể map
    configs.append(_fmt_config(
        "B5", N=12, C=3, G=40, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=open_map(12),
    ))
    configs.append(_fmt_config(
        "B6", N=12, C=3, G=40, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=maze_map(12, 0.20, 22),
    ))
    configs.append(_fmt_config(
        "B7", N=13, C=3, G=45, T=400,
        k_maxes=[3, 3, 3], w_maxes=[20.0, 20.0, 20.0], grid=asymmetric_map(13, 23),
    ))
    configs.append(_fmt_config(
        "B8", N=13, C=3, G=45, T=400,
        k_maxes=[3, 3, 3], w_maxes=[20.0, 20.0, 20.0], grid=bottleneck_map(13),
    ))
    configs.append(_fmt_config(
        "B9", N=14, C=4, G=55, T=480,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0], grid=rooms_4_map(14),
    ))
    configs.append(_fmt_config(
        "B10", N=14, C=4, G=55, T=480,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0], grid=divided_v_map(14, 2),
    ))
    configs.append(_fmt_config(
        "B11", N=15, C=4, G=60, T=480,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
        grid=sparse_map(15, 0.12, 24),
    ))
    configs.append(_fmt_config(
        "B12", N=15, C=4, G=60, T=480,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
        grid=maze_map(15, 0.22, 25),
    ))

    # B13-B16: shipper hỗn hợp
    configs.append(_fmt_config(
        "B13", N=13, C=3, G=40, T=400,
        k_maxes=[3, 2, 1], w_maxes=[20.0, 30.0, 40.0], grid=sparse_map(13, 0.10, 26),
    ))
    configs.append(_fmt_config(
        "B14", N=13, C=4, G=45, T=400,
        k_maxes=[3, 3, 3, 3], w_maxes=[10.0, 10.0, 10.0, 10.0],
        grid=sparse_map(13, 0.10, 27),
    ))
    configs.append(_fmt_config(
        "B15", N=13, C=4, G=45, T=400,
        k_maxes=[1, 1, 2, 3], w_maxes=[40.0, 40.0, 30.0, 20.0],
        grid=sparse_map(13, 0.10, 28),
    ))
    configs.append(_fmt_config(
        "B16", N=12, C=2, G=35, T=360,
        k_maxes=[1, 3], w_maxes=[40.0, 20.0], grid=sparse_map(12, 0.10, 29),
    ))

    # B17-B20: G/T extremes
    map_b17 = sparse_map(13, 0.10, 30)
    configs.append(_fmt_config(
        "B17", N=13, C=3, G=80, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=map_b17,
    ))
    configs.append(_fmt_config(
        "B18", N=13, C=3, G=20, T=480,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=map_b17,
    ))
    configs.append(_fmt_config(
        "B19", N=12, C=3, G=60, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=sparse_map(12, 0.10, 31),
    ))
    configs.append(_fmt_config(
        "B20", N=15, C=4, G=60, T=240,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
        grid=sparse_map(15, 0.10, 32),
    ))

    # -----------------------------------------------------------------------
    # Tier C — Large (15 configs, N=16-20)
    # -----------------------------------------------------------------------

    configs.append(_fmt_config(
        "C1", N=16, C=4, G=65, T=500,
        k_maxes=_k_maxes(4), w_maxes=_w_maxes(4), grid=sparse_map(16, 0.10, 40),
    ))
    configs.append(_fmt_config(
        "C2", N=16, C=5, G=70, T=500,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=sparse_map(16, 0.12, 41),
    ))
    configs.append(_fmt_config(
        "C3", N=17, C=4, G=70, T=550,
        k_maxes=_k_maxes(4), w_maxes=_w_maxes(4), grid=divided_h_map(17, 2),
    ))
    configs.append(_fmt_config(
        "C4", N=17, C=5, G=75, T=550,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=sparse_map(17, 0.12, 42),
    ))
    configs.append(_fmt_config(
        "C5", N=18, C=5, G=80, T=600,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=sparse_map(18, 0.10, 43),
    ))
    configs.append(_fmt_config(
        "C6", N=18, C=6, G=85, T=600,
        k_maxes=_k_maxes(6), w_maxes=_w_maxes(6), grid=sparse_map(18, 0.10, 44),
    ))
    configs.append(_fmt_config(
        "C7", N=19, C=5, G=90, T=650,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=sparse_map(19, 0.12, 45),
    ))
    configs.append(_fmt_config(
        "C8", N=19, C=6, G=95, T=650,
        k_maxes=_k_maxes(6), w_maxes=_w_maxes(6), grid=sparse_map(19, 0.12, 46),
    ))
    configs.append(_fmt_config(
        "C9", N=20, C=5, G=100, T=720,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=sparse_map(20, 0.10, 47),
    ))
    configs.append(_fmt_config(
        "C10", N=20, C=6, G=110, T=720,
        k_maxes=_k_maxes(6), w_maxes=_w_maxes(6), grid=sparse_map(20, 0.12, 48),
    ))
    configs.append(_fmt_config(
        "C11", N=20, C=5, G=100, T=720,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=ring_map(20),
    ))
    configs.append(_fmt_config(
        "C12", N=20, C=5, G=100, T=720,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=grid_obstacles_map(20),
    ))
    configs.append(_fmt_config(
        "C13", N=18, C=5, G=80, T=600,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=bottleneck_map(18),
    ))
    configs.append(_fmt_config(
        "C14", N=20, C=6, G=80, T=720,
        k_maxes=_k_maxes(6), w_maxes=_w_maxes(6), grid=sparse_map(20, 0.08, 49),
    ))
    configs.append(_fmt_config(
        "C15", N=20, C=5, G=150, T=720,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=sparse_map(20, 0.10, 50),
    ))

    # -----------------------------------------------------------------------
    # Tier D — Edge Cases (10 configs)
    # -----------------------------------------------------------------------

    configs.append(_fmt_config(
        "D1", N=8, C=6, G=20, T=200,
        k_maxes=[2]*6, w_maxes=[20.0]*6, grid=open_map(8),
    ))
    configs.append(_fmt_config(
        "D2", N=20, C=1, G=30, T=720,
        k_maxes=[3], w_maxes=[20.0], grid=sparse_map(20, 0.08, 51),
    ))
    configs.append(_fmt_config(
        "D3", N=12, C=3, G=5, T=720,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=sparse_map(12, 0.10, 52),
    ))
    configs.append(_fmt_config(
        "D4", N=20, C=6, G=200, T=720,
        k_maxes=_k_maxes(6), w_maxes=_w_maxes(6), grid=sparse_map(20, 0.10, 53),
    ))

    map_d5 = sparse_map(12, 0.10, 54)
    hs_d5 = [_find_free(map_d5, 3, 3), _find_free(map_d5, 8, 8)]
    configs.append(_fmt_config(
        "D5", N=12, C=3, G=40, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=map_d5,
        surge_amplitude=8.0, surge_windows=[(60, 120), (200, 260)], hotspots=hs_d5,
    ))

    map_d6 = sparse_map(12, 0.10, 55)
    hs_d6 = [_find_free(map_d6, 5, 5)]
    configs.append(_fmt_config(
        "D6", N=12, C=3, G=40, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=map_d6,
        surge_amplitude=3.0, surge_windows=[(40, 80), (130, 170), (240, 290)], hotspots=hs_d6,
    ))

    map_d7 = sparse_map(15, 0.10, 56)
    hs_d7 = [_find_free(map_d7, 1, 1), _find_free(map_d7, 13, 13)]
    configs.append(_fmt_config(
        "D7", N=15, C=4, G=55, T=480,
        k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0], grid=map_d7,
        surge_amplitude=3.0, surge_windows=[(80, 160), (280, 360)], hotspots=hs_d7,
    ))

    configs.append(_fmt_config(
        "D8", N=12, C=3, G=40, T=360,
        k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0], grid=maze_map(12, 0.28, 57),
    ))
    configs.append(_fmt_config(
        "D9", N=12, C=4, G=40, T=360,
        k_maxes=[1, 1, 1, 1], w_maxes=[30.0, 30.0, 30.0, 30.0],
        grid=sparse_map(12, 0.10, 58),
    ))
    configs.append(_fmt_config(
        "D10", N=12, C=4, G=40, T=360,
        k_maxes=[3, 3, 3, 3], w_maxes=[5.0, 5.0, 5.0, 5.0],
        grid=sparse_map(12, 0.10, 59),
    ))

    # -----------------------------------------------------------------------
    # Tier E — Medium-Large (10 configs, N=22-50)
    # -----------------------------------------------------------------------

    configs.append(_fmt_config(
        "E1", N=22, C=5, G=110, T=770,
        k_maxes=_k_maxes(5), w_maxes=_w_maxes(5), grid=sparse_map(22, 0.10, 60),
    ))
    configs.append(_fmt_config(
        "E2", N=25, C=6, G=140, T=875,
        k_maxes=_k_maxes(6), w_maxes=_w_maxes(6), grid=sparse_map(25, 0.10, 61),
    ))
    configs.append(_fmt_config(
        "E3", N=28, C=7, G=165, T=980,
        k_maxes=_k_maxes(7), w_maxes=_w_maxes(7), grid=divided_h_map(28, 3),
    ))
    configs.append(_fmt_config(
        "E4", N=30, C=7, G=185, T=1050,
        k_maxes=_k_maxes(7), w_maxes=_w_maxes(7), grid=sparse_map(30, 0.10, 62),
    ))
    configs.append(_fmt_config(
        "E5", N=32, C=8, G=210, T=1120,
        k_maxes=_k_maxes(8), w_maxes=_w_maxes(8), grid=open_map(32),
    ))
    configs.append(_fmt_config(
        "E6", N=35, C=8, G=240, T=1225,
        k_maxes=_k_maxes(8), w_maxes=_w_maxes(8), grid=sparse_map(35, 0.10, 63),
    ))
    configs.append(_fmt_config(
        "E7", N=38, C=9, G=270, T=1330,
        k_maxes=_k_maxes(9), w_maxes=_w_maxes(9), grid=rooms_4_map(38),
    ))
    configs.append(_fmt_config(
        "E8", N=40, C=10, G=300, T=1400,
        k_maxes=_k_maxes(10), w_maxes=_w_maxes(10), grid=sparse_map(40, 0.10, 64),
    ))
    configs.append(_fmt_config(
        "E9", N=45, C=11, G=340, T=1575,
        k_maxes=_k_maxes(11), w_maxes=_w_maxes(11), grid=sparse_map(45, 0.08, 65),
    ))
    configs.append(_fmt_config(
        "E10", N=50, C=12, G=380, T=1750,
        k_maxes=_k_maxes(12), w_maxes=_w_maxes(12), grid=sparse_map(50, 0.08, 66),
    ))

    # -----------------------------------------------------------------------
    # Tier F — Large (10 configs, N=55-100)
    # -----------------------------------------------------------------------

    configs.append(_fmt_config(
        "F1", N=55, C=13, G=430, T=1925,
        k_maxes=_k_maxes(13), w_maxes=_w_maxes(13), grid=sparse_map(55, 0.08, 70),
    ))
    configs.append(_fmt_config(
        "F2", N=60, C=14, G=480, T=2100,
        k_maxes=_k_maxes(14), w_maxes=_w_maxes(14), grid=sparse_map(60, 0.08, 71),
    ))
    configs.append(_fmt_config(
        "F3", N=65, C=15, G=530, T=2275,
        k_maxes=_k_maxes(15), w_maxes=_w_maxes(15), grid=sparse_map(65, 0.07, 72),
    ))
    configs.append(_fmt_config(
        "F4", N=70, C=16, G=580, T=2450,
        k_maxes=_k_maxes(16), w_maxes=_w_maxes(16), grid=sparse_map(70, 0.07, 73),
    ))
    configs.append(_fmt_config(
        "F5", N=75, C=17, G=630, T=2625,
        k_maxes=_k_maxes(17), w_maxes=_w_maxes(17), grid=sparse_map(75, 0.06, 74),
    ))
    configs.append(_fmt_config(
        "F6", N=80, C=18, G=680, T=2800,
        k_maxes=_k_maxes(18), w_maxes=_w_maxes(18), grid=sparse_map(80, 0.06, 75),
    ))
    configs.append(_fmt_config(
        "F7", N=85, C=19, G=730, T=2975,
        k_maxes=_k_maxes(19), w_maxes=_w_maxes(19), grid=sparse_map(85, 0.06, 76),
    ))
    # F8, F9 dùng open_map để tránh thời gian sinh map quá lâu
    configs.append(_fmt_config(
        "F8", N=90, C=20, G=780, T=3150,
        k_maxes=_k_maxes(20), w_maxes=_w_maxes(20), grid=open_map(90),
    ))
    configs.append(_fmt_config(
        "F9", N=95, C=21, G=830, T=3325,
        k_maxes=_k_maxes(21), w_maxes=_w_maxes(21), grid=open_map(95),
    ))
    configs.append(_fmt_config(
        "F10", N=100, C=22, G=900, T=3500,
        k_maxes=_k_maxes(22), w_maxes=_w_maxes(22), grid=open_map(100),
    ))

    # -----------------------------------------------------------------------
    # Tier X — Spatial adaptability (10 configs)
    #
    # Biến duy nhất: vị trí hotspot (top-left / top-right / bottom-left /
    # bottom-right / center). Mọi tham số khác — N, C, G, T, map grid,
    # surge_amplitude, surge_windows — giống hệt nhau trong mỗi nhóm.
    #
    # Nhóm 1 (X1-X5): N=12, surge=[90,180], amp=3.0
    # Nhóm 2 (X6-X10): N=15, surge=[120,240], amp=3.0
    #
    # Thuật toán tốt phải cho score tương đương trên X1-X5 và X6-X10.
    # Score lệch lớn giữa các config cùng nhóm → thuật toán phụ thuộc
    # vị trí hotspot cố định thay vì thực sự detect động.
    # -----------------------------------------------------------------------

    # Nhóm X1-X5: N=12
    map_x12 = sparse_map(12, 0.10, 100)
    _x12_tl = _find_free(map_x12, 2,  2)   # top-left
    _x12_tr = _find_free(map_x12, 2,  9)   # top-right
    _x12_bl = _find_free(map_x12, 9,  2)   # bottom-left
    _x12_br = _find_free(map_x12, 9,  9)   # bottom-right
    _x12_ct = _find_free(map_x12, 5,  5)   # center

    _x12_base = dict(N=12, C=3, G=40, T=360,
                     k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0],
                     grid=map_x12,
                     surge_amplitude=3.0, surge_windows=[(90, 180)])

    configs.append(_fmt_config("X1", **_x12_base, hotspots=[_x12_tl]))
    configs.append(_fmt_config("X2", **_x12_base, hotspots=[_x12_tr]))
    configs.append(_fmt_config("X3", **_x12_base, hotspots=[_x12_bl]))
    configs.append(_fmt_config("X4", **_x12_base, hotspots=[_x12_br]))
    configs.append(_fmt_config("X5", **_x12_base, hotspots=[_x12_ct]))

    # Nhóm X6-X10: N=15
    map_x15 = sparse_map(15, 0.10, 101)
    _x15_tl = _find_free(map_x15, 2,  2)
    _x15_tr = _find_free(map_x15, 2,  12)
    _x15_bl = _find_free(map_x15, 12, 2)
    _x15_br = _find_free(map_x15, 12, 12)
    _x15_ct = _find_free(map_x15, 7,  7)

    _x15_base = dict(N=15, C=4, G=60, T=480,
                     k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
                     grid=map_x15,
                     surge_amplitude=3.0, surge_windows=[(120, 240)])

    configs.append(_fmt_config("X6",  **_x15_base, hotspots=[_x15_tl]))
    configs.append(_fmt_config("X7",  **_x15_base, hotspots=[_x15_tr]))
    configs.append(_fmt_config("X8",  **_x15_base, hotspots=[_x15_bl]))
    configs.append(_fmt_config("X9",  **_x15_base, hotspots=[_x15_br]))
    configs.append(_fmt_config("X10", **_x15_base, hotspots=[_x15_ct]))

    # -----------------------------------------------------------------------
    # Tier Y — Surge intensity (6 configs)
    #
    # Biến duy nhất: surge_amplitude (0 → 12).
    # Mọi thứ khác cố định: cùng map_x12, cùng hotspot center,
    # cùng surge window [90,270] (wide window để amplitude có đủ thời gian tác động).
    #
    # Y1 (amp=0.0) là baseline không có surge — rate luôn = lambda0 = G/T.
    # So sánh score Y1 → Y6 cho thấy thuật toán mất bao nhiêu % hiệu quả
    # khi surge càng mạnh. Thuật toán adaptive sẽ giữ được score cao hơn.
    # -----------------------------------------------------------------------

    _y_base = dict(N=12, C=3, G=40, T=360,
                   k_maxes=[3, 2, 3], w_maxes=[20.0, 30.0, 20.0],
                   grid=map_x12,          # cùng map với X1-X5
                   surge_windows=[(90, 270)],
                   hotspots=[_x12_ct])    # cùng hotspot center với X5

    configs.append(_fmt_config("Y1", **_y_base, surge_amplitude=0.0))
    configs.append(_fmt_config("Y2", **_y_base, surge_amplitude=1.0))
    configs.append(_fmt_config("Y3", **_y_base, surge_amplitude=3.0))
    configs.append(_fmt_config("Y4", **_y_base, surge_amplitude=5.0))
    configs.append(_fmt_config("Y5", **_y_base, surge_amplitude=8.0))
    configs.append(_fmt_config("Y6", **_y_base, surge_amplitude=12.0))

    # -----------------------------------------------------------------------
    # Tier Z — Temporal adaptability (5 configs)
    #
    # Biến duy nhất: vị trí thời gian của surge window.
    # Mọi thứ khác cố định: cùng map_x15, cùng hotspot center (X10),
    # cùng surge_amplitude=3.0. T=480.
    #
    # Z1: surge sớm  [0.10T, 0.30T] — shipper cần reposition ngay đầu game
    # Z2: surge giữa [0.40T, 0.60T] — surge sau khi shipper đã ở trạng thái ổn định
    # Z3: surge muộn [0.70T, 0.90T] — surge gần cuối, deadline càng gần càng nguy hiểm
    # Z4: surge đầu + cuối — test khả năng "quên" trạng thái surge đầu
    # Z5: 3 surge ngắn rải đều — test phản ứng nhiều lần liên tiếp
    #
    # Thuật toán proactive (reposition trước surge) sẽ nổi bật ở Z1.
    # Thuật toán reactive (detect từ observation) sẽ bắt kịp ở Z2, Z3.
    # Z4/Z5 lọc ra thuật toán không bị "kẹt" ở hotspot cũ sau khi surge kết thúc.
    # -----------------------------------------------------------------------

    _z_base = dict(N=15, C=4, G=60, T=480,
                   k_maxes=[3, 3, 2, 2], w_maxes=[20.0, 20.0, 30.0, 30.0],
                   grid=map_x15,          # cùng map với X6-X10
                   surge_amplitude=3.0,
                   hotspots=[_x15_ct])    # cùng hotspot center với X10

    configs.append(_fmt_config("Z1", **_z_base, surge_windows=[(48, 144)]))     # early
    configs.append(_fmt_config("Z2", **_z_base, surge_windows=[(192, 288)]))    # mid
    configs.append(_fmt_config("Z3", **_z_base, surge_windows=[(336, 432)]))    # late
    configs.append(_fmt_config("Z4", **_z_base, surge_windows=[(48, 120), (360, 432)]))   # early+late
    configs.append(_fmt_config("Z5", **_z_base, surge_windows=[(48, 96), (192, 240), (336, 384)]))  # triple

    return configs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_configs(configs: List[str]) -> bool:
    import sys, tempfile, os
    sys.path.insert(0, ".")
    try:
        from env import load_config
    except ImportError:
        print("[WARN] Không load được env.py để validate; bỏ qua.")
        return True

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(configs))
        tmp = f.name
    try:
        cfgs = load_config(tmp)
        ok = True
        for cfg in cfgs:
            free = [(r, c) for r in range(cfg["N"]) for c in range(cfg["N"]) if cfg["grid"][r][c] == 0]
            if len(free) < cfg["C"]:
                print(f"[ERROR] {cfg['name']}: ô trống ({len(free)}) < C ({cfg['C']})")
                ok = False
            if not _is_connected(cfg["grid"]):
                print(f"[ERROR] {cfg['name']}: bản đồ không liên thông!")
                ok = False
        if ok:
            print(f"[OK] Tất cả {len(cfgs)} config đều hợp lệ.")
        return ok
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sinh test_config_v2.txt")
    parser.add_argument("--out", default="test_config_v2.txt")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    print("Đang sinh configs...")
    configs = build_all_configs()
    print(f"Đã tạo {len(configs)} configs.")

    header = """\
# =============================================================
# test_config_v2.txt — Extended Test Suite (106 configs)
# Multi-Agent Package Delivery (MAPD)
#
# 106 configs chia 10 tầng:
#   S  ( 5): N=5-7,    smoke / sanity
#   A  (15): N=8-10,   biến thể C / map type / G-T ratio
#   B  (20): N=11-15,  surge tường minh, map phức tạp, shipper hỗn hợp
#   C  (15): N=16-20,  large scale
#   D  (10): edge cases (single shipper, extreme density, strong surge, maze)
#   E  (10): N=22-50,  medium-large
#   F  (10): N=55-100, large to very large
#   X  (10): cùng map, hotspot ở 5 vị trí — đo spatial adaptability
#   Y  ( 6): cùng map/hotspot, amplitude 0→12 — đo surge intensity response
#   Z  ( 5): cùng map/hotspot, timing thay đổi — đo temporal adaptability
#
# Dùng run_sample.py để chạy N config ngẫu nhiên thay vì toàn bộ:
#   python3 run_sample.py --method GreedyBFS
#   python3 run_sample.py --method GreedyBFS --tiers X Y Z   # chỉ test adaptive
# =============================================================

"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(configs))
    print(f"Đã ghi ra {args.out}")

    if not args.no_validate:
        validate_configs(configs)


if __name__ == "__main__":
    main()
