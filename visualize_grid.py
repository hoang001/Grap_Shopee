"""
visualize_grid.py -- ASCII visualization for MAPD grid topology analysis.

Features:
  - ASCII grid render with wall / free / passage markers
  - Per-row and per-column passage width annotation
  - Bottleneck highlight (rows/cols below threshold)
  - Topology summary: internal walls, free ratio, passage stats, bottleneck score
  - Optional congestion heatmap overlay (pass a dict {(r,c): int})
  - Optional shipper/order overlay

Usage (standalone):
    python visualize_grid.py --config test_config.txt --name C3
    python visualize_grid.py --config test_stress.txt --name S2_TwoRoom
    python visualize_grid.py --config test_stress.txt --all-configs --no-colour

Programmatic use from solver:
    from visualize_grid import render_grid, render_congestion
    render_grid(grid, N)
    render_congestion(grid, N, solver._congestion)
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]
Position = Tuple[int, int]

# ----------------------- ANSI colours ----------------------- #
_USE_COLOUR = True   # set False for plain-text output (CI / log files)

def _c(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    return "\033[" + code + "m" + text + "\033[0m"

def WALL(s: str)  -> str: return _c("90", s)           # dark grey
def FREE(s: str)  -> str: return _c("37", s)            # white
def BOTTLE(s: str)-> str: return _c("31;1", s)          # bright red -- bottleneck
def HOT(s: str)   -> str: return _c("33;1", s)          # yellow -- congestion medium
def VHOT(s: str)  -> str: return _c("31;1", s)          # bright red -- congestion high
def COLD(s: str)  -> str: return _c("32", s)            # green -- congestion low


# ----------------------- Topology helpers ------------------- #

def _row_free(grid: Grid, N: int, r: int) -> int:
    return sum(1 for c in range(N) if grid[r][c] == 0)

def _col_free(grid: Grid, N: int, c: int) -> int:
    return sum(1 for r in range(N) if grid[r][c] == 0)

def _min_passage(grid: Grid, N: int) -> int:
    vals = [_row_free(grid, N, r) for r in range(1, N - 1)]
    vals += [_col_free(grid, N, c) for c in range(1, N - 1)]
    return min(vals) if vals else 0

def _avg_passage(grid: Grid, N: int) -> float:
    vals = [_row_free(grid, N, r) for r in range(1, N - 1) if _row_free(grid, N, r) > 0]
    vals += [_col_free(grid, N, c) for c in range(1, N - 1) if _col_free(grid, N, c) > 0]
    return sum(vals) / max(len(vals), 1)

def _internal_walls(grid: Grid, N: int) -> int:
    return sum(1 for r in range(1, N - 1) for c in range(1, N - 1) if grid[r][c] == 1)

def _bottleneck_threshold(N: int) -> int:
    return max(3, N // 10)


# ----------------------- Core renderer ---------------------- #

def render_grid(
    grid: Grid,
    N: int,
    *,
    name: str = "",
    congestion: Optional[Dict[Position, int]] = None,
    shippers: Optional[Dict[int, Position]] = None,   # {sid: (r,c)}
    orders: Optional[Dict[int, Tuple[Position, Position]]] = None,  # {oid: (src, dst)}
    colour: bool = True,
) -> str:
    """Return a multi-line string with the ASCII grid + topology annotations."""
    global _USE_COLOUR
    _USE_COLOUR = colour

    thresh  = _bottleneck_threshold(N)
    iw      = _internal_walls(grid, N)
    total   = N * N
    free    = sum(1 for r in range(N) for c in range(N) if grid[r][c] == 0)
    mpc     = _min_passage(grid, N)
    avg_p   = _avg_passage(grid, N)
    b_score = 1.0 - mpc / max(N, 1)
    use_mh  = (iw == 0)

    row_w = [_row_free(grid, N, r) for r in range(N)]
    col_w = [_col_free(grid, N, c) for c in range(N)]

    shipper_pos: Dict[Position, int] = {}
    if shippers:
        for sid, pos in shippers.items():
            shipper_pos[pos] = sid
    order_src: Dict[Position, int] = {}
    order_dst: Dict[Position, int] = {}
    if orders:
        for oid, (src, dst) in orders.items():
            order_src[src] = oid
            order_dst[dst] = oid

    lines: List[str] = []
    sep = "+" + "-" * (N * 2 + 1) + "+"
    title = (" " + name + " ") if name else ""

    lines.append(sep + "  " + title + "N=" + str(N))
    lines.append("  thresh=" + str(thresh)
                 + "  min_passage=" + str(mpc)
                 + "  avg=" + f"{avg_p:.1f}")
    lines.append("  internal_walls=" + str(iw)
                 + "  free_ratio=" + f"{free/total:.2f}")
    lines.append("  bottleneck_score=" + f"{b_score:.3f}"
                 + "  use_manhattan=" + str(use_mh))
    lines.append(sep)

    # Column-width header
    col_parts = []
    for w in col_w:
        token = str(w) if w < 10 else "+"
        col_parts.append(BOTTLE(token) if w <= thresh else token)
    lines.append("| " + " ".join(col_parts) + " |  <- col free-cell counts")
    lines.append(sep)

    # Grid rows
    for r in range(N):
        cells: List[str] = []
        for c in range(N):
            pos = (r, c)
            if grid[r][c] == 1:
                cells.append(WALL("##"))
            elif pos in shipper_pos:
                cells.append(_c("36;1", "S" + str(shipper_pos[pos] % 10)))
            elif pos in order_src:
                cells.append(_c("35;1", "Ps"))
            elif pos in order_dst:
                cells.append(_c("35", "Pd"))
            elif congestion and pos in congestion:
                v = congestion[pos]
                ch = ".." if v == 0 else f"{min(v, 99):2d}"
                if v >= 8:
                    cells.append(VHOT(ch))
                elif v >= 3:
                    cells.append(HOT(ch))
                else:
                    cells.append(COLD(ch))
            else:
                cells.append(FREE("  "))

        rw = row_w[r]
        row_body = "".join(cells)
        if 1 <= r <= N - 2 and rw <= thresh:
            annot = BOTTLE("<- " + f"{rw:2d}")
        else:
            annot = "    " + f"{rw:2d}"
        lines.append("|" + row_body + "| " + annot)

    lines.append(sep)

    # Bottleneck summary
    bottle_rows = [r for r in range(1, N - 1) if row_w[r] <= thresh]
    bottle_cols = [c for c in range(1, N - 1) if col_w[c] <= thresh]
    if bottle_rows or bottle_cols:
        lines.append("  Bottleneck rows: " + str(bottle_rows))
        lines.append("  Bottleneck cols: " + str(bottle_cols))
    else:
        lines.append("  No bottleneck rows/cols detected.")

    return "\n".join(lines)


def render_congestion(
    grid: Grid,
    N: int,
    congestion: Dict[Position, int],
    *,
    name: str = "",
    colour: bool = True,
) -> str:
    """Render grid with congestion heatmap overlay."""
    return render_grid(grid, N, name=name, congestion=congestion, colour=colour)


# ----------------------- Config parser ---------------------- #

def _parse_configs(path: str) -> Dict[str, Tuple[int, Grid]]:
    """Parse test_config.txt / test_stress.txt -> {name: (N, grid)}."""
    configs: Dict[str, Tuple[int, Grid]] = {}
    with open(path, encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\[CONFIG\]", content)
    for block in blocks[1:]:
        name_m = re.search(r"name\s*=\s*(\S+)", block)
        n_m    = re.search(r"N\s*=\s*(\d+)", block)
        map_m  = re.search(r"\[MAP\](.*?)\[END\]", block, re.DOTALL)
        if not (name_m and n_m and map_m):
            continue
        cname   = name_m.group(1)
        N       = int(n_m.group(1))
        map_str = map_m.group(1).strip()
        grid: Grid = []
        for line in map_str.splitlines():
            line = line.strip()
            if not line:
                continue
            row = [int(x) for x in line.split()]
            if len(row) == N:
                grid.append(row)
        if len(grid) == N:
            configs[cname] = (N, grid)
    return configs


# ----------------------- CLI entry point -------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MAPD grid topology")
    parser.add_argument("--config", default="test_config.txt",
                        help="Config file (default: test_config.txt)")
    parser.add_argument("--name", default="",
                        help="Config name to display (empty = first config)")
    parser.add_argument("--no-colour", action="store_true",
                        help="Disable ANSI colour output")
    parser.add_argument("--all-configs", action="store_true",
                        help="Render all configs in the file")
    args = parser.parse_args()

    colour = not args.no_colour
    configs = _parse_configs(args.config)

    if not configs:
        print("No configs found in " + args.config)
        return

    if args.name:
        if args.name not in configs:
            print("Config '" + args.name + "' not found. Available: " + str(list(configs)))
            return
        targets = {args.name: configs[args.name]}
    elif args.all_configs:
        targets = configs
    else:
        first = next(iter(configs))
        targets = {first: configs[first]}
        print("(Showing first config '" + first + "'. Use --name or --all-configs to select.)\n")

    for cname, (N, grid) in targets.items():
        print(render_grid(grid, N, name=cname, colour=colour))
        print()


if __name__ == "__main__":
    main()
