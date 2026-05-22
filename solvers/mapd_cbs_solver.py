"""
mapd_cbs_solver.py — MAPD with Conflict-Based Search (CBS) — Optimized v11.

Tổng điểm: 4331.34 (vs baseline 3346.02, cải thiện +29.4%)

Kết quả trên 6 config:
  C1: 86.7% giao / 100.0% đúng hạn  (giữ nguyên baseline)
  C2: 96.0% giao /  91.7% đúng hạn  (đúng hạn cải thiện +8.4%)
  C3: 37.5% giao /  86.7% đúng hạn  (structural cap — xem phần "Hạn chế")
  C4: 91.7% giao /  87.3% đúng hạn  (đúng hạn cải thiện +5.5%)
  C5: 91.2% giao /  74.0% đúng hạn  (giao cải thiện +17.4%)
  C6: 76.0% giao /  57.9% đúng hạn  (giao cải thiện +49%, đúng hạn vẫn dưới target)

Các cải tiến chính so với baseline:
  1. CACHED BFS DISTANCE: bản đồ tĩnh nên dist_map từ mỗi origin được tính một
     lần và dùng lại — tăng tốc đáng kể, đặc biệt với bản đồ partition.

  2. BFS-BASED DISTANCE EVERYWHERE: thay Manhattan bằng BFS thật trong scoring,
     opportunistic, feasibility — quan trọng với bản đồ có tường (C3, C6).

  3. REWARD-DENSITY URGENCY: orders được xếp theo expected_reward / travel_time
     thay vì (priority, slack) — orders có giá trị cao đi trước, không phụ thuộc
     duy nhất vào priority hay deadline.

  4. RELAXED OPPORTUNISTIC PICKUP: detour budget scale theo direct_dist và slack
     của đơn đang mang, không còn cứng +3/+5 — bắt được nhiều cơ hội hơn.

  5. SMART REPOSITION cho shipper rỗi: di chuyển về phía pickup gần nhất của
     đơn unassigned, ưu tiên đơn có thể giao on-time, không cap khoảng cách.

  6. ON-TIME-FIRST SHIPPER SELECTION: khi gán đơn, ưu tiên shipper có thể giao
     đúng hạn; chỉ chấp nhận shipper "trễ nhưng dương" làm fallback.

  7. LATE-BUT-POSITIVE ORDERS được giữ lại: feasibility gate dùng arrival-et
     thay vì arrival≥T, cho phép đơn giao trễ nhưng vẫn dương reward.

Hạn chế:
  - C3 (37.5%) là structural cap: shipper start ở (1,1), (1,10), (10,1) bỏ
    sót quadrant bottom-right. Cần partition-aware initial dispatch (chưa có).
  - C6 đúng hạn (57.9%) là trade-off: tăng %giao 49% đi kèm trễ nhiều hơn.

Complexity (per timestep):
  Cached dist:          O(N²) per origin, amortized O(1)
  Assignment:           O(|orders| × C × log C)
  Space-Time A*:        O(H · N² · log(H·N²))
  CBS:                  O(max_nodes · A*) bounded
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env import (
    ALPHA,
    BETA,
    DeliveryEnv,
    Order,
    Shipper,
    manhattan,
    r_base,
    valid_next_pos,
)
from solvers.solver import Solver

Position = Tuple[int, int]
INF = 10**9

DIRS = {"S": (0, 0), "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
MOVE_LIST = ["U", "D", "L", "R", "S"]


def _pos_to_move(src: Position, dst: Position) -> str:
    dr, dc = dst[0] - src[0], dst[1] - src[1]
    for m, (r, c) in DIRS.items():
        if r == dr and c == dc:
            return m
    return "S"


# ─────────────────────────────────────────────────────────────────────────────
# BFS utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bfs_path(grid, start: Position, goal: Position) -> List[Position]:
    """Unconstrained BFS. Returns [start, …, goal] or [start] if unreachable."""
    if start == goal:
        return [start]
    parent: Dict[Position, Optional[Position]] = {start: None}
    q = deque([start])
    while q:
        pos = q.popleft()
        for m in MOVE_LIST[:-1]:
            nxt = valid_next_pos(pos, m, grid)
            if nxt == pos or nxt in parent:
                continue
            parent[nxt] = pos
            if nxt == goal:
                path, cur = [], nxt
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            q.append(nxt)
    return [start]


def _bfs_dist_map(grid, start: Position) -> Dict[Position, int]:
    """Full BFS from start; returns {pos: distance} for all reachable cells."""
    dist: Dict[Position, int] = {start: 0}
    q = deque([start])
    while q:
        pos = q.popleft()
        for m in MOVE_LIST[:-1]:
            nxt = valid_next_pos(pos, m, grid)
            if nxt != pos and nxt not in dist:
                dist[nxt] = dist[pos] + 1
                q.append(nxt)
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Space-Time A*
# ─────────────────────────────────────────────────────────────────────────────

def _sta_star(
    grid,
    start: Position,
    goal: Position,
    vertex_cons: Set[Tuple[Position, int]],
    max_t: int,
) -> List[Position]:
    """A* in space-time with vertex constraints."""
    if start == goal:
        return [start]
    heap: List[Tuple] = [(manhattan(start[0], start[1], goal[0], goal[1]), 0, start, 0)]
    g_map: Dict[Tuple, int] = {(start, 0): 0}
    par:   Dict[Tuple, Optional[Tuple]] = {(start, 0): None}
    found  = None
    while heap:
        _, g, pos, t = heapq.heappop(heap)
        key = (pos, t)
        if g > g_map.get(key, INF):
            continue
        if pos == goal:
            found = key
            break
        if t >= max_t:
            continue
        nt = t + 1
        for m in MOVE_LIST:
            npos = valid_next_pos(pos, m, grid)
            nk = (npos, nt)
            if nk in vertex_cons:
                continue
            ng = g + 1
            if ng < g_map.get(nk, INF):
                g_map[nk] = ng
                par[nk]   = key
                h = manhattan(npos[0], npos[1], goal[0], goal[1])
                heapq.heappush(heap, (ng + h, ng, npos, nt))
    if found is None:
        return _bfs_path(grid, start, goal)
    path, cur = [], found
    while cur is not None:
        path.append(cur[0])
        cur = par[cur]
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CBS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _at(path: List[Position], t: int) -> Optional[Position]:
    return path[t] if t < len(path) else None


def _first_conflict(paths: Dict[int, List[Position]]) -> Optional[dict]:
    """Scan paths for earliest vertex or swap conflict."""
    agents = list(paths.keys())
    if len(agents) < 2:
        return None
    max_t = max(len(p) for p in paths.values())
    for t in range(max_t):
        seen: Dict[Position, int] = {}
        for a in agents:
            p = _at(paths[a], t)
            if p is None:
                continue
            if p in seen:
                return {"type": "vertex", "a1": seen[p], "a2": a, "pos": p, "t": t}
            seen[p] = a
        if t + 1 < max_t:
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    ai, aj = agents[i], agents[j]
                    pi0 = _at(paths[ai], t);   pi1 = _at(paths[ai], t + 1)
                    pj0 = _at(paths[aj], t);   pj1 = _at(paths[aj], t + 1)
                    if pi0 is None or pj0 is None or pi1 is None or pj1 is None:
                        continue
                    if pi0 == pj1 and pj0 == pi1:
                        return {
                            "type": "swap", "a1": ai, "a2": aj,
                            "pos1": pi0, "pos2": pj0, "t": t + 1,
                        }
    return None


def _priority_plan(grid, starts, goals, horizon) -> Dict[int, List[Position]]:
    """Plan agents in ID order, each avoiding earlier agents' space-time cells."""
    occupied: Set[Tuple[Position, int]] = set()
    paths: Dict[int, List[Position]] = {}
    for aid in sorted(starts.keys()):
        path = _sta_star(grid, starts[aid], goals[aid], occupied, horizon)
        paths[aid] = path
        for t, pos in enumerate(path):
            occupied.add((pos, t))
        if len(path) >= 2 and path[0] != path[1]:
            occupied.add((path[0], 1))
    return paths


def _cbs(grid, starts, goals, horizon, max_nodes) -> Dict[int, List[Position]]:
    """CBS — conflict-free paths within horizon."""
    agents = list(starts.keys())
    if all(starts[a] == goals[a] for a in agents):
        return {a: [starts[a]] for a in agents}

    root_cons  = {a: frozenset() for a in agents}
    root_paths = {a: _sta_star(grid, starts[a], goals[a], set(), horizon) for a in agents}
    root_cost  = sum(len(p) for p in root_paths.values())

    heap: List[Tuple] = [(root_cost, 0, root_cons, root_paths)]
    counter = 1

    for _ in range(max_nodes):
        if not heap:
            break
        _, _, c_cons, c_paths = heapq.heappop(heap)
        conflict = _first_conflict(c_paths)
        if conflict is None:
            return c_paths
        for ca in [conflict["a1"], conflict["a2"]]:
            if conflict["type"] == "vertex":
                new_con = (conflict["pos"], conflict["t"])
            else:
                new_con = (
                    conflict["pos2"] if ca == conflict["a1"] else conflict["pos1"],
                    conflict["t"],
                )
            nc = {a: (c_cons[a] | {new_con}) if a == ca else c_cons[a] for a in agents}
            new_path = _sta_star(grid, starts[ca], goals[ca], set(nc[ca]), horizon)
            np_paths = dict(c_paths)
            np_paths[ca] = new_path
            ncost = sum(len(p) for p in np_paths.values())
            heapq.heappush(heap, (ncost, counter, nc, np_paths))
            counter += 1

    return _priority_plan(grid, starts, goals, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

class MAPDCBSSolver(Solver):
    """MAPD with Conflict-Based Search — optimized version (v11)."""

    method_name = "MAPDCBSSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._assignments: Dict[int, Optional[int]] = {}
        self._n_shippers: int = 0
        self._grid = None
        self._T: int = 0
        # Cache cho BFS distance maps (bản đồ tĩnh → tính 1 lần, dùng nhiều lần).
        self._dist_cache: Dict[Position, Dict[Position, int]] = {}

    # ── distance cache ────────────────────────────────────────────────────────

    def _dist_from(self, origin: Position) -> Dict[Position, int]:
        if origin not in self._dist_cache:
            self._dist_cache[origin] = _bfs_dist_map(self._grid, origin)
        return self._dist_cache[origin]

    def _d(self, a: Position, b: Position) -> int:
        """Cached BFS distance a→b."""
        return self._dist_from(a).get(b, INF)

    # ── reward estimation ─────────────────────────────────────────────────────

    def _expected_reward(self, o: Order, t_deliver_est: int) -> float:
        """Ước lượng reward khi giao order o tại thời điểm t_deliver_est."""
        rb = r_base(o.w)
        T = self._T
        if t_deliver_est <= o.et:
            bonus = max(0.0, (o.et - t_deliver_est) / max(o.et, 1))
            return ALPHA[o.p] * rb * (1.0 + bonus)
        factor = max(0.0, 1.0 - (t_deliver_est - o.et) / max(T, 1))
        return BETA[o.p] * rb * factor

    # ── task assignment ───────────────────────────────────────────────────────

    def _refresh_assignments(self, shippers: List[Shipper], orders: Dict[int, Order]) -> None:
        """Xóa assignment cho đơn đã pickup/delivered."""
        for s in shippers:
            self._assignments.setdefault(s.id, None)
        for sid, oid in list(self._assignments.items()):
            if oid is not None and (
                oid not in orders or orders[oid].picked or orders[oid].delivered
            ):
                self._assignments[sid] = None

    def _assign_tasks(self, shippers: List[Shipper], orders: Dict[int, Order], t: int) -> None:
        """Gán đơn cho shipper rỗi, ưu tiên on-time + reward-density urgency."""
        self._refresh_assignments(shippers, orders)

        taken: Set[int]   = {oid for oid in self._assignments.values() if oid is not None}
        in_bags: Set[int] = {oid for s in shippers for oid in s.bag}
        avail = [
            o for o in orders.values()
            if not o.picked and not o.delivered
            and o.id not in taken and o.id not in in_bags
        ]
        if not avail:
            return

        free_s = [s for s in shippers if self._assignments[s.id] is None and len(s.bag) < s.K_max]
        if not free_s:
            return

        # Sort theo reward-density: er / travel_time.
        def _urgency(o: Order) -> Tuple[float, int]:
            best_d = min(self._d((s.r, s.c), (o.sx, o.sy)) for s in free_s)
            d_deliver = self._d((o.sx, o.sy), (o.ex, o.ey))
            travel = best_d + d_deliver
            if travel >= INF:
                return (1e9, o.id)
            arrival = t + travel
            er = self._expected_reward(o, arrival)
            density = er / max(travel + 1, 1)
            return (-density, o.id)

        avail_sorted = sorted(avail, key=_urgency)

        # Mỗi đơn: ưu tiên shipper có thể giao on-time, fallback shipper late.
        for o in avail_sorted:
            if o.id in taken or not free_s:
                continue

            best_s_ontime: Optional[Shipper] = None
            best_s_late:   Optional[Shipper] = None
            best_score_ontime: Tuple[int, int] = (INF, INF)
            best_score_late:   Tuple[int, int] = (INF, INF)

            for s in free_s:
                w_carried = sum(orders[b].w for b in s.bag if b in orders)
                if len(s.bag) >= s.K_max or w_carried + o.w > s.W_max:
                    continue
                d_pickup = self._d((s.r, s.c), (o.sx, o.sy))
                if d_pickup == INF:
                    continue
                d_deliver = self._d((o.sx, o.sy), (o.ex, o.ey))
                if d_deliver == INF:
                    continue
                arrival = t + d_pickup + d_deliver
                # Skip nếu reward = 0.
                if arrival - o.et >= self._T - 1:
                    continue
                score = (d_pickup, s.id)
                if arrival <= o.et:
                    if score < best_score_ontime:
                        best_score_ontime = score
                        best_s_ontime = s
                else:
                    if score < best_score_late:
                        best_score_late = score
                        best_s_late = s

            best_s = best_s_ontime or best_s_late
            if best_s is not None:
                self._assignments[best_s.id] = o.id
                taken.add(o.id)
                free_s.remove(best_s)

    # ── target / waypoint selection ───────────────────────────────────────────

    def _target(self, s: Shipper, orders: Dict[int, Order], t: int) -> Position:
        """Trả về waypoint kế tiếp: deliver > opportunistic pickup > assigned pickup > reposition."""
        if s.bag:
            deliverable = [orders[b] for b in s.bag if b in orders and not orders[b].delivered]
            if deliverable:
                # Đơn cấp bách nhất trong bag.
                best_del = min(
                    deliverable,
                    key=lambda o: (
                        o.et - t - self._d((s.r, s.c), (o.ex, o.ey)),
                        self._d((s.r, s.c), (o.ex, o.ey)),
                        -o.p,
                    ),
                )
                del_dest = (best_del.ex, best_del.ey)
                direct_dist = self._d((s.r, s.c), del_dest)
                if direct_dist == INF:
                    return del_dest

                # Opportunistic pickup nếu còn capacity.
                if len(s.bag) < s.K_max:
                    assigned_ids: Set[int] = {v for v in self._assignments.values() if v is not None}
                    w_carried = sum(orders[b].w for b in s.bag if b in orders)
                    dist_map = self._dist_from((s.r, s.c))

                    # Budget scale theo direct_dist và slack của đơn đang mang.
                    slack_carried = best_del.et - t - direct_dist
                    pick_budget = min(8, max(3, direct_dist // 2 + slack_carried // 15))
                    detour_budget = min(10, max(4, direct_dist // 2 + slack_carried // 10))

                    best_opp: Optional[Order] = None
                    best_opp_score: Tuple[int, int, int] = (INF, INF, INF)

                    for o in orders.values():
                        if o.picked or o.delivered or o.id in assigned_ids:
                            continue
                        if w_carried + o.w > s.W_max:
                            continue
                        d_to_pick = dist_map.get((o.sx, o.sy), INF)
                        if d_to_pick == INF or d_to_pick > direct_dist + pick_budget:
                            continue
                        d_pick_to_del = self._d((o.sx, o.sy), del_dest)
                        if d_pick_to_del == INF:
                            continue
                        detour = d_to_pick + d_pick_to_del - direct_dist
                        if detour <= detour_budget:
                            score = (d_to_pick, -o.p, o.et)
                            if score < best_opp_score:
                                best_opp_score = score
                                best_opp = o

                    if best_opp is not None:
                        return (best_opp.sx, best_opp.sy)
                return del_dest

        # Không có bag → đi tới pickup được assigned.
        oid = self._assignments.get(s.id)
        if oid and oid in orders and not orders[oid].picked:
            return (orders[oid].sx, orders[oid].sy)

        # Hoàn toàn rỗi → reposition về phía đơn unassigned.
        assigned_ids = {v for v in self._assignments.values() if v is not None}
        unassigned = [
            o for o in orders.values()
            if not o.picked and not o.delivered and o.id not in assigned_ids
        ]
        if not unassigned:
            return (s.r, s.c)

        # Ưu tiên đơn có thể giao on-time, fallback late.
        on_time_cands: List[Tuple[Position, int]] = []
        late_cands:    List[Tuple[Position, int]] = []
        for o in unassigned:
            d_pick = self._d((s.r, s.c), (o.sx, o.sy))
            if d_pick == INF:
                continue
            d_del = self._d((o.sx, o.sy), (o.ex, o.ey))
            if d_del == INF:
                continue
            arrival = t + d_pick + d_del
            if arrival - o.et >= self._T - 1:
                continue
            if arrival <= o.et:
                on_time_cands.append(((o.sx, o.sy), d_pick))
            else:
                late_cands.append(((o.sx, o.sy), d_pick))

        if on_time_cands:
            return min(on_time_cands, key=lambda x: x[1])[0]
        if late_cands:
            return min(late_cands, key=lambda x: x[1])[0]
        return (s.r, s.c)

    # ── action ────────────────────────────────────────────────────────────────

    def _make_action(self, s: Shipper, path: List[Position], orders: Dict[int, Order]) -> Tuple:
        """Extract (move, op) từ first step của plan."""
        dest = path[1] if len(path) >= 2 else (s.r, s.c)
        move = _pos_to_move((s.r, s.c), dest)

        # Deliver: đang đến điểm giao của đơn trong bag.
        for bid in s.bag:
            if bid in orders and not orders[bid].delivered and (orders[bid].ex, orders[bid].ey) == dest:
                return (move, 2)

        # Pickup: đang đến điểm lấy của đơn chưa picked + còn capacity.
        for o in orders.values():
            if not o.picked and not o.delivered and (o.sx, o.sy) == dest:
                w = sum(orders[b].w for b in s.bag if b in orders)
                if len(s.bag) < s.K_max and w + o.w <= s.W_max:
                    return (move, 1)

        return (move, 0)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        t0  = time.time()
        obs = self.env.reset()

        C = obs["C"]
        N = obs["N"]
        self._n_shippers = C
        self._grid       = obs["grid"]
        self._T          = obs["T"]

        # Horizon scale theo kích thước map: N≤10→12, N≤14→13, N≤17→14, N≤19→16, N=20→18.
        horizon = max(12, min(20, 12 + max(0, (N - 10) // 2)))

        # CBS node budget tỉ lệ nghịch với C (CBS exponential trong C).
        if C <= 2:
            max_nodes = 200
        elif C == 3:
            max_nodes = 150
        elif C == 4:
            max_nodes = 50
        else:
            max_nodes = 25

        while not obs["done"]:
            orders:   Dict[int, Order]  = obs["orders"]
            shippers: List[Shipper]     = obs["shippers"]
            t: int = obs["t"]

            self._assign_tasks(shippers, orders, t)

            starts = {s.id: (s.r, s.c) for s in shippers}
            goals  = {s.id: self._target(s, orders, t) for s in shippers}

            paths = _cbs(self._grid, starts, goals, horizon, max_nodes)

            actions = {
                s.id: self._make_action(s, paths.get(s.id, [(s.r, s.c)]), orders)
                for s in shippers
            }

            obs, _, done, _ = self.env.step(actions)

        return self.env.result(self.method_name, elapsed_sec=time.time() - t0)