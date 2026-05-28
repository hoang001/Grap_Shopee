from __future__ import annotations

import heapq
import time
from collections import deque, OrderedDict
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

SAFE_BUFFER = 1
URGENCY_COEFF = 2.9
AGE_COEFF = 3.0

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

LRU_CACHE_MAX = 100_000


class GreedyBFS(Solver):
    """
    Hệ thống điều phối online phân cấp cho bài toán MAPD trên grid có vật cản.

    Pipeline 5 tầng (thực hiện mỗi bước trong _decide_actions):
      0.   deliver_first / deliver_if_nearby  — giao hàng cơ hội trước khi nhận mới
      1.   Chọn pickup + cam kết              — điểm urgency cao nhất, cam kết N bước (N>20)
      1.5  Pickup cơ hội gần                  — ellipse filter dọc tuyến đường đến goal
      2.   Giao hàng                           — EDF-ordered, có pickup cơ hội en-route (2.5)
      3.   Vị trí chờ thích nghi              — tâm (bottleneck) hoặc phân tán (mở)
      +    Phục hồi bế tắc                    — phát hiện kẹt + bộ nhớ tắc nghẽn nhẹ

    Tìm đường:
      N ≤ 20: BFS + cache đầy đủ (O(1) sau warmup)
      N > 20: A* + heuristic Manhattan + LRU cache (mở rộng đến N=100+)
      Grid hoàn toàn mở (internal_walls=0): shortcut Manhattan thay A*

    Đặc trưng topo (tính một lần lúc init, dùng để dẫn xuất tham số N>20):
      _free_ratio         tỉ lệ ô đi được
      _min_passage_count  hành lang hẹp nhất (bottleneck detector)
      _bottleneck_score   ∈ [0,1], gần 1 khi hành lang rất hẹp
      _avg_passage_width  độ rộng hành lang trung bình
      _internal_walls     tường nội bộ (= 0 ↔ grid hoàn toàn mở)

    Chiến lược N ≤ 20: giá trị cố định theo từng N — đã kiểm thử, không thay đổi.
    Chiến lược N > 20: dẫn xuất từ đặc trưng topo — tổng quát hóa cho grid chưa biết.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._T: int = env.T
        self._N: int = env.N
        self._C: int = env.C
        self._free_cells: List[Position] = [
            (r, c) for r in range(self._N) for c in range(self._N)
            if is_valid_cell((r, c), self.grid)
        ]
        self._grid_center: Optional[Position] = self._find_grid_center()
        self._spread_positions: List[Position] = self._compute_spread_positions(self._C)


        self._free_ratio: float = len(self._free_cells) / max(self._N ** 2, 1)


        self._min_passage_count: int = self._compute_min_passage_count()


        self._internal_walls: int = sum(
            1 for r in range(1, self._N - 1) for c in range(1, self._N - 1)
            if self.grid[r][c] == 1
        )
        self._avg_passage_width: float = self._compute_avg_passage_width()
        self._bottleneck_score: float = 1.0 - self._min_passage_count / max(self._N, 1)


        _narrow_thresh = max(2, self._N // 20)
        self._barrier_row_count: int = sum(
            1 for r in range(1, self._N - 1)
            if sum(1 for c in range(self._N) if self.grid[r][c] == 0) <= _narrow_thresh
        )


        self._path_cache: Dict[Tuple[Position, Position], Tuple[int, Move]] = {}
        if self._N > 20:
            self._lru: Optional[OrderedDict] = OrderedDict()
            self._lru_max: int = min(max(100_000, self._N * self._N * 5), 500_000)
        else:
            self._lru = None
            self._lru_max = 0


        self._strat = self._derive_strategy()
        self._safe_buffer: int = self._strat["safe_buffer"]


        self._committed: Dict[int, Optional[int]] = {}
        self._committed_since: Dict[int, int] = {}


        self._prev_positions: Dict[int, Position] = {}


        self._congestion: Dict[Position, int] = {}
        self._congestion_t: int = 0
        self._congestion_decay_interval: int = 20


        self._m: Dict[str, int] = dict(
            stuck=0, escaped=0, edf_full=0, ub_skip=0, opp=0, idle=0, zone_rep=0
        )


    def _derive_strategy(self) -> dict:

        N, R = self._N, self._free_ratio

        if N <= 20:

            deliver_first = N <= 10

            if N in (12, 15) or N >= 20:
                delivery_mode = "nearest"
            elif N == 18:
                delivery_mode = "blend"
            else:
                delivery_mode = "urgency"

            if N >= 20:
                n_urgency = 3.5
            else:
                n_urgency = URGENCY_COEFF

            if N == 12:
                age_coeff, age_cap = AGE_COEFF, AGE_COEFF
            elif N >= 20:
                age_coeff, age_cap = 0.25, 0.25
            else:
                age_coeff, age_cap = 0.0, 0.0

            if N in (7, 12):
                idle_mode = "center"
            elif N == 10:
                idle_mode = "order_seeking"
            elif N == 15:
                idle_mode = "stay"
            else:
                idle_mode = "spread"

            safe_buffer = 0 if N >= 20 else SAFE_BUFFER
            deliver_nearby = 2


            deliver_nearby_gate = N <= 15


            use_edf = False
            use_manhattan_dist = False
            replan_interval = 1
            opportunistic_radius = 0
            max_detour = 0

        else:


            is_dense = R < 0.60


            narrow_threshold = max(3, self._N // 10)
            is_single_bottleneck = self._min_passage_count <= narrow_threshold

            deliver_first = False
            delivery_mode = "nearest"
            age_coeff = age_cap = 0.25


            is_multi_barrier = self._barrier_row_count >= 3


            is_structured_block = (is_dense and not is_single_bottleneck
                                   and self._min_passage_count >= self._N // 6
                                   and self._N >= 50)
            effective_dense = is_dense and not is_structured_block
            n_urgency = 1.5 if is_dense else URGENCY_COEFF


            is_single_cell_bottleneck = is_single_bottleneck and self._min_passage_count <= 1
            idle_mode = "center" if (effective_dense or is_single_cell_bottleneck) and not is_multi_barrier else "spread"
            safe_buffer = 0
            deliver_nearby = 2
            deliver_nearby_gate = is_dense or is_single_bottleneck
            use_edf = True

            use_manhattan_dist = (self._internal_walls == 0)

            replan_interval = 20 if use_manhattan_dist else (6 if is_multi_barrier else 12)
            opportunistic_radius = 8 if use_manhattan_dist else 5
            max_detour = 6

        return {
            "deliver_first": deliver_first,
            "delivery_mode": delivery_mode,
            "deadline_margin": 0.05,
            "urgency_coeff": n_urgency,
            "age_coeff": age_coeff,
            "age_cap": age_cap,
            "idle_mode": idle_mode,
            "safe_buffer": safe_buffer,
            "deliver_nearby": deliver_nearby,
            "deliver_nearby_gate": deliver_nearby_gate,
            "use_edf": use_edf,
            "use_manhattan_dist": use_manhattan_dist,
            "replan_interval": replan_interval,
            "opportunistic_radius": opportunistic_radius,
            "max_detour": max_detour,
        }

    def _find_grid_center(self) -> Optional[Position]:

        cr, cc = self._N // 2, self._N // 2
        best, best_dist = None, INF
        for r in range(self._N):
            for c in range(self._N):
                if is_valid_cell((r, c), self.grid):
                    d = abs(r - cr) + abs(c - cc)
                    if d < best_dist:
                        best_dist = d
                        best = (r, c)
        return best

    def _compute_spread_positions(self, n: int) -> List[Position]:

        if not self._free_cells or n <= 0:
            return []
        N = self._N
        inner_anchors = [
            (N // 4, N // 4), (N // 4, 3 * N // 4),
            (3 * N // 4, N // 4), (3 * N // 4, 3 * N // 4),
            (N // 2, N // 2),
        ]
        selected: List[Position] = []
        for anchor in inner_anchors:
            if len(selected) >= n:
                break
            cell = min(
                self._free_cells,
                key=lambda x, a=anchor: abs(x[0] - a[0]) + abs(x[1] - a[1])
            )
            if cell not in selected:
                selected.append(cell)
        while len(selected) < n:
            remaining = [c for c in self._free_cells if c not in selected]
            if not remaining:
                break
            cell = max(
                remaining,
                key=lambda x: min(abs(x[0] - y[0]) + abs(x[1] - y[1]) for y in selected)
            )
            selected.append(cell)
        return selected[:n]

    def _compute_min_passage_count(self) -> int:

        best = self._N * self._N
        for r in range(self._N):
            free = sum(1 for c in range(self._N) if self.grid[r][c] == 0)
            if free > 0:
                best = min(best, free)
        for c in range(self._N):
            free = sum(1 for r in range(self._N) if self.grid[r][c] == 0)
            if free > 0:
                best = min(best, free)
        return best

    def _compute_avg_passage_width(self) -> float:

        widths: List[int] = []
        for r in range(1, self._N - 1):
            w = sum(1 for c in range(self._N) if self.grid[r][c] == 0)
            if w > 0:
                widths.append(w)
        for c in range(1, self._N - 1):
            w = sum(1 for r in range(self._N) if self.grid[r][c] == 0)
            if w > 0:
                widths.append(w)
        return sum(widths) / max(len(widths), 1)


    def _bfs_compute(self, start: Position, goal: Position) -> Tuple[int, Move]:

        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF, "S"

        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        queue: deque[Position] = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for move in MOVES:
                nxt = valid_next_pos(cur, move, self.grid)
                if nxt != cur and nxt not in parent:
                    parent[nxt] = (cur, move)
                    queue.append(nxt)

        if goal not in parent:
            return INF, "S"

        dist, first_move, node = 0, "S", goal
        while node != start:
            prev, move = parent[node]
            if prev is None:
                break
            dist += 1
            if prev == start:
                first_move = move
            node = prev

        return dist, first_move

    def _astar(self, start: Position, goal: Position) -> Tuple[int, Move]:

        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF, "S"

        def h(pos: Position) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


        heap = [(h(start), 0, start[0], start[1], "S")]
        g_best: Dict[Position, int] = {start: 0}

        while heap:
            _, g, r, c, first = heapq.heappop(heap)
            cur = (r, c)
            if g > g_best.get(cur, INF):
                continue
            if cur == goal:
                return g, first
            for move in MOVES:
                nxt = valid_next_pos(cur, move, self.grid)
                if nxt == cur:
                    continue
                new_g = g + 1
                if new_g < g_best.get(nxt, INF):
                    g_best[nxt] = new_g
                    new_first = move if cur == start else first
                    heapq.heappush(heap, (new_g + h(nxt), new_g, nxt[0], nxt[1], new_first))

        return INF, "S"

    def _bfs(self, start: Position, goal: Position) -> Tuple[int, Move]:

        if start == goal:
            return 0, "S"
        key = (start, goal)

        if self._N <= 20:
            if key not in self._path_cache:
                self._path_cache[key] = self._bfs_compute(start, goal)
            return self._path_cache[key]
        else:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
            result = self._astar(start, goal)
            self._lru[key] = result
            if len(self._lru) > self._lru_max:
                self._lru.popitem(last=False)
            return result

    def _dist(self, a: Position, b: Position) -> int:

        if self._strat["use_manhattan_dist"]:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        return self._bfs(a, b)[0]

    def _move_to(self, here: Position, goal: Position) -> Tuple[Move, Position]:

        if here == goal:
            return "S", here
        if self._strat["use_manhattan_dist"]:
            dr = goal[0] - here[0]
            dc = goal[1] - here[1]

            move = ("D" if dr > 0 else "U") if abs(dr) >= abs(dc) else ("R" if dc > 0 else "L")
            return move, (here[0] + (1 if move == "D" else -1 if move == "U" else 0),
                          here[1] + (1 if move == "R" else -1 if move == "L" else 0))
        move = self._bfs(here, goal)[1]
        return move, valid_next_pos(here, move, self.grid)


    def _commitment_valid(self, shipper: Shipper, orders: Dict[int, Order],
                          reserved: set, t: int) -> bool:

        sid = shipper.id
        oid = self._committed.get(sid)
        if oid is None or oid in reserved:
            return False
        if oid not in orders:
            return False
        o = orders[oid]
        if o.picked or o.delivered:
            return False

        return (t - self._committed_since.get(sid, t)) < self._strat["replan_interval"]

    def _set_commitment(self, sid: int, oid: int, t: int) -> None:

        self._committed[sid] = oid
        self._committed_since[sid] = t

    def _opportunistic_nearby(self, shipper: Shipper, orders: Dict[int, Order],
                               available: List[Order], reserved: set,
                               committed_goal: Position, t: int,
                               max_detour_override: Optional[int] = None) -> Optional[Order]:

        radius = self._strat["opportunistic_radius"]
        if radius <= 0:
            return None

        max_detour = max_detour_override if max_detour_override is not None else self._strat["max_detour"]
        r0, c0 = shipper.position
        direct = self._dist(shipper.position, committed_goal)
        if direct == 0:
            return None

        best, best_score = None, -float("inf")
        for o in available:
            if o.id in reserved:
                continue
            if abs(r0 - o.sx) + abs(c0 - o.sy) > radius:
                continue
            if not shipper.can_carry(o, orders):
                continue
            if t + abs(r0 - o.sx) + abs(c0 - o.sy) + abs(o.sx - o.ex) + abs(o.sy - o.ey) >= self._T:
                continue


            d_via = (self._dist(shipper.position, (o.sx, o.sy))
                     + self._dist((o.sx, o.sy), committed_goal))
            if d_via - direct > max_detour:
                continue
            if not self._is_safe_to_pickup(shipper, (o.sx, o.sy), orders, t):
                continue
            s = self._score_pickup(shipper.position, o, t)
            if s > best_score:
                best_score = s
                best = o
        return best

    def _opportunistic_corridor(self, shipper: Shipper, orders: Dict[int, Order],
                                available: List[Order], reserved: set,
                                goal: Position, t: int,
                                max_detour: int = 3) -> Optional[Order]:

        if self._strat["opportunistic_radius"] <= 0:
            return None
        r0, c0 = shipper.position
        gr, gc = goal
        direct_m = abs(r0 - gr) + abs(c0 - gc)
        if direct_m == 0:
            return None
        direct = self._dist(shipper.position, goal)
        if direct >= INF:
            return None


        if direct_m > 0 and direct >= 2 * direct_m:
            m_filter_extra = max(1, int(max_detour * direct_m / direct))
        else:
            m_filter_extra = max_detour
        best, best_score = None, -float("inf")
        for o in available:
            if o.id in reserved:
                continue

            m_via = abs(r0 - o.sx) + abs(c0 - o.sy) + abs(o.sx - gr) + abs(o.sy - gc)
            if m_via > direct_m + m_filter_extra:
                continue
            if not shipper.can_carry(o, orders):
                continue
            if t + abs(r0 - o.sx) + abs(c0 - o.sy) + abs(o.sx - o.ex) + abs(o.sy - o.ey) >= self._T:
                continue

            d_via = (self._dist(shipper.position, (o.sx, o.sy))
                     + self._dist((o.sx, o.sy), goal))
            if d_via - direct > max_detour:
                continue
            if not self._is_safe_to_pickup(shipper, (o.sx, o.sy), orders, t):
                continue
            s = self._score_pickup(shipper.position, o, t)
            if s > best_score:
                best_score = s
                best = o
        return best


    def _is_safe_to_pickup(self, shipper: Shipper, pickup_pos: Position,
                           orders: Dict[int, Order], t: int) -> bool:

        if not shipper.bag:
            return True


        sr, sc = shipper.position
        saveable = [
            orders[oid] for oid in shipper.bag
            if oid in orders
            and not orders[oid].delivered
            and t + abs(sr - orders[oid].ex) + abs(sc - orders[oid].ey) <= orders[oid].et
            and t + self._dist(shipper.position, (orders[oid].ex, orders[oid].ey)) <= orders[oid].et
        ]
        if not saveable:
            return True
        self._m["edf_full"] += 1

        d1 = self._dist(shipper.position, pickup_pos)
        t_at_pickup = t + d1
        if self._strat["use_edf"]:


            pos = pickup_pos
            cur_t = t_at_pickup
            for o in sorted(saveable, key=lambda x: x.et):
                d = self._dist(pos, (o.ex, o.ey))
                cur_t += d
                if cur_t > o.et:
                    return False
                pos = (o.ex, o.ey)
            return True
        else:

            for seq in permutations(saveable):
                pos = pickup_pos
                cur_t = t_at_pickup
                ok = True
                for o in seq:
                    d = self._dist(pos, (o.ex, o.ey))
                    cur_t += d
                    if cur_t > o.et - self._safe_buffer:
                        ok = False
                        break
                    pos = (o.ex, o.ey)
                if ok:
                    return True
            return False


    def _score_pickup(self, pos: Position, order: Order, t: int) -> float:

        d1 = self._dist(pos, (order.sx, order.sy))
        d2 = self._dist((order.sx, order.sy), (order.ex, order.ey))
        if d1 >= INF or d2 >= INF:
            return -float("inf")
        if t + d1 + d2 >= self._T:
            return -float("inf")
        est = delivery_reward(order, t + d1 + d2, self._T)
        if est <= 0:
            return -float("inf")
        slack = order.et - t - d1 - d2
        if slack >= 0:
            urgency_factor = 1.0 + self._strat["n_urgency"] / max(slack, 1)
        else:
            urgency_factor = 1.0
        score = est / (d1 + d2 + 1) * urgency_factor
        age_coeff = self._strat["age_coeff"]
        if age_coeff > 0:
            age_steps = t - order.appear_t
            age_factor = 1.0 + min(age_steps * age_coeff / max(self._T, 1),
                                   self._strat["age_cap"])
            score *= age_factor
        return score

    def _pickup_candidate_pool(self, shipper: Shipper, available: List[Order],
                               reserved: set, t: int) -> List[Order]:

        r0, c0 = shipper.position
        if self._N < 40:
            return sorted(available, key=lambda o: abs(r0 - o.sx) + abs(c0 - o.sy))

        cheap: List[Tuple[int, int, int, int, Order]] = []
        for o in available:
            if o.id in reserved:
                continue
            d1_m = abs(r0 - o.sx) + abs(c0 - o.sy)
            d2_m = abs(o.sx - o.ex) + abs(o.sy - o.ey)
            if t + d1_m + d2_m >= self._T:
                continue
            cheap.append((d1_m, o.et, -o.p, o.id, o))

        if len(cheap) <= 96:
            return [item[-1] for item in sorted(cheap)]

        nearby_limit = 64 if self._N <= 50 else 80
        urgent_limit = 24 if self._N <= 50 else 32
        nearby = sorted(cheap)[:nearby_limit]
        urgent = sorted(
            cheap,
            key=lambda item: (
                item[1] - t,
                item[2],
                item[0],
                item[3],
            ),
        )[:urgent_limit]

        seen = set()
        merged: List[Tuple[int, int, int, int, Order]] = []
        for item in nearby + urgent:
            oid = item[4].id
            if oid in seen:
                continue
            seen.add(oid)
            merged.append(item)
        return [item[-1] for item in sorted(merged)]


    def _best_pickup(self, shipper: Shipper, orders: Dict[int, Order],
                     available: List[Order], reserved: set, t: int,
                     reserved_zones: Optional[set] = None) -> Optional[Order]:

        best, best_score = None, -float("inf")
        r0, c0 = shipper.position


        for o in self._pickup_candidate_pool(shipper, available, reserved, t):
            if o.id in reserved:
                continue
            if not shipper.can_carry(o, orders):
                continue

            d1_m = abs(r0 - o.sx) + abs(c0 - o.sy)
            d2_m = abs(o.sx - o.ex) + abs(o.sy - o.ey)
            if t + d1_m + d2_m >= self._T:
                continue


            if best_score > -float("inf"):
                est_ub = delivery_reward(o, t + d1_m + d2_m, self._T)
                if est_ub > 0:
                    score_ub = (est_ub / (d1_m + d2_m + 1)
                                * (1.0 + self._strat["n_urgency"])
                                * (1.0 + self._strat["age_cap"]))
                    if score_ub <= best_score:
                        self._m["ub_skip"] += 1
                        continue

            if not self._is_safe_to_pickup(shipper, (o.sx, o.sy), orders, t):
                continue
            s = self._score_pickup(shipper.position, o, t)


            if reserved_zones and (o.sx, o.sy) in reserved_zones:
                s *= 0.80
                self._m["zone_rep"] += 1
            if s > best_score:
                best_score = s
                best = o
        return best

    def _best_delivery_target(self, shipper: Shipper, orders: Dict[int, Order],
                               t: int) -> Optional[Order]:

        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried:
            return None


        by_dest: Dict[Position, List[Order]] = {}
        for o in carried:
            by_dest.setdefault((o.ex, o.ey), []).append(o)

        mode = self._strat["delivery_mode"]
        d_blend = self._strat["d_blend"]
        best, best_key = None, None

        for dest, group in by_dest.items():
            d = self._dist(shipper.position, dest)
            if d >= INF:
                continue
            most_urgent = min(group, key=lambda o: o.et)
            slack = most_urgent.et - t - d
            flag = 0 if slack >= 0 else 1

            if mode == "nearest":
                key = (flag, d, slack)
            elif mode == "blend":
                key = (flag, slack + d_blend * d)
            else:
                key = (flag, slack, d)

            if best_key is None or key < best_key:
                best_key = key
                best = most_urgent

        return best

    def _idle_target(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Position]:

        mode = self._strat["idle_mode"]

        if mode == "center":

            return self._grid_center

        if mode == "order_seeking":

            best, best_dist = None, INF
            for o in orders.values():
                if o.picked or o.delivered:
                    continue
                d = self._dist(shipper.position, (o.sx, o.sy))
                if d < best_dist and d < INF:
                    best_dist = d
                    best = (o.sx, o.sy)
            return best

        if mode == "spread":

            idx = shipper.id % len(self._spread_positions) if self._spread_positions else 0
            return self._spread_positions[idx] if self._spread_positions else self._grid_center

        return None


    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        t: int = obs["t"]
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]


        all_positions: set = {s.position for s in shippers}


        available: List[Order] = [o for o in orders.values() if not o.picked and not o.delivered]

        actions: Dict[int, Action] = {}
        reserved: set = set()


        reserved_zones: Optional[set] = None
        if len(shippers) > 5:
            reserved_zones = set()
            for _s in shippers:
                coid = self._committed.get(_s.id)
                if coid and coid in orders:
                    _o = orders[coid]
                    if not _o.picked and not _o.delivered:
                        reserved_zones.add((_o.sx, _o.sy))


        for shipper in sorted(shippers, key=lambda s: (len(s.bag), s.id)):
            pos = shipper.position


            if self._strat["deliver_first"] and shipper.bag:
                delivery = self._best_delivery_target(shipper, orders, t)
                if delivery is not None:
                    dest = (delivery.ex, delivery.ey)
                    move, nxt = self._move_to(pos, dest)
                    actions[shipper.id] = (move, 2 if nxt == dest else 0)
                    continue


            gate = self._strat["deliver_nearby_gate"]
            if shipper.bag and (not gate or len(shipper.bag) >= shipper.K_max):
                delivery = self._best_delivery_target(shipper, orders, t)
                if delivery is not None:
                    dest = (delivery.ex, delivery.ey)
                    if self._dist(pos, dest) <= self._strat["deliver_nearby"]:
                        move, nxt = self._move_to(pos, dest)
                        actions[shipper.id] = (move, 2 if nxt == dest else 0)
                        continue


            use_commitment = self._strat["replan_interval"] > 1
            if use_commitment and self._commitment_valid(shipper, orders, reserved, t):

                oid = self._committed[shipper.id]
                o = orders[oid]
                reserved.add(oid)
            else:

                o = self._best_pickup(shipper, orders, available, reserved, t, reserved_zones)
                if o is not None:
                    reserved.add(o.id)
                    if reserved_zones is not None:
                        reserved_zones.add((o.sx, o.sy))
                    if use_commitment:
                        self._set_commitment(shipper.id, o.id, t)

            if o is not None:
                goal = (o.sx, o.sy)


                opp = self._opportunistic_nearby(shipper, orders, available, reserved, goal, t)
                if opp is not None:
                    reserved.add(opp.id)
                    if use_commitment:
                        self._set_commitment(shipper.id, opp.id, t)
                    goal = (opp.sx, opp.sy)
                    self._m["opp"] += 1
                move, nxt = self._move_to(pos, goal)
                actions[shipper.id] = (move, 1 if nxt == goal else 0)
                continue


            if shipper.bag:
                delivery = self._best_delivery_target(shipper, orders, t)
                if delivery is not None:
                    dest = (delivery.ex, delivery.ey)


                    opp = self._opportunistic_corridor(shipper, orders, available, reserved, dest, t,
                                                       max_detour=2)
                    if opp is not None:
                        reserved.add(opp.id)
                        if use_commitment:
                            self._set_commitment(shipper.id, opp.id, t)
                        goal = (opp.sx, opp.sy)
                        self._m["opp"] += 1
                        move, nxt = self._move_to(pos, goal)
                        actions[shipper.id] = (move, 1 if nxt == goal else 0)
                        continue
                    move, nxt = self._move_to(pos, dest)
                    actions[shipper.id] = (move, 2 if nxt == dest else 0)
                    continue


            self._m["idle"] += 1
            idle = self._idle_target(shipper, orders)
            if idle is not None and idle != pos:
                move, _ = self._move_to(pos, idle)
                actions[shipper.id] = (move, 0)
                continue

            actions[shipper.id] = ("S", 0)


        if t - self._congestion_t >= self._congestion_decay_interval and self._congestion:
            self._congestion = {p: v >> 1 for p, v in self._congestion.items() if v > 1}
            self._congestion_t = t

        for shipper in shippers:
            sid = shipper.id
            if sid not in actions:
                continue
            old_move, _ = actions[sid]
            if old_move == "S":
                continue
            if shipper.position != self._prev_positions.get(sid):
                continue
            desired_nxt = valid_next_pos(shipper.position, old_move, self.grid)
            if desired_nxt != shipper.position and desired_nxt not in all_positions:
                continue


            self._congestion[shipper.position] = min(
                self._congestion.get(shipper.position, 0) + 1, 16
            )
            self._m["stuck"] += 1


            best_alt: Optional[Move] = None
            best_cong = float("inf")
            for alt in MOVES:
                alt_nxt = valid_next_pos(shipper.position, alt, self.grid)
                if alt_nxt != shipper.position and alt_nxt not in all_positions:
                    c = self._congestion.get(alt_nxt, 0)
                    if c < best_cong:
                        best_cong = c
                        best_alt = alt
            if best_alt is not None:
                actions[sid] = (best_alt, 0)
                self._m["escaped"] += 1


        for shipper in shippers:
            self._prev_positions[shipper.id] = shipper.position

        return actions


    def run(self) -> dict:

        start_time = time.time()
        obs = self.env.reset()
        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        result = self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
        result["metrics"] = dict(self._m)
        return result
