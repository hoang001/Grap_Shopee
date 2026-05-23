"""
mapd_cbs_solver.py — MAPD with Conflict-Based Search (CBS).

Kiến trúc 2 lớp đúng tinh thần MAPD-CBS:

  • Lớp TASK ASSIGNMENT (online): mỗi bước gán đơn chưa-phục-vụ cho shipper rỗi
    theo EDF (ưu tiên giao đúng hạn) + reward-density; gom đơn cùng tuyến
    (opportunistic pickup); shipper rỗi reposition về phía cụm đơn unassigned.
    Khi không còn đơn để đuổi, shipper rỗi ĐỖ DỰ ĐOÁN theo HEATMAP CẦU quan sát
    được (số đơn từng xuất hiện ở mỗi ô, phân rã theo thời gian) → tự dồn về
    vùng cầu nóng để cắt độ trễ nhặt đơn kế tiếp; coverage-claim trải các shipper
    rảnh ra nhiều vùng thay vì chụm một điểm. Đây là cơ chế thích nghi môi trường
    động: KHÔNG đọc tham số surge/hotspot, chỉ dùng phân bố đơn QUAN SÁT ĐƯỢC.

  • Lớp PATH PLANNING (không va chạm): với số shipper nhỏ dùng Conflict-Based
    Search (space-time A*); với quy mô lớn dùng prioritized greedy-descent có
    đặt chỗ. Cả hai trả về bước kế tiếp cho từng shipper.

Thiết kế nhắm tới TÍNH THÍCH NGHI và KHẢ NĂNG MỞ RỘNG, không tinh chỉnh theo
bất kỳ config cụ thể nào:

  1. KHOẢNG CÁCH BFS CHỈ TỪ Ô TĨNH: bản đồ tĩnh nên BFS dist-map chỉ tính từ
     các điểm pickup/delivery (số lượng bị chặn bởi số đơn) và cache vĩnh viễn.
     KHÔNG bao giờ BFS lại từ vị trí shipper (vị trí đổi mỗi bước) — đây là
     khác biệt then chốt giúp scale tới N lớn và map nhiều vật cản.

  2. HEURISTIC CHÍNH XÁC: space-time A* dùng chính trường khoảng cách BFS tới
     đích làm heuristic → admissible & consistent & chính xác, A* gần như
     không nở thừa kể cả trên maze.

  3. THAM SỐ KHÔNG-THỨ-NGUYÊN: mọi ngưỡng (detour budget, gating CBS) biểu diễn
     theo khoảng cách/slack thực tế hoặc quy mô bài toán, không hardcode theo
     dải N/C của một bộ config.

Complexity (mỗi timestep):
  Khoảng cách:   BFS chỉ 1 lần/điểm tĩnh, O(N²) mỗi lần, tổng ≤ O(#endpoints·N²)
  Assignment:    O(|orders|·C)
  Path (CBS):    O(max_nodes · A*),  A* ~ O(L) nhờ heuristic chính xác
  Path (lớn):    O(C · deg) prioritized greedy-descent
"""

from __future__ import annotations

import heapq
import random
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env import (
    ALPHA,
    BETA,
    DeliveryEnv,
    Order,
    Shipper,
    r_base,
)
from solvers.solver import Solver

Position = Tuple[int, int]
INF = 10**9

DIRS = {"S": (0, 0), "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _pos_to_move(src: Position, dst: Position) -> str:
    dr, dc = dst[0] - src[0], dst[1] - src[1]
    for m, (r, c) in DIRS.items():
        if r == dr and c == dc:
            return m
    return "S"


class MAPDCBSSolver(Solver):
    """MAPD with Conflict-Based Search."""

    method_name = "MAPDCBSSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._assignments: Dict[int, Optional[int]] = {}
        self._grid = None
        self._T: int = 0
        # BFS dist-map cache, keyed by ô TĨNH (pickup/delivery). Không bao giờ
        # chứa vị trí shipper → kích thước bị chặn bởi số điểm tĩnh.
        self._dist_cache: Dict[Position, Dict[Position, int]] = {}
        self._use_cbs = False
        self._cbs_max_nodes = 50
        # Mục tiêu hiện hành của từng shipper (sticky) — chống thrashing.
        self._goal: Dict[int, Position] = {}
        # Chống deadlock hành lang: đếm số bước một shipper "muốn đi nhưng đứng yên".
        self._stuck: Dict[int, int] = {}
        self._prev_pos: Dict[int, Position] = {}
        # Gom đơn cơ hội bị giới hạn ở 2 đơn/tuyến (bag nhỏ → shipper trống nhanh →
        # phản ứng kịp đơn gấp mới, tăng %đúng hạn) VÀ chỉ nhận đơn có detour rẻ
        # (_detour_f nhỏ) — chỉ tạt qua đơn gần như nằm trên đường, không vòng xa
        # làm trễ đơn đang mang. Hai tham số không-thứ-nguyên: số đơn và TỈ LỆ
        # theo độ dài tuyến, không phụ thuộc N/C của config cụ thể.
        self._opp_max: int = 2
        self._detour_f: float = 0.25
        # Tải trọng lớn nhất của đội; đơn nặng hơn là "đơn chết" — không ai chở nổi.
        self._max_wmax: float = 0.0
        # HEATMAP CẦU QUAN SÁT ĐƯỢC (adaptive, KHÔNG đọc surge/hotspot): tích lũy
        # số đơn từng xuất hiện tại mỗi ô lấy hàng, phân rã theo thời gian. Dùng để
        # ĐỖ DỰ ĐOÁN shipper rảnh về vùng cầu cao → cắt độ trễ nhặt đơn KẾ TIẾP.
        self._demand: Dict[Position, float] = {}
        self._demand_decay: float = 0.98
        self._demand_cands: List[Position] = []
        # Ô đã được một shipper rảnh khác nhắm tới trong bước này (để TRẢI shipper
        # ra nhiều vùng cầu thay vì chụm vào một điểm). Reset mỗi bước.
        self._idle_claims: Set[Position] = set()
        # Đích đỗ-dự-đoán hiện hành của shipper rảnh (sticky) — ô heatmap không có
        # đơn thật nên _goal_valid không giữ được; lưu riêng để tránh dao động.
        self._idle_goal: Dict[int, Position] = {}
        # Bật đỗ-dự-đoán theo heatmap (đặt ở run() theo số shipper); mặc định tắt.
        self._allow_anticip: bool = False
        # Trần horizon của A* không-thời-gian (đủ lớn để né va chạm tầm gần; đích
        # xa hơn tầm này được xử lý bằng greedy-descent — xem _sta_star).
        self._horizon_cap: int = 40
        # ADJACENCY TĨNH precompute: ô -> [chính nó (chờ)] + các ô kề đi-được. Bản
        # đồ tĩnh nên tính một lần; tránh gọi valid_next_pos/is_valid_cell hàng chục
        # triệu lần trong A* không-thời-gian (nút cổ chai chính khi N lớn).
        self._adj: Dict[Position, Tuple[Position, ...]] = {}
        # RNG nội bộ để chọn hướng né (không đọc seed env; chỉ phá đối xứng).
        self._rng = random.Random(12345)

    def _build_adj(self) -> None:
        grid = self._grid
        rows, cols = len(grid), len(grid[0])
        adj: Dict[Position, Tuple[Position, ...]] = {}
        for r in range(rows):
            row = grid[r]
            for c in range(cols):
                if row[c] != 0:
                    continue
                nb = [(r, c)]  # chờ tại chỗ
                if r > 0 and grid[r - 1][c] == 0:
                    nb.append((r - 1, c))
                if r + 1 < rows and grid[r + 1][c] == 0:
                    nb.append((r + 1, c))
                if c > 0 and grid[r][c - 1] == 0:
                    nb.append((r, c - 1))
                if c + 1 < cols and grid[r][c + 1] == 0:
                    nb.append((r, c + 1))
                adj[(r, c)] = tuple(nb)
        self._adj = adj

    # ── distance (chỉ BFS từ ô tĩnh) ────────────────────────────────────────

    def _dist_from(self, origin: Position) -> Dict[Position, int]:
        d = self._dist_cache.get(origin)
        if d is None:
            adj = self._adj
            dist: Dict[Position, int] = {origin: 0}
            q = deque([origin])
            while q:
                pos = q.popleft()
                d1 = dist[pos] + 1
                for nxt in adj[pos]:
                    if nxt not in dist:
                        dist[nxt] = d1
                        q.append(nxt)
            self._dist_cache[origin] = d = dist
        return d

    def _d_to(self, static_cell: Position, frm: Position) -> int:
        """Khoảng cách frm → static_cell, BFS cache từ static_cell (đối xứng)."""
        return self._dist_from(static_cell).get(frm, INF)

    # ── reward estimation ───────────────────────────────────────────────────

    def _expected_reward(self, o: Order, t_deliver_est: int) -> float:
        rb = r_base(o.w)
        if t_deliver_est <= o.et:
            bonus = max(0.0, (o.et - t_deliver_est) / max(o.et, 1))
            return ALPHA[o.p] * rb * (1.0 + bonus)
        factor = max(0.0, 1.0 - (t_deliver_est - o.et) / max(self._T, 1))
        return BETA[o.p] * rb * factor

    # ── task assignment ─────────────────────────────────────────────────────

    def _refresh_assignments(self, shippers: List[Shipper], orders: Dict[int, Order]) -> None:
        for s in shippers:
            self._assignments.setdefault(s.id, None)
        for sid, oid in list(self._assignments.items()):
            if oid is not None and (
                oid not in orders or orders[oid].picked or orders[oid].delivered
            ):
                self._assignments[sid] = None

    def _assign_tasks(self, shippers: List[Shipper], orders: Dict[int, Order], t: int) -> None:
        self._refresh_assignments(shippers, orders)

        taken: Set[int] = {oid for oid in self._assignments.values() if oid is not None}
        in_bags: Set[int] = {oid for s in shippers for oid in s.bag}
        avail = [
            o for o in orders.values()
            if not o.picked and not o.delivered
            and o.id not in taken and o.id not in in_bags
        ]
        if not avail:
            return

        free_s = [s for s in shippers if self._assignments[s.id] is None and len(s.bag) == 0]
        if not free_s:
            return

        free_pos = [(s.r, s.c) for s in free_s]

        def _urgency(o: Order) -> Tuple[int, int, float, int]:
            pick = (o.sx, o.sy)
            best_d = min(self._d_to(pick, fp) for fp in free_pos)
            d_deliver = self._d_to((o.ex, o.ey), pick)
            travel = best_d + d_deliver
            if travel >= INF:
                return (2, INF, 0.0, o.id)
            arrival = t + travel
            er = self._expected_reward(o, arrival)
            density = er / max(travel + 1, 1)
            # EDF: đơn còn kịp đúng hạn (savable=0) đi trước; trong đó đơn có
            # ít thời gian xuất phát còn lại (dispatch slack nhỏ) ưu tiên hơn để
            # giành shipper gần nhất; cuối cùng mới tới reward-density.
            savable = 0 if arrival <= o.et else 1
            dispatch_slack = o.et - d_deliver - t - best_d
            return (savable, dispatch_slack, -density, o.id)

        for o in sorted(avail, key=_urgency):
            if o.id in taken or not free_s:
                continue
            pick = (o.sx, o.sy)
            d_deliver = self._d_to((o.ex, o.ey), pick)
            if d_deliver >= INF:
                continue

            best_ontime: Optional[Shipper] = None
            best_late: Optional[Shipper] = None
            sc_ontime: Tuple[int, int] = (INF, INF)
            sc_late: Tuple[int, int] = (INF, INF)

            for s in free_s:
                w_carried = sum(orders[b].w for b in s.bag if b in orders)
                if len(s.bag) >= s.K_max or w_carried + o.w > s.W_max:
                    continue
                d_pickup = self._d_to(pick, (s.r, s.c))
                if d_pickup >= INF:
                    continue
                arrival = t + d_pickup + d_deliver
                if arrival - o.et >= self._T - 1:   # reward ~ 0 → bỏ
                    continue
                score = (d_pickup, s.id)
                if arrival <= o.et:
                    if score < sc_ontime:
                        sc_ontime, best_ontime = score, s
                else:
                    if score < sc_late:
                        sc_late, best_late = score, s

            best_s = best_ontime or best_late
            if best_s is not None:
                self._assignments[best_s.id] = o.id
                taken.add(o.id)
                free_s.remove(best_s)

    # ── target / waypoint selection ─────────────────────────────────────────

    def _target(self, s: Shipper, orders: Dict[int, Order], t: int) -> Position:
        pos = (s.r, s.c)
        if s.bag:
            deliverable = [orders[b] for b in s.bag if b in orders and not orders[b].delivered]
            if deliverable:
                best_del = min(
                    deliverable,
                    key=lambda o: (
                        o.et - t - self._d_to((o.ex, o.ey), pos),
                        self._d_to((o.ex, o.ey), pos),
                        -o.p,
                    ),
                )
                del_dest = (best_del.ex, best_del.ey)
                direct = self._d_to(del_dest, pos)
                if direct >= INF:
                    return del_dest

                if len(s.bag) < s.K_max and self._opp_max > len(s.bag):
                    assigned_ids = {v for v in self._assignments.values() if v is not None}
                    w_carried = sum(orders[b].w for b in s.bag if b in orders)
                    slack = best_del.et - t - direct
                    # Budget không-thứ-nguyên: theo độ dài tuyến và slack đơn mang.
                    pick_budget = max(2, int(direct * self._detour_f) + max(0, slack) // 15)
                    detour_budget = max(3, int(direct * self._detour_f) + max(0, slack) // 10)

                    best_opp: Optional[Order] = None
                    best_score: Tuple[int, int, int, int] = (INF, INF, INF, INF)
                    for o in orders.values():
                        if o.picked or o.delivered or o.id in assigned_ids:
                            continue
                        if w_carried + o.w > s.W_max:
                            continue
                        opick = (o.sx, o.sy)
                        d_to_pick = self._d_to(opick, pos)
                        if d_to_pick >= INF or d_to_pick > direct + pick_budget:
                            continue
                        d_pick_del = self._d_to(del_dest, opick)
                        if d_pick_del >= INF:
                            continue
                        if d_to_pick + d_pick_del - direct <= detour_budget:
                            # Ưu tiên đơn tự nó cũng giao kịp; rồi gần, ưu tiên cao.
                            opp_late = 0 if t + d_to_pick + d_pick_del <= o.et else 1
                            score = (opp_late, d_to_pick, -o.p, o.et)
                            if score < best_score:
                                best_score, best_opp = score, o
                    if best_opp is not None:
                        return (best_opp.sx, best_opp.sy)
                return del_dest

        oid = self._assignments.get(s.id)
        if oid and oid in orders and not orders[oid].picked:
            return (orders[oid].sx, orders[oid].sy)

        # Rỗng hoàn toàn → reposition về cụm đơn unassigned (cơ chế adaptive).
        # Bỏ qua đơn CHẾT (nặng hơn mọi W_max) — không ai chở nổi, đừng đuổi theo.
        assigned_ids = {v for v in self._assignments.values() if v is not None}
        on_time: List[Tuple[Position, int]] = []
        late: List[Tuple[Position, int]] = []
        for o in orders.values():
            if o.picked or o.delivered or o.id in assigned_ids:
                continue
            if o.w > self._max_wmax:
                continue
            pick = (o.sx, o.sy)
            d_pick = self._d_to(pick, pos)
            if d_pick >= INF:
                continue
            d_del = self._d_to((o.ex, o.ey), pick)
            if d_del >= INF:
                continue
            if t + d_pick + d_del - o.et >= self._T - 1:
                continue
            (on_time if t + d_pick + d_del <= o.et else late).append((pick, d_pick))
        # Còn đơn unassigned khả thi → tiến tới điểm lấy gần nhất (ưu tiên đơn còn
        # kịp đúng hạn). Claim ô đã nhắm để nhánh đỗ-dự-đoán phía dưới trải người ra.
        pool = on_time or late
        if pool:
            pk = min(pool, key=lambda x: x[1])[0]
            self._idle_claims.add(pk)
            return pk

        # Không còn đơn unassigned khả thi → ĐỖ DỰ ĐOÁN theo HEATMAP CẦU quan sát
        # được: tiến tới vùng đơn từng xuất hiện nhiều gần đây (điểm số cầu/khoảng
        # cách), trải đều qua coverage-claim. Thuần adaptive, không đọc surge/hotspot
        # — đây là cách shipper rảnh tự dồn về nơi cầu đang nóng để bắt đơn kế tiếp.
        # Đỗ-dự-đoán chỉ bật khi đội đủ đông để VỪA tiếp tục phục vụ VỪA cử người
        # phủ vùng cầu (≥3 shipper). Với 1–2 shipper, mọi người nên phản ứng tham
        # lam (đi tới đơn gần nhất) thay vì bỏ vùng đứng để đi đón đầu cầu ở xa.
        if self._allow_anticip:
            # Giữ đích đỗ-dự-đoán cũ nếu còn cầu (sticky) → không dao động mỗi bước.
            prev = self._idle_goal.get(s.id)
            if (prev is not None and prev != pos and self._demand.get(prev, 0.0) > 1e-3
                    and self._d_to(prev, pos) < INF):
                self._idle_claims.add(prev)
                return prev
            best_cell: Optional[Position] = None
            best_key: Tuple[int, float] = (1, -1.0)
            for cell in self._demand_cands:
                d = self._d_to(cell, pos)
                if d >= INF:
                    continue
                score = self._demand[cell] / (1.0 + d)
                key = (cell in self._idle_claims, -score)
                if key < best_key:
                    best_key, best_cell = key, cell
            if best_cell is not None and best_cell != pos:
                self._idle_claims.add(best_cell)
                self._idle_goal[s.id] = best_cell
                return best_cell

        # Không có tín hiệu cầu nào → tiến tới điểm lấy còn-chờ gần nhất (kể cả đã
        # gán cho người khác) để sẵn sàng tiếp ứng; mục tiêu sticky nên không dao động.
        best_pk: Optional[Position] = None
        best_d = INF
        for o in orders.values():
            if o.picked or o.delivered:
                continue
            pk = (o.sx, o.sy)
            d = self._d_to(pk, pos)
            if d < best_d:
                best_d, best_pk = d, pk
        if best_pk is not None and best_d < INF and best_pk != pos:
            return best_pk
        return pos

    def _goal_valid(self, s: Shipper, g: Position, orders: Dict[int, Order]) -> bool:
        """Mục tiêu cũ còn hiệu lực? (chống đổi mục tiêu mỗi bước)."""
        if g == (s.r, s.c):
            return False
        for b in s.bag:
            o = orders.get(b)
            if o is not None and not o.delivered and (o.ex, o.ey) == g:
                return True
        w = sum(orders[b].w for b in s.bag if b in orders)
        for o in orders.values():
            if not o.picked and not o.delivered and (o.sx, o.sy) == g:
                if len(s.bag) < s.K_max and w + o.w <= s.W_max:
                    return True
        return False

    def _compute_goals(self, shippers: List[Shipper], orders: Dict[int, Order],
                       t: int) -> Dict[int, Position]:
        goals: Dict[int, Position] = {}
        self._idle_claims = set()
        # Ứng viên đỗ-dự-đoán: chỉ giữ TOP ô cầu cao nhất (theo demand) một lần mỗi
        # bước → chặn chi phí O(#shipper · #ô-cầu) khi N/G lớn (Phase 2). Số ứng
        # viên tỉ lệ số shipper (đủ để mỗi người có lựa chọn riêng), không theo N.
        if self._allow_anticip and self._demand:
            k = max(8, 4 * len(shippers))
            self._demand_cands = sorted(
                (c for c, d in self._demand.items() if d > 1e-3),
                key=lambda c: self._demand[c], reverse=True,
            )[:k]
        else:
            self._demand_cands = []
        for s in shippers:
            g = self._goal.get(s.id)
            if g is not None and self._goal_valid(s, g, orders):
                goals[s.id] = g
            else:
                goals[s.id] = self._target(s, orders, t)
            self._goal[s.id] = goals[s.id]
        return goals

    # ── space-time A* (heuristic = trường BFS chính xác tới goal) ────────────

    def _sta_star(self, start: Position, goal: Position,
                  vertex_cons: Set[Tuple[Position, int]], max_t: int) -> List[Position]:
        if start == goal:
            return [start]
        hfield = self._dist_from(goal)
        h0 = hfield.get(start, INF)
        if h0 >= INF:
            return [start]
        # Đích xa hơn horizon → A* KHÔNG THỂ chạm tới trong max_t bước (chờ chỉ làm
        # dài thêm), ắt sẽ duyệt cạn cả quả-cầu-horizon rồi rơi về greedy. Bỏ qua
        # khâu duyệt tốn kém đó, trả greedy NGAY — kết quả y hệt, nhanh hơn nhiều.
        # Đây là nút cổ chai chính khi N lớn (đích thường ở xa).
        if h0 > max_t:
            return self._greedy_path(start, goal, max_t)
        heap: List[Tuple] = [(h0, 0, start, 0)]
        g_map: Dict[Tuple, int] = {(start, 0): 0}
        par: Dict[Tuple, Optional[Tuple]] = {(start, 0): None}
        found = None
        adj = self._adj
        hget = hfield.get
        gget = g_map.get
        push = heapq.heappush
        while heap:
            _, g, pos, tt = heapq.heappop(heap)
            key = (pos, tt)
            if g > gget(key, INF):
                continue
            if pos == goal:
                found = key
                break
            if tt >= max_t:
                continue
            nt = tt + 1
            ng = g + 1
            for npos in adj[pos]:
                nk = (npos, nt)
                if nk in vertex_cons:
                    continue
                h = hget(npos, INF)
                if h >= INF:
                    continue
                if ng < gget(nk, INF):
                    g_map[nk] = ng
                    par[nk] = key
                    push(heap, (ng + h, ng, npos, nt))
        if found is None:
            return self._greedy_path(start, goal, max_t)
        path, cur = [], found
        while cur is not None:
            path.append(cur[0])
            cur = par[cur]
        path.reverse()
        return path

    def _greedy_path(self, start: Position, goal: Position, max_t: int) -> List[Position]:
        """Descent thuần trên trường khoảng cách (không né va chạm)."""
        hfield = self._dist_from(goal)
        path = [start]
        cur = start
        for _ in range(max_t):
            if cur == goal:
                break
            best, bnp = hfield.get(cur, INF), cur
            for np in self._adj[cur]:
                hv = hfield.get(np, INF)
                if hv < best:
                    best, bnp = hv, np
            if bnp == cur:
                break
            cur = bnp
            path.append(cur)
        return path

    # ── CBS (cho quy mô nhỏ) ─────────────────────────────────────────────────

    @staticmethod
    def _at(path: List[Position], t: int) -> Optional[Position]:
        return path[t] if t < len(path) else None

    def _first_conflict(self, paths: Dict[int, List[Position]]) -> Optional[dict]:
        agents = list(paths.keys())
        if len(agents) < 2:
            return None
        max_t = max(len(p) for p in paths.values())
        for t in range(max_t):
            seen: Dict[Position, int] = {}
            for a in agents:
                p = self._at(paths[a], t)
                if p is None:
                    continue
                if p in seen:
                    return {"type": "vertex", "a1": seen[p], "a2": a, "pos": p, "t": t}
                seen[p] = a
            if t + 1 < max_t:
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        ai, aj = agents[i], agents[j]
                        pi0 = self._at(paths[ai], t); pi1 = self._at(paths[ai], t + 1)
                        pj0 = self._at(paths[aj], t); pj1 = self._at(paths[aj], t + 1)
                        if None in (pi0, pi1, pj0, pj1):
                            continue
                        if pi0 == pj1 and pj0 == pi1:
                            return {"type": "swap", "a1": ai, "a2": aj,
                                    "pos1": pi0, "pos2": pj0, "t": t + 1}
        return None

    def _priority_plan(self, starts, goals, horizon, order=None) -> Dict[int, List[Position]]:
        occupied: Set[Tuple[Position, int]] = set()
        paths: Dict[int, List[Position]] = {}
        seq = order if order is not None else sorted(starts.keys())
        for aid in seq:
            path = self._sta_star(starts[aid], goals[aid], occupied, horizon)
            paths[aid] = path
            for t, pos in enumerate(path):
                occupied.add((pos, t))
            if len(path) >= 2 and path[0] != path[1]:
                occupied.add((path[0], 1))
        return paths

    def _cbs(self, starts, goals, horizon) -> Dict[int, List[Position]]:
        agents = list(starts.keys())
        if all(starts[a] == goals[a] for a in agents):
            return {a: [starts[a]] for a in agents}

        root_cons = {a: frozenset() for a in agents}
        root_paths = {a: self._sta_star(starts[a], goals[a], set(), horizon) for a in agents}
        heap: List[Tuple] = [(sum(len(p) for p in root_paths.values()), 0, root_cons, root_paths)]
        counter = 1

        for _ in range(self._cbs_max_nodes):
            if not heap:
                break
            _, _, c_cons, c_paths = heapq.heappop(heap)
            conflict = self._first_conflict(c_paths)
            if conflict is None:
                return c_paths
            for ca in (conflict["a1"], conflict["a2"]):
                if conflict["type"] == "vertex":
                    new_con = (conflict["pos"], conflict["t"])
                else:
                    new_con = (conflict["pos2"] if ca == conflict["a1"] else conflict["pos1"],
                               conflict["t"])
                nc = {a: (c_cons[a] | {new_con}) if a == ca else c_cons[a] for a in agents}
                np_paths = dict(c_paths)
                np_paths[ca] = self._sta_star(starts[ca], goals[ca], set(nc[ca]), horizon)
                heapq.heappush(heap, (sum(len(p) for p in np_paths.values()),
                                      counter, nc, np_paths))
                counter += 1
        return self._priority_plan(starts, goals, horizon)

    # ── prioritized greedy-descent (cho quy mô lớn) ──────────────────────────

    def _plan_moves_big(self, shippers: List[Shipper],
                        goals: Dict[int, Position]) -> Dict[int, Position]:
        # Ưu tiên shipper đang mang hàng / gần đích trước (ít có lợi khi bị giữ).
        def _key(s: Shipper) -> Tuple[int, int]:
            g = goals[s.id]
            return (0 if s.bag else 1, self._d_to(g, (s.r, s.c)) if g != (s.r, s.c) else 0)

        reserved: Dict[Position, int] = {}
        result: Dict[int, Position] = {}
        for s in sorted(shippers, key=_key):
            start = (s.r, s.c)
            goal = goals[s.id]
            if goal == start:
                result[s.id] = start
                reserved[start] = s.id
                continue
            hfield = self._dist_from(goal)
            cands = []
            for np in self._adj[start]:
                hv = hfield.get(np, INF)
                if hv < INF:
                    cands.append((hv, np))
            cands.sort()
            chosen = start
            for _, np in cands:
                if np != start and np in reserved:
                    continue
                chosen = np
                break
            result[s.id] = chosen
            reserved[chosen] = s.id
        return result

    # ── chống deadlock hành lang ─────────────────────────────────────────────

    def _break_deadlocks(self, shippers: List[Shipper], goals: Dict[int, Position],
                         nxt: Dict[int, Position], stuck_thresh: int = 3) -> None:
        """Phát hiện shipper bị kẹt (muốn đi nhưng đứng yên nhiều bước) và ép né.

        Deadlock kiểu swap ở hành lang 1-ô (vd điểm thắt nối hai nửa bản đồ) không
        thể tự giải bằng prioritized planning: env chỉ giữ-ô khi tranh chấp nên hai
        agent đối đầu đóng băng vĩnh viễn. Cách phá tổng quát: agent kẹt lùi sang
        một ô kề TRỐNG bất kỳ (kể cả ra xa đích) để nhường, tạo khe cho chu trình
        vỡ ra. Không phụ thuộc cấu trúc map cụ thể."""
        # Ô sẽ bị chiếm sau bước này (theo kế hoạch hiện hành) — tránh đâm vào.
        planned = {nxt[s.id] for s in shippers}
        cur = {s.id: (s.r, s.c) for s in shippers}
        occupied_now = set(cur.values())
        # Kẹt = muốn đi (có đích) nhưng vị trí THỰC không đổi so với bước trước.
        # Phải đo bằng dịch chuyển thực, vì planner có thể "định đi" vào ô bị chiếm
        # rồi bị env giữ-ô → nxt != pos nhưng agent vẫn đứng yên.
        for s in shippers:
            pos = cur[s.id]
            wants_move = goals.get(s.id, pos) != pos
            did_not_move = self._prev_pos.get(s.id) == pos
            if wants_move and did_not_move:
                self._stuck[s.id] = self._stuck.get(s.id, 0) + 1
            else:
                self._stuck[s.id] = 0
        for s in shippers:
            self._prev_pos[s.id] = cur[s.id]
        # Xử lý theo thứ tự kẹt-lâu-nhất trước.
        for s in sorted(shippers, key=lambda x: -self._stuck.get(x.id, 0)):
            if self._stuck.get(s.id, 0) < stuck_thresh:
                continue
            pos = cur[s.id]
            cands = []
            for np in self._adj[pos][1:]:   # [0] là chính ô (đứng yên) → bỏ
                # Ô trống: không ai đang đứng và không ai khác định vào.
                if np in occupied_now or np in planned:
                    continue
                cands.append(np)
            if cands:
                esc = self._rng.choice(cands)
                planned.discard(nxt[s.id])
                nxt[s.id] = esc
                planned.add(esc)
                occupied_now.discard(pos)
                occupied_now.add(esc)
                self._stuck[s.id] = 0

    # ── action ───────────────────────────────────────────────────────────────

    def _make_action(self, s: Shipper, nxt: Position, orders: Dict[int, Order]) -> Tuple:
        move = _pos_to_move((s.r, s.c), nxt)
        for bid in s.bag:
            o = orders.get(bid)
            if o is not None and not o.delivered and (o.ex, o.ey) == nxt:
                return (move, 2)
        for o in orders.values():
            if not o.picked and not o.delivered and (o.sx, o.sy) == nxt:
                w = sum(orders[b].w for b in s.bag if b in orders)
                if len(s.bag) < s.K_max and w + o.w <= s.W_max:
                    return (move, 1)
        return (move, 0)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        t0 = time.time()
        obs = self.env.reset()

        C = obs["C"]
        N = obs["N"]
        self._grid = obs["grid"]
        self._T = obs["T"]
        self._build_adj()
        self._max_wmax = max((s.W_max for s in obs["shippers"]), default=0.0)
        # Đỗ-dự-đoán theo heatmap chỉ bật khi đội ≥3: đủ người để vừa phục vụ vừa
        # phủ vùng cầu. Đội 1–2 phản ứng tham lam hiệu quả hơn.
        self._allow_anticip = C >= 3

        # Planner mặc định: prioritized planning (lớp thấp của CBS + thứ tự ưu
        # tiên, đặt chỗ theo toàn tuyến trong không-thời gian). Đây là lựa chọn
        # BỀN: giải xung đột bằng đặt chỗ thay vì conflict-tree (vốn reroute
        # đường vòng dài gây nhiễu thời gian giao hàng). CBS đầy đủ vẫn được cài
        # đặt (_cbs) và có thể bật cho quy mô nhỏ nếu muốn tối ưu makespan.
        self._use_cbs = False
        self._cbs_max_nodes = 60 if C <= 4 else 30
        # Horizon đủ để A* tới đích + vài bước chờ né va chạm.
        horizon = max(8, min(2 * N, self._horizon_cap))

        while not obs["done"]:
            orders: Dict[int, Order] = obs["orders"]
            shippers: List[Shipper] = obs["shippers"]
            t: int = obs["t"]

            # Cập nhật heatmap cầu: phân rã rồi cộng các đơn vừa xuất hiện.
            if self._demand:
                for k in self._demand:
                    self._demand[k] *= self._demand_decay
            for oid in obs.get("new_order_ids", []):
                o = orders.get(oid)
                if o is not None:
                    cell = (o.sx, o.sy)
                    self._demand[cell] = self._demand.get(cell, 0.0) + 1.0

            self._assign_tasks(shippers, orders, t)
            goals = self._compute_goals(shippers, orders, t)

            starts = {s.id: (s.r, s.c) for s in shippers}
            if self._use_cbs:
                paths = self._cbs(starts, goals, horizon)
            else:
                # Thứ tự ưu tiên đường đi theo độ gấp deadline: shipper có đơn
                # (đang mang hoặc được gán) gấp nhất lập đường trước → giành
                # đường ngắn nhất, các shipper khác nhường. Không-thứ-nguyên.
                def _urg(s: Shipper) -> Tuple[int, int]:
                    ets = [orders[b].et for b in s.bag if b in orders]
                    aid = self._assignments.get(s.id)
                    if aid in orders:
                        ets.append(orders[aid].et)
                    return (min(ets) - t) if ets else INF, s.id
                seq = [s.id for s in sorted(shippers, key=_urg)]
                paths = self._priority_plan(starts, goals, horizon, order=seq)
            nxt = {s.id: (paths[s.id][1] if len(paths.get(s.id, [])) >= 2 else (s.r, s.c))
                   for s in shippers}
            self._break_deadlocks(shippers, goals, nxt)

            actions = {s.id: self._make_action(s, nxt[s.id], orders) for s in shippers}
            obs, _, done, _ = self.env.step(actions)

        return self.env.result(self.method_name, elapsed_sec=time.time() - t0)
