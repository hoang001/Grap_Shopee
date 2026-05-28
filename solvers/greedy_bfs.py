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
AGE_COEFF = 3.0  # hệ số thưởng tuổi đơn hàng cho grid N=12 có nhiều nút cổ chai (chống bỏ đói đơn hàng)

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

LRU_CACHE_MAX = 100_000  # số entry tối đa trong cache đường đi cho grid lớn


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

        # Tỉ lệ ô đi được — dùng để phân loại grid cho chiến lược thích nghi trên grid lớn
        self._free_ratio: float = len(self._free_cells) / max(self._N ** 2, 1)

        # Số ô tự do nhỏ nhất trong bất kỳ hàng/cột nào — chỉ số phát hiện nút cổ chai
        self._min_passage_count: int = self._compute_min_passage_count()

        # ── Đặc trưng topo — tính một lần, dùng trong _derive_strategy và chẩn đoán ──
        # _internal_walls:    số ô tường bên trong (không kể viền) — 0 nghĩa là grid hoàn toàn mở
        # _avg_passage_width: trung bình số ô tự do trên mỗi hàng/cột nội bộ
        # _bottleneck_score:  ∈ [0,1] — gần 1 khi hành lang rất hẹp, gần 0 khi grid mở rộng
        self._internal_walls: int = sum(
            1 for r in range(1, self._N - 1) for c in range(1, self._N - 1)
            if self.grid[r][c] == 1
        )
        self._avg_passage_width: float = self._compute_avg_passage_width()
        self._bottleneck_score: float = 1.0 - self._min_passage_count / max(self._N, 1)

        # Số hàng có rất ít ô tự do — phát hiện cấu trúc đa vùng ngang (như F10).
        # Grid đa barrier: nhiều hàng chặn ngang, mỗi hàng chỉ có 1-2 khe hở.
        # Chiến lược tốt: phân tán shipper ra các vùng (spread), không tập trung về tâm.
        _narrow_thresh = max(2, self._N // 20)
        self._barrier_row_count: int = sum(
            1 for r in range(1, self._N - 1)
            if sum(1 for c in range(self._N) if self.grid[r][c] == 0) <= _narrow_thresh
        )

        # Tìm đường: cache dict thường cho N<=20, cache LRU giới hạn cho N>20
        # Kích thước LRU tỉ lệ với N² để duy trì tỉ lệ trúng cache hữu ích trên grid lớn
        self._path_cache: Dict[Tuple[Position, Position], Tuple[int, Move]] = {}
        if self._N > 20:
            self._lru: Optional[OrderedDict] = OrderedDict()
            self._lru_max: int = min(max(100_000, self._N * self._N * 5), 500_000)
        else:
            self._lru = None
            self._lru_max = 0

        # Tất cả tham số chiến lược được tính một lần lúc khởi tạo
        self._strat = self._derive_strategy()
        self._safe_buffer: int = self._strat["safe_buffer"]

        # Mục tiêu pickup đã cam kết của từng shipper — phân bổ chi phí _best_pickup qua nhiều timestep.
        # Chỉ hoạt động với N>20 (replan_interval > 1). Khóa là shipper.id.
        self._committed: Dict[int, Optional[int]] = {}     # → order_id đang được cam kết
        self._committed_since: Dict[int, int] = {}         # → timestep khi tạo cam kết

        # Phát hiện kẹt: ghi lại vị trí bước trước để nhận biết shipper bị chặn.
        # Khi bị chặn, thử hướng đi thay thế để phá bế tắc trong hành lang hẹp.
        self._prev_positions: Dict[int, Position] = {}

        # Bộ nhớ tắc nghẽn nhẹ: đếm sự kiện kẹt tại mỗi ô, giảm dần theo thời gian.
        # Dùng để chọn hướng thoát ít tắc nghẽn hơn — không ảnh hưởng đến A* hoặc BFS.
        self._congestion: Dict[Position, int] = {}
        self._congestion_t: int = 0
        self._congestion_decay_interval: int = 20

        # ── Metrics nhẹ: counter-only, O(1)/bước ──
        # stuck:    lần shipper kẹt xác nhận (vị trí không đổi + di chuyển bị chặn)
        # escaped:  lần chọn được hướng thoát thay thế thành công
        # edf_full: lần EDF được gọi với saveable khác rỗng (phép kiểm tra đắt nhất)
        # ub_skip:  ứng viên _best_pickup bị loại bởi score upper-bound (O(1) filter)
        # opp:      pickup cơ hội thành công (step 1.5 + 2.5)
        # idle:     shipper-bước ở trạng thái chờ (step 3 + S)
        self._m: Dict[str, int] = dict(
            stuck=0, escaped=0, edf_full=0, ub_skip=0, opp=0, idle=0, zone_rep=0
        )

    # ------------------------------------------------------------------
    # Xây dựng tham số chiến lược
    # ------------------------------------------------------------------

    def _derive_strategy(self) -> dict:
        """
        Tính tất cả tham số chiến lược.
        N <= 20: giá trị cố định theo từng N (không có rủi ro hồi quy).
        N > 20:  tính từ free_ratio để tổng quát hóa cho grid Phase 2.
        """
        N, R = self._N, self._free_ratio

        if N <= 20:
            # ── Hành vi cố định theo N (đã kiểm thử trên C1-C6) ──
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
            # Grid có nút cổ chai (N<=15): chỉ kích hoạt khi túi đầy để giữ chiến lược batching.
            # Grid mở (N>=18): kích hoạt ngay khi có hàng để tăng tốc độ vòng quay.
            deliver_nearby_gate = N <= 15
            # Kiểm tra an toàn EDF chỉ dùng cho N>20: N=20 có hành lang phức tạp
            # nên cần kiểm tra hoán vị chính xác như các grid Phase 1 khác.
            use_edf = False
            use_manhattan_dist = False  # Tất cả grid Phase 1 đều có cản bên trong
            replan_interval = 1         # Tắt cam kết — tính lại mỗi bước
            opportunistic_radius = 0    # Tắt tìm kiếm cục bộ cơ hội
            max_detour = 0

        else:
            # ── Thích nghi theo đặc trưng topo cho N > 20 ──
            #
            # Đặc trưng topo đã tính trong __init__ (dùng lại, không tính lại):
            #   _free_ratio          : tỉ lệ ô đi được   (R)
            #   _min_passage_count   : hành lang hẹp nhất (bottleneck)
            #   _internal_walls      : tường nội bộ       (= 0 ↔ grid hoàn toàn mở)
            #   _bottleneck_score    : mức độ tắc nghẽn hình học ∈ [0,1]
            #   _avg_passage_width   : độ rộng hành lang trung bình
            #
            # Nguyên tắc dẫn xuất tham số:
            #   - Mật độ tường cao  → urgency thấp hơn, batching được ưu tiên
            #   - Bottleneck hẹp    → idle về tâm, delivery_gate bật
            #   - Grid hoàn toàn mở → Manhattan thay A*, replan dài hơn
            is_dense = R < 0.60
            # Bottleneck thực sự: min_passage <= max(3, N/10)
            # (hàng biên của grid N=100 có 98 ô tự do — ngưỡng 10 tránh nhầm lẫn hàng biên)
            narrow_threshold = max(3, self._N // 10)
            is_single_bottleneck = self._min_passage_count <= narrow_threshold

            deliver_first = False
            delivery_mode = "nearest"
            age_coeff = age_cap = 0.25
            # Đa barrier: ≥3 hàng có ≤max(2,N/20) ô tự do → grid chia nhiều vùng ngang.
            # Chiến lược: phân tán shipper ra các vùng để tránh phải vượt nhiều barrier.
            # Trái với grid bottleneck đơn (corridor) mà center hiệu quả hơn.
            is_multi_barrier = self._barrier_row_count >= 3
            # Block/hive grid (P9-P12): is_dense do nhiều block nhỏ lấp diện tích, KHÔNG phải mê cung khó đi.
            # Nhận biết: is_dense=True + min_passage cao (hành lang đều đặn, dễ đi) + không phải bottleneck.
            # Chiến lược đúng: spread (phân tán tới hotspot góc) + urgency cao (đến đơn kịp thời).
            is_structured_block = (is_dense and not is_single_bottleneck
                                   and self._min_passage_count >= self._N // 6
                                   and self._N >= 50)
            effective_dense = is_dense and not is_structured_block
            n_urgency = 1.5 if is_dense else URGENCY_COEFF
            # Bottleneck đơn + khe DUY NHẤT (min_passage=1): center idle xếp hàng tại khe → hiệu quả.
            # Bottleneck đơn + nhiều khe (min_passage≥2): spread phân agent qua các khe, tránh nghẽn.
            # Ví dụ: P6 (min=1, 1 khe ở tâm) → center; P3 (min=3) và P5 (min=2) → spread.
            is_single_cell_bottleneck = is_single_bottleneck and self._min_passage_count <= 1
            idle_mode = "center" if (effective_dense or is_single_cell_bottleneck) and not is_multi_barrier else "spread"
            safe_buffer = 0
            deliver_nearby = 2
            deliver_nearby_gate = is_dense or is_single_bottleneck
            use_edf = True
            # Manhattan an toàn chỉ khi không có tường nội bộ (Manhattan = BFS trên grid trống)
            use_manhattan_dist = (self._internal_walls == 0)
            # Đa barrier: replan_interval ngắn hơn để phản hồi nhanh với đơn hàng mới trong vùng gần.
            replan_interval = 20 if use_manhattan_dist else (6 if is_multi_barrier else 12)
            opportunistic_radius = 8 if use_manhattan_dist else 5
            max_detour = 6

        return {
            "deliver_first": deliver_first,
            "delivery_mode": delivery_mode,
            "d_blend": 0.05,
            "n_urgency": n_urgency,
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
        """Tìm ô tự do gần tâm grid nhất."""
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
        """Tính n vị trí phân tán trên grid bằng cách dùng neo góc phần tư rồi tối đa hóa khoảng cách tối thiểu."""
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
        """
        Số ô tự do nhỏ nhất trong bất kỳ hàng hoặc cột nào.
        Grid mở N=50: ~48 (toàn bộ hàng giữa là tự do).
        Grid corridor: nhỏ (ví dụ 2 nếu hàng chặn chỉ có 2 khe hở).
        Dùng làm chỉ số độ rộng thông đạo — không phải đếm số đoạn liên tục.
        """
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
        """Trung bình số ô tự do trên mỗi hàng và cột nội bộ (không kể hàng/cột viền).

        Bổ sung cho _min_passage_count: min đo điểm hẹp nhất, avg đo độ mở tổng thể.
        Grid mở N=50: avg ≈ N-2 ≈ 48. Grid có nhiều tường bên trong: avg thấp hơn nhiều.
        """
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

    # ------------------------------------------------------------------
    # Tìm đường — BFS (N<=20) hoặc A* (N>20), cả hai đều có cache
    # ------------------------------------------------------------------

    def _bfs_compute(self, start: Position, goal: Position) -> Tuple[int, Move]:
        """BFS thuần không dùng cache. Chỉ dùng cho N<=20."""
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
        """A* với heuristic Manhattan. Dùng cho N>20. Trả về (khoảng_cách, bước_di_chuyển_đầu_tiên)."""
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF, "S"

        def h(pos: Position) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # (f, g, hàng, cột, bước_đầu) — hàng/cột là int để tránh so sánh chuỗi Move khi bằng nhau
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
        """Tìm đường có cache: BFS+cache_đầy_đủ cho N<=20, A*+LRU cho N>20."""
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
        """Khoảng cách thực giữa hai ô: Manhattan O(1) cho grid mở, BFS/A* có cache cho grid còn lại."""
        if self._strat["use_manhattan_dist"]:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        return self._bfs(a, b)[0]

    def _move_to(self, here: Position, goal: Position) -> Tuple[Move, Position]:
        """Tính bước di chuyển tiếp theo từ here về phía goal. Trả về (hướng_đi, vị_trí_tiếp_theo)."""
        if here == goal:
            return "S", here
        if self._strat["use_manhattan_dist"]:
            dr = goal[0] - here[0]
            dc = goal[1] - here[1]
            # Ưu tiên chiều có delta lớn hơn để đi thẳng nhất có thể
            move = ("D" if dr > 0 else "U") if abs(dr) >= abs(dc) else ("R" if dc > 0 else "L")
            return move, (here[0] + (1 if move == "D" else -1 if move == "U" else 0),
                          here[1] + (1 if move == "R" else -1 if move == "L" else 0))
        move = self._bfs(here, goal)[1]
        return move, valid_next_pos(here, move, self.grid)

    # ------------------------------------------------------------------
    # Cam kết kế hoạch — giảm chi phí A* mỗi timestep cho N>20
    # ------------------------------------------------------------------

    def _commitment_valid(self, shipper: Shipper, orders: Dict[int, Order],
                          reserved: set, t: int) -> bool:
        """Trả về True khi mục tiêu pickup đã cam kết của shipper còn hợp lệ và chưa hết hạn."""
        sid = shipper.id
        oid = self._committed.get(sid)
        if oid is None or oid in reserved:
            return False
        if oid not in orders:
            return False
        o = orders[oid]
        if o.picked or o.delivered:
            return False
        # Hết hạn nếu đã theo đuổi quá replan_interval bước
        return (t - self._committed_since.get(sid, t)) < self._strat["replan_interval"]

    def _set_commitment(self, sid: int, oid: int, t: int) -> None:
        """Ghi lại cam kết pickup mới cho shipper sid tại timestep t."""
        self._committed[sid] = oid
        self._committed_since[sid] = t

    def _opportunistic_nearby(self, shipper: Shipper, orders: Dict[int, Order],
                               available: List[Order], reserved: set,
                               committed_goal: Position, t: int,
                               max_detour_override: Optional[int] = None) -> Optional[Order]:  # [DYNAMIC BUDGET]
        """
        Tìm pickup gần có thể chèn vào đường đến committed_goal với tối đa max_detour bước thêm.

        Bộ lọc không gian Manhattan <= radius giữ tập ứng viên nhỏ; kiểm tra detour dùng
        dist thực để không bị đánh lừa bởi ước lượng không gian mở trên grid có cản.
        """
        radius = self._strat["opportunistic_radius"]
        if radius <= 0:
            return None
        # [DYNAMIC BUDGET] dùng override nếu được truyền vào, ngược lại dùng giá trị cố định từ strategy
        max_detour = max_detour_override if max_detour_override is not None else self._strat["max_detour"]  # [DYNAMIC BUDGET]
        r0, c0 = shipper.position
        direct = self._dist(shipper.position, committed_goal)
        if direct == 0:
            return None  # Đã đến đích, không còn đoạn đường để chèn vào

        best, best_score = None, -float("inf")
        for o in available:
            if o.id in reserved:
                continue
            if abs(r0 - o.sx) + abs(c0 - o.sy) > radius:
                continue  # Ngoài vùng không gian
            if not shipper.can_carry(o, orders):
                continue
            if t + abs(r0 - o.sx) + abs(c0 - o.sy) + abs(o.sx - o.ex) + abs(o.sy - o.ey) >= self._T:
                continue
            # Detour = số bước thêm nếu ghé qua pickup này trước khi đến committed_goal.
            # Với grid mở: O(1); với grid có cản: 2 lần A* thường trúng cache LRU sau vài bước.
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
        """Tìm pickup trong hành lang ellipse dọc theo tuyến đường shipper→goal.

        Khác _opportunistic_nearby (bán kính quanh vị trí hiện tại), phương thức này
        tìm kiếm dọc theo toàn bộ tuyến đường:
            d_M(pos,p) + d_M(p,goal) <= d_M(pos,goal) + max_detour
        Phù hợp với giao hàng đường dài: pickup ở giữa tuyến được phát hiện
        dù cách xa shipper nhưng không tốn thêm bước.
        """
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
        # Tighter pre-filter for high path-stretch grids (many barriers → actual >> Manhattan).
        # On obstacle-heavy grids, most candidates passing direct_m+max_detour would fail
        # the actual A* detour check — just wasting 2 A* calls each.
        # Scale the Manhattan window down proportionally to path_stretch = direct/direct_m.
        if direct_m > 0 and direct >= 2 * direct_m:
            m_filter_extra = max(1, int(max_detour * direct_m / direct))
        else:
            m_filter_extra = max_detour
        best, best_score = None, -float("inf")
        for o in available:
            if o.id in reserved:
                continue
            # Pre-filter nhanh bằng Manhattan trước khi gọi A*
            m_via = abs(r0 - o.sx) + abs(c0 - o.sy) + abs(o.sx - gr) + abs(o.sy - gc)
            if m_via > direct_m + m_filter_extra:
                continue
            if not shipper.can_carry(o, orders):
                continue
            if t + abs(r0 - o.sx) + abs(c0 - o.sy) + abs(o.sx - o.ex) + abs(o.sy - o.ey) >= self._T:
                continue
            # Kiểm tra detour thực bằng BFS/A* (2 lần gọi, thường trúng LRU cache)
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

    # ------------------------------------------------------------------
    # Kiểm tra an toàn khi nhận thêm hàng
    # ------------------------------------------------------------------

    def _is_safe_to_pickup(self, shipper: Shipper, pickup_pos: Position,
                           orders: Dict[int, Order], t: int) -> bool:
        """Trả về True nếu tồn tại thứ tự giao hàng giữ tất cả đơn hàng đang mang đúng hạn."""
        if not shipper.bag:
            return True
        # Trì hoãn tính d1 — tránh gọi A* khi tất cả đơn hàng đang mang đều hết hạn.
        # Manhattan pre-filter (O(1)): nếu Manhattan > et - t thì actual dist cũng vậy.
        # Chỉ gọi A* cho các item còn khả năng giao đúng hạn theo Manhattan.
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
            # N>20: thứ tự EDF là O(bag × BFS) thay vì O(bag! × bag × BFS).
            # EDF gần tối ưu cho bài toán kiểm tra tính khả thi theo deadline.
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
            # N<=20: kiểm tra hoán vị chính xác — rẻ vì N nhỏ và cache đã ấm
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

    # ------------------------------------------------------------------
    # Tính điểm ưu tiên
    # ------------------------------------------------------------------

    def _score_pickup(self, pos: Position, order: Order, t: int) -> float:
        """Điểm = (phần_thưởng_ước_tính / tổng_bước) × hệ_số_urgency × hệ_số_tuổi."""
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
        """
        VRP-inspired lightweight candidate pruning for _best_pickup.

        On large / overloaded maps, avoid running expensive safety and exact
        distance checks for every open order. Keep both nearby candidates for
        throughput and urgent/high-priority candidates as a safety valve.
        """
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

    # ------------------------------------------------------------------
    # Chính sách: chọn pickup và giao hàng
    # ------------------------------------------------------------------

    def _best_pickup(self, shipper: Shipper, orders: Dict[int, Order],
                     available: List[Order], reserved: set, t: int,
                     reserved_zones: Optional[set] = None) -> Optional[Order]:
        """Chọn đơn hàng tốt nhất để nhận: điểm urgency cao nhất, đủ sức chứa, an toàn cho hàng đang mang."""
        best, best_score = None, -float("inf")
        r0, c0 = shipper.position
        # Sắp xếp gần → xa để UB filter loại bỏ đơn hàng xa sớm hơn (giảm số lần gọi A*).
        # Đơn hàng gần nhất được chấm điểm trước → best_score đạt giá trị cao sớm → UB prune hiệu quả hơn.
        # Kết quả lựa chọn cuối cùng giống hệt (UB filter bảo toàn tính chính xác).
        for o in self._pickup_candidate_pool(shipper, available, reserved, t):
            if o.id in reserved:
                continue
            if not shipper.can_carry(o, orders):
                continue
            # Lọc Manhattan: đơn hàng không thể hoàn thành trước khi simulation kết thúc.
            d1_m = abs(r0 - o.sx) + abs(c0 - o.sy)
            d2_m = abs(o.sx - o.ex) + abs(o.sy - o.ey)
            if t + d1_m + d2_m >= self._T:
                continue

            # ── Score upper-bound filter (O(1)) ──
            # UB dùng Manhattan (d1_m ≤ d1, d2_m ≤ d2) → est_ub ≥ actual, denom ≤ actual.
            # urgency_ub = 1 + n_urgency (giả định slack=1, trường hợp urgency cao nhất).
            # age_ub = 1 + age_cap (giả định tuổi tối đa).
            # Nếu UB ≤ best_score → actual score cũng ≤ best_score → bỏ qua an toàn.
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
            # Zone-based soft repulsion: giảm điểm nếu shipper khác đã nhận/cam kết pickup cùng ô.
            # Không chặn truy cập — chỉ làm giảm ưu tiên nhẹ để tránh herding.
            if reserved_zones and (o.sx, o.sy) in reserved_zones:
                s *= 0.80
                self._m["zone_rep"] += 1
            if s > best_score:
                best_score = s
                best = o
        return best

    def _best_delivery_target(self, shipper: Shipper, orders: Dict[int, Order],
                               t: int) -> Optional[Order]:
        """Chọn đơn hàng đang mang tốt nhất để giao, theo chiến lược delivery_mode."""
        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried:
            return None

        # Nhóm theo điểm đến để giao nhiều đơn cùng chỗ một lượt
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
            flag = 0 if slack >= 0 else 1  # ưu tiên đơn hàng còn kịp giao đúng hạn

            if mode == "nearest":
                key = (flag, d, slack)
            elif mode == "blend":
                key = (flag, slack + d_blend * d)
            else:  # "urgency"
                key = (flag, slack, d)

            if best_key is None or key < best_key:
                best_key = key
                best = most_urgent

        return best

    def _idle_target(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Position]:
        """Xác định vị trí chờ thích nghi khi shipper không có việc làm."""
        mode = self._strat["idle_mode"]

        if mode == "center":
            # Tập trung về tâm grid — phù hợp với grid có nút cổ chai hoặc dày đặc
            return self._grid_center

        if mode == "order_seeking":
            # Tiến về pickup gần nhất trong danh sách đơn hàng chờ
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
            # Mỗi shipper về vị trí phân tán được gán sẵn (phủ đều toàn bộ grid)
            idx = shipper.id % len(self._spread_positions) if self._spread_positions else 0
            return self._spread_positions[idx] if self._spread_positions else self._grid_center

        return None  # "stay" — đứng yên

    # ------------------------------------------------------------------
    # Quyết định hành động chính
    # ------------------------------------------------------------------

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        t: int = obs["t"]
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        # Vị trí hiện tại của tất cả shipper — dùng cho phát hiện kẹt và tránh va chạm mềm.
        all_positions: set = {s.position for s in shippers}

        # Xây dựng danh sách available một lần — tránh quét lại tất cả đơn hàng (kể cả đã giao/đã nhận)
        # cho từng shipper. Quan trọng với N lớn khi đơn hàng tích lũy nhiều.
        available: List[Order] = [o for o in orders.values() if not o.picked and not o.delivered]

        actions: Dict[int, Action] = {}
        reserved: set = set()

        # Pre-populate reserved_zones từ cam kết hiện tại (chỉ khi đủ nhiều shipper).
        # Mục đích: giảm điểm các đơn cùng ô pickup với shipper đã cam kết → chống herding.
        # Ngưỡng > 5: bỏ qua grid nhỏ (C1-C6) vì shipper ít nên xung đột zone hiếm.
        reserved_zones: Optional[set] = None
        if len(shippers) > 5:
            reserved_zones = set()
            for _s in shippers:
                coid = self._committed.get(_s.id)
                if coid and coid in orders:
                    _o = orders[coid]
                    if not _o.picked and not _o.delivered:
                        reserved_zones.add((_o.sx, _o.sy))

        # Shipper ít hàng trong túi được ưu tiên chọn pickup trước
        for shipper in sorted(shippers, key=lambda s: (len(s.bag), s.id)):
            pos = shipper.position

            # ── 0. GIAO TRƯỚC cho grid nhỏ: giảm độ trễ giao hàng ──
            if self._strat["deliver_first"] and shipper.bag:
                delivery = self._best_delivery_target(shipper, orders, t)
                if delivery is not None:
                    dest = (delivery.ex, delivery.ey)
                    move, nxt = self._move_to(pos, dest)
                    actions[shipper.id] = (move, 2 if nxt == dest else 0)
                    continue

            # ── 0.5. GIAO NẾU GẦN: dọn túi rẻ trước khi nhận hàng mới ──
            # Chặn: grid nút cổ chai chỉ kích hoạt khi túi đầy (giữ batching).
            # Grid mở kích hoạt ngay khi có hàng (vòng quay nhanh hơn).
            gate = self._strat["deliver_nearby_gate"]
            if shipper.bag and (not gate or len(shipper.bag) >= shipper.K_max):
                delivery = self._best_delivery_target(shipper, orders, t)
                if delivery is not None:
                    dest = (delivery.ex, delivery.ey)
                    if self._dist(pos, dest) <= self._strat["deliver_nearby"]:
                        move, nxt = self._move_to(pos, dest)
                        actions[shipper.id] = (move, 2 if nxt == dest else 0)
                        continue

            # ── 1. NHẬN HÀNG: theo cam kết hoặc tính lại ──
            # Cam kết phân bổ chi phí _best_pickup (đắt với N>20) qua replan_interval bước.
            # N<=20: replan_interval=1 → use_commitment=False → hành vi y hệt trước đây.
            use_commitment = self._strat["replan_interval"] > 1
            if use_commitment and self._commitment_valid(shipper, orders, reserved, t):
                # Dùng cam kết cũ — không gọi _best_pickup
                oid = self._committed[shipper.id]
                o = orders[oid]
                reserved.add(oid)
            else:
                # Tính lại: quét toàn bộ available để tìm đơn hàng tốt nhất
                o = self._best_pickup(shipper, orders, available, reserved, t, reserved_zones)
                if o is not None:
                    reserved.add(o.id)
                    if reserved_zones is not None:
                        reserved_zones.add((o.sx, o.sy))
                    if use_commitment:
                        self._set_commitment(shipper.id, o.id, t)

            if o is not None:
                goal = (o.sx, o.sy)
                # ── 1.5. CƠ HỘI GẦN: nhặt thêm đơn hàng trên đường đi ──
                # Chỉ quét đơn hàng trong bán kính không gian → O(gần) không phải O(tất_cả).
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

            # ── 2. GIAO HÀNG: giao đơn hàng cấp bách nhất đang mang ──
            if shipper.bag:
                delivery = self._best_delivery_target(shipper, orders, t)
                if delivery is not None:
                    dest = (delivery.ex, delivery.ey)
                    # ── 2.5. CƠ HỘI TRÊN ĐƯỜNG GIAO: tìm pickup trong hành lang delivery route ──
                    # Corridor ellipse filter thay thế radius — pickup phải nằm gần tuyến đường,
                    # không chỉ gần shipper. max_detour=2 giữ opportunism thật sự nhỏ.
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

            # ── 3. CHỜ: di chuyển đến vị trí chờ thích nghi ──
            self._m["idle"] += 1
            idle = self._idle_target(shipper, orders)
            if idle is not None and idle != pos:
                move, _ = self._move_to(pos, idle)
                actions[shipper.id] = (move, 0)
                continue

            actions[shipper.id] = ("S", 0)

        # ── Phá bế tắc + bộ nhớ tắc nghẽn nhẹ ──
        # Điều kiện kẹt: (1) vị trí không đổi, (2) hành động là di chuyển thực (≠ "S").
        # Decay thích nghi: bottleneck nặng → decay interval ngắn hơn (tránh tích lũy).
        # Bound tại 16: ngăn giá trị stale thống trị lựa chọn thoát dài hạn.

        if t - self._congestion_t >= self._congestion_decay_interval and self._congestion:
            self._congestion = {p: v >> 1 for p, v in self._congestion.items() if v > 1}
            self._congestion_t = t

        for shipper in shippers:
            sid = shipper.id
            if sid not in actions:
                continue
            old_move, _ = actions[sid]
            if old_move == "S":
                continue  # dừng chủ động — không phải kẹt
            if shipper.position != self._prev_positions.get(sid):
                continue  # vị trí thay đổi — không kẹt
            desired_nxt = valid_next_pos(shipper.position, old_move, self.grid)
            if desired_nxt != shipper.position and desired_nxt not in all_positions:
                continue  # hướng muốn đi không bị chặn bởi shipper khác → giữ nguyên

            # Kẹt xác nhận — ghi nhớ ô này đang bị tắc nghẽn (bounded tại 16)
            self._congestion[shipper.position] = min(
                self._congestion.get(shipper.position, 0) + 1, 16
            )
            self._m["stuck"] += 1

            # Chọn hướng thay thế ít tắc nghẽn nhất trong số các hướng hợp lệ
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

        # Lưu vị trí hiện tại để phát hiện kẹt ở bước tiếp theo
        for shipper in shippers:
            self._prev_positions[shipper.id] = shipper.position

        return actions

    # ------------------------------------------------------------------
    # Vòng lặp chính
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Chạy toàn bộ simulation và trả về kết quả."""
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