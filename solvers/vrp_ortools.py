from __future__ import annotations

import heapq
import itertools
import random
import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from env import (
    DeliveryEnv,
    Order,
    Shipper,
    delivery_reward,
    is_valid_cell,
    manhattan,
    move_cost,
    r_base,
    valid_next_pos,
)
from solver import Solver


# =========================================================
# KIỂU DỮ LIỆU CHUNG
# =========================================================
# Event: một hành động logic trong route
#   - kind: "pickup" hoặc "deliver"
#   - oid: ID đơn hàng
#   - target: ô đích cần đi tới để thực hiện event
Event = Tuple[str, Optional[int], Tuple[int, int]]

# Route: danh sách các event theo thứ tự thực hiện
Route = List[Event]

# RouteKey: khóa rút gọn của route để dùng cho cache
RouteKey = Tuple[Tuple[str, Optional[int]], ...]

# InsertionDelta: kết quả của bài toán chèn đơn vào route
#   - cost_delta: độ thay đổi điểm số nếu chèn đơn vào route
#   - pickup_pos: vị trí chèn pickup
#   - delivery_pos: vị trí chèn delivery
InsertionDelta = Tuple[float, int, int]


# =========================================================
# MÔ HÌNH VRP TẠM THỜI THEO TỪNG STEP
# =========================================================
class VRPModel:
    """
    VRPModel lưu snapshot trạng thái tại thời điểm hiện tại.

    Vai trò trong phương pháp:
    - gom toàn bộ dữ liệu đầu vào của solver vào một object
    - giúp các hàm tối ưu route chỉ cần nhận 1 model thay vì nhiều tham số rời
    - giữ các thuộc tính t, T, grid, shippers, orders và danh sách đơn mở
    """

    def __init__(
        self,
        t: int,
        T: int,
        grid: List[List[int]],
        shippers: List[Shipper],
        orders: Dict[int, Order],
    ):
        self.t = t
        self.T = T
        self.grid = grid
        self.shippers = shippers
        self.orders = orders

        # Danh sách đơn chưa giao, dùng cho bước chọn ứng viên / phân đơn / lập route
        self.open_orders: List[Order] = [
            o for o in orders.values()
            if not o.delivered
        ]


# =========================================================
# SOLVER CHÍNH: VRP theo phong cách OR-Tools heuristic
# =========================================================
class VRPOrToolsSolver(Solver):
    """
    Các thuật toán chính tạo thành phương pháp:
    1. Candidate ranking
       - Xếp hạng đơn theo lợi ích, khoảng cách, rủi ro trễ.
    2. Order-to-shipper assignment
       - Gán đơn cho shipper phù hợp nhất theo tải trọng và sức chứa.
    3. Insertion-based routing
       - Chèn pickup/deliver vào route tại vị trí tốt nhất.
    4. Bundle construction
       - Gom các event cùng vị trí thành "bundle" để giảm không gian tìm kiếm.
    5. Local search
       - Đổi chỗ / di chuyển / đảo đoạn bundle trong một route.
    6. Cross-route improvement
       - Chuyển đơn giữa các shipper nếu tổng điểm tốt hơn.
    7. Large Neighborhood Search (LNS)
       - Phá một phần lời giải rồi sửa lại để thoát local optimum.
    8. A* pathfinding
       - Tìm đường đi ngắn nhất trên grid để biến route logic thành bước di chuyển thực tế.
    """

    # Số ứng viên toàn cục tối đa đem đi xét trong một lượt gán đơn
    _MAX_GLOBAL_CANDIDATES = 80

    # Mỗi đơn chỉ xét một số shipper gần nhất để giảm độ phức tạp
    _MAX_SHIPPER_CANDIDATES_PER_ORDER = 5

    # Nếu số bundle nhỏ, thử toàn bộ hoán vị để tìm thứ tự tốt nhất
    _MAX_ROUTE_BUNDLES_PERMUTE = 4

    # Sau bao nhiêu step thì lập kế hoạch lại
    _MAX_REPLAN_INTERVAL = 3

    # Giới hạn thời gian cho local search mức bundle / cross route
    _MOVE_TIME_CAP = 0.06

    # Tham số cho LNS
    _LNS_DESTROY_RATIO = 0.15
    _LNS_NUM_ITERATIONS = 3
    _LNS_TIME_CAP = 0.03

    def __init__(self, env: DeliveryEnv):
        """
        Khởi tạo solver.
        """
        super().__init__(env)
        self._path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[str]] = {}
        self._dist_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        self._route_eval_cache: Dict[Tuple[int, RouteKey, int, int], float] = {}
        self._current_model: Optional[VRPModel] = None
        self._current_routes: Dict[int, Route] = {}
        self._current_assignment: Dict[int, List[int]] = {}
        self._last_grid_id: Optional[int] = None
        self._last_replan_t: int = -10**9
        self._plan_start: float = 0.0

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(self) -> dict:
        """
        Vòng lặp chính của solver.

        Luồng xử lý:
        1. Reset môi trường
        2. Ở mỗi step:
           - lập kế hoạch hành động
           - chuẩn hóa action cho từng shipper
           - gọi env.step(...)
        3. Khi kết thúc, trả về result của môi trường

        Tác dụng:
            Đây là điểm vào chính của toàn bộ solver.
        """
        start = time.time()

        try:
            obs = self.env.reset()

            while not obs["done"]:
                actions = self._plan_actions(obs)

                fixed_actions: Dict[int, Tuple[str, int]] = {}
                for shipper in sorted(obs["shippers"], key=lambda s: s.id):
                    sid = shipper.id
                    act = actions.get(sid, ("S", 0))

                    # Bảo vệ kiểu dữ liệu đầu ra của action
                    if not isinstance(act, tuple) or len(act) != 2:
                        act = ("S", 0)

                    move, cargo = act
                    if move not in ("S", "L", "R", "U", "D"):
                        move = "S"
                    if not isinstance(cargo, int):
                        cargo = 0

                    fixed_actions[sid] = (move, cargo)

                obs, _, _, _ = self.env.step(fixed_actions)

        except Exception as e:
            print("SOLVER ERROR:", repr(e))
            raise

        elapsed = time.time() - start
        return self.env.result("VRP-OrTools", elapsed)

    def _plan_actions(self, obs: dict) -> Dict[int, Tuple[str, int]]:
        """
        Tạo action cho từng shipper ở một step.
        """
        model = self._build_model(obs)
        self._current_model = model

        need_replan = (
            not self._current_routes
            or (obs["t"] - self._last_replan_t) >= self._MAX_REPLAN_INTERVAL
            or set(self._current_routes.keys()) != {s.id for s in model.shippers}
        )

        if need_replan:
            self._route_eval_cache.clear()
            self._plan_start = time.time()
            self._current_routes = self._solve_vrp(model)
            self._last_replan_t = obs["t"]

        actions: Dict[int, Tuple[str, int]] = {}
        occupied_next: Set[Tuple[int, int]] = set()

        for shipper in sorted(model.shippers, key=lambda s: s.id):
            route = self._current_routes.get(shipper.id, [])
            route = self._sanitize_route_prefix(shipper, route, model.orders)
            self._current_routes[shipper.id] = route

            action = self._action_from_route(
                shipper=shipper,
                route=route,
                grid=model.grid,
                occupied_next=occupied_next,
            )
            actions[shipper.id] = action

            # Đánh dấu ô dự kiến sẽ bị chiếm trong bước kế tiếp
            mv = action[0]
            if mv != "S":
                occupied_next.add(valid_next_pos(shipper.position, mv, model.grid))
            else:
                occupied_next.add(shipper.position)

        return actions

    # =========================================================
    # MODEL BUILDING
    # =========================================================

    def _ensure_grid_cache(self, grid: List[List[int]]) -> None:
        """
        Reset cache path / distance nếu grid đã thay đổi.
        """
        gid = id(grid)
        if self._last_grid_id != gid:
            self._path_cache.clear()
            self._dist_cache.clear()
            self._last_grid_id = gid

    def _build_model(self, obs: dict) -> VRPModel:
        """
        Tạo VRPModel từ obs hiện tại.
        """
        t = obs["t"]
        T = obs["T"]
        grid = obs["grid"]
        shippers: List[Shipper] = obs["shippers"]
        orders: Dict[int, Order] = obs["orders"]

        self._ensure_grid_cache(grid)

        model = VRPModel(
            t=t,
            T=T,
            grid=grid,
            shippers=shippers,
            orders=orders,
        )
        model.profile = self._routing_profile(model)
        return model

    def _routing_profile(self, model: VRPModel) -> Dict[str, float]:
        """
        Tạo bộ trọng số cho hàm đánh giá route.

        Ý nghĩa:
            - Khi gần hết thời gian: tăng phạt trễ, tăng ưu tiên giao nhanh
            - Khi mật độ đơn cao: tăng trọng số cân bằng để tránh lệch tải
        """
        open_orders = model.open_orders
        n_open = len(open_orders)
        total_cells = max(1, len(model.grid) * len(model.grid[0]))
        open_density = n_open / total_cells
        remaining_ratio = (model.T - model.t) / max(1, model.T)

        late_w = 0.055
        priority_w = 0.070
        balance_w = 0.42
        reward_w = 1.0

        if remaining_ratio < 0.35:
            late_w = 0.08
            priority_w = 0.10
            balance_w = 0.55

        if open_density > 0.0025:
            balance_w = max(balance_w, 0.50)

        return {
            "reward_w": reward_w,
            "late_w": late_w,
            "priority_w": priority_w,
            "balance_w": balance_w,
        }

    def _adaptive_plan_budget(self, model: VRPModel) -> float:
        """
        Tính ngân sách thời gian cho 1 lần lập kế hoạch.

        Tác dụng:
            - ít đơn thì cho phép tìm sâu hơn
            - nhiều đơn thì giảm thời gian để đảm bảo chạy nhanh
        """
        n_orders = len(model.open_orders)
        if n_orders >= 110:
            return 0.04
        if n_orders >= 80:
            return 0.05
        if n_orders >= 50:
            return 0.065
        if n_orders >= 25:
            return 0.08
        return 0.10

    # =========================================================
    # SOLVE / PIPELINE
    # =========================================================

    def _solve_vrp(self, model: VRPModel) -> Dict[int, Route]:
        """
        Pipeline giải VRP.

        Trình tự:
        1. Xây lời giải ban đầu bằng assignment + bundle ordering
        2. Local search trong từng route
        3. Cross-route improvement
        4. LNS để thoát local optimum

        Tác dụng:
            Tạo ra lời giải tốt hơn dần qua nhiều tầng tối ưu.
        """
        self._route_eval_cache.clear()
        self._plan_time_budget = self._adaptive_plan_budget(model)

        first_solution = self._first_solution_strategy(model)
        self._current_routes = first_solution

        improved = self._local_search_improvement(
            model=model,
            routes=first_solution,
            assigned=self._current_assignment,
        )
        improved = self._cross_route_improvement(model, improved)
        improved = self._lns_improvement(model, improved)

        model.routes = improved
        return model.routes

    def _first_solution_strategy(self, model: VRPModel) -> Dict[int, Route]:
        """
        Tạo lời giải ban đầu bằng assignment và bundle ordering heuristic.
        """
        routes: Dict[int, Route] = {}
        shipper_list = sorted(model.shippers, key=lambda s: s.id)

        assigned = self._assign_orders_to_shippers(model)
        self._current_assignment = assigned

        for shipper in shipper_list:
            bundles = self._build_shipper_bundles(
                shipper=shipper,
                assigned_order_ids=assigned.get(shipper.id, []),
                orders=model.orders,
            )
            route = self._choose_best_bundle_order(
                shipper=shipper,
                bundles=bundles,
                orders=model.orders,
            )
            routes[shipper.id] = route

        return routes

    def _find_best_insertion(
        self,
        shipper: Shipper,
        route: Route,
        order: Order,
        orders: Dict[int, Order],
        model: VRPModel,
    ) -> InsertionDelta:
        """
        Tìm vị trí chèn tốt nhất theo hàm đánh giá hiện tại.
        """
        best_delta = float("inf")
        best_positions = (0, 0)

        for i in range(len(route) + 1):
            for j in range(i + 1, len(route) + 2):
                delta = self._insert_delta(
                    shipper=shipper,
                    route=route,
                    order=order,
                    orders=orders,
                    pickup_pos=i,
                    delivery_pos=j,
                    model=model,
                )
                if delta < best_delta:
                    best_delta = delta
                    best_positions = (i, j)

        return (best_delta, best_positions[0], best_positions[1])

    def _insert_delta(
        self,
        shipper: Shipper,
        route: Route,
        order: Order,
        orders: Dict[int, Order],
        pickup_pos: int,
        delivery_pos: int,
        model: VRPModel,
    ) -> float:
        """
        Tính chênh lệch điểm số khi chèn 1 đơn vào route.
        """
        temp_route = route[:]
        pickup_event = ("pickup", order.id, (order.sx, order.sy))
        delivery_event = ("deliver", order.id, (order.ex, order.ey))

        temp_route.insert(pickup_pos, pickup_event)
        temp_route.insert(delivery_pos, delivery_event)

        if not self._is_route_feasible(shipper, temp_route, orders):
            return float("inf")

        new_score = self._evaluate_route(
            shipper=shipper,
            route=temp_route,
            orders=orders,
            t=model.t,
            T=model.T,
        )
        old_score = self._evaluate_route(
            shipper=shipper,
            route=route,
            orders=orders,
            t=model.t,
            T=model.T,
        )
        return new_score - old_score

    def _insert_order_to_route(
        self,
        route: Route,
        order: Order,
        pickup_pos: int,
        delivery_pos: int,
    ) -> Route:
        """
        Chèn pickup và deliver của một đơn vào route.
        """
        new_route = route[:]
        pickup_event = ("pickup", order.id, (order.sx, order.sy))
        delivery_event = ("deliver", order.id, (order.ex, order.ey))
        new_route.insert(pickup_pos, pickup_event)
        new_route.insert(delivery_pos, delivery_event)
        return new_route

    def _lns_improvement(
        self,
        model: VRPModel,
        routes: Dict[int, Route],
    ) -> Dict[int, Route]:
        """
        Cải thiện lời giải bằng Large Neighborhood Search.
        Giúp đa dạng hóa lời giải và tránh kẹt ở nghiệm cục bộ.
        """
        if (time.time() - self._plan_start) > self._plan_time_budget * 0.8:
            return routes

        best_routes = {sid: route[:] for sid, route in routes.items()}
        best_score = self._objective(model, best_routes)

        start_lns = time.time()
        for _ in range(self._LNS_NUM_ITERATIONS):
            if (time.time() - start_lns) > self._LNS_TIME_CAP:
                break

            destroyed, removed_orders = self._destroy_routes(routes, model)
            repaired = self._repair_routes_by_insertion(destroyed, removed_orders, model)
            repair_score = self._objective(model, repaired)

            if repair_score > best_score:
                best_routes = {sid: route[:] for sid, route in repaired.items()}
                best_score = repair_score
                routes = {sid: route[:] for sid, route in repaired.items()}

        return best_routes

    def _destroy_routes(
        self,
        routes: Dict[int, Route],
        model: VRPModel,
    ) -> Tuple[Dict[int, Route], List[Order]]:
        """
        Phá một phần route bằng cách xóa ngẫu nhiên một số đơn.
        """
        destroyed = {sid: route[:] for sid, route in routes.items()}
        removed_orders: List[Order] = []

        all_orders_in_routes: List[Tuple[int, int]] = []
        for sid, route in routes.items():
            for kind, oid, _ in route:
                if kind == "pickup" and oid is not None:
                    all_orders_in_routes.append((sid, oid))

        destroy_count = max(1, int(len(all_orders_in_routes) * self._LNS_DESTROY_RATIO))
        orders_to_remove = random.sample(
            all_orders_in_routes,
            min(destroy_count, len(all_orders_in_routes)),
        )

        for sid, oid in orders_to_remove:
            destroyed[sid] = self._remove_order_from_route(destroyed[sid], oid)
            o = model.orders.get(oid)
            if o is not None:
                removed_orders.append(o)

        return destroyed, removed_orders

    def _repair_routes_by_insertion(
        self,
        routes: Dict[int, Route],
        removed_orders: List[Order],
        model: VRPModel,
    ) -> Dict[int, Route]:
        """
        Sửa lời giải sau khi phá bằng cách chèn lại các đơn.
        """
        repaired = {sid: route[:] for sid, route in routes.items()}
        shipper_list = sorted(model.shippers, key=lambda s: s.id)

        sorted_orders = sorted(removed_orders, key=lambda o: (o.et, -o.p, o.w))

        for order in sorted_orders:
            best_shipper = None
            best_delta = float("inf")
            best_pos = (0, 0)

            for shipper in shipper_list:
                sid = shipper.id
                current_route = repaired.get(sid, [])

                delta, p_pos, d_pos = self._find_best_insertion(
                    shipper=shipper,
                    route=current_route,
                    order=order,
                    orders=model.orders,
                    model=model,
                )

                if delta < best_delta:
                    best_delta = delta
                    best_shipper = sid
                    best_pos = (p_pos, d_pos)

            if best_shipper is not None and best_delta < float("inf"):
                route = repaired[best_shipper]
                repaired[best_shipper] = self._insert_order_to_route(
                    route, order, best_pos[0], best_pos[1]
                )

        return repaired

    def _cross_route_improvement(
        self,
        model: VRPModel,
        routes: Dict[int, Route],
    ) -> Dict[int, Route]:
        """
        Tối ưu bằng cách di chuyển đơn giữa các shipper.
        """
        if (time.time() - self._plan_start) > self._plan_time_budget * 0.85:
            return routes

        best_routes = {sid: route[:] for sid, route in routes.items()}
        best_score = self._objective(model, best_routes)

        shipper_list = sorted(model.shippers, key=lambda s: s.id)
        improved = True
        local_start = time.time()

        while improved and (time.time() - local_start) <= self._MOVE_TIME_CAP:
            improved = False

            for i, s1 in enumerate(shipper_list):
                for s2 in shipper_list[i + 1 :]:
                    cand = self._try_cross_relocate(model, best_routes, s1, s2)
                    if cand is not None:
                        cand_score = self._objective(model, cand)
                        if cand_score > best_score:
                            best_routes = cand
                            best_score = cand_score
                            improved = True
                            break
                if improved:
                    break

        return best_routes

    def _try_cross_relocate(
        self,
        model: VRPModel,
        routes: Dict[int, Route],
        s1: Shipper,
        s2: Shipper,
    ) -> Optional[Dict[int, Route]]:
        """
        Thử chuyển một đơn từ route của s1 sang route của s2.
        """
        r1 = routes.get(s1.id, [])
        r2 = routes.get(s2.id, [])

        if not r1:
            return None

        order_id = None
        for kind, oid, _ in r1:
            if kind == "pickup" and oid is not None:
                order_id = oid
                break

        if order_id is None:
            return None

        order = model.orders.get(order_id)
        if order is None:
            return None

        new_r1 = self._remove_order_from_route(r1, order_id)
        if new_r1 is None:
            return None

        delta, p_pos, d_pos = self._find_best_insertion(
            shipper=s2,
            route=r2,
            order=order,
            orders=model.orders,
            model=model,
        )
        if delta >= float("inf"):
            return None

        if not self._is_route_feasible(s1, new_r1, model.orders):
            return None

        new_r2 = self._insert_order_to_route(r2, order, p_pos, d_pos)
        if not self._is_route_feasible(s2, new_r2, model.orders):
            return None

        candidate = {sid: route[:] for sid, route in routes.items()}
        candidate[s1.id] = new_r1
        candidate[s2.id] = new_r2
        return candidate

    def _local_search_improvement(
        self,
        model: VRPModel,
        routes: Dict[int, Route],
        assigned: Dict[int, List[int]],
    ) -> Dict[int, Route]:
        """
        Cải thiện lộ trình của từng shipper bằng tìm kiếm cục bộ trên các bundle.
        """
        shipper_list = sorted(model.shippers, key=lambda s: s.id)
        improved_routes: Dict[int, Route] = {}

        for shipper in shipper_list:
            sid = shipper.id
            base_route = routes.get(sid, [])
            base_score = self._evaluate_route(
                shipper=shipper,
                route=base_route,
                orders=model.orders,
                t=model.t,
                T=model.T,
            )

            if (time.time() - self._plan_start) > self._plan_time_budget:
                improved_routes[sid] = base_route
                continue

            bundles = self._build_shipper_bundles(
                shipper=shipper,
                assigned_order_ids=assigned.get(sid, []),
                orders=model.orders,
            )

            if len(bundles) <= 1:
                improved_routes[sid] = base_route
                continue

            refined_bundles = self._bundle_local_search(
                shipper=shipper,
                bundles=bundles,
                orders=model.orders,
            )
            candidate_route = self._route_from_bundle_order(refined_bundles)

            if self._is_route_feasible(shipper, candidate_route, model.orders):
                candidate_score = self._evaluate_route(
                    shipper=shipper,
                    route=candidate_route,
                    orders=model.orders,
                    t=model.t,
                    T=model.T,
                )
                if candidate_score >= base_score:
                    improved_routes[sid] = candidate_route
                    continue

            improved_routes[sid] = base_route

        return improved_routes

    def _build_shipper_bundles(
        self,
        shipper: Shipper,
        assigned_order_ids: List[int],
        orders: Dict[int, Order],
    ) -> List[List[Event]]:
        """
        Nhóm các sự kiện thành từng gói theo vị trí để giảm không gian tìm kiếm.
        """
        bundles: List[List[Event]] = []

        # Nhóm các đơn đang có sẵn trên bag theo điểm giao
        delivery_groups: DefaultDict[Tuple[int, int], List[Order]] = defaultdict(list)
        for oid in shipper.bag:
            o = orders.get(oid)
            if o is None or o.delivered:
                continue
            delivery_groups[(o.ex, o.ey)].append(o)

        for cell in sorted(delivery_groups.keys()):
            group = delivery_groups[cell]
            group.sort(key=lambda o: (o.et, -o.p, o.id))
            bundles.append([("deliver", o.id, cell) for o in group])

        # Nhóm các đơn được gán cho shipper theo điểm pickup
        pickup_groups: DefaultDict[Tuple[int, int], List[Order]] = defaultdict(list)
        for oid in assigned_order_ids:
            o = orders.get(oid)
            if o is None or o.delivered or o.picked:
                continue
            pickup_groups[(o.sx, o.sy)].append(o)

        for cell in sorted(pickup_groups.keys()):
            group = pickup_groups[cell]
            group.sort(key=lambda o: (-o.p, o.et, -o.w, o.id))

            # Pickup trước
            bundle: List[Event] = [("pickup", o.id, cell) for o in group]

            # Sau đó là các delivery tương ứng
            delivery_orders = sorted(group, key=lambda o: (o.et, -o.p, o.id))
            bundle.extend(("deliver", o.id, (o.ex, o.ey)) for o in delivery_orders)

            bundles.append(bundle)

        return bundles

    def _assign_orders_to_shippers(
        self,
        model: VRPModel,
    ) -> Dict[int, List[int]]:
        """
        Gán các đơn hàng còn trống cho shipper dựa trên mức độ phù hợp và các ràng buộc về năng lực
        """
        shipper_list = sorted(model.shippers, key=lambda s: s.id)
        shipper_by_id = {s.id: s for s in shipper_list}

        open_orders = [o for o in model.open_orders if not o.picked and not o.delivered]
        ranked = self._rank_candidates(model, open_orders)[: self._MAX_GLOBAL_CANDIDATES]

        assigned: Dict[int, List[int]] = {s.id: [] for s in shipper_list}
        shipper_load_w: Dict[int, float] = {}
        shipper_load_k: Dict[int, int] = {}

        # Khởi tạo tải hiện tại của từng shipper
        for shipper in shipper_list:
            shipper_load_w[shipper.id] = sum(model.orders[oid].w for oid in shipper.bag if oid in model.orders)
            shipper_load_k[shipper.id] = sum(
                1 for oid in shipper.bag if oid in model.orders and not model.orders[oid].delivered
            )

        unassigned: Set[int] = {o.id for o in ranked}
        pickup_cell_to_ids: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        for o in ranked:
            pickup_cell_to_ids[(o.sx, o.sy)].append(o.id)

        for order in ranked:
            if order.id not in unassigned:
                continue

            best = None
            pickup = (order.sx, order.sy)
            shipper_candidates = self._candidate_shippers_for_order(shipper_list, pickup)

            for shipper in shipper_candidates:
                sid = shipper.id
                if shipper_load_k[sid] >= shipper.K_max:
                    continue
                if shipper_load_w[sid] + order.w > shipper.W_max:
                    continue

                # Ước lượng nhanh lợi ích nếu shipper này nhận đơn
                approx_gain = self._approx_order_gain(shipper, order, model)
                key = (
                    approx_gain,
                    -manhattan(shipper.position[0], shipper.position[1], pickup[0], pickup[1]),
                    -shipper_load_k[sid],
                    -shipper.id,
                )
                if best is None or key > best[0]:
                    best = (key, sid)

            if best is None:
                continue

            sid = best[1]
            chosen_shipper = shipper_by_id[sid]

            assigned[sid].append(order.id)
            shipper_load_w[sid] += order.w
            shipper_load_k[sid] += 1
            unassigned.discard(order.id)

            # Nếu pickup cùng ô có thêm đơn khác, cố gắng gom luôn nếu còn capacity
            for other_id in pickup_cell_to_ids[pickup]:
                if other_id == order.id or other_id not in unassigned:
                    continue
                other = model.orders.get(other_id)
                if other is None:
                    continue

                if shipper_load_k[sid] >= chosen_shipper.K_max:
                    break
                if shipper_load_w[sid] + other.w > chosen_shipper.W_max:
                    continue
                if self._approx_order_gain(chosen_shipper, other, model) < 0.0:
                    continue

                assigned[sid].append(other.id)
                shipper_load_w[sid] += other.w
                shipper_load_k[sid] += 1
                unassigned.discard(other.id)

        return assigned

    def _candidate_shippers_for_order(
        self,
        shipper_list: List[Shipper],
        pickup: Tuple[int, int],
    ) -> List[Shipper]:
        """
        Lấy danh sách shipper ứng viên gần điểm pickup nhất.
        """
        if len(shipper_list) <= self._MAX_SHIPPER_CANDIDATES_PER_ORDER:
            return shipper_list

        scored: List[Tuple[int, int, Shipper]] = []
        for shipper in shipper_list:
            d = manhattan(shipper.position[0], shipper.position[1], pickup[0], pickup[1])
            scored.append((d, shipper.id, shipper))

        scored.sort(key=lambda x: (x[0], x[1]))
        return [s for _, _, s in scored[: self._MAX_SHIPPER_CANDIDATES_PER_ORDER]]

    def _rank_candidates(self, model: VRPModel, candidates: List[Order]) -> List[Order]:
        """
        Xếp hạng (score) các đơn mở theo tính hấp dẫn.
        """
        if not candidates:
            return []

        nearest_by_cell: Dict[Tuple[int, int], int] = {}
        for o in candidates:
            cell = (o.sx, o.sy)
            if cell not in nearest_by_cell:
                nearest_by_cell[cell] = min(
                    manhattan(s.position[0], s.position[1], cell[0], cell[1]) for s in model.shippers
                )

        def key(o: Order):
            pickup = (o.sx, o.sy)
            drop = (o.ex, o.ey)
            nearest_shipper_d = nearest_by_cell[pickup]
            travel = nearest_shipper_d + manhattan(pickup[0], pickup[1], drop[0], drop[1])
            est_t = model.t + travel
            est_reward = delivery_reward(o, est_t, model.T)
            late_risk = max(0, est_t - o.et) / max(1, model.T)

            affinity = (
                1.25 * est_reward
                - 0.28 * travel
                - 12.0 * late_risk
                + 0.50 * o.p
                - 0.010 * nearest_shipper_d
            )
            return (-affinity, o.et, nearest_shipper_d, o.w, o.id)

        return sorted(candidates, key=key)

    def _approx_order_gain(self, shipper: Shipper, order: Order, model: VRPModel) -> float:
        """
        Ước lượng nhanh (heuristic) lợi ích của một đơn đối với một shipper.
        """
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = manhattan(shipper.position[0], shipper.position[1], pickup[0], pickup[1])
        d2 = manhattan(pickup[0], pickup[1], drop[0], drop[1])
        est_t = model.t + d1 + d2
        reward = delivery_reward(order, est_t, model.T)
        late_risk = max(0, est_t - order.et) / max(1, model.T)
        return reward - 0.20 * (d1 + d2) - 10.0 * late_risk + 0.35 * order.p

    def _route_from_bundle_order(self, bundle_order: List[List[Event]]) -> Route:
        """
        Ghép danh sách bundles thành route phẳng.
        """
        return [ev for bundle in bundle_order for ev in bundle]

    def _bundle_local_search(
        self,
        shipper: Shipper,
        bundles: List[List[Event]],
        orders: Dict[int, Order],
    ) -> List[List[Event]]:
        """
        Cải thiện thứ tự bundle bằng local search.
        """
        if len(bundles) < 2:
            return bundles

        model = self._current_model
        t = model.t if model is not None else 0
        T = model.T if model is not None else 1
        best = bundles[:]
        best_score = self._evaluate_route(shipper, self._route_from_bundle_order(best), orders, t, T)
        start_local = time.time()

        improved = True
        while improved and (time.time() - start_local) <= self._MOVE_TIME_CAP:
            improved = False

            # Swap adjacent bundles
            for i in range(len(best) - 1):
                cand = best[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                score = self._evaluate_route(shipper, self._route_from_bundle_order(cand), orders, t, T)
                if score > best_score or (score == best_score and cand < best):
                    best = cand
                    best_score = score
                    improved = True
                    break

            if improved:
                continue

            # Move a bundle to different position
            for i in range(len(best)):
                for j in range(len(best)):
                    if i == j:
                        continue
                    cand = best[:]
                    item = cand.pop(i)
                    if j > i:
                        j -= 1
                    cand.insert(j, item)
                    score = self._evaluate_route(shipper, self._route_from_bundle_order(cand), orders, t, T)
                    if score > best_score or (score == best_score and cand < best):
                        best = cand
                        best_score = score
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Reverse a segment
            if len(best) >= 4:
                for i in range(len(best) - 2):
                    for j in range(i + 1, len(best) - 1):
                        cand = best[:]
                        cand[i : j + 1] = list(reversed(cand[i : j + 1]))
                        score = self._evaluate_route(shipper, self._route_from_bundle_order(cand), orders, t, T)
                        if score > best_score or (score == best_score and cand < best):
                            best = cand
                            best_score = score
                            improved = True
                            break
                    if improved:
                        break

        return best

    def _choose_best_bundle_order(
        self,
        shipper: Shipper,
        bundles: List[List[Event]],
        orders: Dict[int, Order],
    ) -> Route:
        """
         Chọn thứ tự bundle tốt nhất trong giới hạn thời gian.
        """
        if not bundles:
            return []

        if len(bundles) <= self._MAX_ROUTE_BUNDLES_PERMUTE:
            best_bundle_order: Optional[List[List[Event]]] = None
            best_score = float("-inf")
            start_local = time.time()
            model = self._current_model
            t = model.t if model is not None else 0
            T = model.T if model is not None else 1

            for perm in itertools.permutations(bundles):
                if time.time() - start_local > self._MOVE_TIME_CAP:
                    break
                bundle_order = list(perm)
                route = self._route_from_bundle_order(bundle_order)
                score = self._evaluate_route(shipper, route, orders, t, T)
                if score > best_score or (
                    score == best_score
                    and (best_bundle_order is None or bundle_order < best_bundle_order)
                ):
                    best_score = score
                    best_bundle_order = bundle_order

            if best_bundle_order is None:
                best_bundle_order = bundles[:]
        else:
            remaining = bundles[:]
            ordered: List[List[Event]] = []
            pos = shipper.position

            # Greedy: ưu tiên chọn bundle gần vị trí hiện tại
            while remaining:
                best_idx = 0
                best_key = None
                for i, bundle in enumerate(remaining):
                    anchor = bundle[0][2]
                    d = manhattan(pos[0], pos[1], anchor[0], anchor[1])
                    key = (d, anchor[0], anchor[1], len(bundle), i)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = i
                bundle = remaining.pop(best_idx)
                ordered.append(bundle)
                pos = bundle[-1][2]

            best_bundle_order = ordered

        best_bundle_order = self._bundle_local_search(shipper, best_bundle_order, orders)
        return self._route_from_bundle_order(best_bundle_order)

    def _is_route_feasible(
        self,
        shipper: Shipper,
        route: Route,
        orders: Dict[int, Order],
    ) -> bool:
        """
        Kiểm tra route có khả thi (feasible) không.
        """
        if not route:
            return True

        onboard: Set[int] = {
            oid for oid in shipper.bag if oid in orders and not orders[oid].delivered
        }
        load_w = sum(orders[oid].w for oid in shipper.bag if oid in orders)

        if load_w > shipper.W_max or len(onboard) > shipper.K_max:
            return False

        for kind, oid, target in route:
            if oid is None:
                return False

            o = orders.get(oid)
            if o is None or o.delivered:
                return False

            if kind == "pickup":
                if target != (o.sx, o.sy):
                    return False
                if oid in onboard:
                    return False
                if load_w + o.w > shipper.W_max:
                    return False
                if len(onboard) + 1 > shipper.K_max:
                    return False
                onboard.add(oid)
                load_w += o.w

            elif kind == "deliver":
                if target != (o.ex, o.ey):
                    return False
                if oid not in onboard:
                    return False
                onboard.remove(oid)
                load_w -= o.w

            else:
                return False

        return True

    def _sanitize_route_prefix(self, shipper: Shipper, route: Route, orders: Dict[int, Order]) -> Route:
        """
        Xóa các event đầu route đã không còn cần thiết (stale).
        """
        if not route:
            return route

        cleaned = route[:]
        while cleaned:
            kind, oid, _ = cleaned[0]
            if oid is None:
                cleaned = cleaned[1:]
                continue

            o = orders.get(oid)
            if o is None or o.delivered:
                cleaned = cleaned[1:]
                continue

            if kind == "pickup":
                if oid in shipper.bag:
                    cleaned = cleaned[1:]
                    continue
                break

            if kind == "deliver":
                if oid not in shipper.bag and not o.picked:
                    cleaned = cleaned[1:]
                    continue
                break

            cleaned = cleaned[1:]

        return cleaned

    def _remove_order_from_route(self, route: Route, oid: int) -> Optional[Route]:
        """
        Xóa toàn bộ event của một đơn khỏi route.
        """
        if not any(kind == "pickup" and ev_oid == oid for kind, ev_oid, _ in route):
            return None
        return [ev for ev in route if ev[1] != oid]

    def _objective(self, model: VRPModel, routes: Dict[int, Route]) -> float:
        """
        Tính mục tiêu heuristic tổng của toàn bộ hệ thống.
        """
        total = 0.0
        scores: List[float] = []
        for shipper in model.shippers:
            sid = shipper.id
            score = self._evaluate_route(
                shipper=shipper,
                route=routes.get(sid, []),
                orders=model.orders,
                t=model.t,
                T=model.T,
            )
            scores.append(score)
            total += score

        if scores:
            avg = total / max(1, len(scores))
            imbalance = sum(abs(s - avg) for s in scores)
            total -= 0.01 * imbalance

        return total

    def _evaluate_route(
        self,
        shipper: Shipper,
        route: Route,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> float:
        """
        Tính heuristic score nội bộ dùng để so sánh route.
        """
        if not route:
            return 0.0

        cache_key = (shipper.id, tuple((kind, oid) for kind, oid, _ in route), t, T)
        cached = self._route_eval_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._is_route_feasible(shipper, route, orders):
            return float("-inf")

        pos = shipper.position
        cur_t = t
        model = self._current_model

        if model is not None:
            profile = model.profile
        else:
            profile = {
                "reward_w": 1.0,
                "late_w": 0.06,
                "priority_w": 0.07,
                "balance_w": 0.45,
            }

        onboard: Set[int] = {
            oid for oid in shipper.bag if oid in orders and not orders[oid].delivered
        }
        load_w = sum(orders[oid].w for oid in shipper.bag if oid in orders)

        reward_w = profile["reward_w"]
        late_w = profile["late_w"]
        priority_w = profile["priority_w"]
        balance_w = profile["balance_w"]

        score = 0.0
        total_travel = 0

        for kind, oid, target in route:
            # Chi phí di chuyển đến event tiếp theo
            d = self._plan_dist(pos, target)
            score += d * move_cost(load_w, shipper.W_max)
            score -= 0.001 * d
            total_travel += d
            cur_t += d
            pos = target

            o = orders.get(oid)
            if o is None or o.delivered:
                return float("-inf")

            if kind == "pickup":
                if (o.sx, o.sy) != target or oid in onboard:
                    return float("-inf")
                if load_w + o.w > shipper.W_max:
                    return float("-inf")
                if len(onboard) + 1 > shipper.K_max:
                    return float("-inf")

                # Ưu tiên các đơn quan trọng hơn
                score += priority_w * o.p
                score += 0.004 * r_base(o.w)
                onboard.add(oid)
                load_w += o.w

            elif kind == "deliver":
                if oid not in onboard or (o.ex, o.ey) != target:
                    return float("-inf")

                reward = delivery_reward(o, cur_t, T)
                score += reward_w * reward
                score += 0.01 * o.p
                score -= late_w * max(0, cur_t - o.et)
                onboard.remove(oid)
                load_w -= o.w

            else:
                return float("-inf")

        # Phạt nếu kết thúc route mà vẫn còn hàng trên xe
        if onboard:
            score -= balance_w * len(onboard)

        # Phạt nhẹ theo tổng quãng đường
        score -= 0.0005 * total_travel
        self._route_eval_cache[cache_key] = score
        return score


    def _action_from_route(
        self,
        shipper: Shipper,
        route: Route,
        grid,
        occupied_next: Set[Tuple[int, int]],
    ) -> Tuple[str, int]:
        """
        Chuyển route logic thành action thực thi trong 1 step.
        """
        if not route:
            target = self._dispatch_target(shipper)
            mv = self._next_move_avoiding(grid, shipper.position, target, occupied_next)
            return (mv, 0)

        ev_kind, _, target = route[0]
        mv = self._next_move_avoiding(grid, shipper.position, target, occupied_next)
        nxt = valid_next_pos(shipper.position, mv, grid)

        if ev_kind == "pickup":
            if shipper.position == target:
                return ("S", 1)
            if nxt == target:
                return (mv, 1)
            return (mv, 0)

        if ev_kind == "deliver":
            if shipper.position == target:
                return ("S", 2)
            if nxt == target:
                return (mv, 2)
            return (mv, 0)

        return ("S", 0)

    def _dispatch_target(self, shipper: Shipper) -> Tuple[int, int]:
        """
        Chọn mục tiêu dispatch khi shipper không có route rõ ràng.
        """
        model = self._current_model
        if model is None:
            grid = getattr(self.env, "grid", [[0]])
            rows = len(grid)
            cols = len(grid[0]) if rows and isinstance(grid[0], list) else 1
            return (rows // 2, cols // 2)

        open_orders = [o for o in model.open_orders if not o.delivered and not o.picked]
        if not open_orders:
            rows = len(model.grid)
            cols = len(model.grid[0]) if rows else 1
            return (rows // 2, cols // 2)

        ranked = self._rank_candidates(model, open_orders)[:12]
        if not ranked:
            rows = len(model.grid)
            cols = len(model.grid[0]) if rows else 1
            return (rows // 2, cols // 2)

        best_target = (ranked[0].sx, ranked[0].sy)
        best_utility = float("-inf")

        for o in ranked:
            pickup = (o.sx, o.sy)
            d = manhattan(shipper.position[0], shipper.position[1], pickup[0], pickup[1])
            travel = d + manhattan(pickup[0], pickup[1], o.ex, o.ey)
            est_t = model.t + travel
            reward = delivery_reward(o, est_t, model.T)
            utility = reward - 0.14 * d + 0.25 * o.p - 0.002 * manhattan(
                pickup[0], pickup[1], o.ex, o.ey
            )
            if utility > best_utility:
                best_utility = utility
                best_target = pickup

        return best_target

    def _next_move_avoiding(
        self,
        grid,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        occupied_next: Set[Tuple[int, int]],
    ) -> str:
        """
        Tìm bước di chuyển tiếp theo tránh xung đột ô bị chiếm.
        """
        if start == goal:
            return "S"

        path = self._astar_path(grid, start, goal)
        if path:
            mv = path[0]
            nxt = valid_next_pos(start, mv, grid)
            if nxt not in occupied_next:
                return mv

        best = None
        for mv in ["U", "D", "L", "R"]:
            nxt = valid_next_pos(start, mv, grid)
            if nxt == start:
                continue
            if nxt in occupied_next:
                continue
            d = manhattan(nxt[0], nxt[1], goal[0], goal[1])
            key = (d, mv)
            if best is None or key < best:
                best = key

        if best is None:
            return "S"
        return best[1]

    def _plan_dist(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Ước lượng khoảng cách ngắn nhất trên grid bằng A*, có cache và fallback Manhattan nếu không tìm được đường.
        """
        if start == goal:
            return 0

        key = (start, goal)
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached

        rev = (goal, start)
        cached = self._dist_cache.get(rev)
        if cached is not None:
            self._dist_cache[key] = cached
            return cached

        grid = getattr(self._current_model, "grid", getattr(self.env, "grid", [[0]]))
        path = self._astar_path(grid, start, goal)
        d = len(path) if path else manhattan(start[0], start[1], goal[0], goal[1])
        self._dist_cache[key] = d
        self._dist_cache[rev] = d
        return d

    def _astar_path(
        self,
        grid,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[str]:
        """
        Tìm đường đi ngắn nhất nếu tồn tại từ start tới goal bằng A*.
        """
        if len(self._path_cache) > 40000:
            self._path_cache.clear()
            self._dist_cache.clear()

        key = (start, goal)
        if key in self._path_cache:
            return self._path_cache[key]

        if start == goal:
            self._path_cache[key] = []
            return []

        pq: List[Tuple[int, int, Tuple[int, int]]] = []
        counter = 0
        gscore: Dict[Tuple[int, int], int] = {start: 0}
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        visited: Set[Tuple[int, int]] = set()

        h0 = manhattan(start[0], start[1], goal[0], goal[1])
        heapq.heappush(pq, (h0, counter, start))

        dirs = [("U", -1, 0), ("L", 0, -1), ("R", 0, 1), ("D", 1, 0)]

        while pq:
            _, _, cur = heapq.heappop(pq)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal:
                path: List[str] = []
                while cur != start:
                    prev, mv = parent[cur]
                    path.append(mv)
                    cur = prev
                path.reverse()
                self._path_cache[key] = path
                return path

            cur_g = gscore[cur]
            r, c = cur
            for mv, dr, dc in dirs:
                nxt = (r + dr, c + dc)
                if not is_valid_cell(nxt, grid):
                    continue

                ng = cur_g + 1
                if ng >= gscore.get(nxt, 10**9):
                    continue

                gscore[nxt] = ng
                parent[nxt] = (cur, mv)
                counter += 1
                f = ng + manhattan(nxt[0], nxt[1], goal[0], goal[1])
                heapq.heappush(pq, (f, counter, nxt))

        self._path_cache[key] = []
        return []