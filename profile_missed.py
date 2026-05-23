"""Profile missed and late orders to find improvement ceiling."""
import copy, sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "solvers"))
from env import DeliveryEnv, SEED, load_config, valid_next_pos, delivery_reward
from greedy_bfs import GreedyBFS
from collections import deque
import hashlib

def stable_seed(name, base=SEED):
    return int(hashlib.md5(f"{base}:{name}".encode()).hexdigest()[:8], 16)

def bfs_dist(grid, a, b):
    if a == b: return 0
    visited = {a}
    queue = deque([(a, 0)])
    while queue:
        cur, d = queue.popleft()
        for move in ("U","D","L","R"):
            nxt = valid_next_pos(cur, move, grid)
            if nxt != cur and nxt not in visited:
                if nxt == b: return d + 1
                visited.add(nxt)
                queue.append((nxt, d + 1))
    return 10**9

configs = load_config("test_config.txt")

for cfg in configs:
    N, T, G = cfg["N"], cfg["T"], cfg["G"]
    seed = stable_seed(str(cfg.get("name", "")))

    env = DeliveryEnv(copy.deepcopy(cfg), seed=seed)
    solver = GreedyBFS(env)
    result = solver.run()

    grid = env.grid
    all_orders = env.orders  # all orders after run

    delivered = [o for o in all_orders.values() if o.delivered]
    missed    = [o for o in all_orders.values() if not o.delivered]

    # ── On-time vs late for delivered orders ──
    on_time  = [o for o in delivered if o.deliver_t <= o.et]
    late     = [o for o in delivered if o.deliver_t >  o.et]

    # ── Potential reward if delivered optimally (at appear_t + min_travel) ──
    actual_reward = sum(delivery_reward(o, o.deliver_t, T) for o in delivered)
    optimal_reward = 0.0
    for o in delivered:
        d_min = bfs_dist(grid, (o.sx, o.sy), (o.ex, o.ey))
        best_t = o.appear_t + d_min  # best possible delivery time
        optimal_reward += delivery_reward(o, best_t, T)

    # ── Missed order analysis ──
    tight, near_tight, feasible = 0, 0, 0
    for o in missed:
        d_min = bfs_dist(grid, (o.sx, o.sy), (o.ex, o.ey))
        margin = (o.et - o.appear_t) - d_min
        if margin <= 0:    tight += 1
        elif margin <= 5:  near_tight += 1
        else:              feasible += 1

    # ── Potential from missed feasible orders ──
    missed_potential = 0.0
    for o in missed:
        d_min = bfs_dist(grid, (o.sx, o.sy), (o.ex, o.ey))
        margin = (o.et - o.appear_t) - d_min
        if margin > 0:
            best_t = o.appear_t + d_min
            missed_potential += delivery_reward(o, best_t, T)

    print(f"\n{'='*60}")
    print(f"{cfg['name']} (N={N}, T={T}, G={G})   net={result['net_reward']:.1f}")
    print(f"  Delivered: {len(delivered)}/{G}  ({len(on_time)} on-time, {len(late)} late)")
    if late:
        late_slacks = [o.deliver_t - o.et for o in late]
        print(f"  Late by:   avg={sum(late_slacks)/len(late_slacks):.1f}  max={max(late_slacks)} steps")
    print(f"  Actual reward from delivered:  {actual_reward:.1f}")
    print(f"  Optimal reward (best timing):  {optimal_reward:.1f}  (+{optimal_reward-actual_reward:.1f})")
    print(f"  Missed: {len(missed)} = {tight} impossible + {near_tight} near-tight + {feasible} feasible")
    print(f"  Missed feasible potential:     +{missed_potential:.1f}")
    print(f"  TOTAL CEILING:                 ~{actual_reward + (optimal_reward-actual_reward) + missed_potential:.1f}")
