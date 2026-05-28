"""run_one.py — chạy thử solver trên ĐÚNG MỘT config (theo tên), đo tỉ lệ + thời gian.

Dùng cho N lớn: chạy từng config một để ước lượng thời gian, tránh timeout khi gộp.

Ví dụ:
    python3 run_one.py --name F1
    python3 run_one.py --name F10 --config test_config_v2.txt --seed 42
"""
import sys, hashlib, time, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'solvers')
from env import DeliveryEnv, load_config
from mapd_cbs_solver import MAPDCBSSolver

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True, help='Tên config, vd F1, F10, C5')
ap.add_argument('--config', default='test_config_v2.txt')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--progress', type=int, default=10,
                help='In dòng tiến độ mỗi N bước (0 = tắt). Mặc định 10.')
a = ap.parse_args()

matches = [c for c in load_config(a.config) if c['name'] == a.name]
if not matches:
    sys.exit(f"Không tìm thấy config '{a.name}' trong {a.config}.")
cfg = matches[0]
seed = int(hashlib.md5(f'{a.seed}:{a.name}'.encode()).hexdigest()[:8], 16)
solver = MAPDCBSSolver(DeliveryEnv(cfg, seed=seed))
solver._progress_every = a.progress
t = time.time()
r = solver.run()
el = time.time() - t
print(f"{a.name} N={cfg['N']} C={cfg['C']} G={cfg['G']} T={cfg['T']}: "
      f"giao={r['delivery_rate']:.0f}% ot={r['on_time_rate']:.0f}% "
      f"net={r['net_reward']:.0f} t={el:.1f}s")
