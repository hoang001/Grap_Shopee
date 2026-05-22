# Bài tập nhóm cuối kì: Tối ưu hóa giao hàng đa tác tử thời gian thực

Bài toán mô phỏng hệ thống giao hàng thực tế: một đội **C shipper** hoạt động đồng thời trên bản đồ lưới **N × N**, nhận và giao các kiện hàng có trọng lượng, mức ưu tiên và deadline khác nhau. Đơn hàng xuất hiện liên tục theo thời gian với tốc độ biến động, bao gồm các đợt cao điểm (**surge**) tập trung tại một số khu vực đặc biệt (**hotspot**), tạo ra nút cổ chai cục bộ.

**Nhiệm vụ:** Thiết kế thuật toán phân công và điều phối shipper để **tối đa hóa tổng phần thưởng** trong **T** bước thời gian, cân bằng giữa giao đúng hạn, xử lý đơn ưu tiên cao và chi phí di chuyển.

> **Thang thời gian:** 1 giờ = 10 đơn vị thời gian | 1 ngày = 240 đơn vị thời gian

---

## Files được cấp

| File | Mô tả |
|------|-------|
| `run_test.py` | Grader chính thức, chấm điểm tự động. **Không được sửa.** |
| `test_config.txt` | 6 config Phase 1 kèm bản đồ. **Không được sửa.** |
| `demo_notebook.ipynb` | Kaggle notebook gọi terminal, không chứa thuật toán. |

---

## Phase 1 — Phát triển và nộp bài

Nhóm nhận `test_config.txt` để phát triển và kiểm tra các thuật toán được yêu cầu:

```bash
python run_test.py --config test_config.txt --out results/ --seed 42
```

Code và báo cáo nộp ở Phase 1 là phiên bản chính thức được đánh giá và chấm. Các nhóm cài đặt các thuật toán trong thư mục `solvers/`, upload thư mục lên Kaggle dạng Dataset và để **public** vào hôm deadline Phase 1 để giảng viên có thể xem.

> **Quy tắc:** Nộp trên Kaggle **1 version duy nhất**, vi phạm sẽ bị trừ điểm.

---

## Phase 2 — Config dùng để Ranking

Ba ngày trước deadline Phase 2, giảng viên sẽ công bố file config test dùng để ranking (cập nhật trực tiếp vào `test_config.txt`). Các notebook đã nộp sẽ không được thay đổi.

---

## 1. Mô tả bài toán

Cho bản đồ dạng lưới **A** kích thước **N × N**, trong đó:
- `A[i][j] = 0`: ô trống
- `A[i][j] = 1`: ô vật cản

với `1 <= i, j <= N`.

Tại `t = 0`, có **C shipper** trên bản đồ. Shipper `i` có toạ độ `(x_i, y_i)`, tải trọng tối đa `W_max(i)` và sức chứa `K(i)` đơn. Không có hai shipper nào đứng cùng ô.

---

### 1.1. Tập hành động

Tại mỗi bước `t`, mỗi shipper thực hiện một cặp hành động `(move, cargo_op)`:

| Thành phần | Giá trị | Mô tả |
|-----------|---------|-------|
| `move` | `S` | Đứng yên |
| | `L` / `R` | Di chuyển Tây / Đông |
| | `U` / `D` | Di chuyển Bắc / Nam |
| `cargo_op` | `0` | Không làm gì |
| | `1` | Nhặt đơn tại ô hiện tại |
| | `2 [id]` | Giao đơn `id` đang mang |

> ⚠️ **Mâu thuẫn với code:** `op=2` trong `env.py` **không nhận `id`**. Hàm `_deliver_many()` giao *tất cả* đơn trong túi có điểm đến khớp vị trí hiện tại, không chỉ định order id cụ thể.

> **Thứ tự trong một bước:** Di chuyển → Nhặt hàng → Giao hàng.

---

### 1.2. Mô hình đơn hàng

Một đơn hàng `g_i` được biểu diễn bởi bộ thuộc tính:

```
g_i = <sx_i, sy_i, ex_i, ey_i, et_i, w_i, p_i>
```

| Thuộc tính | Ý nghĩa |
|-----------|---------|
| `sx_i, sy_i` | Toạ độ điểm lấy hàng |
| `ex_i, ey_i` | Toạ độ điểm giao hàng |
| `et_i` | Deadline (đơn vị thời gian); quá hạn sẽ bị phạt |
| `w_i` | Khối lượng kiện hàng (kg) |
| `p_i ∈ {1, 2, 3}` | Mức ưu tiên: 1 = Tiêu chuẩn, 2 = Nhanh, 3 = Hỏa tốc |

---

### 1.3. Mô hình sinh đơn hàng: Surge & Hotspot

Đơn hàng xuất hiện theo **quá trình Poisson không đồng nhất** với tốc độ `λ(t)`:

- Nếu `t ∈ [t_s, t_e]` (trong surge window): `λ(t) = λ₀ × (1 + A)`
- Ngược lại: `λ(t) = λ₀`

| Tham số | Ý nghĩa |
|---------|---------|
| `λ₀ ≈ G / T` | Tốc độ sinh đơn nền (đơn/bước thời gian) |
| `A >= 0` | Biên độ surge (hệ số khuếch đại tốc độ trong cao điểm) |
| `[t_s, t_e]` | Surge window (khoảng thời gian xảy ra cao điểm) |
| Hotspot `(r, c)` | Tâm khu vực đặc biệt: đơn hàng tập trung mạnh gần đây |

**Cơ chế hotspot:** Trong một surge window, với xác suất **70%**, điểm lấy hàng `(sx, sy)` được chọn ngẫu nhiên trong vùng lân cận Manhattan ≤ 3 quanh một hotspot. Xác suất **30%** còn lại vẫn chọn ngẫu nhiên toàn bản đồ.

```
Bình thường (λ₀ = 0.1):      Trong surge (A = 3.0, λ = 0.4):
  . . . . .                     . . . . .
  . . o . .   <- đơn rải đều    . H H H .   <- đơn tập trung
  . o . . .                     . H * H .      quanh hotspot *
  . . . o .                     . H H H .
  . . . . .                     . . . . .
```

> **Phase 1:** Các tham số `λ₀`, `A`, surge windows và hotspots **không được công bố** nhằm khuyến khích thuật toán thích nghi với môi trường động.
>
> **Phase 2:** Tất cả tham số surge và hotspot được công bố đầy đủ.

---

### 1.4. Sức chứa và trọng lượng

Mỗi shipper `i` phải thỏa mãn đồng thời hai ràng buộc:
- Tổng khối lượng: `Σ w_j ≤ W_max(i)` với `j ∈ bag(i)`
- Số đơn: `|bag(i)| ≤ K(i)`

Khi nhặt hàng tại ô có nhiều đơn, **ưu tiên:** hỏa tốc > nhanh > tiêu chuẩn > chỉ số nhỏ hơn.

> ⚠️ **Mâu thuẫn với code:** `Shipper.pickup_best()` trong `env.py` dùng key `(-o.p, o.et, o.id)`, nghĩa là có thêm **deadline sớm hơn** làm tiêu chí phụ thứ hai, trước khi xét `id`.

| Hạng mục | Khối lượng `w` | Chi phí/bước `rc(w)` | Sức chứa `K` |
|----------|---------------|---------------------|-------------|
| Nhẹ | `w ≤ 3 kg` | `-0.01` | 3 đơn |
| Trung bình | `3 < w ≤ 10 kg` | `-0.02` | 2 đơn |
| Nặng | `10 < w ≤ 30 kg` | `-0.04` | 1 đơn |
| Siêu nặng | `w > 30 kg` | `-0.08` | 1 đơn |

> ⚠️ **Mâu thuẫn với code (cột `rc(w)`):** Bảng này không được dùng trong `env.py`. Chi phí thực tế tính theo công thức liên tục ở mục 1.6: `rc = -0.01 × (1 + γ × W_carried / W_max)`, không phân loại theo hạng mục khối lượng gói hàng.
>
> ⚠️ **Mâu thuẫn với code (cột `K`):** `K_max` là thuộc tính cố định **của shipper**, được set trực tiếp trong config độc lập với `W_max`. Trong `test_config.txt`, shipper với `W_max=30.0` có `K_max=2` hoặc `3` — không theo bảng trên.

---

### 1.5. Hàm phần thưởng

Phần thưởng giao đơn `i` tại thời điểm `t_delivery`:

**Giao đúng hạn** (`t_delivery ≤ et_i`):
```
r(i) = α_p × r_base(i) × (1 + bonus)
bonus = max(0, (et_i - t_delivery) / et_i)
```

**Giao trễ** (`t_delivery > et_i`):
```
r(i) = β_p × r_base(i) × max(0, 1 - (t_delivery - et_i) / T)
```

| Loại dịch vụ | `p` | `α_p` | `β_p` |
|-------------|-----|-------|-------|
| Tiêu chuẩn | 1 | 1.0 | 0.1 |
| Nhanh | 2 | 2.0 | 0.3 |
| Hỏa tốc | 3 | 3.0 | 0.5 |

**Phần thưởng cơ bản** `r_base(i) = 10 × f_weight`:

| Trọng lượng `w_i` | `f_weight` | `r_base` |
|------------------|-----------|---------|
| `w ≤ 0.2 kg` | 0.4 | 4 |
| `0.2 < w ≤ 3 kg` | 1.0 | 10 |
| `3 < w ≤ 10 kg` | 1.5 | 15 |
| `10 < w ≤ 30 kg` | 2.0 | 20 |
| `w > 30 kg` | 3.0 | 30 |

---

### 1.6. Chi phí di chuyển

Chi phí di chuyển của shipper `i` tại bước `t` (chỉ tính khi di chuyển bằng L/R/U/D):

```
rc(i, t) = -0.01 × (1 + γ × W_carried(i, t) / W_max(i))
```

với `γ = 1.0`. Đứng yên (`S`) không mất chi phí.

---

### 1.7. Hàm mục tiêu

```
maximize Σ_i [ Σ reward(đơn giao bởi shipper i) + Σ_t rc(i, t) ]
```

---

### 1.8. Các ràng buộc vận hành

- **Va chạm:** Shipper có chỉ số nhỏ hơn được ưu tiên giữ ô khi tranh chấp.
- **Thứ tự:** Di chuyển → Nhặt hàng → Giao hàng trong mỗi bước.
- Shipper không được ra ngoài bản đồ hoặc vào ô vật cản.
- Cả `W_max(i)` và `K(i)` phải được thỏa mãn mọi lúc.

---

## 2. Các phương pháp cần cài đặt

**Yêu cầu với mỗi phương pháp:**
- Trình bày độ phức tạp thời gian và không gian.
- Phân tích mức độ tối ưu: optimal, near-optimal hoặc heuristic, kèm điều kiện đảm bảo nếu có.
- So sánh kết quả định lượng trên các config Phase 1.

### Bắt buộc — 5 điểm/phương pháp
- **Greedy BFS**
- **VRP + OR-Tools**

### Nâng cao — 2.5 điểm/phương pháp
- **Ant Colony Optimization (ACO)**
- **Multi-Agent Pickup and Delivery với Conflict-Based Search (MAPD-CBS)**

---

## 3. Cách nộp bài

```
submission/
├── solvers/               <- thư mục chứa code các thuật toán (phần duy nhất được sửa)
├── run_test.py            <- KHÔNG SỬA
├── test_config.txt        <- KHÔNG SỬA
├── demo_notebook.ipynb    <- notebook Kaggle submit code
└── report.pdf             <- báo cáo kỹ thuật
```

**Quy tắc Kaggle notebook:**
- Share đúng **1 version**.
- Notebook chạy hoàn toàn qua lệnh terminal (`%%bash`), không chứa thuật toán.
- Seed cố định: `--seed 42`.

---

## 4. Thang điểm

| Hạng mục | Điểm | Mô tả |
|----------|------|-------|
| Greedy BFS | 5 | Cài đặt đúng, chạy được trên tất cả config |
| VRP + OR-Tools | 5 | Cài đặt đúng, chạy được trên tất cả config |
| ACO | 2.5 | Kết quả tốt hơn Greedy BFS, có phân tích |
| MAPD-CBS | 2.5 | Cài đặt đúng, xử lý xung đột đa tác tử |
| Báo cáo kỹ thuật | 5 | Theo yêu cầu mục 4.1 |
| Ranking, Phase 2 | 10 | Nhóm cao nhất = 10 điểm, các nhóm khác tỉ lệ tuyến tính |
| Vấn đáp | 20 | Từng thành viên trình bày và trả lời câu hỏi |
| **Tổng** | **50** | |

### 4.1. Yêu cầu báo cáo kỹ thuật (5 điểm)

- Ghi rõ thành viên (tối đa 3) và phân công đóng góp.
- Mô tả từng thuật toán: nguyên lý, độ phức tạp thời gian/không gian, mức độ tối ưu.
- Bảng so sánh kết quả định lượng (net reward, % đơn đúng hạn, thời gian chạy) trên từng config Phase 1.
- Phân tích trade-off giữa các phương pháp.
- *(Nâng cao)* Mô tả chiến lược ứng phó surge và hotspot.

### 4.2. Điểm ranking (10 điểm)

Dựa trên tổng net reward của `run_test.py` với config Phase 2. Nhóm cao nhất = 10 điểm, các nhóm còn lại tỉ lệ tuyến tính.

**Điều kiện chạy lại:**
- Thời gian tối đa: **60 phút**.
- Không được dùng internet khi chạy.
- Không được sử dụng thêm thư viện ngoài (`pip install` bị chặn).
- Các file trong `solvers/` nếu không phải code phải ghi rõ quá trình tạo ra.

> **Cảnh báo:** Vi phạm các điều kiện hoặc code không chạy được → điểm ranking **0/10**.
