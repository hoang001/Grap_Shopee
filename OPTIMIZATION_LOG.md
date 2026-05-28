# Nhật ký tối ưu thuật toán MAPD-CBS

Tài liệu này ghi lại **toàn bộ quá trình** tối ưu solver `solvers/mapd_cbs_solver.py`,
theo trình tự thời gian. Mỗi giai đoạn nêu rõ: (1) ta làm gì, (2) cải thiện đo được,
(3) vấn đề mới phát sinh, (4) cách xử lý. Mục đích là để hiểu *vì sao* thuật toán
có hình dạng hiện tại — đây là một chuỗi thử–sai có căn cứ, không phải lời giải
xuất hiện ngay từ đầu.

> **Nguyên tắc xuyên suốt:** không tối ưu để tối đa net_reward trên một bộ config
> cố định, mà tối ưu để thuật toán **thích nghi với môi trường động**. Tiêu chí
> tối thiểu: ≥70% %giao **và** ≥70% %đúng hạn, đạt trên ~80% số config được test.
> Solver **không bao giờ đọc** tham số surge/hotspot.

---

## 0. Hạ tầng đánh giá (điều kiện tiên quyết)

Trước khi sửa thuật toán, cần một thước đo độ bền. Grader chính thức (`run_test.py`)
chỉ chạy **một** seed cố định — không đo được khả năng thích nghi. Vì vậy ta dựng:

- **`eval_dynamic.py`** — chạy cùng bộ config trên **nhiều `base_seed`**. Mỗi seed
  là một "kịch bản đơn hàng" khác nhau (thời điểm đơn xuất hiện, vị trí, ưu tiên,
  deadline, và realization surge/hotspot). Báo cáo tỉ lệ *pass* theo tiêu chí
  70/70. Đây là cách phát hiện overfit: một thuật toán ăn may 1 seed sẽ lộ ngay
  khi đổi seed.
- **`generate_configs_v3.py` → `test_config_v3.txt`** — bộ 20 config (N≤30) đa dạng
  map (open, sparse, maze, ring, bottleneck, rooms, divided, grid-obstacles) và đa
  dạng động (nhiều mức surge_amplitude, nhiều vị trí/thời điểm hotspot). Surge/hotspot
  được khai báo *trong file config* (hợp lệ — đó là dữ liệu test), solver vẫn không đọc.

**Vì sao quan trọng:** mọi quyết định tối ưu phía sau đều dựa trên đối chiếu hai
nguồn — `test_config.txt` (6 config gốc, dùng grader thật) và `test_config_v3.txt`
(đa seed, đo độ bền). Một thay đổi chỉ được giữ nếu tốt trên **cả hai**.

---

## 1. Hiện trạng ban đầu (baseline)

Thuật toán cũ (v11) là một hybrid: CBS (space-time A* + conflict tree) cho lớp
đường đi, cộng heuristic gán đơn theo reward-density + opportunistic pickup.

**Đo được:**

| Nguồn | Kết quả |
|-------|---------|
| `test_config` tổng net | ~4134 |
| C3 / C6 %giao | 30% / 48% |
| C6 thời gian | **33.3s** |
| `test_config_v3` (seed 42, N≤16) | pass **54.5%**; V9 (maze N=15) mất **43.7s** |

**Hai vấn đề lớn lộ ra ngay:**

1. **Không mở rộng được (scalability).** C6 mất 33s, V9 maze mất 43.7s cho *một* map
   N=15. Phase 2 có N≤100 → gần như chắc chắn timeout (giới hạn 60 phút).
2. **Overfit + kém thích nghi.** Docstring chứa số liệu per-config ("C3: 37.5%
   structural cap", "shipper start ở (1,1)…"), hằng số `horizon`/`max_nodes` dán
   nhãn theo đúng dải N/C của 6 config Phase 1. Trên môi trường động chỉ pass 54.5%.

**Nguyên nhân gốc của sự chậm:**
- BFS khoảng cách được tính lại **từ vị trí shipper mỗi bước** — mà vị trí shipper
  đổi liên tục → cache gần như không hit → mỗi bước một BFS toàn map.
- `sta_star` dùng heuristic **Manhattan**. Trên maze (nhiều tường), Manhattan đánh
  giá thấp xa thực tế → A* nở tung số trạng thái.

---

## 2. Giai đoạn 1 — Viết lại lõi để mở rộng được

**Thay đổi:**

1. **BFS chỉ từ ô TĨNH.** Bản đồ tĩnh, nên chỉ BFS từ các điểm pickup/delivery
   (số lượng bị chặn bởi số đơn) và cache **vĩnh viễn**. Khoảng cách
   `shipper → điểm` lấy từ `dist_from(điểm)[shipper_pos]` (BFS đối xứng trên lưới
   vô hướng). **Không bao giờ** BFS từ vị trí shipper nữa.
2. **Heuristic chính xác.** `sta_star` dùng chính trường khoảng cách BFS tới đích
   làm heuristic → admissible & consistent & *chính xác* → A* gần như tuyến tính,
   kể cả trên maze.
3. **Bỏ tinh chỉnh theo config.** `horizon`/`max_nodes` chuyển sang biểu thức theo
   quy mô bài toán, không dán nhãn dải N/C.

**Cải thiện:**

| | Trước | Sau |
|--|------|-----|
| `test_config` tổng net | 4134 | 3974 (-4%) |
| `test_config` tổng thời gian | 43s | **3s (14×)** |
| C6 thời gian | 33.3s | **1.7s** |
| V9 maze (N=15) | 43.7s | **0.27s** |
| v3 (3 seed, N≤20) pass | — | **60.4%** |

Net giảm nhẹ 4% (bản cũ batching aggressive hơn, được tune cho 6 config này) nhưng
**đổi lại tốc độ gấp 14×** — điều kiện sống còn cho Phase 2.

**Vấn đề mới phát sinh:** chạy v3 đa seed lộ ra **lỗi throughput nghiêm trọng**:
- **V14** (N=20, G=100, C=5, **không** surge) chỉ giao **38%** — tải này lẽ ra rất
  dễ (mỗi shipper ~20 đơn/720 bước). Một config không-surge mà giao thấp ⇒ lỗi
  thuật toán, không phải do tải.
- Chẩn đoán `max_nodes` của CBS: kết quả C6 **dao động hỗn loạn** (1→81%, 30→51%,
  60→84%, 150→50%). Phi đơn điệu = dấu hiệu hệ thống bất ổn định.

---

## 3. Giai đoạn 2 — Truy vết nguyên nhân throughput

**Đo cấu trúc chuyến đi (V14, C6):**
- `idle_ratio` chỉ 17% (shipper bận di chuyển 83% thời gian) nhưng giao rất ít.
- **`retarget` = 24%**: cứ 4 bước thì shipper đổi mục tiêu 1 lần.
- 49/100 đơn **chưa từng được nhặt** dù deadline còn kịp.

**Hai thủ phạm:**

1. **Thrashing mục tiêu.** `_target` tính lại mỗi bước; opportunistic pickup và
   reposition liên tục đổi đích → shipper dao động qua lại, lãng phí bước, và
   *khuếch đại* nhiễu của lớp đường đi.
2. **CBS reroute có hại.** Một xung đột vertex ở thời điểm t nếu để nguyên chỉ tốn
   **1 bước "giữ ô"** (env tự xử lý). Nhưng CBS "giải quyết" nó bằng cách cấm ô đó
   → ép agent đi **đường vòng dài**, làm hỏng thời gian giao hàng. Tùy `max_nodes`,
   các xung đột khác nhau bị reroute khác nhau → kết quả ngẫu nhiên hỗn loạn.

**Kiểm chứng:** thử `_plan_moves_big` (greedy 1-bước) cho mọi config → tệ hơn
(C5 10%): planner 1-bước myopic gây kẹt. Thử `_priority_plan` (CBS với max_nodes=1,
tức prioritized planning thuần, đặt chỗ theo *toàn tuyến* không-thời gian) → C6 lên
81%. ⇒ **prioritized planning ≫ conflict-tree** cho bài toán này.

---

## 4. Giai đoạn 3 — Sticky goal + Prioritized planning (bước nhảy lớn nhất)

**Thay đổi:**

1. **Sticky goal.** Mỗi shipper giữ mục tiêu hiện hành; chỉ tính lại khi mục tiêu
   *mất hiệu lực* (đã tới nơi, đơn đã được nhặt/giao, hết chỗ chứa). Triệt tiêu
   thrashing tận gốc. Opportunistic vẫn xảy ra nhưng *cam kết một lần* mỗi chặng,
   không dao động.
2. **Default planner = `_priority_plan`** (lớp thấp của CBS + thứ tự ưu tiên + đặt
   chỗ toàn tuyến). Bỏ conflict-tree khỏi đường chạy mặc định vì nó reroute có hại.
   CBS đầy đủ (`_cbs`) vẫn được **cài đặt và giữ lại** (đáp ứng yêu cầu đề bài về
   "xử lý xung đột đa tác tử", có thể bật cho quy mô nhỏ).

**Cải thiện (rất lớn):**

| | Trước | Sau |
|--|------|-----|
| C3 %giao | 35% | **92.5%** |
| C6 %giao | 51% | **90%** |
| `test_config` tổng net | 3974 | **4914 (+24%)** |
| v3 (3 seed, N≤20) pass | 60.4% | **85.4%** |
| V14 / V15 %giao | 38% / 21% | **91% / 91%** |

Throughput đã được giải quyết: %giao giờ cao và ổn định (~90%) ở hầu hết config.

**Vấn đề mới phát sinh:** phần thất bại còn lại gần như **toàn bộ là %đúng hạn**
(không phải %giao). Nặng nhất ở map lớn: V15 (ring) đúng hạn 44-66%, và **giảm dần
khi N tăng** (V18–V20 với N=25–30: giao 88-94% nhưng đúng hạn chỉ 50-62%). Trên
`test_config`, C5 đúng hạn 69.9% — trượt ngưỡng đúng **0.1%**.

---

## 5. Giai đoạn 4 — Truy vết nguyên nhân %đúng hạn

Câu hỏi then chốt: lateness trên map lớn là **giới hạn cấu trúc** (deadline quá
ngắn so với quãng đường) hay **dư địa thuật toán**?

**Đo trần đúng hạn lý thuyết** (giả định shipper có mặt *ngay* tại điểm lấy khi đơn
xuất hiện): trần = **89–96%** trên V15/V17/V18/V19/V20. ⇒ **Không** phải giới hạn
cấu trúc — còn rất nhiều dư địa.

**Đo độ trễ nhặt hàng** (pickup latency = thời điểm nhặt − thời điểm đơn xuất hiện):
V20 trung bình **31 bước** (median 19, max 526); V15 trung bình 38. Với deadline
đơn gấp (offset chỉ 10–60 đơn vị), trễ nhặt 20–30 bước là đủ làm hỏng đúng hạn.

⇒ **Kết luận: %đúng hạn mất ở độ trễ nhặt hàng**, không phải ở khâu giao. Hướng
sửa: giảm độ trễ giữa lúc đơn xuất hiện và lúc shipper tới lấy.

---

## 6. Giai đoạn 5 — EDF assignment (giữ lại)

**Thay đổi:** đổi thứ tự gán đơn từ thuần reward-density sang **EDF (Earliest
Deadline First)**. Khóa sắp xếp: `(còn-kịp-đúng-hạn?, dispatch_slack tăng dần,
-density)`. Nghĩa là đơn **còn kịp nhưng gấp nhất** giành shipper gần nhất trước,
rồi mới tới đơn có reward-density cao. Đây là nguyên lý lập lịch deadline kinh điển,
không phụ thuộc config.

**Cải thiện:**

| | Trước | Sau |
|--|------|-----|
| C5 đúng hạn | 69.9% (trượt) | **76.0% (đạt)** |
| C5 %giao | 91.2% | 93.8% |
| `test_config` tổng net | 4914 | **4935** (cao nhất) |
| `test_config` đạt 70/70 | 5/6 | **6/6** |

EDF là một thắng lợi có nguyên lý: cả 6 config gốc giờ đều đạt 70/70, net cao nhất,
v3 gần như không đổi.

---

## 7. Hai ngõ cụt (đã thử và loại bỏ — minh chứng cho quá trình)

Hai ý tưởng dưới đây *nghe hợp lý* nhưng khi đo thì làm hại độ bền → đã **revert**.
Ghi lại để cho thấy quyết định dựa trên số liệu, không theo cảm tính.

### 7a. Giới hạn detour của opportunistic theo slack
**Giả thuyết:** chỉ nhặt thêm đơn khi detour không làm đơn đang mang trễ.
**Kết quả:** giúp C5 nhưng **regression nặng V13** (bottleneck: giao 93.8%→40%) và
*không* sửa được V15. Lý do: trên map đường vòng dài, slack tính theo đường-ngắn-nhất
quá lạc quan; chặn cứng detour khiến throughput sụp ở nơi *bắt buộc* phải batch.
→ **Revert.**

### 7b. "Tạt qua nhặt đơn ngay cạnh" (grab-nearby)
**Giả thuyết:** đang đi giao, nếu có đơn chưa-gán cách ≤2 ô thì tạt lấy để cắt độ
trễ nhặt.
**Kết quả:** bản không-guard giúp v3 (76.7%) nhưng làm *mỏng* biên đúng hạn của
`test_config` (C2, C5 chỉ còn 70.7-70.8% — quá rủi ro cho Phase 2). Bản có slack-guard
thì **backfire** (C6 đúng hạn 63.7% — rớt) vì guard dựa trên khoảng cách lạc quan.
→ **Revert.** Coi việc giảm độ trễ nhặt trên map lớn là *hướng phát triển tiếp*.

---

## 8. Trạng thái hiện tại (chốt)

**`test_config.txt` (grader thật, seed 42):**

| Config | %Giao | %Đúng hạn | Net | t |
|--------|-------|-----------|-----|---|
| C1 | 86.7 | 100.0 | 266.9 | ~0s |
| C2 | 96.0 | 79.2 | 367.4 | ~0s |
| C3 | 92.5 | 89.2 | 799.1 | 0.1s |
| C4 | 93.3 | 83.9 | 939.9 | 0.2s |
| C5 | 93.8 | 76.0 | 1238.7 | 0.4s |
| C6 | 85.0 | 78.8 | 1323.3 | 0.6s |
| **Tổng net** | | | **4935** | **<2s** |

→ **6/6 config đạt 70/70.**

**`test_config_v3.txt` (3 seed, N≤30, 20 config):**
- Pass (config,seed): **43/60 = 71.7%**
- Config đạt mọi seed: 11/20
- Toàn bộ N≤30 chạy < 2.9s/episode (không timeout).

**So với baseline ban đầu:** net 4134→4935 (**+19%**) trong khi thời gian
43s→<2s (**>20×**), và độ bền trên môi trường động 54.5%→71.7%.

---

## 9. Hành trình tóm tắt một dòng mỗi bước

1. Dựng harness đa-seed + bộ config động → **đo được độ bền** (điều kiện tiên quyết).
2. Baseline: nhanh nhưng overfit & **chậm/không scale** (C6 33s, V9 43.7s).
3. BFS-từ-ô-tĩnh + heuristic chính xác → **14× nhanh hơn**; lộ lỗi throughput (V14 38%).
4. Truy vết: **thrashing mục tiêu (24%)** + **CBS reroute có hại** (kết quả hỗn loạn).
5. **Sticky goal + prioritized planning** → C3 35→92.5, C6 51→90, net **+24%**, v3 60→85%.
6. Lộ vấn đề mới: **%đúng hạn** tụt theo N. Truy vết → trần lý thuyết 94%, mất ở **độ trễ nhặt**.
7. **EDF assignment** → C5 đạt ngưỡng, **6/6 test_config pass**, net cao nhất 4935.
8. Thử slack-cap & grab-nearby → **ngõ cụt, revert** (overfit/backfire).

## 9b. Điều tra sâu độ trễ nhặt hàng (4 thử nghiệm, đều không phải cải tiến sạch)

Sau khi xác định %đúng hạn trên map lớn mất ở độ trễ nhặt, ta thử bốn đòn bẩy.
Tất cả đo trên cả `test_config` và `test_config_v3` (nhiều seed). Kết luận: không
hướng nào thắng *bền vững*, nên **giữ nguyên EDF**. Ghi lại để minh chứng quá trình.

1. **Tắt opportunistic pickup.** %giao gần như không đổi (88-93%) ⇒ hệ thống
   **không thực sự capacity-bound**, batching mang lại rất ít. Đây là phát hiện
   quan trọng: latency không do thiếu năng lực chở, mà do *phản ứng chậm*.
2. **Shipper rỗi phản ứng tức thì (non-sticky khi bag rỗng).** Marginal, và
   **làm sập V15 ring** (giao 91→52%) do dao động trên map có vòng/chokepoint.
3. **Gán theo gần-nhất (nearest-first) thay EDF.** Hỏng C5 (giao 53.8% — bỏ đơn
   gấp ở xa). Loại.
4. **Khóa urgency "blend" = et − d_deliver − t** (độ gấp nội tại, không lệch theo
   vị trí shipper). Net `test_config` nhỉnh hơn (4988 vs 4935) và C6 giao 85→91%,
   **nhưng** v3 kém bền hơn (70.0% vs 71.7%). Phần thắng nằm ở net của *config cố
   định* — đúng thứ cần tránh tối ưu — nên revert về EDF.

**Bài học:** trên map có chokepoint, mọi cơ chế "phản ứng lại đơn mới mỗi bước"
đều gây dao động và làm sập throughput. Giảm độ trễ nhặt an toàn đòi hỏi lập
**tuyến (route) có kiểm tra khả thi deadline** thay vì heuristic phản ứng — đây là
hướng phát triển chính, chưa làm vì rủi ro regression cao.

## 10. Giai đoạn 6 — Bốn sửa đổi có nguyên lý (phiên làm việc mới)

Xuất phát: net `test_config` 4935 (6/6 pass), v3 (3 seed) 71.7%, config-pass 11/20.
Truy vết failures cho thấy hai lỗi cấu trúc còn lại (không phải giới hạn tải) và
hai đòn bẩy %đúng hạn an toàn.

### 10a. Lỗi STARVATION pipeline gán đơn (sửa)
**Triệu chứng:** V2 seed7 chỉ giao 36% trên map 10×10/C=2 — 13/25 đơn KHÔNG bao giờ
được nhặt; đo idle-empty cho thấy khi shipper rảnh thì *không còn đơn khả dụng*,
trong khi backlog đầy.
**Nguyên nhân:** `_assign_tasks` cho shipper còn-chỗ-trong-bag (`len(bag)<K_max`)
nhận thêm assignment, NHƯNG `_target` chỉ theo đuổi assignment khi bag RỖNG (nhánh
bag-non-empty đi giao trước, lại còn loại chính đơn đã-gán khỏi opportunistic). Đơn
bị "giữ chỗ" bởi shipper đang bận → shipper khác không lấy được → đơn mới thấy
`free_s` rỗng nên KHÔNG bao giờ được gán.
**Sửa:** chỉ gán cho shipper bag RỖNG (`len(s.bag)==0`) — assignment luôn đi kèm
theo-đuổi-thực. Batching để opportunistic lo.
**Kết quả:** net 4935→5165, v3 71.7→78.3%, config-pass 11→14.

### 10b. DEADLOCK hành lang một-ô (sửa)
**Triệu chứng:** sau 10a, V13 seed123 SỤP còn 5% giao. Trace: 5 shipper đóng băng
vĩnh viễn quanh ô cổng (9,9) nối hai nửa bản đồ — chu trình swap hai chiều mà
prioritized planning không giải được (env chỉ "giữ-ô" khi tranh chấp → hai agent
đối đầu kẹt mãi).
**Sửa:** `_break_deadlocks` — đo dịch-chuyển-THỰC giữa các bước (không phải bước
"định đi"), nếu một shipper muốn-đi mà đứng-yên ≥3 bước thì ép NÉ sang một ô kề
trống bất kỳ (kể cả lùi xa đích) để mở khe phá chu trình. Tổng quát, không gắn
với cấu trúc map.
**Kết quả:** V13 s123 5→92%, V15 (ring) giao 70→~91%, v3 78.3→81.7%.

### 10c. Ưu tiên đường đi theo DEADLINE
**Thay đổi:** `_priority_plan` lập đường theo thứ tự độ-gấp-deadline (đơn đang
mang / được gán gấp nhất đi trước) thay vì theo shipper-id. Shipper gấp giành
đường ngắn, shipper khác nhường.
**Kết quả:** net 5165→5305, C3/C4/C6 %đúng hạn đều vượt 80, v3 config-pass 14→15.

### 10d. Gom đơn: trần 2 + detour rẻ (`_opp_max=2`, `_detour_f=0.25`)
**Lý do:** bag nhỏ → shipper trống nhanh → phản ứng kịp đơn gấp (↑%đúng hạn); chỉ
tạt qua đơn gần-như-trên-đường (detour rẻ) để KHÔNG làm trễ đơn đang mang nhưng
VẪN giữ throughput cho map ít shipper (V2). Cả hai tham số không-thứ-nguyên.
**Kết quả (chốt):** v3 (seed 42/7/123) **86.7%, config-pass 16/20**; kiểm chứng
seed lạ (1/55/777/2024) **83.8%** — không overfit seed.

### 10e. Admission control đơn-trễ — THỬ rồi LOẠI (vi phạm tiêu chí net)
**Giả thuyết:** bỏ gán đơn sẽ giao quá trễ (`arrival-et ≥ frac·T`) để giải phóng
shipper cho đơn savable → ↑%đúng hạn.
**Đo:** trên v3 đúng là ↑ (frac=0.08: pass 86.7→91.7%, net gần như phẳng). NHƯNG
trên `test_config` **net GIẢM** (5263→5206) và C6 %đúng hạn còn tụt.
**Phân tích nguyên lý:** với đơn-trễ, ba chỉ số ĐỐI KHÁNG nhau — giao đơn trễ thì
*↑net, ↑%giao, ↓%đúng hạn*; bỏ nó thì *↓net, ↓%giao, ↑%đúng hạn*. Không có cách
nào dùng đòn bẩy đơn-trễ để vừa ↑%giao vừa ↑%đúng hạn mà không đổi net. Vì tiêu chí
là **không được giảm net**, hướng này bị **loại**. Muốn ↑%đúng hạn mà giữ net thì
phải GIẢM ĐỘ TRỄ của đơn ta vẫn giao (giao nhanh hơn), không phải bỏ đơn.

### 10f. Đỗ chủ động về điểm lấy gần nhất khi rảnh (giữ — thắng mọi mặt)
**Thay đổi:** khi shipper rỗng và không còn đơn unassigned khả thi (mọi đơn còn chờ
đã gán cho người khác), thay vì đứng yên → tiến tới ĐIỂM LẤY CÒN-CHỜ GẦN NHẤT. Cắt
độ trễ phản ứng với đơn kế tiếp; mục tiêu sticky nên không thrashing; thuần adaptive
(chỉ dùng phân bố đơn quan sát được). Đây chính là hướng "giao nhanh hơn" ở 10e —
giảm độ trễ NHẶT mà không bỏ đơn nào.
**Kết quả (net-DƯƠNG mọi mặt):** test_config net 5263→**5491**, %đúng hạn trung bình
87→**91%** (C2 83→100, C3→97, C6 80→85); v3 tune 86.7→**88.3%**, held-out
83.8→**87.5%**, config-pass **16/20 trên CẢ tune lẫn held-out**.

### Trạng thái chốt phiên này

| | test_config (net) | test_config pass | v3 tune | v3 held-out (seed lạ) |
|--|--|--|--|--|
| Trước phiên | 4935 | 6/6 | 71.7% | — |
| Sau phiên | **5491** (+11.3%) | **6/6** | **88.3%** (cfg 16/20) | **87.5%** (cfg 16/20) |

`test_config` %đúng hạn: C1 100, C2 100, C3 97, C4 89, C6 85; chỉ C5 77 còn dưới 80.
%giao mọi config ≥86.7. Toàn bộ thay đổi phiên này là nguyên lý chung (không tinh
chỉnh theo config), kiểm chứng trên seed lạ (1/55/777/2024) để chứng minh không
overfit, và đều giữ/tăng net.

## 11. Giai đoạn 7 — Đỗ-dự-đoán theo heatmap cầu (phiên làm việc mới)

Xuất phát: net `test_config` 5491 (6/6 pass 70/70, nhưng chỉ 2/6 đạt 90/90), v3
(3 seed) 88.3%. Mục tiêu phiên: đẩy %giao & %đúng hạn về 90% và tăng độ bền v3.

### 11a. Chẩn đoán: lateness = ĐỘ TRỄ NHẶT, tập trung ở đơn hỏa-tốc
Đo per-order (appear→pick→deliver) trên C5/C6/V15/V19/V20: đơn TRỄ có độ-trễ-nhặt
cao gấp ~1.5–2× trung vị (vd V15: trung vị 24 vs đơn trễ 49). Leg giao gần như là
quãng đường bắt buộc. %đúng hạn mất chủ yếu ở **p3** (deadline chỉ t+10..60 — quá
ngắn so với map lớn). Net & %đúng hạn ĐỒNG HƯỚNG cho p3 (giao on-time = ALPHA 3.0
vs late BETA 0.5 ⇒ gấp 6×) → một cơ chế cắt độ-trễ-nhặt sẽ ↑ cả hai.

### 11b. Trần cấu trúc đã xác định (không phải lỗi thuật toán)
- **C1 giao 86.7%:** 1 đơn w=40kg trong khi W_max=20 (không chở nổi) + 1 đơn xuất
  hiện ở t=237/T=240. Trần thực ~13/15 → **không thể đạt 90% giao**.
- **C4 giao 93.3%:** 3 đơn 40kg (W_max max=30, không chở nổi) + 1 đơn cuối-giờ →
  93.3% chính là TRẦN. C4 trễ chỉ 1–9 bước ở vài đơn → cắt ~10 bước latency là đạt.

### 11c. Sửa: lọc "đơn chết" ở nhánh reposition (giữ)
Shipper rỗng KHÔNG đuổi theo điểm lấy của đơn nặng hơn mọi `W_max` (không ai chở
nổi) — trước đây bị dụ tới đó, phí quãng đường. Chỉ áp ở nhánh reposition; nhánh
"đỗ tiếp ứng" cuối GIỮ NGUYÊN (lọc nó lại làm tụt C1/C3 — việc đuổi đơn-chết tình
cờ định vị tốt; nay được thay bằng đỗ-dự-đoán có chủ đích bên dưới).

### 11d. Đỗ-dự-đoán theo HEATMAP CẦU quan sát được (thắng chính)
**Cơ chế:** tích lũy `_demand[ô lấy] += 1` mỗi đơn xuất hiện, phân rã ×0.98/bước
(thuần adaptive — KHÔNG đọc surge/hotspot). Shipper rỗng hết đơn-để-đuổi → tiến
tới ô cầu cao theo điểm `demand/(1+dist)`; **sticky** (lưu `_idle_goal` riêng vì ô
heatmap không có đơn thật nên `_goal_valid` không giữ được → tránh dao động);
**coverage-claim** trải nhiều shipper ra nhiều vùng. Gating **C≥3** (đội 1–2 nên
phản ứng tham lam, đừng bỏ vùng đứng đi đón đầu xa — nếu không C1/C2 sập). Chi phí
chặn bằng top-K ứng viên cầu (k=max(8,4·C)) tính một lần/bước → an toàn N lớn.

**Kết quả (net-DƯƠNG, kiểm trên held-out):**

| | test_config net | 90/90 | C5 %đúng hạn | v3 70/70 tune | v3 70/70 held-out |
|--|--|--|--|--|--|
| Trước (gđ 6) | 5491 | 2/6 | 77.3 | 88.3% | 87.5% |
| Sau (gđ 7) | **5639** (+2.7%) | **3/6** | **86.7** (+9.4) | 88.3% | **90.0%** |

C4 lên 92.9% đúng hạn (đạt 90/90), C5 86.7 (+9.4), C3/C2/C1 giữ, v3 90/90 36.7→45%.
**Held-out (seed 1/55/777/2024) = 90.0% ≥ tune 88.3% ⇒ KHÔNG overfit.** Chỉ C6 net
hụt nhẹ (1575→1516, %đúng hạn giữ 84.6).

### 11e. Trọng-số-ưu-tiên cho heatmap — THỬ rồi LOẠI (overfit seed)
`_demand += o.p` (vùng đơn gấp hút shipper mạnh hơn): v3 tune 70/70 **93.3%** rất
hấp dẫn, NHƯNG held-out tụt còn **87.5%** (< count-based 90.0%) và test net giảm
5639→5570. Phần thắng nằm ở seed tune → **revert** về count-based theo đúng kỷ luật
"ưu tiên độ bền held-out".

## 12. Giai đoạn 8 — Tối ưu THỜI GIAN cho N lớn (giữ nguyên chất lượng)

Mục tiêu: giảm thời gian chạy khi N tăng (Phase 2 có 4–6 config N≤100), **N=70 phải
< 6 phút**, và **không được làm tệ kết quả trên test_config / test_config_v3** (đều
N≤30). Đo từng-config-một (E8 N=40, F1 N=55, F4 N=70 trong `test_config_v2.txt`).

### 12a. Định vị nút cổ chai (cProfile trên E8)
`_sta_star` (A* không-thời-gian) chiếm ~95% thời gian. Bên trong: `valid_next_pos`/
`is_valid_cell`/`next_pos` (~56M lần) sinh hàng xóm + `dict.get` (174M) tra heuristic.
14000 lần gọi A* (=C·T) cho E8. F4 (N=70) **timeout >400s**.

### 12b. Precompute ADJACENCY tĩnh (behavior-preserving)
Bản đồ tĩnh → tính sẵn `_adj[ô] = (chính-ô, các-ô-kề-đi-được)` một lần
(`_build_adj`), thay mọi `valid_next_pos(...)` trong `_sta_star`/`_dist_from`/
`_greedy_path`/`_plan_moves_big`/`_break_deadlocks`. Bỏ hẳn ~56M lần gọi
next_pos/is_valid_cell. **E8: 43s → 23s (1.9×)**, kết quả Y HỆT.

### 12c. THEN CHỐT: bỏ qua A* khi đích xa hơn horizon (behavior-preserving)
Quan sát: nếu `h0 = dist(start→goal) > max_t` thì A* KHÔNG THỂ chạm đích trong
`max_t` bước (chờ chỉ làm dài thêm) → chắc chắn duyệt cạn cả "quả cầu horizon"
(O(#ô·max_t) trạng thái) rồi mới rơi về `_greedy_path`. Thêm một dòng
`if h0 > max_t: return self._greedy_path(...)` → trả KẾT QUẢ Y HỆT nhưng bỏ toàn
bộ khâu duyệt vô ích. Đây đúng là tình huống phổ biến khi N lớn (đích thường ở xa).
**E8: 23s → 2.6s; F1 (N=55): 5.9s; F4 (N=70): timeout → 9.6–10.6s.**

### 12d. Kiểm chứng KHÔNG đổi chất lượng
12b+12c chỉ thay cách tính, không đổi đầu ra: `test_config` net **5639** y nguyên,
6 dòng %giao/%đúng hạn y hệt; v3 tune 70/70 **88.3%**, **held-out 90.0%**, 90/90
**45%** — tất cả KHÔNG đổi. (Bench v3 còn chạy nhanh hơn: 33.8s → 19.6s.)

### 12e. ĐÃ THỬ rồi LOẠI
- **Giảm horizon (cap<24):** net SẬP (cap=20 → 4477) vì đích-xa rơi về greedy
  collision-blind gây deadlock ở C5/C6. Không sạch → giữ cap=40 + mẹo 12c.
- **A* cắt-ngắn trả frontier gần-đích nhất:** v3 tụt 88.3→85% (bước-kế-tiếp né-va-
  chạm đôi khi "chờ" trong khi greedy tiến được — throughput quan trọng hơn). Loại.
- **Đổi sang planner 1-bước `_plan_moves_big` cho N lớn:** **THẢM HỌA** C6 giao
  91→18% (deadlock ở cổng hẹp) — phụ thuộc cấu trúc map, không an toàn cho Phase 2.

### Trạng thái chốt phiên này

| N (config) | Trước | Sau |
|--|--|--|
| 40 (E8) | 43s | **2.6s** (16×) |
| 55 (F1) | — | **5.9s** |
| 70 (F4) | **timeout >400s** | **~10s** |

→ N=70 ~10s ≪ 360s; ngoại suy tuyến-tính N=100 ~25–40s. Chất lượng N≤30 KHÔNG đổi.

## 13. Vấn đề còn mở (hướng phát triển)

- **V15 (ring), V19, V20:** vẫn trượt 70/70 ở vài seed (%đúng hạn ~58–67%) — trần
  cấu trúc do p3 + map rất lớn. Dư địa: định tuyến nhiều-đơn có kiểm tra khả thi
  deadline (route insertion), hoặc heatmap có hướng (đón đầu theo dòng đơn).
- **%đúng hạn trên N rất lớn (F1/F4):** với planner hiện tại, đích-xa dùng greedy
  collision-blind → %đúng hạn thấp (F4 ~25%) dù %giao tốt (~91%) và net dương. Nếu
  Phase 2 chấm net thì chấp nhận được; muốn ↑%đúng hạn cần planner tầm-trung rẻ mà
  vẫn né va chạm (vd A* tới waypoint ~horizon bước trên đường ngắn nhất).
