# Báo cáo Day 08 — 2A202603173 - Tạ Xuân Anh Vũ

> Mọi số liệu dưới đây lấy trực tiếp từ `submission/qc/threshold_sweep.md`, `submission/qc/review_stats.md`, `submission/qc/qc_report_auto.md`, `submission/time_log.csv` và `submission/model_log.csv`. Mọi tỉ lệ đều kèm mẫu số.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| `person`, `car`, `bicycle`, `bus`, `truck` | YOLO26m (`yolo26m.pt`) | `0.30` (mặc định) | `r01`, `r02`, `r05`: YOLO26m nhận diện nhanh (0.11s/ảnh trên GPU CUDA) và ổn định 6 lớp COCO. Theo `threshold_sweep.md`, `person` (32 vật gold, recall tối đa 72%) và `car` (11 vật gold, recall tối đa 64%) không đạt mục tiêu recall 90% tại ngưỡng ≤ 0.5; các lớp `bicycle` (3 vật), `bus` (1 vật), `truck` (0 vật) đều có `< 5` vật trong 4 ảnh gold nên giữ ngưỡng mặc định `0.30`, không ép chọn ngưỡng riêng. |
| `motorbike` | YOLO26m (`yolo26m.pt`) | `0.15` | `r01`, `r02`, `r05`: Ở `r02` (`conf=0.5`), số box `motorbike` tụt mạnh từ 55 xuống 32 (sót xe máy ở xa/bị che). Quét ngưỡng trên 13 vật `motorbike` của gold đạt recall **92%** (≥ 90%) và precision **80%** tại ngưỡng `0.15`. |
| `three_wheeler` | YOLOE-26s-seg (`yoloe-26s-seg.pt`, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.15` | `r01`, `r03`, `r06`: YOLO26 không có lớp `three_wheeler` trong COCO nên bỏ sót hoặc gọi nhầm xích lô/tuk-tuk thành `car`/`bicycle`. YOLOE open-vocabulary bắt được xích lô (`r03`: 3 xích lô ở `img_032`) và đạt recall **100%**, precision **78%** trên 7 vật `three_wheeler` của gold tại ngưỡng `0.15`. |

- **Lệnh `merge_prelabels.py` đã dùng:**
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "motorbike=0.15,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM đã chọn:** **SAM 2.1 base (`sam2.1_b.pt`)**. Lý do: cân bằng tốt nhất giữa độ mượt biên mask và tốc độ chạy trên GPU **NVIDIA T1200 Laptop (4 GB VRAM)** (`r04`: 1.38s/ảnh cho 16 mask); không cần xin quyền gated như SAM 3 và cho chất lượng biên vượt trội hơn SAM 1 bản nhẹ (`mobile_sam.pt`).
- **Lỗi không sửa được bằng đổi ngưỡng:** Lỗi **sai lớp do thiếu khái niệm trong tập huấn luyện** (closed-set COCO) — điển hình là YOLO26m gọi đuôi xích lô thành `bicycle` với `conf=0.70` ở `img_032.jpg` (`r01`, `r02`) và gọi xe tuk-tuk lớn thành `car` với `conf=0.42` ở `img_013.jpg` (`r04`). Tăng ngưỡng lên `0.50` vẫn không loại được lỗi `bicycle 0.70` ở `img_032` mà lại làm mất thêm nhiều người/xe máy thật. Cách sửa đúng là **đổi họ model** sang open-vocabulary (YOLOE) cho lớp `three_wheeler` và đặt thứ tự ưu tiên YOLOE trước YOLO26 khi gộp (`merge_prelabels.py`).
- **Ví dụ box điểm cao mà sai / điểm thấp mà đúng:**
  - *Điểm cao mà sai:* `img_032.jpg` (`r01`/`r02`) có box `bicycle` đạt `conf=0.70` nhưng thực chất là phần bánh sau của xích lô (`three_wheeler`); `img_003.jpg` có box `three_wheeler` trùng lặp với `conf=0.80` (`#25`) chỉ ôm phần thùng dưới của xe đẩy đã có box `#26`.
  - *Điểm thấp mà đúng:* `img_032.jpg` (`r03`) có xích lô bên phải được YOLOE phát hiện đúng `three_wheeler` ở `conf=0.27`; `img_023.jpg` có tuk-tuk ở giữa được bắt đúng ở `conf=0.18`. Điều này chứng tỏ điểm `conf` không phải xác suất đúng thực sự ("0.70" không đồng nghĩa 70% chắc chắn đúng).
- **Giới hạn của bước chọn ngưỡng (Part 3):** Quét ngưỡng chỉ thực hiện trên **4 ảnh gold** (mẫu rất nhỏ, một số lớp chỉ 0–3 vật), và **chọn ngưỡng rồi chấm lại trên chính 4 ảnh gold đó (in-sample)**. Do đó precision/recall của máy trên gold là con số **lạc quan**; thực tế khi áp sang 24 ảnh batch, ngưỡng `three_wheeler=0.15` đã gây nhiều box thừa/trùng (`deleted = 23` trên lớp `three_wheeler`).

## 2. AI error profile (từ `submission/qc/review_stats.md`)

*(Đo trên 24 ảnh batch sau khi **loại 27 đối tượng nằm quanh vùng 12 bẫy** — phải loại vì lỗi nhân tạo do ta tự cài không phản ánh chất lượng thật của model. Ngưỡng accepted: IoU ≥ 0.9.)*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 486 | 486/582 = 83.5% | 582 box máy |
| edited (sửa) | 35 | 35/582 = 6.0% | 582 box máy |
| deleted (xoá) | 57 | 57/582 = 9.8% | 582 box máy |
| added (vẽ thêm) | 89 | 89/610 = 14.6% | 610 đối tượng sau duyệt |

- Trong tổng `deleted + added`, có **12 trường hợp đổi lớp** (box đặt đúng vị trí nhưng sai lớp), nên số đối tượng máy **bỏ sót thật** ≈ `89 - 12 = 77` vật (`77/610 = 12.6%`).
- **Lớp máy tệ nhất:**
  - Về **box thừa (`deleted`)**: **`three_wheeler`** tệ nhất với `23` box bị xóa và `7` box phải sửa so với chỉ `12` box được giữ nguyên (`deleted` chiếm `23/42 = 54.8%` box máy của lớp này). Nguyên nhân: ngưỡng `0.15` của YOLOE quá nhạy trên cảnh đường phố đông, hay sinh 2–3 box chồng lên cùng 1 xe ba bánh (`img_003.jpg`) hoặc nhầm xe máy trong ảnh đêm thành `three_wheeler` (`img_022.jpg`).
  - Về **bỏ sót / đổi lớp (`added`)**: **`car`** (`27` box vẽ thêm so với `47` accepted) và **`person`** (`26` box vẽ thêm) do ngưỡng mặc định `0.30` của YOLO26m vẫn bỏ sót xe/người bị che khuất hoặc ở xa; **`truck`** gần như không đáng tin cậy (`accepted=1`, `deleted=4`, `added=4`).
- **So với gold (4 ảnh, mẫu số recall = 67 đối tượng gold):** Máy đạt `TP=47`, `FP=18`, `FN=20` → **Precision = `47/65 = 72.3%`** (KTC 95%: 60–82%), **Recall = `47/67 = 70.1%`** (KTC 95%: 58–80%). Precision và Recall đều ở mức trung bình (và còn mang tính lạc quan do chọn ngưỡng in-sample), giải thích trực tiếp tại sao tỉ lệ `added` (`14.6%`) và `deleted` (`9.8%`) trên lô batch vẫn đáng kể.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Kết quả bắt bẫy:** **`10/12 = 83.3%`** (KTC 95%: 55–95%, đạt ngưỡng tối thiểu 80% của taxonomy):
  - `missing`: **3/3** (100%)
  - `wrong_class`: **3/3** (100%)
  - `loose`: **3/3** (100%)
  - `extra`: **1/3** (33.3%) — **2 bẫy lọt** đều thuộc loại `extra`: **`img_003.jpg`** (box `motorbike` giả trên mặt đường trống góc phải dưới, `conf=0.45`) và **`img_025.jpg`** (box `car` giả).
- **Vì sao để lọt & bước checklist nào bị bỏ:** Cả hai bẫy lọt đều là box `extra` có điểm `conf` trung bình nằm ở rìa/góc ảnh đông vật (`img_003` có tới 28 box, `img_025` có nhiều xe dày). Khi duyệt, mình tập trung nhiều vào việc tìm vật bị sót (`missing`), sửa lớp `three_wheeler` (`wrong_class`) và kéo gọn box lỏng (`loose`) ở trung tâm ảnh, nhưng đã **lướt nhanh bước 2 của checklist (rà từng box máy có sẵn để kiểm tra box thừa)** — thấy box đã có sẵn chữ `motorbike 0.45` ở góc ảnh nên não mặc định tin là hợp lý (**thiên kiến tự động hóa / automation bias**) mà không nhìn kỹ nền bên trong box.
- **`accepted` cao (`83.5%`) cùng bắt bẫy `extra` thấp (`1/3`) nói gì:** `accepted` cao **không** tự động chứng minh cả máy lẫn người duyệt đều tốt; một phần của `83.5%` có thể đến từ việc người duyệt bấm giữ nguyên quá nhanh các box máy có sẵn (nhất là box trùng/thừa của YOLOE mà vòng QC sau đó đã phát hiện tiếp ở `img_003`, `img_013`, `img_022`). Bên cạnh đó, `accepted` không đo được phần vật máy bỏ sót (`added = 89/610 = 14.6%` và recall trên gold chỉ `70.1%`).

## 4. Máy có đáng dùng không?

- **So sánh thời gian (từ `submission/time_log.csv`):**
  - Gán tay từ đầu (`gold_manual`): `30 phút / 4 ảnh` = **7.50 phút/ảnh** (~20.5 vật/ảnh).
  - Duyệt có pre-label (`batch_review`): `52 phút / 24 ảnh` = **2.17 phút/ảnh** (~26.3 vật/ảnh).
  - → Pre-label giúp tăng tốc độ lên khoảng **3.46 lần** so với gán tay hoàn toàn.
  - *Thiên lệch cần lưu ý:* Do mình đã gán 4 ảnh gold trước nên đã quen phím tắt CVAT và quy tắc guideline trước khi bước vào lô batch; tuy nhiên ảnh batch có nhiều cảnh đông xe máy hơn nên hiệu quả tiết kiệm thời gian của pre-label vẫn là có thật.
- **Kết luận:** **CÓ đáng dùng, nhưng có chọn lọc theo lớp.** Rất hiệu quả cho `person` (`accepted=256/272` box máy), `motorbike` (`162/197`) và `car` (`47/51`); với **`three_wheeler`** nên **tăng ngưỡng YOLOE từ `0.15` lên khoảng `0.25–0.30`** (kèm NMS gọn hơn) để cắt lượng box thừa (`deleted=23`), còn với **`truck`** và **`bicycle`** máy hỗ trợ yếu nên người duyệt phải chủ động tự tìm vẽ tay.
- **Ba quy tắc giảm thiên kiến tự động hóa (kiểm tra được bằng số / hành động cụ thể):**
  1. **Quét mù 5 giây đầu mỗi ảnh:** Bắt buộc ẩn toàn bộ box máy (nút con mắt trong CVAT) và đếm nhanh số phương tiện lớn (`three_wheeler`, `car`, `truck`, `bus`) trước khi bật box lên — đảm bảo không bỏ sót vật do bị neo theo nhãn máy.
  2. **Rà ngược mọi box `conf < 0.50` bằng bộ lọc CVAT (Filter):** Trước khi nhấn `F` chuyển ảnh, lọc tất cả box `source == file` có `conf < 0.50` (đặc biệt lớp `three_wheeler` và `motorbike`) và bấm `Tab` đi qua 100% số box đó để xóa sạch box thừa/trùng (`extra`) — nhắm trực tiếp vào điểm yếu `extra 1/3` và 6/9 lỗi `thua` ở QC.
  3. **Chèn bẫy ngẫu nhiên định kỳ (~10% số ảnh, tối thiểu 12 bẫy/lô) + nghỉ 5 phút sau mỗi 10 ảnh:** Giữ tỉ lệ bắt bẫy **≥ 80% trên từng nhóm lỗi** (không để riêng nhóm `extra` dưới 80%); nếu bất kỳ nhóm bẫy nào tụt dưới 80% thì dừng lô và Rework trước khi tiếp tục.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` (tuk-tuk có người lái) | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương, 1 âm) | **Có dính một phần người lái** (đầu, tay, quần dù đã đặt 1 click âm lên áo) và lem nhẹ sang người áo đỏ ngồi phía sau; gộp kính chắn gió. | Lem biên sang người lái và nền phía sau qua khoang hở; lấp kín lỗ hở lớn > 5% giữa khung và mui xe. | `submission/masks/tuktuk_img013.jpg` |
| `img_004.jpg` (tuk-tuk xanh đỗ gần người đứng cạnh) | SAM 2.1 (`sam2.1_b.pt`) | 4 (3 dương, 1 âm) | **Tách sạch người đứng bên trái** nhờ click âm; gộp thùng đồ nhỏ đặt trong sàn xe và kính chắn gió. | Bỏ sót biển chữ "TAXI" màu trắng trên nóc xe; chừa lỗ hở giữa mui và ghế khá tốt nhưng ăn nhẹ một mảng tối sau khung. | `submission/masks/tuktuk_img004.jpg` |
| `img_017.jpg` (tuk-tuk "MR. K" sát xe sau) | SAM 2.1 (`sam2.1_b.pt`) | 4 (3 dương, 1 âm) | Không có người lái; tách rất sạch khỏi chiếc tuk-tuk đậu ngay phía sau nhờ 1 click âm; gộp kính chắn gió. | Bỏ sót biển "TAXI" trên nóc; **phủ kín khoảng trống lớn giữa mui và ghế hành khách**; dính một dải bóng đổ mỏng dưới gầm xe. | `submission/masks/tuktuk_img017.jpg` |

- **Đối chiếu guideline mục 4.2:**
  - *Quy tắc giúp ích:* Mục **4.2.4** (hai vật chạm nhau không chồng lên nhau) và **4.2.3** (bóng đổ không thuộc vật, kính thuộc vật) giúp đánh giá rất rõ chất lượng tách xe ở `img_017` và phần gầm xe. Mục **4.2.2** (lỗ thủng > 5% diện tích phải bỏ ra khỏi polygon) chỉ ra ngay lỗi phổ biến nhất của SAM trên xe tuk-tuk có khoang hở (`img_013`, `img_017`).
  - *Quy tắc còn thiếu/cần làm rõ (đã bổ sung vào v1.1):* Phân biệt rõ giữa **kính chắn gió** (giữ trong mask xe) với **khoang hở không kính** trên tuk-tuk/xích lô (phải đục lỗ/loại bỏ khi > 5%), và xác định rõ biển nóc ("TAXI") gắn cố định thuộc mask xe.
  - *Kết luận về SAM:* Click của con người quyết định biên mask gồm gì ("một click, nhiều mức độ"); ngoài ra từ `r04` (`t_sam/viz/img_013.jpg`) thấy rõ **mask đẹp cũng không cứu được nhãn lớp sai hoặc box detector bị cắt cụt**.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: không cài serverless Nuclio trên CVAT Docker `v2.74.1` nên dùng `scripts/sam2_points.py` chạy trực tiếp `sam2.1_b.pt` trên GPU (ưu điểm: tái lập chính xác tọa độ click bằng dòng lệnh; nhược điểm: mất công chọn tọa độ x,y thủ công thay vì click trực quan ngay trên canvas CVAT).

## 6. Calibration (Part 6)

- **Số quyết định:** **5 quyết định** trong `submission/qc/calibration_decisions.csv` (`CAL-01` đến `CAL-05`, gồm `loi_may`, `cau_hoi_mo`, `bay_lot`, `can_review`).
- **Quyết định quan trọng nhất:**
  - **`CAL-01` (`loi_may` — `img_013`, `img_004`, `img_032`):** Xử lý lỗi hệ thống của pipeline hai model (YOLO26m gọi nhầm tuk-tuk/xích lô là `car`/`bicycle`, còn YOLOE ngưỡng `0.15` lại sinh box `three_wheeler` trùng lặp hoặc đè lên xe máy ở `img_003`).
  - **`CAL-02` (`cau_hoi_mo` — `img_022`, `img_034`):** Chốt ranh giới xe đạp điện / xe máy điện dựa trên biển số và cấu tạo bàn đạp cơ để xử lý dứt điểm ca `can_review` `#583` ở `img_034.jpg`.
- **Quy tắc mới đã cập nhật vào `submission/guidelines/guideline_v1.1_calibrated.md`:** Chốt đủ 7 câu hỏi mở mục 6 (xe đạp điện, cụm xe máy gần camera, hàng nhô ra trên box/mask, tiêu chí thấy rõ đầu+thân trên cho người trong xe có mui, phân biệt kính chắn gió vs khoang hở > 5% trong mask, chỉ giữ 1 box tốt nhất khi ảnh đêm chồng box, không auto-accept theo `conf`) và áp cứng ngưỡng cao ≥ 20 px cho cả vật bị cắt ở mép ảnh (`CAL-05`, `img_023`).
- **Phân biệt nguyên nhân gốc (guideline mơ hồ vs lỗi máy/người):**
  - *Guideline mơ hồ (phải sửa v1.1):* Xe đạp điện (`img_034`), người thấy qua kính/mui (`img_012`), ngưỡng cắt mép `< 20 px` (`img_023`), lỗ hở khung tuk-tuk trong mask (`img_013`, `img_017`).
  - *Lỗi máy & lỗi người duyệt (KHÔNG đổi định nghĩa lớp, phải sửa recipe + checklist):* YOLO26m gọi tuk-tuk là `car` (lỗi đóng tập COCO), YOLOE ngưỡng `0.15` sinh box trùng/nhiễu (lỗi ngưỡng), và bỏ lọt 2 bẫy `extra` ở `img_003`, `img_025` (lỗi thao tác duyệt do thiên kiến tự động hóa).

## 7. QC và quyết định (Part 6)

- **Mẫu QC:** **8/24 ảnh batch** (`3` ảnh có `can_review`: `img_022`, `img_023`, `img_034` + `5` ảnh ngẫu nhiên seed 42: `img_003`, `img_006`, `img_013`, `img_014`, `img_035`), tổng cộng **184 đối tượng** trong mẫu. Người kiểm: chính mình (**hạn chế:** người duyệt tự kiểm lại bài mình dễ mang thiên kiến quen mắt, khó phát hiện các vật mà cả máy lẫn mình cùng bỏ sót).
- **Hai cách đếm tỉ lệ lỗi (từ `submission/qc/qc_report_auto.md`):**
  - Lỗi theo **đối tượng**: **`9/184 = 4.9%`** (mẫu số: 184 đối tượng trong 8 ảnh QC).
  - Lỗi theo **ảnh** (dùng đối chiếu ngưỡng Release ≤ 2%): **`5/8 = 62.5%`** (mẫu số: 8 ảnh QC; 3/8 ảnh sạch: `img_006`, `img_014`, `img_035`).
- **Top lỗi quan sát được (tổng 9 lỗi):**
  1. `thua` (**6/9 lỗi** — lỗi hệ thống lặp lại): `img_003.jpg` (`#5` box `motorbike 0.45` thừa trên mặt đường là bẫy `extra` bị lọt; `#23`, `#24`, `#25` box `three_wheeler` trùng/nhầm của YOLOE), `img_013.jpg` (`#136` box `motorbike` chồng lặp), `img_022.jpg` (`#298` box `three_wheeler` chụp trùm lên cụm xe máy đêm).
  2. `sai_vi_tri_kich_thuoc` (**2/9 lỗi**): `img_023.jpg` (`#310` box `three_wheeler` lỏng góc trên-trái; `#312` box `car` bị mép ảnh cắt chỉ cao 15.1 px < 20 px).
  3. `sai_lop` (**1/9 lỗi**): `img_034.jpg` (`#583` xe máy điện có biển số 29 bị gán `bicycle`).
- **Giới hạn thống kê của mẫu nhỏ:** Với `5/8` ảnh có lỗi, **cận trên 95% (Wilson)** của tỉ lệ ảnh lỗi thật trên cả lô lên tới **86%** (theo `qc_report_auto.md`). Ngay cả khi một lô mẫu nhỏ chỉ 8 ảnh có `0/8` ảnh lỗi, theo quy tắc 3 (`3/n`), tỉ lệ lỗi thật vẫn có thể lên tới `~37.5%`, nên mẫu 8 ảnh **không thể chứng minh** đạt ngưỡng `≤ 2%` mà chỉ đủ để phát hiện sớm lỗi hệ thống (ở đây là lỗi `thua`).
- **Script kiểm được / không kiểm được:**
  - *Script kiểm được:* Cấu trúc file COCO, lớp hợp lệ theo taxonomy, tọa độ box trong khung ảnh, cảnh báo box `< 20 px`, số lượng `accepted/edited/deleted/added`, tỉ lệ bắt 12 bẫy, tổng hợp tỉ lệ lỗi QC và khoảng tin cậy Wilson.
  - *Script không kiểm được:* Box có ôm sát đường biên vật thật không, vật có bị gán sai lớp thực tế hay bị **cả máy lẫn người cùng bỏ sót** ngoài tập gold không, và người duyệt có thực sự thực hiện bước ẩn box quét ảnh trước theo checklist hay không.

**Quyết định: Rework**, ngày `2026-11-17`.
- **Lý do (theo đúng thứ tự cây Hold → Rework → Release):**
  1. *Kiểm Hold:* Dữ liệu đầu vào đầy đủ, hợp lệ (`validate_labels` OK); các câu hỏi mở của guideline v1.0 đã được chốt thành `v1.1`; bắt bẫy đạt `10/12 = 83.3% ≥ 80%` → **không Hold**.
  2. *Kiểm Rework:* Tỉ lệ ảnh có lỗi là **`5/8 = 62.5%` (vượt xa ngưỡng Release `≤ 2%`)**, tỉ lệ lỗi theo đối tượng là **`9/184 = 4.9%`**, và xuất hiện **lỗi hệ thống lặp lại (`thua` chiếm 6/9 lỗi)** do box trùng/nhiễu của YOLOE ở ngưỡng `0.15` cùng bẫy `extra` còn sót → **bắt buộc Rework** (không đủ điều kiện Release).
- **Việc cần làm tiếp cho vòng Rework:**
  1. Trong CVAT, sửa toàn bộ các lỗi đã ghi trên `img_003`, `img_013`, `img_022`, `img_023`, `img_034` và rà lại nhanh 19 ảnh còn lại bằng Filter `conf < 0.50` + xóa các box lẻ `< 20 px` (theo cảnh báo `validate_labels` trên `img_018`, `img_025`, `img_027`).
  2. Xuất các ảnh đã sửa thành `submission/labels/qc_fixed.json`, chạy lại `build_release.py`, lấy mẫu QC lại (`qc_sample.py --force`) và chạy lại `qc_report.py`.
  3. Cho lô ảnh sau: nâng ngưỡng `three_wheeler` của YOLOE lên `0.25–0.30` trước khi nạp CVAT để giảm tải box trùng/thừa.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Có sử dụng trợ lý AI (Antigravity) hỗ trợ thiết lập môi trường Python 3.12 + PyTorch CUDA trong WSL2, sửa lỗi container `traefik` của CVAT Docker bị 404, giải thích quy tắc trong `guideline_v1.0_label_contract.md` (không xem ảnh gold trong lúc gán Part 1), chạy các lệnh script chuẩn của repo và tổng hợp cấu trúc báo cáo từ số liệu thật. **Cách kiểm tra:** Mình tự gán tay hoàn toàn 4 ảnh gold và tự duyệt 24 ảnh batch trên CVAT; đối chiếu trực tiếp toàn bộ số liệu trong báo cáo với file đầu ra gốc của script (`threshold_sweep.md`, `review_stats.md`, `qc_report_auto.md`, `work/runs.csv`) và kiểm tra mắt trực tiếp các ảnh `viz/`, `masks/` và `work/qc/review/`.
- **Gold có bị nhiễm không:** **Không.** 4 ảnh gold (`img_001`, `img_012`, `img_020`, `img_036`) được gán tay hoàn toàn từ đầu trên CVAT trước khi chạy bất kỳ model pre-label nào; không xem pre-label hay nhờ AI nhìn ảnh gold trước khi export `gold_manual.json`.
- **Máy / GPU / CVAT dùng và vướng mắc cài đặt:** Windows + WSL2 (Ubuntu 26.04), GPU **NVIDIA T1200 Laptop GPU (4 GB VRAM, CUDA 12.4)**, **CVAT Docker `v2.74.1`**. Vướng mắc đã xử lý: (1) Ubuntu 26.04 mặc định Python 3.14 chưa có wheel `torch` → tạo lại `.venv` bằng Python 3.12 qua `uv`; (2) `traefik` của CVAT trả lỗi 404 do bind-mount `/var/run/docker.sock` cũ → chạy `docker compose up -d --force-recreate traefik` là hoạt động bình thường; (3) CVAT 2.74.1 báo `Label 'motorbike' is not registered` khi tạo Task chưa gắn Project → dán trực tiếp `cvat/labels_generated.json` vào tab Raw của Task `batch-review`.
- **Dùng cache `data/cache/` thay vì tự chạy?** **Không.** Toàn bộ 6 lượt chạy YOLO26m, YOLOE-26s-seg và SAM 2.1 (`sam2.1_b.pt`) đều tự chạy trực tiếp trên GPU CUDA cục bộ.
- **Lỗi / bug script đã gặp:** Không có bug trong mã nguồn script; chỉ cần cài trước gói `clip` từ GitHub vào `.venv` để YOLOE không bị lỗi thiếu `pip` trong môi trường tạo bằng `uv`.
- **Một điều làm khác lần sau:** Sẽ áp dụng ngay bộ lọc `conf < 0.50` trong CVAT ở cuối mỗi ảnh để quét sạch box thừa/trùng của YOLOE (`three_wheeler`) và xóa các box `< 20 px` ngay trong vòng review đầu tiên, tránh để sót lỗi hệ thống `thua` khiến cả lô phải Rework.
