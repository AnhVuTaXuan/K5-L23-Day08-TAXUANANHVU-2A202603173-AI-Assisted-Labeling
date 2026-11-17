# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603173 - Tạ Xuân Anh Vũ

> **Chỉ ghi phần bổ sung / thay đổi so với `guideline/guideline_v1.0_label_contract.md`**. Mọi ca khó đều gắn ảnh ví dụ (`img_0xx`) và khớp trực tiếp với `submission/qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Xe điện có biển số hoặc dáng xe tay ga/không bàn đạp rõ → `motorbike`; xe đạp trợ lực khung mảnh có bàn đạp cơ rõ → `bicycle`. | `img_034`, `img_022` | `CAL-02` |
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Xe phía trước cao ≥ 40 px và nhìn rõ bánh/yên → gán từng `motorbike`; dãy xe chồng lớp phía sau không tách ranh giới được (hoặc < 20 px) → gom 1 `ignore_region`. | `img_018`, `img_006` | `CAL-05` |
| 3 | Hàng hóa nhô ra trên xe đạp / xe ba gác: box và mask xử lý thế nào? | Giữ quy tắc box không tính phần hàng rời nhô quá ~20% chiều dài xe; với mask SAM chỉ lấy thân xe + thùng/giỏ gắn cố định, loại hàng rời cồng kềnh bằng click âm. | `img_003`, `img_004` | `CAL-01` |
| 4 | Người ngồi trong xe ba bánh có mui che (nhìn qua kính/khoang hở): ngưỡng ≥ 20%? | Giữ ngưỡng ≥ 20% nhưng cụ thể hóa để giảm lệch mắt: thấy rõ **đầu + ít nhất nửa thân trên** thì gán `person` (box ôm phần nhìn thấy); chỉ thấy mảng áo/vai mờ dưới mui thì không gán. | `img_012`, `img_013` | `CAL-03` |
| 5 | Kính chắn gió / cửa sổ xe trong mask: thuộc mask xe hay loại bỏ? | **Kính chắn gió** thuộc mask xe (đồng nhất với mục 4.2.3); riêng **khoảng trống hở không kính** xuyên qua khung xe chiếm > 5% diện tích xe (vd khoang hở tuk-tuk) phải loại bỏ khỏi polygon (mục 4.2.2). | `img_013`, `img_017` | `CAL-03` |
| 6 | Box máy chồng nhau nhiều trong ảnh đêm: gộp hay giữ? | Không giữ nhiều box chồng trên cùng 1 vật: chỉ giữ **1 box khớp biên nhất**, xóa toàn bộ box trùng/nhiễu (đặc biệt box `three_wheeler` conf thấp của YOLOE). | `img_022`, `img_003` | `CAL-04` |
| 7 | Có đặt mức `conf` tối thiểu để tự động chấp nhận box máy không? | **Không.** `conf` chỉ dùng để sắp thứ tự kiểm tra, không bao giờ tự động chấp nhận mà không nhìn ảnh (vì YOLO26 vẫn sai lớp với `conf=0.70` trên xích lô `img_032` và bẫy `extra` vẫn có `conf=0.45` ở `img_003`). | `img_032`, `img_003` | `CAL-04` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_013`, `img_004`, `img_032` | Tuk-tuk, xích lô, xe ba bánh chở hàng có mui/thùng bị nhận nhầm lớp hoặc sinh nhiều box chồng | YOLO26m gọi tuk-tuk là `car` (0.42 ở `img_013`) và gọi đuôi xích lô là `bicycle` (0.70 ở `img_032`); YOLOE ở ngưỡng 0.15 sinh 2–3 box `three_wheeler` chồng lên nhau (`img_003` #24/#25/#26) và đè lên xe máy (`img_003` #23) | Luôn gán đúng 1 box `three_wheeler` bao trọn xe + mui/thùng gắn liền; xóa mọi box trùng hoặc box YOLOE đè lên xe 2 bánh | Mục 2 & mục 5 (`CAL-01`) |
| `img_023`, `img_018` | Vật bị mép ảnh cắt mạnh hoặc ở xa khiến box cao < 20 px (`img_023` #312 `car` cao 15.1 px; `img_018` có 8 box 14–18 px) | YOLO26m vẫn đề xuất box nhỏ < 20 px ở rìa ảnh và phía xa | Áp dụng cứng ngưỡng cao ≥ 20 px cho cả vật bị cắt ở mép ảnh; xóa các box lẻ < 20 px và dùng `ignore_region` cho cụm nhỏ ≥ 5 vật | Mục 4.1.3 & 4.1.6 (`CAL-05`) |
| `img_003`, `img_025` | Box thừa (`extra`) vẽ trên nền đường trống hoặc vùng không có vật | Lỗi nhân tạo (bẫy `extra`: `img_003` #5 `motorbike 0.45`, `img_025` `car`) bị lọt qua vòng review do thiên kiến tự động hóa | Không sửa định nghĩa lớp; đây là **lỗi thao tác review** (không phải guideline mơ hồ) → bổ sung bước kiểm tra riêng nhóm box thừa trên nền trống trước khi chuyển ảnh | Quy trình review, không đổi định nghĩa lớp (`CAL-04`) |
| `img_034` | Xe máy điện / xe tay ga có biển số 29 ở lề phải (`#583`) bị gán nhầm `bicycle` | Gán `can_review` khi review do ranh giới xe đạp điện chưa rõ trong v1.0 | Đổi sang `motorbike` ở vòng Rework vì xe có biển số và thân xe máy điện không có bàn đạp cơ | Mục 2 (`bicycle` / `motorbike`, `CAL-02`) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản label contract gốc do giảng viên cung cấp | — |
| v1.1 | Chốt đủ 7 câu hỏi mở mục 6 (xe đạp điện, cụm xe máy gần, hàng nhô ra, người trong xe có mui, kính/lỗ hở mask, box chồng ảnh đêm, không auto-accept theo `conf`); bổ sung quy tắc xử lý vật cắt mép ảnh < 20 px và xóa box trùng của YOLOE | `CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`, `CAL-05` (dẫn chứng từ `img_003`, `img_004`, `img_012`, `img_013`, `img_017`, `img_018`, `img_022`, `img_023`, `img_025`, `img_032`, `img_034`) |
