# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 582; đối tượng sau duyệt: 610; ngưỡng accepted: IoU >= 0.9
- Đã **loại 27 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 486 | 486/582 = 83.5% | 582 box máy |
| edited (sửa) | 35 | 35/582 = 6.0% | 582 box máy |
| deleted (xoá) | 57 | 57/582 = 9.8% | 582 box máy |
| added (vẽ thêm) | 89 | 89/610 = 14.6% | 610 đối tượng sau duyệt |

Trong số deleted + added, **12 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 77.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 256 | 9 | 7 | 26 |
| bicycle | 5 | 1 | 2 | 8 |
| motorbike | 162 | 17 | 18 | 19 |
| car | 47 | 1 | 3 | 27 |
| bus | 3 | 0 | 0 | 3 |
| truck | 1 | 0 | 4 | 4 |
| three_wheeler | 12 | 7 | 23 | 2 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 67 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 47 | 18 | 20 | 72.3% (KTC 95%: 60–82%) | 70.1% (KTC 95%: 58–80%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - missing: 3/3
  - wrong_class: 3/3
  - extra: 1/3
  - loose: 3/3
  - lọt: img_025.jpg (extra), img_003.jpg (extra)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
