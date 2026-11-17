# Thông tin học viên

- Họ và tên: Tạ Xuân Anh Vũ
- MSHV: 2A202603173
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: AnhVuTaXuan
- Repo bài nộp (URL): https://github.com/AnhVuTaXuan/K5-L23-Day08-TAXUANANHVU-2A202603173-AI-Assisted-Labeling
- Commit chốt bài (hash): 68996a0
- Máy: Windows + WSL2 (Ubuntu 26.04), GPU NVIDIA T1200 Laptop GPU (4GB VRAM, CUDA)
- CVAT (Docker): v2.74.1

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 486/582 = 83.5% / 35/582 = 6.0% / 57/582 = 9.8% (mẫu số: 582 box máy); added: 89/610 = 14.6% (mẫu số: 610 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 47/65 = 72.3% (KTC 95%: 60–82%) / 47/67 = 70.1% (KTC 95%: 58–80%)
- Bắt bẫy: 10/12 = 83.3% (missing 3/3, wrong_class 3/3, loose 3/3, extra 1/3)
- Thời gian: gán tay 7.50 phút/ảnh (30 phút / 4 ảnh); duyệt có pre-label 2.17 phút/ảnh (52 phút / 24 ảnh)
- Lỗi QC: 62.5% theo ảnh (5/8 ảnh, cận trên 95% Wilson: 86%), 4.9% theo đối tượng (9/184 đối tượng)
- **Quyết định:** Rework — lý do một dòng: Tỉ lệ ảnh lỗi 5/8 = 62.5% > ngưỡng 2% và xuất hiện lỗi hệ thống box thừa/trùng (6/9 lỗi, chủ yếu từ YOLOE ngưỡng 0.15 và 1 bẫy extra còn sót).
