# Kế hoạch Phân chia Nhánh & Nhiệm vụ Đồ án CS231

Tài liệu ghi lại kế hoạch phát triển và phân chia nhánh (Git Feature Branches) cho nhóm thực hiện đồ án phân loại vật liệu MINC-2500.

---

## 1. Nhánh `feat/feature-extraction`
- Viết `features/hsv.py`, `features/hog.py` và `features/extractor.py` với hàm `extract_base_vectors(rgb, features_cfg)`.
- Trả về dict `float32` 1 chiều: `hsv` (32), `hog64` (1.764), `hog128` (8.100).
- **HSV**: ảnh 128×128, 16/8/8 bins, L1 từng kênh.
- **HOG**: ảnh xám, 9 hướng, cell 8×8, block 2×2, chuẩn hóa `L2-Hys`.
- Resize trực tiếp từ RGB gốc bằng `INTER_AREA`, không chuẩn hóa L2 ở bước này.
- `uv add scikit-image`, kèm test shape, dtype, giá trị hữu hạn và ảnh hằng.
- Sau khi merge, chạy `extract_features.py` để tạo cache đầy đủ trên dữ liệu thật.

> **Tiến độ**: Đã được Lâm (`Lam210905`) push lên nhánh `origin/feat/feature-extraction` lúc 12:31 ngày 20/11/2026.

---

## 2. Nhánh `feat/fusion-models`
- Viết `features/normalization.py`: chuẩn hóa L2 theo hàng, vector 0 giữ nguyên.
- Viết `features/fusion.py`:
  - `combine_multiscale`: kết hợp HOG đa tỉ lệ (9.864 chiều: HOG-64 + HOG-128).
  - `fuse_features`: kết hợp trọng số $\sqrt{\alpha}$ và $\sqrt{1-\alpha}$, trong đó HOG đứng trước HSV (8.132 chiều và 9.896 chiều).
- Viết `models/svm.py` và `models/knn.py` theo đúng tham số mục 8.2 của tài liệu thiết kế.
- Chuyển `scikit-learn` từ nhóm notebook lên dependency chính (`uv add scikit-learn`).
- Khôi phục các khối `classifier`, `representation` và lưới $\alpha, k$ vào config.
- Test: $\alpha = 0 / 0.5 / 1$, $\alpha$ sai bị từ chối, đúng thứ tự nối, đúng số chiều.

---

## 3. Nhánh `feat/experiment-pipeline` (cần hai nhánh trên)
- Pipeline ghép `default.yaml` với config thí nghiệm, đọc cache bằng `load_cache`.
- Fit trên train, chọn tham số bằng Macro-F1 trên validation, hòa điểm giữ ứng viên đứng trước.
- Đánh giá test đúng một lần sau khi chốt cấu hình.
- Viết `evaluation/metrics.py` và `visualization.py`: Accuracy, Macro-F1 trên đủ 23 lớp, precision/recall/F1 từng lớp, confusion matrix.
- Mỗi run ghi commit Git, revision dataset, cache signature, seed và phiên bản thư viện.
- Script `run_experiment.py` và `summarize_results.py`, kèm test pipeline end-to-end bằng dữ liệu tổng hợp.
- **Mốc kiểm tra**: Chạy C1 qua pipeline phải ra kết quả gần mốc notebook (Test Accuracy ~0.1757, Macro-F1 ~0.1318).

---

## 4. Nhánh `exp/configs-c2-c6`
- Điền 5 file YAML trống theo mẫu `c1_hsv_svm.yaml`:
  - **C2**: HOG-128 (8.100 chiều).
  - **C3**: HOG-64 + HOG-128 (9.864 chiều).
  - **C4**: HSV + HOG-128, $\alpha = 0.5$ (8.132 chiều).
  - **C5**: HSV + HOG đa tỉ lệ, $\alpha = 0.5$ (9.896 chiều).
  - **C6**: HSV + KNN (32 chiều).
- Thêm `c5_tuned_alpha_svm.yaml` ($\alpha$ chọn trên validation) nếu nhóm quyết định báo cáo riêng.
- Kiểm tra số chiều mỗi cấu hình: 8.100, 9.864, 8.132, 9.896 và 32.

---

## 5. Nhánh `exp/run-and-report` (cần các nhánh trên)
- Chạy C1–C6 trên dữ liệu thật, kiểm tra cảnh báo hội tụ của `LinearSVC`.
- Công bố kết quả đã rà soát vào `results/`: `comparison.csv`, `ablation_study.csv` và confusion matrix.
- Làm các so sánh ablation theo tài liệu: C1–C2, C2–C3, C2–C4, C3–C5, C4–C5, C1–C6.
- Ghi rõ lưu ý: các cấu hình đa tỉ lệ cũng đổi số chiều.
