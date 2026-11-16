# CS231 — Phân loại chất liệu bề mặt

Template làm việc nhóm trên MINC-2500. Mỗi thành viên tự viết phần trích xuất đặc trưng và huấn luyện; cùng dùng Split 1 và cách đánh giá trong notebook.

## Bắt đầu

Cần cài uv và một trình mở notebook, ví dụ VS Code với extension Jupyter.

Trong thư mục dự án, tạo môi trường Python và cài thư viện:

```bash
uv sync --locked
```

Tải và giải nén dataset từ nguồn chính thức (các lệnh dưới dùng cho Linux/macOS/WSL):

```bash
mkdir -p data
curl -fL --retry 3 -C - -o data/minc-2500.tar.gz https://opensurfaces-minc.s3.amazonaws.com/minc-2500.tar.gz
tar -xzf data/minc-2500.tar.gz -C data
```

Cấu trúc dữ liệu cần có:

```text
data/minc-2500/
├── categories.txt
├── images/
└── labels/
    ├── train1.txt
    ├── validate1.txt
    └── test1.txt
```

Dataset tải về khoảng 2,1 GB; giải nén khoảng 2,5 GB. Dữ liệu và môi trường `.venv` không đưa lên Git. Mỗi người tải dataset trên máy của mình.

## Làm thí nghiệm

1. Sao chép `experiment_template.ipynb` thành `experiment_<ten>_<phuong_phap>.ipynb`, đặt ngay trong thư mục dự án.
2. Chọn kernel `.venv/bin/python`, chạy notebook từ thư mục dự án.
3. Chạy phần đọc dữ liệu; tự viết phần đặc trưng và huấn luyện.
4. Đánh giá validation để chọn cấu hình; chỉ đánh giá test khi đã chốt phương pháp.
5. Ghi tham số, kết quả và nhận xét ở cuối notebook.

Toàn bộ code cần thiết nằm trong notebook. Sửa `DATA_DIR` nếu dữ liệu nằm ở chỗ khác. Khi đổi phương pháp, dùng **Restart Kernel → Run All** để tránh dùng biến cũ. Trước khi commit notebook, dùng **Clear All Outputs** rồi lưu để repo gọn và dễ xem thay đổi.

## Quy ước đánh giá chung

| Tập | Số ảnh | Vai trò |
|---|---:|---|
| Train | 48.875 | Học đặc trưng/tiền xử lý có tham số và mô hình |
| Validation | 2.875 | Chọn cấu hình bằng Macro-F1 |
| Test | 5.750 | So sánh cuối cùng giữa các phương pháp đã chốt |

- Giữ nguyên thứ tự 23 nhãn trong `categories.txt` và các danh sách Split 1.
- Dự đoán phải khớp thứ tự ảnh. Scaler/PCA chỉ fit trên train; áp dụng lại cho validation và test.
- Báo cáo Accuracy, Macro-F1, Precision/Recall/F1 từng lớp và confusion matrix. Điểm ở thang 0..1; chia cho 0 tính điểm 0.
- Không chỉnh tham số dựa trên điểm test. Template huấn luyện trên train, chưa gộp train và validation.

## Nguồn dữ liệu

[MINC-2500 — Cornell OpenSurfaces](https://opensurfaces.cs.cornell.edu/publications/minc/): 57.500 ảnh thuộc 23 lớp. Nhãn được cấp phép CC BY 4.0; ảnh có giấy phép riêng, xem nguồn chính thức.

Sean Bell, Paul Upchurch, Noah Snavely và Kavita Bala. *Material Recognition in the Wild with the Materials in Context Database*. CVPR 2015.
