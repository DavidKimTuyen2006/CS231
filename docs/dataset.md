# Dataset MINC-2500 — Split 1

## Nguồn

- Hugging Face: [`mcimpoi/minc-2500_split_1`](https://huggingface.co/datasets/mcimpoi/minc-2500_split_1)
- Revision đã ghim: `5ac92ad1af8ef34f9ecd8d54d60a73c0289fe0c3` (lấy từ HF Hub API ngày 2026-11-18, lưu trong `configs/default.yaml`).
- Giấy phép nhãn: CC BY 4.0. Ảnh có giấy phép riêng; xem nguồn chính thức: [Cornell OpenSurfaces](https://opensurfaces.cs.cornell.edu/publications/minc/).

## Schema (đã kiểm tra trên metadata của nguồn)

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `image` | Image | Ảnh, giải mã bằng PIL, chuyển sang RGB `uint8` |
| `label` | ClassLabel (23 lớp) | Thứ tự cố định: xem `dataset.class_names` trong `configs/default.yaml` |

## Split

| Split | Số mẫu |
|---|---:|
| train | 48.875 |
| validation | 2.875 |
| test | 5.750 |

Split được giữ nguyên như nguồn. Không chia lại dữ liệu.

## Lưu trữ cục bộ

- Bản Arrow: `DATASET_ROOT` (mặc định `./data/minc-2500`), tạo bởi `scripts/download_dataset.py` bằng `save_to_disk()`.
- `source.json`: dataset id, revision và số mẫu lúc tải.
- `manifest.json`: tạo bởi `scripts/prepare_data.py`, gồm revision, số mẫu, số ảnh từng lớp và băm nhãn từng split.
- Không đưa dataset lên GitHub.

## Kiểm tra trước khi chốt

- [x] Tên split và cột khớp metadata của nguồn.
- [x] 23 nhãn và số mẫu từng split khớp.
- [ ] Tải đầy đủ và chạy `prepare_data.py` trên máy của nhóm (chưa thực hiện trong repo này).
- [ ] Xác nhận lại revision trước khi nộp.

## Định danh mẫu

Nguồn không có ID mẫu ổn định. Mẫu được định danh bằng `split:index`, theo thứ tự cố định của bản Arrow. Băm nhãn và danh sách ID được lưu trong `manifest.json` và metadata của cache, để phát hiện nếu thứ tự đổi.
