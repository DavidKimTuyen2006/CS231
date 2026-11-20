# CS231 — Chuẩn bị dữ liệu MINC-2500 (Split 1)

Phần dữ liệu của đồ án: tải dataset từ Hugging Face một lần, kiểm tra, rồi trích xuất feature cache để các thành viên khác dùng lại.

Repo dùng **uv** để quản lý Python và thư viện. Dataset và cache lưu cục bộ, không đưa lên GitHub.

---

## 1. Yêu cầu

- Git
- [uv](https://docs.astral.sh/uv/) (uv tự tải Python 3.12 nếu máy chưa có)
- Khoảng **10 GB** trống cho dataset và cache
- Internet cho lần tải đầu tiên

Cài uv:

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux / macOS / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Đóng và mở lại terminal, rồi kiểm tra: `uv --version`.

---

## 2. Cài môi trường (mỗi máy làm một lần)

```bash
git clone <repository-url> minc-material-classification
cd minc-material-classification
uv sync --locked
```

- `uv sync --locked` cài đúng phiên bản trong `uv.lock`.
- Không cần kích hoạt `.venv`: mọi lệnh chạy qua `uv run ...`.

Tạo file cấu hình riêng của máy:

```bash
# Linux / macOS / WSL
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Sửa `.env` nếu muốn đặt dataset và cache ở ổ khác:

```dotenv
DATASET_ROOT=D:/datasets/minc-2500
CACHE_ROOT=D:/cs231-cache
```

Đường dẫn tương đối tính từ thư mục gốc repo. Không ghi đường dẫn riêng vào YAML hay mã nguồn.

---

## 3. Tải và kiểm tra dataset (mỗi máy làm một lần)

Nguồn: [`mcimpoi/minc-2500_split_1`](https://huggingface.co/datasets/mcimpoi/minc-2500_split_1). Revision (commit SHA) được ghim trong `configs/default.yaml`, mọi người dùng cùng một bản.

```bash
uv run python scripts/download_dataset.py
uv run python scripts/prepare_data.py
```

- `download_dataset.py` tải dataset và lưu bản Arrow bằng `save_to_disk()` tại `DATASET_ROOT`.
- `prepare_data.py` kiểm tra số mẫu (48.875 / 2.875 / 5.750), 23 lớp và thứ tự nhãn, rồi ghi `manifest.json`.
- Tải lại khi đã có dataset: `uv run python scripts/download_dataset.py --force`.
- Có sẵn bản Arrow từ máy khác: chép vào `DATASET_ROOT`, rồi chạy `prepare_data.py` để xác thực.
- Có sẵn cache Hugging Face: có thể đặt `DATASET_ROOT=D:/huggingface/datasets/mcimpoi___minc-2500_split_1`. Hàm đọc sẽ tìm cache của revision đã ghim và đọc trực tiếp các shard Arrow, không cần tải lại. Cũng có thể trỏ thẳng tới thư mục chứa `dataset_info.json`.
- Với `notebooks/c1_hsv_svm.ipynb`, cell đầu nạp lại `.env` và cell đọc dataset in đường dẫn đang dùng. Sau khi cập nhật mã nguồn, chọn **Restart Kernel** rồi chạy lại hai cell đầu để kiểm tra dữ liệu.

---

## 4. Trích xuất feature cache

```bash
uv run python scripts/extract_features.py
```

- Cache ghi vào `CACHE_ROOT/minc2500_split1/<signature>/`, theo từng split, dạng `.npy`.
- Chữ ký cache băm revision dataset, danh sách nhãn và cấu hình `features:` trong `configs/default.yaml`. Đổi bất kỳ mục nào trong đó thì phải trích xuất lại.
- Chạy lại khi cache đã hợp lệ sẽ tái sử dụng, không tính lại.
- Ép trích xuất lại: `uv run python scripts/extract_features.py --force`.

**Phụ thuộc nhóm đặc trưng:** script gọi `extract_base_vectors(rgb, features_cfg)` từ `material_classification.features.extractor`. Module này do nhóm đặc trưng viết. Nếu chưa có, script dừng và báo rõ.

Hàm trích xuất nhận ảnh RGB `uint8` và trả về dict `{tên_vector: mảng 1 chiều}`. Mỗi vector được ghi thành `<tên_vector>.npy`.

---

## 5. Cấu trúc repository

```text
minc-material-classification/
├── configs/
│   ├── default.yaml            # dataset (revision ghim, nhãn, số mẫu), tham số đặc trưng cho chữ ký cache
│   └── experiments/
│       └── c1_hsv_svm.yaml     # config cho c1
├── src/material_classification/
│   ├── data/
│   │   ├── dataset.py          # đọc dataset cục bộ, kiểm tra schema, giải mã ảnh
│   │   ├── transforms.py       # resize trực tiếp từ RGB, chuyển HSV và xám
│   │   └── cache.py            # ghi, đọc và xác thực feature cache
│   ├── features/__init__.py    # thuộc nhóm đặc trưng
│   ├── models/__init__.py      # thuộc nhóm mô hình
│   ├── evaluation/__init__.py  # thuộc nhóm đánh giá
│   └── utils.py                # đường dẫn .env, đọc YAML, băm, ghi JSON, thông tin tái lập
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_data.py
│   └── extract_features.py
├── tests/test_dataset.py       # test schema, cache và luồng build cache (không cần MINC-2500)
├── notebooks/experiment_template.ipynb   # template cũ của lớp, không thuộc phần dữ liệu
├── docs/dataset.md             # nguồn, schema, revision, split
├── .github/workflows/tests.yml # CI: uv sync --locked, ruff, pytest
├── .env.example
├── pyproject.toml, uv.lock, .python-version
├── CONTRIBUTING.md
└── README.md
```

---

## 6. Quản lý thư viện với uv

Thư viện khai báo trong `pyproject.toml`, phiên bản khóa trong `uv.lock`. **Commit cả hai file.**

```bash
uv add <package>                      # thêm thư viện (cập nhật pyproject.toml và uv.lock)
uv add --dev <package>                # thêm công cụ phát triển
uv remove <package>                   # gỡ thư viện
uv lock --upgrade-package <package>   # nâng một thư viện trong lock
uv sync --locked                      # cài đúng theo uv.lock
```

Nếu sửa `pyproject.toml` bằng tay, chạy `uv lock`. CI dùng `uv sync --locked` và báo lỗi nếu `uv.lock` không khớp.

---

## 7. Kiểm thử và chất lượng mã

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Test dùng dữ liệu tổng hợp, không cần tải MINC-2500.

---

## 8. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Chưa có dataset cục bộ` | Chạy `download_dataset.py`, hoặc kiểm tra `DATASET_ROOT` trong `.env` |
| `Thứ tự nhãn ... khác configs/default.yaml` | Dataset không đúng bản; kiểm tra revision |
| `Chưa có material_classification.features.extractor` | Chờ nhóm đặc trưng đẩy module lên, rồi `git pull` |
| `Cache ... không khớp cấu hình` | Chạy `extract_features.py --force` |
| `UnicodeEncodeError` khi in tiếng Việt (Windows) | Script đã tự đặt UTF-8; nếu vẫn gặp, chạy `$env:PYTHONUTF8=1` trong PowerShell |
| `uv.lock needs to be updated` trong CI | Chạy `uv lock` cục bộ rồi commit `uv.lock` |

---

## 9. Dữ liệu và trích dẫn

- MINC-2500: [Cornell OpenSurfaces](https://opensurfaces.cs.cornell.edu/publications/minc/). Nhãn CC BY 4.0; ảnh có giấy phép riêng. Chi tiết trong `docs/dataset.md`.
- Sean Bell, Paul Upchurch, Noah Snavely và Kavita Bala. *Material Recognition in the Wild with the Materials in Context Database*. CVPR 2015.
