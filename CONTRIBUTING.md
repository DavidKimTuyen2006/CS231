# Đóng góp vào repo

## Quy trình

1. Cập nhật `main`: `git switch main` rồi `git pull origin main`.
2. Tạo nhánh ngắn theo nhiệm vụ: `git switch -c feat/<ten>` (hoặc `fix/`, `docs/`, `exp/`, `refactor/`).
3. Cài môi trường và viết code: `uv sync --locked`.
4. Trước khi commit: `uv run ruff format .`, `uv run ruff check .`, `uv run pytest`.
5. Mở Pull Request vào `main`, liên kết Issue, mô tả thay đổi và cách kiểm thử.
6. Merge khi test đạt và có ít nhất một người khác review (nếu có thể).

Không push trực tiếp lên `main` khi đã bật bảo vệ nhánh.

## Thư viện

- Dùng `uv add` / `uv remove` để cập nhật `pyproject.toml` và `uv.lock` cùng lúc.
- Commit cả `pyproject.toml` lẫn `uv.lock`. Không sửa `uv.lock` bằng tay.
- Đổi thư viện có ảnh hưởng số học (ví dụ scikit-learn, scikit-image) cần ghi rõ trong PR.

## Không commit

- Dataset, Arrow, Parquet, ảnh gốc (`data/`, `DATASET_ROOT`).
- Feature cache (`cache/`), model, checkpoint, output chi tiết (`outputs/`).
- File `.env`, token Hugging Face, đường dẫn riêng của máy.

Kiểm tra trước khi commit: `git status --short` và `git diff --cached --stat`.

Có thể commit: `src/`, `scripts/`, `configs/`, `tests/`, `docs/`, `README.md`, `CONTRIBUTING.md`, `.env.example`, `.gitignore`, `pyproject.toml`, `uv.lock`, `.github/`, và kết quả tổng hợp nhỏ trong `results/`.

## Quy tắc khi sửa thuật toán

- Bám công thức Chương 3. Fusion dùng căn bậc hai của α: `sqrt(α)·HOG + sqrt(1-α)·HSV`.
- Không resize tuần tự 128 → 64; mỗi độ phân giải resize trực tiếp từ RGB gốc.
- Không fit scaler, PCA hay thống kê nào trên validation hoặc test.
- Không dùng test để chọn tham số. Chốt cấu hình trước, rồi mới đánh giá test.
- Không hard-code đường dẫn cá nhân. Đường dẫn đi qua `.env`.
- Mỗi thay đổi đặc trưng cần test shape, dtype và giá trị hữu hạn.

## Prefix commit

| Prefix | Mục đích |
|---|---|
| `feat:` | Thêm thuật toán hoặc chức năng |
| `fix:` | Sửa lỗi |
| `refactor:` | Tái cấu trúc, không đổi hành vi |
| `test:` | Kiểm thử |
| `docs:` | Tài liệu |
| `exp:` | Cấu hình thí nghiệm |
| `perf:` | Tối ưu hiệu năng |
| `deps:` | Đổi thư viện hoặc lock |
