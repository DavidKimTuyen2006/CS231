"""Tiện ích dùng chung cho phần dữ liệu: đường dẫn .env, đọc YAML, ghi JSON, thông tin tái lập."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Console Windows có thể dùng cp1252; tiếng Việt trong thông báo sẽ làm lỗi khi in.
for _stream in (sys.stdout, sys.stderr):
    if (
        hasattr(_stream, "reconfigure")
        and (_stream.encoding or "").lower().replace("-", "") != "utf8"
    ):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

load_dotenv(REPO_ROOT / ".env")

PACKAGES_FOR_METADATA = (
    "numpy",
    "opencv-python",
    "datasets",
    "pillow",
    "PyYAML",
    "python-dotenv",
)


def local_path(key: str, default: str) -> Path:
    """Đọc đường dẫn từ biến môi trường; đường dẫn tương đối tính từ thư mục gốc repo."""
    path = Path(os.getenv(key, default)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} phải là một mapping YAML")
    return data


def load_default_config() -> dict[str, Any]:
    return read_yaml(DEFAULT_CONFIG)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Băm JSON với khóa đã sắp xếp để kết quả ổn định giữa các máy."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Không ghi được JSON: {type(obj)!r}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_info() -> dict[str, Any]:
    """Commit hiện tại và trạng thái working tree; không báo lỗi nếu không phải repo Git."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES_FOR_METADATA:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions
