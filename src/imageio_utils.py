"""Đọc ảnh an toàn với đường dẫn có dấu tiếng Việt (cv2.imread không đọc được trên Windows)."""
from pathlib import Path

import cv2
import numpy as np

from config import ROOT


def abs_path(rel: str) -> Path:
    """Đường dẫn trong file CSV là tương đối so với thư mục gốc dự án."""
    return ROOT / rel


def read_rgb(path) -> np.ndarray:
    """Đọc ảnh thành mảng RGB uint8 3 kênh (ảnh xám / có kênh alpha đều được đưa về 3 kênh)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
