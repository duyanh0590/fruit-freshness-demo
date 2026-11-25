"""Cấu hình dùng chung cho mọi bước (đường dẫn, lớp, tham số tiền xử lý). Sửa ở đây thay vì sửa trong từng file."""
import sys
from pathlib import Path

# In tiếng Việt không lỗi khi terminal Windows dùng bảng mã cp1252
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"            # nơi giải nén bộ dữ liệu Kaggle
SPLIT_DIR = ROOT / "data" / "splits"       # train.csv, val.csv, test.csv
PROCESSED_DIR = ROOT / "data" / "processed"  # đặc trưng histogram cho baseline
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

SEED = 42

# Thứ tự lớp = nhãn số 0..5 (khớp Bảng 2.2 trong báo cáo)
CLASSES = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
CLASS_VI = ["Táo tươi", "Chuối tươi", "Cam tươi", "Táo hỏng", "Chuối hỏng", "Cam hỏng"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIDE = 32          # ảnh có cạnh nhỏ hơn bị coi là không dùng được
VAL_SIZE = 0.15        # 15% số ảnh gốc làm tập kiểm định (phân tầng theo lớp)
TEST_SIZE = 0.15       # 15% số ảnh gốc làm tập kiểm thử; mọi phiên bản của một ảnh gốc nằm chung một tập

# MobileNetV2
IMG_SIZE = 224         # ảnh được đổi về 224 x 224; chuẩn hóa x / 127.5 - 1 nằm trong mô hình

# Baseline histogram màu
HIST_SIZE = 128        # ảnh được đổi về 128 x 128 trước khi tính histogram
HIST_BINS = 8          # 8 bin mỗi kênh H, S, V -> vector 8^3 = 512 chiều

# Tăng cường dữ liệu (chỉ cho tập huấn luyện) - Bảng 2.4
AUG_ROTATION_DEG = 36   # ±36° (tương đương RandomRotation(0.1) của Keras)
AUG_ZOOM = 0.10         # ±10%
AUG_TRANSLATE = 0.10    # ±10% theo mỗi chiều
AUG_BRIGHTNESS = 0.10   # ±10%
AUG_CONTRAST = 0.10     # ±10%
AUG_FILL_MODE = "nearest"  # vùng trống sau xoay/dịch lấy màu viền gần nhất (nền ảnh chủ yếu trắng)

# ---------------------------------------------------------------- huấn luyện và đánh giá
MODEL_DIR = ROOT / "models"                # mô hình đã huấn luyện (.joblib, .keras, .onnx)
PRED_DIR = REPORT_DIR / "predictions"      # dự đoán trên tập kiểm thử của từng mô hình (.npz)
ROTTEN = [3, 4, 5]                         # nhãn của nhóm "hỏng"

# Baseline k-NN / SVM - Bảng 2.5
CV_FOLDS = 5
KNN_GRID = {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"]}
SVM_GRID = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]}

# MobileNetV2 - mục 2.3, Bảng 2.7
BATCH_SIZE = 32
DROPOUT = 0.2
LR_STAGE1 = 1e-3       # giai đoạn 1: đóng băng backbone, chỉ huấn luyện lớp Dense
LR_STAGE2 = 1e-5       # giai đoạn 2: tinh chỉnh các lớp cuối của backbone
FINE_TUNE_AT = 100     # mở khóa backbone từ lớp thứ 100 trở đi
EPOCHS_STAGE1 = 5      # số epoch tối đa; dừng sớm thường kết thúc trước
EPOCHS_STAGE2 = 5
PATIENCE = 3           # dừng sớm khi val_loss không giảm sau 3 epoch
