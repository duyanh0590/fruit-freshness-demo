"""Demo phân loại độ tươi trái cây: giao diện web Gradio + suy luận ONNX Runtime trên CPU (mục 3.7.2 của báo cáo).

Chạy sau export_onnx.py:  python src/app.py   rồi mở http://127.0.0.1:7860
Giao diện có 3 tab:
  1. Phân loại một ảnh: kết luận đạt / cần loại theo ngưỡng chỉnh được, xác suất 6 lớp, bảng chỉ số chi tiết
     (độ tin cậy, chênh lệch top-1/top-2, entropy, xác suất theo loại quả, thời gian), cảnh báo khi mô hình không
     chắc chắn và bản đồ vùng ảnh ảnh hưởng đến dự đoán (occlusion).
  2. Phân loại hàng loạt: tải nhiều ảnh, thống kê số quả đạt / cần loại, bảng kết quả và tệp CSV tải về.
  3. Thông tin mô hình: kiến trúc, kết quả trên tập kiểm thử, ma trận nhầm lẫn, đồ thị huấn luyện.
Ảnh chỉ cần đổi về 224 x 224 (chuẩn hóa đã nằm trong mô hình).
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort
import pandas as pd

from config import CLASS_VI, FIG_DIR, IMG_SIZE, MODEL_DIR, PRED_DIR, REPORT_DIR, ROOT, ROTTEN, SEED, SPLIT_DIR
from imageio_utils import abs_path, read_rgb, resize

session = ort.InferenceSession(str(MODEL_DIR / "mobilenetv2.onnx"), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
FRUITS = ["Táo", "Chuối", "Cam"]            # lớp k và k + 3 cùng một loại quả
UNSURE_CONF = 0.90                          # dưới ngưỡng này thì khuyên kiểm tra thủ công


def run(batch: np.ndarray) -> np.ndarray:
    return session.run(None, {INPUT_NAME: batch.astype(np.float32)})[0]


def prepare(image: np.ndarray) -> np.ndarray:
    return resize(np.ascontiguousarray(image[..., :3]), IMG_SIZE)


# ------------------------------------------------------------------ các chỉ số cho một ảnh
def describe(proba: np.ndarray, threshold: float) -> dict:
    order = np.argsort(proba)[::-1]
    k, k2 = int(order[0]), int(order[1])
    p_rotten = float(proba[ROTTEN].sum())
    fruit_p = [float(proba[i] + proba[i + 3]) for i in range(3)]
    entropy = float(-(proba * np.log(np.clip(proba, 1e-12, 1))).sum() / np.log(len(proba)))
    return {
        "k": k, "k2": k2, "conf": float(proba[k]), "margin": float(proba[k] - proba[k2]),
        "p_rotten": p_rotten, "reject": p_rotten >= threshold,
        "fruit": int(np.argmax(fruit_p)), "fruit_p": fruit_p, "entropy": entropy,
        "unsure": float(proba[k]) < UNSURE_CONF,
    }


def occlusion_map(img224: np.ndarray, k: int, patch: int = 48, stride: int = 22) -> np.ndarray:
    """Che lần lượt từng ô vuông của ảnh bằng màu xám, đo xác suất lớp k giảm bao nhiêu.

    Ô nào bị che làm xác suất giảm mạnh là vùng mô hình dựa vào để dự đoán. Trả về ảnh phủ bản đồ nhiệt."""
    base = float(run(img224[None])[0][k])
    pos = list(range(0, IMG_SIZE - patch + 1, stride))
    batch = np.repeat(img224[None].astype(np.float32), len(pos) ** 2, axis=0)
    for n, (y, x) in enumerate((y, x) for y in pos for x in pos):
        batch[n, y:y + patch, x:x + patch] = 127.5        # 127,5 -> 0 sau chuẩn hóa x / 127,5 - 1
    drop = np.maximum(base - run(batch)[:, k], 0).reshape(len(pos), len(pos))
    heat = np.zeros((IMG_SIZE, IMG_SIZE), np.float32); cnt = np.zeros_like(heat)
    for i, y in enumerate(pos):
        for j, x in enumerate(pos):
            heat[y:y + patch, x:x + patch] += drop[i, j]; cnt[y:y + patch, x:x + patch] += 1
    heat = cv2.GaussianBlur(heat / np.maximum(cnt, 1), (0, 0), patch / 4)   # làm mượt các ô vuông
    heat = (255 * heat / heat.max()).astype(np.uint8) if heat.max() > 0 else heat.astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img224, 0.55, color, 0.45, 0)


def predict(image: np.ndarray, threshold: float = 0.5, show_map: bool = True):
    empty = (None, "Hãy tải ảnh lên, chụp bằng webcam hoặc bấm một ảnh mẫu.", None, None)
    if image is None:
        return empty
    t0 = time.perf_counter()
    img = prepare(image)
    t1 = time.perf_counter()
    proba = run(img[None])[0]
    t2 = time.perf_counter()
    d = describe(proba, threshold)

    if d["reject"]:
        verdict = f"## ❌ {CLASS_VI[d['k']]} – cần loại khỏi lô hàng"
    else:
        verdict = f"## ✅ {CLASS_VI[d['k']]} – đạt chất lượng"
    verdict += (f"\nĐộ tin cậy {d['conf']:.1%} · xác suất quả hỏng {d['p_rotten']:.1%} "
                f"(ngưỡng loại {threshold:.0%}) · thời gian suy luận {(t2 - t1) * 1000:.0f} ms")
    if d["unsure"]:
        verdict += (f"\n\n> ⚠️ Độ tin cậy dưới {UNSURE_CONF:.0%}: mô hình đang phân vân giữa **{CLASS_VI[d['k']]}** "
                    f"và **{CLASS_VI[d['k2']]}** – nên kiểm tra thủ công.")
    if d["reject"] != (d["k"] in ROTTEN):
        verdict += ("\n\n> ℹ️ Kết luận khác với lớp có xác suất cao nhất vì tổng xác suất của ba lớp hỏng "
                    f"{'vượt' if d['reject'] else 'chưa tới'} ngưỡng loại đã chọn.")

    h, w = image.shape[:2]
    rows = [
        ("Lớp dự đoán", CLASS_VI[d["k"]]),
        ("Độ tin cậy (xác suất lớp dự đoán)", f"{d['conf']:.2%}"),
        ("Lớp đứng thứ hai", f"{CLASS_VI[d['k2']]} ({proba[d['k2']]:.2%})"),
        ("Chênh lệch top-1 – top-2", f"{d['margin']:.2%}"),
        ("Entropy chuẩn hóa (0 = chắc chắn, 1 = phân vân nhất)", f"{d['entropy']:.3f}"),
        ("Xác suất quả tươi / quả hỏng", f"{1 - d['p_rotten']:.2%} / {d['p_rotten']:.2%}"),
        ("Loại quả dự đoán", f"{FRUITS[d['fruit']]} ({d['fruit_p'][d['fruit']]:.2%})"),
        ("Xác suất theo loại quả (táo / chuối / cam)", " / ".join(f"{p:.1%}" for p in d["fruit_p"])),
        ("Kích thước ảnh gốc", f"{w} × {h} điểm ảnh"),
        ("Thời gian tiền xử lý / suy luận", f"{(t1 - t0) * 1000:.1f} ms / {(t2 - t1) * 1000:.1f} ms"),
    ]
    table = pd.DataFrame(rows, columns=["Chỉ số", "Giá trị"])

    heat = occlusion_map(img, d["k"]) if show_map else None
    return {CLASS_VI[i]: float(p) for i, p in enumerate(proba)}, verdict, table, heat


# ------------------------------------------------------------------ phân loại hàng loạt
def predict_batch(files, threshold: float = 0.5):
    if not files:
        return "Hãy chọn một hoặc nhiều ảnh.", None, None, None
    rows, gallery = [], []
    t0 = time.perf_counter()
    for f in files:
        path = Path(f if isinstance(f, str) else f.name)
        try:
            image = read_rgb(path)
        except ValueError:
            rows.append({"Tệp": path.name, "Dự đoán": "Không đọc được ảnh"}); continue
        proba = run(prepare(image)[None])[0]
        d = describe(proba, threshold)
        rows.append({"Tệp": path.name, "Dự đoán": CLASS_VI[d["k"]], "Độ tin cậy": round(d["conf"], 4),
                     "Xác suất hỏng": round(d["p_rotten"], 4),
                     "Kết luận": "Cần loại" if d["reject"] else "Đạt",
                     "Cần kiểm tra thủ công": "Có" if d["unsure"] else ""})
        gallery.append((image, f"{'❌' if d['reject'] else '✅'} {CLASS_VI[d['k']]} {d['conf']:.0%}"))
    total_ms = (time.perf_counter() - t0) * 1000
    df = pd.DataFrame(rows)
    ok = df[df["Dự đoán"] != "Không đọc được ảnh"]
    n_rej = int((ok["Kết luận"] == "Cần loại").sum()) if len(ok) else 0
    n_unsure = int((ok["Cần kiểm tra thủ công"] == "Có").sum()) if len(ok) else 0
    summary = (f"### Đã phân loại {len(ok)} ảnh trong {total_ms / 1000:.2f} s "
               f"(trung bình {total_ms / max(len(rows), 1):.0f} ms/ảnh, gồm cả đọc ảnh)\n"
               f"- ✅ Đạt chất lượng: **{len(ok) - n_rej}** · ❌ Cần loại: **{n_rej}** "
               f"(tỷ lệ loại {n_rej / max(len(ok), 1):.1%}, ngưỡng {threshold:.0%})\n"
               f"- ⚠️ Cần kiểm tra thủ công (độ tin cậy < {UNSURE_CONF:.0%}): **{n_unsure}**")
    if len(ok):
        counts = ok["Dự đoán"].value_counts()
        summary += "\n- Số ảnh theo lớp: " + ", ".join(f"{c} {n}" for c, n in counts.items())
    if len(ok) < len(rows):
        summary += f"\n- Không đọc được: {len(rows) - len(ok)} tệp"
    csv = Path(tempfile.gettempdir()) / "ket_qua_phan_loai.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    return summary, gallery, df, str(csv)


# ------------------------------------------------------------------ thông tin mô hình
def model_info():
    def read_json(p):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            return {}
    pred, onnx = read_json(PRED_DIR / "mobilenetv2.json"), read_json(REPORT_DIR / "onnx_check.json")
    md = ("### MobileNetV2 – transfer learning hai giai đoạn\n"
          "Ảnh 224 × 224 × 3 → chuẩn hóa về [−1, 1] → MobileNetV2 (trọng số ImageNet) → gộp trung bình toàn cục → "
          "Dropout 0,2 → Dense 6 lớp (softmax).  \n"
          "Giai đoạn 1: đóng băng backbone, tốc độ học 1e-3; giai đoạn 2: tinh chỉnh từ lớp thứ 100, tốc độ học 1e-5.  \n"
          "Dữ liệu: 13.599 ảnh (1.511 ảnh gốc × 9 phiên bản), chia 70/15/15 theo ảnh gốc.\n\n")
    if pred:
        md += f"- Số tham số: **{pred.get('params', 0):,}**".replace(",", ".") + "\n"
        md += f"- Accuracy / macro F1 trên tập kiểm định: **{pred.get('val_accuracy', 0):.2%} / {pred.get('val_macro_f1', 0):.3f}**\n"
    if onnx:
        md += (f"- Tệp ONNX: **{onnx.get('onnx_size_mb', 0):.1f} MB** (Keras {onnx.get('keras_size_mb', 0):.1f} MB); "
               f"accuracy ONNX trên tập kiểm thử **{onnx.get('onnx_test_accuracy', 0):.2%}**; "
               f"nhãn trùng với Keras **{onnx.get('label_agreement', 0):.0%}**; "
               f"thời gian suy luận trung vị **{onnx.get('onnx_ms_per_image_median', 0):.1f} ms/ảnh**\n")

    def read_csv(name):
        try:
            return pd.read_csv(REPORT_DIR / name, encoding="utf-8-sig")
        except Exception:
            return None
    per_class = read_csv("bang_3_x_theo_lop_mobilenetv2.csv")
    if per_class is not None and "support" in per_class.columns:
        per_class["support"] = per_class["support"].astype(int)
    figs = [str(FIG_DIR / n) for n in ("hinh_3_x_ma_tran_nham_lan_mobilenetv2.png", "hinh_3_x_do_thi_huan_luyen.png")]
    return md, per_class, *[f if Path(f).exists() else None for f in figs]


def example_images():
    """Mỗi lớp một ảnh của tập kiểm thử để bấm thử nhanh.

    Lấy từ thư mục examples/ đi kèm mã nguồn (đã chép sẵn 6 ảnh), nên demo chạy được cả khi chưa tải bộ dữ liệu;
    nếu thiếu thì chọn ngẫu nhiên từ test.csv như cũ."""
    bundled = sorted((ROOT / "examples").glob("*.png"))
    if bundled:
        return [str(p) for p in bundled]
    df = pd.read_csv(SPLIT_DIR / "test.csv")
    paths = [abs_path(g["path"].sample(1, random_state=SEED).iat[0]) for _, g in df.groupby("label")]
    return [str(p) for p in paths if p.exists()]


INFO = model_info()

with gr.Blocks(title="Phân loại độ tươi trái cây") as demo:
    # font mặc định của Gradio thiếu một số chữ tiếng Việt in đậm (ví dụ "Ủ") nên tiêu đề dùng Segoe UI / Arial
    gr.HTML('<h1 style="font-family: \'Segoe UI\', Arial, sans-serif; font-weight: 700; font-size: 2rem; '
            'margin: 0 0 10px 0;">PHÂN LOẠI ĐỘ TƯƠI CỦA TRÁI CÂY</h1>'
            '<p style="margin: 0;">MobileNetV2 (transfer learning) · ONNX Runtime · 6 lớp: táo, chuối, cam × tươi, hỏng. '
            'Nên dùng ảnh chụp một quả trên nền đơn giản.</p>', padding=False)
    with gr.Tab("Phân loại một ảnh"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Ảnh trái cây", height=340)
                threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Ngưỡng xác suất quả hỏng để loại bỏ",
                                      info="Hạ ngưỡng → loại khắt khe hơn (ít bỏ sót quả hỏng, nhưng loại nhầm nhiều quả tươi hơn)")
                show_map = gr.Checkbox(value=True, label="Hiển thị vùng ảnh ảnh hưởng đến dự đoán (occlusion, thêm ~1 s)")
                button = gr.Button("Phân loại", variant="primary")
                gr.Examples(example_images(), inputs=image, label="Ảnh mẫu từ tập kiểm thử")
            with gr.Column():
                verdict = gr.Markdown("Kết quả sẽ hiển thị ở đây.")
                label = gr.Label(num_top_classes=6, label="Xác suất của 6 lớp")
                heatmap = gr.Image(label="Vùng ảnh mô hình dựa vào (đỏ = che vùng này làm xác suất giảm mạnh nhất)",
                                   height=260, interactive=False)
        details = gr.Dataframe(label="Chỉ số chi tiết", interactive=False, wrap=True, column_widths=["52%", "48%"])
        outputs = [label, verdict, details, heatmap]
        button.click(predict, [image, threshold, show_map], outputs)
        image.change(predict, [image, threshold, show_map], outputs)
        threshold.release(predict, [image, threshold, show_map], outputs)

    with gr.Tab("Phân loại hàng loạt"):
        gr.Markdown("Chọn nhiều ảnh cùng lúc (ví dụ ảnh của một lô hàng) để thống kê số quả đạt và cần loại.")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(file_count="multiple", file_types=["image"], label="Ảnh trái cây")
                threshold_b = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Ngưỡng xác suất quả hỏng để loại bỏ")
                button_b = gr.Button("Phân loại tất cả", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                gallery = gr.Gallery(label="Kết quả", columns=6, height=300)
        table_b = gr.Dataframe(label="Kết quả từng ảnh", interactive=False, wrap=True)
        csv_b = gr.File(label="Tải bảng kết quả (CSV)")
        button_b.click(predict_batch, [files, threshold_b], [summary, gallery, table_b, csv_b])

    with gr.Tab("Thông tin mô hình"):
        gr.Markdown(INFO[0])
        gr.Dataframe(INFO[1], label="MobileNetV2 – kết quả theo từng lớp trên tập kiểm thử (2.043 ảnh)", interactive=False)
        with gr.Row():
            gr.Image(INFO[2], label="Ma trận nhầm lẫn của MobileNetV2", interactive=False, height=420)
            gr.Image(INFO[3], label="Accuracy và loss qua hai giai đoạn huấn luyện", interactive=False, height=420)

if __name__ == "__main__":
    # Trên Render (có biến PORT) phải nghe ở 0.0.0.0, đúng cổng được cấp và không mở trình duyệt;
    # --no-browser: chạy trên máy nhưng không tự mở trình duyệt
    on_cloud = "PORT" in os.environ
    demo.launch(server_name="0.0.0.0" if on_cloud else "127.0.0.1", server_port=int(os.environ.get("PORT", 7860)),
                inbrowser=not on_cloud and "--no-browser" not in sys.argv)
