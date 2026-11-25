# Phân loại độ tươi trái cây – demo trên Render

Demo Gradio phân loại ảnh 6 lớp (táo, chuối, cam × tươi, hỏng) bằng MobileNetV2 (transfer learning) chạy với ONNX
Runtime trên CPU.

- `src/` – mã demo Gradio (`app.py`), cấu hình và hàm đọc, đổi kích thước ảnh
- `models/` – `mobilenetv2.onnx` (khoảng 9 MB)
- `examples/` – 6 ảnh mẫu, mỗi lớp một ảnh
- `reports/` – bảng kết quả, ma trận nhầm lẫn và đồ thị huấn luyện hiển thị ở thẻ thông tin mô hình
- `render.yaml` – cấu hình triển khai trên Render (Python, gói Free, Singapore)

## Triển khai lên Render

1. Tạo kho GitHub mới (Public), đẩy toàn bộ nội dung thư mục này lên (tệp mô hình chỉ khoảng 9 MB nên tải lên bằng
   giao diện web của GitHub được).
2. Vào render.com, đăng nhập bằng GitHub → **New > Blueprint** → chọn kho vừa tạo → **Apply**.
3. Chờ build xong (thẻ **Logs**), mở địa chỉ `https://fruit-freshness-demo.onrender.com` (tên có thể khác nếu đã có
   người dùng).

Gói Free có 512 MB RAM và tự ngủ sau khoảng 15 phút không có truy cập; lần mở đầu sau đó chờ khoảng 30–60 giây.

## Chạy trên máy

```bash
pip install -r requirements.txt
python src/app.py
```
