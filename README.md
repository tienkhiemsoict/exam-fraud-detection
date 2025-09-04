# 🎯 Exam Fraud Detection System

Hệ thống phát hiện gian lận thi cử thông minh sử dụng YOLOv8 và ByteTrack tracking. Hệ thống có thể phát hiện và theo dõi người trong phòng thi, gán vị trí ghế ngồi và phát hiện các hành vi bất thường.

## ✨ Tính năng

- 🔍 **Phát hiện người**: Sử dụng YOLOv8 với độ chính xác cao
- 🎯 **Tracking thông minh**: ByteTrack để theo dõi liên tục
- 📍 **Gán vị trí ghế**: Tự động gán vị trí ghế ngồi cho từng người
- ⚡ **Tối ưu hiệu suất**: Cache thông minh - detect mỗi 5 frame
- 📊 **Thống kê chi tiết**: Báo cáo hiệu suất và kết quả detection
- 🎥 **Dual camera**: Hỗ trợ camera trước và sau
- 💾 **Lưu video**: Xuất video đã xử lý với annotations

## 🚀 Cài đặt nhanh

### 1. Clone repository
```bash
git clone https://github.com/your-username/exam_fraud_detection.git
cd exam_fraud_detection
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy ngay (đã có sẵn video demo và config)
```bash
# Camera sau
python exam_detector_back.py

# Camera trước  
python exam_detector_front.py
```

## 📁 Cấu trúc dự án

```
exam_fraud_detection/
├── 📹 videos/                          # Video input (đã có sẵn demo)
│   ├── back_direction.mp4              # Video demo camera sau
│   └── front_direction.mp4             # Video demo camera trước
├── 📤 output/                          # Video output (được tạo tự động)
│   ├── back_direction_detected.mp4     # Kết quả camera sau
│   └── front_direction_detected.mp4    # Kết quả camera trước
├── 🤖 models/                          # Model weights
│   └── yolov8n.pt                      # Model backup
├── ⚙️ grid_config_back.json            # Cấu hình ghế camera sau
├── ⚙️ grid_config_front.json           # Cấu hình ghế camera trước
├── 🧠 yolov8n.pt                       # YOLO model nhẹ
├── 🧠 yolov8s.pt                       # YOLO model chuẩn
├── 🔧 exam_detector_back.py            # Detector camera sau
├── 🔧 exam_detector_front.py           # Detector camera trước
├── 📐 room_grid_calibrator_back.py     # Calibration camera sau
├── 📐 room_grid_calibrator_front.py    # Calibration camera trước
└── 📋 requirements.txt                 # Dependencies
```

## 🎮 Sử dụng

### Chạy detection với video có sẵn
```bash
# Camera sau (khuyến nghị chạy đầu tiên)
python exam_detector_back.py

# Camera trước
python exam_detector_front.py
```

### Sử dụng video của bạn
1. Đặt video vào thư mục `videos/` với tên:
   - `back_direction.mp4` cho camera sau
   - `front_direction.mp4` cho camera trước

2. Calibrate lại vị trí ghế (nếu cần):
```bash
# Calibrate camera sau
python room_grid_calibrator_back.py

# Calibrate camera trước  
python room_grid_calibrator_front.py
```

3. Chạy detection:
```bash
python exam_detector_back.py
python exam_detector_front.py
```

## 📊 Kết quả mẫu

Sau khi chạy, bạn sẽ thấy output như:

```
Kết quả xử lý video:
- Tổng số frame: 1500
- Số frame đã xử lý: 1500
- Số frame thực sự detect: 300
- Hiệu quả cache: 80.0%
- Thời gian xử lý TB: 0.033s/frame
- Thời gian detect TB: 0.165s/detection
- Số người TB/frame: 12.5
```

## 🎨 Visualization

- 🟢 **Hộp xanh**: Vị trí đã ổn định
- 🟠 **Hộp cam**: Vị trí tạm thời
- 🔴 **Hộp đỏ**: Chưa xác định vị trí
- ⚫ **Điểm xanh**: Vị trí ghế ngồi
- ➖ **Đường nối**: Liên kết người với ghế

## ⚙️ Tùy chỉnh

### Thay đổi tham số detection
Chỉnh sửa trong file detector:

```python
detector = ExamDetectorBack(
    model_path="yolov8s.pt",           # Model: yolov8n.pt (nhanh) hoặc yolov8s.pt (chính xác)
    confidence_threshold=0.25,          # Ngưỡng tin cậy (0.1-0.9)
    input_size=1024,                   # Kích thước input (640, 1024, 1280)
    iou_threshold=0.40,                # Ngưỡng IoU (0.1-0.9)
    position_threshold=150,            # Khoảng cách tối đa đến ghế (pixels)
    detect_interval=5,                 # Detect mỗi N frame (1-10)
    stability_threshold=4,             # Số frame để xác nhận ổn định
    fail_threshold=3                   # Số frame thất bại để reset
)
```

### Calibrate lại vị trí ghế
1. Chạy calibrator:
```bash
python room_grid_calibrator_back.py    # Cho camera sau
python room_grid_calibrator_front.py   # Cho camera trước
```

2. Click để đánh dấu vị trí ghế trên video
3. Nhấn 's' để lưu cấu hình
4. File config sẽ được lưu tự động

## 🔧 Troubleshooting

### Lỗi thường gặp

1. **"No module named 'ultralytics'"**
```bash
pip install ultralytics
```

2. **"CUDA out of memory"**
- Giảm `input_size` xuống 640
- Hoặc dùng `yolov8n.pt` thay vì `yolov8s.pt`

3. **"Video file not found"**
- Kiểm tra file video trong thư mục `videos/`
- Đảm bảo tên file đúng: `back_direction.mp4` hoặc `front_direction.mp4`

4. **"Config file not found"**
- Chạy calibrator trước: `python room_grid_calibrator_back.py`
- Hoặc dùng config mẫu có sẵn

### Tối ưu hiệu suất

- **Máy yếu**: Dùng `yolov8n.pt`, `input_size=640`, `detect_interval=10`
- **Máy mạnh**: Dùng `yolov8s.pt`, `input_size=1280`, `detect_interval=3`
- **Cân bằng**: Cấu hình mặc định

## 📦 Dependencies

- **ultralytics**: YOLOv8 detection và tracking
- **opencv-python**: Xử lý video và hình ảnh  
- **numpy**: Tính toán numerical
- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

## 📄 License

Dự án này được phát hành dưới [MIT License](LICENSE).

## 📞 Liên hệ

- Email: your.email@example.com
- GitHub: [@your-username](https://github.com/your-username)

## 🎯 Demo

![Demo GIF](demo.gif)

*Hệ thống đang hoạt động với detection và tracking real-time*

---

⭐ **Nếu dự án hữu ích, hãy cho một Star!** ⭐
