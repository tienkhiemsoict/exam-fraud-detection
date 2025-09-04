# Cấu trúc
```
exam_fraud_detection/
├── 📹 videos/                          # Video input
│   ├── back_direction.mp4              # Video demo camera sau
│   └── front_direction.mp4             # Video demo camera trước
├── 📤 output/                          # Video output (được tạo tự động)
│   ├── back_direction_detected.mp4     # Kết quả camera sau
│   └── front_direction_detected.mp4    # Kết quả camera trước
├── 🤖 models/                         
│   └── yolov8n.pt                      
├── ⚙️ grid_config_back.json            # Cấu hình ghế camera sau (đã set)
├── ⚙️ grid_config_front.json           # Cấu hình ghế camera trước (đã set)
├── 🧠 yolov8n.pt                       # YOLO model nhẹ
├── 🧠 yolov8s.pt                       # YOLO model chuẩn (dùng model này)
├── 🔧 exam_detector_back.py            # Detector camera sau
├── 🔧 exam_detector_front.py           # Detector camera trước
├── 📐 room_grid_calibrator_back.py     # Calibration camera sau (nếu muốn set lại)
├── 📐 room_grid_calibrator_front.py    # Calibration camera trước (nếu muốn set lại)
└── 📋 requirements.txt                
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



*Hệ thống đang hoạt động với detection và tracking real-time*
