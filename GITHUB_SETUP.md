# 📋 Hướng dẫn đẩy lên GitHub

## 1. Khởi tạo Git repository (trong thư mục exam_fraud_detection)

```bash
# Khởi tạo git
git init

# Thêm tất cả files
git add .

# Commit đầu tiên
git commit -m "🎉 Initial commit - Exam Fraud Detection System

✨ Features:
- YOLOv8 person detection with ByteTrack tracking
- Smart caching (detect every 5 frames)
- Dual camera support (front/back)
- Automatic seat assignment
- Performance statistics
- Ready-to-run with demo videos

📦 Includes:
- Demo videos and configs
- YOLO models (8n, 8s)
- Complete documentation
- Setup scripts for easy installation"
```

## 2. Tạo repository trên GitHub

1. Vào https://github.com
2. Click "New repository" 
3. Đặt tên: `exam-fraud-detection`
4. Mô tả: `🎯 Smart exam fraud detection system using YOLOv8 and ByteTrack - Ready to run with demo videos`
5. Chọn "Public" 
6. **KHÔNG** check "Add README" (vì đã có rồi)
7. Click "Create repository"

## 3. Kết nối và đẩy code

```bash
# Kết nối với GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/exam-fraud-detection.git

# Đẩy code lên
git branch -M main
git push -u origin main
```

## 4. Sau khi đẩy lên

### Kiểm tra repository:
- ✅ README.md hiển thị đẹp
- ✅ Có videos/ folder với demo videos
- ✅ Có models (.pt files)
- ✅ Có configs (.json files)
- ✅ Có requirements.txt

### Tạo release đầu tiên:
1. Vào repository trên GitHub
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0.0`
4. Title: `🎉 First Release - Ready to Run Demo`
5. Description:
```markdown
## 🚀 Exam Fraud Detection v1.0.0

### ✨ Features
- Complete working system with demo videos
- YOLOv8 detection + ByteTrack tracking  
- Smart caching for 5x speed improvement
- Dual camera support
- Automatic seat assignment

### 📦 What's Included
- ✅ Demo videos (back_direction.mp4, front_direction.mp4)
- ✅ Pre-trained models (yolov8n.pt, yolov8s.pt) 
- ✅ Grid configurations (pre-calibrated)
- ✅ Complete documentation
- ✅ Setup scripts

### 🏃‍♂️ Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/exam-fraud-detection.git
cd exam-fraud-detection
pip install -r requirements.txt
python exam_detector_back.py
```

### 🎮 Try the Demo
```bash
python demo.py
```
```

## 5. Tối ưu repository

### Thêm topics (tags):
- `computer-vision`
- `yolo`
- `object-detection`
- `tracking`
- `exam-monitoring`
- `python`
- `opencv`
- `ultralytics`

### Cập nhật description:
`🎯 Smart exam fraud detection system using YOLOv8 and ByteTrack. Ready-to-run with demo videos and pre-trained models. Features smart caching and dual camera support.`

## 6. File sizes cần chú ý

GitHub có giới hạn:
- File < 100MB: OK
- File 100MB+: Cần Git LFS

Kiểm tra file sizes:
```bash
find . -size +50M -type f
```

Nếu có file lớn, dùng Git LFS:
```bash
git lfs install
git lfs track "*.mp4"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS for large files"
```

## 🎉 Hoàn thành!

Repository sẽ có:
- 📖 Documentation đầy đủ  
- 🎬 Demo videos sẵn sàng
- 🤖 Pre-trained models
- ⚙️ Configs đã calibrate
- 🚀 Setup scripts
- 💾 Mọi thứ để clone và chạy ngay
