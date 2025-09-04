#!/usr/bin/env python3
"""
Quick test script để kiểm tra dự án đã setup đúng chưa
"""

import os
import sys

def check_dependencies():
    """Kiểm tra các dependencies"""
    print("🔍 Kiểm tra dependencies...")
    
    try:
        import cv2
        print("✅ OpenCV: OK")
    except ImportError:
        print("❌ OpenCV: MISSING - Run: pip install opencv-python")
        return False
    
    try:
        import numpy
        print("✅ NumPy: OK")
    except ImportError:
        print("❌ NumPy: MISSING - Run: pip install numpy")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics: OK")
    except ImportError:
        print("❌ Ultralytics: MISSING - Run: pip install ultralytics")
        return False
    
    try:
        import torch
        print("✅ PyTorch: OK")
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available - {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA: Not available (CPU only)")
    except ImportError:
        print("❌ PyTorch: MISSING - Run: pip install torch")
        return False
    
    return True

def check_files():
    """Kiểm tra các file cần thiết"""
    print("\n📁 Kiểm tra files...")
    
    required_files = [
        "exam_detector_back.py",
        "exam_detector_front.py",
        "grid_config_back.json", 
        "grid_config_front.json",
        "yolov8n.pt",
        "yolov8s.pt",
        "videos/back_direction.mp4",
        "videos/front_direction.mp4"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: MISSING")
            all_good = False
    
    return all_good

def main():
    print("🚀 Exam Fraud Detection - Quick Test")
    print("=" * 50)
    
    # Kiểm tra dependencies
    deps_ok = check_dependencies()
    
    # Kiểm tra files
    files_ok = check_files()
    
    print("\n" + "=" * 50)
    if deps_ok and files_ok:
        print("🎉 Mọi thứ đã sẵn sàng!")
        print("\n🏃‍♂️ Bạn có thể chạy:")
        print("   python exam_detector_back.py")
        print("   python exam_detector_front.py")
    else:
        print("⚠️ Cần khắc phục một số vấn đề trước khi chạy")
        if not deps_ok:
            print("📦 Chạy: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
