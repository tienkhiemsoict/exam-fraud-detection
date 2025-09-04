#!/usr/bin/env python3
"""
Demo script - Chạy cả 2 camera để demo toàn bộ hệ thống
"""

import os
import subprocess
import sys
import time

def run_demo():
    print("🎯 Exam Fraud Detection System - Full Demo")
    print("=" * 50)
    
    # Kiểm tra files cần thiết
    required_files = [
        "exam_detector_back.py",
        "exam_detector_front.py", 
        "videos/back_direction.mp4",
        "videos/front_direction.mp4",
        "grid_config_back.json",
        "grid_config_front.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Thiếu files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return
    
    print("✅ Tất cả files đã sẵn sàng!")
    print()
    
    # Menu lựa chọn
    print("📋 Chọn demo:")
    print("1. Camera sau (back_direction)")
    print("2. Camera trước (front_direction)")
    print("3. Chạy cả hai (tuần tự)")
    print("4. Test setup")
    print("0. Thoát")
    
    choice = input("\n👉 Nhập lựa chọn (1-4): ").strip()
    
    if choice == "1":
        print("\n🎬 Chạy demo camera sau...")
        subprocess.run([sys.executable, "exam_detector_back.py"])
        
    elif choice == "2":
        print("\n🎬 Chạy demo camera trước...")
        subprocess.run([sys.executable, "exam_detector_front.py"])
        
    elif choice == "3":
        print("\n🎬 Demo camera sau trước...")
        subprocess.run([sys.executable, "exam_detector_back.py"])
        
        print("\n⏳ Chờ 3 giây...")
        time.sleep(3)
        
        print("\n🎬 Demo camera trước...")
        subprocess.run([sys.executable, "exam_detector_front.py"])
        
    elif choice == "4":
        print("\n🧪 Chạy test setup...")
        subprocess.run([sys.executable, "test_setup.py"])
        
    elif choice == "0":
        print("👋 Thoát!")
        
    else:
        print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    run_demo()
