import cv2
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

class RoomGridCalibrator:
    """Công cụ hiệu chỉnh vị trí chỗ ngồi trong phòng thi"""

    def __init__(self):
        # Khởi tạo các biến quản lý điểm mốc
        self.landmarks: List[Dict] = []  # Danh sách các điểm mốc đã đánh dấu
        self.position_counter = 1        # Bộ đếm vị trí từ 1 trở đi
        
        # Khởi tạo các biến hiển thị
        self.window_name = 'Hiệu chỉnh vị trí - ESC để hoàn thành, R để xoá điểm cuối'
        self.original_image = None       # Ảnh gốc không có điểm mốc
        self.image = None                # Ảnh đang hiển thị (có điểm mốc)
        
        # Text hướng dẫn
        self.help_text = [
            "HƯỚNG DẪN:",
            "- Click chuột TRÁI để đánh dấu vị trí",
            "- Nhấn R để xoá điểm vừa đánh dấu",
            "- Nhấn ESC để hoàn thành",
            f"Đang đánh dấu vị trí: {self._get_next_position_label()}"
        ]

    def _get_next_position_label(self) -> str:
        """Tạo nhãn cho vị trí tiếp theo (1, 2, 3,...)"""
        return str(self.position_counter)

    def _update_help_text(self):
        """Cập nhật text hướng dẫn"""
        self.help_text[-1] = f"Đang đánh dấu vị trí: {self._get_next_position_label()}"

    def _advance_position(self):
        """Di chuyển đến vị trí tiếp theo"""
        self.position_counter += 1
        self._update_help_text()

    def _revert_position(self):
        """Quay lại vị trí trước đó"""
        if self.position_counter > 1:
            self.position_counter -= 1
        self._update_help_text()

    def _remove_last_landmark(self):
        """Xoá điểm mốc cuối cùng"""
        if self.landmarks:
            self.landmarks.pop()
            self._revert_position()
            # Vẽ lại hình ảnh
            self._redraw_image()

    def _redraw_image(self):
        """Vẽ lại toàn bộ hình ảnh với các điểm mốc"""
        if self.original_image is not None:
            self.image = self.original_image.copy()
            
            # Vẽ lại các điểm mốc
            for landmark in self.landmarks:
                x, y = landmark['position']
                cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.image, landmark['label'], 
                          (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)

            # Vẽ hướng dẫn
            y_offset = 30
            for i, text in enumerate(self.help_text):
                cv2.putText(self.image, text, (10, y_offset + i*25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            try:
                cv2.imshow(self.window_name, self.image)
            except Exception as e:
                print(f"Lỗi hiển thị hình ảnh: {e}")

    def _mouse_callback(self, event, x, y, flags, param):
        """Xử lý sự kiện chuột"""
        if event == cv2.EVENT_LBUTTONDOWN:  # Click chuột trái
            # Thêm điểm mốc mới
            label = self._get_next_position_label()
            self.landmarks.append({
                'position': (x, y),
                'label': label
            })
            
            # Vẽ điểm mốc
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.image, label, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Di chuyển đến vị trí tiếp theo và cập nhật hiển thị
            self._advance_position()
            cv2.imshow(self.window_name, self.image)

    def calibrate(self, source_path: str) -> List[Dict]:
        """Bắt đầu quá trình hiệu chỉnh
        
        Args:
            source_path: Đường dẫn đến video hoặc ảnh phòng thi
            
        Returns:
            List[Dict]: Danh sách các điểm mốc đã đánh dấu
        """
        # Kiểm tra xem đầu vào là video hay ảnh
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise ValueError("Không thể mở video/ảnh từ đường dẫn đã cho")

        # Nếu là video, lấy frame đầu tiên
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Không thể đọc frame từ video/ảnh")
        cap.release()

        # Lưu ảnh gốc và tạo bản sao để vẽ
        self.original_image = frame
        self.image = frame.copy()
        
        # Tạo cửa sổ và thiết lập callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Hiển thị hướng dẫn ban đầu
        self._redraw_image()
        
        # Vòng lặp chính
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):  # Xoá điểm cuối
                self._remove_last_landmark()
                
        cv2.destroyAllWindows()
        return self.landmarks
        
    def save_landmarks(self, output_path: str):
        """Lưu các điểm mốc vào file JSON
        
        Args:
            output_path: Đường dẫn file JSON đầu ra
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.landmarks, f, ensure_ascii=False, indent=2)
            
    @staticmethod
    def load_landmarks(input_path: str) -> List[Dict]:
        """Đọc các điểm mốc từ file JSON
        
        Args:
            input_path: Đường dẫn file JSON
            
        Returns:
            List[Dict]: Danh sách các điểm mốc
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def main():
    import os
    # Lấy đường dẫn tuyệt đối của thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "videos", "front_direction.mp4")
    config_path = os.path.join(current_dir, "grid_config.json")
    
    print(f"Thư mục hiện tại: {current_dir}")
    print(f"Đường dẫn video: {video_path}")
    
    # Kiểm tra file video có tồn tại
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại {video_path}")
        print("Vui lòng đặt video vào thư mục videos/ với tên front_direction.mp4")
        return
        
    # Tạo thư mục videos nếu chưa có
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
        print(f"Đã tạo thư mục {os.path.dirname(video_path)}")
    
    # Tạo calibrator
    calibrator = RoomGridCalibrator()
    
    # Thực hiện hiệu chỉnh
    try:
        print("Bắt đầu hiệu chỉnh vị trí phòng thi...")
        print("1. Click chuột TRÁI để đánh dấu vị trí")
        print("2. Nhấn R để xoá điểm vừa đánh dấu")
        print("3. Nhấn ESC để hoàn thành")
        
        landmarks = calibrator.calibrate(video_path)
        
        # Lưu kết quả
        if landmarks:
            calibrator.save_landmarks(config_path)
            print(f"Đã lưu {len(landmarks)} vị trí vào {config_path}")
        else:
            print("Không có vị trí nào được đánh dấu")
            
    except Exception as e:
        import traceback
        print(f"Lỗi: {e}")
        print("Chi tiết lỗi:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
    def save_config(self, filename):
        """Lưu cấu hình"""
        config = {
            'lines': self.lines
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f)
        print(f"Đã lưu cấu hình vào {filename}")
    
    def load_config(self, filename):
        """Tải cấu hình"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            self.lines = config['lines']
            print(f"Đã tải cấu hình từ {filename}")
        except Exception as e:
            print(f"Lỗi khi tải cấu hình: {str(e)}")
    
    def calibrate(self, video_path):
        """Chạy quá trình căn chỉnh"""
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        if not ret:
            print("Không thể đọc video!")
            return
            
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== HƯỚNG DẪN SỬ DỤNG ===")
        print("\n1. Vẽ đường thẳng:")
        print("   - Click lần 1: đánh dấu điểm đầu (đỏ)")
        print("   - Di chuyển chuột: xem preview đường thẳng (vàng)")
        print("   - Click lần 2: hoàn thành đường thẳng (xanh lá)")
        print("\n2. Các phím chức năng:")
        print("   - 'Z': Hoàn tác đường vừa vẽ")
        print("   - 'C': Xóa tất cả")
        print("   - 'S': Lưu cấu hình")
        print("   - 'L': Tải cấu hình")
        print("   - 'Q': Thoát")
        
        while True:
            frame_display = self.draw_interface(self.frame)
            cv2.imshow(self.window_name, frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                if self.lines:
                    self.lines.pop()
                    print("Đã hoàn tác đường vừa vẽ")
                self.start_point = None
                self.preview_line = None
            elif key == ord('c'):
                self.lines = []
                self.start_point = None
                self.preview_line = None
                print("Đã xóa tất cả")
            elif key == ord('s'):
                self.save_config('grid_config.json')
            elif key == ord('l'):
                self.load_config('grid_config.json')
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = r"E:\exam_fraud_detection\videos\front_direction.mp4"
    calibrator = RoomGridCalibrator()
    calibrator.calibrate(video_path)

if __name__ == "__main__":
    main()
