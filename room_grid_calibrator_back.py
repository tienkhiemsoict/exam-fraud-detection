import cv2
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

class RoomGridCalibratorBack:
    """Công cụ hiệu chỉnh vị trí chỗ ngồi trong phòng thi - Camera sau"""

    def __init__(self):
        # Khởi tạo các biến quản lý điểm mốc
        self.landmarks: List[Dict] = []  # Danh sách các điểm mốc đã đánh dấu
        self.position_counter = 1        # Bộ đếm vị trí từ 1 trở đi
        
        # Khởi tạo các biến frame và display
        self.window_name = 'Hiệu chỉnh vị trí (Camera Sau) - ESC để hoàn thành, R để xoá điểm cuối'
        self.original_frame = None      # Frame gốc từ video
        self.original_image = None      # Frame đã resize để hiển thị
        self.image = None               # Frame hiển thị với các điểm đánh dấu
        self.display_scale = 1.0        # Tỷ lệ scale giữa frame gốc và hiển thị
        
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
                # Chuyển từ tọa độ gốc sang tọa độ hiển thị
                original_x, original_y = landmark['position']
                display_x = int(original_x * self.display_scale)
                display_y = int(original_y * self.display_scale)
                
                cv2.circle(self.image, (display_x, display_y), 5, (0, 255, 0), -1)
                cv2.putText(self.image, landmark['label'], 
                          (display_x + 10, display_y), cv2.FONT_HERSHEY_SIMPLEX, 
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
            # Chuyển đổi tọa độ click từ frame hiển thị về frame gốc
            original_x = int(x / self.display_scale)
            original_y = int(y / self.display_scale)
            
            print(f"Click tại ({x}, {y}) trên frame hiển thị")
            print(f"Chuyển về tọa độ gốc: ({original_x}, {original_y})")
            
            # Thêm điểm mốc mới với tọa độ gốc
            label = self._get_next_position_label()
            self.landmarks.append({
                'position': (original_x, original_y),
                'label': label
            })
            
            # Vẽ điểm mốc trên ảnh đã resize
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

        # Tạo cửa sổ có thể điều chỉnh kích thước
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Lưu frame gốc
        self.original_frame = frame.copy()
        self.original_height, self.original_width = frame.shape[:2]
        print(f"Kích thước gốc: {self.original_width}x{self.original_height}")

        # Resize frame để dễ nhìn hơn, giữ nguyên tỷ lệ
        target_height = min(800, int(1080 * 0.8))  # assuming 1080p screen
        self.display_scale = target_height / self.original_height
        self.display_width = int(self.original_width * self.display_scale)
        self.display_height = target_height

        frame = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        print(f"Đã resize frame thành {self.display_width}x{self.display_height}, tỷ lệ scale = {self.display_scale:.3f}")

        # Lưu frame hiển thị
        self.original_image = frame.copy()
        self.image = frame.copy()
        
        print("Kiểm tra các thông số:")
        print(f"- Frame gốc: {self.original_frame.shape}")
        print(f"- Frame hiển thị: {frame.shape}")
        print(f"- Tỷ lệ scale: {self.display_scale}")
        
        # Thiết lập callback cho cửa sổ
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Thiết lập kích thước cửa sổ ban đầu
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
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
        # In thông tin debug trước khi lưu
        print("\nThông tin trước khi lưu landmarks:")
        print(f"Số lượng landmarks: {len(self.landmarks)}")
        for idx, landmark in enumerate(self.landmarks):
            print(f"Landmark {idx+1}: position={landmark['position']}, label={landmark['label']}")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.landmarks, f, ensure_ascii=False, indent=2)
            
        print(f"\nĐã lưu landmarks vào {output_path}")
            
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
    video_path = os.path.join(current_dir, "videos", "back_direction.mp4")
    config_path = os.path.join(current_dir, "grid_config_back.json")
    
    print(f"Thư mục hiện tại: {current_dir}")
    print(f"Đường dẫn video: {video_path}")
    
    # Kiểm tra file video có tồn tại
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại {video_path}")
        print("Vui lòng đặt video vào thư mục videos/ với tên back_direction.mp4")
        return
        
    # Tạo thư mục videos nếu chưa có
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
        print(f"Đã tạo thư mục {os.path.dirname(video_path)}")
    
    # Tạo calibrator
    calibrator = RoomGridCalibratorBack()
    
    # Thực hiện hiệu chỉnh
    try:
        print("Bắt đầu hiệu chỉnh vị trí phòng thi (Camera Sau)...")
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
