from ultralytics import YOLO
import cv2
import numpy as np
import json
import time
import os
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

@dataclass(frozen=True)  # frozen=True làm cho class hashable
class Landmark:
    """Điểm mốc trong phòng thi"""
    id: int             # ID của điểm mốc 
    position: Tuple[int, int]  # Tọa độ (x, y) của điểm mốc
    label: str          # Nhãn (số thứ tự ghế)

    def __hash__(self) -> int:
        return hash((self.id, self.position, self.label))

@dataclass
class DetectedPerson:
    """Thông tin về người được phát hiện"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float                 # Độ tin cậy
    track_id: Optional[int] = None    # ID tracking từ ByteTrack
    nearest_landmark: Optional[Landmark] = None  # Điểm mốc gần nhất
    distance_to_landmark: float = float('inf')   # Khoảng cách đến điểm mốc gần nhất
    stable_position: bool = False     # True nếu vị trí đã ổn định qua nhiều frame

class ExamDetectorBack:
    """Hệ thống phát hiện người trong phòng thi - Camera sau"""
    
    def __init__(self, 
                 model_path: str = "yolov8s.pt",
                 confidence_threshold: float = 0.25,
                 input_size: int = 640,
                 iou_threshold: float = 0.3,
                 position_threshold: float = 150,
                 detect_interval: int = 5,        # Số frame giữa mỗi lần detect
                 stability_threshold: int = 4,     # Số lần detect liên tiếp để xác nhận vị trí
                 fail_threshold: int = 3):         # Số lần detect thất bại liên tiếp để bỏ qua
        """Khởi tạo detector với các tham số"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.iou_threshold = iou_threshold
        self.position_threshold = position_threshold
        self.detect_interval = detect_interval
        self.stability_threshold = stability_threshold
        self.fail_threshold = fail_threshold
        
        # Khởi tạo danh sách các điểm mốc
        self.landmarks: List[Landmark] = []
        
        # Lưu trữ các vị trí đã được gán
        self.assigned_positions: Dict[int, str] = {}  # track_id -> landmark label
        
        # Biến đếm frame và theo dõi detect
        self.frame_count = 0                          # Đếm tổng số frame
        self.detect_count = 0                         # Đếm số frame thực sự detect
        self.total_processing_time = 0                # Tổng thời gian xử lý thực tế
        self.previous_detections = []                 # Lưu kết quả detect gần nhất
        
        # Theo dõi trạng thái detect
        self.detection_stability = {}                 # track_id -> số lần detect liên tiếp thành công
        self.detection_failures = {}                  # track_id -> số lần detect liên tiếp thất bại
        self.stable_tracks = set()                   # Tập track_id đã ổn định
        
        # Khởi tạo tracking history
        self.track_history = {}                      # track_id -> list of positions
        self.track_landmarks = {}                    # track_id -> current landmark
        self.position_stability = {}                 # track_id -> số frame ổn định
        self.history_size = 30                      # Giữ lại 30 vị trí gần nhất
        
    def add_landmark(self, position: Tuple[int, int], label: str) -> None:
        """Thêm một điểm mốc mới vào hệ thống"""
        landmark_id = len(self.landmarks)
        new_landmark = Landmark(id=landmark_id, position=position, label=label)
        self.landmarks.append(new_landmark)
        
    def update_tracking_history(self, current_track_positions: Dict[int, Landmark]):
        """Cập nhật lịch sử tracking và kiểm tra độ ổn định của vị trí
        
        Args:
            current_track_positions: Dict[track_id, Landmark] chứa vị trí hiện tại của mỗi track
        """
        # Cập nhật số frame ổn định cho mỗi track
        for track_id, current_landmark in current_track_positions.items():
            if track_id in self.track_landmarks:
                prev_landmark = self.track_landmarks[track_id]
                if prev_landmark == current_landmark:
                    # Tăng số frame ổn định nếu vị trí không thay đổi
                    self.position_stability[track_id] = min(
                        self.position_stability.get(track_id, 0) + 1,
                        self.stability_threshold + 5  # Giới hạn trên
                    )
                else:
                    # Reset nếu vị trí thay đổi
                    self.position_stability[track_id] = 1
                    self.track_landmarks[track_id] = current_landmark
            else:
                # Thêm track mới
                self.track_landmarks[track_id] = current_landmark
                self.position_stability[track_id] = 1
            
            # Cập nhật track history với tâm của bounding box
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            # Giới hạn độ dài lịch sử
            if len(self.track_history[track_id]) > self.history_size:
                self.track_history[track_id] = self.track_history[track_id][-self.history_size:]

    def update_detection_status(self, track_id: int, success: bool):
        """Cập nhật trạng thái detect của một track
        
        Args:
            track_id: ID của track cần cập nhật
            success: True nếu detect thành công, False nếu thất bại
        """
        if success:
            # Reset số lần thất bại
            self.detection_failures[track_id] = 0
            # Tăng số lần thành công
            self.detection_stability[track_id] = self.detection_stability.get(track_id, 0) + 1
            
            # Kiểm tra đạt ngưỡng ổn định
            if self.detection_stability[track_id] >= self.stability_threshold:
                self.stable_tracks.add(track_id)
        else:
            # Reset số lần thành công
            self.detection_stability[track_id] = 0
            # Tăng số lần thất bại
            self.detection_failures[track_id] = self.detection_failures.get(track_id, 0) + 1
            
            # Xóa khỏi tracks ổn định nếu thất bại nhiều
            if self.detection_failures[track_id] >= self.fail_threshold:
                self.stable_tracks.discard(track_id)
                
    def cleanup_tracking_data(self, current_tracks: Set[int]):
        """Dọn dẹp dữ liệu tracking cũ
        
        Args:
            current_tracks: Set of track IDs that are present in the current frame
        """
        # Xóa thông tin của các track_id đã biến mất
        for track_id in list(self.track_history.keys()):
            if track_id not in current_tracks:
                del self.track_history[track_id]
                if track_id in self.position_stability:
                    del self.position_stability[track_id]
                if track_id in self.track_landmarks:
                    del self.track_landmarks[track_id]
                if track_id in self.assigned_positions:
                    del self.assigned_positions[track_id]
                
        # Giới hạn độ dài lịch sử cho mỗi track
        for track_id in self.track_history:
            if len(self.track_history[track_id]) > self.history_size:
                self.track_history[track_id] = self.track_history[track_id][-self.history_size:]
                    
    def calculate_distance(self, bbox: Tuple[int, int, int, int], 
                         landmark: Landmark) -> float:
        """Tính khoảng cách từ người được phát hiện đến điểm mốc"""
        # Tính tâm của bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Tính khoảng cách Euclidean
        dx = center_x - landmark.position[0]
        dy = center_y - landmark.position[1]
        return np.sqrt(dx*dx + dy*dy)
        
    def detect_persons(self, frame: np.ndarray) -> List[DetectedPerson]:
        """Phát hiện người trong frame và gán vị trí"""
        self.frame_count += 1
        
        # Kiểm tra xem có cần detect frame này không
        should_detect = (
            self.frame_count % self.detect_interval == 0 or  # Đến chu kỳ detect
            not self.previous_detections                      # Chưa có kết quả cache nào
        )
        
        # Nếu không cần detect, trả về kết quả cache
        if not should_detect:
            return self.previous_detections
        
        # Thực hiện detection thực sự
        self.detect_count += 1
        self.assigned_positions.clear()
        
        # Thực hiện detection và tracking với YOLO
        start_time = time.time()
        try:
            results = self.model.track(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                classes=[0],  # Chỉ phát hiện người
                persist=True,  # Duy trì track IDs
                verbose=False,
                tracker="bytetrack.yaml"  # Sử dụng ByteTrack với cấu hình mặc định
            )[0]
            
            # Cập nhật thống kê thời gian xử lý thực tế
            self.total_processing_time += time.time() - start_time
            
        except Exception as e:
            print(f"Lỗi khi thực hiện detection: {e}")
            return self.previous_detections  # Trả về cache nếu có lỗi

        # Nếu không có kết quả hoặc không có boxes
        if not hasattr(results, 'boxes') or results.boxes is None:
            self.previous_detections = []
            return []
            
        if not hasattr(results.boxes, 'id') or results.boxes.id is None:
            self.previous_detections = []
            return []

        try:
            # Chuyển đổi sang numpy arrays
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()
        except Exception as e:
            print(f"Lỗi khi xử lý kết quả: {e}")
            return self.previous_detections

        if len(boxes) == 0:
            self.previous_detections = []
            return []

        detected_people = []
        
        # Xử lý từng detection
        for i in range(len(boxes)):
            # Lấy thông tin cho mỗi người
            box = boxes[i]
            confidence = confidences[i]
            track_id = int(track_ids[i])
            
            # Tạo đối tượng DetectedPerson
            person = DetectedPerson(
                bbox=tuple(map(int, box)),
                confidence=float(confidence),
                track_id=track_id
            )
            
            # Tìm điểm mốc gần nhất
            nearest_landmark, min_distance = self.find_nearest_landmark(box)
            
            # Gán vị trí nếu đủ gần
            if nearest_landmark and min_distance <= self.position_threshold:
                person.nearest_landmark = nearest_landmark
                person.distance_to_landmark = min_distance
                self.assigned_positions[track_id] = nearest_landmark.label
                
                # Cập nhật trạng thái detection
                self.update_detection_status(track_id, True)
            else:
                self.update_detection_status(track_id, False)
                
            # Kiểm tra xem track đã ổn định chưa
            person.stable_position = track_id in self.stable_tracks
            
            detected_people.append(person)
            
        # Cập nhật tracking history và dọn dẹp dữ liệu cũ
        current_track_positions = {p.track_id: p.nearest_landmark for p in detected_people if p.nearest_landmark}
        self.update_tracking_history(current_track_positions)
        self.cleanup_tracking_data(set(track_ids))
        
        # Lưu kết quả vào cache để dùng cho các frame không detect
        self.previous_detections = detected_people
        return detected_people
        
    def draw_results(self, frame: np.ndarray, detected_people: List[DetectedPerson]) -> np.ndarray:
        """Vẽ kết quả detection lên frame"""
        frame_with_results = frame.copy()
        
        # Vẽ các điểm mốc
        for landmark in self.landmarks:
            cv2.circle(frame_with_results, landmark.position, 5, (0, 255, 0), -1)
            cv2.putText(frame_with_results, landmark.label, 
                      (landmark.position[0] + 10, landmark.position[1]), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ người được phát hiện và thông tin vị trí
        for person in detected_people:
            # Vẽ bounding box
            x1, y1, x2, y2 = person.bbox
            
            # Màu sắc dựa vào trạng thái
            if person.stable_position:
                color = (0, 255, 0)  # Xanh lá cho vị trí ổn định
            elif person.nearest_landmark:
                color = (0, 165, 255)  # Cam cho vị trí tạm thời
            else:
                color = (0, 0, 255)  # Đỏ cho chưa có vị trí
                
            cv2.rectangle(frame_with_results, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ thông tin
            if person.nearest_landmark:
                # Vẽ đường nối từ tâm người đến điểm mốc
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                landmark_pos = person.nearest_landmark.position
                cv2.line(frame_with_results, (center_x, center_y), 
                        landmark_pos, color, 2)
                
                # Vẽ text thông tin
                text = f"ID:{person.track_id} - Pos:{person.nearest_landmark.label}"
                if person.stable_position:
                    text += " (Stable)"
                cv2.putText(frame_with_results, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Hiển thị track ID nếu chưa có vị trí
                text = f"ID:{person.track_id}"
                cv2.putText(frame_with_results, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        return frame_with_results

    def find_nearest_landmark(self, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[Landmark], float]:
        """Tìm điểm mốc gần nhất với người được phát hiện"""
        min_distance = float('inf')
        nearest_landmark = None
        
        # Tính tâm của bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Tìm điểm mốc gần nhất
        for landmark in self.landmarks:
            dx = center_x - landmark.position[0]
            dy = center_y - landmark.position[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                nearest_landmark = landmark
                
        return nearest_landmark, min_distance

    def process_video(self, video_path: str, output_path: str = None,
                   display: bool = True) -> Dict:
        """Xử lý video và phát hiện người
        
        Args:
            video_path: Đường dẫn đến video đầu vào
            output_path: Đường dẫn để lưu video kết quả (optional)
            display: Có hiển thị video trong quá trình xử lý không
            
        Returns:
            Dict chứa thông tin kết quả xử lý
        """
        print("Bắt đầu xử lý video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Không thể mở video")
        
        # Lấy thông tin video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tạo VideoWriter nếu cần lưu output
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (frame_width, frame_height))
        
        # Khởi tạo biến thống kê
        frame_count = 0
        results = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "persons_detected": 0,
            "processing_time": 0.0
        }
        
        # Reset các biến thống kê của detector trước khi xử lý
        initial_detect_count = self.detect_count
        initial_processing_time = self.total_processing_time
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Phát hiện người và vị trí
                detected_people = self.detect_persons(frame)
                
                # Vẽ kết quả
                frame = self.draw_results(frame, detected_people)
                
                # Hiển thị frame nếu được yêu cầu
                if display:
                    # Resize frame để hiển thị với kích thước phù hợp
                    h, w = frame.shape[:2]
                    target_height = min(800, int(1080 * 0.8))  # 800px hoặc 80% màn hình
                    scale = target_height / h
                    display_w = int(w * scale)
                    display_h = int(h * scale)
                    
                    # Tạo cửa sổ có thể điều chỉnh kích thước nếu chưa có
                    window_name = 'Exam Detection (Back Camera)'
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    
                    # Resize frame và hiển thị
                    display_frame = cv2.resize(frame, (display_w, display_h), 
                                            interpolation=cv2.INTER_AREA)
                    cv2.imshow(window_name, display_frame)
                    
                    # Điều chỉnh kích thước cửa sổ (chỉ cần làm một lần)
                    if frame_count == 0:
                        cv2.resizeWindow(window_name, display_w, display_h)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Lưu frame vào output video
                if out is not None:
                    out.write(frame)
                
                # Cập nhật thống kê
                end_time = time.time()
                frame_processing_time = end_time - start_time
                frame_count += 1
                results["processed_frames"] += 1
                results["persons_detected"] += len(detected_people)
                results["processing_time"] += frame_processing_time
                
        finally:
            # Dọn dẹp
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
        
        # Cập nhật thống kê cuối cùng với thông tin thực tế từ detector
        actual_detections = self.detect_count - initial_detect_count
        actual_processing_time = self.total_processing_time - initial_processing_time
        
        results["actual_detections"] = actual_detections
        results["actual_processing_time"] = actual_processing_time
        results["cache_efficiency"] = (results["processed_frames"] - actual_detections) / results["processed_frames"] * 100 if results["processed_frames"] > 0 else 0
        
        # Tính thống kê cuối cùng
        if results["processed_frames"] > 0:
            results["avg_processing_time"] = results["processing_time"] / results["processed_frames"]
            results["avg_persons_per_frame"] = results["persons_detected"] / results["processed_frames"]
            results["avg_detection_time"] = actual_processing_time / actual_detections if actual_detections > 0 else 0
        else:
            results["avg_processing_time"] = 0
            results["avg_persons_per_frame"] = 0
            results["avg_detection_time"] = 0
            
        return results

def main():
    import os
    
    # Lấy đường dẫn tuyệt đối của thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "videos", "back_direction.mp4")
    config_path = os.path.join(current_dir, "grid_config_back.json")
    output_path = os.path.join(current_dir, "output", "back_direction_detected.mp4")
    
    print(f"Thư mục hiện tại: {current_dir}")
    print(f"Video đầu vào: {video_path}")
    print(f"File cấu hình: {config_path}")
    print(f"Video đầu ra: {output_path}")
    
    # Kiểm tra file video và thư mục
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại {video_path}")
        print("Vui lòng đặt video vào thư mục videos/ với tên back_direction.mp4")
        return
        
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Khởi tạo detector
    detector = ExamDetectorBack(
        model_path="yolov8s.pt",
        confidence_threshold=0.25,
        input_size=640,
        iou_threshold=0.40,
        position_threshold=150,
        detect_interval=5,
        stability_threshold=4,
        fail_threshold=3
    )
    
    # Load các điểm mốc từ file cấu hình
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            landmarks = json.load(f)
            
        # Thêm các điểm mốc vào detector
        for landmark in landmarks:
            detector.add_landmark(
                position=tuple(landmark["position"]),
                label=landmark["label"]
            )
            
        print(f"\nĐã nạp {len(landmarks)} điểm mốc từ file cấu hình")
        
    except FileNotFoundError:
        print(f"\nLỗi: Không tìm thấy file cấu hình tại {config_path}")
        print("Vui lòng chạy room_grid_calibrator_back.py trước để tạo file cấu hình")
        return
    except Exception as e:
        print(f"\nLỗi khi đọc file cấu hình: {e}")
        return
    
    try:
        results = detector.process_video(
            video_path=video_path,
            output_path=output_path,
            display=True
        )
        
        print("\nKết quả xử lý video:")
        print(f"- Tổng số frame: {results['total_frames']}")
        print(f"- Số frame đã xử lý: {results['processed_frames']}")
        print(f"- Số frame thực sự detect: {results['actual_detections']}")
        print(f"- Hiệu quả cache: {results['cache_efficiency']:.1f}%")
        print(f"- Thời gian xử lý TB: {results['processing_time']/results['processed_frames']:.3f}s/frame")
        print(f"- Thời gian detect TB: {results['avg_detection_time']:.3f}s/detection")
        print(f"- Số người TB/frame: {results['persons_detected']/results['total_frames']:.1f}")
        
    except Exception as e:
        print(f"Lỗi khi xử lý video: {e}")

if __name__ == "__main__":
    main()
