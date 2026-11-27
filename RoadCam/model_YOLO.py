# M9_yolo.py
import cv2
import numpy as np
from ultralytics import YOLO

# Các lớp COCO cần phát hiện theo yêu cầu đề bài
PERSON_CLASSES = {0: "person"}
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
TARGET_CLASSES = {**PERSON_CLASSES, **VEHICLE_CLASSES}

# Màu BGR cho hộp bao
COLOR_PERSON = (0, 165, 255)    # Cam cho người đi bộ
COLOR_VEHICLE = (0, 220, 100)   # Xanh lá cho phương tiện an toàn
COLOR_DANGER = (0, 0, 255)      # Đỏ cảnh báo khi xe trong làn quá gần

_model = None


def get_yolo_model(weights_path="yolov8n.pt"):
    """Nạp model YOLO một lần duy nhất (singleton) để tối ưu tốc độ cho video."""
    global _model
    if _model is None:
        _model = YOLO(weights_path)
    return _model


def _ensure_bgr(image):
    """YOLO yêu cầu ảnh 3 kênh BGR."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def sort_roi_quad(points):
    """Chuẩn hóa 4 đỉnh ROI về đúng thứ tự [TL, TR, BR, BL] tránh xoắn ma trận."""
    pts = sorted(points, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


class RoadDistanceEstimator:
    """Quy đổi tọa độ đáy box YOLO (xc, y2) sang khoảng cách mét từ ROI hình thang."""

    def __init__(self):
        self._cached_key = None
        self._H = None
        self.camera_offset_m = 3.0

    def update(self, roi_points, road_width_m=3.5, road_length_m=20.0, camera_offset_m=3.0):
        if roi_points is None or len(roi_points) != 4:
            self._cached_key = None
            self._H = None
            return False

        self.camera_offset_m = float(camera_offset_m)
        key = (
            tuple((int(x), int(y)) for x, y in roi_points),
            float(road_width_m),
            float(road_length_m),
        )
        if key == self._cached_key and self._H is not None:
            return True

        src_pts = sort_roi_quad(roi_points)  # [TL, TR, BR, BL]
        dst_pts = np.array([
            [0.0, float(road_length_m)],
            [float(road_width_m), float(road_length_m)],
            [float(road_width_m), 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        self._H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self._cached_key = key
        return True

    def pixel_to_distance(self, x1, y1, x2, y2):
        if self._H is None:
            return None
        xc = 0.5 * (x1 + x2)
        pt_img = np.array([[[xc, float(y2)]]], dtype=np.float32)
        pt_bev = cv2.perspectiveTransform(pt_img, self._H)[0, 0]
        return max(0.0, self.camera_offset_m + float(pt_bev[1]))


_distance_estimator = RoadDistanceEstimator()


def detect_vehicles_and_people(
    image,
    conf=0.35,
    weights_path="yolov8n.pt",
    roi_points=None,
    road_width_m=3.5,
    road_length_m=20.0,
    camera_offset_m=3.0,
    warn_dist_m=8.0,
):
    """
    Chạy YOLOv8n trên 1 ảnh hoặc 1 khung hình video, kèm đo khoảng cách nếu có roi_points.
    Trả về: (ảnh đã vẽ hộp bao, dict thống kê cho Bảng kết quả)
    """
    if image is None:
        return None, {"Số xe": 0, "Số người": 0}

    bgr = _ensure_bgr(image).copy()
    model = get_yolo_model(weights_path)

    has_roi = _distance_estimator.update(
        roi_points,
        road_width_m=road_width_m,
        road_length_m=road_length_m,
        camera_offset_m=camera_offset_m,
    )
    poly_np = np.array(roi_points, dtype=np.int32) if has_roi else None

    if has_roi:
        cv2.polylines(bgr, [poly_np.reshape(-1, 1, 2)], True, (255, 200, 0), 1)

    results = model.predict(
        source=bgr,
        conf=conf,
        classes=list(TARGET_CLASSES.keys()),
        verbose=False,
    )

    vehicle_count = 0
    person_count = 0
    min_dist_m = None

    if results and len(results) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            xc = (x1 + x2) // 2

            label_name = TARGET_CLASSES.get(cls_id, str(cls_id))
            text = f"{label_name} {score:.2f}"

            if cls_id in PERSON_CLASSES:
                person_count += 1
                color = COLOR_PERSON
            else:
                vehicle_count += 1
                color = COLOR_VEHICLE

                if has_roi:
                    in_lane = cv2.pointPolygonTest(poly_np, (float(xc), float(y2)), False) >= 0
                    if in_lane:
                        dist_m = _distance_estimator.pixel_to_distance(x1, y1, x2, y2)
                        if dist_m is not None:
                            text = f"{label_name} {score:.2f} | {dist_m:.1f}m"
                            if min_dist_m is None or dist_m < min_dist_m:
                                min_dist_m = dist_m
                            if dist_m < warn_dist_m:
                                color = COLOR_DANGER
                            cv2.circle(bgr, (xc, y2), 4, (255, 255, 0), -1)

            # Vẽ hộp bao
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

            # Vẽ nền chữ + tên loại
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1 - 6, th + 6)
            cv2.rectangle(bgr, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
            text_color = (255, 255, 255) if color == COLOR_DANGER else (0, 0, 0)
            cv2.putText(
                bgr, text, (x1 + 3, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA
            )

    info = {
        "Số xe": vehicle_count,
        "Số người": person_count,
    }
    if has_roi:
        info["Xe gần nhất"] = f"{min_dist_m:.1f} m" if min_dist_m is not None else "--"

    return bgr, info