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
COLOR_PERSON = (0, 165, 255)   # Cam cho người đi bộ
COLOR_VEHICLE = (0, 220, 100)  # Xanh lá cho phương tiện

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


def detect_vehicles_and_people(image, conf=0.35, weights_path="yolov8n.pt"):
    """
    Chạy YOLOv8n trên 1 ảnh hoặc 1 khung hình video.
    Trả về: (ảnh đã vẽ hộp bao, {"Số xe": int, "Số người": int})
    """
    if image is None:
        return None, {"Số xe": 0, "Số người": 0}

    bgr = _ensure_bgr(image).copy()
    model = get_yolo_model(weights_path)

    # classes=list(TARGET_CLASSES.keys()) giúp YOLO chỉ trả về 5 lớp mục tiêu
    results = model.predict(
        source=bgr,
        conf=conf,
        classes=list(TARGET_CLASSES.keys()),
        verbose=False
    )

    vehicle_count = 0
    person_count = 0

    if results and len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label_name = TARGET_CLASSES.get(cls_id, str(cls_id))
            if cls_id in PERSON_CLASSES:
                person_count += 1
                color = COLOR_PERSON
            else:
                vehicle_count += 1
                color = COLOR_VEHICLE

            # Vẽ hộp bao
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

            # Vẽ nền chữ + tên loại
            text = f"{label_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1 - 6, th + 6)
            cv2.rectangle(bgr, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(
                bgr, text, (x1 + 3, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )

    info = {
        "Số xe": vehicle_count,
        "Số người": person_count,
    }
    return bgr, info