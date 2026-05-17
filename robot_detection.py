"""
Robot Object Detection using YOLOv8 + OpenVINO
-----------------------------------------------
Detects objects using a YOLOv8 model optimized with OpenVINO and provides
navigation hints based on the detected object's position in the frame.

Requirements:
    pip install ultralytics openvino opencv-python

Usage:
    # Step 0: Set up virtual environment (run once)
    py -3.12 -m venv robot_env
    robot_env\Scripts\activate
    pip install ultralytics openvino opencv-python

    # Step 1: Export YOLOv8 to OpenVINO format (run once)
    python export_openvino.py

    # Step 2: Run detection
    python robot_detection.py --target "person"
"""

import argparse
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_OV_DIR   = "yolov8n_openvino_model"   # OpenVINO export directory (from export_openvino.py)
CAMERA_INDEX   = 0                          # 0 = default webcam
FRAME_SIZE     = 320                        # Must match FRAME_SIZE in export_openvino.py
CONF_THRESHOLD = 0.5                        # Minimum confidence to show a detection
SKIP_FRAMES    = 2                          # Run inference every N frames


# ── Navigation logic ──────────────────────────────────────────────────────────

def get_navigation_hint(box_center_x: float, frame_width: int) -> str:
    """
    Return a simple navigation command based on where the target is
    horizontally in the frame.

        LEFT 35% │ CENTER 30% │ RIGHT 35%
        TURN LEFT │ GO STRAIGHT │ TURN RIGHT
    """
    ratio = box_center_x / frame_width
    if ratio < 0.35:
        return "TURN LEFT"
    elif ratio > 0.65:
        return "TURN RIGHT"
    else:
        return "GO STRAIGHT"


# ── Threaded detector ─────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Runs YOLOv8-OpenVINO inference in a background thread so the
    main (camera) loop is never blocked by inference time.
    """

    def __init__(self, target_label: str):
        print(f"[INFO] Loading OpenVINO model from '{MODEL_OV_DIR}/' ...")
        self.model          = YOLO(f"{MODEL_OV_DIR}/")
        self.target_label   = target_label.lower()
        self.latest_frame   = None
        self.latest_results = None
        self._lock          = threading.Lock()
        self._running       = True

        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print(f"[INFO] Detector started  |  target: '{self.target_label}'")

    def update_frame(self, frame: np.ndarray):
        with self._lock:
            self.latest_frame = frame.copy()

    def get_results(self):
        with self._lock:
            return self.latest_results

    def stop(self):
        self._running = False

    def _inference_loop(self):
        while self._running:
            frame = None
            with self._lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            results = self.model(
                frame,
                imgsz=FRAME_SIZE,
                conf=CONF_THRESHOLD,
                verbose=False,
            )

            with self._lock:
                self.latest_results = results


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, results, target_label: str) -> tuple:
    """
    Draw bounding boxes on frame.
    Returns (annotated_frame, navigation_hint_or_None).
    """
    if results is None:
        return frame, None

    h, w = frame.shape[:2]
    nav_hint = None

    for result in results:
        for box in result.boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = result.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            is_target = (label == target_label)
            color     = (0, 255, 0) if is_target else (200, 200, 200)
            thickness = 2 if is_target else 1

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label
            text = f"{label} {conf:.0%}"
            cv2.putText(frame, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)

            # Center dot (target only)
            if is_target:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                nav_hint = get_navigation_hint(cx, w)

    return frame, nav_hint


def draw_navigation_overlay(frame: np.ndarray, hint: str | None, target_label: str):
    """Draw a semi-transparent status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if hint:
        color  = (0, 255, 100)
        status = f"TARGET: {target_label.upper()}  |  NAV: {hint}"
    else:
        color  = (100, 100, 255)
        status = f"TARGET: {target_label.upper()}  |  SEARCHING ..."

    cv2.putText(frame, status, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_detection(target_label: str):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    detector  = ObjectDetector(target_label)
    frame_idx = 0

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Send every Nth frame to the background detector
        if frame_idx % SKIP_FRAMES == 0:
            detector.update_frame(frame)

        frame_idx += 1

        # Retrieve latest results (may be from a previous frame — that's OK)
        results = detector.get_results()
        frame, nav_hint = draw_detections(frame, results, target_label)
        draw_navigation_overlay(frame, nav_hint, target_label)

        cv2.imshow("Robot Detection (OpenVINO)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + OpenVINO robot detector")
    parser.add_argument(
        "--target",
        type=str,
        default="person",
        help="Object label to navigate toward (default: 'person').",
    )
    args = parser.parse_args()

    run_detection(args.target)
