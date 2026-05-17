"""
Export YOLOv8 to OpenVINO format
---------------------------------
Converts yolov8n.pt to an OpenVINO IR model for faster inference
on Intel hardware (including Intel Iris Plus graphics).

Requirements:
    pip install ultralytics openvino

Usage:
    python export_openvino.py
"""

from ultralytics import YOLO


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PT   = "yolov8n.pt"             # Input: PyTorch weights (auto-downloaded)
MODEL_OV_DIR = "yolov8n_openvino_model"  # Output: OpenVINO export directory
FRAME_SIZE = 320                      # Must match FRAME_SIZE in robot_detection.py


# ── Export ────────────────────────────────────────────────────────────────────

def export_to_openvino():
    """Convert yolov8n.pt to OpenVINO IR format."""
    print(f"[INFO] Loading model '{MODEL_PT}' ...")
    model = YOLO(MODEL_PT)

    print(f"[INFO] Exporting to OpenVINO format (imgsz={FRAME_SIZE}) ...")
    model.export(format="openvino", imgsz=FRAME_SIZE)

    print(f"[INFO] Export complete → {MODEL_OV_DIR}/")
    print(f"[INFO] You can now run: python robot_detection.py --target 'person'")


if __name__ == "__main__":
    export_to_openvino()
