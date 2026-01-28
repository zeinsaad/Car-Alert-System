import cv2
import torch
import time

from eye_closure.detector import EyeDetector
from phone_detector.detector_new import PhoneDetector
from head_pose.detector_final import HeadPoseDetector
from modules.Eye_model import EyeStateCNN

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EYE_MODEL_PATH = "eye_closure/eye_state_cnn.pth"
PHONE_MODEL_PATH = "phone/best (1).pt"
HEAD_MODEL_PATH = "head_pose/biwi_resnet18_yaw_pitch_deg (1).pth"

CAMERA_ID = 0
# ========================================


# -------- Load Eye detector --------
eye_detector = EyeDetector(EYE_MODEL_PATH, DEVICE)

# -------- Load Phone detector --------
phone_detector = PhoneDetector(PHONE_MODEL_PATH, DEVICE)

# -------- Load Head Pose detector --------
# Auto-calibrated at startup
head_detector = HeadPoseDetector(HEAD_MODEL_PATH, DEVICE)

print("✅ All detectors loaded")
print("➡ Press 'q' = quit")

# ================= WEBCAM =================
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -------- FPS tracking --------
prev_time = time.monotonic()
start_time = prev_time
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # -------- Run detectors (ONCE per frame) --------
    t0 = time.monotonic()
    eyes_closed = eye_detector.process(frame)
    t1 = time.monotonic()

    phone_used = phone_detector.process(frame)
    t2 = time.monotonic()

    head_state = head_detector.process(frame)
    t3 = time.monotonic()

    # -------- Per-module timing (debug) --------
    print(
        f"Eye: {(t1 - t0) * 1000:.1f} ms | "
        f"Phone: {(t2 - t1) * 1000:.1f} ms | "
        f"Head: {(t3 - t2) * 1000:.1f} ms"
    )

    # -------- Priority logic --------
    if head_state == "AWAY":
        alert = "DRIVER DISTRACTED"
        color = (0, 0, 255)
    elif phone_used:
        alert = "PHONE USAGE ALERT"
        color = (0, 0, 255)
    elif eyes_closed:
        alert = "DROWSINESS ALERT"
        color = (0, 0, 255)
    else:
        alert = "SAFE"
        color = (0, 255, 0)

    # -------- FPS calculation --------
    now = time.monotonic()
    dt = now - prev_time
    fps = 1.0 / dt if dt > 1e-6 else 0.0
    prev_time = now

    prev_time = now

    avg_fps = frame_count / (now - start_time)

    # -------- Overlay --------
    cv2.putText(
        frame, f"HEAD: {head_state}", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
    )

    cv2.putText(
        frame, alert, (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3
    )
  


    cv2.imshow("Car Alert System (Debug)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
