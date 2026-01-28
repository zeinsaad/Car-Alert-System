import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
import mediapipe as mp
from collections import deque
from modules.HeadPose_model import HeadPoseNet


class HeadPoseDetector:
    def __init__(self, model_path, device, calib_frames=40):
        self.device = device
        self.calib_frames = calib_frames

        # -------- Load trained model --------
        self.model = HeadPoseNet().to(device)

        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        self.model.load_state_dict(state)
        self.model.eval()

        # -------- MediaPipe face detector --------
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

        # -------- Face preprocessing --------
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # -------- Auto-calibration (METHOD 1) --------
        # collect raw yaw/pitch during calibration
        self.calib_yaw = []
        self.calib_pitch = []

        self.yaw_center = None
        self.pitch_center = None

        # -------- Temporal smoothing --------
        self.yaw_hist = deque(maxlen=7)
        self.pitch_hist = deque(maxlen=7)

        # -------- Auto thresholds config --------
        self.K_STD = 3.0           # threshold = K_STD * std
        self.MIN_YAW_THRESH = 8.0  # safety floor
        self.MIN_PITCH_THRESH = 8.0

        # safety floor
        self.MAX_YAW_THRESH = 25.0
        self.MAX_PITCH_THRESH = 20.0

        # -------- Thresholds (DEGREES) --------
        # will be computed after calibration
        self.YAW_THRESH = None
        self.PITCH_THRESH = None

        # -------- Debug values --------
        self.last_yaw = 0.0
        self.last_pitch = 0.0


    # ================= MAIN FUNCTION =================
    def process(self, frame):
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_detector.process(rgb)

        if not result.detections:
            return "NO FACE"

        box = result.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w, int((box.xmin + box.width) * w))
        y2 = min(h, int((box.ymin + box.height) * h))

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return "NO FACE"

        # -------- CNN input --------
        x = self.transform(face).unsqueeze(0).to(self.device)

        # -------- Run CNN --------
        with torch.no_grad():
            yaw_pred, pitch_pred = self.model(x)

        yaw = yaw_pred.item()
        pitch = pitch_pred.item()

        # -------- AUTO-CALIBRATION (center + auto-thresholds) --------
        if self.yaw_center is None:
            self.calib_yaw.append(yaw)
            self.calib_pitch.append(pitch)

            if len(self.calib_yaw) >= self.calib_frames:
                # centers
                self.yaw_center = float(np.mean(self.calib_yaw))
                self.pitch_center = float(np.mean(self.calib_pitch))

                # std of normal forward-looking micro-movements
                yaw_std = float(np.std(self.calib_yaw))
                pitch_std = float(np.std(self.calib_pitch))

                # thresholds from std 
                self.YAW_THRESH = float(
                    np.clip(self.K_STD * yaw_std,
                            self.MIN_YAW_THRESH,
                            self.MAX_YAW_THRESH)
                )

                self.PITCH_THRESH = float(
                    np.clip(self.K_STD * pitch_std,
                            self.MIN_PITCH_THRESH,
                            self.MAX_PITCH_THRESH)
                )


            return "CALIBRATING , PLEASE LOOK STRAIGHT AHEAD AND WAIT ..."

        # -------- Relative angles + smoothing --------
        self.yaw_hist.append(yaw - self.yaw_center)
        self.pitch_hist.append(pitch - self.pitch_center)

        self.last_yaw = float(np.mean(self.yaw_hist))
        self.last_pitch = float(np.mean(self.pitch_hist))

        # -------- Final decision --------
        if abs(self.last_yaw) > self.YAW_THRESH or abs(self.last_pitch) > self.PITCH_THRESH:
            return "AWAY"

        return "CENTER"
