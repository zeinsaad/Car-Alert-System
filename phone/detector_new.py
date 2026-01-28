import cv2
from ultralytics import YOLO


class PhoneDetector:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.device = device

        self.frame_count = 0
        self.run_every = 9
        self.last_detected = False

        # store last box for skipped frames (optional but useful)
        self.last_box = None


    def process(self, frame):
        self.frame_count += 1

        # -------- Frame skipping --------
        if self.frame_count % self.run_every != 0:
            # draw last box if exists
            if self.last_box is not None:
                x1, y1, x2, y2 = self.last_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            return self.last_detected

        # -------- YOLO inference --------
        results = self.model.predict(
            source=frame,
            conf=0.6,
            imgsz=640,
            device=self.device,
            verbose=False # Disable console logs during inference
        )

        r = results[0]

        if r.boxes:
            # take FIRST detected phone
            box = r.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            self.last_box = (x1, y1, x2, y2)
            self.last_detected = True
            return True

        # no detection
        self.last_box = None
        self.last_detected = False
        return False
