import cv2                         
import torch                        
import numpy as np                  
import mediapipe as mp             
from torchvision import transforms  
from modules.Eye_model import EyeStateCNN  

# ---------------- CONFIG ----------------
IMG_SIZE = 64        # Final eye image size expected by the CNN (64x64 pixels)
THRESHOLD = 0.3      # Probability threshold: if predicted probability < 0.3 → eye is considered CLOSED              
RUN_EVERY = 3        # Run eye detection only once every 3 frames 
CLOSED_FRAMES = 6    # Number of consecutive "closed-eye" detections
# --------------------------------------


class EyeDetector:
    def __init__(self, model_path, device):

        # Move model to device and set it to evaluation mode (no dropout, no training behavior)
        self.device = device
        self.model = EyeStateCNN().to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        # Initialize MediaPipe Face Mesh for facial landmark detection
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,              # Only track one face (the driver)
            refine_landmarks=False,       # Disable iris landmarks → faster
            min_detection_confidence=0.5, # Minimum confidence to detect a face
            min_tracking_confidence=0.5   # Minimum confidence to keep tracking the face
        )

        # Landmark indices corresponding to the left and right eyes
        # These indices are FIXED and come from MediaPipe Face Mesh documentation
        # Each number refers to a specific anatomical point around the eye
        self.EYES = [
            [33, 160, 158, 133, 153, 144],    # Left eye landmarks
            [362, 385, 387, 263, 373, 380]    # Right eye landmarks
        ]

        # Image preprocessing pipeline applied to each eye crop
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                 # Convert NumPy array → PIL image
            transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize eye to 64x64
            transforms.ToTensor(),                   # Convert to PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))     # Normalize grayscale values to [-1, 1]
        ])

        self.frame_id = 0          # Counts processed frames (used for frame skipping)
        self.closed_frames = 0     # Counts consecutive frames where eyes are detected closed


    def _eye_tensor(self, frame, lm, idx):
        """
        Extracts one eye region from the frame using landmarks,
        preprocesses it, and returns a tensor ready for CNN inference.
        """

        h, w = frame.shape[:2]  # Get frame height and width in pixels

        # Convert normalized landmark coordinates (range [0,1])
        # into pixel coordinates in the image
        xs = [int(lm[i].x * w) for i in idx]
        ys = [int(lm[i].y * h) for i in idx]

        # Crop the eye region with a small padding (5 pixels)
        # Padding helps include eyelids and avoid tight crops
        eye = frame[
            max(min(ys) - 5, 0) : min(max(ys) + 5, h),
            max(min(xs) - 5, 0) : min(max(xs) + 5, w)
        ]

        # If cropping failed (empty array), return None
        if eye.size == 0:
            return None

        # Convert the eye image to grayscale (CNN was trained on grayscale eyes)
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

        # Improve contrast to handle different lighting conditions
        gray = cv2.equalizeHist(gray)

        #Convert eye crop to square to preserve aspect ratio
        s = max(gray.shape)
        pad = np.zeros((s, s), np.uint8)

        # Center the eye image inside the square canvas
        y, x = (s - gray.shape[0]) // 2, (s - gray.shape[1]) // 2
        pad[y:y+gray.shape[0], x:x+gray.shape[1]] = gray

        # Apply preprocessing transforms:
        return self.transform(pad).unsqueeze(0).to(self.device)


    def process(self, frame):
        """
        Main function called for each video frame.
        Returns:
            True  → driver is drowsy (eyes closed long enough)
            False → driver is not drowsy
        """

        self.frame_id += 1  # Increment frame counter

        # Skip heavy eye processing on intermediate frames
        # We reuse the previous drowsiness state instead
        if self.frame_id % RUN_EVERY:
            return self.closed_frames >= CLOSED_FRAMES

        # Convert frame to RGB because MediaPipe expects RGB images
        res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # If no face is detected:
        # - reset closed-eye counter
        # - report "not drowsy"
        if not res.multi_face_landmarks:
            self.closed_frames = 0
            return False

        # Extract facial landmarks for the detected face
        lm = res.multi_face_landmarks[0].landmark

        probs = []  # Will store eye-closure probabilities from both eyes

        # Disable gradients for faster inference (inference-only mode)
        with torch.no_grad():
            for eye in self.EYES:
                # Extract and preprocess the eye region
                t = self._eye_tensor(frame, lm, eye)

                # If extraction succeeded, run the CNN
                if t is not None:
                    # CNN outputs a logit → apply sigmoid to get probability [0,1]
                    probs.append(torch.sigmoid(self.model(t)).item())

        # If no valid eye predictions were obtained
        if not probs:
            self.closed_frames = 0
            return False

        # Average eye-closure probability across both eyes
        avg_prob = sum(probs) / len(probs)
        #print(avg_prob)

        # If probability indicates closed eyes
        if avg_prob < THRESHOLD:
            self.closed_frames += 1   # Increase consecutive closed-eye counter
        else:
            self.closed_frames = 0    # Reset counter if eyes are open

        # Driver is considered drowsy if eyes stayed closed
        # for enough consecutive processed frames
        return self.closed_frames >= CLOSED_FRAMES
