import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import os
import json
import datetime

IS_DISPLAY = True
IS_WRITE = False
CAMERA_ID = 1
OUTPUT_SIZE = 300
SHOW_NORM_VIEW = True

LABEL_MAP = {0: "Front", 1: "Left", 2: "Right", 3: "Phone"}
LABEL_COLORS = {
    "Front": (0, 220, 0),
    "Left": (0, 220, 220),
    "Right": (0, 140, 255),
    "Phone": (0, 0, 255),
}
MODEL_FILES = [
    "Models/Cubic_SVM.pkl",
    "Models/Neural_Network.pkl",
]

# selected face landmarks for the model
SELECTED_LANDMARKS = [
    0, 1, 2, 2, 4, 5, 10, 17, 33, 37, 39, 40, 54,
    61, 67, 84, 91, 94, 97, 98, 132, 133, 145,
    148, 150, 152, 153, 153, 158, 159, 160, 161,
    162, 163, 168, 172, 185, 195, 234, 251, 263,
    267, 269, 270, 288, 291, 314, 321, 323, 327,
    332, 356, 362, 365, 374, 377, 378, 380, 380,
    385, 386, 387, 388, 390, 409, 468, 469, 470,
    471, 472, 473, 474, 475, 476, 477,
]

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 4

CALIBRATION_FRAMES = 60
CALIBRATION_FILE = "calibration/calibration_ref_transformationM.npz"
CALIBRATION_LOG = "calibration/calibration_log_transformationM.json"

if not os.path.exists("calibration"):
    os.makedirs("calibration")

# smoothing
PROB_EMA_ALPHA = 0.55
HOLD_FRAMES = 2
_IRIS_LM = frozenset(range(468, 478))
_EYELID_LM = frozenset({145, 159, 160, 161, 374, 380, 385, 386, 387, 388})

# Eye aspect ratio (EAR)
_EAR_LEFT = (33, 160, 158, 133, 153, 145)
_EAR_RIGHT = (263, 387, 385, 362, 380, 374)
EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.15

# per-slot alpha for landmark smoothing (handles iris specially)
_ALPHA_BASE = 0.60
_ALPHA_EYELID = 0.40
_ALPHA_IRIS = 0.20
_PER_SLOT_ALPHA = np.array(
    [(_ALPHA_IRIS if lm in _IRIS_LM else
      _ALPHA_EYELID if lm in _EYELID_LM else
      _ALPHA_BASE)
     for lm in SELECTED_LANDMARKS
     for _ in (0, 1)],
    dtype=np.float32
)

# 3D model points for head pose (solvePnP)
_3D_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)
_MODEL_LM_IDS = [4, 152, 33, 263, 61, 291]

# MEDIAPIPE SETUP
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
)

#---------------------------------------------------------------------
def extract_raw(face_landmarks):
    # get (478, 2) raw x,y coords from mediapipe
    return np.array([(lm.x, lm.y) for lm in face_landmarks.landmark],
                    dtype=np.float32)


def extract_raw_3d(face_landmarks):
    # get (478, 3) with depth
    return np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
                    dtype=np.float32)


def compute_ear(lm):
    # eye aspect ratio - averaged over both eyes
    def ear_one(p1, p2, p3, p4, p5, p6):
        A = np.linalg.norm(lm[p2] - lm[p6])
        B = np.linalg.norm(lm[p3] - lm[p5])
        C = np.linalg.norm(lm[p1] - lm[p4])
        return (A + B) / (2.0 * C + 1e-9)
    return float((ear_one(*_EAR_LEFT) + ear_one(*_EAR_RIGHT)) * 0.5)


class LandmarkSmoother:
    def __init__(self):
        self._smooth = None
    
    def update(self, flat_146, ear):
        if self._smooth is None:
            self._smooth = flat_146.copy()
            return self._smooth
        
        alphas = _PER_SLOT_ALPHA.copy()
        
        if ear < EAR_CLOSED_THRESHOLD:
            # blink - freeze iris completely
            alphas = np.where(
                np.array([lm in _IRIS_LM
                           for lm in SELECTED_LANDMARKS
                           for _ in (0, 1)]),
                0.0, alphas
            ).astype(np.float32)
        elif ear < EAR_OPEN_THRESHOLD:
            # partially closed eye - reduce iris alpha
            t = (ear - EAR_CLOSED_THRESHOLD) / (EAR_OPEN_THRESHOLD - EAR_CLOSED_THRESHOLD)
            reduced = _ALPHA_IRIS * t
            alphas = np.where(
                np.array([lm in _IRIS_LM
                           for lm in SELECTED_LANDMARKS
                           for _ in (0, 1)]),
                reduced, alphas
            ).astype(np.float32)
        
        # EMA: smooth = alpha * new + (1-alpha) * prev
        self._smooth = alphas * flat_146 + (1.0 - alphas) * self._smooth
        return self._smooth
    
    def reset(self):
        self._smooth = None


class GazeNormalizer3D:
    def __init__(self):
        self.ref_3d = None
        self.R = None
        self.t = None
        self.has_ref = False
        self.has_bias = False
    
    def calibrate_reference(self, raw_frames_3d):
        # step 1: capture reference at training position
        stacked = np.stack(raw_frames_3d, axis=0)
        self.ref_3d = stacked.mean(axis=0).astype(np.float32)
        self.has_ref = True
        self.has_bias = False
        self.R = None
        self.t = None
        
        std = float(stacked.std(axis=0).mean())
        print(f"\nReference saved as {CALIBRATION_FILE}")
        return std
    
    def calibrate_position(self, raw_frames_3d):
        # step 2: capture at new camera position, compute rotation + translation
        if not self.has_ref:
            print("  ERROR: complete Step 1 first.")
            return False
        
        stacked = np.stack(raw_frames_3d, axis=0)
        cur_3d = stacked.mean(axis=0).astype(np.float32)
        
        # SVD-based rigid alignment: find R, t such that ref_3d ≈ R @ cur_3d.T + t
        cur_center = cur_3d.mean(axis=0, keepdims=True)
        ref_center = self.ref_3d.mean(axis=0, keepdims=True)
        
        cur_centered = cur_3d - cur_center
        ref_centered = self.ref_3d - ref_center
        
        # compute covariance
        H = cur_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        
        # rotation matrix (ensure det = +1 for proper orientation)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # translation vector
        t = (ref_center.T - R @ cur_center.T).reshape(3, 1)
        
        self.R = R.astype(np.float32)
        self.t = t.astype(np.float32)
        self.has_bias = True
        
        std = float(stacked.std(axis=0).mean())
        
        angles_rad = self._euler_from_rotation(self.R)
        angles_deg = np.degrees(angles_rad)
        trans_mag = float(np.linalg.norm(self.t))
        
        print(f"\ncalculated rotation (deg): pitch={angles_deg[0]:.2f}° roll={angles_deg[1]:.2f}° yaw={angles_deg[2]:.2f}°")
        print(f"calculated translation   : {trans_mag:.4f}")
        print(f"Saved as {CALIBRATION_FILE}")
        print(f"\n3D transformation calibration done. Detection should be live!")
        
        self._log({
            "step": "position_bias_3d",
            "timestamp": datetime.datetime.now().isoformat(),
            "n_frames": len(raw_frames_3d),
            "stability_std": std,
            "rotation_pitch_deg": float(angles_deg[0]),
            "rotation_roll_deg": float(angles_deg[1]),
            "rotation_yaw_deg": float(angles_deg[2]),
            "translation_magnitude": trans_mag,
        })
        return True
    
    @staticmethod
    def _euler_from_rotation(R):
        # extract pitch, roll, yaw from rotation matrix
        sin_pitch = np.clip(R[2, 0], -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        
        cos_pitch = np.cos(pitch)
        if np.abs(cos_pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # gimbal lock case
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        
        return np.array([pitch, roll, yaw], dtype=np.float32)
    
    def apply(self, raw_478x3):
        # apply 3D rotation + translation, project to 2D
        if self.has_bias and self.R is not None and self.t is not None:
            corrected_3d = (self.R @ raw_478x3.T + self.t).T
        else:
            corrected_3d = raw_478x3
        
        return corrected_3d[:, :2].astype(np.float32)
    
    def save(self, path):
        if not self.has_ref:
            print("  Nothing to save (no reference yet).")
            return
        d = {"ref_3d": self.ref_3d}
        if self.has_bias and self.R is not None and self.t is not None:
            d["R"] = self.R
            d["t"] = self.t
        np.savez(path, **d)
        print(f"  Saved → {path}")
    
    def load(self, path):
        try:
            d = np.load(path)
            self.ref_3d = d["ref_3d"].astype(np.float32)
            self.has_ref = True
            if "R" in d and "t" in d:
                self.R = d["R"].astype(np.float32)
                self.t = d["t"].astype(np.float32)
                self.has_bias = True
                print(f"  Loaded reference + transformation ← {path}")
            else:
                print(f"  Loaded reference (no transformation yet) ← {path}")
            return True
        except Exception as e:
            print(f"  Could not load {path}: {e}")
            return False
    
    def reset_bias(self):
        self.R = None
        self.t = None
        self.has_bias = False
    
    def _log(self, entry):
        log = []
        if os.path.exists(CALIBRATION_LOG):
            try:
                with open(CALIBRATION_LOG) as f:
                    log = json.load(f)
            except:
                pass
        log.append(entry)
        with open(CALIBRATION_LOG, "w") as f:
            json.dump(log, f, indent=2)


class PredictionSmoother:
    def __init__(self, n_classes=4, alpha=PROB_EMA_ALPHA, hold=HOLD_FRAMES):
        self.alpha = alpha
        self.hold = hold
        self.smooth = np.ones(n_classes, dtype=np.float32) / n_classes
        self.label = 0
        self.pending = 0
        self.count = 0
    
    def update(self, probs):
        probs = np.asarray(probs, dtype=np.float32)
        probs /= probs.sum() + 1e-9
        self.smooth = self.alpha * probs + (1.0 - self.alpha) * self.smooth
        new_label = int(np.argmax(self.smooth))
        
        if new_label == self.label:
            self.pending = new_label
            self.count = 0
        elif new_label == self.pending:
            self.count += 1
            if self.count >= self.hold:
                self.label = new_label
                self.count = 0
        else:
            self.pending = new_label
            self.count = 1
        
        return self.label
    
    @property
    def confidence(self):
        return float(np.max(self.smooth))
    
    def reset(self):
        n = len(self.smooth)
        self.smooth[:] = 1.0 / n
        self.label = 0
        self.pending = 0
        self.count = 0



def run_models(features_478x2, models, smoothed_flat=None):
    # run both models and average probabilities
    if smoothed_flat is not None:
        flat = smoothed_flat.reshape(1, -1)
    else:
        flat = features_478x2[SELECTED_LANDMARKS].flatten().reshape(1, -1)
    
    probs = []
    for m in models:
        try:
            probs.append(m.predict_proba(flat)[0])
        except Exception as e:
            print(f"  Model error: {e}")
    
    return np.mean(probs, axis=0) if probs else np.ones(4) / 4.0

_L_IRIS_CTR = 468
_R_IRIS_CTR = 473
_L_IRIS_RING = [468, 469, 470, 471, 472]
_R_IRIS_RING = [473, 474, 475, 476, 477]
_L_EYE_OUTER = 33
_L_EYE_INNER = 133
_R_EYE_OUTER = 263
_R_EYE_INNER = 362
_EYEBALL_LEFT = np.array([[29.05], [32.7], [-39.5]], dtype=np.float64)
_EYEBALL_RIGHT = np.array([[-29.05], [32.7], [-39.5]], dtype=np.float64)

def _make_camera_matrix(w, h):
    f = float(w)
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1.0]], dtype=np.float64)



def draw_debug_view(raw_478x2, corrected_478x2, label, confidence, has_bias, ear, canvas_size=300):
    # side-by-side view: raw landmarks vs corrected landmarks with 3D gaze rays
    half = canvas_size // 2
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    def draw_half(pts_478x2, x_off, header, dot_clr):
        # map coords to pixels on this half
        mn, mx = pts_478x2.min(axis=0), pts_478x2.max(axis=0)
        rng = np.maximum(mx - mn, 1e-9)
        margin = 16
        
        def to_px(idx):
            p = (pts_478x2[idx] - mn) / rng * (half - 2*margin) + margin
            return (int(p[0]) + x_off, int(p[1]))
        
        all_px = np.array([to_px(i) for i in range(len(pts_478x2))], dtype=np.int32)
        
        # landmark cloud
        for x, y in all_px:
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                cv2.circle(canvas, (x, y), 1, (45, 45, 45), -1)
        
        for idx in set(SELECTED_LANDMARKS):
            x, y = all_px[idx]
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                cv2.circle(canvas, (x, y), 2, dot_clr, -1)
        
        # nose tip
        nx, ny = all_px[NOSE_TIP]
        if 0 <= nx < canvas_size and 0 <= ny < canvas_size:
            cv2.circle(canvas, (nx, ny), 3, (0, 220, 220), -1)
        
        # iris circles
        iris_clr_l = (200, 0, 200)
        iris_clr_r = (0, 200, 200)
        for ring, clr in [(_L_IRIS_RING, iris_clr_l), (_R_IRIS_RING, iris_clr_r)]:
            rpx = np.array([all_px[i] for i in ring])
            ctr = rpx.mean(axis=0).astype(int)
            radius = max(2, int(np.linalg.norm(rpx - ctr, axis=1).mean()))
            if 0 <= ctr[0] < canvas_size and 0 <= ctr[1] < canvas_size:
                cv2.circle(canvas, tuple(ctr), radius, clr, 1)
                cv2.circle(canvas, tuple(ctr), 2, clr, -1)
        
        # header
        cv2.putText(canvas, header,
                    (x_off + 2, canvas_size - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (130, 130, 130), 1)
    
    draw_half(raw_478x2, 0, "RAW", (160, 80, 0))
    draw_half(corrected_478x2, half, "CORRECTED", (0, 160, 220))
    
    # center divider
    cv2.line(canvas, (half, 0), (half, canvas_size), (70, 70, 70), 1)
    
    # top label
    direction = LABEL_MAP.get(label, "?")
    clr = LABEL_COLORS.get(direction, (255, 255, 255))
    cv2.putText(canvas, direction,
                (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)
    
    if not has_bias:
        cv2.putText(canvas, "no bias",
                    (half + 4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                    (100, 100, 200), 1)
    
    # legend
    cv2.putText(canvas, "magenta=L  cyan=R",
                (2, canvas_size - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (120, 120, 120), 1)
    
    return canvas


# MAIN
def main():
    # load models
    models = []
    for path in MODEL_FILES:
        try:
            models.append(load(path))
            print(f"Loaded {path}")
        except Exception as e:
            print(f"Could not load {path}: {e}")
    
    if not models:
        print("No models loaded — exiting.")
        return
    
    norm = GazeNormalizer3D()
    smoother = PredictionSmoother()
    lm_smooth = LandmarkSmoother()
    show_norm = SHOW_NORM_VIEW
    
    ref_loaded = norm.load(CALIBRATION_FILE)
    
    # camera
    cam = cv2.VideoCapture(CAMERA_ID)
    if not cam.isOpened():
        print("Error: could not open camera.")
        return
    print("Camera opened\n")
    
    # state
    cal_buf = []
    label = 0
    raw_probs = np.ones(4) / 4.0
    cal_step = 1 if ref_loaded else 0
    
    # instructions
    print("DRIVER GAZE DETECTION")
    print("F=calibrate | R=reset bias | S=save | D=debug view | Q=quit")
    print()
    
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        # crop center 300x300
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        frame = frame[cy-150:cy+150, cx-150:cx+150]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord("d"), ord("D")):
            show_norm = not show_norm
            if not show_norm:
                cv2.destroyWindow("Debug View")
        
        if results.multi_face_landmarks:
            fl = results.multi_face_landmarks[0]
            raw = extract_raw(fl)
            raw_3d = extract_raw_3d(fl)
            
            # calibration
            if key in (ord("f"), ord("F")):
                cal_buf.append(raw_3d.copy())
                n = len(cal_buf)
                step_lbl = "(reference)" if cal_step == 0 else "(bias)"
                print(f"Calibrating {step_lbl} ... {n}/{CALIBRATION_FRAMES}", end="\r")
                
                if n >= CALIBRATION_FRAMES:
                    if cal_step == 0:
                        norm.calibrate_reference(cal_buf)
                        norm.save(CALIBRATION_FILE)
                        print(f"\nReference saved as {CALIBRATION_FILE}")
                        print("Move camera to new position and hold F to calibrate bias\n")
                        cal_step = 1
                    else:
                        ok = norm.calibrate_position(cal_buf)
                        if ok:
                            norm.save(CALIBRATION_FILE)
                    
                    smoother.reset()
                    lm_smooth.reset()
                    cal_buf.clear()
            
            if key in (ord("s"), ord("S")):
                norm.save(CALIBRATION_FILE)
            
            if key in (ord("r"), ord("R")):
                norm.reset_bias()
                smoother.reset()
                lm_smooth.reset()
                cal_step = 1
                print("\nBias reset. Hold F to re-calibrate.\n")
            
            # process frame
            corrected = norm.apply(raw_3d)
            ear = compute_ear(raw)
            raw_flat = corrected[SELECTED_LANDMARKS].flatten().astype(np.float32)
            smooth_flat = lm_smooth.update(raw_flat, ear)
            raw_probs = run_models(corrected, models, smoothed_flat=smooth_flat)
            
            # flip labels (frame is mirrored)
            raw_probs[1], raw_probs[2] = raw_probs[2].copy(), raw_probs[1].copy()
            
            label = smoother.update(raw_probs)
            
            # display
            if IS_DISPLAY:
                disp = frame.copy()
                
                mp_drawing.draw_landmarks(
                    image=disp,
                    landmark_list=fl,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style(),
                )
                
                direction = LABEL_MAP[label]
                clr = LABEL_COLORS.get(direction, (255, 255, 255))
                cv2.putText(disp, direction, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, clr, 2)
                
                # status bar
                if norm.has_bias:
                    s_txt = "CALIBRATED — per-landmark bias correction active"
                    s_clr = (0, 220, 0)
                elif norm.has_ref:
                    s_txt = "REF ONLY — hold F to calibrate camera position bias"
                    s_clr = (0, 200, 255)
                else:
                    s_txt = "NO REF — hold F (training position, look straight ahead)"
                    s_clr = (80, 80, 255)
                cv2.putText(disp, s_txt, (4, OUTPUT_SIZE - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27, s_clr, 1)
                
                # calibration progress
                if cal_buf:
                    pct = len(cal_buf) / CALIBRATION_FRAMES
                    cv2.rectangle(disp, (0, OUTPUT_SIZE - 5),
                                  (int(OUTPUT_SIZE * pct), OUTPUT_SIZE),
                                  (0, 255, 255), -1)
                
                cv2.imshow("Driver Gaze Detection", disp)
                
                # debug view
                if show_norm:
                    dbg = draw_debug_view(
                        raw, corrected, label, smoother.confidence,
                        norm.has_bias, ear, OUTPUT_SIZE)
                    cv2.imshow("Debug View", dbg)
        
        else:
            if IS_DISPLAY:
                disp = frame.copy()
                cv2.putText(disp, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Driver Gaze Detection", disp)
        
        if IS_WRITE and results.multi_face_landmarks:
            print(f"  {LABEL_MAP[label]:6s}  conf:{smoother.confidence:.2f}"
                  f"  F:{raw_probs[0]:.2f} L:{raw_probs[1]:.2f}"
                  f"  R:{raw_probs[2]:.2f} P:{raw_probs[3]:.2f}")
        
        if key == ord("q"):
            print("\nQuitting...")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()