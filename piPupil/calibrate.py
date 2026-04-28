import cv2
import mediapipe as mp
import numpy as np
import os
import json
import datetime


CAMERA_ID = 1        #change if needed
OUTPUT_SIZE = 300
CALIBRATION_FRAMES = 60
CALIBRATION_FILE = "calibration/calibration_ref_transformationM.npz"
CALIBRATION_LOG = "calibration/calibration_log_transformationM.json"

if not os.path.exists("calibration"):
    os.makedirs("calibration")

# Eye aspect ratio
_EAR_LEFT = (33, 160, 158, 133, 153, 145)
_EAR_RIGHT = (263, 387, 385, 362, 380, 374)
EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.15

# smoothing
_IRIS_LM = frozenset(range(468, 478))
_EYELID_LM = frozenset({145, 159, 160, 161, 374, 380, 385, 386, 387, 388})
_ALPHA_BASE = 0.60
_ALPHA_EYELID = 0.40
_ALPHA_IRIS = 0.20

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

_PER_SLOT_ALPHA = np.array(
    [(_ALPHA_IRIS if lm in _IRIS_LM else
      _ALPHA_EYELID if lm in _EYELID_LM else
      _ALPHA_BASE)
     for lm in SELECTED_LANDMARKS
     for _ in (0, 1)],
    dtype=np.float32
)

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
    # smooths the 146-dim feature vector using per-region EMA alphas
    # iris landmarks get heavier smoothing when eye is closed (extrapolation is noisy)
    
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
    # calibrates 3D transformation (rotation R + translation t) between camera positions
    # uses SVD to find rigid alignment, then applies to correct landmark positions
    
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


def main():
    norm = GazeNormalizer3D()
    lm_smooth = LandmarkSmoother()
    
    ref_loaded = norm.load(CALIBRATION_FILE)
    
    cam = cv2.VideoCapture(CAMERA_ID)
    if not cam.isOpened():
        print("Error: could not open camera.")
        return
    print("Camera opened\n")
    
    cal_buf = []
    cal_step = 1 if ref_loaded else 0
    
    print("GAZE CALIBRATION")
    print("F=calibrate | R=reset bias | S=save | Q=quit")
    print()
    
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        frame = frame[cy-150:cy+150, cx-150:cx+150]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        key = cv2.waitKey(1) & 0xFF
        
        if results.multi_face_landmarks:
            fl = results.multi_face_landmarks[0]
            raw = extract_raw(fl)
            raw_3d = extract_raw_3d(fl)
            
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
                    
                    lm_smooth.reset()
                    cal_buf.clear()
            
            if key in (ord("s"), ord("S")):
                norm.save(CALIBRATION_FILE)
            
            if key in (ord("r"), ord("R")):
                norm.reset_bias()
                lm_smooth.reset()
                cal_step = 1
                print("\nBias reset. Hold F to re-calibrate.\n")
            
            disp = frame.copy()
            mp_drawing.draw_landmarks(
                image=disp,
                landmark_list=fl,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style(),
            )
            
            status = "CALIBRATED" if norm.has_bias else "REF ONLY" if norm.has_ref else "NO REF"
            cv2.putText(disp, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 220, 0), 2)
            
            if cal_buf:
                pct = len(cal_buf) / CALIBRATION_FRAMES
                cv2.rectangle(disp, (0, OUTPUT_SIZE - 5),
                              (int(OUTPUT_SIZE * pct), OUTPUT_SIZE),
                              (0, 255, 255), -1)
            
            cv2.imshow("Calibration", disp)
        
        else:
            disp = frame.copy()
            cv2.putText(disp, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Calibration", disp)
        
        if key == ord("q"):
            print("\nQuitting...")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()