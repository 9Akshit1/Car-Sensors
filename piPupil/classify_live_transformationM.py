import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import os
import json
import datetime

# =====================================================================
# CONFIGURATION
# =====================================================================
IS_DISPLAY        = True
IS_WRITE          = False
CAMERA_ID         = 0
OUTPUT_SIZE       = 300
SHOW_NORM_VIEW    = True

LABEL_MAP = {0: "Front", 1: "Left", 2: "Right", 3: "Phone"}
LABEL_COLORS = {
    "Front": (0, 220, 0),
    "Left":  (0, 220, 220),
    "Right": (0, 140, 255),
    "Phone": (0,   0, 255),
}

MODEL_FILES = [
    "Models/Cubic_SVM.pkl",
    "Models/Neural_Network.pkl",
]

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

CALIBRATION_FRAMES = 60   # more frames = steadier average bias
CALIBRATION_FILE   = "calibration/calibration_ref_transformationM.npz"
CALIBRATION_LOG    = "calibration/calibration_log_transformationM.json"

PROB_EMA_ALPHA = 0.55
HOLD_FRAMES    = 2

# Landmark indices
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP        = 4
FACE_CENTER_IDS = [10, 152, 234, 454]

# Eye landmarks for EAR (Eye Aspect Ratio) computation
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# Open eye ≈ 0.28-0.35 | partial close (phone gaze) ≈ 0.18-0.24 | blink < 0.15
_EAR_LEFT  = (33, 160, 158, 133, 153, 145)   # outer,top-out,top-in,inner,bot-in,bot-out
_EAR_RIGHT = (263, 387, 385, 362, 380, 374)
EAR_OPEN_THRESHOLD  = 0.23   # above this: iris landmarks are reliable
EAR_CLOSED_THRESHOLD = 0.15  # below this: eye is blinking, freeze iris entirely

# Per-region smoothing alphas for the landmark smoother (applied BEFORE the model)
# Iris landmarks (468-477): heavily smoothed because MediaPipe extrapolates
# them (rather than returning None) when eyelids occlude them during downward
# gaze — causing erratic jumps that are pure noise, not real gaze signal.
# Eyelid landmarks: moderate smoothing — they carry real phone-gaze signal
# (eyelids drop when looking down) but are also noisy.
_IRIS_LM   = frozenset(range(468, 478))
_EYELID_LM = frozenset({145,159,160,161,374,380,385,386,387,388})

# Build per-slot alpha array (146 values matching flattened selected landmarks)
_ALPHA_BASE   = 0.60   # most landmarks: responsive
_ALPHA_EYELID = 0.40   # eyelid: medium smoothing
_ALPHA_IRIS   = 0.20   # iris: heavy smoothing to kill extrapolation noise
_PER_SLOT_ALPHA = np.array(
    [(_ALPHA_IRIS   if lm in _IRIS_LM else
      _ALPHA_EYELID if lm in _EYELID_LM else
      _ALPHA_BASE)
     for lm in SELECTED_LANDMARKS
     for _ in (0, 1)],          # two values (x,y) per landmark
    dtype=np.float32
)  # shape (146,)

# solvePnP model for head-pose visualisation
_3D_MODEL_POINTS = np.array([
    [   0.0,    0.0,    0.0],
    [   0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)
_MODEL_LM_IDS = [4, 152, 33, 263, 61, 291]

# =====================================================================
# MEDIAPIPE
# =====================================================================
mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
)


# =====================================================================
# LANDMARK EXTRACTION
# =====================================================================
def extract_raw(face_landmarks) -> np.ndarray:
    """Extract (478, 2) float32 array of raw MediaPipe (x,y) coords."""
    return np.array([(lm.x, lm.y) for lm in face_landmarks.landmark],
                    dtype=np.float32)


def extract_raw_3d(face_landmarks) -> np.ndarray:
    """
    Extract (478, 3) float32 array of MediaPipe landmarks with depth.
    x, y are normalized image coords [0,1].
    z is MediaPipe's relative depth (normalized, no absolute units).
    """
    return np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
                    dtype=np.float32)


# =====================================================================
# EYE ASPECT RATIO
# =====================================================================
def compute_ear(lm: np.ndarray) -> float:
    """
    Eye Aspect Ratio averaged over both eyes.
    lm: (478, 2) raw landmark array.
    Returns float in roughly [0.0, 0.40].
    """
    def ear_one(p1, p2, p3, p4, p5, p6):
        A = np.linalg.norm(lm[p2] - lm[p6])
        B = np.linalg.norm(lm[p3] - lm[p5])
        C = np.linalg.norm(lm[p1] - lm[p4])
        return (A + B) / (2.0 * C + 1e-9)
    return float((ear_one(*_EAR_LEFT) + ear_one(*_EAR_RIGHT)) * 0.5)


# =====================================================================
# LANDMARK SMOOTHER  —  EMA on the feature vector, iris-aware
# =====================================================================
class LandmarkSmoother:
    """
    Smooths the flattened 146-dim feature vector (selected landmarks × xy)
    BEFORE it reaches the model, using per-region EMA alphas.

    Why smooth before the model instead of just smoothing probabilities:
      Probability smoothing averages decisions. Landmark smoothing averages
      inputs — it removes high-frequency jitter in the coordinates that
      makes the model oscillate between nearby class boundaries.

    Iris gating (EAR-based):
      When EAR drops below EAR_OPEN_THRESHOLD the eyelid is partially
      closed (typical downward / phone gaze). MediaPipe does NOT return
      None for iris landmarks in this case — it EXTRAPOLATES them, often
      wildly wrong and noisy. We lower the iris alpha (more history weight)
      dynamically so the smoother holds onto the last reliable iris position
      rather than chasing the extrapolated garbage.

      When EAR < EAR_CLOSED_THRESHOLD (blink) we freeze iris entirely
      (alpha → 0) so a blink doesn't inject a spike into the model input.
    """

    def __init__(self):
        self._smooth = None   # (146,) smoothed feature vector, lazy-init

    def update(self, flat_146: np.ndarray, ear: float) -> np.ndarray:
        """
        flat_146 : (146,) float32 — raw selected landmarks flattened
        ear      : float — current Eye Aspect Ratio
        Returns  : (146,) float32 — smoothed feature vector
        """
        if self._smooth is None:
            self._smooth = flat_146.copy()
            return self._smooth

        # Build effective alphas this frame based on EAR
        alphas = _PER_SLOT_ALPHA.copy()
        if ear < EAR_CLOSED_THRESHOLD:
            # Blink — freeze iris slots completely
            alphas = np.where(
                np.array([lm in _IRIS_LM
                           for lm in SELECTED_LANDMARKS
                           for _ in (0, 1)]),
                0.0, alphas
            ).astype(np.float32)
        elif ear < EAR_OPEN_THRESHOLD:
            # Partial close — linearly reduce iris alpha toward 0
            t = (ear - EAR_CLOSED_THRESHOLD) / (EAR_OPEN_THRESHOLD - EAR_CLOSED_THRESHOLD)
            reduced = _ALPHA_IRIS * t          # scales from 0 → _ALPHA_IRIS
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


# =====================================================================
# GAZE NORMALIZER 3D  —  proper rotation + translation in 3D space
# =====================================================================
class GazeNormalizer3D:
    """
    Stores two calibration snapshots of 3D landmarks (478, 3):

      ref_3d  : (478,3) mean 3D landmarks at TRAINING camera position (forward gaze)
      cur_3d  : (478,3) mean 3D landmarks at CURRENT camera position (forward gaze)

    Computes via SVD a rigid transformation (rotation R + translation t) that
    maps cur_3d → ref_3d:
      ref_3d ≈ R @ cur_3d.T + t  (in matrix form)

    For any live frame:
      1. Extract 3D: raw_3d = (478, 3)
      2. Transform: corrected_3d = (R @ raw_3d.T + t).T  (478, 3)
      3. Project: corrected_2d = corrected_3d[:, :2]  (orthographic, strip z)
      4. Feed to model: selected_2d landmarks from corrected_2d

    Why orthographic projection works:
      - Camera movement is mostly in-plane (left/right/up/down), not far in depth.
      - After rotation + translation, landmarks are in the trained model's 3D space.
      - Stripping z preserves the relative x,y positions that the 2D model was trained on.
      - If camera moved significantly in depth, you'd need camera intrinsics for proper
        perspective projection, but this is usually not necessary for dashcam setups.

    This approach is complete and rigorous: uses all 478 landmarks for alignment,
    solves for optimal rotation + translation via SVD, then projects cleanly to 2D.
    """

    def __init__(self):
        self.ref_3d  = None   # (478,3)  Step 1 snapshot in 3D
        self.R       = None   # (3,3)    Rotation matrix
        self.t       = None   # (3,1)    Translation vector
        self.has_ref = False
        self.has_bias = False  # renamed from has_bias, but kept for API compatibility

    # ------------------------------------------------------------------
    def calibrate_reference(self, raw_frames_3d: list):
        """
        Step 1 — at training camera position, driver looking forward.
        raw_frames_3d: list of (478,3) float32 arrays from extract_raw_3d().
        """
        stacked      = np.stack(raw_frames_3d, axis=0)      # (N, 478, 3)
        self.ref_3d  = stacked.mean(axis=0).astype(np.float32)
        self.has_ref = True
        self.has_bias = False
        self.R       = None
        self.t       = None

        std = float(stacked.std(axis=0).mean())
        print(f"\n  [Step 1] Reference 3D saved.  stability std={std:.5f}"
              f"  (< 0.001 = very steady, ideal)")
        return std

    def calibrate_position(self, raw_frames_3d: list) -> bool:
        """
        Step 2 — at new camera position, driver looking forward.
        Solves for rotation R and translation t via SVD-based rigid point cloud alignment.
        """
        if not self.has_ref:
            print("  ERROR: complete Step 1 first.")
            return False

        stacked  = np.stack(raw_frames_3d, axis=0)            # (N, 478, 3)
        cur_3d   = stacked.mean(axis=0).astype(np.float32)    # (478, 3)

        # --- SVD-based rigid alignment: solve R, t such that ref_3d ≈ R @ cur_3d.T + t ---
        # Center both point clouds
        cur_center = cur_3d.mean(axis=0, keepdims=True)       # (1, 3)
        ref_center = self.ref_3d.mean(axis=0, keepdims=True)  # (1, 3)

        cur_centered = cur_3d - cur_center                    # (478, 3)
        ref_centered = self.ref_3d - ref_center               # (478, 3)

        # SVD on covariance matrix
        H = cur_centered.T @ ref_centered                     # (3, 3)
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation matrix (ensure proper orientation, det(R) = +1)
        R = Vt.T @ U.T                                        # (3, 3)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation vector
        t = (ref_center.T - R @ cur_center.T).reshape(3, 1)  # (3, 1)

        self.R = R.astype(np.float32)
        self.t = t.astype(np.float32)
        self.has_bias = True

        std      = float(stacked.std(axis=0).mean())
        
        # Log rotation angles (Euler decomposition for human readability)
        angles_rad = self._extract_euler_angles(self.R)
        angles_deg = np.degrees(angles_rad)
        
        # Log translation magnitude
        trans_mag = float(np.linalg.norm(self.t))

        print(f"\n  [Step 2] 3D transformation computed.")
        print(f"    stability std : {std:.5f}  (< 0.001 is ideal)")
        print(f"    rotation (deg): pitch={angles_deg[0]:.2f}° roll={angles_deg[1]:.2f}° yaw={angles_deg[2]:.2f}°")
        print(f"    translation   : {trans_mag:.4f}  (magnitude of shift in normalized space)")

        if trans_mag < 0.001:
            print("  ⚠  Tiny translation — camera may be at same position as training.")
        if std > 0.003:
            print("  ⚠  High variance — keep head very still while holding F.")

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
    def _extract_euler_angles(R: np.ndarray) -> np.ndarray:
        """
        Extract Euler angles (pitch, roll, yaw) from rotation matrix R.
        Returns angles in radians.
        Handles gimbal lock gracefully.
        """
        # Clamp to avoid numerical issues in arcsin
        sin_pitch = np.clip(R[2, 0], -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        
        cos_pitch = np.cos(pitch)
        if np.abs(cos_pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock: set roll = 0, solve for yaw
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        
        return np.array([pitch, roll, yaw], dtype=np.float32)

    # ------------------------------------------------------------------
    def apply(self, raw_478x3: np.ndarray) -> np.ndarray:
        """
        Apply rotation + translation to 3D landmarks, then project to 2D.
        
        raw_478x3 : (478, 3) 3D landmarks from extract_raw_3d()
        Returns   : (478, 2) float32 2D landmarks (orthographic projection: strip z)
        
        If no transformation computed, returns 2D slice of raw unchanged.
        """
        if self.has_bias and self.R is not None and self.t is not None:
            # Apply rigid transformation: corrected_3d = (R @ raw_3d.T + t).T
            corrected_3d = (self.R @ raw_478x3.T + self.t).T  # (478, 3)
        else:
            corrected_3d = raw_478x3  # (478, 3)
        
        # Orthographic projection: strip z, keep x,y
        return corrected_3d[:, :2].astype(np.float32)

    # ------------------------------------------------------------------
    def save(self, path: str):
        if not self.has_ref:
            print("  Nothing to save (no reference yet).")
            return
        d = {"ref_3d": self.ref_3d}
        if self.has_bias and self.R is not None and self.t is not None:
            d["R"] = self.R
            d["t"] = self.t
        np.savez(path, **d)
        print(f"  Saved → {path}")

    def load(self, path: str) -> bool:
        try:
            d = np.load(path)
            self.ref_3d = d["ref_3d"].astype(np.float32)
            self.has_ref = True
            if "R" in d and "t" in d:
                self.R = d["R"].astype(np.float32)
                self.t = d["t"].astype(np.float32)
                self.has_bias = True
                print(f"  Loaded reference + rotation/translation ← {path}")
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

    def _log(self, entry: dict):
        log = []
        if os.path.exists(CALIBRATION_LOG):
            try:
                with open(CALIBRATION_LOG) as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append(entry)
        with open(CALIBRATION_LOG, "w") as f:
            json.dump(log, f, indent=2)


# =====================================================================
# PREDICTION SMOOTHER
# =====================================================================
class PredictionSmoother:
    def __init__(self, n_classes=4, alpha=PROB_EMA_ALPHA, hold=HOLD_FRAMES):
        self.alpha   = alpha
        self.hold    = hold
        self.smooth  = np.ones(n_classes, dtype=np.float32) / n_classes
        self.label   = 0
        self.pending = 0
        self.count   = 0

    def update(self, probs: np.ndarray) -> int:
        probs  = np.asarray(probs, dtype=np.float32)
        probs /= probs.sum() + 1e-9
        self.smooth = self.alpha * probs + (1.0 - self.alpha) * self.smooth
        new_label   = int(np.argmax(self.smooth))
        if new_label == self.label:
            self.pending = new_label
            self.count   = 0
        elif new_label == self.pending:
            self.count += 1
            if self.count >= self.hold:
                self.label = new_label
                self.count = 0
        else:
            self.pending = new_label
            self.count   = 1
        return self.label

    @property
    def confidence(self) -> float:
        return float(np.max(self.smooth))

    def reset(self):
        n = len(self.smooth)
        self.smooth[:] = 1.0 / n
        self.label = 0; self.pending = 0; self.count = 0


# =====================================================================
# MODEL INFERENCE
# =====================================================================
def run_models(features_478x2: np.ndarray, models: list,
               smoothed_flat: np.ndarray | None = None) -> np.ndarray:
    """
    features_478x2 : (478,2) bias-corrected landmarks
    smoothed_flat  : (146,) pre-smoothed feature vector (preferred).
                     If None, falls back to raw selection from features_478x2.
    """
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


# =====================================================================
# HEAD-POSE VISUALISATION
# =====================================================================
def draw_nose_axes(img, lm_norm01: np.ndarray, img_w: int, img_h: int,
                   axis_len=55, thickness=2, prefix=""):
    """Draw X/Y/Z axes at nose tip. lm_norm01 must be in 0-1 coords."""
    pts2d = np.array([[lm_norm01[i, 0] * img_w,
                       lm_norm01[i, 1] * img_h]
                      for i in _MODEL_LM_IDS], dtype=np.float64)
    f   = float(img_w)
    cam = np.array([[f, 0, img_w/2], [0, f, img_h/2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(
        _3D_MODEL_POINTS, pts2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return
    axes = np.float32([[0,0,0],[axis_len,0,0],[0,-axis_len,0],[0,0,-axis_len]])
    proj, _ = cv2.projectPoints(axes, rvec, tvec, cam, dist)
    proj    = proj.astype(int).reshape(-1, 2)
    o = tuple(proj[0])
    cv2.line(img, o, tuple(proj[1]), (0,   0, 220), thickness)  # X red
    cv2.line(img, o, tuple(proj[2]), (0, 220,   0), thickness)  # Y green
    cv2.line(img, o, tuple(proj[3]), (220,  0,   0), thickness) # Z blue
    if prefix:
        for pt, lbl in zip(proj[1:], ["X", "Y", "Z"]):
            cv2.putText(img, prefix + lbl, tuple(pt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)


# =====================================================================
# GAZE RAY COMPUTATION  —  proper 3-D method
# =====================================================================
# Iris / eye landmark indices
_L_IRIS_CTR  = 468
_R_IRIS_CTR  = 473
_L_IRIS_RING = [468, 469, 470, 471, 472]
_R_IRIS_RING = [473, 474, 475, 476, 477]
_L_EYE_OUTER = 33
_L_EYE_INNER = 133
_R_EYE_OUTER = 263
_R_EYE_INNER = 362

# 3-D eyeball centres in the same model space as _3D_MODEL_POINTS
# (derived from standard anthropometric measurements, units = mm)
_EYEBALL_LEFT  = np.array([[ 29.05], [32.7], [-39.5]], dtype=np.float64)
_EYEBALL_RIGHT = np.array([[-29.05], [32.7], [-39.5]], dtype=np.float64)

# 6-point head model reused for solvePnP (same as _3D_MODEL_POINTS / _MODEL_LM_IDS)
_GAZE_MODEL_3D = _3D_MODEL_POINTS          # (6,3)
_GAZE_MODEL_IDS = _MODEL_LM_IDS            # [4,152,33,263,61,291]


def _make_camera_matrix(w: int, h: int) -> np.ndarray:
    f = float(w)
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1.0  ]], dtype=np.float64)


def compute_gaze_rays(lm_478x2: np.ndarray, img_w: int, img_h: int):
    """
    Compute head-pose-corrected 3-D gaze rays for both eyes.

    Steps
    -----
    1. solvePnP on 6 stable face landmarks → rotation (rvec) + translation (tvec)
       that maps 3-D model points into camera coords.
    2. estimateAffine3D maps image points → 3-D model space (T_img2world).
    3. Each iris centre (2-D image point) is lifted to 3-D model space via T.
    4. Gaze direction = eyeball_centre → pupil_3d, extended by GAZE_DIST mm.
    5. The 3-D gaze endpoint is projected back to 2-D image coords.
    6. Head-pose correction: subtract the residual head-translation component
       (same technique as the reference code) so the arrow shows EYE movement
       not head movement.

    Returns
    -------
    dict with keys:
      'ok'            : bool — False if solvePnP / estimateAffine3D failed
      'rvec','tvec'   : head pose
      'cam_mat'       : camera matrix used
      'l_pupil_px'    : (2,) int  — left iris centre in image pixels
      'r_pupil_px'    : (2,) int  — right iris centre in image pixels
      'l_gaze_px'     : (2,) int  — left gaze endpoint in image pixels
      'r_gaze_px'     : (2,) int  — right gaze endpoint in image pixels
      'l_gaze_norm'   : (2,) float — normalised gaze offset (units: eye-widths)
      'r_gaze_norm'   : (2,) float — normalised gaze offset
      'combined_norm' : (2,) float — average of both eyes
    """
    GAZE_DIST = 200.0   # mm — how far ahead the gaze ray is projected

    # ---- pixel coords of the 6 model landmarks ----
    pts2d = np.array([[lm_478x2[i, 0] * img_w,
                       lm_478x2[i, 1] * img_h]
                      for i in _GAZE_MODEL_IDS], dtype=np.float64)

    # also as (x,y,0) for estimateAffine3D
    pts2d_z0 = np.hstack([pts2d, np.zeros((6, 1))]).astype(np.float64)

    cam_mat = _make_camera_matrix(img_w, img_h)
    dist    = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        _GAZE_MODEL_3D, pts2d, cam_mat, dist,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"ok": False}

    _, T, _ = cv2.estimateAffine3D(pts2d_z0, _GAZE_MODEL_3D)
    if T is None:
        return {"ok": False}

    # ---- pupil pixel coords ----
    l_px = np.array([lm_478x2[_L_IRIS_CTR, 0] * img_w,
                     lm_478x2[_L_IRIS_CTR, 1] * img_h], dtype=np.float64)
    r_px = np.array([lm_478x2[_R_IRIS_CTR, 0] * img_w,
                     lm_478x2[_R_IRIS_CTR, 1] * img_h], dtype=np.float64)

    def project_gaze(pupil_2d_px, eyeball_3d):
        """Compute head-corrected gaze endpoint in image pixels."""
        # lift pupil from image → 3-D world
        h_coord = np.array([[pupil_2d_px[0]], [pupil_2d_px[1]], [0.0], [1.0]],
                            dtype=np.float64)
        pupil_w = T @ h_coord                        # (3,1) in model/world space

        # Gaze ray: eyeball → pupil, extended GAZE_DIST mm
        direction  = pupil_w - eyeball_3d            # (3,1)
        norm       = np.linalg.norm(direction) + 1e-9
        S          = eyeball_3d + direction / norm * GAZE_DIST  # 3-D gaze point

        # Project 3-D gaze point to image
        gaze_3d, _ = cv2.projectPoints(
            S.T, rvec, tvec, cam_mat, dist)          # → (1,1,2)
        gaze_2d = gaze_3d[0][0]

        # Head-pose correction: project pupil_w at z=40 to find head translation
        head_proj, _ = cv2.projectPoints(
            np.array([[float(pupil_w[0]), float(pupil_w[1]), 40.0]]),
            rvec, tvec, cam_mat, dist)
        head_2d = head_proj[0][0]

        # corrected endpoint = pupil + (gaze - pupil) - (head_proj - pupil)
        tip = pupil_2d_px + (gaze_2d - pupil_2d_px) - (head_2d - pupil_2d_px)
        return tip.astype(np.float64)

    l_tip = project_gaze(l_px, _EYEBALL_LEFT)
    r_tip = project_gaze(r_px, _EYEBALL_RIGHT)

    # ---- normalised gaze offset (eye-width units, for readout) ----
    l_outer_px = np.array([lm_478x2[_L_EYE_OUTER,0]*img_w, lm_478x2[_L_EYE_OUTER,1]*img_h])
    l_inner_px = np.array([lm_478x2[_L_EYE_INNER,0]*img_w, lm_478x2[_L_EYE_INNER,1]*img_h])
    r_outer_px = np.array([lm_478x2[_R_EYE_OUTER,0]*img_w, lm_478x2[_R_EYE_OUTER,1]*img_h])
    r_inner_px = np.array([lm_478x2[_R_EYE_INNER,0]*img_w, lm_478x2[_R_EYE_INNER,1]*img_h])
    l_ew = np.linalg.norm(l_outer_px - l_inner_px) + 1e-9
    r_ew = np.linalg.norm(r_outer_px - r_inner_px) + 1e-9

    l_norm = (l_tip - l_px) / l_ew
    r_norm = (r_tip - r_px) / r_ew

    return {
        "ok":           True,
        "rvec":         rvec,
        "tvec":         tvec,
        "cam_mat":      cam_mat,
        "l_pupil_px":   l_px.astype(int),
        "r_pupil_px":   r_px.astype(int),
        "l_gaze_px":    l_tip.astype(int),
        "r_gaze_px":    r_tip.astype(int),
        "l_gaze_norm":  l_norm,
        "r_gaze_norm":  r_norm,
        "combined_norm": (l_norm + r_norm) * 0.5,
    }


def draw_gaze_on_image(img, gaze_result: dict, color_l=(200,0,200),
                       color_r=(0,200,200), thickness=2):
    """
    Draw iris circles + head-corrected gaze arrows onto img (in-place).
    Works on the main camera frame (full pixel coords).
    """
    if not gaze_result.get("ok"):
        return

    def ipt(arr):
        return (int(arr[0]), int(arr[1]))

    # Gaze arrows
    cv2.arrowedLine(img,
                    ipt(gaze_result["l_pupil_px"]),
                    ipt(gaze_result["l_gaze_px"]),
                    color_l, thickness, tipLength=0.35)
    cv2.arrowedLine(img,
                    ipt(gaze_result["r_pupil_px"]),
                    ipt(gaze_result["r_gaze_px"]),
                    color_r, thickness, tipLength=0.35)

    # Numeric readout
    cn = gaze_result["combined_norm"]
    cv2.putText(img, f"gz({cn[0]:+.2f},{cn[1]:+.2f})",
                (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200,200,200), 1)


# =====================================================================
# DEBUG VIEW — raw vs corrected side by side, with gaze rays
# =====================================================================
def draw_debug_view(raw_478x2: np.ndarray, corrected_478x2: np.ndarray,
                    label: int, confidence: float,
                    has_bias: bool, ear: float, canvas_size=300) -> np.ndarray:
    """
    Left half  : RAW landmarks + 3-D gaze rays (what camera actually sees)
    Right half : CORRECTED landmarks + 3-D gaze rays (what model receives)

    Gaze rays use the full 3-D method: solvePnP → estimateAffine3D → eyeball
    ray → head-pose correction.  The corrected half lets you verify that the
    iris is pointing in the expected direction in training-coord space.

    Colors: magenta = left eye,  cyan = right eye
    """
    half   = canvas_size // 2
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    def draw_half(pts_478x2: np.ndarray, x_off: int, header: str, dot_clr: tuple):
        # ---- map coords → pixels on this half-canvas ----
        mn, mx = pts_478x2.min(axis=0), pts_478x2.max(axis=0)
        rng    = np.maximum(mx - mn, 1e-9)
        margin = 16

        def to_px(idx):
            p = (pts_478x2[idx] - mn) / rng * (half - 2*margin) + margin
            return (int(p[0]) + x_off, int(p[1]))

        all_px = np.array([to_px(i) for i in range(len(pts_478x2))], dtype=np.int32)

        # ---- landmark cloud ----
        for x, y in all_px:
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                cv2.circle(canvas, (x, y), 1, (45, 45, 45), -1)
        for idx in set(SELECTED_LANDMARKS):
            x, y = all_px[idx]
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                cv2.circle(canvas, (x, y), 2, dot_clr, -1)
        # Nose tip
        nx, ny = all_px[NOSE_TIP]
        if 0 <= nx < canvas_size and 0 <= ny < canvas_size:
            cv2.circle(canvas, (nx, ny), 3, (0, 220, 220), -1)

        # ---- iris circles ----
        iris_clr_l = (200, 0, 200)
        iris_clr_r = (0, 200, 200)
        for ring, clr in [(_L_IRIS_RING, iris_clr_l), (_R_IRIS_RING, iris_clr_r)]:
            rpx    = np.array([all_px[i] for i in ring])
            ctr    = rpx.mean(axis=0).astype(int)
            radius = max(2, int(np.linalg.norm(rpx - ctr, axis=1).mean()))
            if 0 <= ctr[0] < canvas_size and 0 <= ctr[1] < canvas_size:
                cv2.circle(canvas, tuple(ctr), radius, clr, 1)
                cv2.circle(canvas, tuple(ctr), 2, clr, -1)

        # ---- 3-D gaze rays ----
        # Build a fake pixel-space landmark array scaled to this half-canvas.
        # The half-canvas IS the image for solvePnP purposes — dimensions half × canvas_size.
        fake_lm = np.zeros_like(pts_478x2)
        fake_lm[:, 0] = (all_px[:, 0] - x_off) / float(half)        # 0-1 in half-width
        fake_lm[:, 1] =  all_px[:, 1]           / float(canvas_size) # 0-1 in height

        gr = compute_gaze_rays(fake_lm, half, canvas_size)

        def clamp2(pt):
            return (int(np.clip(pt[0] + x_off, 0, canvas_size-1)),
                    int(np.clip(pt[1],          0, canvas_size-1)))

        if gr["ok"]:
            # shift x by x_off so we draw on the right half
            lp = clamp2(gr["l_pupil_px"])
            lt = clamp2(gr["l_gaze_px"])
            rp = clamp2(gr["r_pupil_px"])
            rt = clamp2(gr["r_gaze_px"])

            cv2.arrowedLine(canvas, lp, lt, iris_clr_l, 2, tipLength=0.35)
            cv2.arrowedLine(canvas, rp, rt, iris_clr_r, 2, tipLength=0.35)

            cn = gr["combined_norm"]
            cv2.putText(canvas, f"gz({cn[0]:+.2f},{cn[1]:+.2f})",
                        (x_off + 2, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        else:
            cv2.putText(canvas, "gaze N/A",
                        (x_off + 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (80, 80, 80), 1)

        # ---- nose-pose axes ----
        sub = canvas[:, x_off:x_off + half].copy()
        draw_nose_axes(sub, fake_lm, half, canvas_size,
                       axis_len=22, thickness=1, prefix="")
        canvas[:, x_off:x_off + half] = sub

        # ---- header ----
        cv2.putText(canvas, header,
                    (x_off + 2, canvas_size - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (130, 130, 130), 1)

    draw_half(raw_478x2,       0,    "RAW",       (160,  80,   0))
    draw_half(corrected_478x2, half, "CORRECTED", (  0, 160, 220))

    # ---- centre divider ----
    cv2.line(canvas, (half, 0), (half, canvas_size), (70, 70, 70), 1)

    # ---- top label ----
    direction = LABEL_MAP.get(label, "?")
    clr = LABEL_COLORS.get(direction, (255, 255, 255))
    cv2.putText(canvas, f"{direction}  conf:{confidence:.2f}",
                (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)

    if not has_bias:
        cv2.putText(canvas, "no bias",
                    (half + 4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                    (100, 100, 200), 1)

    # ---- legend + EAR bar ----
    cv2.putText(canvas, "magenta=L  cyan=R",
                (2, canvas_size - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (120, 120, 120), 1)

    ear_clr = ((0, 200, 0)   if ear > EAR_OPEN_THRESHOLD   else
               (0, 200, 255) if ear > EAR_CLOSED_THRESHOLD else
               (0,   0, 220))
    bar_w = int(np.clip(ear / 0.35, 0, 1) * (canvas_size - 4))
    cv2.rectangle(canvas,
                  (2, canvas_size - 3), (2 + bar_w, canvas_size - 1),
                  ear_clr, -1)
    iris_state = ("iris:OK"     if ear > EAR_OPEN_THRESHOLD   else
                  "iris:gated"  if ear > EAR_CLOSED_THRESHOLD else
                  "iris:frozen")
    cv2.putText(canvas, f"EAR:{ear:.2f} {iris_state}",
                (half + 2, canvas_size - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, ear_clr, 1)

    return canvas


# =====================================================================
# MAIN
# =====================================================================
def main():
    # ---- load models ------------------------------------------------
    models = []
    for path in MODEL_FILES:
        try:
            models.append(load(path))
            print(f"  Loaded  {path}")
        except Exception as e:
            print(f"  SKIP    {path}: {e}")
    if not models:
        print("No models loaded — exiting.")
        return

    norm      = GazeNormalizer3D()
    smoother  = PredictionSmoother()
    lm_smooth = LandmarkSmoother()
    show_norm = SHOW_NORM_VIEW

    ref_loaded = norm.load(CALIBRATION_FILE)

    # ---- camera -----------------------------------------------------
    cam = cv2.VideoCapture(CAMERA_ID)
    if not cam.isOpened():
        print("Error: could not open camera.")
        return
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH,  OUTPUT_SIZE)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_SIZE)
    print("  Camera opened")

    # ---- state ------------------------------------------------------
    cal_buf   = []
    label     = 0
    raw_probs = np.ones(4) / 4.0
    cal_step  = 1 if ref_loaded else 0  # 0=Step1(ref), 1=Step2(bias)

    # ---- instructions -----------------------------------------------
    print("=" * 64)
    print("  DRIVER GAZE DETECTION")
    print("=" * 64)
    if not ref_loaded:
        print("  FIRST RUN — no reference found.")
        print("  → Place camera at TRAINING position.")
        print("  → Sit in seat, look STRAIGHT AHEAD.")
        print("  → Hold F until bar fills (60 frames) → saves reference.")
    else:
        if norm.has_bias:
            print("  ✓ Reference + bias correction loaded — ready.")
            print("  → If camera moved again: look STRAIGHT AHEAD, hold F.")
        else:
            print("  ✓ Reference loaded.")
            print("  → Camera at training position: just use it (no Step 2 needed).")
            print("  → Camera moved: look STRAIGHT AHEAD, hold F.")
    print()
    print("  F=calibrate | R=reset bias | S=save | D=debug view | Q=quit")
    print("=" * 64 + "\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # Mirror so left/right feel natural when self-monitoring
        frame = cv2.flip(frame, 1)

        # Crop centre 300×300 directly from native camera resolution
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        frame = frame[cy-150:cy+150, cx-150:cx+150]

        '''# Crop centre square
        h, w   = frame.shape[:2]
        side   = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        #frame  = frame[y0:y0+side, x0:x0+side]
        #frame  = cv2.resize(frame, (OUTPUT_SIZE, OUTPUT_SIZE),
        #                    interpolation=cv2.INTER_LINEAR)
        half  = OUTPUT_SIZE // 2
        cy, cx = h // 2, w // 2
        frame  = frame[cy-half:cy+half, cx-half:cx+half]'''

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        key     = cv2.waitKey(1) & 0xFF

        if key in (ord("d"), ord("D")):
            show_norm = not show_norm
            if not show_norm:
                cv2.destroyWindow("Debug View")

        if results.multi_face_landmarks:
            fl  = results.multi_face_landmarks[0]
            raw = extract_raw(fl)   # (478, 2) raw 0-1 coords
            raw_3d = extract_raw_3d(fl)  # (478, 3) with depth

            # ---- CALIBRATION (hold F) --------------------------------
            if key in (ord("f"), ord("F")):
                cal_buf.append(raw_3d.copy())  # Store 3D landmarks for calibration
                n = len(cal_buf)
                step_lbl = "Step 1 (reference)" if cal_step == 0 else "Step 2 (bias)"
                print(f"  Calibrating {step_lbl} ... {n}/{CALIBRATION_FRAMES}",
                      end="\r")

                if n >= CALIBRATION_FRAMES:
                    if cal_step == 0:
                        norm.calibrate_reference(cal_buf)
                        norm.save(CALIBRATION_FILE)
                        print("\n  ✓ Reference saved.")
                        print("  Now move camera to car position, look FORWARD, hold F.")
                        cal_step = 1
                    else:
                        ok = norm.calibrate_position(cal_buf)
                        if ok:
                            norm.save(CALIBRATION_FILE)
                            print("  ✓ 3D transformation calibration done — detection live!")
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
                print("\n  Bias reset. Look STRAIGHT AHEAD, hold F to re-calibrate.")

            # ---- APPLY 3D ROTATION+TRANSLATION + SMOOTH + PREDICT -----
            corrected  = norm.apply(raw_3d)     # (478,2): apply 3D rotation+translation, project to 2D
            ear        = compute_ear(raw)       # eye openness from original raw 2D
            raw_flat   = corrected[SELECTED_LANDMARKS].flatten().astype(np.float32)
            smooth_flat = lm_smooth.update(raw_flat, ear)  # (146,) smoothed
            raw_probs  = run_models(corrected, models, smoothed_flat=smooth_flat)

            # Frame is mirrored → swap model's Left/Right labels
            raw_probs[1], raw_probs[2] = raw_probs[2].copy(), raw_probs[1].copy()

            label = smoother.update(raw_probs)

            # ---- DISPLAY — MAIN WINDOW ------------------------------
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

                draw_nose_axes(disp, raw, OUTPUT_SIZE, OUTPUT_SIZE,
                               axis_len=55, thickness=2, prefix="")

                # 3-D gaze rays on main feed
                gaze_result = compute_gaze_rays(raw, OUTPUT_SIZE, OUTPUT_SIZE)
                draw_gaze_on_image(disp, gaze_result)

                direction = LABEL_MAP[label]
                clr = LABEL_COLORS.get(direction, (255, 255, 255))
                cv2.putText(disp, direction, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, clr, 2)
                cv2.putText(disp, f"Conf: {smoother.confidence:.2f}",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (255, 200, 0), 1)
                prob_txt = (f"F:{raw_probs[0]:.2f} L:{raw_probs[1]:.2f}"
                            f" R:{raw_probs[2]:.2f} P:{raw_probs[3]:.2f}")
                cv2.putText(disp, prob_txt, (10, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

                # EAR indicator — shows eye openness and iris reliability
                ear_clr = ((0, 220, 0)   if ear > EAR_OPEN_THRESHOLD else
                           (0, 200, 255) if ear > EAR_CLOSED_THRESHOLD else
                           (0,   0, 255))
                ear_lbl = ("iris:OK" if ear > EAR_OPEN_THRESHOLD else
                           "iris:gated" if ear > EAR_CLOSED_THRESHOLD else
                           "iris:frozen")
                cv2.putText(disp, f"EAR:{ear:.2f} {ear_lbl}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, ear_clr, 1)

                # Status bar
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

                # Calibration progress bar
                if cal_buf:
                    pct = len(cal_buf) / CALIBRATION_FRAMES
                    cv2.rectangle(disp, (0, OUTPUT_SIZE - 5),
                                  (int(OUTPUT_SIZE * pct), OUTPUT_SIZE),
                                  (0, 255, 255), -1)

                cv2.imshow("Driver Gaze Detection", disp)

                # ---- DEBUG VIEW — raw vs corrected ------------------
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
            print("\nQuitting …")
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()