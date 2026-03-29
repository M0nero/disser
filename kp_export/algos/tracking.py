import numpy as np
import cv2
import math
from time import perf_counter
from typing import List, Dict, Optional, Tuple, Any

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=None, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        One Euro Filter for jitter reduction.
        t0: initial timestamp
        x0: initial value (numpy array)
        dx0: initial derivative (optional)
        min_cutoff: minimum cutoff frequency
        beta: speed coefficient
        d_cutoff: cutoff frequency for derivative
        """
        self.t_prev = t0
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.array(dx0, dtype=np.float32) if dx0 is not None else np.zeros_like(self.x_prev)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        Filter a new value x at timestamp t.
        """
        x = np.array(x, dtype=np.float32)
        t_e = t - self.t_prev
        
        # Prevent division by zero or negative time delta
        if t_e <= 0:
            return self.x_prev

        # Estimate derivative
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Calculate cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        
        # Filter signal
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

class HandTracker:
    def __init__(self, 
                 min_cutoff=0.01, # Very low cutoff for static hands
                 beta=20.0,       # High beta to react quickly to movement
                 d_cutoff=1.0,
                 optical_flow_win_size=(21, 21),
                 optical_flow_max_level=3):
        self.filters: Dict[int, OneEuroFilter] = {}
        self.last_valid_landmarks: Optional[List[Dict[str, float]]] = None
        self.last_valid_ts = 0.0
        self.is_tracking = False
        
        # Filter params
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # Optical Flow params
        self.lk_params = dict(
            winSize=optical_flow_win_size,
            maxLevel=optical_flow_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_gray = None
        self.prev_pts = None

    def reset(self):
        self.filters = {}
        self.last_valid_landmarks = None
        self.is_tracking = False
        self.prev_gray = None
        self.prev_pts = None

    def update(self, landmarks: List[Dict[str, float]], ts: float, image: np.ndarray, score: float = 1.0):
        """
        Update tracker with new detection from MediaPipe.
        """
        self.is_tracking = False # Reset tracking flag as we have a fresh detection
        self.last_valid_landmarks = landmarks
        self.last_valid_ts = ts
        self.last_score = score
        
        # Convert image to gray for next optical flow step
        if image is not None:
            self.prev_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Extract points for optical flow (x, y only)
            h, w = self.prev_gray.shape
            pts = []
            for lm in landmarks:
                px = lm['x'] * w
                py = lm['y'] * h
                pts.append([px, py])
            self.prev_pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

        # Update filters
        filtered_landmarks = []
        for i, lm in enumerate(landmarks):
            val = np.array([lm['x'], lm['y'], lm['z']], dtype=np.float32)
            
            if i not in self.filters:
                self.filters[i] = OneEuroFilter(ts, val, 
                                                min_cutoff=self.min_cutoff, 
                                                beta=self.beta, 
                                                d_cutoff=self.d_cutoff)
                filtered_val = val
            else:
                filtered_val = self.filters[i](ts, val)
            
            filtered_landmarks.append({
                "x": float(filtered_val[0]),
                "y": float(filtered_val[1]),
                "z": float(filtered_val[2]),
                "visibility": lm.get("visibility", 0.0) # Pass through visibility
            })
            
        return filtered_landmarks

    def track(self, image: np.ndarray, ts: float) -> Optional[List[Dict[str, float]]]:
        """
        Attempt to track landmarks using Optical Flow when detection fails.
        Uses Affine Transform to preserve hand shape.
        """
        if self.prev_gray is None or self.prev_pts is None or self.last_valid_landmarks is None:
            return None
            
        if image is None:
            return None

        curr_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pts, None, **self.lk_params)
        
        # Select good points
        good_old = self.prev_pts[st == 1]
        good_new = p1[st == 1]
        
        # Need at least 3 points for Affine
        if len(good_old) < 3:
            return None
            
        # Estimate Affine Transform (Translation + Rotation + Scale)
        # estimateAffinePartial2D is more robust than estimateAffine2D for this as it restricts shear
        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        if M is None:
            return None
            
        # Check if transform is reasonable (not too crazy)
        # Scale should be close to 1.0, rotation small
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        if not (0.8 < scale < 1.2): # Limit scale change to 20% per frame
            return None
            
        # Apply transform to ALL previous points (not just tracked ones)
        # This preserves the shape exactly!
        
        # Prepare all previous points
        h, w = curr_gray.shape
        all_prev_pts = []
        for lm in self.last_valid_landmarks:
            px = lm['x'] * w
            py = lm['y'] * h
            all_prev_pts.append([px, py])
        all_prev_pts = np.array(all_prev_pts, dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform
        transformed_pts = cv2.transform(all_prev_pts, M)
        
        # Update state
        self.prev_gray = curr_gray
        self.prev_pts = transformed_pts # Use transformed points for next tracking step
        self.is_tracking = True
        
        tracked_landmarks = []
        for i, pt in enumerate(transformed_pts):
            x, y = pt.ravel()
            nx = x / w
            ny = y / h
            nz = self.last_valid_landmarks[i]['z'] # Hold Z
            
            # Apply filter
            val = np.array([nx, ny, nz], dtype=np.float32)
            if i in self.filters:
                 filtered_val = self.filters[i](ts, val)
                 nx, ny, nz = filtered_val
            
            tracked_landmarks.append({
                "x": float(nx),
                "y": float(ny),
                "z": float(nz),
                "visibility": self.last_valid_landmarks[i].get("visibility", 0.0)
            })

        self.last_valid_landmarks = tracked_landmarks
        return tracked_landmarks

def _is_frame_record(frame: Any) -> bool:
    return hasattr(frame, "hand_1") and hasattr(frame, "hand_2") and hasattr(frame, "diagnostics")


def _get_hand_landmarks(frame: Any, hand_key: str):
    if _is_frame_record(frame):
        hand = frame.hand_1 if hand_key in {"hand 1", "hand_1"} else frame.hand_2
        return hand.landmarks
    return frame.get(hand_key)


def _set_hand_landmarks(frame: Any, hand_key: str, landmarks):
    if _is_frame_record(frame):
        hand = frame.hand_1 if hand_key in {"hand 1", "hand_1"} else frame.hand_2
        hand.landmarks = landmarks
        frame.both_hands = bool(frame.hand_1.landmarks is not None and frame.hand_2.landmarks is not None)
        return
    frame[hand_key] = landmarks
    frame["both_hands"] = 1 if (frame.get("hand 1") is not None and frame.get("hand 2") is not None) else 0


def smooth_tracks(frames: List[Any], window_size=5):
    """
    Apply global smoothing to the collected frames.
    Simple moving average for now, but can be upgraded to Gaussian.
    """
    if not frames:
        return

    # Extract trajectories
    # We need to smooth x, y, z for hand 1 and hand 2
    
    def get_hand_array(hand_key):
        arr = []
        for fr in frames:
            h = _get_hand_landmarks(fr, hand_key)
            if h is None:
                arr.append(None)
            else:
                # Flatten to [x0, y0, z0, x1, y1, z1, ...]
                flat = []
                for lm in h:
                    flat.extend([lm['x'], lm['y'], lm['z']])
                arr.append(flat)
        return arr

    def set_hand_array(hand_key, smoothed_arr):
        for i, fr in enumerate(frames):
            if smoothed_arr[i] is not None:
                # Reconstruct list of dicts
                flat = smoothed_arr[i]
                original = _get_hand_landmarks(fr, hand_key)
                h = []
                for j in range(0, len(flat), 3):
                    visibility = None
                    if original is not None and j // 3 < len(original):
                        visibility = original[j // 3].get("visibility")
                    h.append({
                        "x": flat[j],
                        "y": flat[j+1],
                        "z": flat[j+2],
                        "visibility": visibility,
                    })

                _set_hand_landmarks(fr, hand_key, h)

    for hand_key in ["hand 1", "hand 2"]:
        data = get_hand_array(hand_key)
        # We can only smooth continuous segments
        # But for global smoothing, we might want to smooth across small gaps?
        # For now, let's just smooth existing data points.
        
        # Convert to numpy for easier handling, but handle Nones
        # Actually, let's just iterate and smooth locally
        
        smoothed_data = [d for d in data] # Copy
        
        L = len(data)
        half_win = window_size // 2
        
        for i in range(L):
            if data[i] is None:
                continue
                
            # Gather window
            window_vals = []
            for j in range(max(0, i - half_win), min(L, i + half_win + 1)):
                if data[j] is not None:
                    window_vals.append(data[j])
            
            if not window_vals:
                continue
                
            # Average
            window_vals = np.array(window_vals)
            avg = np.mean(window_vals, axis=0)
            smoothed_data[i] = avg.tolist()
            
        set_hand_array(hand_key, smoothed_data)
