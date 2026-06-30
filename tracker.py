from collections import deque
from typing import Optional

import cv2
import numpy as np

PERSON_CLASS = 0
# COCO classes that proxy for forklifts before fine-tuning on your own footage.
# car=2, motorcycle=3, bus=5, truck=7 — swap in your custom class IDs once you have them.
DEFAULT_FORKLIFT_CLASSES: frozenset[int] = frozenset({2, 3, 5, 7})

_HISTORY_LEN = 5      # frames of velocity history to average over
_MIN_SPEED = 0.1      # m/s (or px/s without a transform) — skip near-stationary objects
TTC_THRESHOLD = 3.0   # seconds — flag if time-to-collision is below this
SAFETY_RADIUS = 1.5   # meters (or pixels) — also flag if objects are already this close


class ViewTransformer:
    """
    One-time calibration: maps pixel coordinates to real-world floor coordinates
    via a perspective homography.

    pixel_pts: (4, 2) array of pixel corners in order
               [bottom-left, top-left, top-right, bottom-right]
    real_width_m, real_length_m: physical size of that region in meters

    Measure a known rectangle on the floor, mark its four corners in pixel space,
    then pass cv2.getPerspectiveTransform the result. After that, every (x, y) pixel
    inside the quadrilateral converts to meters on the floor plane.
    """

    def __init__(self, pixel_pts: np.ndarray, real_width_m: float, real_length_m: float):
        self._pixel_pts = pixel_pts.astype(np.float32)
        target = np.array(
            [[0, real_length_m], [0, 0], [real_width_m, 0], [real_width_m, real_length_m]],
            dtype=np.float32,
        )
        self._M = cv2.getPerspectiveTransform(self._pixel_pts, target)

    def transform(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Return real-world (x, y) in meters, or None if outside the calibrated region."""
        if cv2.pointPolygonTest(self._pixel_pts, (float(point[0]), float(point[1])), False) < 0:
            return None
        pt = np.array(point, dtype=np.float32).reshape(1, 1, 2)
        return cv2.perspectiveTransform(pt, self._M)[0, 0]


class _KF:
    """4-state constant-velocity Kalman filter (x, y, vx, vy)."""

    def __init__(self, pos: np.ndarray):
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        self.P = np.eye(4) * 1000.0
        self.R = np.eye(2) * 10.0   # measurement noise; increase to trust detections less
        self.Q = np.diag([1.0, 1.0, 10.0, 10.0])  # process noise

    def step(self, z: np.ndarray) -> np.ndarray:
        """Predict + update in one call; return smoothed position."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]


class ObjectTracker:
    """
    Maintains per-object Kalman-filtered positions and rolling velocity vectors
    across frames. Consumes YOLO tracking results (model.track(), not model()).

    When a ViewTransformer is supplied, positions and velocities are in meters.
    Without one they stay in pixels — TTC will still work, but SAFETY_RADIUS and
    TTC_THRESHOLD will need to be tuned in pixel units.
    """

    def __init__(
        self,
        fps: float,
        view_transformer: Optional[ViewTransformer] = None,
        forklift_classes: frozenset[int] = DEFAULT_FORKLIFT_CLASSES,
    ):
        self.fps = fps
        self._vt = view_transformer
        self.forklift_classes = forklift_classes

        self._kf: dict[int, _KF] = {}
        self._last_pos: dict[int, np.ndarray] = {}
        self._cls: dict[int, int] = {}
        self._vel_history: dict[int, deque] = {}

        # public — read these after each update()
        self.positions: dict[int, np.ndarray] = {}   # smoothed position per ID
        self.velocities: dict[int, np.ndarray] = {}  # avg velocity per ID (units/frame)

    def update(self, results) -> None:
        seen: set[int] = set()

        for box in results[0].boxes:
            if box.id is None:
                continue
            obj_id = int(box.id)
            cls = int(box.cls[0])
            if cls != PERSON_CLASS and cls not in self.forklift_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            pixel_pos = np.array([(x1 + x2) / 2, float(y2)])  # bottom-center = floor contact

            if self._vt is not None:
                world_pos = self._vt.transform(pixel_pos)
                if world_pos is None:
                    continue
            else:
                world_pos = pixel_pos

            if obj_id not in self._kf:
                self._kf[obj_id] = _KF(world_pos)
                self._vel_history[obj_id] = deque(maxlen=_HISTORY_LEN)

            smoothed = self._kf[obj_id].step(world_pos)
            self._cls[obj_id] = cls
            seen.add(obj_id)

            if obj_id in self._last_pos:
                self._vel_history[obj_id].append(smoothed - self._last_pos[obj_id])

            self._last_pos[obj_id] = smoothed.copy()
            self.positions[obj_id] = smoothed

            if self._vel_history[obj_id]:
                self.velocities[obj_id] = np.mean(self._vel_history[obj_id], axis=0)

        for stale_id in [i for i in self._kf if i not in seen]:
            for d in (self._kf, self._last_pos, self._cls, self._vel_history,
                      self.positions, self.velocities):
                d.pop(stale_id, None)

    def is_forklift(self, obj_id: int) -> bool:
        return self._cls.get(obj_id) in self.forklift_classes

    def is_person(self, obj_id: int) -> bool:
        return self._cls.get(obj_id) == PERSON_CLASS


def _time_to_closest_approach(
    pos_a: np.ndarray, vel_a: np.ndarray,
    pos_b: np.ndarray, vel_b: np.ndarray,
) -> tuple[float, float]:
    """
    Returns (t_frames, min_separation).
    t_frames is when the two objects would be closest assuming constant velocity.
    Negative t_frames means they are already diverging.
    """
    rel_pos = pos_a - pos_b
    rel_vel = vel_a - vel_b
    speed_sq = float(np.dot(rel_vel, rel_vel))
    if speed_sq < 1e-9:
        return float("inf"), float(np.linalg.norm(rel_pos))
    t = -float(np.dot(rel_pos, rel_vel)) / speed_sq
    min_sep = float(np.linalg.norm(rel_pos + rel_vel * t))
    return t, min_sep


class CollisionDetector:
    """
    Checks all (forklift, person) pairs each frame for imminent collision.

    A pair is flagged when either:
    - time-to-closest-approach < ttc_threshold AND predicted closest distance < safety_radius
    - current separation < safety_radius (already dangerously close regardless of motion)
    """

    def __init__(
        self,
        fps: float,
        ttc_threshold: float = TTC_THRESHOLD,
        safety_radius: float = SAFETY_RADIUS,
    ):
        self.fps = fps
        self.ttc_threshold = ttc_threshold
        self.safety_radius = safety_radius

    def check(self, tracker: ObjectTracker) -> list[tuple[int, int]]:
        """Return list of (forklift_id, person_id) pairs that are flagged as dangerous."""
        forklifts = [i for i in tracker.positions if tracker.is_forklift(i) and i in tracker.velocities]
        persons = [i for i in tracker.positions if tracker.is_person(i) and i in tracker.velocities]

        flagged = []
        for fid in forklifts:
            fvel = tracker.velocities[fid]
            if float(np.linalg.norm(fvel)) * self.fps < _MIN_SPEED:
                continue
            for pid in persons:
                current_sep = float(np.linalg.norm(tracker.positions[fid] - tracker.positions[pid]))
                if current_sep < self.safety_radius:
                    flagged.append((fid, pid))
                    continue

                t_frames, min_sep = _time_to_closest_approach(
                    tracker.positions[fid], fvel,
                    tracker.positions[pid], tracker.velocities[pid],
                )
                ttc_seconds = t_frames / self.fps
                if 0 < ttc_seconds < self.ttc_threshold and min_sep < self.safety_radius:
                    flagged.append((fid, pid))

        return flagged
