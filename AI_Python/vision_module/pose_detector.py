import os
# Matplotlib dùng nội bộ bởi MediaPipe; Agg giảm phụ thuộc GUI (vẫn cần DLL hợp lệ)
os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import mediapipe as mp
import math
import numpy as np
import time
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.lmList = []
        self.prev_points = {}
        self.smooth_alpha = 0.3  # Mượt vừa phải, phản ứng nhanh vừa đủ
        self.status_history = deque(maxlen=30)  # Lấy trạng thái nhiều khung để ổn định

        # Biến cho NGÃ
        self.fall_counter = 0
        self.velocity_history = deque(maxlen=5)
        self.last_y = None

        # Biến cho NGỒI LÂU
        self.sitting_start_time = None
        self.SITTING_LIMIT = 60  # 15 phút, chỉnh test nhanh xuống 10 giây nếu cần

        def _fenv(k, dft):
            try:
                v = os.environ.get(k, "")
                return float(v) if v != "" else dft
            except ValueError:
                return dft
        # Tinh tại máy: VHU_FALL_DEFAULT_* (tư thế thường), VHU_FALL_SITSTOOP_* (ngồi + cúi)
        self._fall_wh_sit_stoop = _fenv("VHU_FALL_SITSTOOP_WH", 0.62)
        self._fall_vel_sit_stoop = _fenv("VHU_FALL_SITSTOOP_VEL", 0.62)
        self._fall_nose_sit_stoop = _fenv("VHU_FALL_SITSTOOP_NOSE", 0.12)
        self._fall_wh_default = _fenv("VHU_FALL_DEFAULT_WH", 0.52)
        self._fall_vel_default = _fenv("VHU_FALL_DEFAULT_VEL", 0.45)
        self._fall_nose_default = _fenv("VHU_FALL_DEFAULT_NOSE", 0.1)
        def _e_yes(k):
            return os.environ.get(k, "").strip().lower() in ("1", "true", "yes")
        self._sit_stoop_allow_nose = _e_yes("VHU_FALL_SITSTOOP_ALLOW_NOSE")
        self._use_world = not _e_yes("VHU_POSE_DISABLE_WORLD")
        try:
            self._world_torso_blend = float(os.environ.get("VHU_WORLD_TORSO_BLEND", "0.58"))
        except ValueError:
            self._world_torso_blend = 0.58
        self._world_torso_blend = max(0.0, min(1.0, self._world_torso_blend))
        self.world_pts = {}

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        self.world_pts = {}
        if self._use_world and self.results.pose_world_landmarks:
            for i, lm in enumerate(self.results.pose_world_landmarks.landmark):
                self.world_pts[i] = np.array([lm.x, lm.y, lm.z], dtype=np.float64)
        if self.results.pose_landmarks and draw:
            mp_drawing.draw_landmarks(
                img, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )
        return img

    def getPosition(self, img, draw=False):
        self.lmList = []
        if not self.results or not self.results.pose_landmarks:
            return self.lmList
        h, w, _ = img.shape
        for i, lm in enumerate(self.results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if i in self.prev_points:
                px, py = self.prev_points[i]
                cx = int(self.smooth_alpha * cx + (1 - self.smooth_alpha) * px)
                cy = int(self.smooth_alpha * cy + (1 - self.smooth_alpha) * py)
            self.prev_points[i] = (cx, cy)
            self.lmList.append([i, cx, cy, lm.visibility])
        return self.lmList

    def get_angle(self, p1, p2, p3):
        try:
            a, b, c = np.array(p1[:2]), np.array(p2[:2]), np.array(p3[:2])
            ba, bc = a - b, c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        except:
            return 0

    def _body_bbox_metrics(self, pts):
        y_coords = [p[1] for p in pts.values()]
        x_coords = [p[0] for p in pts.values()]
        body_h = max(y_coords) - min(y_coords)
        body_w = max(x_coords) - min(x_coords)
        wh_ratio = body_w / max(body_h, 1e-6)
        return body_h, body_w, wh_ratio

    def _torso_inclination_deg(self, pts):
        """2D ảnh: góc hông→vai so với phương 'lên' của khung hình (phụ thuộc góc máy)."""
        if not all(i in pts for i in (11, 12, 23, 24)):
            return 0.0
        sx = (pts[11][0] + pts[12][0]) / 2
        sy = (pts[11][1] + pts[12][1]) / 2
        hx = (pts[23][0] + pts[24][0]) / 2
        hy = (pts[23][1] + pts[24][1]) / 2
        vx, vy = sx - hx, sy - hy
        n = math.hypot(vx, vy)
        if n < 1e-6:
            return 0.0
        vx, vy = vx / n, vy / n
        up_x, up_y = 0.0, -1.0
        cosang = max(-1.0, min(1.0, vx * up_x + vy * up_y))
        return math.degrees(math.acos(cosang))

    def _torso_inclination_deg_world(self):
        """
        3D MediaPipe (trục Y hướng lên trong không gian world của model).
        Ít phụ thuộc góc máy hơn so với 2D — phù hợp té/nằm vs đứng.
        """
        w = self.world_pts
        if not w or not all(i in w for i in (11, 12, 23, 24)):
            return None
        sx = (w[11][0] + w[12][0]) / 2
        sy = (w[11][1] + w[12][1]) / 2
        sz = (w[11][2] + w[12][2]) / 2
        hx = (w[23][0] + w[24][0]) / 2
        hy = (w[23][1] + w[24][1]) / 2
        hz = (w[23][2] + w[24][2]) / 2
        v = np.array([sx - hx, sy - hy, sz - hz], dtype=np.float64)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return None
        v /= n
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cosang = float(np.clip(np.dot(v, up), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _angle_at_b_world(self, idx_a, idx_b, idx_c):
        w = self.world_pts
        if not w or not all(i in w for i in (idx_a, idx_b, idx_c)):
            return None
        a, b, c = w[idx_a], w[idx_b], w[idx_c]
        ba, bc = a - b, c - b
        na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if na < 1e-6 or nc < 1e-6:
            return None
        cosang = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _landmarks_visible(self, pts, indices, min_vis=0.5):
        return all(i in pts and pts[i][2] >= min_vis for i in indices)

    def _effective_torso_deg(self, pts, torso_2d):
        """Kết hợp 2D + world 3D để giảm nhầm khi máy lệch góc."""
        tw = self._torso_inclination_deg_world()
        if tw is None or not self._landmarks_visible(pts, (11, 12, 23, 24), 0.45):
            return torso_2d, False
        b = self._world_torso_blend
        eff = b * tw + (1.0 - b) * torso_2d
        return eff, True

    def is_sitting_pose(self, pts, sh_dist):
        """Ngồi: 2D + (nếu có) góc gối 3D — ổn định hơn khi camera không chính diện."""
        sit_2d = self._is_sitting_pose_2d(pts, sh_dist)
        sit_w = self._is_sitting_pose_world(pts)
        if sit_w is None:
            return sit_2d
        return sit_2d or sit_w

    def _ankle_mid_y(self, pts):
        ys = []
        for i in (27, 28):
            if i in pts and pts[i][2] > 0.45:
                ys.append(pts[i][1])
        if not ys:
            return None
        return sum(ys) / len(ys)

    def _is_sitting_pose_2d(self, pts, sh_dist):
        """Ngồi ghế (heuristic 2D)."""
        if not all(i in pts for i in (23, 24, 25, 26)):
            return False
        _, _, wh_ratio = self._body_bbox_metrics(pts)
        if wh_ratio >= 0.56:
            return False
        hip_knee_l = abs(pts[23][1] - pts[25][1])
        hip_knee_r = abs(pts[24][1] - pts[26][1])
        if min(hip_knee_l, hip_knee_r) >= sh_dist * 1.22:
            return False
        knee_y = min(pts[25][1], pts[26][1])
        hip_y = (pts[23][1] + pts[24][1]) / 2
        if knee_y < hip_y + sh_dist * 0.06:
            return False
        return True

    def _is_sitting_pose_world(self, pts=None):
        """
        Gối gập rõ (góc hông–gối–cổ chân) + thân chưa nằm phẳng trong world.
        pts: dict 2D để kiểm visibility; nếu None vẫn thử world.
        """
        if not self.world_pts:
            return None
        tw = self._torso_inclination_deg_world()
        if tw is not None and tw >= 72:
            return False
        ang_l = self._angle_at_b_world(23, 25, 27)
        ang_r = self._angle_at_b_world(24, 26, 28)
        bends = [a for a in (ang_l, ang_r) if a is not None]
        if not bends:
            return None
        best = min(bends)
        if pts is not None:
            if not (
                self._landmarks_visible(pts, (23, 25, 27), 0.42)
                or self._landmarks_visible(pts, (24, 26, 28), 0.42)
            ):
                return None
        return 55 <= best <= 155

    def check_sitting_duration(self, is_sitting):
        if is_sitting:
            if self.sitting_start_time is None:
                self.sitting_start_time = time.time()
            elapsed = time.time() - self.sitting_start_time
            return elapsed > self.SITTING_LIMIT, int(elapsed)
        else:
            self.sitting_start_time = None
            return False, 0

    def is_falling_advanced(
        self,
        pts,
        sh_dist,
        is_stooping=False,
        is_sitting=False,
        wh_ratio=0.0,
        torso_deg=0.0,
        torso_world=None,
    ):
        """
        Phân biệt té ngã vs ngồi khom:
        - Ngồi ghế ổn định: hông cao hơn mắt cá rõ rệt + torso chưa nằm ngang → không tăng fall chỉ vì cúi người / nhiễu.
        - Té / nằm: tỷ lệ rộng/cao, torso gần ngang, hoặc hông sụp gần mức chân + vận tốc mũi đột biến.
        """
        if 0 in pts and 23 in pts and 24 in pts:
            nose_y = pts[0][1]
            if self.last_y is None:
                self.last_y = nose_y
            vel = nose_y - self.last_y
            self.velocity_history.append(vel)
            self.last_y = nose_y
            avg_vel = sum(self.velocity_history) / len(self.velocity_history)

            hip_y = (pts[23][1] + pts[24][1]) / 2
            ankle_y = self._ankle_mid_y(pts)

            sit_stoop = is_sitting and is_stooping
            if sit_stoop:
                wh_need = self._fall_wh_sit_stoop
                vel_mul = self._fall_vel_sit_stoop
                nose_margin = self._fall_nose_sit_stoop * sh_dist
            else:
                wh_need = self._fall_wh_default
                vel_mul = self._fall_vel_default
                nose_margin = self._fall_nose_default * sh_dist

            lying_like = wh_ratio >= wh_need
            if torso_world is not None and torso_world >= 62:
                lying_like = True
            nose_drops = nose_y > (hip_y + nose_margin)
            rule_nose = nose_drops and lying_like
            if is_stooping and not lying_like:
                rule_nose = False
            if sit_stoop and not self._sit_stoop_allow_nose:
                rule_nose = False

            stable_chair_sit = False
            if ankle_y is not None and is_sitting:
                stable_chair_sit = (
                    hip_y < ankle_y - 0.11 * sh_dist and torso_deg < 58
                )

            collapsed_to_floor = False
            if ankle_y is not None:
                collapsed_to_floor = hip_y > ankle_y - 0.09 * sh_dist

            torso_horizontal = torso_deg >= 60 or (
                torso_world is not None and torso_world >= 58
            )
            strong_pose_fall = lying_like and (
                torso_horizontal or (collapsed_to_floor and (torso_deg >= 48 or (torso_world is not None and torso_world >= 52)))
            )

            rule_vel = avg_vel > sh_dist * vel_mul
            strong_vel = avg_vel > sh_dist * vel_mul * 1.22

            if stable_chair_sit and is_stooping:
                if strong_pose_fall or (strong_vel and lying_like):
                    self.fall_counter += 1
                elif rule_vel or rule_nose:
                    self.fall_counter = max(0, self.fall_counter - 2)
                else:
                    self.fall_counter = max(0, self.fall_counter - 1)
            else:
                trigger = rule_nose or (
                    rule_vel and (lying_like or torso_deg >= 52 or collapsed_to_floor)
                ) or (strong_vel and not sit_stoop)
                if not lying_like and collapsed_to_floor and (
                    torso_deg >= 64 or (torso_world is not None and torso_world >= 60)
                ):
                    trigger = True
                if trigger:
                    self.fall_counter += 1
                else:
                    self.fall_counter = max(0, self.fall_counter - 1)

            return self.fall_counter > 2
        return False

    def is_stooping_strict(self, pts, wh_ratio=0.0, torso_world=None):
        """Cúi / khom: ưu tiên góc 3D vai–hông–gối khi có world (ổn định góc máy)."""
        tw = torso_world if torso_world is not None else self._torso_inclination_deg_world()
        if tw is not None and tw >= 68:
            return False
        if wh_ratio >= 0.54:
            return False
        ang_w_l = self._angle_at_b_world(11, 23, 25)
        ang_w_r = self._angle_at_b_world(12, 24, 26)
        if ang_w_l is not None or ang_w_r is not None:
            aw = min([a for a in (ang_w_l, ang_w_r) if a is not None])
            if self._landmarks_visible(pts, (11, 23, 25), 0.42) or self._landmarks_visible(
                pts, (12, 24, 26), 0.42
            ):
                return aw < 158 and pts.get(11, (0, 0, 0))[2] > 0.45
        if 11 in pts and 23 in pts:
            p_knee = pts[25] if 25 in pts and pts[25][2] > 0.5 else (pts[23][0], pts[23][1] + 100, 0)
            angle = self.get_angle(pts[11], pts[23], p_knee)
            return angle < 162 and pts[11][2] > 0.5
        return False

    def is_waving(self, pts, sh_dist):
        for wrist_id in [15,16]:
            if wrist_id in pts and 0 in pts:
                wrist = pts[wrist_id]
                nose = pts[0]
                if wrist[1] < nose[1] and abs(wrist[0]-nose[0]) > (sh_dist * 0.45):
                    return True
        return False

    def detect_posture(self, frame=None):
        if not self.lmList or len(self.lmList) < 24:
            return "🔍 Đang quét hệ thống...", (200, 200, 200), 0, {}

        pts = {it[0]: (it[1], it[2], it[3]) for it in self.lmList}

        sh_dist = math.hypot(pts[11][0]-pts[12][0], pts[11][1]-pts[12][1]) if 11 in pts and 12 in pts else 100

        _, _, wh_ratio = self._body_bbox_metrics(pts)
        torso_2d = self._torso_inclination_deg(pts)
        torso_world = self._torso_inclination_deg_world()
        torso_deg, used_world = self._effective_torso_deg(pts, torso_2d)
        is_sitting = self.is_sitting_pose(pts, sh_dist)
        too_long, sitting_seconds = self.check_sitting_duration(is_sitting)

        is_stooping = self.is_stooping_strict(pts, wh_ratio=wh_ratio, torso_world=torso_world)
        is_falling = self.is_falling_advanced(
            pts, sh_dist, is_stooping=is_stooping, is_sitting=is_sitting,
            wh_ratio=wh_ratio, torso_deg=torso_deg, torso_world=torso_world,
        )

        shoulder_lean = 11 in pts and 12 in pts and abs(pts[11][1]-pts[12][1]) > (sh_dist * 0.25)

        if is_falling:
            status, color = "🚨 NGUY HIỂM: NGÃ", (0, 0, 255)
            self.sitting_start_time = None

        elif any(pts[i][1] < pts[0][1] for i in [15, 16] if i in pts and pts[i][2] > 0.5):
            status, color = "🆘 CẦN HỖ TRỢ GẤP", (0, 0, 255)

        elif is_stooping:
            if is_sitting:
                status, color = "⚠️ NGỒI KHOM LƯNG", (0, 165, 255)
            else:
                status, color = "🚨 ĐI KHOM NGUY HIỂM", (0, 69, 255)

        elif shoulder_lean:
            status, color = "⚖️ TƯ THẾ LỆCH VAI", (255, 0, 255)

        elif too_long:
            status, color = "⚠️ NỘI NGỒI QUÁ LÂU", (0, 120, 255)

        elif self.is_waving(pts, sh_dist):
            status, color = "👋 ĐANG CHÀO ROBOT", (0, 255, 0)

        elif any(i in pts and abs(pts[i][1] - pts[0][1]) < (sh_dist * 0.2) for i in [15, 16]):
            status, color = "😫 NỘI THẤY MỆT Ư?", (255, 165, 0)

        else:
            if is_sitting:
                status, color = "🧘 ĐANG NGỒI NGHỈ", (255, 255, 255)
            else:
                status, color = "✅ TRẠNG THÁI TỐT", (255, 255, 255)

        self.status_history.append(status)
        final_status = max(set(self.status_history), key=self.status_history.count)

        if is_falling:
            pose_type = "fall"
        elif is_stooping:
            pose_type = "bad_posture"
        elif is_sitting:
            pose_type = "sitting"
        else:
            pose_type = "standing"
        pose_ctx = {
            "is_sitting": is_sitting,
            "is_too_long": too_long,
            "is_stooping": is_stooping,
            "is_falling": is_falling,
            "pose_type": pose_type,
            "sitting_time": int(sitting_seconds),
            "shoulder_lean": shoulder_lean,
            "is_waving": self.is_waving(pts, sh_dist),
            "sitting_seconds": sitting_seconds,
            "status": final_status,
            "wh_ratio": round(wh_ratio, 3),
            "torso_deg": round(torso_deg, 1),
            "torso_deg_2d": round(torso_2d, 1),
            "torso_deg_world": None if torso_world is None else round(torso_world, 1),
            "used_world_lm": used_world,
        }

        return final_status, color, sitting_seconds, pose_ctx
