"""GUI output mode for gesture demo — DVS + RGB side-by-side display.

Shows camera preview with gesture icon overlay. No HUD clutter.
"""

from pathlib import Path

import cv2
import numpy as np

from app.core.demo import OutputMode
from app.core.display import resize_to_height
from app.core.inference.common import BATTLE_MAP

# ---------------------------------------------------------------------------
# Icon loader / overlay helpers (shared by phys_dvs / phys_rgb outputs)
# ---------------------------------------------------------------------------

_ICON_DIR = Path(__file__).parent / "picture"
_ICON_CACHE: dict = {}


def _load_icon(name: str, height: int) -> np.ndarray | None:
    """Load gesture PNG as BGRA, resize to target height. Cached."""
    key = f"{name}_{height}"
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    path = _ICON_DIR / f"{name}.png"
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
    h_orig, w_orig = img.shape[:2]
    new_w = int(w_orig * height / h_orig)
    img = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)
    _ICON_CACHE[key] = img
    return img


def _overlay_icon(panel: np.ndarray, icon: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA icon onto BGR panel at (x, y)."""
    ih, iw = icon.shape[:2]
    ph, pw = panel.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + iw, pw), min(y + ih, ph)
    if x1 >= x2 or y1 >= y2:
        return
    ix1, iy1 = x1 - x, y1 - y
    ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)
    roi = panel[y1:y2, x1:x2]
    alpha = icon[iy1:iy2, ix1:ix2, 3:4].astype(np.float32) / 255.0
    bgr = icon[iy1:iy2, ix1:ix2, :3]
    panel[y1:y2, x1:x2] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)


def _resolve_gesture(stable: str, game_mode: str) -> str:
    """Determine which gesture icon to show."""
    if stable == "none":
        return "none"
    if game_mode == "battle":
        return BATTLE_MAP.get(stable, stable)
    return stable


def _draw_gesture_icon(panel: np.ndarray, gesture: str,
                       icon_h: int = 120) -> None:
    """Overlay gesture icon at bottom-right corner of panel."""
    if gesture == "none":
        return
    icon = _load_icon(gesture, icon_h)
    if icon is None:
        return
    ph, pw = panel.shape[:2]
    x = pw - icon.shape[1] - 10
    y = ph - icon.shape[0] - 10
    _overlay_icon(panel, icon, x, y)


class GestureGUIOutput(OutputMode):
    """Pure display mode — DVS and RGB gesture recognition side by side."""

    def __init__(self, gesture_demo):
        self._demo = gesture_demo
        self._result = None

    def activate(self) -> None:
        self._demo.tracking_enabled = True

    def deactivate(self) -> None:
        pass

    def process(self, result) -> None:
        self._result = result

    def render(self) -> np.ndarray:
        result = self._result
        if result is None:
            return np.zeros((480, 800, 3), dtype=np.uint8)

        dvs_panel = self._render_dvs_panel(result)
        rgb_panel = self._render_rgb_panel(result)

        # Match heights
        target_h = max(dvs_panel.shape[0], rgb_panel.shape[0])
        dvs_panel = resize_to_height(dvs_panel, target_h)
        rgb_panel = resize_to_height(rgb_panel, target_h)

        return np.hstack([dvs_panel, rgb_panel])

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_dvs_panel(result) -> np.ndarray:
        """Render DVS panel — camera image + gesture icon."""
        if result.dvs_display is not None:
            panel = result.dvs_display.copy()
        else:
            panel = np.zeros((480, 492, 3), dtype=np.uint8)
            cv2.putText(panel, "DVS: no model", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            return panel

        target = _resolve_gesture(result.dvs_stable, result.game_mode)
        _draw_gesture_icon(panel, target)
        return panel

    @staticmethod
    def _render_rgb_panel(result) -> np.ndarray:
        """Render RGB panel — camera image + gesture icon."""
        if result.rgb_frame is not None:
            panel = result.rgb_frame.copy()
        else:
            panel = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(panel, "RGB: no model", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            return panel

        target = _resolve_gesture(result.rgb_stable, result.game_mode)
        _draw_gesture_icon(panel, target)
        return panel
