"""Physical DVS output — DVS tracking drives the robotic arm.

When active, DVSReaderThread pushes tracking results to CommandBridge
at ~200fps via set_bridge().  The thread itself never stops — only the
bridge connection is toggled on activate/deactivate.
Display shows single DVS canvas + arm status.
"""

import cv2
import numpy as np

from app.core.demo import OutputMode
from app.core.display import draw_hint_bar, draw_paused_overlay


class TrackingPhysDVSOutput(OutputMode):
    """DVS tracking → arm drawing via CommandBridge."""

    def __init__(self, tracking_demo, bridge, arm_thread):
        self._demo = tracking_demo
        self._bridge = bridge
        self._arm = arm_thread
        self._result = None

    def activate(self) -> None:
        """Enable bridge on DVSReaderThread for 200fps arm push."""
        self._bridge.put(False, 0.5, 0.5)
        # Disable tracking BEFORE connecting bridge to prevent
        # leaked writing commands (~200fps race window).
        self._demo.tracking_enabled = False
        self._demo.dvs_reader.set_bridge(self._bridge)
        print("[PHYS_DVS] Activated — PAUSED (press Space to start tracking)")

    def deactivate(self) -> None:
        """Disconnect bridge from DVSReaderThread (thread keeps running)."""
        if self._demo.dvs_reader:
            self._demo.dvs_reader.set_bridge(None)
        self._bridge.clear()
        print("[PHYS_DVS] Deactivated")

    def on_tracking_changed(self, enabled: bool) -> None:
        pass  # dvs_reader.tracking_enabled is set by demo.tracking_enabled setter

    def process(self, result) -> None:
        self._result = result

    def render(self) -> np.ndarray:
        # Single DVS canvas + arm status (always use demo-level canvas)
        canvas = self._demo.render_dvs_canvas()

        # Arm status hint bar
        canvas = canvas.copy()
        if self._arm:
            pending = self._bridge.pending if self._bridge else 0
            move_c = self._arm.move_count
            fail_c = self._arm.fail_count
            ready = self._arm.is_ready.is_set() if self._arm.is_ready else False
            status = (f"ARM: {'RDY' if ready else 'INIT'} | "
                      f"moves: {move_c} | fails: {fail_c} | queue: {pending}")
            color = (0, 200, 0) if ready else (0, 140, 255)
            hints = [(status, color)]
            if self._arm.error:
                hints.insert(0, (f"ERR: {self._arm.error}", (0, 0, 255)))
            draw_hint_bar(canvas, hints)

        cv2.putText(canvas, "Physical DVS", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)

        if not self._demo.tracking_enabled:
            draw_paused_overlay(canvas)

        return canvas
