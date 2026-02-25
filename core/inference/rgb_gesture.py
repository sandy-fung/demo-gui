"""RGB gesture inference engine (MediaPipe).

Extracted from ``gesture_hand_control_v5.py``.
MediaPipe is lazy-imported inside the class to avoid hard dependency
at module load time.
"""

import os
import time
from typing import Tuple

import cv2
import numpy as np


class MediaPipeGestureInference:
    """MediaPipe GestureRecognizer wrapper (lazy-imports mediapipe).

    Args:
        model_path: Path to ``.task`` model file.
        use_gpu: Try GPU delegate first, fall back to CPU.
    """

    def __init__(self, model_path: str, use_gpu: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Lazy import mediapipe
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp

        delegate = (
            python.BaseOptions.Delegate.GPU if use_gpu
            else python.BaseOptions.Delegate.CPU
        )

        try:
            options = vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_path), delegate=delegate,
                ),
                running_mode=vision.RunningMode.IMAGE,
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            print(f"[RGB_GEST] MediaPipe device: {'GPU' if use_gpu else 'CPU'}")
        except Exception as e:
            if use_gpu:
                print(f"[RGB_GEST] GPU failed ({e}), falling back to CPU")
                options = vision.GestureRecognizerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=python.BaseOptions.Delegate.CPU,
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                )
                self.recognizer = vision.GestureRecognizer.create_from_options(options)
            else:
                raise

        print(f"[RGB_GEST] MediaPipe model loaded: {model_path}")

    def predict(self, frame: np.ndarray) -> Tuple[str, float, float]:
        """Predict gesture from BGR frame.

        Returns:
            ``(gesture_name, confidence, elapsed_seconds)``
        """
        start = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb,
        )
        result = self.recognizer.recognize(mp_image)

        elapsed = time.perf_counter() - start

        if not result.gestures or not result.gestures[0]:
            return "none", 0.0, elapsed

        gesture = result.gestures[0][0]
        name = gesture.category_name.lower() if gesture.category_name else "none"
        return name, gesture.score, elapsed
