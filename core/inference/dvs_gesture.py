"""DVS gesture inference engine.

Extracted from ``realtime_dvs_v3.py`` (create_model + DVSGestureInference)
and ``time_surface.py`` (TimeSurfaceProcessor).
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from app.config import DVS_WIDTH, DVS_HEIGHT


# ============================================================
# Time Surface Processor
# ============================================================

_TS_VALID_MODES = ("fixed", "diff", "ema")


class TimeSurfaceProcessor:
    """Per-pixel time surface blend for DVS event frames.

    Temporally consistent events (hand gestures) are preserved while
    random single-frame noise is suppressed via exponential decay.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        tau: Decay time constant in seconds.
        mode: Event detection mode (``"fixed"``, ``"diff"``, ``"ema"``).
        event_tol: Threshold for event detection.
        bg_value: Fixed background value (``"fixed"`` mode only).
        avg_alpha: EMA smoothing factor (``"ema"`` mode only).
    """

    def __init__(
        self,
        height: int,
        width: int,
        tau: float = 0.02,
        mode: str = "fixed",
        event_tol: float = 20.0,
        bg_value: int = 128,
        avg_alpha: float = 0.05,
    ):
        if mode not in _TS_VALID_MODES:
            raise ValueError(f"mode must be one of {_TS_VALID_MODES}, got '{mode}'")

        self.height = height
        self.width = width
        self.tau = tau
        self.mode = mode
        self.event_tol = event_tol
        self.bg_value = bg_value
        self.avg_alpha = avg_alpha

        self.last_event_time = np.full((height, width), -np.inf, dtype=np.float64)
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_avg: Optional[np.ndarray] = None

    def _detect_events(self, frame_f: np.ndarray) -> np.ndarray:
        if self.mode == "fixed":
            return np.abs(frame_f - self.bg_value) > self.event_tol

        if self.mode == "diff":
            if self.prev_frame is None:
                self.prev_frame = frame_f.copy()
                return np.zeros((self.height, self.width), dtype=bool)
            mask = np.abs(frame_f - self.prev_frame) > self.event_tol
            self.prev_frame = frame_f.copy()
            return mask

        # mode == "ema"
        if self.prev_avg is None:
            self.prev_avg = frame_f.copy()
            return np.zeros((self.height, self.width), dtype=bool)
        mask = np.abs(frame_f - self.prev_avg) > self.event_tol
        self.prev_avg += self.avg_alpha * (frame_f - self.prev_avg)
        return mask

    def process(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Update state and return time-surface blended frame (uint8)."""
        frame_f = frame.astype(np.float32)
        event_mask = self._detect_events(frame_f)
        self.last_event_time[event_mask] = timestamp

        dt = timestamp - self.last_event_time
        ts = np.exp(-dt / self.tau)
        ts[self.last_event_time == -np.inf] = 0.0

        return (frame_f * ts).clip(0, 255).astype(np.uint8)

    def reset(self):
        """Reset internal state."""
        self.last_event_time.fill(-np.inf)
        self.prev_frame = None
        self.prev_avg = None


# ============================================================
# Model Creation (matches training pipeline)
# ============================================================

def create_model(num_classes: int, in_channels: int = 1,
                 model_name: str = "mobilenet_v2") -> nn.Module:
    """Create MobileNet model with custom input channels."""
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        clf_in = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(clf_in, 256), nn.Hardswish(),
            nn.Dropout(p=0.2), nn.Linear(256, num_classes),
        )
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        clf_in = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(clf_in, 256), nn.Hardswish(),
            nn.Dropout(p=0.2), nn.Linear(256, num_classes),
        )
    else:  # mobilenet_v2 (default)
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # Modify first conv layer for non-3 channel input
    if in_channels != 3:
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

    return model


# ============================================================
# DVS Gesture Inference Engine
# ============================================================

class DVSGestureInference:
    """Gesture inference from DVS event frames.

    Loads a checkpoint, builds a MobileNet model, and optionally
    initializes TensorRT or FP16 acceleration.

    The checkpoint may contain a ``config`` dict with ``time_surface``
    parameters — if so, a :class:`TimeSurfaceProcessor` is auto-created
    and used in :meth:`predict`.
    """

    def __init__(
        self,
        model_path: str,
        use_fp16: bool = True,
        use_tensorrt: bool = False,
        rebuild_engine: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model_path = Path(model_path)
        self.image_size = image_size
        self._use_trt_engine = False

        print(f"[DVS_GEST] Device: {self.device}, FP16: {self.use_fp16}")
        self._load_model(model_path)

        if use_tensorrt and self.device.type == "cuda":
            self._setup_tensorrt(rebuild_engine)

        # Transform (single channel normalization)
        transform_list = [transforms.ToPILImage()]
        if image_size is not None:
            transform_list.append(transforms.Resize(image_size))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.transform = transforms.Compose(transform_list)

        # Auto-detect time surface from checkpoint
        self.ts_processor: Optional[TimeSurfaceProcessor] = None
        self._ts_start_time: float = 0.0
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})
        if ckpt_config.get("time_surface", False):
            self.ts_processor = TimeSurfaceProcessor(
                DVS_HEIGHT, DVS_WIDTH,
                tau=ckpt_config.get("ts_tau", 0.02),
                mode=ckpt_config.get("ts_mode", "fixed"),
                event_tol=ckpt_config.get("ts_event_tol", 20.0),
                avg_alpha=ckpt_config.get("ts_avg_alpha", 0.05),
            )
            self._ts_start_time = time.perf_counter()
            print(f"[DVS_GEST] Time Surface: ON (tau={self.ts_processor.tau}, "
                  f"mode={self.ts_processor.mode})")
        else:
            print("[DVS_GEST] Time Surface: OFF")

        self._warmup()
        print("[DVS_GEST] Model ready!")

    def _load_model(self, model_path: str):
        print(f"[DVS_GEST] Loading: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.classes: list = checkpoint.get("classes", [])
        self.in_channels: int = checkpoint.get("in_channels", 1)
        self.model_name: str = checkpoint.get("model_name", "mobilenet_v2")
        num_classes = len(self.classes)

        print(f"[DVS_GEST] Classes: {self.classes}, Model: {self.model_name}")

        self.model = create_model(num_classes, in_channels=self.in_channels,
                                  model_name=self.model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()

    def _setup_tensorrt(self, rebuild_engine: bool = False):
        h, w = self.image_size if self.image_size else (DVS_HEIGHT, DVS_WIDTH)

        model_dir = self.model_path.parent
        precision_suffix = "_fp16" if self.use_fp16 else "_fp32"
        self.onnx_path = model_dir / f"gesture_model{precision_suffix}.onnx"
        self.engine_path = model_dir / f"gesture_model{precision_suffix}.engine"

        # Method 1: torch_tensorrt
        try:
            import torch_tensorrt
            self.model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input(shape=[1, self.in_channels, h, w])],
                enabled_precisions={torch.float16} if self.use_fp16 else {torch.float32},
            )
            self._use_trt_engine = True
            print("[DVS_GEST] TensorRT: Enabled (torch_tensorrt)")
            return
        except ImportError:
            print("[DVS_GEST] torch_tensorrt not found, trying native TensorRT...")
        except Exception as e:
            print(f"[DVS_GEST] torch_tensorrt failed: {e}")

        # Method 2: native TensorRT via ONNX
        try:
            import tensorrt as trt

            if not rebuild_engine and self.engine_path.exists():
                print(f"[DVS_GEST] Loading cached engine: {self.engine_path}")
                self.trt_engine = self._load_trt_engine(str(self.engine_path))
                if self.trt_engine:
                    self._setup_trt_inference(h, w)
                    self._use_trt_engine = True
                    print("[DVS_GEST] TensorRT: Enabled (cached engine)")
                    return

            if rebuild_engine or not self.onnx_path.exists():
                self._export_onnx(h, w)

            self.trt_engine = self._build_trt_engine(str(self.onnx_path), h, w)
            if self.trt_engine:
                self._save_trt_engine(str(self.engine_path))
                self._setup_trt_inference(h, w)
                self._use_trt_engine = True
                print("[DVS_GEST] TensorRT: Enabled (native via ONNX)")
                return

        except ImportError:
            print("[DVS_GEST] TensorRT: Not available")
        except Exception as e:
            print(f"[DVS_GEST] TensorRT: Failed ({e})")

    def _export_onnx(self, h: int, w: int):
        import copy
        dummy_input = torch.randn(1, self.in_channels, h, w, device=self.device)
        if self.use_fp16:
            model_for_export = copy.deepcopy(self.model).float()
            dummy_input = dummy_input.float()
        else:
            model_for_export = self.model

        torch.onnx.export(
            model_for_export, dummy_input, str(self.onnx_path),
            input_names=["dvs_input"], output_names=["gesture_output"],
            opset_version=11,
        )
        print(f"[DVS_GEST] ONNX exported to {self.onnx_path}")
        if self.use_fp16:
            del model_for_export

    def _load_trt_engine(self, engine_path: str):
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as f:
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"[DVS_GEST] Failed to load engine: {e}")
            return None

    def _save_trt_engine(self, engine_path: str):
        try:
            serialized = self.trt_engine.serialize()
            with open(engine_path, "wb") as f:
                f.write(serialized)
        except Exception as e:
            print(f"[DVS_GEST] Failed to save engine: {e}")

    def _build_trt_engine(self, onnx_path: str, h: int, w: int):
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(f"[DVS_GEST] ONNX parse error: {parser.get_error(error)}")
                    return None

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if self.use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            input_shape = (1, self.in_channels, h, w)
            profile.set_shape(input_name, input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)

            print("[DVS_GEST] Building TensorRT engine...")
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                return None

            runtime = trt.Runtime(logger)
            return runtime.deserialize_cuda_engine(serialized_engine)
        except Exception as e:
            print(f"[DVS_GEST] TensorRT engine build failed: {e}")
            return None

    def _setup_trt_inference(self, h: int, w: int):
        import tensorrt as trt
        self.trt_context = self.trt_engine.create_execution_context()
        self._use_new_trt_api = hasattr(self.trt_engine, "num_io_tensors")

        if self._use_new_trt_api:
            num_io = self.trt_engine.num_io_tensors
            self.trt_buffers = {}
            for i in range(num_io):
                name = self.trt_engine.get_tensor_name(i)
                shape = self.trt_engine.get_tensor_shape(name)
                dtype = self.trt_engine.get_tensor_dtype(name)
                mode = self.trt_engine.get_tensor_mode(name)
                is_input = (mode == trt.TensorIOMode.INPUT)
                torch_dtype = torch.float16 if dtype == trt.DataType.HALF else torch.float32
                buffer = torch.zeros(tuple(shape), dtype=torch_dtype, device=self.device)
                self.trt_buffers[name] = buffer
                if is_input:
                    self.trt_input_name = name
                    self.trt_input = buffer
                else:
                    self.trt_output_name = name
                    self.trt_output = buffer
            self.trt_context.set_input_shape(self.trt_input_name,
                                             (1, self.in_channels, h, w))
        else:
            num_bindings = self.trt_engine.num_bindings
            self.trt_bindings = []
            for i in range(num_bindings):
                shape = self.trt_engine.get_binding_shape(i)
                dtype = self.trt_engine.get_binding_dtype(i)
                is_input = self.trt_engine.binding_is_input(i)
                torch_dtype = torch.float16 if dtype == trt.DataType.HALF else torch.float32
                buffer = torch.zeros(tuple(shape), dtype=torch_dtype, device=self.device)
                self.trt_bindings.append(buffer)
                if is_input:
                    self.trt_input_idx = i
                    self.trt_input = buffer
                else:
                    self.trt_output = buffer
            self.trt_context.set_binding_shape(self.trt_input_idx,
                                               (1, self.in_channels, h, w))

    def _predict_trt(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.dtype != self.trt_input.dtype:
            input_tensor = input_tensor.to(self.trt_input.dtype)
        self.trt_input.copy_(input_tensor)

        if self._use_new_trt_api:
            for name, buf in self.trt_buffers.items():
                self.trt_context.set_tensor_address(name, buf.data_ptr())
            self.trt_context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
        else:
            bindings = [b.data_ptr() for b in self.trt_bindings]
            self.trt_context.execute_v2(bindings)

        return self.trt_output.clone()

    def _warmup(self, n: int = 10):
        h, w = self.image_size if self.image_size else (DVS_HEIGHT, DVS_WIDTH)
        dummy = torch.randn(1, self.in_channels, h, w, device=self.device)
        if self.use_fp16:
            dummy = dummy.half()

        with torch.no_grad():
            for _ in range(n):
                if self._use_trt_engine and hasattr(self, "trt_context"):
                    self._predict_trt(dummy)
                else:
                    self.model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def preprocess_dvs(self, dvs_frame: np.ndarray) -> torch.Tensor:
        """Preprocess DVS frame for inference (0-255 grayscale)."""
        if len(dvs_frame.shape) == 3:
            dvs_frame = cv2.cvtColor(dvs_frame, cv2.COLOR_BGR2GRAY)
        tensor = self.transform(dvs_frame).unsqueeze(0).to(self.device)
        if self.use_fp16:
            tensor = tensor.half()
        return tensor

    def predict(self, dvs_frame: np.ndarray) -> Tuple[str, float, np.ndarray, float]:
        """Predict gesture from DVS frame.

        Applies time surface preprocessing if auto-detected from checkpoint.

        Returns:
            ``(gesture, confidence, probs, elapsed_seconds)``
        """
        # Apply time surface if enabled
        if self.ts_processor is not None:
            timestamp = time.perf_counter() - self._ts_start_time
            dvs_frame = self.ts_processor.process(dvs_frame, timestamp)

        start = time.perf_counter()
        tensor = self.preprocess_dvs(dvs_frame)

        with torch.no_grad():
            if self._use_trt_engine and hasattr(self, "trt_context"):
                outputs = self._predict_trt(tensor)
            else:
                outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = probs.max(1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        gesture = self.classes[pred_idx.item()]
        return gesture, confidence.item(), probs[0].cpu().numpy(), elapsed
