"""Entry point: python -m app

Launches the Unified GUI with calibration and tracking tabs.
"""

import time

from app.config import parse_args, setup_sys_path


def main():
    args = parse_args()
    setup_sys_path()

    from app.core.camera import CameraManager
    from app.core.calibration_store import CalibrationStore
    from app.core.event_loop import MainLoop
    from app.core.demo import OutputModeType
    from app.demos.calibration.demo import CalibrationDemo
    from app.demos.tracking.demo import TrackingDemo
    from app.demos.tracking.gui_output import TrackingGUIOutput

    print("=" * 50)
    print("Unified GUI")
    print("=" * 50)

    # ── Step 0: Activate ALL CAN interfaces ──
    from app.core.can_setup import setup_all_can

    can_result = setup_all_can(
        usb_port=args.usb_port,
        skip_arm=args.no_arm,
        skip_hand=args.no_hand,
    )

    # Overwrite args with detected names
    if can_result.arm_can:
        args.can = can_result.arm_can
    elif not args.no_arm:
        print(f"[INIT] Arm CAN failed: {can_result.arm_error} — running without arm")
        args.no_arm = True

    if can_result.hand_can:
        args.hand_can = can_result.hand_can
    elif not args.no_hand:
        print(f"[INIT] Hand CAN failed: {can_result.hand_error} — running without hand")
        args.no_hand = True

    # Shared warm-up for both devices
    if (can_result.arm_can or can_result.hand_can) and args.can_warmup > 0:
        print(f"[INIT] Waiting {args.can_warmup}s for CAN warm-up...")
        time.sleep(args.can_warmup)

    # ── Step 1: Init cameras ──
    camera_mgr = CameraManager(args.dvs_camera, args.rgb_camera, args.rgb_rotate)

    print("[INIT] Starting DVS camera...")
    camera_mgr.init_dvs()

    print("[INIT] Starting RGB camera...")
    camera_mgr.init_rgb()

    # ── Step 2: Shared calibration store ──
    cal_store = CalibrationStore()

    # ── Step 3: Arm (optional) ──
    bridge = None
    arm_thread = None
    if not args.no_arm:
        try:
            from app.core.arm import CommandBridge, ArmThread
            bridge = CommandBridge()
            arm_thread = ArmThread(bridge, args.can, args.speed)
            arm_thread.start()
            print("[INIT] Arm thread started")
        except Exception as e:
            print(f"[INIT] Arm init failed: {e} — running without arm")
            bridge = None
            arm_thread = None
    else:
        print("[INIT] Arm disabled (--no-arm)")

    # ── Step 4: LinkerHand (future) ──
    # Will use args.hand_can when LinkerHand integration is ready

    # ── Step 5: Create demos ──
    cal_demo = CalibrationDemo(cal_store, args, bridge=bridge, arm_thread=arm_thread)
    tracking_demo = TrackingDemo(cal_store, args)

    # ── Step 6: Register output modes for tracking demo ──
    tracking_demo.register_output(
        OutputModeType.GUI,
        TrackingGUIOutput(tracking_demo),
    )

    if bridge and arm_thread:
        from app.demos.tracking.phys_dvs_output import TrackingPhysDVSOutput
        from app.demos.tracking.phys_rgb_output import TrackingPhysRGBOutput
        tracking_demo.register_output(
            OutputModeType.PHYS_DVS,
            TrackingPhysDVSOutput(tracking_demo, bridge, arm_thread),
        )
        tracking_demo.register_output(
            OutputModeType.PHYS_RGB,
            TrackingPhysRGBOutput(tracking_demo, bridge, arm_thread),
        )

    # ── Step 7: Default output = GUI ──
    tracking_demo.switch_output(OutputModeType.GUI)

    # ── Step 8: Run main loop ──
    demos = {"Calibration": cal_demo, "Tracking": tracking_demo}
    loop = MainLoop(camera_mgr, demos, bridge=bridge, arm_thread=arm_thread)

    print()
    print("Controls:")
    print("  Click tab bar    — switch tab")
    print("  [q]              — quit")
    print("  [g/e/r]          — GUI / Physical DVS / Physical RGB mode")
    print("  [h]              — arm go home")
    print("  [w]              — arm go center (draw position)")
    print("  [space]          — toggle tracking")
    print("  [c]              — clear canvas")
    print("  [v]              — cycle layout (GUI mode)")
    print("  [Enter]          — confirm calibration")
    print("  [d]              — re-detect RGB quad")
    print()

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted")
    finally:
        # Cleanup arm
        if arm_thread:
            arm_thread.stop()
            arm_thread.join(timeout=10.0)
            print("[EXIT] Arm thread stopped")
        print("[EXIT] Done")


if __name__ == "__main__":
    main()
