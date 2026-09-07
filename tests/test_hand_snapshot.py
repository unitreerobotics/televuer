import asyncio
from contextlib import contextmanager
from dataclasses import fields
import importlib
from multiprocessing import Array, Value
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np


def load_modules():
    package = types.ModuleType("televuer")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/televuer")]
    vuer = types.ModuleType("vuer")
    vuer.Vuer = object
    schemas = types.ModuleType("vuer.schemas")
    for name in ("ImageBackground", "Hands", "MotionControllers", "WebRTCVideoPlane", "WebRTCStereoVideoPlane"):
        setattr(schemas, name, object)
    with mock.patch.dict(sys.modules, {
        "televuer": package, "vuer": vuer, "vuer.schemas": schemas,
        "cv2": types.ModuleType("cv2"),
    }):
        return importlib.import_module("televuer.televuer"), importlib.import_module("televuer.tv_wrapper")


def hand_event(offset):
    data = np.tile(np.eye(4).reshape(-1), 25)
    for index in range(25):
        data[index * 16 + 12:index * 16 + 15] = [offset, index * 0.01, 0.5]
    return types.SimpleNamespace(value={
        "left": data.tolist(), "right": data.tolist(),
        "leftState": {"pinch": True, "pinchValue": offset},
        "rightState": {"squeeze": True, "squeezeValue": offset},
    })


class PauseBeforeWrite:
    def __init__(self, array, entered, resume):
        self.array, self.entered, self.resume = array, entered, resume

    @contextmanager
    def get_lock(self):
        if threading.current_thread() is not threading.main_thread():
            self.entered.set()
            if not self.resume.wait(2.0):
                raise TimeoutError("test writer was not resumed")
        with self.array.get_lock():
            yield

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value


class HandSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.module, self.wrapper_module = load_modules()
        self.tv = self.module.TeleVuer.__new__(self.module.TeleVuer)
        self.tv.use_hand_tracking = True
        self.tv.head_pose_shared = Array("d", np.eye(4).reshape(-1).tolist())
        self.tv.motion_data_ready_shared = Value("b", False)
        self.tv.motion_sample_seq_shared = Value("L", 0)
        for side in ("left", "right"):
            setattr(self.tv, f"{side}_arm_pose_shared", Array("d", 16))
            setattr(self.tv, f"{side}_hand_position_shared", Array("d", 75))
            setattr(self.tv, f"{side}_hand_orientation_shared", Array("d", 225))
            for name in ("pinch", "squeeze"):
                setattr(self.tv, f"{side}_hand_{name}_shared", Value("b", False))
                setattr(self.tv, f"{side}_hand_{name}Value_shared", Value("d", 0.0))

    def publish(self, offset):
        asyncio.run(self.tv.on_hand_move(hand_event(offset), None))

    def wrapper(self, rotations=False):
        with mock.patch.object(self.wrapper_module, "TeleVuer", return_value=self.tv):
            return self.wrapper_module.TeleVuerWrapper(use_hand_tracking=True, return_hand_rot_data=rotations)

    def assert_same_data(self, left, right):
        for field in fields(left):
            np.testing.assert_equal(getattr(left, field.name), getattr(right, field.name))

    def test_completed_handler_returns_a_consistent_snapshot(self):
        self.publish(0.2)
        snapshot = self.tv.get_hand_motion_snapshot(include_orientations=True)
        self.assertEqual(snapshot["motion_sample_seq"], 2)
        self.assertTrue(snapshot["motion_data_ready"])
        for side in ("left", "right"):
            self.assertAlmostEqual(snapshot[f"{side}_arm_pose"][0, 3], 0.2)
            np.testing.assert_allclose(snapshot[f"{side}_hand_positions"][:, 0], 0.2)
            np.testing.assert_allclose(snapshot[f"{side}_hand_orientations"], np.tile(np.eye(3), (25, 1, 1)))
        self.assertAlmostEqual(snapshot["left_hand_pinchValue"], 0.2)
        self.assertAlmostEqual(snapshot["right_hand_squeezeValue"], 0.2)

    def test_reader_rejects_a_real_handler_paused_between_hands(self):
        self.publish(0.1)
        entered, resume = threading.Event(), threading.Event()
        self.tv.right_arm_pose_shared = PauseBeforeWrite(self.tv.right_arm_pose_shared, entered, resume)
        writer = threading.Thread(target=self.publish, args=(0.4,))
        writer.start()
        try:
            self.assertTrue(entered.wait(1.0))
            # Independent property reads demonstrate the original mixed-update problem.
            self.assertAlmostEqual(self.tv.left_arm_pose[0, 3], 0.4)
            self.assertAlmostEqual(self.tv.right_arm_pose[0, 3], 0.1)
            self.assertIsNone(self.tv.get_hand_motion_snapshot())
        finally:
            resume.set()
            writer.join(timeout=2.0)
        self.assertFalse(writer.is_alive())
        snapshot = self.tv.get_hand_motion_snapshot()
        self.assertEqual(snapshot["motion_sample_seq"], 4)
        np.testing.assert_equal(snapshot["left_arm_pose"], snapshot["right_arm_pose"])

    def test_reader_rejects_a_writer_completing_during_the_copy(self):
        self.publish(0.1)
        original = self.module.TeleVuer.left_hand_positions.fget
        interleaved = False

        def positions(tv):
            nonlocal interleaved
            if not interleaved:
                interleaved = True
                self.publish(0.4)
            return original(tv)

        with mock.patch.object(self.module.TeleVuer, "left_hand_positions", property(positions)):
            self.assertIsNone(self.tv.get_hand_motion_snapshot(max_attempts=1))
        snapshot = self.tv.get_hand_motion_snapshot()
        self.assertAlmostEqual(snapshot["left_arm_pose"][0, 3], 0.4)
        self.assertAlmostEqual(snapshot["right_hand_squeezeValue"], 0.4)

    def test_failed_handler_is_not_committed_and_next_event_recovers(self):
        self.publish(0.1)
        event = hand_event(0.4)
        event.value["left"] = event.value["left"][:32]
        asyncio.run(self.tv.on_hand_move(event, None))
        self.assertEqual(self.tv.motion_sample_seq_shared.value % 2, 1)
        self.assertIsNone(self.tv.get_hand_motion_snapshot())
        self.publish(0.6)
        snapshot = self.tv.get_hand_motion_snapshot()
        self.assertEqual(snapshot["motion_sample_seq"], 6)
        np.testing.assert_allclose(snapshot["left_hand_positions"][:, 0], 0.6)
        np.testing.assert_allclose(snapshot["right_hand_positions"][:, 0], 0.6)

    def test_wrapper_keeps_the_last_complete_pose_fingers_and_gestures(self):
        wrapper = self.wrapper(rotations=True)
        self.publish(0.1)
        before = wrapper.get_tele_data()
        sequence = self.tv._begin_motion_sample()
        with self.tv.left_arm_pose_shared.get_lock():
            self.tv.left_arm_pose_shared[12] = 9.0
        with self.tv.left_hand_position_shared.get_lock():
            self.tv.left_hand_position_shared[:] = [9.0] * 75
        with self.tv.left_hand_orientation_shared.get_lock():
            self.tv.left_hand_orientation_shared[:] = [0.0] * 225
        with self.tv.left_hand_pinchValue_shared.get_lock():
            self.tv.left_hand_pinchValue_shared.value = 0.9
        self.assert_same_data(before, wrapper.get_tele_data())
        self.assertEqual(sequence % 2, 1)
        self.publish(0.5)
        after = wrapper.get_tele_data()
        self.assertNotEqual(before.left_hand_pinchValue, after.left_hand_pinchValue)
        self.assertAlmostEqual(after.left_hand_pinchValue, 50.0)

    def test_startup_with_an_uncommitted_event_uses_not_ready_defaults(self):
        self.tv._begin_motion_sample()
        data = self.wrapper().get_tele_data()
        self.assertFalse(data.motion_data_ready)
        self.assertTrue(np.isfinite(data.left_wrist_pose).all())
        np.testing.assert_equal(data.left_hand_pos, 0.0)

    def test_orientations_are_only_copied_when_requested(self):
        self.publish(0.1)
        with mock.patch.object(self.module.TeleVuer, "left_hand_orientations", new_callable=mock.PropertyMock) as orientations:
            orientations.side_effect = AssertionError("unrequested orientation read")
            self.assertNotIn("left_hand_orientations", self.tv.get_hand_motion_snapshot())

    def test_returned_arrays_do_not_alias_shared_memory(self):
        self.publish(0.2)
        snapshot = self.tv.get_hand_motion_snapshot(include_orientations=True)
        for name in ("left_arm_pose", "left_hand_positions", "left_hand_orientations"):
            snapshot[name][:] = 99.0
            self.assertFalse(np.all(getattr(self.tv, name) == 99.0))

    def test_controller_wrapper_does_not_use_hand_snapshots(self):
        controller = mock.Mock()
        controller.head_pose = np.eye(4)
        controller.left_arm_pose = np.eye(4)
        controller.right_arm_pose = np.eye(4)
        controller.motion_data_ready = True
        for side in ("left", "right"):
            for name in ("triggerValue", "squeezeValue"):
                setattr(controller, f"{side}_ctrl_{name}", 0.2)
            for name in ("trigger", "squeeze", "thumbstick", "aButton", "bButton"):
                setattr(controller, f"{side}_ctrl_{name}", False)
            setattr(controller, f"{side}_ctrl_thumbstickValue", np.zeros(2))
        with mock.patch.object(self.wrapper_module, "TeleVuer", return_value=controller):
            wrapper = self.wrapper_module.TeleVuerWrapper(use_hand_tracking=False)
        data = wrapper.get_tele_data()
        self.assertTrue(data.motion_data_ready)
        self.assertAlmostEqual(data.left_ctrl_triggerValue, 8.0)
        controller.get_hand_motion_snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
