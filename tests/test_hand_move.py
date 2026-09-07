import asyncio
import importlib.util
from multiprocessing import Array, Value
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np


def load_televuer():
    vuer = types.ModuleType("vuer")
    vuer.Vuer = object
    schemas = types.ModuleType("vuer.schemas")
    for name in (
        "ImageBackground", "Hands", "MotionControllers",
        "WebRTCVideoPlane", "WebRTCStereoVideoPlane",
    ):
        setattr(schemas, name, object)
    path = Path(__file__).resolve().parents[1] / "src/televuer/televuer.py"
    with mock.patch.dict("sys.modules", {
        "vuer": vuer, "vuer.schemas": schemas, "cv2": types.ModuleType("cv2"),
    }):
        spec = importlib.util.spec_from_file_location("televuer_hand_move_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.TeleVuer


def hand_pose(offset):
    data = np.tile(np.eye(4).reshape(-1), 25)
    for index in range(25):
        data[index * 16 + 12:index * 16 + 15] = [offset, index * 0.01, 0.5]
    return data.tolist()


class HandMoveTest(unittest.TestCase):
    def setUp(self):
        # Do not construct a Vuer server, shared image memory, or a child process.
        tele_vuer_class = load_televuer()
        self.tv = tele_vuer_class.__new__(tele_vuer_class)
        self.tv.motion_data_ready_shared = Value("b", False)
        for side in ("left", "right"):
            setattr(self.tv, f"{side}_arm_pose_shared", Array("d", 16))
            setattr(self.tv, f"{side}_hand_position_shared", Array("d", 75))
            setattr(self.tv, f"{side}_hand_orientation_shared", Array("d", 225))
            for name in ("pinch", "squeeze"):
                setattr(self.tv, f"{side}_hand_{name}_shared", Value("b", False))
                setattr(self.tv, f"{side}_hand_{name}Value_shared", Value("d", 0.0))

    def publish(self, payload):
        asyncio.run(self.tv.on_hand_move(types.SimpleNamespace(value=payload), None))

    def assert_pose(self, side, expected):
        np.testing.assert_allclose(getattr(self.tv, f"{side}_arm_pose_shared")[:], expected[:16])
        positions = np.asarray(expected).reshape(25, 16)[:, 12:15].reshape(-1)
        np.testing.assert_allclose(getattr(self.tv, f"{side}_hand_position_shared")[:], positions)
        orientations = np.asarray(expected).reshape(25, 16)[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]
        np.testing.assert_allclose(getattr(self.tv, f"{side}_hand_orientation_shared")[:], orientations.reshape(-1))

    def test_complete_event_preserves_pose_and_gesture_values(self):
        self.publish({
            "left": hand_pose(0.2), "right": hand_pose(-0.2),
            "leftState": {"pinch": True, "pinchValue": 0.7},
            "rightState": {"squeeze": True, "squeezeValue": 0.8},
        })
        self.assert_pose("left", hand_pose(0.2))
        self.assert_pose("right", hand_pose(-0.2))
        self.assertTrue(self.tv.left_hand_pinch_shared.value)
        self.assertAlmostEqual(self.tv.left_hand_pinchValue_shared.value, 0.7)
        self.assertTrue(self.tv.right_hand_squeeze_shared.value)
        self.assertAlmostEqual(self.tv.right_hand_squeezeValue_shared.value, 0.8)
        self.assertTrue(self.tv.motion_data_ready)

    def test_missing_gesture_states_does_not_discard_valid_poses(self):
        self.publish({"left": hand_pose(0.2), "right": hand_pose(-0.2)})
        self.assert_pose("left", hand_pose(0.2))
        self.assert_pose("right", hand_pose(-0.2))
        self.assertFalse(self.tv.left_hand_pinch_shared.value)
        self.assertEqual(self.tv.right_hand_squeezeValue_shared.value, 0.0)
        self.assertTrue(self.tv.motion_data_ready)

    def test_null_or_non_object_gesture_states_use_neutral_defaults(self):
        self.publish({
            "left": hand_pose(0.2), "right": hand_pose(-0.2),
            "leftState": None, "rightState": [],
        })
        self.assert_pose("left", hand_pose(0.2))
        self.assert_pose("right", hand_pose(-0.2))
        self.assertTrue(self.tv.motion_data_ready)

    def test_a_single_tracked_hand_is_accepted(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                self.setUp()
                self.publish({side: hand_pose(0.3)})
                self.assert_pose(side, hand_pose(0.3))
                self.assertTrue(self.tv.motion_data_ready)

    def test_missing_hand_keeps_its_last_pose_and_gesture(self):
        self.publish({
            "left": hand_pose(0.2), "right": hand_pose(-0.2),
            "leftState": {}, "rightState": {"pinch": True, "pinchValue": 0.7},
        })
        self.publish({"left": hand_pose(0.4)})
        self.assert_pose("left", hand_pose(0.4))
        self.assert_pose("right", hand_pose(-0.2))
        self.assertTrue(self.tv.right_hand_pinch_shared.value)
        self.assertAlmostEqual(self.tv.right_hand_pinchValue_shared.value, 0.7)

    def test_invalid_pose_does_not_block_the_other_hand(self):
        invalid_poses = [None, [], [0.0] * 399, [0.0] * 400, "invalid", [[1.0] * 16] * 25]
        for value in (float("nan"), float("inf")):
            invalid = hand_pose(0.2)
            invalid[32] = value
            invalid_poses.append(invalid)
        for side in ("left", "right"):
            other = "right" if side == "left" else "left"
            for index, invalid in enumerate(invalid_poses):
                with self.subTest(side=side, case=index):
                    self.setUp()
                    self.publish({
                        side: invalid, other: hand_pose(0.4),
                        "leftState": {}, "rightState": {},
                    })
                    np.testing.assert_array_equal(getattr(self.tv, f"{side}_arm_pose_shared")[:], 0.0)
                    self.assert_pose(other, hand_pose(0.4))
                    self.assertTrue(self.tv.motion_data_ready)

    def test_no_valid_hand_does_not_mark_tracking_ready(self):
        for payload in (None, [], {}, {"left": [], "right": [0.0] * 400}):
            with self.subTest(payload=payload):
                self.publish(payload)
                self.assertFalse(self.tv.motion_data_ready)


if __name__ == "__main__":
    unittest.main()
