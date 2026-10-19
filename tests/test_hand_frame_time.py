"""Test actual event/property code without starting a Vuer server or XR session."""

import ast
import __future__
import asyncio
from dataclasses import dataclass, field, fields
from multiprocessing import Array, Value
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_class(path, name, namespace):
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class HandFrameTimeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Mock(return_value=100.0)
        cls = load_class(ROOT / "src/televuer/televuer.py", "TeleVuer",
                         {"time": SimpleNamespace(monotonic=self.clock), "np": np})
        self.receiver = object.__new__(cls)
        self.receiver.hand_frame_time_shared = Value('d', 0.0)
        self.receiver.motion_data_ready_shared = Value('b', False)
        for side in ("left", "right"):
            for name, size in (("arm_pose", 16), ("hand_position", 75), ("hand_orientation", 225)):
                setattr(self.receiver, f"{side}_{name}_shared", Array('d', size))
            for name in ("pinch", "pinchValue", "squeeze", "squeezeValue"):
                setattr(self.receiver, f"{side}_hand_{name}_shared", Value('d', 0.0))

    def send(self, payload):
        asyncio.run(self.receiver.on_hand_move(SimpleNamespace(value=payload), None))

    def complete(self):
        return {"left": [0.0] * 400, "right": [0.0] * 400, "leftState": {}, "rightState": {}}

    def test_zero_before_first_complete_event(self):
        self.assertEqual(self.receiver.hand_frame_time, 0.0)
        self.clock.assert_not_called()

    def test_complete_events_advance_receive_time(self):
        self.send(self.complete())
        self.assertEqual(self.receiver.hand_frame_time, 100.0)
        self.clock.return_value = 200.0
        self.send(self.complete())
        self.assertEqual(self.receiver.hand_frame_time, 200.0)

    def test_cached_reads_do_not_refresh_receive_time(self):
        self.send(self.complete())
        self.clock.return_value = 999.0
        self.receiver.left_hand_positions
        self.receiver.right_hand_positions
        self.assertEqual(self.receiver.hand_frame_time, 100.0)
        self.clock.assert_called_once()

    def test_missing_and_truncated_events_do_not_refresh_receive_time(self):
        self.send(self.complete())
        self.clock.return_value = 999.0
        self.send({})
        broken = self.complete()
        broken["right"] = [0.0]
        self.send(broken)
        self.assertEqual(self.receiver.hand_frame_time, 100.0)
        self.clock.assert_called_once()

    def test_new_packet_field_is_optional_and_appended(self):
        cls = load_class(ROOT / "src/televuer/tv_wrapper.py", "TeleData",
                         {"np": np, "field": field, "dataclass": dataclass})
        packet = cls(np.eye(4), np.eye(4), np.eye(4))
        self.assertEqual(packet.hand_frame_time, 0.0)
        self.assertEqual(fields(cls)[-1].name, "hand_frame_time")


if __name__ == "__main__":
    unittest.main()
