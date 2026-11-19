"""Calisto Create timeline slice: motion beats bind components, follow sketch paths, store camera angles."""
LEVEL = "core"

import numpy as np

from sketch_machine import timing
from sketch_machine.scene import Curve, NullObject, Scene, StrokeObject


def test_bind_null_proxy_and_serialize(h):
    sc = Scene()
    tm = sc.timing
    beat = timing.bind_selection(sc, selection_ids=[], start=1005, end=1020, annotation="Pass", text="climb left")
    assert beat.target_kind == "null" and beat.target_id is None
    assert beat.annotation == "Pass" and beat.text == "climb left"
    assert tm.active_motion == beat.id
    data = timing.to_dict(tm)
    assert "motions" in data and data["motions"][0]["annotation"] == "Pass"
    loaded = timing.from_dict(data)
    assert len(loaded.motions) == 1
    m = loaded.motions[0]
    assert (m.start, m.end, m.annotation, m.text) == (1005, 1020, "Pass", "climb left")


def test_path_follow_and_camera(h):
    sc = Scene()
    stroke = StrokeObject("Path")
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    stroke.curves.append(Curve(pts, None, np.full(len(pts), 0.02)))
    sc.strokes.append(stroke)
    beat = timing.new_motion(sc.timing, 1001, 1011, annotation="Arc")
    timing.set_motion_path(beat, sc, stroke_id=stroke.id)
    path = timing.path_points(sc, beat)
    assert len(path) == 3
    start = timing.proxy_at(sc, beat, 1001)
    mid = timing.proxy_at(sc, beat, 1006)
    end = timing.proxy_at(sc, beat, 1011)
    assert start is not None and np.allclose(start, pts[0], atol=1e-5)
    assert end is not None and np.allclose(end, pts[-1], atol=1e-5)
    assert mid is not None and mid[0] > 5.0  # past the corner along arc length
    state = (np.array([1.0, 2.0, 3.0]), 12.0, np.eye(3))
    timing.capture_motion_camera(beat, state, 50.0)
    cam = timing.camera_state(beat)
    assert cam is not None and abs(cam[3] - 50.0) < 1e-6
    assert np.allclose(cam[0], state[0])
    roundtrip = timing.from_dict(timing.to_dict(sc.timing))
    assert roundtrip.motions[0].camera is not None
    assert roundtrip.motions[0].path_stroke_id == stroke.id


def test_bind_null_and_stroke(h):
    sc = Scene()
    null = NullObject("Aircraft")
    sc.nulls.append(null)
    beat = timing.bind_selection(sc, selection_ids=[null.id], start=1001, end=1008)
    assert beat.target_kind == "null" and beat.target_id == null.id
    stroke = sc.new_stroke()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    stroke.curves.append(Curve(pts, None))
    beat2 = timing.bind_selection(sc, selection_ids=[stroke.id], start=1010, end=1015, annotation="Wing")
    assert beat2.target_kind == "stroke" and beat2.target_id == stroke.id
    assert timing.target_label(sc, beat2) == "Wing"
