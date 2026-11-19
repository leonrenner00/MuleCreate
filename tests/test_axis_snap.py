"""Alt orbit axis snap: the drag is summed from the start pose, so it leaves an axis view it began in."""
import numpy as np

from sketch_machine.camera import Camera, view_for_back
from sketch_machine.vp.navigation import NavigationMixin, axis_snap_back

FRONT = np.array([0.0, 0.0, 1.0])
TOP = np.array([0.0, 1.0, 0.0])


class _Nav(NavigationMixin):
    """Only what orbit_axis_snap reads: a Camera in an axis view and the snap start mousePress records."""

    def __init__(self, back):
        self.cam = Camera()
        self.cam.rot = view_for_back(back)
        self.cam.ortho = True
        self._persp_view = None
        self._orbit_snap_axis = None
        self._orbit_snap_start = self.cam.state()  # as viewport.mousePressEvent does for an Alt orbit
        self._orbit_snap_acc = [0.0, 0.0]


def _same(a, b):
    return np.allclose(a, b, atol=1e-9)


def test_front_12px_stays_front():
    assert _same(axis_snap_back(view_for_back(FRONT), 12, 0), FRONT)
    nav = _Nav(FRONT)
    target, dist = nav.cam.target.copy(), nav.cam.dist
    nav.orbit_axis_snap(12, 0)
    assert _same(nav.cam.back, FRONT)
    assert nav.cam.ortho
    assert _same(nav.cam.target, target) and nav.cam.dist == dist


def test_front_160px_horizontal_leaves_front():
    assert not _same(axis_snap_back(view_for_back(FRONT), 160, 0), FRONT)
    nav = _Nav(FRONT)
    nav.orbit_axis_snap(160, 0)
    assert not _same(nav.cam.back, FRONT)
    assert nav.cam.ortho


def test_many_12px_steps_match_one_160px_step():
    steps, one = _Nav(FRONT), _Nav(FRONT)
    for d in [12] * 13 + [4]:  # 13 * 12 + 4 = 160 px
        steps.orbit_axis_snap(d, 0)
    one.orbit_axis_snap(160, 0)
    assert _same(steps.cam.rot, one.cam.rot)
    assert _same(steps.cam.target, one.cam.target) and steps.cam.dist == one.cam.dist
    assert not _same(steps.cam.back, FRONT)


def test_top_160px_horizontal_leaves_top():
    assert not _same(axis_snap_back(view_for_back(TOP), 160, 0), TOP)
    nav = _Nav(TOP)
    nav.orbit_axis_snap(160, 0)
    assert not _same(nav.cam.back, TOP)
    assert nav.cam.ortho
