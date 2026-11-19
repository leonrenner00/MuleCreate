"""Alt orbit axis snap: while the button is down the drag is only summed from the start pose; its DIRECTION picks one
neighbour axis view, switched on mouse release (commit_axis_snap). How far the drag goes does not pick more views."""
import numpy as np

from sketch_machine.camera import Camera, view_for_back
from sketch_machine.vp.navigation import NavigationMixin, axis_snap_back

FRONT = np.array([0.0, 0.0, 1.0])
TOP = np.array([0.0, 1.0, 0.0])


class _Nav(NavigationMixin):
    """Only what orbit_axis_snap and commit_axis_snap read: a Camera in an axis view and the snap start."""

    def __init__(self, back):
        self.cam = Camera()
        self.cam.rot = view_for_back(back)
        self.cam.ortho = True
        self._persp_view = None
        self.press()

    def press(self):
        """As viewport.mousePressEvent does for an Alt orbit: a new gesture from the current pose."""
        self._orbit_snap_start = self.cam.state()
        self._orbit_snap_acc = [0.0, 0.0]

    def drag(self, *moves):
        for dx, dy in moves:
            self.orbit_axis_snap(dx, dy)


def _same(a, b):
    return np.allclose(a, b, atol=1e-9)


def test_front_12px_stays_front():
    assert axis_snap_back(view_for_back(FRONT), 12, 0) is None
    nav = _Nav(FRONT)
    rot, target, dist = nav.cam.rot.copy(), nav.cam.target.copy(), nav.cam.dist
    nav.drag((12, 0))
    assert _same(nav.cam.back, FRONT)
    nav.commit_axis_snap()  # mouse release
    assert _same(nav.cam.back, FRONT) and _same(nav.cam.rot, rot)
    assert nav.cam.ortho
    assert _same(nav.cam.target, target) and nav.cam.dist == dist


def test_direction_not_distance():
    a80 = axis_snap_back(view_for_back(FRONT), 80, 0)
    a400 = axis_snap_back(view_for_back(FRONT), 400, 0)
    assert a80 is not None and not _same(a80, FRONT)
    assert _same(a80, a400)
    short, far = _Nav(FRONT), _Nav(FRONT)
    short.drag(*[(10, 0)] * 8)  # 80 px in steps that each stay under the threshold
    far.drag((400, 0))
    short.commit_axis_snap()
    far.commit_axis_snap()
    assert not _same(short.cam.back, FRONT)
    assert abs(float(short.cam.back @ FRONT)) < 1e-9  # a neighbour of Front (90 degrees), not Back
    assert _same(short.cam.rot, far.cam.rot)
    assert short.cam.ortho and far.cam.ortho


def test_commit_on_release_only():
    nav = _Nav(FRONT)
    rot = nav.cam.rot.copy()
    for _ in range(10):  # 400 px in total while the button is down
        nav.orbit_axis_snap(40, 0)
        assert _same(nav.cam.rot, rot) and _same(nav.cam.back, FRONT)
    nav.commit_axis_snap()  # mouse release
    assert not _same(nav.cam.back, FRONT)
    assert abs(float(nav.cam.back @ FRONT)) < 1e-9
    assert nav.cam.ortho


def test_top_direction():
    a80 = axis_snap_back(view_for_back(TOP), 80, 0)
    assert a80 is not None and not _same(a80, TOP)
    assert _same(a80, axis_snap_back(view_for_back(TOP), 400, 0))
    short, far = _Nav(TOP), _Nav(TOP)
    short.drag((80, 0))
    far.drag((400, 0))
    short.commit_axis_snap()
    far.commit_axis_snap()
    assert not _same(short.cam.back, TOP)
    assert abs(float(short.cam.back @ TOP)) < 1e-9
    assert _same(short.cam.rot, far.cam.rot)
    assert short.cam.ortho


def test_second_gesture():
    nav = _Nav(FRONT)
    nav.drag((80, 0))
    nav.commit_axis_snap()  # first release: lands on a neighbour of Front
    first = nav.cam.back.copy()
    assert not _same(first, FRONT)
    nav.press()  # next press, Alt still held: a new start pose
    nav.drag((80, 0))
    assert _same(nav.cam.back, first)  # still down: no move yet
    nav.commit_axis_snap()  # second release
    assert not _same(nav.cam.back, first)
    assert abs(float(nav.cam.back @ first)) < 1e-9  # one neighbour of the view the first gesture settled on
    assert nav.cam.ortho
