"""Camera navigation: fly, focus, views, plane distance, new plane."""
from .common import *  # noqa: F401,F403
from ..i18n import tr




# World-axis look directions (camera looks along -back). Matches Blender View Axis / MACHIN3 View Axis.
AXIS_VIEWS = (
    ("top",    (0.0, 1.0, 0.0)),
    ("bottom", (0.0, -1.0, 0.0)),
    ("front",  (0.0, 0.0, 1.0)),
    ("back",   (0.0, 0.0, -1.0)),
    ("right",  (1.0, 0.0, 0.0)),
    ("left",   (-1.0, 0.0, 0.0)),
)


def nearest_axis_back(back):
    """Nearest world axis for Blender-style Alt orbit snap."""
    back = np.asarray(back, float).reshape(3)
    n = float(np.linalg.norm(back))
    if n < 1e-12:
        return np.array(AXIS_VIEWS[0][1], float)
    back = back / n
    return max((np.array(a, float) for _, a in AXIS_VIEWS), key=lambda a: float(a @ back))


AXIS_SNAP_RAD_PER_PX = 0.02  # Alt orbit snap: ~40 px of drag leaves an axis view (45°), a few px stay put


def axis_snap_back(start_rot, acc_dx, acc_dy):
    """World axis an Alt orbit lands on after a drag of (acc_dx, acc_dy) px in total.
    The whole drag swings the START view along a great circle of its own basis (columns right, up, back), so the
    drag passes on through the axis views (Front -> Right -> Back -> Left ...) also when it began in one. Same
    direction as Camera.orbit: drag right moves the camera toward -right, drag down (screen +y) toward +up."""
    right0, up0, back0 = start_rot[:, 0], start_rot[:, 1], start_rot[:, 2]
    length = math.hypot(acc_dx, acc_dy)
    if length == 0.0:
        return nearest_axis_back(back0)
    swing = (-acc_dx * right0 + acc_dy * up0) / length
    angle = length * AXIS_SNAP_RAD_PER_PX
    return nearest_axis_back(math.cos(angle) * back0 + math.sin(angle) * swing)


class NavigationMixin:
    # --- Camera moves ----------------------------------------------------
    def fly_to(self, target, dist, rot):
        self._flight = (self.cam.state(), (np.asarray(target, float), float(dist), rot), time.monotonic())
        self._flight_timer.start()

    def _step_flight(self):
        start, end, t0 = self._flight
        t = min(1.0, (time.monotonic() - t0) / FLIGHT_SECONDS)
        self.cam.set_state(*Camera.interpolate(start, end, t))
        if t >= 1.0:
            self._stop_flight()
        self.update_view()  # camera only

    def _stop_flight(self):
        self._flight_timer.stop()
        self._flight = None
        if self._flight_ortho is not None:  # axis view reached: now orthographic
            self.cam.ortho = self._flight_ortho
            self._flight_ortho = None

    def focus(self, obj):
        if isinstance(obj, Plane):
            n = obj.normal
            back = n if n @ self.cam.back >= 0 else -n
            dist = FOCUS_DIST * max(obj.size_u, obj.size_v) / 10.0
            self.fly_to(obj.origin, dist, view_for_back(back))
        elif isinstance(obj, NullObject):  # the null with everything below it
            boxes = [o.bounds() for o in self.scene.descendants(obj)
                     if isinstance(o, StrokeObject) and o.curves or isinstance(o, MeshObject) and not o.missing]
            lo = np.min([b[0] for b in boxes] + [obj.origin], axis=0)
            hi = np.max([b[1] for b in boxes] + [obj.origin], axis=0)
            size = max(float(np.linalg.norm(hi - lo)), 1.0)
            self.fly_to((lo + hi) / 2, FOCUS_DIST * size / 10.0, self.cam.rot.copy())
        elif isinstance(obj, StrokeObject) and obj.curves or isinstance(obj, MeshObject) and not obj.missing:
            lo, hi = obj.bounds()
            size = max(float(np.linalg.norm(hi - lo)), 0.1)
            self.fly_to((lo + hi) / 2, FOCUS_DIST * size / 10.0, self.cam.rot.copy())

    def home(self):
        self._persp_view = None
        self.cam.ortho = False
        self.fly_to(np.zeros(3), Camera.HOME_DIST, Camera.home_rot())

    def axis_view(self, back):
        """Move into an axis view; remember the free view before it (for Space + 1)."""
        if getattr(self, "_look", None) is not None:  # timing camera: perspective with its focal length – Esc first
            return self.notify(tr("In einer Shot-Kamera gibt es keine Achsenansichten – Esc verlässt die Kamera"))
        if self._persp_view is None:
            self._persp_view = self.cam.state()
        self.fly_to(self.cam.target, self.cam.dist, view_for_back(back))
        self._flight_ortho = True  # axis views are orthographic as in Houdini
        plane = self.scene.active_plane
        single = not self.multi_mode or [s for s in self._multi_surfaces()] == [plane]
        if self.scene.active_mesh is None and plane.preset and single:
            # only a ground plane was active: the ground plane that looks into this view (own ones stay active)
            facing = max(self.scene.presets, key=lambda pl: abs(float(pl.normal @ back)))
            if facing is not plane:
                self.scene.activate(facing)
                if self.multi_mode:
                    self.scene.multi_ids = [facing.id]
                self.notify(tr("Zeichnet auf: {0}").format(facing.name))


    def snap_axis_view(self, back, animate=False):
        """Snap to a world-axis orthographic view. Instant by default (Alt-orbit snappy feel)."""
        back = np.asarray(back, float)
        if getattr(self, "_look", None) is not None:
            return self.notify(tr("In einer Shot-Kamera gibt es keine Achsenansichten – Esc verlässt die Kamera"))
        if self._persp_view is None and not self.cam.ortho:
            self._persp_view = self.cam.state()
        rot = view_for_back(back)
        if animate:
            self.fly_to(self.cam.target, self.cam.dist, rot)
            self._flight_ortho = True
        else:
            self._stop_flight()
            self.cam.set_state(self.cam.target, self.cam.dist, rot)
            self.cam.ortho = True
            self._flight_ortho = None
        plane = self.scene.active_plane
        single = not self.multi_mode or [s for s in self._multi_surfaces()] == [plane]
        if self.scene.active_mesh is None and plane is not None and plane.preset and single:
            facing = max(self.scene.presets, key=lambda pl: abs(float(pl.normal @ back)))
            if facing is not plane:
                self.scene.activate(facing)
                if self.multi_mode:
                    self.scene.multi_ids = [facing.id]
                self.notify(tr("Zeichnet auf: {0}").format(facing.name))
        self.update_view()

    def orbit_axis_snap(self, dx, dy):
        """Alt orbit (Blender Alt during view rotate): instant orthographic world-axis view. The drag is summed from
        the pose the snap began in (mousePress records it) and swung from there (axis_snap_back) – not orbited step
        by step from the snapped pose, which a mouse step of 10-20 px could never leave."""
        start = self._orbit_snap_start
        if start is None or self.cam.rot is not getattr(self, "_orbit_snap_rot", self.cam.rot):
            # Alt came during the drag, or the camera turned freely since the last snap (Alt let go): begin here
            start = self._orbit_snap_start = self.cam.state()
            self._orbit_snap_acc = [0.0, 0.0]
        if self._persp_view is None and not self.cam.ortho:
            self._persp_view = self.cam.state()
        self._orbit_snap_acc[0] += dx
        self._orbit_snap_acc[1] += dy
        target, dist, rot0 = start
        axis = axis_snap_back(rot0, *self._orbit_snap_acc)
        rot = view_for_back(axis)
        self.cam.set_state(target.copy(), dist, rot)  # copy: set_state keeps the array, the start must stay intact
        self.cam.ortho = True
        self._orbit_snap_rot = rot
        prev = getattr(self, "_orbit_snap_axis", None)
        if prev is None or float(np.linalg.norm(prev - axis)) > 1e-6:
            self._orbit_snap_axis = axis.copy()

    def perspective_view(self):
        """Space + 1 as in Houdini: back to the free view before the last axis view."""
        self.cam.ortho = False
        if self._persp_view is not None:
            self.fly_to(*self._persp_view)
            self._persp_view = None

    def _plane_distance(self):
        """Distance of the camera to the active plane (to the nearest point of the plane), otherwise the orbit distance.
        Mouse wheel and [ ] follow it – not the camera's pivot, which may lie elsewhere."""
        plane = self.scene.active_plane
        if self.scene.active_mesh is not None or plane is None:
            return self.cam.dist
        eye = self.cam.eye
        rel = eye - plane.origin
        u = np.clip(rel @ plane.axis_u, -plane.size_u / 2, plane.size_u / 2)
        w = np.clip(rel @ plane.axis_v, -plane.size_v / 2, plane.size_v / 2)
        return max(float(np.linalg.norm(eye - (plane.origin + u * plane.axis_u + w * plane.axis_v))), NEAR)

    def _plane_step(self):
        """Step for moving the plane: grows with the camera distance, but is always a round
        number (… 0.1, 0.2, 0.5, 1, 2, 5 …) – so the plane lands on smooth heights like 3.0 instead of 2.87."""
        raw = PLANE_STEP * self._plane_distance() / Camera.HOME_DIST
        base = 10 ** math.floor(math.log10(max(raw, 1e-6)))
        return min((m * base for m in (1, 2, 5, 10)), key=lambda x: abs(math.log(x / raw)))

    # --- Actions ---------------------------------------------------------
    def new_plane(self, pos=None):
        """New plane perpendicular to the view direction. If lines are selected, it lies at the depth of their
        center (in an axis view one cannot otherwise tell whether front or back is meant);
        otherwise at the point under the cursor (line, plane or mesh), otherwise at orbit distance.
        Size: fixed fraction of the image height."""
        if pos is None:
            inside = self._last_pos is not None and self.rect().contains(self._last_pos.toPoint())
            pos = self._last_pos if inside else QPointF(self.width() / 2, self.height() / 2)
        origin = None
        if self.edge_selection and self.transform_target is not None:
            lo, hi = self.transform_target.bounds()
            through = Plane("", (lo + hi) / 2, self.cam.right, self.cam.up)
            origin = self._ray_plane(pos, through)[0]
        if origin is None:
            origin = self._pick_point(pos)
        if origin is None:
            orbit_plane = Plane("", self.cam.target, self.cam.right, self.cam.up)
            origin = self._ray_plane(pos, orbit_plane)[0]
        if origin is None:
            origin = self.cam.target
        depth = max((origin - self.cam.eye) @ -self.cam.back, NEAR)
        size = _round_nice(NEW_PLANE_VIEW_FRACTION * self._world_per_px(depth) * self.height())

        self.scene.checkpoint()
        if self.ws.timing_mode:  # in timing: always a flipbook plane (own type, with its own stroke object)
            plane = self.scene.new_flipbook(origin, self.cam.right, self.cam.up)
            self.scene.active_stroke = self.scene.flip_stroke(plane)
        else:
            plane = self.scene.new_plane(origin, self.cam.right, self.cam.up)
        plane.size_u = plane.size_v = size
        # only active, not selected: otherwise a gizmo would sit exactly where one is about to keep drawing (the
        # plane appears under the cursor). Moving works via mouse wheel / [ ], click selects it.
        self.selection = []
        self.edge_selection = []  # determined the position; its gizmo would otherwise be in the way when continuing to draw
        self.notify(tr("{0} – zeichnen; Mausrad oder [ ] verschiebt sie, Klick wählt sie").format(plane.name))
        self._changed()
