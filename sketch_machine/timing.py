"""Timing (prototype): flipbook planes and timing cameras.

Everything holds, nothing is interpolated:
- A drawing on a flipbook plane applies from its frame until the next drawing of the same plane comes (even an
  empty one). Drawings are paint strokes (Curve.paint) with Curve.frame = frame of the drawing; empty drawings are in
  Plane.blank_keys. Which frames have drawings follows from that (nothing stored twice – undo stays simple).
- A camera has poses (position, view direction, focal length) from one frame each; before the first one the first applies.
  Cameras run in parallel over the whole timeline (no cuts) and are attached to timing notes.

Pure data and calculations, no Qt – the controls are in timing_ui.py and vp/timing_view.py.
"""
import bisect
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

START, END, FPS = 1001, 1100, 24  # VFX standard: starts at 1001
SENSOR_W = 36.0  # mm display: full-frame width, image format GATE (fitted horizontally)
GATE = 16 / 9  # image format of the timing cameras (viewfinder, images)
MM_RANGE = (8.0, 600.0)  # focal length slider (500 mm and more: almost orthographic)
CAM_COLOR = (0.33, 0.45, 0.58)  # default color of new cameras – tinting (categories) only by hand
CAM_COLORS = [(0.33, 0.45, 0.58), (0.78, 0.30, 0.25), (0.86, 0.58, 0.18), (0.36, 0.62, 0.32),
              (0.55, 0.38, 0.70), (0.25, 0.62, 0.66), (0.80, 0.42, 0.62), (0.45, 0.45, 0.45)]
CAM_SIZE_RANGE = (0.1, 5.0)  # icon size: global and per camera (factor)
ONION_FADE = 0.3  # onion skin: opacity of the previous drawing (when painting on you see what changes)
ONION_NEXT_FADE = 0.12  # … of the next drawing (only hinted)

_ids = itertools.count(1)


def _new_id():
    return next(_ids)


def _bump_ids(used):
    """Loaded ids: the counter continues after them (no duplicates)."""
    global _ids
    top = max(used, default=0)
    current = next(_ids)
    _ids = itertools.count(max(top, current) + 1)


# --- Focal length --------------------------------------------------------------
def mm_to_tan(mm):
    """Focal length (mm) -> tan of half the vertical angle of view in the image format GATE."""
    return (SENSOR_W / GATE) / 2 / max(float(mm), 1e-3)


def tan_to_mm(tan_half):
    return (SENSOR_W / GATE) / 2 / max(float(tan_half), 1e-9)


# --- Data -------------------------------------------------------------------
@dataclass(eq=False)
class Pose:
    """Pose of a camera from frame: camera state (Camera.state) and focal length. On the board of its shot note
    it appears as a screenshot (board entry with "shot": {"cam", "pose"})."""
    frame: int
    target: np.ndarray
    dist: float
    rot: np.ndarray
    mm: float
    id: int = field(default_factory=_new_id)

    def state(self):
        return self.target.copy(), float(self.dist), self.rot.copy()

    def matches(self, state, tol=1e-6):
        target, dist, rot = state
        return (np.allclose(target, self.target, atol=tol * max(1.0, self.dist)) and abs(dist - self.dist) <= tol * max(1.0, self.dist)
                and np.allclose(rot, self.rot, atol=1e-6))


@dataclass(eq=False)
class TimingCam:
    name: str
    poses: list = field(default_factory=list)  # sorted by frame, at least one
    color: tuple | None = None  # None = default color CAM_COLOR
    size: float = 1.0  # icon size (factor on the global camera size)
    note: int | None = None  # id of the shot note whose board holds its screenshots
    id: int = field(default_factory=_new_id)

    @property
    def rgb(self):
        return tuple(self.color) if self.color else CAM_COLOR


def _global_note():
    from .scene import Note  # (scene imports timing)
    return Note("Global", np.zeros(3))


# Target kinds for motion beats (timeline spans that bind a component + motion intent).
MOTION_KINDS = ("null", "stroke", "mesh", "plane", "surface")


@dataclass(eq=False)
class MotionBeat:
    """First-class timeline span: bind a scene component (or aircraft/null proxy) to a frame range,
    describe the motion with annotation + text, optionally follow a sketch/flipbook path, and store
    a camera angle for shot planning. Pure intent data – no interpolated keyframe animation yet."""
    start: int
    end: int
    target_kind: str = "null"  # MOTION_KINDS
    target_id: int | None = None  # None = free aircraft/part proxy
    annotation: str = ""  # short label on the timeline row
    text: str = ""  # longer motion-intent description
    path_stroke_id: int | None = None  # stroke whose curves define the flight path
    path_plane_id: int | None = None  # flipbook plane: held drawing curves as path
    camera: dict | None = None  # {target, dist, rot, mm} – planning angle for this beat
    id: int = field(default_factory=_new_id)

    def clamp_range(self, tm):
        self.start = tm.clamp(min(self.start, self.end))
        self.end = tm.clamp(max(self.start, self.end))
        return self

    def covers(self, frame):
        f = int(frame)
        return self.start <= f <= self.end

    def progress(self, frame):
        """0..1 along the beat (held at ends)."""
        span = max(int(self.end) - int(self.start), 1)
        return float(min(max(int(frame) - int(self.start), 0), span)) / span


@dataclass(eq=False)
class Timing:
    """Shots: range, frame, cameras and shot notes. Shot notes are notes like in the Notes tab (scene.Note:
    heading in space, leader lines to the flipbook planes, board) – only their own, separate from the normal ones.
    Motion beats (Calisto Create timeline slice) bind components to spans with path + camera planning."""
    start: int = START
    end: int = END
    fps: int = FPS
    frame: int = START  # current frame (stored, not an undo step)
    cam_size: float = 1.0  # global icon size of the cameras
    onion: bool = True
    interp: str = "step"  # export of the cameras: "step" (stepped, as in Mule) or "linear" (export dialog)
    notes: list = field(default_factory=list)  # shot notes (scene.Note)
    global_note: object = field(default_factory=_global_note)  # global shot note ("Global" panel, not in space)
    cams: list = field(default_factory=list)  # TimingCam, each attached to a note (cam.note)
    motions: list = field(default_factory=list)  # MotionBeat – component/path/camera planning spans
    active_motion: int | None = None  # id of the beat selected in the timeline

    def all_notes(self):
        return [self.global_note] + list(self.notes)

    def motion_by_id(self, mid):
        return next((m for m in self.motions if m.id == mid), None)

    def motion_at(self, frame):
        """Beats that cover frame, earliest start first."""
        return [m for m in sorted(self.motions, key=lambda x: (x.start, x.id)) if m.covers(frame)]

    def active_beat(self):
        return self.motion_by_id(self.active_motion) if self.active_motion is not None else None

    def note_by_id(self, nid):
        return next((n for n in self.all_notes() if n.id == nid), None)

    def cameras(self):
        """[(note, camera)] of all cameras – in note order (global first)."""
        order = {n.id: k for k, n in enumerate(self.all_notes())}
        cams = sorted(self.cams, key=lambda c: order.get(c.note, len(order)))
        return [(self.note_by_id(c.note) or self.global_note, c) for c in cams]

    def cams_of(self, note):
        return [c for c in self.cams if c.note == note.id]

    def camera_by_id(self, cid):
        return next((c for c in self.cams if c.id == cid), None)

    def note_of(self, cam):
        return self.note_by_id(cam.note) if cam is not None else None

    def clamp(self, frame):
        return int(min(max(int(frame), self.start), self.end))


# --- Cameras on the boards --------------------------------------------------
def shot_ref(entry):
    """(camera id, pose id) of a board entry or None."""
    s = entry.get("shot") if entry.get("kind") == "image" else None
    return (int(s["cam"]), int(s["pose"])) if s else None


def pose_by_id(cam, pid):
    return next((p for p in cam.poses if p.id == pid), None)


def sync_boards(tm):
    """The board is authoritative: poses whose screenshot is no longer on any board drop out (cameras without
    a pose entirely); a camera belongs to the note on whose board its images lie. Entries whose camera no
    longer exists become normal images. -> True if something changed."""
    seen, where, changed = {}, {}, False
    cams = {c.id: c for c in tm.cams}
    for note in tm.all_notes():
        items = note.board.get("items", [])
        fixed = None
        for k, entry in enumerate(items):
            ref = shot_ref(entry)
            if ref is None:
                continue
            cam = cams.get(ref[0])
            if cam is None or pose_by_id(cam, ref[1]) is None:
                fixed = fixed if fixed is not None else [dict(i) for i in items]
                fixed[k].pop("shot", None)
                continue
            seen.setdefault(cam.id, set()).add(ref[1])
            where.setdefault(cam.id, note.id)
        if fixed is not None:
            note.board = {**note.board, "items": fixed}
            changed = True
    keep = []
    for cam in tm.cams:
        poses = [p for p in cam.poses if p.id in seen.get(cam.id, ())]
        if len(poses) != len(cam.poses):
            cam.poses, changed = poses, True
        if not poses:
            changed = True
            continue
        if cam.note != where[cam.id]:
            cam.note, changed = where[cam.id], True
        keep.append(cam)
    tm.cams = keep
    return changed


def shot_entries(tm, cam):
    """[(note, index, entry)] of the screenshots of a camera."""
    out = []
    for note in tm.all_notes():
        for k, entry in enumerate(note.board.get("items", [])):
            ref = shot_ref(entry)
            if ref is not None and ref[0] == cam.id:
                out.append((note, k, entry))
    return out


def note_layers(scene, note):
    """Flipbook planes to which a shot note is attached: via its leader lines (line of a drawing or point on
    the plane) and the octopus arms – in plane order."""
    planes = {p.id: p for p in flipbook_planes(scene)}
    ids = set()
    for a in note.anchors:
        if a.curve_id is not None:
            c = scene.curve_by_id(a.curve_id)
            if c is not None and c.plane_id in planes:
                ids.add(c.plane_id)
        else:
            for p in planes.values():  # point on a flipbook plane
                if abs(float((np.asarray(a.point, float) - p.origin) @ p.normal)) < 1e-4 * max(p.size_u, p.size_v, 1.0):
                    rel = np.asarray(a.point, float) - p.origin
                    if abs(rel @ p.axis_u) <= p.size_u / 2 + 1e-6 and abs(rel @ p.axis_v) <= p.size_v / 2 + 1e-6:
                        ids.add(p.id)
        for m in a.members:
            if m.obj_id in planes:
                ids.add(m.obj_id)
            elif m.curve_id is not None:
                c = scene.curve_by_id(m.curve_id)
                if c is not None and c.plane_id in planes:
                    ids.add(c.plane_id)
            else:
                s = next((s for s in scene.strokes if s.id == m.obj_id and s.flipbook is not None), None)
                if s is not None and s.flipbook in planes:
                    ids.add(s.flipbook)
    return [p for pid, p in planes.items() if pid in ids]


# --- Cameras -----------------------------------------------------------------
def pose_index_at(cam, frame):
    """Index of the pose that applies at frame (before the first: the first)."""
    frames = [p.frame for p in cam.poses]
    return max(bisect.bisect_right(frames, int(frame)) - 1, 0)


def pose_at(cam, frame):
    return cam.poses[pose_index_at(cam, frame)] if cam.poses else None


def new_camera(name, frame, state, mm, note=None):
    target, dist, rot = state
    return TimingCam(name, [Pose(int(frame), np.array(target, float), float(dist), np.array(rot, float), float(mm))],
                     note=note)


def change_pose(cam, frame, state, pose=None):
    """⟲ Change pose: the pose (otherwise the one applying at frame) gets this view (its frame stays) – no
    animation."""
    p = pose if pose is not None else pose_at(cam, frame)
    p.target, p.dist, p.rot = np.array(state[0], float), float(state[1]), np.array(state[2], float)
    return p


def add_pose(cam, frame, state):
    """➜ New pose from frame (move); if one already exists there, it is overwritten. It inherits the focal length from the
    pose that applied at frame so far."""
    frame = int(frame)
    p = pose_at(cam, frame)
    if p.frame == frame:
        return change_pose(cam, frame, state)
    new = Pose(frame, np.array(state[0], float), float(state[1]), np.array(state[2], float), p.mm)
    cam.poses.append(new)
    cam.poses.sort(key=lambda q: q.frame)
    return new


def remove_pose(cam, pose):
    """Delete pose – the previous one then applies longer. The last pose stays (then delete the camera)."""
    if len(cam.poses) > 1 and pose in cam.poses:
        cam.poses.remove(pose)
        return True
    return False


def set_focal(pose, mm, keep_framing=True):
    """Change the focal length of the pose. keep_framing: hold the framing – the target point stays, the distance is
    recalculated so that the subject at the target point stays the same size (only the perspective changes); otherwise a pure zoom."""
    mm = float(min(max(mm, MM_RANGE[0]), MM_RANGE[1]))
    if keep_framing:
        pose.dist = float(pose.dist) * mm_to_tan(pose.mm) / mm_to_tan(mm)
    pose.mm = mm
    return pose


def pose_view(pose):
    """Camera of a pose in the format of the board screenshots (note_board.camera_view)."""
    return {"target": [round(float(x), 5) for x in pose.target], "dist": round(float(pose.dist), 5),
            "rot": [[round(float(x), 6) for x in row] for row in np.asarray(pose.rot, float)], "ortho": False}


def is_move(cam):
    """Move: more than one pose."""
    return len(cam.poses) > 1


def unique_cam_name(timing, base="Kamera"):
    names = {c.name for _, c in timing.cameras()}
    n = 1
    while f"{base} {n}" in names:
        n += 1
    return f"{base} {n}"


# --- Flipbook planes ---------------------------------------------------------
def flipbook_planes(scene):
    return [p for p in scene.planes if getattr(p, "flipbook", False)]


def flip_keys(scene):
    """{plane id: sorted frames of its drawings} – from the paint strokes (Curve.frame) and the empty drawings
    (Plane.blank_keys)."""
    keys = {p.id: set(p.blank_keys) for p in flipbook_planes(scene)}
    for s in scene.strokes:
        for c in s.curves:
            if c.frame is not None and c.plane_id in keys:
                keys[c.plane_id].add(int(c.frame))
    return {pid: sorted(fr) for pid, fr in keys.items()}


def held_key(frames, t):
    """Drawing that applies at t (last <= t), otherwise None – before the first drawing the plane is empty."""
    i = bisect.bisect_right(frames, int(t)) - 1
    return frames[i] if i >= 0 else None


def neighbor_keys(frames, t):
    """(previous, next) drawing around the one currently applying – for onion skin and ↑/↓."""
    shown = held_key(frames, t)
    if shown is None:
        return None, (frames[0] if frames else None)
    i = frames.index(shown)
    return (frames[i - 1] if i > 0 else None), (frames[i + 1] if i + 1 < len(frames) else None)


def shown_keys(scene, t, keys=None):
    """{plane id: applying drawing or None} at frame t."""
    keys = flip_keys(scene) if keys is None else keys
    return {pid: held_key(fr, t) for pid, fr in keys.items()}


def key_curves(scene, plane_id, frame):
    return [(s, c) for s in scene.strokes for c in s.curves if c.plane_id == plane_id and c.frame == frame]


def move_key(scene, plane, old, new):
    """Move drawing from frame old to new (change timing without repainting). If a drawing already exists at new,
    new then belongs to both – that is why the timeline calls this only with a free target."""
    old, new = int(old), int(new)
    for _, c in key_curves(scene, plane.id, old):
        c.frame = new
    if old in plane.blank_keys:
        plane.blank_keys = sorted({k for k in plane.blank_keys if k != old} | {new})


def delete_key(scene, plane, frame):
    """Delete drawing (strokes and empty drawing) – the previous one then applies longer."""
    for s, c in key_curves(scene, plane.id, frame):
        s.curves.remove(c)
    plane.blank_keys = [k for k in plane.blank_keys if k != int(frame)]


def copy_key(scene, plane, src, dst):
    """Copy drawing src as a new drawing at dst (keep painting: the pose stays there, you erase and paint on it).
    -> {id(old stroke): new stroke}."""
    from .scene import Curve, inherit
    stroke = scene.flip_stroke(plane)
    mapping = {}
    for _, c in key_curves(scene, plane.id, src):
        new = inherit(Curve(c.points.copy(), c.plane_id, c.widths.copy()), c)
        new.normals = None if c.normals is None else c.normals.copy()
        new.frame = int(dst)
        stroke.curves.append(new)
        mapping[id(c)] = new
    return mapping


def add_blank_key(plane, frame):
    """Empty drawing at frame (something disappears)."""
    plane.blank_keys = sorted(set(plane.blank_keys) | {int(frame)})


# --- Motion beats (component bind + path + camera planning) -----------------
def _kind_of(obj):
    """Classify a scene object for MotionBeat.target_kind."""
    from .scene import MeshObject, NullObject, Plane, StrokeObject, SurfaceObject
    if obj is None:
        return "null"
    if isinstance(obj, NullObject):
        return "null"
    if isinstance(obj, MeshObject):
        return "mesh"
    if isinstance(obj, SurfaceObject):
        return "surface"
    if isinstance(obj, StrokeObject):
        return "stroke"
    if isinstance(obj, Plane):
        return "plane"
    return "null"


def resolve_bind_target(scene, selection_ids=None):
    """Pick (kind, id, label) from selection / active objects. Defaults to a free null proxy."""
    ids = list(selection_ids or [])
    for oid in ids:
        obj = scene.object_by_id(oid)
        if obj is None:
            continue
        # Flipbook planes stay drawing surfaces – bind the stroke/mesh/null instead when possible.
        if getattr(obj, "flipbook", False):
            continue
        if getattr(obj, "flipbook", None) is not None and getattr(obj, "curves", None) is not None:
            # stroke owned by a flipbook: usable as a path source, but prefer non-flip as target
            pass
        kind = _kind_of(obj)
        name = getattr(obj, "name", None) or f"{kind} {oid}"
        return kind, obj.id, name
    mesh = scene.active_mesh
    if mesh is not None:
        return "mesh", mesh.id, getattr(mesh, "name", "Mesh")
    stroke = scene.active_stroke
    if stroke is not None and getattr(stroke, "flipbook", None) is None:
        return "stroke", stroke.id, getattr(stroke, "name", "Stroke")
    plane = scene.active_plane
    if plane is not None and not getattr(plane, "flipbook", False) and plane not in getattr(scene, "presets", ()):
        return "plane", plane.id, getattr(plane, "name", "Plane")
    return "null", None, "Proxy"


def new_motion(tm, start, end, kind="null", target_id=None, annotation="", text=""):
    beat = MotionBeat(int(start), int(end), kind if kind in MOTION_KINDS else "null",
                      None if target_id is None else int(target_id),
                      (annotation or "").strip(), (text or "").strip())
    beat.clamp_range(tm)
    tm.motions.append(beat)
    tm.active_motion = beat.id
    return beat


def bind_selection(scene, selection_ids=None, start=None, end=None, annotation="", text=""):
    """Create a motion beat for the current selection (or null proxy) spanning start..end."""
    tm = scene.timing
    start = tm.frame if start is None else int(start)
    end = min(tm.end, start + max(tm.fps, 1) - 1) if end is None else int(end)
    kind, tid, label = resolve_bind_target(scene, selection_ids)
    ann = annotation.strip() if annotation else label
    return new_motion(tm, start, end, kind, tid, ann, text)


def set_motion_path(beat, scene, stroke_id=None, plane_id=None):
    """Attach a path source: a stroke and/or a flipbook plane (held drawing)."""
    beat.path_stroke_id = int(stroke_id) if stroke_id is not None else None
    beat.path_plane_id = int(plane_id) if plane_id is not None else None
    if beat.path_stroke_id is None and beat.path_plane_id is None:
        # Convenience: selected/active stroke, else active flipbook plane
        stroke = scene.active_stroke
        if stroke is not None:
            beat.path_stroke_id = stroke.id
        elif scene.active_plane is not None and scene.active_plane.flipbook:
            beat.path_plane_id = scene.active_plane.id
    return beat


def capture_motion_camera(beat, state, mm):
    """Store the viewport camera angle for shot planning on this beat."""
    target, dist, rot = state
    beat.camera = {"target": _r(target), "dist": round(float(dist), 6),
                   "rot": _r(rot), "mm": round(float(mm), 4)}
    return beat


def camera_state(beat):
    """(target, dist, rot, mm) from a stored beat camera, or None."""
    cam = beat.camera
    if not cam:
        return None
    return (np.array(cam["target"], float), float(cam["dist"]),
            np.array(cam["rot"], float).reshape(3, 3), float(cam.get("mm", 35.0)))


def _curve_points(curves):
    """Concatenate curve polylines into one path (skip tiny stubs)."""
    pts = []
    for c in curves:
        p = np.asarray(c.points, float)
        if len(p) < 2:
            continue
        if pts and np.allclose(pts[-1], p[0], atol=1e-5):
            pts.extend(p[1:].tolist())
        else:
            pts.extend(p.tolist())
    return np.asarray(pts, float) if pts else np.zeros((0, 3))


def path_points(scene, beat, frame=None):
    """World-space polyline for a beat: from path stroke, else flipbook held drawing, else empty."""
    curves = []
    t = scene.timing.frame if frame is None else int(frame)
    if beat.path_stroke_id is not None:
        stroke = scene.object_by_id(beat.path_stroke_id)
        if stroke is not None and getattr(stroke, "curves", None) is not None:
            always = [c for c in stroke.curves if c.frame is None]
            timed = [c for c in stroke.curves if c.frame is not None]
            if timed:
                frames = sorted({int(c.frame) for c in timed})
                held = held_key(frames, t)
                curves = [c for c in timed if c.frame == held] if held is not None else []
            if not curves:
                curves = always
            if not curves:
                curves = list(stroke.curves)
    if not curves and beat.path_plane_id is not None:
        shown = shown_keys(scene, t).get(beat.path_plane_id)
        if shown is not None:
            curves = [c for _, c in key_curves(scene, beat.path_plane_id, shown)]
        else:
            keys = flip_keys(scene).get(beat.path_plane_id, [])
            if keys:
                curves = [c for _, c in key_curves(scene, beat.path_plane_id, keys[0])]
    return _curve_points(curves)


def _arc_resample(pts, u):
    """Point at normalized arc-length u in [0, 1] along polyline pts."""
    if len(pts) == 0:
        return None
    if len(pts) == 1:
        return pts[0].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-12:
        return pts[0].copy()
    dist = float(np.clip(u, 0.0, 1.0)) * total
    acc = 0.0
    for i, length in enumerate(seg):
        if acc + length >= dist - 1e-12:
            t = 0.0 if length < 1e-12 else (dist - acc) / length
            return pts[i] * (1 - t) + pts[i + 1] * t
        acc += length
    return pts[-1].copy()


def proxy_at(scene, beat, frame):
    """World position of the aircraft/part proxy along the beat path at frame, or None if no path."""
    pts = path_points(scene, beat, frame=frame)
    if len(pts) == 0:
        return None
    return _arc_resample(pts, beat.progress(frame))


def target_label(scene, beat):
    if beat.annotation:
        return beat.annotation
    if beat.target_id is not None:
        obj = scene.object_by_id(beat.target_id)
        if obj is not None:
            return getattr(obj, "name", None) or f"{beat.target_kind} {beat.target_id}"
    return "Proxy" if beat.target_kind == "null" else beat.target_kind


def remove_motion(tm, beat):
    if beat in tm.motions:
        tm.motions.remove(beat)
    if tm.active_motion == beat.id:
        tm.active_motion = tm.motions[-1].id if tm.motions else None
    return True


# --- Saving ---------------------------------------------------------------
def _r(a, digits=6):
    return [round(float(x), digits) for x in np.ravel(a)]


def to_dict(timing):
    """Range, settings and cameras – the shot notes are written by the scene (like its normal notes)."""
    def cam(c):
        return {"id": c.id, "name": c.name, "size": round(float(c.size), 4), "note": c.note,
                **({"color": [round(float(x), 4) for x in c.color]} if c.color else {}),
                "poses": [{"id": p.id, "frame": p.frame, "target": _r(p.target), "dist": round(float(p.dist), 6),
                           "rot": _r(p.rot), "mm": round(float(p.mm), 4)} for p in c.poses]}

    def motion(m):
        return {"id": m.id, "start": int(m.start), "end": int(m.end), "kind": m.target_kind,
                "target": m.target_id, "annotation": m.annotation, "text": m.text,
                **({"path_stroke": m.path_stroke_id} if m.path_stroke_id is not None else {}),
                **({"path_plane": m.path_plane_id} if m.path_plane_id is not None else {}),
                **({"camera": m.camera} if m.camera else {})}

    return {"start": timing.start, "end": timing.end, "fps": timing.fps, "frame": timing.frame,
            "cam_size": round(float(timing.cam_size), 4), "onion": bool(timing.onion), "interp": timing.interp,
            "cams": [cam(c) for c in timing.cams],
            **({"motions": [motion(m) for m in timing.motions]} if timing.motions else {}),
            **({"active_motion": timing.active_motion} if timing.active_motion is not None else {})}


def from_dict(data):
    """Timing from the scene file (without the notes, the scene reads those). Files of the first prototype (v0.99.135–140:
    timing notes with text, range and cameras) -> t.legacy = [(note dict, global?, [(camera, {frame: image})])],
    the scene turns these into shot notes with a board."""
    used = []

    def cam(d):
        poses = [Pose(int(p["frame"]), np.array(p["target"], float), float(p["dist"]),
                      np.array(p["rot"], float).reshape(3, 3), float(p.get("mm", 35.0)), int(p.get("id", 0)) or _new_id())
                 for p in d.get("poses", [])]
        poses.sort(key=lambda q: q.frame)
        c = TimingCam(d.get("name", "Kamera"), poses, tuple(d["color"]) if d.get("color") else None,
                      float(d.get("size", 1.0)), d.get("note"), int(d.get("id", 0)) or _new_id())
        used.extend([c.id] + [p.id for p in poses])
        return c

    def motion(d):
        kind = d.get("kind", "null")
        if kind not in MOTION_KINDS:
            kind = "null"
        m = MotionBeat(int(d.get("start", START)), int(d.get("end", START)), kind,
                       None if d.get("target") is None else int(d["target"]),
                       d.get("annotation", "") or "", d.get("text", "") or "",
                       None if d.get("path_stroke") is None else int(d["path_stroke"]),
                       None if d.get("path_plane") is None else int(d["path_plane"]),
                       d.get("camera"), int(d.get("id", 0)) or _new_id())
        used.append(m.id)
        return m

    t = Timing(int(data.get("start", START)), int(data.get("end", END)), int(data.get("fps", FPS)),
               int(data.get("frame", START)), float(data.get("cam_size", 1.0)), bool(data.get("onion", True)))
    t.frame = t.clamp(t.frame)
    t.interp = data.get("interp", "step") if data.get("interp") in ("step", "linear") else "step"
    t.cams = [cam(c) for c in data.get("cams", []) if c.get("poses")]
    t.motions = [motion(m) for m in data.get("motions", [])]
    for m in t.motions:
        m.clamp_range(t)
    am = data.get("active_motion")
    t.active_motion = int(am) if am is not None and t.motion_by_id(int(am)) else (t.motions[-1].id if t.motions else None)
    legacy = []
    if "cams" not in data:  # first prototype
        for d, glob in [(data.get("global_note") or {}, True)] + [(n, False) for n in data.get("notes", [])]:
            cams = []
            for cd in d.get("cameras", []):
                if cd.get("poses"):
                    c = cam(cd)
                    cams.append((c, {int(i["frame"]): i["file"] for i in cd.get("images", [])}))
            if glob and not cams and not (d.get("text") or "").strip():
                continue
            legacy.append((d, glob, cams))
    t.legacy = legacy
    _bump_ids(used)
    return t


def frustum(pose, size, aspect=GATE):
    """Pyramid of a pose in world coordinates: (tip, 4 corners of the image plane) – length size, opening from the
    focal length (long focal length = narrow, long pyramid)."""
    rot = np.asarray(pose.rot, float)
    right, up, back = rot[:, 0], rot[:, 1], rot[:, 2]
    eye = np.asarray(pose.target, float) + float(pose.dist) * back
    tv = mm_to_tan(pose.mm)
    th = tv * aspect
    center = eye - back * size
    corners = [center + right * (sx * th * size) + up * (sy * tv * size) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return eye, corners


def base_size(pose):
    """Base size of the camera icon in world units: a share of the distance to the target point (fits every scene)."""
    return max(float(pose.dist) * 0.12, 1e-3)


def pose_range_label(cam, pose, end):
    """“F1003–F1019”: frames in which the pose applies (until just before the next, the last until the end); a camera with
    only one pose is a still – only its frame."""
    if len(cam.poses) < 2:
        return frame_label(pose.frame)
    k = cam.poses.index(pose)
    last = cam.poses[k + 1].frame - 1 if k + 1 < len(cam.poses) else end
    return frame_label(pose.frame) if last <= pose.frame else f"{frame_label(pose.frame)}–{frame_label(last)}"


def frame_label(frame):
    return f"F{int(frame)}"


def gate_fit(tan_v, width, height):
    """Viewport (width x height) through a camera with vertical tan_v in the image format GATE: (tan_half of the viewport,
    rectangle of the image format in pixels (x, y, w, h)). Wider than GATE: the image format fills the height, otherwise the width."""
    aspect = width / max(height, 1)
    if aspect >= GATE:
        t = tan_v
        gh = float(height)
    else:
        t = tan_v * GATE / aspect
        gh = height * tan_v / t
    gw = gh * GATE
    return t, ((width - gw) / 2, (height - gh) / 2, gw, gh)


def vfov_deg(mm):
    return math.degrees(2 * math.atan(mm_to_tan(mm)))
