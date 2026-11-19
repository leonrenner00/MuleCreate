"""Scene data: planes, stroke objects with their curves, undo/redo, saving/loading.

Undo works with snapshots of the whole scene. The scene is small, this is
simpler and more robust than individual commands (pure caches do not travel along, see SurfaceObject._NO_COPY).

The surface methods of the scene (fill surfaces, lofts, curve surfaces, sync) are in scene_surfaces.py.
"""
import base64
import contextlib
import copy
import gc
import itertools
import json
import operator
import os
import time
import weakref
import zlib
from dataclasses import dataclass, field

import numpy as np


from . import components, fill, geometry, light, loft, network, notes as notes_geo, timing, video
from . import counters, lru
from .geometry import Geometry
from .i18n import tr

_ids = itertools.count(1)
FILE_FORMAT = "sketch-machine-scene"
FILE_VERSION = 5  # 2: meshes, 3: stroke widths, 4: surfaces, 5: surfaces are pure drawing surfaces
PAINT_LIFT = 0.3  # painting on meshes: this far (share of the stroke width) above the surface
UNDO_LIMIT = 150  # this many undo steps (each is a copy of the scene – otherwise memory grows)


def _carry(value, memo):
    """Cache of a surface for its copy: every object that the copy itself also has (deepcopy memo) replaced by
    its counterpart; lists/tuples traversed likewise; everything else (computed arrays, numbers, bytes) shared.
    A comparison by identity against the state of the copy then hits exactly when it would have hit in the original."""
    hit = memo.get(id(value))
    if hit is not None:
        return hit
    t = type(value)
    if t is tuple or t is list:
        get = memo.get  # look up leaves right here (long lists: hundreds of entries per surface)
        out = [_carry(v, memo) if type(v) is tuple or type(v) is list else get(id(v), v) for v in value]
        return tuple(out) if t is tuple else out
    return value


_LOOPS = {}  # triangles of a fill mesh (by identity) -> border points in loop direction


def _boundary_loop(tris):
    """Border points of a triangle mesh in loop direction (edges belonging to only one triangle) – or None if
    the border is not exactly one closed run. Remembered per mesh (fill surfaces share their triangle array)."""
    hit = _LOOPS.get(id(tris))
    if hit is not None and hit[0] is tris:
        return hit[1]
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]).astype(np.int64)
    n = int(e.max()) + 1 if len(e) else 0
    und = np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])
    order = np.argsort(und, kind="stable")
    s = und[order]
    single = np.ones(len(s), bool)
    same = s[1:] == s[:-1]
    single[1:] &= ~same
    single[:-1] &= ~same
    border = e[order[single]]
    nxt = {int(a): int(b) for a, b in border}
    loop = None
    if len(nxt) == len(border) and len(border) >= 3:
        start = int(border[0][0])
        loop, at = [start], nxt.get(start)
        while at is not None and at != start and len(loop) <= len(border):
            loop.append(at)
            at = nxt.get(at)
        if at != start or len(loop) != len(border):
            loop = None
    if len(_LOOPS) > 512:
        _LOOPS.clear()
    _LOOPS[id(tris)] = (tris, None if loop is None else np.array(loop))
    return _LOOPS[id(tris)][1]


def _flat_fill(netmesh):
    """Fill surface that is (practically) flat: only its border, triangulated – instead of the dense mesh needed
    for deforming. Flat means: no mesh point farther than loft.ADAPT_TOL of the surface size from the best-fit plane.
    The border is the same as with the dense mesh. None: not flat (show the dense mesh)."""
    verts, tris = netmesh[0], netmesh[1]
    if len(verts) < 4 or len(tris) < 3:
        return None
    center = verts.mean(axis=0)
    size = float(np.linalg.norm(np.ptp(verts, axis=0)))
    if size <= 1e-12:
        return None
    normal = np.linalg.svd(verts - center, full_matrices=False)[2][2]
    if float(np.abs((verts - center) @ normal).max()) > loft.ADAPT_TOL * size:
        return None
    loop = _boundary_loop(tris)
    if loop is None:
        return None
    return fill.triangulate_loop(verts[loop], resample=False)


def _share_array(array, memo):
    """deepcopy of a numpy array in the undo snapshot: the same array, write-protected (Scene._snapshot)."""
    if array.dtype != object:
        array.flags.writeable = False
        return array
    return array.__deepcopy__(memo)


_MISSING = object()
_ATOMS = frozenset((int, float, str, bool, bytes, type(None), complex))


def _same(a, b, memo):
    """Does a value of the live scene still have exactly the state of its counterpart in the undo copy? Lists, tuples and
    dictionaries piece by piece; arrays only by identity (the counterpart shares them; a new array counts as changed);
    lines via memo (id of the live line -> its valid counterpart). Anything unknown counts as changed."""
    if a is b:
        return True
    t = type(a)
    if t is not type(b):
        return False
    if t is list or t is tuple:
        if len(a) != len(b):
            return False
        get = memo.get
        for x, y in zip(a, b):
            if x is not y:
                if type(x) is Curve:  # (right here: surfaces hold thousands of grid rows)
                    if get(id(x)) is not y:
                        return False
                elif not _same(x, y, memo):
                    return False
        return True
    if t is dict:
        if len(a) != len(b):
            return False
        for k, x in a.items():
            y = b.get(k, _MISSING)
            if y is _MISSING or (x is not y and not _same(x, y, memo)):
                return False
        return True
    if t is Curve:
        return memo.get(id(a)) is b
    if t in _ATOMS:
        return a == b
    return False


def _surface_same(s, twin, memo):
    """Surface unchanged compared to its counterpart: every field compared piece by piece – surfaces change lists and
    dictionaries in place too, a counter is not enough here. The caches count as well: they hold
    lines; a counterpart with lines from earlier would hit nothing after being restored (every surface would
    recompute) and, with the sync stamp, overlook changes to the current lines."""
    if type(s) is not type(twin):
        return False
    skip = type(s)._DROP
    d, t = s.__dict__, twin.__dict__
    n = 0
    for k, v in d.items():
        if k in skip:
            continue
        n += 1
        w = t.get(k, _MISSING)
        if w is _MISSING or (v is not w and not _same(v, w, memo)):
            return False
    return n == sum(1 for k in t if k not in skip)


_SKIPS = {}


def _twin_skip(cls):
    """Verify mode: fields of a surface that do not count when comparing two copies – pure caches (valid
    by identity of their inputs or not at all); the sync stamp and its lines do count."""
    hit = _SKIPS.get(cls)
    if hit is None:
        hit = _SKIPS[cls] = frozenset(cls._NO_COPY) - {"_dep_curves", "_sync_ok"}
    return hit


def _diff(a, b, seen, path):
    """Verify mode (Scene.VERIFY_UNDO): first place where two scene states differ, or None."""
    if a is b:
        return None
    t = type(a)
    if t is not type(b):
        return f"{path}: {t.__name__} / {type(b).__name__}"
    if t is np.ndarray:
        if a.shape != b.shape:
            return f"{path}: Array {a.shape} / {b.shape}"
        if a.dtype == object:
            return next((r for r in (_diff(x, y, seen, path) for x, y in zip(a.ravel(), b.ravel())) if r), None)
        return None if np.array_equal(a, b, equal_nan=a.dtype.kind in "fc") else f"{path}: Array"
    if t is list or t is tuple:
        if len(a) != len(b):
            return f"{path}: Länge {len(a)} / {len(b)}"
        return next((r for r in (_diff(x, y, seen, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))) if r), None)
    if t is dict:
        if a.keys() != b.keys():
            return f"{path}: Schlüssel {sorted(map(str, a.keys() ^ b.keys()))}"
        return next((r for r in (_diff(v, b[k], seen, f"{path}.{k}") for k, v in a.items()) if r), None)
    d = getattr(a, "__dict__", None)
    if d is not None and not isinstance(a, np.generic):
        if (id(a), id(b)) in seen:
            return None
        seen.add((id(a), id(b)))
        skip = _twin_skip(t) if isinstance(a, SurfaceObject) else ("_knot_t",) if t is Curve else ()
        return _diff({k: v for k, v in d.items() if k not in skip},
                     {k: v for k, v in b.__dict__.items() if k not in skip}, seen, f"{path}<{t.__name__}>")
    try:
        ok = bool(a == b) or (a != a and b != b)
    except Exception:
        ok = False
    return None if ok else f"{path}: {a!r} / {b!r}"[:300]


def _freeze_history():
    """Take undo copies out of garbage collection (gc): each copy is thousands of Python objects that the cycle
    collector would otherwise search again on every big pass – after a long session (150 steps, > 1 million
    objects) drawing stuttered (single frames 30 -> 100 ms). The copies contain no cycles, so they are still freed
    when discarded. Collect once beforehand so that no real garbage gets frozen along (afterwards this is
    fast: frozen things are no longer searched)."""
    gc.collect()
    gc.freeze()


def _note_dict(n, r, rel):
    """Note (normal or shot) for the scene file."""
    return {
        "id": n.id, "uid": n.uid, "title": n.title, "pos": r(n.pos), "size": n.size,
        "yaw": None if n.yaw is None else round(float(n.yaw), 5),
        "anchors": [{"point": r(a.point), "curve": a.curve_id, "t": round(float(a.t), 6),
                     **({"members": [{"obj": m.obj_id, "curve": m.curve_id, "t": round(float(m.t), 6),
                                      "point": None if m.point is None else r(m.point)} for m in a.members]}
                        if a.members else {})} for a in n.anchors],
        "board": _board_videos(n.board, rel),
        **({"review": n.review} if n.review else {}),
    }


def _note_from(nd, id_map, absolute):
    """Note from the scene file – rewrite line ids (unknown: 0 -> link via position)."""
    arr = lambda a: np.array(a, float)
    anchors = [NoteAnchor(arr(a["point"]), None if a.get("curve") is None else id_map.get(("curve", a["curve"]), 0),
                          float(a.get("t", 0.0)),
                          members=[NoteMember(id_map.get(m["obj"], 0),  # unknown: 0 = deleted
                                              None if m.get("curve") is None else id_map.get(("curve", m["curve"]), 0),
                                              None if m.get("point") is None else arr(m["point"]),
                                              float(m.get("t", 0.0))) for m in a.get("members", [])])
               for a in nd.get("anchors", [])]
    return Note(nd.get("title", ""), arr(nd["pos"]), nd.get("size", "m"), nd.get("yaw"), anchors,
                _board_videos(nd.get("board") or {}, absolute), review=nd.get("review") or None,
                **({"uid": str(nd["uid"])} if nd.get("uid") else {}))


def _board_videos(board, convert):
    """Board with rewritten video paths (saving: relative to the scene, loading: absolute); without videos the same."""
    items = board.get("items") or ()
    if not any(it.get("video") for it in items):
        return board
    return {**board, "items": [{**it, "video": {**it["video"], "path": convert(it["video"]["path"])}}
                               if it.get("video") else it for it in items]}


class _PaintStep:
    """Undo step for new paint strokes (light: no copy of the scene). A paint stroke changes nothing but itself
    (no points, no net, shapes no surface), so it is enough to take it out again. curves: the strokes
    themselves while they are taken out (redo). The binding to the surface is restored by the sync afterwards."""
    __slots__ = ("stroke_id", "ids", "curves")

    def __init__(self, stroke_id, ids):
        self.stroke_id, self.ids, self.curves = stroke_id, ids, None


class _Selection:
    """Undo step that only holds a selection (light: no copy of the scene)."""
    __slots__ = ("state",)

    def __init__(self, state):
        self.state = state
_PLAIN = frozenset((int, float, bool, str, bytes))  # immutable field values: simply taken over when copying
SURF_EPOCH = [0]  # rises when a field of any surface is set anew (Scene.sync_lofts, lazy)
PLANE_EPOCH = [0]  # rises when a line gets or changes its drawing surface (plane_id) (Scene.curves_on)
POINTS_EPOCH = [0]  # rises when any line or grid row gets new points (is never reset)

# change log (rework plan step 3: report changes instead of asking). Every line, grid row and surface enters itself
# here when one of its fields is set: {id(object): (sequence number, weak reference)}, the most recently
# changed at the back. Whoever wants to know what changed since their last look remembers the sequence
# number and reads from the back up to it (touched_since) – instead of asking every object of the scene.
TOUCH_SEQ = [0]
TOUCH_FLOOR = [0]  # entries up to this number are already discarded: whoever asks older must look at everything
TOUCH_LIMIT = 200000
_TOUCHED = {}


def _touch(obj):
    n = TOUCH_SEQ[0] = TOUCH_SEQ[0] + 1
    key = id(obj)
    old = _TOUCHED.pop(key, None)  # (insert anew: this keeps the log ordered by number)
    # (during a drag the same object reports in every frame: reuse its reference)
    _TOUCHED[key] = (n, old[1] if old is not None and old[1]() is obj else weakref.ref(obj))
    if len(_TOUCHED) > TOUCH_LIMIT:
        for k in list(_TOUCHED)[:TOUCH_LIMIT // 2]:
            TOUCH_FLOOR[0] = _TOUCHED.pop(k)[0]


def touched_since(seq):
    """Lines, grid rows and surfaces on which a field was set since the sequence number seq (live
    objects; what no longer exists is missing). None: the log does not reach back that far – then everything counts as changed."""
    if seq < TOUCH_FLOOR[0]:
        return None
    out = []
    for key in reversed(_TOUCHED):
        n, ref = _TOUCHED[key]
        if n <= seq:
            break
        obj = ref()
        if obj is not None:
            out.append(obj)
    return out


def _epoch_key(fn):
    """Remember the key of a surface (_cache_key) as long as nothing changed anywhere that could change it:
    no line or grid row has new points (POINTS_EPOCH), no field of a surface was set (SURF_EPOCH),
    the same row list in the same length, the same resolution. The key is asked several times per frame per surface
    and so far compared all rows every time (harbor 12.11.: 2,300 asks per frame, ~27 ms)."""
    def cache_key(self):
        d = self.__dict__
        fast = d.get("_key_fast")
        rows = self.curves
        if (fast is not None and fast[0] == POINTS_EPOCH[0] and fast[1] == SURF_EPOCH[0] and fast[2] is rows
                and fast[3] == len(rows) and fast[4] == SurfaceObject.display_res and fast[5] == SurfaceObject.auto_res):
            if SurfaceObject.VERIFY_KEY:
                assert fast[6] == fn(self), f"remembered key of \"{self.name}\" is no longer valid"
            return fast[6]
        key = fn(self)
        d["_key_fast"] = (POINTS_EPOCH[0], SURF_EPOCH[0], rows, len(rows), SurfaceObject.display_res,
                          SurfaceObject.auto_res, key)
        return key
    return cache_key
DEFAULT_WIDTH = 0.02  # stroke width for lines without a value (old files), in world units


IDS_EPOCH = [0]  # rises with every id assigned (and when a line gets another id) – for "does not exist"


def _new_id():
    IDS_EPOCH[0] += 1
    return next(_ids)


def _json_params(params):
    """Settings of a generator as JSON (numbers rounded, lists stay lists)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, float):
            v = round(v, 5)
        elif isinstance(v, (list, tuple, np.ndarray)):
            v = [round(float(x), 5) if isinstance(x, (float, np.floating)) else
                 bool(x) if isinstance(x, (bool, np.bool_)) else x for x in v]
        out[k] = v
    return out


def _scale_points(points, factors, axes, pivot):
    """Scale points along the columns of axes around pivot."""
    m = axes @ np.diag(factors) @ axes.T
    return pivot + (points - pivot) @ m.T


NULL_KINDS = ("null", "symmetry", "array")
SYMMETRY_PLANES = {"zy": 0, "xz": 1, "xy": 2}  # mirror plane (local) -> axis along which to mirror
ARRAY_MODES = ("radial", "linear", "line")


def default_params(kind):
    """Settings of a new generator (symmetry: mirror plane and kept side; array: see generators)."""
    if kind == "symmetry":
        return {"plane": "zy", "side": 1}
    if kind == "array":
        return {"mode": "radial", "count": 6, "angle": 360.0, "offset": [1.0, 0.0, 0.0], "path": None, "up": None,
                "spin": 0.0, "rand_rot": 0.0, "rot_axes": [True, True, True], "rand_off": 0.0,
                "off_axes": [True, True, True], "seed": 1}
    return {}


def unique_name(name, taken):
    """name, or name_1, name_2 … – the first one not yet in taken."""
    if name not in taken:
        return name
    n = 1
    while f"{name}_{n}" in taken:
        n += 1
    return f"{name}_{n}"


def unique_hero_names(scene):
    """Number duplicate names of hero cameras (screenshots with a tick in the note boards) consecutively – the brief and
    set_hero_prompt need unique names. Older scenes on loading."""
    taken = set()
    for note in scene.notes + [scene.global_note]:
        items = note.board.get("items", [])
        changed = False
        for k, item in enumerate(items):
            if item.get("hero") and item.get("hero_name"):
                name = unique_name(item["hero_name"], taken)
                taken.add(name)
                if name != item["hero_name"]:
                    items[k], changed = {**item, "hero_name": name}, True
        if changed:
            note.board = {**note.board, "items": items}


@dataclass(eq=False)  # comparison by identity (numpy fields)
class NullObject:
    """Null object as in Cinema 4D: an axis cross (origin + local axes) that collects other objects under
    itself in the outliner (their parent). If you move it, all objects below go along. The geometry of the children stays
    stored in world coordinates – the null is not a coordinate system, but a handle and a group.

    kind "symmetry" / "array": generator. The children stay the only real objects; mirrored or multiplied
    geometry lives only as an instance (generators.py) until the generator is converted."""
    name: str
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))  # columns: local x, y, z (orthonormal)
    kind: str = "null"
    params: dict = field(default_factory=dict)
    id: int = field(default_factory=_new_id)
    hidden: bool = False
    parent: int | None = None
    color: tuple | None = None  # outliner: color stripe (r, g, b 0..255), applies to everything below up to the next null with a color
    binding: str | None = None  # binding level for the brief (brief.TASK_LEVELS), applies to everything below (outliner)

    scale_axes = (0, 1, 2)

    def __post_init__(self):
        self.params = {**default_params(self.kind), **(self.params or {})}

    def pivot(self):
        return self.origin.copy()

    def bounds(self):
        return self.origin.copy(), self.origin.copy()

    def local_axes(self):
        return self.axes

    def translate(self, delta):
        self.origin = self.origin + delta

    def rotate(self, rot, pivot):
        self.origin = pivot + rot @ (self.origin - pivot)
        self.axes = rot @ self.axes

    def scale(self, factors, axes, pivot):
        self.origin = _scale_points(self.origin[None], factors, axes, pivot)[0]  # axes stay (no size)

    def frame(self):
        """4x4: local -> world."""
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.axes, self.origin
        return m


@dataclass(eq=False)  # comparison by identity (numpy fields)
class Plane:
    name: str
    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    size_u: float = 10.0
    size_v: float = 10.0
    preset: bool = False
    id: int = field(default_factory=_new_id)
    hidden: bool = False  # hidden (eye in the outliner, H): not drawn, not selectable
    picture: dict | None = None  # projected image (pictures.py)
    # flipbook plane (timing): its own type – never becomes a normal plane; only editable in timing mode.
    # blank_keys: frames of empty drawings (something disappears); drawings with strokes follow from Curve.frame
    flipbook: bool = False
    blank_keys: list = field(default_factory=list)

    @property
    def normal(self):
        n = np.cross(self.axis_u, self.axis_v)
        return n / np.linalg.norm(n)

    def corners(self):
        hu, hv = self.axis_u * self.size_u / 2, self.axis_v * self.size_v / 2
        o = self.origin
        return np.array([o - hu - hv, o + hu - hv, o + hu + hv, o - hu + hv])

    def bounds(self):
        c = self.corners()
        return c.min(axis=0), c.max(axis=0)

    def contains(self, point):
        d = point - self.origin
        return abs(d @ self.axis_u) <= self.size_u / 2 and abs(d @ self.axis_v) <= self.size_v / 2

    # --- Transformable (used by the gizmo) ---
    scale_axes = (0, 1)  # planes scale only width/height, always locally

    def pivot(self):
        return self.origin

    def local_axes(self):
        return np.column_stack([self.axis_u, self.axis_v, self.normal])

    def translate(self, delta):
        self.origin = self.origin + delta

    def rotate(self, rot, pivot):
        self.origin = pivot + rot @ (self.origin - pivot)
        self.axis_u = rot @ self.axis_u
        self.axis_v = rot @ self.axis_v

    def scale(self, factors, axes, pivot):
        self.size_u = max(0.05, self.size_u * abs(factors[0]))  # mirrored: plane stays the same size
        self.size_v = max(0.05, self.size_v * abs(factors[1]))


@dataclass(eq=False)  # comparison by identity (numpy fields)
class Curve:
    points: np.ndarray  # (N, 3)
    plane_id: int | None  # drawing surface (plane or mesh) on which it was drawn (info, e.g. for the export)
    widths: np.ndarray | None = None  # (N,) stroke width in world units per point
    generated: bool = False  # connecting line added by filling (open ends)
    id: int = field(default_factory=_new_id)
    knots: tuple | None = None  # (point count, inner node indices) – corners/inserted points, see components.py
    style: str = "pencil"  # "pencil" (pencil, graphite) or "brush" (brush: color, oil texture) – otherwise the same
    color: tuple | None = None  # color of the brush stroke (RGB 0..1); pencil: None
    # paint stroke (paint mode, S): "dead" line, flat on the surface – not selectable or editable, sticks and
    # is erased like a line; normals = surface normal per point (otherwise from the surface), seq = paint order
    paint: bool = False
    normals: np.ndarray | None = None
    seq: int = 0
    lift: float = 0.0  # painting on a mesh: lifted this far along its normal (world units, once)
    hidden: bool = False  # individually hidden (H in edge mode)
    pids: dict | None = None  # curve network: {node index: point ID}, see network.py
    # point attributes (F4): {point ID: (qw, qx, qy, qz, pscale)} – rotation (orient) and size of the points; all lines
    # through a point carry the same entry (network.set_attr), missing = neutral
    pattr: dict | None = None
    frame: int | None = None  # paint stroke of a flipbook plane: frame of its drawing (timing.py); None = always there

    def __post_init__(self):
        if self.widths is None:
            self.widths = np.full(len(self.points), DEFAULT_WIDTH)

    def __deepcopy__(self, memo):
        """Fast copy (undo copies thousands of lines, also the grid rows of every surface): fields individually, simple
        values directly – the general route via __reduce_ex__ cost a multiple per line."""
        new = object.__new__(type(self))
        d = new.__dict__
        share = copy._deepcopy_dispatch.get(np.ndarray) is _share_array  # undo snapshot: share arrays
        if share and self.pids is None and self.pattr is None and len(self.__dict__) == _CURVE_FIELDS:
            # plain line (grid row of a surface, paint stroke): only numbers, tuples and arrays – take over flat
            d.update(self.__dict__)
            for v in (self.points, self.widths, self.normals):
                if v is not None and v.flags.writeable:
                    v.flags.writeable = False
            return new
        for k, v in self.__dict__.items():
            t = type(v)
            if v is None or t in _PLAIN:
                d[k] = v
            elif t is np.ndarray and v.dtype != object:
                if share:
                    v.flags.writeable = False
                    d[k] = v
                else:
                    d[k] = v.copy()
            elif t is dict and k in ("pids", "pattr"):  # {number: number} or {number: tuple}: flat is enough
                d[k] = memo[id(v)] = dict(v)  # in the memo: caches of the surfaces find their counterpart (_carry)
            else:
                d[k] = copy.deepcopy(v, memo)
        return new

    def __setattr__(self, name, value):
        d = self.__dict__
        if name[0] != "_" or "_v" not in d:  # state of the line: counts every assignment to a field (not to memo attributes
            d["_v"] = d.get("_v", 0) + 1      # like _pid_pos) – the display then remembers the entries per object
            _touch(self)                      # … and into the change log (whoever depends on it then looks again)
        if name == "points":
            POINTS_EPOCH[0] += 1
        if name in network.NET_FIELDS:  # curve network must sync again (network.sync)
            network.EPOCH[0] += 1
        elif name == "plane_id":  # directory "lines per drawing surface" (Scene.curves_on) is no longer valid
            PLANE_EPOCH[0] += 1
        elif name == "id":  # (also on creation) – directories that remember "does not exist" ask again
            IDS_EPOCH[0] += 1
        object.__setattr__(self, name, value)

    def translate(self, delta):
        self.points = self.points + delta

    def rotate(self, rot, pivot):
        self.points = pivot + (self.points - pivot) @ rot.T
        if self.normals is not None:
            self.normals = self.normals @ rot.T

    def scale(self, factors, axes, pivot):
        self.points = _scale_points(self.points, factors, axes, pivot)
        self.widths = self.widths * float(np.prod(np.abs(factors)) ** (1 / 3))


_CURVE_FIELDS = len(Curve.__dataclass_fields__) + 1  # this many entries has a line without extra attributes (+ _v)


@dataclass(eq=False)  # comparison by identity (numpy fields)
class StrokeObject:
    """Collects curves until a new stroke object is created."""
    name: str
    curves: list = field(default_factory=list)
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))  # local axes (rotate along)
    id: int = field(default_factory=_new_id)
    hidden: bool = False  # hidden (eye in the outliner, H)
    parent: int | None = None  # id of the null object under which it lies in the outliner
    flipbook: int | None = None  # id of the flipbook plane whose drawings (paint strokes per frame) lie here

    scale_axes = (0, 1, 2)

    def pivot(self):
        if not self.curves:
            return np.zeros(3)
        pts = np.vstack([c.points for c in self.curves])
        return (pts.min(axis=0) + pts.max(axis=0)) / 2

    def bounds(self):
        pts = np.vstack([c.points for c in self.curves]) if self.curves else np.zeros((1, 3))
        return pts.min(axis=0), pts.max(axis=0)

    def local_axes(self):
        return self.axes

    def translate(self, delta):
        for c in self.curves:
            c.translate(delta)

    def rotate(self, rot, pivot):
        for c in self.curves:
            c.rotate(rot, pivot)
        self.axes = rot @ self.axes

    def scale(self, factors, axes, pivot):
        for c in self.curves:
            c.scale(factors, axes, pivot)


LOFT_BIND_TOL = 0.003  # line counts as "on the loft", this close to the surface (share of its size)
_FILL_CACHE = {}  # hash of the outline -> (vertices, triangles, uv); keeps undo snapshots small
_OUTLINE_CACHE = {}  # loft shape -> outline polylines (otherwise recomputed every frame)
_CHAIN_CACHE = {}  # points of the source lines of a loft cross section -> assembled cross section


@dataclass(eq=False)  # comparison by identity (numpy fields)
class SurfaceObject(StrokeObject):
    """Drawing surface – like a drawing plane only an aid to draw on; the drawing
    is solely the lines. Its shape is its own copy of the outline (curves, in running direction,
    not visible), independent of the lines from which it arose."""
    kind: str = "fill"
    # link to the outline (fill surfaces only): [(line id or None = straight gap, reversed)] in running direction.
    # If you move these lines, the surface follows (Scene._sync_fills); lines drawn on it
    # hang on the outline via mean-value coordinates (cage.py) and deform along
    links: list = field(default_factory=list)
    counts: list = field(default_factory=list)  # cage points per outline piece
    rest: np.ndarray | None = None  # rest cage (K, 2): the outline at linking, flat
    link_cover: float = 0.0  # share of the outline covered by the linked lines
    attached: dict = field(default_factory=dict)  # {line id: {"uv": (N,2) rest position, "last": points}}
    # net surface (linked fill surface, netfill.py): (points, triangles, uv) of the surface deformed by its lines;
    # computed from outline and lines, not stored
    netmesh: tuple | None = None
    # display (no real materials): material (theme.MATERIALS), tint (RGB 0..1, None = white like the
    # material) and a fixed random offset of the texture – every surface its own piece, nothing visibly tiles
    material: str = "papier"
    tint: tuple | None = None
    opacity: float = 1.0  # opacity (slider in the outliner: transparency)
    resolution: float = 1.0  # resolution of the net: factor 0.25 … 4 (slider in the outliner), the shape stays
    # helper surface: only a projection surface to draw on (like a helper plane) – no object with material,
    # occludes nothing, shown/hidden with the drawing aids
    aid: bool = False
    # frozen (helper surface, Scene.freeze_surface): ((points, triangles, uv), outlines, sample of the points at
    # freezing) – as long as set, the surface shows exactly this shape and follows no line any more; only the gizmo
    # moves it. frozen_lines: {line id: points} of the lines it hung on when freezing
    frozen: tuple | None = None
    frozen_lines: dict = field(default_factory=dict)
    picture: dict | None = None  # projected image (pictures.py)
    seed: float = field(default_factory=lambda: float(np.random.default_rng().uniform(0, 1000)))
    binding: str | None = None  # binding level for the brief (brief.TASK_LEVELS; None: from the null above or assignment)
    owner: int | None = None  # line object that owns the surface (Scene.assign_owners): stands below it, goes with it

    # viewport quality (global slider, app setting): multiplies the resolution of every surface – only the
    # displayed net, the shape stays (is smoothly resampled every time)
    display_res = 1.0
    # displayed net by curvature (loft.adaptive_grid) instead of in the density of the computation grid: a flat surface
    # gets few triangles, a tight curve many – independent of size (harbor 10.11.: 239 surfaces with
    # 1.57 million triangles together, small and large ones ~6,000 each)
    auto_res = True

    @property
    def res(self):
        """Effective resolution of the displayed net: own slider × viewport quality."""
        return self.resolution * SurfaceObject.display_res

    def __setattr__(self, name, value):
        if name[0] != "_":  # a field of the surface was set anew: the sync must look again (Scene.sync_lofts)
            SURF_EPOCH[0] += 1
            d = self.__dict__
            d["_sv"] = d.get("_sv", 0) + 1  # … namely at this surface (stamp per surface, _surface_stamp)
            _touch(self)
        object.__setattr__(self, name, value)

    # pure caches, each (inputs, result): the undo copy gets them along, their inputs rewritten to the
    # copied objects (_carry) – after undo/redo they stay valid instead of every surface
    # recomputing its net (harbor scene: ~0.35 s per undo). Results are shared (nobody changes arrays in
    # place). Not taken along: the picking proxy (points to its surface).
    _NO_COPY = ("_base", "_sides_cache", "_pid_uv_cache", "_key_memo", "_grid_memo", "_tol_memo", "_tube_key",
                "_drv_key", "_netpatch_key", "_frozen_memo", "_dep_curves", "_sync_ok", "_key_fast")
    VERIFY_KEY = False  # tests: recompute the remembered key (_epoch_key) on every hit
    _DROP = ("_frozen_memo", "_key_fast")

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "curves" and type(v) is list:  # grid rows: thousands – directly, without the detour via deepcopy()
                rows = new.__dict__[k] = []
                for c in v:
                    twin = memo.get(id(c))
                    if twin is None:
                        twin = memo[id(c)] = c.__deepcopy__(memo)
                    rows.append(twin)
            elif k not in self._NO_COPY:
                new.__dict__[k] = copy.deepcopy(v, memo)
        for k in self._NO_COPY:
            if k not in self._DROP and k in self.__dict__:
                new.__dict__[k] = _carry(self.__dict__[k], memo)
        return new

    def loop_points(self):
        pts = []
        for c in self.curves:
            seg = c.points if not pts else c.points[1:]  # shared corner not doubled
            pts.extend(seg)
        return np.asarray(pts, float)

    @_epoch_key
    def _cache_key(self):
        # asked several times per frame per surface (net, picking, extent): remembered as long as outline and net are the same
        # arrays – otherwise per call the bytes of the whole net (harbor: 1,500 calls per mouse move)
        pts = tuple(c.points for c in self.curves)
        net0 = self.netmesh[0] if self.netmesh is not None else None
        auto = bool(SurfaceObject.auto_res)
        memo = self.__dict__.get("_key_memo")
        if (memo is not None and memo[1] is net0 and memo[2] == auto and len(memo[0]) == len(pts)
                and all(map(operator.is_, memo[0], pts))):
            return memo[3]
        key = self.loop_points().tobytes() + (net0.tobytes() if net0 is not None else b"") + bytes([auto])
        self._key_memo = (pts, net0, auto, key)
        return key

    def _build(self):
        if self.netmesh is not None:
            flat = _flat_fill(self.netmesh) if SurfaceObject.auto_res else None
            return self.netmesh if flat is None else flat
        return fill.triangulate_loop(self.loop_points())

    def _cached(self, proxy=True):
        if self.frozen is not None:  # frozen: exactly the held shape
            memo = getattr(self, "_frozen_memo", None)
            if memo is None or memo[0] is not self.frozen:
                memo = self._frozen_memo = [self.frozen, None]
            # proxy only when someone picks: a surface carried along in a drag gets a new shape every frame –
            # building it every time cost ~9 ms per frame in the harbor scene
            if proxy and memo[1] is None:
                mesh = self.frozen[0]
                memo[1] = PickProxy(Geometry(mesh[0], mesh[1]), self)
            return self.frozen[0], memo[1]
        key = (type(self).__name__, self._cache_key())
        if key not in _FILL_CACHE:
            lru.trim(_FILL_CACHE, 4096)
            mesh = self._build()
            proxy = PickProxy(Geometry(mesh[0], mesh[1]), self) if mesh is not None else None
            _FILL_CACHE[key] = (mesh, proxy)
        return _FILL_CACHE[key]

    def outlines(self):
        """Outline polylines for display (thin, like the border of a plane)."""
        if self.frozen is not None:
            return self.frozen[1]
        loop = self.loop_points()
        return [np.vstack([loop, loop[:1]])] if len(loop) > 1 else []

    def fill_mesh(self):
        """(vertices, triangles, uv) of the fill, cached per outline shape."""
        return self._cached(proxy=False)[0]

    def pick_proxy(self, build=True):
        """Proxy for GPU picking (like a mesh); same shape -> same object,
        so that the renderer can keep reusing its buffers. build=False: only if it already exists (frozen
        or carried-along surface) – whoever only wants to know which buffers are still needed builds none."""
        proxy = self._cached(proxy=build)[1]
        if proxy is not None:
            proxy.owner = self  # cache may come from an undo copy
        return proxy

    # frozen: the held shape travels along (the lines' own copies anyway)
    def translate(self, delta):
        super().translate(delta)
        self._move_frozen(lambda p: p + delta)

    def rotate(self, rot, pivot):
        super().rotate(rot, pivot)
        self._move_frozen(lambda p: pivot + (p - pivot) @ rot.T)

    def scale(self, factors, axes, pivot):
        super().scale(factors, axes, pivot)
        self._move_frozen(lambda p: _scale_points(p, factors, axes, pivot))

    def _move_frozen(self, f):
        if self.frozen is not None:
            (verts, tris, uv), lines, sample = self.frozen
            self.frozen = ((f(verts), tris, uv), [f(np.asarray(l, float)) for l in lines], sample)


@dataclass(eq=False)  # comparison by identity (numpy fields)
class ExtrudeSurface(SurfaceObject):
    """Drawing surface of an extrusion: the profile (own copy) pulled as a "curtain". Each segment
    is an offset per profile point (N, 3); row k = profile + sum of the first k segments.
    direction (only on creation): one segment with the same offset. Older files may have several
    segments."""
    kind: str = "extrude"
    segments: list = field(default_factory=list)
    direction: np.ndarray | None = None

    def __post_init__(self):
        if self.direction is not None and not self.segments:
            self.segments = [np.tile(np.asarray(self.direction, float), (len(self.profile()), 1))]
        self.direction = None

    def profile(self):
        return self.loop_points()

    def rows(self):
        """All edge rows: [profile, row 1, ...], each (N, 3)."""
        rows = [self.profile()]
        for seg in self.segments:
            rows.append(rows[-1] + seg)
        return rows

    def _cache_key(self):
        return self.profile().tobytes() + b"".join(np.asarray(seg, float).tobytes() for seg in self.segments)

    def _build(self):
        rows = self.rows()
        n = len(rows[0])
        if n < 2 or len(rows) < 2 or any(len(r) != n for r in rows):
            return None
        verts = np.vstack(rows)
        i = np.arange(n - 1)
        tris = []
        for k in range(len(rows) - 1):
            a, b = k * n, (k + 1) * n
            tris += [np.stack([a + i, a + i + 1, b + i + 1], 1), np.stack([a + i, b + i + 1, b + i], 1)]
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(rows[0], axis=0), axis=1))])
        height = np.zeros(n)
        uv = []
        for k, row in enumerate(rows):
            if k:
                height = height + np.linalg.norm(row - rows[k - 1], axis=1)
            uv.append(np.stack([arc, height], axis=1))
        return verts, np.concatenate(tris).astype(np.int32), np.vstack(uv)

    def edge_list(self, profile=True):
        """[(key, points)] of the rows and side edges – only for converting old files."""
        rows = self.rows()
        n = len(rows[0])
        out = [(("row", k), row) for k, row in enumerate(rows) if k > 0 or profile]
        for c in (0, n - 1):
            out += [(("col", c, k), np.array([rows[k][c], rows[k + 1][c]])) for k in range(len(rows) - 1)]
        return out

    def outlines(self):
        rows = self.rows()
        return [rows[0], rows[-1], np.array([rows[0][0], rows[-1][0]]), np.array([rows[0][-1], rows[-1][-1]])]

    def bounds(self):
        pts = np.vstack(self.rows())
        return pts.min(axis=0), pts.max(axis=0)

    def pivot(self):
        lo, hi = self.bounds()
        return (lo + hi) / 2

    def rotate(self, rot, pivot):
        super().rotate(rot, pivot)
        self.segments = [seg @ rot.T for seg in self.segments]

    def scale(self, factors, axes, pivot):
        super().scale(factors, axes, pivot)
        m = axes @ np.diag(factors) @ axes.T
        self.segments = [seg @ m.T for seg in self.segments]


@dataclass(eq=False)  # comparison by identity (numpy fields)
class LoftSurface(SurfaceObject):
    """Curved drawing surface through several cross sections (loft.py). curves = own copies of the
    cross sections, in order and aligned. sources: per cross section, where it comes from –
    ("lines", [line ids]) or ("surface", surface id) or None. If you move these lines or surfaces,
    the loft surface follows (Scene.sync_lofts); if they are deleted, the last shape stays."""
    kind: str = "loft"
    sources: list = field(default_factory=list)
    closed: bool = True
    rails: list = field(default_factory=list)  # [(line id, column)]: real longitudinal/side lines, follow the surface
    wrap: bool = False  # ring-shaped: the last cross section lofts back to the first (Ctrl+Shift+F)
    # lines drawn on it: {line id: {"uv": (N,2) position on the surface, "on": (N,) on the surface?,
    # "last": points at the last sync}} – they deform along with the loft (Scene._sync_attached)
    attached: dict = field(default_factory=dict)
    # net surface (curve surface): displacement of the grid by the real lines on it (netsurf.py); computed from the
    # lines, not stored
    offset: np.ndarray | None = None
    span: int = loft.SPAN_SAMPLES  # grid rows per section between two cross sections (1: all rows stored)
    # net surface: where the border nodes hang on the border lines (per side fractional point indices, netsurf.boundary_track)
    bnd: list | None = None

    def sections(self):
        secs = [c.points for c in self.curves]
        return secs + secs[:1] if self.wrap and len(secs) >= 3 else secs

    def _same_inputs(self, memo, inputs):
        """Same inputs as remembered? Point arrays by identity (they are replaced, never changed in place),
        the rest by value – cheap instead of the bytes of all grid rows per frame."""
        if memo is None or len(memo) != len(inputs):
            return False
        n = len(self.curves)  # the point arrays of the rows come first: by identity in one go (fast)
        if n and len(inputs) >= n and not all(map(operator.is_, memo[:n], inputs[:n])):
            return False
        return all(a is b if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else a == b
                   for a, b in zip(memo[n:], inputs[n:]))

    def base_grid(self):
        """Grid from the cross sections alone (cached as long as they do not change)."""
        inputs = (*(c.points for c in self.curves), self.wrap, self.span)
        cached = getattr(self, "_base", None)
        if cached is None or not self._same_inputs(cached[0], inputs):
            cached = self._base = (inputs, loft.grid(np.array(self.sections()), self.span))
        gordon = getattr(self, "_gordon", None)  # tube/ring as curve network (longitudinal lines shape it exactly)
        if gordon is not None and gordon.shape == cached[1].shape:
            return gordon
        return cached[1]

    def grid(self):
        """Point grid of the surface (R, M, 3), with the displacement by its lines."""
        g = self.base_grid()
        if self.offset is None or self.offset.shape != g.shape:
            return g
        memo = getattr(self, "_grid_memo", None)  # asked several times per frame: remember the sum (do not change)
        if memo is None or memo[0] is not g or memo[1] is not self.offset:
            memo = self._grid_memo = (g, self.offset, g + self.offset)
        return memo[2]

    @_epoch_key
    def _cache_key(self):
        inputs = (*(c.points for c in self.curves), self.offset, self.closed, self.wrap, self.res,
                  bool(SurfaceObject.auto_res))
        memo = getattr(self, "_key_memo", None)
        if memo is not None and self._same_inputs(memo[0], inputs):
            return memo[1]  # the same bytes object: its hash is remembered too
        off = self.offset.tobytes() if self.offset is not None else b""
        key = (b"".join(np.ascontiguousarray(c.points).tobytes() for c in self.curves)
               + bytes([self.closed, self.wrap, bool(SurfaceObject.auto_res)]) + off + np.float64(self.res).tobytes())
        self._key_memo = (inputs, key)
        return key

    def _build(self):
        if len(self.curves) < 2:
            return None
        g = self.grid()
        geo = getattr(self, "_net_geo", None)  # curve network: separated at hard net lines (the crease stays)
        if SurfaceObject.auto_res:
            # only the displayed net, the shape stays: as fine as its curvature needs (loft.adaptive_grid) –
            # the slider of the surface and the viewport quality make it finer or coarser from there
            br = [r for r, h in zip(geo["rows"], geo["hard_u"]) if h.max() > 1e-3] if geo else ()
            bc = [c for c, h in zip(geo["cols"], geo["hard_v"]) if h.max() > 1e-3] if geo else ()
            g = loft.adaptive_grid(g, self.res, self.closed, br, bc, ring=self.wrap and len(self.curves) >= 3)
        elif self.res != 1.0:  # only the displayed net: smoothly resampled, the shape stays
            br = [r for r, h in zip(geo["rows"], geo["hard_u"]) if h.max() > 1e-3] if geo else ()
            bc = [c for c, h in zip(geo["cols"], geo["hard_v"]) if h.max() > 1e-3] if geo else ()
            g = loft.resample_grid(g, self.res, self.closed, br, bc, ring=self.wrap and len(self.curves) >= 3)
        return loft.build(np.array(self.sections()), self.closed, g=g)

    def _outline_grid(self):
        return self.offset is not None or self.span != loft.SPAN_SAMPLES

    def outlines(self):
        if self.frozen is not None:
            return self.frozen[1]
        if len(self.curves) < 2:
            return []
        key = self._cache_key()
        if key not in _OUTLINE_CACHE:
            lru.trim(_OUTLINE_CACHE, 4096)
            if self._outline_grid():  # net surface: border of the displaced grid
                g = self.grid()
                _OUTLINE_CACHE[key] = ([np.vstack([g[0], g[0][:1]]), np.vstack([g[-1], g[-1][:1]])] if self.closed
                                       else [g[0], g[-1], g[:, 0], g[:, -1]])
            else:
                _OUTLINE_CACHE[key] = loft.outlines(np.array(self.sections()), self.closed)
        return _OUTLINE_CACHE[key]

    def bounds(self):
        mesh = self.fill_mesh()
        pts = mesh[0] if mesh is not None else np.vstack(self.sections())
        return pts.min(axis=0), pts.max(axis=0)

    def pivot(self):
        lo, hi = self.bounds()
        return (lo + hi) / 2


@dataclass(eq=False)
class PatchSurface(LoftSurface):
    """Curve surface (Alt+F, patch.py): smooth surface from bent boundary curves (Coons surface). Stored as a loft
    through rows of the Coons surface; the rows are recomputed from the boundary curves when they change."""
    kind: str = "patch"
    patch_lines: list = field(default_factory=list)  # ids of the boundary curves
    patch_order: list = field(default_factory=list)  # [(index in patch_lines, reversed)] in loop order
    patch_corners: list = field(default_factory=list)  # [(curve in the loop, share)] – the four corners
    patch_arc: bool = False  # share = arc length (new); False: share of the points (older files, is converted)
    # curve network (patchnet.py): grid size and on which row or column each net line lies –
    # {"r", "m", "u": [[[line ids of the chain], row], …], "v": [[[ids], column], …]}; None: lay anew on sync
    net: dict | None = None


@dataclass(eq=False)
class NoteAnchor:
    """End of a leader line: on a line (travels along, t = share of its arc length) or on a
    fixed point (curve_id None). lost: the line is gone – the anchor stays put, dashed."""
    point: np.ndarray
    curve_id: int | None = None
    t: float = 0.0
    lost: bool = False
    members: list = field(default_factory=list)  # note octopus: further parts of the same kind (NoteMember)


@dataclass(eq=False)
class NoteMember:
    """Arm of the note octopus: a further part that the anchor also means – an object (stroke object, surface, mesh)
    or a single line in it (curve_id, travels along like an anchor, t = share of its arc length). point: end
    of the arm (dragged: where it was released; from the selection: the place nearest to the anchor), None: middle of the part.
    Under symmetry/array the arm hangs on the original, the copies are meant as well."""
    obj_id: int
    curve_id: int | None = None
    point: np.ndarray | None = None
    t: float = 0.0


@dataclass(eq=False)
class Note:
    """Note in space (see notes.py): heading, position (lower left corner of the text), size, rotation about Y
    (None: rotates with the camera), leader lines and the note board (text and images)."""
    title: str
    pos: np.ndarray
    size: str = "m"
    yaw: float | None = None
    anchors: list = field(default_factory=list)
    board: dict = field(default_factory=dict)
    id: int = field(default_factory=_new_id)
    # review draft (review.py, from Claude out of a review recording): {"review": folder name, "t", "t_end", "quote"}
    # – until the artist takes it over in the board (None: completely normal note)
    review: dict | None = None
    # fixed identifier across saving/loading (id is reassigned on loading) – review rounds find the note again this way
    uid: str = field(default_factory=lambda: os.urandom(6).hex())


class PickProxy:
    """Turns a surface into a mesh for GPU picking: geometry in world coordinates."""

    def __init__(self, geo, owner):
        self.geo, self.owner = geo, owner
        self.xform = np.eye(4)
        self.missing = False


def inherit(new, old):
    """New line from an old one (split, copy, merge …): take over kind and color."""
    new.style, new.color = old.style, old.color
    new.paint, new.seq, new.hidden = old.paint, old.seq, old.hidden
    new.frame = old.frame  # pieces of a flipbook drawing stay in their drawing
    if old.pattr:
        new.pattr = dict(old.pattr)  # point attributes travel along (surplus entries do no harm)
    return new


def split_curve(curve, point, i, t):
    """Split line at point (on the piece i -> i+1 at t) into two lines -> (first, second) or None at the
    line end. Nodes and net points stay; the cut becomes a new node."""
    if (i == 0 and t < 1e-9) or (i == len(curve.points) - 2 and t > 1 - 1e-9):
        return None
    w = curve.widths[i] + (curve.widths[i + 1] - curve.widths[i]) * t
    first = Curve(np.vstack([curve.points[:i + 1], point]), curve.plane_id,
                  np.append(curve.widths[:i + 1], w), curve.generated)
    second = Curve(np.vstack([point, curve.points[i + 1:]]), curve.plane_id,
                   np.insert(curve.widths[i + 1:], 0, w), curve.generated)
    inherit(first, curve)
    inherit(second, curve)
    if curve.knots is not None and curve.knots[0] == len(curve.points):  # take over nodes instead of detecting anew
        inner = curve.knots[1]
        first.knots = (len(first.points), tuple(k for k in inner if 0 < k <= i))
        second.knots = (len(second.points), tuple(k - i for k in inner if i + 1 <= k < len(curve.points) - 1))
        if curve.pids:  # curve network: old points stay, the cut becomes a new shared point
            first.pids = {k: p for k, p in curve.pids.items() if k <= i}
            second.pids = {k - i: p for k, p in curve.pids.items() if k >= i + 1}
    return first, second


def _rows(points):
    """Point list for the scene file, to 5 places – the whole array rounded at once instead of number by number
    (harbor 11.11.: 1.5 s per save and per automatic backup for this alone)."""
    if type(points) is not np.ndarray or points.dtype != np.float64:
        points = np.asarray(points, float)
    return points.round(5).tolist()


def _curve_dict(c, r):
    return {"id": c.id, "plane": c.plane_id, "points": _rows(c.points), "widths": _rows(c.widths),
            "generated": c.generated, **({"knots": list(c.knots[1])} if c.knots is not None else {}),
            **({"style": c.style, "color": r(c.color)} if c.style != "pencil" and c.color is not None else {}),
            **({"paint": True, "seq": int(c.seq)} if c.paint else {}),
            **({"lift": round(float(c.lift), 6)} if c.paint and c.lift else {}),
            **({"normals": _rows(c.normals)} if c.paint and c.normals is not None
               and len(c.normals) == len(c.points) else {}),
            **({"hidden": True} if c.hidden else {}),
            **({"frame": int(c.frame)} if c.frame is not None else {}),
            **({"pids": [[int(k), int(p)] for k, p in sorted(c.pids.items())]} if c.pids else {}),
            **({"pattr": [[int(p), *r(v)] for p, v in sorted(c.pattr.items())]} if c.pattr else {})}


def _curve_from(c, id_map):
    arr = lambda a: np.array(a, float)
    curve = Curve(arr(c["points"]), id_map.get(c["plane"]), arr(c["widths"]) if "widths" in c else None,
                  c.get("generated", False))
    if len(curve.widths) != len(curve.points):  # older errors (v0.83.1): widths matching the point count
        curve.widths = np.interp(np.linspace(0, 1, len(curve.points)), np.linspace(0, 1, len(curve.widths)), curve.widths)
    if "knots" in c:
        curve.knots = (len(curve.points), tuple(int(i) for i in c["knots"]))
    if c.get("style") == "brush" and c.get("color") is not None:
        curve.style, curve.color = "brush", tuple(float(x) for x in c["color"])
    if c.get("paint"):
        curve.paint, curve.seq = True, int(c.get("seq", 0))
        curve.lift = float(c.get("lift", 0.0))
        if c.get("normals") is not None:
            curve.normals = arr(c["normals"]).reshape(-1, 3)
    curve.hidden = bool(c.get("hidden", False))
    if c.get("frame") is not None and curve.paint:
        curve.frame = int(c["frame"])
    if c.get("pids") and "knots" in c:
        curve.pids = {int(k): int(p) for k, p in c["pids"]}
    if c.get("pattr"):
        curve.pattr = {int(e[0]): tuple(float(x) for x in e[1:6]) for e in c["pattr"]}
    return curve


@dataclass(eq=False)  # comparison by identity (numpy fields)
class MeshObject:
    """Imported reference geometry (one file or, with USD, one mesh prim = one object). The geometry itself is
    immutable and shared between undo snapshots; only xform is transformed."""
    name: str
    path: str  # absolute path; when saving relative to the scene file
    geo: Geometry
    xform: np.ndarray = field(default_factory=lambda: np.eye(4))  # local -> world
    missing: bool = False  # file not found/readable on loading
    id: int = field(default_factory=_new_id)
    hidden: bool = False  # hidden (eye in the outliner, H)
    parent: int | None = None  # id of the null object above
    prim: str | None = None  # USD: path of the mesh prim (None: all meshes of the file in one object, older scenes)
    binding: str | None = None  # binding level for the brief (brief.TASK_LEVELS; None: from the null above or assignment)

    scale_axes = (0, 1, 2)

    def world_bounds(self):
        lo, hi = self.geo.bbox
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = corners @ self.xform[:3, :3].T + self.xform[:3, 3]
        return world.min(axis=0), world.max(axis=0)

    bounds = world_bounds

    def normals_at(self, points):
        """Normal (world) at the nearest point of the mesh for each of the points (world)."""
        rot, t = self.xform[:3, :3], self.xform[:3, 3]
        inv = np.linalg.inv(rot)
        n = self.geo.normals_near((np.asarray(points, float) - t) @ inv.T) @ inv
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    def pivot(self):
        lo, hi = self.world_bounds()
        return (lo + hi) / 2

    def local_axes(self):
        m = self.xform[:3, :3]
        return m / np.maximum(np.linalg.norm(m, axis=0), 1e-12)

    def _apply(self, m3, pivot):
        t = np.eye(4)
        t[:3, :3] = m3
        t[:3, 3] = pivot - m3 @ pivot
        self.xform = t @ self.xform

    def translate(self, delta):
        self.xform = self.xform.copy()
        self.xform[:3, 3] += delta

    def rotate(self, rot, pivot):
        self._apply(rot, pivot)

    def scale(self, factors, axes, pivot):
        self._apply(axes @ np.diag(factors) @ axes.T, pivot)


class TransformSet:
    """What the gizmo moves: one or several objects (planes, stroke objects, meshes, edge groups).
    Everything transforms around a common pivot that belongs to the selection: on first use the
    center of the bounding box, afterwards it travels along (moving takes it along, rotating and scaling
    around it leave it standing). Otherwise the box center would lie elsewhere after every rotation.
    override: () -> (point, axes or None, own axis?) or None; store(point, axes, own).
    """

    def __init__(self, items, override=None, store=None, scene=None):
        self.items = items
        self.override, self.store = override, store
        self.scene = scene
        self._carried = None  # rigidly carried surfaces of this drag (begin_rigid/end_rigid)
        self._free = []  # of those: not in the target themselves, but all their lines – the drag carries their net along here
        self._moved_map = {}  # line id -> line that this drag moves as a whole

    # --- Rigid movement: surfaces whose lines are ALL moved along, do not re-sync in every frame ---
    def begin_rigid(self):
        """At the first step of a drag: every surface of this selection whose inputs (border, net, cross-section,
        longitudinal lines and lines drawn/painted on it) all belong to stroke objects moved along, hold
        for the duration of the drag with its finished net (like a frozen surface: the net travels along, the sync
        skips it). Harbor test 09.11.: moving a hero building (50 surfaces) cost 0.5 s per frame."""
        if self._carried is not None or self.scene is None or not getattr(self, "in_drag", False):
            return
        self._carried = []
        self._carried_new = {}  # id(surface) -> memo attributes that only this drag created
        self._free, self._moved_map = [], {}
        self._M = np.eye(4)
        sc = self.scene
        whole = {c.id: c for o in self.items if isinstance(o, StrokeObject) and not isinstance(o, SurfaceObject)
                 for c in o.curves}  # moved as a whole (rigid)
        moved = set(whole)
        soft = set()  # are only partly dragged along (deform): their surfaces must sync
        for o in self.items:  # points/partial edges: all lines whose points the drag moves (also welded ones)
            if type(o).__name__ == "ComponentGroup":
                moved |= {c.id for c in o.curves.values()}
            elif type(o).__name__ == "EdgeGroup":
                # whole lines (including coupled ones); neighboring lines hanging at their ends are dragged softly along
                whole.update({h.curve.id: h.curve for h in o.handles})
                moved |= {h.curve.id for h in o.handles}
                if o.weld:
                    soft |= {c.id for (c, _j), _end in o._through_links()}
        if not moved:
            return
        soft -= set(whole)
        self._moved_map = whole
        shaped = {}  # id(surface) -> shaped by lines (not just painted/labeled)
        on = {}
        for st in sc.strokes:
            for c in st.curves:
                if c.plane_id is not None:
                    on.setdefault(c.plane_id, []).append(c.id)
        def pieces(cid, depth=0):
            """Still existing lines (pieces) for cid. A line that no longer exists (deleted, or its chain
            is incomplete) cannot change in the drag – only the pieces that are still there count
            (harbor scene: 13 surfaces of a hero building hung on such remains and were never carried along)."""
            if sc.curve_by_id(cid) is not None:
                return [cid]
            if depth > 32:
                return []
            return [x for sub in (sc.split_map.get(cid) or ()) for x in pieces(sub, depth + 1)]

        def inputs(f):
            """Line ids the surface hangs on, or None (hangs on another surface: not safe)."""
            ids = list(getattr(f, "patch_lines", None) or [])
            ids += [i for link, _fl in (getattr(f, "links", None) or []) for i in sc.link_ids(link)]
            srcs = getattr(f, "sources", None) or []
            if any(src is not None and src[0] != "lines" for src in srcs):
                return None
            ids += [i for src in srcs if src is not None for i in src[1]]
            ids += [r[0] for r in (getattr(f, "rails", None) or []) if isinstance(r, (list, tuple))]
            shaped[id(f)] = bool(ids)
            ids += list(getattr(f, "attached", None) or ())
            ids = [x for cid in ids for x in pieces(cid)]
            return ids + on.get(f.id, [])

        def hold(f, idle=False):
            mesh = f.fill_mesh()
            if mesh is None or not len(mesh[0]):
                return
            uv = mesh[2] if len(mesh) > 2 else None
            frozen = ((mesh[0], mesh[1], uv), [np.asarray(l, float) for l in f.outlines()], None)
            if idle:
                # Uninvolved surface: taken out of the sync only for the duration of the drag. Its stamp (_sync_ok,
                # Scene._surface_stamp) stays valid – after release it is not checked again as long as no
                # line it depends on has changed (before, every release re-synced ALL surfaces).
                f.__dict__["frozen"] = frozen
                SURF_EPOCH[0] += 1
                f._idle_hold = True
            else:
                f.__dict__.pop("_idle_hold", None)
                # if it was synced with its lines at the start of the drag, its finished state can simply be moved along on release
                # (Scene.transfer_rigid) instead of arising anew from the lines
                stamp = f.__dict__.get("_sync_ok")
                f._carry_clean = None
                if stamp is not None and "_dep_curves" in f.__dict__ and stamp == sc._surface_stamp(f, world, refs):
                    # … and only if at the end every line it depends on is shifted exactly so and kept its
                    # point IDs (if the drag tears open a weld to a neighboring line, hardness
                    # and connections change – then recompute as before). State of the lines now:
                    f._carry_clean = [(c, c.points, dict(c.pids) if c.pids else None, c.knots)
                                      for c in f.__dict__["_dep_curves"]]
                f._carry_arity = len(mesh)
                f.frozen = frozen
            # (if the flag did not exist before, it disappears again after the drag: otherwise every merely
            # held surface would count as changed for undo and be fully copied at every step – harbor scene
            # 12.11.: 660 uninvolved surfaces, Ctrl+Z about 1 s slower because of it)
            self._carried_new[id(f)] = [k for k in ("_carried", "_carry_base", "_carry_model") if k not in f.__dict__]
            f._carried = True
            self._carried.append(f)

        in_set = {id(o) for o in self.items}
        world, refs = sc._sync_world(False), network.refs_by_pid(sc)
        for f in sc.surfaces:
            if f.frozen is not None or f.aid:
                continue
            ids = inputs(f)
            if ids is None:
                continue
            if any(i in soft for i in ids):
                continue  # one of its lines deforms in the drag: sync normally
            if id(f) in in_set:
                if all(i in moved for i in ids):  # all inputs move along (or it has none and merely travels
                    # itself): carry the net along
                    hold(f)
                    if f.frozen is not None:  # display: net from the start of the drag + position matrix (upload nothing anew)
                        f._carry_base = f.frozen[0]
                        f._carry_model = self._M
            elif shaped.get(id(f)) and all(i in whole for i in ids):
                # surface not selected, but ALL its lines (border, net, painting on it) travel along as a whole:
                # the same as moving the surface – carry the net along instead of recomputing per frame and rebinding the
                # painting (Talking Points 10.11.: 3.4 s per frame with ~1000 paint strokes)
                hold(f)
                if f.frozen is not None:
                    f._carry_base = f.frozen[0]
                    f._carry_model = self._M
                    self._free.append(f)
            elif not any(i in moved for i in ids):
                hold(f, idle=True)  # hangs on nothing moved: does not change in this drag – save the sync
        sc._rigid_active = self._carried

    @staticmethod
    def _moved_by(deps, shift):
        """Are all these lines (state from the drag start: line, points, point IDs, nodes) shifted exactly by shift
        and otherwise unchanged?"""
        for c, points, pids, knots in deps:
            now = c.points
            if now.shape != points.shape or c.knots != knots or (c.pids or None) != pids:
                return False
            if len(now) and float(np.abs(now - points - shift).max()) > 1e-9 * max(1.0, float(np.abs(now).max())):
                return False
        return True

    def end_rigid(self):
        """Drag finished (or no longer rigid): let surfaces follow their lines again – the next sync builds
        them once from the (moved) lines."""
        # only moved (not rotated, not scaled): the finished state of the carried surfaces travels along
        shift = None
        if self._carried and self.scene is not None and np.array_equal(self._M[:3, :3], np.eye(3)):
            shift = self._M[:3, 3].copy()
        own = {id(o) for o in self.items}
        whole = []
        for f in self._carried or []:
            clean = f.__dict__.pop("_carry_clean", None)
            arity = f.__dict__.pop("_carry_arity", 3)
            if getattr(f, "_carried", False):
                if f.__dict__.pop("_idle_hold", False):  # uninvolved: stamp of the sync stays valid (begin_rigid)
                    f.__dict__["frozen"] = None
                    SURF_EPOCH[0] += 1
                    # … but is recomputed once: while it was held, the sync did not
                    # ask it – what reported in the meantime (a neighboring line at its corner moved along) went
                    # past it
                    _touch(f)
                else:
                    f.frozen = None
                    base = f.__dict__.get("_carry_base")
                    if shift is not None and clean is not None and base is not None and self._moved_by(clean, shift):
                        whole.append((f, id(f) in own, base[:arity]))
                f._carried = False
            f._carry_base = f._carry_model = None
            for k in getattr(self, "_carried_new", {}).get(id(f), ()):
                f.__dict__.pop(k, None)
        if whole:
            self.scene.transfer_rigid(whole, shift)
        if self.scene is not None and getattr(self.scene, "_rigid_active", None) is self._carried:
            self.scene._rigid_active = None
        self._carried = None
        self._free = []

    @property
    def scale_axes(self):
        return (0, 1) if all(isinstance(i, Plane) for i in self.items) else (0, 1, 2)

    def bounds(self):
        los, his = zip(*(i.bounds() for i in self.items))
        return np.min(los, axis=0), np.max(his, axis=0)

    def center(self):
        lo, hi = self.bounds()
        return (lo + hi) / 2

    def _override(self):
        return self.override() if self.override is not None else None

    def pivot(self):
        custom = self._override()
        if custom is not None:
            return custom[0].copy()
        center = self.center()
        if self.store is not None:  # from now on fixed for this selection
            self.store(center, None, False)
        return center

    def local_axes(self):
        custom = self._override()
        if custom is not None and custom[1] is not None:
            return custom[1]
        return self.items[0].local_axes()

    def _carry(self, move_point, turn_axes=None):
        """Carry the stored pivot (and own axes) along with the geometry."""
        custom = self._override()
        if custom is None or self.store is None:
            return
        point, axes, own = custom
        if turn_axes is not None and axes is not None:
            axes = turn_axes(axes)
        self.store(move_point(point), axes, own)

    def _accumulate(self, m):
        """Advance the position matrix of the carried surfaces by one step (new array: the display notices the change)."""
        if self._carried:
            self._M = m @ self._M
            for f in self._carried:
                if getattr(f, "_carry_model", None) is not None:
                    f._carry_model = self._M
            for f in self._free:  # not in the target itself: carry its held net along here
                if getattr(f, "_carried", False):
                    f._move_frozen(lambda q: q @ m[:3, :3].T + m[:3, 3])

    def translate(self, delta):
        self.begin_rigid()
        m = np.eye(4)
        m[:3, 3] = delta
        self._accumulate(m)
        for obj in self.items:
            obj.translate(delta)
        self._carry(lambda p: p + delta)
        self._keep_bindings()

    def rotate(self, rot, pivot):
        self.begin_rigid()
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = np.asarray(pivot, float) - rot @ np.asarray(pivot, float)
        self._accumulate(m)
        for obj in self.items:
            obj.rotate(rot, pivot)
        self._carry(lambda p: pivot + rot @ (p - pivot), lambda a: rot @ a)
        self._keep_bindings()

    def _keep_bindings(self, uniform=True):
        """Surface and lines drawn/painted on it were moved together (e.g. a whole group rotated): their
        binding (uv on the surface) stays valid – the sync should not rebind them in every frame (expensive), but
        only follow the lines on the moved surface. Not for non-uniform scaling (rebind there)."""
        if not uniform:
            return
        # painting on a carried surface not selected itself: moved with it, binding stays valid (real
        # lines on it are synced by the sync after the drag itself – their memo state "pos" must then be right)
        for f in self._free:
            for cid, att in (getattr(f, "attached", None) or {}).items():
                curve = self._moved_map.get(cid)
                if curve is not None and curve.paint and isinstance(att, dict) and "last" in att:
                    att["last"] = curve.points.tobytes()
        moved = {c.id: c for o in self.items if isinstance(o, StrokeObject) for c in o.curves}
        for o in self.items:  # whole lines moved with their surface (surface selected: its lines travel along)
            if type(o).__name__ == "EdgeGroup":
                moved.update({h.curve.id: h.curve for h in o.handles})
        if not moved:
            return
        for obj in self.items:
            for cid, att in (getattr(obj, "attached", None) or {}).items():
                curve = moved.get(cid)
                if curve is not None and isinstance(att, dict) and "last" in att:
                    att["last"] = curve.points.tobytes()

    def scale(self, factors, axes, pivot):
        f = np.abs(np.asarray(factors, float))
        if np.allclose(f, f[0]) and np.all(np.asarray(factors, float) > 0):
            self.begin_rigid()
            if self._carried:
                k = float(np.asarray(factors, float)[0])
                m = np.eye(4)
                m[:3, :3] *= k
                m[:3, 3] = np.asarray(pivot, float) * (1 - k)
                self._accumulate(m)
        else:
            self.end_rigid()  # non-uniform: deformed – surfaces must sync
            self._carried = []  # do not hold again in this drag
        single = len(self.items) == 1
        for obj in self.items:
            if isinstance(obj, Plane) and not single:
                # In the group: scale position along, size uniformly
                obj.origin = _scale_points(obj.origin[None], factors, axes, pivot)[0]
                f = float(np.prod(np.abs(factors)) ** (1 / 3))
                obj.scale(np.array([f, f, f]), axes, pivot)
            else:
                obj.scale(factors, axes, pivot)
        self._carry(lambda p: _scale_points(p[None], factors, axes, pivot)[0])
        f = np.abs(np.asarray(factors, float))
        self._keep_bindings(uniform=bool(np.allclose(f, f[0])))


class PivotHandle:
    """Axis mode: the gizmo moves only the pivot and rotates its axes, not the geometry.
    store(point, axes) saves the axis as own; snap (set when dragging with V) takes precedence
    over the freely dragged position."""
    scale_axes = (0, 1, 2)

    def __init__(self, target, store):
        self.target, self._store = target, store
        self._free = None  # freely dragged position during a drag
        self._axes = None  # rotated axes (None: axes of the selection)
        self.snap = None

    @property
    def items(self):  # the selection stays the same (e.g. for highlighting selected edges)
        return self.target.items

    def pivot(self):
        return self.target.pivot()

    def local_axes(self):
        return self.target.local_axes()

    def _commit(self):
        point = self.snap if self.snap is not None else self._free
        custom = self.target._override()
        axes = self._axes if self._axes is not None else (custom[1] if custom is not None else None)
        self._store(np.array(point, float), axes)

    def translate(self, delta):
        self._free = (self.pivot() if self._free is None else self._free) + delta
        self._commit()

    def set_snap(self, point):
        if self._free is None:
            self._free = self.pivot()
        self.snap = point
        self._commit()

    def rotate(self, rot, pivot):
        self._free = self.pivot() if self._free is None else self._free
        self._axes = rot @ self.local_axes()
        self._commit()

    def scale(self, factors, axes, pivot):
        pass  # an axis has no size

    def end(self):
        self._free, self._axes, self.snap = None, None, None


# surface methods of the scene (needs the classes above – hence imported only here)
from .scene_surfaces import SurfaceSyncMixin  # noqa: E402


class Scene(SurfaceSyncMixin):
    def __init__(self):
        x, y, z = np.eye(3)
        self.presets = [
            Plane(tr("Boden (XZ)"), np.zeros(3), z, x, preset=True),
            Plane("Front (XY)", np.zeros(3), x, y, preset=True),
            Plane(tr("Seite (ZY)"), np.zeros(3), y, z, preset=True),
        ]
        self.custom = []
        self.strokes = []
        self.meshes = []
        self.surfaces = []
        self.nulls = []  # null objects and generators (symmetry, array): groups in the outliner, see NullObject
        self.multi_ids = []  # multi set: ids of the surfaces drawn on in multi mode (with undo)
        self.notes = []  # notes (notes mode, N)
        self.global_note = Note(tr("Projekt"), np.zeros(3))  # global note of the scene ("Global" panel): not in space
        self.sun = light.SUN_DEFAULT  # light mode: (azimuth, elevation) in degrees – stored with the scene
        self.light_bands = 0  # light levels (0 = smooth, otherwise number of brightness levels) – stored with the scene
        # note size of the scene: factor on the font sizes Small … Very large (world units). For large scenes: with
        # a harbor of kilometers a 35 cm font is a dot from any usable distance.
        self.note_scale = 1.0
        self.brief_task = {}  # assignment from the brief dialog (brief.task_clean) – stored, no undo step
        # clarified with the AI (MCP write_clarified): {"summary", "answers": [{"question", "answer"}]} – every export
        # writes it into the brief; stored, no undo step
        self.brief_clarified = {}
        self.timing = timing.Timing()  # timing mode: frames, timing notes, cameras (stored, in undo)
        self.split_map = {}  # split lines: old id -> ids of the pieces in running direction (surfaces, lofts, notes follow)
        self.assets = {}  # images of the note boards: file name -> bytes (stored in the scene file)
        self.asset_dir = None  # older scenes: folder of the images next to the file (only read now)
        self.active_plane = self.presets[0]
        self.active_mesh = None  # set: drawing surface is this mesh / this surface instead of active_plane
        self.last_custom = None
        self.active_stroke = None
        self.dirty = False
        self.load_warnings = []
        self.net_report = None  # curve network on loading: points, connected, open ends, taken over?
        self._undo, self._redo = [], []
        self.video_moved = {}  # videos of the boards: old path -> copy next to the scene (adopt_videos)
        self._burst = (None, 0.0)
        self.revision = 0  # counts change steps (checkpoints, undo/redo) – for caches
        self.sel_revision = 0  # counts undo/redo of selection steps
        # (read, set) of the selection – set by the viewport: undo steps remember the selection along
        self.selection_io = None

    # --- Access ---
    @property
    def planes(self):
        return self.presets + self.custom

    @property
    def curves(self):
        """All lines – those of the stroke objects. (Drawing surfaces have no lines, only a shape.)"""
        return [c for s in self.strokes for c in s.curves]

    def pickable_curves(self):
        """Everything that can be clicked, filled and connected like a line."""
        return self.curves

    def curves_on(self, surface_id, also=()):
        """Lines with plane_id == surface_id – plus the lines with the ids as well – in the order of
        pickable_curves(). From a directory that holds until any line changes its drawing surface
        (PLANE_EPOCH), a line is created (sets plane_id) or the lists of the stroke objects change: the
        sync asked all lines of the scene several times per surface (harbor: 118 surfaces x 1,300 lines per change)."""
        lists = [st.curves for st in self.strokes]
        memo = getattr(self, "_on_index", None)
        if (memo is None or memo[0] != PLANE_EPOCH[0] or memo[1] is not self.strokes or len(memo[2]) != len(lists)
                or not all(a is b and len(b) == n for a, b, n in zip(memo[2], lists, memo[3]))):
            index, pos = {}, {}
            at = 0
            for cl in lists:
                for c in cl:
                    pos[c.id] = (at, c)
                    at += 1
                    if c.plane_id is not None:
                        index.setdefault(c.plane_id, []).append((at, c))
            memo = self._on_index = (PLANE_EPOCH[0], self.strokes, lists, [len(cl) for cl in lists], index, pos)
        on = memo[4].get(surface_id, ())
        if not also:
            return [c for _at, c in on]
        pos = memo[5]
        extra = [(pos[i][0] + 1, pos[i][1]) for i in also if i in pos and pos[i][1].plane_id != surface_id]
        if not extra:
            return [c for _at, c in on]
        return [c for _at, c in sorted([*on, *extra], key=lambda x: x[0])]

    @property
    def active_surface(self):
        """Where drawing currently happens: a mesh or a plane."""
        return self.active_mesh or self.active_plane

    def plane_by_id(self, pid):
        return next((p for p in self.planes if p.id == pid), None)

    VERIFY_INDEX = False  # tests: check every answer of the directories (object_by_id, curve_by_id) against the search

    def _lists_stamp(self):
        """Detects whether an object or a line may have been added since the last look: new id assigned,
        undo step, other list, other length. (Hold the lists themselves, not their id: a new list at an
        old address would otherwise not be detected.)"""
        lists = (self.presets, self.custom, self.strokes, self.meshes, self.surfaces, self.nulls)
        return IDS_EPOCH[0], self.revision, lists, [len(x) for x in lists], [len(st.curves) for st in self.strokes]

    @staticmethod
    def _stamp_same(a, b):
        return (a is not None and a[0] == b[0] and a[1] == b[1] and all(x is y for x, y in zip(a[2], b[2]))
                and a[3] == b[3] and a[4] == b[4])

    def object_by_id(self, oid):
        """Object for an id (plane, line object, mesh, surface, null). Via a directory: the remembered position
        is rechecked on every call (lists change in place); "does not exist" holds as long as
        nothing can have been added (_lists_stamp). Before: all lists searched per question (harbor 11.11.: ~1,900
        questions per frame over ~1,000 objects)."""
        if oid is None:
            return None
        lists = (self.presets, self.custom, self.strokes, self.meshes, self.surfaces, self.nulls)
        index = self.__dict__.get("_object_index")
        at = index.get(oid) if index is not None else None
        if at is not None:
            objs = lists[at[0]]
            if at[1] < len(objs) and objs[at[1]].id == oid:
                if self.VERIFY_INDEX:
                    assert objs[at[1]] is self._object_scan(oid), f"directory returns a different object for id {oid}"
                return objs[at[1]]
        miss = self.__dict__.get("_object_miss")
        stamp = self._lists_stamp()
        if miss is not None and oid in miss[1] and self._stamp_same(miss[0], stamp):
            if self.VERIFY_INDEX:
                assert self._object_scan(oid) is None, f"directory does not know id {oid}, but it exists"
            return None
        counters.C["suche_objekt"] += 1
        index = self._object_index = {}
        for n in range(len(lists) - 1, -1, -1):  # backwards: with a duplicate id the first occurrence counts (like the search)
            objs = lists[n]
            for k in range(len(objs) - 1, -1, -1):
                index[objs[k].id] = (n, k)
        at = index.get(oid)
        if at is None:
            if miss is None or not self._stamp_same(miss[0], stamp):
                miss = self._object_miss = (stamp, set())
            miss[1].add(oid)
            return None
        return lists[at[0]][at[1]]

    def _object_scan(self, oid):
        return next((o for o in self.planes + self.strokes + self.meshes + self.surfaces + self.nulls if o.id == oid),
                    None)

    # --- Hierarchy (null objects) ---
    def tree_objects(self):
        """Everything that can lie under a null in the outliner (planes cannot)."""
        return self.strokes + self.surfaces + self.meshes + self.nulls

    def null_by_id(self, nid):
        # remember the position in the list and recheck on every call (list is changed in place): hierarchy questions
        # (is_hidden, ancestors) run thousands of times per frame, with USD hierarchies with dozens of nulls
        index = getattr(self, "_null_index", None) or {}
        k = index.get(nid)
        if k is None or k >= len(self.nulls) or self.nulls[k].id != nid:
            index = self._null_index = {n.id: i for i, n in enumerate(self.nulls)}
            k = index.get(nid)
        return None if k is None else self.nulls[k]

    def parent_of(self, obj):
        pid = getattr(obj, "parent", None)
        return None if pid is None else self.null_by_id(pid)

    def ancestors(self, obj):
        """Nulls above obj, the nearest first."""
        out, seen = [], set()
        node = self.parent_of(obj)
        while node is not None and node.id not in seen:
            seen.add(node.id)
            out.append(node)
            node = self.parent_of(node)
        return out

    def children(self, null):
        """Directly below (null None: top level): nulls first, then surfaces, strokes, meshes."""
        nid = null.id if null is not None else None
        batch = getattr(self, "_kids_batch", None)
        if batch is not None:
            return list(batch.get(nid, ()))
        return [o for o in self.nulls + self.surfaces + self.strokes + self.meshes if getattr(o, "parent", None) == nid]

    @contextlib.contextmanager
    def hierarchy_batch(self):
        """Many hierarchy questions in a row (outliner rebuild, descendants): children from a table built once
        instead of looking through all objects per call (USD hierarchies: thousands of nulls and meshes). Do not change the
        hierarchy in it."""
        if getattr(self, "_kids_batch", None) is not None:
            yield
            return
        kids = {}
        for o in self.nulls + self.surfaces + self.strokes + self.meshes:
            kids.setdefault(getattr(o, "parent", None), []).append(o)
        self._kids_batch = kids
        try:
            yield
        finally:
            self._kids_batch = None

    def descendants(self, null):
        """Everything below null (in depth), without null itself."""
        out, todo = [], [null]
        with self.hierarchy_batch():
            while todo:
                node = todo.pop()
                for c in self.children(node):
                    out.append(c)
                    if isinstance(c, NullObject):
                        todo.append(c)
        return out

    def is_hidden(self, obj):
        """Hidden: itself or via a null above (the eye of a null applies to everything below)."""
        if bool(getattr(obj, "hidden", False)) or any(n.hidden for n in self.ancestors(obj)):
            return True
        owner = getattr(obj, "owner", None)  # surface: hidden with its line object
        if owner is not None and isinstance(obj, SurfaceObject):
            st = self.object_by_id(owner)
            return isinstance(st, StrokeObject) and not isinstance(st, SurfaceObject) and bool(st.hidden)
        return False

    def defining_line_ids(self, surface):
        """ids of the existing lines (pieces) that describe a surface: border, outline, cross sections, longitudinal lines
        (not what merely lies on it)."""
        ids = list(getattr(surface, "patch_lines", None) or [])
        ids += [i for link, _fl in (getattr(surface, "links", None) or []) for i in self.link_ids(link)]
        ids += [i for src in (getattr(surface, "sources", None) or []) if src is not None and src[0] == "lines"
                for i in src[1]]
        ids += [r[0] for r in (getattr(surface, "rails", None) or []) if isinstance(r, (list, tuple))]
        return [sub for cid in ids for sub in self.split_pieces(cid) or []]

    def line_owner_counts(self, surface, owner_of=None):
        """{stroke id: number of lines} the surface hangs on (border, outline, cross sections, longitudinal lines)."""
        owner_of = owner_of if owner_of is not None else {c.id: st for st in self.strokes for c in st.curves}
        count = {}
        for sub in self.defining_line_ids(surface):
            st = owner_of.get(sub)
            if st is not None:
                count[st.id] = count.get(st.id, 0) + 1
        return count

    def assign_owners(self):
        """Phase 4 (harbor test 09.11.): every surface belongs to exactly one line object – the one with most of its lines.
        An existing owner stays as long as it exists and a line of the surface still belongs to it (no
        back and forth). If the line object is moved into another null, the surface goes along; a surface deliberately
        placed elsewhere stays. Returns True if something changed."""
        owner_of = {c.id: st for st in self.strokes for c in st.curves}
        strokes = {st.id: st for st in self.strokes}
        changed = False
        for f in self.surfaces:
            if f.aid:
                continue
            count = self.line_owner_counts(f, owner_of)
            cur = f.owner
            if cur not in strokes or cur not in count:
                new = max(count, key=lambda k: (count[k], -k)) if count else (cur if cur in strokes else None)
                if new != cur:
                    f.owner, changed = new, True
            st = strokes.get(f.owner)
            if st is None:
                continue
            seen = getattr(f, "_owner_parent", ...)
            if seen is not ... and seen != st.parent and f.parent != st.parent:
                f.parent, changed = st.parent, True  # line object was re-parented: its surface goes along
            f._owner_parent = st.parent  # surfaces deliberately placed elsewhere stay where they are
        return changed

    def binding_of(self, obj):
        """Binding level for the brief: its own or that of the nearest null above (None: none – the
        assignment applies). Returns (level, object, at which it is set)."""
        for o in (obj, *self.ancestors(obj)):
            if getattr(o, "binding", None):
                return o.binding, o
        return None, None

    def set_parent(self, objs, null):
        """Hang objs under null (None: top level). A null never comes under itself or its children.
        Returns the objects that were really re-parented."""
        moved = []
        for o in objs:
            if not hasattr(o, "parent"):
                continue  # planes do not belong in the hierarchy
            if null is not None and (o is null or (isinstance(o, NullObject) and null in self.descendants(o))):
                continue
            target = None if null is None else null.id
            if o.parent != target:
                o.parent = target
                moved.append(o)
        return moved

    def new_null(self, kind="null", origin=None, axes=None, parent=None):
        names = {"null": "Null", "symmetry": tr("Symmetrie"), "array": "Array"}
        null = NullObject(self._unique_name(names[kind], self.nulls), np.zeros(3) if origin is None else
                          np.array(origin, float), np.eye(3) if axes is None else np.array(axes, float), kind=kind,
                          parent=parent)
        self.nulls.append(null)
        return null

    @staticmethod
    def common_parent(objs):
        """Common null of all objs (id) – only if all lie in the same null, otherwise None."""
        parents = {getattr(o, "parent", None) for o in objs if o is not None}
        return parents.pop() if len(parents) == 1 else None

    def place_with_lines(self, obj, members):
        """New surface from lines (or surfaces): it lies in the same null as its lines if all lie in the same one –
        a surface from lines under a symmetry is mirrored along that way. members: lines or objects."""
        owners = [m if hasattr(m, "parent") else self.owner_of(m) for m in members if m is not None]
        if owners and all(o is not None for o in owners):
            obj.parent = self.common_parent(owners)
        return obj

    def _clean_parents(self):
        """Resolve references to vanished nulls and cycles (after delete, undo, load): robust instead of silently broken."""
        ids = {n.id for n in self.nulls}
        for o in self.tree_objects():
            if o.parent is not None and o.parent not in ids:
                o.parent = None
        for n in self.nulls:  # cycle (only possible through broken files): break open here
            seen, node = {n.id}, self.parent_of(n)
            while node is not None:
                if node.id in seen:
                    n.parent = None
                    break
                seen.add(node.id)
                node = self.parent_of(node)

    def owner_of(self, curve):
        """Stroke object the line belongs to. Via a directory (line -> object, position): the remembered position
        is rechecked on every call, "belongs to none" holds as long as nothing can have been added
        (_lists_stamp). Before: all lines of all objects searched per question – a selected group with 2,300 lines
        asked for every line when placing the gizmo and after every undo (harbor scene 12.11.: ~0.5 s)."""
        index = self.__dict__.get("_owner_index")
        at = index.get(id(curve)) if index is not None else None
        if at is not None:
            st, j = at
            if j < len(st.curves) and st.curves[j] is curve and self.object_by_id(st.id) is st:
                if self.VERIFY_INDEX:
                    assert st is self._owner_scan(curve), "directory names a different object than the search"
                return st
        miss = self.__dict__.get("_owner_miss")
        stamp = self._lists_stamp()
        if miss is not None and id(curve) in miss[1] and self._stamp_same(miss[0], stamp):
            if self.VERIFY_INDEX:
                assert self._owner_scan(curve) is None, "directory does not know the line, but it belongs to an object"
            return None
        counters.C["suche_besitzer"] += 1
        index = self._owner_index = {}
        for st in reversed(self.strokes):  # backwards: with a duplicate occurrence the first counts (like the search)
            cs = st.curves
            for j in range(len(cs) - 1, -1, -1):
                index[id(cs[j])] = (st, j)
        at = index.get(id(curve))
        if at is None:
            if miss is None or not self._stamp_same(miss[0], stamp):
                miss = self._owner_miss = (stamp, {})
            miss[1][id(curve)] = curve  # (holds the line: its id() stays valid)
            return None
        return at[0]

    def _owner_scan(self, curve):
        return next((s for s in self.strokes if any(c is curve for c in s.curves)), None)

    @staticmethod
    def _closest_on(curve, point):
        """(distance, segment index, parameter) of the nearest point of the line to point."""
        a, b = curve.points[:-1], curve.points[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
        d = np.linalg.norm(a + ab * t[:, None] - point, axis=1)
        i = int(np.argmin(d))
        return d[i], i, t[i]

    def _split_in(self, owner, curve, point, i, t):
        """Split line curve (from owner) at point; the pieces replace it in its place."""
        parts = split_curve(curve, point, i, t)
        if parts is None:
            return  # line end: already a node
        k = next(j for j, c in enumerate(owner.curves) if c is curve)
        owner.curves[k:k + 1] = list(parts)
        self.split_map[curve.id] = [parts[0].id, parts[1].id]

    def insert_knot_at(self, point, tolerance=1e-6):
        """Curve network: create a node at point on every line through point (the line stays whole). If a
        node already lies there, nothing happens; the node lies exactly on point so that it becomes the
        same point as the stroke that snaps there. Returns the number of new nodes."""
        from . import components
        made = 0
        for curve in self.curves:
            if curve.paint or len(curve.points) < 2:
                continue
            d, i, t = self._closest_on(curve, point)
            if d > tolerance:
                continue
            ks = components.knots(curve)
            if any(np.linalg.norm(curve.points[k] - point) < components.WELD_TOL for k in ks):
                continue
            k = components.insert_point(curve, i, t)
            if np.linalg.norm(curve.points[k] - point) > 0:
                pts = curve.points.copy()
                pts[k] = point
                curve.points = pts
                components.knots(curve)
            made += 1
        network.merge_at(self, point)  # deliberately connected: everything exactly there becomes one point
        return made

    def curve_bounds(self, curve):
        """(lo, hi) of the points of a line, remembered as long as its point array is the same (arrays are replaced, never
        changed in place) – for fast pre-selection in loops over all lines."""
        cache = self.__dict__.setdefault("_bounds_cache", {})
        hit = cache.get(id(curve))
        if hit is None or hit[0] is not curve.points:
            if len(cache) > 20000:
                cache.clear()
            hit = cache[id(curve)] = (curve.points, curve.points.min(axis=0), curve.points.max(axis=0), curve)
        return hit[1], hit[2]

    def split_all_at(self, point, tolerance=1e-6):
        """Split every line through point there (T-junction -> real node; overlapping ones stay coupled)."""
        for owner in self.strokes:
            for curve in list(owner.curves):
                if len(curve.points) < 2:
                    continue
                lo, hi = self.curve_bounds(curve)
                if np.any(point < lo - tolerance) or np.any(point > hi + tolerance):
                    continue
                d, i, t = self._closest_on(curve, point)
                if d <= tolerance:
                    self._split_in(owner, curve, point, i, t)

    def split_all_at_many(self, points, tolerance=1e-6):
        """split_all_at for many points at once: bounding boxes of all lines stacked once, per point only
        check the lines whose box contains it (extruding many surfaces: before, all lines per point, seconds)."""
        entries = [(st, c) for st in self.strokes for c in st.curves if len(c.points) >= 2]
        if not entries or not len(points):
            return
        boxes = [self.curve_bounds(c) for _st, c in entries]
        lo = np.array([b[0] for b in boxes]) - tolerance
        hi = np.array([b[1] for b in boxes]) + tolerance
        alive = np.ones(len(entries), bool)
        for point in points:
            point = np.asarray(point, float)
            for k in np.flatnonzero(alive & np.all(lo <= point, axis=1) & np.all(hi >= point, axis=1)):
                owner, curve = entries[k]
                if not any(c is curve for c in owner.curves):
                    alive[k] = False
                    continue
                d, i, t = self._closest_on(curve, point)
                if d > tolerance:
                    continue
                before = len(owner.curves)
                self._split_in(owner, curve, point, i, t)
                if len(owner.curves) == before:
                    continue  # line end: not split
                alive[k] = False
                parts = [self.curve_by_id(pid) for pid in self.split_map.get(curve.id, [])]
                parts = [x for x in parts if x is not None]
                if parts:
                    entries += [(owner, x) for x in parts]
                    pb = [self.curve_bounds(x) for x in parts]
                    lo = np.vstack([lo, np.array([b[0] for b in pb]) - tolerance])
                    hi = np.vstack([hi, np.array([b[1] for b in pb]) + tolerance])
                    alive = np.concatenate([alive, np.ones(len(parts), bool)])

    def next_paint_seq(self):
        """Number for the next paint stroke: the newest lies on top."""
        return 1 + max((c.seq for s in self.strokes for c in s.curves if c.paint), default=0)

    def curve_by_id(self, cid):
        # asked thousands of times per frame: remember the position (stroke, line) and recheck on every call (lists
        # are changed in place), like null_by_id
        index = getattr(self, "_curve_index", None) or {}
        at = index.get(cid)
        if at is not None:
            i, j = at
            if i < len(self.strokes) and j < len(self.strokes[i].curves) and self.strokes[i].curves[j].id == cid:
                return self.strokes[i].curves[j]
        # remember "does not exist" as long as no line can have been added – otherwise every question about a
        # vanished line rebuilds the directory over all lines (harbor 11.11.: 26 times per change, 9,000 lines)
        miss = self.__dict__.get("_curve_miss")
        stamp = self._lists_stamp()
        if miss is not None and cid in miss[1] and self._stamp_same(miss[0], stamp):
            if self.VERIFY_INDEX:
                assert not any(c.id == cid for s in self.strokes for c in s.curves),                     f"directory does not know line {cid}, but it exists"
            return None
        index = self._curve_index = {c.id: (i, j) for i, s in enumerate(self.strokes) for j, c in enumerate(s.curves)}
        counters.C["linien_index"] += len(index)
        at = index.get(cid)
        if at is None:
            if miss is None or not self._stamp_same(miss[0], stamp):
                miss = self._curve_miss = (stamp, set())
            miss[1].add(cid)
            return None
        return self.strokes[at[0]].curves[at[1]]

    # --- Notes ---
    def note_by_id(self, nid):
        """Note (normal or shot, also the global ones) for an id."""
        if nid == self.global_note.id:
            return self.global_note
        return next((n for n in self.notes if n.id == nid), None) or self.timing.note_by_id(nid)

    def is_shot_note(self, note):
        return note is not None and any(n is note for n in self.timing.all_notes())

    def _sync_notes(self):
        """Leader lines travel with their lines; the note follows its first connection. Split or
        replaced lines are relinked via position, deleted ones leave the anchor standing."""
        changed = False
        curves = None
        for note in self.notes + self.timing.notes:  # shot notes hang on lines in the same way (of the flipbook planes)
            for k, anchor in enumerate(note.anchors):
                if anchor.curve_id is None:
                    continue
                curve = self.curve_by_id(anchor.curve_id)
                pieces = self.split_pieces(anchor.curve_id) if curve is None else None
                if pieces:  # split: the piece into which the share t falls (by length of the pieces)
                    lens = [float(np.linalg.norm(np.diff(self.curve_by_id(i).points, axis=0), axis=1).sum()) for i in pieces]
                    total, acc = max(sum(lens), 1e-12), 0.0
                    for i, length in zip(pieces, lens):
                        if anchor.t * total <= acc + length + 1e-12 or i == pieces[-1]:
                            anchor.curve_id, anchor.t = i, min(max((anchor.t * total - acc) / max(length, 1e-12), 0.0), 1.0)
                            break
                        acc += length
                    curve = self.curve_by_id(anchor.curve_id)
                if curve is None:
                    curves = self.pickable_curves() if curves is None else curves
                    curve, t = notes_geo.relink(curves, anchor.point)
                    if curve is None:
                        changed |= not anchor.lost
                        anchor.lost = True
                        continue
                    anchor.curve_id, anchor.t = curve.id, t
                anchor.lost = False
                p = notes_geo.curve_point(curve.points, anchor.t)
                if np.any(np.abs(p - anchor.point) > 1e-9):
                    if k == 0:
                        note.pos = note.pos + (p - anchor.point)
                    anchor.point = p
                    changed = True
            for anchor in note.anchors:  # arms of the note octopus on single lines travel along likewise
                for m in anchor.members:
                    if m.curve_id is None:
                        continue
                    curve = self.curve_by_id(m.curve_id)
                    if curve is None and m.point is not None:  # split/replaced: relink via position
                        curves = self.pickable_curves() if curves is None else curves
                        curve, t = notes_geo.relink(curves, m.point)
                        if curve is None:
                            continue  # deleted: the arm stays (the pre-check reports it)
                        owner = self.owner_of(curve)
                        m.curve_id, m.t, m.obj_id = curve.id, t, owner.id if owner is not None else m.obj_id
                    if curve is None:
                        continue
                    p = notes_geo.curve_point(curve.points, m.t)
                    if m.point is None or np.any(np.abs(p - m.point) > 1e-9):
                        m.point = p
                        changed = True
        return changed

    def asset_bytes(self, name):
        """Image of a note board (from the scene file or memory; older scenes: from the folder next to it)."""
        data = self.assets.get(name)
        if data is None and self.asset_dir:
            try:
                with open(os.path.join(self.asset_dir, name), "rb") as f:
                    data = self.assets[name] = f.read()
            except OSError:
                return None
        return data

    def _note_files(self):
        boards = {item["file"] for n in self.notes + [self.global_note] + self.timing.all_notes()
                  for item in n.board.get("items", []) if item.get("kind") == "image"}
        return boards | {o.picture["asset"] for o in list(self.planes) + list(self.surfaces) if o.picture}

    def _has_timing(self):
        """Anything to save in timing? Scenes without timing stay unchanged."""
        t = self.timing
        return bool(t.notes or t.cams or t.motions or t.global_note.board or timing.flipbook_planes(self)
                    or (t.start, t.end, t.fps, t.frame) != (timing.START, timing.END, timing.FPS, timing.START))

    def new_flipbook(self, origin, axis_u, axis_v):
        """New flipbook plane (timing) together with its stroke object for the drawings."""
        plane = Plane(self._unique_name("Flipbook", self.custom), np.array(origin, float), axis_u.copy(), axis_v.copy(),
                      flipbook=True)
        self.custom.append(plane)
        self.activate(plane)
        self.flip_stroke(plane)
        return plane

    def flip_stroke(self, plane):
        """Stroke object with the drawings of the flipbook plane (if missing, it is created)."""
        found = next((s for s in self.strokes if s.flipbook == plane.id), None)
        if found is None:
            found = StrokeObject(plane.name, flipbook=plane.id)
            self.strokes.append(found)
        return found

    @staticmethod
    def assets_dir_for(path):
        base = path[:-5] if path.lower().endswith(".json") else path
        return base + ".assets"

    def _unique_name(self, prefix, existing):
        names = {o.name for o in existing}
        n = 1
        while f"{prefix} {n}" in names:
            n += 1
        return f"{prefix} {n}"

    # --- Drawing surfaces ---
    def activate(self, plane):
        self.active_plane = plane
        self.active_mesh = None
        if not plane.preset and not plane.flipbook:  # key 4 (last own plane) never leads to a flipbook plane
            self.last_custom = plane

    def activate_mesh(self, mesh):
        self.active_mesh = mesh

    def new_plane(self, origin, axis_u, axis_v):
        plane = Plane(self._unique_name(tr("Ebene"), self.custom), np.array(origin, float), axis_u.copy(), axis_v.copy())
        self.custom.append(plane)
        self.activate(plane)
        return plane

    def _detach_curves(self, surface_id):
        for c in self.curves:
            if c.plane_id == surface_id:
                c.plane_id = None

    def _remove_plane(self, plane):
        self.custom.remove(plane)
        self._detach_curves(plane.id)
        if self.last_custom is plane:
            self.last_custom = self.custom[-1] if self.custom else None
        if self.active_plane is plane:
            self.active_plane = self.presets[0]

    # --- Meshes ---
    def import_mesh(self, path):
        """Load a file and create it as a new mesh object. USD as in Solaris: one mesh per mesh prim, above each group
        (Xform, Scope …) a null at the position of the prim, under a null for the file – show/hide and move
        individually. Returns the topmost object. Errors are raised as an exception."""
        name = os.path.splitext(os.path.basename(path))[0]
        path = os.path.abspath(path)
        if os.path.splitext(path)[1].lower() in geometry.USD_EXT:
            tree = geometry.usd_tree(path)
            if sum(n["kind"] == "mesh" for n in tree) > 1:
                return self._import_usd_tree(path, name, tree)
        mesh = MeshObject(name, path, geometry.load(path))
        self.meshes.append(mesh)
        return mesh

    def _import_usd_tree(self, path, name, tree):
        top = NullObject(name)
        made = {None: top}
        meshes = []
        for node in tree:
            parent = made.get(node["parent"], top)
            if node["kind"] == "group":
                made[node["path"]] = NullObject(node["name"], np.array(node["origin"], float),
                                                np.array(node["axes"], float), parent=parent.id)
            else:
                geo = geometry.load_usd(path, node["path"])
                if geo.triangle_count:
                    meshes.append(MeshObject(node["name"], path, geo, parent=parent.id, prim=node["path"]))
        if not meshes:
            raise ValueError(tr("Die USD-Datei enthält keine sichtbaren Meshes."))
        lo = np.min([m.geo.bbox[0] for m in meshes], axis=0)
        hi = np.max([m.geo.bbox[1] for m in meshes], axis=0)
        top.origin = (lo + hi) / 2 * np.array([1.0, 0.0, 1.0]) + np.array([0.0, lo[1], 0.0])  # center on the ground
        by_id = {n.id: n for n in made.values()}
        alive, todo = set(), [m.parent for m in meshes]
        while todo:  # groups stay if a mesh lies under them (empty prims drop out)
            nid = todo.pop()
            if nid is not None and nid not in alive:
                alive.add(nid)
                todo.append(by_id[nid].parent)
        self.nulls += [n for n in made.values() if n.id in alive]
        self.meshes += meshes
        return top

    # --- Strokes ---
    def new_stroke(self, parent=...):
        """New stroke object; it lies in the same null as the previously active one (drawing continues in the same
        group, e.g. under the symmetry). parent: id or None for a specific place."""
        if parent is ...:
            active = self.active_stroke if self.active_stroke in self.strokes else None
            parent = active.parent if active is not None else None
        stroke = StrokeObject(self._unique_name("Stroke", self.strokes), parent=parent)
        self.strokes.append(stroke)
        self.active_stroke = stroke
        return stroke

    def add_curve(self, points, surface_id, widths=None):
        pts = np.asarray(points, float)
        if len(pts) < 2 or float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) < 1e-7:
            return  # stroke without length (e.g. all corners of a tiny shape snapped to the same point)
        if self.active_stroke is None or self.active_stroke not in self.strokes:
            self.new_stroke()
        widths = None if widths is None else np.asarray(widths, float)
        self.active_stroke.curves.append(Curve(np.asarray(points, float), surface_id, widths))

    def clear_curves(self):
        for s in self.strokes:
            s.curves.clear()

    # --- Deleting ---
    def delete(self, objects):
        """Delete planes and meshes (lines stay) as well as stroke objects (with their lines). A null takes everything
        below it along (as in Cinema 4D; only the null: dissolve_null)."""
        objects = list(objects)
        for obj in list(objects):
            if isinstance(obj, NullObject):
                objects += [o for o in self.descendants(obj) if o not in objects]
        gone = {o.id for o in objects if isinstance(o, StrokeObject) and not isinstance(o, SurfaceObject)}
        objects += [f for f in self.surfaces if f.owner in gone and f not in objects]  # surfaces belong to it
        for obj in list(objects):
            if isinstance(obj, Plane) and obj.flipbook:  # flipbook plane takes its drawings along
                objects += [s for s in self.strokes if s.flipbook == obj.id and s not in objects]
        for obj in objects:
            if isinstance(obj, NullObject):
                if obj in self.nulls:
                    self.nulls.remove(obj)
                continue
            if isinstance(obj, Plane) and not obj.preset and obj in self.custom:
                self._remove_plane(obj)
            elif isinstance(obj, SurfaceObject) and obj in self.surfaces:
                self.surfaces.remove(obj)  # lines on it stay
                self._detach_curves(obj.id)
                if self.active_mesh is obj:
                    self.active_mesh = None
            elif isinstance(obj, StrokeObject) and obj in self.strokes:
                self.strokes.remove(obj)
                if self.active_stroke is obj:
                    self.active_stroke = self.strokes[-1] if self.strokes else None
            elif isinstance(obj, MeshObject) and obj in self.meshes:
                self.meshes.remove(obj)
                self._detach_curves(obj.id)
                if self.active_mesh is obj:
                    self.active_mesh = None
        self._clean_parents()

    def dissolve_null(self, null):
        """Dissolve null: it disappears, its children move up one level (objects stay unchanged)."""
        if null not in self.nulls:
            return []
        kids = self.children(null)
        for o in kids:
            o.parent = null.parent
        self.nulls.remove(null)
        return kids

    # --- Undo / Redo ---
    def checkpoint(self, burst=None):
        """Save the state before a change. The same burst id within a
        second (e.g. mouse wheel notches) is combined into one undo step."""
        self.dirty = True
        self.revision += 1
        now = time.monotonic()
        if burst is None and self._burst[0] == "merge" and now - self._burst[1] < 1.0:
            self._burst = (None, now)  # belongs to the step just saved (merge_next)
            return
        if burst is not None and self._burst[0] == burst and now - self._burst[1] < 1.0:
            self._burst = (burst, now)
            return
        self._undo.append(self._history_copy())
        self._trim_undo()
        self._redo.clear()
        self._burst = (burst, now)

    def _trim_undo(self):
        """Keep at most UNDO_LIMIT scene copies; selection steps do not count (they displace no work)."""
        while sum(1 for e in self._undo if not isinstance(e, _Selection)) > UNDO_LIMIT:
            k = next(i for i, e in enumerate(self._undo) if not isinstance(e, _Selection))
            del self._undo[:k + 1]

    def checkpoint_selection(self, state):
        """Selection has changed (without a change to the scene): remember the previous selection as its own undo step."""
        self._undo.append(_Selection(state))
        self._redo.clear()
        self._burst = (None, 0.0)

    def can_paint_step(self):
        """Is a light undo step enough for the next paint stroke? Not if a new stroke object is created for it
        or it belongs to the step just saved (merge_next)."""
        merging = self._burst[0] == "merge" and time.monotonic() - self._burst[1] < 1.0
        return not merging and self.active_stroke is not None and self.active_stroke in self.strokes

    def checkpoint_paint(self, curves):
        """Hold new paint strokes in the active stroke object as an undo step (instead of checkpoint beforehand)."""
        self.dirty = True
        self.revision += 1
        self._undo.append(_PaintStep(self.active_stroke.id, {c.id for c in curves}))
        self._trim_undo()
        self._redo.clear()
        self._burst = (None, time.monotonic())

    def _toggle_paint(self, step):
        """Take the paint strokes of a _PaintStep out or attach them again (undo, redo)."""
        self.dirty = True
        self.revision += 1
        self._burst = (None, 0.0)
        stroke = next((s for s in self.strokes if s.id == step.stroke_id), None)
        if stroke is None:
            return step
        if step.curves is None:
            step.curves = [c for c in stroke.curves if c.id in step.ids]
            stroke.curves[:] = [c for c in stroke.curves if c.id not in step.ids]
        else:
            stroke.curves.extend(step.curves)
            step.curves = None
        return step

    def _current_selection(self):
        return self.selection_io[0]() if self.selection_io else None

    def merge_next(self):
        """The next checkpoint (right after) belongs to the undo step just saved."""
        self._burst = ("merge", time.monotonic())

    def undo(self):
        if not self._undo:
            return False
        entry = self._undo.pop()
        if isinstance(entry, _Selection):  # only the selection back
            self._redo.append(_Selection(self._current_selection()))
            self._apply_selection(entry.state)
        elif isinstance(entry, _PaintStep):
            self._redo.append(self._toggle_paint(entry))
        else:
            self._redo.append(self._history_copy())
            self._restore(entry, fresh=True)
        self._burst = (None, 0.0)
        return True

    def redo(self):
        if not self._redo:
            return False
        entry = self._redo.pop()
        if isinstance(entry, _Selection):
            self._undo.append(_Selection(self._current_selection()))
            self._apply_selection(entry.state)
        elif isinstance(entry, _PaintStep):
            self._undo.append(self._toggle_paint(entry))
        else:
            self._undo.append(self._history_copy())
            self._restore(entry, fresh=True)
        return True

    def _apply_selection(self, state):
        self.sel_revision += 1
        if state is not None and self.selection_io:
            self.selection_io[1](state)

    def _history_copy(self):
        """Snapshot for the history, taken out of garbage collection (_freeze_history). Collected BEFORE
        copying: afterwards the collector would have to search the tens of thousands of fresh objects of the copy (harbor scene:
        30 ms per step), although they contain no garbage."""
        gc.collect()
        snap = self._snapshot()
        gc.freeze()
        return snap

    def _snapshot(self):
        """Copy of the scene state for the history. Objects, lists and dictionaries are copied, numpy arrays
        SHARED: they are never changed in place in the scene, only replaced (all caches that compare by identity
        also rely on this). Shared arrays become write-protected – a violation aborts
        immediately with an error message instead of silently corrupting an older undo step. After undo, unchanged
        lines and surfaces are thus the same arrays as before: display and snapping recompute nothing."""
        ids = lambda o: o.id if o else None
        state = (self.presets, self.custom, self.strokes, self.meshes, self.surfaces, self.nulls,
                 self.active_plane.id,
                 ids(self.active_mesh), ids(self.last_custom), ids(self.active_stroke),
                 list(self.multi_ids), (self.notes, self.global_note, self.timing),
                 self._current_selection())
        memo, records = self._twin_memo()
        snap = self._shared_copy(state, memo)
        for group in (*(st.curves for st in self.strokes), *(s.curves for s in self.surfaces)):
            for c in group:
                if id(c) not in records:
                    records[id(c)] = (c, c.__dict__.get("_v"), c.__dict__.get("_pid_pos"), memo[id(c)])
        for s in self.surfaces:
            if id(s) not in records:
                records[id(s)] = (s, None, None, memo[id(s)])
        self._twins = records
        if self.VERIFY_UNDO:
            why = _diff(snap, self._shared_copy(state, {}), set(), "scene")
            assert why is None, f"undo copy with shared images differs from the full copy - {why}"
        return snap

    # --- Undo copy only for what changed ---
    # A snapshot is immutable. Every line and every surface of the live scene remembers its counterpart in the
    # last snapshot (_twins). If it is unchanged at the next step, the same counterpart goes in again,
    # instead of copying it anew (harbor 11.11.: 17,500 lines and grid rows, 308 surfaces – 75 ms per stroke, click,
    # release). When restoring, the other way round: what is unchanged stays as a live object; only the rest
    # is copied from the snapshot. The live scene never gets an object from the history itself – otherwise
    # the next change would corrupt all steps that share the same counterpart.
    # Unchanged means for lines: no field set anew (Curve._v), point IDs and point attributes equal in value (they
    # are also changed in place), remembered node position the same. For surfaces: every field piece by piece
    # (_surface_same), plus all their grid rows and the lines of their sync unchanged.
    VERIFY_UNDO = False  # tests: additionally copy every copy and every restore fully and compare

    def _twin_memo(self):
        """({id(live object): valid counterpart}, {id: entry}) for everything unchanged."""
        reg = self.__dict__.get("_twins") or {}
        memo, keep = {}, {}
        if not reg:
            return memo, keep
        for group in (*(st.curves for st in self.strokes), *(s.curves for s in self.surfaces)):
            counters.C["undo_geprueft"] += len(group)
            for c in group:
                rec = reg.get(id(c))
                if rec is None or rec[0] is not c:
                    continue
                d, t = c.__dict__, rec[3].__dict__
                if rec[1] != d.get("_v") or d.get("_pid_pos") is not rec[2]:
                    continue
                p = d.get("pattr")
                if p is not None and p != t.get("pattr"):
                    continue
                p = d.get("pids")
                if p is not None:
                    if p != t.get("pids"):
                        continue
                    memo[id(p)] = t["pids"]  # caches of the surfaces also hold the point IDs by identity
                memo[id(c)] = rec[3]
                keep[id(c)] = rec
        for s in self.surfaces:
            rec = reg.get(id(s))
            if rec is not None and rec[0] is s and _surface_same(s, rec[3], memo):
                memo[id(s)] = rec[3]
                keep[id(s)] = rec
        return memo, keep

    @staticmethod
    def _shared_copy(state, memo):
        dispatch = copy._deepcopy_dispatch
        before = dispatch.get(np.ndarray)
        dispatch[np.ndarray] = _share_array
        try:
            return copy.deepcopy(state, memo)
        finally:
            if before is None:
                del dispatch[np.ndarray]
            else:
                dispatch[np.ndarray] = before

    def _thaw(self, snap, fresh=False):
        """Snapshot as live state: unchanged lines and surfaces stay the live objects, the rest
        is copied (the snapshot itself stays untouched). fresh: the current state was saved immediately before
        (undo/redo) – then every live object is unchanged compared to its counterpart."""
        keep = self.__dict__.get("_twins") or {} if fresh else self._twin_memo()[1]
        memo = {id(rec[3]): rec[0] for rec in keep.values()}
        for rec in keep.values():
            p = rec[0].__dict__.get("pids") if type(rec[0]) is Curve else None
            if p is not None:
                memo[id(rec[3].__dict__["pids"])] = p
        live = self._shared_copy(snap, memo)
        records = {}
        for old, new in zip((*snap[2], *snap[4]), (*live[2], *live[4])):
            for t, c in zip(old.curves, new.curves):
                records[id(c)] = (c, c.__dict__.get("_v"), c.__dict__.get("_pid_pos"), t)
        for t, s in zip(snap[4], live[4]):
            records[id(s)] = (s, None, None, t)
        self._twins = records
        if self.VERIFY_UNDO:
            why = _diff(live, self._shared_copy(snap, {}), set(), "scene")
            assert why is None, f"restored state differs from the snapshot - {why}"
        return live

    def _restore(self, snap, fresh=False):
        self._burst = (None, 0.0)
        self.dirty = True
        self.revision += 1
        snap = self._thaw(snap, fresh)
        presets, custom, strokes, meshes, surfaces, nulls, active_id, mesh_id, last_id, stroke_id, multi, notes, sel = snap
        self.nulls = nulls
        frame = self.timing.frame  # the current frame is not an undo step
        self.notes, self.global_note, self.timing = notes
        self.timing.frame = self.timing.clamp(frame)
        self.multi_ids = multi
        self.presets, self.custom, self.strokes, self.meshes = presets, custom, strokes, meshes
        self.surfaces = surfaces
        self.active_plane = self.plane_by_id(active_id) or self.presets[0]
        self.active_mesh = next((m for m in self.meshes + self.surfaces if m.id == mesh_id), None)
        self.last_custom = self.plane_by_id(last_id)
        self.active_stroke = next((s for s in self.strokes if s.id == stroke_id), None)
        if sel is not None and self.selection_io:  # selection from back then
            self.selection_io[1](sel)

    # --- Saving / loading ---
    def to_dict(self, base_dir):
        r = lambda a: [round(float(x), 5) for x in a]

        def rel(path):
            try:
                return os.path.relpath(path, base_dir).replace("\\", "/")
            except ValueError:  # another drive under Windows
                return path.replace("\\", "/")

        return {
            "format": FILE_FORMAT,
            "version": FILE_VERSION,
            "planes": [{
                "id": p.id, "name": p.name, "preset": p.preset, "origin": r(p.origin),
                "axis_u": r(p.axis_u), "axis_v": r(p.axis_v), "size_u": p.size_u, "size_v": p.size_v,
                **({"hidden": True} if p.hidden else {}),
                **({"picture": p.picture} if p.picture else {}),
                **({"flipbook": True, "blank_keys": [int(k) for k in p.blank_keys]} if p.flipbook else {}),
            } for p in self.planes],
            "nulls": [{
                "id": n.id, "name": n.name, "kind": n.kind, "origin": r(n.origin), "axes": [r(col) for col in n.axes.T],
                "params": _json_params(n.params),
                **({"parent": n.parent} if n.parent is not None else {}),
                **({"hidden": True} if n.hidden else {}),
                **({"color": list(n.color)} if n.color else {}),
                **({"binding": n.binding} if n.binding else {}),
            } for n in self.nulls],
            "meshes": [{
                "id": m.id, "name": m.name, "path": rel(m.path), "xform": [r(row) for row in m.xform],
                **({"prim": m.prim} if m.prim else {}),
                **({"parent": m.parent} if m.parent is not None else {}),
                **({"hidden": True} if m.hidden else {}),
                **({"binding": m.binding} if m.binding else {}),
            } for m in self.meshes],
            "strokes": [{
                "id": s.id, "name": s.name, "axes": [r(col) for col in s.axes.T],
                "curves": [_curve_dict(c, r) for c in s.curves],
                **({"parent": s.parent} if s.parent is not None else {}),
                **({"hidden": True} if s.hidden else {}),
                **({"flipbook": s.flipbook} if s.flipbook is not None else {}),
            } for s in self.strokes],
            "surfaces": [{
                "id": f.id, "name": f.name, "kind": f.kind, "axes": [r(col) for col in f.axes.T],
                "outline": [_curve_dict(c, r) for c in f.curves],
                **({"parent": f.parent} if f.parent is not None else {}),
                **({"hidden": True} if f.hidden else {}),
                "material": f.material, "seed": round(float(f.seed), 3),
                **({"opacity": round(float(f.opacity), 3)} if f.opacity < 1.0 else {}),
                **({"resolution": round(float(f.resolution), 4)} if f.resolution != 1.0 else {}),
                **({"aid": True} if f.aid else {}),
                **({"picture": f.picture} if f.picture else {}),
                **({"frozen": {"verts": _rows(f.frozen[0][0]), "tris": f.frozen[0][1].tolist(),
                               **({"uv": _rows(f.frozen[0][2])} if f.frozen[0][2] is not None else {}),
                               "outlines": [_rows(line) for line in f.frozen[1]],
                               "sample": _rows(f.frozen[2]),
                               "lines": [[cid, _rows(pts)] for cid, pts in f.frozen_lines.items()]}}
                   if f.frozen is not None and not getattr(f, "_carried", False) else {}),  # only carried along in the drag: never save
                **({"binding": f.binding} if f.binding else {}),
                **({"tint": [round(float(x), 4) for x in f.tint]} if f.tint is not None else {}),
                **({"segments": [_rows(seg) for seg in f.segments]}
                   if isinstance(f, ExtrudeSurface) else {}),
                **({"links": [[cid, flipped] for cid, flipped in f.links], "counts": list(f.counts),
                    "rest": _rows(f.rest), "link_cover": f.link_cover,
                    "attached": [[cid, [r(x) for x in a["uv"]]] for cid, a in f.attached.items()]}
                   if type(f) is SurfaceObject and f.rest is not None else {}),
                **({"patch_lines": list(f.patch_lines), "patch_order": [list(o) for o in f.patch_order],
                    "patch_corners": [list(c) for c in f.patch_corners], "patch_arc": bool(f.patch_arc),
                    **({"net": {"r": int(f.net["r"]), "m": int(f.net["m"]),
                                "u": [[list(ids), int(at)] for ids, at in f.net.get("u", [])],
                                "v": [[list(ids), int(at)] for ids, at in f.net.get("v", [])],
                                **({"idx": [[int(a["n"]), [round(float(x), 5) for x in a["fi"]]] for a in f.net["idx"]]}
                                   if f.net.get("idx") else {})}} if f.net else {}),
                    **({"span": int(f.span)} if f.span != loft.SPAN_SAMPLES else {}),
                    **({"bnd": [[int(b["n"]), [round(float(x), 5) for x in b["idx"]]] for b in f.bnd]}
                       if f.bnd and all(b is not None for b in f.bnd) else {})}
                   if isinstance(f, PatchSurface) else {}),
                **({"closed": f.closed, "wrap": f.wrap, "sources": [list(src) if src is not None else None for src in f.sources],
                    "rails": [list(r) for r in f.rails],
                    "attached": [[cid, [r(x) for x in a["uv"]], [bool(x) for x in a["on"]]]
                                 for cid, a in f.attached.items()]}
                   if isinstance(f, LoftSurface) else {}),
            } for f in self.surfaces],
            "global_note": {"title": self.global_note.title, "board": _board_videos(self.global_note.board, rel)},
            "sun": [round(float(x), 2) for x in self.sun],
            "light_bands": int(self.light_bands),
            **({"note_scale": round(float(self.note_scale), 4)} if self.note_scale != 1.0 else {}),
            "brief_task": self.brief_task,
            **({"brief_clarified": self.brief_clarified} if self.brief_clarified else {}),
            **({"timing": {**timing.to_dict(self.timing),
                           "global_note": {"id": self.timing.global_note.id, "title": self.timing.global_note.title,
                                           "board": _board_videos(self.timing.global_note.board, rel)},
                           "notes": [_note_dict(n, r, rel) for n in self.timing.notes]}}
               if self._has_timing() else {}),
            "notes": [_note_dict(n, r, rel) for n in self.notes],
            # split/merged lines: old id -> today's pieces. Surfaces, lofts and notes still refer
            # to the old id; without this list the surface lost its border line on loading (Alpaka tank 14.11.)
            **({"splits": splits} if (splits := self._saved_splits()) else {}),
            "active_plane": self.active_plane.id,
            "active_mesh": self.active_mesh.id if self.active_mesh else None,
            "active_stroke": self.active_stroke.id if self.active_stroke else None,
            # images of the note boards in the file (like PureRef): one file contains everything
            "assets": {name: base64.b64encode(data).decode("ascii") for name in sorted(self._note_files())
                       if (data := self.asset_bytes(name)) is not None},
        }

    def _saved_splits(self):
        """split_map for saving: per old line the current pieces (resolved, only complete chains)."""
        alive = {c.id for s in self.strokes for c in s.curves}  # not curve_by_id: every missing id rebuilt

        def pieces(cid, depth=0):
            if cid in alive:
                return [cid]
            parts = self.split_map.get(cid)
            if not parts or depth > 32:
                return None
            out = []
            for p in parts:
                sub = pieces(p, depth + 1)
                if sub is None:
                    return None
                out += sub
            return out

        return [[cid, sub] for cid in sorted(k for k in self.split_map if isinstance(k, int) and k not in alive)
                if (sub := pieces(cid))]

    @classmethod
    def from_dict(cls, data, base_dir):  # noqa: C901 (one place for all file versions)
        if data.get("format") != FILE_FORMAT:
            raise ValueError(tr("Keine Mule-Szene."))
        scene = cls()
        scene.assets = {name: base64.b64decode(text) for name, text in (data.get("assets") or {}).items()}
        id_map = {}
        arr = lambda a: np.array(a, float)
        presets = [p for p in data["planes"] if p["preset"]]
        for target, p in zip(scene.presets, presets):
            target.origin, target.axis_u, target.axis_v = arr(p["origin"]), arr(p["axis_u"]), arr(p["axis_v"])
            target.hidden = bool(p.get("hidden", False))
            target.picture = p.get("picture")
            id_map[p["id"]] = target.id
        for p in data["planes"]:
            if p["preset"]:
                continue
            plane = Plane(p["name"], arr(p["origin"]), arr(p["axis_u"]), arr(p["axis_v"]), p["size_u"], p["size_v"])
            plane.hidden = bool(p.get("hidden", False))
            plane.picture = p.get("picture")
            if p.get("flipbook"):
                plane.flipbook, plane.blank_keys = True, sorted(int(k) for k in p.get("blank_keys", []))
            scene.custom.append(plane)
            id_map[p["id"]] = plane.id
        for n in data.get("nulls", []):  # rewrite references (parent, lines of the parameters) only at the end
            kind = n.get("kind", "null") if n.get("kind") in NULL_KINDS else "null"
            null = NullObject(n["name"], arr(n["origin"]), arr(n["axes"]).T, kind=kind, params=dict(n.get("params", {})),
                              hidden=bool(n.get("hidden", False)), parent=n.get("parent"),
                              color=tuple(int(x) for x in n["color"]) if n.get("color") else None,
                              binding=n.get("binding"))
            scene.nulls.append(null)
            id_map[n["id"]] = null.id
        failed = set()  # file missing: one warning, not one per mesh prim
        for m in data.get("meshes", []):
            path = os.path.normpath(os.path.join(base_dir, m["path"]))
            prim = m.get("prim")
            try:
                if path in failed:
                    raise OSError
                geo, missing = geometry.load(path, prim), False
            except Exception as exc:
                geo, missing = geometry.EMPTY, True
                if path not in failed:
                    scene.load_warnings.append(f"{m['name'] if not prim else os.path.basename(path)}: {path} ({exc})")
                if not os.path.isfile(path):
                    failed.add(path)
            mesh = MeshObject(m["name"], path, geo, arr(m["xform"]), missing, prim=prim)
            mesh.hidden = bool(m.get("hidden", False))
            mesh.parent = m.get("parent")
            mesh.binding = m.get("binding")
            scene.meshes.append(mesh)
            id_map[m["id"]] = mesh.id
        lofts, fills = [], []
        for f in data.get("surfaces", []):
            outline = [_curve_from(c, id_map) for c in f["outline"]]
            if f.get("kind") in ("loft", "patch"):
                cls_ = PatchSurface if f.get("kind") == "patch" else LoftSurface
                surface = cls_(f["name"], outline, axes=arr(f["axes"]).T, closed=f.get("closed", True),
                                      wrap=bool(f.get("wrap", False)),
                                      sources=[tuple(src) if src else None for src in f.get("sources", [])],
                                      rails=[tuple(r) for r in f.get("rails", [])])
                surface.attached = {cid: {"uv": arr(uv), "on": np.array(on, bool), "last": None}
                                    for cid, uv, on in f.get("attached", [])}
                if isinstance(surface, PatchSurface):
                    surface.patch_lines = list(f.get("patch_lines", []))
                    surface.patch_order = [(int(i), bool(fl)) for i, fl in f.get("patch_order", [])]
                    surface.patch_corners = [(int(j), float(t)) for j, t in f.get("patch_corners", [])]
                    surface.patch_arc = bool(f.get("patch_arc", False))
                    surface.net = f.get("net")  # older files: None -> the grid is laid at the first sync
                    if surface.net and surface.net.get("idx"):
                        surface.net["idx"] = [{"n": int(n), "fi": arr(fi)} for n, fi in surface.net["idx"]]
                    surface.span = int(f.get("span", loft.SPAN_SAMPLES))
                    if f.get("bnd"):
                        surface.bnd = [{"n": int(n), "idx": arr(idx), "pts": None} for n, idx in f["bnd"]]
                lofts.append(surface)
            elif f.get("kind") == "extrude":
                surface = ExtrudeSurface(f["name"], outline, axes=arr(f["axes"]).T,
                                         segments=[arr(seg) for seg in f.get("segments", [])],
                                         direction=arr(f["direction"]) if "direction" in f else None)
            else:
                surface = SurfaceObject(f["name"], outline, axes=arr(f["axes"]).T, kind=f.get("kind", "fill"))
                if "links" in f:
                    surface.links = [(cid, bool(fl)) for cid, fl in f["links"]]
                    surface.counts = [int(c) for c in f.get("counts", [])]
                    surface.rest = arr(f["rest"]).reshape(-1, 2) if f.get("rest") else None
                    surface.link_cover = float(f.get("link_cover", 0.0))
                    surface.attached = {cid: {"uv": arr(uv).reshape(-1, 2), "last": None} for cid, uv in f.get("attached", [])}
                fills.append(surface)
            surface.hidden = bool(f.get("hidden", False))
            surface.parent = f.get("parent")
            surface.binding = f.get("binding")
            surface.material = {"nebel": "spruehfarbe"}.get(f.get("material", "papier"), f.get("material", "papier"))
            # older files without offset: derived fixed from name and position (not a different one on every load)
            surface.seed = float(f["seed"]) if "seed" in f else                 (zlib.crc32(f"{f['name']}/{len(scene.surfaces)}".encode()) % 100000) / 100.0
            surface.tint = tuple(float(x) for x in f["tint"]) if f.get("tint") else None
            surface.opacity = float(f.get("opacity", 1.0))
            surface.resolution = float(f.get("resolution", 1.0))
            surface.aid = bool(f.get("aid", False))
            surface.picture = f.get("picture")
            if f.get("frozen"):
                fz = f["frozen"]
                uv = arr(fz["uv"]) if fz.get("uv") else None
                surface.frozen = ((arr(fz["verts"]), np.array(fz["tris"], np.int32), uv),
                                  [arr(line) for line in fz["outlines"]], arr(fz["sample"]))
                surface.frozen_lines = {cid: arr(pts) for cid, pts in fz.get("lines", [])}
            scene.surfaces.append(surface)
            id_map[f["id"]] = surface.id
        for s in data["strokes"]:
            stroke = StrokeObject(s["name"], axes=arr(s["axes"]).T, parent=s.get("parent"))
            stroke.hidden = bool(s.get("hidden", False))
            stroke.flipbook = id_map.get(s.get("flipbook")) if s.get("flipbook") is not None else None
            id_map[s["id"]] = stroke.id
            stroke.curves = [_curve_from(c, id_map) for c in s["curves"]]
            for c, cd in zip(stroke.curves, s["curves"]):
                id_map[("curve", cd["id"])] = c.id  # lines get new ids: rewrite loft sources
            scene.strokes.append(stroke)
            if s["id"] == data.get("active_stroke"):
                scene.active_stroke = stroke
        for old, pieces in data.get("splits", []):  # old id -> a new one that leads to the pieces via split_map
            ids = [id_map.get(("curve", p)) for p in pieces]
            if ids and all(i is not None for i in ids) and ("curve", old) not in id_map:
                virtual = _new_id()
                scene.split_map[virtual] = ids
                id_map[("curve", old)] = virtual
        for surface in lofts:
            surface.sources = [
                None if src is None else
                ("surface", id_map.get(src[1])) if src[0] == "surface" else
                ("lines", [id_map.get(("curve", cid)) for cid in src[1]])
                for src in surface.sources]
            surface.rails = [(id_map.get(("curve", cid)), col) for cid, col in surface.rails]
            if isinstance(surface, PatchSurface):
                surface.patch_lines = [id_map.get(("curve", cid), 0) for cid in surface.patch_lines]
                if surface.net:
                    surface.net = dict(surface.net, **{key: [[[id_map.get(("curve", cid), 0) for cid in ids], at]
                                                             for ids, at in surface.net.get(key, [])]
                                                       for key in ("u", "v")})
            surface.attached = {id_map.get(("curve", cid)): a for cid, a in surface.attached.items()
                                if id_map.get(("curve", cid)) is not None}
            for cid, a in surface.attached.items():
                curve = scene.curve_by_id(cid)
                if curve is not None:
                    a["last"] = curve.points.tobytes()
                    curve.plane_id = surface.id
        if data.get("sun"):
            scene.sun = tuple(float(x) for x in data["sun"])[:2]
        scene.light_bands = int(data.get("light_bands", 0) or 0)
        scene.note_scale = max(0.01, float(data.get("note_scale", 1.0) or 1.0))
        scene.brief_task = data.get("brief_task") or {}
        scene.brief_clarified = data.get("brief_clarified") or {}
        gn = data.get("global_note") or {}
        scene.global_note.title = gn.get("title", tr("Projekt"))
        absolute = lambda p: os.path.normpath(os.path.join(base_dir, p))
        scene.global_note.board = _board_videos(gn.get("board") or {}, absolute)
        for nd in data.get("notes", []):
            scene.notes.append(_note_from(nd, id_map, absolute))
        if data.get("timing"):
            scene._load_timing(data["timing"], id_map, absolute)
        scene._upgrade_extrude_surfaces(id_map)
        unique_hero_names(scene)
        for surface in fills:  # rewrite line ids; older files: link the outline by position
            if surface.links:
                remap = lambda cid: ([id_map.get(("curve", c)) for c in cid] if isinstance(cid, list)
                                     else id_map.get(("curve", cid)) if cid is not None else None)
                surface.links = [(remap(cid), fl) for cid, fl in surface.links]
                surface.attached = {id_map.get(("curve", cid)): a for cid, a in surface.attached.items()
                                    if id_map.get(("curve", cid)) is not None}
                for cid, a in surface.attached.items():
                    curve = scene.curve_by_id(cid)
                    if curve is not None:
                        a["last"] = curve.points.tobytes()
                        curve.plane_id = surface.id
                pieces = scene._fill_pieces(surface)
                surface._fill_key = None if pieces is None else b"".join(np.ascontiguousarray(q).tobytes() for q in pieces)
            if not surface.links or surface.rest is None or scene._fill_pieces(surface) is None:
                surface.links, surface.rest = [], None
                scene._relink_fill(surface, min_cover=0.6)
        if data.get("version", 1) < 5 and scene.surfaces:
            scene._lines_from_old_surfaces()
        for o in scene.tree_objects():  # hierarchy: old ids -> new (unknown: top level, _clean_parents)
            if o.parent is not None:
                o.parent = id_map.get(o.parent)
        for null in scene.nulls:  # lines in the settings (array along a line)
            for key in ("path", "up"):
                if null.params.get(key) is not None:
                    null.params[key] = id_map.get(("curve", null.params[key]))
        scene._clean_parents()
        active = scene.plane_by_id(id_map.get(data.get("active_plane")))
        if active is not None:
            scene.activate(active)
        scene.last_custom = scene.last_custom or (scene.custom[-1] if scene.custom else None)
        mesh_id = id_map.get(data.get("active_mesh"))
        scene.active_mesh = next((m for m in scene.meshes + scene.surfaces if m.id == mesh_id), None)
        # older file without point IDs? (check before the first sync – the surface sync already assigns them)
        old = bool(network.net_curves(scene)) and not any(c.pids for c in network.net_curves(scene))
        scene.sync_lofts()  # starting state for loft surfaces and lines drawn on them
        # curve network: renumber stored point IDs; older files get them here (takeover)
        net_curves = network.net_curves(scene)
        network.reassign_loaded(net_curves)
        stats = network.sync(scene)
        scene.net_report = dict(stats, migrated=old)
        scene._upgrade_open_lofts()
        scene.lift_mesh_paint()  # older scenes: lift painting on meshes once
        return scene

    def lift_mesh_paint(self, curves=None, toward=None):
        """Lift painting on imported meshes a little (PAINT_LIFT × stroke width) along the mesh normal
        and lay it flat on the surface – otherwise it sits half in the mesh and is hardly visible. Once per stroke
        (curve.lift). toward: viewing direction when painting (side on which it was painted); otherwise the remembered normal."""
        meshes = {m.id: m for m in self.meshes if not m.missing and m.geo.triangle_count}
        for c in self.curves if curves is None else curves:
            if not c.paint or c.lift or c.plane_id not in meshes or len(c.points) == 0:
                continue
            n = meshes[c.plane_id].normals_at(c.points)
            ref = np.broadcast_to(np.asarray(toward, float), n.shape) if toward is not None else \
                c.normals if c.normals is not None and len(c.normals) == len(n) else None
            if ref is not None:
                n = n * np.where(np.einsum("ij,ij->i", n, ref) < 0, -1.0, 1.0)[:, None]
            d = PAINT_LIFT * float(np.mean(c.widths)) if c.widths is not None and len(c.widths) else 1e-3
            c.points = c.points + n * d
            c.normals, c.lift = n, d

    def _lines_from_old_surfaces(self):
        """Files up to version 4: surfaces owned their border lines or extrusion edges. Now
        surfaces are pure drawing surfaces – their edges become normal lines (own stroke object)."""
        stroke = StrokeObject(self._unique_name(tr("Linien aus Flächen"), self.strokes))
        for surface in self.surfaces:
            width = float(surface.curves[0].widths.mean()) if surface.curves else DEFAULT_WIDTH
            stroke.curves += [Curve(c.points.copy(), c.plane_id, c.widths.copy()) for c in surface.curves]
            if isinstance(surface, ExtrudeSurface):
                stroke.curves += [Curve(np.array(pts, float), None, np.full(len(pts), width))
                                  for _, pts in surface.edge_list(profile=False)]
        self.strokes.append(stroke)

    def _video_items(self):
        return [it for n in self.notes + [self.global_note] + self.timing.all_notes() for it in n.board.get("items", [])
                if it.get("video")]

    def _load_timing(self, td, id_map, absolute):
        """Shots from the scene file: range, cameras, shot notes (like normal notes). Note ids are new on loading
        – the cameras then point to the new ones. Files of the first prototype are converted (text and camera images
        onto the board, the note to its first plane)."""
        tm = self.timing = timing.from_dict(td)
        remap = {}
        gd = td.get("global_note") or {}
        if "board" in gd:
            tm.global_note.title = gd.get("title", "Global")
            tm.global_note.board = _board_videos(gd.get("board") or {}, absolute)
            remap[gd.get("id")] = tm.global_note.id
        for nd in td.get("notes", []):
            if "pos" not in nd:
                continue  # first prototype (below)
            note = _note_from(nd, id_map, absolute)
            tm.notes.append(note)
            remap[nd.get("id")] = note.id
        for cam in tm.cams:
            cam.note = remap.get(cam.note, tm.global_note.id)
        for d, glob, cams in getattr(tm, "legacy", []):
            note = tm.global_note if glob else None
            if note is None:
                layers = [self.plane_by_id(id_map.get(i, i)) for i in d.get("layers", [])]
                layers = [p for p in layers if p is not None and p.flipbook]
                where = layers[0].origin + layers[0].axis_v * layers[0].size_v * 0.6 if layers else np.zeros(3)
                anchors = ([NoteAnchor(layers[0].origin.copy(), members=[NoteMember(p.id) for p in layers[1:]])]
                           if layers else [])
                note = Note(d.get("title", "Shot"), where, anchors=anchors)
                tm.notes.append(note)
            items = list(note.board.get("items", []))
            text = (d.get("text") or "").strip()
            if d.get("start") is not None:  # range no longer exists: keep as text
                text = (f"Frames {d['start']}–{d.get('end') or tm.end}\n" + text).strip()
            y = 0.0
            if text:
                items.append({"kind": "text", "x": 0.0, "y": 0.0, "w": 340.0, "text": text})
                y = 160.0
            for cam, images in cams:
                cam.note = note.id
                tm.cams.append(cam)
                for k, pose in enumerate(cam.poses):
                    file = images.get(pose.frame) or next(iter(images.values()), "")
                    items.append({"kind": "image", "x": round(k * 420.0, 1), "y": y, "scale": 0.25, "file": file,
                                  "view": timing.pose_view(pose), "shot": {"cam": cam.id, "pose": pose.id}})
                y += 300.0
            note.board = {"items": items} if items else {}
        tm.legacy = []
        timing.sync_boards(tm)

    def adopt_videos(self, folder):
        """On saving: copy videos of the boards to folder/videos (travels with the scene). video_moved remembers
        where – the elements of an open board point to the new path on the next takeover."""
        for it in self._video_items():
            old = it["video"]["path"]
            if os.path.isfile(old):
                try:
                    it["video"] = {**it["video"], "path": video.adopt(old, folder)}
                    self.video_moved[old] = it["video"]["path"]
                except OSError as exc:  # e.g. disk full: save the scene anyway, the video stays linked
                    self.load_warnings.append(f"{os.path.basename(old)}: {exc}")

    def save(self, path, extra=None):
        self.adopt_videos(os.path.dirname(os.path.abspath(path)))
        data = self.to_dict(os.path.dirname(os.path.abspath(path)))  # images of the boards go into the file too
        if extra:
            data.update(extra)
        text = json.dumps(data, indent=1)  # in one piece: json.dump writes in small chunks and needs a multiple
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.dirty = False

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        scene = cls.from_dict(data, os.path.dirname(os.path.abspath(path)))
        scene.asset_dir = cls.assets_dir_for(os.path.abspath(path))
        scene.assign_owners()  # phase 4: older scenes – every surface gets its line object (and lives in its null)
        return scene, data

