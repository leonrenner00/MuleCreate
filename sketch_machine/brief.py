"""Sketch Brief: export for AI-assisted further processing (e.g. rebuilding in Houdini).

A folder with
  brief.md     - concise, readable description: dimensions, objects, lines, surfaces (material, color), notes
  brief.json   - the same information structured (for scripts, e.g. Python SOPs)
  views/       - rendered views (perspective, top, front, right) without drawing aids, labeled with the
                 top-level groups; per note note_N#.png: where it points
  scene.usd    - the scene as USD (usd_export, binary): hierarchy, surfaces as polymeshes, lines as curves,
                 imported meshes referenced, symmetry/array computed out, notes as points
  notes/       - images from the note boards
Identifiers: L# lines, S# stroke objects, F# drawing surfaces, N# notes, M# imported meshes, H# nulls - the same
everywhere. Only visible things go into the brief.
"""
import contextlib
import datetime
import json
import math
import weakref
import os
import re

import numpy as np

from . import components, i18n, network, paths
from .camera import view_for_back
from .scene import LoftSurface, NullObject, PatchSurface, SurfaceObject

FORMAT = "sketch-brief"
VERSION = 1
LINE_LIST_MAX = 400  # more lines: only summarized in the brief (all in JSON/USD)
VIEWS = (("persp", (0.62, 0.48, 0.62), False), ("top", (0, 1, 0), True), ("front", (0, 0, 1), True),
         ("right", (1, 0, 0), True), ("back", (0, 0, -1), True))
# Hero cameras without any of the artist's own: three default angles (view direction "back" as in VIEWS);
# named neutrally - the tool does not know where the front of the model is
DEFAULT_HEROES = (("hero_1_high", (0.62, 0.48, 0.62)), ("hero_2_high_opposite", (-0.62, 0.40, -0.62)),
                  ("hero_3_low", (-0.72, -0.10, 0.68)))
STRAIGHT_TOL = 0.01  # a line counts as straight if it deviates from the chord by at most this much (fraction of the length)
AXIS_COS = 0.9998  # direction counts as axis-parallel (up to a good 1°; 0.995 already called lines 5.7° off parallel)
TINY = 1e-3  # shorter than this (fraction of the scene size, at least 1 mm): tiny line, not in the table
# Workflow bible: cross-project, in the repo; the brief names the path on this machine, the protocol is in the file
BIBLE = paths.bible_path()


def _r(x, nd=3):
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


def _length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def _axis_name(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    v = np.asarray(v) / n
    for name, axis in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
        if abs(v @ np.array(axis, float)) > AXIS_COS:
            return name
    return None


def _plane_fit(pts):
    """(normal, distance, max. deviation) of a plane through the points."""
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    n = np.linalg.eigh((pts - c).T @ (pts - c))[1][:, 0]  # smallest scatter (3x3 instead of SVD over all points)
    if n @ np.array([0.3, 1.0, 0.2]) < 0:
        n = -n
    return n, float(n @ c), float(np.abs((pts - c) @ n).max())


def _bbox(pts):
    pts = np.asarray(pts, float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return {"min": _r(lo), "max": _r(hi), "size": _r(hi - lo)}


# --- Analysis ------------------------------------------------------------------------------------------

def _visible(scene):
    """What goes into the brief: everything that is not hidden (eye in the outliner, also via a null above;
    individually hidden lines; painting on hidden surfaces/meshes)."""
    hid = scene.is_hidden
    # flipbook drawings (timing prototype) do not belong in the brief and export yet: otherwise all frames would lie on top of each other
    strokes = [st for st in scene.strokes if not hid(st) and st.flipbook is None]
    surfaces = [f for f in scene.surfaces if not hid(f)]
    meshes = [m for m in scene.meshes if not hid(m)]
    nulls = [n for n in scene.nulls if not hid(n)]
    hidden_hosts = {o.id for o in scene.surfaces + scene.meshes if hid(o)}
    curves = [(st, c) for st in strokes for c in st.curves
              if len(c.points) > 1 and not c.hidden and not (c.paint and c.plane_id in hidden_hosts) and c.frame is None]
    return {"strokes": strokes, "surfaces": surfaces, "meshes": meshes, "nulls": nulls, "curves": curves}


def _mesh_corners(m):
    lo, hi = m.world_bounds()
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def index(scene):
    """Only visible things and the identifiers (L#, S#, F#, M#, H#, N#) - fast; analyse and the USD export (usd_export)
    assign them the same way. -> (vis, maps)"""
    vis = _visible(scene)
    maps = {"lid": {c.id: f"L{k + 1}" for k, (_, c) in enumerate(vis["curves"])},
            "sid": {st.id: f"S{k + 1}" for k, st in enumerate(vis["strokes"])},
            "fid": {f.id: f"F{k + 1}" for k, f in enumerate(vis["surfaces"])},
            "mid": {m.id: f"M{k + 1}" for k, m in enumerate(vis["meshes"])},
            "hid": {n.id: f"H{k + 1}" for k, n in enumerate(vis["nulls"])},
            "nid": {n.id: f"N{k + 1}" for k, n in enumerate(scene.notes)}}
    return vis, maps


def analyse(scene, locate=True, focus=None):
    """Scene -> dict with all information of the brief (identifiers L#, S#, F#, N#, M#, H#) - only visible things. locate: assign fixed
    note anchors to the objects they hit; focus: ids of the objects at stake (selection on export; a
    null with everything below it) - the rest is context."""
    vis, maps = index(scene)
    curves = vis["curves"]
    lid, sid, fid, mid, hid = (maps[k] for k in ("lid", "sid", "fid", "mid", "hid"))
    # overall dimensions: everything visible, also imported meshes (helper surfaces are only drawing aids)
    allpts = [c.points for _, c in curves] + [m for f in vis["surfaces"] if not getattr(f, "aid", False)
                                              for m in [_surface_points(f)] if m is not None]
    allpts += [_mesh_corners(m) for m in vis["meshes"] if not m.missing]
    data = {"format": FORMAT, "version": VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "units": "1 unit = 1 m", "up_axis": "Y",
            "ids": "L# lines, S# stroke objects, F# surfaces, N# notes, M# meshes, H# nulls/generators"}
    data["bounds"] = _bbox(np.vstack(allpts)) if allpts else None
    size = float(np.linalg.norm(np.ptp(np.vstack(allpts), axis=0))) if allpts else 1.0
    tol = max(1e-4 * size, 1e-6)
    names = {**{st.id: st.name for st in vis["strokes"]}, **{f.id: f.name for f in vis["surfaces"]},
             **{m.id: m.name for m in vis["meshes"]}, **{n.id: n.name for n in vis["nulls"]}}
    ids = {**sid, **fid, **mid, **hid}

    def path(obj):
        """Position in the hierarchy, from the top: "H5 Glider > H13 W2_L_section_002" (empty: top level)."""
        return " › ".join(f"{ids[n.id]} {n.name}" for n in reversed(scene.ancestors(obj)) if n.id in ids)

    # lines
    lines = []
    by_pid = {}  # point id -> lines through that point (connections only go via shared points)
    for _, c in curves:
        for pid in (c.pids or {}).values():
            by_pid.setdefault(pid, set()).add(lid[c.id])
    tiny = max(TINY * size, 1e-3)
    for st, c in curves:
        pts = c.points
        chord = pts[-1] - pts[0]
        length = _length(pts)
        closed = float(np.linalg.norm(chord)) < tol and len(pts) > 3
        dev = 0.0
        if not closed and np.linalg.norm(chord) > 1e-12:
            u = chord / np.linalg.norm(chord)
            rel = pts - pts[0]
            dev = float(np.linalg.norm(rel - np.outer(rel @ u, u), axis=1).max())
        straight = not closed and dev <= STRAIGHT_TOL * max(length, 1e-9)
        kn = components.knots(c)
        entry = {"id": lid[c.id], "object": sid[st.id], "start": _r(pts[0]), "end": _r(pts[-1]),
                 "length": round(length, 3), "shape": "closed" if closed else "straight" if straight else "curved",
                 "points": len(pts), "corners": len(kn) - 2}
        if c.style == "brush" and c.color is not None:  # brush stroke: color
            entry["style"] = "brush"
            entry["color"] = "#" + "".join(f"{int(round(x * 255)):02x}" for x in c.color)
        if c.paint:  # paint stroke: color on the surface, no construction line (order: higher lies on top)
            entry["paint"] = True
            entry["paint_order"] = int(c.seq)
            if c.plane_id in fid or c.plane_id in mid:  # painted on a surface or an imported mesh
                entry["on"] = ids[c.plane_id]
        axis = _axis_name(chord) if straight else None
        if axis:
            entry["axis"] = axis
        if not straight and len(pts) > 2:
            n, off, err = _plane_fit(pts)
            if err <= 0.01 * max(length, 1e-9):
                entry["plane_normal"] = _r(n, 2)
        joins = {x for pid in (c.pids or {}).values() for x in by_pid.get(pid, ())} - {lid[c.id]}
        if joins:
            entry["joins"] = sorted(joins, key=lambda x: int(x[1:]))
        if length < tiny:
            entry["tiny"] = True  # jitter/remainder, hardly meant
        lines.append(entry)
    data["lines"] = lines

    # connected groups (via shared line ends and ends on other lines)
    data["groups"] = _groups(curves, lid, tol)

    # stroke objects
    data["objects"] = [{"id": sid[st.id], "name": st.name, "lines": [lid[c.id] for c in st.curves if c.id in lid],
                        "bounds": _bbox(np.vstack([c.points for c in st.curves])) if st.curves else None}
                       for st in vis["strokes"]]

    # surfaces
    surfs = []
    for f in vis["surfaces"]:
        pts = _surface_points(f)
        entry = {"id": fid[f.id], "name": f.name}
        if isinstance(f, PatchSurface):  # curve surface: from border curves, possibly with net lines in it
            entry["kind"] = "curve_surface"
            entry["boundary_lines"] = [lid[i] for i in f.patch_lines if i in lid]
            net = f.net or {}
            entry["net_lines"] = [lid[i] for key in ("u", "v") for chain, _ in net.get(key, []) for i in chain
                                  if i in lid]
        elif isinstance(f, LoftSurface):
            entry["kind"] = "loft"
            entry["sections"] = len(f.curves)
            entry["ring"] = bool(f.wrap)
            entry["closed_sections"] = bool(f.closed)
            src = []
            for s in f.sources:
                if s is None:
                    src.append(None)
                elif s[0] == "surface":
                    src.append(fid.get(s[1]))
                else:
                    src.append([lid.get(i) for i in s[1]])
            entry["section_sources"] = src
        elif type(f) is SurfaceObject:
            entry["kind"] = "fill"
            entry["outline_lines"] = [None if cid is None else _one([lid.get(i) for i in scene.link_ids(cid)])
                                      for cid, _ in f.links]
        else:
            entry["kind"] = f.kind
        if pts is not None and len(pts) > 2:
            entry["bounds"] = _bbox(pts)
            n, off, err = _plane_fit(pts)
            if err <= 0.005 * max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1e-9):
                entry["planar"] = True
                entry["plane_normal"] = _r(n, 3)
                entry["plane_offset"] = round(off, 3)
                axis = _axis_name(n)
                if axis:
                    entry["plane"] = {"X": "YZ", "Y": "XZ", "Z": "XY"}[axis] + f" at {axis.lower()}={off * (1 if n @ np.eye(3)['XYZ'.index(axis)] > 0 else -1):.3f}"
            else:
                entry["planar"] = False
            mesh = f.fill_mesh()
            if mesh is not None:
                v, t = mesh[0], mesh[1]
                tri = v[t]
                entry["area"] = round(float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2), 3)
        entry["material"] = getattr(f, "material", "papier")
        if getattr(f, "aid", False):
            entry["helper"] = True  # helper surface: only for drawing on, not part of the result
        if getattr(f, "tint", None) is not None:
            entry["color"] = [round(float(x), 3) for x in f.tint]
        surfs.append(entry)
    data["surfaces"] = surfs

    # meshes
    data["meshes"] = [{"id": mid[m.id], "name": m.name, "file": m.path, "missing": m.missing,
                       "triangles": 0 if m.missing else int(m.geo.triangle_count),
                       **({"prim": m.prim} if m.prim else {}),
                       "bounds": _bbox(m.world_bounds()) if not m.missing else None} for m in vis["meshes"]]

    # hierarchy: nulls and generators (symmetry, array) - their instances are not in lines/surfaces
    for entry, obj in [(e, o) for e, o in zip(data["objects"], vis["strokes"])] + \
            [(e, o) for e, o in zip(surfs, vis["surfaces"])] + [(e, o) for e, o in zip(data["meshes"], vis["meshes"])]:
        if obj.parent in hid:
            entry["parent"] = hid[obj.parent]
    for entry, obj in [(e, o) for e, o in zip(surfs, vis["surfaces"])] + \
            [(e, o) for e, o in zip(data["meshes"], vis["meshes"])]:
        level, _source = scene.binding_of(obj)
        if level in TASK_LEVELS:
            entry["binding"] = level  # bindingness from the outliner (own or from the null above)
    data["hierarchy"] = [_null_entry(scene, n, ids, lid) for n in vis["nulls"]]
    # where the bindingness is set in the outliner - exceptions to the task (nulls: everything below)
    data["bindings"] = [{"id": ids[o.id], "name": o.name, "level": o.binding, "group": isinstance(o, NullObject)}
                        for o in vis["nulls"] + vis["meshes"] + vis["surfaces"]
                        if getattr(o, "binding", None) in TASK_LEVELS and o.id in ids]

    # own helper planes
    data["planes"] = [{"name": p.name, "origin": _r(p.origin), "normal": _r(p.normal, 3), "size": [p.size_u, p.size_v]}
                      for p in scene.custom]

    # notes
    from . import generators
    nid = maps["nid"]
    reach = max(0.005 * size, 1e-3)  # a fixed anchor must lie this close to an object to hit it
    kinds = {**{i: "stroke object" for i in sid}, **{i: "surface" for i in fid}, **{i: "mesh" for i in mid}}
    copied = {oid for oid, seqs in generators.copies_by_object(scene).items()
              if any(not generators.is_original(ops) for _, ops in seqs)}

    def member(m):
        """Arm of the note octopus -> entry; "missing": the part is deleted; None: hidden (never in the brief)."""
        obj = scene.object_by_id(m.obj_id)
        if obj is None or (m.curve_id is not None and scene.curve_by_id(m.curve_id) is None):
            return {"missing": True}
        if isinstance(obj, NullObject):  # group: everything visible in it is meant too
            parts = [ids[c.id] for c in scene.descendants(obj) if c.id in kinds]
            if obj.id not in ids or not parts:
                return None
            e = {"id": ids[obj.id], "name": names[obj.id], "kind": "group", "path": path(obj), "parts": parts}
            if any(c.id in copied for c in scene.descendants(obj)):
                e["with_copies"] = True
            return e
        if obj.id not in kinds or (m.curve_id is not None and m.curve_id not in lid):
            return None
        e = {"id": ids[obj.id], "name": names[obj.id], "kind": kinds[obj.id], "path": path(obj)}
        if m.curve_id is not None:
            e["line"] = lid[m.curve_id]
        if obj.id in copied:
            e["with_copies"] = True  # symmetry/array: the copies are meant too
        return e

    out = []
    for n in scene.notes:
        anchors = []
        for a in n.anchors:
            entry = {"point": _r(a.point)}
            if a.curve_id is not None and a.curve_id in lid:
                entry["line"] = lid[a.curve_id]
                owner = next((st for st, c in curves if c.id == a.curve_id), None)
                if owner is not None:
                    entry["target"] = {"id": sid[owner.id], "name": owner.name, "kind": "stroke object",
                                       "path": path(owner)}
                entry["at"] = round(float(a.t), 3)
                entry["lost"] = bool(a.lost)
            else:
                hit = _locate(vis, a.point, reach) if locate else None
                if hit is not None:
                    kind, obj = hit
                    entry["target"] = {"id": ids[obj.id], "name": names[obj.id], "kind": kind, "path": path(obj)}
            found = [e for e in (member(m) for m in a.members) if e is not None]
            if any(e.get("missing") for e in found):
                entry["members_missing"] = sum(1 for e in found if e.get("missing"))
            found = [e for e in found if not e.get("missing")]
            if found:
                entry["members"] = found  # note octopus: further parts the anchor also means (by hand, possibly incomplete)
            anchors.append(entry)
        out.append({"id": nid[n.id], "uid": n.uid, "title": n.title or "(untitled)", "position": _r(n.pos),
                    "size": n.size,
                    "anchors": anchors, **_board_info(n.board, f"notes/board_{nid[n.id]}.png", scene),
                    **({"review_draft": True} if n.review else {})})
    data["notes"] = out
    g = scene.global_note
    data["project_note"] = {"title": g.title, **_board_info(g.board, "notes/board_project.png", scene)}
    for n in out + [data["project_note"]]:  # parts of the captions: object ids -> identifiers, visible only
        cp = {f: [ids[o] for o in objs if o in ids] for f, objs in n.pop("caption_parts", {}).items()}
        if any(cp.values()):
            n["caption_parts"] = {f: v for f, v in cp.items() if v}
    # the other way round: part -> notes pointing at it (anchor or arm of the octopus) - only reliable with the octopus
    part_notes = {}
    named = {ids[k]: names[k] for k in ids if k in names}
    for n in out:
        for k, a in enumerate(n["anchors"], 1):
            for via, t in ([("anchor", a["target"])] if a.get("target") else []) + \
                    [("also meant", e) for m in a.get("members", [])  # group: also every part in it
                     for e in [m] + [{"id": i, "name": named.get(i, i)} for i in m.get("parts", [])]]:
                e = part_notes.setdefault(t["id"], {"part": t["id"], "name": t["name"], "notes": []})
                ref = {"note": n["id"], "anchor": k, "via": via}
                if ref not in e["notes"]:
                    e["notes"].append(ref)
    for n in out + [data["project_note"]]:  # images whose caption names the part
        for f, ps in n.get("caption_parts", {}).items():
            for i in ps:
                e = part_notes.setdefault(i, {"part": i, "name": named.get(i, i), "notes": []})
                if f not in e.setdefault("pictures", []):
                    e["pictures"].append(f)
    data["part_notes"] = sorted(part_notes.values(), key=lambda e: (e["part"][0], int(e["part"][1:])))
    data["_maps"] = {"lid": lid, "sid": sid, "fid": fid, "nid": nid, "mid": mid, "hid": hid}
    data["_vis"] = vis
    data["_size"] = size
    task = task_clean(getattr(scene, "brief_task", None))
    if task:
        data["task"] = task
    data["_focus"] = set()
    if focus:
        picked = [o for o in (scene.object_by_id(i) for i in focus) if o is not None and o.id in ids]
        want = {o.id for o in picked}
        for o in picked:
            if o.id in hid:
                want |= {d.id for d in scene.descendants(o) if d.id in ids}
        objs = [o for o in vis["strokes"] + vis["surfaces"] + vis["meshes"] if o.id in want]
        pts = _frame_points(objs)
        if objs and pts:
            data["_focus"] = want
            data["focus"] = {"selected": [{"id": ids[o.id], "name": o.name} for o in picked],
                             "ids": sorted((ids[i] for i in want), key=lambda x: (x[0], int(x[1:]))),
                             "bounds": _bbox(np.vstack(pts))}
    return data


def image_captions(scene):
    """Image captions of the scene {file: text} - apply per image file, even if the image
    is on several boards (the first one found)."""
    out = {}
    for note in scene.notes + [scene.global_note]:
        for it in note.board.get("items", []):
            if it.get("kind") == "image" and it.get("caption"):
                out.setdefault(it["file"], it["caption"])
    return out


def image_caption_parts(scene):
    """Parts for the image captions {file: [object ids]} - from the same image entry as image_captions."""
    out = {}
    for note in scene.notes + [scene.global_note]:
        for it in note.board.get("items", []):
            if it.get("kind") == "image" and it.get("caption"):
                out.setdefault(it["file"], list(it.get("caption_parts", [])))
    return out


def video_names(scene):
    """{video file: name in the brief folder videos/} - identical file names from different folders get a number."""
    out = {}
    for it in scene._video_items():
        path = it["video"]["path"]
        if path not in out:
            stem, ext = os.path.splitext(os.path.basename(path))
            name, k = stem + ext, 2
            while name in out.values():
                name, k = f"{stem}_{k}{ext}", k + 1
            out[path] = name
    return out


def _board_videos(items, scene, caps):
    """Videos of a board: file in the brief, preview (the image on the board), overview image, duration, caption."""
    if scene is None:
        return []
    names, out = video_names(scene), []
    for it in items:
        if it.get("video"):
            name = names[it["video"]["path"]]
            entry = {"file": f"videos/{name}", "poster": f"notes/{it['file']}",
                     "overview": f"videos/{os.path.splitext(name)[0]}_overview.jpg",
                     "seconds": round(it["video"].get("duration", 0) / 1000, 1)}
            if it["file"] in caps:
                entry["caption"] = caps[it["file"]]
            if not os.path.isfile(it["video"]["path"]):
                entry["missing"] = True
            out.append(entry)
    return out


def _board_info(board, board_png, scene=None):
    """Content of a note board: texts, images, screenshots from the viewport (with camera), number of painted strokes,
    images missing from the scene file, image captions, reference images without a caption (screenshots do
    not count - they show the sketch itself) and the image of the whole board (export writes it if the board has
    content)."""
    items = board.get("items", [])
    caps = image_captions(scene) if scene is not None else {}
    cparts = image_caption_parts(scene) if scene is not None else {}
    pics = [it for it in items if it.get("kind") == "image"]
    missing = [it["file"] for it in pics if scene is not None and scene.asset_bytes(it["file"]) is None]
    shots = []
    for it in items:
        view = it.get("view") if it.get("kind") == "image" else None
        if view:
            rot = np.array(view["rot"], float)
            target = np.array(view["target"], float)
            shots.append({"image": f"notes/{it['file']}", "camera": {
                "eye": _r(target + float(view["dist"]) * rot[:, 2]), "target": _r(target), "look_dir": _r(-rot[:, 2]),
                "projection": "orthographic" if view.get("ortho") else "perspective"}})
    return {"text": [it["text"] for it in items if it.get("kind") == "text"],
            "images": [f"notes/{it['file']}" for it in items if it.get("kind") == "image"],
            "screenshots": shots, "drawings": sum(1 for it in items if it.get("kind") == "stroke"),
            "missing": [f"notes/{f}" for f in missing],
            "captions": {f"notes/{it['file']}": caps[it["file"]] for it in pics if it["file"] in caps},
            "caption_parts": {f"notes/{it['file']}": cparts[it["file"]] for it in pics if cparts.get(it["file"])},
            "uncaptioned": list(dict.fromkeys(f"notes/{it['file']}" for it in pics if not it.get("view")
                                              and it["file"] not in caps and it["file"] not in missing)),
            "videos": _board_videos(items, scene, caps),
            "board": board_png if items else None}


def _null_entry(scene, n, ids, lid):
    """Null or generator for the brief: position, children and - for generators - what they produce, in Houdini terms.
    Array: additionally the finished matrices per copy (4x4, column vectors, rows here), exact even with randomness."""
    from . import arrays, generators
    e = {"id": ids[n.id], "name": n.name, "kind": n.kind, "origin": _r(n.origin),
         "axes": {"x": _r(n.axes[:, 0]), "y": _r(n.axes[:, 1]), "z": _r(n.axes[:, 2])},
         "parent": ids.get(n.parent), "children": [ids[c.id] for c in scene.children(n) if c.id in ids]}
    if n.kind != "null":
        e["enabled"] = generators.is_on(n)
    if n.kind != "null" and not generators.is_on(n):
        e["note"] = "generator switched off: produces nothing, its children are used as they are"
    elif n.kind == "symmetry":
        nrm, o = generators.mirror_plane(n)
        e["mirror"] = {"plane": n.params.get("plane", "zy").upper(), "origin": _r(o), "normal_kept_side": _r(nrm),
                       "clip": "everything under it is cut at the plane; only the side the normal points to is kept",
                       "houdini": "Clip SOP (keep the normal side) -> Mirror SOP (origin, direction = normal) -> "
                                  "Fuse the points on the plane"}
    elif n.kind == "array":
        p = n.params
        mats = arrays.matrices(scene, n)[1:]
        arr = {"mode": p.get("mode", "radial"), "count": int(p.get("count", 1)), "copies_shown": len(mats),
               "original": "stays where it is (the copies are added)"}
        if arr["mode"] == "radial":
            arr.update(angle_deg=float(p.get("angle", 360.0)), axis=_r(n.axes[:, 1]), pivot=_r(n.origin),
                       houdini="Copy and Transform (rotate about the axis through the pivot) or Copy to Points on a circle")
        elif arr["mode"] == "linear":
            arr.update(step=_r(n.axes @ np.asarray(p.get("offset", [1, 0, 0]), float)),
                       houdini="Copy and Transform (translate by step per copy)")
        else:
            arr.update(path=lid.get(p.get("path")), up_line=lid.get(p.get("up")),
                       houdini="Resample the path line (count points, by arc length), orient: x = tangent, y = up "
                               "(towards the up line, else carried along), then Copy to Points")
        if p.get("spin"):
            arr["spin_deg_per_copy"] = float(p["spin"])
        if p.get("rand_rot") or p.get("rand_off"):
            arr["random"] = {"rotate": float(p.get("rand_rot", 0.0)), "rotate_axes": list(p.get("rot_axes", [])),
                             "offset": float(p.get("rand_off", 0.0)), "offset_axes": list(p.get("off_axes", [])),
                             "seed": int(p.get("seed", 1))}
        arr["copy_matrices"] = [[_r(row, 5) for row in m] for m in mats]
        e["array"] = arr
    return e


def _one(ids):
    return ids[0] if len(ids) == 1 else ids


def _surface_points(f):
    mesh = f.fill_mesh()
    if mesh is not None and len(mesh[0]):
        return mesh[0]
    loop = f.loop_points() if hasattr(f, "loop_points") else None
    return loop if loop is not None and len(loop) else None


def _seg_dist(p, a, b):
    """Distances from p to the segments a[i]-b[i]."""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30), 0, 1)
    return np.linalg.norm(a + ab * t[:, None] - p, axis=1)


def _tri_dist(p, v, tris):
    """Smallest distance from p to the triangles (points v, indices tris)."""
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.maximum(np.linalg.norm(n, axis=1), 1e-30)
    h = np.einsum("ij,ij->i", p - a, n) / nn
    q = p - (h / nn)[:, None] * n  # foot point in the plane of the triangle
    inside = ((np.einsum("ij,ij->i", np.cross(b - a, q - a), n) >= 0) &
              (np.einsum("ij,ij->i", np.cross(c - b, q - b), n) >= 0) &
              (np.einsum("ij,ij->i", np.cross(a - c, q - c), n) >= 0))
    edge = np.minimum(np.minimum(_seg_dist(p, a, b), _seg_dist(p, b, c)), _seg_dist(p, c, a))
    return float(np.where(inside, np.abs(h), edge).min()) if len(tris) else math.inf


def _near_tris(m, p, reach):
    """Triangles of mesh m whose bounding box (in mesh space) comes closer than reach to p - only they can
    undercut the distance (large meshes: otherwise millions of triangles fully computed per anchor)."""
    a = m.xform[:3, :3]
    try:
        local = np.linalg.solve(a, p - m.xform[:3, 3])
        r = reach / max(float(np.linalg.svd(a, compute_uv=False).min()), 1e-12)  # world distance -> mesh space, safe
    except np.linalg.LinAlgError:
        return np.asarray(m.geo.faces)
    if not np.isfinite(r):
        return np.asarray(m.geo.faces)
    faces = np.asarray(m.geo.faces)
    box = _TRI_BOXES.get(m.geo)
    if box is None or box[0] is not faces:  # once per geometry (first note on it): bounding box of all triangles
        tv = m.geo.vertices[faces]  # (n, 3, 3)
        box = (faces, tv.min(axis=1).astype(np.float32), tv.max(axis=1).astype(np.float32))
        _TRI_BOXES[m.geo] = box
    keep = np.all((box[1] <= local + r) & (box[2] >= local - r), axis=1)
    return faces[keep]


_TRI_BOXES = weakref.WeakKeyDictionary()  # geometry -> (faces, min, max) of its triangles (anchor search, _near_tris)


def _locate(vis, point, reach):
    """What lies at the fixed anchor point of a note: (kind, object) of the nearest visible surface, the nearest
    mesh or stroke object (closer than reach), otherwise None."""
    p = np.asarray(point, float)
    best = (reach, None)
    for f in vis["surfaces"]:
        mesh = f.fill_mesh()
        if mesh is not None and len(mesh[1]) and not getattr(f, "aid", False):
            d = _tri_dist(p, np.asarray(mesh[0], float), np.asarray(mesh[1]))
            if d < best[0]:
                best = (d, ("surface", f))
    for m in vis["meshes"]:
        if m.missing or not len(m.geo.faces):
            continue
        lo, hi = m.world_bounds()
        if np.any(p < lo - best[0]) or np.any(p > hi + best[0]):
            continue
        tris = _near_tris(m, p, best[0])
        if not len(tris):
            continue
        used, inv = np.unique(tris, return_inverse=True)  # only the points of these triangles in world coordinates
        v = m.geo.vertices[used].astype(float) @ m.xform[:3, :3].T + m.xform[:3, 3]
        d = _tri_dist(p, v, inv.reshape(tris.shape))
        if d < best[0]:
            best = (d, ("mesh", m))
    for st, c in vis["curves"]:
        if not c.paint:
            d = float(_seg_dist(p, c.points[:-1], c.points[1:]).min())
            if d < best[0]:
                best = (d, ("stroke object", st))
    return best[1]


def _groups(curves, lid, tol):
    """Lines connected via ends (also ends on another line) -> groups."""
    n = len(curves)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ends = [(k, c.points[e]) for k, (_, c) in enumerate(curves) for e in (0, -1)]
    if not ends:
        return []
    pts = np.array([p for _, p in ends])
    owner = np.array([k for k, _ in ends])
    for i in range(len(ends)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.flatnonzero(d < tol):
            a, b = find(owner[i]), find(owner[j])
            if a != b:
                parent[a] = b
    for k, (_, c) in enumerate(curves):  # T-junctions: an end lies on another line
        lo, hi = c.points.min(axis=0) - tol, c.points.max(axis=0) + tol
        near = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1) & (owner != k))  # only ends in their box
        if not len(near):
            continue
        a, ab = c.points[:-1], np.diff(c.points, axis=0)
        ll = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
        rel = pts[near][:, None, :] - a[None]  # (ends, pieces, 3) - all at once
        t = np.clip(np.einsum("esd,sd->es", rel, ab) / ll, 0, 1)
        dist = np.linalg.norm(rel - ab[None] * t[..., None], axis=2).min(axis=1)
        for j in owner[near][dist < tol]:
            x, y = find(j), find(k)
            if x != y:
                parent[x] = y
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    out = []
    for g, members in enumerate(sorted(groups.values(), key=len, reverse=True)):
        pts_g = np.vstack([curves[k][1].points for k in members])
        out.append({"id": f"G{g + 1}", "lines": [lid[curves[k][1].id] for k in members], "bounds": _bbox(pts_g)})
    return out


# --- Views ------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _keep_camera(vp):
    """Restore camera and view after rendering the brief images."""
    cam = vp.cam
    saved = (cam.state(), cam.ortho)
    try:
        yield cam
    finally:
        state, cam.ortho = saved
        cam.set_state(*state)
        vp.update()


def _clean_shot(vp, select=(), lines=(), aspect=None, faint=(), faint_lines=(), ids=False):
    """Image without controls, notes and drawing aids (planes, helper surfaces, axis crosses of nulls); select:
    objects highlighted as selected, lines: lines colored like selected edges; faint/faint_lines: likewise, only
    weaker (arms of the note octopus). ids: (image, color ID per pixel) as clean_image."""
    saved = vp.planes_visible
    vp.planes_visible = False
    try:
        return vp.clean_image(select, lines, aspect=aspect, faint=faint, faint_lines=faint_lines, ids=ids)
    finally:
        vp.planes_visible = saved
        vp.update()


def _object_points(scene, obj, vis_ids):
    """Points of a visible object - for a null those of its whole visible content."""
    if hasattr(obj, "geo"):  # imported mesh
        return None if obj.missing else _mesh_corners(obj)
    if isinstance(obj, SurfaceObject):
        return None if getattr(obj, "aid", False) else _surface_points(obj)
    if hasattr(obj, "curves"):  # stroke object
        pts = [c.points for c in obj.curves if not c.hidden and len(c.points)]
        return np.vstack(pts) if pts else None
    parts = [_object_points(scene, c, vis_ids) for c in scene.children(obj) if c.id in vis_ids]
    parts = [q for q in parts if q is not None and len(q)]
    return np.vstack(parts) if parts else None


def _top_labels(scene, data):
    """[(text, point)] for the top-level objects of the hierarchy (groups with content), top center of the object -
    deeper parts are named by the brief at their group."""
    maps, vis = data["_maps"], data["_vis"]
    ids = {**maps["sid"], **maps["fid"], **maps["mid"], **maps["hid"]}
    out = []
    for obj in vis["nulls"] + vis["strokes"] + vis["surfaces"] + vis["meshes"]:
        if obj.parent in ids:
            continue
        pts = _object_points(scene, obj, ids)
        if pts is None or not len(pts):
            continue
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        name = obj.name if len(obj.name) <= 30 else obj.name[:28] + "…"
        out.append((f"{ids[obj.id]} {name}", np.array([(lo[0] + hi[0]) / 2, hi[1], (lo[2] + hi[2]) / 2])))
    return out


def _free(bg, r, ref, tol=22):
    """Does the rectangle in image bg lie on free background (brightness like ref, samples 6x3)?"""
    if r.left() < 0 or r.top() < 0 or r.right() >= bg.width() or r.bottom() >= bg.height():
        return False
    for i in range(6):
        for j in range(3):
            c = bg.pixelColor(int(r.left() + r.width() * (i + 0.5) / 6), int(r.top() + r.height() * (j + 0.5) / 3))
            if abs(c.lightness() - ref) > tol:
                return False
    return True


def _draw_label(p, c, text, color, dpr, left=False, taken=None, bg=None):
    """Label with leader line; taken: rectangles already occupied - the new one moves on instead of covering them
    (even the AI cannot read overlapping labels); bg: clean image - the label moves up into free
    background instead of covering geometry."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QPen
    w_ = (12 + 7.6 * len(text)) * dpr
    r = (QRectF(c.x() - 8 * dpr - w_, c.y() - 4 * dpr, w_, 17 * dpr) if left
         else QRectF(c.x() + 6 * dpr, c.y() - 20 * dpr, w_, 17 * dpr))
    if bg is not None:
        ref = bg.pixelColor(4, 4).lightness()  # background (paper)
        for step in range(40):  # upward until there is room (otherwise it stays at the object)
            cand = r.translated(0, -14 * dpr * step)
            if _free(bg, cand, ref) and not any(cand.intersects(t) for t in taken or ()):
                r = cand
                break
    if taken is not None:
        for _ in range(12):
            if not any(r.intersects(t) for t in taken):
                break
            r.translate(0, 20 * dpr)
        taken.append(QRectF(r))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 215))
    p.drawRoundedRect(r, 3 * dpr, 3 * dpr)
    p.setPen(QPen(color, 1.2 * dpr))
    p.drawLine(c, QPointF(r.right(), r.top()) if left else QPointF(r.left(), r.bottom()))
    p.setBrush(QColor(255, 255, 255))
    p.drawEllipse(c, 2.5 * dpr, 2.5 * dpr)
    p.drawText(r, Qt.AlignCenter, text)


def _scale_bar(p, img, dpr, px_per_m):
    """Scale bottom right: bar with a round length (about an eighth of the image width) and a figure of 1.8 m."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QPen
    want = img.width() / 8 / px_per_m
    length = next((v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if v >= want * 0.7), 1000)
    w = length * px_per_m
    x1, y = img.width() - 20 * dpr, img.height() - 40 * dpr
    ink = QColor(58, 56, 54)
    p.setPen(QPen(ink, 2 * dpr))
    p.drawLine(QPointF(x1 - w, y), QPointF(x1, y))
    for x in (x1 - w, x1):
        p.drawLine(QPointF(x, y - 5 * dpr), QPointF(x, y + 5 * dpr))
    p.drawText(QRectF(x1 - w, y - 24 * dpr, w, 18 * dpr), Qt.AlignCenter, f"{length:g} m")
    h = 1.8 * px_per_m  # human, 1.8 m
    if h >= 10 * dpr:
        fx, top = x1 - w - 22 * dpr, y - h
        p.setPen(QPen(ink, max(1.0, h / 40)))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(fx, top + h * 0.07), h * 0.07, h * 0.07)
        p.drawLine(QPointF(fx, top + h * 0.14), QPointF(fx, top + h * 0.55))
        p.drawLine(QPointF(fx, top + h * 0.55), QPointF(fx - h * 0.12, y))
        p.drawLine(QPointF(fx, top + h * 0.55), QPointF(fx + h * 0.12, y))
        p.drawLine(QPointF(fx - h * 0.16, top + h * 0.3), QPointF(fx + h * 0.16, top + h * 0.3))
        p.drawText(QRectF(fx - 30 * dpr, top - 18 * dpr, 60 * dpr, 16 * dpr), Qt.AlignCenter, "1.8 m")


def _painter(img):
    from PySide6.QtGui import QFont, QPainter
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    dpr = img.devicePixelRatio() or 1.0
    font = QFont()
    font.setPixelSize(int(13 * dpr))
    font.setBold(True)
    p.setFont(font)
    return p, dpr


def _caption(p, img, dpr, text):
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor
    p.setPen(QColor(58, 56, 54))
    p.drawText(QRectF(10 * dpr, img.height() - 26 * dpr, img.width() - 20 * dpr, 20 * dpr), Qt.AlignLeft, text)


def _frame_points(objs):
    """Points for framing a view: lines, surfaces and (as a sample) meshes."""
    pts = []
    for o in objs:
        if hasattr(o, "geo"):
            if not o.missing and len(o.geo.vertices):
                v = o.geo.vertices[::max(1, len(o.geo.vertices) // 5000)].astype(float)  # a sample suffices
                pts.append(v @ o.xform[:3, :3].T + o.xform[:3, 3])
        elif isinstance(o, SurfaceObject):
            q = None if getattr(o, "aid", False) else _surface_points(o)
            if q is not None:
                pts.append(q)
        elif hasattr(o, "curves"):
            pts += [c.points for c in o.curves if not c.hidden and len(c.points)]
    return pts


def render_views(viewport, data, folder, focus=False):
    """Four overviews (without drawing aids), framed on everything visible; the top-level objects of the hierarchy
    are labeled (where notes point is shown per leader line by render_anchor_views - in the overview it would be crowded).
    focus: the same four views framed on the focus, the focus highlighted, the context in X-ray
    (views/focus_*.png)."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor
    vp = viewport
    vis = data["_vis"]
    b = data["focus"]["bounds"] if focus else data["bounds"]
    lo, hi = (np.array(b["min"]), np.array(b["max"])) if b else (np.full(3, -5.0), np.full(3, 5.0))
    center, radius = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1.0)
    chosen = data["_focus"] if focus else set()
    focus_lines = {c.id for st, c in vis["curves"] if st.id in chosen and not c.paint}  # brushes kept their color
    objs = [o for o in vis["strokes"] + vis["surfaces"] + vis["meshes"] if not focus or o.id in chosen]
    frame_pts = _frame_points(objs)
    frame_pts = np.vstack(frame_pts) if frame_pts else np.vstack([lo, hi])
    labels = [] if focus else [(text, q, False) for text, q in _top_labels(vp.scene, data)]
    files = []
    os.makedirs(folder, exist_ok=True)
    with _keep_camera(vp) as cam:
        for name, back, ortho in VIEWS:
            _frame(vp, cam, back, ortho, center, radius, frame_pts)
            saved_xray = vp.xray
            vp.xray = saved_xray or focus  # focus often lies inside (cockpit in the fuselage): context see-through
            try:
                img = _clean_shot(vp, select=sorted(chosen), lines=[c for _, c in vis["curves"]
                                                                    if focus and c.id in focus_lines])
            finally:
                vp.xray = saved_xray
            px_per_m = img.height() / (2 * cam.dist * math.tan(math.radians(45) / 2))  # before cropping
            img, (ox, oy) = _crop_to_content(vp, img, frame_pts) if ortho else (img, (0, 0))
            bg = img.copy()
            if not focus:  # without labels: as image input (image generation) and for comparing
                os.makedirs(os.path.join(folder, "clean"), exist_ok=True)
                bg.save(os.path.join(folder, "clean", f"{name}.png"))
                files.append(f"views/clean/{name}.png")
            p, dpr = _painter(img)
            pts = np.array([q for _, q, _ in labels]) if labels else np.zeros((0, 3))
            screen, ok = vp.project(pts) if len(pts) else (np.zeros((0, 2)), np.zeros(0, bool))
            taken = []
            for (text, _, _), (x, y), shown in zip(labels, screen, ok):
                if shown and np.isfinite(x):
                    _draw_label(p, QPointF(x * dpr - ox, y * dpr - oy), text, QColor(58, 56, 54), dpr, taken=taken,
                                bg=bg)
            if ortho:  # scale: orthographic is the same everywhere
                _scale_bar(p, img, dpr, px_per_m)
            _caption(p, img, dpr, (f"FOCUS (highlighted, context X-ray) – {name} – " if focus else f"{name.upper()} – ")
                     + f"{'orthographic' if ortho else 'perspective'} · Y up · ids as in brief.md"
                     + ("" if focus else " · where notes point: views/note_N#_k.png"))
            p.end()
            fname = f"focus_{name}.png" if focus else f"{name}.png"
            img.save(os.path.join(folder, fname))
            files.append(f"views/{fname}")
    return files


CROP_MAX_ASPECT = 3.0  # cropped overview at most as long as high (or vice versa)


def _crop_to_content(vp, img, pts):
    """Crop the orthographic overview to the geometry (margin for labels, at the bottom for scale and caption)
    - a flat wing otherwise fills only a strip of the image. -> (image, (x, y) offset in image pixels)"""
    from PySide6.QtCore import QRect
    screen, ok = vp.project(pts)
    screen = screen[ok & np.all(np.isfinite(screen), axis=1)]
    if not len(screen):
        return img, (0, 0)
    dpr = img.devicePixelRatio() or 1.0
    (x0, y0), (x1, y1) = screen.min(axis=0) * dpr, screen.max(axis=0) * dpr
    pad = max(0.08 * max(x1 - x0, y1 - y0), 50 * dpr)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad + 40 * dpr
    w, h = x1 - x0, y1 - y0
    if w > CROP_MAX_ASPECT * h:  # not into a narrow strip: make it taller
        grow = (w / CROP_MAX_ASPECT - h) / 2
        y0, y1 = y0 - grow, y1 + grow
    elif h > CROP_MAX_ASPECT * w:
        grow = (h / CROP_MAX_ASPECT - w) / 2
        x0, x1 = x0 - grow, x1 + grow
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), img.width()), min(int(y1), img.height())
    if (x1 - x0) * (y1 - y0) > 0.9 * img.width() * img.height():
        return img, (0, 0)  # saves hardly anything
    out = img.copy(QRect(x0, y0, x1 - x0, y1 - y0))
    out.setDevicePixelRatio(dpr)
    return out, (x0, y0)


def _frame(vp, cam, back, ortho, center, radius, frame_pts):
    """Frame the camera from direction back (toward the camera) onto frame_pts - measure the visible geometry along the
    view direction (space diagonal or corners of the overall dimensions leave a lot of empty margin with an oblique view)."""
    back = np.array(back, float)
    back /= np.linalg.norm(back)
    cam.set_state(center.copy(), radius / math.tan(math.radians(45) / 2), view_for_back(back))
    cam.ortho = ortho
    rel = frame_pts - center
    aspect = vp.width() / max(vp.height(), 1)
    t = math.tan(math.radians(45) / 2)
    need = np.maximum(np.abs(rel @ cam.up), np.abs(rel @ cam.right) / aspect) * 1.08 / t  # per corner
    depth = np.zeros(len(rel)) if ortho else rel @ cam.back  # perspective: closer to the camera = larger
    cam.set_state(center.copy(), max(float((need + depth).max()), 1.0), view_for_back(back))


def board_heroes(scene):
    """Hero cameras: screenshots in the note boards with the "hero camera" checkbox - [{name, view, aspect, source}] in
    the order of the notes (global note last)."""
    from PySide6.QtGui import QImage
    out = []
    for note in scene.notes + [scene.global_note]:
        for k, item in enumerate(i for i in note.board.get("items", []) if i.get("kind") == "image"):
            if not (item.get("hero") and item.get("view")):
                continue
            blob = scene.asset_bytes(item["file"])
            img = QImage.fromData(blob) if blob else QImage()
            aspect = img.width() / img.height() if not img.isNull() and img.height() else None
            name = item.get("hero_name") or f"{note.title or 'Hero'} {k + 1}"
            out.append({"name": name, "view": item["view"], "aspect": aspect,
                        "source": "project note" if note is scene.global_note else note.title,
                        **({"prompt": item["hero_prompt"]} if item.get("hero_prompt") else {})})
    return out


def frame_all(vp, cam, data, back, ortho):
    """Frame the camera from direction back onto everything visible (like the overviews)."""
    b = data["bounds"]
    lo, hi = (np.array(b["min"]), np.array(b["max"])) if b else (np.full(3, -5.0), np.full(3, 5.0))
    vis = data["_vis"]
    pts = _frame_points(vis["strokes"] + vis["surfaces"] + vis["meshes"])
    _frame(vp, cam, back, ortho, (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1.0),
           np.vstack(pts) if pts else np.vstack([lo, hi]))


def render_heroes(viewport, data, folder):
    """Hero cameras: one image per camera without labels (views/hero_*.png) - the scene's own, otherwise three
    default angles on everything visible. Writes data["cameras"] (position, view direction, field of view) for brief and USD."""
    vp = viewport
    scene = vp.scene
    b = data["bounds"]
    lo, hi = (np.array(b["min"]), np.array(b["max"])) if b else (np.full(3, -5.0), np.full(3, 5.0))
    center, radius = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1.0)
    vis = data["_vis"]
    frame_pts = _frame_points(vis["strokes"] + vis["surfaces"] + vis["meshes"])
    frame_pts = np.vstack(frame_pts) if frame_pts else np.vstack([lo, hi])
    own = board_heroes(scene)
    shots = [(h["name"], h) for h in own] if own else [(name, back) for name, back in DEFAULT_HEROES]
    files, cams, used, id_maps = [], [], set(), []
    os.makedirs(folder, exist_ok=True)
    with _keep_camera(vp) as cam:
        for name, how in shots:
            aspect = None
            if own:
                view = how["view"]
                cam.set_state(np.array(view["target"], float), float(view["dist"]), np.array(view["rot"], float))
                cam.ortho, aspect = bool(view.get("ortho")), how["aspect"]
            else:
                _frame(vp, cam, how, False, center, radius, frame_pts)
            plain = name.lower().translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}))
            key = re.sub(r"[^a-z0-9_]+", "_", plain).strip("_") or "hero"  # file and prim name
            key = key if key.startswith("hero") else f"hero_{key}"
            while key in used:
                key += "_"
            used.add(key)
            img, idm = _clean_shot(vp, aspect=aspect, ids=True)
            img.save(os.path.join(folder, f"{key}.png"))
            files.append(f"views/{key}.png")
            id_maps.append((key, idm))
            # clean_image keeps the vertical field of view and crops to the aspect ratio
            cams.append({"name": name, "id": key, "image": f"views/{key}.png", "default": not own,
                         **({"from": how["source"]} if own else {}),
                         **({"prompt": how["prompt"]} if own and how.get("prompt") else {}),
                         "eye": _r(cam.eye), "target": _r(cam.target), "up": _r(cam.up, 4),
                         "direction": _r(-cam.back, 4), "fov_vertical_deg": round(math.degrees(vp.fov), 2),
                         "aspect": round(aspect or vp.width() / max(vp.height(), 1), 4),
                         **({"orthographic_height": round(2 * cam.dist * math.tan(vp.fov / 2), 4)}
                            if cam.ortho else {})})
    data["cameras"] = cams
    files += _write_id_images(data, id_maps, cams, folder)
    return files


def id_palette(n):
    """n well distinguishable, rich colors (RGB 0-255) for the color ID, all different - never black (= nothing)."""
    import colorsys
    out, seen = [], set()
    for k in range(n):
        s, v = ((0.85, 0.95), (0.55, 0.85), (0.95, 0.6))[k % 3]
        rgb = tuple(round(c * 255) for c in colorsys.hsv_to_rgb((k * 0.618033988749895) % 1.0, s, v))
        while rgb in seen:  # rounded to the same: a touch different
            rgb = (rgb[0], rgb[1], (rgb[2] + 7) % 256)
        seen.add(rgb)
        out.append(rgb)
    return out


def _write_id_images(data, id_maps, cams, folder):
    """Color ID per hero camera (views/<camera>_ids.png): every visible mesh and surface flat in its own color,
    copies (symmetry/array) in that of the original, black = nothing (or only lines). Same color in all
    cameras; data["id_colors"]: color -> part -> notes pointing at it (anchor or arm of the octopus)."""
    from PySide6.QtGui import QImage
    from . import generators
    maps = data["_maps"]
    part_of = {**maps["mid"], **maps["fid"]}  # object id -> identifier (visible only)
    owner = {}  # pixel value -> identifier

    def part(v):
        if v not in owner:
            oid = int(v) if v >= 0 else None
            owner[v] = part_of.get(oid) or part_of.get(generators.vid_owner(oid)) if oid is not None else None
        return owner[v]

    found = []  # (key, values per pixel, identifier per value)
    for key, idm in id_maps:
        if idm is None:
            continue
        vals, inv = np.unique(np.round(idm).astype(np.int64), return_inverse=True)
        found.append((key, inv.reshape(idm.shape), [part(int(v)) for v in vals]))
    parts = sorted({p for _, _, ps in found for p in ps if p}, key=lambda x: (x[0] != "M", x[0], int(x[1:])))
    color = dict(zip(parts, id_palette(len(parts))))
    files = []
    for key, inv, ps in found:
        lut = np.array([color.get(p, (0, 0, 0)) for p in ps], dtype=np.uint8)
        rgb = np.ascontiguousarray(lut[inv])
        h, w = inv.shape
        QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy().save(os.path.join(folder, f"{key}_ids.png"))
        files.append(f"views/{key}_ids.png")
        cam = next(c for c in cams if c["id"] == key)
        cam["id_image"] = f"views/{key}_ids.png"
        cam["id_parts"] = [p for p in parts if p in ps]
    names = {**{e["id"]: e["name"] for e in data.get("meshes", [])}, **{e["id"]: e["name"] for e in data.get("surfaces", [])}}
    notes = {e["part"]: sorted({r["note"] for r in e["notes"]}, key=lambda n: int(n[1:]))
             for e in data.get("part_notes", [])}
    data["id_colors"] = [{"color": "#%02x%02x%02x" % color[p], "part": p, "name": names.get(p, ""),
                          **({"notes": notes[p]} if notes.get(p) else {})} for p in parts]
    return files


_OCCL_CACHE = {}  # id(Geometry) -> (Geometry, a, e1, e2) in Mesh-Koordinaten, float32


def _ray_hits(o, d, a, e1, e2, t_max):
    """Does the ray o + t·d (0 < t < t_max) hit one of the triangles (Möller-Trumbore, all at once)?"""
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    return bool(np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < t_max)))


def _occluded(vis, eye, point):
    """Does a visible surface or mesh lie between camera and point (the object hit itself counts too -
    from behind you do not see the spot either)? The last 2 % before the point do not count (it lies on it)."""
    d = point - eye
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return False
    d = d / length
    t_max = length * 0.98
    for m in vis["meshes"]:
        if m.missing or not len(m.geo.faces):
            continue
        lo, hi = m.world_bounds()
        seg_lo, seg_hi = np.minimum(eye, point), np.maximum(eye, point)
        if np.any(hi < seg_lo) or np.any(lo > seg_hi):
            continue
        hit = _OCCL_CACHE.get(id(m.geo))
        if hit is None or hit[0] is not m.geo:
            if len(_OCCL_CACHE) > 512:
                _OCCL_CACHE.clear()
            v, f = m.geo.vertices.astype(np.float32), np.asarray(m.geo.faces)
            hit = _OCCL_CACHE[id(m.geo)] = (m.geo, v[f[:, 0]], v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        inv = np.linalg.inv(m.xform)  # bring the ray into the mesh (t stays the same)
        lo_o = (inv @ np.append(eye, 1.0))[:3].astype(np.float32)
        lo_d = (inv[:3, :3] @ d).astype(np.float32)
        if _ray_hits(lo_o, lo_d, *hit[1:], t_max):
            return True
    for f in vis["surfaces"]:
        mesh = f.fill_mesh()
        if getattr(f, "aid", False) or mesh is None or not len(mesh[1]):
            continue
        v, t = np.asarray(mesh[0], float), np.asarray(mesh[1])
        if np.any(v.max(0) < np.minimum(eye, point)) or np.any(v.min(0) > np.maximum(eye, point)):
            continue
        a = v[t[:, 0]]
        if _ray_hits(eye, d, a, v[t[:, 1]] - a, v[t[:, 2]] - a, t_max):
            return True
    return False


VIEW_SHARE_OK = 0.5  # this much of the object must lie in front in a view for it to be taken
VIEW_SHARE_XRAY = 0.2  # if every direction stays below that: X-ray


def _visible_share(vp, target):
    """Share of the object that lies in front in the current camera (not occluded): its pixels in the pick buffer with
    everything, divided by its pixels when rendered alone. 0 if it is not in the image."""
    from PySide6.QtCore import QPointF
    if vp.renderer is None:
        return 1.0
    everything = vp._pick_items(vp._pickables())
    mine = [i + 1 for i, m in enumerate(everything) if getattr(m, "owner", m) is target]
    alone = vp._pick_items([target])
    if not mine or not alone:
        return 1.0
    size = vp._physical(QPointF())[2]
    vp.makeCurrent()
    try:
        vp.renderer.update_pick(size, vp.cam, vp._projection(), everything)
        front = int(np.isin(np.round(vp.renderer.read_pick(0, 0, *size)[0][..., 3]), mine).sum())
        vp.renderer.update_pick(size, vp.cam, vp._projection(), alone)
        total = int((np.round(vp.renderer.read_pick(0, 0, *size)[0][..., 3]) > 0).sum())
    finally:
        vp._unbind_vao()
        vp.doneCurrent()
    return front / total if total else 0.0


def _view_dirs(first):
    """View directions (camera -> target reversed: back) to try: first from the direction of the note, then around it
    slightly from above, then from below - sorted by closeness to the first."""
    first = first / max(float(np.linalg.norm(first)), 1e-9)
    ring = []
    for elev, n in ((35, 8), (65, 4), (-25, 6)):
        e = math.radians(elev)
        for k in range(n):
            a = 2 * math.pi * k / n
            ring.append(np.array([math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)]))
    ring.sort(key=lambda b: -float(b @ first))
    return [first] + ring


def _fit_dist(cam, vp, pts, center, back):
    """Distance at which all points fit into the image (perspective) - as with the overviews."""
    cam.set_state(center.copy(), 1.0, view_for_back(back))
    rel = pts - center
    aspect = vp.width() / max(vp.height(), 1)
    t = math.tan(math.radians(45) / 2)
    need = np.maximum(np.abs(rel @ cam.up), np.abs(rel @ cam.right) / aspect) * 1.25 / t
    return max(float((need + rel @ cam.back).max()), 0.05)


def render_anchor_views(viewport, data, folder):
    """One image per leader line of a note: the camera shows the hit object whole (for a line: the line),
    highlighted as selected, with the note's label and a stroke to the spot. First from the direction of the note, otherwise
    around - the first view in which at least half of the object lies in front (line/point: the spot
    free); otherwise the best, and if that is poor too, with X-ray. Without an object (point in empty space): the surroundings of the
    spot. Writes "view" (and "xray") into the anchor."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QPen

    from . import theme
    vp, scene = viewport, viewport.scene
    vis, maps = data["_vis"], data["_maps"]
    by_id = {**{maps["sid"][o.id]: o for o in vis["strokes"]}, **{maps["fid"][o.id]: o for o in vis["surfaces"]},
             **{maps["mid"][o.id]: o for o in vis["meshes"]}}
    curves = {c.id: c for _, c in vis["curves"]}
    files = []
    os.makedirs(folder, exist_ok=True)
    with _keep_camera(vp) as cam:
        cam.ortho = False
        for n, entry in zip(scene.notes, data["notes"]):
            for k, (anchor, a) in enumerate(zip(n.anchors, entry["anchors"]), 1):
                point = np.asarray(a["point"], float)
                target = by_id.get((a.get("target") or {}).get("id"))
                if a.get("line") and anchor.curve_id in curves:
                    pts = curves[anchor.curve_id].points  # line: show the line itself whole
                elif target is not None:
                    pts = _object_points(scene, target, set(_vis_ids(vis)))
                else:
                    pts = None
                if pts is None or not len(pts):
                    r = max(0.06 * data["_size"], 0.3)
                    pts = np.vstack([point - r, point + r])
                faint = [by_id[i] for m in a.get("members", []) if not m.get("line")
                         for i in m.get("parts", [m["id"]]) if i in by_id]  # group: its parts
                faint_lines = [curves[c.curve_id] for c in anchor.members if c.curve_id in curves]
                more = [q for o in faint for q in [_object_points(scene, o, set(_vis_ids(vis)))] if q is not None]
                more += [c.points for c in faint_lines]
                pts = np.vstack([pts, point[None], *more])  # the parts also meant belong in the image
                center = (pts.min(0) + pts.max(0)) / 2
                first = np.asarray(n.pos, float) - point
                if np.linalg.norm(first) < 1e-9:
                    first = np.array([0.62, 0.48, 0.62])
                pickable = hasattr(target, "fill_mesh") or hasattr(target, "geo")  # surface or mesh (None: no)
                best, xray = (-1.0, None), False
                for back in _view_dirs(first):
                    dist = _fit_dist(cam, vp, pts, center, back)
                    cam.set_state(center.copy(), dist, view_for_back(back))
                    if pickable:  # mesh/surface: how much of it can be seen (pick buffer)?
                        share = _visible_share(vp, target)
                    else:  # line or point in empty space: is the spot free?
                        share = 0.0 if _occluded(vis, center + back * dist, point) else 1.0
                    if share > best[0]:
                        best = (share, (back, dist))
                    if share >= VIEW_SHARE_OK:
                        break
                chosen = best[1]
                if best[0] < VIEW_SHARE_XRAY:  # visible well from nowhere: through the hull (X-ray)
                    xray = True
                cam.set_state(center.copy(), chosen[1], view_for_back(chosen[0]))
                saved_xray = vp.xray
                vp.xray = xray or saved_xray
                try:
                    line = curves.get(anchor.curve_id) if a.get("line") else None  # line: color only it
                    img = _clean_shot(vp, select=[target.id] if target is not None and line is None else [],
                                      lines=[line] if line is not None else [],
                                      faint=[o.id for o in faint], faint_lines=faint_lines)
                finally:
                    vp.xray = saved_xray
                p, dpr = _painter(img)
                screen = vp.project(point[None])[0][0]
                at = QPointF(screen[0] * dpr, screen[1] * dpr)
                text = f"{entry['id']} {entry['title']}"
                bw, bh = (16 + 7.6 * len(text)) * dpr, 20 * dpr
                corner = QPointF(14 * dpr, 14 * dpr) if at.x() > img.width() / 2 else \
                    QPointF(img.width() - bw - 14 * dpr, 14 * dpr)  # label into the corner away from the spot
                box = QRectF(corner.x(), corner.y(), bw, bh)
                color = QColor(theme.TERRACOTTA)
                p.setPen(QPen(color, 1.6 * dpr))
                p.drawLine(QPointF(box.center().x(), box.bottom()), at)
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(at, 5 * dpr, 5 * dpr)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 255, 255, 235))
                p.drawRoundedRect(box, 3 * dpr, 3 * dpr)
                p.setPen(QPen(color, 1.4 * dpr))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(box, 3 * dpr, 3 * dpr)
                p.drawText(box, Qt.AlignCenter, text)
                t = a.get("target")
                what = f"{t['kind']} {t['id']} {t['name']}" if t else "no object (free point)"
                also = f" · also meant (lighter): {len(a['members'])} part(s)" if a.get("members") else ""
                _caption(p, img, dpr, f"{entry['id']} – arrow {k} of {len(entry['anchors'])} → {what}"
                                      + (" – highlighted" if target is not None else "") + also
                                      + (" · X-ray: hidden from every side" if xray else ""))
                p.end()
                path = f"views/note_{entry['id']}_{k}.png"
                img.save(os.path.join(folder, os.path.basename(path)))
                a["view"] = path
                if xray:
                    a["xray"] = True
                files.append(path)
    return files


def _vis_ids(vis):
    return [o.id for o in vis["strokes"] + vis["surfaces"] + vis["meshes"] + vis["nulls"]]


# --- USD ------------------------------------------------------------------------------------------------

def _point_attrs(c, refs=None):
    """Point attributes orient/pscale (F4) and hardness (crease along the line: how strongly the lines crossing it kink,
    interpolated proportionally) as primvars per point - only if the line carries any."""
    out = []
    crease = components.crease_along(c, refs) if refs is not None and c.pids else None
    if crease is not None and crease.any():
        out.append(f'            float[] primvars:hardness = {_fl(crease)} (interpolation = "vertex")')
    if not c.pattr or not c.pids:
        return out
    q = np.tile([1.0, 0, 0, 0], (len(c.points), 1))
    s = np.ones(len(c.points))
    for k, pid in c.pids.items():
        if pid in c.pattr and 0 <= k < len(c.points):
            q[k], s[k] = c.pattr[pid][:4], c.pattr[pid][4]
    quats = ", ".join(f"({w:.5g}, {x:.5g}, {y:.5g}, {z:.5g})" for w, x, y, z in q)
    return out + [f'            quatf[] primvars:orient = [{quats}] (interpolation = "vertex")',
                  f'            float[] primvars:pscale = {_fl(s)} (interpolation = "vertex")']


def _fl(arr):
    return "[" + ", ".join(f"{x:.4g}" for x in np.asarray(arr, float).ravel()) + "]"


def write_usd(scene, data, path):
    """scene.usd in the brief: the scene as USD (usd_export) with the same identifiers, resolution as usual for the export; the
    note images lie next to it in the brief folder."""
    from . import usd_export
    usd_export.write(scene, path, data=data, image_root=True)


# --- Brief (Markdown) -------------------------------------------------------------------------------------

ARROW_LABEL = 40  # arrow label in the md at most this long - the whole text is above it in the section


def _caption_md(i, captions=None, parts=None):
    """Caption after the image, with the parts it belongs to: “ - “...” -> parts H3, M5”."""
    cap, ps = (captions or {}).get(i), (parts or {}).get(i)
    return (f" – „{cap}“" if cap else "") + (f" → parts {', '.join(ps)}" if cap and ps else "")


def _video_md(v):
    """The video details after its preview image."""
    if v.get("missing"):
        return f" – poster of a video that is MISSING (`{v['file']}`, {v['seconds']} s) – ask what it showed"
    return f" – poster of the video `{v['file']}` ({v['seconds']} s), overview sheet `{v['overview']}`"


def _image_md(i, shots, captions=None, parts=None):
    """Image on the board; screenshots are only marked as such - their camera is in brief.json (the image shows the
    view anyway)."""
    return f"`{i}`" + (" (viewport screenshot)" if i in shots else "") + _caption_md(i, captions, parts)


def _arrow_md(a):
    def end(e):
        if e is None:
            return "an empty spot"
        if "text" in e:
            text = " ".join(e["text"].split())
            if len(text) > ARROW_LABEL:  # shorten at a word boundary
                text = (text[:ARROW_LABEL].rsplit(" ", 1)[0] or text[:ARROW_LABEL]).rstrip(" ,;:–") + " …"
            return f"„{text}“"
        return f"{'screenshot' if e['screenshot'] else 'image'} `{e['image']}`"
    return f"**{a['id']}** {end(a['from'])} → {end(a['to'])}"


def _sections_md(n):
    """A board in sections: image of the excerpt, texts in reading order, images, arrows (P#). What sections
    and P# are is explained once by the reading aid; sections without text name the open points."""
    shots = {s["image"]: s["camera"] for s in n.get("screenshots", [])}
    md = [f"- whole board: `{n['board']}`"]
    for k, sec in enumerate(n["sections"], 1):
        title = f" – {sec['title']}" if sec.get("title") else ""
        md += ["", f"**Section {k}{title}** – `{sec['image']}`"]
        for t in sec["texts"]:
            md += ["", "> " + t.strip().replace("\n", "\n> ")]
        if sec["images"]:
            missing = set(n.get("missing") or ())
            videos = {v["poster"]: v for v in n.get("videos") or ()}
            md += [""] + [f"- {_image_md(i, shots, n.get('captions'), n.get('caption_parts'))}" +
                          (" – MISSING in the scene file" if i in missing else "") +
                          (_video_md(videos[i]) if i in videos else "") for i in sec["images"]]
        md += [f"- arrow {_arrow_md(a)}" for a in sec["arrows"]]
        if sec["unclear"]:
            md.append(f"- {sec['unclear']} hand-drawn strokes not clearly connected to a text or image – see the picture")
    return md


def _board_md(n):
    """Lines on images, screenshots and painting of a note board."""
    if n.get("sections"):
        return _sections_md(n)
    shots = {s["image"]: s["camera"] for s in n.get("screenshots", [])}
    md = []
    videos = {v["poster"]: v for v in n.get("videos") or ()}
    for i in n["images"]:
        md.append((f"- image: `{i}`" if i not in shots else
                   f"- viewport screenshot: `{i}` – shows the sketch the note is about")
                  + _caption_md(i, n.get("captions"), n.get("caption_parts")) + (_video_md(videos[i]) if i in videos else ""))
    if n.get("board"):
        extra = f", incl. {n['drawings']} hand-drawn marks (arrows, circles, sketches over the images)" \
            if n.get("drawings") else ""
        md.append(f"- whole note board as one picture (images, text{extra}): `{n['board']}`")
    return md


SURFACE_KINDS = {"curve_surface": "curve surface", "loft": "loft", "fill": "fill", "extrude": "extrusion"}


# --- Task (dialog on export, saved in the scene) -----------------------------------------------------
# The dialog sets only what the AI cannot clarify from the scene itself: the task in one sentence, notes on
# this export, "discuss only" and the look pre-stage. Everything else (scope of the round, kinds of work,
# bindingness, tools, way of working) the AI clarifies in the first round with the clarification list (CLARIFY) from the
# notes - previously general dialog sentences stood in for these in the brief, which contradicted the more precise notes
# (blind test Glider 04, 31.10.2026). What is not given is missing from the task - the brief claims nothing about it.
TASK_KINDS = {
    "discuss": ("Nur besprechen – nichts bauen", "do not build anything – review the sketch, answer the artist's "
                "questions and make suggestions"),
}
TASK_LOOK = "look"  # task part look pre-stage: images from the hero cameras before building
# Bindingness per object (outliner column "Brief") - lines, painting and images show intent (reading aid)
TASK_LEVELS = {
    "fixed": ("Fest – nicht anfassen", "fixed – keep them exactly as they are; do not change or rebuild them"),
    "exact": ("Maßgeblich – verfeinern erlaubt", "authoritative – follow them closely; refining (smoother, cleaner, "
              "more detail) is welcome"),
    "basis": ("Grundlage – Ausführung frei", "the basis – proportions, silhouette and placement hold; how you build "
              "them (topology, details) is up to you"),
    "guide": ("Orientierung", "orientation only – look and character; you may rebuild them completely"),
}
TASK_TEXTS = ("task", "notes")
# Clarification list of the first round: what the dialog used to ask - the AI answers it from the notes (with source),
# asks only about open and contradictory points and strikes what does not apply
CLARIFY = (
    "**Goal and this round:** the end goal of the project, and what this round delivers – which parts, and how far "
    "(a blockout to check proportions, complete for design approval, or production-ready). What is explicitly left "
    "for later?",
    "**Kinds of work now:** geometry; moving parts that work (pivots, states as parameters); materials; a rig for an "
    "animator; animation; layout; generated images – which are asked for in this round, which are only context?",
    "**Existing geometry – what holds, what is free:** unless the notes say otherwise, silhouette, proportions, "
    "positions, how the parts connect and the axes of motion hold; topology, cross-sections, wall thickness and the "
    "construction are yours to develop. Which parts are exceptions – kept exactly, or only an orientation?",
    "**Moving or deforming parts:** axes and pivots, range of motion, the states, what moves together, and how "
    "strictly collisions matter.",
    "**Tools and delivery:** which software for what (and whether something must stay procedural or editable), "
    "what is delivered (files, format, hierarchy, scale); for a production handover the technical requirements "
    "(topology, UVs, naming, budget).",
    "**Working mode:** how independently you work between reviews, and what the artist wants to see at a review "
    "(views, display).",
)


def task_clean(task):
    """Only what is given and valid from the saved task settings (empty: no task). Older fields
    (starting point, task parts, build stage, bindingness, environment, way of working) are dropped - the AI
    clarifies that itself now; "discuss only" has no look pre-stage."""
    task = task or {}
    out = {k: task[k].strip() for k in TASK_TEXTS if isinstance(task.get(k), str) and task[k].strip()}
    if task.get("kind") in TASK_KINDS:
        out["kind"] = task["kind"]
    if isinstance(task.get("based_on"), str) and task["based_on"].strip():  # review round: brief folder of the running run (review_round.py)
        out["based_on"] = task["based_on"].strip()
    if TASK_LOOK in (task.get("parts") or []) and out.get("kind") != "discuss":
        out["parts"] = [TASK_LOOK]
    if isinstance(task.get("recipe"), str) and task["recipe"].strip() not in ("", "default"):  # brief recipe (recipes.py)
        out["recipe"] = task["recipe"].strip()
    return out


def _task_md(t, bindings=None, focus=None):
    """Task at its core: sentence and notes of the artist, focus, bindingness per object from the
    outliner; without input a sentence that the first round clarifies the goal."""
    t = t or {}
    md = ["### Task (set by the artist on export)"]
    if t.get("task"):
        md += [f"**In one sentence:** {t['task']}", ""]
    if t.get("notes"):
        md += ["**Artist's notes for this task** (authoritative for this task – where they contradict a note in the "
               "scene, they win; point out the conflict):", *[f"> {l}" if l.strip() else ">"
                                                               for l in t["notes"].splitlines()], ""]
    if not t.get("task") and not t.get("notes"):
        md += ["The artist set no task sentence on export – the first round works out the goal from the notes.", ""]
    if focus:
        md += ["**Focus:** build only the parts listed under „Focus – what to build“ (highlighted in "
               "`views/focus_*.png`); the rest of the scene is context.", ""]
    if bindings:
        md += ["**Set by the artist on single parts** (they override the notes' general rules):"]
        md += [f"- {b['id']} {b['name']}{' and everything in it' if b['group'] else ''}: {TASK_LEVELS[b['level']][1]}."
               for b in bindings]
        md.append("")
    if TASK_LOOK in (t.get("parts") or []):
        md += ["**Look images first:** before anything is built, generate images from the hero cameras and let the "
               "artist pick (see „Look images“).", ""]
    return md


def _look_md(cams, recipe=None):
    """Look pre-stage (ticked in the dialog): how to make the look images is the recipe's (template "look"); without
    one, Mule only passes on the artist's wish and the approved prompts."""
    from . import recipes
    ctx = {"cameras": [{"id": c["id"], "name": c.get("name", c["id"]), "image": c.get("image", ""),
                        "prompt": c.get("prompt") or ""} for c in cams]}
    text = (recipe or recipes.default()).render("look", ctx)
    if text is None:
        text = ("## Look images (before building)" + chr(10) + "The artist asked for look images from the hero cameras "
                "before building." + "".join(chr(10) + f"- `{c['id']}` – prompt approved by the artist: {c['prompt']}"
                                             for c in ctx["cameras"] if c["prompt"]))
    return text.rstrip(chr(10)).split(chr(10))



CORE_FILE = "core.md"  # the core once more as its own file - for rereading at the start of each unit
READ_SECTION = "## Reading the scene"  # the reference material begins here (what has been clarified stands directly before it)


def _core_md(data):
    """Core at the very top (and as core.md): task, first round with clarification list, check units and task list,
    work per unit with build order, checking and reporting. What always applies stands here briefly and in one place; the
    rest of the brief is reference material per part (blind test Glider 04: the long brief was only used
    in parts, priorities and note references got lost in execution)."""
    task = data.get("task") or {}
    builds = task.get("kind") != "discuss"
    # Blind test Glider 003: 6 of ~60 images opened, the rest "when we get to these parts" - hence the count,
    # proof in the first message and a clear separation of what is opened now and what only at the unit
    cams = data.get("cameras") or []
    heroes = 0 if not cams or cams[0]["default"] else len(cams)
    boards = [n for n in data.get("notes", []) + [data.get("project_note") or {}] if n.get("board")]
    secs = sum(len(n.get("sections") or []) for n in boards)
    count = [f"{k} {w}" for k, w in ((heroes, "hero"), (len(boards), "whole-board"), (secs, "section")) if k]
    look = ("- **Pictures first:** open now every hero picture (`views/hero_*.png`), every whole-board picture "
            "(`notes/board_N#.png`) and every section picture (`notes/board_N#_k.png`)"
            + (f" – here {', '.join(count)} pictures" if count else "") + ", all of them, also those of parts "
            "this round does not build: they carry the whole object, its intent and how the parts belong together. "
            "With the Mule tools (MCP server `sketch_machine`) and no export at hand: render_view per hero camera, note_images with "
            "sections. Write one line per section: what you understand there, and say in the first message how many "
            "of these pictures you opened – one you did not open is not read. Open a single picture now where the "
            "section picture does not make it readable or something stays unclear; the rest of the single pictures "
            "you open in full when you work on their parts (see „Working on a unit“).")
    captions = ("- **Captions:** a caption („…“) is the artist's word on what a picture stands for and what matters "
                "in it, not a full description – it does not replace opening the picture. Every reference picture "
                "should have one – for each that has none, propose a caption in this first message: what it stands "
                "for (form, detail, material, mood, function …), what matters in it („only the hinge on the left – the "
                "photo's proportions do not apply“) and the parts it belongs to (ids); if its board already says this, "
                "the caption only points there. The artist confirms (with the Mule tools: store the "
                "confirmed ones with set_image_captions).")
    videos = [v for n in data.get("notes", []) + [data.get("project_note") or {}] for v in n.get("videos") or ()]
    video_step = [("- **Videos:** the boards hold " + (f"{len(videos)} reference videos" if len(videos) > 1 else
                   "a reference video") + " (`videos/`), each with its poster on the board and an overview sheet "
                   "(`videos/*_overview.jpg`: a few frames with timestamps). Look at the sheets and captions now and "
                   "say in one line per video what you take it to show – that is all a video costs by default. Do "
                   "not cut a video into frames on your own: in this first message ask which videos need a closer "
                   "look and what to look for (a motion, its timing, one detail, the mood). Then read only those, "
                   "only the frames that answer it (Mule tool video_frames, otherwise ffmpeg or what you have at "
                   "hand), and name the timestamps you used.")] if videos else []
    clarify = ("- **Clarify:** answer each question below yourself from the notes, the project note, the clarified "
               "points and the files they name (e.g. an existing rig or scene – read it rather than asking what it "
               "contains), naming the source (N#, section, project note, file); mark the ones that do not apply. Send all "
               "answers as one list – the artist confirms or corrects it in one go. Ask openly only what the notes "
               "leave open or where they contradict each other." + (" With the Mule tools, store the "
               "confirmed answers with write_clarified." if builds else ""))
    md = ["## Core – read this first, and again at the start of every unit",
          f"The same text is in `{CORE_FILE}` next to this file. This is a Sketch Brief from Mule, a 3D "
          "sketching tool – what is sketched may be a thing, a creature, a place or a whole scene; „the object“ means "
          "any of them. **It is a prompt, not a finished spec.** This core says what to do and in which order; the "
          "rest of `brief.md` is reference – the scene, the notes and their pictures – that you come back to part by "
          "part.", "", *_task_md(task, data.get("bindings"), data.get("focus"))]
    if not builds:
        md += ["### What to do",
               "**Do not build anything.**", look, captions, *video_step, "- Then explain in your own words what you understood (the object, "
               "its parts, the intent), list open questions and contradictions, and make suggestions. The questions "
               "below show what usually needs to be clear:", *[f"    {k}. {q}" for k, q in enumerate(CLARIFY, 1)],
               "- When you suggest a change to a part: state its intent in one sentence and, for a new shape, show a "
               "quick picture (a section or a view from above). Ask on the level of intent, not about ids.", ""]
        return md
    md += ["### First round – before building",
           look, captions, *video_step, clarify, *[f"    {k}. {q}" for k, q in enumerate(CLARIFY, 1)],
           "- **Review units:** cut the work into units that are judged as a whole – by function and form, not by the "
           "scene hierarchy (e.g. a hose system, a roof, a creature's head, a ramp with its bridge); as few as "
           "readable pictures allow, but split where parts are built differently or checked in different views (e.g. "
           "rigid housings and a soft membrane or cloth). Show them as one overview picture, each unit in its own "
           "color with a legend.",
           "- **Work list:** for each unit, the concrete assignments of this round, one line each: part(s) by id · "
           "source (note and section, or picture file) · what you build, as a visible result (e.g. „bearing housing "
           "round the axis of M12, bolted to the plate“ – not „detail pass“) · the view that will show it · status "
           "(open, done, deferred, replaced). Analysing, testing, review sheets and exports are not lines – they "
           "serve the lines. Every note with an anchor or an „also meant“ part on the unit's parts appears in a line, or is "
           "listed as not for this round with the reason. A precise line in the sketch is matched closely, a loose "
           "gesture only in proportion and character. Keep the list in `review/worklist.md`; new feedback adds "
           "lines, it never pushes out agreed ones.",
           "- Send the answers, the units and the work list in one message and wait for the artist's go. After it, "
           "work through the list on your own until the agreed review; ask again only when the intent contradicts "
           "itself, something fixed would change, or the work grows beyond the list.", "",
           "### Working on a unit",
           "- **Start:** read this core again; open every note with an anchor or an „also meant“ part on the unit's parts (a note usually "
           "means similar parts too) and the single pictures of those notes, now in full size – those with a caption too – and add each to your "
           "picture list; write your own strategy first, then look into the workflow bible (see „How to work with the artist“).",
           "- **Build order:** main form and cross-sections first, then transitions and connections, then the "
           "working assemblies (mounts, joints, hinges, attachments – for a creature: masses, joints, muscles; for a "
           "building: volumes, openings, structure), details last. Details on an unfinished main form are not "
           "progress. Materials go on shapes the artist has approved.",
           "- Infrastructure (setups, scripts, node networks) only where it makes a concrete modelling step possible; "
           "keep tool output short – errors, totals, the parts concerned.", "",
           "### Checking and reporting",
           "- Before a unit counts as done, a reviewer that does not know how you built it (a sub-agent; if you "
           "cannot start one, say so and do a separate pass of your own, labelled as self-check) checks in this "
           "order: 1. the work list – every line done and visible in its view? 2. shape against notes, sketch and "
           "references; 3. geometry and topology. Passing mesh checks is not a shape judgement. Fix what it finds "
           "within the list yourself.",
           "- Report in four groups: done and checked · done but not checkable · open · announced but not built. "
           "Never report done just because technical checks pass.", ""]
    return md


def _intro(data, recipe=None):
    """Reading aid (reference material): what the scene contains (depending on content) and how to read it."""
    from . import recipes
    lines = [l for l in data["lines"] if not l.get("paint")]
    parts = []
    if data["meshes"]:
        parts.append(f"**imported meshes** (M#, {len(data['meshes'])}) – existing model geometry and the starting "
                     "point: develop and build on it rather than starting over; not a finished model")
    if any(not f.get("helper") for f in data["surfaces"]):
        parts.append("**sketched surfaces** (F#) – parts the artist modelled in the tool (lofts, curve surfaces, "
                     "fills, extrusions); their display material and color hint at the intended look")
    if lines:
        parts.append("**lines** (L#, grouped into stroke objects S#) – drawn shapes, edges, paths (e.g. hoses, "
                     "branches, road axes, a camera path) and construction lines")
    if len(lines) < len(data["lines"]):
        parts.append("**paint** – color painted onto surfaces or meshes (feedback, material or detail hints)")
    if data["notes"] or (data.get("project_note") or {}).get("text"):
        parts.append("**notes** (N#) with text, reference images, viewport screenshots and hand-drawn marks – the "
                     "artist's intent; the project note applies to everything")
    # how to read notes and the sketch is the recipe's (template "reading"; Mule Default: today's reading aid)
    reading = (recipe or recipes.default()).render("reading", {})
    reading = reading.rstrip(chr(10)).split(chr(10)) + [""] if reading else []
    return ([READ_SECTION, "The scene contains:"]
            + [f"- {p}" for p in parts]
            + [""] + reading
            + ["Units: 1 unit = 1 m, Y is up. Ids: **H#** groups (nulls), **M#** meshes, **F#** surfaces, **S#** "
               "stroke objects, **L#** lines, **G#** connected line groups, **N#** notes – the same in "
               "`views/*.png`, `scene.usd` (prim names) and `brief.json`. Hidden objects are not part of this brief. "
               "`scene.usd` is the scene itself for Houdini/Blender/Maya: hierarchy as Xforms, surfaces as polymeshes "
               "(quads), lines as curves, imported meshes referenced from their files, symmetry/array copies baked.",
               ""])


def _target_md(t):
    if not t:
        return "no object (free point in space)"
    # only the group directly above - the part tree shows the whole path
    return f"{t['kind']} **{t['id']}** `{t['name']}`" + (f" (in {t['path'].split(' › ')[-1]})" if t.get("path")
                                                         else "")


def _member_md(m):
    out = f"**{m['id']}** `{m['name']}`" + (f" (line {m['line']})" if m.get("line") else "")
    if m.get("parts"):
        out += f" (group – all {len(m['parts'])} part(s) inside: {_ranges_any(m['parts'])})"
    return out + (" with its mirror/array copies" if m.get("with_copies") else "")


def _ranges(ids):
    """["F16", "F17", …, "F46", "F48"] -> "F16–F46, F48"."""
    nums = sorted(int(i[1:]) for i in ids)
    out, start = [], None
    for k, n in enumerate(nums):
        if start is None:
            start = n
        if k + 1 == len(nums) or nums[k + 1] != n + 1:
            out.append(f"{ids[0][0]}{start}" + (f"–{ids[0][0]}{n}" if n != start else ""))
            start = None
    return ", ".join(out)


def _ranges_any(ids):
    """Mixed identifiers (H#, S#, F#, M#) summarized per kind: “F16–F46, M3, S5”."""
    kinds = {}
    for i in ids:
        kinds.setdefault(i[0], []).append(i)
    return ", ".join(_ranges(v) for _, v in sorted(kinds.items()))


def _hex(c):
    return "#" + "".join(f"{int(round(x * 255)):02x}" for x in c)


def _size(b):
    return "–" if not b else f"{b['size'][0]:.2f} × {b['size'][1]:.2f} × {b['size'][2]:.2f} m"


def _count(n):
    return f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n / 1e3:.1f} k" if n >= 1e3 else str(n)


def _surfaces_md(fs):
    """Surfaces of a null in one line: identifiers, kinds, material, colors, overall size (details: brief.json)."""
    from collections import Counter
    kinds = Counter(SURFACE_KINDS.get(f["kind"], f["kind"]) for f in fs)
    mats = Counter(f.get("material", "papier") for f in fs)
    colors = Counter(_hex(f["color"]) if f.get("color") else "no color" for f in fs)
    pts = [np.array([f["bounds"]["min"], f["bounds"]["max"]]) for f in fs if f.get("bounds")]
    size = _size(_bbox(np.vstack(pts))) if pts else "–"
    helper = sum(1 for f in fs if f.get("helper"))
    return (f"{_ranges([f['id'] for f in fs])}: {len(fs)} surface{'s' if len(fs) > 1 else ''} "
            f"({', '.join(f'{v} {k}' for k, v in kinds.items())}), material {', '.join(mats)}, "
            f"color {', '.join(f'{k} ×{v}' if v > 1 else k for k, v in colors.items())}, together {size}"
            + (f", {helper} helper surface(s) – drawing aids only" if helper else ""))


def _components_md(data):
    """Part tree as in the outliner: per null its meshes (name, triangles, size), surfaces (summarized) and
    stroke objects (lines, painting and what on), generators with their settings. Replaces the long lists per
    surface, line group and mesh - those are in brief.json."""
    hier = {h["id"]: h for h in data.get("hierarchy", [])}
    kids = {}
    for key, entries in (("S", data["objects"]), ("F", data["surfaces"]), ("M", data["meshes"])):
        for e in entries:
            kids.setdefault(e.get("parent"), []).append((key, e))
    for h in hier.values():
        kids.setdefault(h["parent"], []).append(("H", h))
    lines = {}
    for l in data["lines"]:
        lines.setdefault(l["object"], []).append(l)
    md = ["## Components (hierarchy as in the outliner)",
          "Groups (H#) with what they contain. Per surface, line and line group details: `brief.json`; exact geometry: "
          "`scene.usd`."]
    files = {}
    for m in data["meshes"]:
        files.setdefault(m["file"], []).append(m)
    for f, ms in files.items():
        tris = sum(m.get("triangles", 0) for m in ms)
        md.append(f"- imported file `{f}`: {len(ms)} meshes ({_ranges([m['id'] for m in ms])}), {_count(tris)} "
                  f"triangles" + (" – FILE MISSING" if any(m["missing"] for m in ms) else ""))

    def content(pid, pad):
        entries = kids.get(pid, [])
        ms = [e for k, e in entries if k == "M"]
        if ms:
            md.append(pad + "- meshes: " + "; ".join(
                f"**{m['id']}** `{m['name']}` ({_count(m.get('triangles', 0))} tris, {_size(m.get('bounds'))})"
                for m in ms))
        fs = [e for k, e in entries if k == "F"]
        if fs:
            md.append(pad + "- " + _surfaces_md(fs))
        for s in (e for k, e in entries if k == "S"):
            ls = lines.get(s["id"], [])
            paint = [l for l in ls if l.get("paint")]
            brush = sorted({l["color"] for l in ls if l.get("style") == "brush" and not l.get("paint")})
            on = sorted({l["on"] for l in paint if l.get("on")}, key=lambda x: (x[0], int(x[1:])))
            md.append(pad + f"- **{s['id']}** {s['name']}: {len(ls) - len(paint)} lines"
                      + (f" (brush colors {', '.join(brush)})" if brush else "")
                      + (f", {len(paint)} painted strokes" + (f" on {', '.join(on)}" if on else "") if paint else "")
                      + f", {_size(s.get('bounds'))}")
        for h in (e for k, e in entries if k == "H"):
            tree(h, pad)

    def tree(h, pad):
        what = {"null": "group", "symmetry": "SYMMETRY", "array": "ARRAY"}.get(h["kind"], h["kind"])
        md.append(f"{pad}- **{h['id']}** {h['name']} – {what}")
        inner = pad + "  "
        if "mirror" in h:
            m = h["mirror"]
            md.append(f"{inner}- mirror plane {m['plane']} through {m['origin']}, kept side normal "
                      f"{m['normal_kept_side']}; {m['clip']}. Houdini: {m['houdini']} (baked in scene.usd)")
        if "array" in h:
            a = h["array"]
            desc = {"radial": lambda: f"{a['count']} copies over {a['angle_deg']}° about axis {a['axis']} "
                                      f"through {a['pivot']}",
                    "linear": lambda: f"{a['count']} copies, step {a['step']}",
                    "line": lambda: f"{a['count']} copies along line {a['path']} (up: {a['up_line'] or 'carried'})"}
            md.append(f"{inner}- {desc[a['mode']]()}; the original {a['original']}. Houdini: {a['houdini']}"
                      + (f"; each copy turns {a['spin_deg_per_copy']}° further" if a.get("spin_deg_per_copy") else "")
                      + (f"; random rotate {a['random']['rotate']}, offset {a['random']['offset']}, seed "
                         f"{a['random']['seed']} – exact matrices in brief.json" if a.get("random") else ""))
        if h.get("note"):
            md.append(f"{inner}- {h['note']}")
        content(h["id"], inner)

    content(None, "")
    return md


def _open_points(data, lang="en"):
    """What the tool itself recognizes as open: missing images, sections only with images, strokes that cannot be assigned,
    notes without a target, missing mesh files - better to name than to silently omit. lang "de": for the
    task dialog (there first, so it can be fixed before the export)."""
    out = []
    boards = [(n["id"] + " " + n["title"], n) for n in data["notes"]]
    pn = data.get("project_note") or {}
    if pn.get("board"):
        boards.insert(0, ("Global-Notiz" if lang == "de" else "project note", pn))
    for name, n in boards:
        if n.get("missing"):
            files = ", ".join(f"`{m}`" for m in n["missing"])
            out.append((f"{name}: {len(n['missing'])} Bild(er) fehlen in der Szenendatei ({files})",
                        f"{name}: {len(n['missing'])} picture(s) missing in the scene file, not in this brief "
                        f"({files}) – ask what they showed"))
        absent = [v["file"] for v in n.get("videos") or () if v.get("missing")]
        if absent:
            out.append((f"{name}: Video-Datei nicht gefunden ({', '.join(absent)})",
                        f"{name}: video file(s) not found, not in this brief ({', '.join(absent)}) – only the poster "
                        "is left; ask what they showed"))
        if n.get("uncaptioned"):
            k = len(n["uncaptioned"])
            out.append((f"{name}: {k} Referenzbild(er) ohne Unterschrift",
                        f"{name}: {k} reference picture(s) without a caption – propose captions in your first "
                        "message (see „Captions“), open each in full when you work on its parts, and add it "
                        "to your picture list"))
        secs = n.get("sections") or []
        empty = ", ".join(str(k) for k, sec in enumerate(secs, 1) if not sec["texts"])
        if empty:
            out.append((f"{name}: Abschnitt(e) {empty} ohne Text – nur Bilder und Skizzen",
                        f"{name}: section(s) {empty} have no text – pictures and sketches only, read them from the "
                        "picture"))
        unclear = sum(sec["unclear"] for sec in secs)
        if unclear:
            out.append((f"{name}: {unclear} Strich(e) im Brett keinem Text oder Bild zuzuordnen",
                        f"{name}: {unclear} hand-drawn stroke(s) not clearly connected to a text or image"))
    for n in data["notes"]:
        name = f"{n['id']} {n['title']}"
        if n.get("review_draft"):
            out.append((f"{name}: Review-Entwurf, noch nicht durchgesehen (im Brett „✓ Übernehmen“)",
                        f"{name}: drafted by an AI from a review recording, not yet checked by the artist – treat "
                        "it as likely but ask when it matters"))
        if not n["anchors"]:
            out.append((f"{name}: zeigt auf nichts in der Szene (keine Hinweislinie)",
                        f"{name}: points at nothing in the scene (no leader line)"))
        free = ", ".join(str(k) for k, anchor in enumerate(n["anchors"], 1) if not anchor.get("target"))
        if free:
            out.append((f"{name}: Hinweislinie(n) {free} enden im Leeren (oder auf Ausgeblendetem)",
                        f"{name}: arrow(s) {free} end in empty space (no object hit)"))
        gone = sum(a.get("members_missing", 0) for a in n["anchors"])
        if gone:
            out.append((f"{name}: {gone} Arm(e) der Notizkrake zeigen auf Gelöschtes (im Notizmodus gestrichelt – "
                        "entfernen oder neu ziehen)",
                        f"{name}: {gone} „also meant“ part(s) were deleted in the sketch – ask what they were"))
    missing = [m["id"] for m in data["meshes"] if m["missing"]]
    if missing:
        out.append((f"Meshes {_ranges(missing)}: Datei fehlt", f"imported meshes {_ranges(missing)}: file missing on "
                                                                 "export"))
    return [de if lang == "de" else en for de, en in out]


# Keywords by which the dialog recognizes whether the scene contains information for the look pre-stage (only a hint for
# the artist, not in the brief - keywords can deceive)
LOOK_WORDS = ("stimmung", "look", "stil", "style", "mood", "material", "farbe", "referenz")


def scene_text(scene, data):
    """Everything written in the scene (notes, global note, names), lowercase - for the keyword hints."""
    vis = data["_vis"]
    parts = [scene.global_note.title, *(data.get("project_note") or {}).get("text", [])]
    for n in data["notes"]:
        parts += [n["title"], *n.get("text", [])]
    parts += [o.name for o in vis["nulls"] + vis["meshes"] + vis["surfaces"] + vis["strokes"]]
    return " ".join(str(x) for x in parts if x).lower()


def _mentions(text, words):
    """Does one of the keywords occur? Short ones (up to 3 characters: uv, rig) only at the start of a word - "Original" is not a rig;
    longer ones also in the middle of a word (Flügelgelenk)."""
    return any(re.search(r"\b" + re.escape(w), text) if len(w) <= 3 else w in text for w in words)


def part_hints(task, text, images=False):
    """Hints in the dialog: no goal recognizable, or look pre-stage without information on mood and style."""
    task = task_clean(task)
    text = " ".join((text, task.get("task", ""), task.get("notes", ""))).lower()
    out = []
    if not task.get("task") and not text.strip():
        out.append(i18n.tr("Kein Auftragssatz und keine Notizen – die KI kennt das Ziel nicht"))
    if TASK_LOOK in task.get("parts", []) and not images and not _mentions(text, LOOK_WORDS):
        out.append(i18n.tr("Look-Vorstufe angekreuzt – keine Notiz nennt Stimmung, Stil oder Materialrichtung"))
    return out


def precheck(scene, data):
    """For the task dialog: open points from the scene (in the language of the interface) and the text for the keyword hints."""
    images = any(n.get("images") for n in data["notes"] + [data.get("project_note") or {}])
    return {"points": _open_points(data, i18n.LANG), "text": scene_text(scene, data), "images": images}


def _later_notes(data):
    """With focus (build brief, package v4): notes whose anchor and "also meant" parts all lie outside the focus
    belong in context.md (notes of later rounds, intent only). Notes without an anchor stay in the brief."""
    fc = data.get("focus")
    if not fc or (data.get("task") or {}).get("kind") == "discuss":
        return set()
    fid = set(fc.get("ids") or ())
    out = set()
    for n in data["notes"]:
        ids = set()
        for a in n["anchors"]:
            ids.add((a.get("target") or {}).get("id"))
            for m in a.get("members") or ():
                ids.add(m.get("id"))
                ids.update(m.get("parts") or ())
        if n["anchors"] and not ids & fid:
            out.add(n["id"])
    return out


def _v4_head(has_context, recipe=None):
    """Head of brief.md in a build brief: where the process is – from the recipe (template brief_head)."""
    from . import recipes
    text = (recipe or recipes.default()).render("brief_head", {"has_context": bool(has_context)})
    if text is None:
        text = "This file holds the notes of this round and the scene; the way of working comes with the recipe."
    return [text.rstrip("\n"), "", PRECEDENCE, ""]


# every build brief, whatever the recipe (2026-11-16): Mule's texts in brief.md never overrule the recipe – but the
# artist's own words in the notes are content and stay the artist's words
PRECEDENCE = ("**This file is content:** the scene, the artist's notes and the pictures. Where a reading aid or "
              "procedure in it disagrees with the way of working this brief came with (`CLAUDE.md` and the files it "
              "names, `project.md`), the way of working wins. The artist's own words in the notes are not reading "
              "aids – they stay what the artist asked for.")


def write_markdown(data, views, path, scene_name):
    """Write brief.md (and for "discuss only" core.md, with focus context.md); returns whether context.md exists."""
    builds = (data.get("task") or {}).get("kind") != "discuss"
    later = _later_notes(data)
    for n in data["notes"]:
        n.pop("later", None)
        if n["id"] in later:
            n["later"] = True
    from . import recipes
    # build brief: the way of working is the recipe's – also the reading aid, the look procedure and how open points
    # and later notes are put; "discuss only" keeps Mule's own texts
    recipe = recipes.get((data.get("task") or {}).get("recipe")) if builds else recipes.default()
    if builds:
        core = _v4_head(bool(later), recipe)
    else:
        core = _core_md(data)
        with open(os.path.join(os.path.dirname(path), CORE_FILE), "w", encoding="utf-8") as fh:
            fh.write("\n".join([f"# Sketch Brief – {scene_name}", "", *core]) + "\n")
    md = [f"# Sketch Brief – {scene_name}", "", *core, *_intro(data, recipe)]
    ctx_head = recipe.render("context_head", {})
    ctx = [f"# Context – notes of later rounds ({scene_name})",
           ctx_head.rstrip(chr(10)) if ctx_head is not None else
           "Read for intent and for how the parts belong together; they are not built in this round.", ""]
    b = data["bounds"]
    md += ["## Overview"]
    if b:
        x, y, z = b["size"]
        md.append(f"- Overall size (all visible geometry, incl. meshes): **{x:.2f} m (X) × {y:.2f} m (Y, up) × "
                  f"{z:.2f} m (Z)**, from {b['min']} to {b['max']}")
    md.append(f"- {len(data['meshes'])} imported meshes, {len(data['surfaces'])} surfaces, {len(data['objects'])} "
              f"stroke objects with {len(data['lines'])} lines, {len(data['notes'])} notes")
    md.append(f"- {len(data['groups'])} connected line groups (G#, listed in brief.json)")
    points = _open_points(data)
    if points:
        ask = recipe.render("open_points", {})
        md += ["", "## Open points the tool noticed", *([ask.rstrip(chr(10))] if ask else []),
               *[f"- {x}" for x in points]]
    if data.get("focus"):
        fc = data["focus"]
        x, y, z = fc["bounds"]["size"]
        md += ["", "## Focus – what to build",
               "The artist selected these parts for this task: "
               + ", ".join(f"**{s['id']}** {s['name']}" for s in fc["selected"])
               + f" ({_ranges_any(fc['ids'])}). **Build only these.** Everything else in the scene is context – fit, "
               "scale, style, connections – not to be rebuilt.",
               f"- focus size: {x:.2f} m (X) × {y:.2f} m (Y) × {z:.2f} m (Z), from {fc['bounds']['min']} to "
               f"{fc['bounds']['max']}",
               "- focus views (highlighted): " + ", ".join(f"`{v}`" for v in views if "focus_" in v)]
    if views:  # live (MCP scene_brief): no rendered files - images are available there via tools
        md += ["", "## Views", *[f"- `{v}`" for v in views if not v.startswith("views/clean/")]]
    if any(v.startswith("views/clean/") for v in views):
        md.append("- the same without labels, caption and scale (for image input or side-by-side comparison): "
                  "`views/clean/*.png`")
    cams = data.get("cameras") or []
    if cams:
        md += ["", "## Hero cameras", ("Standard angles – the artist set none: " if cams[0]["default"] else
                                       "The artist's key angles (screenshots marked as hero camera): ")
               + "use them for look development and image generation and to compare your result from exactly the "
               "same camera. Clean pictures (no labels); the cameras themselves (position, lens, aspect) are in "
               "`scene.usd` under `/sketch/cameras` and in `brief.json`."]
        md += [f"- **{c['name']}**" + (f" (from the board of „{c['from']}“)" if c.get("from") else "")
               + f" `{c['image']}`" + (" – orthographic" if c.get("orthographic_height") else "")
               + (f" · Color-ID `{c['id_image']}`" if c.get("id_image") else "") for c in cams]
        if data.get("id_colors"):
            md += ["", "**Color-ID** (`views/<camera>_ids.png`, same camera and size as the hero picture): every visible "
                   "mesh and surface flat in its own colour, symmetry/array copies in the colour of their original, "
                   "black = nothing or lines only (lines and paint are not in it). No shading, no anti-aliasing – a "
                   "pixel's exact colour value names exactly one part (read the values, e.g. with Python – with many "
                   "parts some colours look alike). Use it to tell which part you see where in the hero picture "
                   "(and in generated look images from that camera). Colour → part → notes that point at it: "
                   "`brief.json` → `id_colors`, per camera `cameras[].id_parts`."]
    if cams and TASK_LOOK in (data.get("task") or {}).get("parts", []):
        md += ["", *_look_md(cams, recipe)]
    md.append("")
    pn = data.get("project_note") or {}
    if pn.get("text") or pn.get("images") or pn.get("drawings"):
        md += [f"## Project note – {pn['title']} (applies to the whole sketch)"]
        for t in [] if pn.get("sections") else pn["text"]:
            md += ["", "> " + t.replace("\n", "\n> ")]
        md += _board_md(pn)
        md.append("")
    if data["notes"]:
        md += ["## Notes (artist intent)"]
        if later:
            ctx += ["## Notes (artist intent)"]
        keep = md
        for n in data["notes"]:
            md = ctx if n.get("later") else keep
            md.append(f"### {n['id']} – {n['title']}")
            for k, a in enumerate(n["anchors"], 1):
                pic = f" – picture `{a['view']}`" + (" (X-ray)" if a.get("xray") else "") if a.get("view") else ""
                if "line" in a:
                    md.append(f"- ({k}) points at line {a['line']} at {a['at']:.0%} of its length"
                              f"{' (line was deleted)' if a.get('lost') else ''}, part of {_target_md(a.get('target'))}"
                              f"{pic}")
                else:
                    md.append(f"- ({k}) points at {_target_md(a.get('target'))} – point {a['point']}{pic}")
                if a.get("members"):
                    md.append("  - also meant: " + ", ".join(_member_md(m) for m in a["members"])
                              + " – marked by the artist, may be incomplete; similar parts are likely meant too")
            for t in [] if n.get("sections") else n["text"]:
                md += ["", "> " + t.replace("\n", "\n> ")]
            md += _board_md(n)
            md.append("")
        md = keep
    if later:
        md += ["", f"Notes of later rounds ({', '.join(sorted(later, key=lambda x: int(x[1:])))}): `context.md`."]
    if data.get("shots"):
        from . import timing_export
        md += timing_export.markdown(data["shots"])
    md += [""] + _components_md(data)
    md += ["", "## Lines"]
    if len(data["lines"]) > LINE_LIST_MAX:
        md.append(f"({len(data['lines'])} lines – listed in brief.json and scene.usd only)")
    else:
        md.append("id | object | shape | length | start → end | notes")
        md.append("---|---|---|---|---|---")
        tiny = [l["id"] for l in data["lines"] if l.get("tiny")]
        for l in data["lines"]:
            if l.get("tiny"):
                continue
            extra = []
            if l.get("joins"):
                extra.append(f"joins {', '.join(l['joins'])}")
            if l.get("axis"):
                extra.append(f"∥{l['axis']}")
            if l.get("corners"):
                extra.append(f"{l['corners']} corners")
            if l.get("plane_normal"):
                extra.append(f"planar n={l['plane_normal']}")
            if l.get("style") == "brush":
                extra.append(f"brush stroke {l['color']}")
            if l.get("paint"):
                extra.append(f"painted{' on ' + l['on'] if l.get('on') else ''} (not a construction line, "
                             f"order {l['paint_order']})")
            md.append(f"{l['id']} | {l['object']} | {l['shape']} | {l['length']} | {l['start']} → {l['end']} | {', '.join(extra)}")
        if tiny:
            md.append(f"({len(tiny)} tiny lines left out – shorter than {max(TINY * data['_size'], 1e-3):.3g} m: "
                      f"{', '.join(tiny)})")
    if data["planes"]:
        md += ["", "## Helper planes (drawing aids)"]
        md += [f"- {p['name']}: origin {p['origin']}, normal {p['normal']}, size {p['size']}" for p in data["planes"]]
    task = data.get("task") or {}
    builds = task.get("kind") != "discuss"
    # details on the core - what always applies stands there; nothing here that contradicts it
    steps = [
        "**Shared names and project file:** keep a short glossary that maps the artist's words for parts to the ids "
        "in this brief (H#, M#, F# …) and clear up words with more than one meaning. Keep it in a file in the project "
        "folder, together with your picture list – one line per reference picture once you opened it in full: the part "
        "ids it is about, what it shows that its caption does not say, what follows for the model – and what you learn about working with the artist.",
        "**Feedback** often comes as several points in one message: add them to the work list and tick off each. "
        "Offer this short form: part · view (camera, state) · kind (design or fault) · goal · what stays fixed."]
    if builds:
        steps[1:1] = [
            "**Matters of taste** (thickness, bulge, how far something reaches): keep them easy to change; a parameter "
            "only where variants or motion need it or the artist asks for one.",
            # check units: by function and shape - examples deliberately from different worlds
            "**Review sheets:** at each agreed review, one sheet per unit that changed: a single picture with the same "
            "labelled views each time – the whole unit, its transitions to its neighbours, one view with wireframe "
            "and, where it moves or deforms, the extremes – saved as `review/<unit>_v<n>.png`; older versions stay. "
            "Between reviews, quick views are enough. Keep `review/review.md`: one row per unit – ids, version, what "
            "changed, a short mesh check (polygon count; open edges where it should be closed, non-manifold edges, "
            "flipped normals), open points.",
            "**Strategy and reviewer:** before a unit or a clearly bounded part of one, write down your own best "
            "strategy, then look up matching entries in the workflow bible, choose and say why. The reviewer checks "
            "that every work-list line of the step can be checked in its view and whether there is a simpler way; "
            "afterwards it checks the result (see the core). If a step fails, work out why and choose a new "
            "strategy; after two failed attempts with similar strategies, change the approach fundamentally, cut "
            "the step differently or ask the artist.",
            f"**Workflow bible:** `{BIBLE.replace(os.sep, '/')}` collects workflows, checks and pitfalls from earlier "
            "Mule projects (protocol at its top). Entries are suggestions, not rules – your own strategy "
            "comes first. Note in the project file which strategies worked or failed; at the end, add what holds in "
            "general (not about this object), signed with your model name, the tools and today's date. If you cannot "
            "open the file, say so and work without it.",
            "**The final check** in detail: the reviewer gets only this brief, its notes and reference pictures, the "
            "work list, fresh renders and the latest sheet of every unit – not your build history. Per unit it "
            "checks **shape** (silhouette, proportions, character of the curves and transitions against sketch, "
            "notes and references, the views the artist named and the hero cameras), **geometry** (no holes where "
            "it should be closed, no flipped or doubled faces, no visible intersections or stray points; density "
            "and scale of detail like its neighbours) and **topology as far as the stage and purpose ask** (edge flow "
            "following the form and – where it bends – the deformation, resolution suited to size and viewing "
            "distance). It also checks every state of moving parts. It lists every difference, every work-list line "
            "not done, every note on the unit's parts that no line covers and every unit without a current sheet."]
    if not builds:  # build brief: the way of working is in the package (conductor.md, rules.md, roles/)
        md += ["", "## How to work with the artist"] + [f"{k}. {s}" for k, s in enumerate(steps, 1)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    cpath = os.path.join(os.path.dirname(path), "context.md")
    if later:
        with open(cpath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(ctx) + "\n")
    elif os.path.isfile(cpath):
        os.remove(cpath)
    return bool(later)


# --- Note boards in sections ------------------------------------------------------------------

SECTION_TEXT_PX = 14  # excerpts so large that the smallest text in them is at least this high (pixels)
SECTION_MAX = 2400  # longest side of an excerpt (pixels)


def _board_sections(scene, board, key, folder):
    """Board into sections (board_links.sections: backdrops, otherwise by proximity): per section a readable image, title
    of the backdrop, texts in reading order, images and the arrows (label or image -> spot, numbered P#
    as in the image). Markings and strokes that cannot be assigned are only counted - better unclear than wrong;
    the image shows them."""
    from PySide6.QtWidgets import QGraphicsScene

    from . import board_links
    from .note_board import TEXT_PX, ImageItem, TextBlock, _populate, render_region

    def end(item):
        if isinstance(item, TextBlock):
            return {"text": item.toPlainText().strip().split("\n")[0].strip()}
        if isinstance(item, ImageItem):
            return {"image": f"notes/{item.file}", "screenshot": bool(item.view)}
        return None

    g = QGraphicsScene()
    _populate(g, scene, board)
    out, num = [], 0
    for k, sec in enumerate(board_links.sections(g), 1):
        texts = [i for i in sec["items"] if isinstance(i, TextBlock)]
        arrows, badges, marks, unclear = [], [], 0, 0
        for _, link in sec["links"]:
            if link["kind"] in ("link", "image"):
                num += 1
                arrow = {"id": f"P{num}", "from": end(link["source"]), "to": end(link["target"])}
                if link["uv"] is not None:
                    arrow["at"] = [round(link["uv"][0], 2), round(link["uv"][1], 2)]
                arrows.append(arrow)
                badges.append((link["tip"], arrow["id"]))
            elif link["kind"] == "mark":
                marks += 1
            else:
                unclear += 1
        rect = sec["rect"].adjusted(-24, -24, 24, 24)
        small = min((t.scale() for t in texts), default=1.0)
        scale = min(max(SECTION_TEXT_PX / (TEXT_PX * small), 0.6), 2.0,
                    SECTION_MAX / max(rect.width(), rect.height()))
        path = f"notes/board_{key}_{k}.png"
        render_region(g, rect, scale, badges).save(os.path.join(folder, path))
        out.append({"image": path, "title": sec["title"], "texts": [t.toPlainText() for t in texts],
                    "images": [f"notes/{i.file}" for i in sec["items"] if isinstance(i, ImageItem)],
                    "arrows": arrows, "marks": marks, "unclear": unclear})
    g.clear()
    return out


# --- Export ---------------------------------------------------------------------------------------------

def _export_videos(scene, folder):
    """Videos of the boards into videos/ (files of the same size already there not again), one overview image each; files
    of earlier exports that no longer occur are removed. The AI reads the images from them only on request."""
    import shutil
    from PySide6.QtWidgets import QApplication

    from . import video
    target, written, keep = os.path.join(folder, video.FOLDER), [], set()
    for path, name in video_names(scene).items():
        if not os.path.isfile(path):
            continue
        os.makedirs(target, exist_ok=True)
        dest = os.path.join(target, name)
        if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(path):
            shutil.copy2(path, dest)
        keep.add(name)
        written.append(f"{video.FOLDER}/{name}")
        sheet = video.overview(path) if QApplication.instance() is not None and video.available() else None
        if sheet is not None:
            name = os.path.splitext(name)[0] + "_overview.jpg"
            sheet.save(os.path.join(target, name), "JPG", 85)
            keep.add(name)
            written.append(f"{video.FOLDER}/{name}")
    if os.path.isdir(target):
        for name in set(os.listdir(target)) - keep:
            os.remove(os.path.join(target, name))
    return written


# MCP tools for the AI in the brief folder: the entries from the repo (.mcp.json for Claude Code, .codex/config.toml
# for Codex) - a session in the brief folder has the same tools, without repo and project memory (blind test)
TOOL_CONFIG_DIR = paths.tool_config_dir()
def _sketch_mcp_url():
    """Live MCP URL (Calisto neighbor default :8772 via SKETCH_MACHINE_MCP_PORT / mcp_server.PORT)."""
    from . import mcp_server
    return f"http://127.0.0.1:{mcp_server.PORT}/mcp"


# Kept for callers that still read the name; prefer _sketch_mcp_url() at write time.
SKETCH_MCP_URL = "http://127.0.0.1:8772/mcp"


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items()) + " }"
    return json.dumps(str(v), ensure_ascii=False)


def _toml_key(k):
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k, ensure_ascii=False)


def _toml_dump(data, prefix=()):
    """Small TOML writer: values first, sub-tables as [a.b] (enough for config.toml)."""
    lines = [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in data.items() if not isinstance(v, dict)]
    for k, v in data.items():
        if isinstance(v, dict):
            path = prefix + (k,)
            if any(not isinstance(x, dict) for x in v.values()) or not v:
                lines += ["", "[" + ".".join(_toml_key(p) for p in path) + "]"]
            lines += _toml_dump(v, path)
    return lines


def write_tool_config(folder, source=None):
    """MCP entries into the brief folder: .mcp.json and .codex/config.toml with the servers from source (repo); the
    Mule server always. What is already there stays - only these servers are set. Returns the files."""
    import tomllib
    source = source or TOOL_CONFIG_DIR

    def read_json(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def read_toml(path):
        try:
            with open(path, "rb") as fh:
                return tomllib.load(fh)
        except (OSError, ValueError):
            return {}

    claude = dict(read_json(os.path.join(source, ".mcp.json")).get("mcpServers") or {})
    url = _sketch_mcp_url()
    claude.setdefault("sketch_machine", {"type": "http", "url": url})
    # Always rewrite sketch_machine URL so a Calisto neighbor port change is not sticky from an old export.
    claude["sketch_machine"] = {"type": "http", "url": url}
    codex = dict(read_toml(os.path.join(source, ".codex", "config.toml")).get("mcp_servers") or {})
    codex["sketch_machine"] = {"url": url}
    path = os.path.join(folder, ".mcp.json")
    data = read_json(path)
    data.setdefault("mcpServers", {}).update(claude)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.makedirs(os.path.join(folder, ".codex"), exist_ok=True)
    path = os.path.join(folder, ".codex", "config.toml")
    data = read_toml(path)
    data.setdefault("mcp_servers", {}).update(codex)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_toml_dump(data)).lstrip("\n") + "\n")
    return [".mcp.json", ".codex/config.toml"]


EXPORT_DIRS = ("views", "notes", "shots")  # written only by the export - emptied beforehand (gen/ etc. stay)


def _clear_old_export(folder):
    """Old export in the same folder: remove its images so that nothing outdated stays behind (a note image or a
    hero camera that no longer exists). Only if a brief really lies there; gen/ (images of the look pre-stage)
    and everything else stay."""
    import shutil
    if not any(os.path.isfile(os.path.join(folder, f)) for f in ("brief.md", "brief.json")):
        return
    for sub in EXPORT_DIRS:
        shutil.rmtree(os.path.join(folder, sub), ignore_errors=True)


def export(scene, viewport, folder, scene_name="sketch", focus=None, data=None, review_round=False):
    """Write the Sketch Brief into folder; returns the list of written files. focus: ids of the selected
    objects - the brief names them as what is to be built (the rest is context). data: analysis already done
    (task dialog) - reused without focus, the task is read anew. review_round: folder is a
    review round (review_round.py) - only review.md (what is new) and the current state, no brief.md, no package."""
    os.makedirs(folder, exist_ok=True)
    _clear_old_export(folder)
    if data is None or focus:
        data = analyse(scene, focus=focus)
    else:
        data.pop("task", None)
        task = task_clean(scene.brief_task)
        if task:
            data["task"] = task
    written = []
    views = render_views(viewport, data, os.path.join(folder, "views")) if viewport is not None else []
    if viewport is not None and data.get("focus"):
        views += render_views(viewport, data, os.path.join(folder, "views"), focus=True)
    written += views
    if viewport is not None:
        written += render_anchor_views(viewport, data, os.path.join(folder, "views"))
        written += render_heroes(viewport, data, os.path.join(folder, "views"))
    note_dir = os.path.join(folder, "notes")
    for n in scene.notes + [scene.global_note]:
        for item in n.board.get("items", []):
            if item.get("kind") == "image":
                blob = scene.asset_bytes(item["file"])
                if blob is not None:
                    os.makedirs(note_dir, exist_ok=True)
                    with open(os.path.join(note_dir, item["file"]), "wb") as fh:
                        fh.write(blob)
                    written.append(f"notes/{item['file']}")
    written += _export_videos(scene, folder)
    from . import timing_export
    if timing_export.has_timing(scene):  # Shots (prototype): cameras, shot notes, flipbook drawings, images
        images, files = timing_export.render_images(scene, viewport, folder)
        boards, more = timing_export.render_boards(scene, folder)
        written += files + more
        data["shots"] = timing_export.brief_data(scene, scene.timing.interp, images, boards)
    else:
        data.pop("shots", None)
    # whole boards as an image (painting over the images is only seen this way by the AI) - needs a running Qt application
    from PySide6.QtWidgets import QApplication
    qt = QApplication.instance() is not None
    if qt:
        from .note_board import render_board
    keys = list(data["_maps"]["nid"].values()) + ["project"]
    for n, entry, key in zip(scene.notes + [scene.global_note], data["notes"] + [data["project_note"]], keys):
        img = render_board(scene, n.board) if entry.get("board") and qt else None
        if img is None:
            entry["board"] = None
            continue
        os.makedirs(note_dir, exist_ok=True)
        img.save(os.path.join(folder, entry["board"]))
        written.append(entry["board"])
        entry["sections"] = _board_sections(scene, n.board, key, folder)
        written += [sec["image"] for sec in entry["sections"]]
    write_usd(scene, data, os.path.join(folder, "scene.usd"))
    if review_round:  # feedback for the running run: no new brief, no rules
        from . import review_round as rr
        public = {k: v for k, v in data.items() if not k.startswith("_")}
        with open(os.path.join(folder, "brief.json"), "w", encoding="utf-8") as fh:
            json.dump(public, fh, indent=1, ensure_ascii=False)
        return [rr.write(public, folder, scene_name), "brief.json", "scene.usd"] + written
    has_context = write_markdown(data, views, os.path.join(folder, "brief.md"), scene_name)
    data.pop("_notice", None)
    if (data.get("task") or {}).get("kind") != "discuss":  # work-method package v4 (rules, roles, hooks, view map)
        from . import brief_v4, recipes
        wanted = (data.get("task") or {}).get("recipe")
        recipe = recipes.get(wanted)
        brief_v4.remove_old_core(folder)
        files, notice = brief_v4.install(folder, scene_name, recipe)
        if wanted and recipe.id != wanted:  # said, never silent (the app asks before; MCP exports read it here)
            notice = (notice + " " if notice else "") + (f"The recipe „{wanted}“ is not installed or not usable – "
                                                         "the brief was made with Mule Default.")
        project = brief_v4.write_project(data, scene_name, folder, has_context, recipe)
        written += files + ([project] if project else [])
        if has_context:
            written.append("context.md")
        if notice:
            data["_notice"] = notice
    clarified = getattr(scene, "brief_clarified", None) or {}
    if clarified.get("summary") or clarified.get("answers"):  # what was clarified with the AI (MCP) still applies
        from . import brief_ai
        brief_ai.write_clarified(folder, clarified.get("summary", ""), [(a["question"], a["answer"])
                                                                      for a in clarified.get("answers", [])],
                                 json_too=False, history=[(a["question"], a["answer"])
                                                          for a in clarified.get("history", [])])
        data["clarified"] = clarified
    public = {k: v for k, v in data.items() if not k.startswith("_")}
    with open(os.path.join(folder, "brief.json"), "w", encoding="utf-8") as fh:
        json.dump(public, fh, indent=1, ensure_ascii=False)
    return ["brief.md", "brief.json", "scene.usd"] + written + write_tool_config(folder)
