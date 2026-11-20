"""Timing mode in the viewport (prototype, data in timing.py, control panels in timing_ui.py).

- Flipbook drawings of all frames live in their own renderer buffer (_flip_draw_data, rebuilt only after update());
  a frame change only selects which ranges are drawn (_flip_frame) – nothing is re-uploaded.
- Looking through a timing camera: the viewport takes on its pose including focal length and jumps along when playing.
  Orbit/pan/zoom only move the view; only ⟲ (change pose) or ➜ (new pose from here) adopt it.
- Camera symbols (pyramids) and viewfinder as QPainter overlay – never in clean images (clean_image).
"""
import hashlib

from .common import *  # noqa: F401,F403
from ..i18n import tr
from ..shortcuts import action_for

timing = timing_mod


class FlipbookSet(TransformSet):
    """Gizmo in timing: the flipbook plane with the drawings of all frames – one fixed object. Pivot and axes are
    those of the plane; scaling acts in its axes on size and drawings alike."""
    scale_axes = (0, 1)

    def __init__(self, plane, curves):
        super().__init__([plane] + list(curves), override=lambda: (plane.origin.copy(), plane.local_axes(), True))
        self.plane, self.curves = plane, list(curves)

    def scale(self, factors, axes, pivot):
        self.plane.scale(factors, axes, pivot)
        for c in self.curves:
            c.scale(factors, axes, pivot)


def _seg_dist(p, a, b):
    """Distance of the image point p to the segment a-b (pixels)."""
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(a + ab * t - p))


SHOT_WIDTH = 420  # camera images on the board this wide (board units)
SHOT_GAP = 30  # distance between the images of a camera (side by side)
SHOT_ROW = 110  # space below an image row (for the settings underneath)


def _shot_scale(width_px):
    return round(min(1.0, SHOT_WIDTH / max(width_px, 1)), 4)


def _shot_entry(cam, pose, file, x, y, scale):
    return {"kind": "image", "x": round(x, 1), "y": round(y, 1), "scale": scale, "file": file,
            "view": timing.pose_view(pose), "shot": {"cam": cam.id, "pose": pose.id}}


def _entry_size(scene, e):
    """(Width, height) of a board entry in board units (rough, for choosing placement)."""
    if e.get("kind") == "image":
        data = scene.asset_bytes(e.get("file", ""))
        img = QImage.fromData(data) if data else None
        w, h = (img.width(), img.height()) if img is not None and not img.isNull() else (160, 110)
        s = float(e.get("scale", 1.0))
        return w * s, h * s + (SHOT_ROW if e.get("shot") else 0)
    if e.get("kind") == "stroke":
        pts = e.get("pts") or [[0, 0]]
        return max(q[0] for q in pts) - min(q[0] for q in pts), max(q[1] for q in pts) - min(q[1] for q in pts)
    if e.get("kind") == "backdrop":
        return float(e.get("w", 0)), float(e.get("h", 0))
    text = e.get("text", "")
    return float(e.get("w", 340)), 30.0 * max(1, text.count("\n") + 1 + len(text) // 40)


def _entry_width(scene, e):
    return _entry_size(scene, e)[0]


def _content_bottom(scene, items):
    """Bottom edge of a board's content (+ gap) – the row of a new camera starts there."""
    bottoms = []
    for e in items:
        if e.get("kind") == "stroke":
            y = min(q[1] for q in (e.get("pts") or [[0, 0]]))
        else:
            y = float(e.get("y", 0))
        bottoms.append(y + _entry_size(scene, e)[1])
    return round(max(bottoms) + 40.0, 1) if bottoms else 0.0


GHOST_ALPHA = 80  # poses that do not apply right now: pale
GATE_DIM = QColor(0, 0, 0, 110)  # viewfinder: darkened outside the image format


class TimingViewMixin:
        # --- Mode ------------------------------------------------------------
    def set_timing_mode(self, on):
        """Fourth basic state timing: paint flipbook planes, set up timing cameras, timeline."""
        on, ws = bool(on), self.ws
        if on == ws.timing_mode:
            return
        if self._mode == "extrude":
            self.cancel_extrude()
        if on:
            if self.notes_mode:
                self._leave_notes()
            ws.select_mode = False
            ws.free_draw = False  # free mode draws real lines – in timing there is only painting
            ws.multi_mode = False
            sc = self.scene
            ws.timing_saved = (sc.active_plane.id, sc.active_mesh.id if sc.active_mesh else None,
                               sc.active_stroke.id if sc.active_stroke else None)
            ws.timing_mode = True
            for vp in ws.viewports:  # gizmo only with T/R/E (the tool from edit stays remembered there)
                vp._edit_tool, vp.gizmo.tool = vp.gizmo.tool, None
            if ws.draw_tool not in ("pen", "brush", "eraser"):
                ws.draw_tool = "brush"
            self._timing_pick_layer()
            self.notify(tr("Shots: Space+N Flipbook · Doppelklick Shot-Notiz · Timeline: Bind/Pfad/Kamera · ←/→ Frame · Alt+A abspielen"))
        else:
            self._leave_timing()
        self.toggles_changed.emit()
        self.ws.timing_changed.emit()
        self._changed()

        # --- Shot notes (like in the Notes tab, own list) ----------------
    def note_list(self):
        """Notes that the viewport shows and edits: in shots the shot notes, otherwise the normal ones."""
        return self.scene.timing.notes if self.ws.timing_mode else self.scene.notes

    @property
    def notes_active(self):
        """Click and edit notes: in notes mode and in shots (there you keep painting next to them)."""
        return self.notes_mode or self.ws.timing_mode

    def _shot_note_hit(self, pos):
        """Shots: does this click go to a shot note (heading, buttons, anchor) instead of painting?"""
        tool = self.notes_tool
        return (tool.editor is not None or (tool.drag is not None and tool.drag["kind"] == "rotate")
                or tool.hit(pos) is not None)

    def _shot_anchor_target(self, pos):
        """Target of a hint line in shots: a stroke of the currently shown drawing of a flipbook plane (ends do not
        snap – the drawing changes, after all), otherwise a point on a flipbook plane, otherwise None."""
        from .. import notes
        from ..scene import NoteAnchor
        sc = self.scene
        planes = {p.id: p for p in timing.flipbook_planes(sc) if not p.hidden}
        items = [(c, c.points) for s in sc.strokes if s.flipbook in planes for c in s.curves
                 if c.frame is not None and len(c.points) >= 2 and self._flip_shown_curve(c)]
        found = self._near_segments(pos, items, PICK_PX) if items else []
        if found:
            _d, curve, i, t = min(found, key=lambda f: f[0])
            point = self._segment_point(pos, curve.points[i], curve.points[i + 1], t)
            return NoteAnchor(np.asarray(point, float), curve.id, notes.curve_param(curve.points, point)[1])
        origin, d = self.ray(pos)
        best = None
        for p in planes.values():
            denom = float(d @ p.normal)
            if abs(denom) < 1e-9:
                continue
            s = float((p.origin - origin) @ p.normal) / denom
            hit = origin + d * s
            rel = hit - p.origin
            if s > 0 and abs(rel @ p.axis_u) <= p.size_u / 2 and abs(rel @ p.axis_v) <= p.size_v / 2:
                if best is None or s < best[0]:
                    best = (s, hit)
        return NoteAnchor(np.asarray(best[1], float)) if best is not None else None

    def _shot_note_key(self, e):
        """Keys for shot notes: Del, F2, R (only over a note – otherwise R rotates the gizmo), running rotation."""
        tool, k = self.notes_tool, e.key()
        rotating = tool.drag is not None and tool.drag["kind"] == "rotate"
        over = tool.hover is not None
        if rotating or k in (Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_F2) or (k == Qt.Key_R and over):
            return tool.key(e)
        return False

    def _leave_timing(self):
        ws, sc = self.ws, self.scene
        for vp in ws.viewports:
            vp.leave_camera()
            vp.gizmo.tool = getattr(vp, "_edit_tool", vp.gizmo.tool)
        ws.timing_mode = False
        ws.playing = False
        saved = getattr(ws, "timing_saved", None) or (None, None, None)
        if sc.active_plane.flipbook:  # the previous drawing surface again (never a flipbook plane)
            plane = sc.plane_by_id(saved[0])
            sc.activate(plane if plane is not None and not plane.flipbook else sc.presets[0])
            mesh = next((m for m in sc.meshes + sc.surfaces if m.id == saved[1]), None)
            if mesh is not None:
                sc.activate_mesh(mesh)
        if sc.active_stroke is not None and sc.active_stroke.flipbook is not None:
            sc.active_stroke = next((s for s in sc.strokes if s.id == saved[2] and s.flipbook is None), None)

    def _timing_pick_layer(self):
        """On entering: if no flipbook plane is active, the first one."""
        sc = self.scene
        if not sc.active_plane.flipbook or sc.active_mesh is not None:
            planes = timing.flipbook_planes(sc)
            if planes:
                self.activate_flipbook(planes[0])

    def activate_flipbook(self, plane):
        sc = self.scene
        sc.activate(plane)
        sc.active_stroke = sc.flip_stroke(plane)
        self.ws.timing_changed.emit()
        self.update_view(all_views=True)

    def _timing_gizmo_target(self):
        """Gizmo target in timing: with T/R/E (tool on) the active flipbook plane together with all drawings – not when
        looking through a camera. Cached until the scene changes (during a drag the target stays)."""
        plane = self._timing_flip_target()
        if self.gizmo.tool is None or self._look is not None or plane is None or plane.hidden:
            return None
        sc = self.scene
        stroke = sc.flip_stroke(plane)
        key = (id(sc), sc.revision, plane.id, id(stroke), len(stroke.curves))
        cached = getattr(self, "_flip_gizmo", None)
        if cached is None or cached[0] != key:
            cached = self._flip_gizmo = (key, FlipbookSet(plane, [c for c in stroke.curves if c.plane_id == plane.id]))
        return cached[1]

    def set_timing_tool(self, tool):
        """T/R/E in timing: gizmo on the active flipbook plane (same key again: off)."""
        if self._timing_flip_target() is None:
            return self.notify(tr("Erst eine Flipbook-Ebene wählen (Name in der Timeline) oder anlegen (Space+N)"))
        self.gizmo.tool = None if self.gizmo.tool == tool else tool
        self.toggles_changed.emit()
        self.update_view(all_views=True)

    def _flip_locked(self, obj):
        """Flipbook plane or its stroke object outside timing: locked (select, delete, rename, gizmo)."""
        flip = (isinstance(obj, Plane) and obj.flipbook) or (isinstance(obj, StrokeObject) and obj.flipbook is not None)
        return flip and not self.ws.timing_mode

    def _timing_allows(self, obj):
        """Flipbook planes (and their stroke object) only in timing – in timing only they."""
        if obj is None:
            return True
        flip = (isinstance(obj, Plane) and obj.flipbook) or (isinstance(obj, StrokeObject) and obj.flipbook is not None)
        return flip if self.ws.timing_mode else not flip

    def _timing_can_draw(self):
        """Left click in timing: drawing only on a flipbook plane (otherwise a hint)."""
        if self.ws.draw_tool == "eraser":
            return True
        surf = self.scene.active_surface
        if not (isinstance(surf, Plane) and surf.flipbook) or surf.hidden:
            self.notify(tr("Shots: gemalt wird nur auf Flipbook-Ebenen – Space+N legt eine an, die Timeline wählt sie"))
            return False
        self.scene.active_stroke = self.scene.flip_stroke(surf)
        return True

        # --- Continue painting -------------------------------------------------
    def _timing_flip_target(self):
        surf = self.scene.active_surface
        return surf if self.ws.timing_mode and isinstance(surf, Plane) and surf.flipbook else None

    def _timing_needs_copy(self, plane):
        """Painting/erasing at a frame without its own drawing: first copy the applicable one here."""
        return self.scene.timing.frame not in timing.flip_keys(self.scene).get(plane.id, [])

    def _timing_copy_key(self, plane):
        """New drawing at the current frame as a copy of the applicable one (before the first: empty) – the caller
        sets the undo step. -> {id(old stroke): copy}"""
        sc, t = self.scene, self.scene.timing.frame
        held = timing.held_key(timing.flip_keys(sc).get(plane.id, []), t)
        mapping = timing.copy_key(sc, plane, held, t) if held is not None else {}
        self.ws.flip_cache = None
        return mapping

    def _timing_erase_targets(self, changed):
        """Eraser in timing: hit strokes of a held drawing belong to an earlier frame – first copy the
        drawing here and erase in the copy (the earlier one stays as it was)."""
        t, out, copies = self.scene.timing.frame, [], {}
        for stroke, curve, runs in changed:
            if curve.frame is not None and curve.frame != t:
                plane = self.scene.plane_by_id(curve.plane_id)
                if plane is None:
                    continue
                if plane.id not in copies:
                    copies[plane.id] = self._timing_copy_key(plane)
                curve = copies[plane.id].get(id(curve))
                if curve is None:
                    continue
            out.append((stroke, curve, runs))
        return out

        # --- Frame filter -----------------------------------------------------
    def _flip_state(self):
        """(Scene, frame, {plane: frames of the drawings}, {plane: applicable drawing}) – determined once per frame."""
        ws, sc = self.ws, self.scene
        t = sc.timing.frame
        c = ws.flip_cache
        if c is None or c[0] is not sc or c[1] != t:
            keys = c[2] if c is not None and c[0] is sc else timing.flip_keys(sc)
            c = ws.flip_cache = (sc, t, keys, {pid: timing.held_key(fr, t) for pid, fr in keys.items()})
        return c

    def _flip_shown_curve(self, curve):
        return self._flip_state()[3].get(curve.plane_id) == curve.frame

    def _flip_draw_data(self):
        """(Entries, {(plane, frame): (first entry, count)}, fingerprint) of all flipbook drawings – sorted by
        plane and frame, so that a frame only draws contiguous ranges of the buffer."""
        sc = self.scene
        planes = {p.id for p in timing.flipbook_planes(sc)}
        if not planes or not self.ws.layers_visible["paint"]:
            return None
        hidden = self._hidden_ids()
        items = [c for s in sc.strokes if not sc.is_hidden(s) for c in s.curves
                 if c.frame is not None and c.plane_id in planes and c.plane_id not in hidden and not c.hidden
                 and len(c.points) >= 2]
        if not items:
            return None
        # depth of the painting by paint order over all flipbook strokes: per plane only one drawing applies
        order = {id(c): (k + 1) / (len(items) + 1) for k, c in enumerate(sorted(items, key=lambda c: (c.seq, c.id)))}
        items.sort(key=lambda c: (c.plane_id, c.frame, c.seq, c.id))
        entries, groups, key = [], {}, []
        for c in items:
            brush = c.style == "brush" and c.color is not None
            color = (*theme.display_rgb(c.color), 1.0) if brush else tuple(QColor(STROKE_COLOR).getRgbF())
            normals = self._paint_normals(c)
            g = groups.setdefault((c.plane_id, int(c.frame)), [len(entries), 0])
            g[1] += 1
            entries.append((c.points, c.widths, color, 2.0 if brush else 1.0, [c.plane_id], (normals, order[id(c)])))
            key.append((c.id, id(c.points), id(c.widths), id(normals), color, int(c.frame), order[id(c)]))
        return entries, {g: tuple(v) for g, v in groups.items()}, tuple(key)

    def _flip_frame(self):
        """For this image: (entries, fingerprint, groups, [(group, opacity)]) – the applicable drawing per
        plane, in timing plus the previous and next one pale (onion skin; not when playing, never in the clean image)."""
        data = getattr(self, "_flip_data", None)
        if not data:
            return None
        entries, groups, key = data
        _, t, keys, shown = self._flip_state()
        onion = (self.ws.timing_mode and self.scene.timing.onion and not self.ws.playing
                 and not getattr(self, "_clean_frame", False))
        jobs = []
        for pid, frames in keys.items():
            if onion:  # previous drawing clearly (what has changed?), next only hinted
                for k, fade in zip(timing.neighbor_keys(frames, t), (timing.ONION_FADE, timing.ONION_NEXT_FADE)):
                    if k is not None and (pid, k) in groups:
                        jobs.append(((pid, k), fade))
            k = shown.get(pid)
            if k is not None and (pid, k) in groups:
                jobs.append(((pid, k), 1.0))
        return entries, key, groups, jobs

        # --- Frames -----------------------------------------------------------
    def set_frame(self, frame):
        """Current frame (all viewports). Only the selection in the flipbook buffer and the timing cameras change –
        no rebuild of the draw lists. False: aborted (keep unsaved camera view)."""
        tm = self.scene.timing
        frame = tm.clamp(frame)
        if frame == tm.frame:
            return True
        for vp in self.ws.viewports:
            if vp._look is not None and vp.look_modified() and not vp._confirm_discard():
                return False
        tm.frame = frame
        for vp in self.ws.viewports:
            if vp._look is not None:
                vp._apply_look()
            vp._frame_painted = False
            QOpenGLWidget.update(vp)
        self.ws.frame_changed.emit(frame)
        return True

    def step_frame(self, delta):
        return self.set_frame(self.scene.timing.frame + int(delta))

    def jump_key(self, direction):
        """↑/↓: to the next/previous drawing of the active flipbook plane (without an active one: of all planes)."""
        sc, t = self.scene, self.scene.timing.frame
        keys = self._flip_state()[2]
        plane = sc.active_plane if sc.active_plane.flipbook else None
        frames = sorted(set(keys.get(plane.id, [])) if plane is not None else {f for fr in keys.values() for f in fr})
        target = (next((f for f in frames if f > t), None) if direction > 0
                  else next((f for f in reversed(frames) if f < t), None))
        if target is not None:
            self.set_frame(target)

        # --- Looking through a timing camera --------------------------------
    def _look_cam(self):
        return self.scene.timing.camera_by_id(self._look) if getattr(self, "_look", None) is not None else None

    def look_through(self, cam_id):
        """Double click on a camera: the viewport takes on its pose at the current frame including focal length."""
        cam = self.scene.timing.camera_by_id(cam_id)
        if cam is None:
            return
        if self._look is not None and self._look != cam_id and self.look_modified() and not self._confirm_discard():
            return
        if self._flight is not None:
            self._stop_flight()
        self._look = cam.id
        self.ws.timing_cam = cam.id
        self._apply_look()
        self.toggles_changed.emit()
        self.ws.timing_changed.emit()
        self.update_view()

    def _apply_look(self):
        cam = self._look_cam()
        if cam is None:
            self._look, self._cam_tan = None, None
            return
        pose = timing.pose_at(cam, self.scene.timing.frame)
        self.cam.ortho = False
        self._flight_ortho = None
        self.cam.set_state(*pose.state())
        self._cam_tan = timing.mm_to_tan(pose.mm)

    def look_modified(self):
        """View moved in the camera but not yet adopted?"""
        cam = self._look_cam()
        if cam is None:
            return False
        return not timing.pose_at(cam, self.scene.timing.frame).matches(self.cam.state())

    def _confirm_discard(self):
        """Discard unsaved camera view? (without asking when playing)"""
        if self.ws.playing:
            return True
        from PySide6.QtWidgets import QMessageBox
        cam = self._look_cam()
        answer = QMessageBox.question(
            self, tr("Ansicht geändert"),
            tr("Die Ansicht in {0} wurde bewegt, aber nicht übernommen. Verwerfen?").format(cam.name if cam else ""),
            QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        return answer == QMessageBox.Discard

    def leave_camera(self):
        """Esc: out of the camera – the viewport keeps the position, the focal length becomes the fixed one again."""
        if getattr(self, "_look", None) is None:
            return
        self._look, self._cam_tan = None, None
        self.toggles_changed.emit()
        self.ws.timing_changed.emit()
        self.update_view()

    def discard_view(self):
        """Discard: back to the pose."""
        if self._look is not None:
            self._apply_look()
            self.update_view()

    def commit_pose(self, new, cam=None, pose=None):
        """⟲ (new=False): the pose (at a camera image; otherwise the applicable one) gets the view through which you
        look – no animation. ➜ (new=True): new pose from the current frame (move), with its own image on the board.
        Only from inside, i.e. when looking through this camera (in any viewport)."""
        from ..note_board import shot_edit, shot_image
        cam = cam or self._look_cam()
        view = next((v for v in self.ws.viewports if cam is not None and v._look == cam.id), None)
        if view is None:
            return self.notify(tr("Erst durch die Kamera schauen (Doppelklick aufs Bild), dann die Stellung setzen"))
        tm = self.scene.timing
        frame, state = tm.frame, view.cam.state()
        note = tm.note_of(cam)

        def edit(items):
            if new:
                p = timing.add_pose(cam, frame, state)
            else:
                p = timing.change_pose(cam, frame, state, pose)
            name, w, h = shot_image(self, cam, p)
            refs = [k for k, e in enumerate(items) if timing.shot_ref(e) == (cam.id, p.id)]
            if refs:  # image of this pose anew
                for k in refs:
                    items[k] = {**items[k], "file": name, "view": timing.pose_view(p)}
            else:  # new pose: image to the right of the camera's last image
                own = [e for e in items if (timing.shot_ref(e) or (None,))[0] == cam.id]
                last = max(own, key=lambda e: float(e.get("x", 0))) if own else None
                scale = _shot_scale(w)
                x = float(last["x"]) + _entry_width(self.scene, last) + SHOT_GAP if last else 0.0
                y = float(last["y"]) if last else _content_bottom(self.scene, items)
                items.append(_shot_entry(cam, p, name, x, y, scale))
            return p

        p = shot_edit(self, note.id, edit)
        view._apply_look()
        self.notify(tr("{0}: neue Stellung ab {1}").format(cam.name, timing.frame_label(p.frame)) if new
                    else tr("{0}: Stellung ab {1} geändert").format(cam.name, timing.frame_label(p.frame)))
        self._timing_changed()

    def begin_focal(self, cam):
        """Focal length slider touched: one undo step (open boards adopt their state beforehand)."""
        from ..note_board import open_boards
        note = self.scene.timing.note_of(cam)
        for b in open_boards(self, note.id) if note is not None else []:
            b.commit()
        self.scene.checkpoint()

    def set_cam_focal(self, cam, mm, keep_framing=True, pose=None):
        """Focal length slider: changes the pose (otherwise the one applicable at this frame), live in the viewport when looking through
        this camera at this pose. Undo step: begin_focal; the image on the board is renewed by end_focal."""
        pose = pose or timing.pose_at(cam, self.scene.timing.frame)
        old_tan = timing.mm_to_tan(pose.mm)
        timing.set_focal(pose, mm, keep_framing)
        for vp in self.ws.viewports:
            if vp._look == cam.id and timing.pose_at(cam, self.scene.timing.frame) is pose:
                if keep_framing:
                    vp.cam.dist = vp.cam.dist * old_tan / timing.mm_to_tan(pose.mm)
                vp._cam_tan = timing.mm_to_tan(pose.mm)
                vp.update_view()
        self._timing_changed(full=False)

    def end_focal(self, cam, pose):
        """Slider released: the image of this pose on the board shows the new focal length (belongs to the same step)."""
        self._refresh_pose_image(cam, pose)

    def _refresh_pose_image(self, cam, pose, checkpoint=False):
        from ..note_board import shot_edit, shot_image
        note = self.scene.timing.note_of(cam)
        if note is None or pose not in cam.poses:
            return

        def edit(items):
            name, _w, _h = shot_image(self, cam, pose)
            for k, e in enumerate(items):
                if timing.shot_ref(e) == (cam.id, pose.id):
                    items[k] = {**e, "file": name, "view": timing.pose_view(pose)}
        shot_edit(self, note.id, edit, checkpoint=checkpoint)
        self._timing_changed()

    def set_cam_props(self, cam, **props):
        """Name, color or symbol size of a camera (one undo step)."""
        from ..note_board import open_boards
        note = self.scene.timing.note_of(cam)
        boards = open_boards(self, note.id) if note is not None else []
        for b in boards:
            b.commit()
        self.scene.checkpoint()
        for key, value in props.items():
            setattr(cam, key, value)
        for b in boards:
            b.refresh_shot_rows()
        self._timing_changed()

    def move_pose(self, cam, pose, frame):
        """Change the frame of a pose (right click on "from F…" or drag in the timeline). If a
        pose of this camera already lies there: not moved. The image then shows the drawings of the new frame."""
        frame = self.scene.timing.clamp(frame)
        if any(p.frame == frame and p is not pose for p in cam.poses):
            return self.notify(tr("{0} hat bei {1} schon eine Stellung – nicht verschoben").format(
                cam.name, timing.frame_label(frame)))
        from ..note_board import open_boards
        note = self.scene.timing.note_of(cam)
        for b in open_boards(self, note.id) if note is not None else []:
            b.commit()
        self.scene.checkpoint()
        pose.frame = int(frame)
        cam.poses.sort(key=lambda q: q.frame)
        for v in self.ws.viewports:
            if v._look == cam.id:
                v._apply_look()
        self._refresh_pose_image(cam, pose)

        # --- Create cameras, images -----------------------------------------
    def new_timing_camera(self, note):
        """Camera button on the board of a shot note (or on the note in the viewport): always a new camera in this
        note, from what you currently see (free: with the viewport focal length; in a camera: as a variant of it),
        its first pose as an image on the board (below the previous content)."""
        from ..note_board import shot_edit, shot_image
        if note is None:
            return None
        tm = self.scene.timing
        inside = self._look_cam()
        mm = timing.pose_at(inside, tm.frame).mm if inside is not None else timing.tan_to_mm(math.tan(FOV / 2))
        if self.cam.ortho:
            self.cam.ortho = False
        state = self.cam.state()

        def edit(items):
            cam = timing.new_camera(timing.unique_cam_name(tm), tm.frame, state, mm, note=note.id)
            if inside is not None:
                cam.color, cam.size = inside.color, inside.size
            tm.cams.append(cam)
            name, w, _h = shot_image(self, cam, cam.poses[0])
            left = min((float(e.get("x", 0)) for e in items), default=0.0)
            items.append(_shot_entry(cam, cam.poses[0], name, left, _content_bottom(self.scene, items), _shot_scale(w)))
            return cam

        cam = shot_edit(self, note.id, edit)
        self.ws.timing_cam = cam.id
        self.ws.active_note = note.id
        self.notify(tr("{0} angelegt in „{1}“ – Bild im Brett, darunter ihre Einstellungen").format(cam.name, note.title or "Global"))
        self._timing_changed()
        return cam

    def camera_jpeg(self, cam, frame=None, pose=None):
        """JPEG bytes through the camera at frame (image format GATE, without controls); pose: this pose instead of the
        one applicable at frame."""
        from ..note_board import snapshot
        tm = self.scene.timing
        frame = tm.frame if frame is None else frame
        pose = pose or timing.pose_at(cam, frame)
        saved = (self.cam.state(), self.cam.ortho, self._cam_tan, tm.frame)
        try:
            tm.frame = frame
            self.ws.flip_cache = None
            self.cam.set_state(*pose.state())
            self.cam.ortho = False
            self._cam_tan, self._cam_exact = timing.mm_to_tan(pose.mm), True
            return snapshot(self, aspect=timing.GATE)[0]
        finally:
            self._cam_exact = False
            state, self.cam.ortho, self._cam_tan, tm.frame = saved
            self.cam.set_state(*state)
            self.ws.flip_cache = None
            self.update_view()

    def _timing_changed(self, full=True):
        """Timing data changed: update panel/timeline, redraw viewports."""
        self.scene.dirty = True
        self.ws.timing_changed.emit()
        if full:
            self._changed()
        else:
            self.update_view(all_views=True)

        # --- Keys -----------------------------------------------------------
    def _timing_key(self, e):
        """Keys in timing. True: handled."""
        k, mods = e.key(), e.modifiers()
        if k == Qt.Key_Escape:
            if self.gizmo.tool is not None:  # first remove the gizmo, then out of the camera, then deselect the note
                self.gizmo.tool = None
                self.toggles_changed.emit()
                self.update_view(all_views=True)
            elif self._look is not None:
                self.leave_camera()
            else:
                self.notes_tool.key(e)
            return True
        if not mods and (tool := action_for(k)):
            self.set_timing_tool(tool)
            return True
        if k in (Qt.Key_Tab, Qt.Key_Backtab):  # as from the notes: back to drawing
            self.set_select_mode(False)
            return True
        if k in (Qt.Key_Left, Qt.Key_Right):
            step = -1 if k == Qt.Key_Left else 1
            if mods & Qt.ShiftModifier:
                tm = self.scene.timing
                self.set_frame(tm.start if step < 0 else tm.end)
            else:
                self.step_frame(step)
            return True
        if k in (Qt.Key_Up, Qt.Key_Down):
            self.jump_key(1 if k == Qt.Key_Up else -1)
            return True
        if k == Qt.Key_A and mods & Qt.AltModifier:
            self.ws.play_requested = not self.ws.playing  # the timeline (timing_ui) plays
            self.ws.timing_changed.emit()
            return True
        if k == Qt.Key_O and not mods:
            tm = self.scene.timing
            tm.onion = not tm.onion
            self.notify(tr("Onion-Skin an") if tm.onion else tr("Onion-Skin aus"))
            self.ws.timing_changed.emit()
            self.update_view(all_views=True)
            return True
        return False

        # --- Clicking cameras in the viewport -----------------------------------
    def _cam_under(self, pos, tol=7.0):
        """(Camera, pose) of the camera symbol under the cursor (edges of the pyramid, also pale poses) or
        None – the nearest wins."""
        tm = self.scene.timing
        p = np.array([pos.x(), pos.y()], float)
        best, best_d = None, tol
        view_eye = self.cam.eye
        for _, cam in tm.cameras():
            if cam.id == self._look:
                continue
            for pose in cam.poses:
                size = timing.base_size(pose) * tm.cam_size * cam.size
                eye, corners = timing.frustum(pose, size)
                if float(np.linalg.norm(eye - view_eye)) < size * 1.5:
                    continue
                pts = np.array([eye] + corners)
                screen, ok = self.project(pts)
                segs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
                for a, b in segs:
                    if not (ok[a] and ok[b]):
                        continue
                    d = _seg_dist(p, screen[a], screen[b])
                    if d < best_d:
                        best, best_d = (cam, pose), d
        return best

    def _pick_cam(self, cam, pose):
        """Click on the camera symbol: select camera (and its note); a pale pose jumps to its frame."""
        tm = self.scene.timing
        self.ws.timing_cam = cam.id
        note = tm.note_of(cam)
        if note is not None:
            self.ws.active_note = note.id  # the Shots panel shows its note
        if pose is not timing.pose_at(cam, tm.frame):
            self.set_frame(pose.frame)
        self.ws.timing_changed.emit()
        self.update_view(all_views=True)

        # --- Overlay: camera symbols and viewfinder --------------------------------
    def _draw_timing(self, p):
        tm = self.scene.timing
        t = tm.frame
        for note, cam in tm.cameras():
            if cam.id == self._look:
                continue  # the one you look through does not show itself
            self._draw_cam_symbol(p, cam, t, tm)
        self._draw_motions(p, tm, t)
        if self._look is not None:
            self._draw_viewfinder(p)

    def _draw_motions(self, p, tm, t):
        """Overlay: path polyline + proxy marker for motion beats covering this frame (planning, not bake)."""
        active = tm.active_beat()
        beats = list(tm.motion_at(t))
        if active is not None and active not in beats:
            beats.append(active)
        for beat in beats:
            pts = timing.path_points(self.scene, beat, frame=t)
            chosen = tm.active_motion == beat.id
            col = QColor(208, 138, 94) if chosen else QColor(94, 148, 168)
            if len(pts) >= 2:
                c = QColor(col)
                c.setAlpha(200 if chosen or beat.covers(t) else 90)
                p.setPen(QPen(c, 2.0 if chosen else 1.3, Qt.DashLine))
                self.draw_polyline(p, pts)
            pos = timing.proxy_at(self.scene, beat, t) if beat.covers(t) else None
            if pos is None and len(pts):
                pos = pts[0]
            if pos is not None and beat.covers(t):
                size = max(float(np.linalg.norm(self.cam.target - self.cam.eye)) * 0.02, 0.05)
                diamond = np.array([pos + (size, 0, 0), pos + (0, size, 0), pos + (-size, 0, 0),
                                    pos + (0, -size, 0), pos + (size, 0, 0)])
                p.setPen(QPen(col, 2.0))
                self.draw_polyline(p, diamond)
                screen, ok = self.project(np.asarray(pos, float)[None])
                if ok[0]:
                    label = beat.annotation or timing.target_label(self.scene, beat)
                    if beat.text:
                        label = f"{label} — {beat.text[:40]}"
                    self._label_at(p, screen[0], label, col)
            # Stored planning camera: pale frustum at the beat start
            state = timing.camera_state(beat)
            if state is not None and (chosen or beat.covers(t)):
                target, dist, rot, mm = state
                pose = timing.Pose(beat.start, target, dist, rot, mm)
                size = timing.base_size(pose) * tm.cam_size * 0.7
                eye, corners = timing.frustum(pose, size)
                c = QColor(col)
                c.setAlpha(110)
                p.setPen(QPen(c, 1.2, Qt.DotLine))
                for corner in corners:
                    self.draw_polyline(p, np.array([eye, corner]))
                self.draw_polyline(p, np.array(corners + corners[:1]))

    def _draw_cam_symbol(self, p, cam, t, tm):
        col = QColor.fromRgbF(*cam.rgb)
        chosen = self.ws.timing_cam == cam.id
        current = timing.pose_at(cam, t)
        eyes = []
        view_eye = self.cam.eye
        for pose in cam.poses:
            size = timing.base_size(pose) * tm.cam_size * cam.size
            eye, corners = timing.frustum(pose, size)
            eyes.append(eye)
            if float(np.linalg.norm(eye - view_eye)) < size * 1.5:  # you stand exactly there (e.g. just left it)
                continue
            c = QColor(col)
            if pose is not current:
                c.setAlpha(GHOST_ALPHA)
            p.setPen(QPen(c, (2.4 if chosen else 1.5) if pose is current else 1.0))
            for corner in corners:
                self.draw_polyline(p, np.array([eye, corner]))
            self.draw_polyline(p, np.array(corners + corners[:1]))
            up = np.asarray(pose.rot, float)[:, 1]
            top = (corners[2] + corners[3]) / 2  # small triangle at the top: where up is
            self.draw_polyline(p, np.array([corners[3], top + up * size * 0.25, corners[2]]))
        if len(eyes) > 1:  # intended move: dashed (Mule does not move the camera along there)
            c = QColor(col)
            c.setAlpha(150)
            p.setPen(QPen(c, 1.2, Qt.DashLine))
            self.draw_polyline(p, np.array(eyes))
        labels = [(timing.pose_at(cam, t) is pose, pose, eye) for pose, eye in zip(cam.poses, eyes)]
        for is_cur, pose, eye in labels:
            screen, ok = self.project(np.asarray(eye, float)[None])
            if not ok[0]:
                continue
            text = cam.name + ("" if not timing.is_move(cam) else f" · {timing.frame_label(pose.frame)}")
            if not is_cur and not timing.is_move(cam):
                continue
            self._label_at(p, screen[0], text, col if is_cur else QColor(col.red(), col.green(), col.blue(), 120))

    def _label_at(self, p, at, text, color):
        font = QFont(self.font())
        font.setPointSize(8)
        w = QFontMetricsF(font).horizontalAdvance(text) + 8

        def draw(q):
            q.setFont(font)
            q.setPen(color)
            q.drawText(QRectF(0, 0, w, 16), Qt.AlignLeft | Qt.AlignVCenter, text)
        self._raster(p, at[0] + 6, at[1] - 18, w, 16, draw)

    def _draw_viewfinder(self, p):
        """Viewfinder: darkened outside the image format, frame in camera color, tag with name, focal length, frame."""
        cam = self._look_cam()
        if cam is None:
            return
        tm = self.scene.timing
        pose = timing.pose_at(cam, tm.frame)
        _, (x, y, gw, gh) = timing.gate_fit(self._cam_tan, self.width(), self.height())
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(GATE_DIM)
        for r in (QRectF(0, 0, w, y), QRectF(0, y + gh, w, h - y - gh), QRectF(0, y, x, gh), QRectF(x + gw, y, w - x - gw, gh)):
            if r.width() > 0.5 and r.height() > 0.5:
                p.drawRect(r)
        col = QColor.fromRgbF(*cam.rgb)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(col, 3))
        p.drawRect(QRectF(x + 1.5, y + 1.5, gw - 3, gh - 3))
        text = f"{cam.name} · {pose.mm:.0f} mm · {timing.frame_label(tm.frame)}"
        if timing.is_move(cam):
            text += "  " + tr("(Stellung ab {0})").format(timing.frame_label(pose.frame))
        modified = self.look_modified()
        hint = tr("Ansicht geändert: ⟲ Stellung ändern · ➜ neue Stellung ab hier · Esc verlässt die Kamera") if modified else ""
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        fm = QFontMetricsF(font)
        bw = max(fm.horizontalAdvance(text), fm.horizontalAdvance(hint) if hint else 0) + 16
        bh = 22 + (18 if hint else 0)

        def draw(q):
            q.setPen(Qt.NoPen)
            q.setBrush(QColor(col.red(), col.green(), col.blue(), 230))
            q.drawRoundedRect(QRectF(0, 0, bw, bh), 4, 4)
            q.setFont(font)
            q.setPen(QColor(255, 255, 255))
            q.drawText(QRectF(8, 0, bw - 8, 22), Qt.AlignLeft | Qt.AlignVCenter, text)
            if hint:
                f2 = QFont(font)
                f2.setBold(False)
                f2.setPointSize(8)
                q.setFont(f2)
                q.drawText(QRectF(8, 20, bw - 8, 18), Qt.AlignLeft | Qt.AlignVCenter, hint)
        self._raster(p, x + 8, y + 8, bw, bh, draw)
