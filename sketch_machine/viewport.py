"""3D viewport: OpenGL for meshes and planes, QPainter overlay for strokes and gizmos.

Navigation like Houdini, axis gizmo like Blender. Drawing happens on the active
drawing surface: a plane (analytic ray intersection) or a mesh (GPU picking).
Several viewports share one workspace (scene, selection, settings); each has
its own camera, perspective or orthographic (axis views).

Structure: this file holds state, Qt events (mouse, keys, tablet), paintGL and the status line. The remaining
tasks live as mixins in sketch_machine/vp/ (shared imports, constants and helpers in vp/common.py):
  selection       selection, selection undo, pivot, rectangle, loop and connected selection
  picking         projection, rays, what lies under the cursor
  component_edit  edit mode points/edges, bevel
  free_draw       free mode (Q)
  tools           extrude, connect
  point_tools     hardness (K), slide points (S)
  snapping        snapping
  cleanup_ops     weld, fuse, cleanup
  surface_ops     loft, curve surface, fill, split; delete, rename, planes
  navigation      camera: fly, focus, views, new plane
  visibility      drawing aids, X-ray, plane switches, hide, paint mode
  drawing         pen, smoothing, pressure, tool choice, straight lines and shapes
  edit_ops        undo/redo, set scene, copy/paste, group
  eraser          eraser
  overlay         QPainter overlay: gizmos, light buttons, markers
  display_lists   lists for the renderer: strokes, surfaces, planes, meshes
  pie             hold-to-invoke radial menus (transform tools, selection kinds)
"""
from .vp.common import *  # noqa: F401,F403
from .vp.selection import SelectionMixin
from .vp.picking import PickingMixin
from .vp.component_edit import ComponentEditMixin
from .vp.free_draw import FreeDrawMixin
from .vp.tools import ToolsMixin
from .vp.point_tools import PointToolsMixin
from .vp.snapping import SnapMixin
from .vp.cleanup_ops import CleanupMixin
from .vp.surface_ops import SurfaceOpsMixin
from .vp.navigation import NavigationMixin
from .vp.visibility import VisibilityMixin
from .vp.drawing import DrawingMixin
from .vp.edit_ops import EditOpsMixin
from .vp.eraser import EraserMixin
from .vp.overlay import OverlayMixin
from .vp.display_lists import DisplayListMixin
from .vp.nulls import NullMixin
from .vp.timing_view import TimingViewMixin
from .vp.pie import PieMixin
from . import i18n, pictures, shortcuts
from .i18n import tr

NAV_MODES = ("orbit", "pan", "dolly")  # mouse camera modes: change only the view, never the scene


class Viewport(SelectionMixin, PickingMixin, ComponentEditMixin, FreeDrawMixin, ToolsMixin, PointToolsMixin, SnapMixin,
               CleanupMixin,
               SurfaceOpsMixin, NavigationMixin, VisibilityMixin, DrawingMixin, EditOpsMixin, EraserMixin,
               OverlayMixin, DisplayListMixin, NullMixin, TimingViewMixin, PieMixin, QOpenGLWidget):
    def __init__(self, workspace=None):
        super().__init__()
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)  # allows line widths > 1
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)  # the renderer does MSAA itself, Qt does not pass it through here
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # State shared by all viewports (scene, selection, pen, modes), see workspace.py
        self.ws = workspace if workspace is not None else Workspace()
        self.ws.viewports.append(self)
        if self.ws.current is None:
            self.ws.current = self
        self.status_changed = self.ws.status_changed
        self.scene_changed = self.ws.scene_changed
        self.toggles_changed = self.ws.toggles_changed
        self.area = None  # ViewArea that shares/closes this viewport (None: single viewport)
        self.notes_tool = NotesTool(self)  # notes mode (N)

        self.fov = FOV
        self.cam = Camera()
        self.gizmo = TransformGizmo(self)
        self.renderer = None
        self._gl_error = None
        self._pen_feedback_until = 0.0
        self._feedback_depth = 10.0  # depth under the cursor when adjusting the pen size (world units)
        self._tip = None  # smoothed pen tip (screen) while drawing
        self._erase_filter = None  # eraser: None = everything, "lines" (Ctrl: no painting)
        self._erase_whole = False  # Shift when starting: delete touched lines entirely
        self._erase_checkpointed = False
        self._erase_last = None
        self._select_hover = None  # (ref, points) of the edge/point under the cursor in S mode
        self._knot_cache = None  # (key, [(ref, line, node)]) of all points for picking
        self._extrude_undo = None  # (sub-edges, edges) before splitting for X - Esc restores them
        self._bevel = None  # running rounding: dict(op, start, depth, size)
        self._free_saved = None  # free stroke running: (active plane, active mesh, multi) - back afterwards
        self._single_saved = False  # multi with only one surface: like isolation during the stroke
        self._free_hover = None  # free mode: start point under the cursor (3D) or None
        self._pressure = None  # pen pressure (0..1) at the last tablet event, None = mouse
        self._pressure_at = 0.0
        self._pressure_trace = []  # pressure curve of the running stroke (for the stroke width)
        self._free_start = None  # free stroke: (start point, on a line?) - a node arises there
        self._band_end = None  # corner of the selection rectangle while dragging
        self._extrude = None  # running drag: dict(curve, origin, axis, axis_name, start_t, length)
        self._val = None  # running value tool (K hardness, J smoothing)
        self._slide = None  # running slide (S held)
        self._slide_key = False  # S held: points slide along their line
        self._fill_last = (None, 0)  # (most recently double-clicked line, loop index) for cycling through

        self.current = None  # points of the stroke currently being drawn
        self._segments = []  # multi mode: already finished pieces [(points, surface)]
        self._current_surface = None  # multi mode: surface of the running piece
        self._straight = False
        self._shape = None  # recognized shape (holding still at the stroke end), readjusted until release
        self._penetrate = False  # P held: straight line on mesh as a real 3D line
        self._ctrl = False  # Ctrl held: gizmo and straight lines snap
        self._shift = False  # with Ctrl: fine steps (tenths)
        self._free_move = False  # Alt mid-drag: move without welding
        self._drag_start = None  # (grip, start point, pivot) of the running gizmo drag
        self._vsnap = False  # V held: pivot snaps to points in axis mode
        self._snap_dot = None  # 3D point where the pen would snap right now (green dot)
        self._arc_end_snap = None  # (point, on_line): snapped end of a recognized arc
        self._last_extrude = None  # ({ids of the copy}, direction): X on the copy continues the same pull
        self._paste_restore = None  # (tool before pasting,): gizmo after pasting only temporarily
        self._dup_last = None  # (selection key, center when pasting) of the most recently pasted
        self._dup_step = None  # offset for repeating duplicate
        self._draw_snaps = []  # [(point, on_line)] snapped start/end of the running stroke
        self._draw_ended = False  # stroke has left the mesh
        self._miss_start = None  # cursor position since which the stroke misses the surface
        self._hold_anchor = None
        self._hold_timer = QTimer(self, singleShot=True, interval=HOLD_MS)
        self._hold_timer.timeout.connect(self._straighten)

        self._last_pos = None
        self._press_pos = None
        self._press_hit = None
        self._press_how = "replace"  # selection on click: replace | add (Shift) | remove (Ctrl)
        self._mode = None  # pending | draw | orbit | pan | dolly | plane | gizmo | transform
        self._space = False
        self._gizmo_hover = None

        self._flight = None  # (start_state, end_state, start_time)
        # Timing (vp/timing_view.py): look through this camera (id), its field of view; flipbook buffer of this viewport
        self._look, self._cam_tan, self._cam_exact = None, None, False
        self._flip_data, self._frame_painted = None, True
        self._flight_ortho = None  # at the end of the flight orthographic (True) / perspective (False)
        self._persp_view = None  # free view before the jump into an axis view
        self._orbit_pivot = None  # point under the cursor at the start of the orbit
        self._orbit_alt_snap = False  # Blender-style Alt axis snap while orbiting
        self._orbit_snap_axis = None
        self._flight_timer = QTimer(self, interval=15)
        self._flight_timer.timeout.connect(self._step_flight)
        self._init_pie()

    def tabletEvent(self, e):
        """Graphics tablet (Wacom): only remember the pressure; do not accept the event - Qt turns it into the
        usual mouse events, all tools work as with the mouse."""
        self._pressure = float(e.pressure())
        self._pressure_at = time.monotonic()
        e.ignore()

    def event(self, e):
        """Alt+letter belongs to the viewport (Alt+F curve surface, Alt+D split loft, Alt+A axis mode), not to the
        menu shortcuts of the menu bar (Alt+D = File ...). After every input: selection change as an undo step."""
        if e.type() == QEvent.ShortcutOverride and e.modifiers() & Qt.AltModifier and \
                e.key() in (Qt.Key_F, Qt.Key_D, Qt.Key_A, Qt.Key_H, Qt.Key_K, Qt.Key_G, Qt.Key_C):
            e.accept()
            return True
        result = super().event(e)
        if e.type() in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.KeyPress, QEvent.KeyRelease):
            self._commit_selection()
        return result

    def focusNextPrevChild(self, forward):
        return False  # Tab switches the mode, not the focus

    def update(self):
        """Redraw - in all viewports, since all show the same scene. Something has changed: the
        draw lists (strokes, surfaces) are rebuilt."""
        self.ws.flip_cache = None
        for vp in self.ws.viewports:
            vp._lists_dirty = True
            vp._lofts_fresh = None  # reconcile surfaces on the next frame (unless _changed just did it)
            QOpenGLWidget.update(vp)

    def update_live(self):
        """Redraw, only the stroke currently being created has changed: lines and surfaces of the scene stay
        as in the last frame (no surface reconciliation, no re-upload of all strokes) - only the live part is
        rebuilt (_live_draw_list). If the scene does change in the process, it calls update() as usual."""
        for vp in self.ws.viewports:
            QOpenGLWidget.update(vp)

    def update_hover(self):
        """Redraw, only the highlight under the mouse has changed (_draw_lists builds the stroke list alone)."""
        self._hover_dirty = True
        QOpenGLWidget.update(self)

    def update_view(self, all_views=False):
        """Redraw, only the camera of this viewport has moved: strokes and surfaces stay as they are
        (draw lists from the last frame) - as in Blender/Houdini, rotating then costs almost only the GPU.
        all_views: also the other viewports (an overlay like a note has changed)."""
        for vp in (self.ws.viewports if all_views else [self]):
            QOpenGLWidget.update(vp)

    def grabFramebuffer(self):
        self._lists_dirty = True  # image for saving/checking: always from the current state
        return super().grabFramebuffer()

    def clean_image(self, select=(), lines=(), aspect=None, look=None, faint=(), faint_lines=(), ids=False):
        """Image of the view as it currently looks (light, X-ray, planes), but without controls: no gizmo,
        no selection, no notes, no drawing aids (planes, helper surfaces, axis crosses), no bars (camera button of the notes). select: these
        objects (ids) highlighted as selected, lines: these lines colored like selected edges (Sketch Brief: the
        object or line a note points at), faint/faint_lines: likewise, only weaker (the arms of the
        note octopus: parts also meant). aspect: aspect ratio (width/height) of the image - the same
        vertical field of view as a viewport of this format, without changing the window (wider: field of view
        widened and cropped top/bottom, narrower: cropped left/right). look: display for this
        image only - {"xray": opacity or None, "wire": opacity or None, "sun": (azimuth, height) = light on}.
        ids: (image, color ID) - plus per pixel the id of the frontmost mesh/surface (renderer.id_map, -1 =
        nothing), row 0 at the top, cropped the same way."""
        ws = self.ws
        if look:
            shown = (ws.xray, ws.xray_opacity, ws.wireframe, ws.wire_opacity, ws.light_on, self.scene.sun)
            ws.xray, ws.wireframe = bool(look.get("xray")), bool(look.get("wire"))
            ws.xray_opacity = float(look.get("xray") or ws.xray_opacity)
            ws.wire_opacity = float(look.get("wire") or ws.wire_opacity)
            if look.get("sun") is not None:
                ws.light_on, self.scene.sun = True, tuple(float(x) for x in look["sun"])
        wide = aspect is not None and aspect > self.width() / max(self.height(), 1)
        self._fov_zoom = aspect * self.height() / max(self.width(), 1) if wide else 1.0
        self._want_ids, self._id_map = ids, None
        try:
            img = self._clean_grab(select, lines, faint, faint_lines)
        finally:
            self._fov_zoom, self._want_ids = 1.0, False
            if look:
                ws.xray, ws.xray_opacity, ws.wireframe, ws.wire_opacity, ws.light_on, self.scene.sun = shown
                self.update()
        idm = None
        if ids:
            idm = self._id_map[::-1] if self._id_map is not None else np.full((img.height(), img.width()), -1.0)
            self._id_map = None
        if aspect is not None:
            w, h = img.width(), img.height()
            cw, ch = (w, round(w / aspect)) if wide else (round(h * aspect), h)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            img = img.copy(x0, y0, cw, ch)
            if idm is not None:
                idm = idm[y0:y0 + ch, x0:x0 + cw]
        return (img, idm) if ids else img

    def _clean_grab(self, select, lines, faint=(), faint_lines=()):
        ws = self.ws
        saved = (ws.select_mode, ws.notes_mode, ws.selection, ws.edge_selection, ws.point_selection,
                 ws.seg_selection, ws.active_note, ws.notes_visible, ws.planes_visible)
        ws.planes_visible = False  # drawing aids (planes, helper surfaces, axis crosses) never in the image
        ws.select_mode, ws.notes_mode = bool(select), False
        ws.selection, ws.edge_selection, ws.point_selection, ws.seg_selection = list(select), [], [], []
        ws.active_note, ws.notes_visible = None, False
        self._clean_frame = True
        weak = QColor(STROKE_COLOR)  # weakly highlighted: between stroke and selection color
        weak.setRgbF(*(a + (b - a) * FAINT_MIX for a, b in zip(QColor(STROKE_COLOR).getRgbF()[:3],
                                                                QColor(SELECTED_STROKE_COLOR).getRgbF()[:3])))
        self._extra_tint = {("line", o.id, c.id): weak for c in faint_lines for o in [self.scene.owner_of(c)]
                            if o is not None}
        self._extra_tint.update({("line", o.id, c.id): SELECTED_STROKE_COLOR for c in lines
                                 for o in [self.scene.owner_of(c)] if o is not None})
        self._faint = set(faint) - set(select)
        try:
            return self.grabFramebuffer()
        finally:
            self._clean_frame, self._extra_tint, self._faint = False, {}, set()
            (ws.select_mode, ws.notes_mode, ws.selection, ws.edge_selection, ws.point_selection, ws.seg_selection,
             ws.active_note, ws.notes_visible, ws.planes_visible) = saved
            self.update()

    def _draw_lists(self):
        """(strokes of the scene, live strokes, surfaces, meshes) for this image - scene and surfaces rebuilt only after update() or
        when scene, mode or Ctrl/Shift have changed, otherwise those of the last frame (rotating, drawing); the
        live part (running stroke) every time. Surfaces first: their reconciliation (sync_lofts) deforms painting on them."""
        # Camera modes show the same lists as without a key: starting and releasing rotate/pan rebuilds nothing;
        # likewise the stroke while it is being created (it is itself in the live part)
        mode = None if self._mode in NAV_MODES + ("pending", "draw") else self._mode
        guard = (id(self.scene), mode, self.select_mode, self._ctrl, self._shift)
        cached = getattr(self, "_lists_cache", None)
        if getattr(self, "_lists_dirty", True) or cached is None or cached[0] != guard:
            surfaces = self._surface_draw_list()
            cached = self._lists_cache = (guard, self._scene_stroke_list(), surfaces, self._mesh_draw_list())
            self._flip_data = self._flip_draw_data()  # flipbook drawings of all frames (the frame only selects ranges)
            self._lists_dirty = False
        elif getattr(self, "_hover_dirty", False):
            strokes = self._scene_stroke_list(reuse=True)
            if self.VERIFY_LISTS:
                self._verify_stroke_list(strokes, self._scene_stroke_list())
            cached = self._lists_cache = (guard, strokes, cached[2], cached[3])
        self._hover_dirty = False
        return cached[1], self._live_draw_list(), cached[2], cached[3]

    def _resolve_pending_paste(self, force=False):
        """A pasted/duplicated copy only counts once it has been moved. If one turns to something else
        while it still lies exactly on the original, it disappears again (no duplicate lines)."""
        pending = self.ws.pending_paste
        if not pending or self._mode == "transform":
            return
        if not force and self.select_mode and set(self.edge_selection) == set(pending):
            return
        self.ws.pending_paste = None
        alive = [r for r in pending if edges.ref_to_curve(self.scene, r) is not None]
        partners = edges.coupled_all(self.scene, alive)
        if not alive or not all(any(o not in pending for o in partners[r]) for r in alive):
            return  # moved (or already gone): the copy stays
        for r in alive:
            c = edges.ref_to_curve(self.scene, r)
            owner = self.scene.owner_of(c)
            if owner is not None:
                owner.curves.remove(c)
        self.ws.detached = frozenset()
        self.notify(tr("Kopie nicht verschoben – sie lag genau auf dem Original und wurde wieder entfernt"))

    def _changed(self, painted=None):
        """The scene has changed. painted: only these new paint strokes were added (no full surface reconciliation,
        no new outliner)."""
        self.ws.flip_cache = None  # drawings of the flipbook planes may have changed
        self.ws.sync_hold = None  # (eraser: reconciliation was deferred up to here)
        self._resolve_pending_paste()
        if painted:
            self.scene.sync_painted(painted)
        else:
            self.scene.sync_lofts(free=self._free_move)
            self.scene.assign_owners()  # new surfaces: their line object; line object rehung: surfaces follow
        synced = getattr(self.scene, "_net_sync", None)
        network.sync(self.scene)  # curve net: reconcile point ids of new/changed lines
        # Surfaces are reconciled now (if the net reconciliation did not move anything more): the next frame does not
        # have to do it again - until something else calls update() (e.g. a gizmo drag without _changed)
        fresh = synced is not None and getattr(self.scene, "_net_sync", None) is synced
        self.ws.points_cache = None
        self.ws.seg_cache = None
        self.selection = [i for i in self.selection if self.scene.object_by_id(i) is not None]
        if self.edge_selection:
            existing = {r for r, _ in self._visible_edges(include_paint=True)}
            self.edge_selection = [r for r in self.edge_selection if r in existing]
        if self.point_selection:
            self.point_selection = [r for r, _ in self._valid_points()]
        if self.seg_selection:
            self.seg_selection = [r for r, _ in self._valid_segs()]
        self.ws.paint_only = bool(painted)
        self.scene_changed.emit()
        self.ws.paint_only = False
        self.emit_status()
        self.update()
        if fresh:
            for vp in self.ws.viewports:
                vp._lofts_fresh = (self.scene, self._free_move)

    # --- Drawing on the active surface ----------------------------------
    def _surface_hit(self, pos):
        """Point on the active drawing surface under the cursor or None."""
        mesh = self.scene.active_mesh
        if mesh is None:
            return self._ray_plane(pos, self.scene.active_plane)[0]
        hit_mesh, point, _ = self._mesh_under(pos, [mesh])  # only the target surface: nothing in front blocks it
        return point if hit_mesh is mesh else None

    # --- Display ------------------------------------------------------
    def initializeGL(self):
        try:
            self.renderer = Renderer(moderngl.create_context(), self.context().functions())
        except Exception as exc:  # e.g. driver without OpenGL 3.3
            self.renderer, self._gl_error = None, str(exc)

    MARKS_EVERY = 4  # in a gizmo drag: read the occlusion of the points back from the graphics card only every this-many frames

    def _marks_visible(self, shared, free):
        """(N,) bool: which points (shared + free) are not occluded by surfaces. Reading back stalls the frame
        until the graphics card is done (harbor scene ~12 ms) - so in a gizmo drag only every MARKS_EVERY-th frame,
        in between the last result applies (same point count = the same points in the same order; a point
        that moves behind a surface vanishes at most a few frames later). Otherwise in every frame."""
        n = len(shared) + len(free)
        if self.xray:
            return np.ones(n, bool)
        last = getattr(self, "_marks_vis", None)
        dragging = self._mode == "transform" and self.gizmo.dragging
        if dragging and last is not None and len(last[0]) == n and last[1] < self.MARKS_EVERY - 1:
            self._marks_vis = (last[0], last[1] + 1)
            return last[0]
        vis = self.renderer.points_visible(np.array(shared + free).reshape(-1, 3))
        self._marks_vis = (vis, 0)
        return vis

    def paintGL(self):
        scene = self.scene
        self._track_pick_order()
        p = QPainter(self)
        if self.renderer is not None:
            # native GL between begin/endNativePainting, otherwise moderngl's GL state disturbs the QPainter
            p.beginNativePainting()
            fbo = self.renderer.ctx.detect_framebuffer(self.defaultFramebufferObject())
            opacity = self.surface_opacity
            # orthographic: light from far behind along the view direction (parallel like the view)
            light = self.cam.eye + self.cam.back * (1e4 if self.cam.ortho else 0.0)
            sun = self.scene.sun
            self.renderer.light = (dict(dir=light_mod.direction(sun), col=light_mod.color(sun), sky=light_mod.SKY_TINT,
                                        away=light_mod.AWAY_DARK, shadow=light_mod.SHADOW_DARK, dome=light_mod.DOME,
                                        bands=self.scene.light_bands)
                                   if self.ws.light_on else None)
            self.renderer.material_scale = self.ws.material_scale
            # (not for images without controls - brief, MCP: they set light and sun themselves)
            self.renderer.shadow_hold = (self._mode == "transform" and self.gizmo.dragging
                                         and not getattr(self, "_clean_frame", False))
            strokes, live, surfaces, meshes = self._draw_lists()  # meshes: the same list when rotating (view-frustum test)
            self.renderer.pictures = [] if getattr(self, "_clean_frame", False) else \
                pictures.draw_list(self.scene, self.planes_visible)
            self.renderer.render(fbo, self.cam, self._projection(), self._focal(), light, meshes,
                                 self._plane_draw_list(), strokes, opacity, live=live,
                                 min_px=self.devicePixelRatioF(), surfaces=surfaces,
                                 view_tan=self._tan_half(),
                                 infinite=self._infinite_plane(), xray=self.xray_opacity if self.xray else None,
                                 wire=self.ws.wire_opacity if self.ws.wireframe else None,
                                 flip=self._flip_frame())
            self._frame_painted = True  # playback: the next frame only after this image (do not lose frames)
            self._frame_marks = None
            if getattr(self, "_want_ids", False):  # clean_image(ids=True): color ID of this image
                self._id_map = self.renderer.id_map()
                fbo.use()
                self.renderer.ctx.viewport = (0, 0, *fbo.size)
            if self.select_mode and self.ws.component in ("point", "edge"):
                # points in edit: do not show those occluded by surfaces (X-ray: all) - the GPU checks them in the
                # coverage map of this image, then draw them into the image again
                shared, free, _ = self._net_marks()
                vis = self._marks_visible(shared, free)
                n = len(shared)
                self._frame_marks = ([q for q, v in zip(shared, vis[:n]) if v], [q for q, v in zip(free, vis[n:]) if v])
                fbo.use()
                self.renderer.ctx.viewport = (0, 0, *fbo.size)
            # notes and their connection lines: parts occluded by geometry as with any line (X-ray: all)
            self.notes_tool.occlusion = None
            if self.ws.notes_visible and self.note_list() and not self.xray:
                self.notes_tool.occlusion = self.notes_tool.visibility(self.renderer.points_visible)
                fbo.use()
                self.renderer.ctx.viewport = (0, 0, *fbo.size)
            self.renderer.release_unused([m.geo for m in scene.meshes] +
                                         [px.geo for s in scene.surfaces
                                          if (px := s.pick_proxy(build=False)) is not None])
            self._unbind_vao()
            p.endNativePainting()
        p.setRenderHint(QPainter.Antialiasing)
        if self.renderer is None:
            p.fillRect(self.rect(), theme.ERROR_BG)
            p.setPen(theme.ERROR_FG)
            p.drawText(self.rect(), Qt.AlignCenter, tr("OpenGL 3.3 nicht verfügbar:\n{0}").format(self._gl_error))
        self._draw_world_axes(p)
        if getattr(self, "_clean_frame", False):  # clean image (clean_image): no controls
            p.end()
            return
        self._draw_nulls(p)
        self._draw_active_outline(p)
        self._draw_plane_spot(p)
        self.notes_tool.draw(p)  # in Shots the shot notes (vp.note_list)
        self._draw_review_pins(p)
        if self.ws.timing_mode:  # Shots: plus camera symbols and viewfinder
            self._draw_timing(p)

        target = self.gizmo_target
        if target is not None:
            self.gizmo.draw(p, target)
            self._draw_pivot(p, target)
        self._draw_axis_gizmo(p)
        self._draw_light_strip(p)
        self._draw_pen_feedback(p)
        if self._mode == "band" and self._band_end is not None:
            p.setPen(QPen(theme.BAND_PEN, 1, Qt.DashLine))
            p.setBrush(theme.BAND_FILL)
            p.drawRect(QRectF(self._press_pos, self._band_end).normalized())
        self._draw_smooth_string(p)
        self._draw_eraser(p)
        self._draw_snap_marks(p)
        self._draw_free(p)
        self._draw_surface_snaps(p)
        self._draw_tool_bar(p)
        self._draw_value_hud(p)
        self._draw_view_label(p)
        self._draw_pie(p)
        p.end()

    # --- Context menu ------------------------------------------------------
    def _context_menu(self, pos, global_pos):
        menu = QMenu(self)
        if self.select_mode:  # edit: the most important first
            menu.addAction(tr("Auswahl in eigenes Objekt\tStrg+G"), self.group_selection).setEnabled(
                bool(self.edge_selection or self.seg_selection or self.point_selection))
            chosen = self._chosen_curves()
            menu.addAction(tr("In Pinselstrich umwandeln (aktuelle Farbe)"), lambda: self.convert_lines("brush")).setEnabled(
                any(c.style != "brush" for c in chosen))
            menu.addAction(tr("In Bleistift umwandeln"), lambda: self.convert_lines("pencil")).setEnabled(
                any(c.style == "brush" for c in chosen))
            menu.addSeparator()
        menu.addAction("New Plane\tSpace+N", lambda: self.new_plane(pos))
        menu.addAction("New Stroke	Space+B", self.new_stroke)
        if not self.select_mode:
            menu.addAction(tr("Auswahl in eigenes Objekt\tStrg+G"), self.group_selection).setEnabled(bool(self.edge_selection))
        menu.addAction(tr("Loft durch Querschnitte\tShift+F"), self.do_loft).setEnabled(
            bool(self.edge_selection) or any(isinstance(o, SurfaceObject) for o in self.selected_objects))
        aid = menu.addAction(tr("Hilfsfläche (nur zum Draufzeichnen)"), self.toggle_surface_aid)
        aid.setEnabled(any(isinstance(o, SurfaceObject) for o in self.selected_objects))
        menu.addAction(tr("Alle Linien löschen"), self.clear_all_curves)
        menu.addSeparator()
        world = menu.addAction(tr("Stiftgröße in Welteinheiten"))
        world.setCheckable(True)
        world.setChecked(self.pen_world_mode)
        world.toggled.connect(self.set_pen_world_mode)
        ext = menu.addAction(tr("Extrudieren erzeugt Flächen"))
        ext.setCheckable(True)
        ext.setChecked(self.ws.extrude_surface)
        ext.toggled.connect(lambda on: setattr(self.ws, "extrude_surface", on))
        upright = menu.addAction(tr("Frei-Modus: aufrechte Ebene (statt zur Kamera)"))
        upright.setCheckable(True)
        upright.setChecked(bool(getattr(self.ws, "free_upright", False)))
        upright.setToolTip(tr("Q: die gedachte Ebene durch den Ansatzpunkt steht senkrecht (zur Kamera gedreht) – nach oben gezogen wird echt senkrecht, auch wenn man schräg von oben schaut"))
        upright.toggled.connect(lambda on: setattr(self.ws, "free_upright", on))
        menu.addAction(tr("Einrast-Schritte …"), self.snap_dialog)
        menu.addAction(tr("Smooth-Maximum …"), self.smooth_dialog)
        menu.addAction(tr("Maximale Strichbreite …"), self.pen_max_dialog)
        axis = menu.addAction(tr("Achsenmodus\tAlt+A"))
        axis.setCheckable(True)
        axis.setChecked(self.ws.axis_mode)
        axis.toggled.connect(self.set_axis_mode)
        menu.addAction(tr("Achse zurücksetzen"), self.reset_pivot).setEnabled(self.has_custom_pivot())
        menu.addSeparator()
        deletable = [o for o in self.selected_objects if not (isinstance(o, Plane) and o.preset)]
        menu.addAction(tr("Delete Selection\tEntf"), self.delete_selection).setEnabled(bool(deletable or self.edge_selection))
        if self.selected is not None:
            menu.addAction("Focus\tSpace+G", lambda: self.focus(self.selected))
        if self.area is not None:
            menu.addSeparator()
            menu.addAction(tr("Split links/rechts"), lambda: self.area.split(self, Qt.Horizontal))
            menu.addAction(tr("Split oben/unten"), lambda: self.area.split(self, Qt.Vertical))
            maximized = self.area.is_maximized()
            menu.addAction((tr("Alle Viewports zeigen") if maximized else tr("Maximieren")) + "\tSpace+F",
                           lambda: self.area.toggle_maximize(self)).setEnabled(self.area.count() > 1)
            menu.addAction(tr("Viewport schließen"), lambda: self.area.close_viewport(self)).setEnabled(self.area.count() > 1)
        menu.exec(global_pos)

    # --- Input ----------------------------------------------------------
    def mousePressEvent(self, e):
        if self.notes_tool.editor is not None or self.notes_tool.dismiss_click:
            self.notes_tool.dismiss_click = False  # tapping a heading: a click beside it only ends the typing
            self.notes_tool.finish_edit()
            self._mode = "consumed"
            return
        self.setFocus()
        self._stop_flight()
        pos = e.position()
        pressure = self._current_pressure()
        self._pressure_trace = [pressure] if pressure is not None else []
        self._press_tablet = pressure is not None  # pen: larger click tolerance (wobbles when tapping)
        self._press_time = time.monotonic()
        self._last_pos = self._press_pos = pos
        self._lazy_start = False  # stroke starts only where it reaches the active surface
        b, mods = e.button(), e.modifiers()
        navigating = bool(mods & Qt.AltModifier) or self._space
        target = self.gizmo_target

        if self._mode == "pie":
            # Click confirms the hovered slice (same as releasing the pie key).
            self.confirm_pie(pos)
            return
        if self._mode == "bevel":  # rounding: left click confirms, right click cancels
            self._finish_bevel(b == Qt.LeftButton)
            self.update()
            return
        if self._mode == "extrude":  # running extrusion: left click confirms, right click cancels
            if b == Qt.LeftButton:
                self._finish_extrude()
                self._extrude = None
                self._mode = "consumed"
            else:
                self.cancel_extrude()
                self._mode = "consumed"
            self.update()
            return
        if self._mode == "value":  # hardness/smoothing: right click cancels, left confirms on release (dragging works)
            if b == Qt.RightButton:
                self.finish_value(False)
                self._mode = "consumed"
            else:
                self._val["pressed"] = True
            self.update()
            return
        if b == Qt.RightButton and not navigating and self._light_hit(pos) in ("light", "sun"):
            self._mode = "consumed"
            return self.show_light_bands(e.globalPosition().toPoint())
        hit = self._light_hit(pos) if b == Qt.LeftButton and not navigating else None
        if hit == "light":
            self._mode = "consumed"
            return self.set_light(not self.ws.light_on)
        if hit == "sun":
            self._mode = "consumed"
            return self.set_sun_edit(not self.ws.sun_edit)
        if hit == "disc":
            self._mode = "sun"
            self._sun_from_disc(pos)
            return
        if b == Qt.LeftButton and not navigating and self._over_close(pos):  # X at the top right
            self._mode = "consumed"
            self.area.close_viewport(self)
            return
        if ((self.notes_mode or self.ws.timing_mode and self._shot_note_hit(pos)) and not navigating
                and b in (Qt.LeftButton, Qt.RightButton) and self.notes_tool.press(e)):
            self._mode = "note"
            self.update()
            return
        if b == Qt.LeftButton and self.erase_mode and not navigating:  # eraser (Alt stays camera)
            self._mode = "erase"
            self._erase_whole = bool(mods & Qt.ShiftModifier and mods & Qt.ControlModifier)  # Ctrl+Shift: delete entirely
            self._erase_checkpointed = False
            self._erase_last = None  # depth of the frontmost hit (sphere), anew on every drag
            self._erase_filter = self._erase_filter_for(mods)
            self._erase(pos, pos)
        elif navigating:
            # Alt/Space: Maya / Industry Compatible — LMB orbit, MMB pan, RMB dolly.
            # With Alt held, orbit uses Blender-style axis snap (see mouseMove).
            self._mode = {Qt.LeftButton: "orbit", Qt.MiddleButton: "pan", Qt.RightButton: "dolly"}.get(b)
            self._orbit_alt_snap = bool(mods & Qt.AltModifier) and self._mode == "orbit"
            self._orbit_snap_axis = None
            self._orbit_snap_start = self.cam.state() if self._orbit_alt_snap else None  # Alt snap swings this pose
            self._orbit_snap_acc = [0.0, 0.0]  # drag summed since the start pose (orbit_axis_snap)
            self._orbit_pivot = self._pick_point(pos) if self._mode == "orbit" else None
            return self.update_view()  # only the camera moves right away: no rebuild at the start
        elif b == Qt.MiddleButton:
            # Blender default (no modifier): MMB orbit, Shift+MMB pan, Ctrl+MMB dolly.
            if mods & Qt.ShiftModifier:
                self._mode = "pan"
            elif mods & Qt.ControlModifier:
                self._mode = "dolly"
            else:
                self._mode = "orbit"
                self._orbit_alt_snap = bool(mods & Qt.AltModifier)
                self._orbit_snap_axis = None
                self._orbit_snap_start = self.cam.state() if self._orbit_alt_snap else None  # Alt snap swings this pose
                self._orbit_snap_acc = [0.0, 0.0]  # drag summed since the start pose (orbit_axis_snap)
                self._orbit_pivot = self._pick_point(pos)
            return self.update_view()
        elif b == Qt.RightButton and mods & Qt.ControlModifier and self._width_targets(mods):
            # Ctrl + RMB drag in edge mode: thickness of the selected lines; Ctrl + Shift in drawing: of all lines
            self._mode = "linewidth"
            self._width_set = self._width_targets(mods)
            self._width_total = 1.0
            self.scene.checkpoint()
        elif b == Qt.RightButton and mods & Qt.ControlModifier:
            self._mode = "pensize"  # Ctrl + RMB drag: pen size (prepared for the Wacom pen)
        elif b == Qt.RightButton and mods & Qt.ShiftModifier:
            self._mode = "plane"
            if self.scene.active_mesh is None:
                self.scene.checkpoint()
        elif b == Qt.RightButton:
            self._context_menu(pos, e.globalPosition().toPoint())
        elif b == Qt.LeftButton and self.select_mode and self.ws.value_tool and not navigating:
            self._mode = "valpending"  # hardness/smoothing on: click selects, dragging adjusts
            self._press_how = self._how(mods)
        elif (b == Qt.LeftButton and self._slide_key and self.select_mode and self.ws.component == "point"
              and self._valid_points()):
            self._begin_slide(pos)  # S held: selected points slide along their line
            if self._mode != "slide":
                self._mode = "consumed"
        elif b == Qt.LeftButton and target is not None and (handle := self.gizmo.hit(pos, target)):
            self.scene.checkpoint()
            self._drag_start = (handle, QPointF(pos), self.ws.selection_pivot)
            self.gizmo.begin(handle, pos, target)
            self._mode = "transform"
        elif b == Qt.LeftButton and self._in_gizmo(pos):
            self._mode = "gizmo"
        elif b == Qt.LeftButton and self.select_mode:
            self._mode = "selpending"
            self._press_how = self._how(mods)
        elif b == Qt.LeftButton and self.ws.timing_mode and (cam_hit := self._cam_under(pos)) is not None:
            self._mode = "consumed"  # timing: click on the camera symbol selects the camera (no stroke)
            self._pick_cam(*cam_hit)
        elif b == Qt.LeftButton and self.ws.timing_mode and not self._timing_can_draw():
            self._mode = "consumed"  # timing: drawing only on flipbook planes (the notice comes from there)
        elif b == Qt.LeftButton:
            self._mode = "pending"
            self._press_how = self._how(mods)
            if self._free_pen() and self._press_how == "replace":
                anchor, on_line = self._free_anchor(pos, with_kind=True)
                if anchor is not None:
                    self._begin_free(anchor)
                    self._free_start = (anchor, on_line)
            elif self.multi_mode and len(self._multi_surfaces()) == 1:
                # only one surface in the set: the stroke arises exactly as in isolation (shapes in the plane etc.)
                only = self._multi_surfaces()[0]
                if only is not self.scene.active_surface:
                    self._activate(only)
                self._single_saved = True
                self.ws.multi_mode = False
            # multi mode: (point, surface), the drag may also start in a gap
            self._press_hit = self._multi_samples([pos])[0] if self.multi_mode else self._surface_hit(pos)
            if not (self.multi_mode or self._single_saved or self._free_saved is not None):
                return self.update_live()  # a stroke begins: nothing has changed in the scene yet
        self.update()

    def _hover_key(self):
        """What a mouse move without a key can change in strokes/surfaces: the highlighted edge or line
        (snap points, cursor rings, gizmo hover are overlays and need no new draw lists)."""
        return self._mode, self._select_hover

    def mouseMoveEvent(self, e):
        before = self._hover_key() if self._mode is None else None
        pos = e.position()
        d = pos - self._last_pos if self._last_pos is not None else QPointF()
        moved = self._press_pos is not None and (pos - self._press_pos).manhattanLength() >= self._click_px()
        self._prev_pos, self._last_pos = self._last_pos, pos
        self._ctrl = bool(e.modifiers() & Qt.ControlModifier)
        self._shift = bool(e.modifiers() & Qt.ShiftModifier)
        if self._mode in ("pending", "draw"):
            pressure = self._current_pressure()
            if pressure is not None:
                self._pressure_trace.append(pressure)
        if self._mode == "sun":  # sky disc: drag the sun
            self._sun_from_disc(pos)
            return
        if self._mode == "pie":
            self.pie_move(pos)
            return
        if self.notes_mode and self._mode in (None, "note") or self.ws.timing_mode and self._mode == "note":
            if self._mode is None:
                self._gizmo_hover = self._gizmo_hit(pos)
            self.notes_tool.move(e)
            self.update_view(all_views=True)  # notes, leader lines and drag preview are overlays
            return
        if self.ws.timing_mode and self._mode is None:  # Shots: buttons on the shot note under the cursor
            before = (self.notes_tool.hover, self.notes_tool.sticky and self.notes_tool.sticky[0])
            self.notes_tool.move(e)
            if before != (self.notes_tool.hover, self.notes_tool.sticky and self.notes_tool.sticky[0]):
                self.update_view(all_views=True)

        if self._mode == "pending" and moved and self._free_pen() and self._free_saved is None:
            self.notify(tr("Frei-Modus: Strich auf einer Linie oder einer Fläche ansetzen (Q: aus)"))
            self._mode = "consumed"
        if (self._mode == "pending" and moved and self._press_hit is None and not self.multi_mode
                and not self._free_pen()):
            # started beside/behind the active surface: the stroke begins where it reaches it
            hit = self._surface_hit(pos)
            if hit is not None:
                # exactly where the stroke reaches the surface (sampled finely between the mouse positions)
                entry = self._edge_point(pos, self._prev_pos) if self._prev_pos is not None else None
                self._press_hit, self._press_pos = (entry if entry is not None else hit), QPointF(pos)
                self._lazy_start = True
        if self._mode == "pending" and moved and (self.multi_mode or self._press_hit is not None):
            self._mode = "draw"
            if self.multi_mode:
                self._multi_start(self._press_hit)
            else:
                self._segments, self.current, self._exit = [], [self._press_hit], None
            # snapping at the start: nearby line end or point on a line
            if self._free_saved is not None and self._free_start is not None:
                snap, on_line = self._free_start  # free mode: exactly at the start point (on a line: node)
            elif self._lazy_start:
                snap, on_line = None, False  # started outside: begins exactly at the border (the connection rule connects)
            else:
                snap, on_line = self._draw_snap(self._press_pos)
            self._draw_snaps = []
            self._start_snapped = snap is not None
            if snap is not None and self.current:
                self.current[0] = snap
                self._draw_snaps.append((snap, on_line))
            self._straight = False
            self._shape = None
            self._draw_ended = False
            self._miss_start = None
            self._tip = QPointF(self._press_pos)
            self._hold_anchor = pos
            self._hold_timer.start()
        idle = self._mode is None and not (self.select_mode or self.erase_mode)
        if self._free_saved is not None and self._mode == "draw":  # free stroke: points/lines at any depth
            shape = self._shape
            self._snap_dot = (self._free_end_snap(pos)[0] if shape is None or shape["kind"] == "arc" and not self._ctrl
                              else None)
        elif self._free_pen():
            self._snap_dot = None  # the ring shows when starting (_free_hover)
        else:
            self._snap_dot = self._end_snap(pos) if idle or self._mode == "draw" and self._shape is None else None
        if self._mode == "draw":
            if self._straight:
                self._update_straight(pos)  # the straight line follows the cursor directly
            else:
                self._pull_tip(pos)
                self._freehand_add(self._tip)
            if not self._straight and (pos - self._hold_anchor).manhattanLength() > HOLD_PX:
                self._hold_anchor = pos
                self._hold_timer.start()  # restart: holding still counts from here
        elif self._mode == "transform":
            alt = bool(e.modifiers() & Qt.AltModifier)
            if alt != self._free_move:  # also detect Alt when the key arrived elsewhere
                self._set_free_move(alt)
            self.gizmo.point_snap = self._move_snap(pos)
            self.gizmo.update(pos, snap=self._snap_step())
            self._update_pivot_snap(pos)
            self.emit_status()
        elif self._mode == "erase":
            self._erase_filter = self._erase_filter_for(e.modifiers())  # can also be switched mid-drag
            self._erase(pos - d, pos)
        elif self._mode == "orbit" or (self._mode == "gizmo" and moved):
            alt = bool(e.modifiers() & Qt.AltModifier) or self._orbit_alt_snap
            if self._mode == "orbit" and alt:
                # Blender: hold Alt while rotating the view → snap through orthographic axes.
                self.orbit_axis_snap(d.x(), d.y())
            else:
                self.cam.orbit(d.x(), d.y(), self._orbit_pivot if self._mode == "orbit" else None)
                self.cam.ortho = False  # rotating leaves the axis view
                self._persp_view = None  # freely rotated: this is now the perspective view
                self._orbit_snap_axis = None
        elif self._mode == "pan":
            self.cam.pan(d.x(), d.y(), 2 * math.atan(self._tan_half()), self.height())
        elif self._mode == "dolly":
            self.cam.dolly((d.x() - d.y()) * 0.006)
        if self._mode in ("orbit", "pan", "dolly"):
            return self.update_view()  # only the camera: strokes and surfaces stay
        if self._mode == "plane":
            self._shift_plane(-d.y() / PIXELS_PER_STEP * self._plane_step())
        elif self._mode == "pensize":
            self._scale_pen(math.exp((d.x() - d.y()) * 0.01))
        elif self._mode == "linewidth":
            self._scale_widths(self._width_set, math.exp((d.x() - d.y()) * 0.01))
        elif self._mode == "extrude":
            self._extrude_update(pos)
            self.emit_status()
        elif self._mode == "value":
            self._value_move(pos)
        elif self._mode == "valpending" and moved:
            hover = self._component_hover(self._press_pos)
            if hover is not None and hover[0] not in (self.point_selection + self.seg_selection + self.edge_selection):
                self._component_click(self._press_pos, "replace")  # under the cursor, not yet selected: select
            self._value_tool_start(self._press_pos)
            if self._mode == "value":
                self._value_move(pos)
        elif self._mode == "slide":
            self._slide_update(pos)
        elif self._mode == "bevel":
            self._bevel_update(pos)
        elif self._mode == "selpending" and moved:
            self._mode = "band"
            self._band_end = pos
        elif self._mode == "band":
            self._band_end = pos
        elif self._mode is None and self.select_mode:
            self._select_hover = self._component_hover(pos)
            target = self.transform_target
            self.gizmo.hover = self.gizmo.hit(pos, target) if target is not None else None
        elif self._mode is None:
            self._gizmo_hover = self._gizmo_hit(pos)
            self.gizmo.hover = None  # draw mode: no transform gizmo
            # Look for the start point here, never when painting the image: the search on surfaces uses GPU picking, and that
            # in the middle of paintGL confuses the GL context (crash)
            self._free_hover = self._free_anchor(pos) if self._free_pen() else None
        if before is not None and self._hover_key() == before:
            return self.update_view()  # only snap point/cursor moved: strokes and surfaces stay
        if before is not None and before[0] is None and self._mode is None and self.select_mode:
            # only the highlighted line has changed: rebuild only the stroke list, with the preparatory work from the last
            # frame - surfaces, edge lines and owners stay (artist 12.11.: every new line under the mouse rebuilt all
            # lists, 0.2-0.5 s; that hit exactly the moment when you start to rotate)
            return self.update_hover()
        if self._mode in ("pending", "draw") and not self.multi_mode:
            return self.update_live()  # only the running stroke (multi: the target surface can change mid-stroke)
        self.update()

    def mouseReleaseEvent(self, e):
        pos = e.position()
        if self._mode == "pie":
            # Mouse click already confirmed on press; ignore the matching release.
            self._press_pos = None
            return
        if self._mode == "note":
            self.notes_tool.release(e)
            self._mode = None
            self._press_pos = None
            self._changed()
            return
        if self._mode in NAV_MODES:  # camera released: nothing has changed in the scene
            if self._mode == "orbit":
                self.commit_axis_snap()  # Alt orbit: switch now to the one axis view the drag's direction picked
            self._orbit_alt_snap = False
            self._orbit_snap_axis = None
            self._mode = None
            self._hold_timer.stop()
            return self.update_view()
        clicked = self._press_pos is not None and (pos - self._press_pos).manhattanLength() < self._click_px()
        if self._mode == "draw" and not self._straight:
            self._finish_line(pos)
            self._trim_exit()
        tip = self._tip  # smoothing: pen tip at the end of the string (the stroke ends there)
        self._tip = None
        free = self._free_saved is not None
        painting = self._mode == "draw" and self._painting()  # paint stroke: does not snap, does not split lines
        free_target = None  # free mode: snapped end -> stroke onto the plane through start and target
        if self._mode == "draw" and free and self._shape is None and not self._draw_ended:
            # free mode: end snaps to points/lines at any depth (own logic, _free_span)
            last = self.current if self.current else (self._segments[-1][0] if self._segments else None)
            snap, on_line = self._free_end_snap(pos)
            if snap is not None and last:
                free_target = snap
                self._draw_snaps.append((snap, on_line))
        if self._mode == "draw" and free and self._shape is not None and self._shape["kind"] == "arc" \
                and self._arc_end_snap:
            self._draw_snaps.append(self._arc_end_snap)
            free_target = self._arc_end_snap[0]
        runs = self._finished_runs() if self._mode == "draw" else []
        # all other strokes: how each piece begins/ends (on the surface, change, gap) - the connections
        # themselves are then set by a single rule on the finished geometry (_resolve_stroke)
        kinds = self._stroke_kinds(runs, pos) if runs and not free and not painting else None
        hint = self._cursor_point(pos) if runs and not free and not painting and self.smooth > 0 else None
        if runs and getattr(self, "_press_tablet", False) and self._shape is None:  # pen tapped: no mini stroke
            pts = np.vstack([np.asarray(r, float) for r, _ in runs])
            screen, ok = self.project(pts)
            if ok.any() and float(np.ptp(screen[ok], axis=0).max()) < TABLET_MIN_STROKE_PX:
                runs = []
                obj = self._object_under(pos)
                if obj is not None and self._is_drawing_surface(obj) and obj is not self.scene.active_surface:
                    self.scene.checkpoint()
                    self._activate(obj)
                    self.notify(tr("Zeichnet auf: {0}").format(obj.name))
        if free and free_target is not None and runs:
            self._free_span(runs[-1][0], free_target)
        elif runs and not free and not painting and self._shape is None and kinds and kinds[-1][1] == "hit":
            # ended on the surface: the end exactly onto the point below (the connection rule makes it shared)
            last = np.asarray(runs[-1][0][-1], float) if len(runs[-1][0]) else None
            screen, ok = self.project(last[None]) if last is not None else (None, [False])
            end = self._end_snap(QPointF(*screen[0])) if ok[0] else None  # from the stroke end, not from the cursor
            if end is not None:
                runs[-1][0][-1] = end
        if free:  # imaginary view plane away again, restore the active ones first (before the checkpoint for undo)
            self._end_free()
            runs = [(points, None) for points, _ in runs]
        if self._single_saved:  # multi with one surface: stroke is finished, a click collects surfaces again
            self._single_saved = False
            self.ws.multi_mode = True
        painted = []  # new paint strokes, if a light undo step suffices for them (light completion, _changed)
        if runs:
            # Timing: if there is no drawing yet at this frame, the current one is copied here (keep painting) -
            # that belongs in the undo step, hence a full one then
            flip = self._timing_flip_target() if painting else None
            copy = flip is not None and self._timing_needs_copy(flip)
            light = painting and not self.ws.pending_paste and self.scene.can_paint_step() and not copy
            if not light:
                self.scene.checkpoint()
            if copy:
                self._timing_copy_key(flip)
            made, made_kinds = [], []  # new lines of this drag and how they begin/end
            for n_run, (points, surface) in enumerate(runs):  # multi mode: one drag can yield several lines
                stroke = self.scene.active_stroke
                before = len(stroke.curves) if stroke is not None else 0
                self.scene.add_curve(points, surface.id if surface is not None else None,
                                     self._widths_for(points, pressure=True))
                stroke = self.scene.active_stroke
                if self.ws.draw_tool == "brush" and stroke is not None and len(stroke.curves) > before:
                    stroke.curves[-1].style, stroke.curves[-1].color = "brush", tuple(self.ws.paint_color)
                if painting and stroke is not None and len(stroke.curves) > before:
                    c = stroke.curves[-1]
                    target = surface if surface is not None else self.scene.active_surface
                    if c.plane_id is None and target is not None:
                        c.plane_id = target.id
                    c.paint, c.seq = True, self.scene.next_paint_seq()
                    if self.ws.timing_mode and getattr(target, "flipbook", False):
                        c.frame = int(self.scene.timing.frame)  # drawing of this frame (if there is none: new)
                    c.normals = None
                    c.normals = np.array(self._paint_normals(c), float)
                    if isinstance(target, MeshObject):  # on a mesh: flat onto its surface, slightly lifted
                        self.scene.lift_mesh_paint([c], toward=self.cam.back)
                    if light:
                        painted.append(c)
                elif stroke is not None and len(stroke.curves) > before:
                    components.knots(stroke.curves[-1])  # fix corners once when drawing (curve net)
                    made.append(stroke.curves[-1])
                    made_kinds.append(kinds[n_run] if kinds else ("hit", "hit"))
            if painted:
                self.scene.checkpoint_paint(painted)
            if not painting:  # also free strokes: their ends lie exactly on the target, the rule connects explicitly
                self._resolve_stroke(made, made_kinds, hint)
        elif self._mode == "pending" and clicked:
            obj = self._object_under(pos)
            if self.multi_mode and obj is not None and self._is_drawing_surface(obj) and self._press_how == "replace":
                self._toggle_multi(obj)
            elif self.multi_mode and obj is None and self._press_how == "replace" and self.scene.multi_ids:
                self.scene.checkpoint()  # click into empty space: multi set empty (Ctrl+Z brings it back)
                self.scene.multi_ids = []
                self.notify(tr("Multi: keine Fläche mehr im Satz – Strg+Z holt die Auswahl zurück"))
            elif obj is not None and self._press_how == "replace":
                # draw mode: a click means "I want to draw here" - only activate, select nothing
                if self._is_drawing_surface(obj) and obj is not self.scene.active_surface:
                    self.scene.checkpoint()  # activating belongs in the history (Ctrl+Z: previous drawing surface)
                self._activate(obj)
                self.notify(tr("Zeichnet auf: {0}").format(obj.name) if not isinstance(obj, StrokeObject) or isinstance(obj, SurfaceObject)
                            else tr("Neue Linien landen in: {0}").format(obj.name))
            else:
                self._select_click(obj, self._press_how)
        elif self._mode == "extrude":  # mouse button released: continues until the click
            return
        elif self._mode == "value":
            if self._val is not None and self._val["pressed"]:
                self.finish_value(True)
            return
        elif self._mode == "valpending" and clicked:
            hover = self._component_hover(pos)
            chosen = self.point_selection + self.seg_selection + self.edge_selection
            if hover is not None and hover[0] in chosen and self._press_how == "replace":
                self.set_value_tool(None)  # clicked on the selection again: tool off, selection stays
            else:
                self._component_click(pos, self._press_how)
        elif self._mode == "slide":
            self.finish_slide(True)
        elif self._mode == "band":
            self._band_select(QRectF(self._press_pos, pos).normalized(), self._press_how)
            self._band_end = None
        elif self._mode == "selpending" and clicked:
            self._component_click(pos, self._press_how)
        elif self._mode == "gizmo" and clicked:
            axis = self._gizmo_hit(pos)
            if axis is not None:
                self.axis_view(axis)
        elif self._mode == "transform":
            target = self.gizmo._drag["target"]
            if self.ws.detached:  # the pasted copy is placed: from now on welded as normal
                self.ws.detached = frozenset()
                self.ws.target_cache = (None, None)
            if isinstance(target, PivotHandle):
                target.end()
            self.gizmo.end()
            if self.gizmo.point_snap is not None:  # V: snapped means connected (curve net)
                self._connect_at(self.gizmo.point_snap[1])
            self.gizmo.point_snap = None
            self._drag_start = None
            self.scene.sync_lofts(free=self._free_move)  # with Alt: detach painted lines before Alt is reset
            if self.scene.rerest_patches():  # redistribute strongly deformed net surfaces evenly (shape stays)
                self._changed()
            free = self._free_move
            self._free_move = False  # applies only to this drag (rebuild the target without Alt)
            if free:
                self.ws.target_cache = (None, None)
            if self._dup_last is not None and self._dup_last[0] == self._selection_key():
                self.notify(tr("Kopie verschoben – Strg+Shift+D: nochmal mit demselben Versatz"))
        self._hold_timer.stop()
        self._end_free()
        if self._single_saved:
            self._single_saved = False
            self.ws.multi_mode = True
        self.current = None
        self._segments = []
        self._straight = False
        self._shape = None
        self._draw_snaps = []
        self._snap_dot = None
        self._mode = None
        self._changed(painted)

    def mouseDoubleClickEvent(self, e):
        if self.ws.timing_mode and e.button() == Qt.LeftButton and not (e.modifiers() & Qt.AltModifier):
            hit = self._cam_under(e.position())
            if hit is not None:  # double-click on the camera symbol: look through the camera
                self._mode = "consumed"
                self.look_through(hit[0].id)
                return
        if ((self.notes_mode or self.ws.timing_mode) and e.button() == Qt.LeftButton
                and not (e.modifiers() & Qt.AltModifier) and self.notes_tool.double_click(e)):
            self._mode = "note"  # the following release only ends (notes branch, no click)
            return
        if self.select_mode and e.button() == Qt.LeftButton and not (e.modifiers() & Qt.AltModifier):
            how = self._how(e.modifiers())
            # point/edge mode: double-click on another object jumps there, on the edited one selects the loop
            if self.ws.component == "face" or not self.jump_to_object(e.position(), how):
                self.select_loop(e.position(), how)
                if self.ws.component != "face":  # point/edge mode: the same as points or edges
                    self._convert_selection(self.ws.component)
            self._mode = "consumed"  # do not count the following release as a click
            self._changed()
            return
        super().mouseDoubleClickEvent(e)

    def enterEvent(self, e):
        """Shortcuts act in the viewport under the mouse (as in Houdini) - unless someone is typing elsewhere."""
        focus = QApplication.focusWidget()
        if focus is None or isinstance(focus, Viewport):
            # held keys (Space, P) travel along, otherwise the tool ends at the viewport edge
            held = None
            if isinstance(focus, Viewport) and focus is not self:
                held = (focus._space, focus._penetrate)
            self.setFocus()
            if held and any(held):
                self._space, self._penetrate = held
                self.emit_status()
        super().enterEvent(e)

    def focusInEvent(self, e):
        self.ws.current = self
        self.update_view(all_views=True)  # only the frame of the active viewport changes
        super().focusInEvent(e)

    def leaveEvent(self, _):
        self._last_pos = None
        self._gizmo_hover = None
        self.update_view()

    def wheelEvent(self, e):
        steps = (e.angleDelta().y() or e.angleDelta().x()) / 120  # with Shift some systems deliver horizontally
        mods = e.modifiers()
        if self._mode == "value":  # hardness/smoothing: mouse wheel adjusts finely
            return self._value_nudge(steps * 0.05)
        if self._mode is None and self.select_mode and self.ws.value_tool:
            return self._value_tool_wheel(steps)
        if mods & Qt.ControlModifier:
            targets = self._width_targets(mods)
            if targets:  # edge mode: selected lines, drawing + Shift: all lines thicker/thinner
                if self.scene._burst[0] != "linewidth" or time.monotonic() - self.scene._burst[1] >= 1.0:
                    self._width_total = 1.0
                self.scene.checkpoint(burst="linewidth")
                self._scale_widths(targets, 1.1 ** steps)
                return
            self._scale_pen(1.15 ** steps)
            return
        self._shift_plane(steps * self._plane_step(), burst=("wheel", self.scene.active_plane.id))
        self.update()

    def keyPressEvent(self, e):
        k, mods = e.key(), e.modifiers()
        if k == Qt.Key_Space:
            if not e.isAutoRepeat():
                self._space = True
            return
        if k == Qt.Key_Control:
            return self._set_ctrl(True)
        if k == Qt.Key_Alt:
            if self._mode == "transform":
                self._set_free_move(True)
                e.accept()
                return
            return super().keyPressEvent(e)
        if k == Qt.Key_P:
            if not e.isAutoRepeat():
                self._set_penetrate(True)
            return
        if self._mode == "value" and self._value_key(e):
            return
        if self._mode == "extrude" and k in (Qt.Key_X, Qt.Key_Y, Qt.Key_Z, Qt.Key_N):
            self._extrude_set_axis({Qt.Key_X: "x", Qt.Key_Y: "y", Qt.Key_Z: "z", Qt.Key_N: "n"}[k])
            self.emit_status()
            self.update()
            return
        if self._space:
            # Space+1..4 like Houdini; Shift+2/3/4 = opposite axis (Blender Ctrl+numpad).
            shift = bool(mods & Qt.ShiftModifier)
            actions = {
                Qt.Key_G: lambda: self.focus(self.selected or self.scene.active_surface),
                Qt.Key_H: self.home,
                Qt.Key_N: lambda: self.new_plane(),
                Qt.Key_B: self.new_stroke,  # new stroke object (in the current group) - sits well like Space+N
                Qt.Key_M: self._toggle_space,
                Qt.Key_F: lambda: self.area and self.area.toggle_maximize(self),  # viewport large (previously Space+B)
                Qt.Key_1: self.perspective_view,
                Qt.Key_2: lambda: self.axis_view(np.array([0, -1.0 if shift else 1.0, 0])),  # Top / Bottom
                Qt.Key_3: lambda: self.axis_view(np.array([0, 0, -1.0 if shift else 1.0])),  # Front / Back
                Qt.Key_4: lambda: self.axis_view(np.array([-1.0 if shift else 1.0, 0, 0])),  # Right / Left
                Qt.Key_5: lambda: self.axis_view(np.array([0, -1.0, 0])),  # Bottom
                Qt.Key_6: lambda: self.axis_view(np.array([0, 0, -1.0])),  # Back
                Qt.Key_7: lambda: self.axis_view(np.array([-1.0, 0, 0])),  # Left
            }
            if k in actions:
                actions[k]()
                self._changed()
            return
        if mods & Qt.ControlModifier:
            if k == Qt.Key_Z and mods & Qt.ShiftModifier or k == Qt.Key_Y:
                self.redo()
            elif k == Qt.Key_Z:
                self.undo()
            elif k in (Qt.Key_Delete, Qt.Key_Backspace):
                self.delete_selection(hard=True)  # Ctrl+Del: delete points (Del dissolves them)
            elif k == Qt.Key_L and not mods & Qt.ShiftModifier and self._mode is None:
                self.select_linked("loop")  # Ctrl+L: loop by net flow (the L branch below is never reached with Ctrl)
                self._changed()
            else:
                return super().keyPressEvent(e)  # Ctrl+S/O/I etc. go to the main window
            return
        if self.ws.timing_mode and self._mode is None:  # Shots: own keys, otherwise only pen/brush/eraser
            if self._shot_note_key(e) or self._timing_key(e):
                return
            if k not in (Qt.Key_A, Qt.Key_D):
                return super().keyPressEvent(e)
        if k == Qt.Key_F and self._mode == "bevel" and self._bevel is not None:  # round <-> straight bevel
            op = self._bevel["op"]
            op.set_flat(not op.flat)
            op.apply(self._bevel["size"])
            self._changed()
            return
        if k in (Qt.Key_Tab, Qt.Key_Backtab) and self._mode is None and not e.isAutoRepeat():
            # Hold Tab → selection pie (point/edge/face); release in center → draw↔edit toggle.
            return self.open_pie("select", k, tap=self.toggle_edit)
        if k == Qt.Key_W and not mods and self._mode is None and not e.isAutoRepeat() and not self.notes_mode:
            # Hold W → transform pie (Move/Rotate/Scale + spaces); machine-free style.
            if not self.select_mode:
                self.set_select_mode(True)
            return self.open_pie("transform", k)
        if (k == Qt.Key_S and not mods and shortcuts.action_for(k) is None and self._mode is None
                and not self.select_mode and not self.notes_mode):
            if not e.isAutoRepeat():
                self.set_paint_mode(not self.ws.paint_mode)  # S in drawing: paint mode on/off
            return
        if k == Qt.Key_H and self._mode is None and not self.notes_mode and not e.isAutoRepeat():
            if mods & Qt.AltModifier:
                return self.unhide_all()
            if mods & Qt.ShiftModifier:
                return self.isolate_selection()
            if not mods:
                return self.hide_selection()
        if k in (Qt.Key_A, Qt.Key_D) and not mods and self._mode is None and not e.isAutoRepeat():
            if k == Qt.Key_A:  # A: pen or brush; if one of them is already on, A switches between both
                tool = self.ws.draw_tool
                if tool in ("pen", "brush") and not self.select_mode and not self.notes_mode:
                    return self.set_draw_tool("brush" if tool == "pen" else "pen")
                return self.set_draw_tool(getattr(self.ws, "last_line_tool", "pen"))
            return self.set_draw_tool("eraser")
        if k == Qt.Key_N and not mods and self._mode is None:
            return self.set_notes_mode(not self.notes_mode)
        if self.notes_mode:  # notes: own keys, otherwise only select/move planes and S
            if self.notes_tool.key(e):
                return
            if k not in (Qt.Key_X, Qt.Key_Y, Qt.Key_Z, Qt.Key_4, Qt.Key_BracketLeft, Qt.Key_BracketRight):
                return super().keyPressEvent(e)
        components_keys = {Qt.Key_1: "point", Qt.Key_2: "edge", Qt.Key_3: "face"}
        # Drawing: base plane by the axis you look at it from - X: side (ZY), Y: ground (XZ), Z: front (XY)
        planes = {Qt.Key_Y: 0, Qt.Key_Z: 1, Qt.Key_X: 2}
        step = ("key", self.scene.active_plane.id)
        tool_key = shortcuts.action_for(k) if not mods else None
        if k in components_keys and self.select_mode:  # edit: point / edge / surface as in Houdini
            return self.set_component(components_keys[k])
        if k in planes and not self.select_mode and not mods:
            self.activate_preset(planes[k])
        elif k == Qt.Key_4:
            self.activate_last_custom()
        elif tool_key is not None and not e.isAutoRepeat():
            self.gizmo.tool = tool_key
            self._paste_restore = None  # deliberately chosen: stays
            self._leave_value_tool()  # transforming ends hardness/smoothing
            if not self.select_mode:  # transforming exists only in selection mode
                return self.set_select_mode(True)
        elif k == Qt.Key_F and mods & Qt.AltModifier:
            return self.do_patch()
        elif k == Qt.Key_D and mods & Qt.AltModifier:
            return self.divide_loft()
        elif k == Qt.Key_F and not self.select_mode and not self.notes_mode:
            return self.colorize_surfaces(reset=bool(mods & Qt.ShiftModifier))  # draw mode: tint
        elif k == Qt.Key_F and mods & Qt.ShiftModifier:
            return self.do_loft()
        elif k == Qt.Key_F:
            return self.do_surface()
        elif k == Qt.Key_Escape and self._mode == "bevel":
            self._finish_bevel(False)
            self._mode = None
            return
        elif k == Qt.Key_Escape and self._mode == "extrude":
            return self.cancel_extrude()
        elif k == Qt.Key_Escape and self._mode == "slide":
            return self.finish_slide(False)
        elif k == Qt.Key_K and not e.isAutoRepeat():
            if mods & Qt.ShiftModifier:
                return self.set_hardness(1.0)
            if mods & Qt.AltModifier:
                return self.set_hardness(0.0)
            return self.set_value_tool("hardness")
        elif k == Qt.Key_J and not mods and not e.isAutoRepeat():
            return self.set_value_tool("smooth")
        elif k == Qt.Key_S and not mods:
            if not e.isAutoRepeat():
                self._slide_key = True
                if self.select_mode and self.ws.component == "point":
                    self.status_changed.emit(tr("S gehalten: gewählte Punkte ziehen – sie gleiten auf ihrer Linie"))
            return
        elif k == Qt.Key_Escape and self._mode == "pie":
            return self.cancel_pie()
        elif k == Qt.Key_Escape and self.select_mode and self.ws.value_tool:
            return self.set_value_tool(None)
        elif k == Qt.Key_Escape and self.select_mode:
            return self.set_select_mode(False)  # Esc always leads back to drawing
        elif k == Qt.Key_Escape:
            self.gizmo.tool = None
            self._paste_restore = None
        elif k == Qt.Key_B:
            return self.set_multi_mode(not self.multi_mode)
        elif k == Qt.Key_Q and not self.select_mode:
            return self.set_free_draw(not self.free_draw)
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selection(hard=bool(mods & Qt.ControlModifier), remove=bool(mods & Qt.ShiftModifier))
        elif k == Qt.Key_BracketRight:
            self._shift_plane(self._plane_step(), burst=step)
        elif k == Qt.Key_BracketLeft:
            self._shift_plane(-self._plane_step(), burst=step)
        elif k == Qt.Key_C and mods & Qt.AltModifier:
            if not e.isAutoRepeat():
                self.convert_generators()
            return
        elif k in (Qt.Key_A, Qt.Key_S, Qt.Key_D) and not mods & Qt.AltModifier:
            return  # held key (repeat): nothing
        elif k == Qt.Key_G and mods & Qt.AltModifier:
            return self.create_null("null")
        elif k == Qt.Key_G and mods & Qt.ShiftModifier:
            return self.dissolve_null()
        elif k == Qt.Key_A and mods & Qt.AltModifier:
            if not self.select_mode:
                self.set_select_mode(True)
            return self.set_axis_mode(not self.ws.axis_mode)
        elif k == Qt.Key_V:
            if not e.isAutoRepeat():
                self._set_vsnap(True)
            return
        elif k == Qt.Key_L:  # L: connected in the own object, Shift+L: across object boundaries, Ctrl+L: loop by net flow
            self.select_linked("loop" if mods & Qt.ControlModifier else "all" if mods & Qt.ShiftModifier else "object")
        elif k == Qt.Key_X:
            return self.start_extrude()
        else:
            return super().keyPressEvent(e)
        self._changed()

    def _set_penetrate(self, on):
        self._penetrate = on
        if self._mode == "draw" and self._straight and self._last_pos is not None:
            self._update_straight(self._last_pos)
            self.update()

    def _snap_step(self):
        """Fraction of the snap step: 0 without Ctrl, 1 with Ctrl, 0.1 with Ctrl+Shift (fine)."""
        if not self._ctrl:
            return 0
        return 0.1 if self._shift or QApplication.keyboardModifiers() & Qt.ShiftModifier else 1

    def _set_free_move(self, on):
        """Alt mid-drag: only the selection moves, welded neighbors stay put (and
        the other way round on release). The drag restarts from the initial state for this - up to the current
        mouse position, you do not notice it."""
        if self._free_move == on:
            return
        self._free_move = on
        if self._mode != "transform" or self._drag_start is None or self._last_pos is None:
            return
        handle, press, pivot = self._drag_start
        self.scene.undo()  # initial state of the drag (checkpoint when starting) ...
        self.scene.checkpoint()  # ... and secure it as an undo step again
        self.ws.selection_pivot = pivot
        target = self.gizmo_target
        if target is None:
            return
        self.gizmo.begin(handle, press, target)
        self.gizmo.update(self._last_pos, snap=self._snap_step())
        self.notify(tr("Alt: nur die Auswahl bewegt sich – verschweißte Linien bleiben stehen") if on
                    else tr("Verschweißte Linien werden wieder mitgezogen"))
        self.update()

    def _set_ctrl(self, on):
        """Pressing/releasing Ctrl takes effect immediately, even without mouse movement (straight line or gizmo drag)."""
        if self._ctrl == on:
            return
        self._ctrl = on
        if self._last_pos is None:
            return
        if self._mode == "draw" and self._straight:
            self._update_straight(self._last_pos)
        elif self._mode == "transform":
            self.gizmo.update(self._last_pos, snap=self._snap_step())
            self.emit_status()
        elif self._mode == "extrude":
            self._extrude_update(self._last_pos)
            self.emit_status()
        elif self._mode == "bevel":
            self._bevel_update(self._last_pos)
        self.update()

    def _set_vsnap(self, on):
        """V held: in axis mode the dragged pivot snaps to points, when moving the
        grabbed point of the selection (takes effect immediately, also mid-drag)."""
        self._vsnap = on
        if self._mode == "transform" and self._last_pos is not None:
            self._update_pivot_snap(self._last_pos)
            if not isinstance(self.gizmo._drag["target"], PivotHandle):
                self.gizmo.point_snap = self._move_snap(self._last_pos)
                self._snap_dot = None if self.gizmo.point_snap is None else self._snap_dot
                self.gizmo.update(self._last_pos, snap=self._snap_step())
            self.update()

    def _update_pivot_snap(self, pos):
        target = self.gizmo._drag["target"] if self.gizmo.dragging else None
        if isinstance(target, PivotHandle):
            snap = self._vsnap and self.gizmo.tool == "translate"
            target.set_snap(self._pivot_snap_point(pos) if snap else None)

    def smooth_dialog(self):
        """Right-click -> Smooth maximum: radius of the string at full smooth slider, in pen radii (saved)."""
        value, ok = QInputDialog.getDouble(self, tr("Smooth-Maximum"), tr("Smooth-Radius bei vollem Regler (Stiftradien):"),
                                           self.smooth_max_factor, 0.5, 100000.0, 1)  # practically unlimited
        if ok:
            self.smooth_max_factor = value
            QSettings("SketchMachine", "SketchMachine").setValue("smooth_max_factor", value)

    def pen_max_dialog(self):
        """Right-click -> Maximum stroke width: upper limit for pen/brush and line thickness in world units (saved)."""
        from .workspace import PEN_MAX_WORLD_LIMIT
        value, ok = QInputDialog.getDouble(self, tr("Maximale Strichbreite"), tr("Größte Strichbreite (Meter, höchstens 5000):"),
                                           self.pen_max_world, 0.01, PEN_MAX_WORLD_LIMIT, 2)
        if ok:
            self.set_pen_max_world(value)
            QSettings("SketchMachine", "SketchMachine").setValue("pen_max_world", value)

    def set_pen_max_world(self, value):
        from .workspace import PEN_MAX_WORLD_LIMIT
        self.pen_max_world = max(0.01, min(PEN_MAX_WORLD_LIMIT, float(value)))
        if self.pen_world > self.pen_max_world:  # the pen shrinks along if the limit lies below it
            self.pen_world = self.pen_max_world
            self.emit_status()

    def snap_dialog(self):
        """Right-click -> Snap steps: step sizes for Ctrl when rotating/straight lines, scaling, moving."""
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Einrast-Schritte (Strg)"))
        form = QFormLayout(dlg)
        fields = []
        for label, attr, lo, hi, decimals, suffix in (
                (tr("Rotieren und Geraden"), "snap_angle", 0.1, 180.0, 1, " °"),
                (tr("Skalieren"), "snap_scale", 0.1, 100.0, 1, " %"),
                (tr("Verschieben"), "snap_move", 0.001, 1000.0, 3, tr(" Einheiten"))):
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSuffix(suffix)
            box.setValue(getattr(self.ws, attr))
            form.addRow(label, box)
            fields.append((attr, box))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        settings = QSettings("SketchMachine", "SketchMachine")
        for attr, box in fields:
            setattr(self.ws, attr, box.value())
            settings.setValue(attr, box.value())
        self.status_changed.emit(tr("Einrasten (Strg): {0:g}° · {1:g} % · {2:g} Einheiten").format(self.ws.snap_angle, self.ws.snap_scale, self.ws.snap_move))

    def _toggle_space(self):
        self.gizmo.space = {"world": "local", "local": "view"}.get(self.gizmo.space, "world")
        self.toggles_changed.emit()
        self._changed()

    def set_transform_tool(self, tool):
        """Toolbar in selection mode: move/rotate/scale (like T/R/E), None = no gizmo."""
        self.gizmo.tool = tool
        self._paste_restore = None
        if tool is not None:
            self._leave_value_tool()  # transforming ends hardness/smoothing
        if tool is not None and not self.select_mode:
            self.set_select_mode(True)
        self.toggles_changed.emit()
        self._changed()

    def set_local_space(self, on):
        if on or self.gizmo.space == "local":
            self.gizmo.space = "local" if on else "world"
        self.toggles_changed.emit()
        self._changed()

    def set_view_space(self, on):
        if on or self.gizmo.space == "view":
            self.gizmo.space = "view" if on else "world"
        self.toggles_changed.emit()
        self._changed()

    def keyReleaseEvent(self, e):
        if self._pie is not None and not e.isAutoRepeat() and e.key() == self._pie.get("key"):
            self.confirm_pie(self._last_pos)
            e.accept()
            return
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
        elif e.key() == Qt.Key_Control:
            self._set_ctrl(False)
        elif e.key() == Qt.Key_Alt and self._free_move:
            self._set_free_move(False)
            e.accept()
        elif e.key() == Qt.Key_V and not e.isAutoRepeat():
            self._set_vsnap(False)
        elif e.key() == Qt.Key_S and not e.isAutoRepeat():
            self._slide_key = False
        elif e.key() == Qt.Key_P and not e.isAutoRepeat():
            self._set_penetrate(False)
        else:
            super().keyReleaseEvent(e)

    def focusOutEvent(self, e):
        # held keys end here (e.g. mouse moved into another viewport)
        if self._pie is not None:
            self.cancel_pie()
        self._space = False
        self._ctrl = False
        self._penetrate = False
        self._slide_key = False
        self.update()
        super().focusOutEvent(e)

    def notify(self, text):
        """Short notice in the status line that stays for NOTICE_S seconds (otherwise the status display
        overwrites it immediately on the next click)."""
        self.ws.notice = (text, time.monotonic())
        self.status_changed.emit(text)
        QTimer.singleShot(int(NOTICE_S * 1000) + 50, self.emit_status)

    def emit_status(self):
        if self.ws.axis_mode:
            tool = TOOLS.get(self.gizmo.tool, "–")
            self.status_changed.emit(
                tr("ACHSENMODUS – Gizmo bewegt nur den Pivot ({0})  |  T: verschieben · R: Achsen drehen · V halten: auf Linienenden/-mitten und Ebenen-Ecken einrasten · Strg: Schritte · Rechtsklick → Achse zurücksetzen · Alt+A: zurück").format(tool))
            return
        notice = self.ws.notice
        if notice is not None and time.monotonic() - notice[1] < NOTICE_S:
            self.status_changed.emit(notice[0])  # short notice takes priority for a few seconds
            return
        if self.notes_mode:
            self.status_changed.emit(
                tr("NOTIZEN ({0})  |  Doppelklick: neue Notiz (auf einer Linie: daran angeheftet) · Ziehen: verschieben (Shift: nur senkrecht) · am Ring bzw. mit Strg ziehen: weitere Hinweislinie · Doppelklick: Überschrift · R: drehen · Rechtsklick: Größe, Ausrichtung · ⤢: Notizbrett groß · Entf: löschen · Esc/N: zurück zum Zeichnen").format(len(self.scene.notes)))
            return
        if self.erase_mode:
            self.status_changed.emit(
                tr("RADIERER – {0:.0f} px  |  Ziehen: Linien und Malerei wegradieren (nur in der Tiefe des vordersten Treffers; P gehalten: durch alles) · Strg: nur Linien · Strg+Shift: ganze Linien löschen · Strg + Mausrad / Strg + RMB: Größe · A: Stift · Tab: Edit").format(self.eraser_px))
            return
        if self._extrude:
            ex = self._extrude
            axis = {"n": tr("Normale"), "x": "X", "y": "Y", "z": "Z"}[ex["axis_name"]]
            self.status_changed.emit(
                tr("EXTRUDIEREN – {0} Linie(n), Länge {1:+.3g} entlang {2}  |  Maus: Länge · X/Y/Z: Achse, N: Normale · Klick: bestätigen · Esc/Rechtsklick: abbrechen").format(len(ex['curves']), ex['length'], axis))
            return
        if self._mode == "bevel" and self._bevel is not None:
            self.status_changed.emit(
                tr("{0} – {1} Ecke(n)/Kante(n), Größe {2:.3g}  |  Maus vom Ansatzpunkt weg: größer · F: rund/gerade · Strg: einrasten (Strg+Shift fein) · Klick: fertig · Esc/Rechtsklick: abbrechen").format(tr("FASE") if self._bevel['op'].flat else tr("ABRUNDEN"), self._bevel['op'].count, self._bevel['size']))
            return
        if self.select_mode:
            comp = self.ws.component
            en = i18n.LANG == "en"
            parts = [(len(self.point_selection), tr("Punkt"), "s" if en else "e"), (len(self.edge_selection) + len(self.seg_selection), tr("Kante"), "s" if en else "n"),
                     (len(self.selection), tr("Objekt"), "s" if en else "e")]
            what = ", ".join(f"{n} {name}{plural if n != 1 else ''}" for n, name, plural in parts if n) or tr("nichts")
            tool = TOOLS.get(self.gizmo.tool, tr("kein Gizmo – T/R/E"))
            hint = {"point": tr("Klick: Punkt · Strg + Klick auf Linie: Punkt einfügen · Doppelklick: Punkte des Loops/der Fläche"),
                    "edge": tr("Klick: Kante · Doppelklick: Loop bzw. Linien der Fläche · F: füllen · X: extrudieren"),
                    "face": tr("Klick: Fläche · Doppelklick: Fläche samt Linien")}[comp]
            self.status_changed.emit(
                tr("EDIT {0} (1/2/3) – {1} gewählt · Gizmo: {2}  |  {3} · L: Verbundenes · Shift: hinzufügen · Strg: abwählen · Ziehen: Rechteck · Tab/Esc: zurück zum Zeichnen").format(COMPONENT_NAMES[comp].upper(), what, tool, hint))
            return
        surface = self.scene.active_surface
        if isinstance(surface, Plane):
            where = tr("Ebene: {0} (Offset {1:+.2f})").format(surface.name, surface.origin @ surface.normal)
        else:
            where = f"{tr('Fläche') if isinstance(surface, SurfaceObject) else 'Mesh'}: {surface.name}"
        if self._free_pen():
            where = tr("FREI (Q) – Strich auf einer Linie oder einer Fläche ansetzen, gezeichnet wird in der Ebene zur Kamera durch diesen Punkt")
        elif self.multi_mode and len(self._multi_surfaces()) > 1:
            n = len(self._multi_surfaces())
            where = tr("MULTI – {0} Flächen – Klick auf Fläche: dazu/weg · B: Multi aus").format(n)
        elif self.multi_mode:
            where = "MULTI – " + where + tr(" – Klick auf weitere Flächen: dazu · B: Multi aus")
        else:
            where += tr(" (Klick wechselt die Fläche · B: Multi – auf mehrere Flächen zugleich)")
        tool = TOOLS.get(self.gizmo.tool, tr("Auswahl"))
        space = {"world": tr("Welt"), "local": tr("Lokal")}.get(self.gizmo.space, tr("Kamera"))
        stroke = self.scene.active_stroke.name if self.scene.active_stroke else "–"
        pen = f"{self.pen_world:.3g} m" if self.pen_world_mode else f"{self.pen_px:.0f} px"
        self.status_changed.emit(
            tr("{0}  |  Stroke: {1}  |  Stift: {2}  |  Auswahl: {3}  |  Werkzeug: {4} [{5}]  |  A Stift · D Radierer · F: Fläche einfärben · Tab: Edit · N: Notizen · Strg: einrasten · P Gerade durchs Mesh").format(where, stroke, pen, len(self.selection), tool, space)
        )


for _name in SHARED:
    setattr(Viewport, _name, _shared(_name))
