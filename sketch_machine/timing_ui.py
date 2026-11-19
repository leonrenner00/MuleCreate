"""Shots mode: timeline (bottom) - prototype + Calisto motion beats.

Timeline: frames, playback (every frame is shown - the next only once the last image is drawn), one row per
flipbook plane with its drawings (drag = change timing), on top the poses of the selected camera, plus a
Motions row for component binds / path / camera-angle planning spans.
Shot notes and cameras live in the Shots panel - a note board like in the Notes tab (note_board.py).
"""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox, QToolButton, QVBoxLayout,
                               QWidget)

from . import timing, ui
from .i18n import tr

NAME_W = 120  # column with the row names
ROW_H = 22
RULER_H = 20
MOTION_FILL = QColor(94, 148, 168, 110)
MOTION_EDGE = QColor(70, 120, 140)
MOTION_ACTIVE = QColor(208, 138, 94, 160)


def _vp(ws):
    return ws.current


class TimelineStrip(QWidget):
    """The timeline itself: ruler, camera row, motions row, one row per flipbook plane."""

    def __init__(self, ws, parent=None):
        super().__init__(parent)
        self.ws = ws
        self.setMouseTracking(True)
        self.setMinimumHeight(RULER_H + ROW_H * 3)
        self._drag = None  # ("scrub",) | ("key", plane, old, new) | ("motion", beat, edge, origin)

    # --- Geometry ---
    @property
    def tm(self):
        return self.ws.scene.timing

    def _rows(self):
        """[(kind, object, title)] in order from the top."""
        sc = self.ws.scene
        cam = sc.timing.camera_by_id(self._cam_id())
        rows = [("cam", cam, cam.name if cam else tr("Kamera"))]
        rows.append(("motions", None, tr("Motion")))
        rows += [("plane", p, p.name) for p in timing.flipbook_planes(sc)]
        return rows

    def _cam_id(self):
        vp = _vp(self.ws)
        return vp._look if vp is not None and vp._look is not None else self.ws.timing_cam

    def sizeHint(self):
        return QSize(600, RULER_H + ROW_H * max(len(self._rows()), 3) + 4)

    def _x(self, frame):
        tm = self.tm
        span = max(tm.end - tm.start, 1)
        w = max(self.width() - NAME_W - 12, 10)
        return NAME_W + 6 + (frame - tm.start) / span * w

    def _frame_at(self, x):
        tm = self.tm
        span = max(tm.end - tm.start, 1)
        w = max(self.width() - NAME_W - 12, 10)
        return tm.clamp(round(tm.start + (x - NAME_W - 6) / w * span))

    def _row_at(self, y):
        i = int((y - RULER_H) // ROW_H)
        rows = self._rows()
        return rows[i] if 0 <= i < len(rows) and y >= RULER_H else None

    # --- Drawing ---
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tm, sc = self.tm, self.ws.scene
        p.fillRect(self.rect(), QColor(ui.PANEL))
        font = QFont(self.font())
        font.setPointSize(8)
        p.setFont(font)
        # ruler
        span = max(tm.end - tm.start, 1)
        step = next((s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500) if span / s <= max(self.width() - NAME_W, 50) / 40), 1000)
        p.setPen(QColor(ui.TEXT_DIM))
        first = tm.start - tm.start % step + (step if tm.start % step else 0)
        for f in range(first, tm.end + 1, step):
            x = self._x(f)
            p.drawLine(QPointF(x, RULER_H - 5), QPointF(x, RULER_H))
            p.drawText(QRectF(x - 30, 0, 60, RULER_H - 5), Qt.AlignCenter, str(f))
        keys = timing.flip_keys(sc)
        active = sc.active_plane.id if sc.active_plane.flipbook else None
        for i, (kind, obj, title) in enumerate(self._rows()):
            y = RULER_H + i * ROW_H
            if kind == "plane" and obj.id == active:
                p.fillRect(QRectF(0, y, self.width(), ROW_H), QColor(208, 138, 94, 40))
            p.setPen(QColor(ui.LINE))
            p.drawLine(QPointF(0, y + ROW_H), QPointF(self.width(), y + ROW_H))
            p.setPen(QColor(ui.ACCENT) if kind == "plane" and obj.id == active else QColor(ui.TEXT))
            p.drawText(QRectF(6, y, NAME_W - 8, ROW_H), Qt.AlignLeft | Qt.AlignVCenter, title)
            cy = y + ROW_H / 2
            if kind == "cam" and obj is not None:
                col = QColor.fromRgbF(*obj.rgb)
                for k, pose in enumerate(obj.poses):
                    x = self._x(pose.frame)
                    nxt = obj.poses[k + 1].frame if k + 1 < len(obj.poses) else tm.end
                    p.fillRect(QRectF(x, cy - 2, max(self._x(nxt) - x, 1), 4), QColor(col.red(), col.green(), col.blue(), 90))
                    p.setPen(QPen(col.darker(140), 1))
                    p.setBrush(col)
                    p.drawEllipse(QPointF(x, cy), 5, 5)
                    prev = obj.poses[k - 1] if k else None
                    if prev is not None and abs(prev.mm - pose.mm) > 0.05:  # focal length changes: small "mm"
                        p.setPen(QColor(ui.TEXT))
                        p.drawText(QRectF(x + 6, y, 30, ROW_H), Qt.AlignLeft | Qt.AlignVCenter, "mm")
            elif kind == "motions":
                for beat in tm.motions:
                    a, b = beat.start, beat.end
                    if self._drag and self._drag[0] == "motion" and self._drag[1] is beat:
                        a, b = self._drag[3], self._drag[4]
                    x0, x1 = self._x(a), self._x(b)
                    if x1 < x0:
                        x0, x1 = x1, x0
                    fill = MOTION_ACTIVE if tm.active_motion == beat.id else MOTION_FILL
                    p.fillRect(QRectF(x0, y + 4, max(x1 - x0, 4), ROW_H - 8), fill)
                    p.setPen(QPen(MOTION_EDGE, 1.2))
                    p.drawRect(QRectF(x0, y + 4, max(x1 - x0, 4), ROW_H - 8))
                    if beat.camera:
                        p.setBrush(QColor(MOTION_EDGE))
                        p.setPen(Qt.NoPen)
                        p.drawEllipse(QPointF(x0 + 5, cy), 3, 3)
                    label = beat.annotation or timing.target_label(sc, beat)
                    p.setPen(QColor(ui.TEXT))
                    p.drawText(QRectF(x0 + 8, y, max(x1 - x0 - 10, 20), ROW_H),
                               Qt.AlignLeft | Qt.AlignVCenter, label)
            elif kind == "plane":
                frames = keys.get(obj.id, [])
                filled = {int(c.frame) for s in sc.strokes for c in s.curves if c.plane_id == obj.id and c.frame is not None}
                for k, f in enumerate(frames):
                    if self._drag and self._drag[0] == "key" and self._drag[1] is obj and self._drag[2] == f:
                        f = self._drag[3]
                    x = self._x(f)
                    nxt = frames[k + 1] if k + 1 < len(frames) else None
                    if f in filled or (self._drag and self._drag[0] == "key" and self._drag[1] is obj
                                       and self._drag[2] in filled and self._drag[3] == f):
                        end = self._x(nxt) if nxt is not None else self._x(tm.end)
                        p.fillRect(QRectF(x, cy - 1.5, max(end - x, 1), 3), QColor(201, 198, 193, 60))  # holds
                    diamond = QPolygonF([QPointF(x, cy - 6), QPointF(x + 6, cy), QPointF(x, cy + 6), QPointF(x - 6, cy)])
                    p.setPen(QPen(QColor(ui.TEXT), 1.2))
                    p.setBrush(QColor(ui.TEXT) if f in filled else Qt.NoBrush)  # empty drawing: hollow
                    p.drawPolygon(diamond)
        # current frame
        x = self._x(tm.frame)
        p.setPen(QPen(QColor(ui.ACCENT), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ui.ACCENT))
        p.drawRoundedRect(QRectF(x - 18, 1, 36, RULER_H - 6), 3, 3)
        p.setPen(QColor(ui.BG))
        p.drawText(QRectF(x - 18, 1, 36, RULER_H - 6), Qt.AlignCenter, str(tm.frame))
        p.end()

    # --- Mouse ---
    def _key_under(self, pos):
        row = self._row_at(pos.y())
        if row is None or row[0] != "plane":
            return None
        for f in timing.flip_keys(self.ws.scene).get(row[1].id, []):
            if abs(self._x(f) - pos.x()) <= 6:
                return row[1], f
        return None

    def _motion_under(self, pos):
        row = self._row_at(pos.y())
        if row is None or row[0] != "motions":
            return None
        x = pos.x()
        for beat in self.tm.motions:
            x0, x1 = self._x(beat.start), self._x(beat.end)
            if x1 < x0:
                x0, x1 = x1, x0
            if x0 - 4 <= x <= x1 + 4:
                edge = "start" if abs(x - x0) <= 6 else "end" if abs(x - x1) <= 6 else "body"
                return beat, edge
        return None

    def mousePressEvent(self, e):
        vp = _vp(self.ws)
        if vp is None:
            return
        pos = e.position()
        row = self._row_at(pos.y())
        if e.button() == Qt.RightButton:
            hit = self._key_under(pos)
            if hit is not None:
                return self._key_menu(hit, e.globalPosition().toPoint())
            mhit = self._motion_under(pos)
            if mhit is not None:
                return self._motion_menu(mhit[0], e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton:
            return
        if pos.x() < NAME_W:  # name: activate plane (painting happens there) or select note
            if row is not None and row[0] == "plane":
                vp.activate_flipbook(row[1])
            self.update()
            return
        mhit = self._motion_under(pos)
        if mhit is not None:
            beat, edge = mhit
            vp.scene.checkpoint()
            self.tm.active_motion = beat.id
            self._drag = ("motion", beat, edge, beat.start, beat.end, beat.start, beat.end)
            vp.set_frame(beat.start if edge == "start" else beat.end if edge == "end" else self._frame_at(pos.x()))
            self.ws.timing_changed.emit()
            return
        hit = self._key_under(pos)
        if hit is not None:
            vp.scene.checkpoint()  # one undo step per drag
            self._drag = ("key", hit[0], hit[1], hit[1])
            vp.set_frame(hit[1])
            return
        self._drag = ("scrub",)
        vp.set_frame(self._frame_at(pos.x()))

    def mouseMoveEvent(self, e):
        vp = _vp(self.ws)
        if vp is None or self._drag is None:
            return
        f = self._frame_at(e.position().x())
        if self._drag[0] == "scrub":
            vp.set_frame(f)  # Qt merges several images into one - dragging stays smooth
        elif self._drag[0] == "motion":
            _, beat, edge, _a, _b, o0, o1 = self._drag
            if edge == "start":
                a, b = min(f, o1), o1
            elif edge == "end":
                a, b = o0, max(f, o0)
            else:
                span = max(o1 - o0, 0)
                mid0 = (o0 + o1) // 2
                a = self.tm.clamp(o0 + (f - mid0))
                b = self.tm.clamp(a + span)
                if b - a < span:
                    a = self.tm.clamp(b - span)
            self._drag = ("motion", beat, edge, a, b, o0, o1)
            self.update()
        else:
            self._drag = (*self._drag[:3], f)
            self.update()

    def mouseReleaseEvent(self, e):
        drag, self._drag = self._drag, None
        vp = _vp(self.ws)
        if not drag or vp is None:
            return
        if drag[0] == "key":
            _, plane, old, new = drag
            if new != old:
                if new in timing.flip_keys(vp.scene).get(plane.id, []):
                    vp.notify(tr("Bei {0} liegt schon eine Zeichnung – nicht verschoben").format(timing.frame_label(new)))
                else:
                    timing.move_key(vp.scene, plane, old, new)
                    vp.set_frame(new)
                    vp._timing_changed()
                    return
            self.update()
            return
        if drag[0] == "motion":
            _, beat, _edge, a, b, o0, o1 = drag
            beat.start, beat.end = int(min(a, b)), int(max(a, b))
            beat.clamp_range(self.tm)
            if (beat.start, beat.end) != (o0, o1):
                vp.scene.dirty = True
                vp._timing_changed(full=False)
            else:
                self.ws.timing_changed.emit()
            self.update()

    def _key_menu(self, hit, at):
        vp = _vp(self.ws)
        plane, frame = hit
        menu = QMenu(self)
        menu.addAction(tr("Zeichnung {0} löschen").format(timing.frame_label(frame)),
                       lambda: (vp.scene.checkpoint(), timing.delete_key(vp.scene, plane, frame), vp._timing_changed()))
        menu.exec(at)

    def _motion_menu(self, beat, at):
        vp = _vp(self.ws)
        menu = QMenu(self)
        menu.addAction(tr("Motion löschen"),
                       lambda: (vp.scene.checkpoint(), timing.remove_motion(self.tm, beat), vp._timing_changed()))
        menu.exec(at)


class TimelinePanel(QWidget):
    """Timeline with control bar; plays back (ws.playing). Motion beats bind components + path + camera."""

    def __init__(self, ws, parent=None):
        super().__init__(parent)
        self.ws = ws
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)
        bar = QHBoxLayout()
        self.play_button = QToolButton(text="▶")
        self.play_button.setToolTip(tr("Abspielen / Anhalten (Alt+A) – jeder Frame wird gezeigt"))
        self.play_button.clicked.connect(self.toggle_play)
        bar.addWidget(self.play_button)
        self.frame_spin = self._spin(tr("Aktueller Frame (←/→, ↑/↓ springt zur nächsten Zeichnung)"))
        self.frame_spin.valueChanged.connect(lambda v: _vp(ws) and _vp(ws).set_frame(v))
        bar.addWidget(QLabel(tr("Frame")))
        bar.addWidget(self.frame_spin)
        self.start_spin = self._spin(tr("Erster Frame"))
        self.end_spin = self._spin(tr("Letzter Frame"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setToolTip(tr("Bilder pro Sekunde"))
        for label, w in ((tr("von"), self.start_spin), (tr("bis"), self.end_spin), ("fps", self.fps_spin)):
            bar.addWidget(QLabel(label))
            bar.addWidget(w)
        self.start_spin.editingFinished.connect(self._range_edited)
        self.end_spin.editingFinished.connect(self._range_edited)
        self.fps_spin.editingFinished.connect(self._range_edited)
        bar.addSpacing(12)
        self.onion_box = QCheckBox(tr("Onion-Skin"))
        self.onion_box.setToolTip(tr("Vorige und nächste Zeichnung blass zeigen (O)"))
        self.onion_box.toggled.connect(self._onion)
        bar.addWidget(self.onion_box)
        for text, tip, slot in ((tr("Leere Zeichnung"), tr("Leere Zeichnung hier: die Ebene ist ab diesem Frame leer"),
                                 self._blank),
                                (tr("Zeichnung löschen"), tr("Die Zeichnung, die hier gilt, löschen – die vorige gilt dann "
                                                             "länger"), self._delete_key)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addSpacing(12)
        bar.addWidget(QLabel(tr("Kameragröße")))
        self.cam_size = QSlider(Qt.Horizontal)
        self.cam_size.setRange(10, 500)
        self.cam_size.setFixedWidth(90)
        self.cam_size.setToolTip(tr("Größe aller Kamerasymbole im Viewport (bei vielen Kameras kleiner)"))
        self.cam_size.valueChanged.connect(self._cam_size)
        bar.addWidget(self.cam_size)
        bar.addStretch(1)
        self.info = QLabel()
        self.info.setStyleSheet(f"color: {ui.TEXT_DIM};")
        bar.addWidget(self.info)
        lay.addLayout(bar)
        # Motion beat bar (Calisto Create timeline slice)
        mbar = QHBoxLayout()
        mbar.setSpacing(4)
        for text, tip, slot in (
                (tr("Bind"), tr("Auswahl (Null/Stroke/Mesh/Plane) an eine Timeline-Spanne binden – ab aktuellem Frame"),
                 self._bind_selection),
                (tr("Pfad"), tr("Pfad aus aktivem Stroke oder Flipbook-Zeichnung für die gewählte Motion setzen"),
                 self._set_path),
                (tr("Kamera"), tr("Aktuelle Viewport-Kamera als Planungswinkel für die Motion speichern"),
                 self._capture_cam),
                (tr("Vorschau"), tr("Gespeicherten Planungswinkel der aktiven Motion anschauen (Esc lässt die Shot-Kamera)"),
                 self._preview_cam),
                (tr("Löschen"), tr("Aktive Motion löschen"), self._delete_motion)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            mbar.addWidget(b)
        mbar.addWidget(QLabel(tr("Annotation")))
        self.ann_edit = QLineEdit()
        self.ann_edit.setPlaceholderText(tr("Kurzlabel der Motion"))
        self.ann_edit.setFixedWidth(120)
        self.ann_edit.editingFinished.connect(self._ann_edited)
        mbar.addWidget(self.ann_edit)
        mbar.addWidget(QLabel(tr("Text")))
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText(tr("Bewegungsabsicht – was macht das Teil hier?"))
        self.text_edit.setMinimumWidth(180)
        self.text_edit.editingFinished.connect(self._text_edited)
        mbar.addWidget(self.text_edit, 1)
        self.motion_info = QLabel()
        self.motion_info.setStyleSheet(f"color: {ui.TEXT_DIM};")
        mbar.addWidget(self.motion_info)
        lay.addLayout(mbar)
        self.strip = TimelineStrip(ws, self)
        lay.addWidget(self.strip, 1)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        ws.timing_changed.connect(self.refresh)
        ws.frame_changed.connect(self._frame_changed)
        ws.scene_changed.connect(self.refresh)
        self.refresh()

    @staticmethod
    def _spin(tip):
        s = QSpinBox()
        s.setRange(-100000, 1000000)
        s.setToolTip(tip)
        s.setKeyboardTracking(False)
        return s

    def refresh(self):
        tm = self.ws.scene.timing
        for spin, value in ((self.frame_spin, tm.frame), (self.start_spin, tm.start), (self.end_spin, tm.end),
                            (self.fps_spin, tm.fps)):
            if spin.value() != value and not spin.hasFocus():
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self.frame_spin.setRange(tm.start, tm.end)
        self.onion_box.blockSignals(True)
        self.onion_box.setChecked(tm.onion)
        self.onion_box.blockSignals(False)
        if not self.cam_size.isSliderDown():
            self.cam_size.blockSignals(True)
            self.cam_size.setValue(round(tm.cam_size * 100))
            self.cam_size.blockSignals(False)
        if getattr(self.ws, "play_requested", None) is not None and self.ws.play_requested != self.ws.playing:
            self.set_playing(self.ws.play_requested)
        self._update_info()
        self._update_motion_fields()
        self.strip.updateGeometry()
        self.strip.update()

    def _frame_changed(self, frame):
        if self.frame_spin.value() != frame:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(frame)
            self.frame_spin.blockSignals(False)
        self._update_info()
        self._update_motion_fields()
        self.strip.update()

    def _update_info(self):
        sc = self.ws.scene
        plane = sc.active_plane if sc.active_plane.flipbook else None
        if plane is None:
            self.info.setText(tr("Keine Flipbook-Ebene aktiv – Space+N im Viewport legt eine an"))
            return
        shown = timing.held_key(timing.flip_keys(sc).get(plane.id, []), sc.timing.frame)
        here = shown == sc.timing.frame
        self.info.setText(tr("{0}: malt in Zeichnung {1}").format(plane.name, timing.frame_label(sc.timing.frame)) if here
                          else tr("{0}: hier gilt {1} – Malen oder Radieren übernimmt sie als neue Zeichnung bei {2}").format(
                              plane.name, timing.frame_label(shown) if shown is not None else tr("nichts"),
                              timing.frame_label(sc.timing.frame)))

    def _update_motion_fields(self):
        beat = self.ws.scene.timing.active_beat()
        for edit, value in ((self.ann_edit, beat.annotation if beat else ""),
                            (self.text_edit, beat.text if beat else "")):
            if not edit.hasFocus() and edit.text() != value:
                edit.blockSignals(True)
                edit.setText(value)
                edit.blockSignals(False)
        if beat is None:
            self.motion_info.setText(tr("Keine Motion – Bind bindet Auswahl an Span"))
            return
        bits = [f"{timing.frame_label(beat.start)}–{timing.frame_label(beat.end)}",
                timing.target_label(self.ws.scene, beat)]
        if beat.path_stroke_id or beat.path_plane_id:
            bits.append(tr("Pfad"))
        if beat.camera:
            bits.append(tr("Kamera"))
        self.motion_info.setText(" · ".join(bits))

    def _active_beat(self, need=True):
        beat = self.ws.scene.timing.active_beat()
        if beat is None and need:
            vp = _vp(self.ws)
            if vp is not None:
                vp.notify(tr("Erst eine Motion wählen oder mit Bind anlegen"))
        return beat

    def _bind_selection(self):
        vp = _vp(self.ws)
        if vp is None:
            return
        vp.scene.checkpoint()
        beat = timing.bind_selection(vp.scene, list(self.ws.selection))
        # Prefer flipbook plane as path when painting in Shots
        plane = vp.scene.active_plane if vp.scene.active_plane.flipbook else None
        if plane is not None:
            timing.set_motion_path(beat, vp.scene, plane_id=plane.id)
        vp.notify(tr("Motion gebunden: {0} ({1}–{2})").format(
            timing.target_label(vp.scene, beat), timing.frame_label(beat.start), timing.frame_label(beat.end)))
        vp._timing_changed()

    def _set_path(self):
        vp = _vp(self.ws)
        beat = self._active_beat()
        if vp is None or beat is None:
            return
        vp.scene.checkpoint()
        stroke = None
        for oid in self.ws.selection:
            obj = vp.scene.object_by_id(oid)
            if obj is not None and getattr(obj, "curves", None) is not None:
                stroke = obj
                break
        if stroke is None and vp.scene.active_stroke is not None:
            stroke = vp.scene.active_stroke
        plane = vp.scene.active_plane if vp.scene.active_plane.flipbook else None
        timing.set_motion_path(beat, vp.scene,
                               stroke_id=stroke.id if stroke is not None else None,
                               plane_id=plane.id if plane is not None and stroke is None else (
                                   plane.id if plane is not None else None))
        if beat.path_stroke_id is None and beat.path_plane_id is None:
            return vp.notify(tr("Kein Pfad – Stroke wählen oder Flipbook bemalen"))
        pts = timing.path_points(vp.scene, beat)
        vp.notify(tr("Pfad gesetzt ({0} Punkte)").format(len(pts)))
        vp._timing_changed()

    def _capture_cam(self):
        vp = _vp(self.ws)
        beat = self._active_beat()
        if vp is None or beat is None:
            return
        vp.scene.checkpoint()
        look = vp._look_cam() if hasattr(vp, "_look_cam") else None
        if look is not None:
            pose = timing.pose_at(look, vp.scene.timing.frame)
            mm = pose.mm
            state = pose.state()
        else:
            tan = getattr(vp, "_cam_tan", None)
            mm = timing.tan_to_mm(tan) if tan is not None else 35.0
            state = vp.cam.state()
        timing.capture_motion_camera(beat, state, mm)
        vp.notify(tr("Kamerawinkel für Motion gespeichert ({0} mm)").format(round(mm, 1)))
        vp._timing_changed()

    def _preview_cam(self):
        vp = _vp(self.ws)
        beat = self._active_beat()
        if vp is None or beat is None:
            return
        state = timing.camera_state(beat)
        if state is None:
            return vp.notify(tr("Diese Motion hat noch keinen Kamerawinkel – Kamera drücken"))
        target, dist, rot, mm = state
        if vp.cam.ortho:
            vp.cam.ortho = False
        vp.cam.set_state(target, dist, rot)
        vp._cam_tan = timing.mm_to_tan(mm)
        vp.update_view()
        vp.notify(tr("Vorschau: Planungswinkel von „{0}“").format(beat.annotation or timing.target_label(vp.scene, beat)))

    def _delete_motion(self):
        vp = _vp(self.ws)
        beat = self._active_beat()
        if vp is None or beat is None:
            return
        vp.scene.checkpoint()
        timing.remove_motion(vp.scene.timing, beat)
        vp._timing_changed()

    def _ann_edited(self):
        beat = self.ws.scene.timing.active_beat()
        if beat is None:
            return
        text = self.ann_edit.text().strip()
        if text != beat.annotation:
            self.ws.scene.checkpoint()
            beat.annotation = text
            self.ws.scene.dirty = True
            self.ws.timing_changed.emit()

    def _text_edited(self):
        beat = self.ws.scene.timing.active_beat()
        if beat is None:
            return
        text = self.text_edit.text().strip()
        if text != beat.text:
            self.ws.scene.checkpoint()
            beat.text = text
            self.ws.scene.dirty = True
            self.ws.timing_changed.emit()

    def _range_edited(self):
        vp = _vp(self.ws)
        tm = self.ws.scene.timing
        start, end, fps = self.start_spin.value(), self.end_spin.value(), self.fps_spin.value()
        if end <= start:
            end = start + 1
        if (start, end, fps) != (tm.start, tm.end, tm.fps):
            tm.start, tm.end, tm.fps = start, end, fps
            tm.frame = tm.clamp(tm.frame)
            self.ws.scene.dirty = True
            if self.ws.playing:
                self.timer.setInterval(round(1000 / tm.fps))
            if vp is not None:
                vp.set_frame(tm.frame)
            self.ws.timing_changed.emit()

    def _cam_size(self, v):
        self.ws.scene.timing.cam_size = v / 100
        self.ws.scene.dirty = True
        vp = _vp(self.ws)
        if vp is not None:
            vp.update_view(all_views=True)

    def _onion(self, on):
        self.ws.scene.timing.onion = bool(on)
        vp = _vp(self.ws)
        if vp is not None:
            vp.update_view(all_views=True)

    def _plane(self):
        vp = _vp(self.ws)
        plane = vp.scene.active_plane if vp is not None else None
        if plane is None or not plane.flipbook:
            if vp is not None:
                vp.notify(tr("Erst eine Flipbook-Ebene wählen (Name in der Timeline) oder anlegen (Space+N)"))
            return None, None
        return vp, plane

    def _blank(self):
        vp, plane = self._plane()
        if plane is None:
            return
        t = vp.scene.timing.frame
        if t in timing.flip_keys(vp.scene).get(plane.id, []):
            return vp.notify(tr("Bei {0} liegt schon eine Zeichnung").format(timing.frame_label(t)))
        vp.scene.checkpoint()
        timing.add_blank_key(plane, t)
        vp._timing_changed()

    def _delete_key(self):
        vp, plane = self._plane()
        if plane is None:
            return
        shown = timing.held_key(timing.flip_keys(vp.scene).get(plane.id, []), vp.scene.timing.frame)
        if shown is None:
            return
        vp.scene.checkpoint()
        timing.delete_key(vp.scene, plane, shown)
        vp._timing_changed()

    # --- Playback ---
    def toggle_play(self):
        self.set_playing(not self.ws.playing)

    def set_playing(self, on):
        self.ws.playing = self.ws.play_requested = bool(on)
        self.play_button.setText("■" if on else "▶")
        if on:
            self.timer.start(round(1000 / max(self.ws.scene.timing.fps, 1)))
        else:
            self.timer.stop()
            vp = _vp(self.ws)
            if vp is not None:
                vp.update_view(all_views=True)  # onion skin again

    def _tick(self):
        if not self.ws.timing_mode:
            return self.set_playing(False)
        vps = [v for v in self.ws.viewports if v.isVisible()]
        if any(not v._frame_painted for v in vps):
            return  # the last image is not drawn yet: no frame is skipped
        tm = self.ws.scene.timing
        nxt = tm.frame + 1 if tm.frame < tm.end else tm.start
        if vps and not vps[0].set_frame(nxt):
            self.set_playing(False)
