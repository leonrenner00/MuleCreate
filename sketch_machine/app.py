import math
import os
import sys

from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QCursor, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMenu, QMessageBox, QSizePolicy, QSlider, QToolBar, QToolButton, QVBoxLayout,
                               QWidget, QWidgetAction)

from . import theme, ui
from .note_board import NoteBoard
from .outliner import Outliner
from .scene import Scene
from .views import ViewArea
from .workspace import CLEANUP_DIST_RANGE, Workspace
from . import assoc, backup, i18n, paths, recorder, review, startup
from .i18n import tr

FILE_FILTER = tr("Mule-Szene (*.mule *.sketch.json);;JSON (*.json)")
GEOMETRY_FILTER = tr("Geometrie (*.obj *.usd *.usda *.usdc *.usdz);;Alle Dateien (*)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QApplication.instance().setStyleSheet(ui.STYLE)  # Anthracite + copper (also menus, tooltips, dialogs)
        self.path = None
        self.last_brief = None  # folder of the last brief export (MCP: write_clarified writes there)
        self.ws = Workspace()
        self.ws.status_changed.connect(self.statusBar().showMessage)
        self.ws.scene_changed.connect(self._update_title)
        self.views = ViewArea(self.ws)
        self.setCentralWidget(self.views)

        self.outliner = Outliner(self.ws)
        dock = QDockWidget("Outliner", self)
        dock.setObjectName("outliner_dock")
        dock.setWidget(self.outliner.panel())  # with a search field above
        self.addDockWidget(Qt.RightDockWidgetArea, dock)  # outliner on the right, toolbar on the left
        # "Note" panel below the outliner: the active note (only in Notes mode; dockable like any panel)
        self.note_panel = NoteBoard(self.views, overlay=False, on_maximize=self._maximize_note,
                                    on_scope=self._set_note_scope, on_pin=self.set_note_pinned)
        self._note_scope = "local"
        self._note_pinned = QSettings("SketchMachine", "SketchMachine").value("note_pinned", False, type=bool)
        self.note_panel.pin_button.setChecked(self._note_pinned)
        self.ws.note_panel = self.note_panel
        self.note_dock = QDockWidget(tr("Notiz"), self)
        self.note_dock.setObjectName("note_dock")
        self.note_dock.setWidget(self.note_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.note_dock)
        self.splitDockWidget(dock, self.note_dock, Qt.Vertical)
        self.note_dock.hide()
        self._notes_shown = False
        self._note_dock_sized = False
        self.ws.scene_changed.connect(self._sync_note_panel)
        # Shots (prototype): timeline at the bottom, Shots panel on the right - visible only in Shots mode. The panel is a
        # note board like the Note panel (Local: clicked shot note, Global: shot note of the whole scene)
        from .timing_ui import TimelinePanel
        self.timeline = TimelinePanel(self.ws)
        self.timeline_dock = QDockWidget(tr("Timeline"), self)
        self.timeline_dock.setObjectName("timeline_dock")
        self.timeline_dock.setWidget(self.timeline)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.timeline_dock)
        self.shots_panel = NoteBoard(self.views, overlay=False, on_maximize=self._maximize_shot,
                                     on_scope=self._set_shot_scope)
        self.shots_panel.pin_button.hide()
        self.shots_panel.empty.setText(tr("Im Viewport eine Shot-Notiz anklicken (Doppelklick legt eine an) – hier "
                                          "stehen dann ihre Kameras, Referenzen und Texte. Global: Notiz für alle Shots."))
        self._shot_scope = "local"
        self.ws.shots_panel = self.shots_panel
        self.timing_dock = QDockWidget("Shots", self)
        self.timing_dock.setObjectName("timing_dock")
        self.timing_dock.setWidget(self.shots_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.timing_dock)
        self.splitDockWidget(dock, self.timing_dock, Qt.Vertical)
        self.ws.toggles_changed.connect(self._sync_timing_docks)
        self.ws.scene_changed.connect(self._sync_shots_panel)
        self.ws.timing_changed.connect(self._sync_shots_panel)

        self._build_menu()
        self._build_toolbar()
        self._build_view_menu()
        self.resize(1400, 850)
        self._default_state = self.saveState()  # factory position of all panels and bars (View -> Reset workspace)
        state = QSettings("SketchMachine", "SketchMachine").value("window_state")
        if state is not None:  # position of the panels (e.g. Note below the viewport)
            self.restoreState(state)
        self.note_dock.hide()
        self.timeline_dock.hide()
        self.timing_dock.hide()
        self.recorder = recorder.Recorder(self)
        self._rec_label = QLabel()
        self._rec_label.setStyleSheet("color: #d23c3c; font-weight: bold; padding: 0 8px;")
        self._rec_label.hide()
        self.statusBar().addPermanentWidget(self._rec_label)
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(500)
        self._rec_timer.timeout.connect(self._update_rec_label)
        self.recorder.changed.connect(self._recorder_changed)
        self.review = review.ReviewSession(self)
        self.review.changed.connect(self._review_changed)
        self.review.transcribed.connect(self._review_transcribed)
        self.review.marked.connect(self._review_marked)
        self.review.unmarked.connect(self._review_unmarked)
        self.review.aiming.connect(self._review_aiming)
        self.review.marking_changed.connect(self._review_marking)
        self.review_sender = review.ReviewSender(self, self.review)
        self.review_sender.state.connect(self._style_review)
        self.review_sender.done.connect(self._review_sent)
        self._send_frame = 0
        self._send_spin = QTimer(self, interval=150)
        self._send_spin.timeout.connect(self._spin_send)
        self.viewport.emit_status()
        if self._note_pinned:
            QTimer.singleShot(0, self._sync_note_panel)
        # automatic backup (settings: on/off, interval; "backups" folder next to the scene)
        self._backup_hash = None
        self._backup_timer = QTimer(self)
        self._backup_timer.timeout.connect(self._backup_when_idle)
        self._restart_backup_timer()
        self._update_title()
        self.viewport.setFocus()

    # --- Backup, recently opened, settings ---
    def _restart_backup_timer(self):
        self._backup_timer.stop()
        if backup.enabled():
            self._backup_timer.start(backup.minutes() * 60 * 1000)

    def _backup_when_idle(self):
        """Backup from the timer: never in the middle of a stroke or drag (in large scenes it takes noticeable time) - while
        a mouse button is pressed, try again shortly."""
        if QApplication.mouseButtons() != Qt.NoButton:
            QTimer.singleShot(500, self._backup_when_idle)
            return
        self.auto_backup()

    def auto_backup(self):
        """Backup of the scene if it was changed since saving or the last backup -> file or None."""
        scene = self.viewport.scene
        if not scene.dirty:
            return None
        try:
            target, self._backup_hash = backup.write(scene, self.path, {"views": self.views.state()}, self._backup_hash)
        except OSError as exc:
            self.statusBar().showMessage(tr("Backup fehlgeschlagen: {0}").format(exc), 8000)
            return None
        if target:
            self.statusBar().showMessage(tr("Backup: {0}").format(target), 4000)
        return target

    def _fill_recent_menu(self):
        menu = self.recent_menu
        menu.clear()
        items = backup.recent()
        for path in items:
            name = os.path.basename(path)
            menu.addAction(f"{name}  —  {os.path.dirname(path)}", lambda p=path: self.open_recent(p))
        if not items:
            menu.addAction(tr("(noch keine)")).setEnabled(False)
        menu.addSeparator()
        menu.addAction(tr("Liste leeren"), backup.clear_recent).setEnabled(bool(items))
        menu.addAction(tr("Backup-Ordner öffnen"), self.open_backup_folder)

    def open_recent(self, path):
        if not os.path.isfile(path):
            QMessageBox.warning(self, tr("Öffnen fehlgeschlagen"), tr("Die Datei gibt es nicht mehr:\n{0}").format(path))
            return
        if self._confirm_discard():
            self.load_path(path)

    def open_backup_folder(self):
        folder = backup.folder_for(self.path)
        os.makedirs(folder, exist_ok=True)
        _open_folder(folder)

    def open_settings(self):
        """Settings: automatic backup on/off and interval in minutes."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QPushButton, QSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Einstellungen"))
        lay = QVBoxLayout(dlg)
        group = QGroupBox(tr("Backup"))
        form = QFormLayout(group)
        on = QCheckBox(tr("Automatisch Backups anlegen"))
        on.setChecked(backup.enabled())
        form.addRow(on)
        spin = QSpinBox()
        spin.setRange(*backup.MINUTES_RANGE)
        spin.setSuffix(tr(" min"))
        spin.setValue(backup.minutes())
        spin.setEnabled(on.isChecked())
        on.toggled.connect(spin.setEnabled)
        form.addRow(tr("Abstand zwischen den Backups:"), spin)
        info = QLabel(tr("Nur wenn sich seit dem Speichern bzw. dem letzten Backup etwas geändert hat. Ordner „{0}“ neben "
                         "der Szenendatei (ungespeichert: Dokumente/Mule/{0}); je Szene bleiben die neuesten {1}.")
                      .format(backup.FOLDER, backup.KEEP))
        info.setWordWrap(True)
        form.addRow(info)
        folder = QPushButton(tr("Backup-Ordner öffnen"))
        folder.clicked.connect(self.open_backup_folder)
        form.addRow(folder)
        lay.addWidget(group)
        rgroup = QGroupBox(tr("Review"))
        rform = QFormLayout(rgroup)
        rkeep = QSpinBox()
        rkeep.setRange(*backup.REVIEW_KEEP_RANGE)
        rkeep.setValue(backup.review_keep())
        rform.addRow(tr("Review-Backups je Szene:"), rkeep)
        rinfo = QLabel(tr("Vor jedem abgeschickten Review sichert Mule die Szene im Ordner „{0}“ neben der Szenendatei; "
                          "ältere werden gelöscht.").format(backup.REVIEW_FOLDER))
        rinfo.setWordWrap(True)
        rform.addRow(rinfo)
        rfolder = QPushButton(tr("Review-Backups öffnen"))
        rfolder.clicked.connect(lambda: (os.makedirs(backup.folder_for(self.path, backup.REVIEW_FOLDER), exist_ok=True),
                                         _open_folder(backup.folder_for(self.path, backup.REVIEW_FOLDER))))
        rform.addRow(rfolder)
        lay.addWidget(rgroup)
        self._review_keep_spin = rkeep  # (Tests)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        self._settings_dialog = dlg  # (Tests)
        if dlg.exec() == QDialog.Accepted:
            backup.set_options(on.isChecked(), spin.value())
            backup.set_review_keep(rkeep.value())
            self._restart_backup_timer()

    @property
    def viewport(self):
        """The most recently used viewport: menu, toolbar and outliner act there."""
        return self.ws.current

    # --- UI ---
    def _build_menu(self):
        file_menu = self.menuBar().addMenu(tr("&Datei"))
        for text, shortcut, slot in (
            (tr("&Neu"), QKeySequence.New, self.new_scene),
            (tr("Ö&ffnen…"), QKeySequence.Open, self.open_scene),
            (None, None, None),
            (tr("&Speichern"), QKeySequence.Save, self.save_scene),
            (tr("Speichern &unter…"), QKeySequence("Ctrl+Shift+S"), self.save_scene_as),
            (tr("Geometrie &importieren…"), QKeySequence("Ctrl+I"), self.import_geometry),
            (tr("Sketch &Brief exportieren…"), QKeySequence("Ctrl+E"), self.export_brief),
        ):
            if text is None:  # recently opened: right below Open
                self.recent_menu = file_menu.addMenu(tr("&Zuletzt geöffnet"))
                self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
                continue
            action = QAction(text, self, shortcut=shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        export_menu = file_menu.addMenu(tr("E&xportieren"))
        for text, shortcut, slot in (
            (tr("Szene als &USD…"), QKeySequence("Ctrl+Shift+E"), self.export_usd),
            (tr("&Kurven…"), QKeySequence(), self.export_curves),
        ):
            action = QAction(text, self, shortcut=shortcut)
            action.triggered.connect(slot)
            export_menu.addAction(action)
        file_menu.addSeparator()
        settings_action = QAction(tr("&Einstellungen…"), self, shortcut=QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)
        # copy & paste: whole objects above the outliner, otherwise the selection in the viewport
        edit_menu = self.menuBar().addMenu(tr("&Bearbeiten"))
        for text, shortcut, slot in (
            (tr("&Kopieren"), QKeySequence.Copy, lambda: self.viewport.copy_selection(objects_only=self._over_outliner())),
            (tr("&Einfügen"), QKeySequence.Paste, lambda: self.viewport.paste(self._paste_mode())),
            (tr("&Duplizieren"), QKeySequence("Ctrl+D"), lambda: self.viewport.duplicate(self._paste_mode())),
            (tr("Duplizieren &wiederholen (gleicher Versatz)"), QKeySequence("Ctrl+Shift+D"),
             lambda: self.viewport.duplicate_repeat(self._paste_mode())),
            (tr("Auswahl in neues &Objekt"), QKeySequence("Ctrl+G"), lambda: self.viewport.group_selection()),
            (tr("Ecken a&brunden (Punkte bzw. Ecken der Kanten)"), QKeySequence("Ctrl+B"), lambda: self.viewport.start_bevel()),
            (tr("&Loft ringförmig (letzter Querschnitt zurück zum ersten)"), QKeySequence("Ctrl+Shift+F"),
             lambda: self.viewport.do_loft(wrap=True)),
        ):
            action = QAction(text, self, shortcut=shortcut)
            if shortcut is not None and QKeySequence(shortcut).toString() not in ("Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z"):
                action.setAutoRepeat(False)  # held key: once (harbor test: Ctrl+B fired ~20x per second)
            action.triggered.connect(slot)
            edit_menu.addAction(action)

    def _ink_color_action(self, menu):
        """View: two color swatches - pencil (lines on paper) and chalk (lines on the chalkboard); saved."""
        from PySide6.QtWidgets import QColorDialog, QPushButton
        settings = QSettings("SketchMachine", "SketchMachine")
        for key, name in (("pencil_color", "pencil"), ("chalk_color", "chalk")):
            value = settings.value(key, "")
            if isinstance(value, str) and len(value) == 7:
                theme.set_stroke_colors(**{name: tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))})
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(12, 4, 12, 4)
        buttons = {}

        def paint():
            for name, rgb in zip(("pencil", "chalk"), theme.stroke_colors()):
                buttons[name].setStyleSheet(f"background: rgb{tuple(rgb)}; border: 1px solid {ui.LINE}; "
                                            "min-width: 28px; min-height: 16px; border-radius: 3px;")

        def pick(name):
            menu.close()
            current = theme.stroke_colors()[0 if name == "pencil" else 1]
            color = QColorDialog.getColor(QColor(*current), self, tr("Bleistiftfarbe") if name == "pencil" else tr("Kreidefarbe"))
            if not color.isValid():
                return
            rgb = color.getRgb()[:3]
            theme.set_stroke_colors(**{name: rgb})
            QSettings("SketchMachine", "SketchMachine").setValue(f"{name}_color", "#%02x%02x%02x" % rgb)
            paint()
            for vp in self.ws.viewports:
                vp._changed()

        def reset():
            theme.set_stroke_colors(theme.PENCIL_DEFAULT, theme.CHALK_DEFAULT)
            s = QSettings("SketchMachine", "SketchMachine")
            s.remove("pencil_color")
            s.remove("chalk_color")
            paint()
            for vp in self.ws.viewports:
                vp._changed()

        for name, text in (("pencil", tr("Bleistift")), ("chalk", tr("Kreide"))):
            lay.addWidget(QLabel(text))
            buttons[name] = QPushButton()
            buttons[name].setToolTip(tr("{0}farbe der Linien ({1})").format(text, tr("Zeichenpapier") if name == 'pencil' else tr("Kreidetafel")))
            buttons[name].clicked.connect(lambda _=False, n=name: pick(n))
            lay.addWidget(buttons[name])
        back = QPushButton("↺")
        back.setToolTip(tr("Standardfarben"))
        back.setFixedWidth(26)
        back.clicked.connect(reset)
        lay.addWidget(back)
        paint()
        holder = QWidgetAction(self)
        holder.setDefaultWidget(box)
        return holder

    def _point_size_action(self):
        """View: slider for the size of the points in edit (point and edge mode), 25-300 %; saved."""
        settings = QSettings("SketchMachine", "SketchMachine")
        self.ws.point_scale = min(max(float(settings.value("point_scale", 1.0)), 0.25), 3.0)
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.addWidget(QLabel(tr("Punktgröße")))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(25, 300)
        slider.setValue(round(self.ws.point_scale * 100))
        slider.setMinimumWidth(120)
        slider.setToolTip(tr("Größe der Punkte im Edit-Modus (Punkt- und Kantenmodus)"))
        value = QLabel(f"{slider.value()} %")
        value.setMinimumWidth(40)

        def changed(v):
            self.ws.point_scale = v / 100
            value.setText(f"{v} %")
            QSettings("SketchMachine", "SketchMachine").setValue("point_scale", self.ws.point_scale)
            for vp in self.ws.viewports:
                vp.update()

        slider.valueChanged.connect(changed)
        lay.addWidget(slider)
        lay.addWidget(value)
        holder = QWidgetAction(self)
        holder.setDefaultWidget(box)
        return holder

    def _build_view_menu(self):
        # Objects: hierarchy like in Cinema 4D - null, symmetry, array (generators with instances)
        obj_menu = self.menuBar().addMenu(tr("&Objekte"))
        for text, slot in (
            # keys act in the viewport (not as menu shortcuts: Shift+G is a capital G when typing a note)
            (tr("&Null (Auswahl gruppieren)\tAlt+G"), lambda: self.viewport.create_null("null")),
            (tr("&Symmetrie (Auswahl spiegeln)"), lambda: self.viewport.create_null("symmetry")),
            (tr("&Array (Auswahl vervielfachen)"), lambda: self.viewport.create_null("array")),
            (tr("Gruppe a&uflösen\tShift+G"), lambda: self.viewport.dissolve_null()),
            (tr("&Konvertieren (echte Geometrie)\tAlt+C"), lambda: self.viewport.convert_generators()),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            obj_menu.addAction(action)
        view_menu = self.menuBar().addMenu(tr("&Ansicht"))
        view_menu.addAction(self.dark_action)
        view_menu.addAction(self.notes_visible_action)
        view_menu.addAction(self.planes_action)
        view_menu.addAction(self.xray_action)
        view_menu.addAction(self.wire_action)
        bands_action = view_menu.addAction(tr("Licht-Stufen …"))  # slider; also via right-click on the light icon
        bands_action.triggered.connect(lambda: self.viewport.show_light_bands(QCursor.pos()))
        start_action = view_menu.addAction(tr("Startanimation"))
        start_action.setCheckable(True)
        start_action.setChecked(startup.enabled())
        start_action.toggled.connect(startup.set_enabled)
        lang_menu = view_menu.addMenu("Sprache / Language")
        lang_group = QActionGroup(self)  # exactly one language: a click selects it, the check mark of the other goes off
        lang_group.setExclusive(True)
        for code, label in i18n.LANGUAGES:
            action = lang_group.addAction(label)
            action.setCheckable(True)
            action.setChecked(code == i18n.saved_language())
            action.triggered.connect(lambda _=False, c=code: self._set_language(c))
            lang_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(self._ink_color_action(view_menu))
        view_menu.addAction(self._point_size_action())
        view_menu.addAction(self.compress_action)
        view_menu.addSeparator()
        self.pin_note_action = view_menu.addAction(tr("Notizbrett in jedem Modus"))
        self.pin_note_action.setToolTip(tr("Das Panel Notiz bleibt auch beim Zeichnen, Malen und im Edit sichtbar (wie 📌 im Panel)"))
        self.pin_note_action.setCheckable(True)
        self.pin_note_action.setChecked(self._note_pinned)
        self.pin_note_action.toggled.connect(self.set_note_pinned)
        reset_action = view_menu.addAction(tr("Arbeitsbereich zurücksetzen"))
        reset_action.setToolTip(tr("Outliner, Notiz, Shots, Timeline und die Leisten wieder an ihren Platz – auch wenn "
                                   "ein Panel abgekoppelt oder geschlossen wurde"))
        reset_action.triggered.connect(self.reset_workspace)
        rec_menu = view_menu.addMenu(tr("Aufnahme"))
        self.rec_action = rec_menu.addAction(tr("Aufnahme starten"))
        self.rec_action.setToolTip(tr("Schreibt jede Eingabe mit Zeitstempel und Rechendauer mit (zum Auswerten der eigenen Arbeit)"))
        self.rec_action.triggered.connect(self.toggle_recording)
        self.rec_obs_action = rec_menu.addAction(tr("OBS mitstarten"))
        self.rec_obs_action.setToolTip(tr("Startet und stoppt die OBS-Aufnahme mit – Protokoll und Video laufen dann synchron"))
        self.rec_obs_action.setCheckable(True)
        self.rec_obs_action.setChecked(recorder.saved_obs_choice())
        self.rec_obs_action.toggled.connect(recorder.save_obs_choice)
        rec_folder_action = rec_menu.addAction(tr("Aufnahmen-Ordner öffnen"))
        rec_folder_action.triggered.connect(lambda: _open_folder(recorder.recordings_dir()))
        review_menu = view_menu.addMenu(tr("Review"))
        self.review_action = QAction(tr("Review starten"), self)  # deliberately without a key (F9 held: crash 14.11.)
        self.review_action.setToolTip(tr("Durch die Szene gehen und Feedback einsprechen: Mule nimmt deine Stimme auf "
                                         "und merkt sich Kamera und Cursor – Claude macht daraus Entwurfs-Notizen"))
        self.review_action.triggered.connect(self.toggle_review)
        review_menu.addAction(self.review_action)
        self._review_mark_action()  # the button next to "Review" uses it too (built beforehand)
        review_menu.addAction(self.review_mark_action)
        self.review_unmark_action = QAction(tr("Letzte Marke zurücknehmen\tStrg+Z beim Markieren"), self)
        self.review_unmark_action.triggered.connect(lambda: self.review.undo_mark())
        self.review_unmark_action.setEnabled(False)
        review_menu.addAction(self.review_unmark_action)
        self.addAction(self.review_action)
        self._mic_menu = review_menu.addMenu(tr("Mikrofon"))
        self._mic_menu.aboutToShow.connect(lambda: self._fill_mic_menu(self._mic_menu))
        review_folder_action = review_menu.addAction(tr("Review-Ordner öffnen"))
        review_folder_action.triggered.connect(lambda: _open_folder(review.reviews_dir()))
        help_menu = self.menuBar().addMenu(tr("&Hilfe"))
        manual_action = QAction(tr("&Handbuch"), self, shortcut=QKeySequence("F1"))
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)
        from . import intro
        help_menu.addAction(tr("&Einführung in Bildern"), lambda: intro.show_intro(self))
        help_menu.addAction(tr("Mit der eigenen &KI fragen (AI_HANDBOOK.md) …"), lambda: intro.show_ai_handbook(self))
        help_menu.addSeparator()
        from . import data_info
        help_menu.addAction(tr("Wo Mule &Dateien ablegt …"), lambda: data_info.show(self))

    def show_manual(self):
        """Manual (Help -> Manual, F1) - opens at the tab of the current mode."""
        if getattr(self, "_manual", None) is None:
            from .manual import ManualWindow
            self._manual = ManualWindow(self)
        vp = self.viewport
        self._manual.open_at("Shots" if vp.timing_mode else tr("Notizen") if vp.notes_mode else "Edit" if vp.select_mode
                             else tr("Zeichnen"))

    def _over_outliner(self):
        o = self.outliner
        return o.isVisible() and o.rect().contains(o.mapFromGlobal(QCursor.pos()))

    def _paste_mode(self):
        return "objects" if self._over_outliner() else "elements"

    def _build_toolbar(self):
        settings = QSettings("SketchMachine", "SketchMachine")
        for attr in ("snap_angle", "snap_scale", "snap_move"):  # snap steps (right-click menu)
            setattr(self.ws, attr, float(settings.value(attr, getattr(self.ws, attr))))
        vp_focus = lambda: self.viewport.setFocus()

        def act(icon_name, tip, checkable=True, text=None, group=None):
            a = QAction(ui.icon(icon_name) if icon_name else QIcon(), text or tip.split(":")[0], self, checkable=checkable)
            a.setToolTip(tip)
            if group is not None:
                group.addAction(a)
            return a

        # --- top: base state Draw | Select, on the right globals ---
        top = QToolBar(tr("Modus"), self)
        top.setObjectName("modes")
        top.setMovable(False)
        top.setToolButtonStyle(Qt.ToolButtonTextOnly)
        modes = QActionGroup(self)
        self.draw_mode_action = QAction(tr("Zeichnen"), self, checkable=True, checked=True)
        self.draw_mode_action.setToolTip(tr("Zeichnen: Stift/Pinsel (A), Radierer (D), F färbt die aktive Fläche ein – Tab wechselt zu Edit"))
        self.select_action = QAction("Edit", self, checkable=True)
        self.select_action.setToolTip(tr("Edit: wählen, transformieren, füllen, extrudieren (Tab wechselt wie in Blender)"))
        self.notes_mode_action = QAction(tr("Notizen"), self, checkable=True)
        self.notes_mode_action.setToolTip(tr("Notizen (N): Überschriften im Raum, mit Hinweislinien an Linien und Punkten, dahinter ein Notizbrett mit Text und Bildern – lokale Prompts"))
        self.timing_mode_action = QAction("Shots", self, checkable=True)
        self.timing_mode_action.setToolTip(tr("Shots: Flipbook, Shot-Notizen, Kameras – plus Motion-Spannen (Bind, Pfad, Kamerawinkel)"))
        for a in (self.draw_mode_action, self.select_action, self.notes_mode_action, self.timing_mode_action):
            modes.addAction(a)
            top.addAction(a)
        self.timing_mode_action.triggered.connect(lambda: (self.viewport.set_timing_mode(True), vp_focus()))
        self.draw_mode_action.triggered.connect(lambda: (self.viewport.set_select_mode(False), vp_focus()))
        self.select_action.triggered.connect(lambda: (self.viewport.set_select_mode(True), vp_focus()))
        self.draw_mode_action.triggered.connect(lambda: (self.viewport.set_notes_mode(False), vp_focus()))
        self.notes_mode_action.triggered.connect(lambda: (self.viewport.set_notes_mode(True), vp_focus()))
        self._build_cleanup_menu(top)
        self._build_review_button(top)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top.addWidget(spacer)
        top.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.planes_action = act("planes", tr("Zeichenhilfen ein-/ausblenden: eigene Ebenen und Zeichenflächen"))
        self.planes_action.setChecked(True)
        self.planes_action.toggled.connect(lambda on: self.viewport.set_planes_visible(on))
        self.xray_action = act("xray", tr("X-Ray: alles halbtransparent, Verdecktes sanft sichtbar"))
        self.xray_action.toggled.connect(lambda on: self.viewport.set_xray(on))
        self.wire_action = act("wire", tr("Drahtgitter: im X-Ray die Kanten importierter Meshes dazu – Rechtsklick: Deckkraft"))
        self.wire_action.toggled.connect(lambda on: (self.viewport.set_wireframe(on), vp_focus()))
        self.wire_slider = QSlider(Qt.Horizontal)
        self.wire_slider.setRange(0, 100)
        self.wire_slider.setValue(round(self.ws.wire_opacity * 100))
        self.wire_slider.setFixedWidth(140)
        self.wire_slider.setToolTip(tr("Drahtgitter: Deckkraft der Kanten"))
        self.wire_slider.valueChanged.connect(lambda v: self.viewport.set_wire_opacity(v / 100))
        self.wire_menu = QMenu(self)
        wire_box = QWidget()
        wire_lay = QHBoxLayout(wire_box)
        wire_lay.setContentsMargins(10, 6, 10, 6)
        wire_label = QLabel()
        wire_label.setPixmap(ui.icon("wire").pixmap(18, 18))
        wire_lay.addWidget(wire_label)
        wire_lay.addWidget(self.wire_slider)
        wire_holder = QWidgetAction(self)
        wire_holder.setDefaultWidget(wire_box)
        self.wire_menu.addAction(wire_holder)
        # X-Ray opacity: right-click on the X-Ray icon
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(60)
        self.opacity_slider.setFixedWidth(140)
        self.opacity_slider.setToolTip(tr("X-Ray: Deckkraft der vorderen Schicht (Verdecktes 40 % davon)"))
        self.opacity_slider.valueChanged.connect(lambda v: self.viewport.set_xray_opacity(v / 100))
        self.mesh_menu = QMenu(self)
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(10, 6, 10, 6)
        mesh_label = QLabel()
        mesh_label.setPixmap(ui.icon("xray").pixmap(18, 18))
        lay.addWidget(mesh_label)
        lay.addWidget(self.opacity_slider)
        holder = QWidgetAction(self)
        holder.setDefaultWidget(box)
        self.mesh_menu.addAction(holder)
        # material scale of the surfaces (logarithmic 1/10 ... 10, middle = 1)
        mat_box = QWidget()
        mat_lay = QHBoxLayout(mat_box)
        mat_lay.setContentsMargins(10, 6, 10, 6)
        mat_lay.addWidget(QLabel("Material"))
        self.material_slider = QSlider(Qt.Horizontal)
        self.material_slider.setRange(-100, 100)
        self.material_slider.setFixedWidth(140)
        self.material_slider.setToolTip(tr("Material-Maßstab der Flächen: links feiner, rechts gröber (nur Anzeige)"))
        start = float(settings.value("material_scale", 1.0))
        self.material_slider.setValue(round(math.log10(max(start, 0.1)) * 100))
        self.viewport.set_material_scale(start)

        def on_material(v):
            scale = 10 ** (v / 100)
            self.viewport.set_material_scale(scale)
            settings.setValue("material_scale", scale)
        self.material_slider.valueChanged.connect(on_material)
        mat_lay.addWidget(self.material_slider)
        mat_holder = QWidgetAction(self)
        mat_holder.setDefaultWidget(mat_box)
        self.mesh_menu.addAction(mat_holder)
        # viewport quality: resolution of all surfaces (x 0.25 ... 4, logarithmic, middle = 1), saved per machine -
        # at the top of the bar
        res_label = QLabel()
        res_label.setPixmap(ui.icon("mesh").pixmap(18, 18))
        res_label.setToolTip(tr("Auflösung aller Flächen"))
        self.res_slider = QSlider(Qt.Horizontal)
        self.res_slider.setRange(0, 100)
        self.res_slider.setFixedWidth(90)
        self.res_slider.setTracking(False)  # recompute only on release (all surfaces)
        self.res_slider.setToolTip(tr("Auflösung aller Flächen (Viewport-Qualität): multipliziert den Regler jeder Fläche im Outliner. Nur das angezeigte Netz – die Form bleibt, Rundungen werden jedes Mal neu berechnet. Der Export hat seinen eigenen Regler."))
        res_value = QLabel()
        res_value.setMinimumWidth(40)
        start = min(max(float(settings.value("surface_res", 1.0)), 0.25), 4.0)
        self.res_slider.setValue(round((math.log(start) / math.log(16) + 0.5) * 100))
        res_value.setText(f"× {start:.2g}")
        self.viewport.set_display_resolution(start)

        def on_res(v):
            factor = 0.25 * 16 ** (v / 100)
            res_value.setText(f"× {factor:.2g}")
            self.viewport.set_display_resolution(factor)
            settings.setValue("surface_res", factor)
        self.res_slider.valueChanged.connect(on_res)
        self.res_slider.sliderMoved.connect(lambda v: res_value.setText(f"× {0.25 * 16 ** (v / 100):.2g}"))
        self.res_slider.sliderReleased.connect(vp_focus)
        # Color Compress: make grays (brush, tinting) readable against the background (display only) - in the
        # View menu (below Point size)
        compress_box = QWidget()
        compress_lay = QHBoxLayout(compress_box)
        compress_lay.setContentsMargins(12, 4, 12, 4)
        compress_lay.addWidget(QLabel("Color Compress"))
        self.compress_slider = QSlider(Qt.Horizontal)
        self.compress_slider.setRange(0, 100)
        self.compress_slider.setFixedWidth(140)
        self.compress_slider.setToolTip(tr("Color Compress: Graustufen lesbar machen – auf der Kreidetafel wird der dunkelste Ton angehoben, auf dem Papier der hellste abgesenkt (nur Anzeige)"))
        self.compress_slider.setValue(round(float(settings.value("color_compress", 0.0)) * 100))
        theme.COLOR_COMPRESS = self.compress_slider.value() / 100
        self.compress_slider.valueChanged.connect(self._set_compress)
        compress_lay.addWidget(self.compress_slider)
        self.compress_action = QWidgetAction(self)
        self.compress_action.setDefaultWidget(compress_box)
        self.dark_action = act("moon", tr("Kreidetafel: dunkler Viewport mit Kreidestrichen (Flächen bleiben gleich)"))
        self.dark_action.toggled.connect(lambda on: self._set_dark(on))
        self.color_view_action = act("palette", tr("Farbansicht: Flächen und Pinsel in Farbe – aus: alles in Schwarz-Weiß"))
        self.color_view_action.setChecked(settings.value("color_view", True, type=bool))
        theme.COLOR_VIEW = self.color_view_action.isChecked()
        self.color_view_action.toggled.connect(lambda on: (self._set_color_view(on), vp_focus()))
        self.notes_visible_action = act("note", tr("Notizen ein-/ausblenden"))
        self.notes_visible_action.setChecked(True)
        self.notes_visible_action.toggled.connect(lambda on: (self.viewport.set_notes_visible(on), vp_focus()))
        for a in (self.notes_visible_action, self.planes_action, self.xray_action, self.wire_action):
            top.addAction(a)
        self._build_layers_button(top)
        top.addSeparator()
        xray_button = top.widgetForAction(self.xray_action)
        xray_button.setContextMenuPolicy(Qt.CustomContextMenu)
        xray_button.customContextMenuRequested.connect(lambda pos: self.mesh_menu.popup(xray_button.mapToGlobal(pos)))
        wire_button = top.widgetForAction(self.wire_action)
        wire_button.setContextMenuPolicy(Qt.CustomContextMenu)
        wire_button.customContextMenuRequested.connect(lambda pos: self.wire_menu.popup(wire_button.mapToGlobal(pos)))
        self.xray_action.setToolTip(tr("X-Ray: alles halbtransparent, Verdecktes sanft sichtbar – Rechtsklick: Deckkraft"))
        top.addWidget(res_label)
        top.addWidget(_spacer(4))
        top.addWidget(self.res_slider)
        top.addWidget(res_value)
        top.addWidget(_spacer(8))
        top.addSeparator()
        top.addAction(self.color_view_action)  # color / black-and-white next to paper / chalkboard
        top.addAction(self.dark_action)
        if settings.value("viewport_dark", False, type=bool):
            self.dark_action.setChecked(True)
        self.addToolBar(Qt.TopToolBarArea, top)
        for a in top.actions():  # buttons with an icon show only the icon, mode switches only text
            w = top.widgetForAction(a)
            if w is not None and a in (self.draw_mode_action, self.select_action, self.notes_mode_action):
                w.setToolButtonStyle(Qt.ToolButtonTextOnly)

        # --- left: toolbar, depending on the mode ---
        rail = QToolBar(tr("Werkzeuge"), self)
        rail.setObjectName("rail")
        rail.setMovable(False)
        rail.setOrientation(Qt.Vertical)
        rail.setIconSize(QSize(26, 26))
        rail.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.rail = rail
        self.draw_items, self.select_items = [], []

        def add(items, *things):
            for t in things:
                if isinstance(t, QAction):
                    rail.addAction(t)
                    items.append(t)
                else:
                    items.append(rail.addWidget(t))

        def sep(items):
            items.append(rail.addSeparator())

        tools = QActionGroup(self)
        self.draw_action = act("pen", tr("Stift (A – nochmal A: Pinsel)"), group=tools)
        self.brush_action = act("brush", tr("Pinsel (A – nochmal A: Stift): zeichnet wie der Stift, in Farbe mit Ölpinsel-Textur"), group=tools)
        self.brush_action.triggered.connect(lambda: (self.viewport.set_draw_tool("brush"), vp_focus()))
        self.draw_action.setChecked(True)
        self.erase_action = act("eraser", tr("Radierer (D): Linien und Malerei, Strg nur Linien, P durch alles"), group=tools)
        self.draw_action.triggered.connect(lambda: (self.viewport.set_draw_tool("pen"), vp_focus()))
        self.erase_action.triggered.connect(lambda: (self.viewport.set_draw_tool("eraser"), vp_focus()))
        self.paint_mode_action = act("paintmode", tr("Malmodus (S): Stift und Pinsel malen Farbe flach auf die Fläche – keine editierbaren Linien, der neueste Strich liegt oben"))
        self.paint_mode_action.toggled.connect(lambda on: (self.viewport.set_paint_mode(on), vp_focus()))
        self.free_action = act("free", tr("Frei-Modus (Q): an einer Linie oder einer Fläche ansetzen, gezeichnet wird in der Ebene zur Kamera durch diesen Punkt"))
        self.free_action.toggled.connect(lambda on: (self.viewport.set_free_draw(on), vp_focus()))
        self.multi_action = act("multi", tr("Multi (B): auf mehrere Flächen zugleich zeichnen – ein Klick nimmt Flächen dazu/heraus, Klick ins Leere leert den Satz. Aus (Standard): nur die aktive Fläche, ein Klick wechselt sie"))
        self.multi_action.toggled.connect(lambda on: self.viewport.set_multi_mode(on))
        self.magnet_action = act("magnet", tr("Magnet: Strichanfang und -ende rasten an Linien ein (Fangradius in Pixeln) – aus: frei zeichnen"))
        self.magnet_action.setChecked(True)
        self.magnet_action.toggled.connect(lambda on: (self.viewport.set_magnet(on), vp_focus()))
        add(self.draw_items, self.draw_action, self.brush_action, self.erase_action)
        sep(self.draw_items)
        # switches that modify the tools (not tools themselves): paint mode, Free, Multi, Magnet
        add(self.draw_items, self.paint_mode_action, self.free_action, self.multi_action, self.magnet_action)
        sep(self.draw_items)
        # tool options: pen/brush = Smooth
        self.smooth_slider = self._rail_slider(tr("Smooth: Stiftspitze hängt an einer Schnur hinter dem Cursor (0 = aus)"))
        self.smooth_slider.valueChanged.connect(lambda v: self.viewport.set_smooth(v / 100))
        self.viewport.smooth_max_factor = float(settings.value("smooth_max_factor", self.viewport.smooth_max_factor))
        self.viewport.set_pen_max_world(settings.value("pen_max_world", self.viewport.pen_max_world))
        smooth_icon = QLabel()
        smooth_icon.setPixmap(ui.icon("smooth").pixmap(20, 20))
        smooth_icon.setToolTip(self.smooth_slider.toolTip())
        smooth_icon.setAlignment(Qt.AlignCenter)
        self.pen_options = [smooth_icon, self.smooth_slider]
        # color (brush, tinting surfaces with F): current color -> color wheel with the recent colors
        recent = settings.value("recent_colors", [], type=list) or []
        self.ws.recent_colors = [tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) for h in recent
                                 if isinstance(h, str) and len(h) == 7][:theme.RECENT_COLORS]
        self.color_action = QAction(self)
        self.color_action.setToolTip(tr("Farbe von Pinsel und Flächen (F im Zeichenmodus färbt ein) – Klick: Farbrad und die letzten Farben"))
        self.color_action.triggered.connect(self._open_color_popup)
        self.swap_action = act("swap", tr("Felder umschalten: Grautöne ⇄ zuletzt benutzte Farben"), checkable=False)
        self.swap_action.triggered.connect(self._swap_swatches)
        self.tone_actions = []
        tone_group = QActionGroup(self)
        for k in range(len(theme.GRAY_TONES)):
            a = QAction(self, checkable=True, checked=k == 0)
            a.triggered.connect(lambda _=False, i=k: (self._pick_swatch(i), vp_focus()))
            tone_group.addAction(a)
            self.tone_actions.append(a)
        self._color_popup = None
        # pressure sensitivity (graphics tablet) per tool, saved in the settings
        for tool in self.ws.pressure:
            self.ws.pressure[tool] = float(settings.value(f"pressure_{tool}", 0.0))
        self.pressure_slider = self._rail_slider(tr("Druckempfindlichkeit"))
        self.pressure_slider.valueChanged.connect(self._set_pressure)
        pressure_icon = QLabel()
        pressure_icon.setPixmap(ui.icon("pressure").pixmap(20, 20))
        pressure_icon.setAlignment(Qt.AlignCenter)
        self.pressure_icon = pressure_icon
        option_items = []
        for wdg in self.pen_options:
            option_items.append(rail.addWidget(wdg))
        self.pen_option_items = option_items[:]
        rail.addAction(self.color_action)  # only the current color; grays/recent colors in the color wheel (swatches gone)
        self.tone_items = [self.color_action]
        self._refresh_swatches()
        self.pressure_items = [rail.addWidget(self.pressure_icon), rail.addWidget(self.pressure_slider)]
        self.draw_items += self.pen_option_items + self.pressure_items  # color: always visible

        transforms = QActionGroup(self)
        transforms.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        self.transform_actions = {}
        for tool, name, tip in (("translate", "translate", tr("Verschieben (T)")), ("rotate", "rotate", tr("Drehen (R)")),
                                ("scale", "scale", tr("Skalieren (E)"))):
            a = act(name, tip, group=transforms)
            a.triggered.connect(lambda on, t=tool: (self.viewport.set_transform_tool(t if on else None), vp_focus()))
            self.transform_actions[tool] = a
        self.axis_action = act("axis", tr("Achsenmodus (Alt+A): das Gizmo verschiebt/dreht nur den Pivot der Auswahl"))
        self.axis_action.toggled.connect(lambda on: self.viewport.set_axis_mode(on))
        self.local_action = act("local", tr("Lokal statt Welt ausrichten (Space+M schaltet Welt, Lokal, Kamera durch)"))
        self.local_action.toggled.connect(lambda on: (self.viewport.set_local_space(on), vp_focus()))
        add(self.select_items, *self.transform_actions.values())
        sep(self.select_items)
        self.view_space_action = act("view_space", tr("Aus Kameraperspektive ausrichten: Achsen wie die Ansicht, für Verschieben, Drehen und Skalieren (Space+M schaltet Welt, Lokal, Kamera durch)"))
        self.view_space_action.toggled.connect(lambda on: (self.viewport.set_view_space(on), vp_focus()))
        add(self.select_items, self.axis_action, self.local_action, self.view_space_action)
        sep(self.select_items)
        comp_group = QActionGroup(self)
        self.component_actions = {}
        for key, (comp, tip) in enumerate((
                ("point", tr("Punkte wählen (1): Linienenden und Ecken; verbundene Linien verformen sich weich mit")),
                ("edge", tr("Kanten wählen (2): Linien bzw. Stücke zwischen zwei Punkten")),
                ("face", tr("Flächen wählen (3): Flächen (Doppelklick: samt Linien darauf)"))), start=1):
            a = act(comp, tip, group=comp_group)
            a.setChecked(comp == "edge")
            a.triggered.connect(lambda _=False, c=comp: (self.viewport.set_component(c), vp_focus()))
            self.component_actions[comp] = a
        add(self.select_items, *self.component_actions.values())
        # drag points: automatic / soft / hard (point and edge mode); hardness (K) and smoothing (J) only via key
        knot_group = QActionGroup(self)
        self.knot_actions = {}
        for mode, tip in (("auto", tr("Punkte ziehen: automatisch – Punkte auf Rundungen weich, an Ecken hart (Standard)")),
                          ("soft", tr("Punkte ziehen: immer weich – die Linie biegt sich ohne Knick bis zu den Nachbarpunkten")),
                          ("hard", tr("Punkte ziehen: immer hart – geradlinig bis zu den Nachbarpunkten, mit Knick am Punkt"))):
            a = act(f"knot_{mode}", tip, group=knot_group)
            a.setChecked(mode == "auto")
            a.triggered.connect(lambda _=False, m=mode: (self.viewport.set_knot_mode(m), vp_focus()))
            self.knot_actions[mode] = a
        self.knot_sep = rail.addSeparator()
        for a in self.knot_actions.values():
            rail.addAction(a)
        self.addToolBar(Qt.LeftToolBarArea, rail)

        self.ws.toggles_changed.connect(self._sync_toggles)
        self.ws.scene_changed.connect(self._sync_toggles)  # T/R/E, Esc etc. by key
        self._sync_toggles()

    def _build_layers_button(self, top):
        """Next to X-Ray: show/hide lines, color lines and painting individually (check marks, the menu stays open).
        The button lights up as long as something is hidden."""
        self.layers_action = QAction(ui.icon("layers"), tr("Ein-/Ausblenden"), self, checkable=True)
        self.layers_action.setToolTip(tr("Ein-/Ausblenden: Linien, Farblinien, Malerei und Flächen einzeln (nur Anzeige)"))
        menu = QMenu(self)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 8, 12, 8)
        self.layer_checks = {}
        for name, text in (("lines", tr("Linien")), ("brush", tr("Farblinien")), ("paint", tr("Malerei")), ("surfaces", tr("Flächen"))):
            check = QCheckBox(text)
            check.setChecked(True)
            check.toggled.connect(lambda on, n=name: self._set_layer(n, on))
            lay.addWidget(check)
            self.layer_checks[name] = check
        holder = QWidgetAction(self)
        holder.setDefaultWidget(box)
        menu.addAction(holder)
        self.layers_action.setMenu(menu)
        top.addAction(self.layers_action)
        button = top.widgetForAction(self.layers_action)
        button.setPopupMode(QToolButton.InstantPopup)
        button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")

    def _set_layer(self, name, on):
        self.viewport.set_layer_visible(name, on)
        self.layers_action.setChecked(not all(c.isChecked() for c in self.layer_checks.values()))

    def _build_cleanup_menu(self, top):
        """Edit: "Cleanup" menu at the top - tidy-up functions for the selection; "Global" = on everything (button glows then)."""
        from PySide6.QtWidgets import QToolButton
        button = QToolButton()
        button.setText("Cleanup")
        button.setPopupMode(QToolButton.InstantPopup)
        button.setToolTip(tr("Aufräumen – wirkt auf die Auswahl; mit „Global“ auf die ganze Szene"))
        menu = QMenu(button)
        self.cleanup_global_action = menu.addAction(tr("Global – auf die ganze Szene anwenden"))
        self.cleanup_global_action.setCheckable(True)
        self.cleanup_global_action.toggled.connect(lambda on: self.viewport.set_cleanup_global(on))
        # snap range: how close (in space, world units) ends or an end and a line must be to be welded,
        # docked or merged - independent of zoom; logarithmic slider
        gap_box = QWidget()
        gap_lay = QHBoxLayout(gap_box)
        gap_lay.setContentsMargins(10, 4, 10, 4)
        gap_lay.addWidget(QLabel(tr("Fangweite")))
        self.cleanup_dist_slider = QSlider(Qt.Horizontal)
        self.cleanup_dist_slider.setRange(0, 100)
        self.cleanup_dist_slider.setFixedWidth(120)
        self.cleanup_dist_slider.setToolTip(tr("Lücken schließen, Verschweißen, Verschmelzen: Enden (bzw. Ende und Linie), die im Raum näher als das liegen, werden verbunden – unabhängig vom Zoom"))
        gap_value = QLabel()
        gap_value.setMinimumWidth(48)
        settings = QSettings("SketchMachine", "SketchMachine")
        lo, hi = CLEANUP_DIST_RANGE
        to_dist = lambda v: lo * (hi / lo) ** (v / 100)
        self.ws.cleanup_dist = min(max(float(settings.value("cleanup_dist", self.ws.cleanup_dist)), lo), hi)
        self.cleanup_dist_slider.setValue(round(math.log(self.ws.cleanup_dist / lo) / math.log(hi / lo) * 100))
        gap_value.setText(_length_text(self.ws.cleanup_dist))

        def on_gap(v):
            self.ws.cleanup_dist = to_dist(v)
            gap_value.setText(_length_text(self.ws.cleanup_dist))
            QSettings("SketchMachine", "SketchMachine").setValue("cleanup_dist", self.ws.cleanup_dist)
        self.cleanup_dist_slider.valueChanged.connect(on_gap)
        gap_lay.addWidget(self.cleanup_dist_slider)
        gap_lay.addWidget(gap_value)
        gap_holder = QWidgetAction(menu)
        gap_holder.setDefaultWidget(gap_box)
        menu.addAction(gap_holder)
        menu.addSeparator()
        for key, text in (("all", tr("Aufräumen: verschweißen + verschmelzen")), ("weld", tr("Verschweißen (Lücken schließen)")),
                          ("crossings", tr("Enden und Kreuzungen verbinden (lose Enden nah an Linien, Kreuzungen im Raum)")),
                          ("fuse", tr("Verschmelzen (geteilte Linien wieder zu einer)")), (None, None),
                          ("straighten", tr("Fast gerade Stücke begradigen")), ("dissolve", tr("Eingefügte Punkte auflösen")),
                          ("duplicates", tr("Doppelte Linien entfernen")), ("tiny", tr("Winzige Linien und Punkte löschen")),
                          ("empty", tr("Leere Stroke-Objekte löschen")), (None, None),
                          ("relink", tr("Flächen wieder an ihre Linien hängen"))):
            if key is None:
                menu.addSeparator()
                continue
            menu.addAction(text, lambda k=key: self.viewport.run_cleanup(k))
        button.setMenu(menu)
        self.cleanup_button = button
        self.cleanup_item = top.addWidget(button)

    def _review_mark_action(self):
        """Action of the "Mark" tool (menu and button), created once."""
        if getattr(self, "review_mark_action", None) is not None:
            return self.review_mark_action
        # "Mark" tool: on at start, every click into a view or onto a board sets a mark
        self.review_mark_action = QAction(tr("Markieren"), self, checkable=True)
        self.review_mark_action.setToolTip(tr("Während des Reviews: jeder Klick in die Ansicht oder auf ein Bild im Brett "
                                              "setzt eine Marke (der Ring zeigt vorher, was getroffen wird). Navigieren "
                                              "mit Alt oder Leertaste wie immer. Aus: normal weiterarbeiten, das Review "
                                              "nimmt weiter auf"))
        self.review_mark_action.toggled.connect(lambda on: self.review.set_marking(on))
        self.review_mark_action.setEnabled(False)
        return self.review_mark_action

    def _build_review_button(self, top):
        """Notes: "Review" menu next to Shots - start/end, Here!, microphone, folder; visible in Notes mode and
        while a review is running (then red with elapsed time)."""
        from PySide6.QtWidgets import QToolButton
        button = QToolButton()
        button.setText(tr("Review"))
        button.setPopupMode(QToolButton.InstantPopup)
        button.setToolTip(tr("Review: durch die Szene gehen und Feedback einsprechen – Claude macht daraus Entwurfs-Notizen"))
        self.review_menu = QMenu(button)
        self.review_menu.aboutToShow.connect(self._fill_review_menu)
        button.setMenu(self.review_menu)
        self.review_button = button
        self.review_item = top.addWidget(button)
        mark = QToolButton()  # "Mark" tool - visible only during the review
        mark.setDefaultAction(self._review_mark_action())
        mark.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.review_mark_button = mark
        self.review_mark_item = top.addWidget(mark)
        self.review_mark_item.setVisible(False)
        send = QToolButton()
        send.setText(tr("Review absenden"))
        send.clicked.connect(self.send_review)
        self.review_send_button = send
        self.review_send_item = top.addWidget(send)

    def _fill_review_menu(self):
        menu = self.review_menu
        menu.clear()
        menu.addAction(self.review_action)
        menu.addAction(self.review_mark_action)
        menu.addAction(self.review_unmark_action)
        menu.addSeparator()
        mics = menu.addMenu(tr("Mikrofon"))
        self._fill_mic_menu(mics)
        ready = review.whisper_python() is not None
        menu.addAction(tr("Transkription eingerichtet ✓ – neu einrichten …") if ready else
                       tr("Transkription einrichten …"), self.open_whisper_setup)
        last = review.find_review()
        info = menu.addAction(tr("Letztes Review: {0}").format(os.path.basename(last)) if last else
                              tr("Noch kein Review aufgenommen"))
        info.setEnabled(False)
        menu.addAction(tr("Review-Ordner öffnen"), lambda: _open_folder(review.reviews_dir()))

    def send_review(self):
        if self.review_sender.phase == "working" and not self.review.active:  # click "Claude is working": off
            self.review_sender.stop_waiting()
            return
        status, folder = self.review_sender.send()
        if status == "none":
            self.statusBar().showMessage(tr("Noch kein Review aufgenommen – Review → Review starten"), 5000)
        elif status == "transcribing":
            self.statusBar().showMessage(tr("Review abgeschickt – wird transkribiert, danach sag Claude Bescheid"), 8000)
        elif status == "ready":
            self.statusBar().showMessage(tr("Review bereit – im Claude-Chat „Review ist da“ schreiben (der Satz liegt in "
                                            "der Zwischenablage, Strg+V)"), 15000)

    def _spin_send(self):
        self._send_frame += 1
        self._style_review()

    def _review_sent(self, name, made):
        self.statusBar().showMessage(tr("Review „{0}“: {1} Entwurf/Entwürfe eingefügt – im Brett ✓ Übernehmen, Strg+Z "
                                        "nimmt alle zurück").format(name, made), 20000)

    def _style_review(self):
        if getattr(self, "review", None) is None or getattr(self, "review_button", None) is None:
            return  # when building the window
        sender = getattr(self, "review_sender", None)
        phase = sender.phase if sender is not None else "idle"
        sending = phase != "idle"
        button = self.review_send_button
        if phase in ("transcribing", "working"):
            if not self._send_spin.isActive():
                self._send_spin.start()
            wait = tr("Transkribiere …") if phase == "transcribing" else tr("Claude arbeitet …")
            button.setText("◐◓◑◒"[self._send_frame % 4] + " " + wait)
            if phase == "working":
                button.setToolTip(tr("Claude ruft gerade Mule-Werkzeuge auf. Kommt zwei Minuten nichts mehr, geht die "
                                     "Anzeige von selbst aus. Klick: Anzeige jetzt ausschalten"))
            button.setStyleSheet(f"QToolButton {{ color: {ui.ACCENT}; font-weight: bold; }}")
        else:
            if getattr(self, "_send_spin", None) is not None:
                self._send_spin.stop()
            if phase == "ready":
                button.setText(tr("Bereit – sag Claude Bescheid"))
                button.setToolTip(tr("Im Claude-Chat „Review ist da“ schreiben – der Satz liegt in der Zwischenablage "
                                     "(Strg+V). Nochmal klicken: Satz erneut kopieren"))
                button.setStyleSheet(f"QToolButton {{ color: {ui.ACCENT}; }}")
            else:
                button.setText(tr("Review absenden"))
                button.setToolTip(tr("Review fertig machen: Aufnahme beenden, transkribieren, Backup der Szene "
                                     "(review_backups) – dann im Claude-Chat Bescheid sagen; die Entwürfe erscheinen "
                                     "auf einmal in der offenen Szene"))
                button.setStyleSheet("")
        on = self.review.active
        if on:
            s = int(self.review.elapsed())
            self.review_button.setText(f"● {tr('Review')} {s // 60:d}:{s % 60:02d}")
            self.review_button.setStyleSheet("QToolButton { background: rgba(210, 60, 60, 0.25); "
                                             "border: 1px solid #d23c3c; color: #d23c3c; font-weight: bold; }")
        else:
            self.review_button.setText(tr("Review"))
            self.review_button.setStyleSheet("")
        vp = self.viewport
        shown = on or sending or (vp.notes_mode and not vp.timing_mode)
        self.review_item.setVisible(shown)
        self.review_send_item.setVisible(shown)
        self.review_mark_item.setVisible(on)
        if on:
            self.review_mark_button.setStyleSheet(
                "QToolButton:checked { background: rgba(225, 0, 160, 0.22); border: 1px solid #e100a0; "
                "color: #e100a0; font-weight: bold; }" if self.review.marking else "")

    def _style_cleanup(self, on):
        self.cleanup_button.setStyleSheet(
            f"QToolButton {{ background: rgba(208, 138, 94, 0.28); border: 1px solid {ui.ACCENT}; color: {ui.ACCENT}; }}"
            if on else "")

    # --- Colors ---
    def _swatch_colors(self):
        """Colors of the five swatches: gray tones or the recent colors."""
        if self.ws.swatch_mode == "color":
            recent = list(self.ws.recent_colors[:len(self.tone_actions)])
            return [tuple(c) for c in recent] + [None] * (len(self.tone_actions) - len(recent))
        return [tuple(theme.gray_tone(t)) for t in theme.GRAY_TONES]

    def _refresh_swatches(self):
        rgb255 = lambda c: [int(round(255 * x)) for x in c]
        self.color_action.setIcon(ui.swatch(rgb255(self.ws.paint_color), current=True))
        current = tuple(round(x, 3) for x in self.ws.paint_color)
        for k, (a, c) in enumerate(zip(self.tone_actions, self._swatch_colors())):
            a.setEnabled(c is not None)
            a.setIcon(ui.swatch(rgb255(c) if c is not None else [60, 60, 62]))
            a.setToolTip((tr("Grauton {0}").format(k + 1) if self.ws.swatch_mode == "gray" else tr("Letzte Farbe {0}").format(k + 1))
                         if c is not None else tr("noch keine Farbe benutzt"))
            a.blockSignals(True)
            a.setChecked(c is not None and tuple(round(x, 3) for x in c) == current)
            a.blockSignals(False)

    def _swap_swatches(self):
        self.ws.swatch_mode = "color" if self.ws.swatch_mode == "gray" else "gray"
        self._refresh_swatches()

    def _pick_swatch(self, k):
        c = self._swatch_colors()[k]
        if c is not None:
            self._use_color(c, remember=self.ws.swatch_mode == "color")

    def _use_color(self, rgb, remember=True, persist=True):
        """Set color; remember: into the recent colors (newest first, remembered globally). Choosing the color changes
        nothing in the scene - in edit only "Apply" in the color picker colors the selection (_apply_color)."""
        self.viewport.set_paint_color(rgb)
        if remember:
            recent = [c for c in self.ws.recent_colors if max(abs(a - b) for a, b in zip(c, rgb)) > 1 / 255]
            self.ws.recent_colors = [tuple(rgb)] + recent[:theme.RECENT_COLORS - 1]
            if persist:
                QSettings("SketchMachine", "SketchMachine").setValue(
                    "recent_colors", ["#" + "".join(f"{int(round(x * 255)):02x}" for x in c) for c in self.ws.recent_colors])
        self._refresh_swatches()

    def _open_color_popup(self):
        from .colorpick import ColorPopup
        if self._color_popup is None:
            self._color_popup = ColorPopup(self)
            self._color_popup.picked.connect(lambda rgb, final: self._use_color(rgb, remember=final))
            self._color_popup.applied.connect(self._apply_color)
        button = self.rail.widgetForAction(self.color_action)
        pos = button.mapToGlobal(button.rect().topRight()) if button is not None else self.cursor().pos()
        self._color_popup.show_at(pos, self.ws.paint_color, self.ws.recent_colors, apply=self.viewport.select_mode)

    def _apply_color(self, rgb, reset=False):
        """"Apply" in the color picker (edit): selected surfaces and brush lines get the color."""
        if not reset:
            self._use_color(rgb, remember=True)
        self.viewport.apply_color(rgb, reset=reset)

    def _set_color_view(self, on, persist=True):
        theme.COLOR_VIEW = bool(on)
        if self.color_view_action.isChecked() != on:
            self.color_view_action.blockSignals(True)
            self.color_view_action.setChecked(on)
            self.color_view_action.blockSignals(False)
        if persist:
            QSettings("SketchMachine", "SketchMachine").setValue("color_view", bool(on))
        for vp in self.ws.viewports:
            vp.update()

    def _set_pressure(self, value, persist=True):
        tool = self.viewport.ws.draw_tool
        self.viewport.set_pressure(tool, value / 100)
        if persist:
            QSettings("SketchMachine", "SketchMachine").setValue(f"pressure_{tool}", value / 100)

    def _set_compress(self, value, persist=True):
        theme.COLOR_COMPRESS = value / 100
        if persist:
            QSettings("SketchMachine", "SketchMachine").setValue("color_compress", value / 100)
        for vp in self.ws.viewports:
            vp.update()

    def _set_dark(self, on, persist=True):
        """Viewport light (drawing paper) or dark (chalkboard) - applies to all viewports."""
        theme.set_dark(on)
        if self.dark_action.isChecked() != on:
            self.dark_action.blockSignals(True)
            self.dark_action.setChecked(on)
            self.dark_action.blockSignals(False)
        if persist:
            QSettings("SketchMachine", "SketchMachine").setValue("viewport_dark", bool(on))
        for vp in self.ws.viewports:
            vp.update()

    def _rail_slider(self, tip):
        s = QSlider(Qt.Vertical)
        s.setRange(0, 100)
        s.setFixedHeight(90)
        s.setToolTip(tip)
        s.sliderReleased.connect(lambda: self.viewport.setFocus())  # keyboard shortcuts stay in the viewport
        return s

    def _sync_toggles(self):
        """Update the buttons when a mode was switched by key."""
        vp = self.viewport
        sel = vp.select_mode
        notes = vp.notes_mode
        timing_on = vp.timing_mode
        checks = [(self.timing_mode_action, timing_on), (self.notes_mode_action, notes), (self.notes_visible_action, vp.notes_visible), (self.multi_action, vp.multi_mode), (self.free_action, vp.free_draw),
                  (self.magnet_action, vp.ws.magnet), (self.planes_action, vp.planes_visible),
                  (self.xray_action, vp.xray), (self.wire_action, vp.ws.wireframe), (self.select_action, sel), (self.draw_mode_action, not sel and not notes and not timing_on),
                  (self.axis_action, vp.ws.axis_mode), (self.local_action, vp.gizmo.space == "local"),
                  (self.view_space_action, vp.gizmo.space == "view"),
                  (self.draw_action, vp.ws.draw_tool == "pen"), (self.brush_action, vp.ws.draw_tool == "brush"),
                  (self.erase_action, vp.ws.draw_tool == "eraser"), (self.paint_mode_action, vp.ws.paint_mode)]
        checks += [(a, vp.gizmo.tool == t) for t, a in self.transform_actions.items()]
        checks += [(a, vp.ws.component == c) for c, a in self.component_actions.items()]
        checks += [(a, vp.ws.knot_mode == m) for m, a in self.knot_actions.items()]
        knots_visible = sel and vp.ws.component in ("point", "edge")
        for a in [self.knot_sep, *self.knot_actions.values()]:
            a.setVisible(knots_visible)
        for action, value in checks:
            if action.isChecked() != value:
                action.blockSignals(True)
                action.setChecked(value)
                action.blockSignals(False)
        tool = vp.ws.draw_tool
        for item in self.draw_items:
            item.setVisible(not sel and not notes)
        self.cleanup_item.setVisible(sel)
        self._style_review()
        for item in self.tone_items:  # color (brush, tinting surfaces with F): in every mode except Notes
            item.setVisible(not notes)
        if self.cleanup_global_action.isChecked() != vp.ws.cleanup_global:
            self.cleanup_global_action.blockSignals(True)
            self.cleanup_global_action.setChecked(vp.ws.cleanup_global)
            self.cleanup_global_action.blockSignals(False)
        self._style_cleanup(vp.ws.cleanup_global)
        for item in self.select_items:
            item.setVisible(sel)
        if not sel and not notes:  # options only for the active tool
            for item in self.pen_option_items:
                item.setVisible(tool in ("pen", "brush"))
            tip = {"pen": tr("Druckempfindlichkeit des Stifts: Strichbreite – 0 = immer gleich, 1 = leichteste Berührung 10 %, festeste 200 % der Stiftgröße"),
                   "brush": tr("Druckempfindlichkeit des Pinsels: Strichbreite – 0 = immer gleich, 1 = 10 % bis 200 %"),
                   "eraser": tr("Druckempfindlichkeit des Radierers: Radiergröße – 0 = immer gleich, 1 = 10 % bis 200 %")}[tool]
            self.pressure_slider.setToolTip(tip)
            self.pressure_icon.setToolTip(tip)
            value = round(vp.ws.pressure[tool] * 100)
            if self.pressure_slider.value() != value:
                self.pressure_slider.blockSignals(True)
                self.pressure_slider.setValue(value)
                self.pressure_slider.blockSignals(False)

    def _sync_timing_docks(self):
        """Timeline and Shots panel: only in Shots mode."""
        on = bool(self.ws.timing_mode)
        for dock in (self.timeline_dock, self.timing_dock):
            if dock.isVisible() != on:
                dock.setVisible(on)
        if on and not getattr(self, "_timing_docks_sized", False):
            self._timing_docks_sized = True  # the first time: the Shots panel gets a good half of the height
            outliner = self.findChild(QDockWidget, "outliner_dock")
            if self.dockWidgetArea(outliner) == self.dockWidgetArea(self.timing_dock) and not self.timing_dock.isFloating():
                h = max(self.timing_dock.parentWidget().height(), 400)
                self.resizeDocks([outliner, self.timing_dock], [int(h * 0.35), int(h * 0.65)], Qt.Vertical)
            self.resizeDocks([self.timeline_dock], [150], Qt.Vertical)
        self._sync_shots_panel()
        if on:
            self.timeline.refresh()
        elif self.ws.playing:
            self.timeline.set_playing(False)

    def _sync_shots_panel(self):
        """Shots panel: in Shots mode the clicked shot note (or the global one) - like the Note panel."""
        panel, vp = self.shots_panel, self.viewport
        if panel.committing:
            return
        if not self.ws.timing_mode:
            if panel.note_id is not None:
                panel.commit()
                panel.show_note(vp, None)
            return
        tm = vp.scene.timing
        if self._shot_scope == "global":
            note = tm.global_note
        else:
            note = tm.note_by_id(vp.ws.active_note) if vp.ws.active_note is not None else None
            note = note if note is not tm.global_note else None
        nid = note.id if note is not None else None
        if nid != panel.note_id:
            panel.commit()
            panel.show_note(vp, nid)
        elif note is not None and not panel.edited() and (note.board != panel.loaded or note.title != panel.loaded_title):
            panel.show_note(vp, nid)  # undo, overlay or heading changed in the viewport
        else:
            panel.refresh_shot_rows()  # camera entered/left: ⟲ ➜ on/off

    def _set_shot_scope(self, scope):
        self._shot_scope = scope
        for key, b in self.shots_panel.scope_buttons.items():
            b.setChecked(key == scope)
        self._sync_shots_panel()

    def _maximize_shot(self):
        nid = self.shots_panel.note_id
        if nid is not None:
            from .note_board import open_board
            open_board(self.viewport, nid)

    def _sync_note_panel(self):
        """Note panel: visible in Notes mode, shows the active note (switching takes over changes)."""
        panel, vp = self.note_panel, self.viewport
        if panel.committing:
            return
        notes = vp.notes_mode or self._note_pinned  # pinned: also while drawing, painting and in edit
        if notes != self._notes_shown:
            self._notes_shown = notes
            self.note_dock.setVisible(notes)
            if notes and not self._note_dock_sized and not self.note_dock.isFloating():
                self._note_dock_sized = True  # the first time: the panel gets a good half of the height
                docks = [self.findChild(QDockWidget, "outliner_dock"), self.note_dock]
                if self.dockWidgetArea(docks[0]) == self.dockWidgetArea(self.note_dock):
                    h = max(self.note_dock.parentWidget().height(), 400)
                    self.resizeDocks(docks, [int(h * 0.42), int(h * 0.58)], Qt.Vertical)
        if not notes:
            if panel.note_id is not None:
                panel.commit()
                panel.show_note(vp, None)
            return
        if self._note_scope == "global":
            note = vp.scene.global_note
        else:
            note = vp.scene.note_by_id(vp.ws.active_note) if vp.ws.active_note is not None else None
            if vp.scene.is_shot_note(note):
                note = None  # shot notes are in the Shots panel
        nid = note.id if note is not None else None
        if nid != panel.note_id:
            panel.commit()
            panel.show_note(vp, nid)
        elif note is not None and not panel.edited() and (note.board != panel.loaded or note.title != panel.loaded_title):
            panel.show_note(vp, nid)  # undo, overlay or heading changed in the viewport

    def reset_workspace(self):
        """View -> Reset workspace: all panels and bars docked again at their factory place. The
        outliner is always there; Note, Shots and timeline show as usual by mode (Notes or pin, Shots)."""
        docks = [self.findChild(QDockWidget, "outliner_dock"), self.note_dock, self.timing_dock, self.timeline_dock]
        for dock in docks:
            dock.setFloating(False)
        self.restoreState(self._default_state)
        docks[0].show()
        for dock in docks[1:]:  # at the factory they were open when remembered: by mode again
            dock.hide()
        self._notes_shown = False
        self._note_dock_sized = self._timing_docks_sized = False  # redistribute heights as on first opening
        self._sync_note_panel()
        self._sync_timing_docks()
        self.viewport.setFocus()
        self.viewport.notify(tr("Arbeitsbereich zurückgesetzt"))

    def set_note_pinned(self, on):
        """Pin the note board: the Note panel stays visible in every mode (pin in the panel, View menu)."""
        on = bool(on)
        self._note_pinned = on
        QSettings("SketchMachine", "SketchMachine").setValue("note_pinned", on)
        for widget in (self.note_panel.pin_button, getattr(self, "pin_note_action", None)):
            if widget is not None and widget.isChecked() != on:
                widget.blockSignals(True)
                widget.setChecked(on)
                widget.blockSignals(False)
        self._notes_shown = not (self.viewport.notes_mode or on)  # forces show/hide, also after the close button of the panel
        self._sync_note_panel()

    # --- Recording ---
    def toggle_recording(self):
        if self.recorder.active:
            self.recorder.stop()
        else:
            self.recorder.start(with_obs=self.rec_obs_action.isChecked())

    def _recorder_changed(self):
        rec = self.recorder
        self.rec_action.setText(tr("Aufnahme beenden") if rec.active else tr("Aufnahme starten"))
        self._rec_label.setVisible(rec.active or self.review.active)
        if rec.active or self.review.active:
            self._rec_timer.start()
            self._update_rec_label()
        else:
            self._rec_timer.stop()
            if rec.folder:
                self.statusBar().showMessage(tr("Aufnahme gespeichert: {0}").format(rec.folder), 10000)

    # --- Review (review.py) ---
    def toggle_review(self):
        if self.review.active:
            self.review.stop()
            return
        self.review.start()
        self._offer_whisper()

    def _offer_whisper(self):
        """Without Whisper, offer once to set up transcription (the review records anyway)."""
        settings = QSettings("SketchMachine", "SketchMachine")
        if review.whisper_python() is not None or settings.value("whisper_ask_off", False, type=bool):
            return
        box = QMessageBox(QMessageBox.Question, tr("Transkription einrichten?"), tr(
            "Das Review nimmt auf. Für das Transkript – damit die KI versteht, was du sagst – braucht Mule einmalig "
            "Whisper (läuft nur auf diesem Rechner, etwa 2–4 GB). Jetzt einrichten? Es läuft im Hintergrund, du kannst "
            "weiter reviewen."), parent=self)
        go = box.addButton(tr("Einrichten …"), QMessageBox.AcceptRole)
        box.addButton(tr("Später"), QMessageBox.RejectRole)
        never = box.addButton(tr("Nicht mehr fragen"), QMessageBox.DestructiveRole)
        box.exec()
        if box.clickedButton() is go:
            self.open_whisper_setup()
        elif box.clickedButton() is never:
            settings.setValue("whisper_ask_off", True)

    def open_whisper_setup(self):
        from . import whisper_setup
        dlg = whisper_setup.SetupDialog(self)
        dlg.setModal(False)  # set up in the background, the recording continues
        dlg.show()
        self._whisper_dialog = dlg

    def _review_marking(self, on):
        """"Mark" tool on/off (keep button, menu and session the same)."""
        if self.review_mark_action.isChecked() != on:
            self.review_mark_action.blockSignals(True)
            self.review_mark_action.setChecked(on)
            self.review_mark_action.blockSignals(False)
        self._style_review()

    def _review_marked(self, n, hit, name):
        if hit:
            self.statusBar().showMessage(tr("Hier! – Marke {0} auf {1}").format(n, name or "?"), 3000)
        else:
            self.statusBar().showMessage(tr("Hier! – Marke {0}: nichts unter dem Cursor getroffen (nur Bild und Stelle)")
                                         .format(n), 3500)

    def _review_unmarked(self, n):
        self.statusBar().showMessage(tr("Marke {0} zurückgenommen").format(n) if n else tr("Keine Marke zum Zurücknehmen"),
                                     3000)

    def _review_aiming(self, name):
        if name == "-":
            return
        self.statusBar().showMessage(tr("Markieren: {0} – Klick setzt die Marke").format(name) if name
                                     else tr("Markieren: nichts unter dem Cursor – ein Klick setzt die Marke trotzdem (nur Bild und Stelle)"),
                                     1500)

    def _fill_mic_menu(self, menu):
        menu.clear()
        chosen = review.saved_microphone()
        group = []
        for dev_id, name in [("", tr("Standard des Systems"))] + review.microphones():
            a = menu.addAction(name)
            a.setCheckable(True)
            a.setChecked(dev_id == chosen)
            a.triggered.connect(lambda _=False, d=dev_id: review.save_microphone(d))
            group.append(a)
        if len(group) == 1:
            menu.addAction(tr("(kein Mikrofon gefunden)")).setEnabled(False)

    def _review_changed(self):
        rv = self.review
        self.review_action.setText(tr("Review beenden") if rv.active else tr("Review starten"))
        self.review_mark_action.setEnabled(rv.active)
        self.review_unmark_action.setEnabled(rv.active)
        self._style_review()
        if rv.active:
            self._rec_label.show()
            self._rec_timer.start()
            self._update_rec_label()
            if not rv.audio.ok:
                QMessageBox.warning(self, tr("Review"), tr("Ohne Ton: {0}. Kamera und Cursor werden trotzdem "
                                                           "aufgezeichnet.").format(rv.audio.error))
        else:
            if not self.recorder.active:
                self._rec_label.hide()
                self._rec_timer.stop()
            if rv.folder:
                msg = tr("Review gespeichert – Transkript läuft im Hintergrund …") if rv.transcribing() else \
                    tr("Review gespeichert: {0}").format(rv.folder)
                self.statusBar().showMessage(msg, 10000)

    def _review_transcribed(self, folder, ok):
        name = os.path.basename(folder)
        if ok:
            self.statusBar().showMessage(tr("Review „{0}“ transkribiert – sag Claude: „Mach aus dem letzten Review "
                                            "Notizen“").format(name), 15000)
        else:
            self.statusBar().showMessage(tr("Review „{0}“: kein Transkript (Whisper nicht gefunden oder Fehler – "
                                            "siehe transkript_fehler.txt). Claude kann es später nochmal "
                                            "versuchen.").format(name), 15000)

    def _update_rec_label(self):
        rec = self.recorder
        parts = []
        if rec.active:
            s = int(rec.elapsed())
            obs = ""
            if rec.obs is not None:
                obs = " · OBS ●" if rec.obs.recording else " · OBS …"
            parts.append(f"● REC {s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}{obs}")
        if self.review.active:
            self._style_review()
            s = int(self.review.elapsed())
            parts.append(f"● REVIEW {s // 60:d}:{s % 60:02d}" + ("" if self.review.audio.ok else tr(" (ohne Ton)"))
                         + (tr(" · Markieren: Klick setzt eine Marke, Strg+Z nimmt sie zurück") if self.review.marking
                            else tr(" · Markieren aus")))
        self._rec_label.setText("   ".join(parts))

    def _set_note_scope(self, scope):
        """Note panel: Local (clicked note) or Global (note of the whole scene)."""
        self._note_scope = scope
        for key, b in self.note_panel.scope_buttons.items():
            b.setChecked(key == scope)
        self._sync_note_panel()

    def _maximize_note(self):
        """Board of the note in the panel across the whole program window (the global note does not become "active" in the process)."""
        nid = self.note_panel.note_id
        if nid is None:
            return
        if nid == self.viewport.scene.global_note.id:
            from .note_board import open_board
            open_board(self.viewport, nid)
        else:
            self.viewport.notes_tool.open_board(nid)

    def _update_title(self):
        name = os.path.basename(self.path) if self.path else tr("Unbenannt")
        star = "*" if self.viewport.scene.dirty else ""
        self.setWindowTitle(f"{name}{star} – Mule Create")

    # --- File ---
    def _confirm_discard(self):
        if not self.viewport.scene.dirty:
            return True
        answer = QMessageBox.question(self, tr("Ungespeicherte Änderungen"), tr("Änderungen speichern?"),
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save_scene()
        return answer == QMessageBox.Discard

    def new_scene(self):
        if self._confirm_discard():
            self.path, self.last_brief = None, None
            self._backup_hash = None
            self.viewport.set_scene(Scene())

    def open_scene(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, tr("Szene öffnen"), "", FILE_FILTER)
        if path:
            self.load_path(path)

    def load_path(self, path):
        """Load a scene from a file (Open dialog, double-click on a scene)."""
        try:
            scene, data = Scene.load(path)
        except Exception as exc:  # file broken or wrong format
            QMessageBox.warning(self, tr("Öffnen fehlgeschlagen"), str(exc))
            return
        self.path, self.last_brief = path, None
        self._backup_hash = None
        backup.add_recent(path)
        if data.get("views"):  # viewport layout with all cameras
            self.views.set_state(data["views"])
        elif data.get("camera"):  # older files: one camera
            self.views.set_single_camera(data["camera"])
        self.viewport.set_scene(scene)
        report = scene.net_report
        if report and report.get("migrated"):  # older file: curve net just created
            self.viewport.notify(tr("Kurvennetz übernommen: {0} Punkte, davon {1} verbunden, {2} offene Linienenden – beim Speichern im neuen Format").format(report['points'], report['shared'], report['free_ends']))
        if scene.load_warnings:
            QMessageBox.warning(self, tr("Geometrie fehlt"),
                                tr("Diese Dateien konnten nicht geladen werden:\n\n") + "\n".join(scene.load_warnings))

    def import_geometry(self):
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Geometrie importieren"), os.path.dirname(self.path or ""),
                                                GEOMETRY_FILTER)
        for path in paths:
            self.statusBar().showMessage(tr("Lade {0} …").format(os.path.basename(path)))
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                mesh = self.viewport.import_mesh(path)
            except Exception as exc:
                QMessageBox.warning(self, tr("Import fehlgeschlagen"), f"{os.path.basename(path)}:\n{exc}")
                continue
            finally:
                QApplication.restoreOverrideCursor()
            parts = [mesh] if hasattr(mesh, "geo") else \
                [o for o in self.viewport.scene.descendants(mesh) if hasattr(o, "geo")]
            tris = sum(m.geo.triangle_count for m in parts)
            self.statusBar().showMessage(tr("Importiert: {0} ({1:,} Dreiecke").format(mesh.name, tris)
                                         + (tr(", {0} Meshes in der Hierarchie der Datei)").format(len(parts)) if len(parts) > 1 else ")"),
                                         5000)

    def save_scene(self, quiet=False):
        """Save; quiet: without a window - an error comes back as OSError (MCP: a dialog blocked the call)."""
        if self.path is None:
            return self.save_scene_as()
        board = getattr(self.views, "_note_board", None)
        if board is not None and board.note_id is not None:
            board.commit()  # open note board first into the note
        self.note_panel.commit()
        self.shots_panel.commit()
        try:
            self.viewport.scene.save(self.path, extra={"views": self.views.state()})
        except OSError as exc:
            if quiet:
                raise
            QMessageBox.warning(self, tr("Speichern fehlgeschlagen"), str(exc))
            return False
        backup.add_recent(self.path)
        self._update_title()
        self.statusBar().showMessage(tr("Gespeichert: {0}").format(self.path), 4000)
        return True

    def scene_base(self):
        return assoc.strip_ext(os.path.basename(self.path or "skizze"))

    def export_brief_to(self, folder=None):
        """Export the brief without a dialog (MCP export_brief): with the saved assignment, without focus. Without folder
        next to the scene as <scene>_brief. Returns the folder."""
        from . import brief
        base = self.scene_base()
        self.note_panel.commit()
        self.shots_panel.commit()
        task = brief.task_clean(self.viewport.scene.brief_task)
        if folder is None and task.get("based_on") and task.get("kind") != "discuss":  # review round
            return self._export_review_round(task, None, None)
        target = folder or os.path.join(os.path.dirname(self.path or os.path.abspath(".")), f"{base}_brief")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            brief.export(self.viewport.scene, self.viewport, target, base)
        finally:
            QApplication.restoreOverrideCursor()
        self.last_brief = target
        self.statusBar().showMessage(tr("Sketch Brief (über Claude): {0}").format(target), 8000)
        return target

    def claude_prompt(self, target):
        """Autoprompt for the AI chat after the export (as in the brief dialog)."""
        from .brief_dialog import autoprompt
        return autoprompt(self.scene_base(), target, self.viewport.scene.brief_task)

    def export_brief(self):
        """Export the Sketch Brief (brief.md, brief.json, views, scene.usd, note images) into a folder."""
        from PySide6.QtWidgets import QDialog
        from . import brief
        from .brief_dialog import BriefDialog
        scene = self.viewport.scene
        # selection as focus only if checked (otherwise a forgotten selection would narrow the brief)
        chosen = [i for i in self.viewport.ws.selection if scene.object_by_id(i) is not None]
        self.note_panel.commit()
        self.shots_panel.commit()
        QApplication.setOverrideCursor(Qt.WaitCursor)  # analysis for the hints in the dialog (reused on export)
        try:
            data = brief.analyse(scene)
        finally:
            QApplication.restoreOverrideCursor()
        base = self.scene_base()
        start = os.path.join(os.path.dirname(self.path or os.path.abspath(".")), f"{base}_brief")
        from . import timing_export
        dlg = BriefDialog(self, scene.brief_task, len(chosen), brief.precheck(scene, data), base, start,
                          scene.brief_clarified, scene.timing.interp if timing_export.has_cameras(scene) else None)
        if dlg.exec() != QDialog.Accepted:
            return
        task = dlg.values()
        if timing_export.has_cameras(scene) and dlg.timing_interp() != scene.timing.interp:
            scene.timing.interp = dlg.timing_interp()  # remembered with the scene (also applies to the USD export)
            scene.dirty = True
        if task != scene.brief_task:
            scene.brief_task = task
            scene.dirty = True
            self._update_title()
        if dlg.clear_clarified:  # discard what was clarified with the AI (in the dialog "Delete")
            scene.brief_clarified = {}
            scene.dirty = True
            self._update_title()
        focus = chosen if dlg.focus.isChecked() else None
        copy_prompt = dlg.copy_prompt.isChecked()
        QSettings("SketchMachine", "SketchMachine").setValue("brief_copy_prompt", copy_prompt)
        if task.get("based_on") and task.get("kind") != "discuss":  # review round for the running chat
            return self._export_review_round(task, focus, data, copy_prompt, dlg)
        folder = QFileDialog.getExistingDirectory(self, tr("Ordner für den Sketch Brief wählen"), os.path.dirname(start))
        if not folder:
            return
        target = os.path.join(folder, f"{base}_brief")
        from . import brief_v4, recipes
        if task.get("kind") != "discuss" and task.get("recipe") and recipes.find(task["recipe"]) is None:
            # the chosen recipe went missing or broke after the dialog (e.g. edited meanwhile): never switch silently
            if QMessageBox.question(self, tr("Brief-Recipe"), tr("Das Recipe „{0}“ ist nicht installiert oder gerade nicht nutzbar. Mit Mule Default exportieren?").format(task["recipe"])) != QMessageBox.Yes:
                return
        if task.get("kind") != "discuss" and brief_v4.running(target, recipes.get(task.get("recipe"))):  # do not swap rules in the middle of a run
            if QMessageBox.question(self, tr("Laufender Lauf"), tr("Im Ordner {0} läuft schon ein Lauf. Regeln, Rollen und Hooks bleiben, wie sie sind; neu geschrieben werden nur Inhalt und Bilder (brief.md, project.md, views, notes). Trotzdem exportieren?").format(brief_v4.work_dir(target))) != QMessageBox.Yes:
                return
        self.note_panel.commit()
        self.shots_panel.commit()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            files = brief.export(self.viewport.scene, self.viewport, target, base, focus=focus, data=data)
        except Exception as exc:
            QMessageBox.warning(self, tr("Export fehlgeschlagen"), str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.last_brief = target
        self.statusBar().showMessage(tr("Sketch Brief: {0} Dateien in {1}").format(len(files), target), 8000)
        if copy_prompt and dlg.prompt_text():  # autoprompt for the AI chat (subscription): paste, the AI reads the brief
            QApplication.clipboard().setText(dlg.prompt_text(target))
            self.statusBar().showMessage(tr("Sketch Brief in {0} – Autoprompt kopiert: neue KI-Sitzung in genau diesem Ordner starten, dann Strg+V").format(target), 10000)

    def _export_review_round(self, task, focus, data, copy_prompt=False, dlg=None):
        """"Builds on": export as a review round into the brief folder of the running chat (review_round.py) - without
        folder choice, without package and hooks. -> folder of the round or None."""
        from . import brief, review_round
        from .brief_dialog import autoprompt
        if not review_round.is_brief(task["based_on"]):
            if dlg is None:  # MCP: message to Claude instead of dialog
                raise ValueError(f"no Sketch Brief (brief.json) in {task['based_on']} – the review round needs the "
                                 "brief folder of the running chat (export dialog, „Baut auf“)")
            QMessageBox.warning(self, tr("Review-Runde"), tr("In {0} liegt kein Sketch Brief (brief.json). Wähl unter „Baut auf“ den Brief-Ordner, in dem der laufende Chat arbeitet.").format(task["based_on"]))
            return None
        base = review_round.base_of(task["based_on"])
        target = review_round.next_folder(base)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            files = brief.export(self.viewport.scene, self.viewport, target, self.scene_base(), focus=focus, data=data,
                                 review_round=True)
        except Exception as exc:
            if dlg is None:
                raise
            QMessageBox.warning(self, tr("Export fehlgeschlagen"), str(exc))
            return None
        finally:
            QApplication.restoreOverrideCursor()
        self.last_brief = base  # what was clarified (write_clarified) belongs into the brief of the running run
        n = review_round.round_number(target)
        self.statusBar().showMessage(tr("Review-Runde {0}: {1} Dateien in {2}").format(n, len(files), target), 8000)
        if copy_prompt:
            own = dlg is not None and not dlg._prompt_auto and dlg.prompt.toPlainText().strip()
            QApplication.clipboard().setText(own or autoprompt(self.scene_base(), target, task))
            self.statusBar().showMessage(tr("Review-Runde {0} in {1} – Satz kopiert: im laufenden Chat einfügen (Strg+V)").format(n, target), 12000)
        return target

    def export_usd(self):
        """Scene as USD (usd_export): hierarchy, surfaces as polymeshes, imported meshes referenced -
        resolution of the surfaces as a multiple of the one in the viewport."""
        from . import usd_export
        base = assoc.strip_ext(os.path.basename(self.path or "skizze"))
        start = os.path.join(os.path.dirname(self.path or os.path.abspath(".")), f"{base}.usd")
        path, _ = QFileDialog.getSaveFileName(self, tr("Szene als USD exportieren"), start,
                                              tr("USD binär (*.usd *.usdc);;USD Text (*.usda)"))
        if not path:
            return
        options = self._usd_options()
        if options is None:
            return
        mult, tubes = options
        self.note_panel.commit()
        self.shots_panel.commit()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            usd_export.write(self.viewport.scene, path, mult=mult, tubes=tubes)
        except Exception as exc:
            QMessageBox.warning(self, tr("Export fehlgeschlagen"), str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(tr("Szene als USD: {0}").format(path), 8000)

    def _usd_options(self):
        """Small dialog: resolution of the surfaces (x viewport) and whether lines additionally come as tubes.
        None: cancelled."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout
        from . import usd_export
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Szene als USD"))
        form = QFormLayout(dlg)
        mult = QDoubleSpinBox(decimals=2, minimum=0.25, maximum=4.0, singleStep=0.25, value=usd_export.DEFAULT_MULT)
        mult.setToolTip(tr("Raster-Flächen (Lofts, Kurvenflächen) im Verhältnis zur Dichte im Viewport – der ist zum glatten Anzeigen sehr dicht; importierte Geometrie bleibt immer, wie sie ist"))
        form.addRow(tr("Auflösung der Flächen (× Viewport):"), mult)
        tubes = QCheckBox(tr("Linien zusätzlich als Röhren (Polymesh, Durchmesser = Strichbreite)"))
        form.addRow(tubes)
        from PySide6.QtWidgets import QComboBox
        from . import timing_export
        scene = self.viewport.scene
        interp = QComboBox()  # timing cameras: only if there are any
        interp.addItem(tr("gestuft – wie in Mule (jede Stellung hält bis zur nächsten)"), "step")
        interp.addItem(tr("linear zwischen den Stellungen (weich machst du danach im 3D-Programm)"), "linear")
        interp.setCurrentIndex(1 if scene.timing.interp == "linear" else 0)
        form.addRow(tr("Shot-Kameras:"), interp)
        form.setRowVisible(interp, timing_export.has_cameras(scene))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return None
        if timing_export.has_cameras(scene) and interp.currentData() != scene.timing.interp:
            scene.timing.interp = interp.currentData()
            scene.dirty = True
        return mult.value(), tubes.isChecked()

    def export_curves(self):
        """Only the curves (curve_export): USD for 3D programs or JSON for scripts and AI chats."""
        from . import curve_export
        options = self._curve_options()
        if options is None:
            return
        kind, hidden, tubes = options
        filters = tr("Kurven als JSON (*.json)") if kind == "json" else tr("USD binär (*.usd *.usdc);;USD Text (*.usda)")
        path, _ = QFileDialog.getSaveFileName(self, tr("Kurven exportieren"),
                                              curve_export.default_path(self.path, kind), filters)
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if kind == "json":
                count, dropped = curve_export.write_json(self.viewport.scene, path, hidden, self.viewport.fov)
            else:
                count, dropped = curve_export.write_usd(self.viewport.scene, path, hidden, tubes, self.viewport.fov)
        except Exception as exc:
            QMessageBox.warning(self, tr("Export fehlgeschlagen"), str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(tr("{0} Kurven, {1} doppelte weggelassen: {2}").format(count, dropped, path), 8000)

    def _curve_options(self):
        """Small dialog: format, hidden items, tubes (USD only). None: cancelled; otherwise (format, hidden, tubes)."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QRadioButton
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Kurven exportieren"))
        form = QFormLayout(dlg)
        usd = QRadioButton(tr("USD – für Houdini, Blender, Maya"), checked=True)
        form.addRow(usd)
        form.addRow(QRadioButton(tr("JSON – für Skripte und KI-Chats (lesbarer Text)")))
        hidden = QCheckBox(tr("Ausgeblendetes mitnehmen"))
        form.addRow(hidden)
        tubes = QCheckBox(tr("Zusätzlich als Röhren (Polymesh, Durchmesser = Strichbreite)"))
        usd.toggled.connect(tubes.setEnabled)
        form.addRow(tubes)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return None
        return ("usd" if usd.isChecked() else "json"), hidden.isChecked(), usd.isChecked() and tubes.isChecked()

    def save_scene_as(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Szene speichern"), self.path or "szene" + assoc.EXT, FILE_FILTER)
        if not path:
            return False
        self.path = assoc.with_ext(path)
        return self.save_scene()

    def _set_language(self, code):
        if code == i18n.saved_language():
            return
        i18n.set_language(code)
        QMessageBox.information(self, "Sprache / Language",  # in both languages: the new one is what is meant
                                "Die neue Sprache gilt nach dem nächsten Start von Mule.\n\n"
                                "The new language applies the next time Mule starts.")

    def closeEvent(self, e):
        self.note_panel.commit()
        self.shots_panel.commit()
        if self._confirm_discard():
            if self.recorder.active:
                self.recorder.stop()
            if self.review.active:
                self.review.stop()  # Claude fetches the transcript later (review_read)
            QSettings("SketchMachine", "SketchMachine").setValue("window_state", self.saveState())
            e.accept()
        else:
            e.ignore()


def main():
    """Everything in one go (self-test, or with the start animation switched off); the normal start goes through
    launch.py, which shows the animation while this module is still loading."""
    paths.start_log()
    if "--selftest" in sys.argv:
        from . import selftest
        selftest.use_throwaway_settings()
    app = startup.make_application()
    window = make_window(app)
    startup.start(window, lambda: show_window(window))
    sys.exit(app.exec())


def make_window(app):
    """The main window with the AI connection, not shown yet. Exits here in a self-test."""
    window = MainWindow()
    if "--selftest" in sys.argv:
        from . import selftest
        sys.exit(selftest.run(window, app, sys.argv[sys.argv.index("--selftest") + 1]))
    from . import mcp_server
    window.mcp = mcp_server.start(window)  # Claude sees the scene at http://127.0.0.1:<PORT>/mcp (PORT from mcp_server)
    assoc.register()  # double-click on a .mule scene opens this Mule (Windows only, per user)
    return window


def show_window(window):
    """Shows the window once the start animation is over; then the scene from a double-click and the intro question."""
    from . import mcp_server
    window.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):  # double-click on a scene; load after showing
        QTimer.singleShot(startup.FADE_MS + 50, lambda: window.load_path(sys.argv[1]))
    from . import intro
    QTimer.singleShot(startup.FADE_MS + 600, lambda: intro.ask_at_first_start(window))  # once: offer the intro
    if window.mcp is None:  # port taken: otherwise the AI silently talks to the other Mule
        QTimer.singleShot(startup.FADE_MS + 100, lambda: window.statusBar().showMessage(tr(
            "KI-Anbindung aus: Port {0} ist belegt (läuft schon eine Mule?) – die KI sieht dann die andere Mule"
        ).format(mcp_server.PORT), 20000))


def _open_folder(path):
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _length_text(d):
    """Length in world units (meters) readable: mm, cm or m."""
    return f"{d * 1000:.2g} mm" if d < 0.01 else f"{d * 100:.2g} cm" if d < 1 else f"{d:.2g} m"


def _spacer(width):
    """Fixed spacing in a bar (sliders do not bump into neighboring icons)."""
    w = QWidget()
    w.setFixedWidth(width)
    return w
