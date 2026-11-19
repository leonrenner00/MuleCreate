"""MCP server of Mule: Claude (Claude Code / desktop app, via the subscription) sees the open scene and writes
what was clarified back – like fxhoudinimcp for Houdini. Runs in the app process at http://127.0.0.1:PORT/mcp (local only,
Streamable HTTP, answers as JSON); every tool runs in the Qt main thread.

Hooking into Claude Code (.mcp.json): {"mcpServers": {"sketch_machine": {"type": "http",
"url": "http://127.0.0.1:<port>/mcp"}}} (Calisto Studio neighbor default port 8772; Studio keeps 8770)."""
import base64
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .i18n import tr
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, Qt, Signal

# Calisto Studio neighbor pin: default off Studio dashboard :8770. Override with SKETCH_MACHINE_MCP_PORT.
def _port_from_env(default=8772):
    raw = (os.environ.get("SKETCH_MACHINE_MCP_PORT") or "").strip()
    if not raw:
        return default
    try:
        port = int(raw)
    except ValueError:
        return default
    return port if 1 <= port <= 65535 else default


PORT = _port_from_env()
VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")  # the newest first
IMAGE_MAX = 1568
MOMENT_MAX = 800  # review_moment for checking: about a quarter of the image tokens; detail=true full size

TOOLS = [
    {"name": "scene_brief",
     "description": "The sketch currently open in Mule, as a Sketch Brief in markdown (live – not the last "
                    "export): task set by the artist, parts and hierarchy, notes with their text, open points. Also "
                    "names the hero cameras (with approved image prompts) and the folder of the last brief export. "
                    "Start here.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "render_view",
     "description": "A picture of the open sketch without UI or labels. view: a hero camera name, one of persp, top, "
                    "front, right, back (framed on everything visible), or current (what the artist sees right now).",
     "inputSchema": {"type": "object", "properties": {
         "view": {"type": "string", "description": "hero camera name, persp/top/front/right/back, or current"},
         "xray": {"type": "boolean", "description": "see-through display (inside parts)"}},
         "required": ["view"], "additionalProperties": False}},
    {"name": "note_images",
     "description": "A note board as a picture (texts, reference images, screenshots and hand-drawn marks as the artist "
                    "arranged them) – live, without export. note: N1, N2 … as in the brief, a note title, or project "
                    "for the project note. sections: also each section of the board as a readable picture (as "
                    "notes/board_N#_k.png in an export: title, texts, arrows P#) – what „Pictures first“ asks for. "
                    "images: also each picture on the board on its own (at most 8).",
     "inputSchema": {"type": "object", "properties": {
         "note": {"type": "string"}, "sections": {"type": "boolean"}, "images": {"type": "boolean"}},
         "required": ["note"], "additionalProperties": False}},
    {"name": "video_frames",
     "description": "Frames of a reference video on a note board – only for a video the artist wants read closely. "
                    "video: its file name as in the brief (videos/…). times: seconds in the video (at most 12); "
                    "without times: an even overview.",
     "inputSchema": {"type": "object", "properties": {
         "video": {"type": "string"}, "times": {"type": "array", "items": {"type": "number"}}},
         "required": ["video"], "additionalProperties": False}},
    {"name": "image_captions",
     "description": "Pictures on the note boards with their file name (as in the brief: notes/…), the notes they are "
                    "on and their caption – by default the reference pictures without a caption (viewport screenshots "
                    "show the sketch itself and need none), at most 8 per call. Look at them with the board "
                    "(note_images) for context, propose one caption each to the artist – what the picture stands for "
                    "here (form, detail, material, mood, function …) and what matters in it (e.g. „only the hinge on "
                    "the left – the photo's proportions do not apply“), one sentence, not a full description; if the "
                    "board already says it, only point there – and the parts it belongs to (ids as in scene_brief: "
                    "H#, M#, F#, S#) – and store the confirmed ones with set_image_captions.",
     "inputSchema": {"type": "object", "properties": {
         "note": {"type": "string", "description": "only this note (N1 … or its title, project for the project note)"},
         "start": {"type": "integer", "description": "skip this many (paging)"},
         "all": {"type": "boolean", "description": "also screenshots and pictures that already have a caption"}},
         "additionalProperties": False}},
    {"name": "set_image_captions",
     "description": "Store captions the artist confirmed on note pictures (file as in the brief or image_captions; "
                    "empty caption removes it and its parts). A caption holds for the picture on every board it is "
                    "on; the brief shows it after the picture with its parts, and lists the picture at each part. "
                    "Only what the artist confirmed.",
     "inputSchema": {"type": "object", "properties": {
         "captions": {"type": "array", "items": {"type": "object", "properties": {
             "file": {"type": "string"}, "caption": {"type": "string"},
             "parts": {"type": "array", "items": {"type": "string"},
                       "description": "ids of the parts the picture belongs to (H#, M#, F#, S# as in scene_brief); "
                                      "[] removes them, left out keeps them"}},
             "required": ["file", "caption"], "additionalProperties": False}}},
         "required": ["captions"], "additionalProperties": False}},
    {"name": "export_brief",
     "description": "Export the Sketch Brief folder (brief.md, brief.json, views/, notes/, scene.usd) with the task "
                    "the artist set. Without folder: next to the scene file as <scene>_brief. Returns the folder – "
                    "read brief.md and the pictures there.",
     "inputSchema": {"type": "object", "properties": {"folder": {"type": "string"}}, "additionalProperties": False}},
    {"name": "write_clarified",
     "description": "After going through the brief with the artist: store the summary the artist confirmed and the "
                    "answers. Adds to what is already clarified (shown in the brief under „Clarified with the artist“): "
                    "the same question gets the new answer, drop takes out questions that no longer apply, a new "
                    "summary replaces the old one (empty: keep it). Replaced and dropped answers move to the history "
                    "(clarified_history.md, outside the brief's core) – so keep what applies now short: drop "
                    "corrections of a finished round and instructions for an earlier round once they are done or "
                    "built in, and merge overlapping answers into one. Saved in the scene (every later export carries "
                    "it) and written into the last exported brief.md. Answer \"decide\" means the artist leaves it "
                    "to the AI that builds. Only what the artist said or confirmed – never your own guesses.",
     "inputSchema": {"type": "object", "properties": {
         "summary": {"type": "string"},
         "answers": {"type": "array", "items": {"type": "object", "properties": {
             "question": {"type": "string"}, "answer": {"type": "string"}},
             "required": ["question", "answer"], "additionalProperties": False}},
         "drop": {"type": "array", "items": {"type": "string"},
                  "description": "questions (as written) that no longer apply – they move to the history"}},
         "required": ["answers"], "additionalProperties": False}},
    {"name": "set_hero_prompt",
     "description": "Store an image prompt the artist approved on a hero camera (a screenshot marked as hero camera). "
                    "Later exports list it as the approved prompt for the look images step. Empty prompt removes it.",
     "inputSchema": {"type": "object", "properties": {"hero": {"type": "string"}, "prompt": {"type": "string"}},
                     "required": ["hero", "prompt"], "additionalProperties": False}},
    {"name": "dictate_note",
     "description": "The artist dictates a note to you: store it as a text on a note board – not word for word, "
                    "but worked through: keep every point the artist made (what, where, why, sizes and numbers, "
                    "open questions), drop filler, repetitions and false starts (a self-correction counts in its "
                    "last version), in the artist's language. Add nothing of your own; mark what is unclear with "
                    "(?). Steps: 1. when the dictation is done, propose it as bullet points (by default – a short "
                    "first line, then one bullet per point, grouped by topic) and ask in the same message whether "
                    "to store it like this or as running text instead, and which note and pictures (say what you "
                    "would pick); 2. store only after the artist's go, in the form they chose; 3. say where it went "
                    "(Ctrl+Z in Mule undoes it). "
                    "note: N1 … or a note title as in scene_brief, project for the project note; left out: the "
                    "board open in Mule right now, else the project note. images: pictures on that board the "
                    "text is about (only when the artist says which, files as in note_images/image_captions) – "
                    "each gets an arrow from the text, read in the brief like an arrow the artist drew; at: where "
                    "in the picture it points (u, v from 0 to 1, from top left; left out: the middle).",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string", "description": "the worked-through note, not the raw dictation"},
         "note": {"type": "string"},
         "images": {"type": "array", "items": {"type": "object", "properties": {
             "file": {"type": "string"},
             "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
             "required": ["file"], "additionalProperties": False}}},
         "required": ["text"], "additionalProperties": False}},
    {"name": "review_read",
     "description": "A review recording made in Mule (menu View → Review): the artist walked through the scene and "
                    "talked feedback, like a supervisor in a dailies session; Mule recorded the voice and where the "
                    "camera was and what the cursor pointed at (no video – every moment can be rendered again). "
                    "Returns the transcript in pieces with time, what was under the cursor (ids as in scene_brief: "
                    "L# line, S# stroke object, F# surface, M# mesh), „Hier!“ marks the artist set on purpose, "
                    "camera changes, the selection and the open note board, what the artist changed themself "
                    "during the review (notes they created, pictures and texts they put on boards, lines they drew), "
                    "plus the drafts already written from this review. Without review: the newest one (list of all "
                    "at the top). The first call after a "
                    "recording may take a few minutes (local transcription). Then turn it into draft notes – see "
                    "review_note.",
     "inputSchema": {"type": "object", "properties": {
         "review": {"type": "string", "description": "name of the review (as listed); left out: the newest"},
         "start": {"type": "integer", "description": "skip this many transcript pieces (paging)"}},
         "additionalProperties": False}},
    {"name": "review_moment",
     "description": "Pictures of moments of a review recording: the sketch rendered from the camera the artist had "
                    "at that time, with the cursor drawn in as a magenta ring – to see what the artist was talking "
                    "about. Pictures cost tokens: look at ONE moment per component first (its best pointing – a "
                    "mark, a spoken „hier“ or a cursor rest) and add more only when text and pointing do not "
                    "settle what is meant. Small by default (800 px – enough to check what is meant); detail=true "
                    "for full size when you must read small things. times: seconds in the review (at most 6).",
     "inputSchema": {"type": "object", "properties": {
         "review": {"type": "string"}, "times": {"type": "array", "items": {"type": "number"}},
         "xray": {"type": "boolean"}, "detail": {"type": "boolean"}},
         "required": ["times"], "additionalProperties": False}},
    {"name": "review_note",
     "description": "Write one feedback point of a review recording into Mule as a DRAFT note – the artist checks "
                    "every draft afterwards in Mule (the board shows the original sentence, ▶ plays the audio, ✓ "
                    "accepts it), so you do not ask before each one. How to work: 1. review_read. 2. Group the talk "
                    "– as few notes as possible, as many as needed: one note per component of the sketch, with a "
                    "bullet group per aspect and a picture per spot (images), never one per sentence or detail. "
                    "What counts as a component depends on the scope of the scene, which you read from the "
                    "transcript, the pictures and scene_brief: if the scene is one object, its main parts are the "
                    "components (a car design: wheel, bonnet, windscreen, mirror); if the scene is a place or a "
                    "group of things, whole objects or kinds of objects are (a motorway: the road, cars, lorries). "
                    "Choose the level at which the artist talks about the scene, name each note after its "
                    "component in general terms, and keep to that level throughout. A further note only for "
                    "another component; rules for the whole scene go to the project note. Things said minutes "
                    "apart about the same component belong together; a later self-correction counts in its last "
                    "version. Plan the grouping first (list the components), then write. 3. Look at each "
                    "component – one review_moment at its best pointing first, more only when it stays unclear – and "
                    "check your bullets against the picture: say what is meant so it reads without the audio („the left wing tip“ instead of "
                    "„here“), but add nothing the artist did not say – unclear stays marked (?). 4. Text like "
                    "dictate_note: in the artist's language, a short first line, then one bullet per point; keep "
                    "every point (what, where, why, sizes, numbers, open questions), drop filler and repetitions. "
                    "5. review_note per point: t = the moment that shows it best – take it from a „cursor on … @time“ "
                    "or a MARK in review_read (or pass mark = its number), never a round number. review_note moves "
                    "t to the nearest real pointing (a mark up to 4 s away, else a spoken „hier/here/dort/there“ up to "
                    "3 s away while the cursor and camera stood still, else a cursor rest 2 s before to 3 s after) and "
                    "leaves the arrow and the cursor anchor out when there is none – then pass anchor with a part id. "
                    "quote = the artist's own words for this point (from the transcript), t_end = "
                    "when they end. Build on what the artist did during the review: a note they created or a board "
                    "they filled is theirs – do not make a second note for the same point; look at it (note_images) "
                    "and add to it (note = its N#) only what they said and is not there yet; lines they drew while "
                    "talking usually show what they mean – look at them in review_moment. t_end = "
                    "when they end. The note hangs at what was under the cursor at t (anchor) – pass anchor with a "
                    "part id when the artist clearly meant another part, „none“ for a point without a place. parts: "
                    "further parts the artist meant too („all these windows“) – the octopus arms. A point about an "
                    "existing note (the artist had its board open or talks about it): note = that note – text and "
                    "pictures are added to its board. General remarks without a place: note = project. images: "
                    "more moments to show (times), default the moment t. Reference pictures: a BOARD MARK in "
                    "review_read is the artist clicking a picture on a note board (often the project note used as "
                    "a reference board) – „the tank should look like this“: pass its number in refs on the note of "
                    "the part they pointed at around it, and the picture goes onto that note's board with an arrow "
                    "on the clicked spot (it stays on its own board too); say in the text what it shows for this "
                    "part. 6. Finally tell the artist in the chat: "
                    "how many drafts, which points were unclear or could not be placed. Prefer review_notes: all "
                    "drafts of a review in ONE call – they appear together in the open scene, one Ctrl+Z. Before "
                    "the first draft of a review Mule saves a backup of the scene (review_backups/ next to it); the "
                    "drafts go live into the open scene, unsaved – the artist saves once they checked them. "
                    "discard: delete a draft you wrote (note = its N#) – only drafts.",
     "inputSchema": {"type": "object", "properties": {
         "review": {"type": "string", "description": "name of the review; left out: the newest"},
         "t": {"type": "number", "description": "seconds in the review: the moment that shows the point"},
         "mark": {"type": "integer", "description": "a „Hier!“ mark number instead of t – the most exact pointing"},
         "t_end": {"type": "number"},
         "text": {"type": "string", "description": "the worked-through note (first line + bullets)"},
         "title": {"type": "string", "description": "title of a new note (short, a few words); left out: first line"},
         "quote": {"type": "string", "description": "the artist's own words for this point"},
         "note": {"type": "string", "description": "an existing note (N# or title, project) to add to instead of a "
                                                   "new draft"},
         "anchor": {"type": "string", "description": "cursor (default), a part id (L#, S#, F#, M#) or none"},
         "parts": {"type": "array", "items": {"type": "string"}},
         "images": {"type": "array", "items": {"type": "number"},
                    "description": "times of the pictures (max 4) – one per spot the note talks about; each gets "
                                   "an arrow where the artist pointed then"},
         "refs": {"type": "array", "items": {"type": "integer"},
                  "description": "BOARD MARK numbers (max 4): the clicked board picture goes onto this note's "
                                 "board, with an arrow on the clicked spot"},
         "discard": {"type": "boolean"}},
         "additionalProperties": False}},
    {"name": "review_notes",
     "description": "All drafts of a review in one go – same fields per draft as review_note (t or mark, text, "
                    "title, quote, t_end, note, anchor, parts, images, refs); they appear together in the open scene as "
                    "ONE undo step, so the artist is not disturbed draft by draft. Use this once you have planned "
                    "the grouping (see review_note). A draft that fails is reported and the others still go in.",
     "inputSchema": {"type": "object", "properties": {
         "review": {"type": "string", "description": "name of the review; left out: the newest"},
         "notes": {"type": "array", "minItems": 1, "maxItems": 40, "items": {"type": "object", "properties": {
             "t": {"type": "number"}, "mark": {"type": "integer"}, "t_end": {"type": "number"},
             "text": {"type": "string"}, "title": {"type": "string"}, "quote": {"type": "string"},
             "note": {"type": "string"}, "anchor": {"type": "string"},
             "parts": {"type": "array", "items": {"type": "string"}},
             "images": {"type": "array", "items": {"type": "number"}},
             "refs": {"type": "array", "items": {"type": "integer"}}},
             "additionalProperties": False}}},
         "required": ["notes"], "additionalProperties": False}},
    {"name": "insert_images",
     "description": "Put picture files from disk on a note board – like the artist dragging them in: reference "
                    "pictures from a project folder, the „Hier!“ pictures of a review (marke_N.jpg in its folder), "
                    "renders of an earlier run. Only pictures you have opened and that clearly belong there – the "
                    "artist named them or the folder, or they are the ones already used for the same thing; when in "
                    "doubt, ask first. The file on disk stays; the picture is stored in the scene (Ctrl+Z undoes "
                    "it). note: N# or a title, project for the project note; left out: the board open in Mule, else "
                    "the project note. files: path (absolute, or relative to the scene file's folder), caption "
                    "(only what the artist said or confirmed, see image_captions), at (u, v from 0 to 1: where an "
                    "arrow from text points). text: optional note text next to the pictures, with an arrow to each "
                    "picture that has at. A picture already on that board is not added twice.",
     "inputSchema": {"type": "object", "properties": {
         "note": {"type": "string"},
         "files": {"type": "array", "maxItems": 12, "items": {"type": "object", "properties": {
             "path": {"type": "string"}, "caption": {"type": "string"},
             "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
             "required": ["path"], "additionalProperties": False}},
         "text": {"type": "string"}},
         "required": ["files"], "additionalProperties": False}},
    {"name": "edit_note",
     "description": "Change the text of a note – ONLY when the artist explicitly asks you to (e.g. to improve a note "
                    "you wrote or one of theirs), or a review draft you are writing right now (review_note); never "
                    "on your own initiative, never to tidy up. Do what was asked, "
                    "nothing more. old: an exact piece of a text on that note's board (as in scene_brief / "
                    "note_images), quoted long enough to be unique; it is replaced by new (empty: deleted; a text "
                    "left empty disappears). title: a new note title (optional). note: N1 … or a note title, project "
                    "for the project note; left out: the board open in Mule right now. Afterwards say what changed; "
                    "Ctrl+Z in Mule undoes it.",
     "inputSchema": {"type": "object", "properties": {
         "old": {"type": "string"}, "new": {"type": "string"}, "title": {"type": "string"},
         "note": {"type": "string"}}, "additionalProperties": False}},
]


def backup_before_review(window, name):
    """Backup of the open scene in review_backups/ (per scene the newest backup.review_keep()), once per review and
    scene -> hint for Claude ("" if already done)."""
    from . import backup
    done = window.__dict__.setdefault("_review_backed", set())
    key = (name, id(window.viewport.scene))
    if key in done:
        return ""
    target, _ = backup.write(window.viewport.scene, window.path, {"views": window.views.state()},
                             folder=backup.REVIEW_FOLDER, keep=backup.review_keep())
    done.add(key)
    return (f" Before the first draft Mule saved a backup of the scene ({target}); the drafts go live into the open "
            "scene, unsaved – the artist saves once they checked them.")


class _MainThread(QObject):
    """Runs calls from the server thread in the Qt main thread and waits for the result."""
    job = Signal(object)

    def __init__(self):
        super().__init__()
        self.job.connect(self._run, Qt.QueuedConnection)

    @staticmethod
    def _run(job):
        fn, box, done = job
        try:
            box["result"] = fn()
        except Exception as exc:  # as a tool error to Claude
            box["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            done.set()

    def call(self, fn, timeout=300):
        box, done = {}, threading.Event()
        self.job.emit((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError(tr("Mule antwortet nicht (beschäftigt oder ein Dialog ist offen)"))
        if "error" in box:
            raise RuntimeError(box["error"])
        return box["result"]


def _png(img, max_px=IMAGE_MAX):
    """QImage -> base64 PNG (longest side at most max_px)."""
    if max(img.width(), img.height()) > max_px:
        img = img.scaled(max_px, max_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return base64.standard_b64encode(bytes(data)).decode("ascii")


class Tools:
    """The tools – run in the main thread, work on the main window (scene, viewport, last export)."""

    def __init__(self, window):
        self.w = window

    @property
    def vp(self):
        return self.w.viewport

    def scene_brief(self):
        from . import brief
        scene = self.vp.scene
        data = brief.analyse(scene)
        path = os.path.join(tempfile.mkdtemp(), "brief.md")
        has_context = brief.write_markdown(data, [], path, self.w.scene_base())
        with open(path, encoding="utf-8") as fh:
            md = fh.read()
        if (data.get("task") or {}).get("kind") != "discuss":  # build brief (package v4): project document first
            from . import brief_v4, recipes
            recipe = recipes.get((data.get("task") or {}).get("recipe"))
            project = "\n".join(brief_v4.project_md(data, self.w.scene_base(), None, has_context, recipe))
            md = project + "\n\n---\n\n" + md
        heroes = brief.board_heroes(scene)
        export = self.w.last_brief
        # Blind test Glider 003: the AI read "exist only in an exported brief folder" in the export folder as "paths
        # do not apply" and never opened the images there – if the folder exists, it is stated here as the location of the files
        if export and os.path.isdir(export):
            files = (f"File paths below (views/…, notes/…) are the files of that export – open them in {export} (as of "
                     "the export; where the scene changed since, render_view and note_images show it live).")
        else:
            files = "File paths below (views/…, notes/…) exist only in an exported brief folder."
        head = [f"Scene file: {self.w.path or '(not saved)'}",
                f"Last brief export: {export or '(none in this session)'}",
                "Pictures: render_view (overviews, hero cameras – one call per camera –, the current view) and "
                "note_images (note boards, sections=true for the section pictures); reference pictures without a "
                "caption: image_captions. " + files,
                "Hero cameras: " + (", ".join(f"„{h['name']}“" + (" (approved prompt)" if h.get("prompt") else "")
                                              for h in heroes) if heroes else "none – exports use standard angles")]
        return [{"type": "text", "text": "\n".join(head) + "\n\n" + md}]

    def render_view(self, view, xray=None):
        from . import brief
        vp = self.vp
        look = None if xray is None else {"xray": 0.6 if xray else None}
        if view == "current":
            return [{"type": "image", "data": _png(vp.clean_image(look=look)), "mimeType": "image/png"}]
        heroes = {h["name"]: h for h in brief.board_heroes(vp.scene)}
        names = {name: (back, ortho) for name, back, ortho in brief.VIEWS}
        if view not in heroes and view not in names:
            raise ValueError(f"unknown view {view!r} – hero cameras: {', '.join(heroes) or 'none'}; or "
                             f"{', '.join(names)}, current")
        data = brief.analyse(vp.scene)
        with brief._keep_camera(vp) as cam:
            aspect = None
            if view in heroes:
                v = heroes[view]["view"]
                import numpy as np
                cam.set_state(np.array(v["target"], float), float(v["dist"]), np.array(v["rot"], float))
                cam.ortho, aspect = bool(v.get("ortho")), heroes[view]["aspect"]
            else:
                brief.frame_all(vp, cam, data, *names[view])
            saved = vp.planes_visible
            vp.planes_visible = False
            try:
                img = vp.clean_image(aspect=aspect, look=look)
            finally:
                vp.planes_visible = saved
                vp.update()
        return [{"type": "image", "data": _png(img), "mimeType": "image/png"}]

    def _notes(self):
        """[(N1 … / project, note)] as in the brief."""
        from . import brief
        scene = self.vp.scene
        nid = brief.index(scene)[1]["nid"]
        return [(nid[n.id], n) for n in scene.notes] + [("project", scene.global_note)]

    def _note(self, note):
        notes = self._notes()
        key = note.strip().lower()
        hit = next(((k, n) for k, n in notes if key in (k.lower(), (n.title or "").strip().lower())), None)
        if hit is None:
            raise ValueError(f"unknown note {note!r} – notes: " + ", ".join(f"{k} {n.title}" for k, n in notes))
        return hit

    def note_images(self, note, sections=False, images=False):
        from PySide6.QtGui import QImage
        from . import brief
        from .note_board import render_board
        scene = self.vp.scene
        name, hit = self._note(note)
        board = render_board(scene, hit.board)
        if board is None:
            return [{"type": "text", "text": f"{name} {hit.title}: the board is empty."}]
        out = [{"type": "text", "text": f"{name} {hit.title} – the whole board:"},
               {"type": "image", "data": _png(board), "mimeType": "image/png"}]
        if sections:  # the same section images as in the export (brief._board_sections), here in a temp folder
            folder = tempfile.mkdtemp()
            os.makedirs(os.path.join(folder, "notes"))
            for k, sec in enumerate(brief._board_sections(scene, hit.board, name, folder), 1):
                arrows = ", ".join(a["id"] for a in sec["arrows"])
                label = (f"section {k}" + (f" „{sec['title']}“" if sec["title"] else "") + f" `{sec['image']}`"
                         + (f" – arrows {arrows}" if arrows else ""))
                out += [{"type": "text", "text": label + ":"},
                        {"type": "image", "data": _png(QImage(os.path.join(folder, sec["image"]))),
                         "mimeType": "image/png"}]
        if images:
            pics = [i for i in hit.board.get("items", []) if i.get("kind") == "image"][:8]
            for k, item in enumerate(pics, 1):
                img = QImage.fromData(scene.asset_bytes(item["file"]) or b"")
                if not img.isNull():
                    label = f"picture {k} `notes/{item['file']}`" + (
                        f" – hero camera „{item['hero_name']}“" if item.get("hero") else
                        " – viewport screenshot" if item.get("view") else "") + (
                        f" – caption „{item['caption']}“" if item.get("caption") else "")
                    out += [{"type": "text", "text": label + ":"},
                            {"type": "image", "data": _png(img), "mimeType": "image/png"}]
        return out

    def video_frames(self, video, times=None):
        from . import brief
        from . import video as video_files
        names = brief.video_names(self.vp.scene)
        wanted = os.path.basename(video.replace("\\", "/"))
        path = next((p for p, name in names.items() if name == wanted), None)
        if path is None:
            raise ValueError(f"unknown video {video!r} – videos: " + (", ".join(names.values()) or "none"))
        at = (lambda d: [t * 1000 for t in times[:12]]) if times else video_files.overview_times
        frames, duration = video_files.read_frames(path, at, video_files.STILL_MAX)
        if not frames:
            raise ValueError(f"could not read {wanted!r}")
        out = [{"type": "text", "text": f"{wanted}: {duration / 1000:.1f} s"}]
        for ms, img in frames:
            out += [{"type": "text", "text": f"at {ms / 1000:.2f} s:"},
                    {"type": "image", "data": _png(img), "mimeType": "image/png"}]
        return out

    def image_captions(self, note=None, start=0, all=False):
        from PySide6.QtGui import QImage
        from . import brief
        scene = self.vp.scene
        notes = [self._note(note)] if note else self._notes()
        caps = brief.image_captions(scene)
        _, maps = brief.index(scene)
        ids = {**maps["sid"], **maps["fid"], **maps["mid"], **maps["hid"]}
        parts = {f: [ids[o] for o in objs if o in ids] for f, objs in brief.image_caption_parts(scene).items()}
        found = {}  # file -> (entry, [notes]) – each image once, even if it lies on several boards
        for key, n in notes:
            for item in n.board.get("items", []):
                if item.get("kind") != "image" or (not all and (item.get("view") or item["file"] in caps)):
                    continue
                found.setdefault(item["file"], (item, []))[1].append(f"{key} {n.title or ''}".strip())
        pics = list(found.items())
        if not pics:
            return [{"type": "text", "text": "Every reference picture has a caption." if not all else "No pictures."}]
        start = max(int(start or 0), 0)
        page = pics[start:start + 8]
        rest = len(pics) - start - len(page)
        out = [{"type": "text", "text": f"Pictures {start + 1}–{start + len(page)} of {len(pics)}"
                                        + (f" (call again with start={start + len(page)} for the rest)" if rest > 0
                                           else "") + ":"}]
        for f, (item, where) in page:
            img = QImage.fromData(scene.asset_bytes(f) or b"")
            label = f"`notes/{f}` – on {', '.join(where)}" + (" – viewport screenshot" if item.get("view") else "") \
                + (f" – caption „{caps[f]}“" if f in caps else "")                 + (f" → parts {', '.join(parts[f])}" if parts.get(f) else "") + (" – MISSING in the scene file" if img.isNull()
                                                                       else "")
            out.append({"type": "text", "text": label})
            if not img.isNull():
                out.append({"type": "image", "data": _png(img), "mimeType": "image/png"})
        return out

    def set_image_captions(self, captions):
        from .note_board import set_image_captions
        from . import brief
        _, maps = brief.index(self.vp.scene)
        by_id = {v: k for m in ("sid", "fid", "mid", "hid") for k, v in maps[m].items()}  # H3 -> object id
        bad = sorted({i for c in captions for i in c.get("parts") or [] if i not in by_id})
        parts = {c["file"]: [by_id[i] for i in c["parts"] if i in by_id] for c in captions if "parts" in c}
        unknown = set_image_captions(self.vp, {c["file"]: c["caption"] for c in captions}, parts)
        self.w._update_title()
        done = len(captions) - len(unknown)
        return [{"type": "text", "text": f"{done} caption(s) stored (saved with the scene, later exports carry them)."
                 + (f" Not on any board: {', '.join(unknown)}" if unknown else "")
                 + (f" Unknown or hidden part ids, left out: {', '.join(bad)}" if bad else "")}]

    def export_brief(self, folder=None):
        target = self.w.export_brief_to(folder)
        from . import review_round
        if target and os.path.isfile(os.path.join(target, review_round.ROUND_FILE)):
            return [{"type": "text", "text": f"Exported review round to {target} – read review.md there (what changed "
                                             "since the last export) and fit it into the running work."}]
        return [{"type": "text", "text": f"Exported to {target} – read brief.md there (and the pictures in views/ "
                                         "and notes/)."}]

    def write_clarified(self, answers, summary="", drop=()):
        from . import brief_ai
        scene = self.vp.scene
        new = [(a["question"], a["answer"]) for a in answers]
        merged = brief_ai.merge_clarified(scene.brief_clarified, summary, new, drop)
        scene.brief_clarified = merged
        scene.dirty = True
        self.w._update_title()
        where = "saved in the scene (later exports carry it)"
        if self.w.last_brief and os.path.isfile(os.path.join(self.w.last_brief, "brief.md")):
            brief_ai.write_clarified(self.w.last_brief, merged.get("summary", ""),
                                     [(a["question"], a["answer"]) for a in merged.get("answers", [])],
                                     history=[(a["question"], a["answer"]) for a in merged.get("history", [])])
            where += f" and written into {self.w.last_brief}/brief.md"
        self.w.statusBar().showMessage(tr("KI: geklärte Punkte übernommen – Szene speichern nicht vergessen"), 8000)
        return [{"type": "text", "text": f"Now {len(merged.get('answers', []))} clarified answers"
                                         f"{' and a summary' if merged.get('summary') else ''}, {where}."}]

    def dictate_note(self, text, note=None, images=()):
        from .note_board import add_dictated_text, open_note_id
        text = text.strip()
        if not text:
            raise ValueError("empty text")
        scene = self.vp.scene
        if note:
            key, hit = self._note(note)
        else:
            nid = open_note_id(self.vp)
            key, hit = next(((k, n) for k, n in self._notes() if n.id == nid), ("project", scene.global_note))
        arrows, missing = add_dictated_text(self.vp, hit, text, [(i["file"], i.get("at")) for i in images or ()])
        self.w._update_title()
        self.w.statusBar().showMessage(tr("KI: Notiz auf „{0}“ eingetragen – Strg+Z nimmt sie zurück")
                                       .format(hit.title or key), 8000)
        on_board = [i["file"] for i in hit.board.get("items", []) if i.get("kind") == "image"]
        return [{"type": "text", "text": f"Stored on {key} {hit.title or ''}".rstrip()
                 + (f" with {arrows} arrow(s) to its pictures" if arrows else "") + "."
                 + (f" Not on this board, no arrow: {', '.join(missing)} (pictures here: "
                    f"{', '.join('notes/' + f for f in on_board) or 'none'})" if missing else "")}]

    def insert_images(self, files, note=None, text=""):
        import hashlib
        from .note_board import add_review_entries, image_file_bytes, open_note_id, _ext_of
        scene = self.vp.scene
        if note:
            key, hit = self._note(note)
        else:
            nid = open_note_id(self.vp)
            key, hit = next(((k, n) for k, n in self._notes() if n.id == nid), ("project", scene.global_note))
        base = os.path.dirname(self.w.path) if self.w.path else os.getcwd()
        on_board = {i.get("file") for i in (hit.board or {}).get("items", []) if i.get("kind") == "image"}
        shots, missing, unreadable, twice = [], [], [], []
        for f in files[:12]:
            path = os.path.expanduser(f["path"])
            path = path if os.path.isabs(path) else os.path.join(base, path)
            if not os.path.isfile(path):
                missing.append(f["path"])
                continue
            data = image_file_bytes(path)
            if data is None:
                unreadable.append(f["path"])
                continue
            if hashlib.sha1(data).hexdigest()[:16] + (_ext_of(data) or ".jpg") in on_board:
                twice.append(os.path.basename(path))
                continue
            uv = tuple(f["at"]) if f.get("at") else None
            shots.append((data, None, uv, (f.get("caption") or "").strip() or None))
        if shots:
            add_review_entries(self.vp, hit, (text or "").strip(), shots)
            self.w._update_title()
            self.w.statusBar().showMessage(tr("KI: {0} Bild(er) auf „{1}“ gelegt – Strg+Z nimmt es zurück")
                                           .format(len(shots), hit.title or key), 8000)
        parts = [f"{len(shots)} picture(s) put on {key} {hit.title or ''}".rstrip()
                 + (" with text" if shots and (text or "").strip() else "") + "."]
        if twice:
            parts.append("Already on that board, not added again: " + ", ".join(twice) + ".")
        if missing:
            parts.append("Not found: " + ", ".join(missing) + f" (relative paths start at {base}).")
        if unreadable:
            parts.append("Not a picture Mule can read: " + ", ".join(unreadable) + ".")
        return [{"type": "text", "text": " ".join(parts)}]

    def edit_note(self, old="", new="", title=None, note=None):
        from .note_board import edit_note_text, open_note_id
        if not old and not (title or "").strip():
            raise ValueError("nothing to change: give old (+ new) or title")
        scene = self.vp.scene
        if note:
            key, hit = self._note(note)
        else:
            nid = open_note_id(self.vp)
            if nid is None:
                raise ValueError("no board is open in Mule – name the note (N1 …, a title or project)")
            key, hit = next(((k, n) for k, n in self._notes() if n.id == nid), ("project", scene.global_note))
        edit_note_text(self.vp, hit, old, new, title)
        self.w._update_title()
        self.w.statusBar().showMessage(tr("KI: Notiz „{0}“ geändert – Strg+Z nimmt es zurück")
                                       .format(hit.title or key), 8000)
        return [{"type": "text", "text": f"Changed {key} {hit.title or ''}".rstrip() + "."}]

    # --- Review (review.py) ---
    def _review(self, review=None):
        """(folder, log) of a review – and Mule shows "Send review" on the button: Claude is working."""
        from . import review as rv
        folder = rv.find_review(review)
        if folder is not None and getattr(self.w, "review_sender", None) is not None:
            self.w.review_sender.working(os.path.basename(folder))
        if folder is None:
            names = [n for n, _ in rv.list_reviews()]
            raise ValueError((f"unknown review {review!r} – " if review else "no review recorded yet – ")
                             + ("reviews: " + ", ".join(names[:12]) if names else
                                "record one in Mule: View → Review → Start review"))
        return folder, rv.read_log(folder)

    def _ids(self):
        from . import brief
        _, maps = brief.index(self.vp.scene)
        return maps

    def _label(self, maps, e):
        """“L12 (S2 wing)” / “M4 fuselage” for a target or marker event."""
        scene = self.vp.scene
        obj = scene.object_by_id(e.get("obj")) if e.get("obj") is not None else None
        oid = None
        if obj is not None:
            for m in ("sid", "fid", "mid", "hid"):
                oid = maps[m].get(obj.id, oid)
        name = f"{oid or '(hidden)'} {getattr(obj, 'name', '') or ''}".strip() if obj is not None else "(deleted)"
        if e.get("curve") is not None:
            lid = maps["lid"].get(e["curve"])
            return f"{lid} ({name})" if lid else f"a line of {name}"
        return name

    def _note_name(self, maps, nid):
        scene = self.vp.scene
        if nid == scene.global_note.id:
            return "the project note"
        n = scene.note_by_id(nid)
        return f"{maps['nid'][nid]} „{n.title}“" if n is not None and nid in maps["nid"] else "a note deleted since"

    def _board_mark_text(self, maps, e):
        """Board marker as short text for review_read."""
        where = self._note_name(maps, e.get("brett"))
        if not e.get("datei"):
            return f"BOARD MARK {e['n']} on {where} (no picture there)"
        u, v = e.get("uv") or (0.5, 0.5)
        cap = f" „{e['unterschrift']}“" if e.get("unterschrift") else ""
        return (f"BOARD MARK {e['n']}: picture{cap} on {where}, clicked at {u:.0%} from left, {v:.0%} from top"
                + ("" if self.vp.scene.asset_bytes(e["datei"]) is not None else " – picture no longer in the scene"))

    def _change_text(self, maps, e):
        """The artist's own change as short text."""
        typ = e["typ"]
        if typ == "notiz_neu":
            return f"created note {self._note_name(maps, e['id'])}"
        if typ == "brett":
            what = [f"+{e[k]} {name}" for k, name in (("bilder", "picture(s)"), ("texte", "text(s)"),
                                                     ("striche", "drawn stroke(s)")) if e.get(k, 0) > 0]
            if e.get("titel"):
                what.append(f"title „{e['titel']}“")
            return f"board of {self._note_name(maps, e['id'])}: " + (", ".join(what) or "changed")
        if typ == "notiz_weg":
            return f"deleted {len(e['ids'])} note(s)"
        if typ == "linien":
            lids = [maps["lid"][c] for c in e.get("neu", []) if c in maps["lid"]]
            objs = [self._label(maps, {"obj": o}) for o in e.get("objekte", [])[:3]]
            return (f"drew {e['anzahl']} line(s)" + (f" {', '.join(lids[:6])}" if lids else "")
                    + (f" in {', '.join(objs)}" if objs else ""))
        if typ == "linien_weg":
            return f"removed {e['anzahl']} line(s)"
        return typ

    def pre_review_read(self, review=None, start=0):
        """In the server thread: create the transcript if it is missing (Mule stays usable)."""
        from . import review as rv
        import time
        folder = rv.find_review(review)
        if folder is None or rv.read_transcript(folder) is not None:
            return
        if not os.path.isfile(os.path.join(folder, rv.AUDIO)):
            return
        busy = os.path.join(folder, "transkript.laeuft")
        for _ in range(1800):  # Mule is transcribing itself right now (after the stop): wait
            if not os.path.isfile(busy) or time.time() - os.path.getmtime(busy) > 3600:
                break
            time.sleep(1)
        if rv.read_transcript(folder) is None:
            ok, msg = rv.transcribe_now(folder)
            if not ok:
                self._transcribe_error = msg

    def review_read(self, review=None, start=0):
        from . import review as rv
        folder, events = self._review(review)
        name = os.path.basename(folder)
        head = events[0] if events and events[0]["typ"] == "start" else {}
        tr_ = rv.read_transcript(folder)
        all_names = [n for n, _ in rv.list_reviews()]
        lines = [f"Review „{name}“ (reviews: {', '.join(all_names[:8])}{' …' if len(all_names) > 8 else ''})",
                 f"Scene then: {head.get('szene') or '(not saved)'} – open now: {self.w.path or '(not saved)'}"
                 + ("" if (head.get("szene") or None) == (self.w.path or None) else
                    " – NOT THE SAME FILE: ids and pictures may not match"),
                 f"Length {rv.clock(events[-1]['t'] if events else 0)}, "
                 f"{sum(1 for e in events if e['typ'] == 'marke')} „Hier!“ mark(s), "
                 f"{sum(1 for e in events if e['typ'] == 'brettmarke')} board mark(s), "
                 f"{len(rv.cameras(events))} camera change(s)."]
        if not head.get("ton", True):
            lines.append(f"No sound was recorded ({head.get('fehler') or 'no microphone'}).")
        drafts = [(k, n) for k, n in self._notes() if n.review and n.review.get("review") == name]
        if drafts:
            lines.append("Drafts already written from this review: " + ", ".join(
                f"{k} „{n.title}“ ({rv.clock(n.review.get('t', 0))})" for k, n in drafts))
        own = [e for e in events if e["typ"] in rv.CHANGES]
        if own:
            maps0 = self._ids()
            lines.append("The artist changed during the review (theirs – build on it, do not duplicate): "
                         + "; ".join(f"{rv.clock(e['t'])} {self._change_text(maps0, e)}" for e in own[:40])
                         + (f"; … {len(own) - 40} more" if len(own) > 40 else ""))
        if tr_ is None:
            err = getattr(self, "_transcribe_error", None)
            self._transcribe_error = None
            lines.append("No transcript" + (f": {err}" if err else " (no sound or no transcription available)")
                         + ". Only the marks and what was pointed at:")
            maps = self._ids()
            for e in events:
                if e["typ"] == "marke":
                    lines.append(f"[{rv.clock(e['t'])}] mark {e['n']}" + (f" on {self._label(maps, e)}"
                                                                           if e.get("obj") is not None else ""))
                elif e["typ"] == "brettmarke":
                    lines.append(f"[{rv.clock(e['t'])}] {self._board_mark_text(maps, e)}")
            return [{"type": "text", "text": "\n".join(lines)}]
        said = rv.word_pointings(events, tr_)
        lines.append(f"Pointing: {sum(1 for e in events if e['typ'] == 'marke')} mark(s), "
                     f"{sum(1 for e in events if e['typ'] == 'ziel')} cursor rest(s), {len(said)} spoken "
                     "„hier/here/dort/there“ with cursor and camera still – take t from these.")
        lines.append(f"Transcript ({tr_.get('language', '?')}, {tr_.get('model', '')}) – per piece: what the cursor "
                     "rested on while it was said, marks, camera numbers (moving = the artist was navigating), the "
                     "selection and the open note board. Look at moments with review_moment.")
        segs = rv.segments_with_context(events, tr_)
        maps = self._ids()
        nid = maps["nid"]
        start = max(int(start or 0), 0)
        page = segs[start:start + 120]
        for s in page:
            bits = []
            if s["targets"]:
                bits.append("cursor on " + ", ".join(
                    f"{self._label(maps, e)} @{rv.clock(e['t'])}" for e in s["targets"][:6]))
            if s["marks"]:
                bits.append("MARK " + ", ".join(f"{e['n']} @{rv.clock(e['t'])}" for e in s["marks"]))
            for e in s.get("board_marks") or ():
                bits.append(f"{self._board_mark_text(maps, e)} @{rv.clock(e['t'])}")
            spoken = [w for w in said if s["start"] - 0.1 <= w["t"] <= s["end"] + 0.1]
            if spoken:
                bits.append("said " + ", ".join(f"„{w['wort']}“ @{rv.clock(w['t'])}" for w in spoken[:4])
                            + " (cursor still)")
            if s.get("changes"):
                bits.append("artist " + ", ".join(self._change_text(maps, e) for e in s["changes"][:4]))
            if s["cams"]:
                bits.append(f"cam #{s['cams'][0]}–#{s['cams'][-1]}" if len(s["cams"]) > 1
                            else f"cam #{s['cams'][0]}")
                if s["moving"]:
                    bits[-1] += " (moving)"
            sel = [self._label(maps, {"obj": i}) for i in s["selection"][:5]]
            if sel:
                bits.append("selected " + ", ".join(sel))
            if s["note"] is not None and s["note"] in nid:
                n = self.vp.scene.note_by_id(s["note"])
                bits.append(f"board {nid[s['note']]} „{getattr(n, 'title', '')}“")
            lines.append(f"[{rv.clock(s['start'])}–{rv.clock(s['end'])}] „{s['text']}“"
                         + (" — " + " · ".join(bits) if bits else ""))
        rest = len(segs) - start - len(page)
        if rest > 0:
            lines.append(f"… {rest} more piece(s): call again with start={start + len(page)}")
        return [{"type": "text", "text": "\n".join(lines)}]

    def _render_at(self, cam, look=None):
        """Image from a recorded camera (aspect ratio of the view back then – the cursor shares fit)."""
        import numpy as np
        from . import brief
        vp = self.vp
        with brief._keep_camera(vp) as c:
            c.set_state(np.array(cam["target"], float), float(cam["dist"]), np.array(cam["rot"], float))
            c.ortho = bool(cam.get("ortho"))
            saved = vp.planes_visible
            vp.planes_visible = False
            try:
                return vp.clean_image(aspect=max(cam.get("w", 1), 1) / max(cam.get("h", 1), 1), look=look)
            finally:
                vp.planes_visible = saved
                vp.update()

    @staticmethod
    def _cursor_uv(m):
        mouse, cam = m["maus"], m["kamera"]
        if mouse is None or cam is None or mouse.get("vp") != cam.get("vp"):
            return None
        return float(mouse["x"]), float(mouse["y"])

    def review_moment(self, times, review=None, xray=None, detail=False):
        from PySide6.QtGui import QColor, QImage, QPainter, QPen
        from . import review as rv
        folder, events = self._review(review)
        if not times:
            raise ValueError("give times (seconds)")
        look = None if xray is None else {"xray": 0.6 if xray else None}
        maps = self._ids()
        out = []
        for t in list(times)[:6]:
            m = rv.moment(events, float(t))
            if m["kamera"] is None:
                out.append({"type": "text", "text": f"{rv.clock(t)}: no camera recorded"})
                continue
            img = self._render_at(m["kamera"], look).convertToFormat(QImage.Format_RGB32)
            uv = self._cursor_uv(m)
            if uv is not None:
                p = QPainter(img)
                p.setRenderHint(QPainter.Antialiasing)
                x, y = uv[0] * img.width(), uv[1] * img.height()
                r = max(img.width(), img.height()) / 60
                for color, w in ((QColor(255, 255, 255), 6), (QColor(225, 0, 160), 3)):
                    p.setPen(QPen(color, w))
                    p.drawEllipse(x - r, y - r, 2 * r, 2 * r)
                p.end()
            cams = rv.cameras(events)
            label = (f"at {rv.clock(t)} – cam #{cams.index(m['kamera']) + 1}"
                     + (f" – cursor on {self._label(maps, m['ziel'])}" if m["ziel"] else
                        " – cursor outside the view" if uv is None else ""))
            out += [{"type": "text", "text": label + ":"},
                    {"type": "image", "data": _png(img, IMAGE_MAX if detail else MOMENT_MAX), "mimeType": "image/png"}]
        return out

    def _part(self, maps, pid):
        """Part identifier (L#, S#, F#, M#, H#) -> (object, line or None)."""
        scene = self.vp.scene
        pid = pid.strip()
        for m in ("sid", "fid", "mid", "hid"):
            for oid, k in maps[m].items():
                if k == pid:
                    return scene.object_by_id(oid), None
        for cid, k in maps["lid"].items():
            if k == pid:
                c = scene.curve_by_id(cid)
                return scene.owner_of(c), c
        raise ValueError(f"unknown part id {pid!r} (ids as in scene_brief)")

    def _pick_moment(self, cam, uv):
        """What lay at the place uv in the recorded view (target as in the log) – only if the view now has
        the same aspect ratio (otherwise the place does not match), else None."""
        import numpy as np
        from PySide6.QtCore import QPointF
        from . import brief
        from . import review as rv
        vp = self.vp
        if cam is None or abs(vp.width() / max(vp.height(), 1) - cam.get("w", 1) / max(cam.get("h", 1), 1)) > 0.01:
            return None
        with brief._keep_camera(vp) as c:
            c.set_state(np.array(cam["target"], float), float(cam["dist"]), np.array(cam["rot"], float))
            c.ortho = bool(cam.get("ortho"))
            hit = rv.pick_at(vp, QPointF(uv[0] * vp.width(), uv[1] * vp.height()))
        vp.update()
        return hit

    def _anchor_for(self, maps, m, anchor):
        """NoteAnchor for the draft: under the cursor (target of the moment), on a part or none."""
        import numpy as np
        from . import notes
        from .note_tool import _member_points
        from .scene import NoteAnchor
        scene = self.vp.scene
        anchor = (anchor or "cursor").strip()
        if anchor.lower() == "none":
            return None
        if anchor.lower() == "cursor":
            e = m["ziel"]
            if e is None:
                return None
            curve = scene.curve_by_id(e["curve"]) if e.get("curve") is not None else None
            point = np.array(e["point"], float)
            if curve is not None and len(curve.points) > 1:
                _, t = notes.curve_param(curve.points, point)
                return NoteAnchor(notes.curve_point(curve.points, t), curve.id, t)
            return NoteAnchor(point)
        obj, curve = self._part(maps, anchor)
        if curve is not None:
            return NoteAnchor(notes.curve_point(curve.points, 0.5), curve.id, 0.5)
        pts = _member_points(obj, scene)
        if pts is None or not len(pts):
            raise ValueError(f"part {anchor} has no geometry to point at")
        c = pts.mean(axis=0)
        return NoteAnchor(pts[int(np.argmin(np.linalg.norm(pts - c, axis=1)))].astype(float))

    def _review_backup(self, folder):
        """Before the first draft of a review (once per scene; "Send review" already does it on sending): backup
        of the open scene in review_backups/ next to it – the drafts then go live into the open scene. -> hint."""
        return backup_before_review(self.w, os.path.basename(folder))

    def _drafted(self, batch, name):
        """Single draft outside review_notes: Mule counts it – if nothing comes for a while afterwards, the
        review counts as finished (review.DRAFT_SETTLE_S)."""
        if not batch and getattr(self.w, "review_sender", None) is not None:
            self.w.review_sender.drafted(name)

    def _activity(self):
        """Every tool call: "Claude is working" stays on as long as Claude uses Mule."""
        if getattr(self.w, "review_sender", None) is not None:
            self.w.review_sender.activity()

    def review_notes(self, notes, review=None):
        scene = self.vp.scene
        self._review(review)  # unknown review: error before anything happens
        self._batch = True
        scene.checkpoint()  # all drafts together: one Ctrl+Z
        out = []
        try:
            for k, spec in enumerate(notes, 1):
                try:
                    msg = self.review_note(review=review, **spec)[0]["text"]
                except Exception as exc:
                    msg = f"FAILED – {exc}"
                out.append(f"{k}. {msg}")
        finally:
            self._batch = False
            self.vp._changed()
            self.w._update_title()
        made = sum(1 for line in out if "FAILED" not in line)
        from . import review as rv
        folder = rv.find_review(review)
        if getattr(self.w, "review_sender", None) is not None and folder:
            self.w.review_sender.finished(os.path.basename(folder), made)
        self.w.statusBar().showMessage(tr("KI: {0} Review-Entwürfe eingefügt – Strg+Z nimmt alle zurück").format(made),
                                       10000)
        return [{"type": "text", "text": f"{made} of {len(notes)} draft(s) in the scene (one undo step):\n"
                 + "\n".join(out)}]

    def review_note(self, t=None, text="", review=None, t_end=None, title=None, quote="", note=None, anchor=None,
                    parts=(), images=None, discard=False, mark=None, refs=()):
        import numpy as np
        from PySide6.QtCore import QBuffer, QIODevice
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QGraphicsScene
        from . import notes
        from . import review as rv
        from .note_board import add_review_entries, review_items, SNAP_MAX, SNAP_QUALITY
        from .note_tool import NEW_OFFSET_PX
        from .scene import Note, NoteMember
        scene, vp = self.vp.scene, self.vp
        if discard:
            if not note:
                raise ValueError("discard needs note (the N# of the draft)")
            key, hit = self._note(note)
            if not hit.review or hit is scene.global_note:
                raise ValueError(f"{key} is not a review draft – only drafts can be discarded here")
            scene.checkpoint()
            scene.notes.remove(hit)
            if vp.ws.active_note == hit.id:
                vp.ws.active_note = None
            vp._changed()
            self.w._update_title()
            return [{"type": "text", "text": f"Draft {key} „{hit.title}“ deleted."}]
        if t is None and mark is None:
            raise ValueError("give t (seconds in the review) or mark")
        folder, events = self._review(review)
        name = os.path.basename(folder)
        text = (text or "").strip()
        board_refs = []
        for r in list(refs or ())[:4]:
            ev = next((e for e in events if e["typ"] == "brettmarke" and e.get("n") == int(r)), None)
            if ev is None:
                raise ValueError(f"no board mark {r} in this review")
            data = scene.asset_bytes(ev["datei"]) if ev.get("datei") else None
            if data is None:
                raise ValueError(f"board mark {r}: no picture there (or it was removed from the scene)")
            board_refs.append((data, None, tuple(ev.get("uv") or (0.5, 0.5)), ev.get("unterschrift")))
        if mark is not None:
            hit = next((e for e in events if e["typ"] == "marke" and e.get("n") == int(mark)), None)
            if hit is None:
                if any(e["typ"] == "brettmarke" and e.get("n") == int(mark) for e in events):
                    raise ValueError(f"mark {mark} is a board mark – pass it in refs, and t or mark of the part")
                raise ValueError(f"no mark {mark} in this review")
            t = hit["t"]
        asked = float(t)
        words = rv.word_pointings(events, rv.read_transcript(folder))
        point = rv.pointing_near(events, asked, words)  # really shown? then exactly this moment
        t = point["t"] if point is not None else asked
        m = rv.moment(events, float(t))
        if point is None:
            m["ziel"] = None  # without showing no anchor at the cursor
        elif point["typ"] == "wort":  # spoken "here": look up what lay under the cursor back then
            m["ziel"] = self._pick_moment(m["kamera"], (point["x"], point["y"]))
        else:
            m["ziel"] = point if point.get("obj") is not None else None
        if m["kamera"] is None:
            raise ValueError("no camera recorded in this review")
        maps = self._ids()

        def shot(at):
            near = rv.pointing_near(events, float(at), words)
            at = near["t"] if near is not None else at
            mm = rv.moment(events, float(at))
            if near is None:
                mm["maus"] = None  # no pointing place: better no arrow than a wrong one
            img = self._render_at(mm["kamera"])
            longest = max(img.width(), img.height())
            if longest > SNAP_MAX:
                img = img.scaled(img.width() * SNAP_MAX // longest, img.height() * SNAP_MAX // longest)
            buf = QBuffer()
            buf.open(QIODevice.WriteOnly)
            img.convertToFormat(QImage.Format_RGB32).save(buf, "JPG", SNAP_QUALITY)
            c = mm["kamera"]
            view = {"target": c["target"], "dist": c["dist"], "rot": c["rot"], "ortho": bool(c.get("ortho"))}
            return bytes(buf.data()), view, self._cursor_uv(mm)

        copied = self._review_backup(folder)
        batch = getattr(self, "_batch", False)
        times = [float(x) for x in (images if images is not None else [t])][:4]
        how = (f" t moved {asked:.1f} -> {t:.1f} s to " if abs(t - asked) > 0.05 else " At ") \
            + ("„Hier!“ mark " + str(point["n"]) if point is not None and point["typ"] == "marke"
               else f"the spoken „{point['wort']}“" if point is not None and point["typ"] == "wort"
               else "a cursor rest" if point is not None else "") \
            + ("." if point is not None else " No pointing near t: no arrow and no cursor anchor.")
        shots = [shot(x) for x in times] + board_refs
        stamp = f"Review {name.rsplit('_', 2)[-2] if name.count('_') >= 2 else name}, {rv.clock(t)}"
        if note:  # to an existing note (or the project note)
            key, hit = self._note(note)
            body = (text + "\n" if text else "") + f"({stamp} – draft)"
            add_review_entries(vp, hit, body, shots, checkpoint=not batch)
            self.w._update_title()
            self._drafted(batch, name)
            return [{"type": "text", "text": f"Added to {key} {hit.title or ''}".rstrip()
                     + f": text and {len(shots)} picture(s), marked „{stamp} – draft“." + how + copied}]
        if not text:
            raise ValueError("empty text")
        anc = self._anchor_for(maps, m, anchor)
        cam = m["kamera"]
        from . import brief
        with brief._keep_camera(vp) as c:  # placement as when created by hand – from the view back then
            c.set_state(np.array(cam["target"], float), float(cam["dist"]), np.array(cam["rot"], float))
            c.ortho = bool(cam.get("ortho"))
            if anc is not None:
                right = notes.right_vector(Note("", anc.point), c.right)
                wpp = float(vp._world_per_px(max(float((anc.point - c.eye) @ -c.back), 1e-3)))
                where = anc.point + right * NEW_OFFSET_PX[0] * wpp + notes.UP * NEW_OFFSET_PX[1] * wpp
            else:
                where = np.array(cam["target"], float)
        members, bad = [], []
        for pid in parts or ():
            try:
                obj, curve = self._part(maps, pid)
            except ValueError:
                bad.append(pid)
                continue
            if obj is not None:
                members.append(NoteMember(obj.id, curve.id if curve is not None else None))
        if anc is not None:
            anc.members = members
        if t_end is None:  # end of the transcript section in which t lies
            segs = (rv.read_transcript(folder) or {}).get("segments") or []
            t_end = next((s["end"] for s in segs if s["start"] - 0.5 <= float(t) <= s["end"] + 0.5), float(t) + 6)
        first = text.splitlines()[0].strip().lstrip("-•*– ").strip()
        title = (title or first).strip()[:60]
        gscene = QGraphicsScene()
        pics, texts = review_items(scene, gscene, text, shots)
        gscene.clear()
        new = Note(title, np.asarray(where, float), "m", anchors=[anc] if anc is not None else [],
                   board={"items": pics + texts},
                   review={"review": name, "t": round(float(t), 2), "t_end": round(float(t_end), 2),
                           "quote": (quote or "").strip()})
        if not batch:
            scene.checkpoint()
        scene.notes.append(new)
        if not batch:
            vp._changed()
            self.w._update_title()
            self.w.statusBar().showMessage(tr("KI: Review-Entwurf „{0}“ angelegt – Strg+Z nimmt ihn zurück")
                                           .format(title), 8000)
        self._drafted(batch, name)
        key = self._ids()["nid"].get(new.id, "N?")
        where_txt = ("at what the cursor was on" if (anchor or "cursor") == "cursor" else f"at {anchor}") \
            if anc is not None else "without a leader line (nothing under the cursor – pass anchor with a part id?)"
        return [{"type": "text", "text": f"Draft {key} „{title}“ created {where_txt}, {len(shots)} picture(s)"
                 + (" with an arrow to the cursor spot" if any(s[2] for s in shots) else "")
                 + (f", {len(members)} octopus arm(s)" if members else "")
                 + (f". Unknown part ids left out: {', '.join(bad)}" if bad else "")
                 + (". parts need an anchor – none added" if members and anc is None else "") + "." + how + copied}]

    def set_hero_prompt(self, hero, prompt):
        from .note_board import set_hero_prompt
        if not set_hero_prompt(self.vp, hero, prompt.strip()):
            from . import brief
            names = ", ".join(h["name"] for h in brief.board_heroes(self.vp.scene)) or "none"
            raise ValueError(f"no hero camera named {hero!r} (hero cameras: {names})")
        self.w._update_title()
        return [{"type": "text", "text": f"Prompt stored on hero camera „{hero}“." if prompt.strip()
                 else f"Prompt removed from „{hero}“."}]


class _Handler(BaseHTTPRequestHandler):
    server_version = "SketchMachineMCP/1"

    def log_message(self, *args):  # silent
        pass

    def _send(self, status, payload=None):
        body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # no server push (SSE): request/response only
        self._send(405)

    def do_DELETE(self):
        self._send(405)

    def do_POST(self):
        if self.path.rstrip("/") != "/mcp":
            return self._send(404)
        try:
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
        except json.JSONDecodeError:
            return self._send(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
        msgs = msg if isinstance(msg, list) else [msg]
        replies = [r for r in (self.server.mcp.handle(m) for m in msgs) if r is not None]
        if not replies:
            return self._send(202)
        self._send(200, replies if isinstance(msg, list) else replies[0])


class Server:
    """JSON-RPC (MCP) over HTTP; start() in the background thread, stop() on exit."""

    def __init__(self, window, port=PORT):
        self.tools = Tools(window)
        self.main = _MainThread()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.mcp = self
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, m):
        """One JSON-RPC message; None for notifications (without id)."""
        if not isinstance(m, dict) or "method" not in m:
            return None
        rid, method, params = m.get("id"), m["method"], m.get("params") or {}
        if rid is None:
            return None  # notifications/initialized etc.
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                result = {"protocolVersion": asked if asked in VERSIONS else VERSIONS[0],
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "sketch_machine", "version": "1"},
                          "instructions": "Mule is a 3D sketching tool whose Sketch Brief is a prompt for "
                                          "building the sketched object. Talk to the artist in the language the artist writes in. Change notes "
                                          "only when the artist explicitly asks (edit_note); write back only what "
                                          "the artist confirmed or dictated to you (dictate_note). Review "
                                          "recordings (review_read) become draft notes (review_note) that the "
                                          "artist checks in Mule."}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                name, args = params.get("name"), params.get("arguments") or {}
                fn = getattr(self.tools, name, None) if any(t["name"] == name for t in TOOLS) else None
                if fn is None:
                    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
                try:
                    pre = getattr(self.tools, "pre_" + name, None)  # long-running work outside the main thread (transcript)
                    if pre is not None:
                        pre(**args)
                    result = {"content": self.main.call(lambda: (fn(**args), self.tools._activity())[0]),
                              "isError": False}
                except Exception as exc:
                    result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            else:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method {method}"}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}


def start(window, port=PORT):
    """Start server; None if the port is taken (e.g. a second Mule is running)."""
    try:
        return Server(window, port).start()
    except OSError:
        return None
