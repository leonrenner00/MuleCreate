"""Tool shortcuts: Mule (T/R/E) and Blender (G/R/S), plus a custom map saved in settings.

Only move, rotate and scale are rebindable. Other keys (Alt+G, Space+G, file shortcuts) stay as they are.
A letter bound to a tool no longer does its old job (Blender S scales; it does not toggle paint).
"""
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel

from .i18n import tr

ACTIONS = ("translate", "rotate", "scale")
PRESETS = {
    "mule": {"translate": "T", "rotate": "R", "scale": "E"},
    "blender": {"translate": "G", "rotate": "R", "scale": "S"},
}
_LABELS = {
    "translate": "Verschieben",
    "rotate": "Drehen",
    "scale": "Skalieren",
}
_current = None  # {"preset": str, "letters": {action: letter}}


def _settings():
    return QSettings("SketchMachine", "SketchMachine")


def _clean(letters):
    out = {}
    for action in ACTIONS:
        letter = str(letters.get(action, PRESETS["mule"][action])).strip().upper()[:1]
        if not ("A" <= letter <= "Z"):
            letter = PRESETS["mule"][action]
        out[action] = letter
    return out


def _which_preset(letters):
    for name, preset in PRESETS.items():
        if letters == preset:
            return name
    return "custom"


def load():
    """The saved map, or the Mule preset when nothing is stored."""
    global _current
    if _current is not None:
        return _current
    s = _settings()
    stored = {action: s.value(f"shortcut_{action}", "") for action in ACTIONS}
    if any(stored.values()):
        letters = _clean(stored)
        preset = str(s.value("shortcut_preset", "") or _which_preset(letters))
        if preset not in ("mule", "blender", "custom"):
            preset = _which_preset(letters)
    else:
        preset, letters = "mule", dict(PRESETS["mule"])
    _current = {"preset": preset, "letters": letters}
    return _current


def save(preset, letters):
    """Remember the map and use it immediately. Rejects a letter used twice."""
    global _current
    letters = _clean(letters)
    if len(set(letters.values())) != len(ACTIONS):
        return tr("Jede Taste darf nur ein Werkzeug haben.")
    preset = preset if preset in ("mule", "blender", "custom") else _which_preset(letters)
    if preset in PRESETS and letters != PRESETS[preset]:
        preset = "custom"
    s = _settings()
    s.setValue("shortcut_preset", preset)
    for action, letter in letters.items():
        s.setValue(f"shortcut_{action}", letter)
    _current = {"preset": preset, "letters": letters}
    return None


def letters():
    return dict(load()["letters"])


def letter(action):
    return load()["letters"][action]


def action_for(key):
    """Transform tool for a bare letter key, or None."""
    text = key_letter(key)
    if text is None:
        return None
    for action, letter_ in load()["letters"].items():
        if letter_ == text:
            return action
    return None


def key_letter(key):
    if Qt.Key_A <= key <= Qt.Key_Z:
        return chr(ord("A") + (key - Qt.Key_A))
    return None


def add_settings_group(layout):
    """Shortcut block for the settings dialog. Returns a callable that saves on OK."""
    state = load()
    draft = {"preset": state["preset"], "letters": dict(state["letters"])}
    group = QGroupBox(tr("Werkzeugkürzel"))
    form = QFormLayout(group)
    preset = QComboBox()
    preset.addItem(tr("Mule (T verschieben, R drehen, E skalieren)"), "mule")
    preset.addItem(tr("Blender (G verschieben, R drehen, S skalieren)"), "blender")
    preset.addItem(tr("Eigen"), "custom")
    boxes = {}

    def sync_preset_label():
        name = _which_preset(draft["letters"])
        draft["preset"] = name
        preset.blockSignals(True)
        preset.setCurrentIndex(max(0, preset.findData(name if name in PRESETS else "custom")))
        preset.blockSignals(False)

    def fill_boxes():
        for action, box in boxes.items():
            box.blockSignals(True)
            box.setCurrentIndex(max(0, box.findText(draft["letters"][action])))
            box.blockSignals(False)

    def choose_preset():
        name = preset.currentData()
        if name in PRESETS:
            draft["letters"] = dict(PRESETS[name])
            draft["preset"] = name
            fill_boxes()

    def choose_letter(action):
        picked = boxes[action].currentText()
        for other, letter_ in list(draft["letters"].items()):
            if other != action and letter_ == picked:
                # Swap so each tool keeps its own key.
                draft["letters"][other] = draft["letters"][action]
        draft["letters"][action] = picked
        fill_boxes()
        sync_preset_label()

    preset.currentIndexChanged.connect(lambda _i: choose_preset())
    form.addRow(tr("Vorgabe:"), preset)
    note = QLabel(tr("Blender legt Verschieben, Drehen und Skalieren auf G, R und S. "
                     "S schaltet dann nicht mehr den Malmodus um. Alt+G und Space+G bleiben wie bisher."))
    note.setWordWrap(True)
    form.addRow(note)
    for action in ACTIONS:
        box = QComboBox()
        for letter_ in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            box.addItem(letter_)
        box.currentIndexChanged.connect(lambda _i, a=action: choose_letter(a))
        boxes[action] = box
        form.addRow(tr(_LABELS[action]) + ":", box)
    fill_boxes()
    sync_preset_label()
    layout.addWidget(group)

    def apply():
        return save(draft["preset"], draft["letters"])

    return apply
