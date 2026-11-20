"""Tool shortcut presets: Mule keeps T/R/E; Blender uses G/R/S for move, rotate and scale."""
from PySide6.QtCore import Qt

from tests.harness import H
from sketch_machine import shortcuts


def _restore():
    shortcuts.save("mule", shortcuts.PRESETS["mule"])


def test_default_is_mule_move_rotate_scale(h: H):
    try:
        _restore()
        assert shortcuts.letters() == {"translate": "T", "rotate": "R", "scale": "E"}
        assert shortcuts.action_for(Qt.Key_T) == "translate"
        assert shortcuts.action_for(Qt.Key_R) == "rotate"
        assert shortcuts.action_for(Qt.Key_E) == "scale"
        assert shortcuts.action_for(Qt.Key_G) is None
    finally:
        _restore()


def test_blender_preset_is_g_r_s(h: H):
    try:
        err = shortcuts.save("blender", shortcuts.PRESETS["blender"])
        assert err is None
        assert shortcuts.letter("translate") == "G"
        assert shortcuts.letter("rotate") == "R"
        assert shortcuts.letter("scale") == "S"
        assert shortcuts.action_for(Qt.Key_G) == "translate"
        assert shortcuts.action_for(Qt.Key_R) == "rotate"
        assert shortcuts.action_for(Qt.Key_S) == "scale"
        assert shortcuts.action_for(Qt.Key_E) is None
        assert shortcuts.action_for(Qt.Key_T) is None
    finally:
        _restore()


def test_custom_letter_and_duplicate_rejected(h: H):
    try:
        _restore()
        err = shortcuts.save("custom", {"translate": "G", "rotate": "R", "scale": "G"})
        assert err
        assert shortcuts.letter("translate") == "T"  # rejected map is not stored
        err = shortcuts.save("custom", {"translate": "G", "rotate": "F", "scale": "E"})
        assert err is None
        assert shortcuts.load()["preset"] == "custom"
        assert shortcuts.action_for(Qt.Key_G) == "translate"
        assert shortcuts.action_for(Qt.Key_F) == "rotate"
    finally:
        _restore()
