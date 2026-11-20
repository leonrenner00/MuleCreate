"""Hold-to-invoke / release-to-confirm radial (pie) menus — MACHIN3 / Blender pie feel.

W  → transform toolset (move / rotate / scale + spaces + axis)
Tab → selection kinds (point / edge / face); short tap still toggles draw ↔ edit
"""
from .common import *  # noqa: F401,F403
from ..i18n import tr
from ..shortcuts import letter

PIE_INNER = 28.0  # dead zone: release here = cancel / tap fallback
PIE_OUTER = 118.0
PIE_LABEL_R = 78.0
PIE_SLICE_GAP = 0.04  # radians visual gap between slices


def _angle_of(dx, dy):
    """Screen angle: 0 = right, CCW, matching atan2(-dy, dx) so up is +90°."""
    return math.atan2(-dy, dx)


def _wrap_delta(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


class PieMixin:
    """Pie menus drawn in the QPainter overlay; driven from key press/release + mouse move."""

    def _init_pie(self):
        self._pie = None  # dict while open, else None

    def pie_open(self):
        return self._pie is not None

    def _pie_items_transform(self):
        """Radial tool functions around Move (and related transform tools)."""
        space = self.gizmo.space
        tool = self.gizmo.tool
        return [
            {"id": "translate", "label": tr("Verschieben"), "hint": letter("translate"),
             "active": tool == "translate",
             "run": lambda: self.set_transform_tool("translate")},
            {"id": "rotate", "label": tr("Drehen"), "hint": letter("rotate"),
             "active": tool == "rotate",
             "run": lambda: self.set_transform_tool("rotate")},
            {"id": "scale", "label": tr("Skalieren"), "hint": letter("scale"),
             "active": tool == "scale",
             "run": lambda: self.set_transform_tool("scale")},
            {"id": "world", "label": tr("Welt"), "hint": "",
             "active": space == "world",
             "run": lambda: self._pie_set_space("world")},
            {"id": "local", "label": tr("Lokal"), "hint": "",
             "active": space == "local",
             "run": lambda: self._pie_set_space("local")},
            {"id": "view", "label": tr("Kamera"), "hint": "",
             "active": space == "view",
             "run": lambda: self._pie_set_space("view")},
            {"id": "axis", "label": tr("Achsenmodus"), "hint": "Alt+A",
             "active": bool(self.ws.axis_mode),
             "run": lambda: self.set_axis_mode(not self.ws.axis_mode)},
            {"id": "snap", "label": tr("Einrast-Schritte …"), "hint": "",
             "active": False,
             "run": self.snap_dialog},
        ]

    def _pie_items_select(self):
        comp = self.ws.component
        return [
            {"id": "point", "label": tr("Punkte"), "hint": "1",
             "active": comp == "point",
             "run": lambda: self._pie_pick_component("point")},
            {"id": "edge", "label": tr("Kanten"), "hint": "2",
             "active": comp == "edge",
             "run": lambda: self._pie_pick_component("edge")},
            {"id": "face", "label": tr("Flächen"), "hint": "3",
             "active": comp == "face",
             "run": lambda: self._pie_pick_component("face")},
        ]

    def _pie_set_space(self, space):
        self.gizmo.space = space
        self.toggles_changed.emit()
        self._changed()

    def _pie_pick_component(self, comp):
        if not self.select_mode:
            self.set_select_mode(True)
        self.set_component(comp)

    def open_pie(self, kind, key, origin=None, tap=None):
        """Open a pie at the cursor (or origin). tap: callable if released in the dead zone."""
        if self._mode not in (None, "consumed"):
            return False
        if kind == "transform":
            items = self._pie_items_transform()
            title = tr("Transformieren")
        elif kind == "select":
            items = self._pie_items_select()
            title = tr("Auswahl")
        else:
            return False
        if not items:
            return False
        pos = origin if origin is not None else (self._last_pos or QPointF(self.width() / 2, self.height() / 2))
        n = len(items)
        # First slice centered on +Y (up) so primary actions feel “north”.
        start = math.pi / 2
        step = 2 * math.pi / n
        for i, item in enumerate(items):
            item["angle"] = start - i * step
            item["index"] = i
        self._pie = {
            "kind": kind,
            "key": key,
            "origin": QPointF(pos),
            "items": items,
            "hover": None,
            "title": title,
            "tap": tap,
            "t0": time.monotonic(),
        }
        self._mode = "pie"
        self.update()
        return True

    def _pie_hover_index(self, pos):
        pie = self._pie
        if pie is None:
            return None
        o = pie["origin"]
        dx, dy = pos.x() - o.x(), pos.y() - o.y()
        dist = math.hypot(dx, dy)
        if dist < PIE_INNER:
            return None
        ang = _angle_of(dx, dy)
        best, best_d = None, 1e9
        for item in pie["items"]:
            d = abs(_wrap_delta(ang, item["angle"]))
            if d < best_d:
                best, best_d = item["index"], d
        half = math.pi / len(pie["items"])
        return best if best_d <= half + 1e-6 else None

    def pie_move(self, pos):
        if self._pie is None:
            return
        hover = self._pie_hover_index(pos)
        if hover != self._pie.get("hover"):
            self._pie["hover"] = hover
            self.update()

    def confirm_pie(self, pos=None):
        """Release-to-confirm: run hovered slice, or tap-fallback / cancel in the dead zone."""
        pie = self._pie
        if pie is None:
            return
        if pos is None:
            pos = self._last_pos or pie["origin"]
        hover = self._pie_hover_index(pos)
        items = pie["items"]
        tap = pie.get("tap")
        self._close_pie(keep_mode=False)
        if hover is not None and 0 <= hover < len(items):
            items[hover]["run"]()
            return
        if tap is not None:
            tap()

    def cancel_pie(self):
        self._close_pie(keep_mode=False)

    def _close_pie(self, keep_mode=False):
        self._pie = None
        if not keep_mode and self._mode == "pie":
            self._mode = None
        self.update()

    def _draw_pie(self, p):
        pie = self._pie
        if pie is None:
            return
        o = pie["origin"]
        n = len(pie["items"])
        step = 2 * math.pi / n
        hover = pie.get("hover")

        # Dim veil so the pie reads clearly over the scene.
        veil = QColor(theme.PAPER)
        veil.setAlpha(90)
        p.fillRect(self.rect(), veil)

        # Outer disc
        p.setPen(QPen(theme.GRAPHITE_SOFT, 1.2))
        fill = QColor(theme.PAPER)
        fill.setAlpha(220)
        p.setBrush(fill)
        p.drawEllipse(o, PIE_OUTER, PIE_OUTER)

        # Inner dead-zone disc
        inner = QColor(theme.PAPER)
        inner.setAlpha(255)
        p.setBrush(inner)
        p.setPen(QPen(theme.GRAPHITE_SOFT, 1.0))
        p.drawEllipse(o, PIE_INNER, PIE_INNER)

        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)

        for item in pie["items"]:
            i = item["index"]
            a0 = item["angle"] + step / 2 - PIE_SLICE_GAP / 2
            a1 = item["angle"] - step / 2 + PIE_SLICE_GAP / 2
            hot = hover == i
            active = item.get("active")
            path = QPainterPath()
            path.moveTo(o)
            # Fan from a0 → a1 (CW visually because Qt Y grows down; angles use math Y-up).
            steps = max(8, int(abs(a0 - a1) / 0.08))
            for s in range(steps + 1):
                t = s / steps
                a = a0 + (a1 - a0) * t
                path.lineTo(QPointF(o.x() + PIE_OUTER * math.cos(a),
                                    o.y() - PIE_OUTER * math.sin(a)))
            path.closeSubpath()
            if hot:
                brush = QColor(theme.TERRACOTTA)
                brush.setAlpha(200)
            elif active:
                brush = QColor(theme.SEA)
                brush.setAlpha(150)
            else:
                brush = QColor(theme.GRAPHITE)
                brush.setAlpha(35)
            p.setBrush(brush)
            p.setPen(QPen(theme.GRAPHITE_SOFT, 1.0))
            p.drawPath(path)

            lx = o.x() + PIE_LABEL_R * math.cos(item["angle"])
            ly = o.y() - PIE_LABEL_R * math.sin(item["angle"])
            label = item["label"]
            if item.get("hint"):
                label = f"{label}  {item['hint']}"
            color = theme.PAPER if hot else theme.GRAPHITE
            p.setPen(color)
            metrics = QFontMetricsF(p.font())
            w = metrics.horizontalAdvance(label)
            h = metrics.height()
            self._text(p, QRectF(lx - w / 2, ly - h / 2, w, h), Qt.AlignCenter, label)

        # Center title
        font.setPointSize(8)
        font.setBold(False)
        p.setFont(font)
        p.setPen(theme.GRAPHITE)
        title = pie["title"]
        metrics = QFontMetricsF(p.font())
        tw = metrics.horizontalAdvance(title)
        th = metrics.height()
        self._text(p, QRectF(o.x() - tw / 2, o.y() - th / 2, tw, th), Qt.AlignCenter, title)
